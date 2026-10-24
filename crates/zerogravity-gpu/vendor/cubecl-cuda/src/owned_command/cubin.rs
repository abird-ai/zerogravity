//! Opt-in pinned cubin commands on the original CUDA actor.
use super::{Bf16BlasSources, PinnedBf16Cubin, PinnedCapturedBf16Cubin};
use cubecl_core::server::ServerError;
use sha2::{Digest, Sha256};
use std::any::Any;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

mod args;
mod attention_gate_policy;
pub(crate) mod attention_policy;
mod canonical47_fa_policy;
mod canonical47_full_policy;
mod canonical47_layer3_policy;
mod canonical47_mlp_policy;
pub(crate) mod canonical47_natural_policy;
mod canonical47_qkkv_policy;
mod canonical47_qkv_policy;
mod elf;
mod full_model_policy;
pub(crate) mod head_policy;
mod policy;
mod qwen2b_mlp_policy;
mod set;
pub(crate) mod short_hf_policy;
mod source_rotary_policy;
mod workspace_clear;
pub use args::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
pub use head_policy::{Bf16VocabularyHead, VocabularyHeadSelection};
pub(crate) use policy::CubinSetPolicy;
pub use set::{CubinWorkspaceView, OwnedCudaBinding, OwnedCudaKernel, OwnedCudaModuleSet};
pub use workspace_clear::CubinWorkspaceClear;

pub(crate) fn failure(message: impl Into<String>) -> ServerError {
    ServerError::graph_state(message)
}
const IMAGE_LIMIT: usize = 32 * 1024 * 1024;
const TOTAL_LIMIT: usize = 256 * 1024 * 1024;
static IMAGE_BYTES: AtomicUsize = AtomicUsize::new(0);

struct ImageCharge {
    bytes: usize,
    _admission_owner: Arc<dyn Any + Send + Sync>,
}
impl ImageCharge {
    fn reserve(
        bytes: usize,
        admission_owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, ServerError> {
        if bytes == 0 || bytes > IMAGE_LIMIT {
            return Err(failure("cubin source image exceeds 32MiB"));
        }
        IMAGE_BYTES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|n| *n <= TOTAL_LIMIT)
            })
            .map_err(|_| failure("cubin retained source images exceed 256MiB"))?;
        Ok(Self {
            bytes,
            _admission_owner: admission_owner,
        })
    }
}
impl Drop for ImageCharge {
    fn drop(&mut self) {
        IMAGE_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

// Declaration order covers reserve refusal and every later validation error.
struct AdmittedImage<T> {
    image: T,
    admission_owner: Arc<dyn Any + Send + Sync>,
}
struct ChargedImage<T> {
    image: T,
    charge: ImageCharge,
}
fn charge_owned_payload<T>(
    image: T,
    bytes: usize,
    admission_owner: Arc<dyn Any + Send + Sync>,
) -> Result<ChargedImage<T>, ServerError> {
    let admitted = AdmittedImage {
        image,
        admission_owner,
    };
    let charge = ImageCharge::reserve(bytes, admitted.admission_owner.clone())?;
    Ok(ChargedImage {
        image: admitted.image,
        charge,
    })
}
fn charge_owned_image(
    image: Vec<u8>,
    admission_owner: Arc<dyn Any + Send + Sync>,
) -> Result<ChargedImage<Vec<u8>>, ServerError> {
    let capacity = image.capacity();
    charge_owned_payload(image, capacity, admission_owner)
}

struct CapturedPayload<T> {
    image: T,
    sha256: [u8; 32],
    binary_sm: u8,
    charge: ImageCharge,
}
fn seal_captured_payload<T: AsRef<[u8]>>(
    image: ChargedImage<T>,
    kind: PinnedCapturedBf16Cubin,
) -> Result<CapturedPayload<T>, ServerError> {
    let bytes = image.image.as_ref();
    let hash: [u8; 32] = Sha256::digest(bytes).into();
    // Refuse non-registry payloads before ELF parsing allocates section metadata.
    kind.require_image(bytes.len(), hash, kind.capture().binary_sm)?;
    let binary_sm = elf::validate(bytes)?;
    kind.require_image(bytes.len(), hash, binary_sm)?;
    Ok(CapturedPayload {
        image: image.image,
        sha256: hash,
        binary_sm,
        charge: image.charge,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubinProvider {
    Cublas,
    CublasLt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CubinImageKey {
    issuer: usize,
    sha256: [u8; 32],
    bytes: usize,
    binary_sm: u8,
}
impl CubinImageKey {
    pub(crate) fn same_issuer(self, other: Self) -> bool {
        self.issuer == other.issuer
    }
}

/// Exact raw provider interval, fixed derivation or separately labelled capture.
/// Retains original host/native-module admission through every actor/image owner.
pub struct VerifiedCubin {
    pub(crate) sources: Arc<Bf16BlasSources>,
    pub(crate) bytes: Vec<u8>,
    sha256: [u8; 32],
    binary_sm: u8,
    derived_from: Option<PinnedBf16Cubin>,
    captured_from: Option<PinnedCapturedBf16Cubin>,
    _charge: ImageCharge,
}
impl std::fmt::Debug for VerifiedCubin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedCubin")
            .field("bytes", &self.bytes.len())
            .field("binary_sm", &self.binary_sm)
            .field("sha256", &self.sha256)
            .field("derived_from", &self.derived_from)
            .field("captured_from", &self.captured_from)
            .finish()
    }
}
impl VerifiedCubin {
    pub(crate) fn read_verified(
        sources: Arc<Bf16BlasSources>,
        admission_owner: Arc<dyn Any + Send + Sync>,
        bytes: usize,
        expected: [u8; 32],
        read: impl FnOnce(&mut [u8]) -> Result<(), ServerError>,
    ) -> Result<Arc<Self>, ServerError> {
        let charge = ImageCharge::reserve(bytes, admission_owner)?;
        let mut image = vec![0u8; bytes];
        read(&mut image)?;
        let hash: [u8; 32] = Sha256::digest(&image).into();
        if hash != expected {
            return Err(failure("cubin raw provider interval SHA differs"));
        }
        let binary_sm = elf::validate(&image)?;
        Ok(Arc::new(Self {
            sources,
            bytes: image,
            sha256: hash,
            binary_sm,
            derived_from: None,
            captured_from: None,
            _charge: charge,
        }))
    }
    pub(crate) fn seal_derived(
        sources: Arc<Bf16BlasSources>,
        admission_owner: Arc<dyn Any + Send + Sync>,
        kind: PinnedBf16Cubin,
        image: Vec<u8>,
    ) -> Result<Arc<Self>, ServerError> {
        let image = charge_owned_image(image, admission_owner)?;
        let hash: [u8; 32] = Sha256::digest(&image.image).into();
        let binary_sm = elf::validate(&image.image)?;
        kind.require_image(image.image.len(), hash, binary_sm)?;
        Ok(Arc::new(Self {
            sources,
            bytes: image.image,
            sha256: hash,
            binary_sm,
            derived_from: Some(kind),
            captured_from: None,
            _charge: image.charge,
        }))
    }
    pub(crate) fn seal_captured(
        sources: Arc<Bf16BlasSources>,
        admission_owner: Arc<dyn Any + Send + Sync>,
        kind: PinnedCapturedBf16Cubin,
        image: Vec<u8>,
    ) -> Result<Arc<Self>, ServerError> {
        let image = charge_owned_image(image, admission_owner)?;
        let image = seal_captured_payload(image, kind)?;
        Ok(Arc::new(Self {
            sources,
            bytes: image.image,
            sha256: image.sha256,
            binary_sm: image.binary_sm,
            derived_from: None,
            captured_from: Some(kind),
            _charge: image.charge,
        }))
    }
    pub(crate) fn require_admission_owner(
        &self,
        owner: &Arc<dyn Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if !Arc::ptr_eq(&self._charge._admission_owner, owner) {
            return Err(failure("cubin image/module-set admission owner differs"));
        }
        Ok(())
    }
    pub(crate) fn image_key(&self) -> CubinImageKey {
        CubinImageKey {
            issuer: Arc::as_ptr(&self.sources) as usize,
            sha256: self.sha256,
            bytes: self.bytes.len(),
            binary_sm: self.binary_sm,
        }
    }
    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    pub fn binary_sm(&self) -> u8 {
        self.binary_sm
    }
    pub fn derived_from(&self) -> Option<PinnedBf16Cubin> {
        self.derived_from
    }
    pub fn captured_from(&self) -> Option<PinnedCapturedBf16Cubin> {
        self.captured_from
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    struct AdmissionDrop(Arc<AtomicUsize>);
    impl Drop for AdmissionDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn image_charge_keeps_original_admission_until_last_owner_drops() {
        let drops = Arc::new(AtomicUsize::new(0));
        let admission = Arc::new(AdmissionDrop(drops.clone()));
        let weak = Arc::downgrade(&admission);
        // Same guard sealed inside VerifiedCubin; no fake source/queue/lease.
        let image_guard = Arc::new(ImageCharge::reserve(1, admission.clone()).unwrap());
        let actor_owner = image_guard.clone();
        let public_owner = image_guard.clone();
        drop((admission, image_guard, public_owner));
        assert!(weak.upgrade().is_some());
        assert_eq!(drops.load(Ordering::Acquire), 0);
        drop(actor_owner);
        assert!(weak.upgrade().is_none());
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }

    #[test]
    fn refused_image_does_not_retain_original_admission() {
        let drops = Arc::new(AtomicUsize::new(0));
        let admission = Arc::new(AdmissionDrop(drops.clone()));
        let weak = Arc::downgrade(&admission);
        assert!(ImageCharge::reserve(IMAGE_LIMIT + 1, admission).is_err());
        assert!(weak.upgrade().is_none());
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }
    #[test]
    fn owned_image_charges_spare_capacity_without_copy_and_retains_admission() {
        let drops = Arc::new(AtomicUsize::new(0));
        let admission = Arc::new(AdmissionDrop(drops.clone()));
        let weak = Arc::downgrade(&admission);
        let mut image = Vec::with_capacity(1000);
        image.extend_from_slice(&[1u8; 8]);
        let pointer = image.as_ptr();
        let capacity = image.capacity();
        let image = charge_owned_image(image, admission.clone()).unwrap();
        drop(admission);
        assert_eq!(image.image.as_ptr(), pointer);
        assert_eq!(image.charge.bytes, capacity);
        assert!(capacity > image.image.len());
        assert!(weak.upgrade().is_some());
        drop(image);
        assert!(weak.upgrade().is_none());
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }
    #[test]
    fn oversized_owned_capacity_refuses_even_with_empty_length() {
        let drops = Arc::new(AtomicUsize::new(0));
        let admission = Arc::new(AdmissionDrop(drops.clone()));
        let weak = Arc::downgrade(&admission);
        let image = Vec::<u8>::with_capacity(IMAGE_LIMIT + 1);
        assert!(image.is_empty());
        assert!(charge_owned_image(image, admission).is_err());
        assert!(weak.upgrade().is_none());
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }

    #[test]
    fn payload_drops_before_original_admission_on_reserve_and_validation_refusal() {
        use std::sync::{Weak, atomic::AtomicBool};
        struct Admission {
            payload_dropped: Arc<AtomicBool>,
        }
        impl Drop for Admission {
            fn drop(&mut self) {
                assert!(self.payload_dropped.load(Ordering::Acquire));
            }
        }
        struct Payload {
            dropped: Arc<AtomicBool>,
            owner: Weak<Admission>,
        }
        impl Drop for Payload {
            fn drop(&mut self) {
                assert!(self.owner.upgrade().is_some());
                self.dropped.store(true, Ordering::Release);
            }
        }
        fn refuse_after_charge(_image: ChargedImage<Payload>) -> Result<(), ServerError> {
            Err(failure("post-admission image validation refusal"))
        }
        for bytes in [0, IMAGE_LIMIT + 1, 1] {
            let dropped = Arc::new(AtomicBool::new(false));
            let admission = Arc::new(Admission {
                payload_dropped: dropped.clone(),
            });
            let weak = Arc::downgrade(&admission);
            let payload = Payload {
                dropped: dropped.clone(),
                owner: weak.clone(),
            };
            let charged = charge_owned_payload(payload, bytes, admission);
            if bytes == 1 {
                assert!(refuse_after_charge(charged.unwrap()).is_err());
            } else {
                assert!(charged.is_err());
            }
            assert!(dropped.load(Ordering::Acquire));
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn captured_seal_refusal_drops_payload_before_original_admission() {
        use std::sync::{Weak, atomic::AtomicBool};
        struct Admission(Arc<AtomicBool>);
        impl Drop for Admission {
            fn drop(&mut self) {
                assert!(self.0.load(Ordering::Acquire));
            }
        }
        struct Payload {
            bytes: [u8; 64],
            dropped: Arc<AtomicBool>,
            owner: Weak<Admission>,
        }
        impl AsRef<[u8]> for Payload {
            fn as_ref(&self) -> &[u8] {
                &self.bytes
            }
        }
        impl Drop for Payload {
            fn drop(&mut self) {
                assert!(self.owner.upgrade().is_some());
                self.dropped.store(true, Ordering::Release);
            }
        }
        for kind in [
            PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80,
            PinnedCapturedBf16Cubin::AttentionSigmoidSm86,
            PinnedCapturedBf16Cubin::AttentionMultiplySm86,
        ] {
            let dropped = Arc::new(AtomicBool::new(false));
            let admission = Arc::new(Admission(dropped.clone()));
            let weak = Arc::downgrade(&admission);
            let payload = Payload {
                bytes: [0; 64],
                dropped: dropped.clone(),
                owner: weak.clone(),
            };
            let charged = charge_owned_payload(payload, 64, admission).unwrap();
            assert!(seal_captured_payload(charged, kind).is_err());
            assert!(dropped.load(Ordering::Acquire));
            assert!(weak.upgrade().is_none());
        }
    }
}
