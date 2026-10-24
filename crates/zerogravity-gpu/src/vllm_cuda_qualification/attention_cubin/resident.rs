//! Fixed original-actor leaf programs; caller owns traversal and completion.
use super::super::projection_cubin::resident::{NativeRetentionWitness, PreparedImages};
use super::super::projection_cubin::{native_error, not_cancelled};
use super::descriptor::AttentionBindings;
use super::manifests::{self, AttentionCase, AttentionRole, AttentionStage};
use crate::{physical_quarantine::CompletionOwner, resident_leases::ResidentInitialization};
use cubecl::cuda::OwnedCudaKernel;
use std::{
    collections::BTreeMap,
    sync::{Arc, Weak, atomic::AtomicBool},
};
use zerogravity_core::{Error, Result};

struct Launch {
    kernel: Arc<OwnedCudaKernel>,
    roles: &'static [AttentionRole],
}
pub(super) struct AttentionPrograms {
    programs: BTreeMap<(AttentionCase, AttentionStage), Vec<Launch>>,
    bindings: Vec<Arc<AttentionBindings>>,
    images: Arc<PreparedImages>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AttentionSubmitted {
    pub(super) native_launches: usize,
    pub(super) device_copies: usize,
}
mod model;
pub(crate) use model::ModelAttentionPrograms;

impl AttentionPrograms {
    /// Every managed root/value/cache/control/readback allocation already exists.
    /// No device allocations, host fixture injection or independent engine here.
    pub(super) fn prepare(
        images: Arc<PreparedImages>,
        bindings: Vec<Arc<AttentionBindings>>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let cases = bindings
            .iter()
            .map(|b| b.descriptor().case())
            .collect::<Vec<_>>();
        manifests::require_closed_cases(&cases)?;
        Self::prepare_bindings(images, bindings, init, cancelled)
    }
    fn prepare_bindings(
        images: Arc<PreparedImages>,
        bindings: Vec<Arc<AttentionBindings>>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let cases = bindings
            .iter()
            .map(|binding| binding.descriptor().case())
            .collect::<Vec<_>>();
        let first = bindings
            .first()
            .ok_or_else(|| Error::Malformed("attention five original cases".into()))?;
        init.require_service(first.service())?;
        images.require_service(first.service())?;
        for b in &bindings {
            if !Arc::ptr_eq(b.service(), first.service())
                || b.descriptor().source() != first.descriptor().source()
                || b.descriptor().scope() != first.descriptor().scope()
            {
                return Err(Error::State(
                    "attention original cohort service/source differs".into(),
                ));
            }
            init.retain_completion_owner(b.clone())?;
        }
        // Retain before first native load and through every partial prepare.
        init.retain_completion_owner(images.clone())?;
        let mut prepared = BTreeMap::<(AttentionCase, AttentionStage), Vec<Launch>>::new();
        let mut reused = BTreeMap::<Vec<u8>, Arc<OwnedCudaKernel>>::new();
        for &case in &cases {
            for &stage in first.descriptor().scope().stages() {
                let mut launches = Vec::new();
                for launch in manifests::launches(case, stage) {
                    not_cancelled(cancelled)?;
                    let image = images.image(launch.image)?;
                    if image.sha256() != launch.image.sha256()
                        || image.byte_len() != launch.image.bytes()
                        || image.binary_sm() != launch.image.binary_sm()
                    {
                        return Err(Error::State(
                            "attention exact immutable source image".into(),
                        ));
                    }
                    // Exact canonical ABI identity includes image. Do not use an
                    // entry-only cache: FA phase structs and launch grids differ.
                    let identity = super::manifest_identity(&launch.manifest, image.sha256());
                    let kernel = if let Some(existing) = reused.get(&identity) {
                        existing.clone()
                    } else {
                        // SAFETY: closed complete source-derived ABI, same admitted
                        // image/owner/set; original real initialization is armed.
                        let kernel = unsafe { images.set().prepare(image, launch.manifest) }
                            .map_err(native_error)?;
                        init.retain_completion_owner(kernel.clone())?;
                        reused.insert(identity, kernel.clone());
                        kernel
                    };
                    launches.push(Launch {
                        kernel,
                        roles: launch.roles,
                    });
                }
                prepared.insert((case, stage), launches);
            }
        }
        if reused.len() != first.descriptor().scope().records() {
            return Err(Error::State(
                "attention exact scoped source manifest records".into(),
            ));
        }
        let result = Arc::new(Self {
            programs: prepared,
            bindings,
            images,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn bindings(&self, case: AttentionCase) -> Result<&Arc<AttentionBindings>> {
        self.bindings
            .iter()
            .find(|b| b.descriptor().case() == case)
            .ok_or_else(|| Error::State("attention case not prepared".into()))
    }
    pub(super) fn completion_owner(self: &Arc<Self>) -> CompletionOwner {
        self.clone()
    }
    /// # Safety
    /// SAME original armed flight retains self/bindings/images and the complete
    /// accesses() union before first enqueue, including both N1 copy leases.
    /// Caller never substitutes this operation for an unqualified model node.
    pub(super) unsafe fn enqueue(
        &self,
        case: AttentionCase,
        stage: AttentionStage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AttentionSubmitted> {
        unsafe { self.enqueue_inner(case, stage, cancelled, None) }
    }
    pub(super) unsafe fn enqueue_cancel_before_copy(
        &self,
        case: AttentionCase,
        flag: &AtomicBool,
    ) -> Result<AttentionSubmitted> {
        if case != AttentionCase::InitialOne || flag.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Error::State(
                "attention closed fresh N1 cancellation cut".into(),
            ));
        }
        unsafe { self.enqueue_inner(case, AttentionStage::Fa, Some(flag), Some(flag)) }
    }
    unsafe fn enqueue_inner(
        &self,
        case: AttentionCase,
        stage: AttentionStage,
        cancelled: Option<&AtomicBool>,
        cancel_before_copy: Option<&AtomicBool>,
    ) -> Result<AttentionSubmitted> {
        not_cancelled(cancelled)?;
        let bindings = self.bindings(case)?;
        unsafe { self.enqueue_bindings(bindings, case, stage, cancelled, cancel_before_copy) }
    }
    unsafe fn enqueue_bindings(
        &self,
        bindings: &Arc<AttentionBindings>,
        case: AttentionCase,
        stage: AttentionStage,
        cancelled: Option<&AtomicBool>,
        cancel_before_copy: Option<&AtomicBool>,
    ) -> Result<AttentionSubmitted> {
        not_cancelled(cancelled)?;
        if bindings.descriptor().case() != case
            || !self
                .bindings
                .iter()
                .any(|prepared| Arc::ptr_eq(prepared, bindings))
        {
            return Err(Error::State(
                "attention original prepared binding identity/case".into(),
            ));
        }
        bindings.service().ensure_usable()?;
        self.images.require_service(bindings.service())?;
        let launches = self
            .programs
            .get(&(case, stage))
            .ok_or_else(|| Error::State("attention stage not prepared".into()))?;
        let mut submitted = AttentionSubmitted::default();
        for launch in launches {
            not_cancelled(cancelled)?;
            let handles = launch
                .roles
                .iter()
                .map(|&role| Ok(bindings.lease(role)?.handle()))
                .collect::<Result<Vec<_>>>()?;
            // SAFETY: same original armed producer flight and sealed bindings.
            unsafe { launch.kernel.enqueue(handles) }.map_err(native_error)?;
            submitted.native_launches += 1;
        }
        if stage == AttentionStage::Fa && case.rows() == 1 {
            if let Some(flag) = cancel_before_copy {
                flag.store(true, std::sync::atomic::Ordering::Release);
            }
            not_cancelled(cancelled)?;
            let last = launches
                .last()
                .ok_or_else(|| Error::State("attention original FA record absent".into()))?;
            // SAFETY: both distinct original4096B leases already occur in this
            // SAME flight; the actor validates service/context/stream/ranges.
            unsafe {
                last.kernel.copy_first_attention_output(
                    bindings.lease(AttentionRole::PrivateO)?.handle(),
                    bindings.lease(AttentionRole::AttentionOutput)?.handle(),
                )
            }
            .map_err(native_error)?;
            submitted.device_copies += 1;
        }
        Ok(submitted)
    }
    pub(super) fn retention_witness(self: &Arc<Self>) -> Result<AttentionRetentionWitness> {
        let kernels = self.unique_kernels();
        Ok(AttentionRetentionWitness {
            prepared: Arc::downgrade(self),
            bindings: self.bindings.iter().map(Arc::downgrade).collect(),
            native: self.images.retention_witness(&kernels)?,
        })
    }
    fn unique_kernels(&self) -> Vec<Arc<OwnedCudaKernel>> {
        let mut result = Vec::new();
        for launch in self.programs.values().flatten() {
            if !result.iter().any(|k| Arc::ptr_eq(k, &launch.kernel)) {
                result.push(launch.kernel.clone());
            }
        }
        result
    }
    /// # Safety
    /// Trusted physical completion for all calls; original initialized-W
    /// read-only flight retains this complete graph through record retirement.
    pub(super) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        for kernel in self.unique_kernels() {
            unsafe { kernel.retire_after_physical_completion() }?;
        }
        Ok(())
    }
    /// Standalone leaf only. Any unload error follows original quarantine.
    pub(super) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        unsafe { self.retire_kernels_after_physical_completion() }?;
        unsafe { self.images.set().retire_after_physical_completion() }
    }
}
pub(super) struct AttentionRetentionWitness {
    prepared: Weak<AttentionPrograms>,
    bindings: Vec<Weak<AttentionBindings>>,
    native: NativeRetentionWitness,
}
impl AttentionRetentionWitness {
    pub(super) fn inspect(
        &self,
    ) -> Result<super::super::projection_cubin::resident::NativeRetentionState> {
        if self.prepared.upgrade().is_none() || self.bindings.iter().any(|b| b.upgrade().is_none())
        {
            return Err(Error::State(
                "attention original armed quarantine lost actual graph".into(),
            ));
        }
        self.native.inspect()
    }
}
