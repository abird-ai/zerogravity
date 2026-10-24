//! Fixed source derivations, not an arbitrary extracted-code issuer.
//! ROOT's pinned cuobjdump outputs were byte-equal to selected CUPTI images.
use super::*;
use crate::owned_command::cubin::{CubinProvider, VerifiedCubin};
use std::any::Any;
use std::os::unix::fs::FileExt;

const TOOL: &str = "8f240516056740fee62c40eec6e1ba5d3d5ab5a309fbba4b0fe26fd54c79767d";
const LT_BYTES: u64 = 749_205_904;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinnedBf16Cubin {
    AmpereBf16Gemm128x64Sm80,
    AmpereBf16Gemm64x64Stages64x5Sm80,
    Bf16SplitKReduceSm86,
    Bf16Gemv2TSm86,
    Bf16GemvxSm86,
}

/// Immutable evidence of an already executed derivation; runtime does not run
/// the extractor. Public copies cannot alter the constructor's closed registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedBf16CubinDerivation {
    pub provider: CubinProvider,
    pub provider_sha256: &'static str,
    pub companion_cublas_sha256: &'static str,
    pub carrier_offset: u64,
    pub carrier_bytes: usize,
    pub carrier_sha256: [u8; 32],
    pub wrapper_index: u32,
    pub extractor_sha256: &'static str,
    pub extract_selector: &'static str,
    pub image_bytes: usize,
    pub image_sha256: [u8; 32],
    pub binary_sm: u8,
}
const fn digest_literal(text: &str) -> [u8; 32] {
    const fn nibble(b: u8) -> u8 {
        match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            _ => panic!("invalid pinned digest"),
        }
    }
    assert!(text.len() == 64);
    let bytes = text.as_bytes();
    let mut out = [0; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = (nibble(bytes[i * 2]) << 4) | nibble(bytes[i * 2 + 1]);
        i += 1;
    }
    out
}
const fn record(
    offset: u64,
    carrier_bytes: usize,
    carrier_sha: &str,
    wrapper: u32,
    image_bytes: usize,
    image_sha: &str,
    sm: u8,
) -> PinnedBf16CubinDerivation {
    PinnedBf16CubinDerivation {
        provider: CubinProvider::CublasLt,
        provider_sha256: LT_HASH,
        companion_cublas_sha256: CUBLAS_HASH,
        carrier_offset: offset,
        carrier_bytes,
        carrier_sha256: digest_literal(carrier_sha),
        wrapper_index: wrapper,
        extractor_sha256: TOOL,
        extract_selector: if sm == 80 {
            "sm_80.cubin"
        } else {
            "sm_86.cubin"
        },
        image_bytes,
        image_sha256: digest_literal(image_sha),
        binary_sm: sm,
    }
}
impl PinnedBf16Cubin {
    pub const fn derivation(self) -> PinnedBf16CubinDerivation {
        match self {
            Self::AmpereBf16Gemm128x64Sm80 => record(
                453_613_968,
                431_104,
                "bc91990604aa63b9c1618551c3e5243faa9cf7cae352fc42bb7f3193272b5d8b",
                3008,
                6_601_808,
                "c1ca8db71001178d04f1b83d7c55883107fa122927e10d41020672678fff2c86",
                80,
            ),
            Self::AmpereBf16Gemm64x64Stages64x5Sm80 => record(
                454_045_072,
                397_608,
                "130ea4555de275883de15cf884a14ae9520b688fedb7b699779dc5fafe815cae",
                3009,
                6_422_224,
                "c3361a22db27ce017fec90a46f66c508cad5d668d606070b214b0351b3cf9588",
                80,
            ),
            Self::Bf16SplitKReduceSm86 => record(
                190_680_584,
                8_553_608,
                "aa9ee7599b38541490418667506ae0264e212a587d4798dc0be32209b14f1845",
                31,
                4_252_960,
                "074807d671d9156ccc56b5a8cbe82f1ba8fae21c8169e674b31b4b8603ba42ec",
                86,
            ),
            Self::Bf16Gemv2TSm86 => record(
                243_887_392,
                4_382_760,
                "a8fbe1d9bd524db6412e42b26d8de2bc50e0787ed032de99cde4e913203145c7",
                55,
                2_818_208,
                "94fc9c63d0562b88395c3d5001397aa1a3a3f572450a00db627750b817c7de56",
                86,
            ),
            Self::Bf16GemvxSm86 => record(
                300_747_584,
                8_872_920,
                "0048dd26b8add68bdc772d2073af3bff80c15598dbc030b56fb8a1652c5b4d37",
                84,
                5_273_248,
                "02cd5eee5a1f93addca2135cde7dfd71b05e499547e227a43806ee953eb2fa99",
                86,
            ),
        }
    }
    pub(crate) fn require_image(
        self,
        bytes: usize,
        digest: [u8; 32],
        sm: u8,
    ) -> Result<(), ServerError> {
        let expected = self.derivation();
        if bytes != expected.image_bytes
            || digest != expected.image_sha256
            || sm != expected.binary_sm
        {
            return Err(failure(
                "derived cubin differs from closed selected source registry",
            ));
        }
        Ok(())
    }
}

fn require_carrier(file: &PinnedFile, entry: PinnedBf16CubinDerivation) -> Result<(), ServerError> {
    file.unchanged()?;
    if file.stamp[2] != LT_BYTES
        || entry.carrier_bytes == 0
        || entry.carrier_bytes > 32 * 1024 * 1024
        || entry
            .carrier_offset
            .checked_add(entry.carrier_bytes as u64)
            .is_none_or(|end| end > file.stamp[2])
    {
        return Err(failure("derived cubin pinned LT carrier extent"));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut read = 0usize;
    while read < entry.carrier_bytes {
        let want = (entry.carrier_bytes - read).min(buffer.len());
        let n = file
            .file
            .read_at(&mut buffer[..want], entry.carrier_offset + read as u64)
            .map_err(|e| failure(format!("derived cubin retained LT carrier read: {e}")))?;
        if n == 0 {
            return Err(failure("derived cubin retained LT carrier EOF"));
        }
        digest.update(&buffer[..n]);
        read += n;
    }
    file.unchanged()?;
    let actual: [u8; 32] = digest.finalize().into();
    if actual != entry.carrier_sha256 {
        return Err(failure("derived cubin retained LT carrier SHA differs"));
    }
    Ok(())
}
impl Bf16BlasSources {
    /// Five proven source derivations only; no caller-chosen expected hash or
    /// missing CUTLASS/module24 fallback. Does not establish inference support.
    ///
    /// # Safety
    /// Caller admitted the input Vec's FULL capacity before allocation. The
    /// exact owner retains original host-image and same-service native-module
    /// admission until real unload/quarantine. Actual source extraction used
    /// the pinned tool/selector; constructor independently matches the sealed
    /// registry and rehashes original carrier before/after sealing, no copy.
    pub unsafe fn derived_cubin(
        self: &Arc<Self>,
        admission_owner: Arc<dyn Any + Send + Sync>,
        kind: PinnedBf16Cubin,
        image: Vec<u8>,
    ) -> Result<Arc<VerifiedCubin>, ServerError> {
        self.require_queue(&self.queue)?;
        let entry = kind.derivation();
        require_carrier(&self.lt, entry)?;
        let sealed = VerifiedCubin::seal_derived(self.clone(), admission_owner, kind, image)?;
        require_carrier(&self.lt, entry)?;
        self.require_queue(&self.queue)?;
        Ok(sealed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const ALL: [PinnedBf16Cubin; 5] = [
        PinnedBf16Cubin::AmpereBf16Gemm128x64Sm80,
        PinnedBf16Cubin::AmpereBf16Gemm64x64Stages64x5Sm80,
        PinnedBf16Cubin::Bf16SplitKReduceSm86,
        PinnedBf16Cubin::Bf16Gemv2TSm86,
        PinnedBf16Cubin::Bf16GemvxSm86,
    ];
    #[test]
    fn closed_derivations_bind_unique_images_to_bounded_original_lt_carriers() {
        let mut hashes = std::collections::HashSet::new();
        let mut intervals = Vec::new();
        for kind in ALL {
            let d = kind.derivation();
            assert_eq!(d.provider, CubinProvider::CublasLt);
            assert_eq!(d.provider_sha256, LT_HASH);
            assert_eq!(d.companion_cublas_sha256, CUBLAS_HASH);
            assert_eq!(d.extractor_sha256, TOOL);
            assert!(hashes.insert(d.image_sha256));
            assert!((1..=32 * 1024 * 1024).contains(&d.image_bytes));
            let end = d
                .carrier_offset
                .checked_add(d.carrier_bytes as u64)
                .unwrap();
            assert!(end <= LT_BYTES);
            assert!(
                !intervals
                    .iter()
                    .any(|&(start, stop)| d.carrier_offset < stop && start < end)
            );
            intervals.push((d.carrier_offset, end));
            assert_eq!(
                d.extract_selector,
                if d.binary_sm == 80 {
                    "sm_80.cubin"
                } else {
                    "sm_86.cubin"
                }
            );
        }
    }
    #[test]
    fn registry_refuses_every_digest_bit_change_cross_variant_and_extent_or_sm_change() {
        for kind in ALL {
            let d = kind.derivation();
            kind.require_image(d.image_bytes, d.image_sha256, d.binary_sm)
                .unwrap();
            for bit in 0..256 {
                let mut digest = d.image_sha256;
                digest[bit / 8] ^= 1 << (bit % 8);
                assert!(
                    kind.require_image(d.image_bytes, digest, d.binary_sm)
                        .is_err()
                );
            }
            assert!(
                kind.require_image(d.image_bytes - 1, d.image_sha256, d.binary_sm)
                    .is_err()
            );
            assert!(
                kind.require_image(
                    d.image_bytes,
                    d.image_sha256,
                    if d.binary_sm == 80 { 86 } else { 80 }
                )
                .is_err()
            );
            for other in ALL {
                if other != kind {
                    let other = other.derivation();
                    assert!(
                        kind.require_image(other.image_bytes, other.image_sha256, other.binary_sm)
                            .is_err()
                    );
                }
            }
        }
    }
}
