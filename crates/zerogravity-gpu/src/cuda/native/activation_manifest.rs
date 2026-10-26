//! Exact five-call original BF16 activation ABI; one static32B template.
//! Captured selected-resource provenance only; no public loader-input equality.
use super::cubin_manifests::ProjectionBinding;
use crate::cuda::gdn::GdnPhase;
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
use zerogravity_core::{Error, Result};

pub(super) const IMAGE_BYTES: usize = 901_664;
pub(super) const IMAGE_SHA256: [u8; 32] = [
    0x7e, 0xc9, 0x41, 0x42, 0x83, 0xb0, 0x99, 0x42, 0xde, 0x90, 0x01, 0x1e, 0xb5, 0xbd, 0xba, 0xeb,
    0x63, 0xb3, 0x52, 0xcd, 0x0d, 0xc5, 0x00, 0xf2, 0xc7, 0xba, 0x38, 0x54, 0xf6, 0x60, 0xfa, 0xdf,
];
pub(super) const IMAGE_HEADER: [u8; 64] = [
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0xbe, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0xc1, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x56, 0x05, 0x56, 0x00, 0x40, 0x00, 0x38, 0x00, 0x04, 0x00, 0x40, 0x00, 0xa9, 0x02, 0x01, 0x00,
];
const ENTRY: &str = "_ZN4vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_bfloat162XadL_ZNS_11silu_kernelIS2_EET_RKS5_fEEXadL_ZNS_18packed_silu_kernelIS3_EES5_S7_fEELb1ELb1ELb0ELb0EEEvPS5_PS6_ifff";
// C0/A8 explicit pointer fields; remaining original scalar bits unchanged.
const TEMPLATE: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x0e, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0x80,
    0x3f, 0, 0, 0, 0,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Activation {
    Rows1,
    Rows64,
    Rows2,
}

impl Activation {
    pub(super) fn for_phase(phase: GdnPhase) -> Result<Self> {
        Ok(match phase {
            GdnPhase::InitialOne | GdnPhase::CachedDecode => Self::Rows1,
            GdnPhase::Initial64 => Self::Rows64,
            GdnPhase::ContinuedTwo => Self::Rows2,
            GdnPhase::FixedPrompt | GdnPhase::TextPrefill(_) => {
                return Err(Error::Unsupported(
                    "original opaque manifest does not admit47 rows".into(),
                ));
            }
        })
    }
    pub(super) const fn rows(self) -> usize {
        match self {
            Self::Rows1 => 1,
            Self::Rows64 => 64,
            Self::Rows2 => 2,
        }
    }
    pub(super) const fn input_bytes(self) -> usize {
        self.rows() * 14_336
    }
    pub(super) const fn output_bytes(self) -> usize {
        self.rows() * 7_168
    }
    pub(super) const fn bindings(self) -> &'static [ProjectionBinding] {
        &[ProjectionBinding::X, ProjectionBinding::Y]
    }
    pub(super) fn manifest(self) -> CubinKernelManifest {
        manifest_rows(self.rows())
    }
}

/// Separately captured exact47 ABI; legacy phase admission remains unchanged.
pub(super) fn fixed_prompt_manifest() -> CubinKernelManifest {
    manifest_rows(47)
}
fn manifest_rows(rows: usize) -> CubinKernelManifest {
    CubinKernelManifest {
        entry: ENTRY.into(),
        binary_sm: 86,
        grid: [rows as u32, 1, 1],
        block: [448, 1, 1],
        dynamic_shared_bytes: 0,
        attributes: CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding {
                bytes: (rows * 14_336) as u64,
                alignment: 16,
                io: CubinIo::Read,
            },
            CubinBinding {
                bytes: (rows * 7_168) as u64,
                alignment: 16,
                io: CubinIo::Write,
            },
        ],
        parameters: [(0, 8), (8, 8), (16, 4), (20, 4), (24, 4), (28, 4)]
            .into_iter()
            .map(|(offset, size)| CubinParameter {
                offset,
                template: TEMPLATE[offset..offset + size].to_vec(),
            })
            .collect(),
        fixups: vec![
            CubinPointerFixup {
                parameter: 0,
                byte_offset: 0,
                binding: 1,
                binding_offset: 0,
                extent: (rows * 7_168) as u64,
                alignment: 16,
                io: CubinIo::Write,
            },
            CubinPointerFixup {
                parameter: 1,
                byte_offset: 0,
                binding: 0,
                binding_offset: 0,
                extent: (rows * 14_336) as u64,
                alignment: 16,
                io: CubinIo::Read,
            },
        ],
    }
}

#[cfg(test)]
#[path = "activation_manifest/tests.rs"]
mod tests;
