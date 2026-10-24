//! Exact original MLP13 kernel ABIs plus two ordered workspace clears.
//! ROOT executed root-closed-mlp-manifests-1; seven unique static templates.
//! Captured data SHAbe5e3b47a93b440cdfb7d4b589045a62fe00da0e7fb2ba47c0b2683f77b50e10.
//! Assets alone grant no numerical/profile or model eligibility.
use super::cubin_manifests::{ProjectionBinding, ProjectionImage, WORKSPACE_BYTES};
use crate::vllm_cuda_gdn::VllmGdnPhase;
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum MlpRole {
    GateUp,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MlpProjection {
    GateUp1,
    Down1,
    GateUp64,
    Down64,
    GateUp2,
    Down2,
}

pub(super) struct MlpLaunch {
    image: ProjectionImage,
    bindings: &'static [ProjectionBinding],
    manifest: CubinKernelManifest,
}
impl MlpLaunch {
    pub(super) fn image(&self) -> ProjectionImage {
        self.image
    }
    pub(super) fn bindings(&self) -> &'static [ProjectionBinding] {
        self.bindings
    }
    pub(super) fn into_manifest(self) -> CubinKernelManifest {
        self.manifest
    }
}

/// Native cuMemsetD8Async prerequisite; not an invented kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MlpWorkspaceClear {
    bytes: usize,
}
impl MlpWorkspaceClear {
    pub(super) const fn binding(self) -> ProjectionBinding {
        ProjectionBinding::Workspace
    }
    pub(super) const fn binding_bytes(self) -> usize {
        WORKSPACE_BYTES
    }
    pub(super) const fn byte_offset(self) -> usize {
        0
    }
    pub(super) const fn bytes(self) -> usize {
        self.bytes
    }
    pub(super) const fn value(self) -> u8 {
        0
    }
    pub(super) const fn io(self) -> CubinIo {
        CubinIo::Write
    }
}

pub(super) enum MlpOperation {
    Clear(MlpWorkspaceClear),
    Kernel(MlpLaunch),
}

const fn literal<const N: usize>(text: &str) -> [u8; N] {
    const fn nibble(value: u8) -> u8 {
        match value {
            b'0'..=b'9' => value - b'0',
            b'a'..=b'f' => value - b'a' + 10,
            _ => panic!("invalid pinned MLP byte literal"),
        }
    }
    assert!(text.len() == N * 2);
    let input = text.as_bytes();
    let mut out = [0; N];
    let mut i = 0;
    while i < N {
        out[i] = nibble(input[i * 2]) << 4 | nibble(input[i * 2 + 1]);
        i += 1;
    }
    out
}

// Actual launch-00-scope-00-gate_up.json; template SHA5f2a15a3146a6ab6aaf6d31278e18c35b6d4c1d9df9ceee56de44441f1030d1e.
const TEMPLATE_0: [u8; 160] = literal(
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040000001c00000004000001000000010000000100000001000000000000000000000000000000000000000000000000000000000000000000803f000000000000000000000000000000000000000000000000000000000000803f00000000",
);
fn launch_0() -> MlpLaunch {
    MlpLaunch { image: ProjectionImage::Bf16Gemv2TSm86, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y],
        manifest: CubinKernelManifest {
            entry: "_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_".into(), binary_sm: ProjectionImage::Bf16Gemv2TSm86.binary_sm(),
            grid: [896, 1, 1], block: [128, 1, 1], dynamic_shared_bytes: 0, attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding { bytes: 14680064, alignment: 16, io: CubinIo::Read },
                CubinBinding { bytes: 2048, alignment: 16, io: CubinIo::Read },
                CubinBinding { bytes: 14336, alignment: 16, io: CubinIo::Write },
            ],
            parameters: vec![
                CubinParameter { offset: 0, template: TEMPLATE_0[0..152].to_vec() },
                CubinParameter { offset: 152, template: TEMPLATE_0[152..156].to_vec() },
                CubinParameter { offset: 156, template: TEMPLATE_0[156..160].to_vec() },
            ],
            fixups: vec![
                CubinPointerFixup { parameter: 0, byte_offset: 0, binding: 0, binding_offset: 0, extent: 14680064, alignment: 16, io: CubinIo::Read },
                CubinPointerFixup { parameter: 0, byte_offset: 16, binding: 1, binding_offset: 0, extent: 2048, alignment: 16, io: CubinIo::Read },
                CubinPointerFixup { parameter: 0, byte_offset: 32, binding: 2, binding_offset: 0, extent: 14336, alignment: 16, io: CubinIo::Write },
                CubinPointerFixup { parameter: 0, byte_offset: 48, binding: 2, binding_offset: 0, extent: 14336, alignment: 16, io: CubinIo::Write },
            ],
        },
    }
}

// Actual launch-01-scope-01-down.json; template SHA85d4f7cfe6c411d8bdf06274cbb49454541d4bbb96b69b25c9349d85259b4c1c.
// Only unread padding260..263 zeroed; scalar256=0xffffffff retained.
const TEMPLATE_1: [u8; 360] = literal(
    "0100000000040000000e0000010000004000000003000000030000000100000003000000c004000000040000000000000000000000000000000e00000000000000380000000000000079feffffffffff0001000000000000000e00000000000000380000000000000079feffffffffff00010000000000000008000000000000004000000000000000c8ffffffffffff00c8ffffffffffff00800000000000000000000000000000008000000000000000800000000000000008000000000000004000000000000000c8ffffffffffff00c8ffffffffffff00800000000000000000000000000000008000000000000000800000000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_1() -> MlpLaunch {
    MlpLaunch { image: ProjectionImage::CutlassWmmaSm80, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y,ProjectionBinding::Workspace],
        manifest: CubinKernelManifest {
            entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
            grid: [8, 8, 3], block: [32, 1, 1], dynamic_shared_bytes: 16896, attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding { bytes: 7340032, alignment: 16, io: CubinIo::Read },
                CubinBinding { bytes: 7168, alignment: 16, io: CubinIo::Read },
                CubinBinding { bytes: 2048, alignment: 16, io: CubinIo::Write },
                CubinBinding { bytes: 8519680, alignment: 16, io: CubinIo::Write },
            ],
            parameters: vec![
                CubinParameter { offset: 0, template: TEMPLATE_1[0..360].to_vec() },
            ],
            fixups: vec![
                CubinPointerFixup { parameter: 0, byte_offset: 280, binding: 1, binding_offset: 0, extent: 7168, alignment: 16, io: CubinIo::Read },
                CubinPointerFixup { parameter: 0, byte_offset: 288, binding: 0, binding_offset: 0, extent: 7340032, alignment: 16, io: CubinIo::Read },
                CubinPointerFixup { parameter: 0, byte_offset: 296, binding: 2, binding_offset: 0, extent: 2048, alignment: 16, io: CubinIo::Write },
                CubinPointerFixup { parameter: 0, byte_offset: 304, binding: 3, binding_offset: 0, extent: 6144, alignment: 16, io: CubinIo::Write },
            ],
        },
    }
}

// Actual launch-02-scope-01-down.json; template SHA2021f6308e1236fbd351cdf8ba655422e9c181da3c4e649029ed7e1317fee194.
// Only inactive host-scalar pointers96..111 zeroed under scalar-mode44=0.
const TEMPLATE_2: [u8; 224] = literal(
    "0000803f00000000000400000100000000040000000400000004000000040000010000000004000003000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_2() -> MlpLaunch {
    MlpLaunch { image: ProjectionImage::Bf16SplitKReduceSm86, bindings: &[ProjectionBinding::Y,ProjectionBinding::Workspace],
        manifest: CubinKernelManifest {
            entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(), binary_sm: ProjectionImage::Bf16SplitKReduceSm86.binary_sm(),
            grid: [32, 1, 1], block: [32, 16, 1], dynamic_shared_bytes: 0, attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding { bytes: 2048, alignment: 16, io: CubinIo::Write },
                CubinBinding { bytes: 8519680, alignment: 16, io: CubinIo::Read },
            ],
            parameters: vec![
                CubinParameter { offset: 0, template: TEMPLATE_2[0..60].to_vec() },
                CubinParameter { offset: 64, template: TEMPLATE_2[64..72].to_vec() },
                CubinParameter { offset: 72, template: TEMPLATE_2[72..80].to_vec() },
                CubinParameter { offset: 80, template: TEMPLATE_2[80..88].to_vec() },
                CubinParameter { offset: 88, template: TEMPLATE_2[88..96].to_vec() },
                CubinParameter { offset: 96, template: TEMPLATE_2[96..104].to_vec() },
                CubinParameter { offset: 104, template: TEMPLATE_2[104..112].to_vec() },
                CubinParameter { offset: 112, template: TEMPLATE_2[112..120].to_vec() },
                CubinParameter { offset: 120, template: TEMPLATE_2[120..128].to_vec() },
                CubinParameter { offset: 128, template: TEMPLATE_2[128..136].to_vec() },
                CubinParameter { offset: 136, template: TEMPLATE_2[136..144].to_vec() },
                CubinParameter { offset: 144, template: TEMPLATE_2[144..152].to_vec() },
                CubinParameter { offset: 152, template: TEMPLATE_2[152..160].to_vec() },
                CubinParameter { offset: 160, template: TEMPLATE_2[160..168].to_vec() },
                CubinParameter { offset: 168, template: TEMPLATE_2[168..176].to_vec() },
                CubinParameter { offset: 176, template: TEMPLATE_2[176..184].to_vec() },
                CubinParameter { offset: 184, template: TEMPLATE_2[184..192].to_vec() },
                CubinParameter { offset: 192, template: TEMPLATE_2[192..200].to_vec() },
                CubinParameter { offset: 200, template: TEMPLATE_2[200..208].to_vec() },
                CubinParameter { offset: 208, template: TEMPLATE_2[208..216].to_vec() },
                CubinParameter { offset: 216, template: TEMPLATE_2[216..224].to_vec() },
            ],
            fixups: vec![
                CubinPointerFixup { parameter: 1, byte_offset: 0, binding: 1, binding_offset: 0, extent: 6144, alignment: 16, io: CubinIo::Read },
                CubinPointerFixup { parameter: 2, byte_offset: 0, binding: 0, binding_offset: 0, extent: 2048, alignment: 16, io: CubinIo::Write },
                CubinPointerFixup { parameter: 3, byte_offset: 0, binding: 0, binding_offset: 0, extent: 2048, alignment: 16, io: CubinIo::Write },
                CubinPointerFixup { parameter: 4, byte_offset: 0, binding: 0, binding_offset: 0, extent: 2048, alignment: 16, io: CubinIo::Write },
            ],
        },
    }
}

// Actual launch-06-scope-04-gate_up.json; template SHAa6d7b9a2894970a1a42a860491622a132e506d774e0868244ff3d9ac2d8432c7.
const TEMPLATE_3: [u8; 176] = literal(
    "000000000000000000000000000000000000000000000000008000000000000000800000000000008080f8ffffffffff8080feffffffffff00000000000000000000000000000000000000000000000000000000000000000004000000040000001c0000001c0000001c000040000000000400000000000000000000000000000004000000100000000000000000000000000000000000000000803f0000000000000000000000000000000000000000",
);
fn launch_3() -> MlpLaunch {
    MlpLaunch {
        image: ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
        bindings: &[
            ProjectionBinding::W,
            ProjectionBinding::X,
            ProjectionBinding::Y,
        ],
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80.binary_sm(),
            grid: [28, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 32768,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 14680064,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 131072,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 917504,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: TEMPLATE_3[0..176].to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 14680064,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 8,
                    binding: 1,
                    binding_offset: 0,
                    extent: 131072,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 16,
                    binding: 2,
                    binding_offset: 0,
                    extent: 917504,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 917504,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    }
}

// Actual launch-07-scope-05-down.json; template SHAd4b9121a83b29b45d0eb437fe420d4bea6895c37c6930448a182ad6a2de265ec.
// Y needs no historical input; laterZ reads its own earlier same-launch writes.
const TEMPLATE_4: [u8; 192] = literal(
    "00000000000000000000000000000000000000000000000000c001000000000000c001000000000080c0f3ffffffffff80c0faffffffffff0007000000000000000700000000000000000000000000000000000000000000000e0000000e000000040000000400000004000040000000000e0000000000000000000000000000000700000c100000000000000000000000000000000000000000803f000000000000000000000000000000000000000000000010000000100000000000000000",
);
fn launch_4() -> MlpLaunch {
    MlpLaunch {
        image: ProjectionImage::AmpereBf16Gemm128x64Sm80,
        bindings: &[
            ProjectionBinding::W,
            ProjectionBinding::X,
            ProjectionBinding::Y,
            ProjectionBinding::Workspace,
        ],
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm128x64Sm80.binary_sm(),
            grid: [8, 1, 2],
            block: [128, 1, 1],
            dynamic_shared_bytes: 24576,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 7340032,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 458752,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 131072,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 8519680,
                    alignment: 16,
                    io: CubinIo::ReadWrite,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: TEMPLATE_4[0..192].to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 7340032,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 8,
                    binding: 1,
                    binding_offset: 0,
                    extent: 458752,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 16,
                    binding: 2,
                    binding_offset: 0,
                    extent: 131072,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 131072,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 120,
                    binding: 3,
                    binding_offset: 0,
                    extent: 32,
                    alignment: 16,
                    io: CubinIo::ReadWrite,
                },
            ],
        },
    }
}

// Actual launch-08-scope-06-gate_up.json; template SHAe374383a3567f867654f319120dc8f5bfe65fafa6f734334bcb7bb69d39aa545.
// Only unread padding260..263 zeroed; scalar256=0xffffffff retained.
const TEMPLATE_5: [u8; 360] = literal(
    "02000000001c00000004000001000000c0010000010000000300000000000000010000000004000000000000000000000000000000000000000400000000000000100000000000000091ffffffffffff0001000000000000000400000000000000100000000000000091ffffffffffff0001000000000000003800000000000000c00100000000000078feffffffffff0078feffffffffff0080030000000000000000000000000000800300000000000080030000000000003800000000000000c00100000000000078feffffffffff0078feffffffffff00800300000000000000000000000000008003000000000000800300000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_5() -> MlpLaunch {
    MlpLaunch { image: ProjectionImage::CutlassWmmaSm80, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y],
        manifest: CubinKernelManifest {
            entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
            grid: [8, 56, 1], block: [32, 1, 1], dynamic_shared_bytes: 16896, attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding { bytes: 14680064, alignment: 16, io: CubinIo::Read },
                CubinBinding { bytes: 4096, alignment: 16, io: CubinIo::Read },
                CubinBinding { bytes: 28672, alignment: 16, io: CubinIo::Write },
            ],
            parameters: vec![
                CubinParameter { offset: 0, template: TEMPLATE_5[0..360].to_vec() },
            ],
            fixups: vec![
                CubinPointerFixup { parameter: 0, byte_offset: 280, binding: 1, binding_offset: 0, extent: 4096, alignment: 16, io: CubinIo::Read },
                CubinPointerFixup { parameter: 0, byte_offset: 288, binding: 0, binding_offset: 0, extent: 14680064, alignment: 16, io: CubinIo::Read },
                CubinPointerFixup { parameter: 0, byte_offset: 296, binding: 2, binding_offset: 0, extent: 28672, alignment: 16, io: CubinIo::Write },
                CubinPointerFixup { parameter: 0, byte_offset: 304, binding: 2, binding_offset: 0, extent: 28672, alignment: 16, io: CubinIo::Write },
            ],
        },
    }
}

// Actual launch-09-scope-07-down.json; template SHAd9fb08dd3496ce4ed01fd1a95d1c8179f75fc1900c55c4ca26606ddca1a611b8.
// Y needs no historical input; laterZ reads its own earlier same-launch writes.
const TEMPLATE_6: [u8; 192] = literal(
    "00000000000000000000000000000000000000000000000000c001000000000000c001000000000080c0faffffffffff80c0faffffffffff0007000000000000000700000000000000000000000000000000000000000000000e0000000e000000040000000400000004000002000000000e0000000000000000000000000000000700000c100000000000000000000000000000000000000000803f000000000000000000000000000000000000000000000010000000100000000000000000",
);
fn launch_6() -> MlpLaunch {
    MlpLaunch {
        image: ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
        bindings: &[
            ProjectionBinding::W,
            ProjectionBinding::X,
            ProjectionBinding::Y,
            ProjectionBinding::Workspace,
        ],
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80.binary_sm(),
            grid: [16, 1, 2],
            block: [128, 1, 1],
            dynamic_shared_bytes: 49152,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 7340032,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 14336,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 4096,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinBinding {
                    bytes: 8519680,
                    alignment: 16,
                    io: CubinIo::ReadWrite,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: TEMPLATE_6[0..192].to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 7340032,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 8,
                    binding: 1,
                    binding_offset: 0,
                    extent: 14336,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 16,
                    binding: 2,
                    binding_offset: 0,
                    extent: 4096,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 4096,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 120,
                    binding: 3,
                    binding_offset: 0,
                    extent: 64,
                    alignment: 16,
                    io: CubinIo::ReadWrite,
                },
            ],
        },
    }
}

impl MlpProjection {
    pub(super) const fn rows(self) -> usize {
        match self {
            Self::GateUp1 | Self::Down1 => 1,
            Self::GateUp64 | Self::Down64 => 64,
            Self::GateUp2 | Self::Down2 => 2,
        }
    }
    pub(super) const fn role(self) -> MlpRole {
        match self {
            Self::GateUp1 | Self::GateUp64 | Self::GateUp2 => MlpRole::GateUp,
            Self::Down1 | Self::Down64 | Self::Down2 => MlpRole::Down,
        }
    }
    pub(super) const fn weight_index(self) -> usize {
        match self.role() {
            MlpRole::GateUp => 0,
            MlpRole::Down => 1,
        }
    }
    pub(super) const fn weight_bytes(self) -> usize {
        [14_680_064, 7_340_032][self.weight_index()]
    }
    pub(super) const fn input_bytes(self) -> usize {
        self.rows() * [2048, 7168][self.weight_index()]
    }
    pub(super) const fn output_bytes(self) -> usize {
        self.rows() * [14336, 2048][self.weight_index()]
    }
    pub(super) fn for_phase(phase: VllmGdnPhase) -> Result<[Self; 2]> {
        Ok(match phase {
            VllmGdnPhase::InitialOne | VllmGdnPhase::CachedDecode => [Self::GateUp1, Self::Down1],
            VllmGdnPhase::Initial64 => [Self::GateUp64, Self::Down64],
            VllmGdnPhase::ContinuedTwo => [Self::GateUp2, Self::Down2],
            VllmGdnPhase::Initial47 | VllmGdnPhase::ShortHfPrefill(_) => {
                return Err(Error::Unsupported(
                    "original opaque manifest does not admit47 rows".into(),
                ));
            }
        })
    }
    pub(super) fn operations(self) -> Vec<MlpOperation> {
        use MlpOperation::{Clear, Kernel};
        match self {
            Self::GateUp1 => vec![Kernel(launch_0())],
            Self::Down1 => vec![Kernel(launch_1()), Kernel(launch_2())],
            Self::GateUp64 => vec![Kernel(launch_3())],
            Self::Down64 => vec![Clear(MlpWorkspaceClear { bytes: 32 }), Kernel(launch_4())],
            Self::GateUp2 => vec![Kernel(launch_5())],
            Self::Down2 => vec![Clear(MlpWorkspaceClear { bytes: 64 }), Kernel(launch_6())],
        }
    }
}

#[cfg(test)]
#[path = "mlp_manifests/tests.rs"]
mod tests;
