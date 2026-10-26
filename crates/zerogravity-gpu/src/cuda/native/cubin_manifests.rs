//! Fixed original19-launch ABI; mathematical inputs come from original leases.
//! Source: ROOT's executed root-closed-launch-manifests-1 (15public roles).
//! Capture data SHA2644a3a108652f9459c9adf34cbe3343cfa84b3a7ec3cb3c31b51fc0e8bafba3.
//! Metadata is not an image issuer or numerical/profile acceptance.
use crate::cuda::gdn::GdnPhase;
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup, PinnedBf16Cubin, PinnedCapturedBf16Cubin,
};
use zerogravity_core::{Error, Result};

pub(super) const WORKSPACE_BYTES: usize = 8_519_680;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Projection {
    Qkvz1,
    Ba1,
    Out1,
    Qkvz64,
    Ba64,
    Out64,
    Qkvz2,
    Ba2,
    Out2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProjectionBinding {
    W,
    X,
    Y,
    Workspace,
}

/// Closed observed image metadata; executable provenance remains explicit.
#[derive(Clone, Copy)]
pub(super) enum ProjectionImageSource {
    Derived(PinnedBf16Cubin),
    Captured(PinnedCapturedBf16Cubin),
}

/// Exact original executable identities; neither kind accepts caller-chosen code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProjectionImage {
    TextFaCausalCombineSm80,
    TextFaSplitCombineSm80,
    TextCutlassBf16Sm80,
    Cohort2BAmpere64SlicedSm80,
    FixedPromptCutlassBf16Sm80,
    RotaryArangeSm86,
    RotaryDivideSm86,
    RotaryFillSm86,
    RotaryPowSm86,
    RotaryReciprocalSm86,
    RotaryCosSm86,
    RotarySinSm86,
    RotaryCatSm86,
    RotaryBf16CastSm86,
    Bf16Gemv2TSm86,
    Bf16GemvxSm86,
    CutlassWmmaSm80,
    Bf16SplitKReduceSm86,
    AmpereBf16Gemm128x64Sm80,
    AmpereBf16Gemm64x64Stages64x5Sm80,
    AttentionAmpere256x128Sm80,
    AttentionQkSm86,
    AttentionKvSm86,
    AttentionFaSplitSm80,
    AttentionFaCausalCombineSm80,
    AttentionSigmoidSm86,
    AttentionMultiplySm86,
}
impl ProjectionImage {
    pub const fn bytes(self) -> usize {
        match self {
            Self::TextFaCausalCombineSm80 => 3_595_872,
            Self::TextFaSplitCombineSm80 => 6_317_920,
            Self::TextCutlassBf16Sm80 => 3_965_472,
            Self::Cohort2BAmpere64SlicedSm80 => 5_857_056,
            Self::FixedPromptCutlassBf16Sm80 => 7_789_728,
            Self::RotaryArangeSm86 => 638536,
            Self::RotaryDivideSm86 => 4066320,
            Self::RotaryFillSm86 => 1942400,
            Self::RotaryPowSm86 => 16041664,
            Self::RotaryReciprocalSm86 => 4246208,
            Self::RotaryCosSm86 => 850848,
            Self::RotarySinSm86 => 848032,
            Self::RotaryCatSm86 => 2103392,
            Self::RotaryBf16CastSm86 => 4557464,
            Self::AttentionAmpere256x128Sm80 => 6352608,
            Self::AttentionQkSm86 => 17584,
            Self::AttentionSigmoidSm86 => 2679648,
            Self::AttentionMultiplySm86 => 3397824,
            Self::AttentionKvSm86 => 1813600,
            Self::AttentionFaSplitSm80 => 6317664,
            Self::AttentionFaCausalCombineSm80 => 3596000,
            Self::Bf16Gemv2TSm86 => 2818208,
            Self::Bf16GemvxSm86 => 5273248,
            Self::CutlassWmmaSm80 => 1428640,
            Self::Bf16SplitKReduceSm86 => 4252960,
            Self::AmpereBf16Gemm128x64Sm80 => 6601808,
            Self::AmpereBf16Gemm64x64Stages64x5Sm80 => 6422224,
        }
    }
    pub const fn sha256(self) -> [u8; 32] {
        match self {
            Self::TextFaCausalCombineSm80 => {
                literal("f739be47a3f566d4bca6fea9009a69251c592e5584ad8f266bb6d352f873325f")
            }
            Self::TextFaSplitCombineSm80 => {
                literal("c777ff76fb18e3d3f30cb94d522432d91755bbd6915e5838d137444bd90c1ee3")
            }
            Self::Cohort2BAmpere64SlicedSm80 => {
                literal("ed95d9920ecf50d7a50a5f94e1986fec41a2555c683fabbe148bc36de40d01f5")
            }
            Self::TextCutlassBf16Sm80 => {
                literal("dd79ea92347b2b68d382fcca514f57d9001e1ec91d228f61b0e7048b2cd86c2b")
            }
            Self::FixedPromptCutlassBf16Sm80 => {
                literal("66c2e61f0cc1d1bf90eeb8ea3245e2849b0c49466c486c8d1daa219166afdc30")
            }
            Self::RotaryArangeSm86 => {
                literal("a2214f06eb1698d79b9073d5aa0b96687db648a1dfcb972ffd8acd55f0463148")
            }
            Self::RotaryDivideSm86 => {
                literal("6bdb6b810e0240cf9b43eea275a62a569a1b413a098021647073a023fd7257ef")
            }
            Self::RotaryFillSm86 => {
                literal("842017f87bae6cdf46487a53d681d766bcdd924ec10d8d8c760311f353126dc5")
            }
            Self::RotaryPowSm86 => {
                literal("3af1ff7b658220ca44e2c21926755e74573439133a9a8279229ae7a9bf6a1d46")
            }
            Self::RotaryReciprocalSm86 => {
                literal("e74e50dcbacecbb9513fd86fb522d9c8be3c6380673894e61f18138aa350c510")
            }
            Self::RotaryCosSm86 => {
                literal("74f16ee1c1a02f9023ef564e667515b28e19e62eec34ca2e88abeac8f8554784")
            }
            Self::RotarySinSm86 => {
                literal("4c8070076b500012feaaf022e79cb08f3cd7114b2985160452402fa040c56034")
            }
            Self::RotaryCatSm86 => {
                literal("16b48f3157ebebadf98cff5c1d49c14481b0329db9341242b89a6350fd740bed")
            }
            Self::RotaryBf16CastSm86 => {
                literal("f57651baaf725abc6147aa5c3b6cdd5f36bd21d094d1e07125ce4af1cfa39bd4")
            }
            Self::AttentionAmpere256x128Sm80 => {
                literal("6ab9468c2d21b9b24c66fbb4380efca47a9057ddcc6adb4eb83997460986a83e")
            }
            Self::AttentionSigmoidSm86 => {
                literal("08c00ee67fa7452b5370a7b2fc2b611afdbe219b5343cebf935aeef6e872ef8b")
            }
            Self::AttentionMultiplySm86 => {
                literal("9277f5b0fa8d3a02004e2c8f02c1991c0bef281b8aa8aebab886797daa13502a")
            }
            Self::AttentionQkSm86 => {
                literal("9049ad21c964126fe32a394f6e5335e9ec0901cb031559db3cab0e63d93b4664")
            }
            Self::AttentionKvSm86 => {
                literal("0781fbbcae3a3f26cf97205af806e6e9001413827e66e298d4a0743cde067035")
            }
            Self::AttentionFaSplitSm80 => {
                literal("915016a9eaee3a391e29bbfcf00ee8de82e4da53c4865d080c04505d0d736345")
            }
            Self::AttentionFaCausalCombineSm80 => {
                literal("b4c2cb3cdcf9d456d5a0c905e468d9fe01a79df46544c8f447b0471ae2b4f6d1")
            }
            Self::Bf16Gemv2TSm86 => {
                literal("94fc9c63d0562b88395c3d5001397aa1a3a3f572450a00db627750b817c7de56")
            }
            Self::Bf16GemvxSm86 => {
                literal("02cd5eee5a1f93addca2135cde7dfd71b05e499547e227a43806ee953eb2fa99")
            }
            Self::CutlassWmmaSm80 => {
                literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d")
            }
            Self::Bf16SplitKReduceSm86 => {
                literal("074807d671d9156ccc56b5a8cbe82f1ba8fae21c8169e674b31b4b8603ba42ec")
            }
            Self::AmpereBf16Gemm128x64Sm80 => {
                literal("c1ca8db71001178d04f1b83d7c55883107fa122927e10d41020672678fff2c86")
            }
            Self::AmpereBf16Gemm64x64Stages64x5Sm80 => {
                literal("c3361a22db27ce017fec90a46f66c508cad5d668d606070b214b0351b3cf9588")
            }
        }
    }
    pub const fn binary_sm(self) -> u8 {
        match self {
            Self::TextFaCausalCombineSm80 => 80,
            Self::TextFaSplitCombineSm80 => 80,
            Self::TextCutlassBf16Sm80 => 80,
            Self::Cohort2BAmpere64SlicedSm80 => 80,
            Self::FixedPromptCutlassBf16Sm80 => 80,
            Self::RotaryArangeSm86
            | Self::RotaryDivideSm86
            | Self::RotaryFillSm86
            | Self::RotaryPowSm86
            | Self::RotaryReciprocalSm86
            | Self::RotaryCosSm86
            | Self::RotarySinSm86
            | Self::RotaryCatSm86
            | Self::RotaryBf16CastSm86 => 86,
            Self::AttentionAmpere256x128Sm80 => 80,
            Self::AttentionQkSm86 | Self::AttentionSigmoidSm86 | Self::AttentionMultiplySm86 => 86,
            Self::AttentionKvSm86 => 86,
            Self::AttentionFaSplitSm80 => 80,
            Self::AttentionFaCausalCombineSm80 => 80,
            Self::Bf16Gemv2TSm86 => 86,
            Self::Bf16GemvxSm86 => 86,
            Self::CutlassWmmaSm80 => 80,
            Self::Bf16SplitKReduceSm86 => 86,
            Self::AmpereBf16Gemm128x64Sm80 => 80,
            Self::AmpereBf16Gemm64x64Stages64x5Sm80 => 80,
        }
    }
    pub(super) const fn provenance(self) -> ProjectionImageSource {
        use ProjectionImageSource::{Captured, Derived};
        match self {
            Self::TextFaCausalCombineSm80 => {
                Captured(PinnedCapturedBf16Cubin::TextFaCausalCombineSm80)
            }
            Self::TextFaSplitCombineSm80 => {
                Captured(PinnedCapturedBf16Cubin::TextFaSplitCombineSm80)
            }
            Self::Cohort2BAmpere64SlicedSm80 => {
                Captured(PinnedCapturedBf16Cubin::Cohort2BAmpere64SlicedSm80)
            }
            Self::TextCutlassBf16Sm80 => Captured(PinnedCapturedBf16Cubin::TextCutlassBf16Sm80),
            Self::FixedPromptCutlassBf16Sm80 => {
                Captured(PinnedCapturedBf16Cubin::FixedPromptCutlassBf16Sm80)
            }
            Self::RotaryArangeSm86 => Captured(PinnedCapturedBf16Cubin::RotaryArangeSm86),
            Self::RotaryDivideSm86 => Captured(PinnedCapturedBf16Cubin::RotaryDivideSm86),
            Self::RotaryFillSm86 => Captured(PinnedCapturedBf16Cubin::RotaryFillSm86),
            Self::RotaryPowSm86 => Captured(PinnedCapturedBf16Cubin::RotaryPowSm86),
            Self::RotaryReciprocalSm86 => Captured(PinnedCapturedBf16Cubin::RotaryReciprocalSm86),
            Self::RotaryCosSm86 => Captured(PinnedCapturedBf16Cubin::RotaryCosSm86),
            Self::RotarySinSm86 => Captured(PinnedCapturedBf16Cubin::RotarySinSm86),
            Self::RotaryCatSm86 => Captured(PinnedCapturedBf16Cubin::RotaryCatSm86),
            Self::RotaryBf16CastSm86 => Captured(PinnedCapturedBf16Cubin::RotaryBf16CastSm86),
            Self::AttentionAmpere256x128Sm80 => {
                Captured(PinnedCapturedBf16Cubin::AttentionAmpere256x128Sm80)
            }
            Self::AttentionSigmoidSm86 => Captured(PinnedCapturedBf16Cubin::AttentionSigmoidSm86),
            Self::AttentionMultiplySm86 => Captured(PinnedCapturedBf16Cubin::AttentionMultiplySm86),
            Self::AttentionQkSm86 => Captured(PinnedCapturedBf16Cubin::AttentionQkSm86),
            Self::AttentionKvSm86 => Captured(PinnedCapturedBf16Cubin::AttentionKvSm86),
            Self::AttentionFaSplitSm80 => Captured(PinnedCapturedBf16Cubin::AttentionFaSplitSm80),
            Self::AttentionFaCausalCombineSm80 => {
                Captured(PinnedCapturedBf16Cubin::AttentionFaCausalCombineSm80)
            }
            Self::Bf16Gemv2TSm86 => Derived(PinnedBf16Cubin::Bf16Gemv2TSm86),
            Self::Bf16GemvxSm86 => Derived(PinnedBf16Cubin::Bf16GemvxSm86),
            Self::CutlassWmmaSm80 => Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
            Self::Bf16SplitKReduceSm86 => Derived(PinnedBf16Cubin::Bf16SplitKReduceSm86),
            Self::AmpereBf16Gemm128x64Sm80 => Derived(PinnedBf16Cubin::AmpereBf16Gemm128x64Sm80),
            Self::AmpereBf16Gemm64x64Stages64x5Sm80 => {
                Derived(PinnedBf16Cubin::AmpereBf16Gemm64x64Stages64x5Sm80)
            }
        }
    }
}

pub(super) struct ProjectionLaunch {
    image: ProjectionImage,
    bindings: &'static [ProjectionBinding],
    manifest: CubinKernelManifest,
}
impl ProjectionLaunch {
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

// Literals only; callers cannot submit a runtime template or manifest.
const fn literal<const N: usize>(text: &str) -> [u8; N] {
    const fn nibble(v: u8) -> u8 {
        match v {
            b'0'..=b'9' => v - b'0',
            b'a'..=b'f' => v - b'a' + 10,
            _ => panic!("invalid pinned byte literal"),
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

// Original launch00, template SHAdfc67abea8f8a38465b46ad4206357941fadc95e5734143f258189bb9bfbd2ec.
const TEMPLATE_0: [u8; 160] = literal(
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040000002000000004000001000000010000000100000001000000000000000000000000000000000000000000000000000000000000000000803f000000000000000000000000000000000000000000000000000000000000803f00000000",
);
fn launch_0() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::Bf16Gemv2TSm86, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y],
    manifest: CubinKernelManifest { entry: "_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_".into(), binary_sm: ProjectionImage::Bf16Gemv2TSm86.binary_sm(),
        grid: [1024, 1, 1], block: [128, 1, 1], dynamic_shared_bytes: 0, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:16777216,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:2048,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:16384,alignment:16,io:CubinIo::Write},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_0[0..152].to_vec()},
            CubinParameter{offset:152,template:TEMPLATE_0[152..156].to_vec()},
            CubinParameter{offset:156,template:TEMPLATE_0[156..160].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:16777216,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:32,binding:2,binding_offset:0,extent:16384,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:0,byte_offset:48,binding:2,binding_offset:0,extent:16384,alignment:16,io:CubinIo::Write},
        ],
    }}
}

// Original launch01, template SHA8099828cc6a580b07f8de648a8600be63031b59eaafd12167824a2ffa7aadf92.
const TEMPLATE_1: [u8; 152] = literal(
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040000200000000004000001000000010000000100000001000000000000000000000000000000000000000000000000000000000000000000803f00000000000000000000000000000000200000000100000000000000",
);
fn launch_1() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::Bf16GemvxSm86, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y],
    manifest: CubinKernelManifest { entry: "_ZN8internal5gemvx6kernelIii13__nv_bfloat16S2_S2_fLb0ELb1ELb1ELb0ELi7ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS2_ES6_S4_IS2_EfEEENSt9enable_ifIXntT5_EvE4typeET11_".into(), binary_sm: ProjectionImage::Bf16GemvxSm86.binary_sm(),
        grid: [8, 1, 1], block: [32, 4, 1], dynamic_shared_bytes: 528, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:65536,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:2048,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:64,alignment:16,io:CubinIo::Write},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_1[0..152].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:0,byte_offset:0,binding:0,binding_offset:0,extent:65536,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:16,binding:1,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:32,binding:2,binding_offset:0,extent:64,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:0,byte_offset:48,binding:2,binding_offset:0,extent:64,alignment:16,io:CubinIo::Write},
        ],
    }}
}

// Original launch02, template SHA0b53216165769e7d5051e5cbdf1d98b52b05cdc18f3c6a96b4ab17975cd23084.
// 260..264: unread padding; preserve scalar256 exactly.
const TEMPLATE_2: [u8; 360] = literal(
    "010000000004000000080000010000004000000003000000030000000100000003000000c002000000040000000000000000000000000000000800000000000000200000000000000021ffffffffffff0001000000000000000800000000000000200000000000000021ffffffffffff00010000000000000008000000000000004000000000000000c8ffffffffffff00c8ffffffffffff00800000000000000000000000000000008000000000000000800000000000000008000000000000004000000000000000c8ffffffffffff00c8ffffffffffff00800000000000000000000000000000008000000000000000800000000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_2() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::CutlassWmmaSm80, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y,ProjectionBinding::Workspace],
    manifest: CubinKernelManifest { entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
        grid: [8, 8, 3], block: [32, 1, 1], dynamic_shared_bytes: 16896, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:4194304,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:2048,alignment:16,io:CubinIo::Write},
            CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Write},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_2[0..360].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:4194304,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:0,byte_offset:304,binding:3,binding_offset:0,extent:6144,alignment:16,io:CubinIo::Write},
        ],
    }}
}

// Original launch03, template SHA2021f6308e1236fbd351cdf8ba655422e9c181da3c4e649029ed7e1317fee194.
// 96..112: host alpha/beta pointer args unreferenced under sealed scalar-mode44=0.
const TEMPLATE_3: [u8; 224] = literal(
    "0000803f00000000000400000100000000040000000400000004000000040000010000000004000003000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_3() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::Bf16SplitKReduceSm86, bindings: &[ProjectionBinding::Y,ProjectionBinding::Workspace],
    manifest: CubinKernelManifest { entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(), binary_sm: ProjectionImage::Bf16SplitKReduceSm86.binary_sm(),
        grid: [32, 1, 1], block: [32, 16, 1], dynamic_shared_bytes: 0, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:2048,alignment:16,io:CubinIo::Write},
            CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Read},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_3[0..60].to_vec()},
            CubinParameter{offset:64,template:TEMPLATE_3[64..72].to_vec()},
            CubinParameter{offset:72,template:TEMPLATE_3[72..80].to_vec()},
            CubinParameter{offset:80,template:TEMPLATE_3[80..88].to_vec()},
            CubinParameter{offset:88,template:TEMPLATE_3[88..96].to_vec()},
            CubinParameter{offset:96,template:TEMPLATE_3[96..104].to_vec()},
            CubinParameter{offset:104,template:TEMPLATE_3[104..112].to_vec()},
            CubinParameter{offset:112,template:TEMPLATE_3[112..120].to_vec()},
            CubinParameter{offset:120,template:TEMPLATE_3[120..128].to_vec()},
            CubinParameter{offset:128,template:TEMPLATE_3[128..136].to_vec()},
            CubinParameter{offset:136,template:TEMPLATE_3[136..144].to_vec()},
            CubinParameter{offset:144,template:TEMPLATE_3[144..152].to_vec()},
            CubinParameter{offset:152,template:TEMPLATE_3[152..160].to_vec()},
            CubinParameter{offset:160,template:TEMPLATE_3[160..168].to_vec()},
            CubinParameter{offset:168,template:TEMPLATE_3[168..176].to_vec()},
            CubinParameter{offset:176,template:TEMPLATE_3[176..184].to_vec()},
            CubinParameter{offset:184,template:TEMPLATE_3[184..192].to_vec()},
            CubinParameter{offset:192,template:TEMPLATE_3[192..200].to_vec()},
            CubinParameter{offset:200,template:TEMPLATE_3[200..208].to_vec()},
            CubinParameter{offset:208,template:TEMPLATE_3[208..216].to_vec()},
            CubinParameter{offset:216,template:TEMPLATE_3[216..224].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:1,byte_offset:0,binding:1,binding_offset:0,extent:6144,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:2,byte_offset:0,binding:0,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:3,byte_offset:0,binding:0,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:4,byte_offset:0,binding:0,binding_offset:0,extent:2048,alignment:16,io:CubinIo::Write},
        ],
    }}
}

// Original launch08, template SHAd0bf6cc00cd24b25b64f8ab9c7954322defff17a946c243b9cc1f1ee4a548b49.
const TEMPLATE_8: [u8; 176] = literal(
    "000000000000000000000000000000000000000000000000008000000000000000800000000000004080fcffffffffff4080feffffffffff0000000000000000000000000000000000000000000000000000000000000000000400000004000000200000002000000020000040000000000400000000000000000000000000000004000000000000000000000000000000000000000000000000803f0000000000000000000000000000000000000000",
);
fn launch_8() -> ProjectionLaunch {
    ProjectionLaunch {
        image: ProjectionImage::AmpereBf16Gemm128x64Sm80,
        bindings: &[
            ProjectionBinding::W,
            ProjectionBinding::X,
            ProjectionBinding::Y,
        ],
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm128x64Sm80.binary_sm(),
            grid: [64, 1, 1],
            block: [64, 1, 1],
            dynamic_shared_bytes: 0,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 16777216,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 131072,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 1048576,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: TEMPLATE_8[0..176].to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 16777216,
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
                    extent: 1048576,
                    alignment: 16,
                    io: CubinIo::Write,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 168,
                    binding: 2,
                    binding_offset: 0,
                    extent: 1048576,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
        },
    }
}

// Original launch09, template SHA32e1e95d1e1b0dea82aee434474f082259a707a402ea75837c5ac651436d66a5.
// 260..264: unread padding; preserve scalar256 exactly.
const TEMPLATE_9: [u8; 360] = literal(
    "4000000020000000000400000200000001000000100000000000000001000000100000004000000000080000000000000000000000000000000400000000000000400000000000000041ffffffffffff0001000000000000000400000000000000400000000000000041ffffffffffff00010000000000004000000000000000000100000000000040ffffffffffffff40ffffffffffffff00040000000000000004000000000000000800000000000000080000000000004000000000000000000100000000000040ffffffffffffff40ffffffffffffff00040000000000000004000000000000000800000000000000080000000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_9() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::CutlassWmmaSm80, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y,ProjectionBinding::Workspace],
    manifest: CubinKernelManifest { entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
        grid: [2, 1, 16], block: [128, 1, 1], dynamic_shared_bytes: 33792, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:65536,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:131072,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},
            CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Write},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_9[0..360].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:131072,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:65536,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:0,byte_offset:304,binding:3,binding_offset:0,extent:65536,alignment:16,io:CubinIo::Write},
        ],
    }}
}

// Original launch10, template SHA4e36ddd7a04dbb6acd3615e945c968117d0caa69c243ba1abc6c1f2f530b1380.
// 96..112: host alpha/beta pointer args unreferenced under sealed scalar-mode44=0.
const TEMPLATE_10: [u8; 224] = literal(
    "0000803f00000000200000004000000020000000200000002000000020000000010000000008000010000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_10() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::Bf16SplitKReduceSm86, bindings: &[ProjectionBinding::Y,ProjectionBinding::Workspace],
    manifest: CubinKernelManifest { entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(), binary_sm: ProjectionImage::Bf16SplitKReduceSm86.binary_sm(),
        grid: [1, 4, 1], block: [32, 16, 1], dynamic_shared_bytes: 0, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},
            CubinBinding{bytes:8519680,alignment:16,io:CubinIo::Read},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_10[0..60].to_vec()},
            CubinParameter{offset:64,template:TEMPLATE_10[64..72].to_vec()},
            CubinParameter{offset:72,template:TEMPLATE_10[72..80].to_vec()},
            CubinParameter{offset:80,template:TEMPLATE_10[80..88].to_vec()},
            CubinParameter{offset:88,template:TEMPLATE_10[88..96].to_vec()},
            CubinParameter{offset:96,template:TEMPLATE_10[96..104].to_vec()},
            CubinParameter{offset:104,template:TEMPLATE_10[104..112].to_vec()},
            CubinParameter{offset:112,template:TEMPLATE_10[112..120].to_vec()},
            CubinParameter{offset:120,template:TEMPLATE_10[120..128].to_vec()},
            CubinParameter{offset:128,template:TEMPLATE_10[128..136].to_vec()},
            CubinParameter{offset:136,template:TEMPLATE_10[136..144].to_vec()},
            CubinParameter{offset:144,template:TEMPLATE_10[144..152].to_vec()},
            CubinParameter{offset:152,template:TEMPLATE_10[152..160].to_vec()},
            CubinParameter{offset:160,template:TEMPLATE_10[160..168].to_vec()},
            CubinParameter{offset:168,template:TEMPLATE_10[168..176].to_vec()},
            CubinParameter{offset:176,template:TEMPLATE_10[176..184].to_vec()},
            CubinParameter{offset:184,template:TEMPLATE_10[184..192].to_vec()},
            CubinParameter{offset:192,template:TEMPLATE_10[192..200].to_vec()},
            CubinParameter{offset:200,template:TEMPLATE_10[200..208].to_vec()},
            CubinParameter{offset:208,template:TEMPLATE_10[208..216].to_vec()},
            CubinParameter{offset:216,template:TEMPLATE_10[216..224].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:1,byte_offset:0,binding:1,binding_offset:0,extent:65536,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:2,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:3,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:4,byte_offset:0,binding:0,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},
        ],
    }}
}

// Original launch11, template SHAf4142787020728fda9be2957984dff6ea9fbac530a661b7fab9377c9e8aeafb2.
const TEMPLATE_11: [u8; 192] = literal(
    "000000000000000000000000000000000000000000000000000001000000000000000100000000008000fdffffffffff8000fdffffffffff0000000000000000000000000000000000000000000000000000000000000000000800000008000000040000000400000004000040000000000800000000000000000000000000000008000000000000000000000000000000000000000000000000803f000000000000000000000000000000000000000000000010000000100000000000000000",
);
fn launch_11() -> ProjectionLaunch {
    ProjectionLaunch {
        image: ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
        bindings: &[
            ProjectionBinding::W,
            ProjectionBinding::X,
            ProjectionBinding::Y,
        ],
        manifest: CubinKernelManifest {
            entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn".into(),
            binary_sm: ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80.binary_sm(),
            grid: [16, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 32768,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                CubinBinding {
                    bytes: 4194304,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 262144,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinBinding {
                    bytes: 131072,
                    alignment: 16,
                    io: CubinIo::Write,
                },
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: TEMPLATE_11[0..192].to_vec(),
            }],
            fixups: vec![
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 0,
                    binding: 0,
                    binding_offset: 0,
                    extent: 4194304,
                    alignment: 16,
                    io: CubinIo::Read,
                },
                CubinPointerFixup {
                    parameter: 0,
                    byte_offset: 8,
                    binding: 1,
                    binding_offset: 0,
                    extent: 262144,
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
            ],
        },
    }
}

// Original launch12, template SHA0747c605f21b9eda0344c5403add5ddf121835421f7bceeebb4fc6f2b84dfc70.
// 260..264: unread padding; preserve scalar256 exactly.
const TEMPLATE_12: [u8; 360] = literal(
    "0200000000200000000400000100000000020000010000000300000000000000010000000004000000000000000000000000000000000000000400000000000000100000000000000091ffffffffffff0001000000000000000400000000000000100000000000000091ffffffffffff0001000000000000004000000000000000000200000000000040feffffffffff0040feffffffffff0000040000000000000000000000000000000400000000000000040000000000004000000000000000000200000000000040feffffffffff0040feffffffffff00000400000000000000000000000000000004000000000000000400000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_12() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::CutlassWmmaSm80, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y],
    manifest: CubinKernelManifest { entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
        grid: [8, 64, 1], block: [32, 1, 1], dynamic_shared_bytes: 8704, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:16777216,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:32768,alignment:16,io:CubinIo::Write},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_12[0..360].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:16777216,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:32768,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:0,byte_offset:304,binding:2,binding_offset:0,extent:32768,alignment:16,io:CubinIo::Write},
        ],
    }}
}

// Original launch13, template SHA823b9646bceaa71b810b539a8cf2517e7ddc0d10f509431e67832d559834f9a1.
// 260..264: unread padding; preserve scalar256 exactly.
const TEMPLATE_13: [u8; 360] = literal(
    "0200000020000000000400000100000002000000010000000100000000000000010000000004000000000000000000000000000000000000000400000000000000100000000000000091ffffffffffff0001000000000000000400000000000000100000000000000091ffffffffffff00010000000000004000000000000000000200000000000040feffffffffffff40feffffffffffff00040000000000000000000000000000000400000000000000040000000000004000000000000000000200000000000040feffffffffffff40feffffffffffff00040000000000000000000000000000000400000000000000040000000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_13() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::CutlassWmmaSm80, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y],
    manifest: CubinKernelManifest { entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
        grid: [2, 1, 1], block: [32, 1, 1], dynamic_shared_bytes: 16896, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:65536,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:4096,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:128,alignment:16,io:CubinIo::Write},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_13[0..360].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:65536,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:128,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:0,byte_offset:304,binding:2,binding_offset:0,extent:128,alignment:16,io:CubinIo::Write},
        ],
    }}
}

// Original launch14, template SHA8a8894d9b0cea8e4d14bbd3224fde87674aeffcb6a4d7f1e49a26f5218f91df5.
// 260..264: unread padding; preserve scalar256 exactly.
const TEMPLATE_14: [u8; 360] = literal(
    "0200000000040000000800000100000020000000010000000300000000000000010000000008000000000000000000000000000000000000000800000000000000800000000000000081feffffffffff0001000000000000000800000000000000800000000000000081feffffffffff00010000000000000008000000000000002000000000000000e8ffffffffffff00e8ffffffffffff00800000000000000080000000000000000001000000000000000100000000000008000000000000002000000000000000e8ffffffffffff00e8ffffffffffff00800000000000000080000000000000000001000000000000000100000000000000803f00000000ffffffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
);
fn launch_14() -> ProjectionLaunch {
    ProjectionLaunch { image: ProjectionImage::CutlassWmmaSm80, bindings: &[ProjectionBinding::W,ProjectionBinding::X,ProjectionBinding::Y],
    manifest: CubinKernelManifest { entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE".into(), binary_sm: ProjectionImage::CutlassWmmaSm80.binary_sm(),
        grid: [8, 4, 1], block: [128, 1, 1], dynamic_shared_bytes: 33792, attributes:CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding{bytes:4194304,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:8192,alignment:16,io:CubinIo::Read},
            CubinBinding{bytes:4096,alignment:16,io:CubinIo::Write},
        ],
        parameters: vec![
            CubinParameter{offset:0,template:TEMPLATE_14[0..360].to_vec()},
        ],
        fixups: vec![
            CubinPointerFixup{parameter:0,byte_offset:280,binding:1,binding_offset:0,extent:8192,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:288,binding:0,binding_offset:0,extent:4194304,alignment:16,io:CubinIo::Read},
            CubinPointerFixup{parameter:0,byte_offset:296,binding:2,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},
            CubinPointerFixup{parameter:0,byte_offset:304,binding:2,binding_offset:0,extent:4096,alignment:16,io:CubinIo::Write},
        ],
    }}
}

impl Projection {
    pub(super) const fn rows(self) -> usize {
        match self {
            Self::Qkvz1 | Self::Ba1 | Self::Out1 => 1,
            Self::Qkvz64 | Self::Ba64 | Self::Out64 => 64,
            Self::Qkvz2 | Self::Ba2 | Self::Out2 => 2,
        }
    }
    pub(super) const fn weight_index(self) -> usize {
        match self {
            Self::Qkvz1 | Self::Qkvz64 | Self::Qkvz2 => 0,
            Self::Ba1 | Self::Ba64 | Self::Ba2 => 1,
            Self::Out1 | Self::Out64 | Self::Out2 => 2,
        }
    }
    pub(super) const fn weight_bytes(self) -> usize {
        [16_777_216, 65_536, 4_194_304][self.weight_index()]
    }
    pub(super) const fn input_bytes(self) -> usize {
        self.rows() * if self.weight_index() == 2 { 4096 } else { 2048 }
    }
    pub(super) const fn output_bytes(self) -> usize {
        self.rows() * [16_384, 64, 2048][self.weight_index()]
    }
    pub(super) fn for_phase(phase: GdnPhase) -> Result<[Self; 3]> {
        Ok(match phase {
            GdnPhase::InitialOne | GdnPhase::CachedDecode => [Self::Qkvz1, Self::Ba1, Self::Out1],
            GdnPhase::Initial64 => [Self::Qkvz64, Self::Ba64, Self::Out64],
            GdnPhase::ContinuedTwo => [Self::Qkvz2, Self::Ba2, Self::Out2],
            GdnPhase::FixedPrompt | GdnPhase::TextPrefill(_) => {
                return Err(Error::Unsupported(
                    "original opaque manifest does not admit47 rows".into(),
                ));
            }
        })
    }
    pub(super) fn launches(self) -> Vec<ProjectionLaunch> {
        match self {
            Self::Qkvz1 => vec![launch_0()],
            Self::Ba1 => vec![launch_1()],
            Self::Out1 => vec![launch_2(), launch_3()],
            Self::Qkvz64 => vec![launch_8()],
            Self::Ba64 => vec![launch_9(), launch_10()],
            Self::Out64 => vec![launch_11()],
            Self::Qkvz2 => vec![launch_12()],
            Self::Ba2 => vec![launch_13()],
            Self::Out2 => vec![launch_14()],
        }
    }
}

#[cfg(test)]
#[path = "cubin_manifests/tests.rs"]
mod tests;
