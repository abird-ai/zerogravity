//! Immutable original-set limits; connected scope accepts only pinned entries.
use super::{
    CubinKernelManifest, PinnedBf16Cubin, PinnedCapturedBf16Cubin, VerifiedCubin, failure,
};
use cubecl_core::server::ServerError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CubinSetPolicy {
    Text08,
    Text08Cohort4,
    Text2B,
    Text2BCachedCohort4,
    SourceRotary,
    ProjectionSix,
    FullQwen08,
    ConnectedQwen08,
    FixedPromptMlp,
    Qwen2bMlp,
    FixedPromptQkv,
    FixedPromptQkKv,
    FixedPromptFa,
    FixedPromptLayer3,
    FixedPromptFull,
    FixedPromptNatural,
    FirstAttentionCaptured,
    FirstAttentionComputedGate,
    VocabularyHead,
}

impl CubinSetPolicy {
    pub(crate) fn limits(
        self,
        modules: usize,
        functions: usize,
        records: usize,
        new_module: bool,
        new_function: bool,
    ) -> Result<(), ServerError> {
        if matches!(
            self,
            Self::Text08
                | Self::Text08Cohort4
                | Self::Text2B
                | Self::Text2BCachedCohort4
        ) {
            return super::text_policy::limits(
                self,
                modules,
                functions,
                records,
                new_module,
                new_function,
            );
        }
        if self == Self::FixedPromptNatural {
            return super::fixed_prompt_natural_policy::limits(
                modules,
                functions,
                records,
                new_module,
                new_function,
            );
        }
        let (max_modules, max_functions, max_records, message) = match self {
            Self::Text08
            | Self::Text08Cohort4
            | Self::Text2B
            | Self::Text2BCachedCohort4 => {
                unreachable!("handled finite short-HF policy")
            }
            Self::FixedPromptNatural => unreachable!("handled closed natural policy"),
            Self::FixedPromptFull => (
                12,
                15,
                16,
                "closed canonical47 full model twelve-module/fifteen-function/sixteen-record capacity",
            ),
            Self::Qwen2bMlp => (
                5,
                7,
                7,
                "closed2B MLP five-module/seven-function/seven-record capacity",
            ),
            Self::FixedPromptLayer3 => (
                11,
                14,
                15,
                "closed canonical47 layer3 eleven-module/fourteen-function/fifteen-record capacity",
            ),
            Self::FixedPromptFa => (
                9,
                12,
                13,
                "closed canonical47 FA nine-module/twelve-function/thirteen-record capacity",
            ),
            Self::FixedPromptQkKv => (
                8,
                11,
                12,
                "closed canonical47 QK/KV eight-module/eleven-function/twelve-record capacity",
            ),
            Self::FixedPromptQkv => (
                6,
                9,
                10,
                "closed canonical47 QKV six-module/nine-function/ten-record capacity",
            ),
            Self::FixedPromptMlp => (
                5,
                7,
                8,
                "closed canonical47 MLP five-module/seven-function/eight-record capacity",
            ),
            Self::FullQwen08 => (
                super::full_model_policy::MODULES,
                super::full_model_policy::FUNCTIONS,
                super::full_model_policy::RECORDS,
                "closed original full24 fourteen-module/twenty-two-function/forty-eight-record capacity",
            ),
            Self::SourceRotary => (
                10,
                11,
                12,
                "closed source rotary ten-module/eleven-function/twelve-record capacity",
            ),
            Self::FirstAttentionComputedGate => (
                11,
                15,
                26,
                "closed attention-gate eleven-module/fifteen-function/twenty-six-record capacity",
            ),
            Self::VocabularyHead => (
                1,
                1,
                1,
                "closed vocabulary-head one-module/function/record capacity",
            ),
            Self::FirstAttentionCaptured => (
                9,
                13,
                20,
                "closed first-attention nine-module/thirteen-function/twenty-record capacity",
            ),
            Self::ProjectionSix => (
                6,
                8,
                19,
                "closed cubin six-module/eight-function/nineteen-record capacity",
            ),
            Self::ConnectedQwen08 => (
                7,
                12,
                21,
                "closed connected qwen08 seven-module/twelve-function/twenty-one-record capacity",
            ),
        };
        if modules > max_modules
            || functions > max_functions
            || records >= max_records
            || (new_module && modules == max_modules)
            || (new_function && functions == max_functions)
        {
            return Err(failure(message));
        }
        Ok(())
    }

    pub(crate) fn require_manifest(
        self,
        image: &VerifiedCubin,
        manifest: &CubinKernelManifest,
    ) -> Result<(), ServerError> {
        if matches!(
            self,
            Self::Text08
                | Self::Text08Cohort4
                | Self::Text2B
                | Self::Text2BCachedCohort4
        ) {
            return super::text_policy::require(self, image, manifest);
        }
        if self == Self::FixedPromptNatural {
            return super::fixed_prompt_natural_policy::require(image, manifest);
        }
        if self == Self::FixedPromptFull {
            return super::fixed_prompt_full_policy::require(image, manifest);
        }
        if self == Self::Qwen2bMlp {
            return super::qwen2b_mlp_policy::require(image, manifest);
        }
        if self == Self::FixedPromptLayer3 {
            return super::fixed_prompt_layer3_policy::require(image, manifest);
        }
        if self == Self::FixedPromptFa {
            return super::fixed_prompt_fa_policy::require(image, manifest);
        }
        if self == Self::FixedPromptQkKv {
            return super::fixed_prompt_qkkv_policy::require(image, manifest);
        }
        if self == Self::FixedPromptQkv {
            return super::fixed_prompt_qkv_policy::require(image, manifest);
        }
        if self == Self::FixedPromptMlp {
            return super::fixed_prompt_mlp_policy::require(image, manifest);
        }
        if self == Self::FullQwen08 {
            return super::full_model_policy::require(image, manifest);
        }
        if self == Self::SourceRotary {
            return super::source_rotary_policy::require(image, manifest);
        }
        if self == Self::FirstAttentionComputedGate {
            return super::attention_gate_policy::require(image, manifest);
        }
        if self == Self::VocabularyHead {
            return super::head_policy::require(image, manifest);
        }
        if self == Self::FirstAttentionCaptured {
            return super::attention_policy::require(image, manifest);
        }
        if self == Self::ProjectionSix {
            // Preserve existing unsafe caller contract and original image routes.
            return Ok(());
        }
        let kind = ConnectedImage::from_provenance(image.derived_from(), image.captured_from())?;
        // Same closed source record as the image's original seal; never admit a
        // raw provider interval or caller-chosen expected hash as connected.
        match image.derived_from() {
            Some(derived) => {
                derived.require_image(image.byte_len(), image.sha256(), image.binary_sm())?
            }
            None => image
                .captured_from()
                .ok_or_else(|| failure("connected captured provenance absent"))?
                .require_image(image.byte_len(), image.sha256(), image.binary_sm())?,
        }
        require_connected_manifest(kind, manifest)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConnectedImage {
    Gemv2T,
    Gemvx,
    Cutlass,
    Reduce,
    Ampere128,
    Ampere64,
    Activation,
}
impl ConnectedImage {
    fn from_provenance(
        derived: Option<PinnedBf16Cubin>,
        captured: Option<PinnedCapturedBf16Cubin>,
    ) -> Result<Self, ServerError> {
        use PinnedBf16Cubin as D;
        use PinnedCapturedBf16Cubin as C;
        match (derived, captured) {
            (Some(D::Bf16Gemv2TSm86), None) => Ok(Self::Gemv2T),
            (Some(D::Bf16GemvxSm86), None) => Ok(Self::Gemvx),
            (Some(D::Bf16SplitKReduceSm86), None) => Ok(Self::Reduce),
            (Some(D::AmpereBf16Gemm128x64Sm80), None) => Ok(Self::Ampere128),
            (Some(D::AmpereBf16Gemm64x64Stages64x5Sm80), None) => Ok(Self::Ampere64),
            (None, Some(C::CutlassWmmaBf16Sm80)) => Ok(Self::Cutlass),
            (None, Some(C::SiluAndMulBf16Sm86)) => Ok(Self::Activation),
            _ => Err(failure(
                "connected qwen08 exact seven sealed image provenance kinds",
            )),
        }
    }
    const fn binary_sm(self) -> u8 {
        match self {
            Self::Gemv2T | Self::Gemvx | Self::Reduce | Self::Activation => 86,
            Self::Cutlass | Self::Ampere128 | Self::Ampere64 => 80,
        }
    }
}

struct EntryRule {
    image: ConnectedImage,
    entry: &'static str,
    params: &'static [(usize, usize)],
    shared: u32,
}

// Pinned executed original GDN19 +MLP13 +activation5 assets. Full names;
// image-label substrings are not function/ABI authority.
const RULES: &[EntryRule] = &[
    EntryRule {
        image: ConnectedImage::Gemv2T,
        entry: "_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_",
        params: &[(0, 152), (152, 4), (156, 4)],
        shared: 0,
    },
    EntryRule {
        image: ConnectedImage::Gemvx,
        entry: "_ZN8internal5gemvx6kernelIii13__nv_bfloat16S2_S2_fLb0ELb1ELb1ELb0ELi7ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS2_ES6_S4_IS2_EfEEENSt9enable_ifIXntT5_EvE4typeET11_",
        params: &[(0, 152)],
        shared: 528,
    },
    EntryRule {
        image: ConnectedImage::Cutlass,
        entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE",
        params: &[(0, 360)],
        shared: 16896,
    },
    EntryRule {
        image: ConnectedImage::Reduce,
        entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_",
        params: &[
            (0, 60),
            (64, 8),
            (72, 8),
            (80, 8),
            (88, 8),
            (96, 8),
            (104, 8),
            (112, 8),
            (120, 8),
            (128, 8),
            (136, 8),
            (144, 8),
            (152, 8),
            (160, 8),
            (168, 8),
            (176, 8),
            (184, 8),
            (192, 8),
            (200, 8),
            (208, 8),
            (216, 8),
        ],
        shared: 0,
    },
    EntryRule {
        image: ConnectedImage::Ampere128,
        entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn",
        params: &[(0, 176)],
        shared: 0,
    },
    EntryRule {
        image: ConnectedImage::Cutlass,
        entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE",
        params: &[(0, 360)],
        shared: 33792,
    },
    EntryRule {
        image: ConnectedImage::Ampere64,
        entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn",
        params: &[(0, 192)],
        shared: 32768,
    },
    EntryRule {
        image: ConnectedImage::Cutlass,
        entry: "_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8EEvNT_6ParamsE",
        params: &[(0, 360)],
        shared: 8704,
    },
    EntryRule {
        image: ConnectedImage::Ampere64,
        entry: "ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_tn",
        params: &[(0, 176)],
        shared: 32768,
    },
    EntryRule {
        image: ConnectedImage::Ampere128,
        entry: "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn",
        params: &[(0, 192)],
        shared: 24576,
    },
    EntryRule {
        image: ConnectedImage::Ampere64,
        entry: "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn",
        params: &[(0, 192)],
        shared: 49152,
    },
    EntryRule {
        image: ConnectedImage::Activation,
        entry: "_ZN4vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_bfloat162XadL_ZNS_11silu_kernelIS2_EET_RKS5_fEEXadL_ZNS_18packed_silu_kernelIS3_EES5_S7_fEELb1ELb1ELb0ELb0EEEvPS5_PS6_ifff",
        params: &[(0, 8), (8, 8), (16, 4), (20, 4), (24, 4), (28, 4)],
        shared: 0,
    },
];

fn require_connected_manifest(
    image: ConnectedImage,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    let rule = RULES
        .iter()
        .find(|rule| rule.image == image && rule.entry == manifest.entry)
        .ok_or_else(|| failure("connected qwen08 exact pinned image/function pair"))?;
    if manifest.binary_sm != image.binary_sm()
        || manifest.dynamic_shared_bytes != rule.shared
        || manifest.parameters.len() != rule.params.len()
        || manifest
            .parameters
            .iter()
            .zip(rule.params)
            .any(|(parameter, &(offset, size))| {
                parameter.offset != offset || parameter.template.len() != size
            })
    {
        return Err(failure(
            "connected qwen08 exact reflected layout/shared allowance",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
