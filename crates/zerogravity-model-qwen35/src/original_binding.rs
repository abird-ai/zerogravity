//! Original HF source roles in the existing loaded owner; execution stays closed.

use crate::accelerator::program::{MatrixAxes, SourceTensor};
use crate::{Error, Result};
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
use zerogravity_core::quant::QuantKind;

#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
mod raw_state;
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
pub use raw_state::{
    OriginalStateDescriptor, OriginalStateEncoding, OriginalStateLayout, OriginalStatePhase,
    OriginalStateView,
};

/// Numerical semantics are independent of the packed source encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenNumericalProfile {
    GgufSourceV1,
    /// Reserved mixed-source CPU target. No trained execution is admitted.
    /// Preserves original F32 parameters; not the all-BF16 loader or CUDA FLA.
    OriginalHfTorchCpuSlowBf16Avx2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum QwenMatrixRole {
    Embedding,
    Output,
    FfnGate(usize),
    FfnUp(usize),
    FfnDown(usize),
    GdnQkv(usize),
    GdnGate(usize),
    GdnAlpha(usize),
    GdnBeta(usize),
    GdnOutput(usize),
    Query(usize),
    Key(usize),
    Value(usize),
    AttentionOutput(usize),
}
impl QwenMatrixRole {
    pub(crate) fn gguf_name(self) -> String {
        let (layer, suffix) = match self {
            Self::Embedding | Self::Output => return "token_embd.weight".into(),
            Self::FfnGate(i) => (i, "ffn_gate.weight"),
            Self::FfnUp(i) => (i, "ffn_up.weight"),
            Self::FfnDown(i) => (i, "ffn_down.weight"),
            Self::GdnQkv(i) => (i, "attn_qkv.weight"),
            Self::GdnGate(i) => (i, "attn_gate.weight"),
            Self::GdnAlpha(i) => (i, "ssm_alpha.weight"),
            Self::GdnBeta(i) => (i, "ssm_beta.weight"),
            Self::GdnOutput(i) => (i, "ssm_out.weight"),
            Self::Query(i) => (i, "attn_q.weight"),
            Self::Key(i) => (i, "attn_k.weight"),
            Self::Value(i) => (i, "attn_v.weight"),
            Self::AttentionOutput(i) => (i, "attn_output.weight"),
        };
        crate::layer_tensor(layer, suffix)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum QwenParameterRole {
    FinalNorm,
    InputNorm(usize),
    FfnNorm(usize),
    GdnConvolution(usize),
    GdnNorm(usize),
    GdnA(usize),
    GdnDt(usize),
    QueryNorm(usize),
    KeyNorm(usize),
}
impl QwenParameterRole {
    fn gguf_name(self) -> String {
        let (layer, suffix) = match self {
            Self::FinalNorm => return "output_norm.weight".into(),
            Self::InputNorm(i) => (i, "attn_norm.weight"),
            Self::FfnNorm(i) => (i, "post_attention_norm.weight"),
            Self::GdnConvolution(i) => (i, "ssm_conv1d.weight"),
            Self::GdnNorm(i) => (i, "ssm_norm.weight"),
            Self::GdnA(i) => (i, "ssm_a"),
            Self::GdnDt(i) => (i, "ssm_dt.bias"),
            Self::QueryNorm(i) => (i, "attn_q_norm.weight"),
            Self::KeyNorm(i) => (i, "attn_k_norm.weight"),
        };
        crate::layer_tensor(layer, suffix)
    }
}

/// Metadata of the unchanged learned tensor, including its original dtype.
pub type QwenParameterDescriptor = zerogravity_core::accelerator::program::SourceParameter;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OriginalGdnGeometry {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub convolution_kernel: usize,
}

/// Owned small metadata only. Original learned payload remains in SourceWeights.
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
pub struct OriginalQwenBinding {
    config: zerogravity_artifacts::model::qwen::QwenTextConfig,
    matrices: std::collections::BTreeMap<QwenMatrixRole, SourceTensor>,
    parameters: std::collections::BTreeMap<QwenParameterRole, QwenParameterDescriptor>,
    gdn: OriginalGdnGeometry,
    config_sha256: String,
    index_sha256: String,
    bundle_sha256: String,
    text_tensors: usize,
    other_tensors: usize,
}
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl OriginalQwenBinding {
    fn bind(
        source: &zerogravity_artifacts::SourceWeights,
        config_json: &[u8],
        config_sha256: &str,
        index_sha256: &str,
    ) -> Result<Self> {
        use QwenMatrixRole as M;
        use QwenParameterRole as P;
        use zerogravity_artifacts::model::qwen_owned::QwenOriginalText;
        let checked = QwenOriginalText::bind(source, config_json, config_sha256, index_sha256)?;
        let mut matrices = std::collections::BTreeMap::new();
        let mut matrix = |role, name: String| -> Result<()> {
            checked.text_tensor(&name)?;
            let tensor = SourceTensor::from_source(source, &name, MatrixAxes::OutputInput)?;
            if tensor.kind != QuantKind::Bf16 {
                return Err(Error::Unsupported(
                    "original HF CPU BF16 target requires original BF16 matrices".into(),
                ));
            }
            matrices.insert(role, tensor);
            Ok(())
        };
        matrix(
            M::Embedding,
            "model.language_model.embed_tokens.weight".into(),
        )?;
        matrix(
            M::Output,
            if checked.config().tie_word_embeddings() {
                "model.language_model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
            .into(),
        )?;
        let mut parameters = std::collections::BTreeMap::new();
        let mut parameter = |role, name: String| -> Result<()> {
            checked.text_tensor(&name)?;
            parameters.insert(role, QwenParameterDescriptor::from_source(source, &name)?);
            Ok(())
        };
        parameter(P::FinalNorm, "model.language_model.norm.weight".into())?;
        for (i, kind) in checked.config().geometry().layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{i}");
            parameter(P::InputNorm(i), format!("{prefix}.input_layernorm.weight"))?;
            parameter(
                P::FfnNorm(i),
                format!("{prefix}.post_attention_layernorm.weight"),
            )?;
            for (role, suffix) in [
                (M::FfnGate(i), "gate_proj"),
                (M::FfnUp(i), "up_proj"),
                (M::FfnDown(i), "down_proj"),
            ] {
                matrix(role, format!("{prefix}.mlp.{suffix}.weight"))?;
            }
            match kind {
                crate::model::LayerKind::LinearAttention => {
                    let prefix = format!("{prefix}.linear_attn");
                    for (role, suffix) in [
                        (M::GdnQkv(i), "in_proj_qkv"),
                        (M::GdnGate(i), "in_proj_z"),
                        (M::GdnAlpha(i), "in_proj_a"),
                        (M::GdnBeta(i), "in_proj_b"),
                        (M::GdnOutput(i), "out_proj"),
                    ] {
                        matrix(role, format!("{prefix}.{suffix}.weight"))?;
                    }
                    for (role, suffix) in [
                        (P::GdnConvolution(i), "conv1d.weight"),
                        (P::GdnNorm(i), "norm.weight"),
                        (P::GdnA(i), "A_log"),
                        (P::GdnDt(i), "dt_bias"),
                    ] {
                        parameter(role, format!("{prefix}.{suffix}"))?;
                    }
                }
                crate::model::LayerKind::FullAttention => {
                    let prefix = format!("{prefix}.self_attn");
                    for (role, suffix) in [
                        (M::Query(i), "q_proj"),
                        (M::Key(i), "k_proj"),
                        (M::Value(i), "v_proj"),
                        (M::AttentionOutput(i), "o_proj"),
                    ] {
                        matrix(role, format!("{prefix}.{suffix}.weight"))?;
                    }
                    parameter(P::QueryNorm(i), format!("{prefix}.q_norm.weight"))?;
                    parameter(P::KeyNorm(i), format!("{prefix}.k_norm.weight"))?;
                }
                crate::model::LayerKind::SlidingAttention => {
                    return Err(Error::Unsupported("original Qwen sliding binding".into()));
                }
            }
        }
        let config = zerogravity_artifacts::model::qwen::QwenTextConfig::from_json(config_json)?;
        let size = |n| {
            usize::try_from(n).map_err(|_| Error::Limit("original HF geometry overflow".into()))
        };
        let gdn = OriginalGdnGeometry {
            key_heads: size(config.geometry().recurrent_key_heads.unwrap_or(0))?,
            value_heads: size(config.geometry().recurrent_value_heads.unwrap_or(0))?,
            key_width: size(config.linear_key_head_dim())?,
            value_width: size(config.linear_value_head_dim())?,
            convolution_kernel: size(config.linear_conv_kernel_dim())?,
        };
        Ok(Self {
            config,
            matrices,
            parameters,
            gdn,
            config_sha256: checked.config_sha256().into(),
            index_sha256: checked.verified_index_sha256().into(),
            bundle_sha256: checked.source_bundle_sha256().into(),
            text_tensors: checked.text_tensor_count(),
            other_tensors: checked.other_tensor_count(),
        })
    }
    pub fn config(&self) -> &zerogravity_artifacts::model::qwen::QwenTextConfig {
        &self.config
    }
    pub fn gdn_geometry(&self) -> OriginalGdnGeometry {
        self.gdn
    }
    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }
    pub fn index_sha256(&self) -> &str {
        &self.index_sha256
    }
    pub fn bundle_sha256(&self) -> &str {
        &self.bundle_sha256
    }
    pub fn text_tensor_count(&self) -> usize {
        self.text_tensors
    }
    pub fn other_tensor_count(&self) -> usize {
        self.other_tensors
    }
}

impl crate::LoadedQwenModel {
    pub fn numerical_profile(&self) -> QwenNumericalProfile {
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        if self.original.is_some() {
            return QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2;
        }
        QwenNumericalProfile::GgufSourceV1
    }
    pub(crate) fn require_execution_profile(&self) -> Result<()> {
        if self.numerical_profile() != QwenNumericalProfile::GgufSourceV1 {
            return Err(Error::Unsupported(
                "original HF source bound; numerical profile/plan operators remain unqualified"
                    .into(),
            ));
        }
        Ok(())
    }
    pub fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        if let Some(original) = &self.original {
            return original.matrices.get(&role).cloned().ok_or_else(|| {
                Error::Unsupported("matrix role absent from original source layer".into())
            });
        }
        SourceTensor::from_source(self, &role.gguf_name(), MatrixAxes::InputOutput)
    }
    pub fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<QwenParameterDescriptor> {
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        if let Some(original) = &self.original {
            return original.parameters.get(&role).cloned().ok_or_else(|| {
                Error::Unsupported("parameter role absent from original source layer".into())
            });
        }
        QwenParameterDescriptor::from_source(self, &role.gguf_name())
    }
    /// Borrow the original role's unchanged dtype, raw axes and learned bytes.
    /// No conversion, numerical admission, learned cache or state allocation.
    pub fn parameter_source(
        &self,
        role: QwenParameterRole,
    ) -> Result<zerogravity_core::source::SourceTensorRef<'_>> {
        let descriptor = self.parameter_descriptor(role)?;
        descriptor.validate_source(&self.weights)?;
        zerogravity_core::source::TensorSource::tensor(&self.weights, &descriptor.name)
    }
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    pub fn original_binding(&self) -> Option<&OriginalQwenBinding> {
        self.original.as_ref()
    }
    /// Cold metadata binding only; never admits CPU/GPU/session execution.
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    pub fn from_original_source(
        weights: zerogravity_artifacts::SourceWeights,
        config_json: &[u8],
        config_sha256: &str,
        index_sha256: &str,
        profile: QwenNumericalProfile,
    ) -> Result<std::sync::Arc<Self>> {
        if profile != QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2 {
            return Err(Error::Unsupported(
                "original source needs explicit HF profile".into(),
            ));
        }
        let original =
            OriginalQwenBinding::bind(&weights, config_json, config_sha256, index_sha256)?;
        let geometry = crate::model::ModelGeometry::from_json(config_json)?;
        let g = original.gdn;
        let overflow = || Error::Limit("original HF state geometry overflow".into());
        let key = g.key_heads.checked_mul(g.key_width).ok_or_else(overflow)?;
        let value = g
            .value_heads
            .checked_mul(g.value_width)
            .ok_or_else(overflow)?;
        let channels = key
            .checked_mul(2)
            .and_then(|x| x.checked_add(value))
            .ok_or_else(overflow)?;
        let conv = channels
            .checked_mul(g.convolution_kernel.checked_sub(1).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        let state = g
            .value_heads
            .checked_mul(g.key_width)
            .and_then(|x| x.checked_mul(g.value_width))
            .ok_or_else(overflow)?;
        let kv_width = usize::try_from(
            geometry
                .num_key_value_heads
                .checked_mul(geometry.head_dim)
                .ok_or_else(overflow)?,
        )
        .map_err(|_| overflow())?;
        let state_specs = geometry
            .layers
            .iter()
            .map(|kind| match kind {
                crate::model::LayerKind::LinearAttention => {
                    Ok(crate::StateSpec::Delta { conv, state })
                }
                crate::model::LayerKind::FullAttention => Ok(crate::StateSpec::Full { kv_width }),
                crate::model::LayerKind::SlidingAttention => {
                    Err(Error::Unsupported("original Qwen sliding state".into()))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let eps = original.config.rms_norm_eps() as f32;
        Ok(std::sync::Arc::new(Self {
            weights,
            geometry,
            // No learned numerical caches or zero-state allocations before qualification.
            layers: Vec::new(),
            final_norm: Vec::new(),
            state_specs,
            original: Some(original),
            #[cfg(feature = "cpu")]
            initial_layers: std::sync::OnceLock::new(),
            eps,
        }))
    }
}

#[cfg(all(test, feature = "original-hf-binding", target_os = "linux"))]
#[path = "original_binding_tests.rs"]
mod tests;
