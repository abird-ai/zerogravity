//! Model-owned exact RMS source policies; observed leaves imply no full-model execution.
// Former binders: FixedPromptRmsSource, Qwen2bRmsSource; final source policy from BoundFinalRmsSource.
use super::*;
use zerogravity_core::accelerator::program::{
    SourceGamma, SourceNumericalContract, SourceRmsGeometry,
};
use zerogravity_core::quant::QuantKind;
const CONFIG: &str = "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4";
const INDEX: &str = "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9";
const SOURCE_BYTES: u64 = 4_548_221_488;
const CHUNK: usize = 1024 * 1024;
const COUNTS: [usize; 5] = [1, 1, 64, 2, 1];
const FINAL_GAMMA: &str = "model.language_model.norm.weight";

const COUNT: usize = 47;
const FIXED_SOURCE_BYTES: u64 = 1_746_942_600;
const FIXED_SOURCE_CHUNK: usize = 1024 * 1024;
const PROMPT: [u32; COUNT] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];

/// Authentic family source/roles and charged cold metadata; no device or root authority.
pub struct PreparedRmsSource {
    model: Arc<OriginalWindowModelSource>,
    profile: QwenSourceProfile,
    embedding: SourceTensor,
    input_gamma: SourceParameter,
    final_gamma: SourceParameter,
    metadata: Arc<WindowWorkspace>,
}
impl PreparedRmsSource {
    pub fn source(&self) -> &VerifiedTensorSource {
        self.model.source()
    }
    pub fn embedding(&self) -> &SourceTensor {
        &self.embedding
    }
    pub fn input_gamma(&self) -> &SourceParameter {
        &self.input_gamma
    }
    pub fn final_gamma(&self) -> &SourceParameter {
        &self.final_gamma
    }
    pub fn metadata(&self) -> Arc<WindowWorkspace> {
        self.metadata.clone()
    }
    pub fn geometry(&self) -> SourceRmsGeometry {
        SourceRmsGeometry {
            width: self.embedding.columns,
            epsilon: 1e-6,
            gamma: SourceGamma::OnePlus,
        }
    }
    pub fn numerical_contract(&self) -> SourceNumericalContract {
        SourceNumericalContract::ReferenceCudaSm86Bf16V1
    }
    pub fn row_counts(&self) -> &[usize] {
        match self.profile {
            QwenSourceProfile::Qwen08 => &[47],
            QwenSourceProfile::Qwen2B => &COUNTS,
        }
    }
    pub fn require_ids(&self, ids: &[u32]) -> Result<()> {
        let valid = match self.profile {
            QwenSourceProfile::Qwen08 => ids == PROMPT,
            QwenSourceProfile::Qwen2B => {
                matches!(ids.len(), 1 | 2 | 64) && ids.iter().all(|id| *id < 248320)
            }
        };
        if !valid {
            return Err(Error::Unsupported(
                "RMS exact family source row policy differs".into(),
            ));
        }
        Ok(())
    }
}
impl OriginalWindowModelSource {
    pub fn prepare_rms_source(
        self: &Arc<Self>,
        profile: QwenSourceProfile,
    ) -> Result<PreparedRmsSource> {
        let metadata = Arc::new(self.source().reserve_workspace(64 * 1024)?);
        match profile {
            QwenSourceProfile::Qwen08 => require_fixed_source(self)?,
            QwenSourceProfile::Qwen2B => require_reference_source(self)?,
        }
        let embedding = self.matrix_descriptor(QwenMatrixRole::Embedding)?;
        let input_gamma = self.parameter_descriptor(QwenParameterRole::InputNorm(0))?;
        let final_gamma = self.parameter_descriptor(QwenParameterRole::FinalNorm)?;
        self.validate_matrix(QwenMatrixRole::Embedding, &embedding)?;
        self.validate_parameter(QwenParameterRole::InputNorm(0), &input_gamma)?;
        self.validate_parameter(QwenParameterRole::FinalNorm, &final_gamma)?;
        Ok(PreparedRmsSource {
            model: self.clone(),
            profile,
            embedding,
            input_gamma,
            final_gamma,
            metadata,
        })
    }
}
/// Original final-only role/config contract; no embedding/input or full source-pin requirement.
pub struct PreparedFinalRmsSource {
    model: Arc<OriginalWindowModelSource>,
    parameter: SourceParameter,
    metadata: Arc<WindowWorkspace>,
}
impl PreparedFinalRmsSource {
    pub fn source(&self) -> &VerifiedTensorSource {
        self.model.source()
    }
    pub fn parameter(&self) -> &SourceParameter {
        &self.parameter
    }
    pub fn geometry(&self) -> SourceRmsGeometry {
        SourceRmsGeometry {
            width: 1024,
            epsilon: 1e-6,
            gamma: SourceGamma::OnePlus,
        }
    }
    pub fn numerical_contract(&self) -> SourceNumericalContract {
        SourceNumericalContract::ReferenceCudaSm86Bf16V1
    }
    pub fn metadata(&self) -> Arc<WindowWorkspace> {
        self.metadata.clone()
    }
}
impl OriginalWindowModelSource {
    pub fn prepare_final_rms_source(self: &Arc<Self>) -> Result<PreparedFinalRmsSource> {
        let metadata = Arc::new(self.source().reserve_workspace(64 * 1024)?);
        let parameter = require_final_parameter(self)?;
        Ok(PreparedFinalRmsSource {
            model: self.clone(),
            parameter,
            metadata,
        })
    }
}
fn require_reference_source(model: &OriginalWindowModelSource) -> Result<()> {
    let owner = model.owner();
    require_pins(
        owner.config_sha256(),
        owner.verified_index_sha256(),
        model.source().source_bytes(),
        model.source().staging_chunk_bytes(),
    )?;
    let config = owner.config();
    let geometry = model.geometry();
    if geometry.family != crate::config::Family::QwenHybrid
        || geometry.hidden_size != 2048usize as u64
        || geometry.vocab_size != 248320
        || geometry.layers.len() != 24
        || geometry.head_dim != 256
        || geometry.num_attention_heads != 8
        || geometry.num_key_value_heads != 2
        || geometry.recurrent_key_heads != Some(16)
        || geometry.recurrent_value_heads != Some(16)
        || config.intermediate_size() != 6144
        || config.linear_key_head_dim() != 128
        || config.linear_value_head_dim() != 128
        || config.linear_conv_kernel_dim() != 4
        || config.rms_norm_eps().to_bits() != 1e-6f64.to_bits()
        || !config.tie_word_embeddings()
        || owner.text_tensor_count() != 320
    {
        return Err(Error::Unsupported(
            "closed original2B RMS config/geometry".into(),
        ));
    }
    // Authenticated original names/axes/dtypes, never fabricated packed roles.
    let matrix = model.source().matrix(
        "model.language_model.embed_tokens.weight",
        MatrixAxes::OutputInput,
    )?;
    if matrix.tensor().kind != QuantKind::Bf16
        || matrix.rows() != 248320
        || matrix.columns() != 2048
    {
        return Err(Error::Unsupported(
            "original2B BF16 embedding geometry".into(),
        ));
    }
    for name in [
        "model.language_model.layers.0.input_layernorm.weight",
        FINAL_GAMMA,
    ] {
        let parameter = model.source().parameter(name)?;
        if parameter.tensor().kind != QuantKind::Bf16
            || parameter.tensor().shape != [2048]
            || parameter.elements() != 2048
        {
            return Err(Error::Unsupported(
                "original2B BF16 raw gamma role/shape".into(),
            ));
        }
    }
    Ok(())
}
fn require_fixed_source(model: &OriginalWindowModelSource) -> Result<()> {
    model.require_small_model_source()?;
    require_fixed_extent(
        model.source().source_bytes(),
        model.source().staging_chunk_bytes(),
    )?;
    let embedding = model.matrix_descriptor(QwenMatrixRole::Embedding)?;
    let gamma = model.parameter_descriptor(QwenParameterRole::InputNorm(0))?;
    if embedding.kind != QuantKind::Bf16
        || embedding.rows != 248320
        || embedding.columns != 1024
        || gamma.kind != QuantKind::Bf16
        || gamma.name != "model.language_model.layers.0.input_layernorm.weight"
        || gamma.shape != [1024 as u64]
        || gamma.bytes != 1024 * 2
    {
        return Err(Error::Unsupported(
            "fixed_prompt original BF16 embedding/gamma".into(),
        ));
    }
    model.validate_matrix(QwenMatrixRole::Embedding, &embedding)?;
    model.validate_parameter(QwenParameterRole::InputNorm(0), &gamma)?;
    require_final_parameter(model)?;
    Ok(())
}
fn require_final_parameter(model: &OriginalWindowModelSource) -> Result<SourceParameter> {
    let parameter = model.parameter_descriptor(QwenParameterRole::FinalNorm)?;
    if model.geometry().hidden_size != 1024
        || model.geometry().vocab_size != 248320
        || model.geometry().layers.len() != 24
        || model.owner().config().rms_norm_eps() != 1e-6
        || parameter.name != "model.language_model.norm.weight"
        || parameter.kind != QuantKind::Bf16
        || parameter.shape != [1024]
        || parameter.bytes != 2048
    {
        return Err(Error::Unsupported(
            "closed original .8B final RMS source/config".into(),
        ));
    }
    model.validate_parameter(QwenParameterRole::FinalNorm, &parameter)?;
    Ok(parameter)
}
fn require_pins(config: &str, index: &str, bytes: u64, chunk: usize) -> Result<()> {
    if config != CONFIG || index != INDEX || bytes != SOURCE_BYTES || chunk != CHUNK {
        return Err(Error::Unsupported(
            "closed original2B config/index/extent/1MiB source opening".into(),
        ));
    }
    Ok(())
}
fn require_fixed_extent(bytes: u64, chunk: usize) -> Result<()> {
    if bytes != FIXED_SOURCE_BYTES || chunk != FIXED_SOURCE_CHUNK {
        return Err(Error::Unsupported(
            "fixed_prompt original source extent/1MiB staging".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original2b_source_pins_cannot_retag_other_width_or_opening() -> Result<()> {
        require_pins(CONFIG, INDEX, SOURCE_BYTES, CHUNK)?;
        for wrong in [
            0,
            SOURCE_BYTES - 1,
            SOURCE_BYTES + 1,
            1_746_942_600,
            u64::MAX,
        ] {
            assert!(require_pins(CONFIG, INDEX, wrong, CHUNK).is_err());
        }
        for wrong in [0, CHUNK - 1, CHUNK + 1, 4 * CHUNK, usize::MAX] {
            assert!(require_pins(CONFIG, INDEX, SOURCE_BYTES, wrong).is_err());
        }
        for wrong in [
            "",
            "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
            INDEX,
        ] {
            assert!(require_pins(wrong, INDEX, SOURCE_BYTES, CHUNK).is_err());
        }
        for wrong in [
            "",
            "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
            CONFIG,
        ] {
            assert!(require_pins(CONFIG, wrong, SOURCE_BYTES, CHUNK).is_err());
        }
        Ok(())
    }

    #[test]
    fn fixed_source_staging_scope_never_inherits_other_extent() {
        assert!(require_fixed_extent(FIXED_SOURCE_BYTES, FIXED_SOURCE_CHUNK).is_ok());
        for (bytes, chunk) in [
            (4_548_221_488, FIXED_SOURCE_CHUNK),
            (FIXED_SOURCE_BYTES, 4 * FIXED_SOURCE_CHUNK),
            (0, FIXED_SOURCE_CHUNK),
            (FIXED_SOURCE_BYTES, 0),
        ] {
            assert!(require_fixed_extent(bytes, chunk).is_err());
        }
    }
}
