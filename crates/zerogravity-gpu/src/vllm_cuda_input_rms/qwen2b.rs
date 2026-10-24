//! Closed original2B RMS diagnostics. No model/profile/publication permit.
use super::{Columns, FusedRmsPrepared, RmsPrepared, Rows, resident::finite_bf16};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::MatrixAxes;
use zerogravity_core::quant::QuantKind;
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

const CONFIG: &str = "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4";
const INDEX: &str = "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9";
const SOURCE_BYTES: u64 = 4_548_221_488;
const CHUNK: usize = 1024 * 1024;
const COUNTS: [usize; 5] = [1, 1, 64, 2, 1];
const FINAL_GAMMA: &str = "model.language_model.norm.weight";

#[derive(Clone)]
pub(super) struct Qwen2bRmsSource {
    model: Arc<OriginalWindowModelSource>,
}
impl Qwen2bRmsSource {
    fn bind(model: Arc<OriginalWindowModelSource>) -> Result<Self> {
        let owner = model.owner();
        require_pins(
            owner.config_sha256(),
            owner.verified_index_sha256(),
            model.source().source_bytes(),
            model.source().staging_chunk_bytes(),
        )?;
        let config = owner.config();
        let geometry = model.geometry();
        if geometry.family != zerogravity_artifacts::model::Family::QwenHybrid
            || geometry.hidden_size != Columns::Qwen2b.count() as u64
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
        Ok(Self { model })
    }
    pub(super) fn source(&self) -> &VerifiedTensorSource {
        self.model.source()
    }
    pub(super) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if !self.source().same_issuer(source)
            || self.model.source_identity() != source.source_identity()
        {
            return Err(Error::State("original2B RMS source issuer differs".into()));
        }
        require_pins(
            self.model.owner().config_sha256(),
            self.model.owner().verified_index_sha256(),
            source.source_bytes(),
            source.staging_chunk_bytes(),
        )
    }
}
fn require_pins(config: &str, index: &str, bytes: u64, chunk: usize) -> Result<()> {
    if config != CONFIG || index != INDEX || bytes != SOURCE_BYTES || chunk != CHUNK {
        return Err(Error::Unsupported(
            "closed original2B config/index/extent/1MiB source opening".into(),
        ));
    }
    Ok(())
}
pub(super) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("original2B RMS cancelled".into()));
    }
    Ok(())
}

/// Observed final branch/residual inputs are explicitly LEAF operands. They
/// establish neither source-produced upstream layers nor a wholemodel route.
pub struct Qwen2bDualRmsLeaf {
    pub branch: Vec<u8>,
    pub residual: Vec<u8>,
}

/// Expected archives may be opened only by this post-completion callback.
pub enum Qwen2bRmsReadback<'a> {
    SourceInput {
        case: usize,
        teacher_ids: &'a [u32],
        embedding: &'a [u8],
        normalized: &'a [u8],
    },
    FinalDualLeaf {
        case: usize,
        sum_f32: &'a [u8],
        normalized: &'a [u8],
        residual: &'a [u8],
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen2bRmsReceipt {
    pub source_embedding_bf16_words: usize,
    pub ordinary_bf16_words: usize,
    pub dual_sum_f32_words: usize,
    pub dual_normalized_bf16_words: usize,
    pub dual_residual_bf16_words: usize,
    pub actual_stage_dispatches: usize,
}

#[path = "qwen2b/qualification.rs"]
mod qualification;
pub use qualification::qualify_qwen2b_rms;

#[cfg(test)]
#[path = "qwen2b/tests.rs"]
mod tests;

#[path = "qwen2b/faults.rs"]
mod faults;
pub use faults::{Qwen2bRmsFault, Qwen2bRmsFaultOutcome, qualify_qwen2b_rms_fault};

#[path = "qwen2b/component.rs"]
mod component;
