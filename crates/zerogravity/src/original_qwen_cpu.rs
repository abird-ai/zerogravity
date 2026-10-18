//! Cold source-bound Qwen original text CPU value adapters.
//!
//! This deliberately does NOT implement a second token loop, Qwen GDN math,
//! attention, tokenization, training-precision rounding or session state.
//! It connects authenticated original HF tensors to the SAME source-typed
//! CPU project/embed and quant source decoder used by existing GGUF math.
//! A compiled original HF tensor layout is NOT a runnable model.
use crate::accelerator::MatVecKernel;
use crate::model::qwen_owned::QwenOriginalText;
use crate::quant::dequantize;
use crate::{Error, Result};
use zerogravity_artifacts::SourceWeights;

/// One borrowed verified Safetensors allocation and exact Qwen text geometry.
/// This does not mint any new model/source identity or resource reservation.
pub struct QwenOriginalCpu<'a> {
    bound: QwenOriginalText<'a>,
}
impl<'a> QwenOriginalCpu<'a> {
    pub fn bind(
        source: &'a SourceWeights,
        config_json: &[u8],
        expected_config_sha256: &str,
        expected_index_sha256: &str,
    ) -> Result<Self> {
        Ok(Self {
            bound: QwenOriginalText::bind(
                source,
                config_json,
                expected_config_sha256,
                expected_index_sha256,
            )?,
        })
    }
    pub fn binding(&self) -> &QwenOriginalText<'a> {
        &self.bound
    }

    /// Original [output,input] HF matrix with unchanged packed source bytes.
    /// This calls the existing qualified CPU/source projection seam and
    /// source-typed optional physical provider, not a new runtime kernel.
    pub fn project(
        &self,
        tensor_name: &str,
        input: &[f32],
        provider: Option<&dyn MatVecKernel>,
    ) -> Result<Vec<f32>> {
        let tensor = self.bound.text_tensor(tensor_name)?;
        crate::original_hf::output_first_projection(
            self.bound.source_identity(),
            tensor_name,
            tensor,
            input,
            provider,
        )
    }

    /// Contiguous original HF token embedding [vocab,hidden], borrowed from
    /// the same immutable verified source. No whole weight transposition.
    pub fn embed(&self, token: u32) -> Result<Vec<f32>> {
        crate::weights::source_reference_embedding(self.bound.embedding_region(token)?)
    }

    /// Exact original source vector (norm/bias/A_log) decode. FP32 CPU
    /// interpretation only; original framework intermediate-rounding remains
    /// a separate numerical acceptance gate.
    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.bound.text_tensor(name)?;
        if tensor.shape.len() != 1 {
            return Err(Error::Malformed(
                "original Qwen parameter vector rank invalid".into(),
            ));
        }
        let count = usize::try_from(tensor.shape[0])
            .map_err(|_| Error::Limit("original Qwen parameter vector width".into()))?;
        if count > 1_000_000 {
            return Err(Error::Limit(
                "original Qwen parameter vector exceeds limit".into(),
            ));
        }
        dequantize(tensor.kind, tensor.packed_bytes, count)
    }
}
