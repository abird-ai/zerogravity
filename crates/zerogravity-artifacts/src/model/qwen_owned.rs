//! Original (not converted GGUF) Qwen3.8 dense-text source binding.
//!
//! One borrowed immutable SourceWeights allocation, the original pinned index
//! from that verified allocation, and caller-authorized full config SHA256
//! define the ordinary 851-tensor Qwen3.8-27B source namespace. The exact
//! QwenTextConfig geometries are reused; no second tensor parser, source
//! issuer, model executor or mathematical schedule is introduced.
//!
//! This is NOT a trained HF inference pathway: tokenization, BF16 numerical
//! schedules, GDN grouped Q/K→V packing, vision/MTP learned components,
//! session and GPU/SSD execution require independent source-real gates.

use super::qwen::{QwenTextConfig, QwenTextStateBytes};
use crate::quant::QuantKind;
use crate::source_weights::SourceWeights;
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use zerogravity_core::accelerator::{EmbeddingRegion, SourceIdentity};
use zerogravity_core::source::{SourceTensorRef, TensorSource};

const MAX_CONFIG: usize = 1024 * 1024;

/// Immutable source-bound *ordinary text* view. It retains NO duplicate
/// large model allocation; the borrow cannot outlive the owner/host-RAM lease.
pub struct QwenOriginalText<'a> {
    source: &'a SourceWeights,
    config: QwenTextConfig,
    text_names: BTreeSet<String>,
    expected_config_sha256: String,
    verified_index_sha256: String,
    other_tensor_count: usize,
}

impl<'a> QwenOriginalText<'a> {
    /// Enforce caller-authorized config and the pinned index of the *same*
    /// original owned shard allocation. Neither digest may be replaced by
    /// a synthetic alias to obtain a model eligibility claim.
    pub fn bind(
        source: &'a SourceWeights,
        config_json: &[u8],
        expected_config_sha256: &str,
        expected_index_sha256: &str,
    ) -> Result<Self> {
        crate::cache::paths::validate_digest(expected_config_sha256)?;
        crate::cache::paths::validate_digest(expected_index_sha256)?;
        if config_json.is_empty()
            || config_json.len() > MAX_CONFIG
            || format!("{:x}", Sha256::digest(config_json)) != expected_config_sha256
        {
            return Err(Error::Malformed(
                "original Qwen config content digest/length disagreement".into(),
            ));
        }
        if source.verified_shard_index_sha256() != Some(expected_index_sha256) {
            return Err(Error::Malformed(
                "Qwen config bound to a different or unverified source index".into(),
            ));
        }
        let config = QwenTextConfig::from_json(config_json)?;
        let specs = config.tensor_specs()?;
        let mut allowed = BTreeSet::new();
        for spec in specs {
            if !allowed.insert(spec.name.clone()) {
                return Err(Error::Malformed(
                    "duplicate source-specific Qwen tensor contract".into(),
                ));
            }
            let info = source.tensor(&spec.name)?;
            spec.validate_header(info)?;
            // Original BF16/F16/F32 is supported for these borrowed
            // source-read tensors. Quant variants have *distinct* encoding
            // semantics and must pass a separate original-weight profile.
            if !matches!(info.kind, QuantKind::Bf16 | QuantKind::F16 | QuantKind::F32) {
                return Err(Error::Unsupported(format!(
                    "unregistered Qwen original text tensor encoding: {}",
                    spec.name
                )));
            }
            TensorSource::tensor(source, &spec.name)?.validate()?;
        }
        let mut other = 0usize;
        for name in source.tensor_names() {
            if !allowed.contains(name) {
                // An extra learned model.language_model/lm_head tensor
                // cannot be treated as vision, MTP or arbitrary metadata.
                if name.starts_with("model.language_model.") || name.starts_with("lm_head.") {
                    return Err(Error::Malformed(format!(
                        "unregistered ordinary-text Qwen tensor: {name}"
                    )));
                }
                other += 1;
            }
        }
        Ok(Self {
            source,
            config,
            text_names: allowed,
            expected_config_sha256: expected_config_sha256.into(),
            verified_index_sha256: expected_index_sha256.into(),
            other_tensor_count: other,
        })
    }
    pub fn config(&self) -> &QwenTextConfig {
        &self.config
    }
    pub fn config_sha256(&self) -> &str {
        &self.expected_config_sha256
    }
    pub fn verified_index_sha256(&self) -> &str {
        &self.verified_index_sha256
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source.source_identity()
    }
    pub fn text_tensor_count(&self) -> usize {
        self.text_names.len()
    }
    pub fn other_tensor_count(&self) -> usize {
        self.other_tensor_count
    }
    pub fn numerical_state_bytes(&self, prefix: u64) -> Result<QwenTextStateBytes> {
        self.config.numerical_state_bytes(prefix)
    }
    /// Source tensor original order: a matrix has [output,input], unlike
    /// GGUF's [input,output]. Never mutate its original TensorInfo to
    /// pretend the weights have been transposed or permuted.
    pub fn text_tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        if !self.text_names.contains(name) {
            return Err(Error::Unsupported(
                "not an ordinary Qwen text tensor".into(),
            ));
        }
        TensorSource::tensor(self.source, name)
    }
    /// Build the existing *source-typed* CPU embedding view directly over
    /// original HF [vocab,hidden] bytes; this does not copy/transpose data.
    /// Embedding rows are complete contiguous packed vectors.
    pub fn embedding_region(&self, token: u32) -> Result<EmbeddingRegion<'_>> {
        let tensor = self.text_tensor("model.language_model.embed_tokens.weight")?;
        if tensor.shape.len() != 2 {
            return Err(Error::Malformed(
                "original Qwen embedding rank invalid".into(),
            ));
        }
        let vocabulary = usize::try_from(tensor.shape[0])
            .map_err(|_| Error::Limit("Qwen source vocabulary overflow".into()))?;
        let width = usize::try_from(tensor.shape[1])
            .map_err(|_| Error::Limit("Qwen source hidden width overflow".into()))?;
        let result = EmbeddingRegion {
            source_identity: self.source_identity(),
            tensor_name: "model.language_model.embed_tokens.weight",
            kind: tensor.kind,
            packed_weights: tensor.packed_bytes,
            token,
            width,
            vocabulary,
        };
        result.validate()?;
        Ok(result)
    }
}
