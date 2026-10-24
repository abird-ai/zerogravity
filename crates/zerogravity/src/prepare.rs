//! Cold preparation joins original source and model-family validation. It does
//! not acquire data, publish a session root, select a backend or run inference.
use crate::artifact::{ArtifactFormat, ArtifactInfo};
use crate::model::{Family, ModelGeometry};
use crate::qwen35::LoadedQwenModel;
use crate::tokenize::QwenTokenizer;
use crate::{Error, Result};
use std::path::Path;
use std::sync::Arc;

pub struct PreparedQwen {
    model: Arc<LoadedQwenModel>,
    tokenizer: QwenTokenizer,
}
impl PreparedQwen {
    /// Pure family routing metadata; no source authority or execution grant.
    pub fn routing_target(g: &ModelGeometry) -> Result<String> {
        if g.family == Family::QwenHybrid
            && g.layers.len() == 24
            && matches!(g.hidden_size, 1024 | 2048)
        {
            Ok(if g.hidden_size == 1024 {
                "qwen3.5-0.8b"
            } else {
                "qwen3.5-2b"
            }
            .into())
        } else {
            Err(Error::Unsupported("generic run CPU adapter currently supports Qwen3.5 0.8B/2B direct geometry; model metadata is not runtime eligibility".into()))
        }
    }
    pub fn routing_bytes(
        info: &ArtifactInfo,
        g: &ModelGeometry,
        tokenizer_bytes: u64,
        context: usize,
    ) -> Result<u64> {
        if info.format != ArtifactFormat::Gguf
            || info.architecture.as_deref() != Some("qwen35")
            || info.file_len > 4 * 1024u64.pow(3)
        {
            return Err(Error::Unsupported(
                "CPU run needs bounded original qwen35 GGUF".into(),
            ));
        }
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or_else(|| Error::Limit("source-shaped CPU planning overflow".into()))
        };
        let mut decoded = 0u64;
        for t in &info.tensors {
            let n = t
                .shape
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d))
                .ok_or_else(|| Error::Limit("tensor shape planning overflow".into()))?;
            decoded = add(
                decoded,
                n.checked_mul(4)
                    .ok_or_else(|| Error::Limit("tensor planning overflow".into()))?,
            )?;
        }
        // Conservative planning, distinct from the existing raw-source tickets.
        // Full decoded tensor population dominates transient matvec rows/caches;
        // the original engine still owns its state/transaction admission.
        let kv = (g
            .layers
            .iter()
            .filter(|kind| **kind == crate::model::LayerKind::FullAttention)
            .count() as u64)
            .checked_mul(g.num_key_value_heads)
            .and_then(|n| n.checked_mul(g.head_dim))
            .and_then(|n| n.checked_mul(context as u64))
            .and_then(|n| n.checked_mul(32))
            .ok_or_else(|| Error::Limit("state planning overflow".into()))?;
        let mut recurrent = 0u64;
        for (i, kind) in g.layers.iter().enumerate() {
            if *kind == crate::model::LayerKind::LinearAttention {
                let name = format!("blk.{i}.ssm_norm.weight");
                let norm = info
                    .tensors
                    .iter()
                    .find(|t| t.name == name && t.shape.len() == 1)
                    .ok_or_else(|| {
                        crate::Error::Malformed("GDN routing needs actual source norm width".into())
                    })?;
                let dim = norm.shape[0];
                let heads = g
                    .recurrent_value_heads
                    .ok_or_else(|| crate::Error::Malformed("GDN routing head count".into()))?;
                let bytes = heads
                    .checked_mul(dim)
                    .and_then(|n| n.checked_mul(dim.checked_add(9)?))
                    .and_then(|n| n.checked_mul(16))
                    .ok_or_else(|| Error::Limit("recurrent planning overflow".into()))?;
                recurrent = add(recurrent, bytes)?;
            }
        }
        let tokenizer = tokenizer_bytes
            .checked_mul(64)
            .ok_or_else(|| Error::Limit("tokenizer planning overflow".into()))?;
        add(
            add(
                add(
                    info.file_len,
                    decoded
                        .checked_mul(4)
                        .ok_or_else(|| Error::Limit("decoded planning overflow".into()))?,
                )?,
                add(kv, recurrent)?,
            )?,
            add(tokenizer, 512 * 1024 * 1024)?,
        )
    }

    /// Existing path preparation, preserving tokenizer-first validation order.
    pub fn from_paths(model: &Path, config: &Path, tokenizer: &Path) -> Result<Self> {
        let tokenizer = QwenTokenizer::from_file(tokenizer)?;
        let model = LoadedQwenModel::open(model, config)?;
        Ok(Self { model, tokenizer })
    }

    /// Continue direct preparation from the same already inspected descriptor
    /// and bounded sidecar bytes. Reuses the original source issuer and family
    /// validation; no metadata view grants model execution authority.
    pub fn from_inspected_gguf(
        artifact: crate::artifact::Artifact,
        config: &[u8],
        tokenizer: &[u8],
    ) -> Result<Self> {
        let geometry = crate::model::ModelGeometry::from_json(config)?;
        LoadedQwenModel::validate_direct_geometry(&geometry)?;
        let tokenizer = QwenTokenizer::from_bytes(tokenizer)?;
        let weights =
            zerogravity_artifacts::source_weights::SourceWeights::from_inspected_artifact(
                artifact,
            )?;
        let model = LoadedQwenModel::from_source(weights, config)?;
        Ok(Self { model, tokenizer })
    }

    pub fn model(&self) -> &Arc<LoadedQwenModel> {
        &self.model
    }
    pub fn tokenizer(&self) -> &QwenTokenizer {
        &self.tokenizer
    }
    pub fn into_parts(self) -> (Arc<LoadedQwenModel>, QwenTokenizer) {
        (self.model, self.tokenizer)
    }

    /// Explicit already-open cache inputs may come from independently pinned
    /// snapshots. Leases stay borrowed until actual owned bytes have been
    /// hashed and parsed. Cache metadata alone grants no model eligibility.
    #[cfg(all(feature = "cache", target_os = "linux"))]
    pub fn from_snapshots(
        model: &crate::cache::SnapshotLease,
        model_file: &str,
        config: &crate::cache::SnapshotLease,
        config_file: &str,
        tokenizer: &crate::cache::SnapshotLease,
        tokenizer_file: &str,
        admission: &crate::cache::TransferAdmission,
    ) -> Result<Self> {
        Self::from_snapshots_with_layout(
            model,
            model_file,
            config,
            config_file,
            tokenizer,
            tokenizer_file,
            admission,
            None,
        )
    }

    /// Optional routing table binds only metadata; original verified owned
    /// source bytes must reproduce it before model decoded allocations.
    #[cfg(all(feature = "cache", target_os = "linux"))]
    #[allow(clippy::too_many_arguments)]
    pub fn from_snapshots_with_layout(
        model: &crate::cache::SnapshotLease,
        model_file: &str,
        config: &crate::cache::SnapshotLease,
        config_file: &str,
        tokenizer: &crate::cache::SnapshotLease,
        tokenizer_file: &str,
        admission: &crate::cache::TransferAdmission,
        expected: Option<&crate::artifact::ArtifactInfo>,
    ) -> Result<Self> {
        use crate::cache::{FileRole, SnapshotLease};
        fn role(lease: &SnapshotLease, path: &str, expected: FileRole) -> Result<()> {
            if lease
                .manifest()
                .entries()
                .iter()
                .find(|entry| entry.path() == path)
                .is_none_or(|entry| entry.role() != expected)
            {
                return Err(crate::Error::Malformed(
                    "cached model component role/path mismatch".into(),
                ));
            }
            Ok(())
        }
        role(model, model_file, FileRole::Weights)?;
        role(config, config_file, FileRole::Config)?;
        role(tokenizer, tokenizer_file, FileRole::Tokenizer)?;
        let entry_len = |lease: &SnapshotLease, path: &str, limit: u64| -> Result<u64> {
            let bytes = lease
                .manifest()
                .entries()
                .iter()
                .find(|entry| entry.path() == path)
                .ok_or_else(|| crate::Error::Malformed("cached sidecar missing".into()))?
                .bytes();
            if bytes > limit {
                return Err(crate::Error::Limit(
                    "cached sidecar exceeds parser bound".into(),
                ));
            }
            Ok(bytes)
        };
        let config_bytes = entry_len(config, config_file, 1024 * 1024)?;
        let tokenizer_bytes = entry_len(tokenizer, tokenizer_file, 64 * 1024 * 1024)?;
        // This accounts the bounded raw sidecar copies and hash scratch through
        // parsing, not all parser/model/session heap allocations or global RAM.
        let sidecar_memory = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[crate::resource_arbiter::AllocationRequest {
                kind: crate::resource_arbiter::AllocationClass::TransferStaging,
                bytes: config_bytes + tokenizer_bytes + 64 * 1024,
            }],
        )?;
        let config_bytes = config.read_verified(config_file, 1024 * 1024)?;
        let geometry = crate::model::ModelGeometry::from_json(&config_bytes)?;
        LoadedQwenModel::validate_direct_geometry(&geometry)?;
        let tokenizer_bytes = tokenizer.read_verified(tokenizer_file, 64 * 1024 * 1024)?;
        let tokenizer = QwenTokenizer::from_bytes(&tokenizer_bytes)?;
        drop(tokenizer_bytes);
        let weights = model.source_weights(model_file, admission)?;
        if let Some(expected) = expected {
            let names = weights.tensor_names().collect::<Vec<_>>();
            if weights.source_len() as u64 != expected.file_len
                || names.len() != expected.tensors.len()
                || expected.tensors.iter().any(|t| {
                    weights.tensor(&t.name).map_or(true, |a| {
                        a.shape != t.shape
                            || a.kind != t.kind
                            || a.byte_offset != t.byte_offset
                            || a.byte_len != t.byte_len
                    })
                })
            {
                return Err(crate::Error::State(
                    "verified owned source differs from routing tensor layout".into(),
                ));
            }
        }
        let model = LoadedQwenModel::from_source(weights, &config_bytes)?;
        drop(config_bytes);
        drop(sidecar_memory);
        Ok(Self { model, tokenizer })
    }
}
