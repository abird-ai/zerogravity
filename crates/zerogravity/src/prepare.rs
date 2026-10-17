//! Cold preparation joins original source and model-family validation. It does
//! not acquire data, publish a session root, select a backend or run inference.
use crate::Result;
use crate::qwen35::LoadedQwenModel;
use crate::tokenize::QwenTokenizer;
use std::path::Path;
use std::sync::Arc;

pub struct PreparedQwen {
    model: Arc<LoadedQwenModel>,
    tokenizer: QwenTokenizer,
}
impl PreparedQwen {
    /// Existing path preparation, preserving tokenizer-first validation order.
    pub fn from_paths(model: &Path, config: &Path, tokenizer: &Path) -> Result<Self> {
        let tokenizer = QwenTokenizer::from_file(tokenizer)?;
        let model = LoadedQwenModel::open(model, config)?;
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
        let model = LoadedQwenModel::from_source(weights, &config_bytes)?;
        drop(config_bytes);
        drop(sidecar_memory);
        Ok(Self { model, tokenizer })
    }
}
