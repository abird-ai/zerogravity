//! Native finite embedding tasks. Explicit prepared owners only; no built-in
//! trained-source/HTTP capability is inferred from the candidate feature.
use std::{collections::BTreeMap, sync::Arc};
pub use zerogravity_engine::{
    embedding::{
        EmbeddingBatch, EmbeddingInput, EmbeddingReceipt, EmbeddingRequest, MAX_DIMENSIONS,
        MAX_INPUTS,
    },
    model_instance::EmbeddingEncoder,
    router::pending::ServingHost,
};
use zerogravity_engine::{
    model_instance::execution::{WorkPolicy, WorkerConfig},
    router::{ServiceError, validate_config},
    scheduler::SchedulerLimits,
};
pub type EncoderModels = BTreeMap<String, Arc<dyn EmbeddingEncoder>>;

/// Cold finite-sequence configuration, lowered to the existing worker
/// bounds. There is no decoder token budget or alternate scheduler.
#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    pub workers: usize,
    pub active_per_worker: usize,
    pub max_context: usize,
    pub host_mib: usize,
    pub max_pending: Option<usize>,
}
impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            workers: 1,
            active_per_worker: 1,
            max_context: 512,
            host_mib: 4096,
            max_pending: Some(16),
        }
    }
}
impl EncoderConfig {
    pub(crate) fn worker(self, limits: SchedulerLimits) -> Result<WorkerConfig, ServiceError> {
        self.worker_with_output(limits, 1)
    }
    pub(crate) fn worker_with_output(
        self,
        limits: SchedulerLimits,
        output: usize,
    ) -> Result<WorkerConfig, ServiceError> {
        validate_config(
            1,
            self.workers,
            self.active_per_worker,
            self.max_context,
            output,
            self.max_pending,
        )?;
        if self
            .workers
            .checked_mul(self.active_per_worker)
            .is_none_or(|n| n > limits.max_active)
            || !limits.allows_pending(self.max_pending)
        {
            return Err(ServiceError::new(
                400,
                "invalid_config",
                "encoder scheduler exceeds runtime limits",
            ));
        }
        Ok(WorkerConfig {
            workers: self.workers,
            active: self.active_per_worker,
            context: self.max_context,
            output,
            max_pending: self.max_pending,
            policy: WorkPolicy::Sequential,
        })
    }
}

#[cfg(all(feature = "model-bert-cpu-candidate", target_os = "linux"))]
mod bert {
    use super::*;
    use crate::{Error, Result};
    use std::sync::atomic::AtomicBool;
    use zerogravity_artifacts::{artifact::ArtifactFormat, cache::SnapshotLease};
    use zerogravity_core::{identity::SourceIdentity, resource_arbiter::TransferAdmission};
    use zerogravity_engine::diagnostics::{ArtifactDescription, ModelDetails};
    use zerogravity_model_bert::BertSentenceEncoder;

    /// Explicit source-bound CPU reference, pending trained-source parity.
    /// No automatic catalog resolver, default product registration or wire
    /// capability. Trusted callers can exercise the actual runtime path.
    pub struct BertEncoderCandidate {
        model: BertSentenceEncoder,
        snapshot: String,
    }
    impl BertEncoderCandidate {
        pub fn prepare(
            snapshot: &SnapshotLease,
            host: &ServingHost,
            alias: &str,
        ) -> Result<Arc<Self>> {
            zerogravity_engine::router::validate_alias(alias).map_err(ServiceError::into_core)?;
            let admission = host
                .transfer_admission(alias, "bert-encoder")
                .map_err(ServiceError::into_core)?;
            Ok(Arc::new(Self {
                model: BertSentenceEncoder::load(snapshot, &admission)?,
                snapshot: snapshot.manifest().digest()?,
            }))
        }
    }
    impl EmbeddingEncoder for BertEncoderCandidate {
        fn source(&self) -> SourceIdentity {
            self.model.source()
        }
        fn max_context(&self) -> usize {
            self.model.config().max_sequence()
        }
        fn vocabulary_size(&self) -> u64 {
            self.model.config().vocabulary_size() as u64
        }
        fn dimensions(&self) -> usize {
            self.model.config().dimensions()
        }
        fn validate_admission(&self, admission: &TransferAdmission) -> Result<()> {
            self.model.validate_admission(admission)
        }
        fn describe(&self) -> Result<ModelDetails> {
            Ok(ModelDetails {
                artifact: Some(ArtifactDescription {
                    family: "bert-sentence-candidate".into(),
                    format: ArtifactFormat::Safetensors,
                    cache_snapshot: Some(self.snapshot.clone()),
                    revision: None,
                }),
                ..Default::default()
            })
        }
        fn embed_into(
            &self,
            input: &EmbeddingInput,
            max_tokens: usize,
            output: &mut [f32],
            cancelled: &AtomicBool,
        ) -> Result<usize> {
            if output.len() != self.dimensions() {
                return Err(Error::InvalidInput("BERT output extent".into()));
            }
            match input {
                EmbeddingInput::Text(text) => {
                    self.model
                        .embed_text_into(text, max_tokens, output, Some(cancelled))
                }
                EmbeddingInput::Tokens(ids) => {
                    self.model
                        .embed_tokens_into(ids, max_tokens, output, Some(cancelled))
                }
            }
        }
    }
}
#[cfg(all(feature = "model-bert-cpu-candidate", target_os = "linux"))]
pub use bert::BertEncoderCandidate;
