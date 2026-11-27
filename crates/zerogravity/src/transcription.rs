//! Native transcription, with explicit prepared speech owners and the
//! original Runtime's admission, registry, worker and terminal receipt.
use std::{collections::BTreeMap, sync::Arc};
pub use zerogravity_engine::transcription::{
    SpeechRecognizer, SpeechSession, Transcript, TranscriptionFinish, TranscriptionReceipt,
    TranscriptionRequest, TranscriptionUsage,
};
use zerogravity_engine::{
    model_instance::execution::WorkerConfig, router::ServiceError, scheduler::SchedulerLimits,
};
pub type SpeechModels = BTreeMap<String, Arc<dyn SpeechRecognizer>>;

#[derive(Clone, Copy, Debug)]
pub struct TranscriptionConfig {
    pub workers: usize,
    pub active_per_worker: usize,
    pub max_context: usize,
    pub max_output_tokens: usize,
    pub host_mib: usize,
    pub max_pending: Option<usize>,
}
impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            workers: 1,
            active_per_worker: 1,
            max_context: 1024,
            max_output_tokens: 64,
            host_mib: 1024,
            max_pending: Some(16),
        }
    }
}
impl TranscriptionConfig {
    pub(crate) fn worker(self, limits: SchedulerLimits) -> Result<WorkerConfig, ServiceError> {
        crate::embedding::EncoderConfig {
            workers: self.workers,
            active_per_worker: self.active_per_worker,
            max_context: self.max_context,
            host_mib: self.host_mib,
            max_pending: self.max_pending,
        }
        .worker_with_output(limits, self.max_output_tokens)
    }
}

#[cfg(all(feature = "model-gemma4-audio-candidate", target_os = "linux"))]
mod gemma {
    use super::*;
    use crate::{Error, Result, embedding::ServingHost};
    use std::sync::atomic::AtomicBool;
    use zerogravity_artifacts::{artifact::ArtifactFormat, cache::SnapshotLease};
    use zerogravity_core::{identity::SourceIdentity, resource_arbiter::TransferAdmission};
    use zerogravity_engine::diagnostics::{ArtifactDescription, ModelDetails};
    use zerogravity_model_gemma4::audio::{
        Gemma4Transcriber, Gemma4Transcription, TranscriptFinish,
    };

    /// Explicit CPU F32 candidate, not an automatic trained ASR catalog or
    /// HTTP capability. Original source, tokenizer and decoder remain family-owned.
    pub struct Gemma4SpeechCandidate {
        model: Arc<Gemma4Transcriber>,
        snapshot: String,
    }
    impl Gemma4SpeechCandidate {
        pub fn prepare(
            snapshot: &SnapshotLease,
            host: &ServingHost,
            alias: &str,
            cancelled: Option<&AtomicBool>,
        ) -> Result<Arc<Self>> {
            zerogravity_engine::router::validate_alias(alias).map_err(ServiceError::into_core)?;
            let admission = host
                .transfer_admission(alias, "gemma4-speech")
                .map_err(ServiceError::into_core)?;
            Ok(Arc::new(Self {
                model: Gemma4Transcriber::load(snapshot, 12 << 30, &admission, cancelled)?,
                snapshot: snapshot.manifest().digest()?,
            }))
        }
        pub fn prompt_tokens(
            &self,
            samples: usize,
            limit: usize,
        ) -> Result<zerogravity_model_gemma4::text::tokenizer::Gemma4EncodedText> {
            self.model.prompt_tokens(samples, limit, None)
        }
    }
    impl SpeechRecognizer for Gemma4SpeechCandidate {
        fn source(&self) -> SourceIdentity {
            self.model.source()
        }
        fn max_samples(&self) -> usize {
            self.model.max_samples()
        }
        fn max_context(&self) -> usize {
            self.model.max_context()
        }
        fn max_output_tokens(&self) -> usize {
            self.model.max_output_tokens()
        }
        fn max_output_bytes(&self) -> usize {
            self.model.max_output_bytes()
        }
        fn vocabulary_size(&self) -> u64 {
            262144
        }
        fn validate_admission(&self, a: &TransferAdmission) -> Result<()> {
            self.model.validate_admission(a)
        }
        fn describe(&self) -> Result<ModelDetails> {
            Ok(ModelDetails {
                artifact: Some(ArtifactDescription {
                    family: "gemma4-audio-candidate".into(),
                    format: ArtifactFormat::Safetensors,
                    cache_snapshot: Some(self.snapshot.clone()),
                    revision: None,
                }),
                ..Default::default()
            })
        }
        fn start(
            self: Arc<Self>,
            samples: &[f32],
            sample_rate: u32,
            context: usize,
            max_tokens: usize,
            cancelled: &AtomicBool,
        ) -> Result<Box<dyn SpeechSession>> {
            Ok(Box::new(Session(self.model.start(
                samples,
                sample_rate,
                context,
                max_tokens,
                Some(cancelled),
            )?)))
        }
    }
    struct Session(Gemma4Transcription);
    impl SpeechSession for Session {
        fn step(
            &mut self,
            output: &mut String,
            cancelled: &AtomicBool,
        ) -> Result<Option<TranscriptionUsage>> {
            let Some(transcript) = self.0.step(Some(cancelled))? else {
                return Ok(None);
            };
            if !output.is_empty() || transcript.text().len() > output.capacity() {
                return Err(Error::Limit("admitted transcript output extent".into()));
            }
            output.push_str(transcript.text());
            Ok(Some(TranscriptionUsage {
                input_tokens: transcript.input_tokens(),
                output_tokens: transcript.output_tokens(),
                finish: match transcript.finish() {
                    TranscriptFinish::Stop => TranscriptionFinish::Stop,
                    TranscriptFinish::Length => TranscriptionFinish::Length,
                },
            }))
        }
    }
}
#[cfg(all(feature = "model-gemma4-audio-candidate", target_os = "linux"))]
pub use gemma::Gemma4SpeechCandidate;
