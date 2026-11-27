//! Finite speech jobs in the original registry, worker FIFO and receipt.
//! Model adapters own audio frontend, prompt, decoder state and presentation.
use crate::router::{
    ModelRead, RequestEvents, ServiceError, Work,
    event_channel::{self, mpsc},
    pending::{self, CancelToken, PendingQueue, ServingCharge, ServingHost},
    statistics::WorkStats,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::{
    identity::SourceIdentity,
    resource_arbiter::{AllocationClass, TransferAdmission},
};

type Result<T> = std::result::Result<T, ServiceError>;
pub const MAX_TRANSCRIPT_BYTES: usize = 65536;
pub const SAMPLE_RATE: u32 = 16000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeechTask {
    Transcription,
    Translation,
}

/// Actual admitted public capability envelope. The wire adapter prices
/// this before decoding PCM; admission rechecks the same registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptionLimits {
    pub registration_id: u64,
    pub task: SpeechTask,
    pub max_samples: usize,
    pub max_tokens: usize,
    pub max_output_bytes: usize,
}

pub struct TranscriptionRequest {
    pub model: String,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub max_tokens: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptionFinish {
    Stop,
    Length,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptionUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub finish: TranscriptionFinish,
}
/// Trusted prepared source; bounded synchronous calls complete before
/// returning. No execution owner, hidden worker or HTTP policy.
pub trait SpeechRecognizer: Send + Sync {
    fn source(&self) -> SourceIdentity;
    fn max_samples(&self) -> usize;
    fn max_context(&self) -> usize;
    fn vocabulary_size(&self) -> u64;
    fn max_output_tokens(&self) -> usize;
    fn max_output_bytes(&self) -> usize;
    /// Trusted explicit wire qualification, captured at registration.
    /// Native candidates never become public just by calculating text.
    fn public_transcription(&self) -> bool {
        false
    }
    /// Native translation and wire qualification are distinct from ASR.
    fn supports_translation(&self) -> bool {
        false
    }
    fn public_translation(&self) -> bool {
        false
    }
    fn start_translation(
        self: Arc<Self>,
        _samples: &[f32],
        _sample_rate: u32,
        _context: usize,
        _max_tokens: usize,
        _cancelled: &AtomicBool,
    ) -> crate::Result<Box<dyn SpeechSession>> {
        Err(crate::Error::Unsupported(
            "speech translation is not supported".into(),
        ))
    }
    fn validate_admission(&self, admission: &TransferAdmission) -> crate::Result<()>;
    fn describe(&self) -> crate::Result<crate::diagnostics::ModelDetails> {
        Ok(Default::default())
    }
    fn start(
        self: Arc<Self>,
        samples: &[f32],
        sample_rate: u32,
        context: usize,
        max_tokens: usize,
        cancelled: &AtomicBool,
    ) -> crate::Result<Box<dyn SpeechSession>>;
}
/// One actual finite model step per quantum. The session may append final
/// decoded text to the caller-admitted String only on terminal completion.
pub trait SpeechSession: Send {
    fn step(
        &mut self,
        output: &mut String,
        cancelled: &AtomicBool,
    ) -> crate::Result<Option<TranscriptionUsage>>;
}
pub struct Transcript {
    task: SpeechTask,
    text: String,
    usage: TranscriptionUsage,
    source: SourceIdentity,
    _charge: ServingCharge,
}
impl Transcript {
    pub fn task(&self) -> SpeechTask {
        self.task
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn usage(&self) -> TranscriptionUsage {
        self.usage
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
}
pub type TranscriptionReceipt = RequestEvents<Result<Transcript>>;

fn overflow() -> ServiceError {
    ServiceError::new(400, "size_overflow", "speech request allocation extent")
}
impl TranscriptionRequest {
    pub(crate) fn validate(
        &self,
        model: &dyn SpeechRecognizer,
        max_bytes: usize,
        context: usize,
        max_output: usize,
    ) -> Result<()> {
        if self.sample_rate != SAMPLE_RATE
            || self.samples.is_empty()
            || self.samples.len() > model.max_samples()
            || self
                .samples
                .iter()
                .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
        {
            return Err(ServiceError::new(
                400,
                "invalid_input",
                "expected bounded normalized mono 16 kHz PCM",
            ));
        }
        if self.samples.len().checked_mul(4).ok_or_else(overflow)? > max_bytes {
            return Err(ServiceError::new(
                413,
                "input_too_large",
                "speech input exceeds configured input bytes",
            ));
        }
        if self.max_tokens == 0
            || self.max_tokens >= context
            || self.max_tokens > max_output.min(model.max_output_tokens())
        {
            return Err(ServiceError::new(
                400,
                "invalid_input",
                "speech output/context limit",
            ));
        }
        Ok(())
    }
}

pub(crate) struct SpeechJob {
    task: SpeechTask,
    request: TranscriptionRequest,
    session: Option<Box<dyn SpeechSession>>,
    model: Arc<dyn SpeechRecognizer>,
    text: String,
    output_charge: Option<ServingCharge>,
    context: usize,
    steps: usize,
    events: mpsc::SyncSender<Result<Transcript>>,
    cancel: CancelToken,
    _request: ServingCharge,
    reader: ModelRead,
}
impl SpeechJob {
    pub(crate) fn prepare(
        intent: (SpeechTask, TranscriptionRequest),
        model: Arc<dyn SpeechRecognizer>,
        reader: ModelRead,
        host: &ServingHost,
        queue: &PendingQueue<Work>,
        id: u64,
        context: usize,
    ) -> Result<(Box<Self>, TranscriptionReceipt)> {
        let reader = reader;
        let model = model;
        let (task, request) = intent;
        let request_bytes = pending::checked_sum(&[
            pending::heap_strings(&[&request.model])?,
            pending::vec_bytes::<f32>(request.samples.capacity())?,
            crate::router::fixed_size::<Self>()?,
            64,
        ])?;
        let charge = host.charge(
            &request.model,
            "speech-request",
            AllocationClass::Scratch,
            request_bytes,
        )?;
        let receipt_charge = host.charge(
            &request.model,
            "speech-receipt",
            AllocationClass::Scratch,
            pending::heap_strings(&[&request.model])?,
        )?;
        let output_charge = host.charge(
            &request.model,
            "speech-output",
            AllocationClass::Output,
            model.max_output_bytes() as u64,
        )?;
        let mut text = String::new();
        text.try_reserve_exact(model.max_output_bytes())
            .map_err(|_| {
                ServiceError::new(503, "queue_allocation", "speech output allocation refused")
            })?;
        let (events, receiver) =
            mpsc::try_sync_channel(host, &request.model, "speech-terminal", 1)?;
        let cancel_charge = host.charge(
            &request.model,
            "cancel-token",
            AllocationClass::Scratch,
            event_channel::cancel_bytes()?,
        )?;
        let cancel = queue.cancel_token(Some(Arc::new(cancel_charge)));
        let receipt = RequestEvents::new(
            id,
            request.model.clone(),
            receiver,
            cancel.clone(),
            Some(receipt_charge),
            None,
        );
        Ok((
            Box::new(Self {
                task,
                request,
                session: None,
                model,
                text,
                output_charge: Some(output_charge),
                context,
                steps: 0,
                events,
                cancel,
                _request: charge,
                reader,
            }),
            receipt,
        ))
    }
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
    fn advance(&mut self) -> Result<Option<TranscriptionUsage>> {
        match &mut self.session {
            None => {
                let model = self.model.clone();
                let samples = &self.request.samples;
                let rate = self.request.sample_rate;
                let max = self.request.max_tokens;
                let cancelled = self.cancel.flag();
                self.session = Some(
                    match self.task {
                        SpeechTask::Transcription => {
                            model.start(samples, rate, self.context, max, cancelled)
                        }
                        SpeechTask::Translation => {
                            model.start_translation(samples, rate, self.context, max, cancelled)
                        }
                    }
                    .map_err(ServiceError::inference)?,
                );
                Ok(None)
            }
            Some(session) => {
                self.steps += 1;
                session
                    .step(&mut self.text, self.cancel.flag())
                    .map_err(ServiceError::inference)
            }
        }
    }
    pub(crate) fn step(&mut self, stats: &WorkStats) -> bool {
        if self.is_cancelled() {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let result = self.advance();
        if self.is_cancelled() {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let result = result.and_then(|usage| {
            if self.text.capacity() > self.model.max_output_bytes()
                || (usage.is_none() && !self.text.is_empty())
            {
                return Err(ServiceError::new(
                    500,
                    "invalid_transcript",
                    "speech adapter output extent or partial terminal",
                ));
            }
            if let Some(u) = usage
                && (u.input_tokens == 0
                    || u.output_tokens == 0
                    || u.output_tokens > self.request.max_tokens
                    || u.output_tokens != self.steps
                    || u.input_tokens
                        .checked_add(u.output_tokens)
                        .is_none_or(|n| n > self.context)
                    || (u.finish == TranscriptionFinish::Length
                        && u.output_tokens != self.request.max_tokens))
            {
                return Err(ServiceError::new(
                    500,
                    "invalid_transcript",
                    "speech adapter usage mismatch",
                ));
            }
            if usage.is_none() && self.steps >= self.request.max_tokens {
                return Err(ServiceError::new(
                    500,
                    "invalid_transcript",
                    "speech adapter exceeded output steps",
                ));
            }
            Ok(usage)
        });
        match result {
            Ok(None) => true,
            Ok(Some(usage)) => {
                // Retire actual decoder before publishing its detached result.
                drop(self.session.take());
                let result = Transcript {
                    task: self.task,
                    text: std::mem::take(&mut self.text),
                    usage,
                    source: self.model.source(),
                    _charge: self.output_charge.take().expect("speech output credit"),
                };
                if self.events.try_send(Ok(result)).is_ok() {
                    let (requests, input, output) = match self.task {
                        SpeechTask::Transcription => (
                            &stats.transcription_requests,
                            &stats.transcription_input_tokens,
                            &stats.transcription_output_tokens,
                        ),
                        SpeechTask::Translation => (
                            &stats.translation_requests,
                            &stats.translation_input_tokens,
                            &stats.translation_output_tokens,
                        ),
                    };
                    requests.fetch_add(1, Ordering::Relaxed);
                    input.fetch_add(usage.input_tokens as u64, Ordering::Relaxed);
                    output.fetch_add(usage.output_tokens as u64, Ordering::Relaxed);
                    self.reader.complete();
                } else {
                    stats.cancelled.fetch_add(1, Ordering::Relaxed);
                }
                false
            }
            Err(error) => {
                drop(self.session.take());
                stats.failed.fetch_add(1, Ordering::Relaxed);
                let _ = self.events.try_send(Err(error));
                false
            }
        }
    }
}
