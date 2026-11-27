//! Shared admitted routing, presentation and worker ownership.
use super::event_channel::mpsc;
#[cfg(test)]
use super::pending::Pop;
use super::pending::{
    self, CancelToken, PendingQueue, QueueSnapshot, ServingCharge, ServingHost as HostPool,
    WorkerQueue,
};
pub use super::registry::ModelRead;
pub(crate) use super::registry::RegisteredOwner;
use super::{host::HostLogits, registry, statistics::WorkStats, waves, worker};
use crate::embedding::{EmbeddingJob, EmbeddingReceipt, EmbeddingRequest, MAX_DIMENSIONS};
pub use crate::model_instance::TextTokenizer;
use crate::model_instance::execution::*;
use crate::model_instance::{EmbeddingEncoder, LogitModel, ModelInstance};
#[cfg(test)]
use crate::model_instance::{PreparedModel, TextSession};
use crate::transcription::{
    MAX_TRANSCRIPT_BYTES, SpeechJob, SpeechRecognizer, SpeechTask, TranscriptionReceipt,
    TranscriptionRequest,
};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;
use zerogravity_core::resource_arbiter::AllocationClass;
#[cfg(test)]
use zerogravity_core::resource_arbiter::GlobalResourceLease;

pub const EVENTS_PER_REQUEST: usize = 16;
pub const MAX_TOKEN_OBSERVATIONS: usize = 512;
pub const MAX_INPUT_BYTES: usize = 1024 * 1024;
pub const MAX_ACTIVE_PER_WORKER: usize = 8;
/// Transport bounds only; each prepared source still limits its own context.
pub const MAX_CONTEXT_TOKENS: usize = 32768;
pub const MAX_OUTPUT_TOKENS: usize = MAX_TOKEN_OBSERVATIONS;

/// Conservative allocator header/rounding for a `Vec`/`VecDeque` backing.
const CONTAINER_ALLOC_OVERHEAD_BYTES: u64 = 64;

pub fn fixed_size<T>() -> Result<u64, ServiceError> {
    u64::try_from(std::mem::size_of::<T>())
        .map_err(|_| ServiceError::new(400, "size_overflow", "container size overflow"))
}

pub fn vecdeque_backing_bytes<T>(capacity: usize) -> Result<u64, ServiceError> {
    let slots = u64::try_from(capacity)
        .ok()
        .and_then(|n| n.checked_mul(fixed_size::<T>().ok()?))
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "queue backing overflow"))?;
    pending::checked_sum(&[
        fixed_size::<VecDeque<T>>()?,
        slots,
        CONTAINER_ALLOC_OVERHEAD_BYTES,
    ])
}

pub use super::error::{FailureDetails, ServiceError};

/// Legacy single-text request. Compatibility adapter into SemanticRequest.
pub struct Request {
    pub model: String,
    pub input: String,
    pub chat: bool,
    pub max_tokens: usize,
    pub stream: bool,
    pub token_metrics: Option<TokenMetricsRequest>,
}

pub use zerogravity_core::task::{ChatMessage, ChatRole, MAX_CHAT_MESSAGES, TextInput};

/// Canonical admitted text intent. Protocol presentation controls never
/// select a different execution path or model owner.
pub struct SemanticRequest {
    pub model: String,
    pub input: TextInput,
    pub max_tokens: usize,
    pub stream: bool,
    pub token_metrics: Option<TokenMetricsRequest>,
}
impl From<Request> for SemanticRequest {
    fn from(request: Request) -> Self {
        Self {
            model: request.model,
            input: TextInput::from_text(request.input, request.chat),
            max_tokens: request.max_tokens,
            stream: request.stream,
            token_metrics: request.token_metrics,
        }
    }
}
impl SemanticRequest {
    /// Legacy embedding services may consume single text without copying.
    /// Never silently flatten a conversation for an older adapter.
    pub fn into_legacy(self) -> Result<Request, ServiceError> {
        let (input, chat) = match self.input {
            TextInput::Raw(text) => (text, false),
            TextInput::User(text) => (text, true),
            TextInput::Conversation(_) => {
                return Err(ServiceError::new(
                    400,
                    "unsupported_chat",
                    "service does not support structured conversation",
                ));
            }
        };
        Ok(Request {
            model: self.model,
            input,
            chat,
            max_tokens: self.max_tokens,
            stream: self.stream,
            token_metrics: self.token_metrics,
        })
    }

    pub fn validate_input(&self, limit: usize) -> Result<(), ServiceError> {
        validate_text_input(&self.input, limit)
    }
}

fn validate_text_input(input: &TextInput, limit: usize) -> Result<(), ServiceError> {
    match input {
        TextInput::Conversation(turns) => {
            if turns.is_empty() || turns.len() > MAX_CHAT_MESSAGES {
                return Err(ServiceError::new(
                    400,
                    "invalid_messages",
                    "messages must contain 1..128 entries",
                ));
            }
            if input.content_bytes().is_none_or(|n| n > limit) {
                return Err(ServiceError::new(
                    413,
                    "input_too_large",
                    "chat input too large",
                ));
            }
        }
        TextInput::Raw(text) | TextInput::User(text) if text.is_empty() || text.len() > limit => {
            return Err(ServiceError::new(
                400,
                "invalid_request",
                "request exceeds input/output bounds",
            ));
        }
        _ => {}
    }
    Ok(())
}

/// Measure and price the original model-owned template for both generation
/// and read-only counting. No tokenization before the caller admits scratch.
fn prompt_bounds(
    tokenizer: &dyn TextTokenizer,
    input: &TextInput,
    input_limit: usize,
) -> Result<crate::model_instance::PromptBounds, ServiceError> {
    let bounds = tokenizer
        .semantic_prompt_bound(input)
        .map_err(|error| match error {
            crate::Error::InvalidInput(message) if input.is_chat() => {
                ServiceError::new(400, "invalid_messages", message)
            }
            crate::Error::Unsupported(message) if input.is_chat() => {
                ServiceError::new(400, "unsupported_chat", message)
            }
            crate::Error::Limit(message) if matches!(input, TextInput::Conversation(_)) => {
                ServiceError::new(413, "input_too_large", message)
            }
            error => ServiceError::new(400, "size_overflow", error.to_string()),
        })?;
    // Preserve the configured structured-chat limit on rendered as well
    // as semantic bytes; legacy single text retains its original contract.
    if matches!(input, TextInput::Conversation(_)) && bounds.input_bytes > input_limit {
        return Err(ServiceError::new(
            400,
            "invalid_request",
            "request exceeds input/output bounds",
        ));
    }
    Ok(bounds)
}

fn prompt_backing_bytes(bounds: &crate::model_instance::PromptBounds) -> Result<u64, ServiceError> {
    pending::checked_sum(&[
        pending::vec_bytes::<u32>(bounds.tokens)?,
        bounds.scratch_bytes,
    ])
}

/// Per-request clock/label only; never an authorization or publication owner.
#[derive(Clone)]
pub struct TokenMetricsRequest {
    pub started: Instant,
    pub wire_request_id: Option<String>,
}
pub struct TokenObservation {
    pub index: usize,
    pub id: u32,
    /// Observed immediately after the original source-root publication.
    pub commit_observed_at: Instant,
    pub gpu: Option<GpuTokenObservation>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct FusedCohortObservation {
    pub publication_sequence: u64,
    pub width: usize,
}
/// Diagnostic values copied only after the original successful publication.
/// They cannot mint a root, attempt, Ready or feedback token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct GpuRootObservation {
    pub session_id: u64,
    pub root: u64,
    pub session_epoch: u64,
}
impl GpuRootObservation {
    pub fn committed(binding: zerogravity_core::session_root::RootBinding) -> Self {
        Self {
            session_id: binding.session().get(),
            root: binding.root(),
            session_epoch: binding.session_epoch(),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct GpuTokenObservation {
    pub producer_frontier: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer_root: Option<GpuRootObservation>,
    pub fused_cohort: Option<FusedCohortObservation>,
}
pub enum ResponseEvent {
    Text(String),
    TokenText {
        text: String,
        token: TokenObservation,
    },
    Done {
        output: String,
        prompt_tokens: usize,
        /// Exact restored prompt prefix from the selected execution owner.
        cached_prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: &'static str,
    },
    Error(ServiceError),
}
pub struct RequestEvents<E = ResponseEvent> {
    pub id: u64,
    pub model: String,
    pub events: mpsc::Receiver<E>,
    pub token_metrics: Option<TokenMetricsRequest>,
    _cancel: CancelToken,
    _request: Option<ServingCharge>,
    _transport: Option<Arc<ServingCharge>>,
}
impl<E> RequestEvents<E> {
    pub fn new(
        id: u64,
        model: String,
        events: mpsc::Receiver<E>,
        cancel: CancelToken,
        request: Option<ServingCharge>,
        transport: Option<Arc<ServingCharge>>,
    ) -> Self {
        Self {
            id,
            model,
            events,
            token_metrics: None,
            _cancel: cancel,
            _request: request,
            _transport: transport,
        }
    }
    pub fn cancel(&self) {
        self._cancel.cancel();
    }
}
impl RequestEvents {
    pub fn observed(mut self, metrics: Option<TokenMetricsRequest>) -> Self {
        self.token_metrics = metrics;
        self
    }
}
impl<E> Drop for RequestEvents<E> {
    fn drop(&mut self) {
        self._cancel.cancel();
    }
}
/// Shared CPU/GPU presentation and backpressure semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presentation {
    Ready,
    Blocked,
    Finished,
}

pub struct GeneratedText {
    tokenizer: Arc<dyn TextTokenizer>,
    events: mpsc::SyncSender<ResponseEvent>,
    max_tokens: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    stream: bool,
    generated: Vec<u32>,
    emitted: String,
    // At most one Text delta and one terminal Done. Retry only unsent events;
    // successfully queued output is never replayed after a GPU retry.
    pending: VecDeque<ResponseEvent>,
    pending_emitted: Option<String>,
    terminal: bool,
    finished: bool,
    failure: Option<ServiceError>,
    token_metrics: Option<TokenMetricsRequest>,
    vocabulary_size: u64,
    last_commit_observed: Option<Instant>,
    // Bundled with `token_metrics` so the wire-ID string and the ingress
    // charge that covers it are released in one owner.
    _metrics_charge: Option<Arc<Vec<ServingCharge>>>,
}
impl GeneratedText {
    pub fn new(
        tokenizer: Arc<dyn TextTokenizer>,
        events: mpsc::SyncSender<ResponseEvent>,
        stream: bool,
        max_tokens: usize,
        prompt_tokens: usize,
    ) -> Self {
        Self::new_owned(tokenizer, events, stream, max_tokens, prompt_tokens)
    }
    pub(super) fn with_cached_prefix(mut self, tokens: usize) -> Self {
        debug_assert!(tokens <= self.prompt_tokens);
        self.cached_prompt_tokens = tokens;
        self
    }
    pub fn new_owned(
        tokenizer: Arc<dyn TextTokenizer>,
        events: mpsc::SyncSender<ResponseEvent>,
        stream: bool,
        max_tokens: usize,
        prompt_tokens: usize,
    ) -> Self {
        Self {
            tokenizer,
            events,
            stream,
            max_tokens,
            prompt_tokens,
            cached_prompt_tokens: 0,
            generated: Vec::new(),
            emitted: String::new(),
            pending: VecDeque::with_capacity(2),
            pending_emitted: None,
            terminal: false,
            finished: false,
            failure: None,
            token_metrics: None,
            vocabulary_size: 0,
            last_commit_observed: None,
            _metrics_charge: None,
        }
    }
    #[cfg(test)]
    pub(crate) fn observed(
        mut self,
        metrics: Option<TokenMetricsRequest>,
        vocabulary: u64,
    ) -> Self {
        self.token_metrics = metrics;
        self.vocabulary_size = vocabulary;
        self
    }
    pub fn observed_charged(
        mut self,
        metrics: Option<TokenMetricsRequest>,
        vocabulary: u64,
        charge: Option<Arc<Vec<ServingCharge>>>,
    ) -> Self {
        self.token_metrics = metrics;
        self.vocabulary_size = vocabulary;
        self._metrics_charge = charge;
        self
    }
    /// Sequential policy: bounded output refusal terminates this presenter.
    #[cfg(test)]
    pub(crate) fn accept(&mut self, id: u32) -> Result<bool, ServiceError> {
        self.accept_committed(id, Instant::now())
    }
    pub fn accept_committed(&mut self, id: u32, committed: Instant) -> Result<bool, ServiceError> {
        self.accept_observed(id, committed, None)
    }
    pub(crate) fn accept_observed(
        &mut self,
        id: u32,
        committed: Instant,
        observation: Option<GpuTokenObservation>,
    ) -> Result<bool, ServiceError> {
        match self.present_observed(id, committed, observation)? {
            Presentation::Ready => Ok(false),
            Presentation::Finished => Ok(true),
            Presentation::Blocked => Err(self.fail(ServiceError::new(
                503,
                "backpressure",
                "client cannot accept streamed output",
            ))),
        }
    }

    pub fn generated_token_count(&self) -> usize {
        self.generated.len()
    }

    /// Stage one already-committed token exactly once. The GPU owner resumes
    /// blocked delivery before advancing this packet or presenting another.
    #[cfg(test)]
    pub(crate) fn present(&mut self, id: u32) -> Result<Presentation, ServiceError> {
        self.present_committed(id, Instant::now())
    }
    pub fn present_committed(
        &mut self,
        id: u32,
        commit_observed_at: Instant,
    ) -> Result<Presentation, ServiceError> {
        self.present_observed(id, commit_observed_at, None)
    }
    pub fn present_observed(
        &mut self,
        id: u32,
        commit_observed_at: Instant,
        gpu: Option<GpuTokenObservation>,
    ) -> Result<Presentation, ServiceError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.finished {
            return Ok(Presentation::Finished);
        }
        if !self.pending.is_empty() || self.terminal {
            return Err(ServiceError::new(
                500,
                "output_pending",
                "resume staged output before presenting another token",
            ));
        }
        if let Some(metrics) = &self.token_metrics
            && (u64::from(id) >= self.vocabulary_size
                || self.generated.len() >= MAX_TOKEN_OBSERVATIONS
                || commit_observed_at < metrics.started
                || commit_observed_at > Instant::now()
                || self
                    .last_commit_observed
                    .is_some_and(|last| commit_observed_at < last))
        {
            return Err(self.fail(ServiceError::new(
                500,
                "invalid_token_observation",
                "invalid token ID, clock or bounded token observation",
            )));
        }
        let token = (self.token_metrics.is_some() && id != self.tokenizer.eos()).then_some(
            TokenObservation {
                index: self.generated.len(),
                id,
                commit_observed_at,
                gpu,
            },
        );
        self.last_commit_observed = Some(commit_observed_at);
        let reason = if id == self.tokenizer.eos() {
            Some("stop")
        } else {
            self.generated.push(id);
            (self.generated.len() >= self.max_tokens).then_some("length")
        };
        let mut token = token;
        if self.stream || reason.is_some() {
            let decoded = match self.tokenizer.decode(&self.generated, true) {
                Ok(decoded) => decoded,
                Err(error) => return Err(self.fail(ServiceError::inference(error))),
            };
            if self.stream {
                if !decoded.starts_with(&self.emitted) {
                    return Err(self.fail(ServiceError::new(
                        500,
                        "decoder_nonmonotonic",
                        "incremental text prefix changed",
                    )));
                }
                // A nonterminal incomplete UTF-8 token remains private. Final
                // decoding retains the original replacement/flush semantics.
                if (reason.is_some() || !decoded.contains('�'))
                    && decoded.len() > self.emitted.len()
                {
                    let text = decoded[self.emitted.len()..].to_owned();
                    self.pending.push_back(match token.take() {
                        Some(token) => ResponseEvent::TokenText { text, token },
                        None => ResponseEvent::Text(text),
                    });
                    self.pending_emitted = Some(decoded.clone());
                }
            }
            if let Some(token) = token.take() {
                self.pending.push_back(ResponseEvent::TokenText {
                    text: String::new(),
                    token,
                });
            }
            if let Some(reason) = reason {
                self.terminal = true;
                self.pending.push_back(ResponseEvent::Done {
                    output: decoded,
                    prompt_tokens: self.prompt_tokens,
                    cached_prompt_tokens: self.cached_prompt_tokens,
                    completion_tokens: self.generated.len(),
                    finish_reason: reason,
                });
            }
        }
        if let Some(token) = token {
            self.pending.push_back(ResponseEvent::TokenText {
                text: String::new(),
                token,
            });
        }
        self.resume()
    }

    /// Retry only unsent bounded events. A full channel has accepted none of
    /// that event; disconnect never causes a new delivery attempt or token.
    pub fn resume(&mut self) -> Result<Presentation, ServiceError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        while let Some(event) = self.pending.pop_front() {
            let text = matches!(&event, ResponseEvent::Text(_))
                || matches!(&event, ResponseEvent::TokenText { text, .. } if !text.is_empty());
            let done = matches!(&event, ResponseEvent::Done { .. });
            match self.events.try_send(event) {
                Ok(()) => {
                    if text {
                        self.emitted = self.pending_emitted.take().expect("staged text frontier");
                    }
                    if done {
                        self.finished = true;
                    }
                }
                Err(mpsc::TrySendError::Full(event)) => {
                    self.pending.push_front(event);
                    return Ok(Presentation::Blocked);
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err(self.fail(ServiceError::new(
                        499,
                        "cancelled",
                        "client disconnected",
                    )));
                }
            }
        }
        Ok(if self.finished {
            Presentation::Finished
        } else {
            Presentation::Ready
        })
    }

    fn fail(&mut self, error: ServiceError) -> ServiceError {
        self.pending.clear();
        self.pending_emitted = None;
        self.failure = Some(error.clone());
        error
    }
}

/// Queued request strings and the ingress charges that cover them. The
/// charges are released only after the strings are gone.
pub struct RequestLease {
    pub request: SemanticRequest,
    pub prefix_sharing: Option<crate::state_cache::PrefixSharingScope>,
    pub _charges: Arc<Vec<ServingCharge>>,
}
/// The metrics wire ID and the original charge covering it travel together
/// into the presenter, so no cloned identifier becomes uncharged.
pub struct MetricsLease {
    pub metrics: Option<TokenMetricsRequest>,
    pub _charges: Option<Arc<Vec<ServingCharge>>>,
}
pub struct Scheduled {
    pub payload: Option<RequestLease>,
    pub metrics: MetricsLease,
    pub events: mpsc::SyncSender<ResponseEvent>,
    pub cancelled: Arc<AtomicBool>,
    pub cancel: CancelToken,
    pub prompt_charge: Option<ServingCharge>,
    pub transport: Option<Arc<ServingCharge>>,
    /// Only direct diagnostic row construction may omit the registry reader.
    pub model_reader: Option<ModelRead>,
}
// Text work remains inline; only encoder work needs the boxed batch
// state. No extra per-token indirection or separate pending queue.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Work {
    Text(Scheduled),
    Finite(FiniteJob),
}
pub(crate) enum FiniteJob {
    Encoder(Box<EmbeddingJob>),
    Speech(Box<SpeechJob>),
}
impl FiniteJob {
    fn is_cancelled(&self) -> bool {
        match self {
            Self::Encoder(j) => j.is_cancelled(),
            Self::Speech(j) => j.is_cancelled(),
        }
    }
    pub(super) fn step(&mut self, stats: &WorkStats) -> bool {
        match self {
            Self::Encoder(j) => j.step(stats),
            Self::Speech(j) => j.step(stats),
        }
    }
}
impl Work {
    pub(crate) fn is_cancelled(&self) -> bool {
        match self {
            Self::Text(job) => job.cancel.is_cancelled(),
            Self::Finite(job) => job.is_cancelled(),
        }
    }
}
pub struct Router {
    models: Arc<dyn registry::Models>,
    host: HostPool,
    queue: PendingQueue<Work>,
    stats: Arc<WorkStats>,
    active_limit: usize,
    workers: Vec<thread::JoinHandle<()>>,
    // Non-payload worker inventory (handles, consumer state, captures, names).
    // Declared after `workers` so the joins/handles pay out first.
    _worker_charge: ServingCharge,
    next_id: AtomicU64,
    config: WorkerConfig,
    _control_charge: ServingCharge,
}
impl Router {
    /// Model-free transport fixture: the original idle CPU worker/queue/host,
    /// with no registered model or inference capability.
    #[cfg(test)]
    pub(crate) fn transport_test() -> (Self, HostPool) {
        let host = HostPool::isolated(256).unwrap();
        let router = Self::start::<HostLogits>(
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            host.clone(),
            WorkerConfig {
                workers: 1,
                active: 1,
                context: 16,
                output: 8,
                max_pending: Some(8),
                policy: WorkPolicy::Sequential,
            },
        )
        .unwrap();
        (router, host)
    }
    #[cfg(test)]
    pub(crate) fn retirement_test<A: Executor>(host: HostPool) -> Self {
        Self::start::<A>(
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            host,
            WorkerConfig {
                workers: 1,
                active: 1,
                context: 16,
                output: 1,
                max_pending: Some(1),
                policy: WorkPolicy::Waves(WavePolicy {
                    max_members: 1,
                    max_tokens: 32,
                    max_scratch: 32 * 1024 * 1024,
                    fairness: Fairness {
                        round_tokens: 33,
                        prefill_age_rounds: 2,
                    },
                    prefill_quantum: 32,
                    unequal_prefill: false,
                    unequal_decode: false,
                    cohorts: Cohorts::None,
                    cohort_width: 0,
                }),
            },
        )
        .unwrap()
    }
    #[cfg(test)]
    pub(crate) fn retirement_counter(&self, previous: u64) {
        self.stats.failed.store(previous, Ordering::Release);
    }
    pub fn new(
        models: BTreeMap<String, Arc<dyn LogitModel>>,
        workers: usize,
        active_per_worker: usize,
        max_context: usize,
        max_output: usize,
        host: HostPool,
        max_pending: Option<usize>,
    ) -> Result<Self, ServiceError> {
        Self::with_executor::<HostLogits>(
            models,
            host,
            WorkerConfig {
                workers,
                active: active_per_worker,
                context: max_context,
                output: max_output,
                max_pending,
                policy: WorkPolicy::Sequential,
            },
        )
    }
    /// One registry, admission path and worker algorithm; executors supply only
    /// typed source/session/completion operations, never a second scheduler.
    pub fn with_executor<A: Executor>(
        models: BTreeMap<String, A::Model>,
        host: HostPool,
        config: WorkerConfig,
    ) -> Result<Self, ServiceError> {
        Self::with_executor_and_embeddings::<A>(models, BTreeMap::new(), host, config)
    }
    /// A cold encoder-only route set, using the same CPU worker algorithm.
    pub fn with_embeddings(
        encoders: BTreeMap<String, Arc<dyn EmbeddingEncoder>>,
        host: HostPool,
        config: WorkerConfig,
    ) -> Result<Self, ServiceError> {
        Self::with_executor_and_embeddings::<HostLogits>(BTreeMap::new(), encoders, host, config)
    }
    /// Mixed prepared owners; one alias set, FIFO, active limit and worker
    /// join. Encoder calls are synchronous, finite, non-text quanta.
    pub fn with_executor_and_embeddings<A: Executor>(
        models: BTreeMap<String, A::Model>,
        encoders: BTreeMap<String, Arc<dyn EmbeddingEncoder>>,
        host: HostPool,
        config: WorkerConfig,
    ) -> Result<Self, ServiceError> {
        Self::with_executor_and_finite_tasks::<A>(models, encoders, BTreeMap::new(), host, config)
    }
    pub fn with_transcription(
        speech: BTreeMap<String, Arc<dyn SpeechRecognizer>>,
        host: HostPool,
        config: WorkerConfig,
    ) -> Result<Self, ServiceError> {
        Self::with_executor_and_finite_tasks::<HostLogits>(
            BTreeMap::new(),
            BTreeMap::new(),
            speech,
            host,
            config,
        )
    }
    /// One alias registry and worker budget across decoder, embedding and speech tasks.
    pub fn with_executor_and_finite_tasks<A: Executor>(
        models: BTreeMap<String, A::Model>,
        encoders: BTreeMap<String, Arc<dyn EmbeddingEncoder>>,
        speech: BTreeMap<String, Arc<dyn SpeechRecognizer>>,
        host: HostPool,
        config: WorkerConfig,
    ) -> Result<Self, ServiceError> {
        validate_config(
            models
                .len()
                .saturating_add(encoders.len())
                .saturating_add(speech.len()),
            config.workers,
            config.active,
            config.context,
            config.output,
            config.max_pending,
        )?;
        validate_aliases(
            models
                .keys()
                .chain(encoders.keys())
                .chain(speech.keys())
                .map(String::as_str),
        )?;
        for (alias, model) in &speech {
            if !model.source().is_bound()
                || model.max_context() == 0
                || model.max_samples() == 0
                || model.max_output_tokens() == 0
                || model.vocabulary_size() == 0
                || !(1..=MAX_TRANSCRIPT_BYTES).contains(&model.max_output_bytes())
            {
                return Err(ServiceError::new(
                    400,
                    "invalid_model",
                    "unbound or unbounded speech recognizer",
                ));
            }
            model
                .validate_admission(&host.transfer_admission(alias, "speech")?)
                .map_err(ServiceError::inference)?;
        }
        for (alias, model) in &encoders {
            if !model.source().is_bound()
                || model.max_context() == 0
                || model.vocabulary_size() == 0
                || !(1..=MAX_DIMENSIONS).contains(&model.dimensions())
            {
                return Err(ServiceError::new(
                    400,
                    "invalid_model",
                    "unbound or unbounded encoder",
                ));
            }
            model
                .validate_admission(&host.transfer_admission(alias, "encoder")?)
                .map_err(ServiceError::inference)?;
        }
        for (alias, handle) in &models {
            validate_alias(alias)?;
            let model = handle.model();
            if !model.source().is_bound()
                || config.context > model.max_context()
                || matches!(A::prefill(handle), Prefill::Chunks(0))
            {
                return Err(ServiceError::new(
                    400,
                    "invalid_model",
                    "unbound source or context/quantum exceeds prepared model admission",
                ));
            }
        }
        if let WorkPolicy::Waves(policy) = config.policy {
            let max_wave_tokens =
                config
                    .context
                    .checked_mul(policy.max_members)
                    .ok_or_else(|| {
                        ServiceError::new(400, "invalid_config", "wave token budget overflow")
                    })?;
            let min_fair_round_tokens =
                policy
                    .max_tokens
                    .checked_add(config.active)
                    .ok_or_else(|| {
                        ServiceError::new(400, "invalid_config", "fair round token budget overflow")
                    })?;
            let max_fair_round_tokens = config
                .context
                .checked_mul(config.active)
                .and_then(|tokens| tokens.checked_add(config.active))
                .ok_or_else(|| {
                    ServiceError::new(400, "invalid_config", "fair round capacity overflow")
                })?;
            if config.workers != 1
                || policy.max_members == 0
                || policy.max_members > config.active
                || policy.max_tokens == 0
                || policy.max_tokens > max_wave_tokens
                || !(1..=512 * 1024 * 1024).contains(&policy.max_scratch)
                || policy.fairness.round_tokens < min_fair_round_tokens
                || policy.fairness.round_tokens > max_fair_round_tokens
                || !(1..=1024).contains(&policy.fairness.prefill_age_rounds)
                || policy.prefill_quantum == 0
                || policy.prefill_quantum > config.context
            {
                return Err(ServiceError::new(
                    400,
                    "invalid_config",
                    "bounded single-worker wave policy required",
                ));
            }
            match policy.cohorts {
                Cohorts::Fused { widths, .. } => {
                    waves::partition_supported_cohorts(0, widths)?;
                    if !widths.contains(&policy.cohort_width)
                        || policy.cohort_width > policy.max_members
                    {
                        return Err(ServiceError::new(
                            400,
                            "invalid_config",
                            "cohort width exceeds admitted wave or supported widths",
                        ));
                    }
                }
                _ if policy.cohort_width != 0 => {
                    return Err(ServiceError::new(
                        400,
                        "invalid_config",
                        "non-cohort policy has a cohort width",
                    ));
                }
                _ => {}
            }
        }
        Self::start::<A>(models, encoders, speech, host, config)
    }
    fn start<A: Executor>(
        models: BTreeMap<String, A::Model>,
        encoders: BTreeMap<String, Arc<dyn EmbeddingEncoder>>,
        speech: BTreeMap<String, Arc<dyn SpeechRecognizer>>,
        host: HostPool,
        config: WorkerConfig,
    ) -> Result<Self, ServiceError> {
        let active_limit = config.workers.checked_mul(config.active).ok_or_else(|| {
            ServiceError::new(400, "invalid_config", "active worker capacity overflow")
        })?;
        let control_charge = host.charge(
            "serve-workers",
            "shared-control",
            AllocationClass::Scratch,
            pending::checked_sum(&[
                arc_bytes::<registry::Registry<A>>()?,
                arc_bytes::<WorkStats>()?,
                // One owner per distinct prepared model (at most alias count).
                // Covers the additional route Arc/state and stable slot.
                (arc_bytes::<registry::Route<A>>()? + fixed_size::<Arc<registry::Route<A>>>()?)
                    .checked_mul((models.len() + encoders.len() + speech.len()) as u64)
                    .ok_or_else(|| {
                        ServiceError::new(400, "size_overflow", "route owner inventory overflow")
                    })?,
            ])?,
        )?;
        let models = Arc::new(registry::Registry::<A>::new(models, encoders, speech)?);
        let stats = Arc::new(WorkStats::default());
        let queue = PendingQueue::new(host.clone(), config.max_pending)?;
        let rows = vecdeque_backing_bytes::<worker::Active<A>>(config.active)?;
        let extra = if matches!(config.policy, WorkPolicy::Waves(_)) {
            // Three persistent queues plus bounded wave/group/retry temporaries.
            // Recursive retry strictly shrinks a band or removes a cancelled row.
            let copies = config
                .active
                .checked_mul(4)
                .and_then(|n| n.checked_add(8))
                .ok_or_else(|| ServiceError::new(400, "size_overflow", "wave scratch overflow"))?;
            pending::checked_sum(&[
                rows.checked_mul(copies as u64).ok_or_else(|| {
                    ServiceError::new(400, "size_overflow", "wave scratch overflow")
                })?,
                pending::vec_bytes::<usize>(
                    config
                        .active
                        .checked_mul(config.active)
                        .and_then(|n| n.checked_mul(16))
                        .ok_or_else(|| {
                            ServiceError::new(400, "size_overflow", "wave index scratch overflow")
                        })?,
                )?,
                A::scratch_bytes(config.active)?,
            ])?
        } else {
            rows
        };
        let extra = if models.has_finite_tasks() {
            pending::checked_sum(&[
                extra,
                vecdeque_backing_bytes::<worker::ActiveFinite>(config.active)?,
            ])?
        } else {
            extra
        };
        let worker_charge = workers_charge(
            &host,
            config.workers,
            std::alloc::Layout::new::<worker::WorkerCaptures<A>>(),
            Some(worker_name_bound(config.workers)),
            extra,
        )?;
        let mut handles: Vec<thread::JoinHandle<()>> = Vec::with_capacity(config.workers);
        for index in 0..config.workers {
            let receiver = match queue.worker() {
                Ok(receiver) => receiver,
                Err(error) => {
                    queue.close();
                    for handle in handles.drain(..) {
                        let _ = handle.join();
                    }
                    return Err(error);
                }
            };
            let captures = worker::WorkerCaptures::<A> {
                queue: receiver,
                models: models.clone(),
                host: host.clone(),
                stats: stats.clone(),
                config,
            };
            match thread::Builder::new()
                .name(format!("zg-decode-{index}"))
                .spawn(move || captures.run())
            {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    queue.close();
                    for handle in handles.drain(..) {
                        let _ = handle.join();
                    }
                    return Err(ServiceError::new(500, "worker_spawn", error.to_string()));
                }
            }
        }
        Ok(Self {
            models,
            host,
            queue,
            stats,
            active_limit,
            workers: handles,
            _worker_charge: worker_charge,
            next_id: AtomicU64::new(1),
            config,
            _control_charge: control_charge,
        })
    }
    pub fn model(&self, alias: &str) -> Option<ModelInstance> {
        registry::instance(self.models.as_ref(), alias)
    }
    // Called under the original runtime lifecycle lock. Nothing is retained
    // from the model by the registration token.
    pub(crate) fn registration(&self, alias: &str) -> Option<RegisteredOwner> {
        self.models.registration(alias)
    }

    pub(crate) fn aliases<'a>(
        &'a self,
        model: &'a ModelInstance,
    ) -> crate::Result<impl Iterator<Item = &'a str> + 'a> {
        let mut aliases = self
            .model_ids()
            .filter(move |id| {
                self.model(id)
                    .is_some_and(|registered| registered.shares_source(model))
            })
            .peekable();
        if aliases.peek().is_none() {
            return Err(crate::Error::State(
                "model is not the original registered owner".into(),
            ));
        }
        Ok(aliases)
    }
    pub(crate) fn begin_unload(
        &self,
        model: &ModelInstance,
    ) -> crate::Result<Option<registry::SelectiveRetirement>> {
        self.models.begin_unload(model)
    }
    pub(crate) fn begin_unload_registration(
        &self,
        alias: &str,
        id: u64,
    ) -> crate::Result<Option<registry::SelectiveRetirement>> {
        self.models.begin_unload_registration(alias, id)
    }
    pub fn host(&self) -> &HostPool {
        &self.host
    }
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        registry::ModelIds {
            models: self.models.as_ref(),
            index: 0,
            public: false,
        }
    }
    /// Text aliases plus explicitly qualified public encoder aliases.
    pub fn serving_model_ids(&self) -> impl Iterator<Item = &str> {
        registry::ModelIds {
            models: self.models.as_ref(),
            index: 0,
            public: true,
        }
    }

    pub fn public_embedding_dimensions(&self, alias: &str) -> Result<usize, ServiceError> {
        let (model, reader) = self.models.acquire_encoder(alias, true)?;
        let dimensions = model.dimensions();
        drop(model);
        drop(reader);
        Ok(dimensions)
    }

    /// A session token retains no model reader or executable. The route is
    /// immutable within a router; a replaced router issues a new registration.
    pub fn realtime_text_registration(&self, alias: &str) -> Result<u64, ServiceError> {
        self.models.realtime_text_registration(alias)
    }

    pub fn request_realtime_text_admitted(
        &self,
        request: SemanticRequest,
        registration: u64,
        ingress: Vec<ServingCharge>,
        sharing: Option<crate::state_cache::PrefixSharingScope>,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_realtime_text_with(request, registration, ingress, sharing, MAX_INPUT_BYTES)
    }

    pub(crate) fn request_realtime_text_with(
        &self,
        request: SemanticRequest,
        registration: u64,
        ingress: Vec<ServingCharge>,
        sharing: Option<crate::state_cache::PrefixSharingScope>,
        input_limit: usize,
    ) -> Result<RequestEvents, ServiceError> {
        if self.realtime_text_registration(&request.model)? != registration {
            return Err(ServiceError::new(
                409,
                "model_changed",
                "realtime model registration changed",
            ));
        }
        if !request.input.is_chat() || !request.stream || request.token_metrics.is_some() {
            return Err(ServiceError::new(
                400,
                "unsupported_task",
                "realtime requires a streamed semantic conversation",
            ));
        }
        self.request_with(request, Some(ingress), sharing, input_limit)
    }

    pub fn validate_chat(&self, alias: &str) -> Result<(), ServiceError> {
        let model = self.models.model(alias).ok_or_else(|| {
            if self.models.registration(alias).is_some() {
                ServiceError::new(400, "unsupported_task", "model is not a text generator")
            } else {
                ServiceError::new(404, "model_not_found", "model is not registered")
            }
        })?;
        model.validate_chat()
    }
    pub fn scheduler_snapshot(&self) -> worker::SchedulerSnapshot {
        worker::SchedulerSnapshot {
            config: self.config,
            queue: self.queue_snapshot(),
            host: self.host.budget_snapshot(),
            active_limit: self.active_limit,
            cohort_width: match self.config.policy {
                WorkPolicy::Sequential => 0,
                WorkPolicy::Waves(p) => p.cohort_width,
            },
            counters: self.stats.snapshot(),
        }
    }
    pub fn catalog_snapshot(&self) -> crate::Result<Vec<crate::diagnostics::ModelSnapshot>> {
        self.models.catalog_snapshot(None)
    }
    pub(crate) fn selected_catalog_snapshot(
        &self,
        registrations: &[u64],
    ) -> crate::Result<Vec<crate::diagnostics::ModelSnapshot>> {
        self.models.catalog_snapshot(Some(registrations))
    }
    pub(crate) fn ready_counts(&self) -> (usize, usize) {
        self.models.ready_counts()
    }
    pub fn max_output(&self) -> usize {
        self.config.output
    }
    pub fn max_context(&self) -> usize {
        self.config.context
    }
    pub fn admission(&self) -> Result<serde_json::Value, ServiceError> {
        let scheduler = self.scheduler_snapshot();
        let mut report = self.models.admission(&scheduler)?;
        // Preserve legacy adapter diagnostics and attach the same typed
        // observations available to the Rust SDK.
        if let Some(fields) = report.as_object_mut() {
            fields.insert(
                "catalog".into(),
                serde_json::to_value(self.catalog_snapshot()?)
                    .map_err(|e| ServiceError::new(500, "diagnostics", e.to_string()))?,
            );
            fields.insert(
                "scheduler".into(),
                serde_json::to_value(scheduler)
                    .map_err(|e| ServiceError::new(500, "diagnostics", e.to_string()))?,
            );
        }
        Ok(report)
    }
    pub fn invalidate_prefix_scope(
        &self,
        sharing: crate::state_cache::PrefixSharingScope,
    ) -> Result<crate::state_cache::PrefixCacheEviction, ServiceError> {
        self.models.invalidate_prefix_scope(sharing)
    }
    pub fn memory_snapshot(&self) -> Option<(u64, u64, u64, u64)> {
        self.host.snapshot()
    }
    pub fn queue_snapshot(&self) -> QueueSnapshot {
        self.queue.snapshot()
    }
    pub fn active_sessions(&self) -> usize {
        self.stats.active.load(Ordering::Acquire)
    }
    pub fn peak_active_sessions(&self) -> usize {
        self.stats.peak.load(Ordering::Acquire)
    }
    pub fn active_limit(&self) -> usize {
        self.active_limit
    }
    pub fn charge_ingress(&self, session: &str, bytes: u64) -> Result<ServingCharge, ServiceError> {
        self.host
            .charge("http-ingress", session, AllocationClass::Scratch, bytes)
    }
    #[allow(dead_code)]
    pub fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
        self.request_semantic(request.into())
    }
    pub fn request_scoped(
        &self,
        request: Request,
        sharing: crate::state_cache::PrefixSharingScope,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_semantic_scoped(request.into(), sharing)
    }
    pub fn request_semantic(
        &self,
        request: SemanticRequest,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_with(request, None, None, MAX_INPUT_BYTES)
    }
    pub fn request_semantic_scoped(
        &self,
        request: SemanticRequest,
        sharing: crate::state_cache::PrefixSharingScope,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_with(request, None, Some(sharing), MAX_INPUT_BYTES)
    }
    /// Transfer already-admitted ingress without copying or re-admitting it.
    pub fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_semantic_admitted(request.into(), ingress)
    }
    pub fn request_admitted_scoped(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
        sharing: crate::state_cache::PrefixSharingScope,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_semantic_admitted_scoped(request.into(), ingress, sharing)
    }
    pub fn request_semantic_admitted(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_with(request, Some(ingress), None, MAX_INPUT_BYTES)
    }
    pub fn request_semantic_admitted_scoped(
        &self,
        request: SemanticRequest,
        ingress: Vec<ServingCharge>,
        sharing: crate::state_cache::PrefixSharingScope,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_with(request, Some(ingress), Some(sharing), MAX_INPUT_BYTES)
    }
    /// Count the selected model's complete rendered input (including its
    /// generation prefix). The borrowed input stays caller-owned. No queue
    /// ticket, session, output reserve, prefix lookup or generation metrics.
    pub fn count_input_tokens(
        &self,
        alias: &str,
        input: &TextInput,
    ) -> Result<usize, ServiceError> {
        self.count_input_tokens_with(alias, input, MAX_INPUT_BYTES)
    }

    pub(crate) fn count_input_tokens_with(
        &self,
        alias: &str,
        input: &TextInput,
        max_input_bytes: usize,
    ) -> Result<usize, ServiceError> {
        // Drop all model/tokenizer handles before releasing the registry
        // reader, including on validation and physical admission failure.
        let (model, reader) = self.models.acquire(alias)?;
        let _reader = reader;
        let model = model;
        let input_limit = max_input_bytes.min(MAX_INPUT_BYTES);
        validate_text_input(input, input_limit)?;
        if input.is_chat() {
            model.validate_chat()?;
        }
        let tokenizer = model.tokenizer();
        let bounds = prompt_bounds(tokenizer.as_ref(), input, input_limit)?;
        let _charge = self.host.charge(
            alias,
            "input-token-count",
            AllocationClass::Scratch,
            prompt_backing_bytes(&bounds)?,
        )?;
        let tokens = tokenizer
            .semantic_prompt_tokens(input)
            .map_err(ServiceError::inference)?;
        if tokens.len() > bounds.tokens {
            return Err(ServiceError::new(
                500,
                "tokenizer_bound",
                "tokenizer exceeded its admitted token bound",
            ));
        }
        if tokens.is_empty() || tokens.len() > self.config.context.min(model.max_context()) {
            return Err(ServiceError::new(
                400,
                "context_overflow",
                "input exceeds bounded model context",
            ));
        }
        Ok(tokens.len())
    }

    fn next_request_id(&self) -> Result<u64, ServiceError> {
        self.next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| ServiceError::new(503, "id_exhausted", "request IDs exhausted"))
    }

    pub fn request_transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionReceipt, ServiceError> {
        self.request_speech_with(SpeechTask::Transcription, request, MAX_INPUT_BYTES, None)
    }
    pub fn request_translation(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionReceipt, ServiceError> {
        self.request_speech_with(SpeechTask::Translation, request, MAX_INPUT_BYTES, None)
    }
    fn speech_limits(
        &self,
        alias: &str,
        model: &dyn SpeechRecognizer,
        task: SpeechTask,
        input_limit: usize,
    ) -> Result<crate::transcription::TranscriptionLimits, ServiceError> {
        let registration = self
            .models
            .registration(alias)
            .ok_or_else(|| ServiceError::new(404, "model_not_found", "model is not registered"))?;
        let context = self.config.context.min(model.max_context());
        let limits = crate::transcription::TranscriptionLimits {
            registration_id: registration.id,
            task,
            max_samples: model
                .max_samples()
                .min(input_limit.min(MAX_INPUT_BYTES) / 4),
            max_tokens: model
                .max_output_tokens()
                .min(self.config.output)
                .min(context.saturating_sub(1)),
            max_output_bytes: model.max_output_bytes(),
        };
        if limits.max_samples == 0 || limits.max_tokens == 0 {
            return Err(ServiceError::new(
                400,
                "unsupported_task",
                "no admitted speech input/output capacity",
            ));
        }
        Ok(limits)
    }
    pub fn public_transcription_limits(
        &self,
        alias: &str,
    ) -> Result<crate::transcription::TranscriptionLimits, ServiceError> {
        self.public_speech_limits_with(SpeechTask::Transcription, alias, MAX_INPUT_BYTES)
    }
    pub fn public_translation_limits(
        &self,
        alias: &str,
    ) -> Result<crate::transcription::TranscriptionLimits, ServiceError> {
        self.public_speech_limits_with(SpeechTask::Translation, alias, MAX_INPUT_BYTES)
    }
    pub(crate) fn public_speech_limits_with(
        &self,
        task: SpeechTask,
        alias: &str,
        input_limit: usize,
    ) -> Result<crate::transcription::TranscriptionLimits, ServiceError> {
        let (model, reader) = self.models.acquire_speech(alias, task, true)?;
        let reader = reader;
        let model = model;
        let limits = self.speech_limits(alias, model.as_ref(), task, input_limit);
        drop(model);
        drop(reader);
        limits
    }
    pub fn request_public_transcription(
        &self,
        request: TranscriptionRequest,
        limits: crate::transcription::TranscriptionLimits,
    ) -> Result<TranscriptionReceipt, ServiceError> {
        self.request_speech_with(
            SpeechTask::Transcription,
            request,
            MAX_INPUT_BYTES,
            Some(limits),
        )
    }
    pub fn request_public_translation(
        &self,
        request: TranscriptionRequest,
        limits: crate::transcription::TranscriptionLimits,
    ) -> Result<TranscriptionReceipt, ServiceError> {
        self.request_speech_with(
            SpeechTask::Translation,
            request,
            MAX_INPUT_BYTES,
            Some(limits),
        )
    }
    pub(crate) fn request_speech_with(
        &self,
        task: SpeechTask,
        request: TranscriptionRequest,
        input_limit: usize,
        expected: Option<crate::transcription::TranscriptionLimits>,
    ) -> Result<TranscriptionReceipt, ServiceError> {
        let (model, reader) =
            self.models
                .acquire_speech(&request.model, task, expected.is_some())?;
        let reader = reader;
        let model = model;
        if let Some(expected) = expected
            && self.speech_limits(&request.model, model.as_ref(), task, input_limit)? != expected
        {
            return Err(ServiceError::new(
                409,
                "model_changed",
                "speech model capability changed",
            ));
        }
        let context = self.config.context.min(model.max_context());
        request.validate(
            model.as_ref(),
            input_limit.min(MAX_INPUT_BYTES),
            context,
            self.config.output,
        )?;
        let id = self.next_request_id()?;
        let ticket = self.queue.reserve()?;
        let (job, receipt) = SpeechJob::prepare(
            (task, request),
            model,
            reader,
            &self.host,
            &self.queue,
            id,
            context,
        )?;
        self.queue
            .enqueue(ticket, Work::Finite(FiniteJob::Speech(job)))
            .map_err(|(error, _)| error)?;
        Ok(receipt)
    }
    pub fn request_embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingReceipt, ServiceError> {
        self.request_embeddings_with(request, MAX_INPUT_BYTES, None)
    }

    /// Wire adapters price the projection before admission. Recheck the
    /// public capability and exact width on the original registry reader,
    /// even if the alias was retired/reloaded after projection preflight.
    pub fn request_public_embeddings(
        &self,
        request: EmbeddingRequest,
        dimensions: usize,
    ) -> Result<EmbeddingReceipt, ServiceError> {
        self.request_embeddings_with(request, MAX_INPUT_BYTES, Some(dimensions))
    }
    pub(crate) fn request_embeddings_with(
        &self,
        request: EmbeddingRequest,
        input_limit: usize,
        public_dimensions: Option<usize>,
    ) -> Result<EmbeddingReceipt, ServiceError> {
        let (model, reader) = self
            .models
            .acquire_encoder(&request.model, public_dimensions.is_some())?;
        let reader = reader;
        let model = model;
        if public_dimensions.is_some_and(|width| width != model.dimensions()) {
            return Err(ServiceError::new(
                409,
                "model_changed",
                "embedding model dimensions changed",
            ));
        }
        let context = self.config.context.min(model.max_context());
        request.validate(
            input_limit.min(MAX_INPUT_BYTES),
            context,
            model.vocabulary_size(),
        )?;
        let id = self.next_request_id()?;
        let ticket = self.queue.reserve()?;
        let (job, receipt) =
            EmbeddingJob::prepare(request, model, reader, &self.host, &self.queue, id, context)?;
        self.queue
            .enqueue(ticket, Work::Finite(FiniteJob::Encoder(job)))
            .map_err(|(error, _)| error)?;
        Ok(receipt)
    }

    pub(crate) fn request_with(
        &self,
        mut request: SemanticRequest,
        ingress: Option<Vec<ServingCharge>>,
        prefix_sharing: Option<crate::state_cache::PrefixSharingScope>,
        max_input_bytes: usize,
    ) -> Result<RequestEvents, ServiceError> {
        let (model, model_reader) = self.models.acquire(&request.model)?;
        let input_limit = max_input_bytes.min(MAX_INPUT_BYTES);
        request.validate_input(input_limit)?;
        if request.max_tokens == 0 || request.max_tokens > self.config.output {
            return Err(ServiceError::new(
                400,
                "invalid_request",
                "request exceeds input/output bounds",
            ));
        }
        // Refuse unsupported model intent before pricing or tokenizing it.
        if request.input.is_chat() {
            model.validate_chat()?;
        }
        let id = self.next_request_id()?;
        // Count/admission reservation precedes every request-derived
        // allocation and queue publication.
        let ticket = self.queue.reserve()?;
        let request_bytes = pending::checked_sum(&[
            pending::heap_strings(&[&request.model])?,
            request
                .input
                .heap_bytes()
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| {
                    ServiceError::new(400, "size_overflow", "semantic input backing overflow")
                })?,
            request
                .token_metrics
                .as_ref()
                .and_then(|m| m.wire_request_id.as_ref())
                .map_or(Ok(0), |id| pending::heap_strings(&[id]))?,
        ])?;
        let charges = Arc::new(match ingress {
            Some(ingress) => ingress,
            None => vec![self.host.charge(
                &request.model,
                "request-copy",
                AllocationClass::Scratch,
                request_bytes,
            )?],
        });
        let event_copy_bytes = pending::checked_sum(&[
            pending::heap_strings(&[&request.model])?,
            request
                .token_metrics
                .as_ref()
                .and_then(|m| m.wire_request_id.as_ref())
                .map_or(Ok(0), |id| pending::heap_strings(&[id]))?,
        ])?;
        let event_copy_charge = self.host.charge(
            &request.model,
            "event-copy",
            AllocationClass::Scratch,
            event_copy_bytes,
        )?;
        let max_tokens = request.max_tokens.min(self.config.context);
        let tokenizer = model.tokenizer();
        let bounds = prompt_bounds(tokenizer.as_ref(), &request.input, input_limit)?;
        let prompt_bytes = pending::checked_sum(&[
            prompt_backing_bytes(&bounds)?,
            // The presenter decodes the whole committed prefix.
            u64::try_from(tokenizer.decode_scratch_bound(max_tokens)).map_err(|_| {
                ServiceError::new(400, "size_overflow", "decode scratch bound overflow")
            })?,
        ])?;
        let prompt_charge = self.host.charge(
            &request.model,
            "prompt-tokens",
            AllocationClass::RecurrentState,
            prompt_bytes,
        )?;
        model.validate_request(&request)?;
        let model_alias = request.model.clone();
        let token_metrics = request.token_metrics.clone();
        let output_bound = output_transport_bound(max_tokens, model.max_token_bytes())?;
        let transport = Arc::new(self.host.charge(
            &model_alias,
            "event-transport",
            AllocationClass::Output,
            output_bound,
        )?);
        let (events_tx, events_rx) = mpsc::try_sync_channel(
            &self.host,
            &model_alias,
            "event-channel",
            EVENTS_PER_REQUEST,
        )?;
        let cancel_charge = self.host.charge(
            &model_alias,
            "cancel-token",
            AllocationClass::Scratch,
            super::event_channel::cancel_bytes()?,
        )?;
        let cancel = self.queue.cancel_token(Some(Arc::new(cancel_charge)));
        let cancelled = Arc::clone(cancel.flag());
        let metrics = MetricsLease {
            metrics: request.token_metrics.take(),
            _charges: token_metrics.is_some().then(|| Arc::clone(&charges)),
        };
        let scheduled = Scheduled {
            payload: Some(RequestLease {
                request: SemanticRequest {
                    max_tokens,
                    ..request
                },
                prefix_sharing,
                _charges: charges,
            }),
            metrics,
            events: events_tx,
            cancelled,
            cancel: cancel.clone(),
            prompt_charge: Some(prompt_charge),
            transport: Some(Arc::clone(&transport)),
            model_reader: Some(model_reader),
        };
        if let Err((error, _)) = self.queue.enqueue(ticket, Work::Text(scheduled)) {
            return Err(error);
        }
        Ok(RequestEvents::new(
            id,
            model_alias,
            events_rx,
            cancel,
            Some(event_copy_charge),
            Some(transport),
        )
        .observed(token_metrics))
    }
}
impl Router {
    pub(crate) fn close_admission(&self) {
        self.queue.close();
    }
    pub fn shutdown(&mut self) -> std::io::Result<()> {
        let before = self.stats.failed.load(Ordering::Acquire);
        self.close_admission();
        let mut failed = false;
        for handle in self.workers.drain(..) {
            failed |= handle.join().is_err();
        }
        if failed {
            Err(std::io::Error::other("CPU serving worker panicked"))
        } else if self.stats.retirement_failed.load(Ordering::Acquire)
            || self.models.retirement_pending()
            || (matches!(self.config.policy, WorkPolicy::Waves(_))
                && self.stats.failed.load(Ordering::Acquire) != before)
        {
            Err(std::io::Error::other("source retirement failed"))
        } else {
            Ok(())
        }
    }
}
impl Drop for Router {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
pub fn output_transport_bound(
    max_tokens: usize,
    max_token_bytes: usize,
) -> Result<u64, ServiceError> {
    let copies = 3u64;
    let json_escape = 6u64;
    let text_per_token = u64::try_from(max_token_bytes)
        .ok()
        .and_then(|n| n.checked_mul(copies))
        .and_then(|n| n.checked_mul(json_escape))
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "output text bound overflow"))?;
    // Generated IDs, presenter events and one metrics record/JSON twin per
    // token are distinct typed owners; text escaping does not cover them.
    let typed_per_token =
        u64::try_from(std::mem::size_of::<u32>() + std::mem::size_of::<TokenObservation>())
            .ok()
            .and_then(|n| n.checked_add(super::token_metrics::record_envelope()))
            .ok_or_else(|| {
                ServiceError::new(400, "size_overflow", "output typed bound overflow")
            })?;
    let per_token = text_per_token
        .checked_add(typed_per_token)
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "output bound overflow"))?;
    // The bounded record vector growth, the serializer buffers and the
    // response-level metric objects are once-per-response owners.
    u64::try_from(max_tokens)
        .ok()
        .and_then(|n| n.checked_mul(per_token))
        .and_then(|n| n.checked_add(super::token_metrics::records_growth()))
        .and_then(|n| n.checked_add(super::token_metrics::response_envelope()))
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "output text bound overflow"))
}
/// Admit every decode worker's application-owned inventory before any of
/// them exists: the pinned std thread-control row (name text, capture closure,
/// runtime handler map), the consumer queue handle and the bounded
/// active-session container. Native stacks, guards, TLS and the OS mapping
/// stay external unquantified headroom; the original builder keeps std's
/// stack=None default.
pub fn workers_charge(
    host: &HostPool,
    workers: usize,
    capture: std::alloc::Layout,
    name: Option<(usize, usize)>,
    extra_per_worker: u64,
) -> Result<ServingCharge, ServiceError> {
    let count = u64::try_from(workers)
        .map_err(|_| ServiceError::new(400, "size_overflow", "worker charge overflow"))?;
    let control = zerogravity_core::resource_arbiter::rust_thread_control_bytes(capture, name)
        .map_err(ServiceError::inference)?;
    let per_worker = pending::checked_sum(&[
        control,
        fixed_size::<WorkerQueue<Work>>()?,
        extra_per_worker,
    ])?;
    let handles = pending::vec_bytes::<thread::JoinHandle<()>>(workers)?;
    let total = per_worker
        .checked_mul(count)
        .and_then(|total| total.checked_add(handles))
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "worker charge overflow"))?;
    host.charge(
        "serve-workers",
        "decode-worker",
        AllocationClass::Scratch,
        total,
    )
}

/// Bound the runtime thread name before formatting it: the fixed prefix plus
/// the widest decimal worker index, with one growth step of capacity slack.
fn worker_name_bound(workers: usize) -> (usize, usize) {
    const PREFIX: &str = "zg-decode-";
    let index = workers.saturating_sub(1);
    let digits = if index == 0 {
        1
    } else {
        (usize::BITS - index.leading_zeros()) as usize
    };
    let len = PREFIX.len() + digits;
    (len + 8, len)
}

pub fn validate_alias(id: &str) -> Result<(), ServiceError> {
    if id.is_empty()
        || id.len() > 96
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(ServiceError::new(
            400,
            "invalid_model",
            "duplicate or invalid model alias",
        ));
    }
    Ok(())
}

/// Validate the complete declaration before source IO, not after an earlier
/// alias has already constructed its source. Bounded, allocation-free.
pub fn validate_aliases<'a>(
    aliases: impl IntoIterator<Item = &'a str>,
) -> Result<(), ServiceError> {
    let mut seen = [""; 16];
    let mut count = 0;
    for alias in aliases {
        validate_alias(alias)?;
        if count == seen.len() || seen[..count].contains(&alias) {
            return Err(ServiceError::new(
                400,
                "invalid_model",
                "duplicate or excessive model aliases",
            ));
        }
        seen[count] = alias;
        count += 1;
    }
    if count == 0 {
        return Err(ServiceError::new(400, "invalid_model", "no model aliases"));
    }
    Ok(())
}

/// Validate before the facade opens a source or starts a worker.
pub fn validate_config(
    models: usize,
    workers: usize,
    active_per_worker: usize,
    max_context: usize,
    max_output: usize,
    max_pending: Option<usize>,
) -> Result<(), ServiceError> {
    if !(1..=16).contains(&models)
        || !(1..=32).contains(&workers)
        || !(1..=MAX_ACTIVE_PER_WORKER).contains(&active_per_worker)
        || !(1..=MAX_CONTEXT_TOKENS).contains(&max_context)
        || !(1..=MAX_OUTPUT_TOKENS).contains(&max_output)
        || max_output > max_context
        || max_pending == Some(0)
    {
        return Err(ServiceError::new(
            400,
            "invalid_config",
            "bounded model/worker/context/pending limits",
        ));
    }
    Ok(())
}

fn arc_bytes<T>() -> Result<u64, ServiceError> {
    let overflow = || ServiceError::new(400, "size_overflow", "shared scheduler layout overflow");
    let (layout, _) = std::alloc::Layout::new::<[AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<T>())
        .map_err(|_| overflow())?;
    layout
        .pad_to_align()
        .size()
        .checked_add(64)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(overflow)
}

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod lifecycle_tests;
#[cfg(test)]
#[path = "cpu_tests.rs"]
mod scheduler_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_context_bound_does_not_imply_source_or_output_capacity() {
        for context in [8192, 16384, MAX_CONTEXT_TOKENS] {
            assert!(validate_config(1, 1, 2, context, 3, Some(64)).is_ok());
        }
        assert!(validate_config(1, 1, 2, MAX_CONTEXT_TOKENS + 1, 3, Some(64)).is_err());
        assert!(validate_config(1, 1, 2, 16384, MAX_OUTPUT_TOKENS + 1, Some(64)).is_err());
        assert!(validate_config(1, 1, MAX_ACTIVE_PER_WORKER + 1, 16384, 3, Some(64)).is_err());
    }

    #[test]
    fn explicit_shutdown_reports_worker_panic_and_joins_remaining_workers() {
        let (mut router, host) = Router::transport_test();
        // Retire the original idle worker before installing controlled joins.
        router.shutdown().unwrap();
        let retired = Arc::new(AtomicBool::new(false));
        let witness = Arc::clone(&retired);
        router
            .workers
            .push(thread::spawn(|| panic!("injected CPU worker panic")));
        router.workers.push(thread::spawn(move || {
            witness.store(true, Ordering::Release);
        }));
        assert!(router.shutdown().is_err());
        assert!(retired.load(Ordering::Acquire));
        assert!(router.workers.is_empty());
        drop(router);
        assert_eq!(host.serving_bytes(), 0);
    }

    fn output_tokenizer(byte_level: bool) -> Arc<dyn TextTokenizer> {
        Arc::new(TestTokenizer { byte_level })
    }
    struct TestTokenizer {
        byte_level: bool,
    }
    impl TextTokenizer for TestTokenizer {
        fn eos(&self) -> u32 {
            1
        }
        fn decode(&self, tokens: &[u32], _: bool) -> crate::Result<String> {
            let mut bytes = Vec::new();
            for token in tokens {
                match token {
                    2 => bytes.push(b'A'),
                    3 => bytes.push(b'B'),
                    4 if self.byte_level => bytes.push(0xc3),
                    5 if self.byte_level => bytes.push(0xa9),
                    _ => return Err(crate::Error::State("test token".into())),
                }
            }
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
        fn prompt_tokens(&self, input: &str, _: bool) -> crate::Result<Vec<u32>> {
            Ok(vec![2; input.len()])
        }
        fn prompt_token_bound(&self, bytes: usize, _: bool) -> usize {
            bytes
        }
        fn admitted_prompt_scratch_bound(&self, _: usize, _: bool) -> crate::Result<u64> {
            Ok(0)
        }
        fn decode_scratch_bound(&self, tokens: usize) -> usize {
            tokens * 4
        }
    }

    fn text(event: ResponseEvent) -> String {
        match event {
            ResponseEvent::Text(value) => value,
            _ => panic!("expected text event"),
        }
    }

    fn done(event: ResponseEvent, expected: &str, tokens: usize, reason: &str) {
        match event {
            ResponseEvent::Done {
                output,
                prompt_tokens,
                completion_tokens,
                finish_reason,
                ..
            } => {
                assert_eq!(output, expected);
                assert_eq!(prompt_tokens, 7);
                assert_eq!(completion_tokens, tokens);
                assert_eq!(finish_reason, reason);
            }
            _ => panic!("expected one terminal event"),
        }
    }

    #[test]
    fn blocked_text_then_terminal_done_resume_without_loss_or_duplicate_tokens() {
        let (sender, receiver) = mpsc::sync_channel(1);
        assert!(
            sender
                .try_send(ResponseEvent::Text("occupied".into()))
                .is_ok()
        );
        let mut output = GeneratedText::new(output_tokenizer(false), sender, true, 2, 7);
        assert_eq!(output.present(2).unwrap(), Presentation::Blocked);
        assert_eq!(output.generated, [2]);
        assert_eq!(output.present(3).unwrap_err().code, "output_pending");
        assert_eq!(output.generated, [2]);
        assert_eq!(output.resume().unwrap(), Presentation::Blocked);
        assert_eq!(text(receiver.recv().unwrap()), "occupied");
        assert_eq!(output.resume().unwrap(), Presentation::Ready);
        assert_eq!(text(receiver.recv().unwrap()), "A");
        assert_eq!(output.resume().unwrap(), Presentation::Ready);
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));

        // Text enters the one-event channel; terminal Done remains private.
        assert_eq!(output.present(3).unwrap(), Presentation::Blocked);
        assert_eq!(output.emitted, "AB");
        assert_eq!(output.generated, [2, 3]);
        assert_eq!(output.pending.len(), 1);
        assert_eq!(output.resume().unwrap(), Presentation::Blocked);
        assert_eq!(text(receiver.recv().unwrap()), "B");
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        done(receiver.recv().unwrap(), "AB", 2, "length");
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        assert_eq!(output.present(2).unwrap(), Presentation::Finished);
        assert_eq!(output.generated, [2, 3]);
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn terminal_token_stages_at_most_text_and_done_when_channel_is_full() {
        let (sender, receiver) = mpsc::sync_channel(1);
        assert!(
            sender
                .try_send(ResponseEvent::Text("occupied".into()))
                .is_ok()
        );
        let mut output = GeneratedText::new(output_tokenizer(false), sender, true, 1, 7);
        assert_eq!(output.present(2).unwrap(), Presentation::Blocked);
        assert_eq!(output.pending.len(), 2);
        assert_eq!(output.generated, [2]);
        assert_eq!(text(receiver.recv().unwrap()), "occupied");
        assert_eq!(output.resume().unwrap(), Presentation::Blocked);
        assert_eq!(output.pending.len(), 1);
        assert_eq!(output.emitted, "A");
        assert_eq!(text(receiver.recv().unwrap()), "A");
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        done(receiver.recv().unwrap(), "A", 1, "length");
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn eos_during_backpressure_finishes_once_without_counting_the_eos() {
        let (sender, receiver) = mpsc::sync_channel(1);
        assert!(
            sender
                .try_send(ResponseEvent::Text("occupied".into()))
                .is_ok()
        );
        let tokenizer = output_tokenizer(false);
        let eos = tokenizer.eos();
        let mut output = GeneratedText::new(tokenizer, sender, true, 2, 7);
        assert_eq!(output.present(eos).unwrap(), Presentation::Blocked);
        assert!(output.generated.is_empty());
        assert_eq!(output.pending.len(), 1);
        assert_eq!(output.resume().unwrap(), Presentation::Blocked);
        assert_eq!(text(receiver.recv().unwrap()), "occupied");
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        done(receiver.recv().unwrap(), "", 0, "stop");
        assert_eq!(output.present(eos).unwrap(), Presentation::Finished);
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn incomplete_utf8_waits_then_emits_the_original_complete_decode_once() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut output = GeneratedText::new(output_tokenizer(true), sender, true, 2, 7);
        assert_eq!(output.present(4).unwrap(), Presentation::Ready);
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        assert_eq!(output.present(5).unwrap(), Presentation::Blocked);
        assert_eq!(text(receiver.recv().unwrap()), "é");
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        done(receiver.recv().unwrap(), "é", 2, "length");
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn final_incomplete_utf8_keeps_original_replacement_flush_semantics() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut output = GeneratedText::new(output_tokenizer(true), sender, true, 1, 7);
        assert_eq!(output.present(4).unwrap(), Presentation::Blocked);
        assert_eq!(text(receiver.recv().unwrap()), "�");
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        done(receiver.recv().unwrap(), "�", 1, "length");
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn cpu_wrapper_refuses_backpressure_and_does_not_resume_a_partial_terminal() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut output = GeneratedText::new(output_tokenizer(false), sender, true, 1, 7);
        let error = output.accept(2).unwrap_err();
        assert_eq!((error.status, error.code), (503, "backpressure"));
        assert_eq!(text(receiver.recv().unwrap()), "A");
        assert_eq!(output.resume().unwrap_err().code, "backpressure");
        assert_eq!(output.accept(3).unwrap_err().code, "backpressure");
        assert_eq!(output.generated, [2]);
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn disconnected_terminal_never_replays_text_or_accepts_another_token() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut output = GeneratedText::new(output_tokenizer(false), sender, true, 1, 7);
        assert_eq!(output.present(2).unwrap(), Presentation::Blocked);
        assert_eq!(text(receiver.recv().unwrap()), "A");
        drop(receiver);
        let error = output.resume().unwrap_err();
        assert_eq!((error.status, error.code), (499, "cancelled"));
        assert_eq!(output.resume().unwrap_err().code, "cancelled");
        assert_eq!(output.present(3).unwrap_err().code, "cancelled");
        assert_eq!(output.generated, [2]);
        assert!(output.pending.is_empty());
    }

    #[test]
    fn nonstreaming_cpu_accept_keeps_original_text_usage_and_eos_behavior() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let tokenizer = output_tokenizer(false);
        let eos = tokenizer.eos();
        let mut output = GeneratedText::new(tokenizer, sender, false, 2, 7);
        assert!(!output.accept(2).unwrap());
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        assert!(output.accept(eos).unwrap());
        done(receiver.recv().unwrap(), "A", 1, "stop");
        assert!(output.accept(3).unwrap());
        assert_eq!(output.generated, [2]);
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn sequential_presentation_preserves_supplied_publication_observation() {
        for stream in [false, true] {
            let (sender, receiver) = mpsc::sync_channel(4);
            let when = Instant::now();
            let observation = GpuTokenObservation {
                producer_frontier: 47,
                producer_root: Some(GpuRootObservation {
                    session_id: 17,
                    root: 3,
                    session_epoch: 2,
                }),
                fused_cohort: None,
            };
            let mut output = GeneratedText::new(output_tokenizer(false), sender, stream, 1, 47)
                .observed(
                    Some(TokenMetricsRequest {
                        started: when,
                        wire_request_id: None,
                    }),
                    7,
                );
            assert!(output.accept_observed(2, when, Some(observation)).unwrap());
            match receiver.recv().unwrap() {
                ResponseEvent::TokenText { token, .. } => {
                    assert_eq!((token.index, token.id), (0, 2));
                    assert_eq!(token.commit_observed_at, when);
                    assert_eq!(token.gpu, Some(observation));
                }
                _ => panic!("original supplied observation missing"),
            }
            match receiver.recv().unwrap() {
                ResponseEvent::Done {
                    output,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    ..
                } => {
                    assert_eq!(
                        (
                            output.as_str(),
                            prompt_tokens,
                            completion_tokens,
                            finish_reason
                        ),
                        ("A", 47, 1, "length")
                    );
                }
                _ => panic!("sequential terminal output missing"),
            }
            assert_eq!(output.generated_token_count(), 1);
        }
    }

    #[test]
    fn request_receiver_cancellation_drops_queued_session_without_inference() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let (_sender, receiver) = mpsc::sync_channel::<ResponseEvent>(1);
        let receipt = RequestEvents {
            id: 7,
            model: "source-owned".into(),
            events: receiver,
            token_metrics: None,
            _cancel: CancelToken::from_flag(Arc::clone(&cancelled)),
            _request: None,
            _transport: None,
        };
        assert!(!cancelled.load(Ordering::Acquire));
        drop(receipt);
        assert!(cancelled.load(Ordering::Acquire));
    }
    #[test]
    fn bounded_cpu_host_budget_refuses_invalid_unbounded_config() {
        for budget in [0, 1, 255, 1_048_577, usize::MAX] {
            let failure = match HostPool::new(budget) {
                Ok(_) => panic!("invalid budget accepted"),
                Err(error) => error,
            };
            assert_eq!(failure.code, "invalid_host_budget");
        }
    }

    #[test]
    fn observed_utf8_tokens_keep_independent_ids_through_credit_retries() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let started = Instant::now() - Duration::from_secs(1);
        let mut output = GeneratedText::new(output_tokenizer(true), sender, true, 2, 7).observed(
            Some(TokenMetricsRequest {
                started,
                wire_request_id: Some("run:byte-pair".into()),
            }),
            7,
        );
        assert_eq!(
            output.present_committed(4, started).unwrap(),
            Presentation::Ready
        );
        assert_eq!(
            output
                .present_committed(5, started + Duration::from_millis(1))
                .unwrap(),
            Presentation::Blocked
        );
        assert_eq!(output.generated, [4, 5]);
        assert_eq!(output.present(2).unwrap_err().code, "output_pending");
        match receiver.recv().unwrap() {
            ResponseEvent::TokenText { text, token } => {
                assert!(text.is_empty());
                assert_eq!((token.index, token.id), (0, 4));
                assert_eq!(token.commit_observed_at, started);
            }
            _ => panic!("missing independent first byte token"),
        }
        assert_eq!(output.resume().unwrap(), Presentation::Blocked);
        match receiver.recv().unwrap() {
            ResponseEvent::TokenText { text, token } => {
                assert_eq!(text, "é");
                assert_eq!((token.index, token.id), (1, 5));
            }
            _ => panic!("missing second token and coalesced UTF8"),
        }
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        done(receiver.recv().unwrap(), "é", 2, "length");
        assert_eq!(output.resume().unwrap(), Presentation::Finished);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn observed_eos_has_no_output_id_and_invalid_id_or_clock_never_appends() {
        let started = Instant::now() - Duration::from_secs(1);
        let (sender, receiver) = mpsc::sync_channel(2);
        let tokenizer = output_tokenizer(false);
        let eos = tokenizer.eos();
        let mut eos_output = GeneratedText::new(tokenizer, sender, true, 2, 7).observed(
            Some(TokenMetricsRequest {
                started,
                wire_request_id: None,
            }),
            7,
        );
        assert_eq!(
            eos_output.present_committed(eos, started).unwrap(),
            Presentation::Finished
        );
        done(receiver.recv().unwrap(), "", 0, "stop");
        assert!(receiver.try_recv().is_err());
        for (id, when) in [
            (u32::MAX, started),
            (2, started - Duration::from_millis(1)),
            (2, Instant::now() + Duration::from_secs(1)),
        ] {
            let (sender, receiver) = mpsc::sync_channel(2);
            let mut output = GeneratedText::new(output_tokenizer(false), sender, true, 2, 7)
                .observed(
                    Some(TokenMetricsRequest {
                        started,
                        wire_request_id: None,
                    }),
                    7,
                );
            assert_eq!(
                output.present_committed(id, when).unwrap_err().code,
                "invalid_token_observation"
            );
            assert!(output.generated.is_empty());
            assert!(receiver.try_recv().is_err());
        }
        let (sender, receiver) = mpsc::sync_channel(2);
        let mut output = GeneratedText::new(output_tokenizer(false), sender, true, 3, 7).observed(
            Some(TokenMetricsRequest {
                started,
                wire_request_id: None,
            }),
            7,
        );
        assert_eq!(
            output
                .present_committed(2, started + Duration::from_millis(20))
                .unwrap(),
            Presentation::Ready
        );
        assert!(matches!(
            receiver.recv().unwrap(),
            ResponseEvent::TokenText { .. }
        ));
        assert_eq!(
            output
                .present_committed(3, started + Duration::from_millis(10))
                .unwrap_err()
                .code,
            "invalid_token_observation"
        );
        assert_eq!(output.generated, [2]);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn worker_inventory_charge_is_retained_until_its_owner_drops() {
        let host = HostPool::isolated(4096).unwrap();
        let before = host.serving_bytes();
        let charge = workers_charge(
            &host,
            2,
            std::alloc::Layout::new::<worker::WorkerCaptures<HostLogits>>(),
            Some((20, 12)),
            128,
        )
        .unwrap();
        assert!(host.serving_bytes() > before);
        drop(charge);
        assert_eq!(host.serving_bytes(), before);
    }

    #[test]
    fn bounded_container_row_prices_header_slots_and_allocator_overhead() {
        let expected = std::mem::size_of::<VecDeque<u32>>() as u64
            + 5 * std::mem::size_of::<u32>() as u64
            + CONTAINER_ALLOC_OVERHEAD_BYTES;
        assert_eq!(vecdeque_backing_bytes::<u32>(5).unwrap(), expected);
    }
}
