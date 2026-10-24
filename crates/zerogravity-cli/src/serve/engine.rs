//! Shared immutable Qwen source, bounded multi-request CPU scheduler.
//! Concurrent worker threads + cooperative prefill/decode within workers.
//! This is NOT yet fused continuous-batch GEMM, paged KV, or GPU serving.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use zerogravity::prepare::PreparedQwen;
use zerogravity::qwen35::{LoadedQwenModel, QwenDenseCpu, StateSpec};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity::tokenize::QwenTokenizer;

pub(super) const QUEUED_PER_WORKER: usize = 16;
pub(super) const EVENTS_PER_REQUEST: usize = 16;
pub(super) const MAX_TOKEN_OBSERVATIONS: usize = 512;
const PREFILL_CHUNK: usize = 32;
pub(super) const MAX_ACTIVE_PER_WORKER: usize = 8;

#[derive(Clone)]
struct HostPool {
    arbiter: ResourceArbiter,
    key: PhysicalDeviceKey,
}
impl HostPool {
    fn new(configured_mib: usize) -> Result<Self, ServiceError> {
        if !(256..=1_048_576).contains(&configured_mib) {
            return Err(ServiceError::new(
                400,
                "invalid_host_budget",
                "host-mib must be 256..=1048576",
            ));
        }
        let bytes = (configured_mib as u64) * 1024 * 1024;
        // One shared process-wide ledger; repeated routers cannot make up
        // separate physical free-memory budgets for the same host pool.
        let arbiter = ResourceArbiter::process_shared();
        let key = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")
            .map_err(ServiceError::inference)?;
        match arbiter.snapshot(&key) {
            Ok(existing) if existing.limit == bytes => {}
            Ok(_) => {
                return Err(ServiceError::new(
                    409,
                    "host_budget_conflict",
                    "existing host admission pool uses a different configured limit",
                ));
            }
            Err(_) => arbiter
                .register(key.clone(), bytes)
                .map_err(ServiceError::inference)?,
        }
        Ok(Self { arbiter, key })
    }
    fn admit(
        &self,
        alias: &str,
        session: &str,
        kind: AllocationClass,
        bytes: u64,
    ) -> Result<GlobalResourceLease, ServiceError> {
        self.arbiter
            .admit(
                &self.key,
                ResourceOwner::new(alias, session).map_err(ServiceError::inference)?,
                &[AllocationRequest { kind, bytes }],
            )
            .map_err(|error| ServiceError::new(503, "host_budget", error.to_string()))
    }
}
fn session_budget(model: &LoadedQwenModel, context: usize) -> Result<u64, ServiceError> {
    let mut bytes = 0u64;
    for spec in model.state_specs() {
        let count = match *spec {
            StateSpec::Delta { conv, state } => {
                conv.checked_add(state).and_then(|n| n.checked_mul(8))
            }
            StateSpec::Full { kv_width } => kv_width
                .checked_mul(context)
                .and_then(|n| n.checked_mul(16)),
        }
        .ok_or_else(|| {
            ServiceError::new(
                400,
                "state_budget",
                "source session state capacity overflow",
            )
        })?;
        bytes = bytes
            .checked_add(count as u64)
            .ok_or_else(|| ServiceError::new(400, "state_budget", "total state budget overflow"))?;
    }
    // Activation/program/logits scratch envelope is conservative, not
    // a claim of actual driver/device memory.
    bytes = bytes
        .checked_add(32 * 1024 * 1024)
        .ok_or_else(|| ServiceError::new(400, "state_budget", "scratch envelope overflow"))?;
    Ok(bytes)
}

#[derive(Debug, Clone)]
pub struct ServiceError {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}
impl ServiceError {
    pub fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
    pub(super) fn inference(error: zerogravity::Error) -> Self {
        Self::new(500, "inference_error", error.to_string())
    }
}

pub struct Model {
    weights: Arc<LoadedQwenModel>,
    tokenizer: Arc<QwenTokenizer>,
    max_session_bytes: u64,
    _weight_charge: GlobalResourceLease,
}
pub struct Request {
    pub model: String,
    pub input: String,
    pub chat: bool,
    pub max_tokens: usize,
    pub stream: bool,
    pub token_metrics: Option<TokenMetricsRequest>,
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
pub struct NativeCohortObservation {
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
    #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
    pub fn committed(binding: zerogravity::session_root::RootBinding) -> Self {
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
    pub native_cohort: Option<NativeCohortObservation>,
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
        completion_tokens: usize,
        finish_reason: &'static str,
    },
    Error(ServiceError),
}
pub struct RequestEvents {
    pub id: u64,
    pub model: String,
    pub events: mpsc::Receiver<ResponseEvent>,
    pub token_metrics: Option<TokenMetricsRequest>,
    cancelled: Arc<AtomicBool>,
}
impl RequestEvents {
    pub(super) fn new(
        id: u64,
        model: String,
        events: mpsc::Receiver<ResponseEvent>,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            id,
            model,
            events,
            token_metrics: None,
            cancelled,
        }
    }
    pub(super) fn observed(mut self, metrics: Option<TokenMetricsRequest>) -> Self {
        self.token_metrics = metrics;
        self
    }
}
impl Drop for RequestEvents {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
/// Shared CPU/GPU presentation and backpressure semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Presentation {
    Ready,
    Blocked,
    Finished,
}

/// Parsed tokenizer and its source admission have one retained owner.
#[derive(Clone)]
pub(super) enum TextTokenizer {
    Shared(Arc<QwenTokenizer>),
    #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
    Canonical(Arc<zerogravity::canonical47::Canonical47Model>),
    #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
    ShortHf(Arc<zerogravity::short_hf::ShortHfModel>),
}
impl std::ops::Deref for TextTokenizer {
    type Target = QwenTokenizer;
    fn deref(&self) -> &QwenTokenizer {
        match self {
            Self::Shared(tokenizer) => tokenizer,
            #[cfg(all(feature = "canonical47-serving", target_os = "linux"))]
            Self::Canonical(model) => model.tokenizer(),
            #[cfg(all(feature = "short-hf-serving", target_os = "linux"))]
            Self::ShortHf(model) => model.tokenizer(),
        }
    }
}

pub(super) struct GeneratedText {
    tokenizer: TextTokenizer,
    events: mpsc::SyncSender<ResponseEvent>,
    max_tokens: usize,
    prompt_tokens: usize,
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
}
impl GeneratedText {
    pub(super) fn new(
        tokenizer: Arc<QwenTokenizer>,
        events: mpsc::SyncSender<ResponseEvent>,
        stream: bool,
        max_tokens: usize,
        prompt_tokens: usize,
    ) -> Self {
        Self::new_owned(
            TextTokenizer::Shared(tokenizer),
            events,
            stream,
            max_tokens,
            prompt_tokens,
        )
    }
    pub(super) fn new_owned(
        tokenizer: TextTokenizer,
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
        }
    }
    pub(super) fn observed(
        mut self,
        metrics: Option<TokenMetricsRequest>,
        vocabulary: u64,
    ) -> Self {
        self.token_metrics = metrics;
        self.vocabulary_size = vocabulary;
        self
    }
    /// CPU compatibility: bounded output refusal terminates this presenter.
    #[cfg(test)]
    pub(super) fn accept(&mut self, id: u32) -> Result<bool, ServiceError> {
        self.accept_committed(id, Instant::now())
    }
    pub(super) fn accept_committed(
        &mut self,
        id: u32,
        committed: Instant,
    ) -> Result<bool, ServiceError> {
        match self.present_committed(id, committed)? {
            Presentation::Ready => Ok(false),
            Presentation::Finished => Ok(true),
            Presentation::Blocked => Err(self.fail(ServiceError::new(
                503,
                "backpressure",
                "client cannot accept streamed output",
            ))),
        }
    }

    #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
    pub(super) fn generated_token_count(&self) -> usize {
        self.generated.len()
    }

    /// Stage one already-committed token exactly once. The GPU owner resumes
    /// blocked delivery before advancing this packet or presenting another.
    #[cfg(test)]
    pub(super) fn present(&mut self, id: u32) -> Result<Presentation, ServiceError> {
        self.present_committed(id, Instant::now())
    }
    pub(super) fn present_committed(
        &mut self,
        id: u32,
        commit_observed_at: Instant,
    ) -> Result<Presentation, ServiceError> {
        self.present_observed(id, commit_observed_at, None)
    }
    pub(super) fn present_observed(
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
    pub(super) fn resume(&mut self) -> Result<Presentation, ServiceError> {
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

pub(super) struct Scheduled {
    pub(super) request: Request,
    pub(super) events: mpsc::SyncSender<ResponseEvent>,
    pub(super) cancelled: Arc<AtomicBool>,
}
pub struct Router {
    models: Arc<BTreeMap<String, Arc<Model>>>,
    host: HostPool,
    workers: Vec<mpsc::SyncSender<Scheduled>>,
    next_worker: AtomicUsize,
    next_id: AtomicU64,
    pub max_context: usize,
    pub max_output: usize,
}
impl Router {
    pub fn new(
        declared: Vec<(String, PathBuf)>,
        workers: usize,
        active_per_worker: usize,
        max_context: usize,
        max_output: usize,
        host_mib: usize,
    ) -> Result<Self, ServiceError> {
        if declared.is_empty()
            || declared.len() > 16
            || !(1..=32).contains(&workers)
            || !(1..=MAX_ACTIVE_PER_WORKER).contains(&active_per_worker)
            || !(1..=8192).contains(&max_context)
            || !(1..=512).contains(&max_output)
            || max_output > max_context
        {
            return Err(ServiceError::new(
                400,
                "invalid_config",
                "bounded model/worker/context limits",
            ));
        }
        let host = HostPool::new(host_mib)?;
        let mut models = BTreeMap::new();
        // Canonical paths reuse the SAME immutable model/host lease across
        // route aliases. A second alias does not republish a source ID.
        let mut unique_sources = BTreeMap::<PathBuf, Arc<Model>>::new();
        for (id, path) in declared {
            if id.is_empty()
                || id.len() > 96
                || !id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                || models.contains_key(&id)
            {
                return Err(ServiceError::new(
                    400,
                    "invalid_model",
                    "duplicate or invalid model alias",
                ));
            }
            let canonical = std::fs::canonicalize(&path).map_err(|_| {
                ServiceError::new(400, "missing_model", "cannot resolve model source")
            })?;
            if let Some(existing) = unique_sources.get(&canonical) {
                models.insert(id, Arc::clone(existing));
                continue;
            }
            let path = canonical;
            let parent = path.parent().unwrap_or(Path::new("."));
            let input_len = std::fs::metadata(&path)
                .map_err(|_| ServiceError::new(400, "missing_model", "model file unavailable"))?
                .len();
            let tokenizer_bytes = std::fs::metadata(parent.join("tokenizer.json"))
                .map_err(|_| {
                    ServiceError::new(400, "missing_tokenizer", "tokenizer file unavailable")
                })?
                .len();
            let charged = input_len
                .checked_add(tokenizer_bytes.saturating_mul(2))
                .and_then(|n| n.checked_add(128 * 1024 * 1024))
                .ok_or_else(|| {
                    ServiceError::new(400, "model_budget", "model source size overflow")
                })?;
            let weight_charge = host.admit(
                &id,
                "immutable-source",
                AllocationClass::ImmutableWeights,
                charged,
            )?;
            let prepared = PreparedQwen::from_paths(
                &path,
                &parent.join("config.json"),
                &parent.join("tokenizer.json"),
            )
            .map_err(ServiceError::inference)?;
            let (weights, tokenizer) = prepared.into_parts();
            let max_session_bytes = session_budget(&weights, max_context)?;
            let shared = Arc::new(Model {
                weights,
                tokenizer: Arc::new(tokenizer),
                max_session_bytes,
                _weight_charge: weight_charge,
            });
            unique_sources.insert(path, Arc::clone(&shared));
            models.insert(id, shared);
        }
        let models = Arc::new(models);
        let mut senders = Vec::with_capacity(workers);
        for index in 0..workers {
            let (tx, rx) = mpsc::sync_channel(QUEUED_PER_WORKER);
            let loaded = Arc::clone(&models);
            let admission = host.clone();
            thread::Builder::new()
                .name(format!("zg-decode-{index}"))
                .spawn(move || worker(rx, loaded, admission, active_per_worker, max_context))
                .map_err(|e| ServiceError::new(500, "worker_spawn", e.to_string()))?;
            senders.push(tx);
        }
        Ok(Self {
            models,
            host,
            workers: senders,
            next_worker: AtomicUsize::new(0),
            next_id: AtomicU64::new(1),
            max_context,
            max_output,
        })
    }
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }
    pub fn memory_snapshot(&self) -> Option<(u64, u64, u64)> {
        self.host
            .arbiter
            .snapshot(&self.host.key)
            .ok()
            .map(|s| (s.limit, s.reserved, s.remaining))
    }
    pub fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
        if !self.models.contains_key(&request.model) {
            return Err(ServiceError::new(
                404,
                "model_not_found",
                "model is not registered",
            ));
        }
        if request.input.is_empty()
            || request.input.len() > 1024 * 1024
            || request.max_tokens == 0
            || request.max_tokens > self.max_output
        {
            return Err(ServiceError::new(
                400,
                "invalid_request",
                "request exceeds input/output bounds",
            ));
        }
        let model_alias = request.model.clone();
        let token_metrics = request.token_metrics.clone();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if id == u64::MAX {
            return Err(ServiceError::new(
                503,
                "id_exhausted",
                "request IDs exhausted",
            ));
        }
        let (events_tx, events_rx) = mpsc::sync_channel(EVENTS_PER_REQUEST);
        let cancelled = Arc::new(AtomicBool::new(false));
        let index = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        self.workers[index]
            .try_send(Scheduled {
                request: Request {
                    max_tokens: request.max_tokens.min(self.max_context),
                    ..request
                },
                events: events_tx,
                cancelled: Arc::clone(&cancelled),
            })
            .map_err(|err| match err {
                mpsc::TrySendError::Full(_) => {
                    ServiceError::new(429, "queue_full", "request admission queue is full")
                }
                mpsc::TrySendError::Disconnected(_) => {
                    ServiceError::new(503, "worker_unavailable", "inference worker unavailable")
                }
            })?;
        Ok(RequestEvents::new(id, model_alias, events_rx, cancelled).observed(token_metrics))
    }
}
struct Active {
    session: QwenDenseCpu,
    _state_charge: GlobalResourceLease,
    events: mpsc::SyncSender<ResponseEvent>,
    cancelled: Arc<AtomicBool>,
    output: GeneratedText,
    prompt: Vec<u32>,
    prompt_pos: usize,
    logits: Vec<f32>,
    logits_commit_observed_at: Instant,
}
impl Active {
    fn new(
        scheduled: Scheduled,
        model: Arc<Model>,
        max_context: usize,
        host: &HostPool,
    ) -> Result<Self, ServiceError> {
        let prompt = model
            .tokenizer
            .prompt_tokens(&scheduled.request.input, scheduled.request.chat)
            .map_err(ServiceError::inference)?;
        if prompt.is_empty()
            || prompt.len() > max_context
            || prompt
                .len()
                .checked_add(scheduled.request.max_tokens - 1)
                .is_none_or(|n| n > max_context)
        {
            return Err(ServiceError::new(
                400,
                "context_overflow",
                "request exceeds bounded model context",
            ));
        }
        let state_charge = host.admit(
            &scheduled.request.model,
            "cpu-request-state",
            AllocationClass::RecurrentState,
            model.max_session_bytes,
        )?;
        let session = model
            .weights
            .new_session()
            .map_err(ServiceError::inference)?;
        let output = GeneratedText::new(
            Arc::clone(&model.tokenizer),
            scheduled.events.clone(),
            scheduled.request.stream,
            scheduled.request.max_tokens,
            prompt.len(),
        )
        .observed(
            scheduled.request.token_metrics.clone(),
            model.weights.geometry().vocab_size,
        );
        Ok(Self {
            session,
            _state_charge: state_charge,
            events: scheduled.events,
            cancelled: scheduled.cancelled,
            output,
            prompt,
            prompt_pos: 0,
            logits: Vec::new(),
            logits_commit_observed_at: Instant::now(),
        })
    }
    /// One cooperative service quantum: one bounded prefill chunk or
    /// one decode token. Other active sessions advance between quanta.
    fn step(&mut self) -> Result<bool, ServiceError> {
        if self.prompt_pos < self.prompt.len() {
            let end = (self.prompt_pos + PREFILL_CHUNK).min(self.prompt.len());
            if end < self.prompt.len() {
                self.session
                    .prefill_chunk_without_logits(&self.prompt[self.prompt_pos..end])
                    .map_err(ServiceError::inference)?;
            } else {
                self.logits = self
                    .session
                    .prefill_transaction(&self.prompt[self.prompt_pos..end])
                    .map_err(ServiceError::inference)?;
                self.logits_commit_observed_at = Instant::now();
            }
            self.prompt_pos = end;
            return Ok(false);
        }
        let id = self
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .ok_or_else(|| ServiceError::new(500, "empty_logits", "inference produced no logits"))?
            .0 as u32;
        if self
            .output
            .accept_committed(id, self.logits_commit_observed_at)?
        {
            return Ok(true);
        }
        self.logits = self.session.forward(id).map_err(ServiceError::inference)?;
        self.logits_commit_observed_at = Instant::now();
        Ok(false)
    }
}
fn worker(
    rx: mpsc::Receiver<Scheduled>,
    models: Arc<BTreeMap<String, Arc<Model>>>,
    host: HostPool,
    active_limit: usize,
    max_context: usize,
) {
    let mut active = VecDeque::<Active>::new();
    loop {
        while active.len() < active_limit {
            let next = if active.is_empty() {
                match rx.recv_timeout(Duration::from_millis(30)) {
                    Ok(job) => Some(job),
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            } else {
                rx.try_recv().ok()
            };
            let Some(job) = next else {
                break;
            };
            if job.cancelled.load(Ordering::Acquire) {
                continue;
            }
            let model = models
                .get(&job.request.model)
                .expect("router validated model");
            let errors = job.events.clone();
            match Active::new(job, Arc::clone(model), max_context, &host) {
                Ok(inference) => active.push_back(inference),
                Err(err) => {
                    let _ = errors.try_send(ResponseEvent::Error(err));
                }
            }
        }
        let Some(mut current) = active.pop_front() else {
            continue;
        };
        if current.cancelled.load(Ordering::Acquire) {
            continue;
        }
        // Disconnect detection without running an extra token is
        // impossible for std::SyncSender; the send path fails closed.
        match current.step() {
            Ok(true) => {}
            Ok(false) => active.push_back(current),
            Err(error) => {
                let _ = current.events.try_send(ResponseEvent::Error(error));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output_tokenizer(byte_level: bool) -> Arc<QwenTokenizer> {
        let decoder = if byte_level {
            serde_json::json!({"type":"ByteLevel", "add_prefix_space":false,
                "trim_offsets":true, "use_regex":true})
        } else {
            serde_json::json!({"type":"Fuse"})
        };
        let source = serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null,
            "normalizer":null, "pre_tokenizer":null, "post_processor":null,
            "added_tokens":[
                {"id":0,"content":"<|im_start|>","single_word":false,
                    "lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":1,"content":"<|im_end|>","single_word":false,
                    "lstrip":false,"rstrip":false,"normalized":false,"special":true}
            ],
            "decoder":decoder,
            "model":{"type":"WordLevel","unk_token":"[UNK]","vocab":{
                "<|im_start|>":0,"<|im_end|>":1,"A":2,"B":3,"Ã":4,"©":5,"[UNK]":6
            }}
        });
        Arc::new(QwenTokenizer::from_bytes(&serde_json::to_vec(&source).unwrap()).unwrap())
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
    fn request_receiver_cancellation_drops_queued_session_without_inference() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let (_sender, receiver) = mpsc::sync_channel::<ResponseEvent>(1);
        let receipt = RequestEvents {
            id: 7,
            model: "source-owned".into(),
            events: receiver,
            token_metrics: None,
            cancelled: Arc::clone(&cancelled),
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
}
