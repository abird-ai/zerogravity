//! Shared prepared sources, bounded multi-request CPU scheduler.
//! Concurrent worker threads + cooperative prefill/decode within workers.
//! This is NOT yet fused continuous-batch GEMM, paged KV, or GPU serving.

use super::event_channel::mpsc;
use super::pending::{
    self, CancelToken, PendingQueue, Pop, QueueSnapshot, ServingCharge, ServingHost as HostPool,
    WorkerQueue,
};
pub use crate::model_instance::TextTokenizer;
use crate::model_instance::{ModelInstance, PreparedModel, TextSession};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use zerogravity_core::resource_arbiter::{AllocationClass, GlobalResourceLease};

pub const EVENTS_PER_REQUEST: usize = 16;
pub const MAX_TOKEN_OBSERVATIONS: usize = 512;
const PREFILL_CHUNK: usize = 32;
pub const MAX_ACTIVE_PER_WORKER: usize = 8;

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
    pub fn inference(error: zerogravity_core::Error) -> Self {
        Self::new(500, "inference_error", error.to_string())
    }
}
impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.status, self.code, self.message)
    }
}
impl std::error::Error for ServiceError {}
impl From<zerogravity_core::Error> for ServiceError {
    fn from(error: zerogravity_core::Error) -> Self {
        Self::inference(error)
    }
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
    _cancel: CancelToken,
    _request: Option<ServingCharge>,
    _transport: Option<Arc<ServingCharge>>,
}
impl RequestEvents {
    pub fn new(
        id: u64,
        model: String,
        events: mpsc::Receiver<ResponseEvent>,
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
    pub fn observed(mut self, metrics: Option<TokenMetricsRequest>) -> Self {
        self.token_metrics = metrics;
        self
    }
}
impl Drop for RequestEvents {
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
    /// CPU policy: bounded output refusal terminates this presenter.
    #[cfg(test)]
    pub(crate) fn accept(&mut self, id: u32) -> Result<bool, ServiceError> {
        self.accept_committed(id, Instant::now())
    }
    pub fn accept_committed(&mut self, id: u32, committed: Instant) -> Result<bool, ServiceError> {
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
    pub request: Request,
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
}
#[derive(Default)]
struct CpuStats {
    active: AtomicUsize,
    peak: AtomicUsize,
}
struct ActiveCharge(Arc<CpuStats>);
impl Drop for ActiveCharge {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}
pub struct Router {
    models: Arc<BTreeMap<String, Arc<dyn PreparedModel>>>,
    host: HostPool,
    queue: PendingQueue<Scheduled>,
    stats: Arc<CpuStats>,
    active_limit: usize,
    workers: Vec<thread::JoinHandle<()>>,
    // Non-payload worker inventory (handles, consumer state, captures, names).
    // Declared after `workers` so the joins/handles pay out first.
    _worker_charge: ServingCharge,
    next_id: AtomicU64,
    pub max_context: usize,
    pub max_output: usize,
}
impl Router {
    /// Model-free transport fixture: the original idle CPU worker/queue/host,
    /// with no registered model or inference capability.
    #[cfg(test)]
    pub(crate) fn transport_test() -> (Self, HostPool) {
        let host = HostPool::isolated(256).unwrap();
        let models = Arc::new(BTreeMap::new());
        let stats = Arc::new(CpuStats::default());
        let queue = PendingQueue::new(host.clone(), Some(8)).unwrap();
        let charge = workers_charge(
            &host,
            1,
            std::alloc::Layout::new::<CpuWorkerCaptures>(),
            Some(worker_name_bound(1)),
            vecdeque_backing_bytes::<Active>(1).unwrap(),
        )
        .unwrap();
        let captures = CpuWorkerCaptures {
            queue: queue.worker().unwrap(),
            models: Arc::clone(&models),
            host: host.clone(),
            stats: Arc::clone(&stats),
            active_limit: 1,
            max_context: 16,
        };
        let worker = thread::Builder::new()
            .name("zg-decode-0".into())
            .spawn(move || captures.run())
            .unwrap();
        (
            Self {
                models,
                host: host.clone(),
                queue,
                stats,
                active_limit: 1,
                workers: vec![worker],
                _worker_charge: charge,
                next_id: AtomicU64::new(1),
                max_context: 16,
                max_output: 8,
            },
            host,
        )
    }

    /// Validate and take the prepared alias map without rebuilding its backing.
    pub fn new(
        models: BTreeMap<String, Arc<dyn PreparedModel>>,
        workers: usize,
        active_per_worker: usize,
        max_context: usize,
        max_output: usize,
        host: HostPool,
        max_pending: Option<usize>,
    ) -> Result<Self, ServiceError> {
        validate_config(
            models.len(),
            workers,
            active_per_worker,
            max_context,
            max_output,
            max_pending,
        )?;
        for (id, model) in &models {
            validate_alias(id)?;
            if !model.source().is_bound() || max_context > model.max_context() {
                return Err(ServiceError::new(
                    400,
                    "invalid_model",
                    "unbound source or context exceeds prepared model admission",
                ));
            }
        }
        let active_limit = workers.checked_mul(active_per_worker).ok_or_else(|| {
            ServiceError::new(400, "invalid_config", "active worker capacity overflow")
        })?;
        let models = Arc::new(models);
        let stats = Arc::new(CpuStats::default());
        let queue = PendingQueue::new(host.clone(), max_pending)?;
        // Charge the whole worker inventory before the handle vector allocates
        // or any thread is created; a partial spawn returns the charge with the
        // router, so no unowned stack stays admitted.
        let active_backing = vecdeque_backing_bytes::<Active>(active_per_worker)?;
        let worker_charge = workers_charge(
            &host,
            workers,
            std::alloc::Layout::new::<CpuWorkerCaptures>(),
            Some(worker_name_bound(workers)),
            active_backing,
        )?;
        let mut handles: Vec<thread::JoinHandle<()>> = Vec::with_capacity(workers);
        let spawn_worker = |index: usize, receiver: WorkerQueue<Scheduled>| {
            let captures = CpuWorkerCaptures {
                queue: receiver,
                models: Arc::clone(&models),
                host: host.clone(),
                stats: Arc::clone(&stats),
                active_limit: active_per_worker,
                max_context,
            };
            let builder = thread::Builder::new().name(format!("zg-decode-{index}"));
            builder.spawn(move || captures.run())
        };
        for index in 0..workers {
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
            let spawned = spawn_worker(index, receiver);
            match spawned {
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
            max_context,
            max_output,
        })
    }
    pub fn model(&self, alias: &str) -> Option<ModelInstance> {
        self.models.get(alias).cloned().map(ModelInstance::new)
    }
    pub fn host(&self) -> &HostPool {
        &self.host
    }
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
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
        self.request_with(request, None)
    }
    /// HTTP handoff: retain the already-charged header/body/parse ingress
    /// through the canonical request strings instead of releasing and
    /// re-admitting them.
    pub fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<ServingCharge>,
    ) -> Result<RequestEvents, ServiceError> {
        self.request_with(request, Some(ingress))
    }
    fn request_with(
        &self,
        mut request: Request,
        ingress: Option<Vec<ServingCharge>>,
    ) -> Result<RequestEvents, ServiceError> {
        let model =
            self.models.get(&request.model).cloned().ok_or_else(|| {
                ServiceError::new(404, "model_not_found", "model is not registered")
            })?;
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
        let id = self
            .next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| ServiceError::new(503, "id_exhausted", "request IDs exhausted"))?;
        // Count/admission reservation precedes every request-derived
        // allocation and queue publication.
        let ticket = self.queue.reserve()?;
        let request_bytes = pending::checked_sum(&[
            pending::heap_strings(&[&request.model, &request.input])?,
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
        let max_tokens = request.max_tokens.min(self.max_context);
        let prompt_bound = model
            .tokenizer()
            .prompt_token_bound(request.input.len(), request.chat);
        let prompt_bytes = pending::checked_sum(&[
            pending::vec_bytes::<u32>(prompt_bound)?,
            model
                .tokenizer()
                .admitted_prompt_scratch_bound(request.input.len(), request.chat)
                .map_err(|error| ServiceError::new(400, "size_overflow", error.to_string()))?,
            // The presenter decodes the whole committed prefix, so the adapter
            // scratch of a full `max_tokens` prefix stays live at each commit.
            u64::try_from(model.tokenizer().decode_scratch_bound(max_tokens)).map_err(|_| {
                ServiceError::new(400, "size_overflow", "decode scratch bound overflow")
            })?,
        ])?;
        let prompt_charge = self.host.charge(
            &request.model,
            "prompt-tokens",
            AllocationClass::RecurrentState,
            prompt_bytes,
        )?;
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
                request: Request {
                    max_tokens,
                    ..request
                },
                _charges: charges,
            }),
            metrics,
            events: events_tx,
            cancelled,
            cancel: cancel.clone(),
            prompt_charge: Some(prompt_charge),
            transport: Some(Arc::clone(&transport)),
        };
        if let Err((error, _)) = self.queue.enqueue(ticket, scheduled) {
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
        self.close_admission();
        let mut failed = false;
        for handle in self.workers.drain(..) {
            failed |= handle.join().is_err();
        }
        if failed {
            Err(std::io::Error::other("CPU serving worker panicked"))
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
struct Active {
    session: Box<dyn TextSession>,
    prompt: Vec<u32>,
    logits: Vec<f32>,
    output: GeneratedText,
    events: mpsc::SyncSender<ResponseEvent>,
    cancelled: Arc<AtomicBool>,
    prompt_pos: usize,
    logits_commit_observed_at: Instant,
    _state_charge: GlobalResourceLease,
    _projection_charge: ServingCharge,
    _prompt_charge: Option<ServingCharge>,
    _transport: Option<Arc<ServingCharge>>,
    _active: ActiveCharge,
    // Keeps the cancel atomic's charged control alive with the flag.
    _cancel: CancelToken,
}
impl Active {
    fn new(
        scheduled: Scheduled,
        model: Arc<dyn PreparedModel>,
        max_context: usize,
        host: &HostPool,
        stats: Arc<CpuStats>,
    ) -> Result<Self, ServiceError> {
        let Scheduled {
            payload,
            metrics,
            events,
            cancelled,
            cancel,
            prompt_charge,
            transport,
        } = scheduled;
        let lease = payload.ok_or_else(|| {
            ServiceError::new(
                500,
                "scheduled_payload",
                "scheduled request lost its payload",
            )
        })?;
        let request = &lease.request;
        let prompt = model
            .tokenizer()
            .prompt_tokens(&request.input, request.chat)
            .map_err(ServiceError::inference)?;
        if prompt.is_empty()
            || prompt.len() > max_context
            || prompt
                .len()
                .checked_add(request.max_tokens - 1)
                .is_none_or(|n| n > max_context)
        {
            return Err(ServiceError::new(
                400,
                "context_overflow",
                "request exceeds bounded model context",
            ));
        }
        let state_charge = host.admit(
            &request.model,
            "cpu-request-state",
            AllocationClass::RecurrentState,
            model.max_session_bytes(),
        )?;
        // A live projection's splitter can grow its two injector blocks.
        let projection_charge = host.charge(
            &request.model,
            "rayon-projection",
            AllocationClass::Scratch,
            model.session_scratch_bytes(),
        )?;
        let session = model
            .clone()
            .new_session()
            .map_err(ServiceError::inference)?;
        let output = GeneratedText::new(
            model.tokenizer(),
            events.clone(),
            request.stream,
            request.max_tokens,
            prompt.len(),
        )
        .observed_charged(metrics.metrics, model.vocabulary_size(), metrics._charges);
        drop(lease);
        let active = stats.active.fetch_add(1, Ordering::AcqRel) + 1;
        stats.peak.fetch_max(active, Ordering::AcqRel);
        Ok(Self {
            session,
            _state_charge: state_charge,
            _projection_charge: projection_charge,
            _prompt_charge: prompt_charge,
            _transport: transport,
            _active: ActiveCharge(Arc::clone(&stats)),
            _cancel: cancel,
            events,
            cancelled,
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
struct CpuWorkerCaptures {
    queue: WorkerQueue<Scheduled>,
    models: Arc<BTreeMap<String, Arc<dyn PreparedModel>>>,
    host: HostPool,
    stats: Arc<CpuStats>,
    active_limit: usize,
    max_context: usize,
}

impl CpuWorkerCaptures {
    /// Consume the exact admitted struct, so the start closure captures only
    /// this priced value: `Layout::new::<CpuWorkerCaptures>()` at admission is
    /// provably the captured width and cannot drift from a field mirror.
    fn run(self) {
        worker(
            self.queue,
            self.models,
            self.host,
            self.stats,
            self.active_limit,
            self.max_context,
        )
    }
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
        fixed_size::<WorkerQueue<Scheduled>>()?,
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

fn worker(
    queue: WorkerQueue<Scheduled>,
    models: Arc<BTreeMap<String, Arc<dyn PreparedModel>>>,
    host: HostPool,
    stats: Arc<CpuStats>,
    active_limit: usize,
    max_context: usize,
) {
    // The container row is charged before spawn and never grows past the
    // configured per-worker active limit.
    let mut active = VecDeque::<Active>::with_capacity(active_limit);
    let mut closing = false;
    let cancelled = |job: &Scheduled| job.cancel.is_cancelled();
    loop {
        // Reclaim cancelled queued payloads even when every active or parked
        // slot prevents promotion. The hint keeps idle rounds cheap.
        let _ = queue.sweep(cancelled);
        if !closing {
            while active.len() < active_limit {
                let next = if active.is_empty() {
                    match queue.pop_wait(Duration::from_millis(30), cancelled) {
                        Pop::Item(job) => Some(job),
                        Pop::Empty => break,
                        Pop::Closed => {
                            closing = true;
                            break;
                        }
                    }
                } else {
                    match queue.try_pop(cancelled) {
                        Pop::Item(job) => Some(job),
                        Pop::Empty => None,
                        Pop::Closed => {
                            closing = true;
                            break;
                        }
                    }
                };
                let Some(job) = next else {
                    break;
                };
                if job.cancelled.load(Ordering::Acquire) {
                    continue;
                }
                let model = models
                    .get(
                        &job.payload
                            .as_ref()
                            .expect("scheduled payload")
                            .request
                            .model,
                    )
                    .expect("router validated model");
                let errors = job.events.clone();
                match Active::new(
                    job,
                    Arc::clone(model),
                    max_context,
                    &host,
                    Arc::clone(&stats),
                ) {
                    Ok(inference) => active.push_back(inference),
                    Err(err) => {
                        let _ = errors.try_send(ResponseEvent::Error(err));
                    }
                }
            }
        }
        if active.is_empty() {
            if closing {
                return;
            }
            continue;
        }
        let Some(mut current) = active.pop_front() else {
            continue;
        };
        if current.cancelled.load(Ordering::Acquire) {
            continue;
        }
        // Delivery detects disconnect and fails closed; no output is replayed.
        match current.step() {
            Ok(true) => {}
            Ok(false) => active.push_back(current),
            Err(error) => {
                let _ = current.events.try_send(ResponseEvent::Error(error));
            }
        }
    }
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
        || !(1..=8192).contains(&max_context)
        || !(1..=512).contains(&max_output)
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
            std::alloc::Layout::new::<CpuWorkerCaptures>(),
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
