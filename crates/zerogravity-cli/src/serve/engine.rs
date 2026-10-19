//! Shared immutable Qwen source, bounded multi-request CPU scheduler.
//! Concurrent worker threads + cooperative prefill/decode within workers.
//! This is NOT yet fused continuous-batch GEMM, paged KV, or GPU serving.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;
use zerogravity::prepare::PreparedQwen;
use zerogravity::qwen35::{LoadedQwenModel, QwenDenseCpu, StateSpec};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity::tokenize::QwenTokenizer;

pub(super) const QUEUED_PER_WORKER: usize = 16;
pub(super) const EVENTS_PER_REQUEST: usize = 16;
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
}
pub enum ResponseEvent {
    Text(String),
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
            cancelled,
        }
    }
}
impl Drop for RequestEvents {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
/// Shared CPU/GPU presentation and backpressure semantics.
pub(super) struct GeneratedText {
    tokenizer: Arc<QwenTokenizer>,
    events: mpsc::SyncSender<ResponseEvent>,
    max_tokens: usize,
    prompt_tokens: usize,
    stream: bool,
    generated: Vec<u32>,
    emitted: String,
}
impl GeneratedText {
    pub(super) fn new(
        tokenizer: Arc<QwenTokenizer>,
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
        }
    }
    /// Returns true once one admitted request has terminated; never
    /// samples again after the final accepted token or an EOS.
    pub(super) fn accept(&mut self, id: u32) -> Result<bool, ServiceError> {
        if id == self.tokenizer.eos() {
            self.finish("stop")?;
            return Ok(true);
        }
        self.generated.push(id);
        if self.stream {
            let decoded = self
                .tokenizer
                .decode(&self.generated, true)
                .map_err(ServiceError::inference)?;
            if !decoded.starts_with(&self.emitted) {
                return Err(ServiceError::new(
                    500,
                    "decoder_nonmonotonic",
                    "incremental text prefix changed",
                ));
            }
            if !decoded.contains('�') {
                let delta = &decoded[self.emitted.len()..];
                if !delta.is_empty() {
                    self.events
                        .try_send(ResponseEvent::Text(delta.to_owned()))
                        .map_err(|e| match e {
                            mpsc::TrySendError::Full(_) => ServiceError::new(
                                503,
                                "backpressure",
                                "client cannot accept streamed output",
                            ),
                            mpsc::TrySendError::Disconnected(_) => {
                                ServiceError::new(499, "cancelled", "client disconnected")
                            }
                        })?;
                    self.emitted = decoded;
                }
            }
        }
        if self.generated.len() >= self.max_tokens {
            self.finish("length")?;
            return Ok(true);
        }
        Ok(false)
    }
    fn finish(&mut self, reason: &'static str) -> Result<(), ServiceError> {
        let output = self
            .tokenizer
            .decode(&self.generated, true)
            .map_err(ServiceError::inference)?;
        if self.stream && !output.starts_with(&self.emitted) {
            return Err(ServiceError::new(
                500,
                "decoder_nonmonotonic",
                "final text differs from streamed prefix",
            ));
        }
        if self.stream && output.len() > self.emitted.len() {
            self.events
                .try_send(ResponseEvent::Text(output[self.emitted.len()..].to_owned()))
                .map_err(|_| ServiceError::new(499, "cancelled", "client closed output stream"))?;
        }
        self.events
            .try_send(ResponseEvent::Done {
                output,
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.generated.len(),
                finish_reason: reason,
            })
            .map_err(|_| ServiceError::new(499, "cancelled", "client closed output stream"))
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
        Ok(RequestEvents::new(id, model_alias, events_rx, cancelled))
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
        if self.output.accept(id)? {
            return Ok(true);
        }
        self.logits = self.session.forward(id).map_err(ServiceError::inference)?;
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

    #[test]
    fn request_receiver_cancellation_drops_queued_session_without_inference() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let (_sender, receiver) = mpsc::sync_channel::<ResponseEvent>(1);
        let receipt = RequestEvents {
            id: 7,
            model: "source-owned".into(),
            events: receiver,
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
}
