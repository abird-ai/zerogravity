//! Shared immutable Qwen source, bounded multi-request CPU scheduler.
//! Concurrent worker threads + cooperative prefill/decode within workers.
//! This is NOT yet fused continuous-batch GEMM, paged KV, or GPU serving.

use super::event_channel::mpsc;
use super::pending::{
    self, CancelToken, PendingQueue, Pop, QueueSnapshot, ServingCharge, ServingHost as HostPool,
    WorkerQueue,
};
use std::collections::{BTreeMap, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use zerogravity::qwen35::{LoadedQwenModel, QwenDenseCpu, StateSpec};
use zerogravity::resource_arbiter::{AllocationClass, GlobalResourceLease};
use zerogravity::tokenize::QwenTokenizer;

pub(super) const EVENTS_PER_REQUEST: usize = 16;
pub(super) const MAX_TOKEN_OBSERVATIONS: usize = 512;
const PREFILL_CHUNK: usize = 32;
pub(super) const MAX_ACTIVE_PER_WORKER: usize = 8;

/// Conservative allocator header/rounding for a `Vec`/`VecDeque` backing.
const CONTAINER_ALLOC_OVERHEAD_BYTES: u64 = 64;

#[derive(PartialEq, Eq)]
struct FileStamp {
    len: u64,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    mtime: i64,
    #[cfg(unix)]
    mtime_nsec: i64,
    #[cfg(unix)]
    ctime: i64,
    #[cfg(unix)]
    ctime_nsec: i64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

fn file_stamp(meta: &std::fs::Metadata) -> Result<FileStamp, ServiceError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(FileStamp {
            len: meta.len(),
            dev: meta.dev(),
            ino: meta.ino(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
            ctime: meta.ctime(),
            ctime_nsec: meta.ctime_nsec(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(FileStamp {
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }
}

/// Read the exact approved tokenizer extent from one held, no-follow
/// descriptor. The allocation is capped at the pinned length, the descriptor
/// must be at EOF immediately after it, and the pre/post descriptor stamp must
/// agree. SHA-256 is checked by `QwenTokenizer::from_approved_bytes` on these
/// same bytes before the parser runs.
pub(super) fn read_approved_tokenizer(path: &Path) -> Result<Vec<u8>, ServiceError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NOFOLLOW: do not authenticate a symlink target while claiming the
        // held path identity.
        options.custom_flags(0o400000);
    }
    let mut file = options
        .open(path)
        .map_err(|_| ServiceError::new(400, "missing_tokenizer", "tokenizer file unavailable"))?;
    let before = file.metadata().map_err(|_| {
        ServiceError::new(400, "missing_tokenizer", "tokenizer metadata unavailable")
    })?;
    if !before.is_file() {
        return Err(ServiceError::new(
            400,
            "missing_tokenizer",
            "tokenizer path is not a regular file",
        ));
    }
    let extent = before.len();
    if extent != zerogravity::tokenize::APPROVED_TOKENIZER_BYTES {
        return Err(ServiceError::new(
            400,
            "tokenizer_inventory",
            "approved tokenizer extent mismatch",
        ));
    }
    let capacity = usize::try_from(extent)
        .map_err(|_| ServiceError::new(400, "tokenizer_inventory", "tokenizer extent overflow"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        ServiceError::new(
            503,
            "tokenizer_inventory",
            "approved tokenizer allocation unavailable",
        )
    })?;
    bytes.resize(capacity, 0);
    file.read_exact(&mut bytes).map_err(|_| {
        ServiceError::new(
            400,
            "tokenizer_inventory",
            "approved tokenizer extent changed during read",
        )
    })?;
    let mut probe = [0u8; 1];
    if file
        .read(&mut probe)
        .map_err(|_| ServiceError::new(400, "tokenizer_inventory", "tokenizer EOF probe failed"))?
        != 0
    {
        return Err(ServiceError::new(
            400,
            "tokenizer_inventory",
            "approved tokenizer extends past its extent",
        ));
    }
    let after = file.metadata().map_err(|_| {
        ServiceError::new(
            400,
            "tokenizer_inventory",
            "tokenizer metadata changed during read",
        )
    })?;
    if file_stamp(&before)? != file_stamp(&after)? || after.len() != extent {
        return Err(ServiceError::new(
            400,
            "tokenizer_inventory",
            "approved tokenizer stamp changed during read",
        ));
    }
    Ok(bytes)
}

pub(super) fn fixed_size<T>() -> Result<u64, ServiceError> {
    u64::try_from(std::mem::size_of::<T>())
        .map_err(|_| ServiceError::new(400, "size_overflow", "container size overflow"))
}

pub(super) fn vecdeque_backing_bytes<T>(capacity: usize) -> Result<u64, ServiceError> {
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
impl From<zerogravity::Error> for ServiceError {
    fn from(error: zerogravity::Error) -> Self {
        Self::inference(error)
    }
}

pub struct Model {
    weights: Arc<LoadedQwenModel>,
    tokenizer: Arc<QwenTokenizer>,
    max_session_bytes: u64,
    max_token_bytes: usize,
    _weight_charge: GlobalResourceLease,
    // Retained `R + G + V` source credit for this actual tokenizer instance:
    // tokenizer-lifetime inventory plus prepaid cache-growth/VM scratch,
    // released only with the source; aliases of this model share one owner.
    _inventory_charge: zerogravity::tokenize_admission::Retained,
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
    _cancel: CancelToken,
    _request: Option<ServingCharge>,
    _transport: Option<Arc<ServingCharge>>,
}
impl RequestEvents {
    pub(super) fn new(
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
    pub(super) fn observed(mut self, metrics: Option<TokenMetricsRequest>) -> Self {
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
    // Bundled with `token_metrics` so the wire-ID string and the ingress
    // charge that covers it are released in one owner.
    _metrics_charge: Option<Arc<Vec<ServingCharge>>>,
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
            _metrics_charge: None,
        }
    }
    #[cfg(test)]
    pub(super) fn observed(
        mut self,
        metrics: Option<TokenMetricsRequest>,
        vocabulary: u64,
    ) -> Self {
        self.token_metrics = metrics;
        self.vocabulary_size = vocabulary;
        self
    }
    pub(super) fn observed_charged(
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

/// Queued request strings and the ingress charges that cover them. The
/// charges are released only after the strings are gone.
pub(super) struct RequestLease {
    pub(super) request: Request,
    pub(super) _charges: Arc<Vec<ServingCharge>>,
}
/// The metrics wire ID and the original charge covering it travel together
/// into the presenter, so no cloned identifier becomes uncharged.
pub(super) struct MetricsLease {
    pub(super) metrics: Option<TokenMetricsRequest>,
    pub(super) _charges: Option<Arc<Vec<ServingCharge>>>,
}
pub(super) struct Scheduled {
    pub(super) payload: Option<RequestLease>,
    pub(super) metrics: MetricsLease,
    pub(super) events: mpsc::SyncSender<ResponseEvent>,
    pub(super) cancelled: Arc<AtomicBool>,
    pub(super) cancel: CancelToken,
    pub(super) prompt_charge: Option<ServingCharge>,
    pub(super) transport: Option<Arc<ServingCharge>>,
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
    models: Arc<BTreeMap<String, Arc<Model>>>,
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
    pub fn new(
        declared: Vec<(String, PathBuf)>,
        workers: usize,
        active_per_worker: usize,
        max_context: usize,
        max_output: usize,
        host_mib: usize,
        max_pending: Option<usize>,
    ) -> Result<Self, ServiceError> {
        if declared.is_empty()
            || declared.len() > 16
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
            let tokenizer_bytes = zerogravity::tokenize::APPROVED_TOKENIZER_BYTES;
            let charged = input_len
                .checked_add(tokenizer_bytes.saturating_mul(2))
                .and_then(|n| n.checked_add(128 * 1024 * 1024))
                .ok_or_else(|| {
                    ServiceError::new(400, "model_budget", "model source size overflow")
                })?;
            // Preflight/admission BEFORE any tokenizer parse, compiled regex,
            // NFC/ByteLevel bootstrap or weight mapping. The approved retained
            // plan needs only the pinned pool capacity and layout constants,
            // so the process-lifetime ByteLevel owner, retained inventory and
            // cold constructor are charged first; the parsed source must then
            // agree with the plan it was admitted under. The immutable weight
            // source is admitted immediately before `LoadedQwenModel::open`,
            // after the constructor scratch/copy are released and verified.
            let tokenizer_admission = host.transfer_admission(&id, "tokenizer-source")?;
            let bootstrap = zerogravity::tokenize_admission::approve_arbiter(
                &tokenizer_admission,
                "bootstrap",
            )?;
            let inventory_charge = bootstrap
                .retain_arbiter(&tokenizer_admission, "retained-inventory")
                .map_err(|error| {
                    let message = error.to_string();
                    match error {
                        zerogravity::Error::Limit(_) => ServiceError::new(
                            503,
                            "host_budget",
                            format!(
                                "stage=retained-inventory owner=model={id} session=retained-inventory: {message}"
                            ),
                        ),
                        other => ServiceError::inference(other),
                    }
                })?;
            // The exact held source copy is priced before allocation; the
            // constructor census and parsed tokenizer are priced from those
            // same bytes. SHA-256 is checked before the parser.
            let tokenizer_copy = host.charge(
                &id,
                "tokenizer-source-copy",
                AllocationClass::Scratch,
                tokenizer_bytes,
            )?;
            let tokenizer_bytes = read_approved_tokenizer(&parent.join("tokenizer.json"))?;
            let tokenizer = {
                let _constructor = bootstrap.constructor_of_arbiter(
                    &tokenizer_admission,
                    "tokenizer-constructor",
                    &tokenizer_bytes,
                )?;
                QwenTokenizer::from_approved_bytes(&tokenizer_bytes)
                    .map_err(ServiceError::inference)?
            };
            // The held copy charge covers these exact bytes: release the
            // payload first, then its precharged source-copy owner.
            drop(tokenizer_bytes);
            drop(tokenizer_copy);
            bootstrap.verify(&tokenizer)?;
            // The unchanged immutable weight source is admitted last, directly
            // before the original weight open; no weight bytes are mapped while
            // only metadata/path work or tokenizer construction is in flight.
            let weight_charge = host
                .admit(
                    &id,
                    "immutable-source",
                    AllocationClass::ImmutableWeights,
                    charged,
                )
                .map_err(|error| {
                    ServiceError::new(
                        error.status,
                        error.code,
                        format!(
                            "stage=immutable-source owner=model={id} session=immutable-source: {}",
                            error.message
                        ),
                    )
                })?;
            let weights = LoadedQwenModel::open(&path, &parent.join("config.json"))
                .map_err(ServiceError::inference)?;
            let max_session_bytes = session_budget(&weights, max_context)?;
            let max_token_bytes = tokenizer
                .max_token_bytes()
                .map_err(ServiceError::inference)?;
            let shared = Arc::new(Model {
                weights,
                tokenizer: Arc::new(tokenizer),
                max_session_bytes,
                max_token_bytes,
                _weight_charge: weight_charge,
                _inventory_charge: inventory_charge,
            });
            unique_sources.insert(path, Arc::clone(&shared));
            models.insert(id, shared);
        }
        let active_limit = workers.checked_mul(active_per_worker).ok_or_else(|| {
            ServiceError::new(400, "invalid_config", "active worker capacity overflow")
        })?;
        // One process-lifetime Rayon owner, priced from the actual configured
        // count, before any decode worker exists. A second router reuses the
        // published owner instead of spawning another global pool.
        let rayon_alias = models
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| ServiceError::new(400, "invalid_model", "no model routes"))?;
        let rayon_admission = host.transfer_admission(&rayon_alias, "global-rayon")?;
        zerogravity::admit_global_rayon(&rayon_admission)?;
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
    pub(super) fn request_admitted(
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
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if id == u64::MAX {
            return Err(ServiceError::new(
                503,
                "id_exhausted",
                "request IDs exhausted",
            ));
        }
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
            .tokenizer
            .prompt_token_bound(request.input.len(), request.chat);
        let prompt_bytes = pending::checked_sum(&[
            pending::vec_bytes::<u32>(prompt_bound)?,
            model
                .tokenizer
                .admitted_prompt_scratch_bound(request.input.len(), request.chat)
                .map_err(|error| ServiceError::new(400, "size_overflow", error.to_string()))?,
            // The presenter decodes the whole committed prefix, so the adapter
            // scratch of a full `max_tokens` prefix stays live at each commit.
            u64::try_from(model.tokenizer.decode_scratch_bound(max_tokens)).map_err(|_| {
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
        let output_bound = output_transport_bound(max_tokens, model.max_token_bytes)?;
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
impl Drop for Router {
    fn drop(&mut self) {
        self.queue.close();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}
pub(super) fn output_transport_bound(
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
    session: QwenDenseCpu,
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
        model: Arc<Model>,
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
            .tokenizer
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
            model.max_session_bytes,
        )?;
        // A live projection's splitter can grow its two injector blocks.
        let projection_charge = host.charge(
            &request.model,
            "rayon-projection",
            AllocationClass::Scratch,
            zerogravity::RAYON_PROJECTION_INJECTOR_BYTES,
        )?;
        let session = model
            .weights
            .new_session()
            .map_err(ServiceError::inference)?;
        let output = GeneratedText::new(
            Arc::clone(&model.tokenizer),
            events.clone(),
            request.stream,
            request.max_tokens,
            prompt.len(),
        )
        .observed_charged(
            metrics.metrics,
            model.weights.geometry().vocab_size,
            metrics._charges,
        );
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
    models: Arc<BTreeMap<String, Arc<Model>>>,
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
pub(super) fn workers_charge(
    host: &HostPool,
    workers: usize,
    capture: std::alloc::Layout,
    name: Option<(usize, usize)>,
    extra_per_worker: u64,
) -> Result<ServingCharge, ServiceError> {
    let count = u64::try_from(workers)
        .map_err(|_| ServiceError::new(400, "size_overflow", "worker charge overflow"))?;
    let control = zerogravity::resource_arbiter::rust_thread_control_bytes(capture, name)
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
    models: Arc<BTreeMap<String, Arc<Model>>>,
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
    fn approved_tokenizer_reader_rejects_wrong_extent_and_symlink() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "zerogravity-tokenizer-reader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("tokenizer.json");
        std::fs::write(&source, b"{}").unwrap();
        assert_eq!(
            read_approved_tokenizer(&source).unwrap_err().code,
            "tokenizer_inventory"
        );
        #[cfg(target_os = "linux")]
        {
            let link = root.join("linked-tokenizer.json");
            std::os::unix::fs::symlink(&source, &link).unwrap();
            assert_eq!(
                read_approved_tokenizer(&link).unwrap_err().code,
                "missing_tokenizer"
            );
        }
        std::fs::remove_dir_all(root).unwrap();
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

    #[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
    mod serving_population {
        use super::*;
        use std::collections::BTreeSet;
        use std::io::{Seek, SeekFrom, Write};

        const VOCAB: u64 = 248_320;
        const HIDDEN: usize = 1024;
        const LAYERS: usize = 24;
        const WORKERS: usize = 2;
        const ACTIVE_PER_WORKER: usize = 4;
        const ACTIVE_LIMIT: usize = WORKERS * ACTIVE_PER_WORKER;
        const HOST_MIB: usize = 4096;
        const MAX_CONTEXT: usize = 8;
        const MAX_OUTPUT: usize = 1;

        fn string(out: &mut Vec<u8>, text: &str) {
            out.extend_from_slice(&(text.len() as u64).to_le_bytes());
            out.extend_from_slice(text.as_bytes());
        }

        /// Genuine legal qwen35 GGUF with the FULL approved vocabulary and
        /// 24 full-attention layers. Token row 0 is the only non-zero
        /// embedding row, so greedy selection is the real id 0 ("!"). All
        /// zero ranges are real F32 weights written as sparse file holes; the
        /// fixture never builds a duplicate 1 GiB payload in memory.
        fn tensors() -> Vec<(String, Vec<u64>, f32)> {
            let mut tensors = Vec::new();
            tensors.push(("token_embd.weight".into(), vec![HIDDEN as u64, VOCAB], 0.0));
            tensors.push(("output_norm.weight".into(), vec![HIDDEN as u64], 1.0));
            for i in 0..LAYERS {
                for (name, shape, value) in [
                    ("attn_norm.weight", vec![HIDDEN as u64], 1.0),
                    ("post_attention_norm.weight", vec![HIDDEN as u64], 1.0),
                    ("attn_q_norm.weight", vec![64], 1.0),
                    ("attn_k_norm.weight", vec![64], 1.0),
                    ("attn_q.weight", vec![HIDDEN as u64, 128], 0.0),
                    ("attn_k.weight", vec![HIDDEN as u64, 64], 0.0),
                    ("attn_v.weight", vec![HIDDEN as u64, 64], 0.0),
                    ("attn_output.weight", vec![64, HIDDEN as u64], 0.0),
                    ("ffn_gate.weight", vec![HIDDEN as u64, 1], 0.0),
                    ("ffn_up.weight", vec![HIDDEN as u64, 1], 0.0),
                    ("ffn_down.weight", vec![1, HIDDEN as u64], 0.0),
                ] {
                    tensors.push((format!("blk.{i}.{name}"), shape, value));
                }
            }
            tensors
        }

        fn gguf(path: &Path) {
            let tensors = tensors();
            let mut infos = Vec::with_capacity(tensors.len());
            let mut payload_len = 0u64;
            for (name, shape, _) in &tensors {
                payload_len = (payload_len + 31) & !31;
                let offset = payload_len;
                let elements = shape.iter().product::<u64>();
                payload_len = payload_len
                    .checked_add(elements * 4)
                    .expect("fixture payload length");
                infos.push((name.clone(), shape.clone(), offset));
            }
            let mut header = b"GGUF".to_vec();
            header.extend_from_slice(&3u32.to_le_bytes());
            header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
            header.extend_from_slice(&1u64.to_le_bytes());
            string(&mut header, "general.architecture");
            header.extend_from_slice(&8u32.to_le_bytes());
            string(&mut header, "qwen35");
            for (name, shape, offset) in &infos {
                string(&mut header, name);
                header.extend_from_slice(&(shape.len() as u32).to_le_bytes());
                for dim in shape {
                    header.extend_from_slice(&dim.to_le_bytes());
                }
                header.extend_from_slice(&0u32.to_le_bytes());
                header.extend_from_slice(&offset.to_le_bytes());
            }
            while !header.len().is_multiple_of(32) {
                header.push(0);
            }
            let payload_start = header.len() as u64;
            let total = payload_start
                .checked_add(payload_len)
                .expect("fixture total length");
            let mut file = std::fs::File::create(path).expect("fixture model");
            file.set_len(total).expect("sparse fixture extent");
            file.seek(SeekFrom::Start(0)).expect("fixture header seek");
            file.write_all(&header).expect("fixture header");
            for ((name, shape, value), (_, _, offset)) in tensors.iter().zip(&infos) {
                let start = payload_start + *offset;
                let elements = shape.iter().product::<u64>() as usize;
                let mut bytes = Vec::new();
                if name.as_str() == "token_embd.weight" {
                    for _ in 0..HIDDEN {
                        bytes.extend_from_slice(&1.0f32.to_le_bytes());
                    }
                } else if *value != 0.0 {
                    bytes.reserve(elements * 4);
                    for _ in 0..elements {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                if !bytes.is_empty() {
                    file.seek(SeekFrom::Start(start))
                        .expect("fixture data seek");
                    file.write_all(&bytes).expect("fixture data");
                }
            }
            file.sync_all().expect("fixture sync");
        }

        fn fixture(tag: &str) -> PathBuf {
            let approved = PathBuf::from(
                std::env::var_os("ZG_APPROVED_TOKENIZER")
                    .expect("set ZG_APPROVED_TOKENIZER to the pinned approved tokenizer.json"),
            );
            let metadata = std::fs::metadata(&approved).expect("approved tokenizer metadata");
            assert_eq!(
                metadata.len(),
                zerogravity::tokenize::APPROVED_TOKENIZER_BYTES,
                "approved tokenizer extent"
            );
            let root = std::env::temp_dir().join(format!("zg-serve-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            // Approved source copied exactly: a regular file with the pinned
            // length/SHA that `read_approved_tokenizer` verifies.
            std::fs::copy(&approved, root.join("tokenizer.json")).unwrap();
            let config = serde_json::json!({
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": HIDDEN as u64,
                    "head_dim": 64,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "vocab_size": VOCAB,
                    "max_position_embeddings": 32768,
                    "num_hidden_layers": LAYERS,
                    "layer_types": vec!["full_attention"; LAYERS],
                    "full_attention_interval": 1,
                    "linear_conv_kernel_dim": 4,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 1
                }
            });
            std::fs::write(
                root.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            gguf(&root.join("model.gguf"));
            root
        }

        fn request() -> Request {
            Request {
                model: "tiny".into(),
                input: "!".into(),
                chat: false,
                max_tokens: MAX_OUTPUT,
                stream: false,
                token_metrics: None,
            }
        }

        /// Drain one real receipt and return its id and actual token counts.
        /// The one terminal event must carry the real decoded id 0 output.
        fn drain(receipt: &RequestEvents) -> (u64, usize, usize, String) {
            let id = receipt.id;
            let mut terminal = None;
            let mut texts = 0;
            while let Ok(event) = receipt.events.recv() {
                match event {
                    ResponseEvent::Text(_) => texts += 1,
                    ResponseEvent::Done {
                        output,
                        prompt_tokens,
                        completion_tokens,
                        finish_reason,
                    } => {
                        assert!(terminal.is_none(), "duplicate terminal event");
                        assert!(prompt_tokens >= 1, "actual prompt token count");
                        assert_eq!(completion_tokens, 1, "actual completion token count");
                        assert_eq!(finish_reason, "length");
                        assert_eq!(output, "!", "decoded greedy id 0");
                        terminal = Some((prompt_tokens, completion_tokens, output));
                    }
                    ResponseEvent::TokenText { .. } => panic!("unexpected token metric event"),
                    ResponseEvent::Error(error) => {
                        panic!("request {id} failed: {} {}", error.status, error.code)
                    }
                }
            }
            assert_eq!(texts, 0, "non-streaming request emits only the terminal");
            let (prompt_tokens, completion_tokens, output) =
                terminal.expect("missing terminal event");
            (id, prompt_tokens, completion_tokens, output)
        }

        fn reexec_with(marker: &str, test: &str, done: &str, envs: &[(&str, &str)]) -> bool {
            if std::env::var_os(marker).is_some() {
                return false;
            }
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--ignored",
                    "--exact",
                    test,
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(marker, "1");
            for (key, value) in envs {
                command.env(key, value);
            }
            let output = command.output().unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(output.status.success(), "child failed: {stdout}\n{stderr}");
            assert!(stdout.contains(done), "child marker missing: {stdout}");
            true
        }

        fn reexec(marker: &str, test: &str, done: &str) -> bool {
            reexec_with(marker, test, done, &[])
        }

        #[test]
        #[ignore = "requires approved tokenizer metadata"]
        fn real_serving_populations_match_actual_router_state() {
            let test = "serve::engine::tests::serving_population::real_serving_populations_match_actual_router_state";
            if reexec("ZG_SERVE_POPULATION_CHILD", test, "POPULATION_CHILD_OK") {
                return;
            }
            let root = fixture("population");
            let model = root.join("model.gguf");
            // Exact current envelope: 4096 MiB configured host and 8 total
            // active slots (2 workers x 4). Every population is offered to
            // the original Router and must complete with real decoded output.
            // Both route names point at the same actual fixture path so the
            // router must share one `Model` owner and one tokenizer credit.
            let router = match Router::new(
                vec![
                    ("tiny".to_string(), model.clone()),
                    ("alias".to_string(), model.clone()),
                ],
                WORKERS,
                ACTIVE_PER_WORKER,
                MAX_CONTEXT,
                MAX_OUTPUT,
                HOST_MIB,
                None,
            ) {
                Ok(router) => router,
                Err(error) => panic!(
                    "4096 MiB / 8 active host refused: status={} code={} message={}",
                    error.status, error.code, error.message
                ),
            };
            assert_eq!(router.active_limit(), ACTIVE_LIMIT);
            assert_eq!(router.queue_snapshot().limit, None);
            let tiny = router.models.get("tiny").expect("actual tiny route");
            let alias = router.models.get("alias").expect("actual alias route");
            assert!(
                Arc::ptr_eq(tiny, alias),
                "same fixture path must share one Model/tokenizer source credit"
            );
            let host = router.host.clone();
            let baseline = host.serving_bytes();
            assert!(baseline > 0, "model/queue/worker serving charges are live");
            println!(
                "baseline serving_bytes={baseline} memory={:?} active_limit={}",
                router.memory_snapshot(),
                router.active_limit()
            );

            for population in [16usize, 32, 64] {
                // Straight offer loop: every real request is offered to the
                // original Router and its receipt is retained before any
                // receipt is drained, received or dropped.
                let mut receipts = Vec::with_capacity(population);
                let mut ids = BTreeSet::new();
                let mut tokens = 0usize;
                for offered in 0..population {
                    let receipt = match router.request(request()) {
                        Ok(receipt) => receipt,
                        Err(error) => panic!(
                            "population {population} offer={offered} refused: status={} code={} message={} queue={:?} memory={:?}",
                            error.status,
                            error.code,
                            error.message,
                            router.queue_snapshot(),
                            router.memory_snapshot()
                        ),
                    };
                    receipts.push(receipt);
                }
                assert_eq!(receipts.len(), population);
                let offered_queue = router.queue_snapshot();
                let offered_active = router.active_sessions();
                assert_eq!(offered_queue.limit, None);
                assert_eq!(offered_queue.consumers, WORKERS);
                assert!(
                    offered_queue.pending <= offered_queue.capacity,
                    "pending fits the ring"
                );
                assert!(offered_active <= router.active_limit());
                println!(
                    "population {population} offered={} active={offered_active} pending={} in_progress={} capacity={} backing={} consumers={}",
                    receipts.len(),
                    offered_queue.pending,
                    offered_queue.in_progress,
                    offered_queue.capacity,
                    offered_queue.backing_bytes,
                    offered_queue.consumers
                );

                let with_receipts = host.serving_bytes();
                assert!(with_receipts > baseline);
                for receipt in &receipts {
                    let (id, prompt, completion, output) = drain(receipt);
                    assert!(ids.insert(id), "duplicate request receipt {id}");
                    assert!(prompt >= 1);
                    assert_eq!(completion, 1);
                    assert_eq!(output, "!");
                    tokens += completion;
                }
                drop(receipts);
                let after_receipts = host.serving_bytes();
                assert!(after_receipts < with_receipts, "receipt charges released");
                // The grown queue backing is legitimately retained while
                // the router and its workers still own it.
                assert!(after_receipts > baseline, "live queue backing retained");
                let queue = router.queue_snapshot();
                let active = router.active_sessions();
                let peak = router.peak_active_sessions();
                assert_eq!(ids.len(), population, "unique completed request ids");
                assert_eq!(tokens, population, "one actual output token per request");
                assert_eq!(queue.pending, 0);
                assert_eq!(queue.in_progress, 0);
                assert_eq!(queue.limit, None);
                assert_eq!(queue.consumers, WORKERS);
                assert!(queue.capacity > 0);
                assert!(queue.backing_bytes > 0);
                assert!(active <= router.active_limit());
                assert!(peak >= 1 && peak <= router.active_limit());
                println!(
                    "population {population} completed: completed={} tokens={tokens} active={active} pending={} in_progress={} capacity={} backing={} peak={peak}",
                    ids.len(),
                    queue.pending,
                    queue.in_progress,
                    queue.capacity,
                    queue.backing_bytes
                );
            }

            let final_serving = host.serving_bytes();
            assert!(final_serving > 0);
            drop(router);
            let closed = host.serving_bytes();
            assert_eq!(
                closed, 0,
                "router drop releases model/request/queue/workers"
            );
            let memory = host.snapshot().expect("host snapshot");
            assert_eq!(memory.3, 0);
            assert!(memory.1 > 0, "permanent bootstrap/Rayon/root reservations");
            let persistent_reserved = memory.1;
            println!(
                "router closure: serving_before={final_serving} serving_after={closed} reserved={} limit={} remaining={}",
                memory.1, memory.0, memory.2
            );

            // Optional pending count cap: hold one real reserve() ticket, so
            // the next actual request is refused by the configured queue cap
            // (429 queue_full), not by the application host budget. Releasing
            // the ticket then admits and serves a real request.
            let capped = match Router::new(
                vec![("tiny".to_string(), model)],
                WORKERS,
                ACTIVE_PER_WORKER,
                MAX_CONTEXT,
                MAX_OUTPUT,
                HOST_MIB,
                Some(1),
            ) {
                Ok(router) => router,
                Err(error) => panic!(
                    "capped 4096 MiB / 8 active router refused: status={} code={} message={}",
                    error.status, error.code, error.message
                ),
            };
            assert_eq!(capped.active_limit(), ACTIVE_LIMIT);
            assert_eq!(capped.queue_snapshot().limit, Some(1));
            let capped_host = capped.host.clone();
            let capped_baseline = capped_host.serving_bytes();
            let ticket = capped
                .queue
                .reserve()
                .expect("real capped queue reservation");
            let held = capped.queue_snapshot();
            assert_eq!(held.limit, Some(1));
            assert_eq!(held.pending, 0);
            assert_eq!(held.in_progress, 1);
            let refused = match capped.request(request()) {
                Err(error) => error,
                Ok(receipt) => panic!(
                    "capped router admitted request {} while a real ticket was held",
                    receipt.id
                ),
            };
            assert_eq!(
                (refused.status, refused.code),
                (429, "queue_full"),
                "actual queue-cap refusal: {}",
                refused.message
            );
            assert_eq!(capped.queue_snapshot().in_progress, 1);
            drop(ticket);
            assert_eq!(capped.queue_snapshot().in_progress, 0);
            let receipt = capped
                .request(request())
                .expect("ticket release admits a real request");
            let (id, prompt, completion, output) = drain(&receipt);
            assert!(id > 0);
            assert!(prompt >= 1);
            assert_eq!(completion, 1);
            assert_eq!(output, "!");
            let accepted_queue = capped.queue_snapshot();
            assert_eq!(accepted_queue.pending, 0);
            assert_eq!(accepted_queue.in_progress, 0);
            let with_receipt = capped_host.serving_bytes();
            assert!(with_receipt > capped_baseline);
            drop(receipt);
            let after_receipt = capped_host.serving_bytes();
            assert!(after_receipt < with_receipt);
            assert!(
                after_receipt > capped_baseline,
                "capped queue backing retained"
            );
            println!(
                "cap: limit=1 held={held:?} refused_status={} refused_code={} accepted_id={id} tokens={completion} serving_receipts={with_receipt} serving_after_receipt={after_receipt}",
                refused.status, refused.code
            );
            drop(capped);
            let capped_closed = capped_host.serving_bytes();
            assert_eq!(
                capped_closed, 0,
                "capped router drop releases serving charges"
            );
            let capped_memory = capped_host.snapshot().expect("capped host snapshot");
            assert_eq!(capped_memory.3, 0);
            assert_eq!(
                capped_memory.1, persistent_reserved,
                "second router releases model/queue/worker charges and reuses globals"
            );
            println!(
                "cap closure: serving_after={capped_closed} reserved={} persistent_reserved={persistent_reserved}",
                capped_memory.1
            );
            std::fs::remove_dir_all(&root).unwrap();
            println!("POPULATION_CHILD_OK");
        }

        #[test]
        #[ignore = "requires approved tokenizer metadata"]
        fn cancelled_requests_keep_source_credit_and_later_genuine_encode_succeeds() {
            let test = "serve::engine::tests::serving_population::cancelled_requests_keep_source_credit_and_later_genuine_encode_succeeds";
            if reexec("ZG_SERVE_CANCEL_CHILD", test, "CANCEL_CHILD_OK") {
                return;
            }
            let root = fixture("cancel");
            let router = Router::new(
                vec![("tiny".to_string(), root.join("model.gguf"))],
                WORKERS,
                ACTIVE_PER_WORKER,
                MAX_CONTEXT,
                MAX_OUTPUT,
                HOST_MIB,
                None,
            )
            .expect("4096 MiB / 8 active router");
            let host = router.host.clone();
            let source = Arc::clone(router.models.get("tiny").expect("actual source route"));
            let baseline = host.snapshot().expect("host snapshot");
            let before = source
                .tokenizer
                .prompt_tokens("!", false)
                .expect("genuine encode before cancellation");
            assert!(!before.is_empty());

            // Dropping the receipt is the request's cancellation owner. Phase
            // timing is intentionally not asserted: whether the row was still
            // queued or already active is not pinned by this test. The source
            // model keeps its `Retained` R+G+V lease independent of
            // cancellation.
            let receipt = router.request(request()).expect("real cancellable request");
            drop(receipt);
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                let queue = router.queue_snapshot();
                if queue.pending == 0 && queue.in_progress == 0 && router.active_sessions() == 0 {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "cancelled request did not retire: queue={queue:?} active={}",
                    router.active_sessions()
                );
                thread::sleep(Duration::from_millis(10));
            }
            let after_cancel = host.snapshot().expect("host snapshot after cancel");
            assert_eq!(
                after_cancel.0, baseline.0,
                "configured host limit is unchanged"
            );
            assert!(
                after_cancel.1 >= baseline.1,
                "source/bootstrap reservations survive request cancellation"
            );
            assert_eq!(
                source
                    .tokenizer
                    .prompt_tokens("!", false)
                    .expect("genuine encode after cancellation"),
                before,
                "the source tokenizer and its retained credit remain live"
            );

            // The same source still serves a real request after cancellation.
            let receipt = router
                .request(request())
                .expect("request after cancellation");
            let (_, prompt, completion, output) = drain(&receipt);
            assert!(prompt >= 1);
            assert_eq!(completion, 1);
            assert_eq!(output, "!");
            drop(receipt);
            assert_eq!(
                source
                    .tokenizer
                    .prompt_tokens("!", false)
                    .expect("genuine encode after serving"),
                before
            );
            drop(source);
            drop(router);
            assert_eq!(host.serving_bytes(), 0);
            std::fs::remove_dir_all(&root).unwrap();
            println!("CANCEL_CHILD_OK");
        }

        #[test]
        #[ignore = "requires approved tokenizer metadata"]
        fn application_host_refusal_is_distinct_from_the_pending_cap() {
            let test = "serve::engine::tests::serving_population::application_host_refusal_is_distinct_from_the_pending_cap";
            if reexec("ZG_SERVE_HOST_CHILD", test, "HOST_CHILD_OK") {
                return;
            }
            // Same genuine 24-layer/full-vocabulary source. Retained `R + G + V`
            // (plus bootstrap/constructor rows) exceeds a 1 GiB application
            // host, so the refusal is the host_budget stage before weights are
            // opened, not the optional pending cap.
            let root = fixture("host");
            let source = root.join("model.gguf");
            let error = match Router::new(
                vec![("tiny".to_string(), source)],
                1,
                1,
                MAX_CONTEXT,
                MAX_OUTPUT,
                1024,
                None,
            ) {
                Err(error) => error,
                Ok(_) => panic!("host admission must refuse the tokenizer retention"),
            };
            assert_eq!(
                (error.status, error.code),
                (503, "host_budget"),
                "actual host refusal: {}",
                error.message
            );
            assert!(
                error.message.contains("stage=retained-inventory"),
                "actual host stage: {}",
                error.message
            );
            assert!(
                error.message.contains("owner=model=tiny"),
                "actual host owner: {}",
                error.message
            );
            assert!(
                error.message.contains("session=retained-inventory"),
                "actual host session: {}",
                error.message
            );
            std::fs::remove_dir_all(&root).unwrap();
            println!("HOST_CHILD_OK: {}", error.message);
        }
    }
}
