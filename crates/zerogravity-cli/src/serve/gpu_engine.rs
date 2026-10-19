//! Resident GPU OpenAI serving: one loaded model source, independent device
//! StateRoots, one physical workspace and terminal completion per GPU step.
//! Opt-in CUDA B2/B4/B8 waves use the original native ForwardPlan; other work
//! interleaves through B1. Device feedback retains each SessionAuthority.

use super::engine::{
    EVENTS_PER_REQUEST, GeneratedText, MAX_ACTIVE_PER_WORKER, Presentation, QUEUED_PER_WORKER,
    Request, RequestEvents, ResponseEvent, Scheduled, ServiceError,
};
use super::gpu_batch::{self, Limits, Phase, Ready};
use crate::GpuBackend;
use futures_lite::future::block_on;
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use zerogravity::gpu::{
    InferenceReadback, PreparedResidentCohort, ResidentModel, ResidentSession, ResidentToken,
    resident_batch::ResidentBatchRows,
};
use zerogravity::qwen35::LoadedQwenModel;
use zerogravity::tokenize::QwenTokenizer;

const GPU_CAPACITY: usize = 512;
const PREFILL_QUANTUM: usize = 32;
const OUTPUT_STALL_LIMIT: Duration = Duration::from_secs(30);
mod cohort;
use cohort::{advance_decode_cohort, finish_prompt_cohort};
#[cfg(all(test, feature = "cuda"))]
mod qualification;
#[cfg(test)]
thread_local! {
    static AFTER_COHORT_READY: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}
pub(super) struct GpuProfile {
    pub workers: usize,
    pub active: usize,
    pub context: usize,
    pub output: usize,
    pub provider_mib: usize,
    pub device_mib: usize,
    pub native_rows: usize,
}

/// CubeCL CUDA's C++ backend requires a toolkit INCLUDE root in addition
/// to the NVIDIA driver/NVRTC library. Validate before the device client:
/// without this, a driver thread can panic/retry indefinitely.
fn check_cuda_toolkit(path: &Path) -> Result<(), ServiceError> {
    for header in ["include/cuda_runtime.h", "include/cuda_fp16.h"] {
        if !path.join(header).is_file() {
            return Err(ServiceError::new(
                503,
                "cuda_toolkit_missing",
                format!("CUDA_PATH={} does not contain {header}", path.display()),
            ));
        }
    }
    if ![
        path.join("include/cccl/cuda/std"),
        path.join("include/cuda/std"),
    ]
    .iter()
    .any(|p| p.is_dir())
    {
        return Err(ServiceError::new(
            503,
            "cuda_cccl_missing",
            format!(
                "CUDA_PATH={} has no CUDA CCCL device headers",
                path.display()
            ),
        ));
    }
    Ok(())
}
fn preflight_cuda_toolkit(backend: GpuBackend) -> Result<(), ServiceError> {
    if backend != GpuBackend::Cuda {
        return Ok(());
    }
    let installed = std::env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .or_else(|| {
            ["/usr/local/cuda", "/opt/cuda"]
                .into_iter()
                .map(PathBuf::from)
                .find(|p| p.is_dir())
        })
        .or_else(|| {
            Path::new("/usr/bin/nvcc")
                .exists()
                .then(|| PathBuf::from("/usr"))
        })
        .ok_or_else(|| {
            ServiceError::new(
                503,
                "cuda_toolkit_missing",
                "CUDA NVRTC requires CUDA_PATH pointing to toolkit include headers",
            )
        })?;
    check_cuda_toolkit(&installed)
}

/// This is one host route to an already resident GPU source/graph.
struct GpuModel {
    resident: Arc<ResidentModel>,
    tokenizer: Arc<QwenTokenizer>,
    group: usize,
    hidden_width: usize,
    native_rows: usize,
}
#[derive(Default)]
struct WaveStats {
    attempted: AtomicU64,
    completed: AtomicU64,
    gpu_forwards: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    queued: AtomicUsize,
    active: AtomicUsize,
    groups: AtomicU64,
    co_scheduled: AtomicU64,
    cohort_publications: AtomicU64,
    cohort_rows: AtomicU64,
    kernel_dispatches: AtomicU64,
    model_tokens: AtomicU64,
    committed_roots: AtomicU64,
    fused_batches: AtomicU64,
    native_bands: [AtomicU64; 3],
    native_admission_refusals: AtomicU64,
    fused_rows: AtomicU64,
    native_projections: AtomicU64,
    native_sampler: AtomicU64,
    discarded_cohorts: AtomicU64,
    output_tokens: AtomicU64,
    parked: AtomicUsize,
}
struct SessionCharge(Arc<WaveStats>);
impl Drop for SessionCharge {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}
pub(super) struct GpuMetrics {
    pub attempted: u64,
    pub completed: u64,
    pub gpu_forwards: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub queued: usize,
    pub active: usize,
    pub groups: u64,
    pub co_scheduled: u64,
    pub cohort_publications: u64,
    pub cohort_rows: u64,
    pub kernel_dispatches: u64,
    pub model_tokens: u64,
    pub committed_roots: u64,
    pub fused_batches: u64,
    pub native_bands: [u64; 3],
    pub native_admission_refusals: u64,
    pub fused_rows: u64,
    pub native_projections: u64,
    pub native_sampler: u64,
    pub discarded_cohorts: u64,
    pub output_tokens: u64,
    pub parked: usize,
    pub native_rows: usize,
}
pub(super) struct GpuRouter {
    models: Arc<BTreeMap<String, Arc<GpuModel>>>,
    sender: mpsc::SyncSender<Scheduled>,
    next_id: AtomicU64,
    stats: Arc<WaveStats>,
    pub(super) max_output: usize,
    native_rows: usize,
}
impl GpuRouter {
    pub(super) fn validate_profile(
        workers: usize,
        active: usize,
        max_context: usize,
        max_output: usize,
        provider_mib: usize,
        device_mib: usize,
    ) -> Result<(), ServiceError> {
        if workers != 1
            || !(1..=MAX_ACTIVE_PER_WORKER).contains(&active)
            || max_context != GPU_CAPACITY
            || !(1..=GPU_CAPACITY).contains(&max_output)
            || !(256..=65536).contains(&provider_mib)
            || !(256..=65536).contains(&device_mib)
            || provider_mib > device_mib
        {
            return Err(ServiceError::new(
                400,
                "invalid_gpu_profile",
                "GPU serve requires workers=1, context=512, active=1..8, \
                 output=1..512, provider/device 256..65536 MiB and provider<=device",
            ));
        }
        Ok(())
    }
    pub(super) fn new(
        declared: Vec<(String, PathBuf)>,
        backend: GpuBackend,
        profile: GpuProfile,
    ) -> Result<Self, ServiceError> {
        let GpuProfile {
            workers,
            active,
            context,
            output,
            provider_mib,
            device_mib,
            native_rows,
        } = profile;
        Self::validate_profile(workers, active, context, output, provider_mib, device_mib)?;
        if native_rows != 0
            && (!matches!(native_rows, 2 | 4 | 8)
                || backend != GpuBackend::Cuda
                || active < native_rows)
        {
            return Err(ServiceError::new(
                400,
                "unsupported_batch_mode",
                "fused serving requires CUDA, rows=2/4/8 and active>=rows",
            ));
        }
        preflight_cuda_toolkit(backend)?;
        if declared.is_empty() || declared.len() > 16 {
            return Err(ServiceError::new(
                400,
                "invalid_models",
                "GPU model routes require 1..16 IDs",
            ));
        }
        // CPU fallback is forbidden: unlinked or absent provider must refuse
        // before a resident model is constructed or HTTP announces readiness.
        let provider_cap = provider_mib.checked_mul(1024 * 1024).ok_or_else(|| {
            ServiceError::new(400, "provider_budget", "GPU provider cap overflow")
        })?;
        let device_cap = (device_mib as u64) * 1024 * 1024;
        let mut models = BTreeMap::new();
        let mut unique = BTreeMap::<PathBuf, Arc<GpuModel>>::new();
        for (alias, path) in declared {
            if alias.is_empty()
                || alias.len() > 96
                || !alias
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                || models.contains_key(&alias)
            {
                return Err(ServiceError::new(
                    400,
                    "invalid_model",
                    "bad/duplicate GPU route name",
                ));
            }
            let path = std::fs::canonicalize(path).map_err(|_| {
                ServiceError::new(400, "missing_model", "GGUF source does not exist")
            })?;
            if let Some(resident) = unique.get(&path) {
                models.insert(alias, Arc::clone(resident));
                continue;
            }
            let parent = path.parent().unwrap_or(Path::new("."));
            let tokenizer = QwenTokenizer::from_file(&parent.join("tokenizer.json"))
                .map_err(ServiceError::inference)?;
            let source = LoadedQwenModel::open(&path, &parent.join("config.json"))
                .map_err(ServiceError::inference)?;
            let hidden_width = usize::try_from(source.geometry().hidden_size).map_err(|_| {
                ServiceError::new(400, "source_hidden", "invalid Qwen source hidden width")
            })?;
            let plan = source
                .forward_plan(GPU_CAPACITY, true, true)
                .map_err(ServiceError::inference)?;
            let device = backend.device().map_err(ServiceError::inference)?;
            let resident = block_on(ResidentModel::load(
                device,
                source,
                plan,
                provider_cap,
                device_cap,
            ))
            .map_err(ServiceError::inference)?;
            let shared = Arc::new(GpuModel {
                resident,
                tokenizer: Arc::new(tokenizer),
                group: unique.len(),
                hidden_width,
                native_rows,
            });
            unique.insert(path, Arc::clone(&shared));
            models.insert(alias, shared);
        }
        let models = Arc::new(models);
        let (sender, receiver) = mpsc::sync_channel(QUEUED_PER_WORKER);
        let sources = Arc::clone(&models);
        let stats = Arc::new(WaveStats::default());
        let worker_stats = Arc::clone(&stats);
        thread::Builder::new()
            .name("zg-resident-gpu".into())
            .spawn(move || worker(receiver, sources, active, worker_stats, native_rows))
            .map_err(|e| ServiceError::new(500, "worker_spawn", e.to_string()))?;
        Ok(Self {
            models,
            sender,
            next_id: AtomicU64::new(1),
            stats,
            max_output: output,
            native_rows,
        })
    }
    pub(super) fn metrics(&self) -> GpuMetrics {
        GpuMetrics {
            attempted: self.stats.attempted.load(Ordering::Relaxed),
            completed: self.stats.completed.load(Ordering::Relaxed),
            gpu_forwards: self.stats.gpu_forwards.load(Ordering::Relaxed),
            failed: self.stats.failed.load(Ordering::Relaxed),
            cancelled: self.stats.cancelled.load(Ordering::Relaxed),
            queued: self.stats.queued.load(Ordering::Acquire),
            active: self.stats.active.load(Ordering::Acquire),
            groups: self.stats.groups.load(Ordering::Relaxed),
            co_scheduled: self.stats.co_scheduled.load(Ordering::Relaxed),
            cohort_publications: self.stats.cohort_publications.load(Ordering::Relaxed),
            cohort_rows: self.stats.cohort_rows.load(Ordering::Relaxed),
            kernel_dispatches: self.stats.kernel_dispatches.load(Ordering::Relaxed),
            model_tokens: self.stats.model_tokens.load(Ordering::Relaxed),
            committed_roots: self.stats.committed_roots.load(Ordering::Relaxed),
            fused_batches: self.stats.fused_batches.load(Ordering::Relaxed),
            native_bands: std::array::from_fn(|i| {
                self.stats.native_bands[i].load(Ordering::Relaxed)
            }),
            native_admission_refusals: self.stats.native_admission_refusals.load(Ordering::Relaxed),
            fused_rows: self.stats.fused_rows.load(Ordering::Relaxed),
            native_projections: self.stats.native_projections.load(Ordering::Relaxed),
            native_sampler: self.stats.native_sampler.load(Ordering::Relaxed),
            discarded_cohorts: self.stats.discarded_cohorts.load(Ordering::Relaxed),
            output_tokens: self.stats.output_tokens.load(Ordering::Relaxed),
            parked: self.stats.parked.load(Ordering::Relaxed),
            native_rows: self.native_rows,
        }
    }
    pub(super) fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }
    pub(super) fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
        if !self.models.contains_key(&request.model) {
            return Err(ServiceError::new(
                404,
                "model_not_found",
                "unregistered GPU model",
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
                "GPU request bounds",
            ));
        }
        let model = request.model.clone();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if id == u64::MAX {
            return Err(ServiceError::new(
                503,
                "id_exhausted",
                "request IDs exhausted",
            ));
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::sync_channel(EVENTS_PER_REQUEST);
        self.stats.queued.fetch_add(1, Ordering::AcqRel);
        self.sender
            .try_send(Scheduled {
                request,
                events: tx,
                cancelled: Arc::clone(&cancelled),
            })
            .map_err(|err| {
                self.stats.queued.fetch_sub(1, Ordering::AcqRel);
                match err {
                    mpsc::TrySendError::Full(_) => {
                        ServiceError::new(429, "queue_full", "GPU queue at capacity")
                    }
                    mpsc::TrySendError::Disconnected(_) => {
                        ServiceError::new(503, "worker_unavailable", "GPU executor unavailable")
                    }
                }
            })?;
        Ok(RequestEvents::new(id, model, rx, cancelled))
    }
}
struct GpuActive {
    _charge: SessionCharge,
    resident: Arc<ResidentModel>,
    session: ResidentSession,
    group: usize,
    hidden_width: usize,
    cancelled: Arc<AtomicBool>,
    errors: mpsc::SyncSender<ResponseEvent>,
    output: GeneratedText,
    prompt: Vec<u32>,
    cursor: usize,
    device_token: Option<ResidentToken>,
    native_rows: usize,
    packet_presented: bool,
    blocked_since: Option<Instant>,
}
impl GpuActive {
    fn new(
        job: Scheduled,
        model: Arc<GpuModel>,
        stats: Arc<WaveStats>,
    ) -> Result<Self, ServiceError> {
        let prompt = model
            .tokenizer
            .prompt_tokens(&job.request.input, job.request.chat)
            .map_err(ServiceError::inference)?;
        if prompt.is_empty()
            || prompt.len() > GPU_CAPACITY
            || prompt
                .len()
                .checked_add(job.request.max_tokens - 1)
                .is_none_or(|total| total > GPU_CAPACITY)
        {
            return Err(ServiceError::new(
                400,
                "context_overflow",
                "resident GPU profile admits at most 512 consumed tokens",
            ));
        }
        let output = GeneratedText::new(
            Arc::clone(&model.tokenizer),
            job.events.clone(),
            job.request.stream,
            job.request.max_tokens,
            prompt.len(),
        );
        // ResidentSession allocates independent, physical source-bound GPU
        // state. Neither its tensor source nor its StateRoot is CPU fallback.
        let session = block_on(model.resident.session()).map_err(ServiceError::inference)?;
        stats.active.fetch_add(1, Ordering::AcqRel);
        Ok(Self {
            _charge: SessionCharge(stats),
            resident: Arc::clone(&model.resident),
            session,
            group: model.group,
            hidden_width: model.hidden_width,
            cancelled: job.cancelled,
            errors: job.events,
            output,
            prompt,
            cursor: 0,
            device_token: None,
            native_rows: model.native_rows,
            packet_presented: false,
            blocked_since: None,
        })
    }
    fn ready(&self) -> Ready {
        let remaining = self.prompt.len().saturating_sub(self.cursor);
        Ready {
            source: self.group,
            phase: if remaining > 0 {
                Phase::Prefill
            } else {
                Phase::Decode
            },
            frontier: self.session.binding().frontier() as usize,
            tokens: if remaining > 0 {
                self.prefill_count()
            } else {
                1
            },
            // This is finite scheduler credit, NOT a claim that this is
            // the true physical GPU workspace footprint.
            scratch_bytes: 32 * 1024 * 1024,
        }
    }
    fn prefill_count(&self) -> usize {
        let remaining = self.prompt.len() - self.cursor;
        if self.native_rows != 0 && remaining > 1 {
            (remaining - 1).min(PREFILL_QUANTUM)
        } else {
            remaining.min(PREFILL_QUANTUM)
        }
    }
    /// Presentation precedes consumption. A Ready-cohort retry retains this
    /// mark and the original packet; only a newly committed packet resets it.
    fn present_decode(&mut self, stats: &WaveStats) -> Result<Presentation, ServiceError> {
        let presented = if self.packet_presented {
            self.output.resume()?
        } else {
            let token = self.device_token.as_ref().ok_or_else(|| {
                ServiceError::new(
                    500,
                    "missing_device_token",
                    "resident GPU feedback is missing",
                )
            })?;
            let before = self.output.generated_token_count();
            let status = self.output.present(token.id)?;
            stats.output_tokens.fetch_add(
                (self.output.generated_token_count() - before) as u64,
                Ordering::Relaxed,
            );
            self.packet_presented = true;
            status
        };
        if presented == Presentation::Blocked {
            let since = self.blocked_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= OUTPUT_STALL_LIMIT {
                return Err(ServiceError::new(
                    503,
                    "backpressure_timeout",
                    "client output credit deadline expired",
                ));
            }
        } else {
            self.blocked_since = None;
        }
        Ok(presented)
    }
    fn install(&mut self, read: InferenceReadback) -> Result<(), ServiceError> {
        let token = read.device_token.ok_or_else(|| {
            ServiceError::new(
                500,
                "missing_device_token",
                "GPU completion has no original greedy packet",
            )
        })?;
        self.device_token = Some(token);
        self.packet_presented = false;
        self.blocked_since = None;
        Ok(())
    }
    /// Original B1 source execution, including all unmatched native-wave rows.
    fn step(&mut self, stats: &WaveStats) -> Result<MemberStep, ServiceError> {
        if self.cursor < self.prompt.len() {
            let end = self.cursor + self.prefill_count();
            let read = if end < self.prompt.len() {
                block_on(
                    self.session
                        .prefill_without_logits(&self.prompt[self.cursor..end], false),
                )
            } else {
                block_on(self.session.forward_greedy(&self.prompt[self.cursor..end]))
            }
            .map_err(ServiceError::inference)?;
            stats.single_completion(&read);
            if end == self.prompt.len() {
                self.install(read)?;
            }
            self.cursor = end;
            return Ok(MemberStep::Forwarded);
        }
        match self.present_decode(stats)? {
            Presentation::Finished => return Ok(MemberStep::Finished),
            Presentation::Blocked => return Ok(MemberStep::Parked),
            Presentation::Ready => {}
        }
        if self.cancelled.load(Ordering::Acquire) {
            return Ok(MemberStep::Cancelled);
        }
        let token = self.device_token.as_ref().ok_or_else(|| {
            ServiceError::new(
                500,
                "missing_device_token",
                "resident GPU feedback is missing",
            )
        })?;
        let read =
            block_on(self.session.forward_device_token(token)).map_err(ServiceError::inference)?;
        stats.single_completion(&read);
        self.install(read)?;
        Ok(MemberStep::Forwarded)
    }
}

enum MemberStep {
    Forwarded,
    Finished,
    Parked,
    Cancelled,
}
impl WaveStats {
    fn single_completion(&self, read: &InferenceReadback) {
        self.gpu_forwards.fetch_add(1, Ordering::Relaxed);
        self.kernel_dispatches
            .fetch_add(read.dispatches as u64, Ordering::Relaxed);
        self.model_tokens.fetch_add(
            (read.state_only_steps + read.logit_steps) as u64,
            Ordering::Relaxed,
        );
        self.committed_roots.fetch_add(1, Ordering::Relaxed);
    }
    fn fail(&self, row: &GpuActive, error: ServiceError) {
        if error.status == 499 && error.code == "cancelled" {
            self.cancelled.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed.fetch_add(1, Ordering::Relaxed);
        }
        let _ = row.errors.try_send(ResponseEvent::Error(error));
    }
}
fn advance_member(next: GpuActive, active: &mut VecDeque<GpuActive>, stats: &WaveStats) {
    let Some(mut next) = eligible_member(next, active, stats) else {
        return;
    };
    stats.attempted.fetch_add(1, Ordering::Relaxed);
    match next.step(stats) {
        Ok(MemberStep::Parked) => active.push_back(next),
        Ok(MemberStep::Cancelled) => {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        Ok(outcome) => {
            stats.completed.fetch_add(1, Ordering::Relaxed);
            if matches!(outcome, MemberStep::Forwarded) {
                active.push_back(next);
            }
        }
        Err(error) => {
            stats.fail(&next, error);
        }
    }
}
/// Resume bounded output before building new GPU waves. Parked rows own
/// no submitted command, queue slot or physical workspace guard.
fn eligible_member(
    mut row: GpuActive,
    parked: &mut VecDeque<GpuActive>,
    stats: &WaveStats,
) -> Option<GpuActive> {
    if row.cancelled.load(Ordering::Acquire) {
        stats.cancelled.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if row.cursor < row.prompt.len() {
        return Some(row);
    }
    match row.present_decode(stats) {
        Ok(Presentation::Ready) => Some(row),
        Ok(Presentation::Blocked) => {
            parked.push_back(row);
            None
        }
        Ok(Presentation::Finished) => {
            stats.attempted.fetch_add(1, Ordering::Relaxed);
            stats.completed.fetch_add(1, Ordering::Relaxed);
            None
        }
        Err(error) => {
            stats.attempted.fetch_add(1, Ordering::Relaxed);
            stats.fail(&row, error);
            None
        }
    }
}

fn worker(
    rx: mpsc::Receiver<Scheduled>,
    models: Arc<BTreeMap<String, Arc<GpuModel>>>,
    active_limit: usize,
    stats: Arc<WaveStats>,
    native_rows: usize,
) {
    let mut active = VecDeque::<GpuActive>::new();
    loop {
        active.retain(|row| {
            if row.cancelled.load(Ordering::Acquire) {
                stats.cancelled.fetch_add(1, Ordering::Relaxed);
                false
            } else {
                true
            }
        });
        while active.len() < active_limit {
            let next = if active.is_empty() {
                match rx.recv_timeout(Duration::from_millis(25)) {
                    Ok(v) => Some(v),
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            } else {
                rx.try_recv().ok()
            };
            let Some(job) = next else { break };
            stats.queued.fetch_sub(1, Ordering::AcqRel);
            if job.cancelled.load(Ordering::Acquire) {
                stats.cancelled.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let Some(model) = models.get(&job.request.model) else {
                continue;
            };
            let errors = job.events.clone();
            match GpuActive::new(job, Arc::clone(model), Arc::clone(&stats)) {
                Ok(session) => active.push_back(session),
                Err(e) => {
                    stats.failed.fetch_add(1, Ordering::Relaxed);
                    let _ = errors.try_send(ResponseEvent::Error(e));
                }
            }
        }
        if active.is_empty() {
            continue;
        }
        let mut eligible = VecDeque::new();
        let mut parked = VecDeque::new();
        for row in active.drain(..) {
            if let Some(row) = eligible_member(row, &mut parked, &stats) {
                eligible.push_back(row);
            }
        }
        active = parked;
        stats.parked.store(active.len(), Ordering::Relaxed);
        if eligible.is_empty() {
            // No locks/queue slot while waiting for bounded transport credit.
            if !active.is_empty() {
                thread::sleep(Duration::from_millis(2));
            }
            continue;
        }
        let ready = eligible.iter().map(GpuActive::ready).collect::<Vec<_>>();
        let waves = match gpu_batch::form(
            &ready,
            Limits {
                max_members: if native_rows != 0 {
                    native_rows
                } else {
                    active_limit
                },
                max_tokens: 64,
                max_scratch: 256 * 1024 * 1024,
            },
        ) {
            Ok(groups) => groups,
            Err(error) => {
                for failed in eligible.drain(..) {
                    stats.failed.fetch_add(1, Ordering::Relaxed);
                    let _ = failed.errors.try_send(ResponseEvent::Error(error.clone()));
                }
                continue;
            }
        };
        let mut pending = eligible.drain(..).map(Some).collect::<Vec<_>>();
        for wave in waves {
            // Validate source/epoch/unique committed session roots BEFORE
            // issuing even the first physical GPU member. P12 source
            // batch admission is shared by native bands and original B1 paths.
            let admitted = wave
                .indices
                .iter()
                .filter_map(|&i| pending[i].as_ref())
                .filter(|row| !row.cancelled.load(Ordering::Acquire))
                .collect::<Vec<_>>();
            if let Some(first) = admitted.first() {
                let roots = admitted
                    .iter()
                    .map(|row| row.session.binding())
                    .collect::<Vec<_>>();
                let verification = ResidentBatchRows::new(&roots, first.hidden_width);
                let verification = verification.and_then(|cohort| {
                    if admitted
                        .iter()
                        .any(|row| row.hidden_width != cohort.row_elements())
                    {
                        return Err(zerogravity::Error::State(
                            "GPU model wave hidden widths differ".into(),
                        ));
                    }
                    cohort.require_current(&roots)?;
                    Ok(cohort)
                });
                if let Err(error) = verification {
                    stats
                        .failed
                        .fetch_add(admitted.len() as u64, Ordering::Relaxed);
                    let refusal = ServiceError::inference(error);
                    for index in wave.indices {
                        if let Some(row) = pending[index].take() {
                            let _ = row.errors.try_send(ResponseEvent::Error(refusal.clone()));
                        }
                    }
                    continue;
                }
            }
            stats.groups.fetch_add(1, Ordering::Relaxed);
            if wave.indices.len() > 1 {
                stats
                    .co_scheduled
                    .fetch_add(wave.indices.len() as u64, Ordering::Relaxed);
            }
            // Exact prompt tails share the existing checked root publisher.
            // Native mode uses multi-row kernels; interleaved mode keeps B1 graphs.
            let batch_ready = wave.indices.len() > 1
                && wave.indices.iter().all(|&i| {
                    pending[i].as_ref().is_some_and(|row| {
                        !row.cancelled.load(Ordering::Acquire)
                            && row.cursor + 1 == row.prompt.len()
                            && row.device_token.is_none()
                    })
                });
            if batch_ready {
                let members = wave
                    .indices
                    .iter()
                    .map(|&i| pending[i].take().expect("unique admitted wave row"))
                    .collect::<Vec<_>>();
                finish_prompt_cohort(members, &mut active, &stats, native_rows);
                continue;
            }
            if native_rows != 0
                && wave.indices.len() > 1
                && wave.indices.iter().all(|&i| {
                    pending[i]
                        .as_ref()
                        .is_some_and(|row| row.cursor == row.prompt.len())
                })
            {
                let members = wave
                    .indices
                    .iter()
                    .map(|&i| pending[i].take().expect("unique admitted decode row"))
                    .collect();
                advance_decode_cohort(members, &mut active, &stats, native_rows);
                continue;
            }
            for index in wave.indices {
                let next = pending[index]
                    .take()
                    .expect("GPU wave planner must cover unique ready slots");
                advance_member(next, &mut active, &stats);
            }
        }
        stats.parked.store(
            active
                .iter()
                .filter(|row| row.blocked_since.is_some())
                .count(),
            Ordering::Relaxed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_requires_runtime_and_cccl_headers_before_device_startup() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "zerogravity-cuda-prerequisite-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("include")).unwrap();
        assert_eq!(
            check_cuda_toolkit(&root).unwrap_err().code,
            "cuda_toolkit_missing"
        );
        std::fs::write(root.join("include/cuda_runtime.h"), b"").unwrap();
        std::fs::write(root.join("include/cuda_fp16.h"), b"").unwrap();
        assert_eq!(
            check_cuda_toolkit(&root).unwrap_err().code,
            "cuda_cccl_missing"
        );
        std::fs::create_dir_all(root.join("include/cccl/cuda/std")).unwrap();
        assert!(check_cuda_toolkit(&root).is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gpu_profile_is_bounded_before_any_device_lookup() {
        assert!(GpuRouter::validate_profile(1, 2, 512, 16, 2048, 4096).is_ok());
        for bad in [
            (2, 2, 512, 16, 2048, 4096),
            (1, 0, 512, 16, 2048, 4096),
            (1, 9, 512, 16, 2048, 4096),
            (1, 2, 8192, 16, 2048, 4096),
            (1, 2, 512, 513, 2048, 4096),
            (1, 2, 512, 16, 4096, 2048),
            (1, 2, 512, 16, 0, 4096),
        ] {
            assert!(GpuRouter::validate_profile(bad.0, bad.1, bad.2, bad.3, bad.4, bad.5).is_err());
        }
    }
}
