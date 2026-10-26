//! Resident GPU OpenAI serving: one loaded model source, independent device
//! StateRoots, one physical workspace and terminal completion per GPU step.
//! Opt-in CUDA B2/B4/B8 waves use the original native ForwardPlan; other work
//! interleaves through B1. Device feedback retains each SessionAuthority.

use super::engine::{
    EVENTS_PER_REQUEST, GeneratedText, GpuTokenObservation, MAX_ACTIVE_PER_WORKER, MetricsLease,
    NativeCohortObservation, Presentation, Request, RequestEvents, RequestLease, ResponseEvent,
    Scheduled, ServiceError, TextTokenizer, output_transport_bound, read_approved_tokenizer,
    vecdeque_backing_bytes, workers_charge,
};
use super::event_channel::mpsc;
use super::gpu_batch::{self, Limits, Phase, Ready};
use super::pending::{
    self, PendingQueue, Pop, QueueSnapshot, ServingCharge, ServingHost, WorkerQueue,
};
use crate::GpuBackend;
use futures_lite::future::block_on;
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use zerogravity::gpu::{
    InferenceReadback, PreparedResidentCohort, ResidentModel, ResidentSession, ResidentToken,
    resident_batch::ResidentBatchRows,
};
use zerogravity::qwen35::LoadedQwenModel;
use zerogravity::resource_arbiter::AllocationClass;
use zerogravity::tokenize::QwenTokenizer;

const GPU_CAPACITY: usize = 512;
const PREFILL_QUANTUM: usize = 32;
const OUTPUT_STALL_LIMIT: Duration = Duration::from_secs(30);
mod cohort;
mod route;
use route::{ModelEngine, RequestEngine};
#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
mod fixed_prompt;
#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
mod qwen35_cuda;
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
    pub host_mib: usize,
    pub max_pending: Option<usize>,
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
    engine: ModelEngine,
    tokenizer: TextTokenizer,
    group: usize,
    hidden_width: usize,
    vocabulary_size: u64,
    native_rows: usize,
    max_token_bytes: usize,
    /// Retained `R + G + V` source credit owned by the model that holds it.
    _inventory: Option<zerogravity::tokenize_admission::Retained>,
    readers: Arc<AtomicUsize>,
}
#[derive(Default)]
struct WaveStats {
    attempted: AtomicU64,
    completed: AtomicU64,
    gpu_forwards: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    active: AtomicUsize,
    peak_active: AtomicUsize,
    groups: AtomicU64,
    co_scheduled: AtomicU64,
    cohort_publications: AtomicU64,
    cohort_rows: AtomicU64,
    kernel_dispatches: AtomicU64,
    model_tokens: AtomicU64,
    committed_roots: AtomicU64,
    fused_batches: AtomicU64,
    fused_state_only_batches: AtomicU64,
    fused_state_only_rows: AtomicU64,
    fused_greedy_batches: AtomicU64,
    native_bands: [AtomicU64; 3],
    native_admission_refusals: AtomicU64,
    fused_rows: AtomicU64,
    native_projections: AtomicU64,
    native_sampler: AtomicU64,
    discarded_cohorts: AtomicU64,
    output_tokens: AtomicU64,
    parked: AtomicUsize,
}
impl WaveStats {
    fn admit_session(self: &Arc<Self>) -> SessionCharge {
        let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak_active.fetch_max(active, Ordering::AcqRel);
        SessionCharge(self.clone())
    }
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
    pub queue: QueueSnapshot,
    pub host: Option<(u64, u64, Option<u64>, u64, u64)>,
    pub active: usize,
    pub peak_active: usize,
    pub active_limit: usize,
    pub canonical_resources: Option<serde_json::Value>,
    pub hf_source: Option<serde_json::Value>,
    pub hf_resources: Option<serde_json::Value>,
    pub groups: u64,
    pub co_scheduled: u64,
    pub cohort_publications: u64,
    pub cohort_rows: u64,
    pub kernel_dispatches: u64,
    pub model_tokens: u64,
    pub committed_roots: u64,
    pub fused_batches: u64,
    pub fused_state_only_batches: u64,
    pub fused_state_only_rows: u64,
    pub fused_greedy_batches: u64,
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
    queue: PendingQueue<Scheduled>,
    host: ServingHost,
    next_id: AtomicU64,
    stats: Arc<WaveStats>,
    pub(super) max_output: usize,
    native_rows: usize,
    active_limit: usize,
    worker: Option<thread::JoinHandle<()>>,
    // Worker handle/consumer/closure inventory; released after the join
    // handle. Native stack/TLS/OS mapping is external unquantified headroom.
    _worker_charge: ServingCharge,
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
            host_mib,
            max_pending,
        } = profile;
        if !(256..=1_048_576).contains(&host_mib) {
            return Err(ServiceError::new(
                400,
                "invalid_host_budget",
                "host-mib must be 256..=1048576",
            ));
        }
        if max_pending == Some(0) {
            return Err(ServiceError::new(
                400,
                "invalid_config",
                "max-pending-requests must be a positive count or none",
            ));
        }
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
        // The host ledger exists before any source is parsed so the
        // process-global tokenizer bootstrap, the retained inventory and the
        // cold constructor are admitted ahead of the work they price.
        let host = ServingHost::new(host_mib)?;
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
            // One process-global tokenizer owner, admitted before any source
            // parse: ByteLevel statics, retained inventory, cold constructor.
            let tokenizer_admission = host.transfer_admission(&alias, "tokenizer-source")?;
            let bootstrap = zerogravity::tokenize_admission::approve_arbiter(
                &tokenizer_admission,
                "bootstrap",
            )?;
            let inventory_charge =
                bootstrap.retain_arbiter(&tokenizer_admission, "retained-inventory")?;
            // Approved source identity (12,807,982 B + SHA-256), the bounded
            // BPE-cache serving profile and the census of the exact bytes.
            // Price the capped held copy before allocating it; the no-follow
            // reader enforces extent/EOF/stamp and the parser checks SHA-256
            // on these same bytes before any JSON is parsed.
            let tokenizer_copy = host.charge(
                &alias,
                "tokenizer-source-copy",
                AllocationClass::Scratch,
                zerogravity::tokenize::APPROVED_TOKENIZER_BYTES,
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
            // Release the held bytes before their source-copy charge, then
            // validate the parsed inventory against the admitted plan.
            drop(tokenizer_bytes);
            drop(tokenizer_copy);
            bootstrap.verify(&tokenizer)?;
            let source = LoadedQwenModel::open(&path, &parent.join("config.json"))
                .map_err(ServiceError::inference)?;
            let hidden_width = usize::try_from(source.geometry().hidden_size).map_err(|_| {
                ServiceError::new(400, "source_hidden", "invalid Qwen source hidden width")
            })?;
            let vocabulary_size = source.geometry().vocab_size;
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
            let max_token_bytes = tokenizer
                .max_token_bytes()
                .map_err(ServiceError::inference)?;
            let shared = Arc::new(GpuModel {
                engine: ModelEngine::Gguf(resident),
                tokenizer: TextTokenizer::Shared(Arc::new(tokenizer)),
                group: unique.len(),
                hidden_width,
                vocabulary_size,
                native_rows,
                max_token_bytes,
                _inventory: Some(inventory_charge),
                readers: Arc::new(AtomicUsize::new(0)),
            });
            unique.insert(path, Arc::clone(&shared));
            models.insert(alias, shared);
        }
        Self::start(models, active, output, native_rows, host, max_pending)
    }
    fn start(
        models: BTreeMap<String, Arc<GpuModel>>,
        active: usize,
        output: usize,
        native_rows: usize,
        host: ServingHost,
        max_pending: Option<usize>,
    ) -> Result<Self, ServiceError> {
        let models = Arc::new(models);
        let queue = PendingQueue::new(host.clone(), max_pending)?;
        let sources = Arc::clone(&models);
        let stats = Arc::new(WaveStats::default());
        let worker_stats = Arc::clone(&stats);
        // Charge the resident worker's application-owned handle/consumer/
        // closure inventory and its three bounded container rows before the
        // consumer handle, join handle or thread exists; the router retains it
        // until join. The builder keeps std's default stack=None.
        let container_rows = pending::checked_sum(&[
            vecdeque_backing_bytes::<GpuActive>(active)?,
            vecdeque_backing_bytes::<GpuActive>(active)?,
            vecdeque_backing_bytes::<GpuActive>(active)?,
        ])?;
        let worker_charge = workers_charge(
            &host,
            1,
            std::alloc::Layout::new::<GpuWorkerCaptures>(),
            Some(("zg-resident-gpu".len(), "zg-resident-gpu".len())),
            container_rows,
        )?;
        let receiver = match queue.worker() {
            Ok(receiver) => receiver,
            Err(error) => {
                queue.close();
                return Err(error);
            }
        };
        let builder = thread::Builder::new().name("zg-resident-gpu".into());
        let captures = GpuWorkerCaptures {
            queue: receiver,
            models: sources,
            active_limit: active,
            stats: worker_stats,
            native_rows,
        };
        let worker = builder.spawn(move || captures.run()).map_err(|error| {
            queue.close();
            ServiceError::new(500, "worker_spawn", error.to_string())
        })?;
        Ok(Self {
            models,
            queue,
            host,
            next_id: AtomicU64::new(1),
            stats,
            max_output: output,
            native_rows,
            active_limit: active,
            worker: Some(worker),
            _worker_charge: worker_charge,
        })
    }
    pub(super) fn metrics(&self) -> GpuMetrics {
        let queue = self.queue.snapshot();
        let host_snapshot = self.host.snapshot().unwrap_or((0, 0, 0, 0));
        let (limit, reserved, _remaining, serving) = host_snapshot;
        #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
        let hf_health = qwen35_cuda::health(&self.models, host_snapshot);
        #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
        let backend = hf_health
            .as_ref()
            .and_then(|(_, resources)| resources.get("backend_host_reserved_bytes"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        #[cfg(not(all(feature = "qwen35-cuda-qualification", target_os = "linux")))]
        let backend = 0;
        #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
        let host = if qwen35_cuda::models_present(&self.models) && hf_health.is_none() {
            None
        } else {
            Some({
                let model = reserved
                    .checked_sub(backend)
                    .and_then(|bytes| bytes.checked_sub(serving));
                (limit, reserved, model, backend, serving)
            })
        };
        #[cfg(not(all(feature = "qwen35-cuda-qualification", target_os = "linux")))]
        let host = Some({
            let model = reserved.checked_sub(serving);
            (limit, reserved, model, backend, serving)
        });
        GpuMetrics {
            attempted: self.stats.attempted.load(Ordering::Relaxed),
            completed: self.stats.completed.load(Ordering::Relaxed),
            gpu_forwards: self.stats.gpu_forwards.load(Ordering::Relaxed),
            failed: self.stats.failed.load(Ordering::Relaxed),
            cancelled: self.stats.cancelled.load(Ordering::Relaxed),
            queue,
            host,
            active: self.stats.active.load(Ordering::Acquire),
            peak_active: self.stats.peak_active.load(Ordering::Acquire),
            active_limit: self.active_limit,
            #[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
            canonical_resources: fixed_prompt::resources(&self.models, host_snapshot),
            #[cfg(not(all(feature = "qwen35-cuda-serving", target_os = "linux")))]
            canonical_resources: None,
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            hf_source: hf_health.as_ref().map(|h| h.0.clone()),
            #[cfg(not(all(feature = "qwen35-cuda-qualification", target_os = "linux")))]
            hf_source: None,
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            hf_resources: hf_health.map(|h| h.1),
            #[cfg(not(all(feature = "qwen35-cuda-qualification", target_os = "linux")))]
            hf_resources: None,
            groups: self.stats.groups.load(Ordering::Relaxed),
            co_scheduled: self.stats.co_scheduled.load(Ordering::Relaxed),
            cohort_publications: self.stats.cohort_publications.load(Ordering::Relaxed),
            cohort_rows: self.stats.cohort_rows.load(Ordering::Relaxed),
            kernel_dispatches: self.stats.kernel_dispatches.load(Ordering::Relaxed),
            model_tokens: self.stats.model_tokens.load(Ordering::Relaxed),
            committed_roots: self.stats.committed_roots.load(Ordering::Relaxed),
            fused_batches: self.stats.fused_batches.load(Ordering::Relaxed),
            fused_state_only_batches: self.stats.fused_state_only_batches.load(Ordering::Relaxed),
            fused_state_only_rows: self.stats.fused_state_only_rows.load(Ordering::Relaxed),
            fused_greedy_batches: self.stats.fused_greedy_batches.load(Ordering::Relaxed),
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
    pub(super) fn charge_ingress(
        &self,
        session: &str,
        bytes: u64,
    ) -> Result<ServingCharge, ServiceError> {
        self.host
            .charge("http-ingress", session, AllocationClass::Scratch, bytes)
    }
    #[allow(dead_code)]
    pub(super) fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
        self.request_with(request, None)
    }
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
                ServiceError::new(404, "model_not_found", "unregistered GPU model")
            })?;
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
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if id == u64::MAX {
            return Err(ServiceError::new(
                503,
                "id_exhausted",
                "request IDs exhausted",
            ));
        }
        let ticket = self.queue.reserve()?;
        let prompt_bound = model
            .tokenizer
            .prompt_token_bound(request.input.len(), request.chat);
        let prompt_scratch = model
            .tokenizer
            .admitted_prompt_scratch_bound(request.input.len(), request.chat)
            .map_err(|error| ServiceError::new(400, "size_overflow", error.to_string()))?;
        let prompt_bytes = pending::checked_sum(&[
            pending::vec_bytes::<u32>(prompt_bound)?,
            prompt_scratch,
            // The presenter decodes the whole committed prefix, so a full
            // `max_tokens` prefix's adapter scratch stays live at each commit.
            u64::try_from(model.tokenizer.decode_scratch_bound(request.max_tokens)).map_err(
                |_| ServiceError::new(400, "size_overflow", "decode scratch bound overflow"),
            )?,
        ])?;
        let prompt_charge = self.host.charge(
            &request.model,
            "prompt-tokens",
            AllocationClass::RecurrentState,
            prompt_bytes,
        )?;
        // Canonical/Qwen CUDA validation runs only after count/host admission.
        #[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
        model.engine.validate_request(&model.tokenizer, &request)?;
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
        let model_alias = request.model.clone();
        let token_metrics = request.token_metrics.clone();
        let transport = Arc::new(self.host.charge(
            &model_alias,
            "event-transport",
            AllocationClass::Output,
            output_transport_bound(request.max_tokens, model.max_token_bytes)?,
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
                request,
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
impl GpuRouter {
    pub(super) fn shutdown(&mut self) -> std::io::Result<()> {
        let failures = self.stats.failed.load(Ordering::Acquire);
        self.queue.close();
        if self
            .worker
            .take()
            .is_some_and(|worker| worker.join().is_err())
        {
            Err(std::io::Error::other("GPU serving worker panicked"))
        } else if self.stats.failed.load(Ordering::Acquire) != failures
            || self
                .models
                .values()
                .any(|model| model.readers.load(Ordering::Acquire) != 0)
        {
            // Original retirement records failures in WaveStats and retains
            // quarantine. Joining its worker cannot turn that into success.
            Err(std::io::Error::other("GPU serving retirement failed"))
        } else {
            Ok(())
        }
    }
}
impl Drop for GpuRouter {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
struct SourceReaders(Arc<AtomicUsize>);
impl SourceReaders {
    fn enter(readers: &Arc<AtomicUsize>) -> Self {
        readers.fetch_add(1, Ordering::AcqRel);
        Self(Arc::clone(readers))
    }
}
impl Drop for SourceReaders {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
struct GpuActive {
    engine: RequestEngine,
    group: usize,
    hidden_width: usize,
    cancelled: Arc<AtomicBool>,
    errors: mpsc::SyncSender<ResponseEvent>,
    output: GeneratedText,
    prompt: Vec<u32>,
    cursor: usize,
    device_token: Option<ResidentToken>,
    token_commit_observed_at: Option<Instant>,
    token_gpu_observation: Option<GpuTokenObservation>,
    native_rows: usize,
    packet_presented: bool,
    blocked_since: Option<Instant>,
    _charge: SessionCharge,
    _prompt_charge: Option<ServingCharge>,
    _transport: Option<Arc<ServingCharge>>,
    // Dropped after the engine/session so source retirement cannot race a live
    // resident/session reference.
    _readers: SourceReaders,
    // Keeps the cancel atomic's charged control alive with the flag.
    _cancel: super::pending::CancelToken,
}
impl GpuActive {
    fn new(
        job: Scheduled,
        model: Arc<GpuModel>,
        stats: Arc<WaveStats>,
    ) -> Result<Self, ServiceError> {
        let Scheduled {
            payload,
            metrics,
            events,
            cancelled,
            cancel,
            prompt_charge,
            transport,
        } = job;
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
            || prompt.len() > GPU_CAPACITY
            || prompt
                .len()
                .checked_add(request.max_tokens - 1)
                .is_none_or(|total| total > GPU_CAPACITY)
        {
            return Err(ServiceError::new(
                400,
                "context_overflow",
                "resident GPU profile admits at most 512 consumed tokens",
            ));
        }
        let output = GeneratedText::new_owned(
            model.tokenizer.clone(),
            events.clone(),
            request.stream,
            request.max_tokens,
            prompt.len(),
        )
        .observed_charged(metrics.metrics, model.vocabulary_size, metrics._charges);
        // ResidentSession allocates independent, physical source-bound GPU
        // state. Neither its tensor source nor its StateRoot is CPU fallback.
        let readers = SourceReaders::enter(&model.readers);
        let engine = model.engine.session(&cancelled)?;
        drop(lease);
        let charge = stats.admit_session();
        Ok(Self {
            _charge: charge,
            _prompt_charge: prompt_charge,
            _transport: transport,
            _readers: readers,
            _cancel: cancel,
            engine,
            group: model.group,
            hidden_width: model.hidden_width,
            cancelled,
            errors: events,
            output,
            prompt,
            cursor: 0,
            device_token: None,
            token_commit_observed_at: None,
            token_gpu_observation: None,
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
            frontier: self.engine.binding().frontier() as usize,
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
        if self.engine.is_original_hf() {
            return remaining;
        }
        if self.native_rows != 0 {
            // Native cohorts and unmatched B1 rows consume the same truthful
            // one-token wave credit, then release the worker to other rows.
            remaining.min(1)
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
            let token_id = self.engine.presentation_id(self.device_token.as_ref())?;
            let before = self.output.generated_token_count();
            let committed = self.token_commit_observed_at.ok_or_else(|| {
                ServiceError::new(
                    500,
                    "missing_commit_observation",
                    "resident packet has no publication observation",
                )
            })?;
            let status =
                self.output
                    .present_observed(token_id, committed, self.token_gpu_observation)?;
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
    fn install(
        &mut self,
        read: InferenceReadback,
        commit_observed_at: Instant,
        native_cohort: Option<NativeCohortObservation>,
    ) -> Result<(), ServiceError> {
        let token = read.device_token.ok_or_else(|| {
            ServiceError::new(
                500,
                "missing_device_token",
                "GPU completion has no original greedy packet",
            )
        })?;
        // install follows successful original publication; this is the exact
        // packet-producing committed binding, including a B1 retry.
        self.token_gpu_observation = Some(GpuTokenObservation {
            producer_frontier: self.engine.binding().frontier(),
            producer_root: None,
            native_cohort,
        });
        self.device_token = Some(token);
        self.token_commit_observed_at = Some(commit_observed_at);
        self.packet_presented = false;
        self.blocked_since = None;
        Ok(())
    }
    /// Original B1 source execution, including all unmatched native-wave rows.
    fn step(&mut self, stats: &WaveStats) -> Result<MemberStep, ServiceError> {
        #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
        if self.engine.is_qwen35_cuda() {
            return self.step_qwen35_cuda(stats);
        }
        #[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
        if self.engine.is_canonical() {
            return self.step_canonical(stats);
        }
        if self.cursor < self.prompt.len() {
            let end = self.cursor + self.prefill_count();
            let read = if end < self.prompt.len() {
                block_on(
                    self.engine
                        .legacy_mut()
                        .expect("legacy step selected")
                        .prefill_without_logits(&self.prompt[self.cursor..end], false),
                )
            } else {
                block_on(
                    self.engine
                        .legacy_mut()
                        .expect("legacy step selected")
                        .forward_greedy(&self.prompt[self.cursor..end]),
                )
            }
            .map_err(ServiceError::inference)?;
            let committed = Instant::now();
            stats.single_completion(&read);
            if end == self.prompt.len() {
                self.install(read, committed, None)?;
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
        let read = block_on(
            self.engine
                .legacy_mut()
                .expect("legacy step selected")
                .forward_device_token(token),
        )
        .map_err(ServiceError::inference)?;
        let committed = Instant::now();
        stats.single_completion(&read);
        self.install(read, committed, None)?;
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

fn retire_if_drained(models: &BTreeMap<String, Arc<GpuModel>>, stats: &WaveStats) {
    #[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
    fixed_prompt::retire_sources(models, stats);
    #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
    qwen35_cuda::retire_sources(models, stats);
}
struct GpuWorkerCaptures {
    queue: WorkerQueue<Scheduled>,
    models: Arc<BTreeMap<String, Arc<GpuModel>>>,
    active_limit: usize,
    stats: Arc<WaveStats>,
    native_rows: usize,
}

impl GpuWorkerCaptures {
    /// Consume the exact admitted struct so the start closure captures only
    /// this priced value; the admitted `Layout::new::<GpuWorkerCaptures>()`
    /// cannot drift from the spawn capture.
    fn run(self) {
        worker(
            self.queue,
            self.models,
            self.active_limit,
            self.stats,
            self.native_rows,
        )
    }
}
fn worker(
    queue: WorkerQueue<Scheduled>,
    models: Arc<BTreeMap<String, Arc<GpuModel>>>,
    active_limit: usize,
    stats: Arc<WaveStats>,
    native_rows: usize,
) {
    // Three container rows are charged before spawn: active sessions, the
    // current eligible set, and the parked set. Swapping the drained active
    // queue with parked reuses both allocations instead of growing either.
    let mut active = VecDeque::<GpuActive>::with_capacity(active_limit);
    let mut eligible = VecDeque::<GpuActive>::with_capacity(active_limit);
    let mut parked = VecDeque::<GpuActive>::with_capacity(active_limit);
    let mut closing = false;
    let cancelled = |job: &Scheduled| job.cancel.is_cancelled();
    loop {
        // Reclaim cancelled queued payloads even when every active or parked
        // slot prevents promotion. The hint keeps idle rounds cheap.
        let _ = queue.sweep(cancelled);
        active.retain(|row| {
            if row.cancelled.load(Ordering::Acquire) {
                stats.cancelled.fetch_add(1, Ordering::Relaxed);
                false
            } else {
                true
            }
        });
        if !closing {
            while active.len() < active_limit {
                let next = if active.is_empty() {
                    match queue.pop_wait(Duration::from_millis(25), cancelled) {
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
                let Some(job) = next else { break };
                if job.cancelled.load(Ordering::Acquire) {
                    stats.cancelled.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let request = job
                    .payload
                    .as_ref()
                    .expect("scheduled payload")
                    .request
                    .model
                    .clone();
                let Some(model) = models.get(&request) else {
                    continue;
                };
                let errors = job.events.clone();
                match GpuActive::new(job, Arc::clone(model), Arc::clone(&stats)) {
                    Ok(session) => active.push_back(session),
                    Err(e) => {
                        if e.status == 499 && e.code == "cancelled" {
                            stats.cancelled.fetch_add(1, Ordering::Relaxed);
                        } else {
                            stats.failed.fetch_add(1, Ordering::Relaxed);
                        }
                        let _ = errors.try_send(ResponseEvent::Error(e));
                    }
                }
            }
        }
        if active.is_empty() {
            if closing {
                retire_if_drained(&models, &stats);
                return;
            }
            continue;
        }
        eligible.clear();
        parked.clear();
        for row in active.drain(..) {
            if let Some(row) = eligible_member(row, &mut parked, &stats) {
                eligible.push_back(row);
            }
        }
        std::mem::swap(&mut active, &mut parked);
        stats.parked.store(active.len(), Ordering::Relaxed);
        if eligible.is_empty() {
            // No locks/queue slot while waiting for bounded transport credit.
            if !active.is_empty() {
                thread::sleep(Duration::from_millis(2));
            }
            continue;
        }
        let ready = eligible.iter().map(GpuActive::ready).collect::<Vec<_>>();
        let bounds = Limits {
            max_members: if native_rows != 0 {
                native_rows
            } else {
                active_limit
            },
            max_tokens: if eligible.iter().all(|row| row.engine.is_qwen35_cuda()) {
                72 * native_rows.max(1)
            } else {
                64
            },
            max_scratch: 256 * 1024 * 1024,
        };
        let planned = if native_rows != 0 && eligible.iter().all(|row| row.engine.is_qwen35_cuda())
        {
            // Every admitted ready row gets one original fair quantum/round.
            // Actual native eligibility is checked separately for each member.
            gpu_batch::form_cached_with_prefill_quantum(&ready, bounds, 72)
        } else if native_rows != 0 {
            gpu_batch::form_native(&ready, bounds)
        } else if eligible.iter().all(|row| row.engine.is_original_hf()) {
            // Pure source preflight already sealed the complete prompt. Charge
            // its actual atomic length; two47-token requests need two64-token waves.
            let quantum = eligible
                .iter()
                .map(|row| row.prompt.len())
                .max()
                .expect("nonempty eligible source wave");
            gpu_batch::form_with_prefill_quantum(&ready, bounds, quantum)
        } else {
            gpu_batch::form(&ready, bounds)
        };
        let waves = match planned {
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
                    .map(|row| row.engine.binding())
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
            #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
            if native_rows == 4
                && wave.indices.iter().all(|&index| {
                    pending[index]
                        .as_ref()
                        .is_some_and(|row| row.engine.is_qwen35_cuda())
                })
            {
                let members = wave
                    .indices
                    .iter()
                    .map(|&index| pending[index].take().expect("unique original HF wave"))
                    .collect();
                qwen35_cuda::advance_cohort(members, &mut active, &stats);
                continue;
            }
            // Prefix and tail readouts share the original checked publisher.
            // The cohort owner partitions state-only and greedy barriers.
            let batch_ready = wave.indices.len() > 1
                && wave.indices.iter().all(|&i| {
                    pending[i].as_ref().is_some_and(|row| {
                        !row.cancelled.load(Ordering::Acquire)
                            && !row.engine.is_original_hf()
                            && row.cursor < row.prompt.len()
                            && (native_rows != 0 || row.cursor + 1 == row.prompt.len())
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
    fn shutdown_rejects_new_retirement_failures_after_join() {
        // Exercise original queue closure and a real worker join without any
        // registered model, device or inference capability.
        for (previous, fail) in [(7, false), (7, true), (u64::MAX, true)] {
            let host = ServingHost::isolated(256).unwrap();
            let queue = PendingQueue::<Scheduled>::new(host.clone(), Some(1)).unwrap();
            let stats = Arc::new(WaveStats::default());
            stats.failed.store(previous, Ordering::Release);
            let captures = (queue.worker().unwrap(), Arc::clone(&stats), fail);
            let run = move || {
                let (control, stats, fail) = captures;
                loop {
                    match control.pop_wait(Duration::from_millis(10), |_| false) {
                        Pop::Closed => {
                            if fail {
                                stats.failed.fetch_add(1, Ordering::Relaxed);
                            }
                            return;
                        }
                        Pop::Empty => {}
                        Pop::Item(_) => panic!("shutdown fixture admitted an unexpected request"),
                    }
                }
            };
            let name = "zg-shutdown-test";
            let charge = workers_charge(
                &host,
                1,
                std::alloc::Layout::for_value(&run),
                Some((name.len(), name.len())),
                0,
            )
            .unwrap();
            let worker = thread::Builder::new().name(name.into()).spawn(run).unwrap();
            let mut router = GpuRouter {
                models: Arc::new(BTreeMap::new()),
                queue,
                host: host.clone(),
                next_id: AtomicU64::new(1),
                stats,
                max_output: 1,
                native_rows: 0,
                active_limit: 1,
                worker: Some(worker),
                _worker_charge: charge,
            };
            assert_eq!(router.shutdown().is_err(), fail);
            assert!(router.worker.is_none());
            drop(router);
            assert_eq!(host.serving_bytes(), 0);
        }
    }

    #[test]
    fn admitted_session_high_water_survives_warmup_and_completed_drops() {
        let stats = Arc::new(WaveStats::default());
        let warmup = stats.admit_session();
        assert_eq!(stats.active.load(Ordering::Acquire), 1);
        drop(warmup);
        assert_eq!(stats.active.load(Ordering::Acquire), 0);
        assert_eq!(stats.peak_active.load(Ordering::Acquire), 1);
        let mut sessions = (0..8).map(|_| stats.admit_session()).collect::<Vec<_>>();
        assert_eq!(stats.active.load(Ordering::Acquire), 8);
        assert_eq!(stats.peak_active.load(Ordering::Acquire), 8);
        sessions.truncate(2);
        assert_eq!(stats.active.load(Ordering::Acquire), 2);
        let next = stats.admit_session();
        assert_eq!(stats.peak_active.load(Ordering::Acquire), 8);
        drop(next);
        drop(sessions);
        assert_eq!(stats.active.load(Ordering::Acquire), 0);
        assert_eq!(stats.peak_active.load(Ordering::Acquire), 8);
    }

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
