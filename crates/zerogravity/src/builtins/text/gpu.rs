//! Ordinary resident GGUF composition; no captured recipes or scheduler.
use super::{arc_owner_bytes, local};
use crate::gpu::{
    InferenceReadback, PreparedResidentCohort, ResidentModel, ResidentSession, ResidentToken,
    resident_batch::ResidentBatchRows,
};
use crate::{devices::GpuBackend, identity::SourceIdentity, resource_arbiter::AllocationClass};
use futures_lite::future::block_on;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};
use zerogravity_engine::model_instance::{PreparedModel, TextTokenizer, execution::*};
use zerogravity_engine::router::{
    GpuTokenObservation, MAX_ACTIVE_PER_WORKER, Router, ServiceError,
    pending::ServingHost,
    worker::{Active, SchedulerSnapshot},
};
const GPU_CAPACITY: usize = 512;
mod load;
pub use load::load_gpu_router;
pub(crate) use load::load_gpu_router_declared;
#[cfg(any(
    all(test, feature = "cuda", feature = "cpu"),
    all(feature = "qwen35-cuda-qualification", target_os = "linux")
))]
pub(crate) use load::preflight_cuda_toolkit;
mod cohort;
pub(crate) mod report;
#[derive(Clone, Copy, Debug)]
pub struct GpuProfile {
    pub workers: usize,
    pub active: usize,
    pub context: usize,
    pub output: usize,
    pub provider_mib: usize,
    pub device_mib: usize,
    pub cohort_width: usize,
    pub host_mib: usize,
    pub max_pending: Option<usize>,
}

struct Model {
    resident: Mutex<Option<Arc<ResidentModel>>>,
    source: SourceIdentity,
    execution: zerogravity_core::provider::ExecutionDescription,
    source_host_reserved_bytes: Option<u64>,
    cache_snapshot: Option<String>,
    tokenizer: Arc<dyn TextTokenizer>,
    group: usize,
    hidden_width: usize,
    vocabulary_size: u64,
    cohort_width: usize,
    max_token_bytes: usize,
    readers: Arc<AtomicUsize>,
    _metadata: Arc<zerogravity_engine::router::pending::ServingCharge>,
}
impl Model {
    fn resident(&self) -> Result<Arc<ResidentModel>, ServiceError> {
        self.resident
            .lock()
            .map_err(|_| ServiceError::new(500, "source_owner", "resident model lock poisoned"))?
            .as_ref()
            .cloned()
            .ok_or_else(|| ServiceError::new(503, "model_retired", "resident model retired"))
    }
}
impl PreparedModel for Model {
    fn source(&self) -> SourceIdentity {
        self.source
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        self.tokenizer.clone()
    }
    fn max_context(&self) -> usize {
        GPU_CAPACITY
    }
    fn max_token_bytes(&self) -> usize {
        self.max_token_bytes
    }
    fn vocabulary_size(&self) -> u64 {
        self.vocabulary_size
    }
    fn describe(&self) -> crate::Result<zerogravity_engine::diagnostics::ModelDetails> {
        use zerogravity_engine::diagnostics::{ModelDetails, ModelMemory};
        let resident = self.resident().map_err(ServiceError::into_core)?;
        Ok(ModelDetails {
            artifact: Some(local::artifact_description(self.cache_snapshot.clone())),
            execution: Some(self.execution.clone()),
            state_policy: Some(zerogravity_core::provider::ResidentStatePolicy::Contiguous),
            prefix_sharing: Some(false),
            source_host_reserved_bytes: self.source_host_reserved_bytes,
            memory: Some(ModelMemory::Resident(resident.memory_snapshot()?)),
        })
    }
}
pub(crate) fn adapter_metadata<T>(
    host: &ServingHost,
    alias: &str,
    role: &str,
) -> Result<Arc<zerogravity_engine::router::pending::ServingCharge>, ServiceError> {
    let charge = host.charge(
        alias,
        role,
        AllocationClass::ImmutableWeights,
        zerogravity_engine::router::pending::checked_sum(&[
            arc_owner_bytes::<T>()?,
            arc_owner_bytes::<AtomicUsize>()?,
            arc_owner_bytes::<zerogravity_engine::router::pending::ServingCharge>()?,
        ])?,
    )?;
    Ok(Arc::new(charge))
}
pub(crate) struct SourceReaders {
    readers: Arc<AtomicUsize>,
    // The counter's allocation credit follows the last reader, not the model.
    _metadata: Arc<zerogravity_engine::router::pending::ServingCharge>,
}
impl SourceReaders {
    pub(crate) fn enter(
        readers: &Arc<AtomicUsize>,
        metadata: &Arc<zerogravity_engine::router::pending::ServingCharge>,
    ) -> Self {
        readers.fetch_add(1, Ordering::AcqRel);
        Self {
            readers: readers.clone(),
            _metadata: metadata.clone(),
        }
    }
}
impl Drop for SourceReaders {
    fn drop(&mut self) {
        self.readers.fetch_sub(1, Ordering::AcqRel);
    }
}

struct NativeText;
struct State {
    session: ResidentSession,
    device_token: Option<ResidentToken>,
}
type Row = Active<NativeText>;
fn policy(active: usize, cohort_width: usize) -> WorkPolicy {
    WorkPolicy::Waves(WavePolicy {
        max_members: if cohort_width != 0 {
            cohort_width
        } else {
            active
        },
        max_tokens: 64,
        max_scratch: 256 * 1024 * 1024,
        fairness: Fairness {
            round_tokens: 64 + active,
            prefill_age_rounds: 2,
        },
        prefill_quantum: 32,
        unequal_prefill: false,
        unequal_decode: cohort_width != 0,
        cohort_width,
        cohorts: if cohort_width == 0 {
            Cohorts::PromptTails
        } else {
            Cohorts::Fused {
                widths: &[8, 4, 2, 1],
                split_prefix: true,
                retry: RetryOrder::Regroup,
            }
        },
    })
}
fn start(
    models: BTreeMap<String, Arc<Model>>,
    active: usize,
    output: usize,
    cohort_width: usize,
    host: ServingHost,
    max_pending: Option<usize>,
) -> Result<Router, ServiceError> {
    Router::with_executor::<NativeText>(
        models,
        host,
        WorkerConfig {
            workers: 1,
            active,
            context: GPU_CAPACITY,
            output,
            max_pending,
            policy: policy(active, cohort_width),
        },
    )
}
fn retire_model(model: &Arc<Model>) -> Result<(), ServiceError> {
    if model.readers.load(Ordering::Acquire) != 0 {
        return Err(ServiceError::new(
            500,
            "source_readers",
            "resident model still has active readers",
        ));
    }
    let resident = model
        .resident
        .lock()
        .map_err(|_| ServiceError::new(500, "source_owner", "resident model lock poisoned"))?
        .take();
    if let Some(resident) = resident {
        block_on(resident.shutdown_tensor_source()).map_err(ServiceError::inference)?;
    }
    Ok(())
}
impl Executor for NativeText {
    type Model = Arc<Model>;
    type State = State;
    type Lease = SourceReaders;
    type Ready = PreparedResidentCohort;
    fn open(
        model: &Self::Model,
        _: &str,
        cancelled: &Arc<AtomicBool>,
        _: &ServingHost,
    ) -> Result<(State, SourceReaders), ServiceError> {
        if cancelled.load(Ordering::Acquire) {
            return Err(ServiceError::new(499, "cancelled", "request cancelled"));
        }
        let readers = SourceReaders::enter(&model.readers, &model._metadata);
        let session = block_on(model.resident()?.session()).map_err(ServiceError::inference)?;
        Ok((
            State {
                session,
                device_token: None,
            },
            readers,
        ))
    }
    fn prefill(model: &Self::Model) -> Prefill {
        Prefill::Chunks(match model.cohort_width {
            0 => 32,
            2 | 4 | 8 => 4,
            _ => 1,
        })
    }
    fn group(model: &Self::Model) -> usize {
        model.group
    }
    fn frontier(state: &State) -> usize {
        state.session.binding().frontier() as usize
    }
    fn selected(state: &mut State) -> Result<u32, ServiceError> {
        state.device_token.as_ref().map(|t| t.id).ok_or_else(|| {
            ServiceError::new(
                500,
                "missing_device_token",
                "computed committed packet absent",
            )
        })
    }
    fn has_feedback(state: &State) -> bool {
        state.device_token.is_some()
    }
    fn cohort_eligible(row: &Row) -> bool {
        matches!(row.model.cohort_width, 2 | 4 | 8)
            && row.cursor < row.prompt.len()
            && row.state.device_token.is_none()
    }
    fn compatible(first: &Row, row: &Row) -> bool {
        Arc::ptr_eq(
            first.state.session.resident_model(),
            row.state.session.resident_model(),
        ) && first.model.hidden_width == row.model.hidden_width
            && first.cursor < first.prompt.len()
            && row.cursor < row.prompt.len()
            && first.prefill_count() == row.prefill_count()
    }
    fn single(row: &mut Row) -> Result<SingleCompleted, ServiceError> {
        let (read, consumed, readout) = if row.cursor < row.prompt.len() {
            let end = row.cursor + row.prefill_count();
            let read = if end < row.prompt.len() {
                block_on(
                    row.state
                        .session
                        .prefill_without_logits(&row.prompt[row.cursor..end], false),
                )
            } else {
                block_on(
                    row.state
                        .session
                        .forward_greedy(&row.prompt[row.cursor..end]),
                )
            }
            .map_err(ServiceError::inference)?;
            (read, end - row.cursor, end == row.prompt.len())
        } else {
            let token = row.state.device_token.as_ref().ok_or_else(|| {
                ServiceError::new(
                    500,
                    "missing_device_token",
                    "resident GPU feedback is missing",
                )
            })?;
            let read = block_on(row.state.session.forward_device_token(token))
                .map_err(ServiceError::inference)?;
            (read, 1, true)
        };
        let committed = Instant::now();
        let completion = Completion {
            forwards: 1,
            dispatches: read.dispatches as u64,
            model_tokens: (read.state_only_steps + read.logit_steps) as u64,
            committed_roots: 1,
            ..Completion::default()
        };
        let published = if readout {
            install(row, read, consumed, committed)
        } else {
            Ok(Published {
                consumed,
                readout: false,
                observed_at: committed,
                observation: None,
            })
        };
        Ok(SingleCompleted {
            completion,
            published,
        })
    }
    fn validate_wave(rows: &[&Row]) -> Result<(), ServiceError> {
        cohort::validate_wave(rows)
    }
    fn validate_cohort(rows: &[Row], work: CohortWork) -> Result<(), ServiceError> {
        cohort::validate(rows, work)
    }
    fn prepare(
        rows: &mut [Row],
        work: CohortWork,
    ) -> Result<Preparation<Self::Ready>, ServiceError> {
        cohort::prepare(rows, work)
    }
    fn commit(
        ready: Self::Ready,
        rows: &mut [Row],
        work: CohortWork,
    ) -> Result<Publication, ServiceError> {
        cohort::commit(ready, rows, work)
    }
    fn retirement_pending(models: &BTreeMap<String, Arc<Model>>) -> bool {
        models.values().any(|m| {
            m.readers.load(Ordering::Acquire) != 0
                || m.resident
                    .lock()
                    .map_or(true, |resident| resident.is_some())
        })
    }
    fn selective_retirement() -> Option<ModelRetirement<Self::Model>> {
        Some(retire_model)
    }
    fn retire(models: &BTreeMap<String, Arc<Model>>) -> Result<(), ServiceError> {
        for model in models.values() {
            retire_model(model)?;
        }
        Ok(())
    }
    fn diagnostics(
        _: &BTreeMap<String, Arc<Model>>,
        snapshot: &SchedulerSnapshot,
    ) -> Result<serde_json::Value, ServiceError> {
        Ok(report::render(
            snapshot,
            report::Resources::ordinary(snapshot),
        ))
    }
    fn scratch_bytes(active: usize) -> Result<u64, ServiceError> {
        zerogravity_engine::router::pending::vec_bytes::<(
            crate::session_root::RootBinding,
            usize,
            usize,
            usize,
            usize,
        )>(
            active.checked_mul(8).ok_or_else(|| {
                ServiceError::new(400, "size_overflow", "adapter scratch overflow")
            })?,
        )
    }
}
fn install(
    row: &mut Row,
    read: InferenceReadback,
    consumed: usize,
    committed: Instant,
) -> Result<Published, ServiceError> {
    let token = read.device_token.ok_or_else(|| {
        ServiceError::new(
            500,
            "missing_device_token",
            "GPU completion has no original greedy packet",
        )
    })?;
    let observation = GpuTokenObservation {
        producer_frontier: row.state.session.binding().frontier(),
        producer_root: None,
        fused_cohort: None,
    };
    row.state.device_token = Some(token);
    Ok(Published {
        consumed,
        readout: true,
        observed_at: committed,
        observation: Some(observation),
    })
}
#[cfg(test)]
thread_local! {
    static AFTER_COHORT_READY: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(all(test, feature = "cuda", feature = "cpu"))]
mod qualification;
