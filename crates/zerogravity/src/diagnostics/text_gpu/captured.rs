//! Finite captured execution only; queues, grouping and publication accounting
//! belong to the shared engine. Device feedback stays in the original sessions.
use super::*;
pub(super) use crate::builtins::text::gpu::preflight_cuda_toolkit;
use crate::builtins::text::gpu::{SourceReaders, adapter_metadata, report};
use crate::gpu::{cuda::native::TextPendingCohort, resident_batch::ResidentBatchRows};
pub(super) use futures_lite::future::block_on;
use std::time::Instant;
pub(super) use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use zerogravity_engine::model_instance::{PreparedModel, TextTokenizer, execution::*};
pub(super) use zerogravity_engine::router::pending::ServingHost;
use zerogravity_engine::router::{
    GpuRootObservation, GpuTokenObservation, Request,
    pending::ServingCharge,
    worker::{Active, SchedulerSnapshot},
};
mod cohort;

pub(super) enum ModelEngine {
    FixedPrompt(Arc<crate::qwen35_cuda::fixed_prompt::Model>),
    Qwen35Cuda(Arc<crate::qwen35_cuda::Model>),
}
pub(super) struct CapturedModel {
    pub engine: ModelEngine,
    pub tokenizer: Arc<dyn TextTokenizer>,
    pub group: usize,
    pub hidden_width: usize,
    pub vocabulary_size: u64,
    pub cohort_width: usize,
    pub max_token_bytes: usize,
    pub readers: Arc<AtomicUsize>,
    pub _metadata: Arc<ServingCharge>,
}
impl PreparedModel for CapturedModel {
    fn source(&self) -> crate::identity::SourceIdentity {
        match &self.engine {
            ModelEngine::FixedPrompt(m) => m.source_identity(),
            ModelEngine::Qwen35Cuda(m) => m.source_identity(),
        }
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        self.tokenizer.clone()
    }
    fn max_context(&self) -> usize {
        512
    }
    fn max_token_bytes(&self) -> usize {
        self.max_token_bytes
    }
    fn vocabulary_size(&self) -> u64 {
        self.vocabulary_size
    }
    fn validate_request(&self, request: &Request) -> Result<(), ServiceError> {
        let ids = self
            .tokenizer
            .prompt_tokens(&request.input, request.chat)
            .map_err(ServiceError::inference)?;
        match &self.engine {
            ModelEngine::FixedPrompt(m) => m.validate_prompt(&ids, request.max_tokens),
            ModelEngine::Qwen35Cuda(m) => m.validate_prompt(&ids, request.max_tokens),
        }
        .map_err(|e| ServiceError::new(400, "unsupported_source_request", e.to_string()))
    }
}
pub(super) fn metadata_charge(
    host: &ServingHost,
    alias: &str,
) -> Result<Arc<ServingCharge>, ServiceError> {
    adapter_metadata::<CapturedModel>(host, alias, "captured-adapter")
}

enum State {
    Fixed {
        model: Arc<crate::qwen35_cuda::fixed_prompt::Model>,
        session: crate::qwen35_cuda::fixed_prompt::Session,
        selected: Option<u32>,
    },
    Qwen {
        model: Arc<crate::qwen35_cuda::Model>,
        session: crate::qwen35_cuda::Session,
        selected: Option<u32>,
    },
}
impl State {
    fn binding(&self) -> crate::session_root::RootBinding {
        match self {
            Self::Fixed { session, .. } => session.binding(),
            Self::Qwen { session, .. } => session.binding(),
        }
    }
    fn selected(&self) -> Option<u32> {
        match self {
            Self::Fixed { selected, .. } | Self::Qwen { selected, .. } => *selected,
        }
    }
}
struct CapturedText;
type Row = Active<CapturedText>;
pub(super) fn start(
    models: BTreeMap<String, Arc<CapturedModel>>,
    active: usize,
    output: usize,
    cohort_width: usize,
    host: ServingHost,
    max_pending: Option<usize>,
) -> Result<Router, ServiceError> {
    let qwen = qwen35_cuda::models_present(&models);
    let policy = WorkPolicy::Waves(WavePolicy {
        max_members: if cohort_width != 0 {
            cohort_width
        } else {
            active
        },
        max_tokens: if qwen { 72 * cohort_width.max(1) } else { 64 },
        max_scratch: 256 * 1024 * 1024,
        fairness: Fairness {
            round_tokens: if qwen {
                72 * cohort_width.max(1) + active
            } else {
                64 + active
            },
            prefill_age_rounds: 2,
        },
        prefill_quantum: if qwen { 72 } else { 47 },
        unequal_decode: cohort_width != 0,
        unequal_prefill: false,
        cohort_width,
        cohorts: if cohort_width == 4 && qwen {
            Cohorts::Fused {
                widths: &[4, 1],
                split_prefix: false,
                retry: RetryOrder::NextRound,
            }
        } else {
            Cohorts::None
        },
    });
    Router::with_executor::<CapturedText>(
        models,
        host,
        WorkerConfig {
            workers: 1,
            active,
            context: 512,
            output,
            max_pending,
            policy,
        },
    )
}
impl Executor for CapturedText {
    type Model = Arc<CapturedModel>;
    type State = State;
    type Lease = SourceReaders;
    type Ready = TextPendingCohort;
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
        let state = match &model.engine {
            ModelEngine::FixedPrompt(m) => State::Fixed {
                model: m.clone(),
                session: block_on(m.session(Some(cancelled)))
                    .map_err(|e| fixed_prompt::inference_error(e, m, cancelled))?,
                selected: None,
            },
            ModelEngine::Qwen35Cuda(m) => State::Qwen {
                model: m.clone(),
                session: block_on(m.session(Some(cancelled)))
                    .map_err(|e| qwen35_cuda::inference_error(e, m, cancelled))?,
                selected: None,
            },
        };
        Ok((state, readers))
    }
    fn prefill(_: &Self::Model) -> Prefill {
        Prefill::Atomic
    }
    fn group(model: &Self::Model) -> usize {
        model.group
    }
    fn frontier(state: &State) -> usize {
        state.binding().frontier() as usize
    }
    fn selected(state: &mut State) -> Result<u32, ServiceError> {
        state.selected().ok_or_else(|| {
            ServiceError::new(
                500,
                "missing_device_token",
                "computed committed packet absent",
            )
        })
    }
    fn has_feedback(state: &State) -> bool {
        state.selected().is_some()
    }
    fn single(row: &mut Row) -> Result<SingleCompleted, ServiceError> {
        let prefill = row.cursor == 0;
        let (root, dispatches, consumed, observe_root) = match &mut row.state {
            State::Fixed {
                model,
                session,
                selected,
            } => {
                let result = if prefill {
                    block_on(session.infer_prompt(&row.prompt, Some(&row.cancelled)))
                } else {
                    block_on(session.infer_next(Some(&row.cancelled)))
                };
                let r =
                    result.map_err(|e| fixed_prompt::inference_error(e, model, &row.cancelled))?;
                if r.root != session.binding()
                    || r.executed_nodes != 255
                    || r.native_launches == 0
                    || r.consumed_tokens != if prefill { 47 } else { 1 }
                {
                    return Err(ServiceError::new(
                        500,
                        "canonical_receipt",
                        "incomplete original source traversal/commit",
                    ));
                }
                *selected = Some(r.selected_id);
                (r.root, r.dispatches, r.consumed_tokens, false)
            }
            State::Qwen {
                model,
                session,
                selected,
            } => {
                // Drive submitted work to original completion/quarantine even
                // when HTTP cancellation arrives; never race/drop this future.
                let result = if prefill {
                    block_on(session.infer_prompt(&row.prompt, Some(&row.cancelled)))
                } else {
                    block_on(session.infer_next(Some(&row.cancelled)))
                };
                let r =
                    result.map_err(|e| qwen35_cuda::inference_error(e, model, &row.cancelled))?;
                if r.root != session.binding()
                    || r.executed_nodes != 255
                    || r.native_launches == 0
                    || r.consumed_tokens != if prefill { row.prompt.len() } else { 1 }
                {
                    return Err(ServiceError::new(
                        500,
                        "qwen35_cuda_receipt",
                        "incomplete actual source traversal/commit",
                    ));
                }
                *selected = Some(r.selected_id);
                (r.root, r.dispatches, r.consumed_tokens, true)
            }
        };
        Ok(SingleCompleted {
            completion: Completion {
                forwards: 1,
                dispatches: dispatches as u64,
                model_tokens: consumed as u64,
                committed_roots: 1,
                ..Completion::default()
            },
            published: Ok(Published {
                consumed,
                readout: true,
                observed_at: Instant::now(),
                observation: Some(GpuTokenObservation {
                    producer_frontier: root.frontier(),
                    producer_root: observe_root.then(|| GpuRootObservation::committed(root)),
                    fused_cohort: None,
                }),
            }),
        })
    }
    fn validate_wave(rows: &[&Row]) -> Result<(), ServiceError> {
        let Some(first) = rows.first() else {
            return Ok(());
        };
        let roots = rows.iter().map(|r| r.state.binding()).collect::<Vec<_>>();
        let cohort = ResidentBatchRows::new(&roots, first.model.hidden_width)
            .map_err(ServiceError::inference)?;
        if rows
            .iter()
            .any(|r| r.model.hidden_width != cohort.row_elements())
        {
            return Err(ServiceError::inference(crate::Error::State(
                "GPU model wave hidden widths differ".into(),
            )));
        }
        cohort
            .require_current(&roots)
            .map_err(ServiceError::inference)
    }
    fn cohort_eligible(row: &Row) -> bool {
        cohort::eligible(row)
    }
    fn compatible(first: &Row, row: &Row) -> bool {
        cohort::compatible(first, row)
    }
    fn validate_cohort(rows: &[Row], _: CohortWork) -> Result<(), ServiceError> {
        if rows.len() != 4 || rows.iter().any(|r| !cohort::compatible(&rows[0], r)) {
            Err(ServiceError::new(
                500,
                "bf16_cohort_membership",
                "source B4 eligibility changed",
            ))
        } else {
            Ok(())
        }
    }
    fn prepare(
        rows: &mut [Row],
        _: CohortWork,
    ) -> Result<Preparation<TextPendingCohort>, ServiceError> {
        cohort::prepare(rows)
    }
    fn commit(
        ready: TextPendingCohort,
        rows: &mut [Row],
        _: CohortWork,
    ) -> Result<Publication, ServiceError> {
        cohort::commit(ready, rows)
    }
    fn retirement_pending(models: &BTreeMap<String, Arc<CapturedModel>>) -> bool {
        models
            .values()
            .any(|m| m.readers.load(Ordering::Acquire) != 0)
    }
    fn retire(models: &BTreeMap<String, Arc<CapturedModel>>) -> Result<(), ServiceError> {
        let mut failure = None;
        for (index, model) in models.values().enumerate() {
            if models.values().take(index).any(|m| m.group == model.group)
                || models
                    .values()
                    .any(|m| m.group == model.group && m.readers.load(Ordering::Acquire) != 0)
            {
                continue;
            }
            let result = match &model.engine {
                ModelEngine::FixedPrompt(m) => {
                    block_on(m.resident().clone().close(None)).map(|_| ())
                }
                ModelEngine::Qwen35Cuda(m) => {
                    block_on(m.resident().clone().close(None)).map(|_| ())
                }
            };
            if let Err(error) = result {
                failure.get_or_insert_with(|| ServiceError::inference(error));
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    fn diagnostics(
        models: &BTreeMap<String, Arc<CapturedModel>>,
        snapshot: &SchedulerSnapshot,
    ) -> Result<serde_json::Value, ServiceError> {
        let host_snapshot = snapshot.host.unwrap_or((0, 0, 0, 0));
        let hf = qwen35_cuda::health(models, host_snapshot);
        let backend = hf
            .as_ref()
            .and_then(|(_, r)| r.get("backend_host_reserved_bytes"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let host = if qwen35_cuda::models_present(models) && hf.is_none() {
            None
        } else {
            let (limit, reserved, _, serving) = host_snapshot;
            Some((
                limit,
                reserved,
                reserved
                    .checked_sub(backend)
                    .and_then(|v| v.checked_sub(serving)),
                backend,
                serving,
            ))
        };
        let resources = report::Resources {
            host,
            fixed: fixed_prompt::resources(models, host_snapshot),
            hf_source: hf.as_ref().map(|h| h.0.clone()),
            hf_resources: hf.map(|h| h.1),
        };
        Ok(report::render(snapshot, resources))
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
