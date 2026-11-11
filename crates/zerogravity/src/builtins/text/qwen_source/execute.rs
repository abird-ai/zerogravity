use super::*;
#[cfg(feature = "qwen35-cuda")]
use crate::gpu::{SourceCallCohortPending, SourceCallCohortPreparation};
use crate::{
    accelerator::program::SourceCallMode,
    gpu::{
        InferenceReadback, ResidentOutput, ResidentToken, SourceCallPreparation,
        SourceCallPublication,
    },
    numerics::{ActivationPrecision, greedy_finite},
    session_root::RootBinding,
};
use futures_lite::future::block_on;
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
#[cfg(feature = "qwen35-cuda")]
use zerogravity_engine::router::waves::Phase;
use zerogravity_engine::{
    model_instance::execution::*,
    router::{
        GpuRootObservation, GpuTokenObservation,
        worker::{Active, SchedulerSnapshot},
    },
};
pub(super) struct SourceText;
#[cfg(feature = "qwen35-cuda")]
type CohortReady = SourceCallCohortPending;
#[cfg(not(feature = "qwen35-cuda"))]
type CohortReady = std::convert::Infallible;
struct Feedback {
    root: RootBinding,
    id: u32,
    device: Option<ResidentToken>,
}
#[cfg(feature = "qwen35-cuda")]
fn native_phase_compatible(
    first_prefill: Option<(u64, usize)>,
    row_prefill: Option<(u64, usize)>,
) -> bool {
    match (first_prefill, row_prefill) {
        (Some(first), Some(row)) => first == row,
        (None, None) => true,
        _ => false,
    }
}
pub(super) struct State {
    session: SourceCallSession,
    // Computed from this original committed readback; never presentation IDs.
    feedback: Option<Feedback>,
}
fn cancelled() -> ServiceError {
    ServiceError::new(499, "cancelled", "request cancelled")
}
fn phase(base: usize, rows: usize) -> SourceCallMode {
    if base != 0 && rows == 1 {
        SourceCallMode::Decode
    } else {
        SourceCallMode::Prefill
    }
}
fn select(logits: &[f32], vocabulary: u64) -> crate::Result<u32> {
    if logits.len() as u64 != vocabulary {
        return Err(crate::Error::State(
            "source full vocabulary readback extent".into(),
        ));
    }
    ActivationPrecision::Bf16WidenedF32.validate_values(logits)?;
    Ok(greedy_finite(logits)?.0)
}
fn completion(read: &InferenceReadback, consumed: usize, readout: bool) -> Completion {
    Completion {
        forwards: 1,
        dispatches: read.dispatches as u64,
        model_tokens: consumed as u64,
        state_only: !readout,
        ..Completion::default()
    }
}
impl Executor for SourceText {
    type Model = Arc<Model>;
    type State = State;
    type Lease = Arc<Model>;
    type Ready = CohortReady;
    fn open(
        model: &Self::Model,
        _: &str,
        flag: &Arc<AtomicBool>,
        _: &ServingHost,
    ) -> Result<(State, Self::Lease), ServiceError> {
        if flag.load(Ordering::Acquire) {
            return Err(cancelled());
        }
        let session = block_on(model.resident()?.session()).map_err(ServiceError::inference)?;
        Ok((
            State {
                session,
                feedback: None,
            },
            model.clone(),
        ))
    }
    fn prefill(model: &Self::Model) -> Prefill {
        Prefill::Chunks(model.source.plan.max_chunk_rows())
    }
    fn group(_: &Self::Model) -> usize {
        0
    }
    fn frontier(state: &State) -> usize {
        state.session.binding().frontier() as usize
    }
    fn selected(state: &mut State) -> Result<u32, ServiceError> {
        state
            .feedback
            .as_ref()
            .filter(|feedback| feedback.root == state.session.binding())
            .map(|feedback| feedback.id)
            .ok_or_else(|| {
                ServiceError::new(
                    500,
                    "source_feedback",
                    "original committed feedback absent or stale",
                )
            })
    }
    fn has_feedback(state: &State) -> bool {
        state.feedback.is_some()
    }
    #[cfg(feature = "qwen35-cuda")]
    fn cohort_eligible(row: &Active<Self>) -> bool {
        if row.cursor < row.prompt.len() {
            let rows = row.prefill_count();
            let frontier = row.state.session.binding().frontier() as usize;
            rows != 0
                && rows <= row.model.source.plan.max_chunk_rows()
                && !row.packet_presented
                && row.blocked_since.is_none()
                && row.state.feedback.is_none()
                && frontier == row.cursor
                && frontier
                    .checked_add(rows)
                    .is_some_and(|end| end <= row.model.source.plan.context_capacity_tokens())
        } else {
            row.packet_presented
                && row.blocked_since.is_none()
                && row.state.feedback.as_ref().is_some_and(|feedback| {
                    feedback.root == row.state.session.binding() && feedback.device.is_some()
                })
        }
    }
    #[cfg(feature = "qwen35-cuda")]
    fn compatible(first: &Active<Self>, row: &Active<Self>) -> bool {
        let prefill = |row: &Active<Self>| {
            (row.cursor < row.prompt.len())
                .then(|| (row.state.session.binding().frontier(), row.prefill_count()))
        };
        Arc::ptr_eq(&first.model, &row.model)
            && native_phase_compatible(prefill(first), prefill(row))
    }
    #[cfg(feature = "qwen35-cuda")]
    fn validate_cohort(rows: &[Active<Self>], work: CohortWork) -> Result<(), ServiceError> {
        let phase_matches = match work.phase {
            Phase::Prefill => rows.iter().all(|row| row.cursor < row.prompt.len()),
            Phase::Decode => rows.iter().all(|row| row.cursor == row.prompt.len()),
        };
        let output_matches = match work.phase {
            Phase::Prefill => rows
                .iter()
                .all(|row| work.state_only == row.prefill_has_successor()),
            Phase::Decode => !work.state_only,
        };
        if !matches!(work.native_rows, 2 | 4 | 8)
            || rows.len() != work.native_rows
            || !phase_matches
            || !output_matches
            || rows.iter().any(|row| !Self::cohort_eligible(row))
            || rows[1..].iter().any(|row| !Self::compatible(&rows[0], row))
        {
            return Err(ServiceError::new(
                500,
                "source_cohort",
                "source CUDA cohort requires aligned native prefill chunks or compatible cached-decode rows",
            ));
        }
        Ok(())
    }
    #[cfg(feature = "qwen35-cuda")]
    fn prepare(
        rows: &mut [Active<Self>],
        work: CohortWork,
    ) -> Result<Preparation<Self::Ready>, ServiceError> {
        Self::validate_cohort(rows, work)?;
        let tokens_per_row = match work.phase {
            Phase::Prefill => rows[0].prefill_count(),
            Phase::Decode => 1,
        };
        let model_tokens = tokens_per_row
            .checked_mul(rows.len())
            .and_then(|tokens| u64::try_from(tokens).ok())
            .ok_or_else(|| {
                ServiceError::new(500, "source_cohort", "cohort model-token count overflow")
            })?;
        let model = rows[0].model.resident()?;
        let ready = match work.phase {
            Phase::Prefill => {
                let invocations = rows
                    .iter()
                    .map(|row| {
                        let end = row.cursor.checked_add(row.prefill_count()).ok_or_else(|| {
                            crate::Error::Limit("source cohort prefill cursor".into())
                        })?;
                        row.model.source.plan.bind_invocation(
                            SourceCallMode::Prefill,
                            Self::frontier(&row.state),
                            Arc::from(&row.prompt[row.cursor..end]),
                            !work.state_only,
                        )
                    })
                    .collect::<crate::Result<Vec<_>>>()
                    .map_err(ServiceError::inference)?;
                let inputs = rows
                    .iter()
                    .zip(&invocations)
                    .map(|(row, invocation)| (&row.state.session, invocation.bound_invocation()))
                    .collect::<Vec<_>>();
                match block_on(model.prepare_prefill_cohort_admitted(&inputs))
                    .map_err(ServiceError::inference)?
                {
                    SourceCallCohortPreparation::Capacity => return Ok(Preparation::Capacity),
                    SourceCallCohortPreparation::Ready(ready) => ready,
                }
            }
            Phase::Decode => {
                let invocations = rows
                    .iter()
                    .map(|row| {
                        let feedback = row.state.feedback.as_ref().expect("validated feedback");
                        row.model.source.plan.bind_invocation(
                            SourceCallMode::Decode,
                            Self::frontier(&row.state),
                            Arc::from([feedback.id]),
                            true,
                        )
                    })
                    .collect::<crate::Result<Vec<_>>>()
                    .map_err(ServiceError::inference)?;
                let inputs = rows
                    .iter()
                    .zip(&invocations)
                    .map(|(row, invocation)| {
                        (
                            &row.state.session,
                            invocation.bound_invocation(),
                            row.state
                                .feedback
                                .as_ref()
                                .and_then(|feedback| feedback.device.as_ref())
                                .expect("validated device feedback"),
                        )
                    })
                    .collect::<Vec<_>>();
                match block_on(model.prepare_greedy_decode_cohort_admitted(&inputs))
                    .map_err(ServiceError::inference)?
                {
                    SourceCallCohortPreparation::Capacity => return Ok(Preparation::Capacity),
                    SourceCallCohortPreparation::Ready(ready) => ready,
                }
            }
        };
        let native = ready.fused_dispatches() != 0;
        Ok(Preparation::Ready(Completed {
            completion: Completion {
                quanta: rows.len() as u64,
                forwards: 1,
                dispatches: ready.dispatches() as u64,
                model_tokens,
                committed_roots: 0,
                native_projections: ready.fused_dispatches() as u64,
                native_sampler: 0,
                native_rows: if native { rows.len() } else { 0 },
                state_only: work.state_only,
            },
            ready,
            native,
        }))
    }
    #[cfg(feature = "qwen35-cuda")]
    fn commit(
        ready: Self::Ready,
        rows: &mut [Active<Self>],
        work: CohortWork,
    ) -> Result<Publication, ServiceError> {
        Self::validate_cohort(rows, work)?;
        let consumed = match work.phase {
            Phase::Prefill => rows[0].prefill_count(),
            Phase::Decode => 1,
        };
        let mut sessions = rows
            .iter_mut()
            .map(|row| &mut row.state.session)
            .collect::<Vec<_>>();
        let receipts = ready
            .commit(&mut sessions)
            .map_err(ServiceError::inference)?;
        drop(sessions);
        if receipts.len() != rows.len() {
            return Err(ServiceError::new(
                500,
                "source_cohort_receipts",
                "source cohort publisher returned a different row count",
            ));
        }
        let committed = Instant::now();
        let published = rows
            .iter_mut()
            .zip(receipts)
            .map(|(row, read)| {
                if work.state_only {
                    if read.state_only_steps != 1
                        || read.logit_steps != 0
                        || !read.logits.is_empty()
                        || read.device_token.is_some()
                    {
                        return Err(ServiceError::new(
                            500,
                            "source_cohort_prefix_readout",
                            "state-only source cohort produced a generated readout",
                        ));
                    }
                    return Ok(Published {
                        consumed,
                        readout: false,
                        observed_at: committed,
                        observation: None,
                    });
                }
                let token = read.device_token.clone().ok_or_else(|| {
                    ServiceError::new(
                        500,
                        "source_cohort_token",
                        "source cohort completion has no device token",
                    )
                })?;
                if read.logit_steps != 1 || !read.logits.is_empty() {
                    return Err(ServiceError::new(
                        500,
                        "source_cohort_readout",
                        "source cohort completion returned invalid readout",
                    ));
                }
                let root = row.state.session.binding();
                row.state.feedback = Some(Feedback {
                    root,
                    id: token.id,
                    device: Some(token),
                });
                Ok(Published {
                    consumed,
                    readout: true,
                    observed_at: committed,
                    observation: Some(GpuTokenObservation {
                        producer_frontier: root.frontier(),
                        producer_root: Some(GpuRootObservation::committed(root)),
                        native_cohort: None,
                    }),
                })
            })
            .collect();
        Ok(Publication::Committed(published))
    }
    fn single(row: &mut Active<Self>) -> Result<SingleCompleted, ServiceError> {
        let (ids, consumed, readout): (Arc<[u32]>, usize, bool) = if row.cursor < row.prompt.len() {
            let end = row.cursor + row.prefill_count();
            (
                Arc::from(&row.prompt[row.cursor..end]),
                end - row.cursor,
                end == row.prompt.len(),
            )
        } else {
            (Arc::from([Self::selected(&mut row.state)?]), 1, true)
        };
        let base = Self::frontier(&row.state);
        let mode = phase(base, consumed);
        let device_greedy = readout && row.state.session.supports_device_greedy();
        let device_input = if mode == SourceCallMode::Decode {
            row.state
                .feedback
                .as_ref()
                .filter(|feedback| feedback.root == row.state.session.binding())
                .and_then(|feedback| feedback.device.clone())
        } else {
            None
        };
        let invocation = row
            .model
            .source
            .plan
            .bind_invocation(mode, base, ids, readout)
            .map_err(ServiceError::inference)?;
        // Always drive submitted work to original completion/quarantine. No
        // cancellation race, dropped submitted future, or retry after an error.
        let pending = match block_on(async {
            if device_greedy {
                if let Some(token) = device_input.as_ref() {
                    row.state
                        .session
                        .prepare_greedy_device_token(
                            invocation.bound_invocation(),
                            mode,
                            token,
                            Some(row.cancelled.clone()),
                        )
                        .await
                } else {
                    row.state
                        .session
                        .prepare_greedy(
                            invocation.bound_invocation(),
                            mode,
                            Some(row.cancelled.clone()),
                        )
                        .await
                }
            } else {
                row.state
                    .session
                    .prepare(
                        invocation.bound_invocation(),
                        mode,
                        if readout {
                            ResidentOutput::Logits
                        } else {
                            ResidentOutput::StateOnly
                        },
                        Some(row.cancelled.clone()),
                    )
                    .await
            }
        })
        .map_err(ServiceError::inference)?
        {
            SourceCallPreparation::Ready(pending) => pending,
            SourceCallPreparation::Capacity => {
                return Err(ServiceError::new(
                    503,
                    "source_capacity",
                    "original prelaunch capacity refused",
                ));
            }
            SourceCallPreparation::CancelledBeforeLaunch => return Err(cancelled()),
        };
        let mut completed = completion(pending.readback(), consumed, readout);
        // Validate actual completed logits before publishing. Failed selection
        // drops only a known-completed candidate and still counts its work.
        let selected = if readout {
            let selected = if device_greedy {
                pending
                    .readback()
                    .device_token
                    .as_ref()
                    .map(|token| token.id)
                    .ok_or_else(|| crate::Error::State("ordinary CUDA greedy packet absent".into()))
            } else {
                select(&pending.readback().logits, row.model.vocabulary_size())
            };
            match selected {
                Ok(id) => Some(id),
                Err(error) => {
                    return Ok(SingleCompleted {
                        completion: completed,
                        published: Err(ServiceError::inference(error)),
                    });
                }
            }
        } else {
            None
        };
        let published = match row.state.session.commit(pending) {
            Err(error) => Err(ServiceError::inference(error)),
            Ok(SourceCallPublication::Discarded(_readback)) => Err(cancelled()),
            Ok(SourceCallPublication::Committed(readback)) => {
                completed.committed_roots = 1;
                let root = row.state.session.binding();
                row.state.feedback = selected.map(|id| Feedback {
                    root,
                    id,
                    device: readback.device_token.clone(),
                });
                Ok(Published {
                    consumed,
                    readout,
                    observed_at: Instant::now(),
                    observation: readout.then(|| GpuTokenObservation {
                        producer_frontier: root.frontier(),
                        producer_root: Some(GpuRootObservation::committed(root)),
                        native_cohort: None,
                    }),
                })
            }
        };
        Ok(SingleCompleted {
            completion: completed,
            published,
        })
    }
    fn retire(models: &BTreeMap<String, Self::Model>) -> Result<(), ServiceError> {
        for model in models.values() {
            let resident = model.resident.lock().map_err(|_| poison())?.take();
            if let Some(resident) = resident {
                block_on(resident.shutdown()).map_err(ServiceError::inference)?;
            }
        }
        Ok(())
    }
    fn diagnostics(
        models: &BTreeMap<String, Self::Model>,
        snapshot: &SchedulerSnapshot,
    ) -> Result<serde_json::Value, ServiceError> {
        let mut report = super::super::gpu::report::render(
            snapshot,
            super::super::gpu::report::Resources::ordinary(snapshot),
        );
        // One physical host ledger; no observed backend/source subdivision.
        report["model_host_reserved_bytes"] = serde_json::Value::Null;
        report["backend_host_reserved_bytes"] = serde_json::Value::Null;
        report["model_and_backend_host_reserved_bytes"] = serde_json::json!(
            snapshot
                .host
                .and_then(|(_, reserved, _, serving)| reserved.checked_sub(serving))
        );
        report["source_models"] = serde_json::Value::Array(models.iter().map(|(alias, model)| {
            let metadata = &model.source.metadata;
            serde_json::json!({"alias": alias, "family": metadata.family, "repository": metadata.repository,
                "revision": metadata.revision, "hidden_width": metadata.hidden_width,
                "context": model.max_context(), "prefill_rows": 64,
                "numerical_contract": format!("{:?}", model.source.plan.source_seal().numerical_contract())})
        }).collect());
        Ok(report)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initialized_single_prompt_tail_is_decode_not_cached_chunk() {
        assert_eq!(phase(0, 1), SourceCallMode::Prefill);
        assert_eq!(phase(64, 1), SourceCallMode::Decode);
        assert_eq!(phase(64, 17), SourceCallMode::Prefill);
        assert_eq!(phase(47, 64), SourceCallMode::Prefill);
    }
    #[cfg(feature = "qwen35-cuda")]
    #[test]
    fn native_prefill_stays_aligned_while_decode_accepts_private_frontiers() {
        assert!(native_phase_compatible(Some((47, 17)), Some((47, 17))));
        assert!(!native_phase_compatible(Some((47, 17)), Some((48, 17))));
        assert!(!native_phase_compatible(Some((47, 17)), Some((47, 16))));
        assert!(!native_phase_compatible(Some((47, 17)), None));
        assert!(native_phase_compatible(None, None));
    }
    #[test]
    fn feedback_uses_complete_finite_bf16_logits_with_original_zero_tie() {
        assert_eq!(select(&[-0.0, 0.0], 2).unwrap(), 0);
        assert_eq!(select(&[1.0, 2.0, 2.0], 3).unwrap(), 1);
        for (values, n) in [
            (&[1.0][..], 2),
            (&[f32::INFINITY][..], 1),
            (&[f32::from_bits(1)][..], 1),
        ] {
            assert!(select(values, n).is_err());
        }
    }
}
