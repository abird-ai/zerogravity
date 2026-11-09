use super::*;
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
use zerogravity_engine::{
    model_instance::execution::*,
    router::{
        GpuRootObservation, GpuTokenObservation,
        worker::{Active, SchedulerSnapshot},
    },
};
pub(super) struct SourceText;
struct Feedback {
    root: RootBinding,
    id: u32,
    device: Option<ResidentToken>,
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
    type Ready = std::convert::Infallible;
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
    fn prefill(_: &Self::Model) -> Prefill {
        Prefill::Chunks(64)
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
                "context": model.max_context(), "prefill_rows": 64, "native_session_rows": 0,
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
