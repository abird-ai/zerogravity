//! Host-logit adapter. Original model math/session publication remains external.
use super::pending::ServingHost as HostPool;
use super::worker::{Active, SchedulerSnapshot};
use super::*;
use crate::model_instance::{LogitModel, TextSession, execution::*};
use std::result::Result;
use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};
use zerogravity_core::resource_arbiter::{AllocationClass, GlobalResourceLease};

pub struct HostLogits;
pub struct State {
    session: Box<dyn TextSession>,
    logits: Vec<f32>,
    selected: Option<u32>,
}
pub struct Lease {
    _state: GlobalResourceLease,
    _projection: pending::ServingCharge,
}
impl Executor for HostLogits {
    type Model = Arc<dyn LogitModel>;
    type State = State;
    type Lease = Lease;
    type Ready = std::convert::Infallible;

    fn open(
        model: &Self::Model,
        alias: &str,
        _: &Arc<AtomicBool>,
        host: &HostPool,
    ) -> Result<(State, Lease), ServiceError> {
        let state_charge = host.admit(
            alias,
            "cpu-request-state",
            AllocationClass::RecurrentState,
            model.max_session_bytes(),
        )?;
        let projection = host.charge(
            alias,
            "rayon-projection",
            AllocationClass::Scratch,
            model.session_scratch_bytes(),
        )?;
        let session = model
            .clone()
            .new_session()
            .map_err(ServiceError::inference)?;
        Ok((
            State {
                session,
                logits: Vec::new(),
                selected: None,
            },
            Lease {
                _state: state_charge,
                _projection: projection,
            },
        ))
    }
    fn describe_model(model: &Self::Model) -> crate::Result<crate::diagnostics::ModelDetails> {
        let mut details = model.describe()?;
        details.memory = Some(crate::diagnostics::ModelMemory::Host {
            session_state_bytes: model.max_session_bytes(),
            session_scratch_bytes: model.session_scratch_bytes(),
        });
        Ok(details)
    }
    fn prefill(_: &Self::Model) -> Prefill {
        Prefill::Chunks(32)
    }
    fn group(_: &Self::Model) -> usize {
        0
    }
    fn frontier(_: &State) -> usize {
        0
    } // Sequential work never forms physical cohorts.
    fn selected(state: &mut State) -> Result<u32, ServiceError> {
        let id = state
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .ok_or_else(|| ServiceError::new(500, "empty_logits", "inference produced no logits"))?
            .0 as u32;
        state.selected = Some(id);
        Ok(id)
    }
    fn has_feedback(state: &State) -> bool {
        !state.logits.is_empty()
    }
    fn single(row: &mut Active<Self>) -> Result<SingleCompleted, ServiceError> {
        let (consumed, readout) = if row.cursor < row.prompt.len() {
            let end = row.cursor + row.prefill_count();
            let tokens = &row.prompt[row.cursor..end];
            if end < row.prompt.len() {
                row.state
                    .session
                    .prefill_chunk_without_logits(tokens)
                    .map_err(ServiceError::inference)?;
            } else {
                row.state.logits = row
                    .state
                    .session
                    .prefill_transaction(tokens)
                    .map_err(ServiceError::inference)?;
            }
            (end - row.cursor, end == row.prompt.len())
        } else {
            let id = row.state.selected.take().ok_or_else(|| {
                ServiceError::new(
                    500,
                    "missing_feedback",
                    "host-logit selection was not presented",
                )
            })?;
            row.state.logits = row
                .state
                .session
                .forward(id)
                .map_err(ServiceError::inference)?;
            (1, true)
        };
        Ok(SingleCompleted {
            // CPU adapters do not invent native dispatch/root receipt counters.
            completion: Completion::default(),
            published: Ok(Published {
                consumed,
                readout,
                observed_at: Instant::now(),
                observation: None,
            }),
        })
    }
    fn selective_retirement() -> Option<ModelRetirement<Self::Model>> {
        // CPU source weights are retained by the original final Arc readers.
        Some(|_| Ok(()))
    }
    fn diagnostics(
        _: &BTreeMap<String, Self::Model>,
        snapshot: &SchedulerSnapshot,
    ) -> Result<serde_json::Value, ServiceError> {
        let (limit, reserved, remaining, serving) = snapshot
            .host
            .as_ref()
            .map(|host| host.totals())
            .ok_or_else(|| {
                ServiceError::new(500, "host_state", "host admission snapshot unavailable")
            })?;
        let q = &snapshot.queue;
        Ok(serde_json::json!({"kind":"configured_host_cpu",
            "host_limit_bytes":limit,"total_host_limit_bytes":limit,"total_host_reserved_bytes":reserved,
            "reserved_bytes":reserved,"remaining_bytes":remaining,"model_host_reserved_bytes":reserved.checked_sub(serving),
            "backend_host_reserved_bytes":0,"serving_host_reserved_bytes":serving,
            "queued_requests":q.pending,"pending_admission_in_progress":q.in_progress,
            "pending_queue_capacity":q.capacity,"pending_queue_charged_bytes":q.backing_bytes,
            "pending_queue_consumers":q.consumers,"queued_requests_limit":q.limit,
            "active_sessions":snapshot.counters.active,"peak_active_sessions":snapshot.counters.peak,
            "active_sessions_limit":snapshot.active_limit}))
    }
}
