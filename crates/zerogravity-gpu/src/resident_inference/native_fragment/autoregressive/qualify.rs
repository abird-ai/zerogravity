//! Existing autoregressive physical preparation; model-owned loops supply actual calls.
use super::completion::{AutoregressiveCommit, AutoregressivePrepared, AutoregressiveSubmission};
use super::*;
use crate::resident_inference::native_fragment::{
    AutoregressiveReuseStorage, ColdModelSlots, SourceModelReadback,
};
use crate::resident_program::native_full::FullModelProgram;
use zerogravity_core::device_events::BufferLeaseId;

#[derive(Debug)]
pub struct AutoregressiveReceipt {
    pub phases: usize,
    pub roots: [RootBinding; 2],
    pub phase_roots: Vec<RootBinding>,
    pub short_tokens: [u32; 2],
    pub long_tokens: [u32; 2],
    pub prefix64_emissions: usize,
    pub native_launches: usize,
    pub workspace_clears: usize,
    pub device_copies: usize,
    pub dispatches: usize,
    pub source_weight_windows: usize,
    pub tied_weight_windows: usize,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}

/// Same service observation owner previously retained by each qualification loop.
/// No enqueue, driver, root or source constructor is exposed.
pub struct AutoregressiveObserver {
    service: Arc<ResidentService>,
}
impl AutoregressiveObserver {
    pub(crate) fn new(model: &ResidentModel) -> Self {
        Self {
            service: model.service.clone(),
        }
    }
    pub fn finish(self) -> Result<(usize, u64)> {
        let resident = self.service.resident_bytes()?;
        if resident != 0 || self.service.pending_events()? != 0 {
            return Err(Error::State(
                "autoregressive completed owners failed to retire".into(),
            ));
        }
        let key = self.service.physical_key();
        drop(self);
        let physical =
            zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
        if physical.reserved != 0 || physical.fenced {
            return Err(Error::State(
                "autoregressive physical envelope failed final0".into(),
            ));
        }
        Ok((resident, physical.reserved))
    }
}
/// Copied observations of the SAME original completed body; no native permit.
pub struct AutoregressiveStep {
    prepared: AutoregressivePrepared,
    pub native_launches: usize,
    pub workspace_clears: usize,
    pub device_copies: usize,
    pub value_ids: Option<Vec<BufferLeaseId>>,
    pub candidate_ids: Option<Vec<BufferLeaseId>>,
    pub packet_id: Option<BufferLeaseId>,
}
impl AutoregressiveStep {
    fn new(prepared: AutoregressivePrepared, body: &FullModelProgram) -> Self {
        Self {
            prepared,
            native_launches: body.native_launches(),
            workspace_clears: body.workspace_clears(),
            device_copies: body.device_copies(),
            value_ids: None,
            candidate_ids: None,
            packet_id: None,
        }
    }
    pub fn commit(
        self,
        session: &mut ResidentSession,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AutoregressiveCommit> {
        session.commit_autoregressive_call(self.prepared, cancelled)
    }
}
impl ColdModelSlots {
    pub async fn prepare_prompt<F>(
        self: &Arc<Self>,
        session: &ResidentSession,
        record: Arc<SourceBindings>,
        slot: AutoregressiveSlot,
        pool: &Arc<AutoregressiveStatePool>,
        validate: F,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AutoregressiveStep>
    where
        F: FnMut(SourceModelReadback<'_>) -> Result<()>,
    {
        let input = session.bind_autoregressive_prompt(record, slot, cancelled)?;
        self.prepare_input(session, input, pool, validate, cancelled)
            .await
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "qualification API keeps current session, immutable source record, feedback authority, state pool and validation callback distinct"
    )]
    pub async fn prepare_feedback<F>(
        self: &Arc<Self>,
        session: &ResidentSession,
        record: Arc<SourceBindings>,
        slot: AutoregressiveSlot,
        request: AutoregressiveFeedback<'_>,
        pool: &Arc<AutoregressiveStatePool>,
        validate: F,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AutoregressiveStep>
    where
        F: FnMut(SourceModelReadback<'_>) -> Result<()>,
    {
        let input = session.bind_autoregressive_feedback(record, slot, request, cancelled)?;
        self.prepare_input(session, input, pool, validate, cancelled)
            .await
    }
    async fn prepare_input<F>(
        self: &Arc<Self>,
        session: &ResidentSession,
        input: Arc<AutoregressiveInput>,
        pool: &Arc<AutoregressiveStatePool>,
        validate: F,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AutoregressiveStep>
    where
        F: FnMut(SourceModelReadback<'_>) -> Result<()>,
    {
        let state = session.bind_autoregressive_state(input.clone(), pool)?;
        let program = self.bind_call(session, input.clone(), state.clone(), cancelled)?;
        let body = program.full_model_body()?.clone();
        let prepared = session
            .prepare_autoregressive_call(
                AutoregressiveSubmission {
                    input,
                    state,
                    program,
                    cold_owner: self.completion_owner(),
                },
                validate,
                cancelled,
            )
            .await?;
        Ok(AutoregressiveStep::new(prepared, &body))
    }
}
impl AutoregressiveReuseStorage {
    pub async fn prepare_prompt<F>(
        self: &Arc<Self>,
        session: &ResidentSession,
        record: Arc<SourceBindings>,
        slot: AutoregressiveSlot,
        validate: F,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AutoregressiveStep>
    where
        F: FnMut(SourceModelReadback<'_>) -> Result<()>,
    {
        self.require_open(session)?;
        let input = session.bind_autoregressive_prompt(record, slot, cancelled)?;
        self.prepare_input(session, input, validate, cancelled)
            .await
    }
    pub async fn prepare_feedback<F>(
        self: &Arc<Self>,
        session: &ResidentSession,
        record: Arc<SourceBindings>,
        slot: AutoregressiveSlot,
        request: AutoregressiveFeedback<'_>,
        validate: F,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AutoregressiveStep>
    where
        F: FnMut(SourceModelReadback<'_>) -> Result<()>,
    {
        self.require_open(session)?;
        let input = session.bind_autoregressive_feedback(record, slot, request, cancelled)?;
        self.prepare_input(session, input, validate, cancelled)
            .await
    }
    async fn prepare_input<F>(
        self: &Arc<Self>,
        session: &ResidentSession,
        input: Arc<AutoregressiveInput>,
        validate: F,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AutoregressiveStep>
    where
        F: FnMut(SourceModelReadback<'_>) -> Result<()>,
    {
        let state = session
            .bind_fresh_autoregressive_state(input.clone(), cancelled)
            .await?;
        state.require_reusable_state()?;
        let candidate_ids = state.outputs().iter().map(ResidentLease::id).collect();
        let mut observed_body = None;
        let prepared = session
            .prepare_autoregressive_call_bound(
                input.clone(),
                state.clone(),
                self.completion_owner(),
                || {
                    let program = self.bind_call(session, input, state, cancelled)?;
                    observed_body = Some(program.full_model_body()?.clone());
                    Ok(program)
                },
                validate,
                cancelled,
            )
            .await?;
        let body = observed_body
            .ok_or_else(|| Error::State("reusable actual locked binding absent".into()))?;
        let mut step = AutoregressiveStep::new(prepared, &body);
        step.value_ids = Some(super::reuse::value_ids(&body)?);
        step.candidate_ids = Some(candidate_ids);
        step.packet_id = Some(body.publication_packet()?.id());
        Ok(step)
    }
    pub fn observer(&self) -> Result<AutoregressiveObserver> {
        let model = self.model()?;
        Ok(AutoregressiveObserver::new(&model))
    }
    pub async fn verify_packets(
        self: &Arc<Self>,
        packets: &[Arc<CommittedBf16Feedback>],
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        super::reuse::verify_packets(&self.model()?, self, packets, cancelled).await
    }
}
