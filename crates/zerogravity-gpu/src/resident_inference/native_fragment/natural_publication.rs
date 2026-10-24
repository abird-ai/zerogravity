//! Same original SessionAuthority publication and completed-packet feedback.
//! Closed qualification coordinates; the production quantum32 route stays closed.
use crate::resident_inference::native_fragment::continuation::{
    AttemptPhaseBinding, GdnAttemptSchedule, PrefixPhaseEnqueue,
};
use crate::resident_inference::{
    Domain, InferenceReadback, PreparedInference, ResidentModel, ResidentSession, ResidentToken,
};
use crate::resident_leases::{ResidentLease, ResidentService};
use std::sync::{
    Arc, OnceLock, Weak,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    GdnPrefixLayer, RecordedVllm08Model, Vllm08Call, VllmBf16Invocation, VllmBf16Teachers,
};
use zerogravity_core::device_events::BufferRole;
use zerogravity_core::session_root::{RootAttempt, RootBinding};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

mod completion;
mod coordinates;
mod faults;
mod qualify;
mod reuse;
pub use reuse::{
    Vllm08NaturalReuseFault, Vllm08NaturalReuseFaultExecution, Vllm08NaturalReuseFaultOutcome,
    Vllm08NaturalReuseReceipt, Vllm08NaturalReuseRound, qualify_vllm08_natural_reuse,
    qualify_vllm08_natural_reuse_fault,
};
mod state;
pub(crate) use coordinates::NaturalSlot;
pub use faults::{
    Vllm08NaturalFault, Vllm08NaturalFaultExecution, Vllm08NaturalFaultOutcome,
    qualify_vllm08_natural_fault,
};
pub use qualify::{Vllm08NaturalReceipt, qualify_vllm08_natural_model};
pub(crate) use state::{NaturalCallState, NaturalStatePool};

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State(
            "natural full24 cancelled before publication".into(),
        ));
    }
    Ok(())
}
fn same_model(expected: &Weak<ResidentModel>, session: &ResidentSession) -> Result<()> {
    let model = expected
        .upgrade()
        .ok_or_else(|| Error::State("natural original model owner retired".into()))?;
    if !Arc::ptr_eq(&model, &session.model) {
        return Err(Error::State(
            "natural foreign original ResidentModel".into(),
        ));
    }
    session.model.service.ensure_usable()
}

/// No constructor accepts public ResidentToken.id/max_bits or caller output IDs.
/// Created only by completion::commit after original ResidentSession::commit.
pub(crate) struct CommittedBf16Feedback {
    source: Arc<OriginalWindowModelSource>,
    model: Weak<ResidentModel>,
    token: ResidentToken,
    root: RootBinding,
    selected_id: u32,
    selected_bits: u32,
    packet_bytes: [u8; 12],
    allocation_generation: u64,
    service_generation: u64,
}
impl CommittedBf16Feedback {
    fn require_current(
        &self,
        session: &ResidentSession,
        source: &Arc<OriginalWindowModelSource>,
    ) -> Result<u32> {
        same_model(&self.model, session)?;
        if !Arc::ptr_eq(&self.source, source)
            || session.binding() != self.root
            || self.token.producer != Some(self.root)
            || !Arc::ptr_eq(self.token.lease.service(), &session.model.service)
            || self.token.lease.role() != BufferRole::Output
            || self.token.lease.bytes() != 12
            || !self.token.lease.is_initialized()
            || self.token.lease.allocation_generation() != self.allocation_generation
            || self.token.lease.service_generation() != self.service_generation
            || self.selected_id >= 248320
        {
            return Err(Error::State(
                "natural feedback is stale/foreign or lacks original completed packet".into(),
            ));
        }
        Ok(self.selected_id)
    }
    pub(crate) fn selected(&self) -> (u32, u32) {
        (self.selected_id, self.selected_bits)
    }
}

enum InputOrigin {
    Prompt,
    Feedback(Arc<CommittedBf16Feedback>),
}
/// Only the actual ResidentSession below mints this late immutable request.
/// Weak model identity avoids Body→input→Model.workspace→Body ownership cycles.
pub(crate) struct NaturalCallInput {
    source: Arc<OriginalWindowModelSource>,
    model: Weak<ResidentModel>,
    service: Arc<ResidentService>,
    attempt: RootAttempt,
    record: Arc<RecordedVllm08Model>,
    schedule: Arc<GdnAttemptSchedule>,
    phase: Arc<AttemptPhaseBinding>,
    slot: NaturalSlot,
    origin: InputOrigin,
    _metadata: WindowWorkspace,
}
impl NaturalCallInput {
    pub(crate) fn source(&self) -> &Arc<OriginalWindowModelSource> {
        &self.source
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    pub(crate) fn call(&self) -> &Vllm08Call {
        self.record.call()
    }
    pub(crate) fn ids(&self) -> &[u32] {
        self.call().request().teachers().ids()
    }
    pub(crate) fn attempt(&self) -> &RootAttempt {
        &self.attempt
    }
    pub(crate) fn slot(&self) -> NaturalSlot {
        self.slot
    }
    pub(in crate::resident_inference) fn phase_binding(&self) -> &Arc<AttemptPhaseBinding> {
        &self.phase
    }
    pub(crate) fn require_session(&self, session: &ResidentSession) -> Result<()> {
        same_model(&self.model, session)?;
        self.slot
            .require_current(session.binding(), &self.attempt, self.call())?;
        if !Arc::ptr_eq(&self.service, &session.model.service)
            || self.source.source_identity() != session.binding().source()
            || session.model.capacity != 512
            || session.model.vocab != 248320
        {
            return Err(Error::State(
                "natural exact source/service/current call differs".into(),
            ));
        }
        self.phase
            .require_full_model_binding(session, &self.attempt, self.call())?;
        match &self.origin {
            InputOrigin::Prompt if !self.slot.feedback() => {}
            InputOrigin::Feedback(feedback) if self.slot.feedback() => {
                if self.ids() != [feedback.require_current(session, &self.source)?] {
                    return Err(Error::State(
                        "natural source input differs from actual completed BF16 packet".into(),
                    ));
                }
            }
            _ => return Err(Error::State("natural prompt/feedback role differs".into())),
        }
        Ok(())
    }
    pub(crate) fn permit<'a>(
        &'a self,
        session: &'a ResidentSession,
    ) -> Result<PrefixPhaseEnqueue<'a>> {
        self.require_session(session)?;
        PrefixPhaseEnqueue::new(session, &self.schedule, 0, None)
    }
}
impl ResidentSession {
    pub(crate) fn bind_natural_prompt(
        &self,
        source: Arc<OriginalWindowModelSource>,
        slot: NaturalSlot,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<NaturalCallInput>> {
        if slot.feedback() {
            return Err(Error::State(
                "natural cached input requires original completed feedback, not public IDs".into(),
            ));
        }
        self.bind_natural_input(source, slot, ids, InputOrigin::Prompt, cancelled)
    }
    pub(crate) fn bind_natural_feedback(
        &self,
        source: Arc<OriginalWindowModelSource>,
        slot: NaturalSlot,
        feedback: Arc<CommittedBf16Feedback>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<NaturalCallInput>> {
        not_cancelled(cancelled)?;
        if !slot.feedback() {
            return Err(Error::State(
                "natural prompt input cannot be replaced with sampled feedback".into(),
            ));
        }
        let selected = feedback.require_current(self, &source)?;
        self.bind_natural_input(
            source,
            slot,
            &[selected],
            InputOrigin::Feedback(feedback),
            cancelled,
        )
    }
    fn bind_natural_input(
        &self,
        source: Arc<OriginalWindowModelSource>,
        slot: NaturalSlot,
        ids: &[u32],
        origin: InputOrigin,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<NaturalCallInput>> {
        not_cancelled(cancelled)?;
        source.require_vllm08_full_source()?;
        self.model.service.ensure_usable()?;
        if source.source_identity() != self.binding().source()
            || source.source_identity() != self.model.service.source_identity()
            || self.model.capacity != 512
            || self.model.vocab != 248320
            || ids.len() != slot.rows()
            || ids.iter().any(|id| *id >= 248320)
            || self.binding().frontier() != slot.base() as u64
        {
            return Err(Error::Unsupported(
                "closed natural .8B original current source/span".into(),
            ));
        }
        let metadata = source.source().reserve_workspace(4 * 1024 * 1024)?;
        let attempt = self.authority.begin_advance(slot.rows() as u64)?;
        let call = Vllm08Call::new(VllmBf16Invocation::new(
            GdnPrefixLayer::new(source.source_identity(), 0)?,
            slot.phase(),
            VllmBf16Teachers::new(source.source_identity(), slot.base(), 512, ids)?,
        )?)?;
        slot.require_current(self.binding(), &attempt, &call)?;
        let record = Arc::new(source.record_vllm08_model(call.clone())?);
        let schedule = GdnAttemptSchedule::new_published_call(&source, attempt.clone(), call)?;
        let phase = AttemptPhaseBinding::new(schedule.clone(), 0, None)?;
        let input = Arc::new(NaturalCallInput {
            source,
            model: Arc::downgrade(&self.model),
            service: self.model.service.clone(),
            attempt,
            record,
            schedule,
            phase,
            slot,
            origin,
            _metadata: metadata,
        });
        input.require_session(self)?;
        Ok(input)
    }
}
