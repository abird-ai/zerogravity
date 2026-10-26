//! Same original SessionAuthority publication and completed-packet feedback.
//! Closed qualification coordinates; the production quantum32 route stays closed.
use crate::resident_inference::native_fragment::continuation::{
    AttemptPhaseBinding, GdnAttemptSchedule, PrefixPhaseEnqueue,
};
use crate::resident_inference::{
    Domain, InferenceReadback, PreparedInference, ResidentModel, ResidentSession, ResidentToken,
};
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::resident_program::source_bindings::SourceBindings;
use std::sync::{
    Arc, OnceLock, Weak,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::SourceCall;
use zerogravity_core::device_events::BufferRole;
use zerogravity_core::session_root::{RootAttempt, RootBinding};
use zerogravity_core::{Error, Result};

mod completion;
mod coordinates;
mod faults;
mod qualify;
mod reuse;
pub use reuse::{
    NaturalReuseFault, NaturalReuseFaultExecution, NaturalReuseFaultOutcome, NaturalReuseReceipt,
    NaturalReuseRound, qualify_natural_reuse_fault,
};
mod state;
pub use completion::NaturalCommitted;
pub use coordinates::NaturalSlot;
pub use faults::{NaturalFault, NaturalFaultExecution, NaturalFaultOutcome, qualify_natural_fault};
pub use qualify::{NaturalObserver, NaturalReceipt, NaturalStep};
pub(crate) use state::NaturalCallState;
pub use state::NaturalStatePool;

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
pub struct CommittedBf16Feedback {
    source: VerifiedTensorSource,
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
        source: &VerifiedTensorSource,
    ) -> Result<u32> {
        same_model(&self.model, session)?;
        require_packet_words(
            self.selected_id,
            self.selected_bits,
            self.token.id,
            self.token.max_bits,
            &self.packet_bytes,
        )?;
        if !self.source.same_issuer(source)
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
    pub fn selected(&self) -> (u32, u32) {
        (self.selected_id, self.selected_bits)
    }
}

fn require_packet_words(
    id: u32,
    bits: u32,
    token_id: u32,
    token_bits: u32,
    raw: &[u8; 12],
) -> Result<()> {
    if id >= 248320
        || id != token_id
        || bits != token_bits
        || raw[..4] != id.to_le_bytes()
        || raw[4..8] != bits.to_le_bytes()
        || raw[8..] != [0; 4]
        || bits & 0xffff != 0
        || bits & 0x7f80_0000 == 0x7f80_0000
    {
        return Err(Error::State(
            "natural actual committed BF16 packet words differ".into(),
        ));
    }
    Ok(())
}
/// Borrows one genuine committed packet and its current session; data is not a permit.
pub struct NaturalFeedbackRequest<'a> {
    session: &'a ResidentSession,
    packet: &'a Arc<CommittedBf16Feedback>,
    root: RootBinding,
}
impl CommittedBf16Feedback {
    pub fn request<'a>(
        self: &'a Arc<Self>,
        session: &'a ResidentSession,
    ) -> Result<NaturalFeedbackRequest<'a>> {
        self.require_current(session, &self.source)?;
        Ok(NaturalFeedbackRequest {
            session,
            packet: self,
            root: session.binding(),
        })
    }
}
impl NaturalFeedbackRequest<'_> {
    pub fn selected_id(&self) -> u32 {
        self.packet.selected_id
    }
    pub fn base(&self) -> usize {
        self.packet.root.frontier() as usize
    }
    fn require(
        &self,
        session: &ResidentSession,
        source: &VerifiedTensorSource,
    ) -> Result<&Arc<CommittedBf16Feedback>> {
        if !std::ptr::eq(session, self.session) || self.root != session.binding() {
            return Err(Error::State(
                "natural feedback request belongs to another current session".into(),
            ));
        }
        self.packet.require_current(session, source)?;
        Ok(self.packet)
    }
}

enum InputOrigin {
    Prompt,
    Feedback(Arc<CommittedBf16Feedback>),
}
/// Only the actual ResidentSession below mints this late immutable request.
/// Weak model identity avoids Body→input→Model.workspace→Body ownership cycles.
pub(crate) struct NaturalCallInput {
    source: VerifiedTensorSource,
    model: Weak<ResidentModel>,
    service: Arc<ResidentService>,
    attempt: RootAttempt,
    record: Arc<SourceBindings>,
    schedule: Arc<GdnAttemptSchedule>,
    phase: Arc<AttemptPhaseBinding>,
    slot: NaturalSlot,
    origin: InputOrigin,
    _metadata: WindowWorkspace,
}
impl NaturalCallInput {
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn record(&self) -> &Arc<SourceBindings> {
        &self.record
    }
    pub(crate) fn call(&self) -> &SourceCall {
        self.record.call()
    }
    pub(crate) fn ids(&self) -> &[u32] {
        self.call().rows().ids()
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
            || !self.source.same_issuer(self.record.source())
            || self.call().rows().capacity() != session.model.capacity
            || u64::try_from(self.call().rows().vocabulary()).ok() != Some(session.model.vocab)
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
        record: Arc<SourceBindings>,
        slot: NaturalSlot,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<NaturalCallInput>> {
        if slot.feedback() {
            return Err(Error::State(
                "natural cached input requires original committed feedback".into(),
            ));
        }
        self.bind_natural_input(record, slot, InputOrigin::Prompt, cancelled)
    }
    pub(crate) fn bind_natural_feedback(
        &self,
        record: Arc<SourceBindings>,
        slot: NaturalSlot,
        request: NaturalFeedbackRequest<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<NaturalCallInput>> {
        not_cancelled(cancelled)?;
        if !slot.feedback() {
            return Err(Error::State(
                "natural prompt input cannot be replaced with sampled feedback".into(),
            ));
        }
        let feedback = request.require(self, record.source())?;
        if record.call().rows().ids() != [feedback.selected_id] {
            return Err(Error::State(
                "natural prepared call differs from original committed ID".into(),
            ));
        }
        self.bind_natural_input(
            record,
            slot,
            InputOrigin::Feedback(feedback.clone()),
            cancelled,
        )
    }
    fn bind_natural_input(
        &self,
        record: Arc<SourceBindings>,
        slot: NaturalSlot,
        origin: InputOrigin,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<NaturalCallInput>> {
        not_cancelled(cancelled)?;
        self.model.service.ensure_usable()?;
        let source = record.source().clone();
        let call = record.call();
        if source.source_identity() != self.binding().source()
            || source.source_identity() != self.model.service.source_identity()
            || self.model.capacity != 512
            || self.model.vocab != 248320
            || call.rows().capacity() != 512
            || call.rows().vocabulary() != 248320
            || call.rows().ids().len() != slot.rows()
            || call.rows().ids().iter().any(|id| *id >= 248320)
            || self.binding().frontier() != slot.base() as u64
        {
            return Err(Error::Unsupported(
                "closed natural original current source/span".into(),
            ));
        }
        // Same original input-wrapper allowance; graph metadata belongs to record.
        let metadata = source.reserve_workspace(4 * 1024 * 1024)?;
        let attempt = self.authority.begin_advance(slot.rows() as u64)?;
        slot.require_current(self.binding(), &attempt, call)?;
        let schedule =
            GdnAttemptSchedule::new_published_call(&source, attempt.clone(), call.clone())?;
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

#[cfg(test)]
mod packet_tests {
    use super::*;
    fn words(id: u32, bits: u32) -> [u8; 12] {
        let mut bytes = [0; 12];
        bytes[..4].copy_from_slice(&id.to_le_bytes());
        bytes[4..8].copy_from_slice(&bits.to_le_bytes());
        bytes
    }
    #[test]
    fn immutable_feedback_words_cannot_substitute_id_bits_or_padding() {
        let id = 353;
        let bits = 0x3f80_0000;
        let raw = words(id, bits);
        require_packet_words(id, bits, id, bits, &raw).unwrap();
        assert!(require_packet_words(id, bits, id + 1, bits, &raw).is_err());
        assert!(require_packet_words(id, bits, id, bits + 0x10000, &raw).is_err());
        let mut wrong = raw;
        wrong[0] ^= 1;
        assert!(require_packet_words(id, bits, id, bits, &wrong).is_err());
        wrong = raw;
        wrong[4] ^= 1;
        assert!(require_packet_words(id, bits, id, bits, &wrong).is_err());
        wrong = raw;
        wrong[8] = 1;
        assert!(require_packet_words(id, bits, id, bits, &wrong).is_err());
        for bits in [0x3f80_0001, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000] {
            assert!(require_packet_words(id, bits, id, bits, &words(id, bits)).is_err());
        }
        assert!(require_packet_words(248320, 0, 248320, 0, &words(248320, 0)).is_err());
    }
}
