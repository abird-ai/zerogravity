//! Fresh immutable inputs for original47+7 calls; completion/commit owns feedback.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule, PrefixPhaseEnqueue};
use super::*;
use crate::cuda::native::fixed_prompt_model::{FixedPromptCallKind, FixedPromptModelBinding};
use crate::resident_program::source_bindings::SourceBindings;
use std::sync::Weak;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::SourceCall;
use zerogravity_core::accelerator::program::SourceCallMode;

mod state;
pub(crate) use state::FixedPromptCallState;
mod readback;
pub use readback::FixedPromptNaturalReadback;

enum Origin {
    Prompt,
    Feedback(Arc<CommittedPacket>),
}

// No public constructor accepts ResidentToken.id or caller-selected sampled IDs.
pub(crate) struct CommittedPacket {
    source: VerifiedTensorSource,
    model: Weak<ResidentModel>,
    token: ResidentToken,
    root: RootBinding,
    packet_bytes: [u8; 12],
    allocation_generation: u64,
    service_generation: u64,
}

impl CommittedPacket {
    fn require_current(
        &self,
        session: &ResidentSession,
        source: &VerifiedTensorSource,
    ) -> Result<u32> {
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("canonical feedback original model retired".into()))?;
        let id = u32::from_le_bytes(self.packet_bytes[..4].try_into().unwrap());
        let bits = u32::from_le_bytes(self.packet_bytes[4..8].try_into().unwrap());
        if !Arc::ptr_eq(&model, &session.model)
            || !self.source.same_issuer(source)
            || self.root != session.binding()
            || self.token.producer != Some(self.root)
            || id != self.token.id
            || bits != self.token.max_bits
            || id >= 248320
            || bits & 0xffff != 0
            || bits & 0x7f80_0000 == 0x7f80_0000
            || self.packet_bytes[8..] != [0; 4]
            || !Arc::ptr_eq(self.token.lease.service(), &session.model.service)
            || self.token.lease.role() != BufferRole::Output
            || self.token.lease.bytes() != 12
            || !self.token.lease.is_initialized()
            || self.token.lease.allocation_generation() != self.allocation_generation
            || self.token.lease.service_generation() != self.service_generation
        {
            return Err(Error::State(
                "canonical feedback is stale/foreign or not the original committed packet".into(),
            ));
        }
        session.model.service.ensure_usable()?;
        Ok(id)
    }
}

/// Borrowed from the actual committed packet; model preparation gains no root permit.
pub struct FixedPromptFeedbackRequest<'a> {
    session: &'a FixedPromptSession,
    packet: &'a Arc<CommittedPacket>,
    base: usize,
}
impl FixedPromptFeedbackRequest<'_> {
    pub fn selected_id(&self) -> u32 {
        self.packet.token.id
    }
    pub fn base(&self) -> usize {
        self.base
    }
    fn require(
        &self,
        session: &FixedPromptSession,
        source: &VerifiedTensorSource,
    ) -> Result<&Arc<CommittedPacket>> {
        if !std::ptr::eq(self.session, session)
            || !session
                .feedback
                .as_ref()
                .is_some_and(|p| Arc::ptr_eq(p, self.packet))
            || self.base as u64 != session.binding().frontier()
        {
            return Err(Error::State(
                "fixed prompt feedback request is stale/foreign".into(),
            ));
        }
        self.packet.require_current(&session.original, source)?;
        Ok(self.packet)
    }
}

pub(super) fn tied_member(
    record: &SourceBindings,
) -> Result<zerogravity_core::accelerator::program::SourceTensor> {
    use zerogravity_core::accelerator::program::SourceOperatorKind;
    let op = record
        .operator(0)?
        .ok_or_else(|| Error::State("actual embedding operator absent".into()))?;
    if op.kind() != SourceOperatorKind::EmbeddingRows || op.matrices().len() != 1 {
        return Err(Error::State(
            "actual single embedding source member required".into(),
        ));
    }
    Ok(op.matrices()[0].clone())
}

// Weak<Model> prevents workspace→Body→input→Model→workspace cycles.
pub(crate) struct FixedPromptCallInput {
    model: Weak<ResidentModel>,
    source: VerifiedTensorSource,
    service: Arc<ResidentService>,
    attempt: RootAttempt,
    scope: Arc<FixedPromptModelBinding>,
    schedule: Arc<GdnAttemptSchedule>,
    phase: Arc<AttemptPhaseBinding>,
    origin: Origin,
    _metadata: WindowWorkspace,
}

impl FixedPromptCallInput {
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn scope(&self) -> &Arc<FixedPromptModelBinding> {
        &self.scope
    }
    pub(crate) fn record(&self) -> &Arc<SourceBindings> {
        self.scope.record()
    }
    pub(crate) fn call(&self) -> &SourceCall {
        self.record().call()
    }
    pub(crate) fn ids(&self) -> &[u32] {
        self.call().rows().ids()
    }
    pub(crate) fn attempt(&self) -> &RootAttempt {
        &self.attempt
    }
    pub(crate) fn kind(&self) -> FixedPromptCallKind {
        self.scope.kind()
    }
    pub(crate) fn base(&self) -> usize {
        self.call().rows().base()
    }
    pub(crate) fn rows(&self) -> usize {
        self.call().rows().len()
    }
    pub(in crate::resident_inference) fn phase(&self) -> &Arc<AttemptPhaseBinding> {
        &self.phase
    }

    pub(crate) fn require_session(&self, session: &ResidentSession) -> Result<()> {
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("canonical input original model retired".into()))?;
        if !Arc::ptr_eq(&model, &session.model)
            || !Arc::ptr_eq(&self.service, &session.model.service)
            || !self.source.same_issuer(self.scope.source())
            || self.service.source_identity() != self.call().source()
            || self.source.source_identity() != session.binding().source()
            || session.model.capacity != 512
            || session.model.vocab != 248320
            || self.kind() != FixedPromptCallKind::of(self.call())?
        {
            return Err(Error::State(
                "canonical immutable source/model/service/call differs".into(),
            ));
        }
        self.phase
            .require_fixed_prompt_published_binding(session, &self.attempt, self.call())?;
        match (&self.origin, self.kind()) {
            (Origin::Prompt, FixedPromptCallKind::Prefill) => {}
            (Origin::Feedback(packet), FixedPromptCallKind::Decode { .. }) => {
                if self.ids() != [packet.require_current(session, &self.source)?] {
                    return Err(Error::State(
                        "canonical input is not actual completed BF16 feedback".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::State(
                    "canonical prompt/feedback origin differs".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn require_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
    ) -> Result<()> {
        self.phase
            .require_fixed_prompt_published_permit(permit, service, self.call())
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
    pub(crate) fn bind_fixed_prompt_prompt(
        &self,
        source: VerifiedTensorSource,
        record: Arc<SourceBindings>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<FixedPromptCallInput>> {
        self.bind_fixed_prompt_input(
            source,
            record,
            SourceCallMode::Prefill,
            Origin::Prompt,
            cancelled,
        )
    }

    pub(crate) fn bind_fixed_prompt_feedback(
        &self,
        source: VerifiedTensorSource,
        record: Arc<SourceBindings>,
        packet: Arc<CommittedPacket>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<FixedPromptCallInput>> {
        not_cancelled(cancelled)?;
        let id = packet.require_current(self, &source)?;
        if record.call().rows().ids() != [id] {
            return Err(Error::State(
                "fixed prompt call is not actual committed feedback".into(),
            ));
        }
        self.bind_fixed_prompt_input(
            source,
            record,
            SourceCallMode::Decode,
            Origin::Feedback(packet),
            cancelled,
        )
    }

    fn bind_fixed_prompt_input(
        &self,
        source: VerifiedTensorSource,
        record: Arc<SourceBindings>,
        mode: SourceCallMode,
        origin: Origin,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<FixedPromptCallInput>> {
        not_cancelled(cancelled)?;
        let call = record.call();
        if !source.same_issuer(record.source())
            || source.source_identity() != self.binding().source()
            || self.model.capacity != 512
            || self.model.vocab != 248320
            || call.mode() != mode
            || call.rows().base() as u64 != self.binding().frontier()
        {
            return Err(Error::State(
                "fixed prompt actual source/call/current root differs".into(),
            ));
        }
        FixedPromptCallKind::of(call)?;
        let metadata = source.reserve_workspace(4 * 1024 * 1024)?;
        let attempt = self.authority.begin_advance(call.rows().len() as u64)?;
        let scope = FixedPromptModelBinding::bind(record)?;
        let schedule = GdnAttemptSchedule::new_fixed_prompt_published(&scope, attempt.clone())?;
        let phase = AttemptPhaseBinding::new(schedule.clone(), 0, None)?;
        let input = Arc::new(FixedPromptCallInput {
            model: Arc::downgrade(&self.model),
            source,
            service: self.model.service.clone(),
            attempt,
            scope,
            schedule,
            phase,
            origin,
            _metadata: metadata,
        });
        input.require_session(self)?;
        Ok(input)
    }
}

pub(super) type NaturalValidation<'a> =
    &'a mut (dyn for<'b> FnMut(FixedPromptNaturalReadback<'b>) -> Result<()> + 'a);

mod completion;
mod loaded;
pub use loaded::{
    FixedPromptCallReceipt, FixedPromptCloseReceipt, FixedPromptEmptySnapshot,
    FixedPromptLifecycleProfile, FixedPromptLoaded, FixedPromptLoadedObservation,
    FixedPromptLoadedSnapshot, FixedPromptNaturalFault, FixedPromptNaturalFaultOutcome,
    FixedPromptPending, FixedPromptSession, fixed_prompt_natural_host_bound,
    prepare_fixed_prompt_natural, prepare_fixed_prompt_natural_with_profile,
};
