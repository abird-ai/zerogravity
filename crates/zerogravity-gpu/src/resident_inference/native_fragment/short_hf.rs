//! Actual source-issued short spans and original current-root ownership.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule, PrefixPhaseEnqueue};
use super::*;
use crate::vllm_cuda_qualification::short_hf_model::{ShortHfModelBinding, ShortHfModelBindings};
use std::sync::Weak;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    RecordedVllm08Model, ShortHfSpan, ShortHfWorkload, Vllm08Call, VllmBf16Mode,
};
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;

enum Origin {
    Prompt,
    Feedback(Arc<CommittedShortHfPacket>),
}

// No public constructor accepts ResidentToken.id or caller-selected sampled IDs.
pub(crate) struct CommittedShortHfPacket {
    workload: ShortHfWorkload,
    source: Arc<OriginalWindowModelSource>,
    model: Weak<ResidentModel>,
    token: ResidentToken,
    root: RootBinding,
    packet_bytes: [u8; 12],
    allocation_generation: u64,
    service_generation: u64,
}

impl CommittedShortHfPacket {
    fn require_current(
        &self,
        session: &ResidentSession,
        source: &BoundFullModelSource,
    ) -> Result<u32> {
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("short-HF feedback original model retired".into()))?;
        let id = u32::from_le_bytes(self.packet_bytes[..4].try_into().unwrap());
        let bits = u32::from_le_bytes(self.packet_bytes[4..8].try_into().unwrap());
        if !Arc::ptr_eq(&model, &session.model)
            || !Arc::ptr_eq(&self.source, source.source())
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
                "short-HF feedback is stale/foreign or not the original committed packet".into(),
            ));
        }
        session.model.service.ensure_usable()?;
        Ok(id)
    }
}

// No strong Model reference: native/current-flight owners cannot form a Body cycle.
pub(crate) struct ShortHfCallInput {
    model: Weak<ResidentModel>,
    source: BoundFullModelSource,
    service: Arc<ResidentService>,
    attempt: RootAttempt,
    scope: Arc<ShortHfModelBinding>,
    schedule: Arc<GdnAttemptSchedule>,
    phase: Arc<AttemptPhaseBinding>,
    origin: Origin,
    _metadata: WindowWorkspace,
}
impl ShortHfCallInput {
    pub(crate) fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    pub(crate) fn scope(&self) -> &Arc<ShortHfModelBinding> {
        &self.scope
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        self.scope.record()
    }
    pub(crate) fn call(&self) -> &Vllm08Call {
        self.record().call()
    }
    /// The original immutable committed packet, never a host-ID upload. Its
    /// genuine source/root/generation was checked when binding this call and is
    /// rechecked by require_session before any cohort permit is issued.
    pub(crate) fn feedback_lease(&self) -> Result<Option<&ResidentLease>> {
        match &self.origin {
            Origin::Prompt => Ok(None),
            Origin::Feedback(packet)
                if packet.root == self.attempt.base()
                    && packet.token.producer == Some(packet.root) =>
            {
                Ok(Some(&packet.token.lease))
            }
            Origin::Feedback(_) => {
                Err(Error::State("cohort feedback producer/root differs".into()))
            }
        }
    }
    pub(crate) fn ids(&self) -> &[u32] {
        self.call().request().teachers().ids()
    }
    pub(crate) fn attempt(&self) -> &RootAttempt {
        &self.attempt
    }
    pub(in crate::resident_inference) fn phase(&self) -> &Arc<AttemptPhaseBinding> {
        &self.phase
    }
    pub(crate) fn base(&self) -> usize {
        self.scope.call().span().base()
    }
    pub(crate) fn rows(&self) -> usize {
        self.scope.call().span().len()
    }
    pub(crate) fn workload(&self) -> ShortHfWorkload {
        self.scope.call().span().workload()
    }
    pub(crate) fn call_ordinal(&self) -> usize {
        if self.scope.call().span().mode() == VllmBf16Mode::Prefill {
            0
        } else {
            self.base() - self.workload().prompt_rows() + 1
        }
    }
    pub(crate) fn require_session(&self, session: &ResidentSession) -> Result<()> {
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("short-HF source model retired".into()))?;
        if !Arc::ptr_eq(&model, &session.model)
            || !Arc::ptr_eq(&self.service, &session.model.service)
            || !Arc::ptr_eq(self.source.source(), self.scope.source().source())
            || self.source.model_profile() != self.scope.call().model_profile()
            || self.service.source_identity() != self.call().source()
            || self.source.source().source_identity() != session.binding().source()
            || session.model.capacity != 512
            || session.model.vocab != 248320
        {
            return Err(Error::State(
                "short-HF SAME source/model/service/current call".into(),
            ));
        }
        self.scope.call().validate()?;
        self.phase
            .require_short_hf_binding(session, &self.attempt, self.call())?;
        match (&self.origin, self.scope.call().span().mode()) {
            (Origin::Prompt, VllmBf16Mode::Prefill) => {}
            (Origin::Feedback(packet), VllmBf16Mode::Decode) => {
                if packet.workload != self.workload()
                    || self.ids() != [packet.require_current(session, &self.source)?]
                {
                    return Err(Error::State(
                        "short-HF actual committed greedy feedback/workload differs".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::State(
                    "short-HF prompt/feedback origin differs".into(),
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
            .require_short_hf_permit(permit, service, self.call())
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
    pub(crate) fn bind_short_hf_prompt(
        &self,
        prepared: Arc<ShortHfModelBindings>,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ShortHfCallInput>> {
        let workload = ShortHfWorkload::from_prompt_rows(ids.len())?;
        let span = ShortHfSpan::prefill(self.binding().source(), workload, ids)?;
        self.bind_short_hf_input(prepared, span, Origin::Prompt, cancelled)
    }
    pub(crate) fn bind_short_hf_feedback(
        &self,
        prepared: Arc<ShortHfModelBindings>,
        packet: Arc<CommittedShortHfPacket>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ShortHfCallInput>> {
        not_cancelled(cancelled)?;
        let id = packet.require_current(self, prepared.source())?;
        let base = usize::try_from(self.binding().frontier())
            .map_err(|_| Error::Limit("short-HF root frontier".into()))?;
        let span = ShortHfSpan::cached(self.binding().source(), packet.workload, base, id)?;
        self.bind_short_hf_input(prepared, span, Origin::Feedback(packet), cancelled)
    }
    fn bind_short_hf_input(
        &self,
        prepared: Arc<ShortHfModelBindings>,
        span: ShortHfSpan,
        origin: Origin,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ShortHfCallInput>> {
        not_cancelled(cancelled)?;
        let source = prepared.source().clone();
        if source.source().source_identity() != self.binding().source()
            || self.model.capacity != 512
            || self.model.vocab != 248320
        {
            return Err(Error::State(
                "short-HF original source/session differs".into(),
            ));
        }
        let metadata = source
            .source()
            .source()
            .reserve_workspace(RecordedVllm08Model::BINDING_METADATA_BYTES)?;
        let bound = source.bind_short_hf_span(span)?;
        let record = Arc::new(bound.record_prepared()?);
        let scope = prepared.bind(bound, record)?;
        let attempt = self
            .authority
            .begin_advance(scope.call().span().len() as u64)?;
        let schedule = GdnAttemptSchedule::new_short_hf(&scope, attempt.clone())?;
        let phase = AttemptPhaseBinding::new(schedule.clone(), 0, None)?;
        let input = Arc::new(ShortHfCallInput {
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

mod state;
pub(crate) use state::ShortHfCallState;
mod readback;
pub use readback::ShortHfReadback;

mod loaded;
pub use loaded::{
    ShortHfCallReceipt, ShortHfCloseReceipt, ShortHfCohortFault, ShortHfCohortFaultOutcome,
    ShortHfCohortFaultRetention, ShortHfCohortPreparation, ShortHfCohortReadback,
    ShortHfCohortReceipt, ShortHfCohortRowReceipt, ShortHfEmptySnapshot, ShortHfFault,
    ShortHfFaultOutcome, ShortHfLoaded, ShortHfLoadedObservation, ShortHfLoadedSnapshot,
    ShortHfPending, ShortHfPendingCohort, ShortHfQualificationInputs, ShortHfResourceInventory,
    ShortHfSession, prepare_short_hf, prepare_short_hf_b4,
};
mod completion;
type ShortHfValidation<'a> = &'a mut dyn for<'b> FnMut(ShortHfReadback<'b>) -> Result<()>;
