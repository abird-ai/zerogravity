//! Fresh immutable inputs for original47+7 calls; completion/commit owns feedback.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule, PrefixPhaseEnqueue};
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::{
    Canonical47CallKind, Canonical47ModelBinding,
};
use std::sync::Weak;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    GdnPrefixLayer, RecordedVllm08Model, TeacherSpan, Vllm08Call, VllmBf16Mode, VllmBf16Teachers,
};
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;

mod state;
pub(crate) use state::CanonicalCallState;
mod readback;
pub use readback::Canonical47NaturalReadback;

enum Origin {
    Prompt,
    Feedback(Arc<CommittedPacket>),
}

// No public constructor accepts ResidentToken.id or caller-selected sampled IDs.
pub(crate) struct CommittedPacket {
    source: Arc<OriginalWindowModelSource>,
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
        source: &BoundFullModelSource,
    ) -> Result<u32> {
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("canonical feedback original model retired".into()))?;
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
                "canonical feedback is stale/foreign or not the original committed packet".into(),
            ));
        }
        session.model.service.ensure_usable()?;
        Ok(id)
    }
}

// Weak<Model> prevents workspace→Body→input→Model→workspace cycles.
pub(crate) struct CanonicalCallInput {
    model: Weak<ResidentModel>,
    source: BoundFullModelSource,
    service: Arc<ResidentService>,
    attempt: RootAttempt,
    scope: Arc<Canonical47ModelBinding>,
    schedule: Arc<GdnAttemptSchedule>,
    phase: Arc<AttemptPhaseBinding>,
    origin: Origin,
    _metadata: WindowWorkspace,
}

impl CanonicalCallInput {
    pub(crate) fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    pub(crate) fn scope(&self) -> &Arc<Canonical47ModelBinding> {
        &self.scope
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        self.scope.record()
    }
    pub(crate) fn call(&self) -> &Vllm08Call {
        self.record().call()
    }
    pub(crate) fn ids(&self) -> &[u32] {
        self.call().request().teachers().ids()
    }
    pub(crate) fn attempt(&self) -> &RootAttempt {
        &self.attempt
    }
    pub(crate) fn kind(&self) -> Canonical47CallKind {
        self.scope.kind()
    }
    pub(crate) fn base(&self) -> usize {
        self.call().request().teachers().base()
    }
    pub(crate) fn rows(&self) -> usize {
        self.call().rows()
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
            || !Arc::ptr_eq(self.source.source(), self.scope.source().source())
            || self.service.source_identity() != self.call().source()
            || self.source.source().source_identity() != session.binding().source()
            || session.model.capacity != 512
            || session.model.vocab != 248320
            || self.kind() != Canonical47CallKind::of(self.call().request())?
        {
            return Err(Error::State(
                "canonical immutable source/model/service/call differs".into(),
            ));
        }
        self.phase
            .require_canonical47_published_binding(session, &self.attempt, self.call())?;
        match (&self.origin, self.kind()) {
            (Origin::Prompt, Canonical47CallKind::Prefill) => {}
            (Origin::Feedback(packet), Canonical47CallKind::Decode { .. }) => {
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
            .require_canonical47_published_permit(permit, service, self.call())
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
    pub(crate) fn bind_canonical47_prompt(
        &self,
        source: BoundFullModelSource,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<CanonicalCallInput>> {
        self.bind_canonical47_input(
            source,
            VllmBf16Mode::Prefill,
            ids,
            Origin::Prompt,
            cancelled,
        )
    }

    pub(crate) fn bind_canonical47_feedback(
        &self,
        source: BoundFullModelSource,
        packet: Arc<CommittedPacket>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<CanonicalCallInput>> {
        not_cancelled(cancelled)?;
        let id = packet.require_current(self, &source)?;
        self.bind_canonical47_input(
            source,
            VllmBf16Mode::Decode,
            &[id],
            Origin::Feedback(packet),
            cancelled,
        )
    }

    fn bind_canonical47_input(
        &self,
        source: BoundFullModelSource,
        mode: VllmBf16Mode,
        ids: &[u32],
        origin: Origin,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<CanonicalCallInput>> {
        not_cancelled(cancelled)?;
        source.source().require_vllm08_full_source()?;
        if source.source().source_identity() != self.binding().source()
            || self.model.capacity != 512
            || self.model.vocab != 248320
        {
            return Err(Error::Unsupported(
                "canonical original .8B source/session only".into(),
            ));
        }
        let base = usize::try_from(self.binding().frontier())
            .map_err(|_| Error::Limit("canonical frontier extent".into()))?;
        // Admit the actual new recipe/control metadata before minting its teacher issuer.
        let metadata = source
            .source()
            .source()
            .reserve_workspace(4 * 1024 * 1024)?;
        let teachers = if ids.len() == 1 {
            VllmBf16Teachers::new(self.binding().source(), base, self.model.capacity, ids)?
        } else {
            VllmBf16Teachers::from_span(TeacherSpan::new(
                self.binding().source(),
                base,
                self.model.capacity,
                248320,
                ids,
            )?)?
        };
        let request = VllmBf16Invocation::from_span(
            GdnPrefixLayer::new(self.binding().source(), 0)?,
            mode,
            teachers,
        )?;
        Canonical47CallKind::of(&request)?;
        let attempt = self.authority.begin_advance(ids.len() as u64)?;
        let record = Arc::new(
            source
                .source()
                .record_vllm08_model(Vllm08Call::from_span(request)?)?,
        );
        let scope = Canonical47ModelBinding::bind(source.clone(), record)?;
        let schedule = GdnAttemptSchedule::new_canonical47_published(&scope, attempt.clone())?;
        let phase = AttemptPhaseBinding::new(schedule.clone(), 0, None)?;
        let input = Arc::new(CanonicalCallInput {
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
    &'a mut (dyn for<'b> FnMut(Canonical47NaturalReadback<'b>) -> Result<()> + 'a);

mod completion;
mod loaded;
pub use loaded::{
    Canonical47CallReceipt, Canonical47CloseReceipt, Canonical47EmptySnapshot,
    Canonical47LifecycleProfile, Canonical47Loaded, Canonical47LoadedObservation,
    Canonical47LoadedSnapshot, Canonical47NaturalFault, Canonical47NaturalFaultOutcome,
    Canonical47Pending, Canonical47Session, canonical47_natural_host_bound,
    prepare_canonical47_natural, prepare_canonical47_natural_with_profile,
};
