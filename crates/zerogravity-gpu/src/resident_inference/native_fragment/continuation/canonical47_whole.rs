//! Exactly one initial47 original attempt,24layers; old3-layer scope untouched.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind;
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;
use zerogravity_core::accelerator::program::Vllm08Call;
impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_canonical47_whole(
        cut: &Arc<Canonical47PrefixCut>,
        attempt: RootAttempt,
    ) -> Result<Arc<Self>> {
        cut.require_whole()?;
        let source = cut.scope();
        if source.kind() != Canonical47CallKind::Prefill
            || attempt.base().source() != source.record().call().source()
            || attempt.base().source() != source.source().source().source_identity()
            || attempt.base().frontier() != 0
            || attempt.frontier() != 47
        {
            return Err(Error::Unsupported(
                "whole initial47 original RootAttempt only".into(),
            ));
        }
        source.source().source().require_vllm08_full_source()?;
        Ok(Arc::new(Self {
            base: attempt.base(),
            attempt,
            invocations: vec![source.record().call().request().clone()],
            completed_layers: 24,
        }))
    }
}
impl AttemptPhaseBinding {
    fn require_canonical47_whole_shape(&self, call: &Vllm08Call) -> Result<()> {
        if Canonical47CallKind::of(call.request())? != Canonical47CallKind::Prefill
            || self.phase_index != 0
            || self.prior.is_some()
            || self.schedule.completed_layers != 24
            || self.schedule.invocations.len() != 1
            || self.schedule.base.frontier() != 0
            || self.schedule.attempt.frontier() != 47
            || self.schedule.invocation(0)? != call.request()
        {
            return Err(Error::State(
                "whole24 exact original47 schedule/source issuer".into(),
            ));
        }
        Ok(())
    }
    pub(in crate::resident_inference) fn require_canonical47_whole_binding(
        &self,
        session: &ResidentSession,
        attempt: &RootAttempt,
        call: &Vllm08Call,
    ) -> Result<()> {
        self.require_canonical47_whole_shape(call)?;
        self.require_binding(session, attempt, call.request())
    }
    pub(crate) fn require_canonical47_whole_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
        call: &Vllm08Call,
    ) -> Result<()> {
        self.require_canonical47_whole_shape(call)?;
        self.require_permit(permit, service, call.request())
    }
    pub(in crate::resident_inference) fn same_canonical47_whole_request(
        &self,
        other: &Self,
    ) -> bool {
        self.schedule.completed_layers == 24
            && other.schedule.completed_layers == 24
            && self.phase_index == 0
            && other.phase_index == 0
            && self.prior.is_none()
            && other.prior.is_none()
            && self.schedule.invocations.len() == 1
            && other.schedule.invocations.len() == 1
            && self.schedule.base.frontier() == 0
            && self.schedule.attempt.frontier() == 47
            && Arc::ptr_eq(&self.schedule, &other.schedule)
    }
    pub(in crate::resident_inference) fn canonical47_whole_position_scope(&self) -> bool {
        self.schedule.completed_layers == 24
            && self.schedule.invocations.len() == 1
            && self.schedule.base.frontier() == 0
            && self.schedule.attempt.frontier() == 47
    }
}
