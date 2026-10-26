//! Exactly one initial47 original attempt,24layers; old3-layer scope untouched.
use super::*;
use crate::cuda::native::fixed_prompt_model::FixedPromptCallKind;
use crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut;
use zerogravity_core::accelerator::program::SourceCall;
impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_fixed_prompt_whole(
        cut: &Arc<FixedPromptPrefixCut>,
        attempt: RootAttempt,
    ) -> Result<Arc<Self>> {
        cut.require_whole()?;
        let source = cut.scope();
        if source.kind() != FixedPromptCallKind::Prefill
            || attempt.base().source() != source.record().call().source()
            || attempt.base().source() != source.source().source_identity()
            || attempt.base().frontier() != 0
            || attempt.frontier() != 47
        {
            return Err(Error::Unsupported(
                "whole initial47 original RootAttempt only".into(),
            ));
        }
        Ok(Arc::new(Self {
            base: attempt.base(),
            attempt,
            invocations: vec![source.record().call().clone()],
            completed_layers: 24,
        }))
    }
}
impl AttemptPhaseBinding {
    fn require_fixed_prompt_whole_shape(&self, call: &SourceCall) -> Result<()> {
        if FixedPromptCallKind::of(call)? != FixedPromptCallKind::Prefill
            || self.phase_index != 0
            || self.prior.is_some()
            || self.schedule.completed_layers != 24
            || self.schedule.invocations.len() != 1
            || self.schedule.base.frontier() != 0
            || self.schedule.attempt.frontier() != 47
            || self.schedule.invocation(0)? != call
        {
            return Err(Error::State(
                "whole24 exact original47 schedule/source issuer".into(),
            ));
        }
        Ok(())
    }
    pub(in crate::resident_inference) fn require_fixed_prompt_whole_binding(
        &self,
        session: &ResidentSession,
        attempt: &RootAttempt,
        call: &SourceCall,
    ) -> Result<()> {
        self.require_fixed_prompt_whole_shape(call)?;
        self.require_binding(session, attempt, call, 0)
    }
    pub(crate) fn require_fixed_prompt_whole_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
        call: &SourceCall,
    ) -> Result<()> {
        self.require_fixed_prompt_whole_shape(call)?;
        self.require_permit(permit, service, call, 0)
    }
    pub(in crate::resident_inference) fn same_fixed_prompt_whole_request(
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
    pub(in crate::resident_inference) fn fixed_prompt_whole_position_scope(&self) -> bool {
        self.schedule.completed_layers == 24
            && self.schedule.invocations.len() == 1
            && self.schedule.base.frontier() == 0
            && self.schedule.attempt.frontier() == 47
    }
}
