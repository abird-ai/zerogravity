//! Genuine one-call Prefill47 attempt, exactly three GDN layers, no publication.
use super::*;
use crate::cuda::native::fixed_prompt_model::{FixedPromptCallKind, FixedPromptModelBinding};
use zerogravity_core::accelerator::program::SourceCall;
impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_fixed_prompt_model_prefix(
        source: &FixedPromptModelBinding,
        attempt: RootAttempt,
    ) -> Result<Arc<Self>> {
        if source.kind() != FixedPromptCallKind::Prefill
            || source.record().call().source() != attempt.base().source()
            || attempt.base().source() != source.source().source_identity()
            || attempt.base().frontier() != 0
            || attempt.frontier() != 47
        {
            return Err(Error::Unsupported(
                "genuine canonical47 first3 initial model attempt only".into(),
            ));
        }
        Ok(Arc::new(Self {
            base: attempt.base(),
            attempt,
            invocations: vec![source.record().call().clone()],
            completed_layers: 3,
        }))
    }
}
impl AttemptPhaseBinding {
    fn require_fixed_prompt_model_prefix_shape(&self, call: &SourceCall) -> Result<()> {
        if FixedPromptCallKind::of(call)? != FixedPromptCallKind::Prefill
            || self.phase_index != 0
            || self.prior.is_some()
            || self.schedule.completed_layers != 3
            || self.schedule.invocations.len() != 1
            || self.schedule.base.frontier() != 0
            || self.schedule.attempt.frontier() != 47
            || self.schedule.invocation(0)? != call
        {
            return Err(Error::State(
                "exact original canonical47 first3 model schedule required".into(),
            ));
        }
        Ok(())
    }
    pub(in crate::resident_inference) fn require_fixed_prompt_model_prefix_binding(
        &self,
        session: &ResidentSession,
        attempt: &RootAttempt,
        call: &SourceCall,
    ) -> Result<()> {
        self.require_fixed_prompt_model_prefix_shape(call)?;
        self.require_binding(session, attempt, call, 0)
    }
    pub(crate) fn require_fixed_prompt_model_prefix_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
        call: &SourceCall,
    ) -> Result<()> {
        self.require_fixed_prompt_model_prefix_shape(call)?;
        self.require_permit(permit, service, call, 0)
    }
    pub(in crate::resident_inference) fn same_fixed_prompt_model_prefix_request(
        &self,
        other: &Self,
    ) -> bool {
        self.schedule.completed_layers == 3
            && other.schedule.completed_layers == 3
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::SourceCallMode;
    use zerogravity_core::accelerator::program::TeacherSpan;
    fn call(source: SourceIdentity) -> SourceCall {
        let rows = zerogravity_core::accelerator::program::SourceRows::from_teacher_span(
            TeacherSpan::new(source, 0, 512, 248320, &[11; 47]).unwrap(),
            248320,
        )
        .unwrap();
        SourceCall::new(rows, SourceCallMode::Prefill).unwrap()
    }
    fn schedule(call: &SourceCall, layers: usize, tokens: u64) -> Arc<GdnAttemptSchedule> {
        let authority = SessionAuthority::new(call.source()).unwrap();
        Arc::new(GdnAttemptSchedule {
            base: authority.binding(),
            attempt: authority.begin_advance(tokens).unwrap(),
            invocations: vec![call.clone()],
            completed_layers: layers,
        })
    }
    #[test]
    fn exact_first3_shape_is_separate_from_full24_and_wrong_extent() {
        let call = call(SourceIdentity::for_new_allocation().unwrap());
        let actual = AttemptPhaseBinding::new(schedule(&call, 3, 47), 0, None).unwrap();
        actual
            .require_fixed_prompt_model_prefix_shape(&call)
            .unwrap();
        for (layers, tokens) in [(1, 47), (24, 47), (3, 46), (3, 48)] {
            assert!(
                AttemptPhaseBinding::new(schedule(&call, layers, tokens), 0, None)
                    .unwrap()
                    .require_fixed_prompt_model_prefix_shape(&call)
                    .is_err()
            );
        }
        assert_eq!(call.rows().len(), 47);
    }
    #[test]
    fn equal_teacher_words_from_new_issuer_cannot_replace_scheduled_call() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let a = call(source);
        let b = call(source);
        assert_eq!(a.rows().ids(), b.rows().ids());
        assert_ne!(a, b);
        let binding = AttemptPhaseBinding::new(schedule(&a, 3, 47), 0, None).unwrap();
        assert!(binding.require_fixed_prompt_model_prefix_shape(&b).is_err());
    }
    #[test]
    fn state_cache_scope_requires_same_original_schedule_arc() {
        let call = call(SourceIdentity::for_new_allocation().unwrap());
        let original = schedule(&call, 3, 47);
        let a = AttemptPhaseBinding::new(original.clone(), 0, None).unwrap();
        let b = AttemptPhaseBinding::new(original, 0, None).unwrap();
        assert!(a.same_fixed_prompt_model_prefix_request(&b));
        let foreign = AttemptPhaseBinding::new(schedule(&call, 3, 47), 0, None).unwrap();
        assert!(!a.same_fixed_prompt_model_prefix_request(&foreign));
    }
}
