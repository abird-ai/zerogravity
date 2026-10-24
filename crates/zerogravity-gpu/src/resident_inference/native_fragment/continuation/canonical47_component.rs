//! One genuine unpublished47 attempt; never RootReady/commit/frontier promotion.
use super::*;
use crate::resident_program::qwen2b_component::ComponentKind;
use crate::vllm_cuda_gdn::Canonical47GdnArithmetic;
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;
impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_initial_canonical47_component(
        source: &BoundFullModelSource,
        attempt: RootAttempt,
        invocation: VllmBf16Invocation,
        arithmetic: Canonical47GdnArithmetic,
    ) -> Result<Arc<Self>> {
        let kind = ComponentKind::Canonical47(arithmetic);
        kind.require_request(&invocation)?;
        source.source().require_vllm08_full_source()?;
        let base = attempt.base();
        if source.model_profile() != kind.profile()
            || base.source() != source.source().source_identity()
            || invocation.layer().source() != base.source()
            || base.frontier() != 0
            || attempt.frontier() != 47
        {
            return Err(Error::Unsupported(
                "genuine current original47 component attempt only".into(),
            ));
        }
        Ok(Arc::new(Self {
            base,
            attempt,
            invocations: vec![invocation],
            completed_layers: 1,
        }))
    }
}
impl AttemptPhaseBinding {
    pub(in crate::resident_inference) fn require_initial_canonical47_component(
        &self,
        invocation: &VllmBf16Invocation,
    ) -> Result<()> {
        ComponentKind::Canonical47(Canonical47GdnArithmetic::OwnedCanonical47Solve4)
            .require_request(invocation)?;
        if self.phase_index != 0
            || self.prior.is_some()
            || self.schedule.completed_layers != 1
            || self.schedule.invocations.len() != 1
            || self.schedule.base.frontier() != 0
            || self.schedule.attempt.frontier() != 47
            || self.schedule.invocation(0)? != invocation
        {
            return Err(Error::State(
                "exact original47 unpublished schedule binding required".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident_program::qwen2b_component::CANONICAL47_PROMPT;
    use zerogravity_core::{
        accelerator::{
            SourceIdentity,
            program::{GdnPrefixLayer, TeacherSpan, VllmBf16Mode, VllmBf16Teachers},
        },
        session_root::SessionAuthority,
    };
    fn invocation(source: SourceIdentity) -> VllmBf16Invocation {
        VllmBf16Invocation::from_span(
            GdnPrefixLayer::new(source, 0).unwrap(),
            VllmBf16Mode::Prefill,
            VllmBf16Teachers::from_span(
                TeacherSpan::new(source, 0, 512, 248320, &CANONICAL47_PROMPT).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn make_binding(
        layers: usize,
    ) -> (
        SessionAuthority,
        Arc<AttemptPhaseBinding>,
        VllmBf16Invocation,
    ) {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let authority = SessionAuthority::new(source).unwrap();
        let request = invocation(source);
        let attempt = authority.begin_advance(47).unwrap();
        let schedule = Arc::new(GdnAttemptSchedule {
            base: authority.binding(),
            attempt,
            invocations: vec![request.clone()],
            completed_layers: layers,
        });
        let binding = AttemptPhaseBinding::new(schedule, 0, None).unwrap();
        (authority, binding, request)
    }
    #[test]
    fn original47_component_never_issues_or_publishes_a_root() {
        let (authority, binding, request) = make_binding(1);
        binding
            .require_initial_canonical47_component(&request)
            .unwrap();
        assert_eq!(authority.binding().frontier(), 0);
        assert_eq!(authority.binding().root(), 0);
        assert_eq!(binding.schedule.attempt.frontier(), 47);
        let new_issuer = invocation(authority.binding().source());
        assert_ne!(request.teachers(), new_issuer.teachers());
        assert!(
            binding
                .require_initial_canonical47_component(&new_issuer)
                .is_err()
        );
        let (_, full, request) = make_binding(24);
        assert!(
            full.require_initial_canonical47_component(&request)
                .is_err()
        );
    }
}
