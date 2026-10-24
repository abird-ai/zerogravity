//! Source-issued whole call on the original current RootAttempt.
use super::*;
use crate::vllm_cuda_qualification::short_hf_model::ShortHfModelBinding;
use zerogravity_core::accelerator::program::{ShortHfCall, Vllm08Call};

fn coordinates(base: RootBinding, attempt: &RootAttempt, call: &ShortHfCall) -> Result<()> {
    call.validate()?;
    let start = call.span().base();
    let rows = call.span().len();
    if attempt.base() != base
        || call.span().source() != base.source()
        || base.frontier() != start as u64
        || attempt.frontier() != (start + rows) as u64
    {
        return Err(Error::State(
            "short-HF actual current RootAttempt coordinates".into(),
        ));
    }
    Ok(())
}
impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_short_hf(
        scope: &Arc<ShortHfModelBinding>,
        attempt: RootAttempt,
    ) -> Result<Arc<Self>> {
        if scope.source().model_profile() != scope.call().model_profile()
            || scope.source().source().source_identity() != scope.record().call().source()
        {
            return Err(Error::State("short-HF exact model/source schedule".into()));
        }
        coordinates(attempt.base(), &attempt, scope.call())?;
        Ok(Arc::new(Self {
            base: attempt.base(),
            attempt,
            invocations: vec![scope.call().request().clone()],
            completed_layers: 24,
        }))
    }
}
impl AttemptPhaseBinding {
    fn require_short_hf_shape(&self, call: &Vllm08Call) -> Result<()> {
        let span = call
            .request()
            .teachers()
            .short_hf_span()
            .ok_or_else(|| Error::State("short-HF real semantic issuer absent".into()))?;
        let typed = ShortHfCall::new(call.model_profile(), span.clone())?;
        if typed.as_vllm_call() != call
            || self.phase_index != 0
            || self.prior.is_some()
            || self.schedule.completed_layers != 24
            || self.schedule.invocations.len() != 1
            || self.schedule.invocation(0)? != call.request()
        {
            return Err(Error::State(
                "short-HF exact one actual whole-call schedule".into(),
            ));
        }
        coordinates(self.schedule.base, &self.schedule.attempt, &typed)
    }
    pub(in crate::resident_inference) fn require_short_hf_binding(
        &self,
        session: &ResidentSession,
        attempt: &RootAttempt,
        call: &Vllm08Call,
    ) -> Result<()> {
        self.require_short_hf_shape(call)?;
        self.require_binding(session, attempt, call.request())
    }
    pub(crate) fn require_short_hf_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
        call: &Vllm08Call,
    ) -> Result<()> {
        self.require_short_hf_shape(call)?;
        self.require_permit(permit, service, call.request())
    }
    pub(in crate::resident_inference) fn same_short_hf_request(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.schedule, &other.schedule)
            && self.phase_index == 0
            && other.phase_index == 0
            && self.prior.is_none()
            && other.prior.is_none()
            && self.schedule.completed_layers == 24
            && self.schedule.invocations.len() == 1
    }
    pub(in crate::resident_inference) fn require_retained_short_hf(
        &self,
        base: RootBinding,
        call: &Vllm08Call,
    ) -> Result<()> {
        self.require_short_hf_shape(call)?;
        if self.schedule.base != base {
            return Err(Error::State(
                "short-HF retained original root differs".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::{
        accelerator::{
            SourceIdentity,
            program::{FullModelProfile, ShortHfSpan, ShortHfWorkload},
        },
        session_root::SessionAuthority,
    };
    #[test]
    fn actual_five_prompts_and_seven_feedback_advances_keep_original_authority() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        for rows in [47, 57, 62, 67, 72] {
            let workload = ShortHfWorkload::from_prompt_rows(rows)?;
            for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
                let mut authority = SessionAuthority::new(source)?;
                let call = ShortHfCall::new(
                    profile,
                    ShortHfSpan::prefill(source, workload, &vec![11; rows])?,
                )?;
                let attempt = authority.begin_advance(rows as u64)?;
                coordinates(authority.binding(), &attempt, &call)?;
                assert_eq!(authority.binding().frontier(), 0);
                authority.prepare_publish(attempt)?.commit();
                for base in workload.cached_bases() {
                    let before = authority.binding();
                    let attempt = authority.begin_advance(1)?;
                    let call = ShortHfCall::new(
                        profile,
                        ShortHfSpan::cached(source, workload, base, 11)?,
                    )?;
                    coordinates(before, &attempt, &call)?;
                    assert_eq!(authority.binding(), before);
                    authority.prepare_publish(attempt)?.commit();
                }
                assert_eq!(authority.binding().frontier(), (rows + 7) as u64);
            }
        }
        Ok(())
    }
    #[test]
    fn source_session_wrong_advance_and_stale_coordinates_refuse() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let mut a = SessionAuthority::new(source)?;
        let b = SessionAuthority::new(source)?;
        let call = ShortHfCall::new(
            FullModelProfile::Qwen2B,
            ShortHfSpan::prefill(source, ShortHfWorkload::Prompt72, &[11; 72])?,
        )?;
        assert!(coordinates(a.binding(), &b.begin_advance(72)?, &call).is_err());
        assert!(coordinates(a.binding(), &a.begin_advance(71)?, &call).is_err());
        let attempt = a.begin_advance(72)?;
        coordinates(a.binding(), &attempt, &call)?;
        a.prepare_publish(attempt.clone())?.commit();
        assert!(coordinates(a.binding(), &attempt, &call).is_err());
        Ok(())
    }
}
