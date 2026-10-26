//! Source-issued whole call on the original current RootAttempt.
use super::*;
use crate::cuda::native::text_model::TextModelBinding;
use zerogravity_core::accelerator::program::SourceCall;

fn coordinates(base: RootBinding, attempt: &RootAttempt, call: &SourceCall) -> Result<()> {
    let start = call.rows().base();
    let rows = call.rows().len();
    if attempt.base() != base
        || call.source() != base.source()
        || base.frontier() != start as u64
        || attempt.frontier() != (start + rows) as u64
    {
        return Err(Error::State(
            "text actual current RootAttempt coordinates".into(),
        ));
    }
    Ok(())
}
impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_text(
        scope: &Arc<TextModelBinding>,
        attempt: RootAttempt,
    ) -> Result<Arc<Self>> {
        if scope.source().source_identity() != scope.record().call().source()
            || scope.call() != scope.record().call()
            || scope.record().selection().completed_layers() != 24
            || scope.gdn_sites().len() + scope.attention_sites().len() != 24
        {
            return Err(Error::State(
                "text exact source/whole-graph schedule".into(),
            ));
        }
        coordinates(attempt.base(), &attempt, scope.call())?;
        Ok(Arc::new(Self {
            base: attempt.base(),
            attempt,
            invocations: vec![scope.call().clone()],
            completed_layers: 24,
        }))
    }
}
impl AttemptPhaseBinding {
    fn require_prompt_shape(&self, call: &SourceCall) -> Result<()> {
        if self.phase_index != 0
            || self.prior.is_some()
            || self.schedule.completed_layers != 24
            || self.schedule.invocations.len() != 1
            || self.schedule.invocation(0)? != call
        {
            return Err(Error::State(
                "text exact one actual whole-call schedule".into(),
            ));
        }
        coordinates(self.schedule.base, &self.schedule.attempt, call)
    }
    pub(in crate::resident_inference) fn require_text_binding(
        &self,
        session: &ResidentSession,
        attempt: &RootAttempt,
        call: &SourceCall,
    ) -> Result<()> {
        self.require_prompt_shape(call)?;
        self.require_binding(session, attempt, call, 0)
    }
    pub(crate) fn require_text_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
        call: &SourceCall,
    ) -> Result<()> {
        self.require_prompt_shape(call)?;
        self.require_permit(permit, service, call, 0)
    }
    pub(in crate::resident_inference) fn same_text_request(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.schedule, &other.schedule)
            && self.phase_index == 0
            && other.phase_index == 0
            && self.prior.is_none()
            && other.prior.is_none()
            && self.schedule.completed_layers == 24
            && self.schedule.invocations.len() == 1
    }
    pub(in crate::resident_inference) fn require_retained_text(
        &self,
        base: RootBinding,
        call: &SourceCall,
    ) -> Result<()> {
        self.require_prompt_shape(call)?;
        if self.schedule.base != base {
            return Err(Error::State("text retained original root differs".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::{SourceCallMode, SourceRows, TextCallGeometry};
    use zerogravity_core::session_root::SessionAuthority;
    fn call(
        source: SourceIdentity,
        mode: SourceCallMode,
        base: usize,
        ids: &[u32],
    ) -> Result<SourceCall> {
        SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, 248320, 512)?,
                base,
                Arc::from(ids),
            )?,
            mode,
        )
    }
    #[test]
    fn five_prompts_and_seven_feedback_advances_keep_original_authority() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        for rows in [47, 57, 62, 67, 72] {
            let mut authority = SessionAuthority::new(source)?;
            let prompt = call(source, SourceCallMode::Prefill, 0, &vec![11; rows])?;
            let attempt = authority.begin_advance(rows as u64)?;
            coordinates(authority.binding(), &attempt, &prompt)?;
            assert_eq!(authority.binding().frontier(), 0);
            authority.prepare_publish(attempt)?.commit();
            for base in rows..rows + 7 {
                let before = authority.binding();
                let attempt = authority.begin_advance(1)?;
                let feedback = call(source, SourceCallMode::Decode, base, &[11])?;
                coordinates(before, &attempt, &feedback)?;
                assert_eq!(authority.binding(), before);
                authority.prepare_publish(attempt)?.commit();
            }
            assert_eq!(authority.binding().frontier(), (rows + 7) as u64);
        }
        Ok(())
    }
    #[test]
    fn source_session_wrong_advance_and_stale_coordinates_refuse() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let mut a = SessionAuthority::new(source)?;
        let b = SessionAuthority::new(source)?;
        let c = call(source, SourceCallMode::Prefill, 0, &[11; 72])?;
        assert!(coordinates(a.binding(), &b.begin_advance(72)?, &c).is_err());
        assert!(coordinates(a.binding(), &a.begin_advance(71)?, &c).is_err());
        let attempt = a.begin_advance(72)?;
        coordinates(a.binding(), &attempt, &c)?;
        a.prepare_publish(attempt.clone())?.commit();
        assert!(coordinates(a.binding(), &attempt, &c).is_err());
        Ok(())
    }
}
