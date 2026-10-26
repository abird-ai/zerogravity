//! One genuine unpublished component attempt; no Ready/root/publication.
// Previous issuers: fixed_prompt_component, qwen2b_component.
use super::*;
use crate::resident_program::qwen2b_component::ComponentRecipe;

fn coordinates(base: RootBinding, attempt: &RootAttempt, call: &SourceCall) -> Result<()> {
    if call.source() != base.source()
        || attempt.base() != base
        || base.frontier() != 0
        || call.rows().base() != 0
        || call.rows().capacity() != 512
        || call.rows().vocabulary() != 248320
        || call.mode() != zerogravity_core::accelerator::program::SourceCallMode::Prefill
        || !matches!(call.rows().len(), 1 | 47 | 64)
        || attempt.frontier() != call.rows().len() as u64
    {
        return Err(Error::State(
            "component actual source/current initial attempt differs".into(),
        ));
    }
    Ok(())
}
impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_initial_component(
        recipe: &Arc<ComponentRecipe>,
        attempt: RootAttempt,
    ) -> Result<Arc<Self>> {
        let call = recipe.bindings().call();
        recipe.kind().require_call(call)?;
        if recipe.source().source_identity() != call.source() {
            return Err(Error::State(
                "component actual source issuer differs".into(),
            ));
        }
        coordinates(attempt.base(), &attempt, call)?;
        Ok(Arc::new(Self {
            base: attempt.base(),
            attempt,
            invocations: vec![call.clone()],
            completed_layers: 1,
        }))
    }
}
impl AttemptPhaseBinding {
    pub(in crate::resident_inference) fn require_initial_component_call(
        &self,
        call: &SourceCall,
    ) -> Result<()> {
        if self.phase_index != 0
            || self.prior.is_some()
            || self.schedule.completed_layers != 1
            || self.schedule.invocations.len() != 1
            || self.schedule.invocation(0)? != call
        {
            return Err(Error::State(
                "component exact unpublished one-phase issuer differs".into(),
            ));
        }
        coordinates(self.schedule.base, &self.schedule.attempt, call)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::{SourceCallMode, SourceRows, TextCallGeometry};
    use zerogravity_core::session_root::SessionAuthority;
    fn call(source: SourceIdentity, rows: usize) -> SourceCall {
        let rows = SourceRows::new(
            TextCallGeometry::new(source, 248320, 512).unwrap(),
            0,
            Arc::from(vec![11; rows]),
        )
        .unwrap();
        SourceCall::new(rows, SourceCallMode::Prefill).unwrap()
    }
    #[test]
    fn pure_coordinates_do_not_publish_and_foreign_attempts_refuse() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let a = SessionAuthority::new(source)?;
        let b = SessionAuthority::new(source)?;
        for rows in [1, 47, 64] {
            let request = call(source, rows);
            coordinates(a.binding(), &a.begin_advance(rows as u64)?, &request)?;
            assert!(coordinates(a.binding(), &b.begin_advance(rows as u64)?, &request).is_err());
            assert!(
                coordinates(a.binding(), &a.begin_advance(rows as u64 + 1)?, &request).is_err()
            );
            assert_eq!(a.binding().frontier(), 0);
        }
        for rows in [2, 46, 48, 63, 65] {
            assert!(
                coordinates(
                    a.binding(),
                    &a.begin_advance(rows as u64)?,
                    &call(source, rows)
                )
                .is_err()
            );
        }
        Ok(())
    }
    #[test]
    fn equal_words_new_issuer_and_whole_model_scope_cannot_replace_component() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let authority = SessionAuthority::new(source)?;
        let actual = call(source, 47);
        let reminted = call(source, 47);
        assert_eq!(actual.rows().ids(), reminted.rows().ids());
        assert_ne!(actual, reminted);
        for layers in [1, 24] {
            let schedule = Arc::new(GdnAttemptSchedule {
                base: authority.binding(),
                attempt: authority.begin_advance(47)?,
                invocations: vec![actual.clone()],
                completed_layers: layers,
            });
            let phase = AttemptPhaseBinding::new(schedule, 0, None)?;
            assert_eq!(
                phase.require_initial_component_call(&actual).is_ok(),
                layers == 1
            );
            assert!(phase.require_initial_component_call(&reminted).is_err());
        }
        assert_eq!(authority.binding().frontier(), 0);
        Ok(())
    }
}
