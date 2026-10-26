//! Original prior/candidate domains with exact short-HF source/current-root seals.
use super::*;
use crate::cuda::native::text_model::TextModelBinding;
impl ResidentSession {
    pub(in crate::resident_inference) fn bind_text_state(
        &self,
        attempt: &RootAttempt,
        scope: &Arc<TextModelBinding>,
        layer: usize,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        if !scope
            .record()
            .plan()
            .required_domains()
            .iter()
            .any(|slot| match slot {
                StateSlot::Conv(domain) => domain.0 == layer,
                StateSlot::Recurrent(domain) => domain.0 == layer,
                StateSlot::Kv(domain) => domain.0 == layer,
            })
        {
            return Err(Error::State("short-HF actual state layer".into()));
        }
        let result = self.bind_model_state_scoped(
            attempt,
            scope.record(),
            layer,
            candidate,
            phase,
            ModelStateScope::Text(scope.clone()),
        )?;
        result.require_text(scope)?;
        Ok(result)
    }
}
impl BoundModelState {
    pub(crate) fn require_text(&self, scope: &Arc<TextModelBinding>) -> Result<()> {
        let ModelStateScope::Text(actual) = &self.scope else {
            return Err(Error::State("short-HF state scope absent".into()));
        };
        if !Arc::ptr_eq(actual, scope)
            || self.call != *scope.record().call()
            || self.base.source() != self.call.source()
            || self.base.frontier() != scope.call().rows().base() as u64
        {
            return Err(Error::State(
                "short-HF exact actual source/call/layer/prior root".into(),
            ));
        }
        self.phase.require_retained_text(self.base, &self.call)?;
        if self.kv_domain().is_ok() {
            scope.attention_site(self.layer)?;
            if self.effects.len() != 1
                || self.effects[0].read.slot != StateSlot::Kv(KvDomain(self.layer))
            {
                return Err(Error::State("short-HF actual sole KV write effect".into()));
            }
        } else {
            scope.gdn_site(self.layer)?;
            if self.effects.len() != 2 {
                return Err(Error::State("short-HF actual Conv/H effect pair".into()));
            }
        }
        Ok(())
    }
}
