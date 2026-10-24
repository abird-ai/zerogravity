//! Original prior/candidate domains with exact short-HF source/current-root seals.
use super::*;
use crate::vllm_cuda_qualification::short_hf_model::ShortHfModelBinding;
impl ResidentSession {
    pub(in crate::resident_inference) fn bind_short_hf_state(
        &self,
        attempt: &RootAttempt,
        scope: &Arc<ShortHfModelBinding>,
        layer: Vllm08Layer,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        if layer != scope.record().call().layer(layer.index())? {
            return Err(Error::State("short-HF actual state layer".into()));
        }
        let result = self.bind_model_state_scoped(
            attempt,
            scope.record(),
            layer,
            candidate,
            phase,
            ModelStateScope::ShortHf(scope.clone()),
        )?;
        result.require_short_hf(scope)?;
        Ok(result)
    }
}
impl BoundModelState {
    pub(crate) fn require_short_hf(&self, scope: &Arc<ShortHfModelBinding>) -> Result<()> {
        let ModelStateScope::ShortHf(actual) = &self.scope else {
            return Err(Error::State("short-HF state scope absent".into()));
        };
        if !Arc::ptr_eq(actual, scope)
            || self.call != *scope.record().call()
            || self.layer != self.call.layer(self.layer.index())?
            || self.base.source() != self.call.source()
            || self.base.frontier() != scope.call().span().base() as u64
        {
            return Err(Error::State(
                "short-HF exact actual source/call/layer/prior root".into(),
            ));
        }
        self.phase
            .require_retained_short_hf(self.base, &self.call)?;
        if self.layer.is_attention() {
            scope.attention_site(self.layer.index())?;
            if self.effects.len() != 1
                || self.effects[0].read.slot != StateSlot::Kv(KvDomain(self.layer.index()))
            {
                return Err(Error::State("short-HF actual sole KV write effect".into()));
            }
        } else {
            scope.gdn_site(self.layer.index())?;
            if self.effects.len() != 2 {
                return Err(Error::State("short-HF actual Conv/H effect pair".into()));
            }
        }
        Ok(())
    }
}
