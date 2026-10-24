//! Full24 current-attempt canonical domains; no alternative Root issuer.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;
impl ResidentSession {
    pub(in crate::resident_inference) fn bind_canonical47_whole_state(
        &self,
        attempt: &RootAttempt,
        cut: &Arc<Canonical47PrefixCut>,
        layer: Vllm08Layer,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        cut.require_whole()?;
        if layer != cut.scope().record().call().layer(layer.index())? {
            return Err(Error::State("whole actual source layer".into()));
        }
        let result = self.bind_model_state_scoped(
            attempt,
            cut.scope().record(),
            layer,
            candidate,
            phase,
            ModelStateScope::Canonical47Whole(cut.clone()),
        )?;
        result.require_canonical47_whole(cut)?;
        Ok(result)
    }
}
impl BoundModelState {
    pub(crate) fn whole_cut(&self) -> Result<&Arc<Canonical47PrefixCut>> {
        match &self.scope {
            ModelStateScope::Canonical47Whole(cut) => Ok(cut),
            _ => Err(Error::State("whole original state scope absent".into())),
        }
    }
    pub(crate) fn require_canonical47_whole(&self, cut: &Arc<Canonical47PrefixCut>) -> Result<()> {
        cut.require_whole()?;
        if !Arc::ptr_eq(self.whole_cut()?, cut)
            || self.call != *cut.scope().record().call()
            || self.layer != self.call.layer(self.layer.index())?
            || self.base.source() != self.call.source()
            || self.base.frontier() != 0
        {
            return Err(Error::State(
                "whole state source/call/layer/root differs".into(),
            ));
        }
        if self.layer.is_attention() {
            let site = cut.attention_site(self.layer.index())?;
            if self.effects.as_slice() != [site.effect()] {
                return Err(Error::State("whole KV own source effect".into()));
            }
        } else {
            cut.scope().gdn_site(self.layer.index())?;
            if self.effects.len() != 2 {
                return Err(Error::State("whole GDN effect pair".into()));
            }
        }
        Ok(())
    }
    pub(crate) fn require_canonical47_gdn_scope(
        &self,
        scope: &crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding,
    ) -> Result<()> {
        match &self.scope {
            ModelStateScope::Canonical47Whole(cut) => {
                if !std::ptr::eq(cut.scope().as_ref(), scope) || self.layer.is_attention() {
                    return Err(Error::State("whole GDN scope/layer".into()));
                }
                self.require_canonical47_whole(cut)
            }
            _ => self.require_canonical47_model_prefix(scope),
        }
    }
}
