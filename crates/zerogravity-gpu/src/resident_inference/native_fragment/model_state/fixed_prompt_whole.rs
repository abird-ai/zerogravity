//! Full24 current-attempt canonical domains; no alternative Root issuer.
use super::*;
use crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut;
impl ResidentSession {
    pub(in crate::resident_inference) fn bind_fixed_prompt_whole_state(
        &self,
        attempt: &RootAttempt,
        cut: &Arc<FixedPromptPrefixCut>,
        layer: usize,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        cut.require_whole()?;
        if !cut
            .scope()
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
            return Err(Error::State("whole actual source layer".into()));
        }
        let result = self.bind_model_state_scoped(
            attempt,
            cut.scope().record(),
            layer,
            candidate,
            phase,
            ModelStateScope::FixedPromptWhole(cut.clone()),
        )?;
        result.require_fixed_prompt_whole(cut)?;
        Ok(result)
    }
}
impl BoundModelState {
    pub(crate) fn whole_cut(&self) -> Result<&Arc<FixedPromptPrefixCut>> {
        match &self.scope {
            ModelStateScope::FixedPromptWhole(cut) => Ok(cut),
            _ => Err(Error::State("whole original state scope absent".into())),
        }
    }
    pub(crate) fn require_fixed_prompt_whole(&self, cut: &Arc<FixedPromptPrefixCut>) -> Result<()> {
        cut.require_whole()?;
        if !Arc::ptr_eq(self.whole_cut()?, cut)
            || self.call != *cut.scope().record().call()
            || self.base.source() != self.call.source()
            || self.base.frontier() != 0
        {
            return Err(Error::State(
                "whole state source/call/layer/root differs".into(),
            ));
        }
        if self.kv_domain().is_ok() {
            let site = cut.attention_site(self.layer)?;
            if self.effects.as_slice() != [site.effect()] {
                return Err(Error::State("whole KV own source effect".into()));
            }
        } else {
            cut.scope().gdn_site(self.layer)?;
            if self.effects.len() != 2 {
                return Err(Error::State("whole GDN effect pair".into()));
            }
        }
        Ok(())
    }
    pub(crate) fn require_fixed_prompt_gdn_scope(
        &self,
        scope: &crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding,
    ) -> Result<()> {
        match &self.scope {
            ModelStateScope::FixedPromptWhole(cut) => {
                if !std::ptr::eq(cut.scope().as_ref(), scope) || self.kv_domain().is_ok() {
                    return Err(Error::State("whole GDN scope/layer".into()));
                }
                self.require_fixed_prompt_whole(cut)
            }
            _ => self.require_fixed_prompt_model_prefix(scope),
        }
    }
}
