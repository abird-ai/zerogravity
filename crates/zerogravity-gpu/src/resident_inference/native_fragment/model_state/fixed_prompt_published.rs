//! Original published prior/candidate domains; no full-model publication authority.
use super::*;
use crate::cuda::native::fixed_prompt_model::{FixedPromptCallKind, FixedPromptModelBinding};

impl ResidentSession {
    pub(in crate::resident_inference) fn bind_fixed_prompt_published_state(
        &self,
        attempt: &RootAttempt,
        scope: &Arc<FixedPromptModelBinding>,
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
            return Err(Error::State(
                "canonical published actual layer issuer".into(),
            ));
        }
        let result = self.bind_model_state_scoped(
            attempt,
            scope.record(),
            layer,
            candidate,
            phase,
            ModelStateScope::FixedPromptPublished(scope.clone()),
        )?;
        result.require_fixed_prompt_published(scope)?;
        Ok(result)
    }
}

impl BoundModelState {
    // Metadata only: uncertainty must not prevent proving retained exact owners.
    pub(crate) fn require_fixed_prompt_published(
        &self,
        scope: &Arc<FixedPromptModelBinding>,
    ) -> Result<()> {
        let ModelStateScope::FixedPromptPublished(actual) = &self.scope else {
            return Err(Error::State(
                "canonical published state scope absent".into(),
            ));
        };
        let start = match scope.kind() {
            FixedPromptCallKind::Prefill => 0,
            FixedPromptCallKind::Decode { base } => base,
        };
        if !Arc::ptr_eq(actual, scope)
            || self.call != *scope.record().call()
            || self.base.source() != self.call.source()
            || self.base.frontier() != start as u64
        {
            return Err(Error::State(
                "canonical published state source/call/layer/root differs".into(),
            ));
        }
        self.phase
            .require_retained_fixed_prompt_published(self.base, &self.call)?;
        if self.kv_domain().is_ok() {
            if self.effects.len() != 1
                || self.effects[0].read.slot != StateSlot::Kv(KvDomain(self.layer))
            {
                return Err(Error::State(
                    "canonical published exact KV site effect".into(),
                ));
            }
        } else {
            scope.gdn_site(self.layer)?;
            if self.effects.len() != 2 {
                return Err(Error::State(
                    "canonical published exact Conv/H effects".into(),
                ));
            }
        }
        Ok(())
    }
}
