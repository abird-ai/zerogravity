//! Original published prior/candidate domains; no full-model publication authority.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::{
    Canonical47CallKind, Canonical47ModelBinding,
};

impl ResidentSession {
    pub(in crate::resident_inference) fn bind_canonical47_published_state(
        &self,
        attempt: &RootAttempt,
        scope: &Arc<Canonical47ModelBinding>,
        layer: Vllm08Layer,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        if layer != scope.record().call().layer(layer.index())? {
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
            ModelStateScope::Canonical47Published(scope.clone()),
        )?;
        result.require_canonical47_published(scope)?;
        Ok(result)
    }
}

impl BoundModelState {
    // Metadata only: uncertainty must not prevent proving retained exact owners.
    pub(crate) fn require_canonical47_published(
        &self,
        scope: &Arc<Canonical47ModelBinding>,
    ) -> Result<()> {
        let ModelStateScope::Canonical47Published(actual) = &self.scope else {
            return Err(Error::State(
                "canonical published state scope absent".into(),
            ));
        };
        let start = match scope.kind() {
            Canonical47CallKind::Prefill => 0,
            Canonical47CallKind::Decode { base } => base,
        };
        if !Arc::ptr_eq(actual, scope)
            || self.call != *scope.record().call()
            || self.layer != self.call.layer(self.layer.index())?
            || self.base.source() != self.call.source()
            || self.base.frontier() != start as u64
        {
            return Err(Error::State(
                "canonical published state source/call/layer/root differs".into(),
            ));
        }
        self.phase
            .require_retained_canonical47_published(self.base, &self.call)?;
        if self.layer.is_attention() {
            if self.effects.len() != 1
                || self.effects[0].read.slot != StateSlot::Kv(KvDomain(self.layer.index()))
            {
                return Err(Error::State(
                    "canonical published exact KV site effect".into(),
                ));
            }
        } else {
            scope.gdn_site(self.layer.index())?;
            if self.effects.len() != 2 {
                return Err(Error::State(
                    "canonical published exact Conv/H effects".into(),
                ));
            }
        }
        Ok(())
    }
}
