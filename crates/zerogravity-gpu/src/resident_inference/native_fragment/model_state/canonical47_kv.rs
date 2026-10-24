//! Original Kv3 priors/candidates for the exact35-node unpublished source cut.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::{
    Canonical47CallKind, Canonical47ModelBinding,
};
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;
use zerogravity_core::accelerator::program::ValueId;

impl ResidentSession {
    pub(in crate::resident_inference) fn bind_canonical47_attention_kv_state(
        &self,
        attempt: &RootAttempt,
        cut: &Arc<Canonical47PrefixCut>,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        if !cut.is_attention_cut() {
            return Err(Error::Unsupported(
                "canonical47 original35-node Kv3 state cut".into(),
            ));
        }
        let scope = cut.scope();
        if scope.kind() != Canonical47CallKind::Prefill {
            return Err(Error::Unsupported(
                "canonical47 initial Kv3 source call".into(),
            ));
        }
        let bound = self.bind_model_state_impl(
            attempt,
            scope.record(),
            scope.record().call().layer(3)?,
            candidate,
            phase,
            true,
        )?;
        bound.require_canonical47_attention_kv(scope)?;
        Ok(bound)
    }
}
impl BoundModelState {
    pub(crate) fn require_canonical47_attention_kv(
        &self,
        scope: &Canonical47ModelBinding,
    ) -> Result<()> {
        let call = scope.record().call();
        let node = scope
            .record()
            .plan()
            .nodes()
            .get(34)
            .ok_or_else(|| Error::State("canonical47 actual Kv3 node absent".into()))?;
        if !matches!(&self.scope, ModelStateScope::Canonical47Prefix)
            || scope.kind() != Canonical47CallKind::Prefill
            || self.call != *call
            || self.layer != call.layer(3)?
            || !self.layer.is_attention()
            || self.base.source() != scope.source().source().source_identity()
            || self.base.frontier() != 0
            || self.effects.len() != 1
            || node.inputs.as_slice() != [ValueId(43), ValueId(41)]
            || !node.outputs.is_empty()
            || node.state != self.effects.first().copied()
            || !matches!(&node.operation,PlanOperation::TypedVllm08 {descriptor,state:None}
                if descriptor.call()==call && descriptor.layer()==Some(self.layer)
                    && descriptor.kind()==Vllm08Kind::KvAppend)
            || self.effects[0].read.slot != StateSlot::Kv(KvDomain(3))
            || self.effects[0].write.slot != StateSlot::Kv(KvDomain(3))
            || self.effects[0].read.ordinal != 0
            || self.effects[0].write.ordinal != 1
        {
            return Err(Error::State(
                "canonical47 same original call/root/Kv3 version/effect".into(),
            ));
        }
        Ok(())
    }
}
