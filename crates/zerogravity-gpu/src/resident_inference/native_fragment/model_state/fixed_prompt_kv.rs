//! Original Kv3 priors/candidates for the exact35-node unpublished source cut.
use super::*;
use crate::cuda::native::fixed_prompt_model::{FixedPromptCallKind, FixedPromptModelBinding};
use crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut;
use zerogravity_core::accelerator::program::SourceOperatorKind;
use zerogravity_core::accelerator::program::ValueId;

impl ResidentSession {
    pub(in crate::resident_inference) fn bind_fixed_prompt_attention_kv_state(
        &self,
        attempt: &RootAttempt,
        cut: &Arc<FixedPromptPrefixCut>,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        if !cut.is_attention_cut() {
            return Err(Error::Unsupported(
                "canonical47 original35-node Kv3 state cut".into(),
            ));
        }
        let scope = cut.scope();
        if scope.kind() != FixedPromptCallKind::Prefill {
            return Err(Error::Unsupported(
                "canonical47 initial Kv3 source call".into(),
            ));
        }
        let bound =
            self.bind_model_state_impl(attempt, scope.record(), 3, candidate, phase, true)?;
        bound.require_fixed_prompt_attention_kv(scope)?;
        Ok(bound)
    }
}
impl BoundModelState {
    pub(crate) fn require_fixed_prompt_attention_kv(
        &self,
        scope: &FixedPromptModelBinding,
    ) -> Result<()> {
        let call = scope.record().call();
        let node = scope
            .record()
            .plan()
            .nodes()
            .get(34)
            .ok_or_else(|| Error::State("canonical47 actual Kv3 node absent".into()))?;
        if !matches!(&self.scope, ModelStateScope::FixedPromptPrefix)
            || scope.kind() != FixedPromptCallKind::Prefill
            || self.call != *call
            || self.layer != 3
            || self.kv_domain()? != KvDomain(3)
            || self.base.source() != scope.source().source_identity()
            || self.base.frontier() != 0
            || self.effects.len() != 1
            || node.inputs.as_slice() != [ValueId(43), ValueId(41)]
            || !node.outputs.is_empty()
            || node.state != self.effects.first().copied()
            || (!matches!(
                &node.operation,
                PlanOperation::SourceOperator { state: None, .. }
            ) || !scope.record().operator(34)?.is_some_and(|descriptor| {
                descriptor.call() == call
                    && matches!(descriptor.kind(), SourceOperatorKind::KvAppend { domain, .. }
                        if domain == KvDomain(3))
            }))
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
