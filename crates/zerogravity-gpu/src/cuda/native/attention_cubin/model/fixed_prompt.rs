//! Exact raw original Q/K gamma binding for the computed fixed_prompt node33.
use super::super::super::fixed_prompt_model_prefix::FixedPromptPrefixCut;
use super::*;
use zerogravity_core::accelerator::program::PlanOperation;
impl ModelAttentionParameters {
    pub(crate) fn require_fixed_prompt_qk(
        &self,
        cut: &Arc<FixedPromptPrefixCut>,
        values: &Arc<ModelValues>,
    ) -> Result<[ResidentLease; 2]> {
        cut.scope().require_values(values)?;
        let node = cut
            .scope()
            .record()
            .plan()
            .nodes()
            .get(33)
            .ok_or_else(|| Error::State("fixed_prompt actual QK gamma node absent".into()))?;
        let Some(descriptor) = &cut.scope().record().operator(33)? else {
            return Err(Error::State(
                "fixed_prompt QK gamma stateless operator".into(),
            ));
        };
        if !matches!(
            &node.operation,
            PlanOperation::SourceOperator { state: None, .. }
        ) {
            return Err(Error::State(
                "source operator state-read contract differs".into(),
            ));
        }
        if !cut.includes_kv()
            || !matches!(descriptor.kind(), SourceOperatorKind::QkRmsRotaryGated(_))
            || descriptor.call() != values.record().call()
            || !self.source.same_issuer(cut.scope().source())
            || descriptor.parameters().len() != 2
            || !descriptor.matrices().is_empty()
        {
            return Err(Error::State(
                "fixed_prompt actual original QK parameter/source/call owners".into(),
            ));
        }
        self.require_operator(descriptor, values.service())
    }
}
