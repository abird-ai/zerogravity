//! Exact current-call source gamma binding; no legacy span conversion.
use super::*;
use crate::resident_inference::native_fragment::fixed_prompt_natural::FixedPromptCallInput;
use zerogravity_core::accelerator::program::PlanOperation;
impl ModelAttentionParameters {
    pub(crate) fn require_fixed_prompt_call(
        &self,
        input: &FixedPromptCallInput,
        values: &Arc<ModelValues>,
        node: usize,
    ) -> Result<[ResidentLease; 2]> {
        input.scope().require_values(values)?;
        let actual = input
            .record()
            .plan()
            .nodes()
            .get(node)
            .ok_or_else(|| Error::State("current QK node absent".into()))?;
        let Some(descriptor) = &input.record().operator(node)? else {
            return Err(Error::State("current QK descriptor absent".into()));
        };
        if !matches!(
            &actual.operation,
            PlanOperation::SourceOperator { state: None, .. }
        ) {
            return Err(Error::State("current QK descriptor absent".into()));
        }
        if !matches!(descriptor.kind(), SourceOperatorKind::QkRmsRotaryGated(_))
            || descriptor.call() != input.call()
            || !self.source.same_issuer(input.source())
            || descriptor.parameters().len() != 2
            || !descriptor.matrices().is_empty()
        {
            return Err(Error::State("current source QK parameter seal".into()));
        }
        self.require_operator(descriptor, values.service())
    }
}
