//! Six exact source gamma pairs; no layer3 parameter reuse.
use super::*;
use crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut;
use zerogravity_core::accelerator::program::PlanOperation;
impl ModelAttentionParameters {
    pub(crate) fn require_whole_qk(
        &self,
        cut: &Arc<FixedPromptPrefixCut>,
        values: &Arc<ModelValues>,
        layer: usize,
    ) -> Result<[ResidentLease; 2]> {
        cut.require_whole()?;
        cut.scope().require_values(values)?;
        let site = cut.attention_site(layer)?;
        let node = &cut.scope().record().plan().nodes()[site.nodes()[1]];
        let Some(descriptor) = &cut.scope().record().operator(site.nodes()[1])? else {
            return Err(Error::State("whole source QK operator".into()));
        };
        if !matches!(
            &node.operation,
            PlanOperation::SourceOperator { state: None, .. }
        ) {
            return Err(Error::State(
                "source operator state-read contract differs".into(),
            ));
        }
        if !matches!(descriptor.kind(), SourceOperatorKind::QkRmsRotaryGated(_))
            || descriptor.call() != values.record().call()
            || !self.source.same_issuer(cut.scope().source())
            || descriptor.parameters().len() != 2
            || !descriptor.matrices().is_empty()
        {
            return Err(Error::State("whole QK actual source gamma owner".into()));
        }
        self.require_operator(descriptor, values.service())
    }
}
