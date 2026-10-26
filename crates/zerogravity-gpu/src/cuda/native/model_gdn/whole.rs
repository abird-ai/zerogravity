//! Full47 exact original GDN source/parameter/SSA/current-state joins.
use super::*;
use crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut;
impl ModelGdnPrepared {
    pub(crate) async fn prepare_fixed_prompt_whole(
        cut: &Arc<FixedPromptPrefixCut>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        domains: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        declared: crate::cuda::gdn::FixedPromptGdnArithmetic,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        cut.require_whole()?;
        domains.require_fixed_prompt_whole(cut)?;
        if domains.conv_domain()? != ConvDomain(domains.layer())
            || domains.recurrent_domain()? != RecurrentDomain(domains.layer())
        {
            return Err(Error::State("whole GDN actual nonattention layer".into()));
        }
        let scope = cut.scope();
        scope.require_values(&values)?;
        scope.kind().require_prefill_gdn(declared)?;
        let nodes = scope.gdn_site(domains.layer())?.nodes();
        let result = Self::prepare_impl(
            values,
            parameters,
            domains,
            arena,
            nodes,
            Some(scope.clone()),
            cancelled,
        )
        .await?;
        result.require_fixed_prompt_binding(scope, declared)?;
        Ok(result)
    }
}
