//! Full47 exact original GDN source/parameter/SSA/current-state joins.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut;
impl ModelGdnPrepared {
    pub(crate) async fn prepare_canonical47_whole(
        cut: &Arc<Canonical47PrefixCut>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        domains: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        declared: crate::vllm_cuda_gdn::Canonical47GdnArithmetic,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        cut.require_whole()?;
        domains.require_canonical47_whole(cut)?;
        if domains.layer().is_attention() {
            return Err(Error::State("whole GDN actual nonattention layer".into()));
        }
        let scope = cut.scope();
        scope.require_values(&values)?;
        scope.kind().require_prefill_gdn(declared)?;
        let nodes = scope.gdn_site(domains.layer().index())?.nodes();
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
        result.require_canonical47_binding(scope, declared)?;
        Ok(result)
    }
}
