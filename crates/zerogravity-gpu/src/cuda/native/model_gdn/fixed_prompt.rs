//! Exact full-record producer/source/layer/state/code join; no new factory permit.
use super::*;
use crate::cuda::gdn::FixedPromptGdnArithmetic;
use crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding;
impl ModelGdnPrepared {
    pub(crate) fn require_fixed_prompt_binding(
        &self,
        scope: &FixedPromptModelBinding,
        declared: FixedPromptGdnArithmetic,
    ) -> Result<()> {
        scope.require_values(&self.binding.values)?;
        let layer = self.binding.layer();
        let site = scope.gdn_site(layer)?;
        if !self.binding.source().same_issuer(scope.source())
            || self.binding.call() != scope.record().call()
            || site.layer() != layer
            || self.binding.nodes != site.nodes()
            || self.domains.layer() != self.binding.layer()
        {
            return Err(Error::State(
                "fixed_prompt actual full-record GDN source/layer/node/state issuer".into(),
            ));
        }
        for (id, lease) in site.operands().into_iter().zip(&self.binding.operands) {
            if scope.record().plan().source_value_provenance(id)?.is_none()
                || self.binding.values.lease(id)?.id() != lease.id()
            {
                return Err(Error::State(
                    "fixed_prompt exact full-record GDN ValueId/physical producer".into(),
                ));
            }
        }
        // Original check seals all5 producer and4 real learned parameter leases,
        // genuine domains/call and actual prepared arithmetic. Family alone is insufficient.
        self.binding
            .require_prepared(&self.prepared, &self.domains)?;
        scope.require_gdn_arithmetic(layer, self.prepared.arithmetic(), declared)
    }
}
