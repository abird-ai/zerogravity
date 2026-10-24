//! Exact full-record producer/source/layer/state/code join; no new factory permit.
use super::*;
use crate::vllm_cuda_gdn::Canonical47GdnArithmetic;
use crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding;
impl ModelGdnPrepared {
    pub(crate) fn require_canonical47_binding(
        &self,
        scope: &Canonical47ModelBinding,
        declared: Canonical47GdnArithmetic,
    ) -> Result<()> {
        scope.require_values(&self.binding.values)?;
        let layer = self.binding.layer().index();
        let site = scope.gdn_site(layer)?;
        if !Arc::ptr_eq(self.binding.source(), scope.source().source())
            || self.binding.call() != scope.record().call()
            || site.layer() != layer
            || self.binding.nodes != site.nodes()
            || self.domains.layer() != self.binding.layer()
        {
            return Err(Error::State(
                "canonical47 actual full-record GDN source/layer/node/state issuer".into(),
            ));
        }
        for (id, lease) in site.operands().into_iter().zip(&self.binding.operands) {
            if scope.record().plan().vllm_value_provenance(id)?.is_none()
                || self.binding.values.lease(id)?.id() != lease.id()
            {
                return Err(Error::State(
                    "canonical47 exact full-record GDN ValueId/physical producer".into(),
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
