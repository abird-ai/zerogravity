//! Bind a new true call to existing gamma/math storage; no upload/allocation.
use super::*;
impl FusedRmsPrepared {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        binding: BoundModelRmsSource,
        attention: &ResidentLease,
        residual: &ResidentLease,
    ) -> Result<Arc<Self>> {
        if self.pooled.is_some() {
            return Err(Error::Unsupported(
                "pooled RMS requires fresh text epoch binding".into(),
            ));
        }
        let old = self
            .model_binding
            .as_ref()
            .or(self.owner._model_binding.as_ref())
            .ok_or_else(|| Error::State("natural RMS needs original full model gamma".into()))?;
        let (a, b) = (old.operator(), binding.operator());
        binding.require_source(&self.owner.source)?;
        if a.numerical_contract() != b.numerical_contract()
            || a.matrices() != b.matrices()
            || a.call().source() != b.call().source()
            || a.kind() != b.kind()
            || a.parameters() != b.parameters()
            || a.call().rows().capacity() != b.call().rows().capacity()
            || a.call().rows().vocabulary() != b.call().rows().vocabulary()
            || a.call().rows().len() != b.call().rows().len()
            || a.call().mode() != b.call().mode()
            || a.call().rows().base() != b.call().rows().base()
        {
            return Err(Error::State(
                "natural RMS source/site/geometry differs from cold storage".into(),
            ));
        }
        crate::resident_program::natural_binding::same_lease(
            self.lease(Buffer::Attention),
            attention,
        )?;
        crate::resident_program::natural_binding::same_lease(
            self.lease(Buffer::PriorResidual),
            residual,
        )?;
        Ok(Arc::new(Self {
            service: self.service.clone(),
            leases: self.leases.clone(),
            stages: self.stages.clone(),
            accesses: self.accesses.clone(),
            owner: self.owner.clone(),
            model_binding: Some(binding),
            pooled: None,
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            _text_binding: None,
        }))
    }
}
