//! True call/state wrapper over the original source/native attention slot.
use super::*;
use crate::resident_program::natural_binding::same_recipe;
impl ModelAttentionBinding {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
        domains: Arc<BoundModelState>,
        positions: Arc<TextPositionsPrepared>,
    ) -> Result<Arc<Self>> {
        same_recipe(self.values.record(), values.record())?;
        if self.layer != domains.layer()
            || domains.call() != values.record().call()
            || !Arc::ptr_eq(values.service(), self.values.service())
        {
            return Err(Error::State(
                "natural attention original value/state slot".into(),
            ));
        }
        positions.require_invocation(values.record().call())?;
        crate::resident_program::natural_binding::same_lease(
            self.positions.prepared_lease(),
            positions.prepared_lease(),
        )?;
        for index in 0..327 {
            crate::resident_program::natural_binding::same_lease(
                self.values
                    .lease(zerogravity_core::accelerator::program::ValueId(index))?,
                values.lease(zerogravity_core::accelerator::program::ValueId(index))?,
            )?;
        }
        self.rotary
            .require_call(&self.source, values.record().call(), values.service())?;
        let kv = self.kv.bind_natural(domains.clone())?;
        Ok(Arc::new(Self {
            values,
            layer: self.layer,
            case: self.case,
            nodes: self.nodes.clone(),
            native: self.native.clone(),
            positions,
            rotary: self.rotary.clone(),
            kv,
            domains,
            source: self.source.clone(),
            _weights: self._weights.clone(),
            _parameters: self._parameters.clone(),
            _cache: self._cache.clone(),
            _controls: self._controls.clone(),
        }))
    }
}
impl ModelAttentionBinding {
    pub(crate) fn publication_position_readback(
        &self,
    ) -> Result<crate::resident_leases::ResidentReadback> {
        self.positions.readback()
    }
    pub(crate) fn validate_publication_positions(
        &self,
        raw: &[cubecl::bytes::Bytes],
    ) -> Result<()> {
        self.positions.validate_readback(raw)
    }
}
