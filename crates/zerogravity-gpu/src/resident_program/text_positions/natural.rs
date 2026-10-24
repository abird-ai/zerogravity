//! Positions depend on genuine call/base, not the sampled token value.
use super::*;
impl TextPositionsPrepared {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        binding: Arc<BoundTextPositions>,
    ) -> Result<Arc<Self>> {
        if !Arc::ptr_eq(binding.service(), self.lease.service())
            || !binding
                .source()
                .source()
                .same_issuer(self.payload.binding.source().source())
            || binding.bytes()?.as_slice() != self.payload.bytes.as_slice()
        {
            return Err(Error::State(
                "natural original integer position geometry differs".into(),
            ));
        }
        let host = binding
            .source()
            .source()
            .reserve_workspace(3 * POSITION_BYTES)?;
        let payload = Arc::new(PositionPayload {
            bytes: self.payload.bytes.clone(),
            binding,
            _host: host,
        });
        Ok(Arc::new(Self {
            lease: self.lease.clone(),
            payload,
        }))
    }
}
