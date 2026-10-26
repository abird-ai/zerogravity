//! Fresh immutable output packet; only reducer temporaries are shared.
use super::*;
use crate::resident_program::natural_binding::{same_lease, same_recipe};

impl PreparedBf16Greedy {
    pub(crate) fn bind_reusable(self: &Arc<Self>, values: Arc<ModelValues>) -> Result<Arc<Self>> {
        same_recipe(self.owner.values.record(), values.record())?;
        require_head(values.record())?;
        same_lease(
            &self.logits,
            values.lease(values.record().selection().output())?,
        )?;
        same_lease(&self.status, values.status())?;
        let metadata = self.owner.source.reserve_workspace(METADATA_BYTES)?;
        // Original service admission precedes allocation. No fake initialization;
        // the enclosing actual sampler Write is its only content producer.
        let packet = self
            .service
            .allocate_uninitialized(PACKET_BYTES, BufferRole::Output)?;
        if packet.is_initialized() || packet.id() == self.packet.id() {
            return Err(Error::State(
                "reusable greedy requires fresh private packet".into(),
            ));
        }
        let call = values.record().call().clone();
        let owner = Arc::new(HostOwner {
            values,
            source: self.owner.source.clone(),
            _metadata: metadata,
        });
        let accesses = vec![
            (self.logits.clone(), DeviceAccess::Read),
            (self.partials.clone(), DeviceAccess::Write),
            (packet.clone(), DeviceAccess::Write),
            (self.status.clone(), DeviceAccess::Write),
        ];
        let outputs = vec![self.partials.clone(), packet.clone()];
        let result = Arc::new(Self {
            service: self.service.clone(),
            logits: self.logits.clone(),
            partials: self.partials.clone(),
            packet,
            status: self.status.clone(),
            accesses,
            outputs,
            call,
            logits_value: self.logits_value,
            owner,
        });
        result.validate_bindings()?;
        Ok(result)
    }
}
