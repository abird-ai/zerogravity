//! Same computed BF16 Head/status/partials, fresh immutable output packet.
use super::*;
use crate::resident_program::natural_binding::same_lease;

impl PreparedBf16Greedy {
    pub(crate) fn bind_canonical_call(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
    ) -> Result<Arc<Self>> {
        crate::resident_program::canonical47_binding::same_recipe(
            &self.owner.source,
            self.owner.values.record(),
            values.record(),
        )?;
        require_head(values.record())?;
        same_lease(&self.logits, values.lease(values.record().logits())?)?;
        same_lease(&self.status, values.status())?;
        let metadata = self
            .owner
            .source
            .source()
            .reserve_workspace(METADATA_BYTES)?;
        let packet = self
            .service
            .allocate_uninitialized(PACKET_BYTES, BufferRole::Output)?;
        if packet.is_initialized()
            || [
                self.packet.id(),
                self.partials.id(),
                self.status.id(),
                self.logits.id(),
            ]
            .contains(&packet.id())
        {
            return Err(Error::State(
                "canonical greedy requires fresh actual immutable packet".into(),
            ));
        }
        let call = values.record().call().clone();
        let owner = Arc::new(HostOwner {
            values,
            source: self.owner.source.clone(),
            _metadata: metadata,
        });
        let result = Arc::new(Self {
            service: self.service.clone(),
            logits: self.logits.clone(),
            partials: self.partials.clone(),
            packet: packet.clone(),
            status: self.status.clone(),
            accesses: vec![
                (self.logits.clone(), DeviceAccess::Read),
                (self.partials.clone(), DeviceAccess::Write),
                (packet.clone(), DeviceAccess::Write),
                (self.status.clone(), DeviceAccess::Write),
            ],
            outputs: vec![self.partials.clone(), packet],
            call,
            logits_value: self.logits_value,
            owner,
        });
        result.validate_bindings()?;
        Ok(result)
    }
}
