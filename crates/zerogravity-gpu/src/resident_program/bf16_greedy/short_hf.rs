//! Same actual full248320 BF16 reducer; fresh packet/current record/status.
use super::*;
use crate::resident_program::natural_binding::same_lease;
impl PreparedBf16Greedy {
    pub(crate) fn bind_short_hf_call(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
    ) -> Result<Arc<Self>> {
        let cold = self.owner.values.short_hf_storage_scope()?;
        let scope = values.short_hf_storage_scope()?;
        scope.require_values(&values)?;
        if Arc::ptr_eq(cold.record(), scope.record())
            || !Arc::ptr_eq(cold.source().source(), scope.source().source())
            || cold.call().model_profile() != scope.call().model_profile()
            || cold.call().span().len() != scope.call().span().len()
            || !Arc::ptr_eq(values.service(), &self.service)
        {
            return Err(Error::State(
                "short-HF fresh exact source/fullHead/geometry reducer binding".into(),
            ));
        }
        require_head(values.record())?;
        same_lease(&self.logits, values.lease(values.record().logits())?)?;
        let status = values.status().clone();
        if status.bytes() != 4
            || status.role() != BufferRole::Scratch
            || status.id() == self.partials.id()
            || status.id() == self.logits.id()
            || !Arc::ptr_eq(status.service(), &self.service)
        {
            return Err(Error::State(
                "short-HF actual independent current status".into(),
            ));
        }
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
                status.id(),
                self.logits.id(),
            ]
            .contains(&packet.id())
        {
            return Err(Error::State(
                "short-HF actual fresh immutable output packet".into(),
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
            status: status.clone(),
            accesses: vec![
                (self.logits.clone(), DeviceAccess::Read),
                (self.partials.clone(), DeviceAccess::Write),
                (packet.clone(), DeviceAccess::Write),
                (status, DeviceAccess::Write),
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

impl PreparedBf16Greedy {
    pub(crate) fn short_hf_allocation_layout() -> [usize; 2] {
        [PARTIAL_BYTES, PACKET_BYTES]
    }
}
