//! Same actual full248320 BF16 reducer; fresh packet/current record/status.
use super::*;
use crate::resident_program::natural_binding::same_lease;
impl PreparedBf16Greedy {
    pub(crate) fn bind_prompt_call(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
    ) -> Result<Arc<Self>> {
        let cold = self.owner.values.text_storage_scope()?;
        let scope = values.text_storage_scope()?;
        scope.require_values(&values)?;
        if Arc::ptr_eq(cold.record(), scope.record())
            || !cold.source().same_issuer(scope.source())
            || cold.native_capture() != scope.native_capture()
            || cold.call().rows().len() != scope.call().rows().len()
            || !Arc::ptr_eq(values.service(), &self.service)
        {
            return Err(Error::State(
                "short-HF fresh exact source/fullHead/geometry reducer binding".into(),
            ));
        }
        require_head(values.record())?;
        same_lease(
            &self.logits,
            values.lease(values.record().selection().output())?,
        )?;
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
        let metadata = self.owner.source.reserve_workspace(METADATA_BYTES)?;
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
    pub(crate) fn text_allocation_layout() -> [usize; 2] {
        [PARTIAL_BYTES, PACKET_BYTES]
    }
}
