//! Fresh root/source binding over original preallocated copy/status resources.
use super::*;
impl ModelGdnAdapter {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        binding: Arc<ModelGdnBinding>,
        domains: Arc<BoundModelState>,
        prepared: &Arc<GdnPrepared>,
    ) -> Result<Arc<Self>> {
        binding.require_prepared(prepared, &domains)?;
        let service = self.cache.service();
        if domains.layer() != self.domains.layer()
            || GdnPhase::for_reference_call(binding.call())?
                != GdnPhase::for_reference_call(self.binding.call())?
            || binding.call().rows().base() != self.binding.call().rows().base()
            || self.cache.id() != prepared.bound_lease(GdnBuffer::CacheArena)?.id()
        {
            return Err(Error::State(
                "natural GDN adapter cold geometry differs".into(),
            ));
        }
        let leases = domains.leases();
        for (index, (lease, _)) in leases.iter().enumerate() {
            let bytes = if index < 2 {
                CONV_ELEMENTS * 4
            } else {
                H_ELEMENTS * 4
            };
            if !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != service.source_identity()
                || lease.service_generation() != service.generation()
                || lease.bytes() != bytes
                || lease.id() == self.cache.id()
                || lease.id() == self.status.id()
            {
                return Err(Error::State(
                    "natural GDN actual canonical root extents".into(),
                ));
            }
        }
        let metadata = binding.source().reserve_workspace(64 * 1024)?;
        let conv = plane_handle(
            &self.cache,
            self.slot.convolution_offset(),
            CONV_STATE_BYTES,
        )?;
        let recurrent = plane_handle(
            &self.cache,
            self.slot.recurrent_offset(),
            RECURRENT_STATE_BYTES,
        )?;
        // Pure checked kernel/handle binding only. Existing exact source/grid
        // keys are unchanged; no client allocation/launch/init or new native record.
        let copies = [
            StateTransport::prepare(
                service,
                StateCopy::ConvIn,
                leases[0].0.handle(),
                conv.clone(),
                &self.status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::ConvOut,
                conv,
                leases[1].0.handle(),
                &self.status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::RecurrentIn,
                leases[2].0.handle(),
                recurrent.clone(),
                &self.status,
            )?,
            StateTransport::prepare(
                service,
                StateCopy::RecurrentOut,
                recurrent,
                leases[3].0.handle(),
                &self.status,
            )?,
        ];
        Ok(Arc::new(Self {
            copies,
            domains,
            binding,
            cache: self.cache.clone(),
            status: self.status.clone(),
            slot: self.slot,
            _transport: None,
            _metadata: metadata,
        }))
    }
}
