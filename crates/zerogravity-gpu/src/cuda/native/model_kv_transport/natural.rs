//! Same KV source/grid kernels, bound to the CURRENT original canonical root.
use super::*;
impl ModelKvTransport {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        binding: Arc<BoundModelState>,
    ) -> Result<Arc<Self>> {
        let (a, b) = (self.binding.call(), binding.call());
        if self.binding.layer() != binding.layer()
            || a.source() != b.source()
            || a.mode() != b.mode()
            || a.rows().base() != b.rows().base()
            || a.rows().len() != b.rows().len()
        {
            return Err(Error::State("natural KV original geometry differs".into()));
        }
        let metadata = self._source.reserve_workspace(64 * 1024)?;
        let service = self.cache.service();
        let leases = binding.leases();
        for (lease, _) in &leases {
            if !Arc::ptr_eq(lease.service(), service)
                || lease.bytes() != CANONICAL_BYTES
                || lease.source_identity() != service.source_identity()
                || lease.service_generation() != service.generation()
                || lease.id() == self.cache.id()
                || lease.id() == self.status.id()
            {
                return Err(Error::State(
                    "natural KV current canonical leases differ".into(),
                ));
            }
        }
        let make = |old: &Copy, incoming: bool| -> Result<Copy> {
            let mut handles = vec![
                leases[0].0.handle(),
                leases[2].0.handle(),
                self.cache.handle(),
            ];
            if !incoming {
                handles.extend([leases[1].0.handle(), leases[3].0.handle()]);
            }
            handles.push(self.status.handle());
            old.kernel.validate(service.client(), old.grid, &handles)?;
            Ok(Copy {
                kernel: old.kernel.clone(),
                grid: old.grid,
                handles,
            })
        };
        Ok(Arc::new(Self {
            incoming: self
                .incoming
                .as_ref()
                .map(|copy| make(copy, true))
                .transpose()?,
            outgoing: make(&self.outgoing, false)?,
            binding,
            cache: self.cache.clone(),
            _arena: self._arena.clone(),
            status: self.status.clone(),
            _source: self._source.clone(),
            _metadata: metadata,
        }))
    }
}
