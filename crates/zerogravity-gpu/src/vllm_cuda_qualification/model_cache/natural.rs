//! Shared defined transport arena; actual canonical history belongs to each Session.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding;

pub(crate) struct SharedModelCache {
    source: Arc<OriginalWindowModelSource>,
    layer: Vllm08Layer,
    lease: ResidentLease,
}
impl SharedModelCache {
    /// Caller owns its original cold fair slot. No BoundModelState/RootAttempt is
    /// needed to initialize private zero storage; this creates no history authority.
    pub(crate) async fn prepare(
        source: Arc<OriginalWindowModelSource>,
        service: &Arc<ResidentService>,
        layer: Vllm08Layer,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        source.require_vllm08_full_source()?;
        if layer.source() != source.source_identity()
            || service.source_identity() != source.source_identity()
        {
            return Err(Error::State(
                "shared canonical arena actual source/layer/service".into(),
            ));
        }
        let host = source.source().reserve_workspace(BYTES * 3 + 64 * 1024)?;
        let owner = Arc::new(ZeroSource {
            bytes: vec![0; BYTES],
            _source: source.clone(),
            _host: host,
        });
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(owner.clone())?;
        let lease = init.upload_bytes(&owner.bytes, BufferRole::RecurrentState)?;
        let result = Arc::new(Self {
            source,
            layer,
            lease,
        });
        init.retain_completion_owner(result.clone())?;
        drop(init.complete_sync().await?);
        not_cancelled(cancelled)?;
        Ok(result)
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn require_binding(
        &self,
        scope: &Arc<Canonical47ModelBinding>,
        state: &BoundModelState,
    ) -> Result<()> {
        self.retained_binding(scope, state)?;
        self.lease.service().ensure_usable()
    }
    pub(crate) fn retained_binding(
        &self,
        scope: &Arc<Canonical47ModelBinding>,
        state: &BoundModelState,
    ) -> Result<()> {
        state.require_canonical47_published(scope)?;
        if !Arc::ptr_eq(&self.source, scope.source().source())
            || state.layer() != self.layer
            || self.layer != scope.record().call().layer(self.layer.index())?
            || self.lease.bytes() != BYTES
            || !self.lease.is_initialized()
            || self.lease.source_identity() != scope.record().call().source()
            || self.lease.role() != BufferRole::RecurrentState
            || state.leases().iter().any(|(lease, _)| {
                lease.id() == self.lease.id()
                    || !Arc::ptr_eq(lease.service(), self.lease.service())
                    || lease.service_generation() != self.lease.service_generation()
            })
        {
            return Err(Error::State(
                "shared canonical arena source/layer/lease differs".into(),
            ));
        }
        Ok(())
    }
}
