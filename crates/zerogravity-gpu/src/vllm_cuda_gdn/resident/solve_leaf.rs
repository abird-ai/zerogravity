//! Reuse the existing exact ZeroInverse copy; no second zero implementation.
use super::*;
pub(crate) struct InverseZero(Copy);
impl InverseZero {
    pub(crate) fn prepare(service: &Arc<ResidentService>, inverse: &ResidentLease) -> Result<Self> {
        if !Arc::ptr_eq(service, inverse.service())
            || inverse.bytes() != GdnBuffer::Inverse.bytes(VllmGdnPhase::Initial47)
            || inverse.role() != BufferRole::Scratch
        {
            return Err(Error::State(
                "original47 zero inverse original lease".into(),
            ));
        }
        let buffers = BTreeMap::from([(GdnBuffer::Inverse, inverse.clone())]);
        Copy::prepare(
            service,
            VllmGdnPhase::Initial47,
            GdnCacheSlot::One,
            CopyKind::ZeroInverse,
            &buffers,
        )
        .map(Self)
    }
    pub(crate) fn prepare_2b(
        service: &Arc<ResidentService>,
        inverse: &ResidentLease,
    ) -> Result<Self> {
        if !Arc::ptr_eq(service, inverse.service())
            || inverse.bytes() != GdnBuffer::Inverse.bytes(VllmGdnPhase::Initial64)
            || inverse.role() != BufferRole::Scratch
        {
            return Err(Error::State(
                "original2B/T64 zero inverse original lease".into(),
            ));
        }
        let buffers = BTreeMap::from([(GdnBuffer::Inverse, inverse.clone())]);
        Copy::prepare(
            service,
            VllmGdnPhase::Initial64,
            GdnCacheSlot::Four,
            CopyKind::ZeroInverse,
            &buffers,
        )
        .map(Self)
    }
    pub(crate) fn enqueue(&self, client: &Client) -> Result<()> {
        self.0.launch(client)
    }
}
