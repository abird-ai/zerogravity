//! One bounded physical producer read, without retiring the original flight.
use super::*;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};

pub(crate) struct ResidentFrameCheckpoint {
    source: VerifiedTensorSource,
    pending: Mutex<Option<WindowWorkspace>>,
    registered_event: Mutex<Option<DeviceEventId>>,
    _control: WindowWorkspace,
}
impl ResidentFrameCheckpoint {
    pub(crate) fn new(source: VerifiedTensorSource) -> Result<Arc<Self>> {
        let control = source.reserve_workspace(4096)?;
        Ok(Arc::new(Self {
            source,
            pending: Mutex::new(None),
            registered_event: Mutex::new(None),
            _control: control,
        }))
    }
    /// The borrowed frame and comparison retire before returning. Callback
    /// failure leaves the original event armed and cannot mint Ready/root.
    pub(crate) async fn compare(
        self: &Arc<Self>,
        flight: &mut ResidentSubmission<'_>,
        read: &ResidentReadback,
        validate: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<()> {
        if read.bytes == 0
            || read.bytes > source_page::PAGE_BYTES
            || read.lease.source_identity() != self.source.source_identity()
        {
            return Err(Error::Limit(
                "one authentic completed producer frame <=1MiB".into(),
            ));
        }
        flight.require_accesses([(&read.lease, DeviceAccess::Read)])?;
        {
            let mut event = self
                .registered_event
                .lock()
                .map_err(|_| Error::State("producer frame event owner poisoned".into()))?;
            if *event != Some(flight.event) {
                flight.retain_completion_owner(self.clone())?;
                *event = Some(flight.event);
            }
        }
        {
            let mut pending = self
                .pending
                .lock()
                .map_err(|_| Error::State("producer frame owner poisoned".into()))?;
            if pending.is_some() {
                return Err(Error::State(
                    "producer frame outstanding before reuse".into(),
                ));
            }
            *pending = Some(self.source.reserve_workspace(3 * read.bytes)?);
        }
        let values = flight
            .readback_checked(std::slice::from_ref(read), |_| Ok(()))
            .await?;
        let result = validate(&values[0]);
        drop(values);
        let reservation = self
            .pending
            .lock()
            .map_err(|_| Error::State("producer frame owner poisoned".into()))?
            .take();
        drop(reservation);
        result?;
        flight.require_accesses([(&read.lease, DeviceAccess::Read)])
    }
}
