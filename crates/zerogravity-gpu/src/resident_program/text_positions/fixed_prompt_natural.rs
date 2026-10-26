//! Current-call integer positions on one admitted storage lease.
use super::*;
use crate::resident_inference::native_fragment::fixed_prompt_natural::FixedPromptCallInput;
use crate::resident_leases::ResidentService;
use zerogravity_core::device_events::DeviceAccess;

pub(crate) struct FixedPromptPositionsStorage {
    lease: ResidentLease,
}
pub(crate) struct FixedPromptPositions {
    storage: Arc<FixedPromptPositionsStorage>,
    input: Arc<FixedPromptCallInput>,
    bytes: [u8; POSITION_BYTES],
    submitted: AtomicBool,
    _host: WindowWorkspace,
}
impl FixedPromptPositionsStorage {
    pub(crate) fn prepare(init: &mut ResidentInitialization<'_>) -> Result<Arc<Self>> {
        let result = Arc::new(Self {
            lease: init.allocate_empty(POSITION_BYTES, BufferRole::Activation)?,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn bind(
        self: &Arc<Self>,
        input: Arc<FixedPromptCallInput>,
    ) -> Result<Arc<FixedPromptPositions>> {
        if self.lease.source_identity() != input.call().source()
            || self.lease.bytes() != POSITION_BYTES
            || self.lease.service_generation() != self.lease.service().generation()
        {
            return Err(Error::State(
                "canonical positions source/storage generation differs".into(),
            ));
        }
        let host = input.source().reserve_workspace(3 * POSITION_BYTES)?;
        let mut bytes = [0; POSITION_BYTES];
        for axis in 0..3 {
            for row in 0..input.rows() {
                let value = i64::try_from(input.base() + row)
                    .map_err(|_| Error::Limit("canonical position overflow".into()))?;
                let offset = (axis * 65 + row) * 8;
                bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
        }
        Ok(Arc::new(FixedPromptPositions {
            storage: self.clone(),
            input,
            bytes,
            submitted: AtomicBool::new(false),
            _host: host,
        }))
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
}
impl FixedPromptPositions {
    pub(crate) fn lease(&self) -> &ResidentLease {
        self.storage.lease()
    }
    pub(crate) fn accesses(&self) -> [(ResidentLease, DeviceAccess); 1] {
        [(self.lease().clone(), DeviceAccess::Write)]
    }
    // ORIGINAL armed call flight retains this owner and lists the lease Write
    // before this overwrite. No eager initialized flag or recovery operation.
    pub(crate) fn enqueue(
        &self,
        session: &ResidentSession,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        if cancelled.is_some_and(|v| v.load(Ordering::Acquire)) {
            return Err(Error::State("canonical positions cancelled".into()));
        }
        self.input.require_session(session)?;
        self.lease().service().ensure_usable()?;
        if self.submitted.swap(true, Ordering::AcqRel) {
            return Err(Error::State("canonical positions already submitted".into()));
        }
        self.lease().service().client().write(
            &self.lease().handle(),
            Bytes::from_bytes_vec(self.bytes.to_vec()),
        );
        Ok(1)
    }
    pub(crate) fn require_enqueue(
        &self,
        session: &ResidentSession,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        if cancelled.is_some_and(|v| v.load(Ordering::Acquire)) {
            return Err(Error::State("canonical positions cancelled".into()));
        }
        self.input.require_session(session)?;
        self.input.require_permit(permit, self.lease().service())?;
        if !self.submitted.load(Ordering::Acquire) {
            return Err(Error::State(
                "canonical positions writer has not been submitted".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn readback(&self) -> Result<ResidentReadback> {
        self.lease().readback(0, POSITION_BYTES)
    }
    pub(crate) fn validate(&self, raw: &[u8]) -> Result<()> {
        if raw != self.bytes || !self.submitted.load(Ordering::Acquire) {
            return Err(Error::State("canonical positions transport changed".into()));
        }
        Ok(())
    }
    pub(crate) fn retained(
        &self,
        input: &Arc<FixedPromptCallInput>,
        service: &Arc<ResidentService>,
    ) -> bool {
        Arc::ptr_eq(input, &self.input)
            && Arc::ptr_eq(service, self.lease().service())
            && self.lease().service_generation() == service.generation()
            && self.lease().bytes() == POSITION_BYTES
    }
}
