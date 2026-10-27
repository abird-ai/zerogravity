//! Actual short-HF graph-position stride513, retained on the original call flight.
use super::*;
use crate::resident_inference::native_fragment::text::TextCallInput;
use crate::resident_leases::ResidentService;
use zerogravity_core::device_events::DeviceAccess;

// Previous constants: SHORT_HF_POSITION_PITCH / SHORT_HF_POSITION_BYTES.
pub(crate) const TEXT_POSITION_PITCH: usize = 513;
pub(crate) const TEXT_POSITION_BYTES: usize = 3 * TEXT_POSITION_PITCH * 8;
pub(crate) struct TextPositionsStorage {
    lease: ResidentLease,
}
pub(crate) struct TextPositions {
    storage: Arc<TextPositionsStorage>,
    input: Arc<TextCallInput>,
    bytes: Box<[u8; TEXT_POSITION_BYTES]>,
    submitted: AtomicBool,
    _host: WindowWorkspace,
}
fn encode(base: usize, rows: usize) -> Result<Box<[u8; TEXT_POSITION_BYTES]>> {
    if !matches!(rows, 1 | 47 | 48 | 49 | 50 | 57 | 62 | 67 | 72)
        || base.checked_add(rows).is_none_or(|end| end > 79)
    {
        return Err(Error::Unsupported(
            "short-HF actual finite position span".into(),
        ));
    }
    let mut bytes = Box::new([0; TEXT_POSITION_BYTES]);
    for axis in 0..3 {
        for row in 0..rows {
            let value = i64::try_from(base + row)
                .map_err(|_| Error::Limit("short-HF position overflow".into()))?;
            let offset = (axis * TEXT_POSITION_PITCH + row) * 8;
            bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}
impl TextPositionsStorage {
    pub(crate) fn prepare(init: &mut ResidentInitialization<'_>) -> Result<Arc<Self>> {
        let result = Arc::new(Self {
            lease: init.allocate_empty(TEXT_POSITION_BYTES, BufferRole::Activation)?,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn bind(self: &Arc<Self>, input: Arc<TextCallInput>) -> Result<Arc<TextPositions>> {
        if self.lease.source_identity() != input.call().source()
            || self.lease.bytes() != TEXT_POSITION_BYTES
            || self.lease.service_generation() != self.lease.service().generation()
        {
            return Err(Error::State(
                "short-HF actual source/position storage generation".into(),
            ));
        }
        let host = input
            .source()
            .reserve_workspace(3 * TEXT_POSITION_BYTES + 64 * 1024)?;
        let bytes = encode(input.base(), input.rows())?;
        Ok(Arc::new(TextPositions {
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
impl TextPositions {
    pub(crate) fn lease(&self) -> &ResidentLease {
        self.storage.lease()
    }
    pub(crate) fn accesses(&self) -> [(ResidentLease, DeviceAccess); 1] {
        [(self.lease().clone(), DeviceAccess::Write)]
    }
    pub(crate) fn enqueue(
        &self,
        session: &ResidentSession,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        if cancelled.is_some_and(|v| v.load(Ordering::Acquire)) {
            return Err(Error::State("short HF request cancelled".into()));
        }
        self.input.require_session(session)?;
        self.input.require_permit(permit, self.lease().service())?;
        self.lease().service().ensure_usable()?;
        if self.submitted.swap(true, Ordering::AcqRel) {
            return Err(Error::State("short-HF positions already submitted".into()));
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
            return Err(Error::State("short HF request cancelled".into()));
        }
        self.input.require_session(session)?;
        self.input.require_permit(permit, self.lease().service())?;
        if !self.submitted.load(Ordering::Acquire) {
            return Err(Error::State(
                "short-HF position writer not submitted".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn readback(&self) -> Result<ResidentReadback> {
        self.lease().readback(0, TEXT_POSITION_BYTES)
    }
    pub(crate) fn validate(&self, raw: &[u8]) -> Result<()> {
        if raw != self.bytes.as_slice() || !self.submitted.load(Ordering::Acquire) {
            return Err(Error::State(
                "short-HF actual position transport changed".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn retained(
        &self,
        input: &Arc<TextCallInput>,
        service: &Arc<ResidentService>,
    ) -> bool {
        Arc::ptr_eq(input, &self.input)
            && Arc::ptr_eq(service, self.lease().service())
            && self.lease().service_generation() == service.generation()
            && self.lease().bytes() == TEXT_POSITION_BYTES
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_actual_stride_and_every_active_position_are_source_derived() -> Result<()> {
        assert_eq!(TEXT_POSITION_BYTES, 12312);
        for rows in [1, 47, 48, 49, 50, 57, 62, 67, 72] {
            for base in if rows == 1 {
                vec![47, 53, 57, 63, 62, 68, 67, 73, 72, 78]
            } else {
                vec![0]
            } {
                let bytes = encode(base, rows)?;
                for axis in 0..3 {
                    for row in 0..513 {
                        let start = (axis * 513 + row) * 8;
                        let value = i64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
                        assert_eq!(value, if row < rows { (base + row) as i64 } else { 0 });
                    }
                }
            }
        }
        for (base, rows) in [
            (0, 0),
            (0, 64),
            (79, 1),
            (78, 2),
            (1, usize::MAX),
            (usize::MAX, 1),
        ] {
            assert!(encode(base, rows).is_err());
        }
        Ok(())
    }
}
