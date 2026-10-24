//! One exact real-reader release boundary; no retained GPU/source owners.
use super::*;
use std::sync::{OnceLock, atomic::AtomicUsize};

struct Probe {
    generation: u64,
    lease: BufferLeaseId,
    fault: bool,
    release_hits: AtomicUsize,
    native_retirements: AtomicUsize,
    hits_before_native_retirement: AtomicUsize,
}
impl Probe {
    fn hit(&self, generation: u64, lease: BufferLeaseId) -> bool {
        if generation != self.generation || lease != self.lease {
            return false;
        }
        let first = self.release_hits.fetch_add(1, Ordering::AcqRel) == 0;
        self.fault && first
    }
    fn native_retirement(&self) {
        if self.native_retirements.fetch_add(1, Ordering::AcqRel) == 0 {
            self.hits_before_native_retirement
                .store(self.release_hits.load(Ordering::Acquire), Ordering::Release);
        }
    }
    fn finish(&self) -> Result<ReaderResult> {
        let result = ReaderResult {
            release_hits: self.release_hits.load(Ordering::Acquire),
            native_retirements: self.native_retirements.load(Ordering::Acquire),
            hits_before_native_retirement: self
                .hits_before_native_retirement
                .load(Ordering::Acquire),
        };
        if result.release_hits != 1
            || result.native_retirements != 1
            || result.hits_before_native_retirement != 1
        {
            return Err(Error::State(format!(
                "final-reader/native order differs: releases={} native={} prior_readers={}",
                result.release_hits,
                result.native_retirements,
                result.hits_before_native_retirement
            )));
        }
        Ok(result)
    }
}
static PROBE: OnceLock<Probe> = OnceLock::new();

pub(crate) struct ReaderProbe {
    pub(crate) lease: String,
    pub(crate) generation: u64,
    pub(crate) bytes: usize,
    pub(crate) aliases_inside_wrapper: usize,
}
pub(crate) struct ReaderResult {
    pub(crate) release_hits: usize,
    pub(crate) native_retirements: usize,
    pub(crate) hits_before_native_retirement: usize,
}
pub(crate) fn arm(
    lease: &ResidentLease,
    aliases_inside_wrapper: usize,
    fault: bool,
) -> Result<ReaderProbe> {
    lease.0.service.ensure_usable()?;
    if aliases_inside_wrapper == 0 || Arc::strong_count(&lease.0) != aliases_inside_wrapper {
        return Err(Error::State(
            "target reader has owners outside the final wrapper".into(),
        ));
    }
    if lease.0.quarantined.load(Ordering::Acquire) {
        return Err(Error::State("final root reader already quarantined".into()));
    }
    PROBE
        .set(Probe {
            generation: lease.0.service.generation,
            lease: lease.id(),
            fault,
            release_hits: AtomicUsize::new(0),
            native_retirements: AtomicUsize::new(0),
            hits_before_native_retirement: AtomicUsize::new(usize::MAX),
        })
        .map_err(|_| Error::State("one original final-reader probe per process".into()))?;
    Ok(ReaderProbe {
        lease: format!("{:?}", lease.id()),
        generation: lease.0.service.generation,
        bytes: lease.bytes(),
        aliases_inside_wrapper,
    })
}
pub(super) fn refuse_release(generation: u64, lease: BufferLeaseId) -> bool {
    PROBE
        .get()
        .is_some_and(|probe| probe.hit(generation, lease))
}
pub(crate) fn native_retirement() {
    if let Some(probe) = PROBE.get() {
        probe.native_retirement();
    }
}
pub(crate) fn result() -> Result<ReaderResult> {
    let probe = PROBE
        .get()
        .ok_or_else(|| Error::State("original final-reader probe absent".into()))?;
    probe.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn new_probe(fault: bool) -> (Probe, BufferLeaseId) {
        let mut ledger = DeviceEventLedger::new(19, 1024).unwrap();
        let selected = ledger.admit(64, BufferRole::RecurrentState).unwrap();
        let other = ledger.admit(64, BufferRole::RecurrentState).unwrap();
        (
            Probe {
                generation: 7,
                lease: selected,
                fault,
                release_hits: AtomicUsize::new(0),
                native_retirements: AtomicUsize::new(0),
                hits_before_native_retirement: AtomicUsize::new(usize::MAX),
            },
            other,
        )
    }
    #[test]
    fn final_reader_fault_cannot_match_another_service_or_buffer() {
        let (probe, other) = new_probe(true);
        for generation in [0, 6, 8, u64::MAX] {
            assert!(!probe.hit(generation, probe.lease));
        }
        assert!(!probe.hit(7, other));
        assert_eq!(probe.release_hits.load(Ordering::Acquire), 0);
        assert!(probe.hit(7, probe.lease));
        probe.native_retirement();
        assert!(probe.finish().is_ok());
    }
    #[test]
    fn known_reader_observation_never_requests_a_retirement_failure() {
        let (probe, _) = new_probe(false);
        assert!(!probe.hit(7, probe.lease));
        probe.native_retirement();
        assert!(probe.finish().is_ok());
    }
    #[test]
    fn native_retirement_before_the_actual_reader_is_rejected() {
        let (probe, _) = new_probe(true);
        probe.native_retirement();
        assert!(probe.hit(7, probe.lease));
        assert!(probe.finish().is_err());
    }
    #[test]
    fn duplicate_reader_or_native_boundary_is_rejected() {
        let (probe, _) = new_probe(true);
        assert!(probe.hit(7, probe.lease));
        assert!(!probe.hit(7, probe.lease));
        probe.native_retirement();
        assert!(probe.finish().is_err());
        let (probe, _) = new_probe(false);
        assert!(!probe.hit(7, probe.lease));
        probe.native_retirement();
        probe.native_retirement();
        assert!(probe.finish().is_err());
    }
}
