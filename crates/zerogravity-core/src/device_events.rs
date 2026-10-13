//! Typed asynchronous-device event and buffer-lease protocol.
//!
//! This is the device-INDEPENDENT control/data ownership contract, not a GPU
//! implementation. A driver must report completion only after a real backend
//! event/fence has established that all memory effects are complete.
//!
//! Until then, buffers stay pinned even after task cancellation or an epoch
//! switch. An uncertain device failure poisons the affected leases: neither
//! CPU fallback, buffer reuse nor StateRoot publication is permitted.
//!
//! Only the engine's logical coordinator may publish a root. The event ledger is an
//! admission/fencing precondition, never a competing state writer.

use crate::state::{CommittedOutput, PreparedDomain, TaskCoordinator};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferLeaseId(u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceEventId {
    epoch: u64,
    sequence: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferRole {
    ImmutableWeights,
    Activation,
    Scratch,
    RecurrentState,
    Output,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceAccess {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventPhase {
    Submitted,
    CancellationRequested,
    /// Backend signaled actual completion; memory remains pinned until
    /// publication, explicit cancellation or event retirement.
    Ready,
    /// The backend cannot prove physical completion after failure. Its
    /// leases must remain fenced until trusted device teardown.
    CompletionUnknown,
}

struct Lease {
    bytes: usize,
    #[allow(dead_code)]
    role: BufferRole,
    readers: usize,
    writer: bool,
    fenced: bool,
}
struct Event {
    expected_root: u64,
    accesses: Vec<(BufferLeaseId, DeviceAccess)>,
    phase: EventPhase,
}

/// Single-device event control. One instance belongs to one actual service
/// and task epoch; a higher ResourceArbiter may own coarse admission.
pub struct DeviceEventLedger {
    epoch: u64,
    budget_bytes: usize,
    resident_bytes: usize,
    next_lease: u64,
    next_event: u64,
    leases: BTreeMap<BufferLeaseId, Lease>,
    events: BTreeMap<DeviceEventId, Event>,
}

impl DeviceEventLedger {
    pub fn new(epoch: u64, budget_bytes: usize) -> Result<Self> {
        if budget_bytes == 0 {
            return Err(Error::Limit("device lease budget must be positive".into()));
        }
        Ok(Self {
            epoch,
            budget_bytes,
            resident_bytes: 0,
            next_lease: 1,
            next_event: 1,
            leases: BTreeMap::new(),
            events: BTreeMap::new(),
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
    pub fn pending_events(&self) -> usize {
        self.events.len()
    }

    pub fn admit(&mut self, bytes: usize, role: BufferRole) -> Result<BufferLeaseId> {
        if bytes == 0 || bytes > self.budget_bytes.saturating_sub(self.resident_bytes) {
            return Err(Error::Limit("device byte admission budget exceeded".into()));
        }
        let next = self
            .next_lease
            .checked_add(1)
            .ok_or_else(|| Error::Limit("device lease sequence exhausted".into()))?;
        let id = BufferLeaseId(self.next_lease);
        self.next_lease = next;
        self.leases.insert(
            id,
            Lease {
                bytes,
                role,
                readers: 0,
                writer: false,
                fenced: false,
            },
        );
        self.resident_bytes += bytes;
        Ok(id)
    }

    /// Admission is all-or-nothing. A device-buffer write cannot overlap a
    /// read or write; multiple immutable reads can share one pinned lease.
    pub fn submit(
        &mut self,
        expected_root: u64,
        accesses: &[(BufferLeaseId, DeviceAccess)],
    ) -> Result<DeviceEventId> {
        if accesses.is_empty() {
            return Err(Error::State(
                "device submission requires buffer leases".into(),
            ));
        }
        let mut unique = BTreeSet::new();
        for &(id, access) in accesses {
            if !unique.insert(id) {
                return Err(Error::State(
                    "duplicate buffer lease in one device event".into(),
                ));
            }
            let lease = self
                .leases
                .get(&id)
                .ok_or_else(|| Error::State("unknown device buffer lease".into()))?;
            if lease.fenced || lease.writer || (access == DeviceAccess::Write && lease.readers > 0)
            {
                return Err(Error::State("device lease not safely available".into()));
            }
        }
        let next = self
            .next_event
            .checked_add(1)
            .ok_or_else(|| Error::Limit("device event sequence exhausted".into()))?;
        let id = DeviceEventId {
            epoch: self.epoch,
            sequence: self.next_event,
        };
        self.next_event = next;
        for &(lease_id, access) in accesses {
            let lease = self
                .leases
                .get_mut(&lease_id)
                .expect("all source leases admitted before pinning");
            match access {
                DeviceAccess::Read => lease.readers += 1,
                DeviceAccess::Write => lease.writer = true,
            }
        }
        self.events.insert(
            id,
            Event {
                expected_root,
                accesses: accesses.to_vec(),
                phase: EventPhase::Submitted,
            },
        );
        Ok(id)
    }

    pub fn phase(&self, id: DeviceEventId) -> Result<EventPhase> {
        self.events
            .get(&id)
            .map(|v| v.phase)
            .ok_or_else(|| Error::State("event absent, retired or already published".into()))
    }

    /// Requests cancellation, but does NOT release or reuse pending GPU
    /// buffers. Backend must still report physical completion or teardown.
    pub fn request_cancel(&mut self, id: DeviceEventId) -> Result<()> {
        let phase = self.phase(id)?;
        match phase {
            EventPhase::Submitted | EventPhase::CancellationRequested => {
                self.events.get_mut(&id).expect("observed event").phase =
                    EventPhase::CancellationRequested;
                Ok(())
            }
            EventPhase::Ready => self.retire(id),
            EventPhase::CompletionUnknown => Err(Error::State(
                "device completion unknown; leases remain fenced".into(),
            )),
        }
    }

    /// Must ONLY be called after the physical device event was actually
    /// signaled. A cancelled in-flight event is retired only NOW.
    pub fn report_physical_ready(&mut self, id: DeviceEventId) -> Result<()> {
        match self.phase(id)? {
            EventPhase::Submitted => {
                self.events.get_mut(&id).expect("observed event").phase = EventPhase::Ready;
                Ok(())
            }
            EventPhase::CancellationRequested => self.retire(id),
            _ => Err(Error::State(
                "duplicate, stale or uncertain completion".into(),
            )),
        }
    }

    /// Retire a completed physical event while preserving admitted allocations.
    /// Resident outputs retain their leases and are published separately by the
    /// logical session authority after all root/domain checks. This method
    /// neither initializes a buffer nor proves that a proposed root is valid.
    pub fn retire_physical_ready(&mut self, id: DeviceEventId) -> Result<()> {
        if id.epoch != self.epoch || self.phase(id)? != EventPhase::Ready {
            return Err(Error::State(
                "event is not current physically Ready work".into(),
            ));
        }
        self.retire(id)
    }

    /// An unobserved GPU failure cannot be treated as completed. Poison the
    /// associated leases and retain their ownership until provider teardown.
    pub fn report_completion_unknown(&mut self, id: DeviceEventId) -> Result<()> {
        let phase = self.phase(id)?;
        if phase != EventPhase::Submitted && phase != EventPhase::CancellationRequested {
            return Err(Error::State("cannot lose already completed event".into()));
        }
        let refs = self
            .events
            .get(&id)
            .expect("observed event")
            .accesses
            .clone();
        for &(lease_id, _) in &refs {
            self.leases
                .get_mut(&lease_id)
                .expect("tracked lease")
                .fenced = true;
        }
        self.events.get_mut(&id).expect("observed event").phase = EventPhase::CompletionUnknown;
        Ok(())
    }

    /// Epoch fencing cancels still-inflight old work; cancelled buffers
    /// remain pinned until their own physical ready event arrives.
    pub fn advance_epoch(&mut self, next_epoch: u64) -> Result<()> {
        if next_epoch <= self.epoch {
            return Err(Error::State("device epoch did not advance".into()));
        }
        let ready = self
            .events
            .iter()
            .filter(|(_, v)| v.phase == EventPhase::Ready)
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        for id in ready {
            self.retire(id)?;
        }
        for event in self.events.values_mut() {
            if event.phase == EventPhase::Submitted {
                event.phase = EventPhase::CancellationRequested;
            }
        }
        self.epoch = next_epoch;
        Ok(())
    }

    /// A Ready event gates publication, but TaskCoordinator remains the
    /// SINGLE writer and independently validates domains, root, frontier.
    /// On an invalid candidate the Ready event stays pinned for cancellation
    /// or a corrected publication, never becomes a partially written root.
    pub fn publish_ready(
        &mut self,
        id: DeviceEventId,
        coordinator: &mut TaskCoordinator,
        frontier: u64,
        domains: Vec<PreparedDomain>,
        payload: Vec<u8>,
    ) -> Result<CommittedOutput> {
        let event = self
            .events
            .get(&id)
            .ok_or_else(|| Error::State("unknown ready event".into()))?;
        if id.epoch != self.epoch
            || coordinator.epoch != self.epoch
            || event.phase != EventPhase::Ready
        {
            return Err(Error::State(
                "not current-epoch physically ready device result".into(),
            ));
        }
        let result = coordinator.publish(event.expected_root, frontier, domains, payload)?;
        self.retire(id)?;
        Ok(result)
    }

    /// Release a buffer only if no asynchronous event still references it.
    pub fn release_lease(&mut self, id: BufferLeaseId) -> Result<()> {
        let lease = self
            .leases
            .get(&id)
            .ok_or_else(|| Error::State("unknown or already released buffer".into()))?;
        if lease.fenced || lease.writer || lease.readers != 0 {
            return Err(Error::State(
                "buffer still pinned or completion uncertain".into(),
            ));
        }
        let bytes = lease.bytes;
        self.leases.remove(&id);
        self.resident_bytes -= bytes;
        Ok(())
    }

    fn retire(&mut self, id: DeviceEventId) -> Result<()> {
        let event = self
            .events
            .remove(&id)
            .ok_or_else(|| Error::State("event already retired".into()))?;
        if event.phase == EventPhase::CompletionUnknown {
            self.events.insert(id, event);
            return Err(Error::State(
                "unknown completion event cannot retire".into(),
            ));
        }
        for (lease_id, access) in event.accesses {
            let lease = self
                .leases
                .get_mut(&lease_id)
                .expect("registered lease pinned until physical retirement");
            match access {
                DeviceAccess::Read => lease.readers -= 1,
                DeviceAccess::Write => lease.writer = false,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn physical_retirement_preserves_resident_output_admission() {
        let mut ledger = DeviceEventLedger::new(1, 64).unwrap();
        let output = ledger.admit(32, BufferRole::RecurrentState).unwrap();
        let event = ledger.submit(0, &[(output, DeviceAccess::Write)]).unwrap();
        assert!(ledger.retire_physical_ready(event).is_err());
        assert!(ledger.release_lease(output).is_err());
        ledger.report_physical_ready(event).unwrap();
        ledger.retire_physical_ready(event).unwrap();
        assert_eq!(ledger.pending_events(), 0);
        assert_eq!(ledger.resident_bytes(), 32);
        let reader = ledger.submit(0, &[(output, DeviceAccess::Read)]).unwrap();
        ledger.report_completion_unknown(reader).unwrap();
        assert!(ledger.retire_physical_ready(reader).is_err());
        assert!(ledger.release_lease(output).is_err());
        assert_eq!(ledger.resident_bytes(), 32);
    }

    use super::*;

    fn domain(name: &str, schema: u64, frontier: u64) -> PreparedDomain {
        PreparedDomain {
            name: name.into(),
            schema,
            frontier,
            bytes: vec![frontier as u8],
        }
    }
    #[test]
    fn event_admission_ready_publication_and_read_write_fencing() {
        let mut ledger = DeviceEventLedger::new(7, 100).unwrap();
        let weights = ledger.admit(40, BufferRole::ImmutableWeights).unwrap();
        let output = ledger.admit(20, BufferRole::Output).unwrap();
        assert!(ledger.admit(41, BufferRole::Scratch).is_err());
        let event = ledger
            .submit(
                0,
                &[(weights, DeviceAccess::Read), (output, DeviceAccess::Write)],
            )
            .unwrap();
        assert_eq!(ledger.pending_events(), 1);
        assert!(ledger.release_lease(output).is_err());
        assert!(ledger.submit(0, &[(output, DeviceAccess::Read)]).is_err());
        // An independent reader may access the same immutable weights.
        let second = ledger.submit(0, &[(weights, DeviceAccess::Read)]).unwrap();
        assert!(ledger.report_physical_ready(second).is_ok());
        let mut coordinator = TaskCoordinator::new(7, [("gpu".into(), 1)]).unwrap();
        assert!(
            ledger
                .publish_ready(
                    event,
                    &mut coordinator,
                    1,
                    vec![domain("gpu", 1, 1)],
                    vec![99]
                )
                .is_err()
        );
        assert_eq!(coordinator.root().id, 0);
        ledger.report_physical_ready(event).unwrap();
        // Incomplete candidate must fail without consuming the event/root.
        assert!(
            ledger
                .publish_ready(event, &mut coordinator, 1, vec![], vec![])
                .is_err()
        );
        assert_eq!(coordinator.root().id, 0);
        let out = ledger
            .publish_ready(
                event,
                &mut coordinator,
                1,
                vec![domain("gpu", 1, 1)],
                vec![99],
            )
            .unwrap();
        assert_eq!((out.root_id, out.epoch, out.sequence), (1, 7, 1));
        assert!(
            ledger
                .publish_ready(
                    event,
                    &mut coordinator,
                    2,
                    vec![domain("gpu", 1, 2)],
                    vec![]
                )
                .is_err()
        );
        ledger.request_cancel(second).unwrap();
        assert_eq!(ledger.pending_events(), 0);
        ledger.release_lease(weights).unwrap();
        ledger.release_lease(output).unwrap();
        assert_eq!(ledger.resident_bytes(), 0);
    }

    #[test]
    fn cancellation_and_epoch_change_cannot_free_unsignalled_work() {
        let mut ledger = DeviceEventLedger::new(1, 64).unwrap();
        let lease = ledger.admit(32, BufferRole::Activation).unwrap();
        let e = ledger.submit(0, &[(lease, DeviceAccess::Write)]).unwrap();
        ledger.request_cancel(e).unwrap();
        assert_eq!(ledger.phase(e).unwrap(), EventPhase::CancellationRequested);
        assert!(ledger.release_lease(lease).is_err());
        ledger.advance_epoch(2).unwrap();
        let mut c = TaskCoordinator::new(2, [("state".into(), 1)]).unwrap();
        assert!(
            ledger
                .publish_ready(e, &mut c, 1, vec![domain("state", 1, 1)], vec![])
                .is_err()
        );
        assert_eq!(c.root().id, 0);
        ledger.report_physical_ready(e).unwrap();
        assert!(ledger.phase(e).is_err());
        ledger.release_lease(lease).unwrap();
        assert!(ledger.advance_epoch(2).is_err());
    }

    #[test]
    fn uncertain_gpu_failure_poison_prevents_reuse_and_state_commit() {
        let mut ledger = DeviceEventLedger::new(3, 128).unwrap();
        let lease = ledger.admit(60, BufferRole::RecurrentState).unwrap();
        let event = ledger.submit(0, &[(lease, DeviceAccess::Write)]).unwrap();
        ledger.report_completion_unknown(event).unwrap();
        assert_eq!(ledger.phase(event).unwrap(), EventPhase::CompletionUnknown);
        assert!(ledger.release_lease(lease).is_err());
        assert!(ledger.submit(0, &[(lease, DeviceAccess::Read)]).is_err());
        assert!(ledger.request_cancel(event).is_err());
        assert!(ledger.report_physical_ready(event).is_err());
        let mut c = TaskCoordinator::new(3, [("state".into(), 1)]).unwrap();
        assert!(
            ledger
                .publish_ready(event, &mut c, 1, vec![domain("state", 1, 1)], vec![])
                .is_err()
        );
        assert_eq!(c.root().id, 0);
    }

    #[test]
    fn asynchronous_completion_after_cancel_never_reuses_buffer_early() {
        use std::sync::{Arc, Mutex, mpsc};
        use std::thread;

        let ledger = Arc::new(Mutex::new(DeviceEventLedger::new(11, 128).unwrap()));
        let (id, lease) = {
            let mut l = ledger.lock().unwrap();
            let lease = l.admit(64, BufferRole::Output).unwrap();
            let id = l.submit(0, &[(lease, DeviceAccess::Write)]).unwrap();
            (id, lease)
        };
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (completed_tx, completed_rx) = mpsc::channel::<()>();
        let worker_ledger = ledger.clone();
        let worker = thread::spawn(move || {
            go_rx.recv().unwrap();
            // Test backend: after a mocked physical event signaled,
            // report ready and notify host. No actual GPU is involved.
            worker_ledger
                .lock()
                .unwrap()
                .report_physical_ready(id)
                .unwrap();
            completed_tx.send(()).unwrap();
        });
        {
            let mut l = ledger.lock().unwrap();
            l.request_cancel(id).unwrap();
            assert_eq!(l.phase(id).unwrap(), EventPhase::CancellationRequested);
            assert!(l.release_lease(lease).is_err());
            assert!(l.submit(0, &[(lease, DeviceAccess::Read)]).is_err());
        }
        go_tx.send(()).unwrap();
        completed_rx.recv().unwrap();
        worker.join().unwrap();
        let mut l = ledger.lock().unwrap();
        assert_eq!(l.pending_events(), 0);
        l.release_lease(lease).unwrap();
        assert_eq!(l.resident_bytes(), 0);
    }

    #[test]
    fn out_of_order_ready_cannot_override_committed_root() {
        let mut l = DeviceEventLedger::new(12, 96).unwrap();
        let weights = l.admit(32, BufferRole::ImmutableWeights).unwrap();
        let a = l.admit(16, BufferRole::Output).unwrap();
        let b = l.admit(16, BufferRole::Output).unwrap();
        let first = l
            .submit(
                0,
                &[(weights, DeviceAccess::Read), (a, DeviceAccess::Write)],
            )
            .unwrap();
        let second = l
            .submit(
                0,
                &[(weights, DeviceAccess::Read), (b, DeviceAccess::Write)],
            )
            .unwrap();
        // Worker 2 becomes ready first, but only coordinator can publish.
        l.report_physical_ready(second).unwrap();
        l.report_physical_ready(first).unwrap();
        let mut coord = TaskCoordinator::new(12, [("gpu".into(), 1)]).unwrap();
        let committed = l
            .publish_ready(second, &mut coord, 1, vec![domain("gpu", 1, 1)], vec![2])
            .unwrap();
        assert_eq!((committed.root_id, committed.sequence), (1, 1));
        assert!(
            l.publish_ready(first, &mut coord, 2, vec![domain("gpu", 1, 2)], vec![1])
                .is_err()
        );
        assert_eq!(coord.root().id, 1);
        l.request_cancel(first).unwrap();
        for lease in [weights, a, b] {
            l.release_lease(lease).unwrap();
        }
        assert_eq!(l.resident_bytes(), 0);
    }

    #[test]
    fn bad_submission_is_atomic_and_stale_root_is_rejected() {
        let mut ledger = DeviceEventLedger::new(8, 90).unwrap();
        assert!(ledger.admit(0, BufferRole::Scratch).is_err());
        let a = ledger.admit(40, BufferRole::ImmutableWeights).unwrap();
        let b = ledger.admit(40, BufferRole::Output).unwrap();
        assert!(
            ledger
                .submit(0, &[(a, DeviceAccess::Read), (a, DeviceAccess::Write)])
                .is_err()
        );
        assert!(
            ledger
                .submit(
                    0,
                    &[
                        (a, DeviceAccess::Read),
                        (BufferLeaseId(999), DeviceAccess::Write)
                    ]
                )
                .is_err()
        );
        assert_eq!(ledger.pending_events(), 0);
        let event = ledger
            .submit(123, &[(a, DeviceAccess::Read), (b, DeviceAccess::Write)])
            .unwrap();
        ledger.report_physical_ready(event).unwrap();
        let mut coordinator = TaskCoordinator::new(8, [("gpu".into(), 1)]).unwrap();
        assert!(
            ledger
                .publish_ready(
                    event,
                    &mut coordinator,
                    1,
                    vec![domain("gpu", 1, 1)],
                    vec![]
                )
                .is_err()
        );
        assert_eq!(coordinator.root().id, 0);
        ledger.request_cancel(event).unwrap();
        ledger.release_lease(a).unwrap();
        ledger.release_lease(b).unwrap();
        assert_eq!(ledger.resident_bytes(), 0);
    }
}
