//! P07-G: bounded, fair *control-plane* queue for heterogeneous device tasks.
//!
//! One logical scheduler owns this queue. It is not a CUDA/ROCm/Vulkan
//! worker: dispatch creates a global RAII memory reservation before a task
//! could be handed to a real DeviceWorker; it does not execute model math,
//! issue GPU commands, or publish StateRoot.
//!
//! Round-robin fairness is per (model,session) group, with explicit queue
//! capacity/backpressure, cancelled and expired task removal, epoch fencing,
//! and the ability to skip a temporarily memory-blocked group. A separate
//! physical backend must own and signal actual completion/worker receipts.

use crate::resource_arbiter::{
    AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Debug)]
pub struct DeviceTask {
    pub id: u64,
    pub epoch: u64,
    pub owner: ResourceOwner,
    pub allocations: Vec<AllocationRequest>,
    pub deadline_tick: Option<u64>,
}

pub struct AdmittedTask {
    pub task: DeviceTask,
    /// Cannot launch physical work without first marking this submitted.
    /// Drop before submission releases reservation; dropping while in flight
    /// fences the resource service until trustworthy recovery.
    pub reservation: GlobalResourceLease,
}

pub enum DispatchChoice {
    Admitted(AdmittedTask),
    Expired(DeviceTask),
    NoCapacity,
    Empty,
}

/// Fair admission of tasks whose static and transient physical allocations
/// are ALREADY covered by an attested, held provider envelope. No extra
/// global reservation is made: doing so would double-charge one GPU buffer.
/// The physical worker must verify the envelope and Ready/fence before launch.
pub enum PrepaidChoice {
    Dispatched(DeviceTask),
    Expired(DeviceTask),
    NoCapacity,
    Empty,
}

/// Control-only queues, no durable workers or remote task handles.
pub struct FairDeviceQueue {
    epoch: u64,
    capacity: usize,
    outstanding: usize,
    groups: BTreeMap<(String, String), VecDeque<DeviceTask>>,
    next_groups: VecDeque<(String, String)>,
    ids: BTreeSet<u64>,
}
impl FairDeviceQueue {
    pub fn new(epoch: u64, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Limit("device queue capacity must be nonzero".into()));
        }
        Ok(Self {
            epoch,
            capacity,
            outstanding: 0,
            groups: BTreeMap::new(),
            next_groups: VecDeque::new(),
            ids: BTreeSet::new(),
        })
    }

    pub fn queued(&self) -> usize {
        self.outstanding
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn enqueue(&mut self, task: DeviceTask) -> Result<()> {
        if task.id == 0 || task.epoch != self.epoch || task.allocations.is_empty() {
            return Err(Error::State("invalid or stale device task".into()));
        }
        if self.outstanding >= self.capacity {
            return Err(Error::Limit(
                "bounded device queue full: backpressure".into(),
            ));
        }
        if !self.ids.insert(task.id) {
            return Err(Error::State("duplicate device task ID".into()));
        }
        let key = (task.owner.model.clone(), task.owner.session.clone());
        let bucket = self.groups.entry(key.clone()).or_default();
        if bucket.is_empty() {
            self.next_groups.push_back(key);
        }
        bucket.push_back(task);
        self.outstanding += 1;
        Ok(())
    }

    /// Cancelling a queued task cannot affect a physically dispatched one;
    /// the DeviceEventLedger/ResourceLease owns actual in-flight cancellation.
    pub fn cancel_queued(&mut self, id: u64) -> bool {
        if !self.ids.contains(&id) {
            return false;
        }
        let key = self.groups.iter().find_map(|(key, v)| {
            if v.iter().any(|t| t.id == id) {
                Some(key.clone())
            } else {
                None
            }
        });
        let Some(key) = key else { return false };
        let items = self.groups.get_mut(&key).expect("tracked group");
        let idx = items.iter().position(|x| x.id == id).expect("tracked task");
        items.remove(idx);
        self.outstanding -= 1;
        self.ids.remove(&id);
        if items.is_empty() {
            self.groups.remove(&key);
            self.next_groups.retain(|x| x != &key);
        }
        true
    }

    /// New task epochs discard *queued* work. In-flight GPU work is NOT
    /// managed here and remains fenced by its own lease/event owner.
    pub fn advance_epoch(&mut self, next: u64) -> Result<usize> {
        if next <= self.epoch {
            return Err(Error::State("stale queue epoch".into()));
        }
        let discarded = self.outstanding;
        self.epoch = next;
        self.outstanding = 0;
        self.groups.clear();
        self.next_groups.clear();
        self.ids.clear();
        Ok(discarded)
    }

    /// Consider at most one head item per model/session per dispatch. A
    /// transient device memory refusal cannot monopolize the worker.
    /// Expired requests are surfaced for caller notification, not executed.
    pub fn next(
        &mut self,
        now_tick: u64,
        arbiter: &ResourceArbiter,
        device: &PhysicalDeviceKey,
    ) -> Result<DispatchChoice> {
        if self.outstanding == 0 {
            return Ok(DispatchChoice::Empty);
        }
        let visits = self.next_groups.len();
        for _ in 0..visits {
            let key = self
                .next_groups
                .pop_front()
                .ok_or_else(|| Error::State("queue group index corrupted".into()))?;
            let task = self
                .groups
                .get(&key)
                .and_then(|items| items.front())
                .ok_or_else(|| Error::State("nonempty queue group missing head".into()))?
                .clone();
            if task.deadline_tick.is_some_and(|d| d < now_tick) {
                self.take_front(&key)?;
                return Ok(DispatchChoice::Expired(task));
            }
            match arbiter.admit(device, task.owner.clone(), &task.allocations) {
                Ok(reservation) => {
                    self.take_front(&key)?;
                    return Ok(DispatchChoice::Admitted(AdmittedTask { task, reservation }));
                }
                Err(Error::Limit(_)) => {
                    // Try another group's smaller request without letting a
                    // memory-blocked task reorder its own request stream.
                    self.next_groups.push_back(key);
                }
                Err(e) => {
                    self.next_groups.push_front(key);
                    return Err(e);
                }
            }
        }
        Ok(DispatchChoice::NoCapacity)
    }

    /// One RR choice with PREPAID physical memory. `can_dispatch` must
    /// validate an existing live owner lease, local suballocation upper bound
    /// and physical device fence; a raw `DeviceTask` is NOT proof of any of
    /// those conditions. Workers release the ready slot only after backend
    /// quiescence (or terminal physical quarantine).
    pub fn next_prepaid(
        &mut self,
        now_tick: u64,
        mut can_dispatch: impl FnMut(&DeviceTask) -> Result<bool>,
    ) -> Result<PrepaidChoice> {
        if self.outstanding == 0 {
            return Ok(PrepaidChoice::Empty);
        }
        let visits = self.next_groups.len();
        for _ in 0..visits {
            let key = self
                .next_groups
                .pop_front()
                .ok_or_else(|| Error::State("queue group index corrupted".into()))?;
            let task = self
                .groups
                .get(&key)
                .and_then(|items| items.front())
                .ok_or_else(|| Error::State("nonempty queue group missing head".into()))?
                .clone();
            if task.deadline_tick.is_some_and(|d| d < now_tick) {
                self.take_front(&key)?;
                return Ok(PrepaidChoice::Expired(task));
            }
            match can_dispatch(&task) {
                Ok(true) => {
                    self.take_front(&key)?;
                    return Ok(PrepaidChoice::Dispatched(task));
                }
                Ok(false) => self.next_groups.push_back(key),
                Err(error) => {
                    self.next_groups.push_front(key);
                    return Err(error);
                }
            }
        }
        Ok(PrepaidChoice::NoCapacity)
    }

    fn take_front(&mut self, key: &(String, String)) -> Result<()> {
        let items = self
            .groups
            .get_mut(key)
            .ok_or_else(|| Error::State("queue selected group missing".into()))?;
        let task = items
            .pop_front()
            .ok_or_else(|| Error::State("queue selected task missing".into()))?;
        self.ids.remove(&task.id);
        self.outstanding -= 1;
        if items.is_empty() {
            self.groups.remove(key);
        } else {
            self.next_groups.push_back(key.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_arbiter::AllocationClass;
    fn setup(capacity: usize) -> (FairDeviceQueue, ResourceArbiter, PhysicalDeviceKey) {
        let device = PhysicalDeviceKey::new("cuda", "0").unwrap();
        let arbiter = ResourceArbiter::new();
        arbiter.register(device.clone(), 100).unwrap();
        (FairDeviceQueue::new(3, capacity).unwrap(), arbiter, device)
    }
    fn task(id: u64, model: &str, session: &str, bytes: u64) -> DeviceTask {
        DeviceTask {
            id,
            epoch: 3,
            owner: ResourceOwner::new(model, session).unwrap(),
            allocations: vec![AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes,
            }],
            deadline_tick: None,
        }
    }
    #[test]
    fn round_robin_between_models_not_infinite_priority_stealing() {
        let (mut queue, a, gpu) = setup(5);
        for id in 1..=3 {
            queue.enqueue(task(id, "model-A", "s1", 10)).unwrap();
        }
        queue.enqueue(task(4, "model-B", "s2", 10)).unwrap();
        let first = queue.next(0, &a, &gpu).unwrap();
        let DispatchChoice::Admitted(first) = first else {
            panic!("first not admitted")
        };
        assert_eq!(first.task.id, 1);
        let second = queue.next(0, &a, &gpu).unwrap();
        let DispatchChoice::Admitted(second) = second else {
            panic!("second not admitted")
        };
        assert_eq!(second.task.id, 4);
        first.reservation.release().unwrap();
        second.reservation.release().unwrap();
        let third = queue.next(0, &a, &gpu).unwrap();
        let DispatchChoice::Admitted(third) = third else {
            panic!("third not admitted")
        };
        assert_eq!(third.task.id, 2);
        third.reservation.release().unwrap();
    }
    #[test]
    fn resource_blocked_large_request_does_not_starve_small_model() {
        let (mut queue, a, gpu) = setup(3);
        let occupied = a
            .admit(
                &gpu,
                ResourceOwner::new("other", "s").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ImmutableWeights,
                    bytes: 80,
                }],
            )
            .unwrap();
        queue.enqueue(task(1, "big", "s", 40)).unwrap();
        queue.enqueue(task(2, "small", "s", 10)).unwrap();
        let selected = queue.next(0, &a, &gpu).unwrap();
        let DispatchChoice::Admitted(selected) = selected else {
            panic!("small was blocked")
        };
        assert_eq!(selected.task.id, 2);
        selected.reservation.release().unwrap();
        assert!(matches!(
            queue.next(0, &a, &gpu).unwrap(),
            DispatchChoice::NoCapacity
        ));
        occupied.release().unwrap();
        let DispatchChoice::Admitted(next) = queue.next(0, &a, &gpu).unwrap() else {
            panic!("capacity not released")
        };
        assert_eq!(next.task.id, 1);
        next.reservation.release().unwrap();
    }
    #[test]
    fn queue_backpressure_cancellation_deadline_and_epoch() {
        let (mut queue, a, gpu) = setup(2);
        let mut first = task(1, "model-A", "s1", 5);
        first.deadline_tick = Some(10);
        queue.enqueue(first).unwrap();
        queue.enqueue(task(2, "model-B", "s2", 5)).unwrap();
        assert!(queue.enqueue(task(3, "model-C", "s3", 5)).is_err());
        assert!(queue.enqueue(task(2, "model-B", "s2", 5)).is_err());
        let first = queue.next(11, &a, &gpu).unwrap();
        let DispatchChoice::Expired(expired) = first else {
            panic!("deadline not checked")
        };
        assert_eq!(expired.id, 1);
        assert!(queue.cancel_queued(2));
        assert!(!queue.cancel_queued(2));
        assert!(matches!(
            queue.next(11, &a, &gpu).unwrap(),
            DispatchChoice::Empty
        ));
        queue.enqueue(task(3, "model-C", "s3", 5)).unwrap();
        assert_eq!(queue.advance_epoch(4).unwrap(), 1);
        assert_eq!(queue.queued(), 0);
        assert!(queue.enqueue(task(4, "model-A", "s1", 5)).is_err());
    }
    #[test]
    fn prepaid_work_reuses_fair_order_without_reserving_memory_twice() {
        let (mut queue, arbiter, card) = setup(5);
        // The actual CubeCL resident source and scratch are subdivisions of
        // an existing provider envelope, not extra device allocations.
        let envelope = arbiter
            .admit(
                &card,
                ResourceOwner::new("qwen", "resident").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: 90,
                }],
            )
            .unwrap();
        queue.enqueue(task(1, "a", "s1", 20)).unwrap();
        queue.enqueue(task(2, "a", "s1", 20)).unwrap();
        queue.enqueue(task(3, "b", "s2", 15)).unwrap();
        assert_eq!(queue.queued(), 3);
        let PrepaidChoice::Dispatched(first) = queue
            .next_prepaid(0, |t| Ok(t.allocations[0].bytes <= 20))
            .unwrap()
        else {
            panic!("first prepaid task not dispatched")
        };
        assert_eq!(first.id, 1);
        let PrepaidChoice::Dispatched(second) = queue
            .next_prepaid(0, |t| Ok(t.allocations[0].bytes <= 20))
            .unwrap()
        else {
            panic!("other model was starved")
        };
        assert_eq!(second.id, 3);
        let PrepaidChoice::Dispatched(third) = queue
            .next_prepaid(0, |t| Ok(t.allocations[0].bytes <= 20))
            .unwrap()
        else {
            panic!("remaining task not dispatched")
        };
        assert_eq!(third.id, 2);
        // The device still holds exactly ONE physical reservation.
        let snapshot = arbiter.snapshot(&card).unwrap();
        assert_eq!(snapshot.reserved, 90);
        assert_eq!(snapshot.reservations, 1);
        envelope.release().unwrap();
        assert_eq!(arbiter.snapshot(&card).unwrap().reserved, 0);
    }

    #[test]
    fn prepaid_declined_and_deadline_tasks_preserve_fairness_and_boundedness() {
        let (mut queue, _, _) = setup(2);
        let mut old = task(1, "a", "s1", 32);
        old.deadline_tick = Some(4);
        queue.enqueue(old).unwrap();
        queue.enqueue(task(2, "b", "s2", 5)).unwrap();
        let PrepaidChoice::Expired(expired) = queue.next_prepaid(5, |_| Ok(true)).unwrap() else {
            panic!("expired queued task executed")
        };
        assert_eq!(expired.id, 1);
        queue.enqueue(task(3, "a", "s1", 32)).unwrap();
        let PrepaidChoice::Dispatched(small) = queue
            .next_prepaid(5, |t| Ok(t.allocations[0].bytes <= 10))
            .unwrap()
        else {
            panic!("large blocked request starved a small model")
        };
        assert_eq!(small.id, 2);
        assert!(matches!(
            queue.next_prepaid(5, |_| Ok(false)).unwrap(),
            PrepaidChoice::NoCapacity
        ));
        assert_eq!(queue.queued(), 1);
        assert!(queue.cancel_queued(3));
    }

    #[test]
    fn unknown_device_failure_aborts_without_dequeue_or_root_publication() {
        let (mut queue, a, gpu) = setup(2);
        queue.enqueue(task(1, "model", "s1", 50)).unwrap();
        let poisoned = a
            .admit(
                &gpu,
                ResourceOwner::new("gpu", "w").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 10,
                }],
            )
            .unwrap();
        poisoned.mark_submitted().unwrap();
        drop(poisoned);
        assert!(queue.next(0, &a, &gpu).is_err());
        assert_eq!(queue.queued(), 1);
        assert!(a.snapshot(&gpu).unwrap().fenced);
    }
}
