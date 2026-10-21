//! P07-G2b: one bounded, fair, cooperative GPU submission service per
//! attested physical card (not per CubeCL backend, model or session).
//!
//! Every queued job holds a BORROW of its immutable GGUF source region in the
//! awaiting caller. There is no repeated megabyte-sized source copy, extra
//! host worker inference engine, detached GPU future, or foreign runtime.
//! The selected Future drives the existing P07-E physical CubeCL readback;
//! the sole Qwen TaskCoordinator still publishes StateRoot after this returns.
//!
//! The global provider envelope has ALREADY reserved physical GPU memory;
//! the G1 fair queue is used in prepaid mode and must never charge it twice.
//! At most one physical region per card is dispatched by this conservative
//! service. Cancellation of a queued Future is safe and does not poison GPU.
//! Cancellation of an active Future is terminal and globally fences the card.
//! Physical real-device concurrency/cost tuning remains future work.

use crate::physical_service::PhysicalServiceBudget;
use std::{
    collections::{BTreeMap, BTreeSet},
    future::poll_fn,
    sync::{Arc, Mutex, OnceLock, Weak},
    task::{Context, Poll, Waker},
    time::Instant,
};
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::device_queue::{DeviceTask, FairDeviceQueue, PrepaidChoice};
use zerogravity_core::resource_arbiter::{
    AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

const DEFAULT_QUEUE_CAPACITY: usize = 32;

struct QueueState {
    queue: FairDeviceQueue,
    next_id: u64,
    active: Option<u64>,
    failed: BTreeSet<u64>,
    wakers: BTreeMap<u64, Waker>,
    selected: u64,
    completed: u64,
    cancelled_queued: u64,
    expired: u64,
}

pub struct PhysicalQueue {
    key: PhysicalDeviceKey,
    capacity: usize,
    epoch_origin: Instant,
    state: Mutex<QueueState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueStats {
    pub queued: usize,
    pub capacity: usize,
    pub active: bool,
    /// Tasks chosen by fair scheduling, BEFORE any physical GPU command.
    /// A recoverable source rejection may increment this but NOT complete.
    pub selected: u64,
    pub completed: u64,
    pub cancelled_queued: u64,
    pub expired: u64,
    pub epoch: u64,
}

impl PhysicalQueue {
    fn new(key: PhysicalDeviceKey, capacity: usize) -> Result<Self> {
        Ok(Self {
            key,
            capacity,
            epoch_origin: Instant::now(),
            state: Mutex::new(QueueState {
                queue: FairDeviceQueue::new(1, capacity)?,
                next_id: 1,
                active: None,
                failed: BTreeSet::new(),
                wakers: BTreeMap::new(),
                selected: 0,
                completed: 0,
                cancelled_queued: 0,
                expired: 0,
            }),
        })
    }

    fn now_tick(&self) -> u64 {
        u64::try_from(self.epoch_origin.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// The provider's live noncloneable resource envelope is REQUIRED: a
    /// caller cannot invent a prepaid byte count or use another model's GPU.
    pub fn submit(
        self: &Arc<Self>,
        service: &PhysicalServiceBudget,
        owner: ResourceOwner,
        allocations: &[AllocationRequest],
        deadline_tick: Option<u64>,
    ) -> Result<WaitingTicket> {
        let prepaid_bytes = service.prepaid_bytes_for(&self.key, &owner)?;
        self.enqueue_prepaid(owner, allocations, prepaid_bytes, deadline_tick)
    }

    /// G1 fairness mechanics only. Physical production callers must hold an
    /// actual PhysicalServiceBudget and use submit(); the isolated host tests
    /// exercise this lower-level queue policy without making GPU claims.
    fn enqueue_prepaid(
        self: &Arc<Self>,
        owner: ResourceOwner,
        allocations: &[AllocationRequest],
        prepaid_bytes: u64,
        deadline_tick: Option<u64>,
    ) -> Result<WaitingTicket> {
        if allocations.is_empty() || prepaid_bytes == 0 {
            return Err(Error::Limit("prepaid device task lacks an envelope".into()));
        }
        let total = allocations.iter().try_fold(0u64, |total, a| {
            if a.bytes == 0 {
                return Err(Error::Limit("zero prepaid device allocation".into()));
            }
            total
                .checked_add(a.bytes)
                .ok_or_else(|| Error::Limit("prepaid task bytes overflow".into()))
        })?;
        if total > prepaid_bytes {
            return Err(Error::Limit(
                "queued physical scratch exceeds held provider envelope".into(),
            ));
        }
        if ResourceArbiter::process_shared().is_fenced(&self.key)? {
            return Err(Error::State(
                "physical card fenced before device enqueue".into(),
            ));
        }
        let wake = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::State("physical GPU worker queue poisoned".into()))?;
            let id = state.next_id;
            let next = id
                .checked_add(1)
                .ok_or_else(|| Error::Limit("physical queue task id exhausted".into()))?;
            let epoch = state.queue.epoch();
            let task = DeviceTask {
                id,
                epoch,
                owner,
                allocations: allocations.to_vec(),
                deadline_tick,
            };
            state.queue.enqueue(task)?;
            state.next_id = next;
            // Wake earlier sleeping callers; selection still happens only in
            // the fair queue, under this short lock.
            let wakes = state.wakers.values().cloned().collect::<Vec<_>>();
            (id, epoch, wakes)
        };
        for waker in wake.2 {
            waker.wake();
        }
        Ok(WaitingTicket {
            queue: self.clone(),
            id: wake.0,
            epoch: wake.1,
            armed: true,
        })
    }

    fn poll_acquire(
        self: &Arc<Self>,
        id: u64,
        epoch: u64,
        cx: &Context<'_>,
    ) -> Poll<Result<ActiveSlot>> {
        // Query the single authoritative physical charge outside the worker
        // queue lock. No global lock is held across a future or device launch.
        match ResourceArbiter::process_shared().is_fenced(&self.key) {
            Ok(true) => {
                return Poll::Ready(Err(Error::State("physical GPU service fenced".into())));
            }
            Err(e) => return Poll::Ready(Err(e)),
            _ => {}
        }
        let mut to_wake = Vec::new();
        let outcome = {
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => {
                    return Poll::Ready(Err(Error::State("physical GPU queue poisoned".into())));
                }
            };
            if epoch != state.queue.epoch() {
                return Poll::Ready(Err(Error::State(
                    "stale physical queue ticket epoch".into(),
                )));
            }
            if state.failed.remove(&id) {
                return Poll::Ready(Err(Error::State(
                    "queued GPU task expired or stale epoch".into(),
                )));
            }
            state.wakers.insert(id, cx.waker().clone());
            if state.active.is_none() {
                // A single scheduling pass can discard multiple expired
                // tasks, but never runs model math under the queue mutex.
                loop {
                    let eligible = state.wakers.keys().copied().collect::<BTreeSet<_>>();
                    let choice = match state
                        .queue
                        .next_prepaid(self.now_tick(), |task| Ok(eligible.contains(&task.id)))
                    {
                        Ok(choice) => choice,
                        Err(err) => return Poll::Ready(Err(err)),
                    };
                    match choice {
                        PrepaidChoice::Dispatched(task) => {
                            state.active = Some(task.id);
                            state.selected += 1;
                            if let Some(waker) = state.wakers.get(&task.id) {
                                to_wake.push(waker.clone());
                            }
                            break;
                        }
                        PrepaidChoice::Expired(task) => {
                            state.expired += 1;
                            state.failed.insert(task.id);
                            if let Some(waker) = state.wakers.get(&task.id) {
                                to_wake.push(waker.clone());
                            }
                        }
                        PrepaidChoice::NoCapacity | PrepaidChoice::Empty => break,
                    }
                }
            }
            if state.active == Some(id) {
                state.wakers.remove(&id);
                Poll::Ready(Ok(ActiveSlot {
                    queue: self.clone(),
                    id,
                    epoch: state.queue.epoch(),
                    armed: true,
                    submitted: false,
                }))
            } else if state.failed.remove(&id) {
                state.wakers.remove(&id);
                Poll::Ready(Err(Error::State("queued GPU task expired".into())))
            } else {
                Poll::Pending
            }
        };
        for waker in to_wake {
            waker.wake();
        }
        outcome
    }

    fn cancel_waiting(&self, id: u64) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.queue.cancel_queued(id) {
            state.cancelled_queued += 1;
        } else if state.active == Some(id) {
            // Selected but never polled/accepted by its original owner: no
            // physical GPU work was launched, so this can be cancelled safely.
            state.active = None;
        }
        state.failed.remove(&id);
        state.wakers.remove(&id);
        let wakes = state.wakers.values().cloned().collect::<Vec<_>>();
        drop(state);
        for waker in wakes {
            waker.wake();
        }
    }

    fn retire(&self, id: u64, epoch: u64, completed: bool) -> Result<bool> {
        let (stale, wakes) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::State("physical GPU queue poisoned".into()))?;
            if state.active != Some(id) {
                return Err(Error::State(
                    "device worker slot not owned by retiring task".into(),
                ));
            }
            state.active = None;
            if completed {
                state.completed += 1;
            }
            let stale = state.queue.epoch() != epoch;
            let wakes = state.wakers.values().cloned().collect::<Vec<_>>();
            (stale, wakes)
        };
        for waker in wakes {
            waker.wake();
        }
        Ok(stale)
    }

    /// Global physical-card generation change. Cancels *every* queued model
    /// and session on the card, not merely the caller's own requests. Per-
    /// session cancellation must drop its tickets, not revoke this epoch.
    pub fn advance_epoch(&self, next: u64) -> Result<usize> {
        let (discarded, wakes) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::State("physical GPU queue poisoned".into()))?;
            let count = state.queue.advance_epoch(next)?;
            let active = state.active;
            let queued = state
                .wakers
                .keys()
                .copied()
                .filter(|id| Some(*id) != active)
                .collect::<Vec<_>>();
            state.failed.extend(queued);
            let wakes = state.wakers.values().cloned().collect::<Vec<_>>();
            (count, wakes)
        };
        for waker in wakes {
            waker.wake();
        }
        Ok(discarded)
    }

    pub fn stats(&self) -> Result<QueueStats> {
        let state = self
            .state
            .lock()
            .map_err(|_| Error::State("physical GPU queue poisoned".into()))?;
        Ok(QueueStats {
            queued: state.queue.queued(),
            capacity: self.capacity,
            active: state.active.is_some(),
            selected: state.selected,
            completed: state.completed,
            cancelled_queued: state.cancelled_queued,
            expired: state.expired,
            epoch: state.queue.epoch(),
        })
    }

    pub fn key(&self) -> &PhysicalDeviceKey {
        &self.key
    }
}

/// The globally shared scheduler is indexed by PHYSICAL card, not backend
/// ServiceId. Its global weak registry never owns a GPU client or source
/// model; the last provider naturally tears down an idle queue.
pub fn queue_for(key: PhysicalDeviceKey) -> Result<Arc<PhysicalQueue>> {
    static QUEUES: OnceLock<Mutex<BTreeMap<PhysicalDeviceKey, Weak<PhysicalQueue>>>> =
        OnceLock::new();
    let registry = QUEUES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut entries = registry
        .lock()
        .map_err(|_| Error::State("physical GPU worker registry poisoned".into()))?;
    if let Some(queue) = entries.get(&key).and_then(Weak::upgrade) {
        if queue.capacity != DEFAULT_QUEUE_CAPACITY {
            return Err(Error::State(
                "incompatible physical worker queue capacity".into(),
            ));
        }
        return Ok(queue);
    }
    let queue = Arc::new(PhysicalQueue::new(key.clone(), DEFAULT_QUEUE_CAPACITY)?);
    entries.insert(key, Arc::downgrade(&queue));
    Ok(queue)
}

#[must_use]
pub struct WaitingTicket {
    queue: Arc<PhysicalQueue>,
    id: u64,
    epoch: u64,
    armed: bool,
}
impl WaitingTicket {
    pub async fn wait(mut self) -> Result<ActiveSlot> {
        let slot = poll_fn(|cx| self.queue.poll_acquire(self.id, self.epoch, cx)).await?;
        self.armed = false;
        Ok(slot)
    }
}
impl Drop for WaitingTicket {
    fn drop(&mut self) {
        if self.armed {
            self.queue.cancel_waiting(self.id);
        }
    }
}

#[must_use]
pub struct ActiveSlot {
    queue: Arc<PhysicalQueue>,
    id: u64,
    epoch: u64,
    armed: bool,
    submitted: bool,
}
impl ActiveSlot {
    /// This must be called just before the first possible device command.
    /// A dropped acquired slot with no submission is ordinary cancellation,
    /// but a submitted slot is a terminal physical resource fence.
    pub fn begin_submission(&mut self) -> Result<()> {
        if self.submitted {
            return Err(Error::State("device worker slot already submitted".into()));
        }
        {
            let state = self
                .queue
                .state
                .lock()
                .map_err(|_| Error::State("physical GPU worker queue poisoned".into()))?;
            if state.queue.epoch() != self.epoch || state.active != Some(self.id) {
                return Err(Error::State(
                    "stale/cancelled GPU queue slot must not submit physical commands".into(),
                ));
            }
        }
        if ResourceArbiter::process_shared().is_fenced(&self.queue.key)? {
            return Err(Error::State(
                "physical GPU fenced before worker submission".into(),
            ));
        }
        self.submitted = true;
        Ok(())
    }

    /// Call only AFTER physical GPU readback/event retirement, or after a
    /// provider error whose own RAII guards have quarantined uncertain work.
    /// A changed session/queue epoch refuses publication even after Ready.
    pub fn finish(mut self) -> Result<()> {
        if !self.submitted {
            return Err(Error::State(
                "GPU worker cannot report Ready before submission".into(),
            ));
        }
        let stale = self.queue.retire(self.id, self.epoch, true)?;
        self.armed = false;
        if stale {
            Err(Error::State(
                "stale GPU queue epoch after physical Ready".into(),
            ))
        } else {
            Ok(())
        }
    }
}
impl Drop for ActiveSlot {
    fn drop(&mut self) {
        if self.armed {
            if self.submitted {
                // A physically submitted Future was cancelled: no reusable
                // memory, and no further model on this card until recovery.
                let _ = ResourceArbiter::process_shared().fence_unknown_device(&self.queue.key);
            }
            let _ = self.queue.retire(self.id, self.epoch, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{future::Future, pin::Pin, task::Waker};
    use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

    fn setup(id: &str) -> Arc<PhysicalQueue> {
        let key = PhysicalDeviceKey::new("test-pci", id).unwrap();
        ResourceArbiter::process_shared()
            .register_matching(key.clone(), 4096)
            .unwrap();
        Arc::new(PhysicalQueue::new(key, 4).unwrap())
    }
    fn ticket(queue: &Arc<PhysicalQueue>, id: &str) -> WaitingTicket {
        queue
            .enqueue_prepaid(
                ResourceOwner::new(id, "session").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 32,
                }],
                1024,
                None,
            )
            .unwrap()
    }
    fn poll<T: Future>(future: Pin<&mut T>) -> Poll<T::Output> {
        let waker = Waker::noop();
        let mut ctx = Context::from_waker(waker);
        future.poll(&mut ctx)
    }
    #[test]
    fn real_lease_prepaid_fairness_and_no_memory_double_charge() {
        let q = setup("queue-fair");
        let arbiter = ResourceArbiter::process_shared();
        let existing = arbiter
            .admit(
                q.key(),
                ResourceOwner::new("static", "owner").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: 256,
                }],
            )
            .unwrap();
        let first = ticket(&q, "a");
        let second = ticket(&q, "a");
        let third = ticket(&q, "b");
        let mut first = Box::pin(first.wait());
        let mut second = Box::pin(second.wait());
        let mut third = Box::pin(third.wait());
        let Poll::Ready(Ok(slot)) = poll(first.as_mut()) else {
            panic!("first not selected")
        };
        assert!(matches!(poll(second.as_mut()), Poll::Pending));
        assert!(matches!(poll(third.as_mut()), Poll::Pending));
        let mut slot = slot;
        slot.begin_submission().unwrap();
        slot.finish().unwrap();
        // RR chooses b after a's first dispatched request.
        assert!(matches!(poll(second.as_mut()), Poll::Pending));
        let Poll::Ready(Ok(mut slot)) = poll(third.as_mut()) else {
            panic!("other model starved")
        };
        slot.begin_submission().unwrap();
        slot.finish().unwrap();
        let Poll::Ready(Ok(mut slot)) = poll(second.as_mut()) else {
            panic!("a2 not resumed")
        };
        slot.begin_submission().unwrap();
        slot.finish().unwrap();
        assert_eq!(arbiter.snapshot(q.key()).unwrap().reserved, 256);
        assert_eq!(q.stats().unwrap().completed, 3);
        existing.release().unwrap();
    }
    #[test]
    fn cancelled_pending_request_never_fences_physical_gpu() {
        let q = setup("queue-cancel-pending");
        let mut first = Box::pin(ticket(&q, "a").wait());
        let Poll::Ready(Ok(mut slot)) = poll(first.as_mut()) else {
            panic!("first not ready")
        };
        let mut waiting = Box::pin(ticket(&q, "b").wait());
        assert!(matches!(poll(waiting.as_mut()), Poll::Pending));
        drop(waiting);
        assert_eq!(q.stats().unwrap().queued, 0);
        slot.begin_submission().unwrap();
        slot.finish().unwrap();
        assert!(
            !ResourceArbiter::process_shared()
                .snapshot(q.key())
                .unwrap()
                .fenced
        );
    }
    #[test]
    fn selected_unsubmitted_or_epoch_stale_slot_cancels_without_gpu_fence() {
        let q = setup("queue-unsubmitted");
        let mut pending = Box::pin(ticket(&q, "a").wait());
        let Poll::Ready(Ok(slot)) = poll(pending.as_mut()) else {
            panic!("slot not selected")
        };
        drop(slot);
        assert!(
            !ResourceArbiter::process_shared()
                .snapshot(q.key())
                .unwrap()
                .fenced
        );
        assert!(!q.stats().unwrap().active);
        let mut pending = Box::pin(ticket(&q, "a").wait());
        let Poll::Ready(Ok(mut slot)) = poll(pending.as_mut()) else {
            panic!("slot not selected")
        };
        q.advance_epoch(2).unwrap();
        assert!(slot.begin_submission().is_err());
        drop(slot);
        assert!(
            !ResourceArbiter::process_shared()
                .snapshot(q.key())
                .unwrap()
                .fenced
        );
    }

    #[test]
    fn cancelled_inflight_request_fences_card_and_refuses_other_models() {
        let q = setup("queue-cancel-active");
        let mut first = Box::pin(ticket(&q, "a").wait());
        let Poll::Ready(Ok(mut slot)) = poll(first.as_mut()) else {
            panic!("first not ready")
        };
        slot.begin_submission().unwrap();
        drop(slot);
        assert!(
            ResourceArbiter::process_shared()
                .snapshot(q.key())
                .unwrap()
                .fenced
        );
        assert!(
            q.enqueue_prepaid(
                ResourceOwner::new("b", "s").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 8
                }],
                256,
                None
            )
            .is_err()
        );
    }
    #[test]
    fn epoch_change_invalidates_old_ready_without_publishing() {
        let q = setup("queue-epoch");
        let mut first = Box::pin(ticket(&q, "a").wait());
        let Poll::Ready(Ok(mut slot)) = poll(first.as_mut()) else {
            panic!("first not ready")
        };
        let waiting = ticket(&q, "b");
        slot.begin_submission().unwrap();
        assert_eq!(q.advance_epoch(2).unwrap(), 1);
        // Simulated trusted physical Ready arrives after epoch revocation:
        // it releases the slot, but cannot publish the obsolete result.
        assert!(slot.finish().is_err());
        assert!(futures_lite::future::block_on(waiting.wait()).is_err());
        assert!(
            !ResourceArbiter::process_shared()
                .snapshot(q.key())
                .unwrap()
                .fenced
        );
    }
}
