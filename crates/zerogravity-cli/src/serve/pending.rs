//! One process-router serving FIFO and the host-storage owners it admits.
//!
//! The queue is intentionally generic: CPU and GPU workers consume the same
//! primitive instead of compiling a separate bounded channel per worker.
//! Count limits are optional; host admission and exact per-entry storage are
//! always required before a request is published to a worker.

use super::engine::ServiceError;
use AllocationClass::Scratch;
use std::alloc::Layout;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};

/// One process-wide observation ledger per original arbiter/key domain.
/// Serving updates and observations hold `lock` so the ledger and the arbiter
/// snapshot move together; model/session admissions only touch the arbiter.
struct ServingDomain {
    lock: Mutex<()>,
    bytes: AtomicU64,
}
fn domains() -> &'static Mutex<BTreeMap<PhysicalDeviceKey, Arc<ServingDomain>>> {
    static DOMAINS: OnceLock<Mutex<BTreeMap<PhysicalDeviceKey, Arc<ServingDomain>>>> =
        OnceLock::new();
    DOMAINS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Process-shared configured host domain used by model, session and serving
/// storage. Clones share one arbiter/key/domain ledger.
#[derive(Clone)]
pub(super) struct ServingHost {
    arbiter: ResourceArbiter,
    key: PhysicalDeviceKey,
    domain: Arc<ServingDomain>,
}
impl ServingHost {
    pub(super) fn new(configured_mib: usize) -> Result<Self, ServiceError> {
        if !(256..=1_048_576).contains(&configured_mib) {
            return Err(ServiceError::new(
                400,
                "invalid_host_budget",
                "host-mib must be 256..=1048576",
            ));
        }
        let bytes = (configured_mib as u64) * 1024 * 1024;
        let arbiter = ResourceArbiter::process_shared();
        let key = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")
            .map_err(ServiceError::inference)?;
        let domain = domains()
            .lock()
            .map_err(|_| ServiceError::new(500, "host_state", "serving domain lock poisoned"))?
            .entry(key.clone())
            .or_insert_with(|| {
                Arc::new(ServingDomain {
                    lock: Mutex::new(()),
                    bytes: AtomicU64::new(0),
                })
            })
            .clone();
        match arbiter.register_matching(key.clone(), bytes) {
            Ok(()) => {}
            Err(_) => {
                return Err(ServiceError::new(
                    409,
                    "host_budget_conflict",
                    "existing host admission pool uses a different configured limit",
                ));
            }
        }
        // One process-retained runtime thread-control root allowance in this
        // original arbiter/key domain, before any worker or HTTP connection
        // thread exists. A later router verifies the same domain and charge
        // instead of admitting another root.
        zerogravity::resource_arbiter::admit_thread_control_root(
            &arbiter,
            &key,
            ResourceOwner::new("serve-root", "thread-control").map_err(ServiceError::inference)?,
        )
        .map_err(ServiceError::inference)?;
        Ok(Self {
            arbiter,
            key,
            domain,
        })
    }
    /// Isolated domain for unit tests; never the production process ledger.
    #[cfg(test)]
    pub(super) fn isolated(configured_mib: usize) -> Result<Self, ServiceError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let bytes = (configured_mib as u64) * 1024 * 1024;
        let key = PhysicalDeviceKey::new(
            "host",
            &format!("isolated-{}", NEXT.fetch_add(1, Ordering::Relaxed)),
        )
        .map_err(ServiceError::inference)?;
        let arbiter = ResourceArbiter::new();
        arbiter
            .register(key.clone(), bytes)
            .map_err(ServiceError::inference)?;
        Ok(Self {
            arbiter,
            key,
            domain: Arc::new(ServingDomain {
                lock: Mutex::new(()),
                bytes: AtomicU64::new(0),
            }),
        })
    }
    /// Model/session/backend storage. These owners are part of the model
    /// component, not the serving queue.
    pub(super) fn admit(
        &self,
        alias: &str,
        session: &str,
        kind: AllocationClass,
        bytes: u64,
    ) -> Result<GlobalResourceLease, ServiceError> {
        let total = checked_sum(&[bytes, self.lease_envelope(alias, session)?])?;
        self.reserve_lease(alias, session, kind, total)
    }
    /// Serving ingress/queue/event/output/transport storage.
    pub(super) fn charge(
        &self,
        alias: &str,
        session: &str,
        kind: AllocationClass,
        bytes: u64,
    ) -> Result<ServingCharge, ServiceError> {
        if bytes == 0 {
            return Ok(ServingCharge {
                bytes: 0,
                domain: self.domain.clone(),
                lease: None,
            });
        }
        let total = checked_sum(&[bytes, self.lease_envelope(alias, session)?])?;
        let _state =
            self.domain.lock.lock().map_err(|_| {
                ServiceError::new(500, "host_state", "serving ledger lock poisoned")
            })?;
        let lease = self.reserve_lease(alias, session, kind, total)?;
        self.domain.bytes.fetch_add(total, Ordering::AcqRel);
        Ok(ServingCharge {
            bytes: total,
            domain: self.domain.clone(),
            lease: Some(lease),
        })
    }
    #[cfg(test)]
    pub(super) fn serving_bytes(&self) -> u64 {
        self.domain.bytes.load(Ordering::Acquire)
    }
    /// One coherent (limit, reserved, remaining, serving) observation. The
    /// domain lock is the seam: every serving charge/drop updates both sides
    /// under it, so `reserved.saturating_sub(serving)` is never a fabricated
    /// residual while an admission is in flight.
    pub(super) fn snapshot(&self) -> Option<(u64, u64, u64, u64)> {
        let _state = self.domain.lock.lock().ok()?;
        let snapshot = self.arbiter.snapshot(&self.key).ok()?;
        let serving = self.domain.bytes.load(Ordering::Acquire);
        Some((
            snapshot.limit,
            snapshot.reserved,
            snapshot.remaining,
            serving,
        ))
    }
    /// Reusable typed-admission seam for tokenizer/bootstrap owners. Clones
    /// the existing arbiter/key and constructs the owner only: no domain,
    /// registration or reservation is created here, and the caller performs
    /// the only admission. Available wherever the cache transfer type exists.
    #[cfg(feature = "cache")]
    /// Borrow the serving host's original arbiter/key domain. The core type
    /// keeps one public admission borrower without a second cache-only copy.
    pub(super) fn transfer_admission(
        &self,
        alias: &str,
        session: &str,
    ) -> Result<zerogravity::resource_arbiter::TransferAdmission, ServiceError> {
        Ok(zerogravity::resource_arbiter::TransferAdmission {
            arbiter: self.arbiter.clone(),
            resource: self.key.clone(),
            owner: ResourceOwner::new(alias, session).map_err(ServiceError::inference)?,
        })
    }
}

impl ServingHost {
    /// One arbiter lease's non-payload inventory: detached lease, reservation
    /// node, owner strings, allocation vector and BTree/Arc control slack.
    /// Every charge and model/session lease uses this same envelope.
    fn lease_envelope(&self, alias: &str, session: &str) -> Result<u64, ServiceError> {
        let overflow = || ServiceError::new(400, "size_overflow", "lease envelope overflow");
        let typed = zerogravity::resource_arbiter::lease_inventory_bytes(alias, session, 1)
            .map_err(|_| overflow())?;
        let device = u64::try_from(self.key.backend.len())
            .ok()
            .and_then(|backend| {
                u64::try_from(self.key.device.len())
                    .ok()
                    .and_then(|device| backend.checked_add(device))
            })
            .ok_or_else(overflow)?;
        let fixed = usize::checked_add(
            std::mem::size_of::<ServingCharge>(),
            std::mem::size_of::<GlobalResourceLease>(),
        )
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(overflow)?;
        fixed
            .checked_add(typed)
            .and_then(|total| total.checked_add(device))
            .ok_or_else(overflow)
    }
    fn reserve_lease(
        &self,
        alias: &str,
        session: &str,
        kind: AllocationClass,
        bytes: u64,
    ) -> Result<GlobalResourceLease, ServiceError> {
        self.arbiter
            .admit(
                &self.key,
                ResourceOwner::new(alias, session).map_err(ServiceError::inference)?,
                &[AllocationRequest { kind, bytes }],
            )
            .map_err(|error| ServiceError::new(503, "host_budget", error.to_string()))
    }
}

/// A charged serving allocation. The payload it covers must be dropped before
/// this owner. Drop takes the domain seam before releasing the arbiter lease;
/// no observer can see the ledger without the matching reservation.
pub(super) struct ServingCharge {
    bytes: u64,
    domain: Arc<ServingDomain>,
    lease: Option<GlobalResourceLease>,
}
impl Drop for ServingCharge {
    fn drop(&mut self) {
        // A poisoned seam must not leak the reservation; release both sides
        // under the poisoned guard without panicking in Drop.
        let _state = self
            .domain
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.domain.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        drop(self.lease.take());
    }
}

pub(super) fn slot_bytes<T>(capacity: usize) -> Result<u64, ServiceError> {
    let size = std::mem::size_of::<Option<T>>();
    if size == 0 {
        return Ok(0);
    }
    let count = u64::try_from(capacity).map_err(|_| queue_alloc_error())?;
    let bytes = count
        .checked_mul(size as u64)
        .ok_or_else(queue_alloc_error)?;
    Ok(bytes)
}

/// Exact byte allocation; no hidden spare capacity is published.
pub(super) fn allocate_bytes(len: usize) -> Result<Box<[u8]>, ServiceError> {
    if len == 0 {
        return Ok(Vec::new().into_boxed_slice());
    }
    let layout = Layout::array::<u8>(len).map_err(|_| queue_alloc_error())?;
    // SAFETY: non-zero layout for `len` bytes, initialized before exposure.
    let raw = unsafe { std::alloc::alloc(layout) };
    if raw.is_null() {
        return Err(queue_alloc_error());
    }
    unsafe { std::ptr::write_bytes(raw, 0, len) };
    Ok(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(raw, len)) })
}

/// Exact fixed-layout backing store. `Box<[Option<T>]>` owns exactly the
/// requested length; no hidden spare capacity is published or retained.
pub(super) fn allocate_slots<T>(capacity: usize) -> Result<Box<[Option<T>]>, ServiceError> {
    if capacity == 0 {
        return Ok(Vec::new().into_boxed_slice());
    }
    if std::mem::size_of::<Option<T>>() == 0 {
        return Ok(std::iter::repeat_with(|| None)
            .take(capacity)
            .collect::<Vec<_>>()
            .into_boxed_slice());
    }
    let layout = Layout::array::<Option<T>>(capacity).map_err(|_| queue_alloc_error())?;
    // SAFETY: layout is non-zero and valid for `capacity` initialized slots.
    // Every slot is written as `None` before the boxed slice is exposed.
    let raw = unsafe { std::alloc::alloc(layout) } as *mut Option<T>;
    if raw.is_null() {
        return Err(queue_alloc_error());
    }
    for index in 0..capacity {
        unsafe { raw.add(index).write(None) };
    }
    Ok(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(raw, capacity)) })
}

pub(super) fn queue_alloc_error() -> ServiceError {
    ServiceError::new(503, "queue_allocation", "serving storage allocation failed")
}

pub(super) fn checked_sum(values: &[u64]) -> Result<u64, ServiceError> {
    values
        .iter()
        .try_fold(0u64, |total, value| total.checked_add(*value))
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "serving size overflow"))
}

pub(super) fn heap_strings(values: &[&String]) -> Result<u64, ServiceError> {
    let mut total = 0u64;
    for value in values {
        total =
            total
                .checked_add(u64::try_from(value.capacity()).map_err(|_| {
                    ServiceError::new(400, "size_overflow", "serving size overflow")
                })?)
                .ok_or_else(|| ServiceError::new(400, "size_overflow", "serving size overflow"))?;
    }
    Ok(total)
}

pub(super) fn vec_bytes<T>(count: usize) -> Result<u64, ServiceError> {
    let count = u64::try_from(count)
        .map_err(|_| ServiceError::new(400, "size_overflow", "serving size overflow"))?;
    let size = u64::try_from(std::mem::size_of::<T>())
        .map_err(|_| ServiceError::new(400, "size_overflow", "serving size overflow"))?;
    count
        .checked_mul(size)
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "serving size overflow"))
}

struct ControlState {
    pending: usize,
    in_progress: usize,
    closed: bool,
    capacity: usize,
    backing_bytes: u64,
    consumers: usize,
}
struct Control {
    state: Mutex<ControlState>,
    ready: Arc<Condvar>,
    max_pending: Option<usize>,
    cancelled_hint: AtomicBool,
    _charge: ServingCharge,
}

/// Queue wake/cancel handle. It weakly notifies the queue wake state and
/// cancellation hint; it never owns a request or the queue itself.
#[derive(Clone)]
pub(super) struct CancelToken {
    flag: Arc<AtomicBool>,
    control: Option<Weak<Control>>,
    _charge: Option<Arc<ServingCharge>>,
}
impl std::fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}
impl CancelToken {
    fn new(control: &Arc<Control>, charge: Option<Arc<ServingCharge>>) -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            control: Some(Arc::downgrade(control)),
            _charge: charge,
        }
    }
    #[cfg(test)]
    pub(super) fn from_flag(flag: Arc<AtomicBool>) -> Self {
        Self {
            flag,
            control: None,
            _charge: None,
        }
    }
    #[cfg(test)]
    pub(super) fn standalone() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            control: None,
            _charge: None,
        }
    }
    pub(super) fn flag(&self) -> &Arc<AtomicBool> {
        &self.flag
    }
    pub(super) fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
    pub(super) fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        if let Some(control) = self.control.as_ref().and_then(Weak::upgrade) {
            // A wake hint is a flag, not a counter; saturation/overflow
            // must never cause a cancelled request to become invisible.
            control.cancelled_hint.store(true, Ordering::Release);
            control.ready.notify_all();
        }
    }
}

/// One array and the exact charge covering its allocation. Keeping the charge
/// with the array lets the sweep move an array between the live ring and the
/// retirement spare without ever leaving it uncharged.
struct Slots<T> {
    slots: Box<[Option<T>]>,
    charge: Option<ServingCharge>,
    // Backing identity. A retired array may return as the ring's spare only
    // for the same generation and capacity; a concurrent growth creates a new
    // generation, so a smaller old array can never be adopted.
    generation: u64,
}
fn next_generation() -> u64 {
    static GENERATION: AtomicU64 = AtomicU64::new(1);
    GENERATION.fetch_add(1, Ordering::Relaxed)
}
impl<T> Slots<T> {
    fn allocated(
        capacity: usize,
        host: &ServingHost,
        generation: u64,
    ) -> Result<Self, ServiceError> {
        let bytes = slot_bytes::<T>(capacity)?;
        let charge = if bytes == 0 {
            None
        } else {
            Some(host.charge("serving-pending", "queue-backing", Scratch, bytes)?)
        };
        let slots = allocate_slots::<T>(capacity)?;
        Ok(Self {
            slots,
            charge,
            generation,
        })
    }
    fn capacity(&self) -> usize {
        self.slots.len()
    }
}

/// FIFO backing plus an equal-capacity, pre-admitted spare array. The spare is
/// empty at rest; cancellation compaction swaps live entries into it and uses
/// the old array as the retirement owner, so no allocation occurs at cancel
/// time. Both arrays carry independent exact charges, so growth and retirement
/// can never release one array's charge while the other array is still live.
struct Ring<T> {
    live: Slots<T>,
    spare: Option<Slots<T>>,
    head: usize,
    len: usize,
    generation: u64,
}
impl<T> Ring<T> {
    fn capacity(&self) -> usize {
        self.live.capacity()
    }
    fn index(&self, offset: usize) -> usize {
        (self.head + offset) % self.live.capacity()
    }
    fn push(&mut self, item: T) {
        let index = self.index(self.len);
        self.live.slots[index] = Some(item);
        self.len += 1;
    }
    fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let index = self.head;
        let value = self.live.slots[index].take();
        self.head = (self.head + 1) % self.live.capacity();
        self.len -= 1;
        value
    }
}
struct Payload<T> {
    ring: Option<Ring<T>>,
}
struct PushOutcome<T> {
    old: Option<Ring<T>>,
    capacity: usize,
    bytes: u64,
}
type PushResult<T> = Result<PushOutcome<T>, (ServiceError, T)>;
impl<T> Payload<T> {
    fn push(&mut self, item: T, host: &ServingHost) -> PushResult<T> {
        let current = self.ring.as_ref().map_or(0, Ring::capacity);
        let len = self.ring.as_ref().map_or(0, |ring| ring.len);
        if self
            .ring
            .as_ref()
            .is_some_and(|ring| ring.len < ring.capacity())
        {
            let ring = self.ring.as_mut().expect("checked ring");
            ring.push(item);
            return Ok(PushOutcome {
                old: None,
                capacity: ring.capacity(),
                bytes: 0,
            });
        }
        let next = match current.checked_mul(2) {
            Some(doubled) => current.max(16).max(doubled).max(len.saturating_add(1)),
            None => return Err((queue_alloc_error(), item)),
        };
        // Allocate and charge both the live array and its compaction spare
        // before the item is published. The old ring keeps its own charges
        // until after publication, so the old+new growth peak is real.
        // One backing generation covers the new live array and its spare so
        // the pair can be swapped and returned without mixing generations.
        let generation = next_generation();
        let live = match Slots::allocated(next, host, generation) {
            Ok(slots) => slots,
            Err(error) => return Err((error, item)),
        };
        let spare = match Slots::allocated(next, host, generation) {
            Ok(slots) => slots,
            Err(error) => return Err((error, item)),
        };
        let bytes = match checked_sum(&[
            live.charge.as_ref().map_or(0, |charge| charge.bytes),
            spare.charge.as_ref().map_or(0, |charge| charge.bytes),
        ]) {
            Ok(bytes) => bytes,
            Err(error) => return Err((error, item)),
        };
        let mut replacement = Ring {
            live,
            spare: Some(spare),
            head: 0,
            len: 0,
            generation,
        };
        let old = self.ring.take();
        if let Some(mut old) = old {
            while let Some(value) = old.pop_front() {
                replacement.push(value);
            }
            replacement.push(item);
            self.ring = Some(replacement);
            Ok(PushOutcome {
                old: Some(old),
                capacity: next,
                bytes,
            })
        } else {
            replacement.push(item);
            self.ring = Some(replacement);
            Ok(PushOutcome {
                old: None,
                capacity: next,
                bytes,
            })
        }
    }
}

/// Ticket reserves optional count capacity and admission-in-progress. It is
/// committed only after storage admission, allocation and queue publication.
pub(super) struct AdmissionTicket {
    control: Arc<Control>,
    committed: bool,
}
impl AdmissionTicket {
    fn commit(&mut self) {
        self.committed = true;
    }
}
impl Drop for AdmissionTicket {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Ok(mut state) = self.control.state.lock() {
            state.in_progress = state.in_progress.saturating_sub(1);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct QueueSnapshot {
    pub limit: Option<usize>,
    pub pending: usize,
    pub in_progress: usize,
    pub capacity: usize,
    pub backing_bytes: u64,
    pub consumers: usize,
}

pub(super) enum Pop<T> {
    Item(T),
    Empty,
    Closed,
}

pub(super) struct PendingQueue<T> {
    // Payload is declared before control so the last handle destroys the ring
    // and its backing charges before releasing the queue-control/header charge.
    // Rust drops fields in declaration order; no Drop impl is required.
    payload: Arc<Mutex<Payload<T>>>,
    control: Arc<Control>,
    host: ServingHost,
}
impl<T> Clone for PendingQueue<T> {
    fn clone(&self) -> Self {
        Self {
            payload: Arc::clone(&self.payload),
            control: Arc::clone(&self.control),
            host: self.host.clone(),
        }
    }
}

/// Worker-side consumer handle. Dropping the last consumer closes admission,
/// so a panicked or exited worker cannot leave accepted orphan jobs.
pub(super) struct WorkerQueue<T> {
    queue: PendingQueue<T>,
    control: Arc<Control>,
}
impl<T> std::ops::Deref for WorkerQueue<T> {
    type Target = PendingQueue<T>;
    fn deref(&self) -> &Self::Target {
        &self.queue
    }
}
impl<T> Drop for WorkerQueue<T> {
    fn drop(&mut self) {
        let Ok(mut state) = self.control.state.lock() else {
            return;
        };
        state.consumers = state.consumers.saturating_sub(1);
        if state.consumers != 0 {
            return;
        }
        state.closed = true;
        // Last consumer: close admission and drain accepted queued jobs so a
        // panicked/exited worker cannot leave them orphaned. The payload ring
        // is released after both queue locks, not under them.
        let mut dropped = None;
        if let Ok(mut payload) = self.queue.payload.lock() {
            dropped = payload.ring.take();
            state.pending = 0;
            state.capacity = 0;
            state.backing_bytes = 0;
        }
        self.control.ready.notify_all();
        drop(state);
        drop(dropped);
    }
}

impl<T> PendingQueue<T> {
    pub(super) fn new(host: ServingHost, max_pending: Option<usize>) -> Result<Self, ServiceError> {
        let control_bytes = std::mem::size_of::<Control>()
            .checked_add(std::mem::size_of::<ControlState>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Payload<T>>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<Mutex<Payload<T>>>()))
            .and_then(|n| {
                std::mem::size_of::<Condvar>()
                    .checked_mul(2)
                    .and_then(|extra| n.checked_add(extra))
            })
            .and_then(|n| {
                std::mem::size_of::<usize>()
                    .checked_mul(6)
                    .and_then(|extra| n.checked_add(extra))
            })
            .ok_or_else(|| ServiceError::new(400, "size_overflow", "queue control overflow"))?;
        let charge = host.charge(
            "serving-queue",
            "queue-control",
            Scratch,
            u64::try_from(control_bytes)
                .map_err(|_| ServiceError::new(400, "size_overflow", "queue control overflow"))?,
        )?;
        Ok(Self {
            control: Arc::new(Control {
                state: Mutex::new(ControlState {
                    pending: 0,
                    in_progress: 0,
                    closed: false,
                    capacity: 0,
                    backing_bytes: 0,
                    consumers: 0,
                }),
                ready: Arc::new(Condvar::new()),
                max_pending,
                cancelled_hint: AtomicBool::new(false),
                _charge: charge,
            }),
            payload: Arc::new(Mutex::new(Payload { ring: None })),
            host,
        })
    }
    pub(super) fn worker(&self) -> Result<WorkerQueue<T>, ServiceError> {
        let mut state = self.control.state.lock().expect("pending queue control");
        if state.closed {
            return Err(ServiceError::new(
                503,
                "worker_unavailable",
                "pending request queue is closed",
            ));
        }
        state.consumers += 1;
        drop(state);
        Ok(WorkerQueue {
            queue: self.clone(),
            control: Arc::clone(&self.control),
        })
    }
    pub(super) fn cancel_token(&self, charge: Option<Arc<ServingCharge>>) -> CancelToken {
        CancelToken::new(&self.control, charge)
    }
    pub(super) fn reserve(&self) -> Result<AdmissionTicket, ServiceError> {
        let mut state = self.control.state.lock().expect("pending queue control");
        if state.closed || state.consumers == 0 {
            return Err(ServiceError::new(
                503,
                "worker_unavailable",
                "no serving worker is consuming the pending queue",
            ));
        }
        if let Some(limit) = self.control.max_pending {
            let total = state
                .pending
                .checked_add(state.in_progress)
                .ok_or_else(|| {
                    ServiceError::new(429, "queue_full", "pending request count limit reached")
                })?;
            if total >= limit {
                return Err(ServiceError::new(
                    429,
                    "queue_full",
                    "pending request count limit reached",
                ));
            }
        }
        state.in_progress += 1;
        Ok(AdmissionTicket {
            control: Arc::clone(&self.control),
            committed: false,
        })
    }
    pub(super) fn enqueue(
        &self,
        mut ticket: AdmissionTicket,
        item: T,
    ) -> Result<(), (ServiceError, T)> {
        let mut state = self.control.state.lock().expect("pending queue control");
        if state.closed {
            drop(state);
            return Err((
                ServiceError::new(503, "worker_unavailable", "pending request queue is closed"),
                item,
            ));
        }
        let mut payload = self.payload.lock().expect("pending queue payload");
        let PushOutcome {
            old,
            capacity,
            bytes: new_bytes,
        } = match payload.push(item, &self.host) {
            Ok(value) => value,
            Err((error, item)) => {
                drop(payload);
                drop(state);
                return Err((error, item));
            }
        };
        state.pending += 1;
        state.in_progress = state.in_progress.saturating_sub(1);
        if new_bytes != 0 {
            state.capacity = capacity;
            // The old ring is released after publication; only the new
            // backing remains charged. The peak covered both reservations.
            state.backing_bytes = new_bytes;
        } else if capacity > state.capacity {
            state.capacity = capacity;
        }
        ticket.commit();
        drop(payload);
        drop(state);
        self.control.ready.notify_one();
        drop(old);
        Ok(())
    }

    /// Remove all currently cancelled entries from anywhere in the FIFO. The
    /// predicate is evaluated once per queued entry in stable FIFO order. Live
    /// entries move into the pre-admitted same-capacity spare; the old array
    /// becomes the retirement owner, is emptied and destroyed after both queue
    /// locks are released, then returns as the empty spare. No allocation
    /// happens at cancellation time. If the spare is briefly busy, the hint is
    /// restored so a later worker round retries rather than losing the sweep.
    pub(super) fn sweep<F: Fn(&T) -> bool>(&self, cancelled: F) -> Result<usize, ServiceError> {
        if !self.control.cancelled_hint.swap(false, Ordering::AcqRel) {
            return Ok(0);
        }
        let mut state = self.control.state.lock().expect("pending queue control");
        if state.pending == 0 {
            // Cancellation may precede publication of its reserved ticket.
            // Keep the hint until a nonempty sweep observes that item.
            self.restore_cancel_hint();
            return Ok(0);
        }
        let mut payload = self.payload.lock().expect("pending queue payload");
        let Some(ring) = payload.ring.as_mut() else {
            self.restore_cancel_hint();
            return Ok(0);
        };
        let Some(mut spare) = ring.spare.take() else {
            self.restore_cancel_hint();
            return Ok(0);
        };
        // Compact every surviving entry in FIFO order. The swap happens even
        // when nothing was removed: the review's active-cancel-hint case (a
        // cancelled job that is not pending) must keep every accepted entry in
        // the live array instead of leaving it in the retirement spare.
        let mut live = 0usize;
        for offset in 0..ring.len {
            let index = ring.index(offset);
            let Some(item) = ring.live.slots[index].as_ref() else {
                continue;
            };
            if !cancelled(item) {
                let item = ring.live.slots[index].take().expect("checked live slot");
                spare.slots[live] = Some(item);
                live += 1;
            }
        }
        let removed = ring.len - live;
        // Swap the compacted live array in. The old live array carries its own
        // exact charge with it, so retirement cannot release the charge before
        // the array is empty even if another enqueue grows the ring meanwhile.
        let generation = ring.generation;
        let mut retired = std::mem::replace(&mut ring.live, spare);
        ring.head = 0;
        ring.len = live;
        state.pending = state.pending.saturating_sub(removed);
        drop(payload);
        drop(state);
        for slot in retired.slots.iter_mut() {
            slot.take();
        }
        let mut payload = self.payload.lock().expect("pending queue payload");
        let mut orphan = None;
        if let Some(ring) = payload.ring.as_mut() {
            // Only the same backing generation/capacity may become the ring's
            // spare. A smaller old array from before a concurrent growth is
            // dropped outside both queue locks rather than adopted.
            if ring.spare.is_none()
                && retired.generation == generation
                && retired.capacity() == ring.live.capacity()
            {
                ring.spare = Some(retired);
            } else {
                orphan = Some(retired);
            }
        } else {
            orphan = Some(retired);
        }
        drop(payload);
        drop(orphan);
        self.control.ready.notify_all();
        Ok(removed)
    }

    fn restore_cancel_hint(&self) {
        self.control.cancelled_hint.store(true, Ordering::Release);
    }

    pub(super) fn try_pop<F: Fn(&T) -> bool>(&self, cancelled: F) -> Pop<T> {
        let _ = self.sweep(&cancelled);
        let mut state = self.control.state.lock().expect("pending queue control");
        if state.pending == 0 {
            return if state.closed {
                Pop::Closed
            } else {
                Pop::Empty
            };
        }
        let mut payload = self.payload.lock().expect("pending queue payload");
        let item = payload.ring.as_mut().and_then(Ring::pop_front);
        if item.is_some() {
            state.pending = state.pending.saturating_sub(1);
        }
        let closed = state.closed && state.pending == 0;
        drop(payload);
        drop(state);
        match item {
            Some(item) => Pop::Item(item),
            None if closed => Pop::Closed,
            None => Pop::Empty,
        }
    }
    pub(super) fn pop_wait<F: Fn(&T) -> bool>(&self, wait: Duration, cancelled: F) -> Pop<T> {
        let deadline = Instant::now() + wait;
        loop {
            match self.try_pop(&cancelled) {
                Pop::Empty => {}
                other => return other,
            }
            let state = self.control.state.lock().expect("pending queue control");
            if state.closed && state.pending == 0 {
                return Pop::Closed;
            }
            if state.pending > 0 {
                drop(state);
                continue;
            }
            let now = Instant::now();
            if now >= deadline {
                return Pop::Empty;
            }
            let _ = self
                .control
                .ready
                .wait_timeout(state, deadline - now)
                .expect("pending queue wait");
        }
    }
    pub(super) fn snapshot(&self) -> QueueSnapshot {
        let state = self.control.state.lock().expect("pending queue control");
        QueueSnapshot {
            limit: self.control.max_pending,
            pending: state.pending,
            in_progress: state.in_progress,
            capacity: state.capacity,
            backing_bytes: state.backing_bytes,
            consumers: state.consumers,
        }
    }
    pub(super) fn close(&self) {
        let mut state = self.control.state.lock().expect("pending queue control");
        state.closed = true;
        self.control.ready.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(limit_mib: usize) -> ServingHost {
        ServingHost::isolated(limit_mib).unwrap()
    }
    /// Small isolated ledger for pressure/refusal cases; production `new`
    /// enforces the 256 MiB configuration floor, which is too large here.
    fn tiny_host(bytes: u64) -> ServingHost {
        let arbiter = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("host", "tiny-test").unwrap();
        arbiter.register(key.clone(), bytes).unwrap();
        ServingHost {
            arbiter,
            key,
            domain: Arc::new(ServingDomain {
                lock: Mutex::new(()),
                bytes: AtomicU64::new(0),
            }),
        }
    }
    fn item(cancelled: bool) -> (CancelToken, Vec<u8>) {
        let token = CancelToken::standalone();
        if cancelled {
            token.cancel();
        }
        (token, vec![0; 8])
    }

    #[test]
    fn cancelled_reserved_ticket_survives_an_empty_sweep_until_publication() {
        let admission = host(512);
        let queue = PendingQueue::<(CancelToken, usize)>::new(admission.clone(), None).unwrap();
        let worker = queue.worker().unwrap();
        let ticket = queue.reserve().unwrap();
        let cancel = queue.cancel_token(None);
        cancel.cancel();
        // The job is admitted, but it is not in the queue yet.
        assert_eq!(queue.sweep(|entry| entry.0.is_cancelled()).unwrap(), 0);
        queue.enqueue(ticket, (cancel, 7)).unwrap();
        assert_eq!(queue.snapshot().pending, 1);
        assert_eq!(queue.sweep(|entry| entry.0.is_cancelled()).unwrap(), 1);
        assert_eq!(queue.snapshot().pending, 0);
        drop((worker, queue));
        assert_eq!(admission.serving_bytes(), 0);
    }

    #[test]
    fn default_queue_is_not_count_capped_and_grows_exactly() {
        let admission = host(512);
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(admission.clone(), None).unwrap();
        let worker = queue.worker().unwrap();
        for index in 0..300 {
            let (token, payload) = item(false);
            let ticket = queue.reserve().unwrap();
            queue.enqueue(ticket, (token, payload)).unwrap();
            if index == 0 {
                assert!(queue.snapshot().capacity >= 1);
            }
        }
        let snapshot = queue.snapshot();
        assert_eq!(snapshot.pending, 300);
        assert!(snapshot.limit.is_none());
        assert!(snapshot.capacity >= 300);
        assert!(snapshot.backing_bytes > 0);
        for _ in 0..300 {
            assert!(matches!(queue.try_pop(|_| false), Pop::Item(_)));
        }
        assert!(matches!(queue.try_pop(|_| false), Pop::Empty));
        drop(queue);
        drop(worker);
        assert_eq!(admission.serving_bytes(), 0);
        let snapshot = admission.snapshot().unwrap();
        assert_eq!(
            (snapshot.0, snapshot.1, snapshot.3),
            (512 * 1024 * 1024, 0, 0)
        );
    }

    #[test]
    fn host_budget_bounds_and_allocation_failure_are_typed() {
        // Explicit configured HOST budget: 256 MiB ..= 1 TiB, never implicit.
        assert_eq!(ServingHost::new(255).err().unwrap().status, 400);
        assert_eq!(ServingHost::new(1_048_577).err().unwrap().status, 400);
        let allocation = queue_alloc_error();
        assert_eq!(
            (allocation.status, allocation.code),
            (503, "queue_allocation")
        );
    }

    #[test]
    fn two_queues_share_one_host_ledger_and_release_independently() {
        let admission = host(512);
        let first = PendingQueue::<(CancelToken, Vec<u8>)>::new(admission.clone(), None).unwrap();
        let first_worker = first.worker().unwrap();
        let first_charge = admission.serving_bytes();
        let second = PendingQueue::<(CancelToken, Vec<u8>)>::new(admission.clone(), None).unwrap();
        let second_worker = second.worker().unwrap();
        let second_charge = admission.serving_bytes();
        assert!(second_charge > first_charge);
        let snapshot = admission.snapshot().unwrap();
        assert_eq!(snapshot.1, snapshot.3);
        drop(first);
        drop(first_worker);
        let after_first = admission.serving_bytes();
        assert!(after_first > 0 && after_first < second_charge);
        drop(second);
        drop(second_worker);
        assert_eq!(admission.serving_bytes(), 0);
    }

    fn ring_charge(ring: &Ring<Vec<u8>>) -> u64 {
        ring.live.charge.as_ref().map_or(0, |charge| charge.bytes)
            + ring
                .spare
                .as_ref()
                .and_then(|spare| spare.charge.as_ref())
                .map_or(0, |charge| charge.bytes)
    }

    #[test]
    fn growth_keeps_old_and_new_backing_charged_until_publication() {
        let admission = host(512);
        let mut payload = Payload::<Vec<u8>> { ring: None };
        for index in 0..16 {
            payload.push(vec![index as u8; 8], &admission).unwrap();
        }
        let before = admission.serving_bytes();
        let outcome = payload.push(vec![0; 8], &admission).unwrap();
        let old = outcome.old.expect("seventeenth entry grows backing");
        let old_charge = ring_charge(&old);
        let new_charge = ring_charge(payload.ring.as_ref().unwrap());
        assert_eq!(before, old_charge);
        assert_eq!(admission.serving_bytes(), old_charge + new_charge);
        drop(old);
        assert_eq!(admission.serving_bytes(), new_charge);
        drop(payload);
        assert_eq!(admission.serving_bytes(), 0);
    }

    #[test]
    fn concurrent_tickets_admit_at_most_the_optional_count() {
        let queue =
            Arc::new(PendingQueue::<(CancelToken, Vec<u8>)>::new(host(512), Some(1)).unwrap());
        let _worker = queue.worker().unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let queue = Arc::clone(&queue);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                match queue.reserve() {
                    Ok(ticket) => {
                        let (token, payload) = item(false);
                        queue.enqueue(ticket, (token, payload)).unwrap();
                        true
                    }
                    Err(error) => {
                        assert_eq!(error.status, 429);
                        false
                    }
                }
            }));
        }
        let admitted = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .filter(|admitted| *admitted)
            .count();
        assert_eq!(admitted, 1);
        assert_eq!(queue.snapshot().pending, 1);
        assert_eq!(queue.snapshot().in_progress, 0);
    }

    #[test]
    fn optional_count_cap_counts_pending_and_admission_in_progress() {
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(host(512), Some(1)).unwrap();
        let _worker = queue.worker().unwrap();
        let ticket = queue.reserve().unwrap();
        assert!(queue.reserve().is_err());
        let (token, payload) = item(false);
        queue.enqueue(ticket, (token, payload)).unwrap();
        assert!(queue.reserve().is_err());
        assert!(matches!(queue.try_pop(|_| false), Pop::Item(_)));
        assert!(queue.reserve().is_ok());
    }

    #[test]
    fn cancelled_entries_are_removed_without_running() {
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(host(512), None).unwrap();
        let _worker = queue.worker().unwrap();
        let cancelled = queue.cancel_token(None);
        cancelled.cancel();
        let payload = vec![0; 8];
        let ticket = queue.reserve().unwrap();
        queue.enqueue(ticket, (cancelled.clone(), payload)).unwrap();
        let (live, payload) = item(false);
        let ticket = queue.reserve().unwrap();
        queue.enqueue(ticket, (live, payload)).unwrap();
        assert_eq!(queue.snapshot().pending, 2);
        assert_eq!(queue.sweep(|(token, _)| token.is_cancelled()).unwrap(), 1);
        assert_eq!(queue.snapshot().pending, 1);
        assert!(matches!(
            queue.try_pop(|(token, _)| token.is_cancelled()),
            Pop::Item(_)
        ));
        assert_eq!(queue.snapshot().pending, 0);
    }

    #[test]
    fn sweep_reclaims_head_middle_tail_without_reordering_live_fifo() {
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(host(512), None).unwrap();
        let _worker = queue.worker().unwrap();
        let mut tokens = Vec::new();
        for index in 0..6u8 {
            let token = queue.cancel_token(None);
            let ticket = queue.reserve().unwrap();
            queue
                .enqueue(ticket, (token.clone(), vec![index; 8]))
                .unwrap();
            tokens.push(token);
        }
        tokens[0].cancel();
        tokens[2].cancel();
        tokens[5].cancel();
        assert_eq!(queue.snapshot().pending, 6);
        assert_eq!(queue.sweep(|(token, _)| token.is_cancelled()).unwrap(), 3);
        assert_eq!(queue.snapshot().pending, 3);
        let mut order = Vec::new();
        while let Pop::Item((_, payload)) = queue.try_pop(|(token, _)| token.is_cancelled()) {
            order.push(payload[0]);
        }
        assert_eq!(order, vec![1, 3, 4]);
    }

    #[test]
    fn active_cancel_hint_keeps_every_uncancelled_pending_entry() {
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(host(512), None).unwrap();
        let _worker = queue.worker().unwrap();
        let mut tokens = Vec::new();
        for index in 0..5u8 {
            let token = queue.cancel_token(None);
            let ticket = queue.reserve().unwrap();
            queue
                .enqueue(ticket, (token.clone(), vec![index; 8]))
                .unwrap();
            tokens.push(token);
        }
        // Cancelled job that is running, not pending: only the hint is raised.
        let active = queue.cancel_token(None);
        active.cancel();
        assert_eq!(queue.snapshot().pending, 5);
        assert_eq!(queue.sweep(|(token, _)| token.is_cancelled()).unwrap(), 0);
        assert_eq!(queue.snapshot().pending, 5);
        let mut order = Vec::new();
        while let Pop::Item((_, payload)) = queue.try_pop(|(token, _)| token.is_cancelled()) {
            order.push(payload[0]);
        }
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn sweep_after_advanced_head_keeps_live_fifo() {
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(host(512), None).unwrap();
        let _worker = queue.worker().unwrap();
        let mut tokens = Vec::new();
        for index in 0..8u8 {
            let token = queue.cancel_token(None);
            let ticket = queue.reserve().unwrap();
            queue
                .enqueue(ticket, (token.clone(), vec![index; 8]))
                .unwrap();
            tokens.push(token);
        }
        for _ in 0..3 {
            assert!(matches!(queue.try_pop(|_| false), Pop::Item(_)));
        }
        for index in 8..12u8 {
            let token = queue.cancel_token(None);
            let ticket = queue.reserve().unwrap();
            queue
                .enqueue(ticket, (token.clone(), vec![index; 8]))
                .unwrap();
            tokens.push(token);
        }
        tokens[4].cancel();
        tokens[10].cancel();
        assert_eq!(queue.sweep(|(token, _)| token.is_cancelled()).unwrap(), 2);
        let mut order = Vec::new();
        while let Pop::Item((_, payload)) = queue.try_pop(|(token, _)| token.is_cancelled()) {
            order.push(payload[0]);
        }
        assert_eq!(order, vec![3, 5, 6, 7, 8, 9, 11]);
    }

    #[test]
    fn last_consumer_closes_and_drains_accepted_jobs() {
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(host(512), None).unwrap();
        let worker = queue.worker().unwrap();
        let (token, payload) = item(false);
        let ticket = queue.reserve().unwrap();
        queue.enqueue(ticket, (token, payload)).unwrap();
        assert_eq!(queue.snapshot().pending, 1);
        drop(worker);
        assert_eq!(queue.snapshot().pending, 0);
        assert_eq!(queue.snapshot().capacity, 0);
        assert!(queue.reserve().is_err());
        assert!(matches!(queue.try_pop(|_| false), Pop::Closed));
    }

    #[test]
    fn close_drains_remaining_entries_then_reports_closed() {
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(host(512), None).unwrap();
        let worker = queue.worker().unwrap();
        let (token, payload) = item(false);
        let ticket = queue.reserve().unwrap();
        queue.enqueue(ticket, (token, payload)).unwrap();
        queue.close();
        assert!(matches!(queue.try_pop(|_| false), Pop::Item(_)));
        assert!(matches!(queue.try_pop(|_| false), Pop::Closed));
        drop(worker);
        assert!(queue.reserve().is_err());
    }

    #[test]
    fn last_owner_drops_payload_before_control_charge() {
        struct Probe {
            host: ServingHost,
            observed: Arc<AtomicU64>,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                self.observed
                    .store(self.host.serving_bytes(), Ordering::Release);
            }
        }
        let admission = host(512);
        let queue = PendingQueue::<Probe>::new(admission.clone(), None).unwrap();
        // Admitted consumer without a worker handle: the last PendingQueue
        // destruction is the owner that retires the ring.
        queue.control.state.lock().unwrap().consumers = 1;
        let observed = Arc::new(AtomicU64::new(u64::MAX));
        let ticket = queue.reserve().unwrap();
        assert!(
            queue
                .enqueue(
                    ticket,
                    Probe {
                        host: admission.clone(),
                        observed: Arc::clone(&observed),
                    },
                )
                .is_ok()
        );
        let charged = admission.serving_bytes();
        assert!(charged > 0);
        drop(queue);
        // Ring and control charges were both still present while the payload
        // item dropped; only then did the control charge retire.
        assert_eq!(observed.load(Ordering::Acquire), charged);
        assert_eq!(admission.serving_bytes(), 0);
    }

    #[test]
    fn payload_drop_reenters_the_host_ledger_outside_queue_locks() {
        struct Reentrant {
            host: ServingHost,
            reentered: Arc<AtomicBool>,
        }
        impl Drop for Reentrant {
            fn drop(&mut self) {
                // Takes the domain seam and arbiter lock. Dropping payloads
                // under the control/payload locks would invert that order.
                let charge = self
                    .host
                    .charge("reentrant", "drop", AllocationClass::Scratch, 16)
                    .unwrap();
                let (limit, reserved, _remaining, serving) = self.host.snapshot().unwrap();
                assert!(limit > 0);
                assert!(reserved >= serving);
                drop(charge);
                self.reentered.store(true, Ordering::Release);
            }
        }
        let admission = host(512);
        let queue = PendingQueue::<Reentrant>::new(admission.clone(), None).unwrap();
        queue.control.state.lock().unwrap().consumers = 1;
        let reentered = Arc::new(AtomicBool::new(false));
        let ticket = queue.reserve().unwrap();
        assert!(
            queue
                .enqueue(
                    ticket,
                    Reentrant {
                        host: admission.clone(),
                        reentered: Arc::clone(&reentered),
                    },
                )
                .is_ok()
        );
        drop(queue);
        assert!(reentered.load(Ordering::Acquire));
        assert_eq!(admission.serving_bytes(), 0);
    }

    #[cfg(feature = "cache")]
    #[test]
    fn transfer_admission_clones_the_domain_without_charging_it() {
        let admission = host(512);
        let before = admission.snapshot().unwrap();
        let transfer = admission.transfer_admission("alias", "session").unwrap();
        assert_eq!(transfer.resource, admission.key);
        assert_eq!(transfer.owner.model, "alias");
        assert_eq!(transfer.owner.session, "session");
        assert_eq!(admission.snapshot().unwrap(), before);
        // Callers perform the only admission; the seam itself never reserves.
        let lease = transfer
            .arbiter
            .admit(
                &transfer.resource,
                transfer.owner.clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 32,
                }],
            )
            .unwrap();
        assert_eq!(admission.snapshot().unwrap().1, before.1 + 32);
        drop(lease);
        assert_eq!(admission.snapshot().unwrap(), before);
    }

    #[test]
    fn health_ledger_separates_model_leases_from_serving_charges() {
        let admission = host(512);
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(admission.clone(), None).unwrap();
        let worker = queue.worker().unwrap();
        let model = admission
            .admit("model", "session", AllocationClass::ImmutableWeights, 4096)
            .unwrap();
        let (_, reserved, _, serving) = admission.snapshot().unwrap();
        assert!(serving > 0);
        assert_eq!(reserved, serving + model.reserved_bytes().unwrap());
        drop(model);
        let (_, reserved, _, serving) = admission.snapshot().unwrap();
        // No fabricated retired baseline: the model lease is simply absent.
        assert_eq!(reserved, serving);
        drop(queue);
        drop(worker);
        assert_eq!(admission.serving_bytes(), 0);
        assert_eq!(admission.snapshot().unwrap().1, 0);
    }

    #[test]
    fn admission_closed_between_reserve_and_enqueue_returns_the_item() {
        let admission = host(512);
        let queue = PendingQueue::<(CancelToken, Vec<u8>)>::new(admission.clone(), None).unwrap();
        let _worker = queue.worker().unwrap();
        let ticket = queue.reserve().unwrap();
        queue.close();
        let (token, payload) = item(false);
        let (error, returned) = queue.enqueue(ticket, (token, payload)).unwrap_err();
        assert_eq!(error.status, 503);
        assert_eq!(error.code, "worker_unavailable");
        assert_eq!(returned.1.len(), 8);
        assert_eq!(queue.snapshot().pending, 0);
        assert_eq!(queue.snapshot().in_progress, 0);
    }

    #[test]
    fn slot_inventory_refuses_unrepresentable_capacity() {
        assert!(slot_bytes::<u16>(usize::MAX).is_err());
    }

    #[test]
    fn ring_growth_refuses_before_publishing_under_host_pressure() {
        let admission = tiny_host(24 * 1024);
        let queue = PendingQueue::<[u8; 1024]>::new(admission.clone(), None).unwrap();
        let _worker = queue.worker().unwrap();
        let charged = admission.serving_bytes();
        let ticket = queue.reserve().unwrap();
        let (error, returned) = queue.enqueue(ticket, [7u8; 1024]).unwrap_err();
        // 16 live slots fit; the required 16-slot compaction spare does not.
        assert_eq!(error.code, "host_budget");
        assert_eq!(returned[0], 7);
        assert_eq!(queue.snapshot().pending, 0);
        assert_eq!(queue.snapshot().in_progress, 0);
        assert_eq!(queue.snapshot().capacity, 0);
        assert_eq!(admission.serving_bytes(), charged);
    }
}
