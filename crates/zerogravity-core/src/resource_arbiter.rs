//! P07-G control-plane slice: source-neutral multi-provider device admission.
//!
//! Explicitly NO GPU driver or foreign serving dependency. This type shares
//! capacity across multiple providers/models on the SAME identified physical
//! device and refuses unknown device completion instead of releasing its
//! reservations prematurely. It does NOT identify hardware automatically,
//! schedule CUDA kernels, or publish StateRoot.
//!
//! The optional CubeCL G2a adapter now attests PCI/LUID/registry identity
//! and uses `process_shared()` to bind all providers and future fair queues
//! to one process-local budget authority. This is NOT cross-process OS VRAM
//! admission, measured free-memory reporting, or a physical GPU worker.

use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock};

mod host_workspace;
mod partition;
pub use host_workspace::HostWorkspace;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalDeviceKey {
    /// Explicit backend/service namespace; must correspond to one physical
    /// resource domain, not merely a model name.
    pub backend: String,
    pub device: String,
}

impl PhysicalDeviceKey {
    pub fn new(backend: &str, device: &str) -> Result<Self> {
        if backend.trim().is_empty() || device.trim().is_empty() {
            return Err(Error::Malformed("empty physical device identity".into()));
        }
        Ok(Self {
            backend: backend.to_string(),
            device: device.to_string(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AllocationClass {
    ImmutableWeights,
    RecurrentState,
    KvCache,
    Activations,
    Output,
    Scratch,
    TransferStaging,
    /// Conservative reservation that already covers a provider's internal
    /// weight, activation, scratch, staging and output subdivisions.
    ProviderEnvelope,
    Other,
}

#[derive(Clone, Debug)]
pub struct AllocationRequest {
    pub kind: AllocationClass,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceOwner {
    pub model: String,
    pub session: String,
}

impl ResourceOwner {
    pub fn new(model: &str, session: &str) -> Result<Self> {
        if model.is_empty() || session.is_empty() {
            return Err(Error::Malformed(
                "resource owner must include model and session".into(),
            ));
        }
        Ok(Self {
            model: model.to_string(),
            session: session.to_string(),
        })
    }
}

/// Borrow an existing, caller-registered resource domain. Never creates an
/// arbiter, registers a physical key, or submits a GPU operation; the borrowing
/// owner performs the only admission in its own domain.
#[derive(Clone)]
pub struct TransferAdmission {
    pub arbiter: ResourceArbiter,
    pub resource: PhysicalDeviceKey,
    pub owner: ResourceOwner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeasePhase {
    Reserved,
    Submitted,
    Ready,
    Unknown,
}
struct Reservation {
    owner: ResourceOwner,
    bytes: u64,
    allocations: Vec<AllocationRequest>,
    phase: LeasePhase,
}
struct DeviceBudget {
    limit: u64,
    committed: u64,
    /// Unknown GPU work fences the ENTIRE device service, not only the model.
    fenced: bool,
    reservations: BTreeMap<u64, Reservation>,
}
struct ArbiterState {
    next_id: u64,
    budgets: BTreeMap<PhysicalDeviceKey, DeviceBudget>,
}

/// One `BTreeMap<u64, Reservation>` node upper bound from the actual std
/// layout: a 16-byte header (parent pointer + length/height), 11 key slots, 12
/// edges and the inline value slots (leaf 1072 B, internal 1168 B). A live
/// lease owns at most one internal node.
const RESERVATION_BTREE_KEYS: usize = 11;
const RESERVATION_BTREE_EDGES: usize = 12;
const RESERVATION_NODE_HEADER: usize = 16;
const RESERVATION_SIZE: usize = std::mem::size_of::<Reservation>();
const RESERVATION_NODE_UPPER: usize = RESERVATION_NODE_HEADER
    + RESERVATION_BTREE_EDGES * std::mem::size_of::<usize>()
    + RESERVATION_BTREE_KEYS * (std::mem::size_of::<u64>() + RESERVATION_SIZE);
const _: () = assert!(RESERVATION_SIZE == 88);
const _: () = assert!(RESERVATION_NODE_UPPER == 1168);

/// Non-payload inventory of one live reservation, priced from the private
/// typed layout instead of a hand-counted pointer guess. The `Vec` copy is
/// exact because `admit` builds it from the caller slice; the owner `String`
/// payload is the only variable part. Every conversion, sum and product is
/// checked so a huge `allocations` count can never wrap into a small bound.
/// This is admission-control storage only; the caller still passes the actual
/// payload bytes to `admit`, and the helper never reserves recursively.
pub fn lease_inventory_bytes(alias: &str, session: &str, allocations: usize) -> Result<u64> {
    let overflow = || Error::Limit("lease inventory overflow".into());
    let fixed = u64::try_from(RESERVATION_SIZE)
        .ok()
        .and_then(|size| {
            u64::try_from(RESERVATION_NODE_UPPER)
                .ok()
                .and_then(|node| size.checked_add(node))
        })
        .ok_or_else(overflow)?;
    let alias = u64::try_from(alias.len()).map_err(|_| overflow())?;
    let session = u64::try_from(session.len()).map_err(|_| overflow())?;
    let allocations = u64::try_from(allocations).map_err(|_| overflow())?;
    let request =
        u64::try_from(std::mem::size_of::<AllocationRequest>()).map_err(|_| overflow())?;
    let vector = allocations.checked_mul(request).ok_or_else(overflow)?;
    fixed
        .checked_add(alias)
        .and_then(|total| total.checked_add(session))
        .and_then(|total| total.checked_add(vector))
        .ok_or_else(overflow)
}

/// Actual admission inventory for one lease, from the values `admit` consumes.
///
/// [`lease_inventory_bytes`] prices the reservation node, the owner `String`
/// lengths and the `Vec` copy. `admit` additionally CLONES `key.backend`/
/// `key.device` into the lease (each clone allocates exactly its length), and a
/// by-value [`ResourceOwner`] can carry `String` spare capacity that the lease
/// then keeps alive. Both are real, so both are priced here. The inline
/// `GlobalResourceLease` value belongs to the caller's own struct and is NOT
/// added again. Every term is checked.
pub fn lease_actual_inventory_bytes(
    key: &PhysicalDeviceKey,
    owner: &ResourceOwner,
    allocations: usize,
) -> Result<u64> {
    let overflow = || Error::Limit("lease inventory overflow".into());
    let base = lease_inventory_bytes(&owner.model, &owner.session, allocations)?;
    let payload = |text: &String| u64::try_from(text.len()).map_err(|_| overflow());
    let spare = |text: &String| {
        u64::try_from(text.capacity().saturating_sub(text.len())).map_err(|_| overflow())
    };
    let key_payload = payload(&key.backend)?
        .checked_add(payload(&key.device)?)
        .ok_or_else(overflow)?;
    let spare_payload = spare(&owner.model)?
        .checked_add(spare(&owner.session)?)
        .ok_or_else(overflow)?;
    base.checked_add(key_payload)
        .and_then(|total| total.checked_add(spare_payload))
        .ok_or_else(overflow)
}

/// Pinned Rust 1.95 thread-control heap rows for one spawned thread.
///
/// Source geometry: `Thread`s private `Inner{name,id,parker}`, the lifecycle
/// `Packet`/`ThreadInit`/`rust_start` boxes, the inline spawn-hook value, the
/// runtime handler-map node and its process-retained map root. These are
/// application/runtime control allocations only; native stacks, guards,
/// alternate signal stacks, TLS, glibc caches, allocator arenas and OS
/// mappings stay external unquantified headroom.
///
/// Scope: the rows are pinned Rust 1.95 `target_os = "linux"` with 64-bit
/// pointers (the `AtomicU32` Parker and the `MAP_NODE` node width). Other
/// targets get the same checked arithmetic from their own `Layout`s; no other
/// OS or pointer width carries a numeric proof.
mod thread_control {
    use super::{
        AllocationClass, AllocationRequest, Error, GlobalResourceLease, PhysicalDeviceKey,
        ResourceArbiter, ResourceOwner, Result,
    };
    use std::alloc::Layout;
    use std::sync::Mutex;

    /// One `BTreeMap<usize, ThreadInfo>` internal node for the pinned B=6 tree.
    const MAP_NODE: u64 = 640;
    /// Process-retained runtime handler-map root allowance, admitted once.
    pub const ROOT_BYTES: u64 = MAP_NODE;

    static ROOT: Mutex<Option<GlobalResourceLease>> = Mutex::new(None);

    fn overflow() -> Error {
        Error::Limit("thread control inventory overflow".into())
    }

    fn round(size: u64, align: u64) -> Result<u64> {
        size.checked_add(align - 1)
            .map(|value| value / align * align)
            .ok_or_else(overflow)
    }

    /// `L(fields)`: alignment is the maximum field alignment and size is the
    /// checked sum of each field rounded to it. Bounds field reordering and
    /// padding without assuming tuple layout equivalence.
    fn row(fields: &[(u64, u64)]) -> Result<(u64, u64)> {
        let align = fields.iter().map(|field| field.1).max().unwrap_or(1);
        let mut total = 0u64;
        for field in fields {
            total = total
                .checked_add(round(field.0, align)?)
                .ok_or_else(overflow)?;
        }
        Ok((total, align))
    }

    /// `A(X)=round_a(round_a(2*sizeof(usize))+size(X))`, `a=max(align(X),8)`:
    /// one `Arc` allocation including its two reference counts.
    fn arc(payload: (u64, u64)) -> Result<(u64, u64)> {
        let align = payload.1.max(core::mem::align_of::<usize>() as u64);
        let base = round(2 * core::mem::size_of::<usize>() as u64, align)?;
        let size = round(base.checked_add(payload.0).ok_or_else(overflow)?, align)?;
        Ok((size, align))
    }

    fn of<T>() -> (u64, u64) {
        (
            core::mem::size_of::<T>() as u64,
            core::mem::align_of::<T>() as u64,
        )
    }

    fn named_bytes(name: Option<(usize, usize)>) -> Result<u64> {
        let Some((capacity, len)) = name else {
            return Ok(0);
        };
        if len > capacity {
            return Err(Error::Malformed(
                "thread name length exceeds its bounded capacity".into(),
            ));
        }
        let capacity = u64::try_from(capacity).map_err(|_| overflow())?;
        let len = u64::try_from(len).map_err(|_| overflow())?;
        // Input/backing growth, CString shrink/copy and the handler's own
        // copied `Box<str>` name.
        let growth = capacity
            .checked_mul(2)
            .ok_or_else(overflow)?
            .max(len.checked_add(1).ok_or_else(overflow)?)
            .max(8);
        capacity
            .checked_add(growth)
            .and_then(|total| total.checked_add(len + 1))
            .and_then(|total| total.checked_add(len))
            .ok_or_else(overflow)
    }

    /// Checked application thread-control row for one spawn whose capture
    /// closure has `capture`'s layout and whose bounded name is
    /// `name = Some((capacity, len))`. `START` includes the capture, so callers
    /// must not charge a separate capture width.
    pub fn rust_thread_control_bytes(capture: Layout, name: Option<(usize, usize)>) -> Result<u64> {
        // `NAME_INLINE=L(usize,CString)`; the explicit Option discriminant is
        // covered, no private niche assumption.
        let name_inline = row(&[of::<usize>(), of::<std::ffi::CString>()])?;
        // `INNER=L(NAME_INLINE,ThreadId,AtomicU32)`, minimum alignment 8. The
        // Parker is the pinned Linux futex `AtomicU32`, not a pointer stand-in.
        let inner = row(&[
            name_inline,
            of::<std::thread::ThreadId>(),
            of::<std::sync::atomic::AtomicU32>(),
            (0, 8),
        ])?;
        let thread = arc(inner)?;
        // `PACKET` uses the public `Option`/`Result` layout, not a guessed width.
        let packet = arc(row(&[
            of::<Option<std::sync::Arc<()>>>(),
            of::<Option<std::thread::Result<()>>>(),
        ])?)?;
        let init = row(&[
            of::<std::thread::Thread>(),
            of::<Box<dyn FnOnce() + Send>>(),
        ])?;
        let hooks = row(&[
            of::<Option<std::sync::Arc<()>>>(),
            of::<Vec<Box<dyn FnOnce() + Send>>>(),
        ])?;
        let start = row(&[
            (capture.size() as u64, capture.align() as u64),
            hooks,
            of::<std::sync::Arc<()>>(),
        ])?;
        let named = named_bytes(name)?;
        thread
            .0
            .checked_add(packet.0)
            .and_then(|total| total.checked_add(init.0))
            .and_then(|total| total.checked_add(start.0))
            .and_then(|total| total.checked_add(named))
            // Two map nodes per live thread: the leaf and split overlap.
            .and_then(|total| total.checked_add(2 * MAP_NODE))
            .ok_or_else(overflow)
    }

    /// `A(AtomicBool)`: the initializer's one-per-attempt control allocation.
    pub fn thread_attempt_flag_bytes() -> Result<u64> {
        Ok(arc(of::<std::sync::atomic::AtomicBool>())?.0)
    }

    /// Admit the single process-retained thread-control root allowance in the
    /// caller's existing arbiter/key domain. Reuse verifies the same domain and
    /// actual charge, so a second router neither admits nor multiplies it.
    pub fn admit_thread_control_root(
        arbiter: &ResourceArbiter,
        key: &PhysicalDeviceKey,
        owner: ResourceOwner,
    ) -> Result<u64> {
        admit_root_once(&ROOT, arbiter, key, owner)
    }

    /// The production admission above, over an explicit root slot so a test can
    /// hold its own fresh slot instead of clearing the process root.
    fn admit_root_once(
        root: &Mutex<Option<GlobalResourceLease>>,
        arbiter: &ResourceArbiter,
        key: &PhysicalDeviceKey,
        owner: ResourceOwner,
    ) -> Result<u64> {
        let mut root = root
            .lock()
            .map_err(|_| Error::State("thread-control root lock poisoned".into()))?;
        if let Some(lease) = root.as_ref() {
            return lease.bytes_in_domain(arbiter, key);
        }
        let metadata = super::lease_actual_inventory_bytes(key, &owner, 1)?;
        let bytes = ROOT_BYTES.checked_add(metadata).ok_or_else(overflow)?;
        let lease = arbiter.admit(
            key,
            owner,
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes,
            }],
        )?;
        let bytes = lease.reserved_bytes()?;
        *root = Some(lease);
        Ok(bytes)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
        #[test]
        fn pinned_control_rows_match_the_linux_x86_64_geometry() {
            // THREAD 56 + PACKET 48 + INIT 24 + START 40 + N 0 + MAP 1280.
            assert_eq!(
                rust_thread_control_bytes(Layout::new::<()>(), None).unwrap(),
                1_448
            );
            assert_eq!(thread_attempt_flag_bytes().unwrap(), 24);
            let named = rust_thread_control_bytes(Layout::new::<()>(), Some((7, 7))).unwrap();
            assert_eq!(named - 1_448, 7 + 14 + 8 + 7);
            assert!(rust_thread_control_bytes(Layout::new::<()>(), Some((3, 4))).is_err());
        }

        #[test]
        fn root_prices_the_cloned_key_and_owner_spare_capacity() {
            let root = Mutex::new(None);
            let arbiter = ResourceArbiter::new();
            // Long domain strings plus an owner that carries spare capacity:
            // the base typed subtotal ALONE fits, the actual admission does not.
            let key = PhysicalDeviceKey {
                backend: "host-with-a-deliberately-long-backend-name".into(),
                device: "root-spare-capacity".into(),
            };
            // `String::clone` allocates exactly `len`, so every attempt gets a
            // freshly constructed owner that really carries spare capacity.
            let owner = || {
                let mut model = String::with_capacity(96);
                model.push_str("model");
                let mut session = String::with_capacity(96);
                session.push_str("session");
                ResourceOwner { model, session }
            };
            let base =
                super::super::lease_inventory_bytes(&owner().model, &owner().session, 1).unwrap();
            let actual = super::super::lease_actual_inventory_bytes(&key, &owner(), 1).unwrap();
            assert!(actual > base, "key clone and spare capacity are real");
            let first = owner();
            let expected_delta = (key.backend.len() + key.device.len() + first.model.capacity()
                - first.model.len()
                + first.session.capacity()
                - first.session.len()) as u64;
            assert_eq!(actual - base, expected_delta);
            arbiter.register(key.clone(), ROOT_BYTES + base).unwrap();
            let refused = admit_root_once(&root, &arbiter, &key, owner());
            assert!(matches!(refused, Err(Error::Limit(_))), "{refused:?}");
            assert!(root.lock().unwrap().is_none());
            assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);

            // The exact composed inventory admits once and reuses unchanged.
            let exact = ResourceArbiter::new();
            exact.register(key.clone(), ROOT_BYTES + actual).unwrap();
            let admitted = admit_root_once(&root, &exact, &key, owner()).unwrap();
            assert_eq!(admitted, ROOT_BYTES + actual);
            assert_eq!(exact.snapshot(&key).unwrap().reserved, admitted);
            assert_eq!(
                admit_root_once(&root, &exact, &key, owner()).unwrap(),
                admitted
            );
            assert_eq!(exact.snapshot(&key).unwrap().reserved, admitted);
        }

        #[test]
        fn root_is_admitted_once_in_the_original_domain() {
            // A fresh slot keeps the process root intact and makes every step
            // a real assertion instead of tolerating an already-primed Err.
            let root = Mutex::new(None);
            let arbiter = ResourceArbiter::new();
            let key = PhysicalDeviceKey::new("host", "root-once").unwrap();
            arbiter.register(key.clone(), 1_000_000).unwrap();
            let owner = ResourceOwner::new("model", "session").unwrap();
            let first = admit_root_once(&root, &arbiter, &key, owner.clone()).unwrap();
            let metadata =
                crate::resource_arbiter::lease_actual_inventory_bytes(&key, &owner, 1).unwrap();
            assert_eq!(first, ROOT_BYTES + metadata);
            assert_eq!(arbiter.snapshot(&key).unwrap().reserved, first);
            // Same original domain reuses the retained lease: no new charge.
            assert_eq!(
                admit_root_once(&root, &arbiter, &key, owner.clone()).unwrap(),
                first
            );
            assert_eq!(arbiter.snapshot(&key).unwrap().reserved, first);
            // A different domain is refused, not silently re-rooted.
            let other = ResourceArbiter::new();
            let other_key = PhysicalDeviceKey::new("host", "root-once-other").unwrap();
            other.register(other_key.clone(), 1_000_000).unwrap();
            assert!(admit_root_once(&root, &other, &other_key, owner).is_err());
            assert_eq!(other.snapshot(&other_key).unwrap().reserved, 0);
        }
    }
}

pub use thread_control::{
    ROOT_BYTES as THREAD_CONTROL_ROOT_BYTES, admit_thread_control_root, rust_thread_control_bytes,
    thread_attempt_flag_bytes,
};

#[derive(Clone)]
pub struct ResourceArbiter {
    shared: Arc<Mutex<ArbiterState>>,
}

#[derive(Clone, Debug)]
pub struct DeviceBudgetSnapshot {
    pub limit: u64,
    pub reserved: u64,
    pub remaining: u64,
    pub reservations: usize,
    pub fenced: bool,
    pub per_class: BTreeMap<AllocationClass, u64>,
    pub active_models: BTreeSet<String>,
}

impl ResourceArbiter {
    /// One authoritative process-level admission instance shared by the
    /// GPU service adapter AND any G1 fair queues/workers using that card.
    /// Tests may construct isolated Arbiters with `new`, but production
    /// physical-memory owners must not independently create device budgets.
    pub fn process_shared() -> Self {
        static SHARED: OnceLock<ResourceArbiter> = OnceLock::new();
        SHARED.get_or_init(Self::new).clone()
    }

    /// Inspect the original owner identity; no reservation or budget mutation.
    pub fn is_process_shared(&self) -> bool {
        Arc::ptr_eq(&self.shared, &Self::process_shared().shared)
    }

    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(ArbiterState {
                next_id: 1,
                budgets: BTreeMap::new(),
            })),
        }
    }

    /// The real backend must map all providers sharing one physical memory
    /// pool to precisely one PhysicalDeviceKey. Configuration refusal is
    /// preferable to inventing a hardware limit from GTT addressability.
    pub fn register(&self, key: PhysicalDeviceKey, limit: u64) -> Result<()> {
        if limit == 0 {
            return Err(Error::Limit(
                "physical device budget must be positive".into(),
            ));
        }
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        if state.budgets.contains_key(&key) {
            return Err(Error::State("physical device already registered".into()));
        }
        state.budgets.insert(
            key,
            DeviceBudget {
                limit,
                committed: 0,
                fenced: false,
                reservations: BTreeMap::new(),
            },
        );
        Ok(())
    }

    /// Idempotent registration for multiple independently initialized
    /// backends that have independently attested the SAME physical budget.
    /// A different limit for the same device is not silently reconciled:
    /// otherwise two providers could believe incompatible memory headrooms.
    pub fn register_matching(&self, key: PhysicalDeviceKey, limit: u64) -> Result<()> {
        if limit == 0 {
            return Err(Error::Limit(
                "physical device budget must be positive".into(),
            ));
        }
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        if let Some(existing) = state.budgets.get(&key) {
            return if existing.limit == limit {
                Ok(())
            } else {
                Err(Error::State(
                    "conflicting physical device budget attestation".into(),
                ))
            };
        }
        state.budgets.insert(
            key,
            DeviceBudget {
                limit,
                committed: 0,
                fenced: false,
                reservations: BTreeMap::new(),
            },
        );
        Ok(())
    }

    /// Transactional reservation for an entire bundle of static/dynamic
    /// resources, not one allocation at a time. No partial budget capture.
    pub fn admit(
        &self,
        key: &PhysicalDeviceKey,
        owner: ResourceOwner,
        allocations: &[AllocationRequest],
    ) -> Result<GlobalResourceLease> {
        if allocations.is_empty() || allocations.iter().any(|x| x.bytes == 0) {
            return Err(Error::Limit("empty/zero resource request".into()));
        }
        let total = allocations.iter().try_fold(0u64, |total, x| {
            total
                .checked_add(x.bytes)
                .ok_or_else(|| Error::Limit("resource reservation sum overflow".into()))
        })?;
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        // Validate everything before mutating counters.
        {
            let budget = state
                .budgets
                .get(key)
                .ok_or_else(|| Error::State("unknown physical device service".into()))?;
            if budget.fenced {
                return Err(Error::State(
                    "physical device completion uncertain: fenced".into(),
                ));
            }
            if total > budget.limit.saturating_sub(budget.committed) {
                return Err(Error::Limit(
                    "physical device memory admission refused".into(),
                ));
            }
        }
        let id = state.next_id;
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| Error::Limit("resource reservation ID exhausted".into()))?;
        let budget = state
            .budgets
            .get_mut(key)
            .expect("previously validated device");
        budget.committed += total;
        budget.reservations.insert(
            id,
            Reservation {
                owner,
                bytes: total,
                allocations: allocations.to_vec(),
                phase: LeasePhase::Reserved,
            },
        );
        Ok(GlobalResourceLease {
            shared: self.shared.clone(),
            device: key.clone(),
            id,
            released: false,
        })
    }

    /// Cheap hot-path physical fence check. Unlike snapshot(), this does not
    /// allocate or enumerate per-model/per-class reservations under the lock.
    /// It is only a point-in-time observation; actual device launch is still
    /// guarded by the noncloneable retained physical resource lease.
    pub fn is_fenced(&self, key: &PhysicalDeviceKey) -> Result<bool> {
        let state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        Ok(state
            .budgets
            .get(key)
            .ok_or_else(|| Error::State("unknown physical device".into()))?
            .fenced)
    }

    /// Linearize a short synchronous publication with every physical fence.
    /// The authoritative arbiter mutex stays held from the unfenced check
    /// through `publish`; a concurrent fence can occur only before or after it.
    /// Unknown devices and poisoned/fenced arbiters never invoke the closure.
    ///
    /// The closure must not wait for GPU work, allocate physical resources,
    /// call ANY method which re-enters this arbiter (including indirectly),
    /// or drop resource-owning values. Complete all fallible validation before
    /// assigning a payload/root. Use `mem::replace` and RETURN displaced GPU
    /// leases/payloads, then drop them AFTER this method returns; their Drop
    /// implementations may re-enter resource admission/fencing. The closure
    /// remains responsible for its payload; this method cannot roll it back.
    pub fn with_unfenced<T>(
        &self,
        key: &PhysicalDeviceKey,
        publish: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        let budget = state
            .budgets
            .get(key)
            .ok_or_else(|| Error::State("unknown physical device".into()))?;
        if budget.fenced {
            return Err(Error::State("physical device is fenced".into()));
        }
        let result = publish();
        drop(state);
        result
    }

    pub fn snapshot(&self, key: &PhysicalDeviceKey) -> Result<DeviceBudgetSnapshot> {
        let state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        let b = state
            .budgets
            .get(key)
            .ok_or_else(|| Error::State("unknown device budget".into()))?;
        let mut per_class: BTreeMap<AllocationClass, u64> = BTreeMap::new();
        let mut active_models = BTreeSet::new();
        for reservation in b.reservations.values() {
            active_models.insert(reservation.owner.model.clone());
            for part in &reservation.allocations {
                *per_class.entry(part.kind).or_default() += part.bytes;
            }
        }
        Ok(DeviceBudgetSnapshot {
            limit: b.limit,
            reserved: b.committed,
            remaining: b.limit - b.committed,
            reservations: b.reservations.len(),
            fenced: b.fenced,
            per_class,
            active_models,
        })
    }

    /// Reusable resident leases need one atomic Ready/Reserved→Submitted
    /// transition per dispatch, without a failed phase-change lock attempt.
    fn submit_retained(&self, key: &PhysicalDeviceKey, id: u64) -> Result<()> {
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        let budget = state
            .budgets
            .get_mut(key)
            .ok_or_else(|| Error::State("unknown physical device".into()))?;
        if budget.fenced {
            return Err(Error::State("physical device is fenced".into()));
        }
        let reservation = budget
            .reservations
            .get_mut(&id)
            .ok_or_else(|| Error::State("unknown/released memory lease".into()))?;
        if !matches!(reservation.phase, LeasePhase::Reserved | LeasePhase::Ready) {
            return Err(Error::State("resource already submitted or unsafe".into()));
        }
        reservation.phase = LeasePhase::Submitted;
        Ok(())
    }

    /// Terminal process-wide fence when a physical device worker is dropped
    /// while a submitted job may still be executing. The physical provider's
    /// own guard retains actual Handle keepalives and its reservation. This
    /// does not permit early freeing, timed recovery, or falsely signal Ready.
    pub fn fence_unknown_device(&self, key: &PhysicalDeviceKey) -> Result<()> {
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        let budget = state
            .budgets
            .get_mut(key)
            .ok_or_else(|| Error::State("unknown physical device".into()))?;
        budget.fenced = true;
        Ok(())
    }

    /// There is deliberately NO public "unfence" or forced teardown. A real
    /// backend must have proved all GPU work has stopped before recovering.
    fn phase_change(
        &self,
        key: &PhysicalDeviceKey,
        id: u64,
        from: LeasePhase,
        to: LeasePhase,
    ) -> Result<()> {
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        let b = state
            .budgets
            .get_mut(key)
            .ok_or_else(|| Error::State("unknown physical device".into()))?;
        if b.fenced {
            return Err(Error::State("physical device is fenced".into()));
        }
        let reservation = b
            .reservations
            .get_mut(&id)
            .ok_or_else(|| Error::State("unknown/released memory lease".into()))?;
        if reservation.phase != from {
            return Err(Error::State(
                "invalid resource lease phase transition".into(),
            ));
        }
        reservation.phase = to;
        Ok(())
    }

    fn mark_unknown(&self, key: &PhysicalDeviceKey, id: u64) {
        let Ok(mut state) = self.shared.lock() else {
            return;
        };
        let Some(device) = state.budgets.get_mut(key) else {
            return;
        };
        if let Some(reservation) = device.reservations.get_mut(&id) {
            reservation.phase = LeasePhase::Unknown;
            device.fenced = true;
        }
    }

    fn retire(&self, key: &PhysicalDeviceKey, id: u64) -> Result<()> {
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        let device = state
            .budgets
            .get_mut(key)
            .ok_or_else(|| Error::State("unknown device budget".into()))?;
        let reservation = device
            .reservations
            .get(&id)
            .ok_or_else(|| Error::State("unknown/released resource lease".into()))?;
        if !matches!(reservation.phase, LeasePhase::Reserved | LeasePhase::Ready) {
            return Err(Error::State(
                "cannot release in-flight or unknown physical memory".into(),
            ));
        }
        let size = reservation.bytes;
        device.reservations.remove(&id);
        device.committed -= size;
        Ok(())
    }
}
impl Default for ResourceArbiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Owner-bound RAII lease. Mark submitted before physical dispatch, ready
/// ONLY after a trusted backend completion receipt. A dropped submitted
/// lease conservatively fences the entire device and retains its charge.
#[must_use]
pub struct GlobalResourceLease {
    shared: Arc<Mutex<ArbiterState>>,
    device: PhysicalDeviceKey,
    id: u64,
    released: bool,
}

impl GlobalResourceLease {
    fn arbiter(&self) -> ResourceArbiter {
        ResourceArbiter {
            shared: self.shared.clone(),
        }
    }
    /// Verify a detached charge without exposing its original arbiter/key.
    pub fn bytes_in_domain(
        &self,
        arbiter: &ResourceArbiter,
        key: &PhysicalDeviceKey,
    ) -> Result<u64> {
        if self.released || !Arc::ptr_eq(&self.shared, &arbiter.shared) || &self.device != key {
            return Err(Error::State(
                "detached reservation belongs to another admission domain".into(),
            ));
        }
        self.reserved_bytes()
    }
    /// Current charge of this retained lease, including an unknown reservation.
    pub fn reserved_bytes(&self) -> Result<u64> {
        if self.released {
            return Err(Error::State("detached reservation released".into()));
        }
        let state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        let reservation = state
            .budgets
            .get(&self.device)
            .and_then(|b| b.reservations.get(&self.id))
            .ok_or_else(|| Error::State("detached reservation absent".into()))?;
        Ok(reservation.bytes)
    }
    pub fn mark_submitted(&self) -> Result<()> {
        // A retained physical allocation can be resubmitted after genuine
        // Ready while remaining globally charged to its original owner.
        self.arbiter().submit_retained(&self.device, self.id)
    }
    /// Terminal, service-wide fence for a submitted GPU operation whose
    /// completion cannot be proved. Never used as an ordinary cancellation.
    pub fn fence_unknown(&self) {
        self.arbiter().mark_unknown(&self.device, self.id);
    }
    /// The caller must have independently observed real device completion.
    /// This ledger is not itself a GPU event source.
    pub fn mark_ready(&self) -> Result<()> {
        self.arbiter().phase_change(
            &self.device,
            self.id,
            LeasePhase::Submitted,
            LeasePhase::Ready,
        )
    }
    pub fn release(mut self) -> Result<()> {
        self.arbiter().retire(&self.device, self.id)?;
        self.released = true;
        Ok(())
    }
}
impl Drop for GlobalResourceLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        // If submitted work's physical completion is still uncertain, retain
        // the memory charge and fence the entire physical service.
        let arbiter = self.arbiter();
        if arbiter.retire(&self.device, self.id).is_err() {
            arbiter.mark_unknown(&self.device, self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn device(name: &str) -> PhysicalDeviceKey {
        PhysicalDeviceKey::new("vulkan", name).unwrap()
    }
    fn owner(model: &str) -> ResourceOwner {
        ResourceOwner::new(model, "session-1").unwrap()
    }
    fn req(kind: AllocationClass, bytes: u64) -> AllocationRequest {
        AllocationRequest { kind, bytes }
    }

    #[test]
    fn lease_inventory_is_checked_and_prices_the_typed_layout() {
        let fixed = u64::try_from(RESERVATION_SIZE).unwrap()
            + u64::try_from(RESERVATION_NODE_UPPER).unwrap();
        assert_eq!(lease_inventory_bytes("m", "s", 0).unwrap(), fixed + 2);
        let request = u64::try_from(std::mem::size_of::<AllocationRequest>()).unwrap();
        assert_eq!(
            lease_inventory_bytes("mo", "se", 2).unwrap(),
            fixed + 4 + 2 * request
        );
        assert!(lease_inventory_bytes("m", "s", usize::MAX).is_err());
        // The inventory is control-only and charged exactly once per lease;
        // it never recursively admits storage to account for itself.
        let a = ResourceArbiter::new();
        let gpu = device("inventory");
        a.register(gpu.clone(), 1_000_000).unwrap();
        let cost = lease_inventory_bytes("owner", "session", 1).unwrap();
        let lease = a
            .admit(&gpu, owner("owner"), &[req(AllocationClass::Scratch, cost)])
            .unwrap();
        assert_eq!(a.snapshot(&gpu).unwrap().reserved, cost);
        lease.release().unwrap();
        assert_eq!(a.snapshot(&gpu).unwrap().reserved, 0);
    }

    #[test]
    fn compiler_context_detached_lease_requires_same_arbiter_and_key() {
        let a = ResourceArbiter::new();
        let b = ResourceArbiter::new();
        let first = device("detached-a");
        let other = device("detached-b");
        for arb in [&a, &b] {
            arb.register(first.clone(), 64).unwrap();
        }
        a.register(other.clone(), 64).unwrap();
        let lease = a
            .admit(
                &first,
                owner("detached"),
                &[req(AllocationClass::Scratch, 32)],
            )
            .unwrap();
        assert_eq!(lease.bytes_in_domain(&a, &first).unwrap(), 32);
        assert!(lease.bytes_in_domain(&a, &other).is_err());
        assert!(lease.bytes_in_domain(&b, &first).is_err());
        assert_eq!(a.snapshot(&first).unwrap().reserved, 32);
        drop(lease);
        assert_eq!(a.snapshot(&first).unwrap().reserved, 0);
    }
    #[test]
    fn physical_budget_enforces_cross_model_shared_admission() {
        let a = ResourceArbiter::new();
        let gpu = device("gpu-0");
        let other = device("gpu-1");
        a.register(gpu.clone(), 100).unwrap();
        a.register(other.clone(), 100).unwrap();
        assert!(a.register(gpu.clone(), 100).is_err());
        let first = a
            .admit(
                &gpu,
                owner("qwen-0.8b"),
                &[
                    req(AllocationClass::ImmutableWeights, 55),
                    req(AllocationClass::Scratch, 15),
                ],
            )
            .unwrap();
        assert!(
            a.admit(&gpu, owner("qwen-2b"), &[req(AllocationClass::KvCache, 40)])
                .is_err()
        );
        assert_eq!(a.snapshot(&gpu).unwrap().reserved, 70);
        let second = a
            .admit(
                &other,
                owner("qwen-2b"),
                &[req(AllocationClass::RecurrentState, 70)],
            )
            .unwrap();
        assert_eq!(a.snapshot(&other).unwrap().reserved, 70);
        first.release().unwrap();
        let shared = a
            .admit(&gpu, owner("qwen-2b"), &[req(AllocationClass::KvCache, 40)])
            .unwrap();
        assert_eq!(a.snapshot(&gpu).unwrap().reserved, 40);
        drop(shared);
        second.release().unwrap();
        assert_eq!(a.snapshot(&gpu).unwrap().reserved, 0);
        assert_eq!(a.snapshot(&other).unwrap().reserved, 0);
    }

    #[test]
    fn atomic_multi_class_and_overflow_refuse_without_charges() {
        let a = ResourceArbiter::new();
        let gpu = device("gpu-zero");
        a.register(gpu.clone(), 120).unwrap();
        assert!(a.admit(&gpu, owner("model"), &[]).is_err());
        assert!(
            a.admit(
                &gpu,
                owner("model"),
                &[
                    req(AllocationClass::Scratch, 20),
                    req(AllocationClass::Activations, 0)
                ]
            )
            .is_err()
        );
        assert!(
            a.admit(
                &gpu,
                owner("model"),
                &[
                    req(AllocationClass::Scratch, u64::MAX),
                    req(AllocationClass::Activations, 1)
                ]
            )
            .is_err()
        );
        assert_eq!(a.snapshot(&gpu).unwrap().reserved, 0);
        let lease = a
            .admit(
                &gpu,
                owner("model"),
                &[
                    req(AllocationClass::ImmutableWeights, 20),
                    req(AllocationClass::TransferStaging, 30),
                    req(AllocationClass::Activations, 5),
                ],
            )
            .unwrap();
        let view = a.snapshot(&gpu).unwrap();
        assert_eq!(view.reserved, 55);
        assert_eq!(view.per_class[&AllocationClass::TransferStaging], 30);
        lease.release().unwrap();
    }

    #[test]
    fn submitted_drop_fences_all_other_models_and_preserves_charge() {
        let a = ResourceArbiter::new();
        let gpu = device("gpu-shared");
        a.register(gpu.clone(), 200).unwrap();
        let lease = a
            .admit(
                &gpu,
                owner("model-A"),
                &[req(AllocationClass::ImmutableWeights, 120)],
            )
            .unwrap();
        lease.mark_submitted().unwrap();
        drop(lease);
        let snapshot = a.snapshot(&gpu).unwrap();
        assert!(snapshot.fenced);
        assert_eq!(snapshot.reserved, 120);
        assert!(
            a.admit(&gpu, owner("model-B"), &[req(AllocationClass::Scratch, 16)])
                .is_err()
        );
        // Unknown physical GPU allocations are never released by timeout.
    }

    #[test]
    fn gpu_and_device_queue_can_share_one_process_arbiter() {
        let a = ResourceArbiter::process_shared();
        let b = ResourceArbiter::process_shared();
        assert!(Arc::ptr_eq(&a.shared, &b.shared));
    }

    #[test]
    fn matching_physical_registration_reuses_one_budget_not_two() {
        let a = ResourceArbiter::new();
        let gpu = device("pci:0000:07:00.0");
        a.register_matching(gpu.clone(), 100).unwrap();
        a.register_matching(gpu.clone(), 100).unwrap();
        assert!(a.register_matching(gpu.clone(), 101).is_err());
        let l1 = a
            .admit(&gpu, owner("a"), &[req(AllocationClass::Other, 75)])
            .unwrap();
        assert!(
            a.admit(&gpu, owner("b"), &[req(AllocationClass::Other, 50)])
                .is_err()
        );
        l1.release().unwrap();
        assert_eq!(a.snapshot(&gpu).unwrap().remaining, 100);
    }

    #[test]
    fn resubmitting_a_ready_retained_allocation_requires_a_new_ready() {
        let a = ResourceArbiter::new();
        let gpu = device("pci:0000:07:00.1");
        a.register(gpu.clone(), 100).unwrap();
        let l = a
            .admit(&gpu, owner("a"), &[req(AllocationClass::Other, 50)])
            .unwrap();
        l.mark_submitted().unwrap();
        l.mark_ready().unwrap();
        l.mark_submitted().unwrap();
        assert!(l.mark_submitted().is_err());
        assert!(
            a.admit(&gpu, owner("b"), &[req(AllocationClass::Scratch, 60)])
                .is_err()
        );
        l.mark_ready().unwrap();
        l.release().unwrap();
        assert_eq!(a.snapshot(&gpu).unwrap().reserved, 0);
    }

    #[test]
    fn confirmed_ready_releases_budget_and_double_transition_refused() {
        let a = ResourceArbiter::new();
        let gpu = device("gpu-ready");
        a.register(gpu.clone(), 128).unwrap();
        let lease = a
            .admit(&gpu, owner("model"), &[req(AllocationClass::Output, 64)])
            .unwrap();
        assert!(lease.mark_ready().is_err());
        lease.mark_submitted().unwrap();
        assert!(lease.mark_submitted().is_err());
        lease.mark_ready().unwrap();
        lease.release().unwrap();
        assert_eq!(a.snapshot(&gpu).unwrap().reserved, 0);
        assert!(!a.snapshot(&gpu).unwrap().fenced);
    }

    #[test]
    fn unfenced_publication_refuses_unknown_and_fenced_without_invoking_closure() {
        let arbiter = ResourceArbiter::new();
        let gpu = device("publication-fenced");
        let other = device("publication-other");
        let mut calls = 0;
        assert!(
            arbiter
                .with_unfenced(&gpu, || {
                    calls += 1;
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(calls, 0);
        arbiter.register(gpu.clone(), 100).unwrap();
        arbiter.register(other.clone(), 100).unwrap();
        arbiter.fence_unknown_device(&gpu).unwrap();
        assert!(
            arbiter
                .with_unfenced(&gpu, || {
                    calls += 1;
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(calls, 0);
        assert_eq!(
            arbiter
                .with_unfenced(&other, || {
                    calls += 1;
                    Ok(42)
                })
                .unwrap(),
            42
        );
        assert_eq!(calls, 1);
        assert_eq!(arbiter.snapshot(&other).unwrap().reserved, 0);
    }

    #[test]
    fn publication_holds_authoritative_mutex_until_root_assignment_then_fence_wins() {
        use std::sync::{
            TryLockError,
            atomic::{AtomicU64, Ordering},
            mpsc,
        };
        use std::time::Duration;
        let arbiter = ResourceArbiter::new();
        let gpu = device("publication-concurrent");
        arbiter.register(gpu.clone(), 100).unwrap();
        let root = Arc::new(AtomicU64::new(0));
        let (enter, entered) = mpsc::channel();
        let (checked, check) = mpsc::channel();
        let (done, completion) = mpsc::channel();
        let other_arbiter = arbiter.clone();
        let other_gpu = gpu.clone();
        let other_root = root.clone();
        let worker = std::thread::spawn(move || {
            entered.recv_timeout(Duration::from_secs(5)).unwrap();
            // Prove that publication holds the SAME mutex used by fencing;
            // no probabilistic sleep/scheduler assumption establishes order.
            assert!(matches!(
                other_arbiter.shared.try_lock(),
                Err(TryLockError::WouldBlock)
            ));
            checked.send(()).unwrap();
            other_arbiter.fence_unknown_device(&other_gpu).unwrap();
            assert_eq!(other_root.load(Ordering::Acquire), 1);
            done.send(()).unwrap();
        });
        arbiter
            .with_unfenced(&gpu, || {
                enter.send(()).unwrap();
                check.recv_timeout(Duration::from_secs(5)).unwrap();
                assert!(matches!(
                    completion.try_recv(),
                    Err(mpsc::TryRecvError::Empty)
                ));
                root.store(1, Ordering::Release);
                Ok(())
            })
            .unwrap();
        completion.recv_timeout(Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        assert!(arbiter.is_fenced(&gpu).unwrap());
        assert!(
            arbiter
                .with_unfenced(&gpu, || {
                    root.store(2, Ordering::Release);
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(root.load(Ordering::Acquire), 1);
    }

    #[test]
    fn publication_error_releases_guard_and_panic_poison_refuses_later_publication() {
        let arbiter = ResourceArbiter::new();
        let gpu = device("publication-error");
        arbiter.register(gpu.clone(), 100).unwrap();
        let result = arbiter.with_unfenced(&gpu, || -> Result<()> {
            Err(Error::State("payload validation refused".into()))
        });
        assert!(
            matches!(result, Err(Error::State(message)) if message == "payload validation refused")
        );
        assert!(!arbiter.is_fenced(&gpu).unwrap());
        assert_eq!(arbiter.with_unfenced(&gpu, || Ok(7)).unwrap(), 7);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = arbiter.with_unfenced(&gpu, || -> Result<()> {
                panic!("injected synchronous publication panic");
            });
        }));
        assert!(panic.is_err());
        let mut invoked = false;
        assert!(
            arbiter
                .with_unfenced(&gpu, || {
                    invoked = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!invoked);
    }
}
