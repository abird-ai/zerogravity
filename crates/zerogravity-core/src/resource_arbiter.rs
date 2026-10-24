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
