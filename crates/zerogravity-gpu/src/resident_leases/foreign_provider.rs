//! Opaque provider allocations subdivide the SAME prepaid resident envelope.
use super::*;
use std::ops::Deref;

/// One lock linearizes managed-buffer and opaque-provider admission.
pub(crate) struct ResidentLedger {
    events: DeviceEventLedger,
    foreign_bytes: usize,
    cap: usize,
    models: BTreeMap<u64, ModelUsage>,
    leases: BTreeMap<BufferLeaseId, (u64, usize)>,
    backings: BTreeMap<zerogravity_core::device_events::DeviceBackingId, (u64, usize)>,
    event_models: BTreeMap<DeviceEventId, u64>,
}
struct ModelUsage {
    cap: usize,
    bytes: usize,
}
impl ResidentLedger {
    pub(super) fn physical(generation: u64, cap: usize) -> Result<Self> {
        Ok(Self {
            events: DeviceEventLedger::new(generation, cap)?,
            foreign_bytes: 0,
            cap,
            models: BTreeMap::new(),
            leases: BTreeMap::new(),
            backings: BTreeMap::new(),
            event_models: BTreeMap::new(),
        })
    }
    #[cfg(test)]
    pub(super) fn new(generation: u64, cap: usize) -> Result<Self> {
        let mut ledger = Self::physical(generation, cap)?;
        ledger.attach_model(0, cap)?;
        Ok(ledger)
    }
    pub(super) fn attach_model(&mut self, model: u64, cap: usize) -> Result<()> {
        if cap == 0 || self.models.contains_key(&model) || cap > self.available_model_capacity()? {
            return Err(Error::Limit(
                "physical model sub-admission exceeds original remainder".into(),
            ));
        }
        self.models.insert(model, ModelUsage { cap, bytes: 0 });
        Ok(())
    }
    pub(super) fn available_model_capacity(&self) -> Result<usize> {
        let used = self.models.values().try_fold(0usize, |n, m| {
            n.checked_add(m.cap)
                .ok_or_else(|| Error::Limit("model quota sum overflow".into()))
        })?;
        self.cap
            .checked_sub(used)
            .ok_or_else(|| Error::State("model quotas exceed physical remainder".into()))
    }
    pub(super) fn close_model(&mut self, model: u64) -> Result<()> {
        if self.model_bytes(model)? != 0 || self.model_events(model) != 0 {
            return Err(Error::State(
                "model allocations/readers/modules remain live".into(),
            ));
        }
        self.models
            .remove(&model)
            .ok_or_else(|| Error::State("model namespace already closed".into()))?;
        Ok(())
    }
    pub(super) fn model_bytes(&self, model: u64) -> Result<usize> {
        self.models
            .get(&model)
            .map(|m| m.bytes)
            .ok_or_else(|| Error::State("model namespace closed or foreign".into()))
    }
    pub(super) fn model_events(&self, model: u64) -> usize {
        self.event_models.values().filter(|&&m| m == model).count()
    }
    fn check_request(&self, model: u64, bytes: usize) -> Result<()> {
        let owned = self
            .models
            .get(&model)
            .ok_or_else(|| Error::State("model namespace closed or foreign".into()))?;
        if bytes == 0
            || owned.bytes.checked_add(bytes).is_none_or(|n| n > owned.cap)
            || self
                .resident_bytes()
                .checked_add(bytes)
                .is_none_or(|n| n > self.cap)
        {
            return Err(Error::Limit(
                "aggregate/model managed+native admission exceeded".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn admit_for(
        &mut self,
        model: u64,
        bytes: usize,
        role: BufferRole,
    ) -> Result<BufferLeaseId> {
        self.check_request(model, bytes)?;
        let id = self.events.admit(bytes, role)?;
        self.models.get_mut(&model).expect("checked model").bytes += bytes;
        self.leases.insert(id, (model, bytes));
        Ok(id)
    }
    #[cfg(test)]
    pub(super) fn admit(&mut self, bytes: usize, role: BufferRole) -> Result<BufferLeaseId> {
        self.admit_for(0, bytes, role)
    }
    pub(super) fn admit_backing_for(
        &mut self,
        model: u64,
        bytes: usize,
        role: BufferRole,
    ) -> Result<zerogravity_core::device_events::DeviceBackingId> {
        self.check_request(model, bytes)?;
        let id = self.events.admit_backing(bytes, role)?;
        self.models.get_mut(&model).expect("checked model").bytes += bytes;
        self.backings.insert(id, (model, bytes));
        Ok(id)
    }
    pub(super) fn admit_backing_view(
        &mut self,
        id: zerogravity_core::device_events::DeviceBackingId,
        offset: usize,
        bytes: usize,
        role: BufferRole,
    ) -> Result<BufferLeaseId> {
        let model = self
            .backings
            .get(&id)
            .ok_or_else(|| Error::State("foreign backing namespace".into()))?
            .0;
        let view = self.events.admit_backing_view(id, offset, bytes, role)?;
        self.leases.insert(view, (model, 0));
        Ok(view)
    }
    pub(super) fn release_lease(&mut self, id: BufferLeaseId) -> Result<()> {
        let (model, bytes) = *self
            .leases
            .get(&id)
            .ok_or_else(|| Error::State("foreign model lease".into()))?;
        self.events.release_lease(id)?;
        self.leases.remove(&id);
        self.models.get_mut(&model).expect("live lease model").bytes -= bytes;
        Ok(())
    }
    pub(super) fn release_backing(
        &mut self,
        id: zerogravity_core::device_events::DeviceBackingId,
    ) -> Result<()> {
        let (model, bytes) = *self
            .backings
            .get(&id)
            .ok_or_else(|| Error::State("foreign model backing".into()))?;
        self.events.release_backing(id)?;
        self.backings.remove(&id);
        self.models
            .get_mut(&model)
            .expect("live backing model")
            .bytes -= bytes;
        Ok(())
    }
    pub(super) fn submit(
        &mut self,
        root: u64,
        accesses: &[(BufferLeaseId, DeviceAccess)],
    ) -> Result<DeviceEventId> {
        let first = accesses
            .first()
            .ok_or_else(|| Error::State("empty model event".into()))?;
        let model = self
            .leases
            .get(&first.0)
            .ok_or_else(|| Error::State("foreign model event lease".into()))?
            .0;
        if accesses
            .iter()
            .any(|(id, _)| self.leases.get(id).is_none_or(|&(m, _)| m != model))
        {
            return Err(Error::State("event mixes source/model namespaces".into()));
        }
        let id = self.events.submit(root, accesses)?;
        self.event_models.insert(id, model);
        Ok(id)
    }
    pub(super) fn report_completion_unknown(&mut self, id: DeviceEventId) -> Result<()> {
        self.events.report_completion_unknown(id)
    }
    pub(super) fn report_physical_ready(&mut self, id: DeviceEventId) -> Result<()> {
        self.events.report_physical_ready(id)?;
        if self.events.phase(id).is_err() {
            self.event_models.remove(&id);
        }
        Ok(())
    }
    pub(super) fn retire_physical_ready(&mut self, id: DeviceEventId) -> Result<()> {
        self.events.retire_physical_ready(id)?;
        self.event_models.remove(&id);
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn request_cancel(&mut self, id: DeviceEventId) -> Result<()> {
        self.events.request_cancel(id)?;
        if self.events.phase(id).is_err() {
            self.event_models.remove(&id);
        }
        Ok(())
    }
    pub(super) fn resident_bytes(&self) -> usize {
        self.events.resident_bytes() + self.foreign_bytes
    }
    #[cfg(any(
        test,
        feature = "cuda-native",
        all(
            target_os = "linux",
            any(feature = "hip", feature = "vulkan"),
            any(
                feature = "gemma-bf16-qualification",
                feature = "qwen-source-qualification",
                all(test, feature = "qualification", feature = "hip")
            )
        )
    ))]
    fn reserve_foreign_for(&mut self, model: u64, bytes: usize) -> Result<()> {
        self.check_request(model, bytes)?;
        self.foreign_bytes += bytes;
        self.models.get_mut(&model).expect("checked model").bytes += bytes;
        Ok(())
    }
    #[cfg(any(
        test,
        feature = "cuda-native",
        all(
            target_os = "linux",
            any(feature = "hip", feature = "vulkan"),
            any(
                feature = "gemma-bf16-qualification",
                feature = "qwen-source-qualification",
                all(test, feature = "qualification", feature = "hip")
            )
        )
    ))]
    fn release_foreign_for(&mut self, model: u64, bytes: usize) -> Result<()> {
        let used = self
            .models
            .get_mut(&model)
            .ok_or_else(|| Error::State("native model namespace closed".into()))?;
        if bytes == 0 || used.bytes < bytes || self.foreign_bytes < bytes {
            return Err(Error::State("native model accounting underflow".into()));
        }
        self.foreign_bytes -= bytes;
        used.bytes -= bytes;
        Ok(())
    }
}

impl Deref for ResidentLedger {
    type Target = DeviceEventLedger;
    fn deref(&self) -> &Self::Target {
        &self.events
    }
}
// The host-only owner exercises this SAME charge implementation without a
// fabricated device/client/handle. Production retains its detached original
// global envelope and SAME ledger, without a Client/actor reference.
#[cfg(any(
    test,
    feature = "cuda-native",
    all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    )
))]
pub(crate) trait ForeignProviderOwner: Send + Sync {
    fn foreign_ledger(&self) -> &Mutex<ResidentLedger>;
    fn foreign_model(&self) -> u64 {
        0
    }
    fn foreign_release_failed(self: &Arc<Self>);
}
#[cfg(any(
    test,
    feature = "cuda-native",
    all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    )
))]
pub(crate) struct ForeignProviderCharge<
    O: ForeignProviderOwner = super::native_cache::NativeCacheBudget,
> {
    bytes: usize,
    owner: Arc<O>,
}
#[cfg(any(
    test,
    feature = "cuda-native",
    all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    )
))]
impl<O: ForeignProviderOwner> ForeignProviderCharge<O> {
    pub(super) fn reserve(owner: Arc<O>, bytes: usize) -> Result<Arc<Self>> {
        owner
            .foreign_ledger()
            .lock()
            .map_err(|_| Error::State("opaque provider ledger poisoned".into()))?
            .reserve_foreign_for(owner.foreign_model(), bytes)?;
        Ok(Arc::new(Self { bytes, owner }))
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}
#[cfg(feature = "cuda-native")]
impl ForeignProviderCharge<super::native_cache::NativeCacheBudget> {
    pub(crate) fn require_service(&self, service: &Arc<ResidentService>) -> Result<()> {
        self.owner.require_service(service)
    }
}
#[cfg(any(
    test,
    feature = "cuda-native",
    all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    )
))]
impl<O: ForeignProviderOwner> Drop for ForeignProviderCharge<O> {
    fn drop(&mut self) {
        let released = self
            .owner
            .foreign_ledger()
            .lock()
            .ok()
            .is_some_and(|mut ledger| {
                ledger
                    .release_foreign_for(self.owner.foreign_model(), self.bytes)
                    .is_ok()
            });
        if !released {
            self.owner.foreign_release_failed();
        }
    }
}
#[cfg(feature = "cuda-native")]
impl ResidentService {
    /// Reserve BEFORE a foreign allocator runs. Retain through actual provider
    /// destruction AND pool retirement; unknown completion must quarantine it.
    pub(crate) fn reserve_foreign_provider(
        self: &Arc<Self>,
        bytes: usize,
    ) -> Result<Arc<ForeignProviderCharge>> {
        self.ensure_usable()?;
        self.native_cache_budget()?.reserve(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Barrier, atomic::AtomicUsize};
    use zerogravity_core::resource_arbiter::{
        AllocationClass, AllocationRequest, GlobalResourceLease, ResourceArbiter,
    };

    struct HostOwner {
        ledger: Mutex<ResidentLedger>,
        release_failures: AtomicUsize,
        _envelope: GlobalResourceLease,
    }
    impl ForeignProviderOwner for HostOwner {
        fn foreign_ledger(&self) -> &Mutex<ResidentLedger> {
            &self.ledger
        }
        fn foreign_release_failed(self: &Arc<Self>) {
            self.release_failures.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn fixture(cap: usize) -> (Arc<HostOwner>, ResourceArbiter, PhysicalDeviceKey) {
        let arbiter = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("host", "opaque-provider-test").unwrap();
        arbiter.register(key.clone(), cap as u64).unwrap();
        let envelope = arbiter
            .admit(
                &key,
                ResourceOwner::new("model", "original-service").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: cap as u64,
                }],
            )
            .unwrap();
        (
            Arc::new(HostOwner {
                ledger: Mutex::new(ResidentLedger::new(1, cap).unwrap()),
                release_failures: AtomicUsize::new(0),
                _envelope: envelope,
            }),
            arbiter,
            key,
        )
    }

    #[test]
    fn opaque_and_managed_allocations_share_cap_and_one_original_envelope() {
        let (owner, arbiter, key) = fixture(100);
        let buffer = owner
            .ledger
            .lock()
            .unwrap()
            .admit(40, BufferRole::Output)
            .unwrap();
        let charge = ForeignProviderCharge::reserve(owner.clone(), 60).unwrap();
        assert_eq!(charge.bytes(), 60);
        assert_eq!(owner.ledger.lock().unwrap().resident_bytes(), 100);
        assert!(
            owner
                .ledger
                .lock()
                .unwrap()
                .admit(1, BufferRole::Scratch)
                .is_err()
        );
        assert!(ForeignProviderCharge::reserve(owner.clone(), 1).is_err());
        let snapshot = arbiter.snapshot(&key).unwrap();
        assert_eq!(snapshot.reserved, 100);
        assert_eq!(snapshot.per_class.len(), 1);
        assert_eq!(
            snapshot.per_class.get(&AllocationClass::ProviderEnvelope),
            Some(&100)
        );
        drop(charge);
        assert_eq!(owner.ledger.lock().unwrap().resident_bytes(), 40);
        let next = owner
            .ledger
            .lock()
            .unwrap()
            .admit(60, BufferRole::Activation)
            .unwrap();
        owner.ledger.lock().unwrap().release_lease(next).unwrap();
        owner.ledger.lock().unwrap().release_lease(buffer).unwrap();
        assert_eq!(owner.ledger.lock().unwrap().resident_bytes(), 0);
        drop(owner);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }

    #[test]
    fn invalid_requests_and_failed_allocation_release_the_reservation() {
        let (owner, _, _) = fixture(128);
        for bytes in [0, 129, usize::MAX] {
            assert!(ForeignProviderCharge::reserve(owner.clone(), bytes).is_err());
        }
        assert_eq!(owner.ledger.lock().unwrap().resident_bytes(), 0);
        let allocator_calls = AtomicUsize::new(0);
        let allocation = || -> Result<()> {
            let _charge = ForeignProviderCharge::reserve(owner.clone(), 128)?;
            allocator_calls.fetch_add(1, Ordering::SeqCst);
            Err(Error::State("bounded allocator refusal".into()))
        };
        assert!(allocation().is_err());
        assert_eq!(allocator_calls.load(Ordering::SeqCst), 1);
        assert_eq!(owner.ledger.lock().unwrap().resident_bytes(), 0);
        let charge = ForeignProviderCharge::reserve(owner.clone(), 128).unwrap();
        assert!(allocation().is_err());
        assert_eq!(
            allocator_calls.load(Ordering::SeqCst),
            1,
            "pressure refuses before allocator"
        );
        drop(charge);
        assert_eq!(owner.release_failures.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn last_arc_and_type_erased_completion_owner_keep_original_envelope_alive() {
        let (owner, arbiter, key) = fixture(128);
        let weak = Arc::downgrade(&owner);
        let charge = ForeignProviderCharge::reserve(owner.clone(), 96).unwrap();
        let cloned = charge.clone();
        let completion_owner: crate::physical_quarantine::CompletionOwner = charge;
        let retained_unknown = vec![completion_owner];
        drop(owner);
        drop(cloned);
        assert!(weak.upgrade().is_some());
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 128);
        assert_eq!(
            weak.upgrade()
                .unwrap()
                .ledger
                .lock()
                .unwrap()
                .resident_bytes(),
            96
        );
        // Only this host fixture is cleared; real uncertain owners stay quarantined.
        drop(retained_unknown);
        assert!(weak.upgrade().is_none());
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }

    #[test]
    fn managed_and_foreign_threads_cannot_both_spend_the_same_capacity() {
        let (owner, _, _) = fixture(100);
        let start = Arc::new(Barrier::new(3));
        let held = Arc::new(Barrier::new(3));
        let successes = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            {
                let (owner, start, held, successes) = (
                    owner.clone(),
                    start.clone(),
                    held.clone(),
                    successes.clone(),
                );
                scope.spawn(move || {
                    start.wait();
                    let charge = ForeignProviderCharge::reserve(owner, 60).ok();
                    if charge.is_some() {
                        successes.fetch_add(1, Ordering::SeqCst);
                    }
                    held.wait();
                    held.wait();
                    drop(charge);
                });
            }
            {
                let (owner, start, held, successes) = (
                    owner.clone(),
                    start.clone(),
                    held.clone(),
                    successes.clone(),
                );
                scope.spawn(move || {
                    start.wait();
                    let buffer = owner
                        .ledger
                        .lock()
                        .unwrap()
                        .admit(60, BufferRole::Output)
                        .ok();
                    if buffer.is_some() {
                        successes.fetch_add(1, Ordering::SeqCst);
                    }
                    held.wait();
                    held.wait();
                    if let Some(buffer) = buffer {
                        owner.ledger.lock().unwrap().release_lease(buffer).unwrap();
                    }
                });
            }
            start.wait();
            held.wait();
            let admitted = successes.load(Ordering::SeqCst);
            let resident = owner.ledger.lock().unwrap().resident_bytes();
            held.wait();
            assert_eq!(admitted, 1);
            assert_eq!(resident, 60);
        });
        assert_eq!(owner.ledger.lock().unwrap().resident_bytes(), 0);
    }

    #[test]
    fn arithmetic_overflow_refuses_before_any_admission_change() {
        let mut ledger = ResidentLedger::new(1, usize::MAX).unwrap();
        ledger.reserve_foreign_for(0, usize::MAX).unwrap();
        assert!(ledger.reserve_foreign_for(0, 1).is_err());
        assert!(ledger.admit(1, BufferRole::Output).is_err());
        assert_eq!(ledger.resident_bytes(), usize::MAX);
        ledger.release_foreign_for(0, usize::MAX).unwrap();
        assert!(ledger.release_foreign_for(0, 1).is_err());
        assert_eq!(ledger.resident_bytes(), 0);
    }

    #[test]
    fn poisoned_release_preserves_charge_and_invokes_owner_fencing() {
        let (owner, _, _) = fixture(128);
        let charge = ForeignProviderCharge::reserve(owner.clone(), 96).unwrap();
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ledger = owner.ledger.lock().unwrap();
            panic!("source-free ledger poison");
        }));
        assert!(poisoned.is_err());
        drop(charge);
        assert_eq!(owner.release_failures.load(Ordering::SeqCst), 1);
        let poisoned = owner.ledger.lock().err().unwrap();
        assert_eq!(poisoned.into_inner().resident_bytes(), 96);
    }
    #[test]
    fn idle_backing_and_live_views_share_original_foreign_cap_without_double_charge() {
        let mut ledger = ResidentLedger::new(1, 128).unwrap();
        ledger.reserve_foreign_for(0, 32).unwrap();
        let backing = ledger
            .admit_backing_for(0, 96, BufferRole::RecurrentState)
            .unwrap();
        let view = ledger
            .admit_backing_view(backing, 0, 96, BufferRole::RecurrentState)
            .unwrap();
        assert_eq!(ledger.resident_bytes(), 128);
        assert!(ledger.reserve_foreign_for(0, 1).is_err());
        assert!(ledger.admit(1, BufferRole::Output).is_err());
        ledger.release_lease(view).unwrap();
        assert_eq!(ledger.resident_bytes(), 128);
        assert!(ledger.admit_backing_for(0, 1, BufferRole::Scratch).is_err());
        ledger.release_backing(backing).unwrap();
        assert_eq!(ledger.resident_bytes(), 32);
        ledger.release_foreign_for(0, 32).unwrap();
        assert_eq!(ledger.resident_bytes(), 0);
    }
}
