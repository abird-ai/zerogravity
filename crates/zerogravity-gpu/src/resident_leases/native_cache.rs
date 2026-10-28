//! Native admission retains original charges without retaining its actor.
use super::foreign_provider::{ForeignProviderCharge, ForeignProviderOwner};
use super::*;
use cubecl_common::device::ServiceId;
use zerogravity_core::resource_arbiter::{GlobalResourceLease, ResourceArbiter};

/// No Client/queue/model/cache references: module → charge → this original
/// ledger/envelope is acyclic. Actual source/host owners accompany the module.
pub(crate) struct NativeCacheBudget {
    ledger: Arc<Mutex<ResidentLedger>>,
    envelope: Arc<GlobalResourceLease>,
    arbiter: ResourceArbiter,
    fence: Arc<AtomicBool>,
    physical: PhysicalDeviceKey,
    source: SourceIdentity,
    service: ServiceId,
    generation: u64,
    physical_generation: u64,
    admission: Arc<ModelAdmission>,
}
impl NativeCacheBudget {
    pub(crate) fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Arc<ForeignProviderCharge>> {
        // Same physical publication/fence lock, then SAME local ledger. No
        // second reservation or observer-created allocation authority.
        self.arbiter.with_unfenced(&self.physical, || {
            self.admission.ensure_open()?;
            if self.fence.load(Ordering::Acquire) {
                return Err(Error::State("native original envelope fenced".into()));
            }
            ForeignProviderCharge::reserve(self.clone(), bytes)
        })
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    ))]
    pub(crate) fn require_usable(&self) -> Result<()> {
        self.arbiter.with_unfenced(&self.physical, || {
            self.admission.ensure_open()?;
            if self.fence.load(Ordering::Acquire) {
                Err(Error::State("native original envelope fenced".into()))
            } else {
                Ok(())
            }
        })
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    ))]
    pub(crate) fn fence_unknown(self: &Arc<Self>) {
        self.foreign_release_failed();
    }
    #[cfg(any(
        feature = "cuda-cublas",
        feature = "cuda-native-qualification",
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
    pub(crate) fn require_service(&self, service: &Arc<ResidentService>) -> Result<()> {
        if !Arc::ptr_eq(&self.admission, &service.admission)
            || self.source != service.source
            || self.service != service.client.service_id()
            || self.generation != service.generation
            || self.physical_generation != service.physical.generation()
            || self.physical != service.physical_key()
            || !Arc::ptr_eq(&self.ledger, &service.ledger)
            || !Arc::ptr_eq(&self.envelope, &service.budget.retained_envelope())
            || !Arc::ptr_eq(&self.fence, &service.budget.fence_flag())
        {
            return Err(Error::State(
                "native original charged service/source/generation differs".into(),
            ));
        }
        service.ensure_usable()
    }
}
impl ForeignProviderOwner for NativeCacheBudget {
    fn foreign_ledger(&self) -> &Mutex<ResidentLedger> {
        &self.ledger
    }
    fn foreign_model(&self) -> u64 {
        self.admission.namespace
    }
    fn foreign_release_failed(self: &Arc<Self>) {
        self.envelope.fence_unknown();
        self.fence.store(true, Ordering::Release);
        // Unknown release retains the detached actual proof, not a fresh lease
        // or a Client/actor cycle. Process exit remains the recovery boundary.
        std::mem::forget(self.clone());
    }
}
#[cfg(any(
    feature = "cuda-cublas",
    feature = "cuda-native-qualification",
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
impl ResidentService {
    pub(crate) fn native_cache_budget(self: &Arc<Self>) -> Result<Arc<NativeCacheBudget>> {
        self.ensure_usable()?;
        Ok(Arc::new(NativeCacheBudget {
            ledger: self.ledger.clone(),
            envelope: self.budget.retained_envelope(),
            arbiter: ResourceArbiter::process_shared(),
            fence: self.budget.fence_flag().clone(),
            physical: self.physical_key(),
            source: self.source,
            service: self.client.service_id(),
            generation: self.generation,
            physical_generation: self.physical.generation(),
            admission: self.admission.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::device::DeviceId;
    use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
    fn fixture() -> (Arc<NativeCacheBudget>, ResourceArbiter, PhysicalDeviceKey) {
        let arbiter = ResourceArbiter::new();
        let physical = PhysicalDeviceKey::new("host", "detached-native-budget").unwrap();
        arbiter.register(physical.clone(), 128).unwrap();
        let lease = arbiter
            .admit(
                &physical,
                ResourceOwner::new("source", "original-envelope").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: 128,
                }],
            )
            .unwrap();
        let ledger = Arc::new(Mutex::new(ResidentLedger::physical(901, 128).unwrap()));
        let envelope = Arc::new(lease);
        let fence = Arc::new(AtomicBool::new(false));
        let source = SourceIdentity::for_new_allocation().unwrap();
        let admission = ModelAdmission::test_new(
            ledger.clone(),
            envelope.clone(),
            arbiter.clone(),
            physical.clone(),
            fence.clone(),
            7,
            source,
            128,
            None,
        )
        .unwrap();
        let budget = Arc::new(NativeCacheBudget {
            ledger,
            envelope,
            arbiter: arbiter.clone(),
            fence,
            physical: physical.clone(),
            source,
            service: ServiceId::of::<()>(DeviceId::new(0, 0)),
            generation: 7,
            physical_generation: 901,
            admission,
        });
        assert!(budget.source.is_bound());
        assert_eq!(budget.service, ServiceId::of::<()>(DeviceId::new(0, 0)));
        assert_eq!(budget.generation, 7);
        assert_eq!(budget.physical_generation, 901);
        (budget, arbiter, physical)
    }
    #[test]
    fn module_charge_survives_caller_drop_without_client_or_second_envelope() {
        let (budget, arbiter, key) = fixture();
        let ledger = Arc::downgrade(&budget.ledger);
        let envelope = Arc::downgrade(&budget.envelope);
        let charge = budget.reserve(64).unwrap();
        assert_eq!(budget.ledger.lock().unwrap().resident_bytes(), 64);
        drop(budget);
        assert!(ledger.upgrade().is_some());
        assert!(envelope.upgrade().is_some());
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 128);
        drop(charge);
        assert!(ledger.upgrade().is_none());
        assert!(envelope.upgrade().is_none());
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
    #[test]
    fn same_local_ledger_limits_native_and_managed_charge() {
        let (budget, arbiter, key) = fixture();
        let charge = budget.reserve(64).unwrap();
        assert!(
            budget
                .ledger
                .lock()
                .unwrap()
                .admit_for(7, 65, BufferRole::Scratch)
                .is_err()
        );
        assert!(budget.reserve(65).is_err());
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 128);
        drop(charge);
        assert_eq!(budget.ledger.lock().unwrap().resident_bytes(), 0);
        drop(budget);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
    #[test]
    fn unknown_release_fences_same_envelope_and_retains_detached_proof() {
        let (budget, arbiter, key) = fixture();
        let envelope = Arc::downgrade(&budget.envelope);
        budget.foreign_release_failed();
        assert!(budget.fence.load(Ordering::Acquire));
        assert!(arbiter.snapshot(&key).unwrap().fenced);
        assert!(budget.reserve(1).is_err());
        drop(budget);
        assert!(envelope.upgrade().is_some());
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 128);
    }
}
