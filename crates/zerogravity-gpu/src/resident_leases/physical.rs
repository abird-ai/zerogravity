//! One original Client/envelope/queue; source admissions retain no actor backedge.
use super::*;
use zerogravity_core::resource_arbiter::{GlobalResourceLease, ResourceArbiter};

pub(crate) enum ResidentSource {
    Tensor(Arc<dyn zerogravity_core::source::TensorSource>),
    #[cfg(all(
        any(
            feature = "cuda-cublas",
            feature = "cuda-native-qualification",
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification"
        ),
        target_os = "linux"
    ))]
    Windows(zerogravity_artifacts::source_weights::VerifiedTensorSource),
}
impl ResidentSource {
    fn identity(&self) -> SourceIdentity {
        match self {
            Self::Tensor(source) => source.source_identity(),
            #[cfg(all(
                any(
                    feature = "cuda-cublas",
                    feature = "cuda-native-qualification",
                    feature = "gemma-bf16-qualification",
                    feature = "qwen-source-qualification"
                ),
                target_os = "linux"
            ))]
            Self::Windows(source) => source.source_identity(),
        }
    }
}
/// Physical identity and namespace generation differ from each model binding.
pub struct ResidentPhysical {
    device: Device,
    pub(super) client: Client,
    #[cfg(feature = "qualification")]
    pub(super) provider: crate::backend::Provider,
    #[cfg(all(
        feature = "cuda",
        any(feature = "qualification", feature = "cuda-cublas")
    ))]
    pub(super) pinned_cuda_queue: Option<Arc<cubecl::cuda::OwnedCudaQueue>>,
    pub(super) budget: PhysicalServiceBudget,
    pub(super) queue: Arc<PhysicalQueue>,
    pub(super) ledger: Arc<Mutex<ResidentLedger>>,
    pub(super) next_allocation: AtomicU64,
    pub(super) poisoned: AtomicBool,
    generation: u64,
    #[cfg(all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    ))]
    _native_context: Option<Arc<crate::physical_service::NativeContextBudget>>,
}
/// Detached quota/source owner: no Client, physical owner, service or cache Arc.
pub(crate) struct ModelAdmission {
    pub(super) ledger: Arc<Mutex<ResidentLedger>>,
    envelope: Arc<GlobalResourceLease>,
    arbiter: ResourceArbiter,
    physical: PhysicalDeviceKey,
    fence: Arc<AtomicBool>,
    pub(super) namespace: u64,
    source: SourceIdentity,
    owner: ResourceOwner,
    cap: usize,
    closed: Arc<AtomicBool>,
    _source: Option<ResidentSource>,
}
impl ModelAdmission {
    fn register(self) -> Result<Arc<Self>> {
        self.arbiter.with_unfenced(&self.physical, || {
            if self.fence.load(Ordering::Acquire) {
                return Err(Error::State(
                    "physical owner fenced before model attach".into(),
                ));
            }
            self.ledger
                .lock()
                .map_err(|_| Error::State("physical resident ledger poisoned".into()))?
                .attach_model(self.namespace, self.cap)
        })?;
        self.closed.store(false, Ordering::Release);
        Ok(Arc::new(self))
    }
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn test_new(
        ledger: Arc<Mutex<ResidentLedger>>,
        envelope: Arc<GlobalResourceLease>,
        arbiter: ResourceArbiter,
        physical: PhysicalDeviceKey,
        fence: Arc<AtomicBool>,
        namespace: u64,
        source: SourceIdentity,
        cap: usize,
        retention: Option<ResidentSource>,
    ) -> Result<Arc<Self>> {
        Self {
            ledger,
            envelope,
            arbiter,
            physical,
            fence,
            namespace,
            source,
            owner: ResourceOwner::new("source", "model")?,
            cap,
            closed: Arc::new(AtomicBool::new(true)),
            _source: retention,
        }
        .register()
    }
    pub(super) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub(super) fn ensure_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) || self.fence.load(Ordering::Acquire) {
            return Err(Error::State(
                "source binding closed or physical owner fenced".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn require_binding(
        &self,
        source: SourceIdentity,
        owner: &ResourceOwner,
        namespace: u64,
        cap: usize,
    ) -> Result<()> {
        self.ensure_open()?;
        if source != self.source
            || owner != &self.owner
            || namespace != self.namespace
            || cap != self.cap
        {
            return Err(Error::State(
                "source admission identity/namespace/cap mismatch".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn close(&self) -> Result<()> {
        self.arbiter.with_unfenced(&self.physical, || {
            self.ensure_open()?;
            self.ledger
                .lock()
                .map_err(|_| Error::State("physical resident ledger poisoned".into()))?
                .close_model(self.namespace)?;
            self.closed.store(true, Ordering::Release);
            Ok(())
        })
    }
}
impl Drop for ModelAdmission {
    fn drop(&mut self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        // Release local quota only after all actual lease/module readers retired.
        let safe = self.close().is_ok();
        if !safe {
            self.envelope.fence_unknown();
            self.fence.store(true, Ordering::Release);
            std::mem::forget(self.envelope.clone());
            std::mem::forget(self.ledger.clone());
            if let Some(source) = self._source.take() {
                std::mem::forget(source);
            }
        }
    }
}
impl ResidentPhysical {
    pub fn new(device: Device, provider_cap: usize, device_cap: u64) -> Result<Arc<Self>> {
        crate::backend::validate_provider(&device)?;
        let provider = crate::backend::Provider::selected(&device)?;
        let client = device.client();
        Self::from_client(device, client, provider_cap, device_cap, provider).map(Arc::new)
    }
    fn from_client(
        device: Device,
        client: Client,
        provider_cap: usize,
        device_cap: u64,
        provider: crate::backend::Provider,
    ) -> Result<Self> {
        #[cfg(not(feature = "qualification"))]
        let _ = provider;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let budget = PhysicalServiceBudget::attach(
            &client,
            device_cap,
            provider_cap as u64,
            ResourceOwner::new("backend/physical", "original-owner")?,
        )?;
        Self::from_budget(device, client, provider_cap, provider, budget)
    }
    #[cfg(all(
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification"
        ),
        feature = "vulkan",
        target_os = "linux"
    ))]
    fn from_client_preadmitted(
        device: Device,
        provider_cap: usize,
        device_cap: u64,
        pre: &crate::physical_service::PreAdmittedDevice,
    ) -> Result<Self> {
        crate::backend::validate_provider(&device)?;
        let provider = crate::backend::Provider::selected(&device)?;
        let client = device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let budget = PhysicalServiceBudget::attach_preadmitted(
            &client,
            pre,
            device_cap,
            provider_cap as u64,
            ResourceOwner::new("backend/physical", "original-owner")?,
        )?;
        Self::from_budget(device, client, provider_cap, provider, budget)
    }
    fn from_budget(
        device: Device,
        client: Client,
        provider_cap: usize,
        provider: crate::backend::Provider,
        budget: PhysicalServiceBudget,
    ) -> Result<Self> {
        let _ = provider_cap;
        #[cfg(not(feature = "qualification"))]
        let _ = provider;
        #[cfg(not(any(feature = "qualification", feature = "cuda-cublas")))]
        let _ = provider;
        let generation = issue(&NEXT_SERVICE_GENERATION)?;
        let queue = queue_for(budget.physical_key())?;
        Ok(Self {
            device,
            client,
            #[cfg(feature = "qualification")]
            provider,
            #[cfg(all(
                feature = "cuda",
                any(feature = "qualification", feature = "cuda-cublas")
            ))]
            pinned_cuda_queue: None,
            ledger: Arc::new(Mutex::new(ResidentLedger::physical(
                generation,
                usize::try_from(budget.provider_capacity())
                    .map_err(|_| Error::Limit("physical provider capacity overflow".into()))?,
            )?)),
            budget,
            queue,
            next_allocation: AtomicU64::new(1),
            poisoned: AtomicBool::new(false),
            generation,
            #[cfg(all(
                target_os = "linux",
                any(feature = "hip", feature = "vulkan"),
                any(
                    feature = "gemma-bf16-qualification",
                    feature = "qwen-source-qualification",
                    all(test, feature = "qualification", feature = "hip")
                )
            ))]
            _native_context: None,
        })
    }
    #[cfg(all(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification",
        feature = "hip",
        target_os = "linux"
    ))]
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }
    #[cfg(all(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification",
        feature = "hip",
        target_os = "linux"
    ))]
    pub(crate) fn physical_key(&self) -> PhysicalDeviceKey {
        self.budget.physical_key()
    }
    pub(crate) fn device(&self) -> Device {
        self.device.clone()
    }
    pub(crate) fn attach_model(
        self: &Arc<Self>,
        source: ResidentSource,
        owner: ResourceOwner,
        cap: usize,
    ) -> Result<Arc<ResidentService>> {
        self.attach_inner(source.identity(), owner, cap, Some(source))
            .map(Arc::new)
    }
    #[cfg(all(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification",
        feature = "hip",
        target_os = "linux"
    ))]
    pub(crate) fn attach_native_model(
        self: &Arc<Self>,
        source: ResidentSource,
        owner: ResourceOwner,
        cap: usize,
        registered: &cubecl::hip::RegisteredHipNative,
    ) -> Result<Arc<ResidentService>> {
        if registered.client().service_id() != self.client.service_id() {
            return Err(Error::State(
                "HIP scoped source view belongs to another original actor".into(),
            ));
        }
        let mut service = self.attach_inner(source.identity(), owner, cap, Some(source))?;
        service.client = registered.source_client();
        Ok(Arc::new(service))
    }
    pub(super) fn attach_inner(
        self: &Arc<Self>,
        source: SourceIdentity,
        owner: ResourceOwner,
        cap: usize,
        retention: Option<ResidentSource>,
    ) -> Result<ResidentService> {
        ResidentService::validate_owner(source, &owner)?;
        if self.poisoned.load(Ordering::Acquire) || self.budget.fence_flag().load(Ordering::Acquire)
        {
            return Err(Error::State(
                "physical owner fenced before source binding".into(),
            ));
        }
        let namespace = issue(&NEXT_SERVICE_GENERATION)?;
        let admission = ModelAdmission {
            ledger: self.ledger.clone(),
            envelope: self.budget.retained_envelope(),
            arbiter: ResourceArbiter::process_shared(),
            physical: self.budget.physical_key(),
            fence: self.budget.fence_flag(),
            namespace,
            source,
            owner: owner.clone(),
            cap,
            closed: Arc::new(AtomicBool::new(true)),
            _source: retention,
        }
        .register()?;
        let budget =
            self.budget
                .source_binding(owner.clone(), cap as u64, admission.closed.clone())?;
        Ok(ResidentService {
            physical: self.clone(),
            client: Arc::new(self.client.clone()),
            budget,
            source,
            owner,
            generation: namespace,
            admission,
            uploaded_value_bytes: AtomicU64::new(0),
            cap,
            retained_forever: AtomicBool::new(false),
        })
    }
    pub fn available_model_capacity(&self) -> Result<usize> {
        self.ledger
            .lock()
            .map_err(|_| Error::State("physical resident ledger poisoned".into()))?
            .available_model_capacity()
    }
    pub fn resident_bytes(&self) -> Result<usize> {
        self.ledger
            .lock()
            .map(|l| l.resident_bytes())
            .map_err(|_| Error::State("physical resident ledger poisoned".into()))
    }
    pub fn generation(&self) -> u64 {
        self.generation
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
    pub(crate) fn new_with_native_context(
        device: Device,
        provider_cap: usize,
        device_cap: u64,
        policy: Arc<str>,
        context_bytes: u64,
    ) -> Result<(Arc<Self>, Arc<crate::physical_service::NativeContextBudget>)> {
        let backend = match &device {
            Device::Hip(_) => "backend/hip",
            #[cfg(feature = "vulkan")]
            Device::Wgpu(wgpu) if wgpu.backend == cubecl::device::WgpuBackend::Vulkan => {
                "backend/vulkan"
            }
            _ => {
                return Err(Error::Limit(
                    "native physical context policy/device/extent before bootstrap".into(),
                ));
            }
        };
        if policy.is_empty() || context_bytes == 0 || context_bytes >= provider_cap as u64 {
            return Err(Error::Limit(
                "native physical context policy/device/extent before bootstrap".into(),
            ));
        }
        crate::backend::validate_provider(&device)?;
        #[cfg(all(
            any(
                feature = "gemma-bf16-qualification",
                feature = "qwen-source-qualification"
            ),
            feature = "vulkan",
            target_os = "linux"
        ))]
        let mut physical = {
            if matches!(&device, Device::Wgpu(wgpu) if wgpu.backend == cubecl::device::WgpuBackend::Vulkan)
            {
                let preflight = crate::vulkan::native::preflight_vulkan_device(&device)?;
                let pre = PhysicalServiceBudget::preadmit_device(
                    preflight.key().clone(),
                    preflight.max_memory(),
                    device_cap,
                    provider_cap as u64,
                    ResourceOwner::new("backend/physical", "original-owner")?,
                )?;
                NATIVE_CONTEXT_CONSTRUCTED
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .map_err(|_| {
                        let name = if backend == "backend/hip" {
                            "HIP"
                        } else {
                            "Vulkan"
                        };
                        Error::State(format!(
                            "{name} physical context constructor already attempted"
                        ))
                    })?;
                if let Err(error) = preflight.register(&device) {
                    pre.fence_and_retain();
                    return Err(error);
                }
                match Self::from_client_preadmitted(device, provider_cap, device_cap, &pre) {
                    Ok(physical) => Arc::new(physical),
                    Err(error) => {
                        pre.fence_and_retain();
                        return Err(error);
                    }
                }
            } else {
                NATIVE_CONTEXT_CONSTRUCTED
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .map_err(|_| {
                        let name = if backend == "backend/hip" {
                            "HIP"
                        } else {
                            "Vulkan"
                        };
                        Error::State(format!(
                            "{name} physical context constructor already attempted"
                        ))
                    })?;
                Self::new(device, provider_cap, device_cap)?
            }
        };
        #[cfg(not(all(
            any(
                feature = "gemma-bf16-qualification",
                feature = "qwen-source-qualification"
            ),
            feature = "vulkan",
            target_os = "linux"
        )))]
        let mut physical = {
            NATIVE_CONTEXT_CONSTRUCTED
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| {
                    let name = if backend == "backend/hip" {
                        "HIP"
                    } else {
                        "Vulkan"
                    };
                    Error::State(format!(
                        "{name} physical context constructor already attempted"
                    ))
                })?;
            Self::new(device, provider_cap, device_cap)?
        };
        let original = Arc::get_mut(&mut physical).expect("cold physical owner exclusive");
        let context = original
            .budget
            .partition_native_context(backend, policy, context_bytes)?;
        original._native_context = Some(context.clone());
        let remainder = usize::try_from(original.budget.provider_capacity())
            .map_err(|_| Error::Limit("physical context remainder overflow".into()))?;
        original.ledger = Arc::new(Mutex::new(ResidentLedger::physical(
            original.generation,
            remainder,
        )?));
        Ok((physical, context))
    }
    #[cfg(all(
        feature = "cuda",
        any(feature = "qualification", feature = "cuda-cublas")
    ))]
    #[allow(dead_code)] // Staged ordinary CUDA constructor; selector remains refused.
    pub(crate) fn new_pinned_cuda(
        device: Device,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<Arc<Self>> {
        if !matches!(device, Device::Cuda(_)) {
            return Err(Error::State("pinned physical owner requires CUDA".into()));
        }
        crate::backend::validate_provider(&device)?;
        let provider = crate::backend::Provider::selected(&device)?;
        // SAFETY: original Client pinned cold before its first own clone/allocation.
        let (client, pinned) =
            unsafe { cubecl::cuda::OwnedCudaQueue::pin_cold(device.client()) }
                .map_err(|e| Error::State(format!("pin original CUDA queue: {e}")))?;
        let mut physical = Self::from_client(device, client, provider_cap, device_cap, provider)?;
        physical.pinned_cuda_queue = Some(Arc::new(pinned));
        Ok(Arc::new(physical))
    }
}

#[cfg(test)]
mod tests {
    use super::super::foreign_provider::{ForeignProviderCharge, ForeignProviderOwner};
    use super::*;
    use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
    use zerogravity_core::source::{SourceTensorRef, TensorSource};

    struct Source(SourceIdentity);
    impl TensorSource for Source {
        fn source_identity(&self) -> SourceIdentity {
            self.0
        }
        fn tensor(&self, _: &str) -> Result<SourceTensorRef<'_>> {
            Err(Error::Unsupported("no tensor in lifetime fixture".into()))
        }
    }
    struct PhysicalFixture {
        arbiter: ResourceArbiter,
        key: PhysicalDeviceKey,
        envelope: Arc<GlobalResourceLease>,
        fence: Arc<AtomicBool>,
        ledger: Arc<Mutex<ResidentLedger>>,
    }
    impl PhysicalFixture {
        fn new() -> Self {
            let arbiter = ResourceArbiter::new();
            let key = PhysicalDeviceKey::new("host", "shared-original-physical").unwrap();
            arbiter.register(key.clone(), 128).unwrap();
            let envelope = Arc::new(
                arbiter
                    .admit(
                        &key,
                        ResourceOwner::new("physical", "original").unwrap(),
                        &[AllocationRequest {
                            kind: AllocationClass::ProviderEnvelope,
                            bytes: 128,
                        }],
                    )
                    .unwrap(),
            );
            Self {
                arbiter,
                key,
                envelope,
                fence: Arc::new(AtomicBool::new(false)),
                ledger: Arc::new(Mutex::new(ResidentLedger::physical(901, 128).unwrap())),
            }
        }
        fn attach(
            &self,
            namespace: u64,
            cap: usize,
            source: Option<Arc<Source>>,
        ) -> Result<Arc<ModelAdmission>> {
            let identity = source
                .as_ref()
                .map_or_else(|| SourceIdentity::for_new_allocation().unwrap(), |s| s.0);
            ModelAdmission::test_new(
                self.ledger.clone(),
                self.envelope.clone(),
                self.arbiter.clone(),
                self.key.clone(),
                self.fence.clone(),
                namespace,
                identity,
                cap,
                source.map(|s| ResidentSource::Tensor(s)),
            )
        }
    }
    struct Module(Arc<ModelAdmission>);
    impl ForeignProviderOwner for Module {
        fn foreign_ledger(&self) -> &Mutex<ResidentLedger> {
            &self.0.ledger
        }
        fn foreign_model(&self) -> u64 {
            self.0.namespace
        }
        fn foreign_release_failed(self: &Arc<Self>) {
            self.0.envelope.fence_unknown();
            self.0.fence.store(true, Ordering::Release);
            std::mem::forget(self.clone());
        }
    }
    #[test]
    fn original_context_partition_and_all_models_fit_the_unchanged_physical_cap() {
        let mut physical = PhysicalFixture::new();
        let context = Arc::get_mut(&mut physical.envelope)
            .unwrap()
            .partition_reserved(
                ResourceOwner::new("backend-context", "persistent").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: 32,
                }],
            )
            .unwrap();
        physical.ledger = Arc::new(Mutex::new(ResidentLedger::physical(901, 96).unwrap()));
        let a = physical.attach(51, 64, None).unwrap();
        let b = physical.attach(52, 32, None).unwrap();
        assert!(physical.attach(53, 1, None).is_err());
        assert_eq!(
            physical.arbiter.snapshot(&physical.key).unwrap().reserved,
            128
        );
        a.close().unwrap();
        b.close().unwrap();
        drop(a);
        drop(b);
        assert_eq!(
            physical.arbiter.snapshot(&physical.key).unwrap().reserved,
            128
        );
        let arbiter = physical.arbiter.clone();
        let key = physical.key.clone();
        drop(physical);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 32);
        drop(context);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
    #[test]
    fn concurrent_model_subadmissions_cannot_each_receive_the_original_remainder() {
        let physical = PhysicalFixture::new();
        let barrier = std::sync::Barrier::new(2);
        let admitted = std::thread::scope(|scope| {
            let run = |namespace| {
                let admitted = physical.attach(namespace, 80, None);
                barrier.wait();
                let success = admitted.is_ok();
                drop(admitted);
                success
            };
            let a = scope.spawn(move || run(61));
            let b = scope.spawn(move || run(62));
            usize::from(a.join().unwrap()) + usize::from(b.join().unwrap())
        });
        assert_eq!(admitted, 1);
        assert!(!physical.arbiter.snapshot(&physical.key).unwrap().fenced);
        assert_eq!(
            physical
                .ledger
                .lock()
                .unwrap()
                .available_model_capacity()
                .unwrap(),
            128
        );
    }
    #[test]
    fn model_close_waits_for_managed_foreign_and_event_retirement() {
        let physical = PhysicalFixture::new();
        let model = physical.attach(81, 96, None).unwrap();
        let foreign = ForeignProviderCharge::reserve(Arc::new(Module(model.clone())), 16).unwrap();
        let managed = physical
            .ledger
            .lock()
            .unwrap()
            .admit_for(81, 32, BufferRole::Output)
            .unwrap();
        let event = physical
            .ledger
            .lock()
            .unwrap()
            .submit(0, &[(managed, DeviceAccess::Read)])
            .unwrap();

        assert!(model.close().is_err());
        drop(foreign);
        assert!(model.close().is_err());

        {
            let mut ledger = physical.ledger.lock().unwrap();
            ledger.report_physical_ready(event).unwrap();
            ledger.retire_physical_ready(event).unwrap();
            ledger.release_lease(managed).unwrap();
        }
        model.close().unwrap();
        assert!(model.ensure_open().is_err());
    }

    #[test]
    fn sum_of_model_quotas_and_native_managed_charges_share_one_original_envelope() {
        let physical = PhysicalFixture::new();
        let a = physical.attach(11, 80, None).unwrap();
        assert!(physical.attach(12, 49, None).is_err());
        assert!(!physical.fence.load(Ordering::Acquire));
        assert!(!physical.arbiter.snapshot(&physical.key).unwrap().fenced);
        let b = physical.attach(12, 48, None).unwrap();
        assert!(physical.attach(13, 1, None).is_err());
        let native = ForeignProviderCharge::reserve(Arc::new(Module(a.clone())), 32).unwrap();
        let mut ledger = physical.ledger.lock().unwrap();
        let managed = ledger
            .admit_for(11, 48, BufferRole::ImmutableWeights)
            .unwrap();
        assert!(ledger.admit_for(11, 1, BufferRole::Scratch).is_err());
        let sibling = ledger.admit_for(12, 48, BufferRole::Output).unwrap();
        assert_eq!(ledger.resident_bytes(), 128);
        assert_eq!(
            physical.arbiter.snapshot(&physical.key).unwrap().reserved,
            128
        );
        ledger.release_lease(managed).unwrap();
        ledger.release_lease(sibling).unwrap();
        drop(ledger);
        drop(native);
        a.close().unwrap();
        b.close().unwrap();
        assert_eq!(
            physical
                .ledger
                .lock()
                .unwrap()
                .available_model_capacity()
                .unwrap(),
            128
        );
    }
    #[test]
    fn global_ids_namespace_events_and_independent_close_reload_preserve_readers() {
        let physical = PhysicalFixture::new();
        let a = physical.attach(21, 64, None).unwrap();
        let b = physical.attach(22, 64, None).unwrap();
        let mut ledger = physical.ledger.lock().unwrap();
        let first = ledger.admit_for(21, 32, BufferRole::Output).unwrap();
        let sibling = ledger.admit_for(22, 32, BufferRole::Output).unwrap();
        assert_ne!(first, sibling);
        assert!(
            ledger
                .submit(
                    0,
                    &[(first, DeviceAccess::Read), (sibling, DeviceAccess::Read)]
                )
                .is_err()
        );
        let reader = ledger.submit(0, &[(first, DeviceAccess::Read)]).unwrap();
        let sibling_reader = ledger.submit(0, &[(sibling, DeviceAccess::Read)]).unwrap();
        ledger.request_cancel(sibling_reader).unwrap();
        assert_eq!(ledger.model_events(22), 1);
        drop(ledger);
        assert!(a.close().is_err());
        assert!(b.ensure_open().is_ok());
        let mut ledger = physical.ledger.lock().unwrap();
        assert!(ledger.release_lease(first).is_err());
        ledger.report_physical_ready(reader).unwrap();
        assert!(ledger.release_lease(first).is_err());
        ledger.retire_physical_ready(reader).unwrap();
        ledger.release_lease(first).unwrap();
        drop(ledger);
        a.close().unwrap();
        assert!(a.ensure_open().is_err());
        assert!(b.ensure_open().is_ok());
        let reload = physical.attach(23, 64, None).unwrap();
        let mut ledger = physical.ledger.lock().unwrap();
        let fresh = ledger.admit_for(23, 32, BufferRole::Output).unwrap();
        assert_ne!(fresh, first);
        assert!(ledger.admit_for(21, 1, BufferRole::Scratch).is_err());
        assert_eq!(ledger.model_bytes(22).unwrap(), 32);
        ledger.release_lease(fresh).unwrap();
        assert!(ledger.release_lease(sibling).is_err());
        ledger.report_physical_ready(sibling_reader).unwrap();
        assert_eq!(ledger.model_events(22), 0);
        ledger.release_lease(sibling).unwrap();
        drop(ledger);
        reload.close().unwrap();
        b.close().unwrap();
    }
    #[test]
    fn delayed_event_reader_retains_source_and_quota_through_actual_ledger_retirement() {
        let physical = PhysicalFixture::new();
        let source = Arc::new(Source(SourceIdentity::for_new_allocation().unwrap()));
        let weak = Arc::downgrade(&source);
        let model = physical.attach(71, 64, Some(source.clone())).unwrap();
        drop(source);
        // Production flights retain their source service and this same admission.
        let flight_admission = model.clone();
        drop(model);
        let mut ledger = physical.ledger.lock().unwrap();
        let lease = ledger.admit_for(71, 32, BufferRole::Output).unwrap();
        let event = ledger.submit(0, &[(lease, DeviceAccess::Read)]).unwrap();
        drop(ledger);
        assert!(flight_admission.close().is_err());
        assert!(weak.upgrade().is_some());
        let mut ledger = physical.ledger.lock().unwrap();
        ledger.report_physical_ready(event).unwrap();
        assert!(ledger.release_lease(lease).is_err());
        drop(ledger);
        assert!(flight_admission.close().is_err());
        let mut ledger = physical.ledger.lock().unwrap();
        ledger.retire_physical_ready(event).unwrap();
        ledger.release_lease(lease).unwrap();
        drop(ledger);
        drop(flight_admission);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            physical
                .ledger
                .lock()
                .unwrap()
                .available_model_capacity()
                .unwrap(),
            128
        );
    }
    #[test]
    fn source_owner_and_quota_outlive_source_actor_until_actual_module_retirement() {
        let physical = PhysicalFixture::new();
        let source = Arc::new(Source(SourceIdentity::for_new_allocation().unwrap()));
        let weak = Arc::downgrade(&source);
        let a = physical.attach(31, 64, Some(source.clone())).unwrap();
        drop(source);
        a.require_binding(a.source, &a.owner, 31, 64).unwrap();
        assert!(
            a.require_binding(
                SourceIdentity::for_new_allocation().unwrap(),
                &a.owner,
                31,
                64
            )
            .is_err()
        );
        assert!(
            a.require_binding(
                a.source,
                &ResourceOwner::new("foreign", "model").unwrap(),
                31,
                64
            )
            .is_err()
        );
        assert!(a.require_binding(a.source, &a.owner, 32, 64).is_err());
        assert!(a.require_binding(a.source, &a.owner, 31, 65).is_err());
        let charge = ForeignProviderCharge::reserve(Arc::new(Module(a.clone())), 32).unwrap();
        assert!(a.close().is_err());
        drop(a);
        assert!(weak.upgrade().is_some());
        assert_eq!(
            physical
                .ledger
                .lock()
                .unwrap()
                .available_model_capacity()
                .unwrap(),
            64
        );
        drop(charge);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            physical
                .ledger
                .lock()
                .unwrap()
                .available_model_capacity()
                .unwrap(),
            128
        );
    }
    #[test]
    fn unknown_module_release_fences_siblings_and_retains_original_source_quota() {
        let physical = PhysicalFixture::new();
        let source = Arc::new(Source(SourceIdentity::for_new_allocation().unwrap()));
        let weak = Arc::downgrade(&source);
        let a = physical.attach(41, 64, Some(source.clone())).unwrap();
        let b = physical.attach(42, 64, None).unwrap();
        drop(source);
        let module = Arc::new(Module(a.clone()));
        module.foreign_release_failed();
        drop(module);
        drop(a);
        assert!(b.ensure_open().is_err());
        assert!(b.close().is_err());
        assert!(physical.attach(43, 1, None).is_err());
        assert!(physical.arbiter.snapshot(&physical.key).unwrap().fenced);
        assert_eq!(
            physical.arbiter.snapshot(&physical.key).unwrap().reserved,
            128
        );
        assert!(weak.upgrade().is_some());
        assert_eq!(
            physical
                .ledger
                .lock()
                .unwrap()
                .available_model_capacity()
                .unwrap(),
            0
        );
    }
}
