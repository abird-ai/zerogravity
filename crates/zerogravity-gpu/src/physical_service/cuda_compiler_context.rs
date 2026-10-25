//! Source-neutral original host/device reservations for generic CUDA code.
use super::*;
use cubecl_common::device::ServiceId;

pub(crate) const CUDA_COMPILER_HOST_BYTES: u64 =
    cubecl::cuda::CudaCompilerContextPolicy::HOST_BYTES;
pub(crate) const CUDA_COMPILER_DEVICE_BYTES: u64 =
    cubecl::cuda::CudaCompilerContextPolicy::DEVICE_BYTES;
pub(crate) const CUDA_COMPILER_POLICY: &str = "cuda-compiler-context-v1";

fn backend_host_fits(limit: u64) -> bool {
    CUDA_COMPILER_HOST_BYTES <= limit
}

/// No Client/physical service/model/source/cache backedge. The CUDA actor owns
/// this after installation; original process exit is its destruction boundary.
pub(crate) struct CudaCompilerContextBudget {
    host: Arc<GlobalResourceLease>,
    device: Arc<GlobalResourceLease>,
    host_limit: u64,
    physical: PhysicalDeviceKey,
    service: ServiceId,
    fence: Arc<AtomicBool>,
    adopted: Arc<AtomicBool>,
}
impl CudaCompilerContextBudget {
    pub(crate) fn admit(
        original: &PhysicalServiceBudget,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    ) -> Result<Arc<Self>> {
        if original.total_limit != 5120 * 1024 * 1024
            || original
                .provider_cap
                .checked_add(CUDA_COMPILER_DEVICE_BYTES)
                .is_none_or(|n| n > original.total_limit)
            || original.fence_flag.load(Ordering::Acquire)
        {
            return Err(Error::Limit(
                "CUDA model+backend context exceeds original5GiB physical cap".into(),
            ));
        }
        // Capture the actual finite source-issued workspace limit. No generic
        // 4/8 GiB whitelist: the source domain decides its own finite cap, and
        // the detached 384MiB backend reservation must fit inside it.
        let host_limit = source.workspace_limit_bytes()?;
        if !backend_host_fits(host_limit) {
            return Err(Error::Limit(
                "CUDA compiler HOST context exceeds the captured source workspace limit".into(),
            ));
        }
        let host = source.reserve_detached_host_workspace(
            host_limit,
            ResourceOwner::new("backend/cuda", CUDA_COMPILER_POLICY)?,
            CUDA_COMPILER_HOST_BYTES,
        )?;
        let device = ResourceArbiter::process_shared().admit(
            &original.key,
            ResourceOwner::new("backend/cuda", CUDA_COMPILER_POLICY)?,
            &[AllocationRequest {
                kind: AllocationClass::ProviderEnvelope,
                bytes: CUDA_COMPILER_DEVICE_BYTES,
            }],
        )?;
        Ok(Arc::new(Self {
            host: Arc::new(host),
            device: Arc::new(device),
            host_limit,
            physical: original.key.clone(),
            service: original.client.service_id(),
            fence: original.fence_flag.clone(),
            adopted: Arc::new(AtomicBool::new(false)),
        }))
    }
    pub(crate) fn observed_reservations(
        &self,
        original: &PhysicalServiceBudget,
    ) -> Result<(u64, u64)> {
        if self.physical != original.key || self.service != original.client.service_id() {
            return Err(Error::State(
                "CUDA compiler observation belongs to another context".into(),
            ));
        }
        Ok((self.host.reserved_bytes()?, self.device.reserved_bytes()?))
    }
    pub(crate) fn require_host(
        &self,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    ) -> Result<u64> {
        // Same captured source domain and limit that issued this detached
        // reservation. A later source/limit change cannot redirect the charge.
        if !backend_host_fits(self.host_limit) {
            return Err(Error::Limit(
                "CUDA compiler HOST context captured an insufficient source workspace limit".into(),
            ));
        }
        let bytes = source.detached_host_workspace_bytes(&self.host, self.host_limit)?;
        if bytes != CUDA_COMPILER_HOST_BYTES {
            return Err(Error::State(
                "CUDA retained HOST context extent differs".into(),
            ));
        }
        Ok(bytes)
    }
    pub(crate) fn adoption_flag(&self) -> Arc<AtomicBool> {
        self.adopted.clone()
    }
    pub(crate) fn mark_adopted(&self) {
        self.adopted.store(true, Ordering::Release);
    }
    pub(crate) fn require(&self, original: &PhysicalServiceBudget) -> Result<()> {
        if original
            .provider_cap
            .checked_add(CUDA_COMPILER_DEVICE_BYTES)
            .is_none_or(|n| n > original.total_limit)
            || self.physical != original.key
            || self.service != original.client.service_id()
            || original.fence_flag.load(Ordering::Acquire)
            || self.fence.load(Ordering::Acquire)
            || ResourceArbiter::process_shared().is_fenced(&self.physical)?
        {
            return Err(Error::State(
                "CUDA compiler original context budget identity/fence".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn fence_unknown(&self) {
        self.mark_adopted();
        self.host.fence_unknown();
        self.device.fence_unknown();
        self.fence.store(true, Ordering::Release);
    }
}
impl Drop for CudaCompilerContextBudget {
    fn drop(&mut self) {
        // Neither model closure nor cache invalidation proves context death.
        if self.adopted.load(Ordering::Acquire) {
            std::mem::forget(self.host.clone());
            std::mem::forget(self.device.clone());
        }
    }
}

/// Synchronous compiler effects on the original physical owner. No buffer,
/// logical write event, inference flight or empty-stream completion is invented.
pub(crate) struct CudaCompilerPreparation<'a> {
    work: Option<PhysicalWork<'a>>,
    context: Arc<CudaCompilerContextBudget>,
    client: Client,
    owners: Vec<crate::physical_quarantine::CompletionOwner>,
    settled: bool,
}
impl CudaCompilerContextBudget {
    pub(crate) fn begin_preparation<'a>(
        self: &Arc<Self>,
        original: &'a PhysicalServiceBudget,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
        owners: Vec<crate::physical_quarantine::CompletionOwner>,
    ) -> Result<CudaCompilerPreparation<'a>> {
        self.require(original)?;
        self.require_host(source)?;
        if owners.is_empty() || owners.len() > 8 {
            return Err(Error::Limit("CUDA compiler original owner count".into()));
        }
        let work = original.begin(&original.client)?;
        Ok(CudaCompilerPreparation {
            work: Some(work),
            context: self.clone(),
            client: original.client.clone(),
            owners,
            settled: false,
        })
    }
}
impl CudaCompilerPreparation<'_> {
    /// Every synchronous actor compile/load/function lookup returned Result::Ok
    /// and NVRTC program destruction settled. Persistent code stays admitted.
    pub(crate) fn complete(mut self) -> Result<()> {
        self.work
            .take()
            .ok_or_else(|| Error::State("CUDA compiler preparation already settled".into()))?
            .complete()?;
        self.settled = true;
        Ok(())
    }
}
impl Drop for CudaCompilerPreparation<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.context.fence_unknown();
            crate::physical_quarantine::quarantine_uncertain_with_owners(
                self.client.clone(),
                Vec::new(),
                std::mem::take(&mut self.owners),
            );
            // Armed original PhysicalWork independently fences its envelope.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (
        ResourceArbiter,
        PhysicalDeviceKey,
        PhysicalDeviceKey,
        Arc<CudaCompilerContextBudget>,
    ) {
        let arbiter = ResourceArbiter::new();
        let host = PhysicalDeviceKey::new("host", "compiler-context-lifetime").unwrap();
        let device = PhysicalDeviceKey::new("test", "compiler-context-lifetime").unwrap();
        for key in [&host, &device] {
            arbiter.register(key.clone(), 64).unwrap();
        }
        let lease = |key, kind| {
            Arc::new(
                arbiter
                    .admit(
                        key,
                        ResourceOwner::new("backend/cuda", "test").unwrap(),
                        &[AllocationRequest { kind, bytes: 32 }],
                    )
                    .unwrap(),
            )
        };
        let owner = Arc::new(CudaCompilerContextBudget {
            host: lease(&host, AllocationClass::Scratch),
            device: lease(&device, AllocationClass::ProviderEnvelope),
            host_limit: 4096 * 1024 * 1024,
            physical: device.clone(),
            service: ServiceId::of::<SelfContainedTest>(cubecl_common::device::DeviceId::new(0, 0)),
            fence: Arc::new(AtomicBool::new(false)),
            adopted: Arc::new(AtomicBool::new(false)),
        });
        (arbiter, host, device, owner)
    }
    struct SelfContainedTest;
    #[test]
    fn generic_backend_captures_an_actual_finite_source_limit_not_a_4or8_whitelist() {
        assert_eq!(CUDA_COMPILER_HOST_BYTES, 384 * 1024 * 1024);
        assert_eq!(CUDA_COMPILER_DEVICE_BYTES, 512 * 1024 * 1024);
        assert!(backend_host_fits(384 * 1024 * 1024));
        assert!(backend_host_fits(512 * 1024 * 1024));
        assert!(backend_host_fits(4096 * 1024 * 1024));
        assert!(backend_host_fits(5120 * 1024 * 1024));
        assert!(!backend_host_fits(384 * 1024 * 1024 - 1));
    }
    #[test]
    fn compiler_context_known_preinstall_refusal_releases_both_reservations() {
        let (arbiter, host, device, owner) = fixture();
        drop(owner);
        for key in [&host, &device] {
            let s = arbiter.snapshot(key).unwrap();
            assert_eq!(s.reserved, 0);
            assert!(!s.fenced);
        }
    }
    #[test]
    fn compiler_context_actor_adoption_retains_only_detached_charges_after_owner_drop() {
        let (arbiter, host, device, owner) = fixture();
        let weak = Arc::downgrade(&owner);
        let actor_flag = owner.adoption_flag();
        actor_flag.store(true, Ordering::Release);
        drop(owner);
        assert!(weak.upgrade().is_none()); // no leaked model/service/source owner
        for key in [&host, &device] {
            let s = arbiter.snapshot(key).unwrap();
            assert_eq!(s.reserved, 32);
            assert!(!s.fenced);
        }
    }
    #[test]
    fn compiler_context_unknown_marks_adopted_and_fences_both_original_domains() {
        let (arbiter, host, device, owner) = fixture();
        owner.fence_unknown();
        drop(owner);
        for key in [&host, &device] {
            let s = arbiter.snapshot(key).unwrap();
            assert_eq!(s.reserved, 32);
            assert!(s.fenced);
        }
    }
}
