//! P07-G2: bind CubeCL services to one reported physical memory domain.
//!
//! CubeCL's ServiceId identifies a *runtime service*, not a physical GPU:
//! CUDA and Vulkan clients may address the same PCI card. We only accept the
//! vendor/OS-reported physical PCI/LUID/registry identity, never adapter name,
//! ordinal, GGUF identity, virtual GTT addressability or a debug TypeId string.
//!
//! Physical capacity is a conservative OPERATOR cap checked against CubeCL's
//! driver-reported maximum. It is not a report of currently free VRAM. Each
//! provider obtains one physical envelope from the process-global arbiter;
//! P07-F subdivides that envelope into cached weights + dynamic/output buffers.
//! These nested local byte checks are NOT a second physical reservation.
//! This conservative coarse admission can strand capacity; fine-grained
//! weight/scratch charges and real fair async workers are separate future work.

use cubecl::{client::Client, ir::PhysicalDevice};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};

/// PCI across CUDA/HIP/Vulkan on Linux, LUID on Windows and registry ID on
/// macOS. Reject anonymous/software devices and ambiguous adapter names.
pub(crate) fn physical_key(physical: Option<&PhysicalDevice>) -> Result<PhysicalDeviceKey> {
    let identity = physical.ok_or_else(|| {
        Error::State("CubeCL backend has no reported physical device identity".into())
    })?;
    if let Some(pci) = identity.pci_address {
        return PhysicalDeviceKey::new("pci", &pci.to_string());
    }
    if let Some(luid) = identity.luid {
        return PhysicalDeviceKey::new("luid", &format!("{:02x?}", luid.bytes()));
    }
    if let Some(registry) = identity.registry_entry_id {
        return PhysicalDeviceKey::new("registry", &registry.get().to_string());
    }
    Err(Error::State(
        "CubeCL physical device lacks PCI/LUID/registry identity; shared VRAM cannot be attested"
            .into(),
    ))
}

/// Backend property alone may describe allocated/addressable memory rather
/// than currently free memory. Require an explicit conservative operator cap.
fn checked_caps(reported: Option<u64>, operator: u64, provider: u64) -> Result<()> {
    let maximum = reported.ok_or_else(|| {
        Error::State("CubeCL backend has no physical memory budget; refusing GTT guess".into())
    })?;
    if operator == 0 || provider == 0 || operator > maximum || provider > operator {
        return Err(Error::Limit(
            "operator/device/provider physical memory budget absent or inconsistent with reported memory".into()
        ));
    }
    Ok(())
}

#[cfg(any(
    test,
    all(feature = "qualification", any(feature = "hip", feature = "vulkan"))
))]
mod native_context;
#[cfg(all(feature = "qualification", any(feature = "hip", feature = "vulkan")))]
pub(crate) use native_context::NativeContextBudget;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod cuda_compiler_context;
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub(crate) use cuda_compiler_context::{
    CUDA_COMPILER_DEVICE_BYTES, CUDA_COMPILER_HOST_BYTES, CudaCompilerContextBudget,
};

pub struct PhysicalServiceBudget {
    client: Client,
    key: PhysicalDeviceKey,
    total_limit: u64,
    provider_cap: u64,
    owner_model: String,
    model_closed: Option<Arc<AtomicBool>>,
    // Conservatively charges the whole provider cap once, not again at each
    // source weight / activation / output suballocation.
    lease: Arc<GlobalResourceLease>,
    fence_flag: Arc<AtomicBool>,
}

/// Physical device admission taken before any context or device exists. It
/// carries the one original lease into `PhysicalServiceBudget` without a
/// second charge.
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "vulkan",
    target_os = "linux"
))]
pub(crate) struct PreAdmittedDevice {
    key: PhysicalDeviceKey,
    reported_max: u64,
    lease: Arc<GlobalResourceLease>,
    fence_flag: Arc<AtomicBool>,
}

#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "vulkan",
    target_os = "linux"
))]
impl PreAdmittedDevice {
    /// Bootstrap uncertainty: retain the charge and fence the card instead of
    /// crediting an admission whose context may have been constructed.
    pub(crate) fn fence_and_retain(&self) {
        self.lease.fence_unknown();
        self.fence_flag.store(true, Ordering::Release);
        std::mem::forget(self.lease.clone());
        std::mem::forget(self.fence_flag.clone());
    }
}

impl PhysicalServiceBudget {
    /// Admit the selected adapter's actual identity/device-local capacity
    /// before `device.client()` or native context construction.
    #[cfg(all(
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification"
        ),
        feature = "vulkan",
        target_os = "linux"
    ))]
    pub(crate) fn preadmit_device(
        key: PhysicalDeviceKey,
        reported_max: u64,
        operator_device_cap: u64,
        provider_cap: u64,
        owner: ResourceOwner,
    ) -> Result<PreAdmittedDevice> {
        checked_caps(Some(reported_max), operator_device_cap, provider_cap)?;
        let arbiter = ResourceArbiter::process_shared();
        arbiter.register_matching(key.clone(), operator_device_cap)?;
        let lease = arbiter.admit(
            &key,
            owner,
            &[AllocationRequest {
                kind: AllocationClass::ProviderEnvelope,
                bytes: provider_cap,
            }],
        )?;
        Ok(PreAdmittedDevice {
            key,
            reported_max,
            lease: Arc::new(lease),
            fence_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Attach the already-admitted physical charge to the actual client. The
    /// client's selected identity and capacity must match the preflight; no
    /// second arbiter admission is made.
    #[cfg(all(
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification"
        ),
        feature = "vulkan",
        target_os = "linux"
    ))]
    pub(crate) fn attach_preadmitted(
        client: &Client,
        pre: &PreAdmittedDevice,
        operator_device_cap: u64,
        provider_cap: u64,
        owner: ResourceOwner,
    ) -> Result<Self> {
        let properties = client.properties();
        let key = physical_key(properties.identity.physical.as_ref())?;
        if key != pre.key
            || properties
                .memory
                .max_memory()
                .is_none_or(|n| n < pre.reported_max || n < operator_device_cap)
            || operator_device_cap == 0
            || provider_cap == 0
            || provider_cap > operator_device_cap
        {
            return Err(Error::State(
                "pre-admitted physical identity/capacity changed before context".into(),
            ));
        }
        crate::physical_quarantine::ensure_service_safe(client)?;
        Ok(Self {
            client: client.clone(),
            key: pre.key.clone(),
            total_limit: operator_device_cap,
            provider_cap,
            owner_model: owner.model.clone(),
            model_closed: None,
            lease: pre.lease.clone(),
            fence_flag: pre.fence_flag.clone(),
        })
    }

    pub fn attach(
        client: &Client,
        operator_device_cap: u64,
        provider_cap: u64,
        owner: ResourceOwner,
    ) -> Result<Self> {
        let properties = client.properties();
        let key = physical_key(properties.identity.physical.as_ref())?;
        checked_caps(
            properties.memory.max_memory(),
            operator_device_cap,
            provider_cap,
        )?;
        let arbiter = ResourceArbiter::process_shared();
        arbiter.register_matching(key.clone(), operator_device_cap)?;
        let lease = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::ProviderEnvelope, // a single physical envelope
                bytes: provider_cap,
            }],
        )?;
        Ok(Self {
            client: client.clone(),
            key,
            total_limit: operator_device_cap,
            provider_cap,
            owner_model: owner.model.clone(),
            model_closed: None,
            lease: Arc::new(lease),
            fence_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    /// This verification also catches a provider accidentally switching
    /// CubeCL services after its physical envelope was admitted.
    pub fn begin<'a>(&'a self, client: &Client) -> Result<PhysicalWork<'a>> {
        if self
            .model_closed
            .as_ref()
            .is_some_and(|closed| closed.load(Ordering::Acquire))
        {
            return Err(Error::State(
                "source model admission closed before physical work".into(),
            ));
        }
        if client.service_id() != self.client.service_id()
            || physical_key(client.properties().identity.physical.as_ref())? != self.key
            || client
                .properties()
                .memory
                .max_memory()
                .is_none_or(|n| n < self.total_limit)
        {
            return Err(Error::State(
                "CubeCL physical service identity/capacity changed".into(),
            ));
        }
        crate::physical_quarantine::ensure_service_safe(client)?;
        let mut work = PhysicalWork::begin(&self.lease)?;
        work.fence_flag = Some(self.fence_flag.clone());
        Ok(work)
    }

    pub fn fence_flag(&self) -> Arc<AtomicBool> {
        self.fence_flag.clone()
    }

    pub fn physical_key(&self) -> PhysicalDeviceKey {
        self.key.clone()
    }

    /// Proof of pre-existing physical charge for this model and card. The
    /// noncloneable owning lease remains inside this live service object.
    pub fn prepaid_bytes_for(&self, key: &PhysicalDeviceKey, owner: &ResourceOwner) -> Result<u64> {
        if key != &self.key
            || owner.model != self.owner_model
            || self
                .model_closed
                .as_ref()
                .is_some_and(|closed| closed.load(Ordering::Acquire))
        {
            return Err(Error::State(
                "GPU queued work belongs to another model or physical card".into(),
            ));
        }
        if ResourceArbiter::process_shared().is_fenced(key)? {
            return Err(Error::State("GPU card fenced before prepaid work".into()));
        }
        Ok(self.provider_cap)
    }

    /// Partition while exclusively owned, before local admission/submission.
    /// The backend policy supplies this fixed context allowance; model adapters
    /// cannot construct the persistent proof or grow it during execution.
    #[cfg(all(feature = "qualification", any(feature = "hip", feature = "vulkan")))]
    pub(crate) fn partition_native_context(
        &mut self,
        backend: &'static str,
        policy: Arc<str>,
        bytes: u64,
    ) -> Result<Arc<NativeContextBudget>> {
        let remainder = self
            .provider_cap
            .checked_sub(bytes)
            .filter(|n| *n != 0)
            .ok_or_else(|| Error::Limit("native context must leave model capacity".into()))?;
        let lease = Arc::get_mut(&mut self.lease).ok_or_else(|| {
            Error::State("native context partition requires exclusive envelope".into())
        })?;
        let context = NativeContextBudget::partition(
            lease,
            self.key.clone(),
            self.client.service_id(),
            self.fence_flag.clone(),
            backend,
            policy,
            bytes,
        )?;
        self.provider_cap = remainder;
        Ok(context)
    }

    pub(crate) fn provider_capacity(&self) -> u64 {
        self.provider_cap
    }

    /// Same original physical envelope, detached from Client/actor lifetime.
    pub(crate) fn retained_envelope(&self) -> Arc<GlobalResourceLease> {
        self.lease.clone()
    }

    // Called only after the original resident ledger registered this model quota.
    pub(crate) fn source_binding(
        &self,
        owner: ResourceOwner,
        cap: u64,
        closed: Arc<AtomicBool>,
    ) -> Result<Self> {
        if cap == 0 || cap > self.provider_cap || owner.model.is_empty() || owner.session.is_empty()
        {
            return Err(Error::Limit(
                "source view exceeds original physical envelope".into(),
            ));
        }
        self.prepaid_bytes_for(
            &self.key,
            &ResourceOwner::new(&self.owner_model, "physical-owner")?,
        )?;
        Ok(Self {
            client: self.client.clone(),
            key: self.key.clone(),
            total_limit: self.total_limit,
            provider_cap: cap,
            owner_model: owner.model,
            model_closed: Some(closed),
            lease: self.lease.clone(),
            fence_flag: self.fence_flag.clone(),
        })
    }
    pub fn remaining(&self) -> Result<u64> {
        Ok(ResourceArbiter::process_shared()
            .snapshot(&self.key)?
            .remaining)
    }
}

/// A single submission owns its original physical envelope until CubeCL
/// read_async and the P07-E logical event ledger both report completion.
/// An aborted Future fences EVERY service sharing this physical card.
/// Physical handles remain independently alive in P07-F quarantine.
#[must_use]
pub struct PhysicalWork<'a> {
    lease: &'a GlobalResourceLease,
    armed: bool,
    fence_flag: Option<Arc<AtomicBool>>,
}
impl<'a> PhysicalWork<'a> {
    fn begin(lease: &'a GlobalResourceLease) -> Result<Self> {
        lease.mark_submitted()?;
        Ok(Self {
            lease,
            armed: true,
            fence_flag: None,
        })
    }
    pub fn complete(mut self) -> Result<()> {
        self.lease.mark_ready()?;
        self.armed = false;
        Ok(())
    }
}
impl Drop for PhysicalWork<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.lease.fence_unknown();
            if let Some(flag) = &self.fence_flag {
                flag.store(true, Ordering::Release);
            }
        }
    }
}

/// Intentional execution bootstrap only. Unlike device inventory this creates
/// the original runtime Client/primary context. Existing finite profiles retain
/// their documented pre-admission context gap; this grants no memory budget.
#[cfg(feature = "cuda")]
pub fn initialize_cuda_physical_identity(device: &cubecl::Device) -> Result<PhysicalDeviceKey> {
    if !matches!(device, cubecl::Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "CUDA identity bootstrap requires CUDA device".into(),
        ));
    }
    crate::backend::validate_provider(device)?;
    let client = device.client();
    physical_key(client.properties().identity.physical.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{AdapterLuid, PciAddress, RegistryEntryId};

    #[test]
    fn backend_reported_physical_identity_is_stable_across_service_names() {
        let mut pci = PhysicalDevice::default();
        pci.pci_address = Some(PciAddress {
            domain: 0,
            bus: 7,
            device: 0,
            function: 0,
        });
        assert_eq!(
            physical_key(Some(&pci)).unwrap(),
            PhysicalDeviceKey::new("pci", "0000:07:00.0").unwrap()
        );
        assert!(physical_key(None).is_err());
        assert!(physical_key(Some(&PhysicalDevice::default())).is_err());
        let mut luid = PhysicalDevice::default();
        luid.luid = Some(AdapterLuid::from_parts(123, 456));
        assert_eq!(physical_key(Some(&luid)).unwrap().backend, "luid");
        let mut registry = PhysicalDevice::default();
        registry.registry_entry_id = Some(RegistryEntryId::new(1));
        assert_eq!(physical_key(Some(&registry)).unwrap().backend, "registry");
    }

    #[test]
    fn unknown_or_inconsistent_physical_capacity_is_refused() {
        assert!(checked_caps(None, 32, 16).is_err());
        assert!(checked_caps(Some(1024), 0, 16).is_err());
        assert!(checked_caps(Some(1024), 1025, 16).is_err());
        assert!(checked_caps(Some(1024), 512, 513).is_err());
        assert!(checked_caps(Some(1024), 512, 0).is_err());
        checked_caps(Some(1024), 512, 256).unwrap();
    }

    #[test]
    fn dropped_work_marks_cached_source_quarantine_flag_and_retains_charge() {
        let a = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("pci", "0000:07:00.0").unwrap();
        a.register(key.clone(), 128).unwrap();
        let lease = a
            .admit(
                &key,
                ResourceOwner::new("m", "s").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: 64,
                }],
            )
            .unwrap();
        let quarantine = Arc::new(AtomicBool::new(false));
        let mut work = PhysicalWork::begin(&lease).unwrap();
        work.fence_flag = Some(quarantine.clone());
        drop(work);
        assert!(quarantine.load(Ordering::Acquire));
        assert!(a.snapshot(&key).unwrap().fenced);
        drop(lease);
        assert_eq!(a.snapshot(&key).unwrap().reserved, 64);
    }

    #[test]
    fn completed_work_allows_reuse_cancelled_work_fences_shared_resource() {
        let a = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("pci", "0000:07:00.0").unwrap();
        a.register_matching(key.clone(), 256).unwrap();
        let one = a
            .admit(
                &key,
                ResourceOwner::new("qwen.8", "s1").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: 160,
                }],
            )
            .unwrap();
        let two = a
            .admit(
                &key,
                ResourceOwner::new("qwen.2", "s2").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: 80,
                }],
            )
            .unwrap();
        assert_eq!(a.snapshot(&key).unwrap().remaining, 16);
        assert_eq!(
            a.snapshot(&key)
                .unwrap()
                .per_class
                .get(&AllocationClass::ProviderEnvelope),
            Some(&240)
        );
        PhysicalWork::begin(&one).unwrap().complete().unwrap();
        PhysicalWork::begin(&one).unwrap().complete().unwrap();
        let unknown = PhysicalWork::begin(&two).unwrap();
        drop(unknown); // cancelled after possible physical dispatch
        assert!(a.snapshot(&key).unwrap().fenced);
        assert!(PhysicalWork::begin(&one).is_err());
        drop(one);
        drop(two);
        // Submitted or unknown memory stays charged; never time-reclaimed.
        assert!(a.snapshot(&key).unwrap().reserved >= 80);
    }

    #[test]
    fn matching_operator_caps_share_one_budget_and_contradictions_refuse() {
        let a = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("pci", "0000:07:00.0").unwrap();
        a.register_matching(key.clone(), 128).unwrap();
        a.register_matching(key.clone(), 128).unwrap();
        assert!(a.register_matching(key.clone(), 64).is_err());
        assert_eq!(a.snapshot(&key).unwrap().limit, 128);
    }
}
