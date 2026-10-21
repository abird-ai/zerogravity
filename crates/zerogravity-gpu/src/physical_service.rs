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
fn physical_key(physical: Option<&PhysicalDevice>) -> Result<PhysicalDeviceKey> {
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

pub struct PhysicalServiceBudget {
    client: Client,
    key: PhysicalDeviceKey,
    total_limit: u64,
    provider_cap: u64,
    owner_model: String,
    // Conservatively charges the whole provider cap once, not again at each
    // source weight / activation / output suballocation.
    lease: GlobalResourceLease,
    fence_flag: Arc<AtomicBool>,
}

impl PhysicalServiceBudget {
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
            lease,
            fence_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    /// This verification also catches a provider accidentally switching
    /// CubeCL services after its physical envelope was admitted.
    pub fn begin<'a>(&'a self, client: &Client) -> Result<PhysicalWork<'a>> {
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
        if key != &self.key || owner.model != self.owner_model {
            return Err(Error::State(
                "GPU queued work belongs to another model or physical card".into(),
            ));
        }
        if ResourceArbiter::process_shared().is_fenced(key)? {
            return Err(Error::State("GPU card fenced before prepaid work".into()));
        }
        Ok(self.provider_cap)
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
