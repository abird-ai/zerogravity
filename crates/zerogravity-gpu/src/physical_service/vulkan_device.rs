//! Exact selected adapter for the existing guarded cold Vulkan bootstrap.
use super::*;
use cubecl::{Device, wgpu::WgpuAdapterPreflight};

/// Owns the selected adapter, before a device or queue exists.
pub(crate) struct VulkanDevicePreflight {
    key: PhysicalDeviceKey,
    max_memory: u64,
    setup: WgpuAdapterPreflight,
}

impl VulkanDevicePreflight {
    pub(crate) fn select(device: &Device) -> Result<Self> {
        let Device::Wgpu(wgpu) = device else {
            return Err(Error::Unsupported(
                "Vulkan physical selection requires Vulkan".into(),
            ));
        };
        if wgpu.backend != cubecl::device::WgpuBackend::Vulkan {
            return Err(Error::Unsupported(
                "Vulkan physical selection requires explicit Vulkan backend".into(),
            ));
        }
        let setup = cubecl::wgpu::try_preflight_adapter::<cubecl::wgpu::Vulkan>(wgpu)
            .map_err(|error| Error::State(format!("Vulkan adapter preflight: {error}")))?;
        if setup.backend() != cubecl::wgpu::wgpu::Backend::Vulkan {
            return Err(Error::State(
                "Vulkan adapter preflight selected another backend".into(),
            ));
        }
        let key = physical_key(setup.physical().as_ref())?;
        if key.backend != "pci" {
            return Err(Error::Unsupported(
                "Vulkan local VRAM observation requires exact PCI identity".into(),
            ));
        }
        let max_memory = setup.max_memory().ok_or_else(|| {
            Error::State("Vulkan adapter preflight has no device-local capacity".into())
        })?;
        Ok(Self {
            key,
            max_memory,
            setup,
        })
    }

    pub(crate) fn key(&self) -> &PhysicalDeviceKey {
        &self.key
    }

    pub(crate) fn preadmit(
        &self,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<PreAdmittedDevice> {
        let local = zerogravity_core::placement::LocalVramObservation::observe(
            &std::path::Path::new("/sys/bus/pci/devices").join(&self.key.device),
        );
        let (total, available) = local.total.zip(local.available()).ok_or_else(|| {
            Error::Unsupported("Vulkan local VRAM capacity/current usage unavailable".into())
        })?;
        let ceiling = self.max_memory.min(total);
        PhysicalServiceBudget::preadmit_available_device(
            self.key().clone(),
            ceiling,
            available.min(ceiling),
            device_cap,
            provider_cap as u64,
            ResourceOwner::new("backend/physical", "original-owner")?,
        )
    }

    /// Register this exact adapter once on the guarded cold path. Do not
    /// call this for an already initialized selector; subsequent model
    /// admissions share its original ResidentPhysical.
    pub(crate) fn register(self, device: &Device) -> Result<()> {
        let Device::Wgpu(wgpu) = device else {
            return Err(Error::State(
                "Vulkan preflighted context device kind differs".into(),
            ));
        };
        if wgpu.backend != cubecl::device::WgpuBackend::Vulkan {
            return Err(Error::State(
                "Vulkan preflighted context backend differs".into(),
            ));
        }
        cubecl::wgpu::try_init_preflight::<cubecl::wgpu::Vulkan>(
            wgpu,
            self.setup,
            cubecl::wgpu::RuntimeOptions::default(),
        )
        .map(|_| ())
        .map_err(|error| Error::State(format!("Vulkan preflighted context: {error}")))
    }
}
