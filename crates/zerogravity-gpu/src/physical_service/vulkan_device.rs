//! Exact named Vulkan adapter or its already registered service.
use super::*;
use cubecl::{
    Device,
    device::{WgpuBackend, WgpuDevice, WgpuDeviceKind},
    wgpu::WgpuAdapterPreflight,
};

enum Selected {
    Existing(Client),
    Cold(WgpuAdapterPreflight),
}

/// No second registry. Hold the exact original Client or uninitialized adapter.
pub(crate) struct VulkanDevicePreflight {
    key: PhysicalDeviceKey,
    max_memory: u64,
    selector: WgpuDevice,
    selected: Selected,
}

fn named(device: &Device) -> Result<&WgpuDevice> {
    match device {
        Device::Wgpu(wgpu)
            if wgpu.backend == WgpuBackend::Vulkan
                && !matches!(wgpu.kind, WgpuDeviceKind::Existing(_)) =>
        {
            Ok(wgpu)
        }
        _ => Err(Error::Unsupported(
            "Vulkan physical selection requires explicit named adapter".into(),
        )),
    }
}

impl VulkanDevicePreflight {
    pub(crate) fn select(device: &Device) -> Result<Self> {
        let selector = named(device)?;
        let (physical, max_memory, selected) = if let Some(client) =
            cubecl::wgpu::existing_client(selector)
        {
            (
                client.properties().identity.physical.clone(),
                client.properties().memory.max_memory(),
                Selected::Existing(client),
            )
        } else {
            let setup = cubecl::wgpu::try_preflight_adapter::<cubecl::wgpu::Vulkan>(selector)
                .map_err(|error| Error::State(format!("Vulkan adapter preflight: {error}")))?;
            if setup.backend() != cubecl::wgpu::wgpu::Backend::Vulkan {
                return Err(Error::State(
                    "Vulkan preflight selected another backend".into(),
                ));
            }
            (setup.physical(), setup.max_memory(), Selected::Cold(setup))
        };
        let key = physical_key(physical.as_ref())?;
        if key.backend != "pci" {
            return Err(Error::Unsupported(
                "Vulkan local VRAM observation requires exact PCI identity".into(),
            ));
        }
        let max_memory = max_memory
            .ok_or_else(|| Error::State("Vulkan adapter has no device-local capacity".into()))?;
        Ok(Self {
            key,
            max_memory,
            selector: selector.clone(),
            selected,
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

    /// Return the actual registered service, constructing only on the original
    /// registry miss. A racing initializer is checked by the admission owner.
    pub(crate) fn client(self, device: &Device, before_create: impl FnOnce()) -> Result<Client> {
        let selector = named(device)?;
        if selector != &self.selector {
            return Err(Error::State("Vulkan preflighted selector differs".into()));
        }
        match self.selected {
            Selected::Existing(client) => Ok(client),
            Selected::Cold(setup) => cubecl::wgpu::try_client_preflight::<cubecl::wgpu::Vulkan>(
                selector,
                setup,
                cubecl::wgpu::RuntimeOptions::default(),
                before_create,
            )
            .map_err(|error| Error::State(format!("Vulkan preflighted context: {error}"))),
        }
    }
}
