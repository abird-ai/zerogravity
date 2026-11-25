//! Context-free CUDA selection by reported physical identity, never ordinal alone.
use super::*;

pub(crate) struct ObservedCudaDevice {
    pub(crate) key: PhysicalDeviceKey,
    pub(crate) total_bytes: u64,
    #[cfg(feature = "cuda-cublas")]
    pub(crate) compute_capability: (i32, i32),
}

impl ObservedCudaDevice {
    pub(crate) fn select(device: &cubecl::Device) -> Result<Self> {
        let cubecl::Device::Cuda(selected) = device else {
            return Err(Error::Unsupported(
                "CUDA physical selection requires CUDA".into(),
            ));
        };
        let devices = cubecl::cuda::observe_cuda_devices()
            .map_err(|error| Error::Unsupported(format!("CUDA device observation: {error}")))?;
        let actual = devices
            .iter()
            .find(|view| view.ordinal == selected.index)
            .ok_or_else(|| Error::Unsupported("selected CUDA device unavailable".into()))?;
        let pci = |view: &cubecl::cuda::CudaDeviceMetadata| -> Result<cubecl::ir::PciAddress> {
            view.pci_bus_id
                .parse()
                .map_err(|error| Error::State(format!("CUDA physical PCI identity: {error}")))
        };
        let key = pci(actual)?;
        for other in &devices {
            if other.ordinal != actual.ordinal && pci(other)? == key {
                return Err(Error::Unsupported(
                    "CUDA physical PCI has ambiguous device views".into(),
                ));
            }
        }
        Ok(Self {
            key: PhysicalDeviceKey::new("pci", &key.to_string())?,
            total_bytes: actual.total_bytes,
            #[cfg(feature = "cuda-cublas")]
            compute_capability: actual.compute_capability,
        })
    }

    pub(crate) fn preadmit(
        &self,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<PreAdmittedDevice> {
        PhysicalServiceBudget::preadmit_device(
            self.key.clone(),
            self.total_bytes,
            device_cap,
            provider_cap as u64,
            ResourceOwner::new("backend/physical", "original-owner")?,
        )
    }
}
