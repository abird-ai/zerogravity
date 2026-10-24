use cubecl::{Device, device::WgpuBackend};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Provider {
    Cuda,
    Hip,
    Vulkan,
}
impl Provider {
    pub(crate) fn selected(device: &Device) -> Result<Self> {
        match device {
            Device::Cuda(_) => Ok(Self::Cuda),
            Device::Hip(_) => Ok(Self::Hip),
            Device::Wgpu(device) if device.backend == WgpuBackend::Vulkan => Ok(Self::Vulkan),
            _ => Err(Error::Unsupported(
                "resident execution requires explicit Vulkan, CUDA or HIP selection".into(),
            )),
        }
    }
    pub(super) const fn compiled(self) -> bool {
        match self {
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Hip => cfg!(feature = "hip"),
            Self::Vulkan => cfg!(feature = "vulkan"),
        }
    }
    pub(super) fn require_compiled(self, compiled: bool) -> Result<()> {
        if !compiled {
            let name = match self {
                Self::Cuda => "CUDA",
                Self::Hip => "HIP",
                Self::Vulkan => "Vulkan",
            };
            return Err(Error::Unsupported(format!(
                "resident {name} provider is not compiled into this build"
            )));
        }
        Ok(())
    }
}
