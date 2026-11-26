//! Explicit backend selection; inspection/construction does not probe hardware.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend {
    VulkanAmd,
    VulkanNvidia,
    Cuda,
    Hip,
}

impl GpuBackend {
    pub fn command(self) -> &'static str {
        match self {
            Self::VulkanAmd => "run-vulkan-amd",
            Self::VulkanNvidia => "run-vulkan-nvidia",
            Self::Cuda => "run-cuda",
            Self::Hip => "run-rocm",
        }
    }

    pub fn feature(self) -> &'static str {
        match self {
            Self::VulkanAmd | Self::VulkanNvidia => "vulkan",
            Self::Cuda => "cuda",
            Self::Hip => "hip",
        }
    }

    pub fn compiled(self) -> bool {
        match self {
            Self::VulkanAmd | Self::VulkanNvidia => cfg!(feature = "vulkan"),
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Hip => cfg!(feature = "hip"),
        }
    }

    /// Can the original backend loader open its runtime libraries?
    /// No explicit device initialization, client, context or model. None
    /// means no compiled probe; success is not a physical binding or ABI,
    /// compiler/provider or execution qualification.
    pub fn runtime_libraries(self) -> Option<bool> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => Some(crate::gpu::cuda_driver_available()),
            #[cfg(feature = "hip")]
            Self::Hip => Some(crate::gpu::hip_runtime_libraries_available()),
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            Self::VulkanAmd | Self::VulkanNvidia => Some(crate::gpu::vulkan_loader_available()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    #[cfg(all(
        feature = "model-qwen35",
        feature = "text",
        any(feature = "vulkan", feature = "cuda", feature = "hip")
    ))]
    pub fn device(self) -> crate::Result<crate::gpu::Device> {
        use crate::gpu::Device;
        let device = match self {
            #[cfg(feature = "vulkan")]
            Self::VulkanAmd => Device::vulkan(crate::gpu::WgpuDeviceKind::IntegratedGpu(0)),
            #[cfg(feature = "vulkan")]
            Self::VulkanNvidia => Device::vulkan(crate::gpu::WgpuDeviceKind::DiscreteGpu(0)),
            #[cfg(feature = "cuda")]
            Self::Cuda => Device::cuda(0),
            #[cfg(feature = "hip")]
            Self::Hip => Device::rocm(0),
            #[cfg(not(all(feature = "vulkan", feature = "cuda", feature = "hip")))]
            _ => return Err(self.unavailable()),
        };
        device.map_err(|error| {
            crate::Error::Unsupported(format!("{} device unavailable: {error}", self.command(),))
        })
    }

    pub fn unavailable(self) -> crate::Error {
        crate::Error::Unsupported(format!(
            "{} requires compiled features {}, model-qwen35 and text",
            self.command(),
            self.feature(),
        ))
    }
}
