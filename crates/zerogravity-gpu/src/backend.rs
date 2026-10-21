//! Cold validation for the explicitly selected production provider.
use cubecl::Device;
use zerogravity_core::{Error, Result};

/// Raw CubeCL device variants can name runtimes that were compiled out. Check
/// our explicit provider contract without enumerating hardware or creating a
/// client: CubeCL's `client` call panics for an unlinked runtime.
pub(crate) fn validate_provider(device: &Device) -> Result<()> {
    use cubecl::device::WgpuBackend;

    let (name, compiled) = match device {
        Device::Cuda(_) => ("CUDA", cfg!(feature = "cuda")),
        Device::Hip(_) => ("HIP", cfg!(feature = "hip")),
        Device::Wgpu(device) if device.backend == WgpuBackend::Vulkan => {
            ("Vulkan", cfg!(feature = "vulkan"))
        }
        _ => {
            return Err(Error::Unsupported(
                "resident execution requires explicit Vulkan, CUDA or HIP selection".into(),
            ));
        }
    };
    if !compiled {
        return Err(Error::Unsupported(format!(
            "resident {name} provider is not compiled into this build"
        )));
    }
    Ok(())
}
