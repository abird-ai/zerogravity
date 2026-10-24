//! Cold validation for the explicitly selected production provider.
//! No source arithmetic, layout, device-memory or completion permission.
use cubecl::Device;
use zerogravity_core::Result;

mod contract;
mod environment;
#[cfg(test)]
mod tests;

pub(crate) use contract::Provider;

/// Reject unlinked providers and unavailable HIP libraries before CubeCL's
/// client path, which otherwise panics on a missing HIP runtime. Device
/// capabilities and source-numerical contracts remain separately required.
pub(crate) fn validate_provider(device: &Device) -> Result<()> {
    validate_with(device, Provider::compiled, environment::hip_available)
}

fn validate_with(
    device: &Device,
    compiled: impl FnOnce(Provider) -> bool,
    hip_available: impl FnOnce() -> bool,
) -> Result<()> {
    let provider = Provider::selected(device)?;
    provider.require_compiled(compiled(provider))?;
    environment::require_runtime(provider, hip_available)
}

#[cfg(test)]
pub(crate) fn validate_build_provider(device: &Device) -> Result<()> {
    let provider = Provider::selected(device)?;
    provider.require_compiled(provider.compiled())
}

#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "hip",
    target_os = "linux"
))]
pub(crate) mod hip_native_policy;
