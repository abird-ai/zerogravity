//! ROCm (HIP) provider policies and physical qualification.
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "hip",
    target_os = "linux"
))]
pub(crate) mod native;
#[cfg(all(test, feature = "qualification", feature = "hip", target_os = "linux"))]
mod qualification;
