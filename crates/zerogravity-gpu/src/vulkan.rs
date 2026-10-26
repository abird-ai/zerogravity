//! Vulkan provider policy; shared provider selection stays in backend.
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "vulkan",
    target_os = "linux"
))]
pub(crate) mod native;
