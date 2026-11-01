//! Cold composition inventory and source-bound execution adapters.
use crate::diagnostics::{BuiltinStatus, MigrationStatus};

#[cfg(feature = "text")]
pub mod text;
pub fn components() -> [BuiltinStatus; 6] {
    [
        ("cpu", cfg!(feature = "cpu")),
        ("cuda", cfg!(feature = "cuda")),
        ("hip", cfg!(feature = "hip")),
        ("vulkan", cfg!(feature = "vulkan")),
        ("qwen35", cfg!(feature = "model-qwen35")),
        ("gemma4", cfg!(feature = "model-gemma4")),
    ]
    .map(|(component, compiled)| BuiltinStatus {
        component,
        compiled,
        fabric: if cfg!(all(feature = "cpu", feature = "text", feature = "cache"))
            && matches!(component, "cpu" | "qwen35")
        {
            MigrationStatus::CpuTextAdapter
        } else {
            MigrationStatus::AwaitingAdapter
        },
    })
}
