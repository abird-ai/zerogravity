//! Cold composition inventory. F0 registers source-bound execution adapters here.
use crate::diagnostics::{BuiltinStatus, MigrationStatus};
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
        fabric: MigrationStatus::AwaitingAdapter,
    })
}
