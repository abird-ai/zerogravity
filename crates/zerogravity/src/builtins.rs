//! Cold composition inventory and source-bound execution adapters.
use crate::diagnostics::{BuiltinStatus, MigrationStatus};
mod registration;
pub(crate) use registration::{load_declared_text, load_text};

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
        fabric: if compiled
            && cfg!(feature = "text")
            && match component {
                "cpu" => cfg!(feature = "cache"),
                "cuda" | "hip" | "vulkan" => true,
                "qwen35" => {
                    cfg!(all(feature = "cpu", feature = "cache"))
                        || cfg!(any(feature = "cuda", feature = "hip", feature = "vulkan"))
                }
                _ => false,
            } {
            MigrationStatus::TextAdapter
        } else {
            MigrationStatus::AwaitingAdapter
        },
    })
}
