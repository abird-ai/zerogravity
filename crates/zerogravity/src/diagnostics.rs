//! Migration/capability observations are not execution permits.
pub use zerogravity_engine::diagnostics::RuntimeStatus;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationStatus {
    AwaitingAdapter,
    /// Shared text wiring compiled; source guards and physical qualification are separate.
    TextAdapter,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuiltinStatus {
    pub component: &'static str,
    pub compiled: bool,
    pub fabric: MigrationStatus,
}

#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
pub mod text_gpu;

/// Explicit original CPU source/audit driver; not a scheduling or serving owner.
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
pub mod cpu_text;

/// Finite native route classification/refusal; execution requires qualification.
pub mod qwen_route;

/// Explicit original Gemma CPU/optional HIP candidates, never serving eligibility.
#[cfg(all(feature = "model-gemma4-cpu-candidate", target_os = "linux"))]
pub mod gemma_text;
