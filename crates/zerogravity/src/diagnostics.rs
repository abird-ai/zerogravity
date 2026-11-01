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

#[cfg(all(feature = "qwen35-cuda", target_os = "linux"))]
pub mod text_gpu;
