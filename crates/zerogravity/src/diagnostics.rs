//! Migration/capability observations are not execution permits.
pub use zerogravity_engine::diagnostics::RuntimeStatus;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationStatus {
    AwaitingAdapter,
    /// Local CPU text wiring compiled; actual source/profile guards still apply.
    CpuTextAdapter,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuiltinStatus {
    pub component: &'static str,
    pub compiled: bool,
    pub fabric: MigrationStatus,
}
