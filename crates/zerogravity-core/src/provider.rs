//! Coarse compilation seams. No runtime execution authority from a descriptor.
mod selection;
use crate::{
    Result,
    placement::PlacementRequest,
    semantic::{ForwardPlan, RegionRequest},
};
pub use selection::{
    ExecutionMode, SelectedMode, SelectionPolicy, ShapeMissPolicy, StateBridge, SwitchingBoundary,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
    GraphOnly,
    Remote,
}

/// Retains the original semantic program and provider-owned prepared resources.
/// Execution/Ready adapters remain with existing providers until S2/F0.
pub trait PreparedExecutable: Send + Sync {
    fn program(&self) -> &ForwardPlan;
}
pub trait RegionExecutionProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn prepare(
        &self,
        region: &RegionRequest<'_>,
        placement: &PlacementRequest,
    ) -> Result<Box<dyn PreparedExecutable>>;
}
