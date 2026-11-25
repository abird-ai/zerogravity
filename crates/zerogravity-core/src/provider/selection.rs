//! Cold selection descriptions. No online tuner or qualified candidate issuer.
use super::ProviderKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchingBoundary {
    ColdPreparation,
    CommittedRoot,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateBridge {
    OriginalLayout,
    ConvertAtCommittedFrontier,
    RecomputeAtCommittedFrontier,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeMissPolicy {
    Refuse,
    QualifiedDecomposition,
}
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Portable,
    Native,
    FusedRegion,
    /// Native command capture/replay, not external graph compilation.
    Graph,
    /// Selected compiler/runtime implementation, independently of the device.
    Compiled(CompiledProvider),
    Remote,
}

/// Optional implementation identity only. These variants do not imply an
/// installed SDK, supported model, available device or qualified executable.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledProvider {
    Migraphx,
    TensorRt,
    Iree,
    OnnxRuntime,
}

/// Hysteresis and switching constraints for a future finite qualified candidate set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionPolicy {
    pub switching: SwitchingBoundary,
    pub stable_observations: std::num::NonZeroU32,
    pub shape_miss: ShapeMissPolicy,
    pub state_bridge: StateBridge,
}
/// Selected physical mode description; not evidence, a state bridge or Ready.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedMode {
    pub provider: ProviderKind,
    pub mode: ExecutionMode,
}
