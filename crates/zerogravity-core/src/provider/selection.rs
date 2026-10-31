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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Portable,
    FusedRegion,
    Graph,
    Remote,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedMode {
    pub provider: ProviderKind,
    pub mode: ExecutionMode,
}
