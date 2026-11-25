//! On-demand observations of the original recorded and prepared contracts.
//! No graph clone, physical handle, source reader or execution permission.
use super::{ObservationSupport, PhysicalRegionPort, ProviderRequirements, StateStorage};
use crate::{
    accelerator::program::{SourcePlanGeometry, ValueId},
    identity::SourceIdentity,
    semantic::{ForwardPlan, InvocationPorts, RegionStatePort},
};
use std::ops::{Range, RangeInclusive};

/// Original recording geometry, not a content hash or new plan identity.
/// Indices refer to this recording only; equality grants no binding rights.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramDescription {
    pub source: SourceIdentity,
    pub node_count: usize,
    pub value_count: usize,
    pub state_node_count: usize,
    pub state_domain_count: usize,
    pub context_capacity: usize,
    pub logits: Option<ValueId>,
    pub frontier_delta: usize,
    pub source_template: Option<SourcePlanGeometry>,
}
impl ProgramDescription {
    pub fn recorded(plan: &ForwardPlan) -> Self {
        Self {
            source: plan.source(),
            node_count: plan.nodes().len(),
            value_count: plan.values().len(),
            state_node_count: plan.state_nodes().len(),
            state_domain_count: plan.required_domains().len(),
            context_capacity: plan.prefix_capacity(),
            logits: plan.logits(),
            frontier_delta: plan.frontier_delta(),
            source_template: plan.source_plan_geometry(),
        }
    }
}

/// Exact selected closed-region contracts, copied only on explicit request.
/// The selected implementation, source and full recording are reported by
/// the enclosing ExecutionDescription. No actual storage or completion claim.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionDescription {
    pub nodes: Range<usize>,
    pub supported_rows: RangeInclusive<usize>,
    pub inputs: Box<[PhysicalRegionPort]>,
    pub outputs: Box<[PhysicalRegionPort]>,
    pub state_inputs: Box<[RegionStatePort]>,
    pub state_outputs: Box<[RegionStatePort]>,
    pub state_storage: Box<[StateStorage]>,
    pub invocation_ports: InvocationPorts,
    pub observations: ObservationSupport,
    pub requirements: ProviderRequirements,
}
