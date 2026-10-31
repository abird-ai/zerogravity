//! The canonical semantic program is the existing ForwardPlan, not another IR.
//! Prepared executables may replace closed regions, never source/state laws.
pub use crate::accelerator::program::{
    BoundSourceInvocation, ForwardPlan, MathematicalBackend, PlanNode, PlanOperation,
    SourceInvocation, SourceOperatorTemplate, StateEffect, StateSlot, StateVersion, ValueId,
    ValueShape,
};
pub use crate::pipeline::ValueKind;

/// An inspection range, not a closed-region replacement permit.
#[derive(Clone, Debug)]
pub struct RegionRequest<'a> {
    pub program: &'a ForwardPlan,
    pub nodes: std::ops::Range<usize>,
}
