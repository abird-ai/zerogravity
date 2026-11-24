//! The canonical semantic program is the existing ForwardPlan, not another IR.
//! Prepared executables may replace closed regions, never source/state laws.
pub use crate::accelerator::program::{
    BoundInvocation, BoundTextCall, ForwardPlan, InvocationInput, MathematicalBackend, PlanNode,
    PlanOperation, SourceOperatorTemplate, StateEffect, StateSlot, StateVersion, ValueId,
    ValueShape,
};
pub use crate::pipeline::ValueKind;

mod call;
mod region;
mod state;
pub use call::InvocationPorts;
pub use region::{BoundRegion, ClosedRegion, RegionBinding, RegionObservationPort, RegionPort};
pub use state::{RegionStatePort, SourceStateDomain, StateLayout};
