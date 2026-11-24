//! Closed, dense source-template regions; the original plan remains authoritative.
use super::{
    BoundInvocation, ForwardPlan, PlanNode, PlanOperation, RegionStatePort, ValueId, ValueShape,
};
use crate::{
    Error, Result,
    accelerator::program::{SourceOperatorKind, SourceRowSelection, SourceStateAccess, TensorView},
    numerics::ActivationPrecision,
};
use std::{ops::Range, sync::Arc};

/// Logical layout, not a device allocation or storage encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionPort {
    pub value: ValueId,
    pub precision: ActivationPrecision,
    pub width: usize,
    pub rows: SourceRowSelection,
}
impl RegionPort {
    fn binding(self, active_rows: usize) -> RegionBinding {
        RegionBinding {
            value: self.value,
            precision: self.precision,
            layout: TensorView {
                offset: 0,
                rows: match self.rows {
                    SourceRowSelection::AllActive => active_rows,
                    SourceRowSelection::LastActive => 1,
                },
                width: self.width,
                row_stride: self.width,
            },
        }
    }
}

/// Exact logical invocation binding. Physical format, backing and leases are
/// independently checked by the prepared implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionBinding {
    pub value: ValueId,
    pub precision: ActivationPrecision,
    pub layout: TensorView,
}

/// A contiguous dense region of the original invocation-free program.
/// All operations remain required. Stateful operators explicitly export
/// original versioned state ports and use the bound invocation's frontier.
/// Embedding, rotary, arbitrary observations and output-policy crossings
/// are not yet supported by this region contract.
pub struct ClosedRegion {
    plan: Arc<ForwardPlan>,
    range: Range<usize>,
    inputs: Vec<RegionPort>,
    outputs: Vec<RegionPort>,
    state_inputs: Vec<RegionStatePort>,
    state_outputs: Vec<RegionStatePort>,
    max_rows: usize,
}
impl ClosedRegion {
    pub(crate) fn derive(plan: Arc<ForwardPlan>, range: Range<usize>) -> Result<Self> {
        let geometry = plan.source_plan_geometry().ok_or_else(|| {
            Error::Unsupported("closed region requires invocation-free source template".into())
        })?;
        if range.is_empty() || range.end > plan.nodes().len() {
            return Err(Error::Malformed("closed region node range".into()));
        }
        let frontier = plan.state_nodes().len();
        if range.start < frontier && range.end > frontier {
            return Err(Error::Unsupported(
                "region crosses output-policy boundary".into(),
            ));
        }
        let max_rows = geometry.max_rows();
        let mut producers = vec![None; plan.values().len()];
        for (index, node) in plan.nodes().iter().enumerate() {
            for value in &node.outputs {
                let producer = producers
                    .get_mut(value.0 as usize)
                    .ok_or_else(|| Error::State("region value outside original plan".into()))?;
                if producer.replace(index).is_some() {
                    return Err(Error::State("region value has multiple producers".into()));
                }
            }
        }
        let producer = |value: ValueId| {
            producers
                .get(value.0 as usize)
                .copied()
                .flatten()
                .ok_or_else(|| Error::State("region value has no original producer".into()))
        };
        let port = |value: ValueId, elements: usize, rows: SourceRowSelection| {
            let capacity_rows = if rows == SourceRowSelection::AllActive {
                max_rows
            } else {
                1
            };
            if elements == 0
                || !elements.is_multiple_of(capacity_rows)
                || plan.values().get(value.0 as usize) != Some(&ValueShape::Vector(elements))
                || plan.precisions().get(value.0 as usize)
                    != Some(&ActivationPrecision::Bf16WidenedF32)
            {
                return Err(Error::State(
                    "region dense value geometry/precision differs".into(),
                ));
            }
            Ok(RegionPort {
                value,
                precision: ActivationPrecision::Bf16WidenedF32,
                width: elements / capacity_rows,
                rows,
            })
        };
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut state_inputs = Vec::new();
        let mut state_outputs = Vec::new();
        let domains = if plan.nodes()[range.clone()].iter().any(|node| {
            node.state.is_some()
                || matches!(
                    &node.operation,
                    PlanOperation::SourceOperatorTemplate { state: Some(_), .. }
                )
        }) {
            plan.source_state_domains()?
        } else {
            Vec::new()
        };
        let mut escapes = vec![false; plan.values().len()];
        for node in &plan.nodes()[range.end..] {
            for value in &node.inputs {
                if range.contains(&producer(*value)?) {
                    escapes[value.0 as usize] = true;
                }
            }
        }
        if let Some(logits) = plan.logits() {
            escapes[logits.0 as usize] = true;
        }
        for (index, node) in plan.nodes()[range.clone()].iter().enumerate() {
            let index = range.start + index;
            let PlanOperation::SourceOperatorTemplate { descriptor, state } = &node.operation
            else {
                return Err(Error::Unsupported(
                    "region operation requires an explicit effect contract".into(),
                ));
            };
            if !matches!(
                descriptor.kind(),
                SourceOperatorKind::Projection { selected_row: None }
                    | SourceOperatorKind::Rms(_)
                    | SourceOperatorKind::ResidualRms(_)
                    | SourceOperatorKind::Sigmoid { .. }
                    | SourceOperatorKind::Multiply { .. }
                    | SourceOperatorKind::SiluAndMul { .. }
                    | SourceOperatorKind::CausalConvSplit { .. }
                    | SourceOperatorKind::GatedDelta { .. }
                    | SourceOperatorKind::KvAppend { .. }
                    | SourceOperatorKind::Attention { .. }
            ) {
                return Err(Error::Unsupported(
                    "region operation is not supported dense math/state".into(),
                ));
            }
            if descriptor.source() != plan.source()
                || descriptor.max_rows() != max_rows
                || descriptor.vocabulary() != geometry.vocabulary()
                || descriptor.numerical_contract().program_profile() != plan.numerical_profile()
            {
                return Err(Error::State(
                    "region source template geometry differs".into(),
                ));
            }
            let lookup = |slot| {
                domains
                    .iter()
                    .find(|domain| domain.slot() == slot)
                    .ok_or_else(|| Error::State("region original state domain absent".into()))
            };
            match descriptor.state_access() {
                SourceStateAccess::None => {
                    if state.is_some() || node.state.is_some() {
                        return Err(Error::State("region unexpected state effect".into()));
                    }
                }
                SourceStateAccess::Write(slot) => {
                    let domain = lookup(slot)?;
                    state_inputs.push(RegionStatePort {
                        version: domain.effect.read,
                        layout: domain.layout,
                    });
                    state_outputs.push(RegionStatePort {
                        version: domain.effect.write,
                        layout: domain.layout,
                    });
                }
                SourceStateAccess::Read(slot) => {
                    let domain = lookup(slot)?;
                    if !range.contains(&domain.writer) {
                        let port = RegionStatePort {
                            version: domain.effect.write,
                            layout: domain.layout,
                        };
                        if !state_inputs.contains(&port) {
                            state_inputs.push(port);
                        }
                    }
                }
            }
            let input_elements = descriptor.input_elements()?;
            let output_elements = descriptor.output_elements()?;
            if input_elements.len() != node.inputs.len()
                || output_elements.len() != node.outputs.len()
            {
                return Err(Error::State("region operation port arity differs".into()));
            }
            for (&value, elements) in node.inputs.iter().zip(input_elements) {
                let origin = producer(value)?;
                if origin >= index {
                    return Err(Error::State("region input producer order differs".into()));
                }
                let PlanOperation::SourceOperatorTemplate {
                    descriptor: origin_descriptor,
                    ..
                } = &plan.nodes()[origin].operation
                else {
                    return Err(Error::Unsupported(
                        "region input producer layout is not supported".into(),
                    ));
                };
                if origin_descriptor.row_selection() != SourceRowSelection::AllActive {
                    return Err(Error::Unsupported(
                        "region input requires all-active rows".into(),
                    ));
                }
                let port = port(value, elements, SourceRowSelection::AllActive)?;
                if origin < range.start && !inputs.contains(&port) {
                    inputs.push(port);
                }
            }
            for (&value, elements) in node.outputs.iter().zip(output_elements) {
                let port = port(value, elements, descriptor.row_selection())?;
                if escapes[value.0 as usize] {
                    outputs.push(port);
                }
            }
        }
        if (inputs.is_empty() && state_inputs.is_empty())
            || (outputs.is_empty() && state_outputs.is_empty())
        {
            return Err(Error::Unsupported(
                "region requires explicit input and output ports".into(),
            ));
        }
        Ok(Self {
            plan,
            range,
            inputs,
            outputs,
            state_inputs,
            state_outputs,
            max_rows,
        })
    }

    pub fn program(&self) -> &Arc<ForwardPlan> {
        &self.plan
    }
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn nodes(&self) -> &[PlanNode] {
        &self.plan.nodes()[self.range.clone()]
    }
    pub fn inputs(&self) -> &[RegionPort] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[RegionPort] {
        &self.outputs
    }
    pub fn state_inputs(&self) -> &[RegionStatePort] {
        &self.state_inputs
    }
    pub fn state_outputs(&self) -> &[RegionStatePort] {
        &self.state_outputs
    }
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn capacity_bindings(&self) -> impl Iterator<Item = RegionBinding> + '_ {
        self.inputs
            .iter()
            .chain(&self.outputs)
            .map(|port| port.binding(self.max_rows))
    }

    /// Late logical binding to the exact plan and active invocation schedule.
    /// No allocation, state mutation or submission.
    pub fn bind<'a>(&'a self, invocation: &'a BoundInvocation) -> Result<BoundRegion<'a>> {
        let rows = invocation.input().token_count();
        if !Arc::ptr_eq(&self.plan, invocation.plan()) {
            return Err(Error::State(
                "region invocation belongs to another program".into(),
            ));
        }
        if rows == 0 || rows > self.max_rows || self.range.end > invocation.active_nodes().len() {
            return Err(Error::Unsupported(
                "region invocation outside active shape/schedule".into(),
            ));
        }
        Ok(BoundRegion {
            region: self,
            invocation,
        })
    }
}

pub struct BoundRegion<'a> {
    region: &'a ClosedRegion,
    invocation: &'a BoundInvocation,
}
impl BoundRegion<'_> {
    pub fn invocation(&self) -> &BoundInvocation {
        self.invocation
    }
    pub fn state_inputs(&self) -> &[RegionStatePort] {
        self.region.state_inputs()
    }
    pub fn state_outputs(&self) -> &[RegionStatePort] {
        self.region.state_outputs()
    }
    pub fn inputs(&self) -> impl Iterator<Item = RegionBinding> + '_ {
        self.region
            .inputs
            .iter()
            .map(|port| port.binding(self.invocation.input().token_count()))
    }
    pub fn outputs(&self) -> impl Iterator<Item = RegionBinding> + '_ {
        self.region
            .outputs
            .iter()
            .map(|port| port.binding(self.invocation.input().token_count()))
    }
    pub fn require_bindings(
        &self,
        inputs: &[RegionBinding],
        outputs: &[RegionBinding],
    ) -> Result<()> {
        if !self.inputs().eq(inputs.iter().copied()) || !self.outputs().eq(outputs.iter().copied())
        {
            return Err(Error::State(
                "region invocation port bindings differ".into(),
            ));
        }
        Ok(())
    }
}
