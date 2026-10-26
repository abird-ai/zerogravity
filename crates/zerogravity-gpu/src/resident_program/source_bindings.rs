//! Checked actual source/graph/storage joins; no family or native permit.
use super::{StateLayout, StateRequirement};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    ActivationPrecision, ForwardPlan, PlanOperation, SourceCall, SourceOperator,
    SourceOperatorKind, SourceStateAccess, StateSlot, ValueId,
};
use zerogravity_core::{Error, Result};

/// Observation/fault selections in the one actual graph. No recipe or operator IR.
#[derive(Clone, PartialEq, Eq)]
pub struct SourceSelection {
    output: ValueId,
    residual_output: Option<ValueId>,
    normalized: Option<ValueId>,
    boundaries: Box<[[ValueId; 2]]>,
    selected_row: Option<usize>,
    completed_layers: usize,
    first_mlp_down_node: Option<usize>,
}
impl SourceSelection {
    /// Move already admitted selector storage; this constructor allocates nothing.
    pub fn new(
        output: ValueId,
        residual_output: Option<ValueId>,
        normalized: Option<ValueId>,
        boundaries: Box<[[ValueId; 2]]>,
        selected_row: Option<usize>,
        completed_layers: usize,
        first_mlp_down_node: Option<usize>,
    ) -> Self {
        Self {
            output,
            residual_output,
            normalized,
            boundaries,
            selected_row,
            completed_layers,
            first_mlp_down_node,
        }
    }
    pub fn output(&self) -> ValueId {
        self.output
    }
    pub fn residual_output(&self) -> Option<ValueId> {
        self.residual_output
    }
    pub fn normalized(&self) -> Option<ValueId> {
        self.normalized
    }
    pub fn boundaries(&self) -> &[[ValueId; 2]] {
        &self.boundaries
    }
    pub fn selected_row(&self) -> Option<usize> {
        self.selected_row
    }
    pub fn completed_layers(&self) -> usize {
        self.completed_layers
    }
    pub fn first_mlp_down_node(&self) -> Option<usize> {
        self.first_mlp_down_node
    }
    fn require_plan(&self, plan: &ForwardPlan) -> Result<()> {
        require_value(plan, self.output)?;
        for id in [self.residual_output, self.normalized]
            .into_iter()
            .flatten()
        {
            require_value(plan, id)?;
        }
        for pair in &self.boundaries {
            if pair[0] == pair[1] {
                return Err(Error::State("source boundary outputs alias".into()));
            }
            for &id in pair {
                require_value(plan, id)?;
            }
        }
        if self.completed_layers > plan.nodes().len()
            || self.boundaries.len() > self.completed_layers
            || self.selected_row.is_some_and(|row| {
                plan.source_call()
                    .is_none_or(|call| row >= call.rows().len())
            })
        {
            return Err(Error::State(
                "source selection extent outside actual graph/call".into(),
            ));
        }
        if let Some(index) = self.first_mlp_down_node {
            let Some(node) = plan.nodes().get(index) else {
                return Err(Error::State(
                    "source fault node outside actual graph".into(),
                ));
            };
            if !matches!(&node.operation, PlanOperation::SourceOperator { descriptor, .. }
                if matches!(descriptor.kind(), SourceOperatorKind::Projection { .. }))
            {
                return Err(Error::State(
                    "source selected projection node differs".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Owns the original plan once. Public metadata is checked data, never authority
/// to bypass source-family eligibility, physical preparation or state publication.
pub struct SourceBindings {
    source: VerifiedTensorSource,
    plan: Arc<ForwardPlan>,
    selection: SourceSelection,
    states: Vec<StateRequirement>,
    source_record_admission: Option<Arc<WindowWorkspace>>,
}
impl SourceBindings {
    /// Caller admits metadata before allocating selector/state storage. Existing
    /// metadata retained by the consumed plan stays owned and charged.
    pub fn new(
        source: VerifiedTensorSource,
        plan: ForwardPlan,
        selection: SourceSelection,
        states: Vec<StateRequirement>,
        metadata: WindowWorkspace,
    ) -> Result<Self> {
        let bytes = Self::metadata_bytes(&plan, selection.boundaries.len(), states.capacity())?;
        if !source.same_issuer(metadata.source()) || bytes > metadata.bytes() {
            return Err(Error::Limit(
                "source binding admitted metadata/issuer differs".into(),
            ));
        }
        if plan.prepared_metadata_bytes().unwrap_or(0) < plan.source_binding_metadata_bytes()? {
            return Err(Error::Limit(
                "source graph retained metadata charge absent or too small".into(),
            ));
        }
        let call = plan
            .source_call()
            .ok_or_else(|| Error::State("source binding actual call absent".into()))?;
        if plan.numerical_profile()
            != zerogravity_core::accelerator::program::ProgramNumericalProfile::SourceCudaSm86Bf16V1
            || plan.source() != source.source_identity()
            || call.source() != source.source_identity()
            || call.rows().capacity() != plan.prefix_capacity()
            || plan.frontier_delta() != 0
            || plan.logits().is_some()
            || plan.values().len() != plan.precisions().len()
            || plan
                .precisions()
                .iter()
                .any(|p| *p != ActivationPrecision::Bf16WidenedF32)
        {
            return Err(Error::State(
                "source binding unpublished BF16 graph/source differs".into(),
            ));
        }
        selection.require_plan(&plan)?;
        require_states(&plan, &states)?;
        for node in plan.nodes() {
            let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
                return Err(Error::Unsupported(
                    "source binding mathematical operator only".into(),
                ));
            };
            descriptor.validate()?;
            if descriptor.call() != call
                || descriptor.numerical_contract() != zerogravity_core::accelerator::program::SourceNumericalContract::ReferenceCudaSm86Bf16V1
            {
                return Err(Error::State(
                    "source binding foreign operator call issuer".into(),
                ));
            }
            for matrix in descriptor.matrices() {
                let actual = source.matrix(&matrix.name, matrix.axes)?;
                if matrix.source != source.source_identity()
                    || actual.source_identity() != matrix.source
                    || actual.tensor().name != matrix.name
                    || actual.tensor().kind != matrix.kind
                    || actual.axes() != matrix.axes
                    || actual.rows() != matrix.rows
                    || actual.columns() != matrix.columns
                {
                    return Err(Error::State(
                        "source binding actual matrix header differs".into(),
                    ));
                }
                source.require_matrix_binding(&actual)?;
            }
            for parameter in descriptor.parameters() {
                let actual = match descriptor.kind() {
                    SourceOperatorKind::CausalConvSplit { .. } => {
                        source.conv4_parameter(&parameter.name)?
                    }
                    _ => source.parameter(&parameter.name)?,
                };
                actual.validate_source_parameter(parameter)?;
            }
            for &id in node.inputs.iter().chain(&node.outputs) {
                require_value(&plan, id)?;
            }
        }
        Ok(Self {
            source,
            plan: Arc::new(plan.retain_binding_metadata(metadata.into_host_workspace()?)?),
            selection,
            states,
            source_record_admission: None,
        })
    }
    pub fn metadata_bytes(
        plan: &ForwardPlan,
        boundary_count: usize,
        state_capacity: usize,
    ) -> Result<usize> {
        Self::metadata_overhead_bytes(boundary_count, state_capacity)?
            .checked_add(plan.source_binding_metadata_bytes()?)
            .ok_or_else(|| Error::Limit("source binding graph metadata overflow".into()))
    }
    fn metadata_overhead_bytes(boundary_count: usize, state_capacity: usize) -> Result<usize> {
        boundary_count
            .checked_mul(std::mem::size_of::<[ValueId; 2]>())
            .and_then(|n| {
                state_capacity
                    .checked_mul(std::mem::size_of::<StateRequirement>())
                    .and_then(|s| n.checked_add(s))
            })
            .and_then(|n| {
                n.checked_add(
                    std::mem::size_of::<Self>()
                        + std::mem::size_of::<ForwardPlan>()
                        + std::mem::size_of::<zerogravity_core::resource_arbiter::HostWorkspace>()
                        + 6 * std::mem::size_of::<usize>(),
                )
            })
            .ok_or_else(|| Error::Limit("source binding metadata capacity overflow".into()))
    }
    /// New actual row issuer over the same immutable math. The model/app checks
    /// finite request eligibility first; native/root authority remains separate.
    pub fn rebind(&self, call: SourceCall, metadata: WindowWorkspace) -> Result<Self> {
        if !self.source.same_issuer(metadata.source())
            || metadata.bytes() < self.rebind_metadata_bytes()?
        {
            return Err(Error::Limit(
                "source rebind admitted metadata/issuer differs".into(),
            ));
        }
        // Core checks exact source, mode, base, row count, capacity and vocabulary.
        let plan = self.plan.rebind_source_call(call)?;
        let selection = self.selection.clone();
        let states = self.states.clone();
        selection.require_plan(&plan)?;
        Ok(Self {
            source: self.source.clone(),
            plan: Arc::new(plan.retain_binding_metadata(metadata.into_host_workspace()?)?),
            selection,
            states,
            source_record_admission: self.source_record_admission.clone(),
        })
    }
    pub fn rebind_metadata_bytes(&self) -> Result<usize> {
        Self::metadata_bytes(
            &self.plan,
            self.selection.boundaries.len(),
            self.states.capacity(),
        )
    }
    /// Retain the caller's original admitted record allocation through uploads
    /// and uncertain completion; no new charge or source owner is minted.
    pub fn retain_source_record_admission(mut self, ticket: Arc<WindowWorkspace>) -> Result<Self> {
        if self.source_record_admission.is_some()
            || !self.source.same_issuer(ticket.source())
            || ticket.bytes() > 32 * 1024 * 1024
            || ticket.bytes() < self.plan.source_binding_metadata_bytes()?
        {
            return Err(Error::State(
                "source record admission issuer/extent differs".into(),
            ));
        }
        self.source_record_admission = Some(ticket);
        Ok(self)
    }
    pub fn source_record_admission(&self) -> Option<&Arc<WindowWorkspace>> {
        self.source_record_admission.as_ref()
    }
    pub(crate) fn shares_prepared_semantics(&self, other: &Self) -> bool {
        self.source.same_issuer(&other.source)
            && self.plan.shares_semantics(&other.plan)
            && self.selection == other.selection
            && self.states.len() == other.states.len()
            && self.states.iter().zip(&other.states).all(|(a, b)| {
                a.slot == b.slot && a.bytes == b.bytes && require_layout(a.layout, b.layout).is_ok()
            })
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub(crate) fn shared_plan(&self) -> Arc<ForwardPlan> {
        self.plan.clone()
    }
    pub fn call(&self) -> &SourceCall {
        self.plan.source_call().expect("checked actual call")
    }
    pub fn selection(&self) -> &SourceSelection {
        &self.selection
    }
    pub fn state_requirements(&self) -> &[StateRequirement] {
        &self.states
    }
    pub fn operator(&self, index: usize) -> Result<Option<&SourceOperator>> {
        let node = self
            .plan
            .nodes()
            .get(index)
            .ok_or_else(|| Error::State("source operator index outside actual graph".into()))?;
        match &node.operation {
            PlanOperation::SourceOperator { descriptor, .. } => Ok(Some(descriptor)),
            _ => Ok(None),
        }
    }
}
fn require_value(plan: &ForwardPlan, id: ValueId) -> Result<()> {
    let shape = plan
        .values()
        .get(id.0 as usize)
        .ok_or_else(|| Error::State("source selected value outside actual graph".into()))?;
    if shape.capacity_elements()? == 0
        || plan.precisions().get(id.0 as usize) != Some(&ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State("source value extent/precision differs".into()));
    }
    let tag = plan
        .source_value_provenance(id)?
        .ok_or_else(|| Error::State("source value producer absent".into()))?;
    let node = plan
        .nodes()
        .get(tag.node_index())
        .ok_or_else(|| Error::State("source value producer outside graph".into()))?;
    if Some(tag.call()) != plan.source_call()
        || node.outputs.get(tag.output_index()) != Some(&id)
        || tag.inputs() != node.inputs
    {
        return Err(Error::State(
            "source value producer/input/call issuer differs".into(),
        ));
    }
    Ok(())
}
fn require_states(plan: &ForwardPlan, requirements: &[StateRequirement]) -> Result<()> {
    if requirements.len() != plan.required_domains().len()
        || requirements
            .iter()
            .map(|r| r.slot)
            .ne(plan.required_domains().iter().copied())
        || requirements
            .windows(2)
            .any(|pair| pair[0].slot >= pair[1].slot)
    {
        return Err(Error::State(
            "source complete sorted state-domain storage set differs".into(),
        ));
    }
    for requirement in requirements {
        require_storage(requirement)?;
        if matches!(requirement.layout, StateLayout::Kv { capacity, .. } if capacity != plan.prefix_capacity())
        {
            return Err(Error::State(
                "source KV storage capacity differs from actual plan".into(),
            ));
        }
    }
    for node in plan.nodes() {
        let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
            continue;
        };
        let slot = match descriptor.state_access() {
            SourceStateAccess::None => continue,
            SourceStateAccess::Write(slot) | SourceStateAccess::Read(slot) => slot,
        };
        let index = requirements
            .binary_search_by_key(&slot, |r| r.slot)
            .map_err(|_| Error::State("source operator state storage absent".into()))?;
        let requirement = &requirements[index];
        let expected = match descriptor.kind() {
            SourceOperatorKind::CausalConvSplit { geometry, .. } => Some(StateLayout::Vector {
                elements: geometry
                    .qkv_channels
                    .checked_mul(
                        geometry
                            .kernel
                            .checked_sub(1)
                            .ok_or_else(|| Error::State("source convolution kernel".into()))?,
                    )
                    .ok_or_else(|| Error::Limit("source convolution state extent".into()))?,
            }),
            SourceOperatorKind::GatedDelta { geometry, .. } => Some(StateLayout::Vector {
                elements: geometry
                    .value_heads
                    .checked_mul(geometry.key_width)
                    .and_then(|n| n.checked_mul(geometry.value_width))
                    .ok_or_else(|| Error::Limit("source recurrent state extent".into()))?,
            }),
            SourceOperatorKind::KvAppend { key_width, .. } => Some(StateLayout::Kv {
                slab: key_width,
                capacity: plan.prefix_capacity(),
            }),
            SourceOperatorKind::FlashAttention { geometry, .. } => Some(StateLayout::Kv {
                slab: geometry
                    .kv_heads
                    .checked_mul(geometry.width)
                    .ok_or_else(|| Error::Limit("source attention state extent".into()))?,
                capacity: plan.prefix_capacity(),
            }),
            // Carry-only domains retain the model owner's explicit checked layout.
            SourceOperatorKind::StateCarry { .. } => None,
            _ => {
                return Err(Error::State(
                    "source operator has unexpected state access".into(),
                ));
            }
        };
        if let Some(expected) = expected {
            require_layout(requirement.layout, expected)?;
        }
    }
    Ok(())
}
fn require_storage(requirement: &StateRequirement) -> Result<()> {
    let (planes, words) = requirement.layout.planes()?;
    let bytes = planes
        .checked_mul(words)
        .and_then(|n| n.checked_mul(4))
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("source F32 state storage extent".into()))?;
    if requirement.bytes != bytes
        || !matches!(
            (requirement.slot, requirement.layout),
            (
                StateSlot::Conv(_) | StateSlot::Recurrent(_),
                StateLayout::Vector { .. }
            ) | (
                StateSlot::Kv(_),
                StateLayout::Kv {
                    slab: 1..,
                    capacity: 1..
                }
            )
        )
    {
        return Err(Error::State(
            "source state storage precision/planes/bytes differs".into(),
        ));
    }
    Ok(())
}
fn require_layout(actual: StateLayout, expected: StateLayout) -> Result<()> {
    let equal = match (actual, expected) {
        (StateLayout::Vector { elements: a }, StateLayout::Vector { elements: b }) => a == b,
        (
            StateLayout::Kv {
                slab: a,
                capacity: b,
            },
            StateLayout::Kv {
                slab: c,
                capacity: d,
            },
        ) => a == c && b == d,
        _ => false,
    };
    if !equal {
        return Err(Error::State("source operator state layout differs".into()));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::program::{
        ConvDomain, KvDomain, RecurrentDomain, SourceRows,
    };
    #[test]
    fn state_bytes_are_exact_f32_planes_and_slot_kinds() {
        for (slot, layout, bytes) in [
            (
                StateSlot::Conv(ConvDomain(2)),
                StateLayout::Vector { elements: 9 },
                36,
            ),
            (
                StateSlot::Recurrent(RecurrentDomain(2)),
                StateLayout::Vector { elements: 15 },
                60,
            ),
            (
                StateSlot::Kv(KvDomain(3)),
                StateLayout::Kv {
                    slab: 5,
                    capacity: 7,
                },
                280,
            ),
        ] {
            assert!(
                require_storage(&StateRequirement {
                    slot,
                    layout,
                    bytes
                })
                .is_ok()
            );
            assert!(
                require_storage(&StateRequirement {
                    slot,
                    layout,
                    bytes: bytes / 2
                })
                .is_err()
            );
        }
        assert!(
            require_storage(&StateRequirement {
                slot: StateSlot::Kv(KvDomain(0)),
                layout: StateLayout::Vector { elements: 8 },
                bytes: 32
            })
            .is_err()
        );
        assert!(
            require_storage(&StateRequirement {
                slot: StateSlot::Conv(ConvDomain(0)),
                layout: StateLayout::Vector { elements: 0 },
                bytes: 0
            })
            .is_err()
        );
        assert!(
            require_storage(&StateRequirement {
                slot: StateSlot::Kv(KvDomain(0)),
                layout: StateLayout::Kv {
                    slab: usize::MAX,
                    capacity: 2
                },
                bytes: 8
            })
            .is_err()
        );
    }
    #[test]
    fn metadata_capacity_overflow_and_layout_substitution_refuse() {
        assert!(SourceBindings::metadata_overhead_bytes(usize::MAX, 1).is_err());
        assert!(SourceBindings::metadata_overhead_bytes(1, usize::MAX).is_err());
        assert!(
            require_layout(
                StateLayout::Kv {
                    slab: 4,
                    capacity: 8
                },
                StateLayout::Kv {
                    slab: 8,
                    capacity: 4
                }
            )
            .is_err()
        );
        assert!(
            require_layout(
                StateLayout::Vector { elements: 32 },
                StateLayout::Kv {
                    slab: 4,
                    capacity: 8
                }
            )
            .is_err()
        );
    }
    fn graph(required: Vec<StateSlot>) -> ForwardPlan {
        use zerogravity_core::accelerator::{SourceIdentity, program::*};
        use zerogravity_core::quant::QuantKind;
        let source = SourceIdentity::for_new_allocation().unwrap();
        let call = SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, 8, 16).unwrap(),
                0,
                Arc::from([3]),
            )
            .unwrap(),
            SourceCallMode::Prefill,
        )
        .unwrap();
        let mut builder = PlanBuilder::new_source_program(call.clone(), required.clone()).unwrap();
        let embedding = SourceOperator::new(
            call.clone(),
            SourceOperatorKind::EmbeddingRows,
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            vec![SourceTensor {
                source,
                name: "fixture.embedding".into(),
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                rows: 8,
                columns: 4,
            }],
            vec![],
        )
        .unwrap();
        builder
            .record_source_operator("fixture.embedding", &embedding, &[])
            .unwrap();
        for slot in required {
            let carry = SourceOperator::new(
                call.clone(),
                SourceOperatorKind::StateCarry { slot },
                SourceNumericalContract::ReferenceCudaSm86Bf16V1,
                vec![],
                vec![],
            )
            .unwrap();
            builder
                .record_source_operator("fixture.carry", &carry, &[])
                .unwrap();
        }
        builder.finish_source_program().unwrap()
    }
    #[test]
    fn selection_requires_actual_value_producer_indices_and_rows() {
        let plan = graph(vec![]);
        let selection = |output, boundaries, selected_row, node| {
            SourceSelection::new(output, None, None, boundaries, selected_row, 0, node)
        };
        assert!(
            selection(ValueId(0), Box::new([]), Some(0), None)
                .require_plan(&plan)
                .is_ok()
        );
        assert!(
            selection(ValueId(1), Box::new([]), None, None)
                .require_plan(&plan)
                .is_err()
        );
        assert!(
            selection(ValueId(0), Box::new([]), Some(1), None)
                .require_plan(&plan)
                .is_err()
        );
        assert!(
            selection(ValueId(0), Box::new([[ValueId(0), ValueId(0)]]), None, None)
                .require_plan(&plan)
                .is_err()
        );
        assert!(
            selection(ValueId(0), Box::new([]), None, Some(99))
                .require_plan(&plan)
                .is_err()
        );
        assert!(
            selection(ValueId(0), Box::new([]), None, Some(0))
                .require_plan(&plan)
                .is_err()
        );
        let call = plan.source_call().unwrap();
        let foreign = SourceCall::new(
            SourceRows::new(
                zerogravity_core::accelerator::program::TextCallGeometry::new(call.source(), 8, 16)
                    .unwrap(),
                0,
                Arc::from([3]),
            )
            .unwrap(),
            zerogravity_core::accelerator::program::SourceCallMode::Prefill,
        )
        .unwrap();
        assert_ne!(&foreign, call); // Equal token IDs do not reproduce the issuer.
        let rebound = plan.rebind_source_call(foreign).unwrap();
        assert!(plan.shares_semantics(&rebound));
        assert!(require_value(&rebound, ValueId(0)).is_ok());
    }
    #[test]
    fn carry_only_storage_requires_the_complete_original_domain_set() {
        let conv = StateSlot::Conv(ConvDomain(2));
        let kv = StateSlot::Kv(KvDomain(3));
        let plan = graph(vec![conv, kv]);
        let requirements = [
            StateRequirement {
                slot: conv,
                layout: StateLayout::Vector { elements: 9 },
                bytes: 36,
            },
            StateRequirement {
                slot: kv,
                layout: StateLayout::Kv {
                    slab: 5,
                    capacity: 16,
                },
                bytes: 640,
            },
        ];
        assert!(require_states(&plan, &requirements).is_ok());
        assert!(require_states(&plan, &requirements[..1]).is_err());
        assert!(require_states(&plan, &[requirements[1], requirements[0]]).is_err());
        assert!(require_states(&plan, &[requirements[0], requirements[0]]).is_err());
        let mut wrong = requirements;
        wrong[1].bytes = 320;
        assert!(require_states(&plan, &wrong).is_err());
    }
}
