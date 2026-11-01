//! Typed source-record storage, with a separately guarded finite adapter.
use super::*;
use zerogravity_core::accelerator::program::{
    ProgramNumericalProfile, SourceParameter, StandaloneExpProfile, ValueStorageBinding,
    ValueStorageLimits, ValueStorageRetention, ValueStorageReusePolicy,
};

pub(super) const ALIGNMENT: usize = 256;
pub(super) const RECORD_BYTES: usize = 4 * 1024 * 1024;
pub(super) const METADATA_BYTES: usize = 4 * 1024 * 1024;
#[derive(Clone, Debug)]
pub struct TextSourceResourceInventory {
    pub value_slot_bytes: usize,
    pub value_slots: usize,
    pub parameter_bytes: usize,
    pub math_scratch_bytes: usize,
    pub source_transport_bytes: usize,
    pub owned_device_bytes: usize,
    pub greedy_workspace_bytes: usize,
    pub feedback_peak_bytes: usize,
    pub root_payload_bytes: usize,
    pub six_root_payload_bytes: usize,
    pub root_initialization_host_bytes: usize,
    pub root_pool_metadata_bytes: usize,
    pub native_host_bytes: usize,
    pub host_upper_bytes: usize,
    pub native_device_bytes: usize,
    pub admitted_host_domain_bytes: usize,
    pub complete_device_bytes: usize,
    pub largest_binding_bytes: usize,
}
pub(super) struct Layout {
    pub slots: Vec<usize>,
    pub bindings: Vec<ValueStorageBinding>,
    pub parameters: BTreeMap<String, SourceParameter>,
    pub math: Vec<usize>,
    pub states: Vec<StateRequirement>,
    pub value_bytes: usize,
    pub parameter_bytes: usize,
    pub owned_bytes: usize,
    pub root_bytes: usize,
    pub max_columns: usize,
}
pub(super) fn aligned(n: usize) -> Result<usize> {
    n.checked_add(ALIGNMENT - 1)
        .map(|n| n & !(ALIGNMENT - 1))
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("source alignment overflow".into()))
}
pub(super) fn sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| Error::Limit("source complete owner sum overflow".into()))
    })
}
impl Layout {
    pub(super) fn require_source_record(recorded: &TextStepBindings) -> Result<()> {
        let p = recorded.plan();
        if p.numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || p.standalone_exp_profile() != Some(StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1)
            || p.is_operator_slice()
            || !(1..=64).contains(&recorded.capacity())
            || p.prefix_capacity() != recorded.capacity()
            || recorded.position() >= recorded.capacity()
            || p.frontier_delta() != 1
            || p.required_domains().is_empty()
            || p.required_domains().len() > 128
            || p.required_domains().len() != p.proposed_domains().len()
            || p.nodes().len() > 8192
            || p.values().len() > 8192
        {
            return Err(Error::Unsupported(
                "source single-row arithmetic/state bound".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn require_record(recorded: &TextStepBindings) -> Result<()> {
        let plan = recorded.plan();
        if plan.numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || plan.standalone_exp_profile()
                != Some(StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1)
            || plan.is_operator_slice()
            || plan.prefix_capacity() != 2
            || plan.frontier_delta() != 1
            || plan.required_domains().len() != 112
            || plan.proposed_domains().len() != 112
            || plan.nodes().len() > 8192
            || plan.values().len() > 8192
            || recorded.position() > 1
        {
            return Err(Error::Unsupported(
                "finite full64 explicit source arithmetic record".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn derive(recorded: &TextStepBindings) -> Result<Self> {
        Self::require_record(recorded)?;
        let layout = Self::derive_source(recorded)?;
        if layout.parameters.len() != 353
            || layout.parameter_bytes > 11 * 1024 * 1024
            || layout.states.iter().any(|state| match state.layout {
                StateLayout::Kv { slab, capacity } => slab != 1024 || capacity != 2,
                StateLayout::Vector { elements } => {
                    elements
                        != match state.slot {
                            StateSlot::Conv(_) => 10240 * 4,
                            StateSlot::Recurrent(_) => 48 * 128 * 128,
                            _ => return true,
                        }
                }
            })
        {
            return Err(Error::Unsupported(
                "finite original parameter/state inventory".into(),
            ));
        }
        Ok(layout)
    }
    pub(super) fn derive_source(recorded: &TextStepBindings) -> Result<Self> {
        Self::require_source_record(recorded)?;
        let plan = recorded.plan();
        let physical = plan
            .values()
            .iter()
            .zip(plan.precisions())
            .map(|(v, precision)| {
                if !matches!(*precision, zerogravity_core::accelerator::program::ActivationPrecision::F32
                    | zerogravity_core::accelerator::program::ActivationPrecision::Bf16WidenedF32) {
                    return Err(Error::Unsupported("source explicit widened-F32 physical storage".into()));
                }
                ResidentBufferLayout::new(ResidentPhysicalType::F32, v.capacity_elements()?)
            })
            .collect::<Result<Vec<_>>>()?;
        let storage = ResidentBufferLayout::plan_values(
            plan,
            &physical,
            ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &[],
                taps: &[],
            },
            ValueStorageLimits {
                max_nodes: 8192,
                max_values: 8192,
                max_slots: 512,
                max_slot_bytes: 4 * 1024 * 1024,
                max_total_bytes: 256 * 1024 * 1024,
                alignment: ALIGNMENT,
                reuse_policy: ValueStorageReusePolicy::ExactAlignedSize,
            },
        )?;
        storage.require_plan(plan)?;
        let mut parameters = BTreeMap::<String, SourceParameter>::new();
        let mut insert = |p: &SourceParameter| -> Result<()> {
            if p.source != plan.source()
                || p.name.is_empty()
                || p.bytes > 1024 * 1024
                || !matches!(
                    p.kind,
                    StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32)
                        | StoredTensorEncoding::Scalar(
                            zerogravity_core::encoding::ScalarDType::Bf16
                        )
                )
            {
                return Err(Error::State("authentic finite source parameter".into()));
            }
            if let Some(old) = parameters.get(&p.name) {
                if old != p {
                    return Err(Error::State("source parameter binding changed".into()));
                }
            } else {
                parameters.insert(p.name.clone(), p.clone());
            }
            Ok(())
        };
        let mut states = BTreeMap::new();
        for node in plan.nodes() {
            match &node.operation {
                PlanOperation::TypedOrdinaryRms { parameter, .. } => insert(parameter)?,
                PlanOperation::TypedTorchGdn { contract, .. } => {
                    contract.validate(plan.source())?;
                    for p in [
                        &contract.convolution,
                        &contract.a_log,
                        &contract.dt_bias,
                        &contract.norm,
                    ] {
                        insert(p)?;
                    }
                }
                PlanOperation::TypedSourceStateWrite { binding } => {
                    let effect = node
                        .state
                        .ok_or_else(|| Error::State("source write effect absent".into()))?;
                    if effect.write.slot != binding.slot()
                        || effect.read.ordinal != 0
                        || effect.write.ordinal != 1
                    {
                        return Err(Error::State("source original state-effect chain".into()));
                    }
                    let layout = super::state_layout::derive(recorded, node, binding)?;
                    let bytes = match layout {
                        StateLayout::Vector { elements } => elements * 4,
                        StateLayout::Kv { slab, capacity } => 2 * slab * capacity * 4,
                    };
                    if states
                        .insert(
                            binding.slot(),
                            StateRequirement {
                                slot: binding.slot(),
                                layout,
                                physical: ResidentPhysicalType::F32,
                                bytes,
                            },
                        )
                        .is_some()
                    {
                        return Err(Error::State("source duplicate proposed domain".into()));
                    }
                }
                _ => {}
            }
        }
        if states.len() != plan.required_domains().len()
            || states
                .keys()
                .copied()
                .ne(plan.required_domains().iter().copied())
        {
            return Err(Error::State("source state owner inventory".into()));
        }
        let max_columns = plan
            .nodes()
            .iter()
            .filter_map(|n| match &n.operation {
                PlanOperation::TypedProject { tensor, .. } => Some(tensor.columns),
                _ => None,
            })
            .max()
            .ok_or_else(|| Error::State("source projection list empty".into()))?;
        if max_columns > 32768 || !max_columns.is_multiple_of(64) {
            return Err(Error::Unsupported(
                "source full-K finite packed width".into(),
            ));
        }
        let parameter_bytes = sum(parameters
            .values()
            .map(|p| {
                let elements = p.shape.iter().product::<u64>() as usize;
                aligned(elements * 4)
            })
            .collect::<Result<Vec<_>>>()?)?;
        let mut math = math::scratch_layout().to_vec();
        for node in plan.nodes() {
            if let PlanOperation::TypedTorchAttention { geometry, position } = node.operation {
                let sizes = crate::source_attention::single_row::scratch_bytes(geometry, position)?;
                for (index, bytes) in [21, 22, 25, 26, 27, 28, 23, 24].into_iter().zip(sizes) {
                    math[index] = math[index].max(bytes);
                }
            }
        }
        let math = math.into_iter().map(aligned).collect::<Result<Vec<_>>>()?;
        let slots = storage
            .slots()
            .iter()
            .map(|s| s.reserved_bytes)
            .collect::<Vec<_>>();
        let root_bytes = sum(states.values().map(|s| s.bytes))?;
        let tables = sum(math::immutable_tables()
            .iter()
            .map(|t| aligned(t.len()))
            .collect::<Result<Vec<_>>>()?)?;
        let owned_bytes = sum([
            storage.total_bytes(),
            parameter_bytes,
            sum(math.iter().copied())?,
            source_page::PAGE_BYTES,
            aligned(max_columns * 2)?,
            2 * aligned(2048 * 4)?,
            2 * ALIGNMENT,
            tables,
        ])?;
        Ok(Self {
            slots,
            bindings: storage.bindings().to_vec(),
            parameters,
            math,
            states: states.into_values().collect(),
            value_bytes: storage.total_bytes(),
            parameter_bytes,
            owned_bytes,
            root_bytes,
            max_columns,
        })
    }
}
