//! Source-derived owners for both finite token records; no admission authority.
use super::*;
use zerogravity_core::accelerator::program::{
    ProgramNumericalProfile, SourceParameter, StandaloneExpProfile, ValueStorageBinding,
    ValueStorageLimits, ValueStorageRetention, ValueStorageReusePolicy,
};

pub(super) const ALIGNMENT: usize = 256;
pub(super) const RECORD_BYTES: usize = 4 * 1024 * 1024;
pub(super) const METADATA_BYTES: usize = 4 * 1024 * 1024;
#[derive(Clone, Debug)]
pub struct QwenSourceResourceInventory {
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
    pub(super) fn derive(recorded: &RecordedWindowToken) -> Result<Self> {
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
        let physical = plan
            .values()
            .iter()
            .map(|v| ResidentBufferLayout::new(ResidentPhysicalType::F32, v.capacity_elements()?))
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
                || !matches!(p.kind, QuantKind::F32 | QuantKind::Bf16)
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
                    let layout = match binding.slot() {
                        StateSlot::Kv(_) => {
                            let n = binding.components()[0];
                            if n != binding.components()[1] || n != (recorded.position() + 1) * 1024
                            {
                                return Err(Error::State(
                                    "source authentic compact KV planes".into(),
                                ));
                            }
                            StateLayout::Kv {
                                slab: 1024,
                                capacity: 2,
                            }
                        }
                        StateSlot::Conv(_) => {
                            if binding.components() != [10240 * 4] {
                                return Err(Error::State("source fullConv4 state".into()));
                            }
                            StateLayout::Vector {
                                elements: 10240 * 4,
                            }
                        }
                        StateSlot::Recurrent(_) => {
                            if binding.components() != [48 * 128 * 128] {
                                return Err(Error::State("source H48 key-major state".into()));
                            }
                            StateLayout::Vector {
                                elements: 48 * 128 * 128,
                            }
                        }
                    };
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
        if states.len() != 112
            || states
                .keys()
                .copied()
                .ne(plan.required_domains().iter().copied())
        {
            return Err(Error::State("full112 source state owner inventory".into()));
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
        // The authenticated complete64 record has129 ordinary hidden RMS,
        // 32 attention Q/K gammas and48 sets of4 original GDN parameters.
        if parameters.len() != 353 || parameter_bytes > 11 * 1024 * 1024 {
            return Err(Error::Unsupported(
                "finite original raw parameter inventory".into(),
            ));
        }
        let math = math::scratch_layout()
            .iter()
            .map(|n| aligned(*n))
            .collect::<Result<Vec<_>>>()?;
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
