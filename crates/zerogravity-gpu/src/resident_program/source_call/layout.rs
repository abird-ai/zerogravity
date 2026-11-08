//! Allocation-free lowering of the original CPU-arithmetic source template.
use super::*;
use zerogravity_core::accelerator::program::{
    ProgramNumericalProfile, SourceAttentionMask, SourceGamma, SourceGdnContract,
    SourceOperatorKind, SourceParameter, SourceRowSelection, SourceStateAccess, SourceTensor,
    StandaloneExpProfile, ValueStorageBinding, ValueStorageLimits, ValueStorageRetention,
    ValueStorageReusePolicy,
};
use zerogravity_core::encoding::{ScalarDType, StoredTensorEncoding};

pub(super) const ALIGNMENT: usize = 256;
pub(super) const MAX_ROWS: usize = 64;
pub(super) const RMS: std::ops::Range<usize> = 0..8;
pub(super) const PROJECT: usize = 8;
pub(super) const QUERY: usize = 9;
pub(super) const KEY: usize = 10;
pub(super) const VALUE: usize = 11;
pub(super) const BETA_INPUT: usize = 12;
pub(super) const BETA: usize = 13;
pub(super) const LOG_DECAY: usize = 14;
pub(super) const A_EXP: usize = 15;
pub(super) const CORE: usize = 16;
pub(super) const NORMAL: usize = 17;
pub(super) const DECAYED: usize = 18;
pub(super) const MEMORY: usize = 19;
pub(super) const DELTA: usize = 20;
pub(super) const DECAY: usize = 21;
pub(super) const ANGLES: usize = 22;
pub(super) const TRIG: usize = 23;
pub(super) const SIGMOID_SHIFT: usize = 24;
pub(super) const CHUNK: std::ops::Range<usize> = 25..44;
pub(super) const CHUNK_SCRATCH: std::ops::Range<usize> = 44..46;
pub(super) const ATTENTION: std::ops::Range<usize> = 46..53;

pub(super) struct Layout {
    pub plan: Arc<ForwardPlan>,
    pub bindings: Vec<ValueStorageBinding>,
    pub slots: Vec<usize>,
    pub matrices: BTreeMap<String, SourceTensor>,
    pub parameters: BTreeMap<String, SourceParameter>,
    pub scratch: Vec<usize>,
    pub max_columns: usize,
    pub projection_scratch_words: usize,
    pub vocabulary: usize,
    pub preflight: ResidentPreflight,
}

pub(super) fn aligned(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(ALIGNMENT - 1)
        .map(|n| n & !(ALIGNMENT - 1))
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("source-call aligned extent".into()))
}
pub(super) fn sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| Error::Limit("source-call owner byte sum".into()))
    })
}
fn f32_vector(slot: StateSlot, elements: usize) -> Result<StateRequirement> {
    let layout = ResidentBufferLayout::new(ResidentPhysicalType::F32, elements)?;
    Ok(StateRequirement {
        slot,
        layout: StateLayout::Vector { elements },
        physical: ResidentPhysicalType::F32,
        bytes: layout.bytes()?,
    })
}
fn insert<T: PartialEq + Clone>(
    map: &mut BTreeMap<String, T>,
    name: &str,
    value: &T,
) -> Result<()> {
    if let Some(old) = map.get(name) {
        if old != value {
            return Err(Error::State(
                "source-call immutable descriptor changed".into(),
            ));
        }
    } else {
        map.insert(name.to_owned(), value.clone());
    }
    Ok(())
}

impl Layout {
    pub(super) fn derive(plan: Arc<ForwardPlan>) -> Result<Self> {
        if plan.source_call().is_some()
            || plan.frontier_delta() != 0
            || plan.is_operator_slice()
            || plan.numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || plan.standalone_exp_profile()
                != Some(StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1)
            || !(1..=32768).contains(&plan.prefix_capacity())
            || plan.nodes().is_empty()
            || plan.nodes().len() > 8192
            || plan.values().is_empty()
            || plan.values().len() > 8192
            || plan.required_domains().is_empty()
            || plan.required_domains().len() > 128
            || plan.required_domains().len() != plan.proposed_domains().len()
            || plan.logits().is_none()
        {
            return Err(Error::Unsupported(
                "source-call static CPU arithmetic/state template".into(),
            ));
        }
        let physical = plan
            .values()
            .iter()
            .map(|s| ResidentBufferLayout::new(ResidentPhysicalType::F32, s.capacity_elements()?))
            .collect::<Result<Vec<_>>>()?;
        let storage = ResidentBufferLayout::plan_values(
            &plan,
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
                max_slot_bytes: 16 * 1024 * 1024,
                max_total_bytes: 256 * 1024 * 1024,
                alignment: ALIGNMENT,
                reuse_policy: ValueStorageReusePolicy::ExactAlignedSize,
            },
        )?;
        storage.require_plan(&plan)?;
        let slots = storage
            .slots()
            .iter()
            .map(|s| s.reserved_bytes)
            .collect::<Vec<_>>();
        let bindings = storage.bindings().to_vec();
        let mut matrices = BTreeMap::new();
        let mut parameters = BTreeMap::new();
        let mut states = BTreeMap::new();
        let mut max_columns = 0;
        let mut max_matrix_rows = 0;
        let mut project = 0;
        let mut sigmoid_width = 0;
        let mut channels = 0;
        let mut heads = 0;
        let mut qheads = 0;
        let mut kvheads = 0;
        let mut rms = 0;
        let mut vocabulary = 0;
        for node in plan.nodes() {
            let PlanOperation::SourceOperatorTemplate { descriptor: d, .. } = &node.operation
            else {
                return Err(Error::Unsupported(
                    "source-call contains a non-template operation".into(),
                ));
            };
            if d.source() != plan.source()
                || d.numerical_contract() != SourceNumericalContract::TorchCpuBf16Avx2MixedSourceV1
                || d.max_rows() != MAX_ROWS
                || vocabulary != 0 && vocabulary != d.vocabulary()
            {
                return Err(Error::State(
                    "source-call template source/contract/geometry".into(),
                ));
            }
            vocabulary = d.vocabulary();
            for m in d.matrices() {
                if m.kind != StoredTensorEncoding::Scalar(ScalarDType::Bf16)
                    || m.columns == 0
                    || !m.columns.is_multiple_of(64)
                {
                    return Err(Error::Unsupported(
                        "source-call original BF16 full-K matrix".into(),
                    ));
                }
                insert(&mut matrices, &m.name, m)?;
                max_columns = max_columns.max(m.columns);
                max_matrix_rows = max_matrix_rows.max(m.rows);
            }
            for p in d.parameters() {
                if !matches!(
                    p.kind,
                    StoredTensorEncoding::Scalar(ScalarDType::Bf16 | ScalarDType::F32)
                ) {
                    return Err(Error::Unsupported(
                        "source-call exact learned parameter codec".into(),
                    ));
                }
                insert(&mut parameters, &p.name, p)?;
            }
            let state = match d.kind() {
                SourceOperatorKind::EmbeddingRows => None,
                SourceOperatorKind::Projection { selected_row: None } => {
                    project = project.max(d.output_elements()?.into_iter().max().unwrap_or(0));
                    None
                }
                SourceOperatorKind::Rms(g) | SourceOperatorKind::ResidualRms(g) => {
                    if g.width == 0
                        || g.width > 5120
                        || !g.width.is_multiple_of(32)
                        || g.epsilon.to_bits() != 1e-6f32.to_bits()
                        || g.gamma != SourceGamma::OnePlus
                    {
                        return Err(Error::Unsupported(
                            "source-call source RMS geometry/gamma".into(),
                        ));
                    }
                    rms = rms.max(g.width);
                    None
                }
                SourceOperatorKind::CausalConvSplit {
                    domain,
                    geometry: g,
                } => {
                    if g.kernel != 4
                        || g.qkv_channels == 0
                        || g.qkv_channels > 10240
                        || !g.qkv_channels.is_multiple_of(128)
                        || g.z_channels == 0
                        || g.z_channels > g.qkv_channels
                        || g.z_channels > 6144
                        || !g.z_channels.is_multiple_of(128)
                        || g.coefficient_channels != 2 * g.z_channels / 128
                    {
                        return Err(Error::Unsupported(
                            "source-call complete Conv4 geometry".into(),
                        ));
                    }
                    channels = channels.max(g.qkv_channels);
                    Some(f32_vector(StateSlot::Conv(domain), g.qkv_channels * 4)?)
                }
                SourceOperatorKind::GatedDelta {
                    domain,
                    geometry: g,
                } => {
                    if g.contract != SourceGdnContract::TorchCpuAvx2L2SiluBf16V1
                        || g.key_width != 128
                        || g.value_width != 128
                        || !(1..=16).contains(&g.key_heads)
                        || !(1..=48).contains(&g.value_heads)
                        || !g.value_heads.is_multiple_of(g.key_heads)
                        || !g.value_heads.is_multiple_of(16)
                        || g.epsilon.to_bits() != 1e-6f32.to_bits()
                    {
                        return Err(Error::Unsupported(
                            "source-call original chunk/cached GDN geometry".into(),
                        ));
                    }
                    heads = heads.max(g.value_heads);
                    rms = rms.max(g.value_heads * 128);
                    Some(f32_vector(
                        StateSlot::Recurrent(domain),
                        g.value_heads * 16384,
                    )?)
                }
                SourceOperatorKind::QkRmsRotaryGated(g) => {
                    crate::source_attention::rows::Geometry::new(
                        g.attention.query_heads,
                        g.attention.kv_heads,
                        plan.prefix_capacity(),
                    )?;
                    if g.attention.width != 256
                        || g.rotary_dimensions != 64
                        || g.theta.to_bits() != 10_000_000f32.to_bits()
                        || g.epsilon.to_bits() != 1e-6f32.to_bits()
                    {
                        return Err(Error::Unsupported(
                            "source-call source RMS/rotary profile".into(),
                        ));
                    }
                    qheads = qheads.max(g.attention.query_heads);
                    kvheads = kvheads.max(g.attention.kv_heads);
                    rms = rms.max(g.attention.query_heads * 256);
                    None
                }
                SourceOperatorKind::KvAppend {
                    domain,
                    key_width,
                    packed_value,
                } => {
                    if key_width == 0
                        || key_width > 1024
                        || !key_width.is_multiple_of(256)
                        || packed_value.width != key_width
                    {
                        return Err(Error::Unsupported("source-call original KV extent".into()));
                    }
                    let layout = StateLayout::Kv {
                        slab: key_width,
                        capacity: plan.prefix_capacity(),
                    };
                    Some(StateRequirement {
                        slot: StateSlot::Kv(domain),
                        layout,
                        physical: ResidentPhysicalType::F32,
                        bytes: key_width * plan.prefix_capacity() * 8,
                    })
                }
                SourceOperatorKind::FlashAttention {
                    geometry: g,
                    scale,
                    mask,
                    ..
                } => {
                    crate::source_attention::rows::Geometry::new(
                        g.query_heads,
                        g.kv_heads,
                        plan.prefix_capacity(),
                    )?;
                    if g.width != 256
                        || scale.to_bits() != 0.0625f32.to_bits()
                        || mask != SourceAttentionMask::Causal
                    {
                        return Err(Error::Unsupported(
                            "source-call complete attention profile".into(),
                        ));
                    }
                    qheads = qheads.max(g.query_heads);
                    kvheads = kvheads.max(g.kv_heads);
                    None
                }
                SourceOperatorKind::Sigmoid { width }
                | SourceOperatorKind::Multiply { width }
                | SourceOperatorKind::SiluAndMul { width } => {
                    if width == 0 || width > 32768 || !width.is_multiple_of(32) {
                        return Err(Error::Unsupported("source-call activation geometry".into()));
                    }
                    if matches!(d.kind(), SourceOperatorKind::Sigmoid { .. }) {
                        sigmoid_width = sigmoid_width.max(width);
                    }
                    None
                }
                _ => {
                    return Err(Error::Unsupported(
                        "source-call kernel implementation absent".into(),
                    ));
                }
            };
            if let Some(s) = state {
                s.plane_layout()?;
                let effect = node
                    .state
                    .ok_or_else(|| Error::State("source-call state effect absent".into()))?;
                if effect.read.slot != s.slot
                    || effect.write.slot != s.slot
                    || effect.read.ordinal != 0
                    || effect.write.ordinal != 1
                    || d.state_access() != SourceStateAccess::Write(s.slot)
                    || states.insert(s.slot, s).is_some()
                {
                    return Err(Error::State(
                        "source-call original state effect chain".into(),
                    ));
                }
            }
        }
        let mut ordered = Vec::new();
        for (slot, version) in plan.required_domains().iter().zip(plan.proposed_domains()) {
            if version.slot != *slot || version.ordinal != 1 {
                return Err(Error::State("source-call proposed domain version".into()));
            }
            ordered.push(
                states
                    .remove(slot)
                    .ok_or_else(|| Error::State("source-call missing state definition".into()))?,
            );
        }
        if !states.is_empty()
            || heads == 0
            || qheads == 0
            || channels == 0
            || rms == 0
            || matrices.is_empty()
        {
            return Err(Error::Unsupported(
                "source-call incomplete source math population".into(),
            ));
        }
        let last = plan
            .nodes()
            .last()
            .ok_or_else(|| Error::State("source-call head absent".into()))?;
        let PlanOperation::SourceOperatorTemplate {
            descriptor: head, ..
        } = &last.operation
        else {
            unreachable!()
        };
        if head.row_selection() != SourceRowSelection::LastActive
            || head.output_elements()? != [vocabulary]
            || last.outputs.as_slice() != [plan.logits().unwrap()]
        {
            return Err(Error::Unsupported(
                "source-call last-row complete-vocabulary readout".into(),
            ));
        }
        let v = MAX_ROWS * heads * 128;
        let a = MAX_ROWS * qheads * 256;
        let rms = rms * MAX_ROWS;
        let mut scratch = vec![
            rms,
            64,
            64,
            64,
            64,
            64,
            rms,
            rms,
            project,
            v.max(a),
            v.max(MAX_ROWS * kvheads * 256),
            v.max(a),
            MAX_ROWS * heads,
            MAX_ROWS * heads,
            MAX_ROWS * heads,
            heads,
            v.max(a),
            v.max(a),
            heads * 16384,
            heads * 128,
            heads * 128,
            heads,
            32,
            64,
        ];
        // Shift words are F64, all other scratch words are F32.
        scratch.push(MAX_ROWS * sigmoid_width.max(heads) * 2);
        let g = crate::source_gdn::chunk::execution::Geometry::new(MAX_ROWS, heads)?;
        scratch.extend(g.output_words());
        scratch.extend(g.scratch_words());
        let attention =
            crate::source_attention::rows::Geometry::new(qheads, kvheads, plan.prefix_capacity())?;
        scratch.extend(attention.scratch_words());
        if scratch.len() != ATTENTION.end {
            return Err(Error::State("source-call scratch schema".into()));
        }
        let scratch = scratch
            .into_iter()
            .map(|n| aligned(n * 4))
            .collect::<Result<Vec<_>>>()?;
        let source_bytes = sum(matrices
            .values()
            .map(|m| {
                usize::try_from(m.packed_bytes()?)
                    .map_err(|_| Error::Limit("source-call matrix bytes".into()))
            })
            .collect::<Result<Vec<_>>>()?)?;
        let parameter_bytes = sum(parameters
            .values()
            .map(|p| {
                let n = p
                    .shape
                    .iter()
                    .try_fold(1usize, |a, b| {
                        a.checked_mul(usize::try_from(*b).ok()?).filter(|n| *n > 0)
                    })
                    .ok_or_else(|| Error::Limit("source-call parameter shape".into()))?;
                aligned(
                    n.checked_mul(4)
                        .ok_or_else(|| Error::Limit("source-call parameter bytes".into()))?,
                )
            })
            .collect::<Result<Vec<_>>>()?)?;
        let root_bytes = sum(ordered.iter().map(|s| s.bytes))?;
        // Largest complete projection output also bounds each constituent matrix.
        let projection_scratch_words = max_matrix_rows.max(project);
        let workspace_bytes = sum([
            sum(scratch.iter().copied())?,
            crate::resident_leases::source_page::PAGE_BYTES,
            aligned(MAX_ROWS * max_columns * 2)?,
            2 * aligned(projection_scratch_words * 4)?,
        ])?;
        let tables = super::tables()
            .into_iter()
            .map(|t| aligned(t.len()))
            .collect::<Result<Vec<_>>>()?;
        let preflight = ResidentPreflight {
            source: plan.source(),
            source_bytes,
            value_bytes: sum(slots.iter().copied())?,
            workspace_bytes,
            status_bytes: ALIGNMENT,
            parameter_bytes: sum([parameter_bytes, sum(tables)?, ALIGNMENT])?,
            prior_state_bytes: root_bytes,
            candidate_state_bytes: root_bytes,
            states: ordered,
            scratch: [0; super::super::SCRATCH_SLOTS],
        };
        preflight.total_with_state()?;
        Ok(Self {
            plan,
            bindings,
            slots,
            matrices,
            parameters,
            scratch,
            max_columns,
            projection_scratch_words,
            vocabulary,
            preflight,
        })
    }
}
