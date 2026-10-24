# Paused continuation text — volume 18

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-gpu/src/resident_program.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 106702; SHA256 `478a0ce0df73dcb043195e9d0deaf84ddb881dda6e39e1ee1ebd352c3333a679`.

````text
//! H16 typed ForwardPlan -> resident physical-kernel bindings.
//!
//! This module contains no Qwen layer schedule and no activation mathematics on
//! the host. It does not admit work, fence a device, publish a root, or declare
//! Ready. The enclosing physical owner must admit the complete preflight,
//! retain `handles_for_attempt` BEFORE launch, and retain them after launch
//! until actual completion or quarantine. One mutable program is one execution
//! workspace; reuse is permitted only after that owner's completion fence.

use crate::{
    activation_quant::ActivationLayout,
    device_activation::DeviceActivation,
    host_descriptor::StaticSource,
    source_accumulation::{self, SourceAccumulation},
};
use cubecl::{client::Client, prelude::*, server::Handle};
use cubecl_common::device::ServiceId;
use cubecl_runtime::memory_management::ManagedMemoryId;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::Arc,
};
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::{
    MatVecRegion, SourceIdentity,
    program::{
        ForwardPlan, PlanNode, PlanOperation, RopeGeometry, SourceTensor, StateSlot, StateVersion,
        TensorView, ValueId,
    },
};
use zerogravity_core::source::TensorSource;
#[cfg(all(test, feature = "qualification"))]
use zerogravity_model_qwen35::LoadedQwenModel;

const SCRATCH_SLOTS: usize = 12;

mod batch;
pub(crate) use batch::BatchProgram;
#[cfg(all(feature = "qualification", feature = "cuda"))]
pub(crate) use batch::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay};

#[cfg(feature = "qualification")]
#[path = "resident_projection_qualification.rs"]
mod projection_qualification;

/// Allocation coordinates only. Original handles and admitted leases remain
/// the lifetime owners; this copyable metadata never pins or promotes memory.
#[derive(Clone, Debug)]
struct BufferRange {
    service: ServiceId,
    memory: ManagedMemoryId,
    bytes: Range<u64>,
}
impl BufferRange {
    fn new(handle: &Handle) -> Result<Self> {
        let start = handle.offset_start.unwrap_or(0);
        let end = handle.size().checked_sub(handle.offset_end.unwrap_or(0));
        let end = end
            .filter(|&end| start < end)
            .ok_or_else(|| Error::State("resident buffer byte range invalid".into()))?;
        Ok(Self {
            service: handle.service,
            memory: handle.clone().binding().memory.id(),
            bytes: start..end,
        })
    }
    fn overlaps(&self, other: &Self) -> bool {
        self.service == other.service
            && self.memory == other.memory
            && self.bytes.start < other.bytes.end
            && other.bytes.start < self.bytes.end
    }
}

/// Exact immutable ranges, prepared once. Distinct slices of a shared
/// allocation remain separate; only identical aliases are deduplicated.
struct OwnedRanges {
    service: ServiceId,
    by_memory: BTreeMap<ManagedMemoryId, Vec<Range<u64>>>,
}
impl OwnedRanges {
    fn new(service: ServiceId) -> Self {
        Self {
            service,
            by_memory: BTreeMap::new(),
        }
    }
    fn insert(&mut self, range: BufferRange) -> Result<()> {
        if range.service != self.service {
            return Err(Error::State("resident owned range service differs".into()));
        }
        let ranges = self.by_memory.entry(range.memory).or_default();
        if !ranges.contains(&range.bytes) {
            ranges.push(range.bytes);
        }
        Ok(())
    }
    fn overlaps(&self, range: &BufferRange) -> bool {
        range.service == self.service
            && self.by_memory.get(&range.memory).is_some_and(|ranges| {
                ranges
                    .iter()
                    .any(|r| r.start < range.bytes.end && range.bytes.start < r.end)
            })
    }
}
fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| Error::Limit("resident program byte sum overflow".into()))
    })
}
fn bytes(elements: usize, width: usize) -> Result<usize> {
    elements
        .checked_mul(width)
        .filter(|_| u32::try_from(elements).is_ok())
        .ok_or_else(|| Error::Limit("resident program buffer index/byte overflow".into()))
}
fn grid(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
}
fn threads() -> CubeDim {
    CubeDim::new_1d(32)
}
fn arg(handle: &Handle, elements: usize) -> BufferArg {
    // SAFETY: all call sites are internal to the checked typed lowering;
    // preflight/state binding validates the physical byte ranges separately.
    unsafe { BufferArg::from_raw_parts(handle.clone(), elements) }
}

/// Every View is already physically packed by this lowerer. Contiguous can
/// therefore name that same immutable dense value without allocating/copying.
/// Resolve chains to earlier original values; never alias a range or a state.
fn contiguous_aliases(plan: &ForwardPlan) -> Result<Vec<Option<ValueId>>> {
    let mut aliases = vec![None; plan.values().len()];
    let mut produced = BTreeSet::new();
    for node in plan.nodes() {
        if matches!(node.operation, PlanOperation::Contiguous) {
            let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                return Err(Error::Malformed("resident contiguous alias arity".into()));
            };
            let shape = |id: ValueId| {
                plan.values()
                    .get(id.0 as usize)
                    .ok_or_else(|| Error::Malformed("resident alias value index".into()))
            };
            if node.state.is_some()
                || input.0 >= output.0
                || !produced.contains(&input.0)
                || shape(*input)? != shape(*output)?
            {
                return Err(Error::State(
                    "resident contiguous alias is not an equal-shape earlier value".into(),
                ));
            }
            aliases[output.0 as usize] = Some(aliases[input.0 as usize].unwrap_or(*input));
        }
        produced.extend(node.outputs.iter().map(|id| id.0));
    }
    Ok(aliases)
}

#[derive(Clone, Debug)]
pub(crate) struct DeviceBuffer {
    pub handle: Handle,
    pub elements: usize,
}
impl DeviceBuffer {
    pub fn new(handle: Handle, elements: usize) -> Self {
        Self { handle, elements }
    }
    fn check(&self, client: &Client, expected: usize) -> Result<BufferRange> {
        let range = BufferRange::new(&self.handle)?;
        if self.elements != expected
            || range.service != client.service_id()
            || range.bytes.end - range.bytes.start < bytes(expected, 4)? as u64
        {
            return Err(Error::State(
                "resident program binding size/service mismatch".into(),
            ));
        }
        Ok(range)
    }
}
#[derive(Clone, Debug)]
pub(crate) enum StateBuffers {
    Vector(DeviceBuffer),
    Kv {
        keys: DeviceBuffer,
        values: DeviceBuffer,
    },
}
impl StateBuffers {
    pub fn buffers(&self) -> Vec<&DeviceBuffer> {
        match self {
            Self::Vector(v) => vec![v],
            Self::Kv { keys, values } => vec![keys, values],
        }
    }
    pub fn handles(&self) -> Vec<Handle> {
        match self {
            Self::Vector(v) => vec![v.handle.clone()],
            Self::Kv { keys, values } => vec![keys.handle.clone(), values.handle.clone()],
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct StateBinding {
    pub slot: StateSlot,
    pub prior: StateBuffers,
    pub candidate: StateBuffers,
    /// Committed prefix length; meaningful for paired KV only.
    pub prior_tokens: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct RuntimeParameters {
    pub token: u32,
    pub position: usize,
    pub valid_tokens: usize,
}
enum CausalMask<'a> {
    Uniform(usize),
    Rows {
        positions: &'a Handle,
        heads_per_row: usize,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidentOutput {
    StateOnly,
    Logits,
}
impl ResidentOutput {
    fn nodes(self, plan: &ForwardPlan) -> &[PlanNode] {
        match self {
            Self::StateOnly => plan.state_nodes(),
            Self::Logits => plan.nodes(),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum StateLayout {
    Vector { elements: usize },
    Kv { slab: usize, capacity: usize },
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct StateRequirement {
    pub slot: StateSlot,
    pub layout: StateLayout,
    pub bytes: usize,
}

/// Exact logical allocations, excluding allocator page rounding. Prior and
/// candidate state are external bindings and reported separately to prevent
/// double charging an already-resident committed root.
#[derive(Clone, Debug)]
pub(crate) struct ResidentPreflight {
    pub source: SourceIdentity,
    pub source_bytes: usize,
    pub value_bytes: usize,
    pub workspace_bytes: usize,
    pub status_bytes: usize,
    pub parameter_bytes: usize,
    pub prior_state_bytes: usize,
    pub candidate_state_bytes: usize,
    pub states: Vec<StateRequirement>,
    scratch: [usize; SCRATCH_SLOTS],
}
impl ResidentPreflight {
    pub fn owned_bytes(&self) -> Result<usize> {
        checked_sum([
            self.source_bytes,
            self.value_bytes,
            self.workspace_bytes,
            self.status_bytes,
            self.parameter_bytes,
        ])
    }
    pub fn total_with_state(&self) -> Result<usize> {
        checked_sum([
            self.owned_bytes()?,
            self.prior_state_bytes,
            self.candidate_state_bytes,
        ])
    }
}
struct ResidentTensor {
    packed: Handle,
    elements: usize,
    scales: Handle,
    biases: Handle,
    layout: Option<ActivationLayout>,
}
struct NodeStatus {
    block: Option<Handle>,
    row: Option<Handle>,
}

pub(crate) struct ResidentProgram {
    // The immutable original allocation remains alive for every resident handle.
    _source_owner: Arc<dyn TensorSource>,
    plan: ForwardPlan,
    preflight: ResidentPreflight,
    tensors: BTreeMap<String, ResidentTensor>,
    constants: Vec<Handle>,
    values: Vec<DeviceBuffer>,
    scratch: [Handle; SCRATCH_SLOTS],
    status: Handle,
    node_status: Vec<NodeStatus>,
    // Immutable per-operation views of existing originals. Preparation neither
    // allocates memory nor initializes values/statuses skipped by state-only.
    projections: Vec<Option<Box<DeviceActivation>>>,
    token: Handle,
    exp_tables: Handle,
    trig_tables: Handle,
    owned_ranges: OwnedRanges,
}
#[derive(Clone, Debug)]
pub(crate) struct ResidentSubmission {
    pub logits: Option<DeviceBuffer>,
    pub proposals: Vec<(StateVersion, StateBuffers)>,
    pub status: Handle,
    /// Every dynamic-quant status pair, in plan order. Zero is success.
    pub quant_status: Vec<(Handle, usize, Handle, usize)>,
    pub embedding_status: Vec<(Handle, usize)>,
    pub dispatches: usize,
    /// Conservative subset: actual native multi-row projection launches.
    pub fused_dispatches: usize,
}

impl ResidentProgram {
    fn require_legacy_contract(plan: &ForwardPlan) -> Result<()> {
        use zerogravity_core::accelerator::program::{
            ActivationPrecision, ProgramNumericalProfile,
        };
        if plan.numerical_profile() != ProgramNumericalProfile::GgufSourceV1
            || plan.is_operator_slice()
            || plan
                .precisions()
                .iter()
                .any(|precision| *precision != ActivationPrecision::F32)
            || plan.nodes().iter().any(|node| {
                matches!(
                    node.operation,
                    PlanOperation::TypedBulkF32Input { .. }
                        | PlanOperation::TypedBulkF32Matmul { .. }
                        | PlanOperation::TypedEmbed { .. }
                        | PlanOperation::TypedProject { .. }
                        | PlanOperation::TypedOrdinaryRms { .. }
                        | PlanOperation::TypedCausalConvSilu { .. }
                )
            })
        {
            return Err(Error::Unsupported(
                "resident backend has no qualified typed numerical contract".into(),
            ));
        }
        Ok(())
    }
    /// Pure metadata/source validation; no GPU allocation or live math.
    pub fn preflight(model: &dyn TensorSource, plan: &ForwardPlan) -> Result<ResidentPreflight> {
        Self::require_legacy_contract(plan)?;
        if plan.source() != model.source_identity()
            || !plan.source().is_bound()
            || plan.frontier_delta() != 1
            || plan.prefix_capacity() == 0
            || plan.prefix_capacity() > 512
            || !plan.prefix_capacity().is_multiple_of(256)
        {
            return Err(Error::Unsupported(
                "resident plan source/capacity/frontier unsupported".into(),
            ));
        }
        let aliases = contiguous_aliases(plan)?;
        let mut source_bytes = checked_sum(plan.constants().iter().map(|c| c.values.len() * 4))?;
        source_bytes = checked_sum([
            source_bytes,
            crate::source_transcendentals::TABLES.len() * 8,
            crate::source_trig::TRIG_TABLES.len() * 8,
            4,
        ])?; // shared no-bias dummy
        let mut seen = BTreeMap::<String, SourceTensor>::new();
        let mut projected = BTreeSet::new();
        let mut scratch = [4usize; SCRATCH_SLOTS];
        let mut status_bytes = 4usize;
        let mut state_specs = BTreeMap::new();
        for constant in plan.constants() {
            if constant.source != plan.source() || constant.values.iter().any(|x| !x.is_finite()) {
                return Err(Error::Malformed(
                    "resident static vector source/finite mismatch".into(),
                ));
            }
        }
        for node in plan.nodes() {
            use zerogravity_core::accelerator::program::ParameterSlot;
            let parameter = match &node.operation {
                PlanOperation::Embed { token, .. } => Some((token.0, ParameterSlot::Token)),
                PlanOperation::Rope { position, .. }
                | PlanOperation::KvPrefixGuard { position, .. } => {
                    Some((position.0, ParameterSlot::Position))
                }
                PlanOperation::AttentionQk { valid_tokens, .. }
                | PlanOperation::AttentionSoftmax { valid_tokens, .. }
                | PlanOperation::AttentionValue { valid_tokens, .. } => {
                    Some((valid_tokens.0, ParameterSlot::ValidTokens))
                }
                _ => None,
            };
            if parameter.is_some_and(|(actual, expected)| actual != expected) {
                return Err(Error::Unsupported(
                    "resident symbolic parameter differs from source schedule".into(),
                ));
            }
            match &node.operation {
                PlanOperation::Recurrent { width }
                    if *width == 0 || !width.is_multiple_of(32) || *width > 256 =>
                {
                    return Err(Error::Unsupported(
                        "resident recurrence requires source width 32..256 multiple32".into(),
                    ));
                }
                PlanOperation::AttentionQk { geometry, .. }
                | PlanOperation::AttentionValue { geometry, .. }
                    if geometry.width == 0
                        || !geometry.width.is_multiple_of(32)
                        || geometry.width > 512
                        || geometry.query_heads > 64
                        || geometry.kv_heads > 64 =>
                {
                    return Err(Error::Unsupported(
                        "resident attention exceeds source geometry bounds".into(),
                    ));
                }
                PlanOperation::Rope { geometry, .. }
                    if geometry.rotate > 64
                        || geometry.attention.width > 512
                        || geometry.attention.query_heads > 64
                        || geometry.attention.kv_heads > 64 =>
                {
                    return Err(Error::Unsupported(
                        "resident RoPE exceeds source geometry bounds".into(),
                    ));
                }
                PlanOperation::Coefficients { dt, a, .. }
                    if plan.constants()[dt.0 as usize]
                        .values
                        .iter()
                        .any(|x| x.abs() > 128.0)
                        || plan.constants()[a.0 as usize]
                            .values
                            .iter()
                            .any(|x| *x > 0.0 || x.abs() > 1_000_000.0) =>
                {
                    return Err(Error::Unsupported(
                        "resident source GDN dt/A outside attested bounds".into(),
                    ));
                }
                _ => {}
            }
            if let PlanOperation::Embed { tensor, .. } | PlanOperation::Project { tensor } =
                &node.operation
            {
                if tensor.source != plan.source() {
                    return Err(Error::State("resident tensor source mismatch".into()));
                }
                // Operator support is independent of source-storage reuse. A
                // tensor first seen by Project must still satisfy Embed's
                // narrower source codec/geometry contract on every use.
                let embedding = if matches!(node.operation, PlanOperation::Embed { .. }) {
                    Some(crate::embedding::EmbeddingLayout::new(
                        tensor.kind,
                        tensor.rows,
                        tensor.columns,
                    )?)
                } else {
                    None
                };
                tensor.validate_source(model)?;
                if matches!(node.operation, PlanOperation::Project { .. })
                    && projected.insert(tensor.name.clone())
                {
                    let block = if tensor.kind == QuantKind::Q6K {
                        256
                    } else {
                        32
                    };
                    if tensor.kind != QuantKind::F32 {
                        source_bytes = checked_sum([
                            source_bytes,
                            bytes(
                                (tensor.columns / block) * tensor.rows,
                                if tensor.kind == QuantKind::Q4_1 { 8 } else { 4 },
                            )?,
                        ])?;
                    }
                }
                if let Some(prior) = seen.get(&tensor.name) {
                    if prior != tensor {
                        return Err(Error::State(
                            "resident source name reused with different geometry".into(),
                        ));
                    }
                } else {
                    let region = region(model, tensor)?;
                    let source = if let Some(embedding) = embedding {
                        embedding.source_bytes
                    } else {
                        StaticSource::gpu_bytes_for(region)?
                    };
                    source_bytes = checked_sum([source_bytes, source])?;
                    seen.insert(tensor.name.clone(), tensor.clone());
                }
            }
            let n = |id: ValueId| -> Result<usize> {
                plan.values()
                    .get(id.0 as usize)
                    .ok_or_else(|| Error::Malformed("resident plan value index invalid".into()))?
                    .capacity_elements()
            };
            let input = |i: usize| -> Result<usize> {
                n(*node
                    .inputs
                    .get(i)
                    .ok_or_else(|| Error::Malformed("resident plan input missing".into()))?)
            };
            let cap = plan.prefix_capacity();
            if let PlanOperation::AttentionQk {
                state, geometry, ..
            }
            | PlanOperation::AttentionValue {
                state, geometry, ..
            } = &node.operation
            {
                let slab = geometry
                    .kv_heads
                    .checked_mul(geometry.width)
                    .ok_or_else(|| Error::Limit("resident attention KV slab overflow".into()))?;
                if !matches!(
                    state_specs.get(&state.slot),
                    Some(StateLayout::Kv { slab: stored, capacity })
                        if *stored == slab && *capacity == cap
                ) {
                    return Err(Error::State(
                        "resident attention KV slab differs from geometry".into(),
                    ));
                }
            }
            for id in node.inputs.iter().chain(node.outputs.iter()) {
                if n(*id)? > crate::program_kernels::MAX_WORD_OPS {
                    return Err(Error::Unsupported(
                        "resident value exceeds bounded transport domain".into(),
                    ));
                }
            }
            match &node.operation {
                PlanOperation::Rms { groups, .. }
                    if *groups > 4096 || input(0)? / groups > 16_384 =>
                {
                    return Err(Error::Unsupported(
                        "resident RMS exceeds source bounds".into(),
                    ));
                }
                PlanOperation::Silu { .. }
                | PlanOperation::Add
                | PlanOperation::SigmoidGate
                | PlanOperation::Coefficients { .. }
                    if input(0)? > 65_536 =>
                {
                    return Err(Error::Unsupported(
                        "resident elementwise/coefficient width exceeds source bounds".into(),
                    ));
                }
                PlanOperation::Conv { .. } if input(0)? > 16_384 => {
                    return Err(Error::Unsupported(
                        "resident convolution exceeds source bounds".into(),
                    ));
                }
                PlanOperation::Recurrent { width } if input(0)? / width > 64 => {
                    return Err(Error::Unsupported(
                        "resident recurrent head count exceeds source bounds".into(),
                    ));
                }
                PlanOperation::AttentionSoftmax { heads, .. } if *heads > 64 => {
                    return Err(Error::Unsupported(
                        "resident softmax head count exceeds source bounds".into(),
                    ));
                }
                _ => {}
            }
            let sizes: Vec<usize> = match &node.operation {
                PlanOperation::Embed { tensor, .. } => {
                    status_bytes = checked_sum([status_bytes, tensor.columns * 4])?;
                    vec![]
                }
                PlanOperation::Project { tensor } if tensor.kind != QuantKind::F32 => {
                    let l = ActivationLayout::new(tensor.kind, tensor.columns, tensor.rows)?;
                    status_bytes = checked_sum([status_bytes, (l.blocks + l.rows) * 4])?;
                    vec![
                        l.blocks * 16,
                        l.columns * 4,
                        l.columns * 4,
                        l.composite_count * 4,
                        if l.with_bias {
                            l.composite_count * 4
                        } else {
                            4
                        },
                    ]
                }
                PlanOperation::Rms { groups, .. } => vec![input(0)? * 4, groups * 8],
                PlanOperation::Silu { .. } => vec![input(0)? * crate::silu::SCRATCH * 4],
                PlanOperation::SigmoidGate => vec![
                    input(0)? * crate::elementwise::SIGMOID_SCRATCH * 4,
                    input(0)? * 8,
                ],
                PlanOperation::Conv { .. } => {
                    state_specs.insert(
                        node.state
                            .ok_or_else(|| Error::Malformed("conv missing effect".into()))?
                            .read
                            .slot,
                        StateLayout::Vector {
                            elements: input(0)? * 3,
                        },
                    );
                    vec![input(0)? * 16]
                }
                PlanOperation::Coefficients { .. } => vec![
                    input(0)? * 16,
                    input(0)? * 24,
                    input(0)? * 24,
                    input(0)? * 16,
                ],
                PlanOperation::Recurrent { width } => {
                    let v = input(0)?;
                    state_specs.insert(
                        node.state
                            .ok_or_else(|| Error::Malformed("recurrence missing effect".into()))?
                            .read
                            .slot,
                        StateLayout::Vector {
                            elements: v * width,
                        },
                    );
                    vec![v * 12, v * width * 4, v * 4, v * 128, v * 64]
                }
                PlanOperation::Rope { geometry, .. } => {
                    let g = geometry.attention;
                    let half = geometry.rotate / 2;
                    let h = g.query_heads + g.kv_heads;
                    let count = input(0)? + input(1)? + input(2)?;
                    vec![
                        count * 4,
                        half * 4,
                        half * 8,
                        h * half * 8,
                        h * half * 16,
                        h * half * 16,
                        count * 4,
                    ]
                }
                PlanOperation::AppendKv => {
                    let slab = input(0)?;
                    state_specs.insert(
                        node.state
                            .ok_or_else(|| Error::Malformed("KV append missing effect".into()))?
                            .read
                            .slot,
                        StateLayout::Kv {
                            slab,
                            capacity: cap,
                        },
                    );
                    vec![]
                }
                PlanOperation::AttentionQk { geometry: g, .. } => {
                    let c = g.query_heads * cap;
                    vec![
                        (g.query_heads * g.width + cap * g.kv_heads * g.width) * 4,
                        c * 128,
                        c * 128,
                        c * 256,
                        c * 256,
                        c * 8,
                    ]
                }
                PlanOperation::AttentionSoftmax { heads: h, .. } => {
                    let n = h * cap;
                    let groups = cap.div_ceil(8);
                    vec![
                        n * 40,
                        n * 4,
                        n * 8,
                        n * 8,
                        h * 4,
                        h * groups * 16,
                        h * groups * 8,
                        n * 4,
                        h * 8,
                        h * 8,
                        h * 4,
                    ]
                }
                PlanOperation::AttentionValue {
                    geometry: g,
                    padding,
                    ..
                } => {
                    if *padding != 256 {
                        return Err(Error::Unsupported(
                            "resident attention value padding differs from source".into(),
                        ));
                    }
                    let c = g.query_heads * g.width;
                    vec![
                        (g.query_heads * cap + cap * g.kv_heads * g.width) * 4,
                        c * 128,
                        c * 128,
                        c * 256,
                        c * 256,
                    ]
                }
                PlanOperation::View(view) => {
                    let ([source], [destination]) =
                        (node.inputs.as_slice(), node.outputs.as_slice())
                    else {
                        return Err(Error::Malformed("resident View arity".into()));
                    };
                    if node.state.is_some()
                        || view.rows.checked_mul(view.width) != Some(n(*destination)?)
                        || source.0 >= destination.0
                    {
                        return Err(Error::Malformed(
                            "resident View must fully write one dense private value".into(),
                        ));
                    }
                    view.validate(input(0)?)?;
                    crate::program_kernels::checked_copy_words(
                        view.rows,
                        view.width,
                        view.row_stride,
                        view.width,
                    )?;
                    vec![]
                }
                _ => vec![],
            };
            for (slot, size) in sizes.into_iter().enumerate() {
                if u32::try_from(size / 4).is_err() {
                    return Err(Error::Limit(
                        "resident workspace exceeds shader index domain".into(),
                    ));
                }
                scratch[slot] = scratch[slot].max(size);
            }
        }
        if state_specs.len() != plan.required_domains().len()
            || plan
                .required_domains()
                .iter()
                .any(|s| !state_specs.contains_key(s))
        {
            return Err(Error::State(
                "resident plan state requirements incomplete".into(),
            ));
        }
        let mut states = Vec::new();
        for (slot, layout) in state_specs {
            let b = match layout {
                StateLayout::Vector { elements } => bytes(elements, 4)?,
                StateLayout::Kv { slab, capacity } => bytes(slab * capacity, 8)?,
            };
            states.push(StateRequirement {
                slot,
                layout,
                bytes: b,
            });
        }
        let state_bytes = checked_sum(states.iter().map(|s| s.bytes))?;
        let value_bytes = checked_sum(
            plan.values()
                .iter()
                .zip(&aliases)
                .filter(|(_, alias)| alias.is_none())
                .map(|(s, _)| s.capacity_elements().and_then(|n| bytes(n, 4)))
                .collect::<Result<Vec<_>>>()?,
        )?;
        Ok(ResidentPreflight {
            source: plan.source(),
            source_bytes,
            value_bytes,
            workspace_bytes: checked_sum(scratch)?,
            status_bytes,
            parameter_bytes: 4,
            prior_state_bytes: state_bytes,
            candidate_state_bytes: state_bytes,
            states,
            scratch,
        })
    }

    /// Check the complete resident shader profile before physical submission.
    pub fn ensure_capability(client: &Client) -> Result<()> {
        source_accumulation::ensure_capability(client, SourceAccumulation::WideF64)?;
        crate::activation_quant_fused::ensure_selected_capability(client)?;
        for t in [
            cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64),
            cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I64),
        ] {
            if !client.properties().features.supports_type(t) {
                return Err(Error::Unsupported(
                    "resident source arithmetic requires shader Int64/UInt64".into(),
                ));
            }
        }
        let ty = cubecl::ir::Type::atomic(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U32));
        let usage = client.properties().atomic_type_usage(ty);
        if !usage.contains(cubecl::ir::features::AtomicUsage::LoadStore)
            || !usage.contains(cubecl::ir::features::AtomicUsage::Bitwise)
        {
            return Err(Error::Unsupported(
                "resident certificates require U32 atomic load/store and bitwise operations".into(),
            ));
        }
        Ok(())
    }

    /// Upload each immutable tensor/vector once, and allocate the exact logical
    /// preflight. Caller admits/retains these allocations before any launch.
    pub fn prepare_with_allocator(
        client: &Client,
        model: Arc<dyn TensorSource>,
        plan: ForwardPlan,
        mut allocate: impl FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>,
    ) -> Result<Self> {
        let preflight = Self::preflight(model.as_ref(), &plan)?;
        Self::ensure_capability(client)?;
        let mut allocation = ProgramAllocator {
            client,
            allocate: &mut allocate,
        };
        let dummy = allocation.upload(ResidentBufferRole::Source, f32::as_bytes(&[0.0]))?;
        let mut tensors = BTreeMap::new();
        let projected: BTreeSet<_> = plan
            .nodes()
            .iter()
            .filter_map(|n| match &n.operation {
                PlanOperation::Project { tensor } => Some(tensor.name.clone()),
                _ => None,
            })
            .collect();
        for node in plan.nodes() {
            let (tensor, embedding) = match &node.operation {
                PlanOperation::Embed { tensor, .. } => (tensor, true),
                PlanOperation::Project { tensor } => (tensor, false),
                _ => continue,
            };
            if tensors.contains_key(&tensor.name) {
                continue;
            }
            let (packed, elements, scales, biases, layout) = if embedding
                && !projected.contains(&tensor.name)
            {
                let b = model.tensor(&tensor.name)?.packed_bytes;
                (
                    allocation.upload(ResidentBufferRole::Source, b)?,
                    b.len(),
                    dummy.clone(),
                    dummy.clone(),
                    None,
                )
            } else {
                let (descriptor, b) = StaticSource::from_region(region(model.as_ref(), tensor)?)?;
                let scales = if descriptor.weight_scales().is_empty() {
                    dummy.clone()
                } else {
                    allocation.upload(
                        ResidentBufferRole::Source,
                        f32::as_bytes(descriptor.weight_scales()),
                    )?
                };
                let biases = if descriptor.biases().is_empty() {
                    dummy.clone()
                } else {
                    allocation.upload(
                        ResidentBufferRole::Source,
                        f32::as_bytes(descriptor.biases()),
                    )?
                };
                (
                    allocation.upload(ResidentBufferRole::Source, &b)?,
                    descriptor.gpu_elements,
                    scales,
                    biases,
                    if tensor.kind == QuantKind::F32 {
                        None
                    } else {
                        Some(ActivationLayout::new(
                            tensor.kind,
                            tensor.columns,
                            tensor.rows,
                        )?)
                    },
                )
            };
            tensors.insert(
                tensor.name.clone(),
                ResidentTensor {
                    packed,
                    elements,
                    scales,
                    biases,
                    layout,
                },
            );
        }
        let constants = plan
            .constants()
            .iter()
            .map(|c| allocation.upload(ResidentBufferRole::Source, f32::as_bytes(&c.values)))
            .collect::<Result<Vec<_>>>()?;
        let aliases = contiguous_aliases(&plan)?;
        let mut values: Vec<DeviceBuffer> = Vec::with_capacity(plan.values().len());
        for (shape, alias) in plan.values().iter().zip(&aliases) {
            let n = shape.capacity_elements()?;
            values.push(if let Some(original) = alias {
                // Clone the complete admitted original, preserving its service,
                // allocation and byte range. No second lease or upload exists.
                values[original.0 as usize].clone()
            } else {
                DeviceBuffer {
                    handle: allocation.empty(ResidentBufferRole::Value, bytes(n, 4)?)?,
                    elements: n,
                }
            });
        }
        let scratch: [Handle; SCRATCH_SLOTS] = (0..SCRATCH_SLOTS)
            .map(|i| allocation.empty(ResidentBufferRole::Workspace, preflight.scratch[i]))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .expect("fixed scratch slot count");
        let node_status = plan
            .nodes()
            .iter()
            .map(|n| {
                Ok(match &n.operation {
                    PlanOperation::Embed { tensor, .. } => NodeStatus {
                        block: Some(
                            allocation.empty(ResidentBufferRole::Status, tensor.columns * 4)?,
                        ),
                        row: None,
                    },
                    PlanOperation::Project { tensor } => {
                        if let Some(l) = tensors[&tensor.name].layout {
                            NodeStatus {
                                block: Some(
                                    allocation.empty(ResidentBufferRole::Status, l.blocks * 4)?,
                                ),
                                row: Some(
                                    allocation.empty(ResidentBufferRole::Status, l.rows * 4)?,
                                ),
                            }
                        } else {
                            NodeStatus {
                                block: None,
                                row: None,
                            }
                        }
                    }
                    _ => NodeStatus {
                        block: None,
                        row: None,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut program = Self {
            _source_owner: model,
            plan,
            preflight,
            tensors,
            constants,
            values,
            scratch,
            node_status,
            projections: Vec::new(),
            status: allocation.empty(ResidentBufferRole::Status, 4)?,
            token: allocation.empty(ResidentBufferRole::Parameter, 4)?,
            exp_tables: allocation.upload(
                ResidentBufferRole::Source,
                f64::as_bytes(&crate::source_transcendentals::TABLES),
            )?,
            trig_tables: allocation.upload(
                ResidentBufferRole::Source,
                f64::as_bytes(&crate::source_trig::TRIG_TABLES),
            )?,
            owned_ranges: OwnedRanges::new(client.service_id()),
        };
        program.projections = program
            .plan
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, node)| match &node.operation {
                PlanOperation::Project { tensor } if tensor.kind != QuantKind::F32 => program
                    .projection_binding(client, index, tensor)
                    .map(|binding| Some(Box::new(binding))),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        for handle in program.owned_handles() {
            program.owned_ranges.insert(BufferRange::new(&handle)?)?;
        }
        Ok(program)
    }

    /// Cold-only construction from the same original per-operation bindings
    /// previously rebuilt at every quantized Project. All handles already exist
    /// and remain owned by the original admitted program/attempt cohort.
    fn projection_binding(
        &self,
        client: &Client,
        index: usize,
        tensor: &SourceTensor,
    ) -> Result<DeviceActivation> {
        let t = &self.tensors[&tensor.name];
        let layout = t
            .layout
            .ok_or_else(|| Error::State("resident quant projection layout absent".into()))?;
        let flags = &self.node_status[index];
        let input = self.plan.nodes()[index].inputs[0];
        DeviceActivation::from_bound(
            client,
            &layout,
            self.values[input.0 as usize].handle.clone(),
            t.scales.clone(),
            t.biases.clone(),
            [
                self.scratch[0].clone(),
                self.scratch[1].clone(),
                self.scratch[2].clone(),
                self.scratch[3].clone(),
                self.scratch[4].clone(),
                flags.block.as_ref().expect("quant block status").clone(),
                flags.row.as_ref().expect("quant row status").clone(),
            ],
        )
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn budget(&self) -> &ResidentPreflight {
        &self.preflight
    }
    pub fn values(&self) -> &[DeviceBuffer] {
        &self.values
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.plan.source()
    }
    /// No handles are created after this cohort is retained. Includes all
    /// original source, static, intermediate, status and external state handles.
    pub fn owned_handles(&self) -> Vec<Handle> {
        self.handles_for_attempt(&[])
    }

    /// Shared source/scratch remain pinned; skipped output values and per-node
    /// statuses are absent so their uninitialized storage cannot become Ready.
    /// This is cold binding metadata, not a second allocation or program.
    pub fn execution_handles(&self, output: ResidentOutput) -> Vec<Handle> {
        let nodes = output.nodes(&self.plan);
        let mut handles = vec![
            self.status.clone(),
            self.token.clone(),
            self.exp_tables.clone(),
            self.trig_tables.clone(),
        ];
        for t in self.tensors.values() {
            handles.extend([t.packed.clone(), t.scales.clone(), t.biases.clone()]);
        }
        handles.extend(self.constants.iter().cloned());
        handles.extend(self.scratch.iter().cloned());
        // SSA aliases retain the same original handle. The enclosing owner
        // selects each admitted lease once; all alias producers precede it in
        // this same closed prefix, including during state-only prefill.
        handles.extend(
            nodes
                .iter()
                .flat_map(|n| &n.outputs)
                .map(|id| self.values[id.0 as usize].handle.clone()),
        );
        for s in &self.node_status[..nodes.len()] {
            handles.extend(s.block.iter().cloned());
            handles.extend(s.row.iter().cloned());
        }
        handles
    }

    pub fn handles_for_attempt(&self, bindings: &[StateBinding]) -> Vec<Handle> {
        let mut handles = self.execution_handles(ResidentOutput::Logits);
        for b in bindings {
            handles.extend(b.prior.handles());
            handles.extend(b.candidate.handles());
        }
        handles
    }
    pub fn validate_runtime(
        &self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
    ) -> Result<()> {
        if self.owned_ranges.service != client.service_id() {
            return Err(Error::State(
                "resident program used with another physical service".into(),
            ));
        }
        if p.valid_tokens
            != p.position
                .checked_add(1)
                .ok_or_else(|| Error::Limit("resident position overflow".into()))?
            || p.valid_tokens > self.plan.prefix_capacity()
            || p.position > 32768
        {
            return Err(Error::Limit(
                "resident runtime prefix exceeds prepared capacity".into(),
            ));
        }
        let mut seen = BTreeMap::new();
        let mut priors = Vec::new();
        let mut candidates = Vec::new();
        for b in bindings {
            if seen.insert(b.slot, b).is_some() {
                return Err(Error::State("duplicate resident state binding".into()));
            }
        }
        if bindings.len() != self.preflight.states.len() {
            return Err(Error::State("resident state binding count mismatch".into()));
        }
        for r in &self.preflight.states {
            let b = seen
                .get(&r.slot)
                .ok_or_else(|| Error::State("missing resident state domain".into()))?;
            for (state, ranges) in [(&b.prior, &mut priors), (&b.candidate, &mut candidates)] {
                match (r.layout, state) {
                    (StateLayout::Vector { elements }, StateBuffers::Vector(v)) => {
                        ranges.push(v.check(client, elements)?);
                    }
                    (StateLayout::Kv { slab, capacity }, StateBuffers::Kv { keys, values }) => {
                        ranges.push(keys.check(client, slab * capacity)?);
                        ranges.push(values.check(client, slab * capacity)?);
                        if b.prior_tokens != p.position {
                            return Err(Error::State(
                                "resident KV prefix differs from bound position".into(),
                            ));
                        }
                    }
                    _ => {
                        return Err(Error::State(
                            "resident state binding has wrong typed layout".into(),
                        ));
                    }
                }
            }
        }
        // Reject overlapping candidate writes against EVERY committed input,
        // sibling candidate and program-owned allocation, even across domains.
        for (i, c) in candidates.iter().enumerate() {
            if self.owned_ranges.overlaps(c)
                || priors
                    .iter()
                    .chain(candidates[..i].iter())
                    .any(|h| c.overlaps(h))
            {
                return Err(Error::State(
                    "resident private proposal aliases an input/workspace/proposal".into(),
                ));
            }
        }
        Ok(())
    }

    /// Bind metadata, launch GPU-only work, and return UNCOMPLETED handles.
    /// There is no CPU fallback and no readback anywhere in this method.
    pub fn launch(
        &mut self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        self.launch_impl(client, p, bindings, None, output)
    }

    /// Direct device-token binding (for example the first word of the compact
    /// GPU greedy packet). Caller retains this extra original before launch.
    pub fn launch_device_token(
        &mut self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        token: &DeviceBuffer,
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        token.check(client, 1)?;
        self.launch_impl(client, p, bindings, Some(token), output)
    }
    fn launch_impl(
        &mut self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        token: Option<&DeviceBuffer>,
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        self.validate_runtime(client, p, bindings)?;
        let mut dispatches = 0;
        unsafe {
            crate::program_kernels::clear_step_status::launch_unchecked(
                client,
                grid(1),
                threads(),
                arg(&self.status, 1),
            );
            if let Some(token) = token {
                self.copy::<false>(
                    client,
                    &token.handle,
                    1,
                    &self.token,
                    1,
                    TensorView::vector(0, 1),
                    0,
                );
            } else {
                crate::resident_bindings::write_token::launch_unchecked(
                    client,
                    grid(1),
                    threads(),
                    arg(&self.token, 1),
                    p.token,
                );
            }
        }
        dispatches += 2;
        let nodes = output.nodes(&self.plan);
        for (index, node) in nodes.iter().enumerate() {
            dispatches += self.launch_node(client, index, node, p, bindings)?;
            // A preflight-validated View fully writes its dense destination and
            // certifies every copied word in the same kernel. All other output
            // certificates (including Contiguous aliases) remain separate.
            if matches!(node.operation, PlanOperation::View(_)) {
                continue;
            }
            for (out_index, id) in node.outputs.iter().enumerate() {
                let value = &self.values[id.0 as usize];
                if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
                    && out_index == 1
                {
                    unsafe {
                        crate::resident_bindings::certify_causal::launch_unchecked(
                            client,
                            grid(value.elements),
                            threads(),
                            arg(&value.handle, value.elements),
                            arg(&self.status, 1),
                            p.valid_tokens as u32,
                            geometry.query_heads,
                            self.plan.prefix_capacity(),
                        );
                    }
                } else {
                    self.certify(client, &value.handle, value.elements);
                }
                dispatches += 1;
            }
        }
        let logits = self
            .plan
            .logits()
            .filter(|_| output == ResidentOutput::Logits);
        if let Some(logits) = logits {
            let output = &self.values[logits.0 as usize];
            self.certify(client, &output.handle, output.elements);
            dispatches += 1;
        }
        // Certify all private proposals without moving a state scalar to host.
        for b in bindings {
            for value in b.candidate.buffers() {
                self.certify(client, &value.handle, value.elements);
                dispatches += 1;
            }
        }
        let mut proposals = Vec::new();
        for version in self.plan.proposed_domains() {
            let b = bindings
                .iter()
                .find(|b| b.slot == version.slot)
                .expect("validated state binding");
            proposals.push((*version, b.candidate.clone()));
        }
        let mut quant_status = Vec::new();
        let mut embedding_status = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            match &node.operation {
                PlanOperation::Project { tensor } => {
                    if let Some(l) = self.tensors[&tensor.name].layout {
                        quant_status.push((
                            self.node_status[i]
                                .block
                                .as_ref()
                                .expect("allocated block status")
                                .clone(),
                            l.blocks,
                            self.node_status[i]
                                .row
                                .as_ref()
                                .expect("allocated row status")
                                .clone(),
                            l.rows,
                        ));
                    }
                }
                PlanOperation::Embed { tensor, .. } => embedding_status.push((
                    self.node_status[i]
                        .block
                        .as_ref()
                        .expect("allocated embedding status")
                        .clone(),
                    tensor.columns,
                )),
                _ => {}
            }
        }
        Ok(ResidentSubmission {
            logits: logits.map(|v| self.values[v.0 as usize].clone()),
            proposals,
            status: self.status.clone(),
            quant_status,
            embedding_status,
            dispatches,
            fused_dispatches: 0,
        })
    }
    #[allow(clippy::too_many_arguments)] // Explicit typed source/destination ranges.
    fn copy<const CERTIFY_FINITE: bool>(
        &self,
        client: &Client,
        source: &Handle,
        source_len: usize,
        dest: &Handle,
        dest_len: usize,
        view: TensorView,
        offset: usize,
    ) {
        // Geometry and range checks are resolved by typed plan/preflight; the
        // physical helper also detects bad offsets before touching device RAM.
        unsafe {
            crate::program_kernels::copy_strided_words::launch_unchecked(
                client,
                grid(view.rows * view.width),
                threads(),
                arg(source, source_len),
                arg(dest, dest_len),
                arg(&self.status, 1),
                view.offset as u32,
                offset as u32,
                view.row_stride as u32,
                view.width as u32,
                view.rows,
                view.width,
                CERTIFY_FINITE,
            );
        }
    }
    fn certify(&self, client: &Client, source: &Handle, n: usize) {
        unsafe {
            crate::program_kernels::certify_finite_strided::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(source, n),
                arg(&self.status, 1),
                0,
                n as u32,
                1,
                n,
            );
        }
    }
    fn launch_node(
        &self,
        client: &Client,
        index: usize,
        node: &PlanNode,
        p: RuntimeParameters,
        bindings: &[StateBinding],
    ) -> Result<usize> {
        let input = |i: usize| &self.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.values[node.outputs[i].0 as usize];
        let s = |i: usize| &self.scratch[i];
        let state = |version: StateVersion| -> &StateBuffers {
            let b = bindings
                .iter()
                .find(|b| b.slot == version.slot)
                .expect("validated domain");
            if version.ordinal == 0 {
                &b.prior
            } else {
                &b.candidate
            }
        };
        let vector = |version: StateVersion| -> &DeviceBuffer {
            match state(version) {
                StateBuffers::Vector(v) => v,
                _ => unreachable!("typed vector effect"),
            }
        };
        let kv = |version: StateVersion| -> (&DeviceBuffer, &DeviceBuffer) {
            match state(version) {
                StateBuffers::Kv { keys, values } => (keys, values),
                _ => unreachable!("typed KV effect"),
            }
        };
        let cap = self.plan.prefix_capacity();
        // SAFETY: immutable typed plan, preflight and validated state bindings
        // bound every offset/count. Parent retains the complete original cohort.
        unsafe {
            match &node.operation {
                PlanOperation::TypedBulkF32Input { .. }
                | PlanOperation::TypedBulkF32Matmul { .. }
                | PlanOperation::TypedEmbed { .. }
                | PlanOperation::TypedProject { .. }
                | PlanOperation::TypedOrdinaryRms { .. }
                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                    "resident typed operation has no qualified lowering".into(),
                )),
                PlanOperation::Embed { tensor, .. } => {
                    let t = &self.tensors[&tensor.name];
                    let status = self.node_status[index]
                        .block
                        .as_ref()
                        .expect("embedding status");
                    crate::embedding::gather_quantized::launch_unchecked(
                        client,
                        grid(tensor.columns),
                        threads(),
                        arg(&t.packed, t.elements),
                        arg(&self.token, 1),
                        arg(&output(0).handle, tensor.columns),
                        arg(status, tensor.columns),
                        tensor.rows,
                        tensor.columns,
                        tensor.kind == QuantKind::Q6K,
                    );
                    crate::resident_bindings::merge_status::launch_unchecked(
                        client,
                        grid(tensor.columns),
                        threads(),
                        arg(status, tensor.columns),
                        arg(&self.status, 1),
                        tensor.columns,
                    );
                    Ok(2)
                }
                PlanOperation::Project { tensor } => {
                    let t = &self.tensors[&tensor.name];
                    let out = output(0);
                    if tensor.kind == QuantKind::F32 {
                        crate::mixed_ops::f32_ggml_dot_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&input(0).handle, tensor.columns),
                            arg(&out.handle, tensor.rows),
                            tensor.columns,
                            tensor.rows,
                            true,
                        );
                        return Ok(1);
                    }
                    let l = t.layout.expect("quant layout");
                    let dynamic = self.projections[index]
                        .as_ref()
                        .expect("prepared quant projection binding");
                    let preparation_dispatches = dynamic.launch_prepare(client, &l)?;
                    crate::resident_bindings::merge_status::launch_unchecked(
                        client,
                        grid(l.blocks),
                        threads(),
                        arg(&dynamic.block_status, l.blocks),
                        arg(&self.status, 1),
                        l.blocks,
                    );
                    crate::resident_bindings::merge_status::launch_unchecked(
                        client,
                        grid(l.rows),
                        threads(),
                        arg(&dynamic.row_status, l.rows),
                        arg(&self.status, 1),
                        l.rows,
                    );
                    match tensor.kind {
                        QuantKind::Q4_0 => crate::q4_q8::source_q4_q8_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&dynamic.scales, l.composite_count),
                            arg(&dynamic.codes, l.columns),
                            arg(&out.handle, tensor.rows),
                            l.blocks,
                            tensor.rows,
                            true,
                        ),
                        QuantKind::Q4_1 => crate::mixed_ops::q4_1_q8_1_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&dynamic.scales, l.composite_count),
                            arg(&dynamic.offsets, l.composite_count),
                            arg(&dynamic.codes, l.columns),
                            arg(&out.handle, tensor.rows),
                            l.blocks,
                            tensor.rows,
                            true,
                        ),
                        QuantKind::Q8_0 => crate::mixed_ops::q8_0_q8_0_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&dynamic.scales, l.composite_count),
                            arg(&dynamic.codes, l.columns),
                            arg(&out.handle, tensor.rows),
                            l.blocks,
                            tensor.rows,
                            true,
                        ),
                        QuantKind::Q6K => crate::mixed_ops::q6_k_q8_k_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&dynamic.scales, l.composite_count),
                            arg(&dynamic.codes, l.columns),
                            arg(&out.handle, tensor.rows),
                            l.blocks,
                            tensor.rows,
                            true,
                        ),
                        _ => {
                            return Err(Error::Unsupported(
                                "resident matrix codec unsupported".into(),
                            ));
                        }
                    }
                    Ok(preparation_dispatches + 3)
                }
                PlanOperation::View(view) => {
                    self.copy::<true>(
                        client,
                        &input(0).handle,
                        input(0).elements,
                        &output(0).handle,
                        output(0).elements,
                        *view,
                        0,
                    );
                    Ok(1)
                }
                PlanOperation::Contiguous => {
                    // Cold binding aliases the entire already-packed input.
                    // launch_impl still certifies this semantic output.
                    Ok(0)
                }
                PlanOperation::Rms { gamma, eps, groups } => {
                    let n = input(0).elements;
                    let width = n / groups;
                    crate::rms_norm::rms_squares::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(s(0), n),
                        n,
                    );
                    crate::rms_norm::rms_inverse::launch_unchecked(
                        client,
                        grid(*groups),
                        threads(),
                        arg(s(0), n),
                        arg(s(1), groups * 2),
                        *eps,
                        width,
                        *groups,
                    );
                    crate::rms_norm::rms_round_inverse::launch_unchecked(
                        client,
                        grid(*groups),
                        threads(),
                        arg(s(1), groups * 2),
                        *groups,
                    );
                    crate::rms_norm::rms_apply::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(&self.constants[gamma.0 as usize], width),
                        arg(s(1), groups * 2),
                        arg(&output(0).handle, n),
                        width,
                        *groups,
                    );
                    Ok(4)
                }
                PlanOperation::Silu { multiplied } => {
                    let n = input(0).elements;
                    crate::resident_bindings::certify_interval::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(&self.status, 1),
                        -256.0,
                        256.0,
                        n,
                    );
                    for stage in 0..12 {
                        crate::silu::silu_step::launch_unchecked(
                            client,
                            grid(n),
                            threads(),
                            arg(&input(0).handle, n),
                            arg(s(0), n * crate::silu::SCRATCH),
                            arg(&output(0).handle, n),
                            n,
                            stage,
                        );
                    }
                    if *multiplied {
                        crate::silu::silu_gate::launch_unchecked(
                            client,
                            grid(n),
                            threads(),
                            arg(&input(1).handle, n),
                            arg(&output(0).handle, n),
                            n,
                        );
                    }
                    Ok(13 + usize::from(*multiplied))
                }
                PlanOperation::Add => {
                    let n = input(0).elements;
                    crate::elementwise::residual_add::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(&input(1).handle, n),
                        arg(&output(0).handle, n),
                        n,
                    );
                    Ok(1)
                }
                PlanOperation::SigmoidGate => {
                    let n = input(0).elements;
                    for stage in 0..crate::elementwise::SIGMOID_STAGES {
                        crate::elementwise::sigmoid_gate_step::launch_unchecked(
                            client,
                            grid(n),
                            threads(),
                            arg(&input(1).handle, n),
                            arg(&input(0).handle, n),
                            arg(
                                &self.exp_tables,
                                crate::source_transcendentals::TABLES.len(),
                            ),
                            arg(s(0), n * crate::elementwise::SIGMOID_SCRATCH),
                            arg(s(1), n),
                            arg(&output(0).handle, n),
                            n,
                            stage,
                        );
                    }
                    Ok(crate::elementwise::SIGMOID_STAGES)
                }
                PlanOperation::Conv { weights } => {
                    let n = input(0).elements;
                    let effect = node.state.expect("conv effect");
                    crate::gdn_conv::gdn_conv_products::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(&self.constants[weights.0 as usize], n * 4),
                        arg(&vector(effect.read).handle, n * 3),
                        arg(s(0), n * 4),
                        arg(&vector(effect.write).handle, n * 3),
                        n,
                    );
                    for level in 0..3 {
                        crate::gdn_conv::gdn_conv_reduce::launch_unchecked(
                            client,
                            grid(n),
                            threads(),
                            arg(s(0), n * 4),
                            arg(&output(0).handle, n),
                            n,
                            level,
                        );
                    }
                    Ok(4)
                }
                PlanOperation::Coefficients { dt, a, scale } => {
                    let h = input(0).elements;
                    for i in 0..2 {
                        crate::resident_bindings::certify_interval::launch_unchecked(
                            client,
                            grid(h),
                            threads(),
                            arg(&input(i).handle, h),
                            arg(&self.status, 1),
                            -128.0,
                            128.0,
                            h,
                        );
                    }
                    crate::resident_bindings::coefficient_input::launch_unchecked(
                        client,
                        grid(h),
                        threads(),
                        arg(&input(0).handle, h),
                        arg(&input(1).handle, h),
                        arg(&self.constants[dt.0 as usize], h),
                        arg(&self.constants[a.0 as usize], h),
                        arg(s(0), h * 4),
                        h,
                    );
                    for stage in 0..8 {
                        crate::gdn_coefficients::coefficient_step::launch_unchecked(
                            client,
                            grid(h),
                            threads(),
                            arg(s(0), h * 4),
                            arg(
                                &self.exp_tables,
                                crate::source_transcendentals::TABLES.len(),
                            ),
                            arg(s(1), h * 6),
                            arg(s(2), h * 3),
                            arg(s(3), h * 4),
                            h,
                            stage,
                        );
                    }
                    crate::resident_bindings::coefficient_output::launch_unchecked(
                        client,
                        grid(h),
                        threads(),
                        arg(s(3), h * 4),
                        arg(&output(0).handle, h * 3),
                        arg(&output(1).handle, h),
                        arg(&output(2).handle, h),
                        *scale,
                        h,
                    );
                    Ok(12)
                }
                PlanOperation::Recurrent { width } => {
                    let n = input(0).elements;
                    let m = n * width;
                    let effect = node.state.expect("recurrent effect");
                    for i in 0..3 {
                        self.copy::<false>(
                            client,
                            &input(i).handle,
                            n,
                            s(0),
                            n * 3,
                            TensorView::vector(0, n),
                            n * i,
                        );
                    }
                    crate::gdn_recurrent::decay::launch_unchecked(
                        client,
                        grid(m),
                        threads(),
                        arg(&vector(effect.read).handle, m),
                        arg(&input(3).handle, n / width * 3),
                        arg(s(1), m),
                        *width,
                        m,
                    );
                    let mut count = 4;
                    count += self.recurrent_dot(client, s(1), s(0), s(3), s(4), s(2), *width, n, n);
                    crate::gdn_recurrent::residual::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(s(0), n * 3),
                        arg(s(2), n),
                        n,
                    );
                    crate::gdn_recurrent::beta::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(3).handle, n / width * 3),
                        arg(s(2), n),
                        *width,
                        n,
                    );
                    crate::gdn_recurrent::update::launch_unchecked(
                        client,
                        grid(m),
                        threads(),
                        arg(s(1), m),
                        arg(s(0), n * 3),
                        arg(s(2), n),
                        arg(&vector(effect.write).handle, m),
                        *width,
                        n,
                        m,
                    );
                    count += 3;
                    count += self.recurrent_dot(
                        client,
                        &vector(effect.write).handle,
                        s(0),
                        s(3),
                        s(4),
                        &output(0).handle,
                        *width,
                        n,
                        0,
                    );
                    crate::gdn_recurrent::scale::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(3).handle, n / width * 3),
                        arg(&output(0).handle, n),
                        *width,
                        n,
                    );
                    Ok(count + 1)
                }
                PlanOperation::KvPrefixGuard { .. } => Ok(0), // exact position checked before first launch
                PlanOperation::Rope { geometry, .. } => {
                    let g = geometry.attention;
                    let h = g.query_heads + g.kv_heads;
                    let half = geometry.rotate / 2;
                    let q = input(0).elements;
                    let k = input(1).elements;
                    let v = input(2).elements;
                    let count = q + k + v;
                    let ordered_angles = crate::rope_angles::PreparedAngles::single_for_cuda(
                        client,
                        s(1),
                        p.position as f32,
                        geometry.theta_scale,
                        half,
                    )?;
                    for (i, offset) in [(0, 0), (1, q), (2, q + k)] {
                        self.copy::<false>(
                            client,
                            &input(i).handle,
                            input(i).elements,
                            s(0),
                            count,
                            TensorView::vector(0, input(i).elements),
                            offset,
                        );
                    }
                    let angle_dispatches = if let Some(angles) = ordered_angles {
                        angles.launch(client)?
                    } else {
                        for stage in 0..half {
                            crate::rope_kv::angle_step::launch_unchecked(
                                client,
                                grid(1),
                                threads(),
                                arg(s(1), half),
                                p.position as f32,
                                geometry.theta_scale,
                                stage,
                            );
                        }
                        half
                    };
                    crate::source_trig::sincos_angles::launch_unchecked(
                        client,
                        grid(half),
                        threads(),
                        arg(s(1), half),
                        arg(&self.trig_tables, crate::source_trig::TRIG_TABLES.len()),
                        arg(s(2), half * 2),
                        half,
                    );
                    crate::rope_kv::rope_products::launch_unchecked(
                        client,
                        grid(h * half),
                        threads(),
                        arg(s(0), count),
                        arg(s(2), half * 2),
                        arg(s(3), h * half * 2),
                        g.width,
                        half,
                        h,
                        g.query_heads,
                        1,
                    );
                    crate::rope_kv::rope_sums::launch_unchecked(
                        client,
                        grid(count),
                        threads(),
                        arg(s(0), count),
                        arg(s(2), half * 2),
                        arg(s(3), h * half * 2),
                        arg(s(4), h * half * 2),
                        arg(s(6), count),
                        g.width,
                        half,
                        h,
                        count,
                        g.query_heads,
                        1,
                    );
                    crate::rope_kv::rope_subtract::launch_unchecked(
                        client,
                        grid(h * half * 2),
                        threads(),
                        arg(s(0), count),
                        arg(s(2), half * 2),
                        arg(s(3), h * half * 2),
                        arg(s(4), h * half * 2),
                        arg(s(5), h * half * 2),
                        g.width,
                        half,
                        h,
                        g.query_heads,
                        1,
                    );
                    crate::rope_kv::rope_apply::launch_unchecked(
                        client,
                        grid(h * half * 2),
                        threads(),
                        arg(s(0), count),
                        arg(s(2), half * 2),
                        arg(s(3), h * half * 2),
                        arg(s(4), h * half * 2),
                        arg(s(5), h * half * 2),
                        arg(s(6), count),
                        g.width,
                        half,
                        h,
                        g.query_heads,
                        1,
                    );
                    for (i, offset) in [(0, 0), (1, q), (2, q + k)] {
                        self.copy::<false>(
                            client,
                            s(6),
                            count,
                            &output(i).handle,
                            output(i).elements,
                            TensorView::vector(offset, output(i).elements),
                            0,
                        );
                    }
                    Ok(angle_dispatches + 11)
                }
                PlanOperation::AppendKv => {
                    let effect = node.state.expect("KV append effect");
                    let (previous_k, previous_v) = kv(effect.read);
                    let (candidate_k, candidate_v) = kv(effect.write);
                    let slab = input(0).elements;
                    crate::resident_bindings::append_kv::launch_unchecked(
                        client,
                        grid(slab * cap),
                        threads(),
                        arg(&previous_k.handle, slab * cap),
                        arg(&previous_v.handle, slab * cap),
                        arg(&input(0).handle, slab),
                        arg(&input(1).handle, slab),
                        arg(&candidate_k.handle, slab * cap),
                        arg(&candidate_v.handle, slab * cap),
                        p.position as u32,
                        slab,
                        cap,
                    );
                    Ok(1)
                }
                PlanOperation::AttentionQk {
                    state: version,
                    geometry: g,
                    scale,
                    ..
                } => {
                    let (keys, _) = kv(*version);
                    let query = g.query_heads * g.width;
                    let key = cap * g.kv_heads * g.width;
                    let columns = g.query_heads * cap;
                    self.copy::<false>(
                        client,
                        &input(0).handle,
                        query,
                        s(0),
                        query + key,
                        TensorView::vector(0, query),
                        0,
                    );
                    self.copy::<false>(
                        client,
                        &keys.handle,
                        key,
                        s(0),
                        query + key,
                        TensorView::vector(0, key),
                        query,
                    );
                    for block in 0..g.width / 32 {
                        let (previous, lanes) = if block % 2 == 0 {
                            (s(2), s(1))
                        } else {
                            (s(1), s(2))
                        };
                        for stage in 0..3 {
                            crate::attention_qk::accumulate::launch_unchecked(
                                client,
                                grid(columns * 32),
                                threads(),
                                arg(s(0), query + key),
                                arg(previous, columns * 32),
                                arg(lanes, columns * 32),
                                arg(s(3), columns * 32),
                                arg(s(4), columns * 32),
                                g.width,
                                cap,
                                g.query_heads,
                                g.kv_heads,
                                block,
                                stage,
                            );
                        }
                    }
                    let last = if (g.width / 32 - 1) % 2 == 0 {
                        s(1)
                    } else {
                        s(2)
                    };
                    let other = if (g.width / 32 - 1) % 2 == 0 {
                        s(2)
                    } else {
                        s(1)
                    };
                    let mut count = 2 + g.width / 32 * 3;
                    count += self.attention_reduce(client, last, other, s(5), columns);
                    crate::attention_qk::scale::launch_unchecked(
                        client,
                        grid(columns),
                        threads(),
                        arg(s(5), columns * 2),
                        *scale,
                        columns,
                    );
                    self.copy::<false>(
                        client,
                        s(5),
                        columns * 2,
                        &output(0).handle,
                        columns,
                        TensorView::vector(0, columns),
                        0,
                    );
                    crate::resident_bindings::masked_scores::launch_unchecked(
                        client,
                        grid(columns),
                        threads(),
                        arg(s(5), columns * 2),
                        arg(&output(1).handle, columns),
                        p.valid_tokens as u32,
                        g.query_heads,
                        cap,
                    );
                    Ok(count + 3)
                }
                PlanOperation::AttentionSoftmax { heads, .. } => {
                    self.softmax(client, input(0), output(0), *heads, cap, p.valid_tokens)
                }
                PlanOperation::AttentionValue {
                    state: version,
                    geometry: g,
                    padding,
                    ..
                } => {
                    let (_, values) = kv(*version);
                    let probabilities = g.query_heads * cap;
                    let value = cap * g.kv_heads * g.width;
                    let columns = g.query_heads * g.width;
                    self.copy::<false>(
                        client,
                        &input(0).handle,
                        probabilities,
                        s(0),
                        probabilities + value,
                        TensorView::vector(0, probabilities),
                        0,
                    );
                    self.copy::<false>(
                        client,
                        &values.handle,
                        value,
                        s(0),
                        probabilities + value,
                        TensorView::vector(0, value),
                        probabilities,
                    );
                    // Exact source padded length is RUNTIME metadata; extra all-zero
                    // FMA blocks can change -0 to +0 and must never execute.
                    let blocks = p.valid_tokens.div_ceil(*padding) * padding / 32;
                    for block in 0..blocks {
                        let (previous, lanes) = if block % 2 == 0 {
                            (s(2), s(1))
                        } else {
                            (s(1), s(2))
                        };
                        for stage in 0..3 {
                            crate::attention_value::accumulate::launch_unchecked(
                                client,
                                grid(columns * 32),
                                threads(),
                                arg(s(0), probabilities + value),
                                arg(previous, columns * 32),
                                arg(lanes, columns * 32),
                                arg(s(3), columns * 32),
                                arg(s(4), columns * 32),
                                block as u32,
                                g.width,
                                cap,
                                g.query_heads,
                                g.kv_heads,
                                stage,
                            );
                        }
                    }
                    let last = if (blocks - 1) % 2 == 0 { s(1) } else { s(2) };
                    let other = if (blocks - 1) % 2 == 0 { s(2) } else { s(1) };
                    Ok(2 + blocks * 3
                        + self.attention_reduce(client, last, other, &output(0).handle, columns))
                }
                PlanOperation::CheckFinite { .. } => {
                    self.certify(client, &input(0).handle, input(0).elements);
                    Ok(1)
                }
                PlanOperation::Observe { .. } | PlanOperation::ObserveState { .. } => Ok(0),
            }
        }
    }
    #[allow(clippy::too_many_arguments)] // Exact already-qualified recurrent buffer roles.
    fn recurrent_dot(
        &self,
        client: &Client,
        state: &Handle,
        qkv: &Handle,
        lanes: &Handle,
        reduced: &Handle,
        output: &Handle,
        width: usize,
        values: usize,
        offset: usize,
    ) -> usize {
        unsafe {
            for block in 0..width / 32 {
                crate::gdn_recurrent::dot_lanes::launch_unchecked(
                    client,
                    grid(values * 32),
                    threads(),
                    arg(state, values * width),
                    arg(qkv, values * 3),
                    arg(lanes, values * 32),
                    width,
                    values,
                    offset,
                    block,
                );
            }
            for level in 0..5 {
                let input = if level % 2 == 0 { lanes } else { reduced };
                let destination = if level == 4 {
                    output
                } else if level % 2 == 0 {
                    reduced
                } else {
                    lanes
                };
                let in_len = if level % 2 == 0 {
                    values * 32
                } else {
                    values * 16
                };
                let out_len = if level == 4 {
                    values
                } else if level % 2 == 0 {
                    values * 16
                } else {
                    values * 32
                };
                crate::gdn_recurrent::dot_reduce::launch_unchecked(
                    client,
                    grid(values * (16 >> level)),
                    threads(),
                    arg(input, in_len),
                    arg(destination, out_len),
                    values,
                    level,
                );
            }
        }
        width / 32 + 5
    }
    fn attention_reduce(
        &self,
        client: &Client,
        lanes: &Handle,
        reduced: &Handle,
        output: &Handle,
        columns: usize,
    ) -> usize {
        unsafe {
            for level in 0..5 {
                let input = if level % 2 == 0 { lanes } else { reduced };
                let destination = if level == 4 {
                    output
                } else if level % 2 == 0 {
                    reduced
                } else {
                    lanes
                };
                let in_len = if level % 2 == 0 {
                    columns * 32
                } else {
                    columns * 16
                };
                let out_len = if level == 4 {
                    columns
                } else if level % 2 == 0 {
                    columns * 16
                } else {
                    columns * 32
                };
                crate::attention_qk::reduce::launch_unchecked(
                    client,
                    grid(columns * (16 >> level)),
                    threads(),
                    arg(input, in_len),
                    arg(destination, out_len),
                    columns,
                    level,
                );
            }
        }
        5
    }
    fn softmax(
        &self,
        client: &Client,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        h: usize,
        cap: usize,
        valid: usize,
    ) -> Result<usize> {
        self.softmax_masked(client, input, output, h, cap, CausalMask::Uniform(valid))
    }

    fn softmax_masked(
        &self,
        client: &Client,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        h: usize,
        cap: usize,
        mask: CausalMask<'_>,
    ) -> Result<usize> {
        use crate::attention_softmax as sm;
        let n = h * cap;
        let groups = cap.div_ceil(8);
        let s = |i: usize| &self.scratch[i];
        // The original finite valid prefix is followed by -Inf scores. Maxima
        // and source polynomial stages therefore preserve all valid bits; the
        // padded exponentials become canonical +0 before the original tree.
        unsafe {
            sm::maxima::launch_unchecked(
                client,
                grid(h),
                threads(),
                arg(&input.handle, n),
                arg(s(4), h),
                h,
                cap,
            );
            for stage in 0..2 {
                sm::prepare::launch_unchecked(
                    client,
                    grid(n),
                    threads(),
                    arg(&input.handle, n),
                    arg(s(4), h),
                    arg(s(0), n * sm::SCRATCH),
                    n,
                    cap,
                    stage,
                );
            }
            for step in 0..8 {
                if step == 1 || step == 3 {
                    sm::prepare::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input.handle, n),
                        arg(s(4), h),
                        arg(s(0), n * sm::SCRATCH),
                        n,
                        cap,
                        if step == 1 { 2 } else { 3 },
                    );
                }
                for stage in 0..3 {
                    sm::polynomial::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(s(0), n * sm::SCRATCH),
                        arg(s(1), n),
                        arg(s(2), n),
                        arg(s(3), n),
                        n,
                        step,
                        stage,
                    );
                }
            }
            sm::exponentials::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(s(0), n * sm::SCRATCH),
                arg(s(7), n),
                n,
            );
            match mask {
                CausalMask::Uniform(valid) => {
                    crate::resident_bindings::zero_future::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(s(7), n),
                        valid as u32,
                        h,
                        cap,
                    );
                }
                CausalMask::Rows {
                    positions,
                    heads_per_row,
                } => {
                    crate::resident_bindings::zero_future_rows::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(s(7), n),
                        arg(positions, h / heads_per_row),
                        heads_per_row,
                        h / heads_per_row,
                        cap,
                    );
                }
            }
            for level in 0..3 {
                let (source, destination, source_len, destination_len) = if level == 0 {
                    (s(7), s(5), n, h * groups * 4)
                } else if level == 1 {
                    (s(5), s(6), h * groups * 4, h * groups * 2)
                } else {
                    (s(6), s(5), h * groups * 2, h * groups * 4)
                };
                sm::reduce::launch_unchecked(
                    client,
                    grid(h * groups * (4 >> level)),
                    threads(),
                    arg(source, source_len),
                    arg(destination, destination_len),
                    h,
                    cap,
                    level,
                );
            }
            // Source totals are positive. Extra all-+0 groups leave their f64
            // sum unchanged, retaining one fixed capacity shader specialization.
            sm::totals::launch_unchecked(
                client,
                grid(h),
                threads(),
                arg(s(5), h * groups * 4),
                arg(s(8), h),
                h,
                groups,
            );
            sm::reciprocal::launch_unchecked(
                client,
                grid(h),
                threads(),
                arg(s(8), h),
                arg(s(9), h),
                h,
            );
            sm::inverses::launch_unchecked(
                client,
                grid(h),
                threads(),
                arg(s(9), h),
                arg(s(10), h),
                h,
            );
            sm::probabilities::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(s(7), n),
                arg(s(10), h),
                arg(&output.handle, n),
                n,
                cap,
            );
        }
        Ok(38)
    }
}

fn region<'a>(model: &'a dyn TensorSource, t: &'a SourceTensor) -> Result<MatVecRegion<'a>> {
    Ok(MatVecRegion {
        source_identity: t.source,
        tensor_name: &t.name,
        kind: t.kind,
        columns: t.columns,
        rows: t.rows,
        packed_weights: model.tensor(&t.name)?.packed_bytes,
        activation: &[],
    })
}
/// Allocation is delegated to the already-armed physical initialization
/// owner. It admits and retains Empty before writing Upload bytes into it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ResidentBufferRole {
    Source,
    Value,
    Workspace,
    Status,
    Parameter,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum AllocationInit<'a> {
    Empty,
    Upload(&'a [u8]),
}
struct ProgramAllocator<'a, F> {
    client: &'a Client,
    allocate: &'a mut F,
}
impl<F: FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>>
    ProgramAllocator<'_, F>
{
    fn check(&self, handle: Handle, bytes: usize) -> Result<Handle> {
        if handle.service != self.client.service_id() || handle.size_in_used() < bytes as u64 {
            return Err(Error::State(
                "resident allocator returned invalid service/size".into(),
            ));
        }
        Ok(handle)
    }
    fn empty(&mut self, role: ResidentBufferRole, bytes: usize) -> Result<Handle> {
        let handle = (self.allocate)(role, bytes, AllocationInit::Empty)?;
        self.check(handle, bytes)
    }
    fn upload(&mut self, role: ResidentBufferRole, bytes: &[u8]) -> Result<Handle> {
        let handle = (self.allocate)(role, bytes.len(), AllocationInit::Upload(bytes))?;
        self.check(handle, bytes.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "qualification")]
    use std::path::Path;

    #[test]
    fn prepared_ranges_match_independent_byte_coverage_with_gaps_and_aliases() {
        let service = ServiceId::of::<()>(cubecl_common::device::DeviceId::new(0, 0));
        let memory = cubecl_runtime::memory_management::ManagedMemoryHandle::new()
            .binding()
            .id();
        let range = |bytes| BufferRange {
            service,
            memory,
            bytes,
        };
        let mut owned = OwnedRanges::new(service);
        for bytes in [2..4, 8..10, 2..4] {
            owned.insert(range(bytes)).unwrap();
        }
        assert_eq!(owned.by_memory[&memory].len(), 2);
        for start in 0..12 {
            for end in start + 1..=12 {
                // Independent oracle enumerates byte membership, including
                // adjacency and the unowned gap in a single allocation.
                let expected = (start..end).any(|i| [2, 3, 8, 9].contains(&i));
                assert_eq!(owned.overlaps(&range(start..end)), expected);
            }
        }
        let gap = range(4..8);
        assert!(!gap.overlaps(&range(2..4)));
        assert!(!gap.overlaps(&range(8..10)));
        assert!(gap.overlaps(&range(7..9)));
    }

    #[test]
    fn prepared_ranges_keep_service_and_allocation_identity_separate() {
        let device = cubecl_common::device::DeviceId::new(0, 0);
        let service = ServiceId::of::<()>(device);
        let foreign = ServiceId::of::<u8>(device);
        let id = || {
            cubecl_runtime::memory_management::ManagedMemoryHandle::new()
                .binding()
                .id()
        };
        let a = BufferRange {
            service,
            memory: id(),
            bytes: 2..8,
        };
        let b = BufferRange {
            memory: id(),
            ..a.clone()
        };
        let c = BufferRange {
            service: foreign,
            ..a.clone()
        };
        let mut owned = OwnedRanges::new(service);
        owned.insert(a.clone()).unwrap();
        assert!(owned.overlaps(&a));
        assert!(!owned.overlaps(&b));
        assert!(!owned.overlaps(&c));
        assert!(!a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(owned.insert(c).is_err());
        assert_eq!(owned.by_memory.len(), 1);
    }

    #[test]
    fn budget_arithmetic_refuses_overflow_and_shader_index_excess() {
        assert_eq!(checked_sum([4, 8, 16]).unwrap(), 28);
        assert!(checked_sum([usize::MAX, 1]).is_err());
        assert!(bytes(usize::MAX, 8).is_err());
        assert_eq!(bytes(262_144, 4).unwrap(), 1_048_576);
    }

    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires both existing trained checkpoints; metadata only, no GPU or token math"]
    fn real_source_preflight_counts_tied_embedding_once_and_all_private_domains() {
        let cache = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models");
        for name in ["qwen35-0.8b", "qwen35-2b"] {
            let dir = cache.join(name);
            let model =
                LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
            for cap in [256, 512] {
                for logits in [false, true] {
                    let plan = model.forward_plan(cap, logits, false).unwrap();
                    let b = ResidentProgram::preflight(model.as_ref(), &plan).unwrap();
                    let aliases = contiguous_aliases(&plan).unwrap();
                    let alias_count = aliases.iter().flatten().count();
                    let fused_views = plan
                        .nodes()
                        .iter()
                        .filter(|node| matches!(node.operation, PlanOperation::View(_)))
                        .count();
                    assert!(fused_views > 0);
                    let prepared_projections = plan
                        .nodes()
                        .iter()
                        .filter(|node| {
                            matches!(&node.operation, PlanOperation::Project { tensor } if tensor.kind != QuantKind::F32)
                        })
                        .count();
                    assert!(prepared_projections > 0);
                    // Ordinary schedule has two packs per full-attention
                    // layer; complete GDN diagnostics add two per GDN layer.
                    assert!([12, 48].contains(&alias_count));
                    let unaliased: usize = plan
                        .values()
                        .iter()
                        .map(|s| s.capacity_elements().unwrap() * 4)
                        .sum();
                    let saved: usize = aliases
                        .iter()
                        .enumerate()
                        .filter(|(_, alias)| alias.is_some())
                        .map(|(i, alias)| {
                            let original = alias.unwrap().0 as usize;
                            assert!(original < i);
                            assert!(aliases[original].is_none());
                            assert_eq!(plan.values()[i], plan.values()[original]);
                            plan.values()[i].capacity_elements().unwrap() * 4
                        })
                        .sum();
                    assert!(saved > 0);
                    assert_eq!(b.value_bytes, unaliased - saved);
                    assert_eq!(b.states.len(), 42);
                    assert_eq!(b.prior_state_bytes, b.candidate_state_bytes);
                    assert!(b.states.iter().all(|s| s.bytes > 0));
                    let mut tensors = BTreeMap::<String, &SourceTensor>::new();
                    let mut projections = BTreeSet::new();
                    for node in plan.nodes() {
                        match &node.operation {
                            PlanOperation::Embed { tensor, .. } => {
                                tensors.insert(tensor.name.clone(), tensor);
                            }
                            PlanOperation::Project { tensor } => {
                                tensors.insert(tensor.name.clone(), tensor);
                                projections.insert(tensor.name.clone());
                            }
                            _ => {}
                        }
                    }
                    let mut source = 4
                        + crate::source_transcendentals::TABLES.len() * 8
                        + crate::source_trig::TRIG_TABLES.len() * 8;
                    for t in tensors.values() {
                        source += StaticSource::gpu_bytes_for(region(model.as_ref(), t).unwrap())
                            .unwrap();
                        if projections.contains(&t.name) && t.kind != QuantKind::F32 {
                            source += t.rows
                                * (t.columns / if t.kind == QuantKind::Q6K { 256 } else { 32 })
                                * if t.kind == QuantKind::Q4_1 { 8 } else { 4 };
                        }
                    }
                    source += plan
                        .constants()
                        .iter()
                        .map(|c| c.values.len() * 4)
                        .sum::<usize>();
                    assert_eq!(b.source_bytes, source, "{name} cap={cap} logits={logits}");
                    assert_eq!(
                        b.total_with_state().unwrap(),
                        b.owned_bytes().unwrap() + b.prior_state_bytes + b.candidate_state_bytes
                    );
                    println!(
                        "RESIDENT_METADATA_BUDGET model={name} capacity={cap} logits={logits} source={} values={} workspace={} status={} prior={} proposed={} total={} aliases={alias_count} saved_value_bytes={saved} fused_view_certificates={fused_views} prepared_projection_bindings={prepared_projections}",
                        b.source_bytes,
                        b.value_bytes,
                        b.workspace_bytes,
                        b.status_bytes,
                        b.prior_state_bytes,
                        b.candidate_state_bytes,
                        b.total_with_state().unwrap()
                    );
                }
            }
            for cap in [1, 255, 257, 768] {
                let plan = model.forward_plan(cap, true, false).unwrap();
                assert!(ResidentProgram::preflight(model.as_ref(), &plan).is_err());
            }
        }
    }
}

#[cfg(test)]
#[path = "resident_program/typed_refusal_tests.rs"]
mod typed_refusal_tests;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11185; SHA256 `176bf0662d83b16ec1d3bcbe24cddd80cbe5f993277b94b9b55e03dba60bfcf0`.

````text
//! Pure explicit default-bulk F32 fragment, not a model producer/forward permit.
use crate::accelerator::program::{
    BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Origin, BulkF32Side,
    PlanBuilder, ProgramNumericalProfile, RecordedOperatorSlice, TeacherSpan,
};
use crate::original_binding::{QwenMatrixRole, QwenNumericalProfile};
use crate::{Error, LoadedQwenModel, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};

const META: u64 = 16384;
/// Existing host ledger only; no budget registration or outer RAM claim.
#[derive(Clone)]
pub struct BulkF32HostAdmission {
    pub arbiter: ResourceArbiter,
    pub resource: PhysicalDeviceKey,
    pub owner: ResourceOwner,
}
/// An explicit actual allocation view. External input is never claimed to be
/// current model state or a mathematical producer from the present Rust graph.
pub struct ExternalBulkF32<'a> {
    pub name: &'a str,
    pub layout: BulkF32Layout,
    pub backing: &'a [f32],
}

pub(crate) struct BulkOwner {
    pub(crate) model: Arc<LoadedQwenModel>,
    pub(crate) matrix: BulkF32Matrix,
    pub(crate) admission: BulkF32HostAdmission,
    _metadata: GlobalResourceLease,
}
pub(crate) enum LeftInput {
    External(Vec<f32>),
    Source,
}
pub(crate) struct BulkInputs {
    // Payloads precede staging; source/metadata owner remains last.
    left: LeftInput,
    right: Vec<f32>,
    _staging: GlobalResourceLease,
    pub(crate) owner: Arc<BulkOwner>,
}
impl BulkInputs {
    pub(crate) fn values(&self, side: BulkF32Side) -> Result<&[f32]> {
        if side == BulkF32Side::Right {
            return Ok(&self.right);
        }
        match &self.left {
            LeftInput::External(values) => Ok(values),
            LeftInput::Source => {
                let BulkF32Origin::SourceMatrix(tensor) = self.owner.matrix.left().origin() else {
                    return Err(Error::State("bulk learned backing descriptor".into()));
                };
                tensor.validate_source(self.owner.model.as_ref())?;
                let bytes = self.owner.model.tensor_bytes(&tensor.name)?;
                // SAFETY: all F32 bit patterns are valid; source raw kind/length
                // and target little endian are checked. Returned borrow retains
                // the SAME immutable model allocation; no transposition/cache.
                let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
                if !cfg!(target_endian = "little")
                    || !prefix.is_empty()
                    || !suffix.is_empty()
                    || values.len() != self.owner.matrix.left().layout().backing_words()
                {
                    return Err(Error::Unsupported(
                        "bulk original F32 alignment/byte geometry".into(),
                    ));
                }
                Ok(values)
            }
        }
    }
}

pub struct BulkF32Slice {
    pub(crate) inputs: Arc<BulkInputs>,
}
impl BulkF32Slice {
    /// Exactly one immutable original GGUF model and actual consecutive IDs.
    /// These constructors confer pure-operator ownership, never SessionAuthority.
    #[allow(clippy::too_many_arguments)] // Explicit source, temporal and physical scope.
    pub fn observed_pair(
        model: Arc<LoadedQwenModel>,
        base: usize,
        capacity: usize,
        ids: &[u32],
        left: ExternalBulkF32<'_>,
        right: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::bind(
            model,
            base,
            capacity,
            ids,
            None,
            Some(left),
            right,
            admission,
            cancelled,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn source_projection(
        model: Arc<LoadedQwenModel>,
        base: usize,
        capacity: usize,
        ids: &[u32],
        role: QwenMatrixRole,
        right: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::bind(
            model,
            base,
            capacity,
            ids,
            Some(role),
            None,
            right,
            admission,
            cancelled,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn bind(
        model: Arc<LoadedQwenModel>,
        base: usize,
        capacity: usize,
        ids: &[u32],
        source_role: Option<QwenMatrixRole>,
        external_left: Option<ExternalBulkF32<'_>>,
        right: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        if model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
            || admission.resource.backend != "host"
        {
            return Err(Error::Unsupported(
                "bulk pure GGUF F32 existing host scope".into(),
            ));
        }
        // Bounded original metadata is admitted BEFORE span/descriptor allocation.
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        let span = TeacherSpan::new(
            model.source_identity(),
            base,
            capacity,
            model.geometry().vocab_size,
            ids,
        )?;
        let left = match (source_role, &external_left) {
            (Some(role), None) => {
                let tensor = model.matrix_descriptor(role)?;
                BulkF32Operand::source_matrix(span.clone(), model.as_ref(), &tensor.name)?
            }
            (None, Some(value)) => {
                BulkF32Operand::external(span.clone(), BulkF32Side::Left, value.name, value.layout)?
            }
            _ => return Err(Error::State("bulk left source transport ambiguity".into())),
        };
        let rhs = BulkF32Operand::external(span, BulkF32Side::Right, right.name, right.layout)?;
        let matrix = BulkF32Matrix::new(left, rhs, BulkF32Contract::HaswellTinyBlasF32V1)?;
        for (operand, payload) in [
            (matrix.left(), external_left.as_ref().map(|v| v.backing)),
            (matrix.right(), Some(right.backing)),
        ] {
            if payload.is_some_and(|data| data.len() != operand.layout().backing_words()) {
                return Err(Error::Malformed(
                    "bulk exact external allocation backing".into(),
                ));
            }
        }
        let input_words = right
            .backing
            .len()
            .checked_add(external_left.as_ref().map_or(0, |v| v.backing.len()))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Limit("bulk aggregate staging bytes".into()))?;
        // Both private external payloads share ONE exact staging ticket before
        // either allocation. Learned raw matrix remains borrowed from its owner.
        let staging = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: input_words as u64,
            }],
        )?;
        let owner = Arc::new(BulkOwner {
            model,
            matrix,
            admission,
            _metadata: metadata,
        });
        let copy = |data: &[f32]| -> Result<Vec<f32>> {
            not_cancelled(cancelled)?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(data.len())
                .map_err(|_| Error::Limit("bulk external payload allocation".into()))?;
            if values.capacity() != data.len() {
                return Err(Error::Limit("bulk external actual capacity".into()));
            }
            values.extend_from_slice(data);
            not_cancelled(cancelled)?;
            Ok(values)
        };
        let left = match external_left {
            Some(value) => LeftInput::External(copy(value.backing)?),
            None => LeftInput::Source,
        };
        let inputs = Arc::new(BulkInputs {
            owner,
            left,
            right: copy(right.backing)?,
            _staging: staging,
        });
        // Raw source alignment and all logical finite words are subsequently
        // checked by the unchanged primitive before its own output allocation.
        inputs.values(BulkF32Side::Left)?;
        not_cancelled(cancelled)?;
        Ok(Self { inputs })
    }
    pub fn matrix(&self) -> &BulkF32Matrix {
        &self.inputs.owner.matrix
    }
    pub fn record(&self) -> Result<RecordedOperatorSlice> {
        let mut builder = PlanBuilder::new_operator_slice(
            self.matrix().source(),
            ProgramNumericalProfile::GgufSourceV1,
        )?;
        let output = crate::forward::bulk_f32_slice(&mut builder, self.matrix())?;
        builder.finish_operator_slice(output)
    }
    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<BulkF32Output> {
        self.execute_with_provider(None, cancelled)
    }
    pub fn execute_with_provider(
        &self,
        kernel: Option<&dyn crate::accelerator::MatVecKernel>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<BulkF32Output> {
        not_cancelled(cancelled)?;
        if kernel.is_some() {
            return Err(Error::Unsupported(
                "unaware provider cannot execute Haswell matrix contract".into(),
            ));
        }
        let mut math = crate::host_math::HostMath::for_bulk_f32_slice(self, cancelled);
        let value = crate::forward::bulk_f32_slice(&mut math, self.matrix())?;
        not_cancelled(cancelled)?;
        Ok(BulkF32Output {
            value,
            shape: self.matrix().output().shape(),
            source: self.matrix().source(),
        })
    }
}
#[derive(Clone)]
pub struct BulkF32Output {
    value: crate::host_math::HostValue,
    shape: [usize; 4],
    source: crate::accelerator::SourceIdentity,
}
impl BulkF32Output {
    pub fn values(&self) -> Result<&[f32]> {
        self.value.as_slice()
    }
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn contract(&self) -> BulkF32Contract {
        BulkF32Contract::HaswellTinyBlasF32V1
    }
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<&[f32]> {
        self.values()?
            .get(range)
            .ok_or_else(|| Error::Malformed("bulk borrowed output range".into()))
    }
}
pub(crate) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("closed bulk matrix cancelled".into()));
    }
    Ok(())
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-model-qwen35/src/forward/bulk_f32.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 447; SHA256 `fd2e3353db051a2a5ad8efd5254e69655b555c0d970dd7802a9c3d90a40739ff`.

````text
//! One closed matrix fragment via the existing generic mathematical engine.
use crate::Result;
use crate::accelerator::program::{BulkF32Matrix, MathematicalBackend};
pub(crate) fn bulk_f32_slice<M: MathematicalBackend>(
    math: &mut M,
    matrix: &BulkF32Matrix,
) -> Result<M::Value> {
    let left = math.bulk_f32_input(matrix.left())?;
    let right = math.bulk_f32_input(matrix.right())?;
    math.bulk_f32_matmul(matrix, &left, &right)
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-model-qwen35/src/forward.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 23918; SHA256 `c363384dfc8fce2d5730db47db880e3be307bbb94ffc70079a42add88d439215`.

````text
//! Single Qwen forward schedule over host values or recorded values.
//! No backend owns a copied Qwen token/layer loop.

use super::{CachedLayer, LoadedQwenModel, layer_tensor};
use crate::accelerator::program::{
    AttentionGeometry, ConvDomain, ForwardInputs, ForwardPlan, KvDomain, MathematicalBackend,
    OrdinaryRmsContract, PlanBuilder, RecurrentDomain, RopeGeometry, SourceVector, StateSlot,
    TensorView,
};
use crate::model::LayerKind;
use crate::operator_slice::OperatorSliceSource;
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, Result};

/// One family-neutral Qwen **layer traversal** shared by the existing
/// GGUF CPU/recorded MathematicalBackend and source-verified original HF
/// analytical execution. Layer mathematics remain profile-specific; this
/// iterator never allocates a session, buffers a model or publishes a root.
pub fn run_layer_sequence<V>(
    layers: &[LayerKind],
    mut residual: V,
    mut evaluate_layer: impl FnMut(usize, LayerKind, V) -> Result<V>,
) -> Result<V> {
    if layers.is_empty() || layers.len() > 4096 {
        return Err(Error::Limit(
            "Qwen layer schedule exceeds qualified bounds".into(),
        ));
    }
    for (index, &kind) in layers.iter().enumerate() {
        residual = evaluate_layer(index, kind, residual)?;
    }
    Ok(residual)
}

fn vector<'a>(model: &LoadedQwenModel, name: &'a str, values: &'a [f32]) -> SourceVector<'a> {
    SourceVector {
        source: model.source_identity(),
        name,
        values,
    }
}
fn project<B: MathematicalBackend>(
    model: &impl OperatorSliceSource,
    math: &mut B,
    role: QwenMatrixRole,
    input: &B::Value,
) -> Result<B::Value> {
    math.project_profiled(
        model.matrix_descriptor(role)?,
        input,
        model.projection_contract(),
    )
}

/// Pure fragment of this same source-role schedule. It never touches state,
/// normalizes a hidden activation or represents a complete Qwen layer/token.
pub(crate) fn embedding_projection_slice<B: MathematicalBackend>(
    model: &impl OperatorSliceSource,
    math: &mut B,
    token: B::Index,
    role: QwenMatrixRole,
) -> Result<(B::Value, B::Value)> {
    let embedding = math.embed_profiled(
        model.matrix_descriptor(QwenMatrixRole::Embedding)?,
        token,
        model.embedding_contract(),
    )?;
    let projected = project(model, math, role, &embedding)?;
    let expected = match model.program_numerical_profile() {
        crate::accelerator::program::ProgramNumericalProfile::GgufSourceV1 => {
            crate::accelerator::program::ActivationPrecision::F32
        }
        crate::accelerator::program::ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1 => {
            crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
        }
    };
    if math.precision(&embedding) != expected || math.precision(&projected) != expected {
        return Err(Error::State(
            "source operator slice activation precision".into(),
        ));
    }
    Ok((embedding, projected))
}
/// Pure original layer0 input prefix through this same backend/schedule seam.
/// No layer loop, cache proposal, token advance or full-HF execution admission.
pub(crate) fn embedding_input_norm_projection_slice<B: MathematicalBackend>(
    model: &impl OperatorSliceSource,
    math: &mut B,
    token: B::Index,
) -> Result<(B::Value, B::Value, B::Value)> {
    use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
    if model.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || model.rms_epsilon().to_bits() != 1e-6f32.to_bits()
    {
        return Err(Error::Unsupported(
            "original ordinary RMS slice profile/epsilon".into(),
        ));
    }
    let embedding = math.embed_profiled(
        model.matrix_descriptor(QwenMatrixRole::Embedding)?,
        token,
        model.embedding_contract(),
    )?;
    let normalized = math.ordinary_rms_profiled(
        model.parameter_descriptor(QwenParameterRole::InputNorm(0))?,
        &embedding,
        OrdinaryRmsContract::TorchCpuBf16Avx2,
    )?;
    let projected = project(model, math, QwenMatrixRole::GdnQkv(0), &normalized)?;
    if [&embedding, &normalized, &projected]
        .iter()
        .any(|value| math.precision(value) != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State(
            "ordinary RMS slice activation precision".into(),
        ));
    }
    Ok((embedding, normalized, projected))
}

/// Pure prefix of the existing attention role schedule, before rotary/state.
pub(crate) fn attention_projection_norm_slice<B: MathematicalBackend>(
    model: &impl OperatorSliceSource,
    math: &mut B,
    layer: usize,
    input: &B::Value,
) -> Result<crate::operator_slice::attention::AttentionProjectionValues<B::Value>> {
    use crate::accelerator::program::ActivationPrecision;
    use crate::operator_slice::attention::{AttentionProjectionValues, attention_roles};
    let roles = attention_roles(model, layer)?;
    if math.elements(input) != roles.hidden
        || math.precision(input) != ActivationPrecision::Bf16WidenedF32
    {
        return Err(Error::Malformed(
            "attention slice typed full hidden input".into(),
        ));
    }
    let query_gate_projection =
        math.project_profiled(roles.query, input, model.projection_contract())?;
    let key_projection = math.project_profiled(roles.key, input, model.projection_contract())?;
    let value_projection =
        math.project_profiled(roles.value, input, model.projection_contract())?;
    for (value, elements) in [
        (&query_gate_projection, roles.query_heads * roles.width * 2),
        (&key_projection, roles.kv_heads * roles.width),
        (&value_projection, roles.kv_heads * roles.width),
    ] {
        expect_elements(math, value, elements, "attention slice projected geometry")?;
    }
    let query_view = math.view(
        &query_gate_projection,
        TensorView {
            offset: 0,
            rows: roles.query_heads,
            width: roles.width,
            row_stride: roles.width * 2,
        },
    )?;
    let query = math.contiguous(&query_view)?;
    let gate_view = math.view(
        &query_gate_projection,
        TensorView {
            offset: roles.width,
            rows: roles.query_heads,
            width: roles.width,
            row_stride: roles.width * 2,
        },
    )?;
    let gate = math.contiguous(&gate_view)?;
    let normalized_query = math.ordinary_rms_profiled(
        roles.query_norm,
        &query,
        OrdinaryRmsContract::TorchCpuBf16Avx2,
    )?;
    let normalized_key = math.ordinary_rms_profiled(
        roles.key_norm,
        &key_projection,
        OrdinaryRmsContract::TorchCpuBf16Avx2,
    )?;
    if [
        &query_gate_projection,
        &key_projection,
        &value_projection,
        &query,
        &gate,
        &normalized_query,
        &normalized_key,
    ]
    .iter()
    .any(|value| math.precision(value) != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State("attention slice output precision".into()));
    }
    Ok(AttentionProjectionValues {
        query_gate_projection,
        key_projection,
        value_projection,
        query,
        gate,
        normalized_query,
        normalized_key,
    })
}

fn expect_elements<B: MathematicalBackend>(
    math: &B,
    value: &B::Value,
    expected: usize,
    message: &str,
) -> Result<()> {
    if math.elements(value) != expected {
        return Err(Error::Malformed(message.into()));
    }
    Ok(())
}

pub(super) fn forward_token<B: MathematicalBackend>(
    model: &LoadedQwenModel,
    math: &mut B,
    input: ForwardInputs<B::Index>,
    need_logits: bool,
) -> Result<Option<B::Value>> {
    model.require_execution_profile()?;
    let x = math.embed_profiled(
        model.matrix_descriptor(QwenMatrixRole::Embedding)?,
        input.token,
        model.embedding_contract(),
    )?;
    math.observe(usize::MAX, "embedding", &x)?;
    let x = run_layer_sequence(&model.geometry.layers, x, |i, kind, mut x| {
        let layer = &model.layers[i];
        let name = layer_tensor(i, "attn_norm.weight");
        let normal = math.rms(&name, &x, vector(model, &name, &layer.attn_norm), model.eps)?;
        math.observe(i, "attention_input_norm", &normal)?;
        let attn = match kind {
            LayerKind::LinearAttention => gdn_layer(model, math, i, &normal, layer)?,
            LayerKind::FullAttention => attention_layer(model, math, i, &normal, layer, input)?,
            LayerKind::SlidingAttention => {
                return Err(Error::Unsupported(
                    "Qwen dense forward has no sliding layer".into(),
                ));
            }
        };
        expect_elements(
            math,
            &attn,
            math.elements(&x),
            "attention output hidden width",
        )?;
        math.observe(i, "attention_output", &attn)?;
        x = math.add(&layer_tensor(i, "attn_residual"), &x, &attn)?;
        math.observe(i, "post_attention_residual", &x)?;
        let name = layer_tensor(i, "post_attention_norm.weight");
        let norm = math.rms(&name, &x, vector(model, &name, &layer.ffn_norm), model.eps)?;
        math.observe(i, "ffn_input_norm", &norm)?;
        let gate = project(model, math, QwenMatrixRole::FfnGate(i), &norm)?;
        let up = project(model, math, QwenMatrixRole::FfnUp(i), &norm)?;
        expect_elements(math, &gate, layer.ffn_dim, "FFN projection mismatch")?;
        expect_elements(math, &up, layer.ffn_dim, "FFN projection mismatch")?;
        let fused = math.silu(&layer_tensor(i, "ffn_silu"), &gate, Some(&up))?;
        let ff = project(model, math, QwenMatrixRole::FfnDown(i), &fused)?;
        math.observe(i, "ffn_output", &ff)?;
        x = math.add(&layer_tensor(i, "ffn_residual"), &x, &ff)?;
        math.assert_finite(i, &x)?;
        math.observe(i, "post_ffn_residual", &x)?;
        Ok(x)
    })?;
    // Private proposed metadata advances at the same point as the old CPU
    // schedule. Later final norm/vocabulary failures still abort the draft.
    math.advance_frontier()?;
    if !need_logits {
        return Ok(None);
    }
    let name = "output_norm.weight";
    let norm = math.rms(name, &x, vector(model, name, &model.final_norm), model.eps)?;
    math.observe(usize::MAX, "final_norm", &norm)?;
    Ok(Some(project(model, math, QwenMatrixRole::Output, &norm)?))
}

fn gdn_layer<B: MathematicalBackend>(
    model: &LoadedQwenModel,
    math: &mut B,
    i: usize,
    x: &B::Value,
    layer: &CachedLayer,
) -> Result<B::Value> {
    let heads = model.geometry.recurrent_value_heads.unwrap() as usize;
    let d = layer.ssm_norm.as_ref().unwrap().len();
    let width = heads * d;
    let qkv = project(model, math, QwenMatrixRole::GdnQkv(i), x)?;
    let z = project(model, math, QwenMatrixRole::GdnGate(i), x)?;
    let alpha = project(model, math, QwenMatrixRole::GdnAlpha(i), x)?;
    let beta = project(model, math, QwenMatrixRole::GdnBeta(i), x)?;
    expect_elements(math, &qkv, width * 3, "GDN projected dimensions mismatch")?;
    expect_elements(math, &z, width, "GDN projected dimensions mismatch")?;
    expect_elements(math, &alpha, heads, "GDN projected dimensions mismatch")?;
    expect_elements(math, &beta, heads, "GDN projected dimensions mismatch")?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "gdn_qkv_raw", &qkv)?;
        math.observe(i, "gdn_z_raw", &z)?;
    }
    let name = layer_tensor(i, "ssm_conv1d.weight");
    let raw = math.conv_step(
        &name,
        ConvDomain(i),
        &qkv,
        vector(model, &name, layer.conv_kernel.as_ref().unwrap()),
    )?;
    let mixed = math.silu(&layer_tensor(i, "gdn_conv_silu"), &raw, None)?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "gdn_conv_raw", &raw)?;
        math.observe(i, "gdn_conv_silu", &mixed)?;
    }
    let dt_name = layer_tensor(i, "ssm_dt.bias");
    let a_name = layer_tensor(i, "ssm_a");
    let coefficients = math.coefficients(
        &layer_tensor(i, "gdn_coefficients"),
        &alpha,
        &beta,
        vector(model, &dt_name, layer.dt.as_ref().unwrap()),
        vector(model, &a_name, layer.a.as_ref().unwrap()),
        1.0 / (d as f32).sqrt(),
    )?;
    let qk = math.view(&mixed, TensorView::vector(0, width * 2))?;
    let scale = vec![1.0 / (d as f32).sqrt(); d];
    let name = layer_tensor(i, "gdn_qk_l2");
    let normalized_qk = math.grouped_rms(
        &name,
        &qk,
        vector(model, &name, &scale),
        heads * 2,
        model.eps / d as f32,
    )?;
    let q = math.view(&normalized_qk, TensorView::vector(0, width))?;
    let k = math.view(&normalized_qk, TensorView::vector(width, width))?;
    let v = math.view(&mixed, TensorView::vector(width * 2, width))?;
    let core = math.recurrent_step(
        &layer_tensor(i, "gdn_recurrent"),
        RecurrentDomain(i),
        &q,
        &k,
        &v,
        &coefficients.packed,
        d,
    )?;
    let name = layer_tensor(i, "ssm_norm.weight");
    let normalized = math.grouped_rms(
        &name,
        &core,
        vector(model, &name, layer.ssm_norm.as_ref().unwrap()),
        heads,
        model.eps,
    )?;
    let output = math.silu(&layer_tensor(i, "gdn_output_gate"), &z, Some(&normalized))?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        if i == 0 {
            math.observe_state(i, "gdn_state", StateSlot::Recurrent(RecurrentDomain(i)))?;
        }
        math.observe(i, "gdn_q_norm", &q)?;
        math.observe(i, "gdn_k_norm", &k)?;
        math.observe(i, "gdn_v", &v)?;
        math.observe(i, "gdn_alpha", &alpha)?;
        math.observe(i, "gdn_alpha_softplus", &coefficients.softplus)?;
        math.observe(i, "gdn_gate_raw", &coefficients.gate_raw)?;
        let gain = math.view(
            &coefficients.packed,
            TensorView {
                offset: 0,
                rows: heads,
                width: 1,
                row_stride: 3,
            },
        )?;
        let gain = math.contiguous(&gain)?;
        math.observe(i, "gdn_gain", &gain)?;
        let beta = math.view(
            &coefficients.packed,
            TensorView {
                offset: 1,
                rows: heads,
                width: 1,
                row_stride: 3,
            },
        )?;
        let beta = math.contiguous(&beta)?;
        math.observe(i, "gdn_beta", &beta)?;
        math.observe(i, "gdn_core_output", &core)?;
        math.observe(i, "gdn_gated_output", &output)?;
    }
    project(model, math, QwenMatrixRole::GdnOutput(i), &output)
}

fn attention_layer<B: MathematicalBackend>(
    model: &LoadedQwenModel,
    math: &mut B,
    i: usize,
    x: &B::Value,
    layer: &CachedLayer,
    input: ForwardInputs<B::Index>,
) -> Result<B::Value> {
    let n_q = model.geometry.num_attention_heads as usize;
    let n_kv = model.geometry.num_key_value_heads as usize;
    let d = model.geometry.head_dim as usize;
    let q_proj = project(model, math, QwenMatrixRole::Query(i), x)?;
    let k_proj = project(model, math, QwenMatrixRole::Key(i), x)?;
    let v_proj = project(model, math, QwenMatrixRole::Value(i), x)?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "full_qg_raw", &q_proj)?;
        math.observe(i, "full_k_raw", &k_proj)?;
        math.observe(i, "full_v_raw", &v_proj)?;
    }
    if n_kv == 0 || !n_q.is_multiple_of(n_kv) {
        return Err(Error::Malformed("GQA tensor head count mismatch".into()));
    }
    expect_elements(math, &q_proj, n_q * d * 2, "GQA tensor head count mismatch")?;
    expect_elements(math, &k_proj, n_kv * d, "GQA tensor head count mismatch")?;
    expect_elements(math, &v_proj, n_kv * d, "GQA tensor head count mismatch")?;
    let q_view = math.view(
        &q_proj,
        TensorView {
            offset: 0,
            rows: n_q,
            width: d,
            row_stride: d * 2,
        },
    )?;
    let raw_q = math.contiguous(&q_view)?;
    let g_view = math.view(
        &q_proj,
        TensorView {
            offset: d,
            rows: n_q,
            width: d,
            row_stride: d * 2,
        },
    )?;
    let gates = math.contiguous(&g_view)?;
    let name = layer_tensor(i, "attn_q_norm.weight");
    let norm_q = math.grouped_rms(
        &name,
        &raw_q,
        vector(model, &name, layer.q_norm.as_ref().unwrap()),
        n_q,
        model.eps,
    )?;
    let name = layer_tensor(i, "attn_k_norm.weight");
    let norm_k = math.grouped_rms(
        &name,
        &k_proj,
        vector(model, &name, layer.k_norm.as_ref().unwrap()),
        n_kv,
        model.eps,
    )?;
    math.validate_kv_prefix(KvDomain(i), input.position)?;
    let geometry = AttentionGeometry {
        width: d,
        query_heads: n_q,
        kv_heads: n_kv,
    };
    let rotated = math.rope(
        &layer_tensor(i, "attn_rope_kv"),
        &norm_q,
        &norm_k,
        &v_proj,
        input.position,
        RopeGeometry {
            attention: geometry,
            rotate: 64,
            theta_scale: 10_000_000.0f32.powf(-2.0 / 64.0),
        },
    )?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "full_q_norm", &norm_q)?;
        math.observe(i, "full_k_norm", &norm_k)?;
        math.observe(i, "full_q_rope", &rotated.query)?;
        math.observe(i, "full_k_rope", &rotated.key)?;
        math.observe(i, "full_gate_raw", &gates)?;
    }
    math.append_kv(KvDomain(i), &rotated.key, &rotated.value)?;
    let qk = math.attention_qk(
        &layer_tensor(i, "attn_qk"),
        KvDomain(i),
        &rotated.query,
        input.valid_tokens,
        geometry,
        1.0 / (d as f32).sqrt(),
    )?;
    let probabilities = math.attention_softmax(
        &layer_tensor(i, "attn_softmax"),
        &qk.scores,
        n_q,
        input.valid_tokens,
    )?;
    let result = math.attention_value(
        &layer_tensor(i, "attn_value"),
        KvDomain(i),
        &probabilities,
        input.valid_tokens,
        geometry,
    )?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "full_qk_dots", &qk.dots)?;
        math.observe(i, "full_scores", &qk.scores)?;
        math.observe(i, "full_probs", &probabilities)?;
        math.observe(i, "full_pregate", &result)?;
    }
    let gated = math.sigmoid_mul(&layer_tensor(i, "attn_gate"), &result, &gates)?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    math.observe(i, "full_gated", &gated)?;
    project(model, math, QwenMatrixRole::AttentionOutput(i), &gated)
}

impl LoadedQwenModel {
    /// Record one bounded symbolic forward. The result contains no live
    /// device/session handles and is not evidence that any backend executed.
    pub fn forward_plan(
        &self,
        prefix_capacity: usize,
        need_logits: bool,
        capture: bool,
    ) -> Result<ForwardPlan> {
        self.require_execution_profile()?;
        if prefix_capacity == 0 || prefix_capacity > self.geometry.max_position_embeddings as usize
        {
            return Err(Error::Limit(
                "forward-plan prefix capacity exceeds model".into(),
            ));
        }
        let mut domains = Vec::new();
        for (layer, kind) in self.geometry.layers.iter().enumerate() {
            match kind {
                LayerKind::LinearAttention => {
                    domains.push(StateSlot::Conv(ConvDomain(layer)));
                    domains.push(StateSlot::Recurrent(RecurrentDomain(layer)));
                }
                LayerKind::FullAttention => domains.push(StateSlot::Kv(KvDomain(layer))),
                LayerKind::SlidingAttention => {
                    return Err(Error::Unsupported("Qwen forward-plan sliding state".into()));
                }
            }
        }
        let mut builder =
            PlanBuilder::new(self.source_identity(), prefix_capacity, domains, capture)?;
        let output = forward_token(self, &mut builder, PlanBuilder::inputs(), need_logits)?;
        builder.finish(output)
    }
}

#[cfg(all(test, feature = "cpu"))]
mod matrix_axes_tests {
    use super::*;
    use crate::accelerator::program::{MatrixAxes, SourceTensor};
    use crate::accelerator::{MatVecRegion, SourceIdentity};
    use zerogravity_core::quant::QuantKind;
    use zerogravity_core::source::{SourceTensorRef, TensorSource};

    struct OriginalRows {
        source: SourceIdentity,
        shape: [u64; 2],
        bytes: Vec<u8>,
    }
    impl TensorSource for OriginalRows {
        fn source_identity(&self) -> SourceIdentity {
            self.source
        }
        fn tensor(&self, _name: &str) -> Result<SourceTensorRef<'_>> {
            Ok(SourceTensorRef {
                kind: QuantKind::F32,
                shape: &self.shape,
                packed_bytes: &self.bytes,
            })
        }
    }

    #[test]
    fn gguf_and_hf_raw_axes_use_identical_original_cpu_row_math() {
        for (axes, shape) in [
            (MatrixAxes::InputOutput, [3, 2]),
            (MatrixAxes::OutputInput, [2, 3]),
        ] {
            let source = OriginalRows {
                source: SourceIdentity::for_new_allocation().unwrap(),
                shape,
                bytes: [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            };
            let tensor = SourceTensor::from_source(&source, "matrix", axes).unwrap();
            let actual = crate::weights::source_reference_projection(MatVecRegion {
                source_identity: tensor.source,
                tensor_name: &tensor.name,
                kind: tensor.kind,
                columns: tensor.columns,
                rows: tensor.rows,
                packed_weights: source.tensor("matrix").unwrap().packed_bytes,
                activation: &[7.0, 8.0, 9.0],
            })
            .unwrap();
            assert_eq!(actual, [50.0, 122.0]);
            assert_eq!(source.shape, shape);
        }
    }
}

/// Pure original window only; proposed history is not a StateEffect.
pub(crate) fn gdn_convolution_silu_slice<B: MathematicalBackend>(
    source: &impl OperatorSliceSource,
    math: &mut B,
    layer: usize,
    input: &B::Value,
    history: &B::Value,
) -> Result<crate::accelerator::program::CausalConvValues<B::Value>> {
    use crate::accelerator::program::{ActivationPrecision, CausalConvSiluContract};
    let (parameter, channels) = crate::operator_slice::gdn_conv::convolution_role(source, layer)?;
    if math.elements(input) != channels
        || math.elements(history) != channels * 3
        || [input, history]
            .iter()
            .any(|v| math.precision(v) != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State(
            "original convolution slice geometry/precision".into(),
        ));
    }
    let values = math.causal_conv_silu_profiled(
        parameter,
        input,
        history,
        CausalConvSiluContract::TorchCpuBf16Avx2,
    )?;
    for (value, elements) in [
        (&values.raw, channels),
        (&values.activated, channels),
        (&values.next_history, channels * 3),
    ] {
        if math.elements(value) != elements
            || math.precision(value) != ActivationPrecision::Bf16WidenedF32
        {
            return Err(Error::State(
                "original convolution region result geometry/precision".into(),
            ));
        }
    }
    Ok(values)
}

#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
#[path = "forward/bulk_f32.rs"]
mod bulk_f32;
#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
pub(crate) use bulk_f32::bulk_f32_slice;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8355; SHA256 `05d8495048f7af0dd01192fb203f893ae168a30e2e1be5fc687952be1a7f0fef`.

````text
//! Closed input/result backing under the SAME HostMath, no layer/token engine.
use super::*;
use crate::accelerator::program::{BulkF32Matrix, BulkF32Operand, BulkF32Side};
use crate::operator_slice::bulk_f32::{BulkF32Slice, BulkInputs, BulkOwner, not_cancelled};
use std::sync::atomic::AtomicBool;
use zerogravity_cpu::math::haswell_bulk_f32::{
    BulkF32Admission, BulkF32Completed, BulkF32Profile, BulkF32Request, F32View,
};

pub(super) struct BulkCall<'a> {
    slice: &'a BulkF32Slice,
    cancelled: Option<&'a AtomicBool>,
}
pub(super) enum BulkBacking {
    Input {
        inputs: Arc<BulkInputs>,
        side: BulkF32Side,
    },
    Output {
        completed: BulkF32Completed,
        owner: Arc<BulkOwner>,
    },
}
impl BulkBacking {
    pub(super) fn values(&self) -> Result<&[f32]> {
        match self {
            Self::Input { inputs, side } => inputs.values(*side),
            Self::Output { completed, .. } => Ok(completed.values()),
        }
    }
    fn owner(&self) -> &Arc<BulkOwner> {
        match self {
            Self::Input { inputs, .. } => &inputs.owner,
            Self::Output { owner, .. } => owner,
        }
    }
}
fn fp_environment() -> Result<()> {
    let mut controls = 0u32;
    // SAFETY: read-only control instruction writes one valid stack word.
    unsafe {
        std::arch::asm!("stmxcsr [{at}]",at=in(reg)&mut controls,options(nostack,preserves_flags));
    }
    if controls & 0xffc0 != 0x1f80 {
        return Err(Error::Unsupported(
            "closed Haswell profile requires unchanged RNE/noFTZ/noDAZ controls".into(),
        ));
    }
    Ok(())
}
fn request_view<'a>(
    source: crate::accelerator::SourceIdentity,
    operand: &'a BulkF32Operand,
    values: &'a [f32],
) -> F32View<'a> {
    let layout = operand.layout();
    F32View {
        source,
        name: operand.name(),
        values,
        offset: layout.offset(),
        shape: layout.shape(),
        strides: layout.strides(),
    }
}
impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
    pub(crate) fn for_bulk_f32_slice(
        slice: &'engine BulkF32Slice,
        cancelled: Option<&'engine AtomicBool>,
    ) -> Self {
        let mut math = Self::for_operator_slice(slice.inputs.owner.model.as_ref(), None);
        math.bulk_call = Some(BulkCall { slice, cancelled });
        math
    }
    fn bulk_call(&self) -> Result<&BulkCall<'engine>> {
        if self.engine.is_some()
            || self.state.is_some()
            || self.region_kernel.is_some()
            || self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
        {
            return Err(Error::Unsupported(
                "bulk F32 private same-engine host slice only".into(),
            ));
        }
        self.bulk_call
            .as_ref()
            .ok_or_else(|| Error::Unsupported("bulk F32 has no bound temporal input issuer".into()))
    }
    pub(super) fn import_bulk_f32(&self, operand: &BulkF32Operand) -> Result<HostValue> {
        let call = self.bulk_call()?;
        not_cancelled(call.cancelled)?;
        fp_environment()?;
        let owner = &call.slice.inputs.owner;
        let expected = match operand.side() {
            BulkF32Side::Left => owner.matrix.left(),
            BulkF32Side::Right => owner.matrix.right(),
        };
        if operand != expected || operand.span().source() != self.source.source_identity() {
            return Err(Error::State(
                "bulk input exact bound owner/span/layout".into(),
            ));
        }
        let backing = BulkBacking::Input {
            inputs: call.slice.inputs.clone(),
            side: operand.side(),
        };
        if backing.values()?.len() != operand.layout().backing_words() {
            return Err(Error::State("bulk actual input storage extent".into()));
        }
        HostValue::from_bulk(
            backing,
            operand.layout().backing_words(),
            self.source.source_identity(),
        )
    }
    pub(super) fn execute_bulk_f32(
        &self,
        matrix: &BulkF32Matrix,
        left: &HostValue,
        right: &HostValue,
    ) -> Result<HostValue> {
        let call = self.bulk_call()?;
        let owner = &call.slice.inputs.owner;
        not_cancelled(call.cancelled)?;
        fp_environment()?;
        if matrix != &owner.matrix {
            return Err(Error::State("bulk exact bound matrix/temporal span".into()));
        }
        for (value, operand) in [(left, matrix.left()), (right, matrix.right())] {
            let backing = value.data.bulk.as_ref().ok_or_else(|| {
                Error::State("bulk input has no admitted original backing".into())
            })?;
            if !Arc::ptr_eq(backing.owner(), owner)
                || value.source != Some(matrix.source())
                || value.profile != Some(ProgramNumericalProfile::GgufSourceV1)
                || value.precision != ActivationPrecision::F32
                || value.elements != operand.layout().backing_words()
                || value.view != TensorView::vector(0, value.elements)
                || !matches!(backing,BulkBacking::Input{side,..}if *side==operand.side())
            {
                return Err(Error::State(
                    "bulk foreign/partial/retagged input value".into(),
                ));
            }
        }
        let a = left.as_slice()?;
        let b = right.as_slice()?;
        let extent = |values: &[f32]| -> Result<std::ops::Range<usize>> {
            let start = values.as_ptr() as usize;
            let end = values
                .len()
                .checked_mul(4)
                .and_then(|n| start.checked_add(n))
                .ok_or_else(|| Error::Limit("bulk actual allocation range".into()))?;
            Ok(start..end)
        };
        let ar = extent(a)?;
        let br = extent(b)?;
        if ar.start < br.end && br.start < ar.end {
            return Err(Error::State(
                "bulk independent input backing aliases".into(),
            ));
        }
        // Unchanged source-derived primitive reserves actual result+scratch
        // BEFORE its Vec allocation. No old MatVec/provider fallback exists.
        let completed = zerogravity_cpu::math::haswell_bulk_f32::execute(
            BulkF32Request {
                profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
                source: matrix.source(),
                left: request_view(matrix.source(), matrix.left(), a),
                right: request_view(matrix.source(), matrix.right(), b),
            },
            BulkF32Admission {
                arbiter: &owner.admission.arbiter,
                resource: &owner.admission.resource,
                owner: &owner.admission.owner,
            },
            call.cancelled,
        )?;
        fp_environment()?;
        not_cancelled(call.cancelled)?;
        if completed.source() != matrix.source()
            || completed.shape() != matrix.output().shape()
            || completed.values().len() != matrix.output().backing_words()
        {
            return Err(Error::State(
                "bulk actual primitive result geometry/source".into(),
            ));
        }
        let value = HostValue::from_bulk(
            BulkBacking::Output {
                completed,
                owner: owner.clone(),
            },
            matrix.output().backing_words(),
            matrix.source(),
        )?;
        not_cancelled(call.cancelled)?;
        Ok(value)
    }
}
impl HostValue {
    fn from_bulk(
        backing: BulkBacking,
        elements: usize,
        source: crate::accelerator::SourceIdentity,
    ) -> Result<Self> {
        if !source.is_bound() || backing.values()?.len() != elements {
            return Err(Error::State("bulk retained backing extent/source".into()));
        }
        Ok(Self {
            data: Arc::new(HostStorage {
                values: Vec::new(),
                #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
                workspace: None,
                bulk: Some(backing),
            }),
            view: TensorView::vector(0, elements),
            elements,
            precision: ActivationPrecision::F32,
            source: Some(source),
            profile: Some(ProgramNumericalProfile::GgufSourceV1),
        })
    }
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-model-qwen35/src/host_math.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 42009; SHA256 `ba4b725f9ec132b9b3c019b4a2ed1348de6f099d719f8329be225033d0a656c8`.

````text
//! CPU/optional-region adapter for the single generic forward schedule.
//! The existing QwenState remains private and unchanged in representation.

use super::{
    LayerObservation, LayerState, QwenDenseCpu, QwenState, observe, source_gdn_recurrent_step,
};
use crate::accelerator::program::{
    ActivationPrecision, AttentionGeometry, CoefficientValues, ConvDomain, EmbeddingContract,
    KvDomain, MathematicalBackend, OrdinaryRmsContract, ProgramNumericalProfile,
    ProjectionContract, QkValues, RecurrentDomain, RopeGeometry, RopeValues, SourceParameter,
    SourceTensor, SourceVector, StateSlot, TensorView,
};
use crate::accelerator::{
    AttentionQkRegion, AttentionSoftmaxRegion, AttentionValueRegion, BinaryOperation, BinaryRegion,
    GdnCoefficientRegion, GdnCoefficients, GdnRecurrentExecution, GdnRecurrentRegion, RopeKvRegion,
};
use crate::operator_slice::OperatorSliceSource;
use crate::{Error, Result};
use std::sync::Arc;
#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
use zerogravity_artifacts::source_weights::WindowWorkspace;

struct HostStorage {
    values: Vec<f32>,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    bulk: Option<bulk_f32::BulkBacking>,
    // Payload drops before its actual same-source host reservation.
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    workspace: Option<Arc<WindowWorkspace>>,
}
impl std::ops::Deref for HostStorage {
    type Target = Vec<f32>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

enum HostSource<'a> {
    Loaded(&'a super::LoadedQwenModel),
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    Windows(&'a crate::original_binding::OriginalWindowOperators),
}
impl HostSource<'_> {
    fn metadata(&self) -> &dyn OperatorSliceSource {
        match self {
            Self::Loaded(source) => *source,
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::Windows(source) => *source,
        }
    }
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.metadata().source_identity()
    }
    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        self.metadata().program_numerical_profile()
    }
    fn rms_epsilon(&self) -> f32 {
        self.metadata().rms_epsilon()
    }
    fn matrix_descriptor(
        &self,
        role: crate::original_binding::QwenMatrixRole,
    ) -> Result<SourceTensor> {
        self.metadata().matrix_descriptor(role)
    }
    fn validate_tensor(&self, tensor: &SourceTensor) -> Result<()> {
        match self {
            Self::Loaded(source) => tensor.validate_source(*source),
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            Self::Windows(source) => source.validate_matrix(tensor),
        }
    }
}

type HostKvView<'a> = (&'a [Arc<[f32]>], &'a [Arc<[f32]>]);

/// Host views share a Vec allocation without cloning numerical payload. A
/// strided view is packed explicitly by contiguous(), never by hidden math.
#[derive(Clone)]
pub(super) struct HostValue {
    data: Arc<HostStorage>,
    view: TensorView,
    elements: usize,
    precision: ActivationPrecision,
    source: Option<crate::accelerator::SourceIdentity>,
    profile: Option<ProgramNumericalProfile>,
}
impl HostValue {
    fn new(values: Vec<f32>) -> Self {
        let elements = values.len();
        Self {
            data: Arc::new(HostStorage {
                values,
                #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
                bulk: None,
                #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
                workspace: None,
            }),
            view: TensorView::vector(0, elements),
            elements,
            precision: ActivationPrecision::F32,
            source: None,
            profile: None,
        }
    }
    fn typed(
        values: Vec<f32>,
        precision: ActivationPrecision,
        source: crate::accelerator::SourceIdentity,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        if !source.is_bound() {
            return Err(Error::State("unbound typed host activation".into()));
        }
        precision.validate_values(&values)?;
        let mut value = Self::new(values);
        value.precision = precision;
        value.source = Some(source);
        value.profile = Some(profile);
        Ok(value)
    }
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    fn retain_workspace(mut self, workspace: Option<Arc<WindowWorkspace>>) -> Result<Self> {
        Arc::get_mut(&mut self.data)
            .ok_or_else(|| Error::State("new host backing unexpectedly shared".into()))?
            .workspace = workspace;
        Ok(self)
    }
    pub(super) fn as_slice(&self) -> Result<&[f32]> {
        if !self.view.is_contiguous() {
            return Err(Error::State(
                "host operator requires explicit contiguous input".into(),
            ));
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let values = match &self.data.bulk {
            Some(backing) => backing.values()?,
            None => &self.data.values,
        };
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        let values = &self.data.values;
        values
            .get(
                self.view.offset
                    ..self
                        .view
                        .offset
                        .checked_add(self.elements)
                        .ok_or_else(|| Error::Limit("host view end overflow".into()))?,
            )
            .ok_or_else(|| Error::State("invalid private host tensor view".into()))
    }
    pub fn into_vec(self) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if self.data.bulk.is_some() {
            return Err(Error::Unsupported(
                "leased bulk result has no Vec escape".into(),
            ));
        }
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        if self.data.workspace.is_some() {
            return Err(Error::Unsupported(
                "leased window values require borrowed output access".into(),
            ));
        }
        if self.view.offset == 0 && self.view.is_contiguous() && self.elements == self.data.len() {
            return Ok(match Arc::try_unwrap(self.data) {
                Ok(data) => data.values,
                Err(data) => data.values.clone(),
            });
        }
        Ok(self.as_slice()?.to_vec())
    }
}

pub(super) struct HostMath<'engine, 'state, 'trace> {
    source: HostSource<'engine>,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    bulk_call: Option<bulk_f32::BulkCall<'engine>>,
    engine: Option<&'engine QwenDenseCpu>,
    state: Option<&'state mut QwenState>,
    region_kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
    trace: Option<&'trace mut Vec<LayerObservation>>,
    position: usize,
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    cancelled: Option<&'engine std::sync::atomic::AtomicBool>,
}
impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
    pub fn new(
        engine: &'engine QwenDenseCpu,
        state: &'state mut QwenState,
        trace: Option<&'trace mut Vec<LayerObservation>>,
    ) -> Self {
        let position = state.position;
        Self {
            source: HostSource::Loaded(engine.model.as_ref()),
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            bulk_call: None,
            engine: Some(engine),
            state: Some(state),
            region_kernel: engine.kernel.as_deref(),
            trace,
            position,
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            cancelled: None,
        }
    }
    pub fn for_operator_slice(
        source: &'engine super::LoadedQwenModel,
        kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
    ) -> Self {
        Self {
            source: HostSource::Loaded(source),
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            bulk_call: None,
            engine: None,
            state: None,
            region_kernel: kernel,
            trace: None,
            position: 0,
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            cancelled: None,
        }
    }
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    pub fn for_window_operator_slice(
        source: &'engine crate::original_binding::OriginalWindowOperators,
        cancelled: Option<&'engine std::sync::atomic::AtomicBool>,
        kernel: Option<&'engine dyn crate::accelerator::MatVecKernel>,
    ) -> Self {
        Self {
            source: HostSource::Windows(source),
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            bulk_call: None,
            engine: None,
            state: None,
            region_kernel: kernel,
            trace: None,
            position: 0,
            cancelled,
        }
    }
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    fn prepare_output(&self, input: Option<&HostValue>) -> Result<Option<Arc<WindowWorkspace>>> {
        let HostSource::Windows(source) = &self.source else {
            return Ok(None);
        };
        if self
            .cancelled
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(Error::State(
                "window host operation cancelled before admission".into(),
            ));
        }
        if let Some(input) = input {
            let prior = input.data.workspace.as_ref().ok_or_else(|| {
                Error::State("window activation has no retained physical backing".into())
            })?;
            if !source.source().same_issuer(prior.source()) {
                return Err(Error::State(
                    "window activation belongs to another source issuer".into(),
                ));
            }
        }
        Ok(Some(Arc::new(source.workspace()?)))
    }
    /// Explicit operator input transport, never a model producer or root.
    pub(super) fn import_attention_input(
        &self,
        input: crate::operator_slice::attention::AttentionSliceInput<'_>,
        hidden: usize,
    ) -> Result<HostValue> {
        input.validate()?;
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "typed attention CPU input provider".into(),
            ));
        }
        if input.source_identity() != self.source.source_identity()
            || input.numerical_profile() != self.source.program_numerical_profile()
            || self.source.metadata().geometry().hidden_size != hidden as u64
            || input.bytes().len()
                != hidden
                    .checked_mul(2)
                    .ok_or_else(|| Error::Limit("attention private input width".into()))?
        {
            return Err(Error::State(
                "attention input source/profile/full hidden width".into(),
            ));
        }
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let workspace = self.prepare_output(None)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(hidden)
            .map_err(|_| Error::Limit("private attention input transport".into()))?;
        values.extend(
            input.bytes().chunks_exact(2).map(|word| {
                f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
            }),
        );
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        if self
            .cancelled
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(Error::State(
                "attention input cancelled after transport".into(),
            ));
        }
        let output = HostValue::typed(
            values,
            input.precision(),
            input.source_identity(),
            input.numerical_profile(),
        )?;
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn engine(&self) -> Result<&'engine QwenDenseCpu> {
        self.engine
            .ok_or_else(|| Error::Unsupported("operator slice has no session engine".into()))
    }
    fn state(&self) -> Result<&QwenState> {
        self.state
            .as_deref()
            .ok_or_else(|| Error::State("operator slice has no session state".into()))
    }
    fn state_mut(&mut self) -> Result<&mut QwenState> {
        self.state
            .as_deref_mut()
            .ok_or_else(|| Error::State("operator slice has no session state".into()))
    }
    fn legacy_precision(&self, values: &[&HostValue]) -> Result<()> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if values.iter().any(|v| v.data.bulk.is_some()) {
            return Err(Error::Unsupported(
                "bulk F32 values require explicit matrix contract".into(),
            ));
        }
        if self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
            || values
                .iter()
                .any(|value| value.precision != ActivationPrecision::F32)
        {
            return Err(Error::Unsupported(
                "legacy host operation lacks explicit numerical contract".into(),
            ));
        }
        Ok(())
    }
    fn tensor(&self, tensor: &SourceTensor) -> Result<()> {
        self.source.validate_tensor(tensor)
    }
    fn vector(&self, vector: SourceVector<'_>) -> Result<()> {
        if vector.source != self.source.source_identity() {
            return Err(Error::State(
                "host forward static vector changed source".into(),
            ));
        }
        Ok(())
    }
    fn kv(&self, domain: KvDomain, tokens: usize) -> Result<HostKvView<'_>> {
        match self.state()?.layers.get(domain.0) {
            Some(LayerState::Full { keys, values })
                if keys.len() == tokens && values.len() == tokens =>
            {
                Ok((keys, values))
            }
            _ => Err(Error::State(
                "attention KV prefix does not match source position".into(),
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)] // Mirrors already qualified closed-op inputs.
impl MathematicalBackend for HostMath<'_, '_, '_> {
    type Value = HostValue;
    type Index = usize;
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn bulk_f32_input(
        &mut self,
        operand: &crate::accelerator::program::BulkF32Operand,
    ) -> Result<HostValue> {
        self.import_bulk_f32(operand)
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn bulk_f32_matmul(
        &mut self,
        matrix: &crate::accelerator::program::BulkF32Matrix,
        left: &HostValue,
        right: &HostValue,
    ) -> Result<HostValue> {
        self.execute_bulk_f32(matrix, left, right)
    }
    fn elements(&self, value: &HostValue) -> usize {
        value.elements
    }
    fn precision(&self, value: &HostValue) -> ActivationPrecision {
        value.precision
    }
    fn embed_profiled(
        &mut self,
        tensor: SourceTensor,
        token: usize,
        contract: EmbeddingContract,
    ) -> Result<HostValue> {
        if contract == EmbeddingContract::Source {
            return self.embed(tensor, token);
        }
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "physical provider does not declare typed CPU profile".into(),
            ));
        }
        self.tensor(&tensor)?;
        if self.source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || tensor.kind != zerogravity_core::quant::QuantKind::Bf16
            || tensor.axes != crate::accelerator::program::MatrixAxes::OutputInput
            || tensor
                != self
                    .source
                    .matrix_descriptor(crate::original_binding::QwenMatrixRole::Embedding)?
        {
            return Err(Error::Unsupported(
                "host typed embedding source/contract".into(),
            ));
        }
        let token = u32::try_from(token)
            .map_err(|_| Error::Malformed("typed embedding token domain".into()))?;
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let workspace = self.prepare_output(None)?;
        let values = match &self.source {
            HostSource::Loaded(source) => {
                crate::weights::source_reference_embedding(source.embedding_source(token)?)?
            }
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            HostSource::Windows(source) => source.embed(&tensor, token as usize, self.cancelled)?,
        };
        let output = HostValue::typed(
            values,
            ActivationPrecision::Bf16WidenedF32,
            self.source.source_identity(),
            self.source.program_numerical_profile(),
        )?;
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn project_profiled(
        &mut self,
        tensor: SourceTensor,
        input: &HostValue,
        contract: ProjectionContract,
    ) -> Result<HostValue> {
        if contract == ProjectionContract::Source {
            return self.project(tensor, input);
        }
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "physical provider does not declare typed CPU profile".into(),
            ));
        }
        self.tensor(&tensor)?;
        if self.source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || input.precision != ActivationPrecision::Bf16WidenedF32
            || input.source != Some(self.source.source_identity())
            || input.profile != Some(self.source.program_numerical_profile())
            || tensor.kind != zerogravity_core::quant::QuantKind::Bf16
            || tensor.axes != crate::accelerator::program::MatrixAxes::OutputInput
        {
            return Err(Error::Unsupported(
                "host typed projection source/contract".into(),
            ));
        }
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let workspace = self.prepare_output(Some(input))?;
        let values = match &self.source {
            HostSource::Loaded(source) => {
                zerogravity_cpu::original_hf::output_first_projection_with_profile(
                    self.source.source_identity(),
                    &tensor.name,
                    zerogravity_core::source::TensorSource::tensor(*source, &tensor.name)?,
                    input.as_slice()?,
                    self.region_kernel,
                    crate::weights::SourceProjectionProfile::TorchCpuBf16Avx2,
                )?
            }
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            HostSource::Windows(source) => {
                source.project(&tensor, input.as_slice()?, self.cancelled)?
            }
        };
        let values = values
            .into_iter()
            .map(round_bf16)
            .collect::<Result<Vec<_>>>()?;
        let output = HostValue::typed(
            values,
            ActivationPrecision::Bf16WidenedF32,
            self.source.source_identity(),
            self.source.program_numerical_profile(),
        )?;
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn ordinary_rms_profiled(
        &mut self,
        parameter: SourceParameter,
        input: &HostValue,
        contract: OrdinaryRmsContract,
    ) -> Result<HostValue> {
        if self.region_kernel.is_some() {
            return Err(Error::Unsupported(
                "physical provider does not declare typed ordinary RMS contract".into(),
            ));
        }
        if self.source.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            || self.source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
            || input.precision != ActivationPrecision::Bf16WidenedF32
            || input.source != Some(self.source.source_identity())
            || input.profile != Some(self.source.program_numerical_profile())
        {
            return Err(Error::Unsupported(
                "host ordinary RMS numerical profile/epsilon/source/precision".into(),
            ));
        }
        let width = contract.validate(&parameter, input.elements)?;
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let workspace = self.prepare_output(Some(input))?;
        let values = match &self.source {
            HostSource::Loaded(source) => {
                parameter.validate_source(*source)?;
                let raw = zerogravity_core::source::TensorSource::tensor(*source, &parameter.name)?;
                let output = zerogravity_cpu::original_hf::norm::ordinary_rms(
                    self.source.source_identity(),
                    raw,
                    input.as_slice()?,
                    width,
                )?;
                output.output_bf16().to_vec()
            }
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            HostSource::Windows(source) => {
                source.normalize(&parameter, input.as_slice()?, self.cancelled)?
            }
        };
        let output = HostValue::typed(
            values,
            ActivationPrecision::Bf16WidenedF32,
            self.source.source_identity(),
            self.source.program_numerical_profile(),
        )?;
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn causal_conv_silu_profiled(
        &mut self,
        parameter: SourceParameter,
        input: &HostValue,
        history: &HostValue,
        contract: crate::accelerator::program::CausalConvSiluContract,
    ) -> Result<crate::accelerator::program::CausalConvValues<HostValue>> {
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        {
            self.execute_causal_conv_silu(parameter, input, history, contract)
        }
        #[cfg(not(all(feature = "original-hf-binding", target_os = "linux")))]
        {
            let _ = (parameter, input, history, contract);
            Err(Error::Unsupported(
                "causal convolution requires leased original windows".into(),
            ))
        }
    }
    fn embed(&mut self, tensor: SourceTensor, token: usize) -> Result<HostValue> {
        self.legacy_precision(&[])?;
        self.tensor(&tensor)?;
        if tensor.name != "token_embd.weight" {
            return Err(Error::State("unexpected Qwen embedding role".into()));
        }
        let token = u32::try_from(token)
            .map_err(|_| Error::Malformed("embedding ID exceeds source token domain".into()))?;
        Ok(HostValue::new(self.engine()?.embedding_region(token)?))
    }
    fn project(&mut self, tensor: SourceTensor, input: &HostValue) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        self.tensor(&tensor)?;
        Ok(HostValue::new(
            self.engine()?.project(&tensor.name, input.as_slice()?)?,
        ))
    }
    fn view(&mut self, input: &HostValue, view: TensorView) -> Result<HostValue> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if input.data.bulk.is_some() {
            return Err(Error::Unsupported(
                "bulk geometry cannot be reinterpreted by legacy view".into(),
            ));
        }
        // A view of an already strided view must first use explicit packing.
        if !input.view.is_contiguous() {
            return Err(Error::State(
                "nested strided host view requires packing".into(),
            ));
        }
        view.validate(input.elements)?;
        let elements = view.elements()?;
        let offset = input
            .view
            .offset
            .checked_add(view.offset)
            .ok_or_else(|| Error::Limit("host tensor offset overflow".into()))?;
        Ok(HostValue {
            data: input.data.clone(),
            view: TensorView { offset, ..view },
            elements,
            precision: input.precision,
            source: input.source,
            profile: input.profile,
        })
    }
    fn contiguous(&mut self, input: &HostValue) -> Result<HostValue> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if input.data.bulk.is_some() && input.view != TensorView::vector(0, input.elements) {
            return Err(Error::Unsupported(
                "bulk geometry cannot be implicitly packed".into(),
            ));
        }
        if input.view.is_contiguous() {
            return Ok(input.clone());
        }
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let workspace = self.prepare_output(Some(input))?;
        let mut values = Vec::with_capacity(input.elements);
        for row in 0..input.view.rows {
            let offset = input.view.offset + row * input.view.row_stride;
            values.extend_from_slice(&input.data[offset..offset + input.view.width]);
        }
        let output = match input.source {
            Some(source) => HostValue::typed(
                values,
                input.precision,
                source,
                input
                    .profile
                    .ok_or_else(|| Error::State("typed host profile missing".into()))?,
            ),
            None => Ok(HostValue::new(values)),
        }?;
        #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
        let output = output.retain_workspace(workspace)?;
        Ok(output)
    }
    fn rms(
        &mut self,
        name: &str,
        input: &HostValue,
        gamma: SourceVector<'_>,
        eps: f32,
    ) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        self.vector(gamma)?;
        if eps.to_bits() != self.engine()?.eps.to_bits() {
            return Err(Error::State(
                "host residual RMS changed source epsilon".into(),
            ));
        }
        Ok(HostValue::new(self.engine()?.normalize_region(
            name,
            input.as_slice()?,
            gamma.values,
        )?))
    }
    fn grouped_rms(
        &mut self,
        name: &str,
        input: &HostValue,
        gamma: SourceVector<'_>,
        groups: usize,
        eps: f32,
    ) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        self.vector(gamma)?;
        Ok(HostValue::new(self.engine()?.normalize_grouped_region(
            name,
            input.as_slice()?,
            gamma.values,
            groups,
            eps,
        )?))
    }
    fn silu(
        &mut self,
        name: &str,
        input: &HostValue,
        multiplier: Option<&HostValue>,
    ) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        if let Some(value) = multiplier {
            self.legacy_precision(&[value])?;
        }
        let multiplier = multiplier.map(HostValue::as_slice).transpose()?;
        Ok(HostValue::new(self.engine()?.silu_region(
            name,
            input.as_slice()?,
            multiplier,
        )?))
    }
    fn add(&mut self, name: &str, left: &HostValue, right: &HostValue) -> Result<HostValue> {
        self.legacy_precision(&[left, right])?;
        Ok(HostValue::new(self.engine()?.binary_region(
            BinaryRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                kind: BinaryOperation::Add,
                left: left.as_slice()?,
                right: right.as_slice()?,
            },
        )?))
    }
    fn sigmoid_mul(
        &mut self,
        name: &str,
        value: &HostValue,
        gate: &HostValue,
    ) -> Result<HostValue> {
        self.legacy_precision(&[value, gate])?;
        Ok(HostValue::new(self.engine()?.binary_region(
            BinaryRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                kind: BinaryOperation::SigmoidGate,
                left: value.as_slice()?,
                right: gate.as_slice()?,
            },
        )?))
    }
    fn conv_step(
        &mut self,
        name: &str,
        domain: ConvDomain,
        input: &HostValue,
        weights: SourceVector<'_>,
    ) -> Result<HostValue> {
        self.legacy_precision(&[input])?;
        self.vector(weights)?;
        let engine = self.engine()?;
        let Some(LayerState::Delta { conv, .. }) = self.state_mut()?.layers.get_mut(domain.0)
        else {
            return Err(Error::State("invalid private convolution domain".into()));
        };
        let input = input.as_slice()?;
        let mut output = vec![0.0; input.len()];
        engine.gdn_conv_region(
            name,
            input,
            weights.values,
            Arc::make_mut(conv).as_mut_slice(),
            &mut output,
        )?;
        Ok(HostValue::new(output))
    }
    fn coefficients(
        &mut self,
        name: &str,
        alpha: &HostValue,
        beta: &HostValue,
        dt: SourceVector<'_>,
        a: SourceVector<'_>,
        scale: f32,
    ) -> Result<CoefficientValues<HostValue>> {
        self.legacy_precision(&[alpha, beta])?;
        self.vector(dt)?;
        self.vector(a)?;
        let ready = self
            .engine()?
            .gdn_coefficient_region(GdnCoefficientRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                alpha: alpha.as_slice()?,
                beta: beta.as_slice()?,
                dt: dt.values,
                a: a.values,
                scale,
            })?;
        let packed = ready
            .coefficients
            .iter()
            .flat_map(|c| [c.gain, c.beta, c.scale])
            .collect();
        Ok(CoefficientValues {
            packed: HostValue::new(packed),
            softplus: HostValue::new(ready.softplus),
            gate_raw: HostValue::new(ready.gate_raw),
        })
    }
    fn recurrent_step(
        &mut self,
        name: &str,
        domain: RecurrentDomain,
        q: &HostValue,
        k: &HostValue,
        v: &HostValue,
        coefficients: &HostValue,
        width: usize,
    ) -> Result<HostValue> {
        self.legacy_precision(&[q, k, v, coefficients])?;
        let q = q.as_slice()?;
        let k = k.as_slice()?;
        let v = v.as_slice()?;
        let packed = coefficients.as_slice()?;
        if width == 0
            || !q.len().is_multiple_of(width)
            || (q.len() / width).checked_mul(3) != Some(packed.len())
        {
            return Err(Error::Malformed("host recurrent coefficient layout".into()));
        }
        let coefficients: Vec<_> = packed
            .chunks_exact(3)
            .map(|c| GdnCoefficients {
                gain: c[0],
                beta: c[1],
                scale: c[2],
            })
            .collect();
        let engine = self.engine()?;
        let source_identity = self.source.source_identity();
        let Some(LayerState::Delta { state, .. }) = self.state_mut()?.layers.get_mut(domain.0)
        else {
            return Err(Error::State("invalid private recurrent domain".into()));
        };
        let recurrent = Arc::make_mut(state).as_mut_slice();
        let result = engine.gdn_recurrent_region(GdnRecurrentRegion {
            source_identity,
            operation: name,
            width,
            q,
            k,
            v,
            coefficients: &coefficients,
            state: recurrent,
        })?;
        let mut output = vec![0.0; q.len()];
        match result {
            GdnRecurrentExecution::Completed(ready) => {
                output.copy_from_slice(&ready.output);
                recurrent.copy_from_slice(&ready.state);
            }
            GdnRecurrentExecution::Unsupported => {
                for (h, &coeff) in coefficients.iter().enumerate() {
                    source_gdn_recurrent_step(
                        &mut recurrent[h * width * width..(h + 1) * width * width],
                        &q[h * width..(h + 1) * width],
                        &k[h * width..(h + 1) * width],
                        &v[h * width..(h + 1) * width],
                        coeff,
                        &mut output[h * width..(h + 1) * width],
                    )?;
                }
            }
        }
        Ok(HostValue::new(output))
    }
    fn validate_kv_prefix(&mut self, domain: KvDomain, position: usize) -> Result<()> {
        self.legacy_precision(&[])?;
        self.kv(domain, position)?;
        Ok(())
    }
    fn rope(
        &mut self,
        name: &str,
        q: &HostValue,
        k: &HostValue,
        v: &HostValue,
        position: usize,
        geometry: RopeGeometry,
    ) -> Result<RopeValues<HostValue>> {
        self.legacy_precision(&[q, k, v])?;
        let ready = self.engine()?.rope_kv_region(RopeKvRegion {
            source_identity: self.source.source_identity(),
            operation: name,
            query: q.as_slice()?,
            key: k.as_slice()?,
            value: v.as_slice()?,
            head_width: geometry.attention.width,
            query_heads: geometry.attention.query_heads,
            kv_heads: geometry.attention.kv_heads,
            rotate: geometry.rotate,
            position,
            theta_scale: geometry.theta_scale,
        })?;
        Ok(RopeValues {
            query: HostValue::new(ready.query),
            key: HostValue::new(ready.key),
            value: HostValue::new(ready.value),
        })
    }
    fn append_kv(&mut self, domain: KvDomain, k: &HostValue, v: &HostValue) -> Result<()> {
        self.legacy_precision(&[k, v])?;
        let Some(LayerState::Full { keys, values }) = self.state_mut()?.layers.get_mut(domain.0)
        else {
            return Err(Error::State("invalid private KV domain".into()));
        };
        keys.push(Arc::from(k.as_slice()?));
        values.push(Arc::from(v.as_slice()?));
        Ok(())
    }
    fn attention_qk(
        &mut self,
        name: &str,
        domain: KvDomain,
        query: &HostValue,
        valid_tokens: usize,
        geometry: AttentionGeometry,
        scale: f32,
    ) -> Result<QkValues<HostValue>> {
        self.legacy_precision(&[query])?;
        let (keys, _) = self.kv(domain, valid_tokens)?;
        let ready = self.engine()?.attention_qk_region(AttentionQkRegion {
            source_identity: self.source.source_identity(),
            operation: name,
            query: query.as_slice()?,
            keys,
            head_width: geometry.width,
            query_heads: geometry.query_heads,
            kv_heads: geometry.kv_heads,
            scale,
        })?;
        Ok(QkValues {
            dots: HostValue::new(ready.dots),
            scores: HostValue::new(ready.scores),
        })
    }
    fn attention_softmax(
        &mut self,
        name: &str,
        scores: &HostValue,
        heads: usize,
        valid_tokens: usize,
    ) -> Result<HostValue> {
        self.legacy_precision(&[scores])?;
        let ready = self
            .engine()?
            .attention_softmax_region(AttentionSoftmaxRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                scores: scores.as_slice()?,
                heads,
                width: valid_tokens,
            })?;
        Ok(HostValue::new(ready.probabilities))
    }
    fn attention_value(
        &mut self,
        name: &str,
        domain: KvDomain,
        probabilities: &HostValue,
        valid_tokens: usize,
        geometry: AttentionGeometry,
    ) -> Result<HostValue> {
        self.legacy_precision(&[probabilities])?;
        let (_, values) = self.kv(domain, valid_tokens)?;
        Ok(HostValue::new(self.engine()?.attention_value_region(
            AttentionValueRegion {
                source_identity: self.source.source_identity(),
                operation: name,
                probabilities: probabilities.as_slice()?,
                values,
                head_width: geometry.width,
                query_heads: geometry.query_heads,
                kv_heads: geometry.kv_heads,
                padded_tokens: valid_tokens.div_ceil(256) * 256,
            },
        )?))
    }
    fn assert_finite(&mut self, layer: usize, value: &HostValue) -> Result<()> {
        if !value.as_slice()?.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed(format!(
                "nonfinite layer {layer} computation"
            )));
        }
        Ok(())
    }
    fn observe(&mut self, layer: usize, stage: &'static str, value: &HostValue) -> Result<()> {
        if self.trace.is_some() {
            observe(
                &mut self.trace,
                self.position,
                layer,
                stage,
                value.as_slice()?,
            );
        }
        Ok(())
    }
    fn observe_state(
        &mut self,
        layer: usize,
        stage: &'static str,
        domain: StateSlot,
    ) -> Result<()> {
        if self.trace.is_some() {
            let state = self
                .state
                .as_deref()
                .ok_or_else(|| Error::State("operator slice has no observed state".into()))?;
            let values = match domain {
                StateSlot::Conv(ConvDomain(i)) => match state.layers.get(i) {
                    Some(LayerState::Delta { conv, .. }) => conv.as_slice(),
                    _ => return Err(Error::State("invalid observed convolution domain".into())),
                },
                StateSlot::Recurrent(RecurrentDomain(i)) => match state.layers.get(i) {
                    Some(LayerState::Delta { state, .. }) => state.as_slice(),
                    _ => return Err(Error::State("invalid observed recurrent domain".into())),
                },
                StateSlot::Kv(_) => {
                    return Err(Error::Unsupported(
                        "KV state observation needs explicit slab export".into(),
                    ));
                }
            };
            observe(&mut self.trace, self.position, layer, stage, values);
        }
        Ok(())
    }
    fn advance_frontier(&mut self) -> Result<()> {
        if self.state()?.position != self.position {
            return Err(Error::State(
                "duplicate host proposed frontier advance".into(),
            ));
        }
        let next = self
            .state()?
            .position
            .checked_add(1)
            .ok_or_else(|| Error::Limit("host proposed position overflow".into()))?;
        self.state_mut()?.position = next;
        Ok(())
    }
}

fn round_bf16(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(Error::Malformed("nonfinite typed projection output".into()));
    }
    let bits = value.to_bits();
    let rounded = f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000);
    if !rounded.is_finite() {
        return Err(Error::Limit("typed BF16 projection output overflow".into()));
    }
    Ok(rounded)
}

#[cfg(test)]
mod typed_value_tests {
    use super::*;
    #[test]
    fn typed_values_require_explicit_source_precision_and_keep_profile() {
        let source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
        let profile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
        assert!(
            HostValue::typed(
                vec![1.0],
                ActivationPrecision::Bf16WidenedF32,
                crate::accelerator::SourceIdentity::unbound(),
                profile
            )
            .is_err()
        );
        assert!(
            HostValue::typed(
                vec![1.0001],
                ActivationPrecision::Bf16WidenedF32,
                source,
                profile
            )
            .is_err()
        );
        let typed = HostValue::typed(
            vec![-0.0, 1.0],
            ActivationPrecision::Bf16WidenedF32,
            source,
            profile,
        )
        .unwrap();
        assert_eq!(typed.source, Some(source));
        assert_eq!(typed.profile, Some(profile));
        assert_eq!(typed.as_slice().unwrap()[0].to_bits(), (-0.0f32).to_bits());
        let legacy = HostValue::new(vec![1.0]);
        assert_eq!(legacy.precision, ActivationPrecision::F32);
        assert_eq!(legacy.source, None);
        assert_eq!(legacy.profile, None);
    }
    #[test]
    fn projection_boundary_rounds_bf16_rne_and_refuses_finite_overflow() {
        for (input, expected) in [
            (0x3f80_8000, 0x3f80_0000),
            (0x3f81_8000, 0x3f82_0000),
            (0x8000_0000, 0x8000_0000),
        ] {
            assert_eq!(
                round_bf16(f32::from_bits(input)).unwrap().to_bits(),
                expected
            );
        }
        for value in [f32::MAX, f32::NAN, f32::INFINITY] {
            assert!(round_bf16(value).is_err());
        }
    }
}

#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
mod gdn_conv;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod bulk_f32;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-model-qwen35/src/operator_slice.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8780; SHA256 `8f862f16fe2a9d3c7b2f1af64d5a84bae2ce53c8a6bbdb32f98e48587785f29b`.

````text
//! State-free operators through the existing schedule/backend, never a token.
use crate::accelerator::program::{
    EmbeddingContract, PlanBuilder, ProgramNumericalProfile, ProjectionContract,
    RecordedOperatorSlice,
};
use crate::original_binding::{QwenMatrixRole, QwenNumericalProfile};
use crate::{Error, LoadedQwenModel, Result};

/// Pure role metadata only; never a layer/token/session execution capability.
pub(crate) trait OperatorSliceSource {
    fn geometry(&self) -> &crate::model::ModelGeometry;
    fn source_identity(&self) -> crate::accelerator::SourceIdentity;
    fn program_numerical_profile(&self) -> ProgramNumericalProfile;
    fn rms_epsilon(&self) -> f32;
    fn matrix_descriptor(
        &self,
        role: QwenMatrixRole,
    ) -> Result<crate::accelerator::program::SourceTensor>;
    fn parameter_descriptor(
        &self,
        role: crate::original_binding::QwenParameterRole,
    ) -> Result<crate::accelerator::program::SourceParameter>;
    fn embedding_contract(&self) -> EmbeddingContract;
    fn projection_contract(&self) -> ProjectionContract;
}
impl OperatorSliceSource for LoadedQwenModel {
    fn geometry(&self) -> &crate::model::ModelGeometry {
        LoadedQwenModel::geometry(self)
    }
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        LoadedQwenModel::source_identity(self)
    }
    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        LoadedQwenModel::program_numerical_profile(self)
    }
    fn rms_epsilon(&self) -> f32 {
        self.eps
    }
    fn matrix_descriptor(
        &self,
        role: QwenMatrixRole,
    ) -> Result<crate::accelerator::program::SourceTensor> {
        LoadedQwenModel::matrix_descriptor(self, role)
    }
    fn parameter_descriptor(
        &self,
        role: crate::original_binding::QwenParameterRole,
    ) -> Result<crate::accelerator::program::SourceParameter> {
        LoadedQwenModel::parameter_descriptor(self, role)
    }
    fn embedding_contract(&self) -> EmbeddingContract {
        LoadedQwenModel::embedding_contract(self)
    }
    fn projection_contract(&self) -> ProjectionContract {
        LoadedQwenModel::projection_contract(self)
    }
}

impl LoadedQwenModel {
    /// Numerical contract only; the full execution/session guard is separate.
    pub fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        match self.numerical_profile() {
            QwenNumericalProfile::GgufSourceV1 => ProgramNumericalProfile::GgufSourceV1,
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2 => {
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            }
        }
    }
    pub(crate) fn embedding_contract(&self) -> EmbeddingContract {
        match self.numerical_profile() {
            QwenNumericalProfile::GgufSourceV1 => EmbeddingContract::Source,
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2 => {
                EmbeddingContract::Bf16OutputInput
            }
        }
    }
    pub(crate) fn projection_contract(&self) -> ProjectionContract {
        match self.numerical_profile() {
            QwenNumericalProfile::GgufSourceV1 => ProjectionContract::Source,
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2 => {
                ProjectionContract::TorchCpuBf16Avx2
            }
        }
    }
    fn require_operator_slice(&self) -> Result<()> {
        if self.program_numerical_profile()
            != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported(
                "closed original embedding/projection profile".into(),
            ));
        }
        Ok(())
    }
    /// Record a pure source embedding→projection fragment, without state domains,
    /// token frontier advance, logits publication or a complete model claim.
    pub fn embedding_projection_slice(
        &self,
        role: QwenMatrixRole,
    ) -> Result<RecordedOperatorSlice> {
        self.require_operator_slice()?;
        let mut builder = PlanBuilder::new_operator_slice(
            self.source_identity(),
            self.program_numerical_profile(),
        )?;
        let (_, output) = crate::forward::embedding_projection_slice(
            self,
            &mut builder,
            PlanBuilder::inputs().token,
            role,
        )?;
        builder.finish_operator_slice(output)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_embedding_projection_slice(
        &self,
        token: u32,
        role: QwenMatrixRole,
    ) -> Result<EmbeddingProjectionOutput> {
        self.require_operator_slice()?;
        let mut math = crate::host_math::HostMath::for_operator_slice(self, None);
        let (embedding, projected) =
            crate::forward::embedding_projection_slice(self, &mut math, token as usize, role)?;
        Ok(EmbeddingProjectionOutput {
            source: self.source_identity(),
            profile: self.program_numerical_profile(),
            embedding: embedding.into_vec()?,
            projected: projected.into_vec()?,
        })
    }
    /// Record embedding→raw InputNorm(0)→GdnQkv(0), without any state effects.
    pub fn embedding_input_norm_projection_slice(&self) -> Result<RecordedOperatorSlice> {
        self.require_operator_slice()?;
        let mut builder = PlanBuilder::new_operator_slice(
            self.source_identity(),
            self.program_numerical_profile(),
        )?;
        let (_, _, output) = crate::forward::embedding_input_norm_projection_slice(
            self,
            &mut builder,
            PlanBuilder::inputs().token,
        )?;
        builder.finish_operator_slice(output)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_embedding_input_norm_projection_slice(
        &self,
        token: u32,
    ) -> Result<EmbeddingInputNormProjectionOutput> {
        self.require_operator_slice()?;
        let mut math = crate::host_math::HostMath::for_operator_slice(self, None);
        let (embedding, normalized, projected) =
            crate::forward::embedding_input_norm_projection_slice(self, &mut math, token as usize)?;
        Ok(EmbeddingInputNormProjectionOutput {
            source: self.source_identity(),
            profile: self.program_numerical_profile(),
            embedding: embedding.into_vec()?,
            normalized: normalized.into_vec()?,
            projected: projected.into_vec()?,
        })
    }
}

/// Private source-bound prefix values; no session, StateRoot or logits receipt.
#[cfg(feature = "cpu")]
pub struct EmbeddingInputNormProjectionOutput {
    source: crate::accelerator::SourceIdentity,
    profile: ProgramNumericalProfile,
    embedding: Vec<f32>,
    normalized: Vec<f32>,
    projected: Vec<f32>,
}
#[cfg(feature = "cpu")]
impl EmbeddingInputNormProjectionOutput {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn precision(&self) -> crate::accelerator::program::ActivationPrecision {
        crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
    }
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }
    pub fn normalized(&self) -> &[f32] {
        &self.normalized
    }
    pub fn projected(&self) -> &[f32] {
        &self.projected
    }
}

/// Private source-bound operator results; no SessionAuthority or root receipt.
#[cfg(feature = "cpu")]
pub struct EmbeddingProjectionOutput {
    source: crate::accelerator::SourceIdentity,
    profile: ProgramNumericalProfile,
    embedding: Vec<f32>,
    projected: Vec<f32>,
}
#[cfg(feature = "cpu")]
impl EmbeddingProjectionOutput {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn precision(&self) -> crate::accelerator::program::ActivationPrecision {
        crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
    }
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }
    pub fn projected(&self) -> &[f32] {
        &self.projected
    }
}

#[cfg(all(
    test,
    feature = "cpu",
    feature = "original-hf-binding",
    target_os = "linux"
))]
#[path = "operator_slice_tests.rs"]
mod tests;

/// Pure attention projections/QK norms; no rotary/cache/session admission.
#[path = "attention_slice.rs"]
pub mod attention;

#[path = "gdn_conv_slice.rs"]
pub mod gdn_conv;

/// Explicit external/source F32 temporal matrix, no model/state eligibility.
#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
#[path = "bulk_f32_slice.rs"]
pub mod bulk_f32;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/run_replay.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6783; SHA256 `9bd219e08683005092efb2929a544d1e90980a914eefcbd224f95eb4f82df53a`.

````text
"""ROOT ONLY: bounded owned SAME engine pure operator replay; no model forward."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import time
from contracts import HERE, digest, frozen, load, parents, read, require, write
from contracts import verify_transport
from sigterm_guard import SigtermCancellation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--build-attestation', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--new-run', type=Path, required=True)
    args = p.parse_args()
    pin = frozen()
    build = read(args.build_attestation)
    binary = digest(args.binary, 64 * 1024 * 1024)
    require(args.binary.is_absolute() and binary['sha256'] == args.binary_sha256,
            'actual bounded fresh ROOT binary')
    require(build['status'] == 'ROOT_COMPILED_SAME_ENGINE_BULK_F32_ONLY' and build['build_exit_code'] == 0
            and build['freeze_sha256'] == pin and build['binary'] == str(args.binary)
            and build['binary_digest'] == binary and len(build['effective_lock_sha256']) == 64
            and build['main_digest'] == digest(HERE / 'main.rs'), 'actual ROOT build/source binding')
    proposed = HERE / 'proposed'
    actual_files = []
    for path in sorted(proposed.rglob('*.rs')):
        actual = HERE.parents[2] / path.relative_to(proposed)
        expected = digest(path)
        require(digest(actual) == expected, 'ROOT actual merged primitive bytes')
        actual_files.append({'path':str(actual),'digest':expected})
    primitive=HERE.parent/'numerics-and-context-bulk-f32-primitive-20261020'
    dependencies=[]
    for relative in ('crates/zerogravity-cpu/src/math.rs','crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs',
                     'crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs'):
        frozen_source=primitive/('followup-attribution/proposed' if relative.endswith('haswell_bulk_f32.rs')else'proposed')/relative
        expected=digest(frozen_source);actual=HERE.parents[2]/relative
        require(digest(actual)==expected,'actual unchanged primitive + Mozilla attribution')
        dependencies.append({'path':str(actual),'digest':expected})
    require(build['production_files'] == actual_files and build['primitive_dependency_files']==dependencies,
            'actual ROOT compiled production/dependency source pins')
    require(args.fixtures.is_absolute() and args.fixtures.parent.resolve() == HERE,
            'own authentic fixture transport')
    parents()
    _,inputs = verify_transport(args.fixtures)
    fixture = digest(args.fixtures / 'manifest.json')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, 'four available CPUs')
    require(args.new_run.is_absolute() and args.new_run.parent.resolve() == HERE
            and not args.new_run.exists(), 'fresh own run')
    args.new_run.mkdir(mode=0o700)
    output = args.new_run / 'output'
    model_args=[]
    for row in inputs['models']: model_args.extend((row['gguf'],row['config']))
    argv = [str(args.binary),str(args.fixtures),str(output),*model_args,str(args.fixtures/'teacher64.u32le')]
    receipt = {'status':'UNQUALIFIED','freeze_sha256':pin,'argv':argv,'binary':binary,
               'build_attestation':build,'fixture_manifest':fixture,'actual_affinity_expected':cpus,
               'wall_seconds':300,'as_limit_bytes':4*1024**3,'math_scope':'SAME HostMath pure matrix; original source materialized; no forward/session',
               'provider_model_GPU_execution':False}
    write(args.new_run / 'prepared-invocation.json', receipt)
    h = load(HERE / 'owned_lifetime.py','bulk_f32_owned_lifetime')
    env = {'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','RAYON_NUM_THREADS':'4',
           'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':'-1',
           'NVIDIA_VISIBLE_DEVICES':'void','LD_PRELOAD':'','LD_AUDIT':''}
    def limits():
        os.sched_setaffinity(0,cpus)
        for kind,value in ((resource.RLIMIT_AS,4*1024**3),(resource.RLIMIT_CPU,1200),
                           (resource.RLIMIT_NOFILE,64),(resource.RLIMIT_CORE,0),
                           (resource.RLIMIT_FSIZE,8*1024**2),(resource.RLIMIT_STACK,8*1024**2)):
            resource.setrlimit(kind,(value,value))
    child = log = failure = None
    start = time.monotonic()
    with SigtermCancellation() as cancellation:
      try:
        cancellation.check()
        child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True,preexec_fn=limits)
        receipt['owned_cli_pid'] = child.pid
        cancellation.check()
        log = h.Log(child.stdout,args.new_run / 'execution.log')
        receipt['exit_code'] = cancellation.observe(log,child,300)
      except BaseException as error:
        failure = error
      finally:
        errors = receipt['cleanup_errors'] = []
        if child is not None:
            try:
                h.cleanup_group(child,receipt)
            except BaseException as error:
                errors.append('owned foreground:'+repr(error))
        if log is not None:
            try:
                log.finish()
            except BaseException as error:
                errors.append('log:'+repr(error))
            finally:
                receipt['log_cleanup'] = {'pipe_eof':log.eof,'all_closed':log.closed,
                    'total_bytes':log.total,'retained_bytes':log.retained}
        elif child is not None and child.stdout is not None:
            try:
                child.stdout.close()
            except BaseException as error:
                errors.append('uninitialized pipe:'+repr(error))
        receipt['elapsed_seconds_not_benchmark'] = time.monotonic()-start
        receipt['sigterm_received'] = cancellation.received
    if failure is not None:
        receipt['failure'] = repr(failure)
    good = failure is None and receipt.get('exit_code') == 0 and not receipt['cleanup_errors'] and not receipt['sigterm_received']
    if good:
        receipt['status'] = 'EXECUTED_SAME_ENGINE_PURE_BULK_REQUIRES_VALIDATION'
    write(args.new_run / 'invocation.json', receipt)
    require(good, 'failed source/control/cleanup attempt retained; no retry/fallback')
    require(digest(args.binary) == binary and digest(args.fixtures / 'manifest.json') == fixture,
            'unchanged binary/transport after known retirement')
    parents()
    verify_transport(args.fixtures)
    require(frozen() == pin, 'unchanged source after replay')

if __name__ == '__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/run_replay_v2.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7029; SHA256 `edbc7f3e43b44cfb938db9a19ac007fe50ec40bec49501e41b5f006b585fa3b9`.

````text
"""ROOT ONLY: bounded owned SAME engine pure operator replay; no model forward."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import time
from contracts import HERE, digest, frozen, load, parents, read, require, write
from contracts import verify_transport
from sigterm_guard_v2 import SigtermCancellation
from execution_v2 import verify_execution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--build-attestation', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--new-run', type=Path, required=True)
    args = p.parse_args()
    pin = frozen()
    execution_pin = verify_execution()
    build = read(args.build_attestation)
    binary = digest(args.binary, 64 * 1024 * 1024)
    require(args.binary.is_absolute() and binary['sha256'] == args.binary_sha256,
            'actual bounded fresh ROOT binary')
    require(build['status'] == 'ROOT_COMPILED_SAME_ENGINE_BULK_F32_ONLY' and build['build_exit_code'] == 0
            and build['freeze_sha256'] == pin and build['binary'] == str(args.binary)
            and build['binary_digest'] == binary and len(build['effective_lock_sha256']) == 64
            and build['main_digest'] == digest(HERE / 'main.rs'), 'actual ROOT build/source binding')
    proposed = HERE / 'proposed'
    actual_files = []
    for path in sorted(proposed.rglob('*.rs')):
        actual = HERE.parents[2] / path.relative_to(proposed)
        expected = digest(path)
        require(digest(actual) == expected, 'ROOT actual merged primitive bytes')
        actual_files.append({'path':str(actual),'digest':expected})
    primitive=HERE.parent/'numerics-and-context-bulk-f32-primitive-20261020'
    dependencies=[]
    for relative in ('crates/zerogravity-cpu/src/math.rs','crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs',
                     'crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs'):
        frozen_source=primitive/('followup-attribution/proposed' if relative.endswith('haswell_bulk_f32.rs')else'proposed')/relative
        expected=digest(frozen_source);actual=HERE.parents[2]/relative
        require(digest(actual)==expected,'actual unchanged primitive + Mozilla attribution')
        dependencies.append({'path':str(actual),'digest':expected})
    require(build['production_files'] == actual_files and build['primitive_dependency_files']==dependencies,
            'actual ROOT compiled production/dependency source pins')
    require(args.fixtures.is_absolute() and args.fixtures.parent.resolve() == HERE,
            'own authentic fixture transport')
    parents()
    _,inputs = verify_transport(args.fixtures)
    fixture = digest(args.fixtures / 'manifest.json')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, 'four available CPUs')
    require(args.new_run.is_absolute() and args.new_run.parent.resolve() == HERE
            and not args.new_run.exists(), 'fresh own run')
    args.new_run.mkdir(mode=0o700)
    output = args.new_run / 'output'
    model_args=[]
    for row in inputs['models']: model_args.extend((row['gguf'],row['config']))
    argv = [str(args.binary),str(args.fixtures),str(output),*model_args,str(args.fixtures/'teacher64.u32le')]
    receipt = {'status':'UNQUALIFIED','freeze_sha256':pin,'argv':argv,'binary':binary,
               'build_attestation':build,'fixture_manifest':fixture,'actual_affinity_expected':cpus,
               'wall_seconds':300,'as_limit_bytes':4*1024**3,'math_scope':'SAME HostMath pure matrix; original source materialized; no forward/session',
               'provider_model_GPU_execution':False,'execution_freeze_sha256':execution_pin,
               'record_only_cancellation_signals':['SIGTERM','SIGINT']}
    write(args.new_run / 'prepared-invocation.json', receipt)
    h = load(HERE / 'owned_lifetime.py','bulk_f32_owned_lifetime')
    env = {'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','RAYON_NUM_THREADS':'4',
           'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':'-1',
           'NVIDIA_VISIBLE_DEVICES':'void','LD_PRELOAD':'','LD_AUDIT':''}
    def limits():
        os.sched_setaffinity(0,cpus)
        for kind,value in ((resource.RLIMIT_AS,4*1024**3),(resource.RLIMIT_CPU,1200),
                           (resource.RLIMIT_NOFILE,64),(resource.RLIMIT_CORE,0),
                           (resource.RLIMIT_FSIZE,8*1024**2),(resource.RLIMIT_STACK,8*1024**2)):
            resource.setrlimit(kind,(value,value))
    child = log = failure = None
    start = time.monotonic()
    with SigtermCancellation() as cancellation:
      try:
        cancellation.check()
        child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True,preexec_fn=limits)
        receipt['owned_cli_pid'] = child.pid
        cancellation.check()
        log = h.Log(child.stdout,args.new_run / 'execution.log')
        receipt['exit_code'] = cancellation.observe(log,child,300)
      except BaseException as error:
        failure = error
      finally:
        errors = receipt['cleanup_errors'] = []
        if child is not None:
            try:
                h.cleanup_group(child,receipt)
            except BaseException as error:
                errors.append('owned foreground:'+repr(error))
        if log is not None:
            try:
                log.finish()
            except BaseException as error:
                errors.append('log:'+repr(error))
            finally:
                receipt['log_cleanup'] = {'pipe_eof':log.eof,'all_closed':log.closed,
                    'total_bytes':log.total,'retained_bytes':log.retained}
        elif child is not None and child.stdout is not None:
            try:
                child.stdout.close()
            except BaseException as error:
                errors.append('uninitialized pipe:'+repr(error))
        receipt['elapsed_seconds_not_benchmark'] = time.monotonic()-start
        receipt['sigterm_received'] = cancellation.received
    if failure is not None:
        receipt['failure'] = repr(failure)
    good = failure is None and receipt.get('exit_code') == 0 and not receipt['cleanup_errors'] and not receipt['sigterm_received']
    if good:
        receipt['status'] = 'EXECUTED_SAME_ENGINE_PURE_BULK_REQUIRES_VALIDATION'
    write(args.new_run / 'invocation.json', receipt)
    require(good, 'failed source/control/cleanup attempt retained; no retry/fallback')
    require(digest(args.binary) == binary and digest(args.fixtures / 'manifest.json') == fixture,
            'unchanged binary/transport after known retirement')
    parents()
    verify_transport(args.fixtures)
    require(frozen() == pin and verify_execution() == execution_pin, 'unchanged source/execution after replay')

if __name__ == '__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/sigterm_guard.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1303; SHA256 `c60e3cf48390ae7192be4415b7feead46a0f6b3eea1b4fd732908b0df5ffbd81`.

````text
"""SIGTERM checkpoints around the original bounded log drain, not new ownership."""
import signal


class SigtermCancellation:
    def __init__(self):
        self.received = False
        self.previous = None

    def _record(self, signum, frame):
        # Never asynchronously raise across Popen assignment or retirement.
        self.received = True

    def __enter__(self):
        self.previous = signal.signal(signal.SIGTERM, self._record)
        return self

    def __exit__(self, kind, value, traceback):
        signal.signal(signal.SIGTERM, self.previous)
        return False

    def exception(self):
        return InterruptedError("SIGTERM: retire original owned startup before failure")

    def check(self):
        if self.received:
            raise self.exception()

    def observe(self, log, process, seconds):
        original_drain = log.drain

        def checked_drain(timeout):
            self.check()
            original_drain(timeout)
            self.check()

        self.check()
        log.drain = checked_drain
        try:
            result = log.observe(process, seconds)
            self.check()
            return result
        finally:
            # Original cleanup/finish drain proceeds even after repeated SIGTERM.
            log.drain = original_drain

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/sigterm_guard_v2.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1428; SHA256 `00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128`.

````text
"""SIGTERM/SIGINT checkpoints around the original bounded log drain, not new ownership."""
import signal


class SigtermCancellation:
    def __init__(self):
        self.received = False
        self.previous = {}

    def _record(self, signum, frame):
        # Never asynchronously raise across Popen assignment or retirement.
        self.received = True

    def __enter__(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.previous[signum] = signal.signal(signum, self._record)
        return self

    def __exit__(self, kind, value, traceback):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
        return False

    def exception(self):
        return InterruptedError("interrupted: retire original owned serving regression before failure")

    def check(self):
        if self.received:
            raise self.exception()

    def observe(self, log, process, seconds):
        original_drain = log.drain

        def checked_drain(timeout):
            self.check()
            original_drain(timeout)
            self.check()

        self.check()
        log.drain = checked_drain
        try:
            result = log.observe(process, seconds)
            self.check()
            return result
        finally:
            # Original cleanup/finish drain proceeds even after repeated SIGTERM.
            log.drain = original_drain

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/test_contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3833; SHA256 `6dde5061dda1a1577fa06100abf6ae0337bca9637dbfed91db122fd7244520e4`.

````text
"""Prepared synthetic validator fixtures; no raw models/provider or arithmetic."""
import copy
import unittest
from validate import report, META, CACHE, CONTEXT, TRANSPORT, INPUTS, WORDS
from contracts import CASES, MODELS

def ledger(metadata=0,staging=0,output=0,weights=0,reservations=0):
    scratch=256 if output else 0
    return dict(reserved=metadata+staging+output+weights+scratch,reservations=reservations,fenced=False,
                metadata=metadata,staging=staging,output=output,weights=weights,scratch=scratch)

def synthetic():
    rows=[]
    for case in CASES:
        weights=563036064 if case=='av'else 1221139712;size=WORDS[case]*4
        rows.append(dict(case=case,model_sha256=MODELS[case],words=WORDS[case],unequal_words=0,source_bytes=weights,
            source_bound=True,expected_opened_after_math=True,output_clone_retains_source=True,
            pre_cancel_constructor=True,pre_cancel_execution=True,real_one_byte_pressure_refused=True,provider_callbacks=0,
            plan_nodes=3,frontier_delta=0,state_domains=0,left_shape=[256,256,2,1]if case=='av'else[2048,16,1,1],
            left_strides=[1,8448,2162688,4325376]if case=='av'else[1,2048,32768,32768],
            right_shape=[256,64,8,1]if case=='av'else[2048,64,1,1],
            live=ledger(META+CACHE+CONTEXT,TRANSPORT+INPUTS[case],size,weights,6),
            retained_output=ledger(META+CACHE+CONTEXT,0,size,weights,4),after_case=ledger(metadata=META,reservations=1)))
    return dict(status='PASS_SAME_ENGINE_PURE_BULK_F32_ONLY',words=132096,default_forward_changed=False,
        fullmodel_or_StateRoot_qualified=False,historical_prefix_recovered=False,actual_mid_operation_cancel_qualified=False,
        physical_RAM_admission_qualified=False,actual_affinity=[0,1,2,3],mxcsr_before=0x1f80,mxcsr_after=0x1f80,
        before=ledger(),final=ledger(),cases=rows)

class Contracts(unittest.TestCase):
    def test_literal_positive_is_only_parser_fixture(self):report(synthetic())
    def test_cannot_drop_actual_output_words(self):
        value=synthetic();value['cases'][0]['words']-=1
        with self.assertRaises(ValueError):report(value)
    def test_cannot_publish_or_relax_exactness(self):
        for key,replacement in(('unequal_words',1),('frontier_delta',64),('state_domains',42),('plan_nodes',1)):
            value=synthetic();value['cases'][0][key]=replacement
            with self.assertRaises(ValueError):report(value)
    def test_actual_temporal_n_not_cohort_b(self):
        value=synthetic();value['cases'][0]['right_shape'][1]=8
        with self.assertRaises(ValueError):report(value)
    def test_strided_v_never_dense_retagged(self):
        value=synthetic();value['cases'][0]['left_strides']=[1,256,65536,131072]
        with self.assertRaises(ValueError):report(value)
    def test_live_payload_cannot_escape_original_tickets(self):
        value=synthetic();value['cases'][0]['retained_output']['output']=0
        with self.assertRaises(ValueError):report(value)
    def test_metadata_only_claim_cannot_be_promoted(self):
        for key in('default_forward_changed','fullmodel_or_StateRoot_qualified','historical_prefix_recovered','physical_RAM_admission_qualified'):
            value=synthetic();value[key]=True
            with self.assertRaises(ValueError):report(value)
    def test_bool_cannot_substitute_integer_proof(self):
        value=synthetic();value['cases'][0]['provider_callbacks']=False
        with self.assertRaises(ValueError):report(value)
    def test_source_cases_cannot_omit_duplicate_or_swap(self):
        for rows in([synthetic()['cases'][0]],[synthetic()['cases'][0]]*2,list(reversed(synthetic()['cases']))):
            value=synthetic();value['cases']=copy.deepcopy(rows)
            with self.assertRaises(ValueError):report(value)
if __name__=='__main__':unittest.main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/test_execution_v2.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 2091; SHA256 `16d362017cbbf4351975200bfc6f39f76337a8e090927ff72c627c8c131a7786`.

````text
"""Prepared harmless mocked guards, no real signals/children/provider/container."""
import signal
import unittest
from unittest.mock import patch
from sigterm_guard_v2 import SigtermCancellation

class SignalFixtures(unittest.TestCase):
    def test_both_signal_handlers_only_record_until_checkpoint(self):
        prior={signal.SIGTERM:object(),signal.SIGINT:object()};calls=[]
        def exchange(number,handler):calls.append((number,handler));return prior[number]
        with patch('signal.signal',side_effect=exchange):
            with SigtermCancellation()as guard:
                self.assertEqual([n for n,_ in calls],[signal.SIGTERM,signal.SIGINT])
                guard._record(signal.SIGINT,None)
                self.assertTrue(guard.received)
                with self.assertRaises(InterruptedError):guard.check()
            self.assertEqual(calls[-2:],[(signal.SIGTERM,prior[signal.SIGTERM]),(signal.SIGINT,prior[signal.SIGINT])])
    def test_cancellation_does_not_interrupt_child_assignment_or_finally(self):
        cleaned=[];owned=None
        with patch('signal.signal',return_value=signal.SIG_DFL):
            with SigtermCancellation()as guard:
                try:
                    guard._record(signal.SIGTERM,None)
                    owned=object() # Mock the successful Popen return assignment.
                    guard.check()
                except InterruptedError:pass
                finally:cleaned.append(owned)
        self.assertEqual(cleaned,[owned]);self.assertIsNotNone(owned)
    def test_original_drain_is_restored_for_cleanup_after_cancel(self):
        class Log:
            def drain(self,timeout):pass
            def observe(self,process,seconds):self.drain(0);return 0
        log=Log();old=log.drain
        with patch('signal.signal',return_value=signal.SIG_DFL):
            with SigtermCancellation()as guard:
                guard._record(signal.SIGINT,None)
                with self.assertRaises(InterruptedError):guard.observe(log,None,1)
                self.assertEqual(log.drain,old)
if __name__=='__main__':unittest.main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/validate.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7060; SHA256 `cff5bf92b15b2e50eb189b0d2dcdec6520d430476907e0fc3e95f70a3bfb6cb4`.

````text
"""ROOT ONLY independent whole-output/three-node/source/ledger validator."""
import argparse
from pathlib import Path
from contracts import HERE, CASES, MODELS, digest, frozen, parents, read, require, verify_transport, write

META=4*1024**2
CACHE=16*1024**2
CONTEXT=16384
TRANSPORT=32*1024**2
WORDS={'av':131072,'alpha':1024}
INPUTS={'av':(4317184+131072)*4,'alpha':131072*4}

def ledger(row,*,metadata=0,staging=0,output=0,weights=0,reservations=0):
    scratch=256 if output else 0
    require(set(row)=={'reserved','reservations','fenced','output','scratch','staging','metadata','weights'},'exact live ledger fields')
    require(all(type(row[k])is int for k in row if k!='fenced')and row['fenced']is False,'actual ledger scalar types')
    require(row=={'reserved':metadata+staging+output+scratch+weights,'reservations':reservations,'fenced':False,
                  'metadata':metadata,'staging':staging,'output':output,'scratch':scratch,'weights':weights},
            'actual class/retained owner/charge conservation')

def report(value):
    require(value['status']=='PASS_SAME_ENGINE_PURE_BULK_F32_ONLY'and type(value['words'])is int and value['words']==132096,
            'complete pure operator scope')
    for key in('default_forward_changed','fullmodel_or_StateRoot_qualified','historical_prefix_recovered',
               'actual_mid_operation_cancel_qualified','physical_RAM_admission_qualified'):
        require(value[key]is False,'no broader capability/schedule/history/admission claim')
    cpus=value['actual_affinity'];require(len(cpus)==4 and len(set(cpus))==4 and all(type(n)is int and 0<=n<1024 for n in cpus),'actual four CPU set')
    for key in('mxcsr_before','mxcsr_after'):
        require(type(value[key])is int and value[key]&0xffc0==0x1f80,'unchanged RNE/noFTZ/noDAZ controls')
    ledger(value['before']);ledger(value['final'])
    require([r['case']for r in value['cases']]==list(CASES),'complete distinct authentic population')
    for row in value['cases']:
        case=row['case'];size=WORDS[case]*4
        require(row['model_sha256']==MODELS[case]and type(row['words'])is int and row['words']==WORDS[case]
                and type(row['unequal_words'])is int and row['unequal_words']==0,'all unchanged default raw output words')
        require(type(row['source_bytes'])is int and 0<row['source_bytes']<2*1024**3,'actual bounded source envelope')
        require(row['source_bound']is True and row['expected_opened_after_math']is True and row['output_clone_retains_source']is True,
                'actual source/result lifetime and expected isolation')
        for key in('pre_cancel_constructor','pre_cancel_execution','real_one_byte_pressure_refused'):
            require(row[key]is True,'actual cancellation/admission refusal')
        require(type(row['provider_callbacks'])is int and row['provider_callbacks']==0,'unaware provider not called')
        require(type(row['plan_nodes'])is int and row['plan_nodes']==3 and type(row['frontier_delta'])is int and row['frontier_delta']==0
                and type(row['state_domains'])is int and row['state_domains']==0,'same three-node no-state plan')
        require(all(type(n)is int for key in('left_shape','left_strides','right_shape')for n in row[key]),'exact integer geometry')
        require(row['left_shape']==([256,256,2,1]if case=='av'else[2048,16,1,1])
                and row['left_strides']==([1,8448,2162688,4325376]if case=='av'else[1,2048,32768,32768])
                and row['right_shape']==([256,64,8,1]if case=='av'else[2048,64,1,1]),'actual fullK/original stride/N shapes')
        ledger(row['live'],metadata=META+CACHE+CONTEXT,staging=TRANSPORT+INPUTS[case],output=size,weights=row['source_bytes'],reservations=6)
        ledger(row['retained_output'],metadata=META+CACHE+CONTEXT,output=size,weights=row['source_bytes'],reservations=4)
        ledger(row['after_case'],metadata=META,reservations=1)

def invocation(value):
    require(value['status']=='EXECUTED_SAME_ENGINE_PURE_BULK_REQUIRES_VALIDATION'and value['exit_code']==0
            and value['cleanup_errors']==[]and value['sigterm_received']is False
            and value['provider_model_GPU_execution']is False and value['wall_seconds']==300 and value['as_limit_bytes']==4*1024**3
            and value['math_scope']=='SAME HostMath pure matrix; original source materialized; no forward/session','actual bounded invocation')
    g=value['foreground_cleanup'];log=value['log_cleanup']
    require(g['pgid']==value['owned_cli_pid']and g['leader_reaped']is True and g['group_absent']is True and g['stage_errors']==[]
            and g['all_nonzero_signals_before_reap']is True and g['kill_attempts_before_reap']==2 and g['leader_reap_attempts']in(1,2),
            'original unreaped owner/group retirement')
    require(log['pipe_eof']is True and log['all_closed']is True and log['total_bytes']==log['retained_bytes']<=4*1024**2,
            'bounded complete owned pipe drain/close')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True)
    a=p.parse_args();pin=frozen();parents();manifest,inputs=verify_transport(a.fixtures)
    require(a.run.is_absolute()and a.run.parent.resolve()==HERE and a.fixtures.is_absolute()and a.fixtures.parent.resolve()==HERE,'own ROOT artifacts')
    called=read(a.run/'invocation.json');invocation(called)
    expected_argv=[str(a.fixtures),str(a.run/'output')]
    for row in inputs['models']:expected_argv.extend((row['gguf'],row['config']))
    expected_argv.append(str(a.fixtures/'teacher64.u32le'))
    require(called['freeze_sha256']==pin and called['fixture_manifest']==digest(a.fixtures/'manifest.json')
            and called['binary']==digest(called['argv'][0],64*1024**2)and called['argv'][1:]==expected_argv,'actual full invocation/source binding')
    value=read(a.run/'output/report.json');report(value)
    require(all(row['source_bytes']==model['gguf_digest']['bytes']for row,model in zip(value['cases'],inputs['models'],strict=True)),
            'actual materialized source envelope equals authenticated file length')
    require(value['actual_affinity']==called['actual_affinity_expected'],'same actual CPU set')
    for row in manifest['cases']:
        actual=a.run/'output'/f"{row['case']}.actual.f32le";expected=a.fixtures/row['expected']['file']
        require(digest(actual,1024**2)==row['expected']['digest']==digest(expected,1024**2)
                and actual.read_bytes()==expected.read_bytes(),'complete independent DEFAULT raw bytes; no tolerance')
    require(frozen()==pin,'unchanged reviewed source')
    write(a.run/'validation.json',{'status':'PASS_SAME_ENGINE_PURE_BULK_F32_ONLY','freeze_sha256':pin,'words':132096,
          'cases':list(CASES),'invocation':digest(a.run/'invocation.json'),'report':digest(a.run/'output/report.json'),
          'fullmodel_default_or_state_qualification':False,'historical_scope_qualified':False,'outer_RAM_admission_qualified':False})
if __name__=='__main__':main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/validate_v2.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1203; SHA256 `d54e35c1be96fd47f1a75707c256690ff024e4c16399f0c3831c4254828c3bfb`.

````text
"""ROOT ONLY V1 complete math validation +additive two-signal execution binding."""
import argparse
from pathlib import Path
import validate as original
from contracts import digest,read,require,write
from execution_v2 import verify_execution

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True)
    a=p.parse_args();pin=verify_execution();called=read(a.run/'invocation.json')
    require(called['execution_freeze_sha256']==pin and called['record_only_cancellation_signals']==['SIGTERM','SIGINT'],
            'actual fresh runner receipt binds both record-only signals')
    original.main() # Unchanged complete source/bytes/ledger/lifecycle validator.
    require(verify_execution()==pin,'same execution source after independent validation')
    write(a.run/'validation-execution-v2.json',{'status':'PASS_BOUND_EXECUTION_ONLY_V2','execution_freeze_sha256':pin,
          'original_math_validation':digest(a.run/'validation.json'),'invocation':digest(a.run/'invocation.json'),
          'actual_signal_injection_qualified':False,'math_source_changed':False})
if __name__=='__main__':main()

````

## Source: .agents/runs/numerics-and-context-default-bulk-isolated-20261020/baseline/Cargo.toml

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 506; SHA256 `1764ad9286e46dcd24fb2c92d303a1e093672fb06a916f2c2ab0e02517e52e80`.

````text
[package]
name = "zg-same-engine-bulk-f32-replay"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[[bin]]
name = "bulk-f32-typed-replay"
path = "main.rs"

[dependencies]
serde = { version = "=1.0.229", features = ["derive"] }
serde_json = "=1.0.151"
zerogravity-core = { path = "../../../../crates/zerogravity-core" }
zerogravity-model-qwen35 = { path = "../../../../crates/zerogravity-model-qwen35", features = ["cpu"] }
libc = "=0.2.190"

[profile.release]
lto = false
codegen-units = 1

````

