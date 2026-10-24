# Paused continuation text — volume 14

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/baseline/crates/zerogravity-gpu/src/resident_program.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 106452; SHA256 `32ae5012e22a62570127b93f336a00c4be740e9bdc29c751dc8c5e9c8fa501a6`.

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
                    PlanOperation::TypedEmbed { .. }
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
                PlanOperation::TypedEmbed { .. }
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

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/baseline/crates/zerogravity-model-qwen35/src/forward.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 23682; SHA256 `a96a993f73df31ff56ec20b3704480f48c4dc03c8e62d016fa83956d88aa5820`.

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

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/baseline/crates/zerogravity-model-qwen35/src/host_math.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 39194; SHA256 `8937d8d8efb7269ef15e5a9bb8338dacc60ae8483c29050c8f2f018818d36355`.

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
        self.data
            .get(self.view.offset..self.view.offset + self.elements)
            .ok_or_else(|| Error::State("invalid private host tensor view".into()))
    }
    pub fn into_vec(self) -> Result<Vec<f32>> {
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

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/baseline/crates/zerogravity-model-qwen35/src/operator_slice.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8579; SHA256 `3f0d399bf936c6db3ee02315558fd626e09e956ad5d97d28b01a827fdb0d4686`.

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

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4989; SHA256 `5be8a53cf66f04d8b7319b1e75d945befb9e5d79d5682f06fffbbcf2e1593fb2`.

````text
"""ROOT-only source/input binding. Import performs no hashing/provider/model calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PRIMITIVE=HERE.parent/'numerics-and-context-bulk-f32-primitive-20261020'
CASES=('av','alpha')
MODELS={'av':'57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf',
        'alpha':'91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6'}

def require(ok,message):
    if not ok: raise ValueError(message)

def digest(path,cap=None):
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode)and(cap is None or before.st_size<=cap),'bounded regular file')
        h=hashlib.sha256();count=0
        while raw:=f.read(1024*1024):h.update(raw);count+=len(raw)
        after=os.fstat(f.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)and count==before.st_size,
                'same immutable FD')
        return {'bytes':count,'sha256':h.hexdigest()}

def read(path,cap=1024*1024):
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:
        require(os.fstat(f.fileno()).st_size<=cap,'bounded JSON')
        raw=f.read(cap+1)
    require(len(raw)<=cap,'actual JSON cap')
    def unique(rows):
        out={}
        for key,value in rows:require(key not in out,'duplicate key');out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=unique)

def write(path,value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    require(len(raw)<=1024*1024,'bounded report')
    with Path(path).open('xb')as f:os.chmod(path,0o600);f.write(raw)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m

def original():
    """Bind unchanged original prepare.py to ITS unchanged contracts, not ours."""
    old=load(PRIMITIVE/'contracts.py','typed_bulk_immutable_primitive_contracts')
    prior=sys.modules.get('contracts')
    sys.modules['contracts']=old
    try:prep=load(PRIMITIVE/'prepare.py','typed_bulk_immutable_primitive_prepare')
    finally:
        if prior is None:sys.modules.pop('contracts',None)
        else:sys.modules['contracts']=prior
    return old,prep

def frozen():
    pin=read(HERE/'freeze-manifest.json')
    for row in pin['files']:
        require(digest(HERE/row['path'],2*1024*1024)==row['digest'],'reviewed proposal/helper changed')
    for row in pin['source_pins']:
        require(digest(row['path'],2*1024*1024)==row['digest'],'original source/metadata changed')
    return digest(HERE/'freeze-manifest.json')['sha256']

def parents():
    """ROOT ONLY: original full authentic source/readsets include model/DSO hashes."""
    old,prep=original();old.frozen();av,alpha=old.parents()
    return old,prep,av,alpha

def verify_transport(folder):
    old,prep=original()
    manifest=prep.verify_transport(folder)
    typed=read(folder/'typed-inputs.json')
    require(typed['status']=='AUTHENTIC_SAME_ENGINE_BULK_F32_INPUTS_ONLY'
            and typed['arithmetic_executed']is False and typed['sourceprofile']=='GGUF_ORIGINAL'
            and typed['base']==0 and typed['capacity']==8448 and typed['teachers']==64,'closed pure64teacher scope')
    require(all(type(typed[k])is int for k in('base','capacity','teachers')),'exact teacher geometry types')
    require(digest(folder/'teacher64.u32le',256)==typed['teacher_digest'],'actual64 teacher bits')
    for row,case in zip(typed['models'],CASES,strict=True):
        require(row['case']==case and row['gguf']==str(REPO/'target/models'/('qwen35-0.8b'if case=='av'else'qwen35-2b')/'model.gguf')
                and row['gguf_digest']['sha256']==MODELS[case],'original model source namespace')
        # ROOT ONLY full source verification; no operator/model computation.
        require(digest(row['gguf'])==row['gguf_digest']and digest(row['config'],1024*1024)==row['config_digest'],
                'actual immutable whole GGUF/config')
    av_ids=old.AV.parent/'numerics-and-context-bulk-causality-20261020/root-first64-ar1-budget/input_ids.u32le'
    alpha=load(old.ALPHA/'prepare.py','typed_bulk_original_alpha_metadata')
    alpha_ids=alpha.CAPTURE/'input_ids.u32le'
    require(digest(av_ids,32768)==typed['teacher_sources'][0]['digest']and
            digest(alpha_ids,32768)==typed['teacher_sources'][1]['digest'],'original teacher source bindings')
    a=av_ids.read_bytes();b=alpha_ids.read_bytes()
    require(len(a)==len(b)==32768 and a==b and a[:256]==(folder/'teacher64.u32le').read_bytes(),
            'same actual full8192 teachers; first64 belongs to BOTH original captures')
    return manifest,typed

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/main.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 19992; SHA256 `b424884ef3f5e442bf9706ec9c60d0816b7fdc180f2bd38f56c66a74ea33bc55`.

````text
//! ROOT-only SAME engine pure matrix replay; no forward/session/token loop.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_core::accelerator::program::{BulkF32Layout, PlanOperation};
use zerogravity_core::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_model_qwen35::operator_slice::bulk_f32::{
    BulkF32HostAdmission, BulkF32Slice, ExternalBulkF32,
};
use zerogravity_model_qwen35::{LoadedQwenModel, original_binding::QwenMatrixRole};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const META: u64 = 4 * 1024 * 1024;
const STAGING: u64 = 32 * 1024 * 1024;
const LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const CACHE_METADATA: u64 = 16 * 1024 * 1024;

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Digest {
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Part {
    file: String,
    name: String,
    shape: [usize; 4],
    strides: [usize; 4],
    digest: Digest,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    case: String,
    model_sha256: String,
    left: Part,
    right: Part,
    expected: Part,
    source_provenance: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    status: String,
    profile: String,
    cases: Vec<Case>,
    arithmetic_executed: bool,
    expected_from_candidate: bool,
    gap_storage: String,
}
fn snapshot(arbiter: &ResourceArbiter, key: &PhysicalDeviceKey) -> Result<Value> {
    let s = arbiter.snapshot(key)?;
    Ok(
        json!({"reserved":s.reserved,"reservations":s.reservations,"fenced":s.fenced,
        "output":s.per_class.get(&AllocationClass::Output).copied().unwrap_or(0),
        "scratch":s.per_class.get(&AllocationClass::Scratch).copied().unwrap_or(0),
        "staging":s.per_class.get(&AllocationClass::TransferStaging).copied().unwrap_or(0),
        "metadata":s.per_class.get(&AllocationClass::Other).copied().unwrap_or(0),
        "weights":s.per_class.get(&AllocationClass::ImmutableWeights).copied().unwrap_or(0)}),
    )
}
fn mxcsr() -> u32 {
    let mut value = 0u32;
    // SAFETY: read-only CPU-control instruction writes one valid local word.
    unsafe {
        std::arch::asm!("stmxcsr [{pointer}]", pointer = in(reg) &mut value,
        options(nostack, preserves_flags));
    }
    value
}
fn affinity() -> Result<Vec<usize>> {
    let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    need(
        unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&mask), &mut mask) } == 0,
        "actual CPU mask",
    )?;
    let actual = (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .collect::<Vec<_>>();
    need(actual.len() == 4, "ROOT exact four available CPUs")?;
    Ok(actual)
}
fn read_words(folder: &Path, part: &Part) -> Result<Vec<f32>> {
    need(
        Path::new(&part.file).components().count() == 1 && !part.file.starts_with('.'),
        "closed file component",
    )?;
    need(
        part.digest.bytes > 0
            && part.digest.bytes <= 1024 * 1024
            && part.digest.bytes.is_multiple_of(4)
            && part.digest.sha256.len() == 64,
        "bounded complete raw F32 file",
    )?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(folder.join(&part.file))?;
    let before = file.metadata()?;
    need(
        before.is_file() && before.len() == part.digest.bytes as u64,
        "actual exact raw length",
    )?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(part.digest.bytes)?;
    (&mut file)
        .take(part.digest.bytes as u64 + 1)
        .read_to_end(&mut raw)?;
    use std::os::unix::fs::MetadataExt;
    let after = file.metadata()?;
    need(
        raw.len() == part.digest.bytes
            && before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.len() == after.len()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.mtime() == after.mtime()
            && before.ctime_nsec() == after.ctime_nsec()
            && before.ctime() == after.ctime(),
        "same immutable input FD",
    )?;
    let mut words = Vec::new();
    words.try_reserve_exact(raw.len() / 4)?;
    for value in raw.chunks_exact(4) {
        let v = f32::from_le_bytes(value.try_into()?);
        need(
            v.is_finite(),
            "actual finite F32 words; no canonicalization",
        )?;
        words.push(v);
    }
    Ok(words)
}
fn expand(words: &[f32], part: &Part) -> Result<Vec<f32>> {
    let logical = part
        .shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v).ok_or("logical overflow"))?;
    need(words.len() == logical, "complete captured logical source")?;
    let span = part
        .shape
        .iter()
        .zip(part.strides)
        .try_fold(1usize, |n, (&v, stride)| {
            v.checked_sub(1)
                .and_then(|v| v.checked_mul(stride))
                .and_then(|v| n.checked_add(v))
                .ok_or("source span overflow")
        })?;
    need(
        span <= 8 * 1024 * 1024 && part.strides[0] == 1,
        "bounded original physical span",
    )?;
    let mut result = Vec::new();
    result.try_reserve_exact(span)?;
    result.resize(span, 0.0);
    let mut at = 0;
    for sequence in 0..part.shape[3] {
        for head in 0..part.shape[2] {
            for row in 0..part.shape[1] {
                let offset =
                    row * part.strides[1] + head * part.strides[2] + sequence * part.strides[3];
                result[offset..offset + part.shape[0]]
                    .copy_from_slice(&words[at..at + part.shape[0]]);
                at += part.shape[0];
            }
        }
    }
    Ok(result)
}

struct DeniedProvider(AtomicUsize);
impl MatVecKernel for DeniedProvider {
    fn try_execute(&self, _region: MatVecRegion<'_>) -> zerogravity_core::Result<RegionExecution> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(RegionExecution::Unsupported)
    }
}
fn run(
    fixtures: &Path,
    output: &Path,
    models: &[(PathBuf, PathBuf); 2],
    teacher: &Path,
) -> Result<()> {
    let cpus = affinity()?;
    let fp_before = mxcsr();
    need(fp_before & 0xffc0 == 0x1f80, "actual RNE/noFTZ/noDAZ")?;
    need(!output.exists(), "fresh output")?;
    std::fs::create_dir(output)?;
    let raw = std::fs::read(teacher)?;
    need(raw.len() == 256, "actual64 original teacher IDs")?;
    let ids = raw
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(fixtures.join("manifest.json"))?;
    need(
        file.metadata()?.len() <= 512 * 1024,
        "bounded two-case metadata",
    )?;
    let manifest: Manifest = serde_json::from_reader(file.take(512 * 1024 + 1))?;
    need(
        manifest.status == "AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY"
            && manifest.profile == "GgmlD812350HaswellTinyBlas"
            && !manifest.arithmetic_executed
            && !manifest.expected_from_candidate
            && manifest.cases.len() == 2
            && manifest.gap_storage
                == "actual dense logical V restored at original strides; uncaptured gaps zero-filled",
        "authentic unchanged default reference scope",
    )?;
    let arbiter = ResourceArbiter::process_shared();
    let key = PhysicalDeviceKey::new("host", "same-engine-default-bulk-F32")?;
    arbiter.register_matching(key.clone(), LIMIT)?;
    let owner = ResourceOwner::new("original-GGUF-pure-matrix", "ROOT-qualification")?;
    let before = snapshot(&arbiter, &key)?;
    let meta = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: META,
        }],
    )?;
    let mut reports = Vec::new();
    for (index, case) in manifest.cases.iter().enumerate() {
        let id = if index == 0 { "av" } else { "alpha" };
        need(case.case == id, "two distinct ordered cases")?;
        need(
            case.model_sha256
                == if index == 0 {
                    "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf"
                } else {
                    "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6"
                },
            "original source SHA metadata",
        )?;
        need(
            case.source_provenance.is_object(),
            "original producer/reference provenance",
        )?;
        let source_bytes = std::fs::metadata(&models[index].0)?.len();
        // Diagnostic envelope before materialization; actual cached-vector/metadata
        // capacity and outer process physical RAM remain outside this claim.
        let source_ticket = arbiter.admit(
            &key,
            owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::ImmutableWeights,
                    bytes: source_bytes,
                },
                AllocationRequest {
                    kind: AllocationClass::Other,
                    bytes: CACHE_METADATA,
                },
            ],
        )?;
        let model = LoadedQwenModel::open(&models[index].0, &models[index].1)?;
        let weak = Arc::downgrade(&model);
        let source = model.source_identity();
        let staging = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: STAGING,
            }],
        )?;
        let dense = read_words(fixtures, &case.left)?;
        let left = expand(&dense, &case.left)?;
        drop(dense);
        let right = read_words(fixtures, &case.right)?;
        let ll = BulkF32Layout::new(case.left.shape, case.left.strides, 0, left.len())?;
        let rl = BulkF32Layout::dense(case.right.shape)?;
        need(
            rl.strides() == case.right.strides && rl.backing_words() == right.len(),
            "actual full right contiguous geometry",
        )?;
        if index == 1 {
            let original = model.tensor_bytes("blk.0.ssm_alpha.weight")?;
            need(
                original.len() == left.len() * 4
                    && original
                        .chunks_exact(4)
                        .zip(&left)
                        .all(|(raw, value)| raw == value.to_le_bytes()),
                "all32768 original source alpha words",
            )?;
        }
        let admission = BulkF32HostAdmission {
            arbiter: arbiter.clone(),
            resource: key.clone(),
            owner: owner.clone(),
        };
        let make = |flag: Option<&AtomicBool>| -> zerogravity_core::Result<BulkF32Slice> {
            let rhs = ExternalBulkF32 {
                name: &case.right.name,
                layout: rl,
                backing: &right,
            };
            if index == 0 {
                BulkF32Slice::observed_pair(
                    model.clone(),
                    0,
                    8448,
                    &ids,
                    ExternalBulkF32 {
                        name: &case.left.name,
                        layout: ll,
                        backing: &left,
                    },
                    rhs,
                    admission.clone(),
                    flag,
                )
            } else {
                BulkF32Slice::source_projection(
                    model.clone(),
                    0,
                    8448,
                    &ids,
                    QwenMatrixRole::GdnAlpha(0),
                    rhs,
                    admission.clone(),
                    flag,
                )
            }
        };
        let cancelled = AtomicBool::new(true);
        let prior = snapshot(&arbiter, &key)?;
        need(
            make(Some(&cancelled)).is_err() && snapshot(&arbiter, &key)? == prior,
            "pre-cancel before any private context lease",
        )?;
        let slice = make(None)?;
        need(
            slice.matrix().source() == source && slice.matrix().left().layout() == ll,
            "same immutable source/full-stride descriptor",
        )?;
        let recorded = slice.record()?;
        let plan = recorded.plan();
        need(
            plan.nodes().len() == 3
                && matches!(
                    plan.nodes()[0].operation,
                    PlanOperation::TypedBulkF32Input { .. }
                )
                && matches!(
                    plan.nodes()[1].operation,
                    PlanOperation::TypedBulkF32Input { .. }
                )
                && matches!(
                    plan.nodes()[2].operation,
                    PlanOperation::TypedBulkF32Matmul { .. }
                )
                && plan.is_operator_slice()
                && plan.required_domains().is_empty()
                && plan.proposed_domains().is_empty()
                && plan.frontier_delta() == 0
                && plan.logits().is_none(),
            "SAME exact three-node pure ForwardPlan",
        )?;
        drop(recorded);
        let baseline = snapshot(&arbiter, &key)?;
        need(
            slice.execute(Some(&cancelled)).is_err() && snapshot(&arbiter, &key)? == baseline,
            "pre-cancel execution retains exact input ledger",
        )?;
        let provider = DeniedProvider(AtomicUsize::new(0));
        need(
            slice.execute_with_provider(Some(&provider), None).is_err()
                && provider.0.load(Ordering::Relaxed) == 0
                && snapshot(&arbiter, &key)? == baseline,
            "unaware provider no callback or output allocation",
        )?;
        let out_words = case.expected.shape.iter().product::<usize>();
        let out_bytes = (out_words * 4) as u64;
        let reserved = arbiter.snapshot(&key)?.reserved;
        let pressure = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: LIMIT - reserved - out_bytes - 256 + 1,
            }],
        )?;
        let pressure_state = snapshot(&arbiter, &key)?;
        need(
            matches!(slice.execute(None),Err(zerogravity_core::Error::Limit(ref text))if text=="physical device memory admission refused")
                && snapshot(&arbiter, &key)? == pressure_state,
            "real one-byte output/scratch admission refusal",
        )?;
        drop(pressure);
        let completed = slice.execute(None)?;
        need(
            completed.source_identity() == source
                && completed.shape() == case.expected.shape
                && completed.values()?.len() == out_words,
            "source-bound complete actual matrix result",
        )?;
        let live = snapshot(&arbiter, &key)?;
        // No expected arithmetic bytes are read until SAME HostMath has returned.
        let expected = read_words(fixtures, &case.expected)?;
        need(
            expected.len() == completed.values()?.len(),
            "complete expected cardinality",
        )?;
        let unequal = completed
            .values()?
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mut actual = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join(format!("{id}.actual.f32le")))?;
        for word in completed.values()? {
            actual.write_all(&word.to_le_bytes())?;
        }
        actual.sync_all()?;
        drop(actual);
        drop(expected);
        let clone = completed.clone();
        drop(completed);
        need(
            clone.view(0..out_words)?.len() == out_words
                && clone.view(out_words..out_words + 1).is_err(),
            "borrowed retained output range",
        )?;
        drop(slice);
        drop(model);
        drop(left);
        drop(right);
        drop(staging);
        need(
            weak.upgrade().is_some(),
            "actual output retains original immutable model source",
        )?;
        let retained = snapshot(&arbiter, &key)?;
        need(
            retained["staging"] == 0
                && retained["output"] == out_bytes
                && retained["scratch"] == 256
                && retained["metadata"] == META + CACHE_METADATA + 16384,
            "input payloads gone; clone retains output ticket + issuer",
        )?;
        drop(clone);
        need(
            weak.upgrade().is_none(),
            "final output releases original source owner",
        )?;
        need(
            arbiter.snapshot(&key)?.reserved == META + source_bytes + CACHE_METADATA,
            "only external source envelope remains",
        )?;
        drop(source_ticket);
        let after = snapshot(&arbiter, &key)?;
        need(after["reserved"] == META, "case resources returned")?;
        reports.push(json!({"case":id,"model_sha256":case.model_sha256,"words":out_words,"unequal_words":unequal,
            "source_bound":source.is_bound(),"source_bytes":source_bytes,"left_shape":case.left.shape,"left_strides":case.left.strides,
            "right_shape":case.right.shape,"plan_nodes":3,"frontier_delta":0,"state_domains":0,"expected_opened_after_math":true,
            "pre_cancel_constructor":true,"pre_cancel_execution":true,"provider_callbacks":0,"real_one_byte_pressure_refused":true,
            "live":live,"retained_output":retained,"after_case":after,"output_clone_retains_source":true}));
    }
    drop(meta);
    let final_state = snapshot(&arbiter, &key)?;
    need(
        final_state["reserved"] == 0 && final_state["fenced"] == false,
        "all original leases final0",
    )?;
    need(mxcsr() & 0xffc0 == 0x1f80, "unchanged source FP controls")?;
    let exact = reports.iter().all(|row| row["unequal_words"] == 0);
    let report = json!({"status":if exact{"PASS_SAME_ENGINE_PURE_BULK_F32_ONLY"}else{"FAIL_SOURCE_BITS"},"cases":reports,
        "before":before,"final":final_state,"actual_affinity":cpus,"mxcsr_before":fp_before,"mxcsr_after":mxcsr(),
        "words":132096,"default_forward_changed":false,"fullmodel_or_StateRoot_qualified":false,"historical_prefix_recovered":false,
        "actual_mid_operation_cancel_qualified":false,"physical_RAM_admission_qualified":false});
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.join("report.json"))?;
    serde_json::to_writer_pretty(&mut f, &report)?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    need(
        exact,
        "unchanged default public outputs must all match; no tolerance",
    )
}
fn main() {
    let args = std::env::args().map(PathBuf::from).collect::<Vec<_>>();
    let result = (|| -> Result<()> {
        need(
            args.len() == 8,
            "FIXTURES NEW_OUTPUT GGUF08 CONFIG08 GGUF2 CONFIG2 TEACHER64",
        )?;
        run(
            &args[1],
            &args[2],
            &[
                (args[3].clone(), args[4].clone()),
                (args[5].clone(), args[6].clone()),
            ],
            &args[7],
        )
    })();
    if let Err(error) = result {
        eprintln!("BULK_TYPED_UNQUALIFIED:{error}");
        std::process::exit(2)
    }
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/owned_lifetime.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11840; SHA256 `53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d`.

````text
"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.

Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
before leader reap: its reserved PID prevents accidental process-group reuse.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
CAP = 4 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def command(argv, timeout=10):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
    return result


def leader_status_unreaped(process):
    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if observed is None:
        return None
    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
            "owned leader exit observation")
    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status


class Log:
    """Main-thread bounded drain; no thread/future can survive cleanup."""
    def __init__(self, pipe, path):
        self.pipe, self.path = pipe, Path(path)
        self.selector = self.file = None
        self.retained = self.total = 0
        self.eof = self.closed = False
        self.errors = []
        try:
            self.file = self.path.open("xb")
            os.chmod(self.path, 0o600)
            os.set_blocking(pipe.fileno(), False)
            self.selector = selectors.DefaultSelector()
            self.selector.register(pipe, selectors.EVENT_READ)
        except BaseException:
            if self.selector is not None:
                self.selector.close()
            if self.file is not None:
                self.file.close()
            raise

    def drain(self, timeout):
        for key, _ in self.selector.select(timeout):
            # One bounded block per wake; the exit/deadline remains observable
            # even when an owned writer continuously produces output.
            try:
                raw = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not raw:
                self.selector.unregister(key.fileobj)
                self.eof = True
                continue
            self.total += len(raw)
            keep = raw[:max(0, CAP - self.retained)]
            if keep:
                try:
                    self.file.write(keep)
                    self.retained += len(keep)
                except OSError as error:
                    self.errors.append(repr(error))

    def observe(self, process, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.drain(0.05)
            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
            status = leader_status_unreaped(process)
            if status is not None:
                return status
        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")

    def finish(self):
        failures = []
        try:
            deadline = time.monotonic() + 3
            while not self.eof and time.monotonic() < deadline:
                self.drain(0.05)
            require(self.eof, "owned stdout pipe EOF deadline")
            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException as error:
            failures.append("drain/evidence: " + repr(error))
        finally:
            # Close all three even when EOF, evidence I/O or one close fails.
            for name, close in (("selector", self.selector.close), ("log", self.file.close),
                                ("pipe", self.pipe.close)):
                try:
                    close()
                except BaseException as error:
                    failures.append(name + ": " + repr(error))
            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
        if failures:
            raise RuntimeError("owned log cleanup: " + "; ".join(failures))


def owned_container(name, nonce, cidfile):
    target, recorded_cid = name, None
    if cidfile.exists():
        cid = cidfile.read_text().strip()
        if cid:
            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
            target = recorded_cid = cid
    exists = command(PODMAN + ["container", "exists", target])
    if exists.returncode == 1:
        return None
    require(exists.returncode == 0, "owned container existence check failed")
    inspected = command(PODMAN + ["container", "inspect", target])
    require(inspected.returncode == 0, "owned container inspect failed")
    value, = json.loads(inspected.stdout)
    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
            "container name/nonce mismatch; no signal")
    cid = value["Id"]
    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
    if recorded_cid is not None:
        require(recorded_cid == cid, "owned CID changed; no signal")
    return cid


def cleanup_container(name, nonce, cidfile):
    errors = []
    try:
        cid = owned_container(name, nonce, cidfile)
    except BaseException as error:
        return ["container owner binding: " + repr(error)]
    if cid is None:
        return errors
    try:
        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
        if stopped.returncode != 0:
            errors.append("owned stop returned nonzero")
    except BaseException as error:
        errors.append("container stop: " + repr(error))
    try:
        # Fresh exact owner binding is required even after a failed/timedout
        # stop. A stop failure must not skip the bounded force-remove attempt.
        if owned_container(name, nonce, cidfile) is not None:
            removed = command(PODMAN + ["rm", "--force", cid], 15)
            require(removed.returncode == 0, "owned container force cleanup failed")
    except BaseException as error:
        errors.append("container force cleanup: " + repr(error))
    try:
        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
    except BaseException as error:
        errors.append("container final absence: " + repr(error))
    return errors


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group_unreaped(process, sig):
    # waitid authenticates that this is still our unreaped child. start_new_session
    # made it the session/group leader; no foreign group is inferred from a PID.
    leader_status_unreaped(process)
    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
            "owned leader session/PGID binding")
    os.killpg(process.pid, sig)


def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
    """Independent bounded stages; no nonzero signal after any reap attempt."""
    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
    errors = []
    try:
        signal_group_unreaped(process, signal.SIGTERM)
        state["term_sent"] = True
    except ProcessLookupError:
        state["term_group_absent_before_reap"] = True
    except BaseException as error:
        errors.append("owned TERM: " + repr(error))
    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            observe(process)
            pause(0.05)
    except BaseException as error:
        errors.append("owned grace: " + repr(error))
    # Both independently authenticated KILL attempts precede ANY wait/reap.
    for attempt in range(2):
        state["kill_attempts_before_reap"] += 1
        try:
            signal_group_unreaped(process, signal.SIGKILL)
            state["kill_sent_before_reap"] = True
        except ProcessLookupError:
            state["kill_group_absent_before_reap"] = True
        except BaseException as error:
            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
    state["all_nonzero_signals_before_reap"] = True
    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
    try:
        state["leader_reap_attempts"] += 1
        state["exit_code"] = process.wait(timeout=5)
        state["leader_reaped"] = True
    except BaseException as error:
        errors.append("owned first leader reap: " + repr(error))
    finally:
        if process.returncode is None:
            try:
                state["leader_reap_attempts"] += 1
                state["exit_code"] = process.wait(timeout=5)
                state["leader_reaped"] = True
            except BaseException as error:
                errors.append("owned final leader reap: " + repr(error))
        else:
            state["exit_code"] = process.returncode
            state["leader_reaped"] = True
    try:
        require(state["leader_reaped"], "owned leader was not reaped")
        deadline = time.monotonic() + 5
        while group_exists(process.pid) and time.monotonic() < deadline:
            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
        state["group_absent"] = not group_exists(process.pid)
        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
    except BaseException as error:
        errors.append("owned group absence: " + repr(error))
    state["stage_errors"] = list(errors)
    if errors:
        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))


def cleanup(process, name, nonce, cidfile, receipt):
    errors = []
    try:
        errors.extend(cleanup_container(name, nonce, cidfile))
    except BaseException as error:
        errors.append("initial owned container cleanup: " + repr(error))
    try:
        cleanup_group(process, receipt)
    except BaseException as error:
        errors.append("owned foreground cleanup: " + repr(error))
    # Rediscover startup races using only the original nonce/name/CID binding.
    try:
        errors.extend(cleanup_container(name, nonce, cidfile))
    except BaseException as error:
        errors.append("final owned container cleanup: " + repr(error))
    try:
        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
        receipt["owned_container_absent_after_reap"] = True
    except BaseException as error:
        errors.append("final owned absence: " + repr(error))
    return errors

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/prepare.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 2978; SHA256 `2771c29d4d02019957dda1b5be2e42598b19ea9620a6cec20ac29031f493c554`.

````text
"""ROOT ONLY copy original operands/defaults unchanged; no model/operator runtime."""
import argparse
from pathlib import Path
from contracts import HERE, REPO, CASES, digest, frozen, parents, require, verify_transport, write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new-fixtures',type=Path,required=True)
    a=p.parse_args();pin=frozen();old,prep,av,alpha=parents()
    require(a.new_fixtures.is_absolute()and a.new_fixtures.parent.resolve()==HERE and not a.new_fixtures.exists(),'fresh own fixture directory')
    a.new_fixtures.mkdir(mode=0o700);rows=[]
    for case in CASES:
        row,paths=prep.transport_descriptor(case);rows.append(row)
        for key,path in zip(('left','right','expected'),paths,strict=True):
            require(digest(path,1024*1024)==row[key]['digest'],'original full raw source bytes')
            target=a.new_fixtures/row[key]['file']
            with path.open('rb')as src,target.open('xb')as dst:
                remaining=row[key]['digest']['bytes']
                while remaining:
                    raw=src.read(min(remaining,1024*1024));require(raw,'complete raw source');dst.write(raw);remaining-=len(raw)
                require(not src.read(1),'source did not grow')
            target.chmod(0o600)
    write(a.new_fixtures/'manifest.json',{'status':'AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY',
          'profile':'GgmlD812350HaswellTinyBlas','cases':rows,'arithmetic_executed':False,'expected_from_candidate':False,
          'gap_storage':'actual dense logical V restored at original strides; uncaptured gaps zero-filled'})
    av_ids=old.AV.parent/'numerics-and-context-bulk-causality-20261020/root-first64-ar1-budget/input_ids.u32le'
    alpha_ids=alpha.CAPTURE/'input_ids.u32le';require(av_ids.read_bytes()==alpha_ids.read_bytes(),'actual same teachers')
    ids=av_ids.read_bytes();require(len(ids)==32768,'full8192 source IDs')
    (a.new_fixtures/'teacher64.u32le').write_bytes(ids[:256]);(a.new_fixtures/'teacher64.u32le').chmod(0o600)
    models=[]
    for case,alias in zip(CASES,('qwen35-0.8b','qwen35-2b'),strict=True):
        model=REPO/'target/models'/alias/'model.gguf';config=model.with_name('config.json')
        models.append({'case':case,'gguf':str(model),'gguf_digest':digest(model),
                       'config':str(config),'config_digest':digest(config,1024*1024)})
    write(a.new_fixtures/'typed-inputs.json',{'status':'AUTHENTIC_SAME_ENGINE_BULK_F32_INPUTS_ONLY',
          'arithmetic_executed':False,'sourceprofile':'GGUF_ORIGINAL','base':0,'capacity':8448,'teachers':64,
          'teacher_digest':digest(a.new_fixtures/'teacher64.u32le',256),
          'teacher_sources':[{'path':str(path),'digest':digest(path,32768)}for path in(av_ids,alpha_ids)],'models':models})
    verify_transport(a.new_fixtures);require(frozen()==pin,'same reviewed source after transport')
    print('AUTHENTIC_SAME_ENGINE_BULK_F32_INPUTS_ONLY')
if __name__=='__main__':main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/proposed/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6711; SHA256 `fbbd8b45b26c201e558d555dbc41335010bf391e5578ceb0e50c9cc8a028c1fc`.

````text
use super::*;
fn span() -> TeacherSpan {
    TeacherSpan::new(
        SourceIdentity::for_new_allocation().unwrap(),
        0,
        8448,
        248320,
        &(0..64).collect::<Vec<_>>(),
    )
    .unwrap()
}
fn matrix(span: &TeacherSpan) -> BulkF32Matrix {
    let left = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Left,
        "actual external V",
        BulkF32Layout::new([256, 256, 2, 1], [1, 8448, 2162688, 4325376], 0, 4317184).unwrap(),
    )
    .unwrap();
    let right = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "actual external P",
        BulkF32Layout::dense([256, 64, 8, 1]).unwrap(),
    )
    .unwrap();
    BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap()
}
#[test]
fn actual_temporal_n_is_not_cohort_width_and_full_k_is_preserved() {
    let span = span();
    let operation = matrix(&span);
    assert_eq!(operation.left().layout().shape(), [256, 256, 2, 1]);
    assert_eq!(operation.right().layout().shape(), [256, 64, 8, 1]);
    assert_eq!(operation.output().shape(), [256, 64, 8, 1]);
    assert_eq!(operation.output().backing_words(), 131072);
    // Real N2 is a different declared teacher span, never a B2 request label.
    let wrong = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "P",
        BulkF32Layout::dense([256, 2, 8, 1]).unwrap(),
    );
    assert!(wrong.is_err());
    assert!(TeacherSpan::new(span.source(), 0, 8448, 248320, &[11]).is_err());
    assert!(TeacherSpan::new(span.source(), 8447, 8448, 248320, &[11, 353]).is_err());
}
#[test]
fn same_ids_source_names_do_not_merge_issuers_and_wrong_full_k_refuses() {
    let span = span();
    let operation = matrix(&span);
    let foreign = TeacherSpan::new(
        span.source(),
        span.base(),
        span.capacity(),
        248320,
        span.ids(),
    )
    .unwrap();
    let right = BulkF32Operand::external(
        foreign,
        BulkF32Side::Right,
        "actual external P",
        operation.right().layout(),
    )
    .unwrap();
    assert!(
        BulkF32Matrix::new(
            operation.left().clone(),
            right,
            BulkF32Contract::HaswellTinyBlasF32V1
        )
        .is_err()
    );
    let right = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "P",
        BulkF32Layout::dense([128, 64, 8, 1]).unwrap(),
    )
    .unwrap();
    assert!(
        BulkF32Matrix::new(
            operation.left().clone(),
            right,
            BulkF32Contract::HaswellTinyBlasF32V1
        )
        .is_err()
    );
}
#[test]
fn real_nonzero_offsets_and_gaps_need_full_checked_storage() {
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 47).is_ok());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 46).is_err());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 7, 28, 28], 0, 32).is_err());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], usize::MAX, 64).is_err());
    assert!(BulkF32Layout::dense([8192, 4096, 8, 1]).is_err());
    let span = span();
    assert!(
        BulkF32Operand::external(
            span,
            BulkF32Side::Right,
            "strided P",
            BulkF32Layout::new([8, 64, 1, 1], [1, 12, 768, 768], 0, 764).unwrap()
        )
        .is_err()
    );
}
#[test]
fn same_builder_three_nodes_stay_pure_and_legacy_use_or_foreign_values_refuse() {
    let span = span();
    let operation = matrix(&span);
    let mut b =
        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let left = b.bulk_f32_input(operation.left()).unwrap();
    let right = b.bulk_f32_input(operation.right()).unwrap();
    let mut other =
        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let foreign = other.bulk_f32_input(operation.right()).unwrap();
    assert!(b.bulk_f32_matmul(&operation, &left, &foreign).is_err());
    assert!(b.silu("legacy", &right, None).is_err());
    let output = b.bulk_f32_matmul(&operation, &left, &right).unwrap();
    let slice = b.finish_operator_slice(output).unwrap();
    assert_eq!(slice.plan().nodes().len(), 3);
    assert!(slice.plan().required_domains().is_empty());
    assert!(slice.plan().proposed_domains().is_empty());
    assert_eq!(slice.plan().frontier_delta(), 0);
    assert!(slice.plan().logits().is_none());
    assert!(slice.plan().is_operator_slice());
    let slot = StateSlot::Conv(ConvDomain(0));
    let mut full = PlanBuilder::new(span.source(), 8448, vec![slot], false).unwrap();
    assert!(full.bulk_f32_input(operation.left()).is_err());
}

struct Original {
    identity: SourceIdentity,
    kind: QuantKind,
    shape: [u64; 2],
    bytes: Vec<u8>,
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        if name != "original.weight" {
            return Err(Error::Malformed("unknown original name".into()));
        }
        Ok(SourceTensorRef {
            kind: self.kind,
            shape: &self.shape,
            packed_bytes: &self.bytes,
        })
    }
}
#[test]
fn whole_raw_source_matrix_keeps_axes_dtype_extent_and_exact_issuer() {
    let original = Original {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        kind: QuantKind::F32,
        shape: [8, 4],
        bytes: vec![0; 128],
    };
    let span = TeacherSpan::new(original.identity, 0, 8, 248320, &[11, 353]).unwrap();
    let left = BulkF32Operand::source_matrix(span.clone(), &original, "original.weight").unwrap();
    assert_eq!(left.layout().shape(), [8, 4, 1, 1]);
    let BulkF32Origin::SourceMatrix(raw) = left.origin() else {
        panic!("original raw source expected")
    };
    assert_eq!(raw.axes, MatrixAxes::InputOutput);
    raw.validate_source(&original).unwrap();
    let foreign = Original {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        ..original
    };
    assert!(BulkF32Operand::source_matrix(span.clone(), &foreign, "original.weight").is_err());
    let bf16 = Original {
        identity: span.source(),
        kind: QuantKind::Bf16,
        shape: [8, 4],
        bytes: vec![0; 64],
    };
    assert!(BulkF32Operand::source_matrix(span.clone(), &bf16, "original.weight").is_err());
    let short = Original {
        identity: span.source(),
        kind: QuantKind::F32,
        shape: [8, 4],
        bytes: vec![0; 124],
    };
    assert!(BulkF32Operand::source_matrix(span, &short, "original.weight").is_err());
}

````

