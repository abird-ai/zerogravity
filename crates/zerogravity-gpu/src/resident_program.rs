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

fn supported_prefix_capacity(capacity: usize) -> bool {
    matches!(capacity, 256 | 512)
        || (cfg!(all(feature = "qualification", feature = "cuda"))
            && matches!(capacity, 768 | 1024))
}

mod physical_layout;
pub use physical_layout::{ResidentBufferLayout, ResidentPhysicalType};

mod batch;
pub(crate) use batch::BatchProgram;
#[cfg(all(feature = "qualification", target_os = "linux"))]
pub(crate) mod typed_slice;
#[cfg(all(feature = "qualification", feature = "cuda"))]
pub(crate) use batch::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay};

#[cfg(feature = "qualification")]
#[path = "resident_projection_qualification.rs"]
mod projection_qualification;

/// Allocation coordinates only. Original handles and admitted leases remain
/// the lifetime owners; this copyable metadata never pins or promotes memory.
#[derive(Clone, Debug)]
pub(crate) struct BufferRange {
    service: ServiceId,
    memory: ManagedMemoryId,
    bytes: Range<u64>,
}
impl BufferRange {
    pub(crate) fn new(handle: &Handle) -> Result<Self> {
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
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        self.service == other.service
            && self.memory == other.memory
            && self.bytes.start < other.bytes.end
            && other.bytes.start < self.bytes.end
    }
    #[cfg(all(feature = "qualification", target_os = "linux"))]
    pub(crate) fn require_exact_len(&self, bytes: usize) -> Result<()> {
        if self.bytes.end - self.bytes.start != bytes as u64 {
            return Err(Error::State("resident exact logical buffer length".into()));
        }
        Ok(())
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

#[derive(Clone)]
pub(crate) struct DeviceBuffer {
    pub handle: Handle,
    pub elements: usize,
    physical: ResidentBufferLayout,
    // Typed external binding retains its ORIGINAL admitted lease. Workspace
    // allocations instead remain in the enclosing existing owner/lease list.
    lease: Option<crate::resident_leases::ResidentLease>,
}
impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("handle", &self.handle)
            .field("elements", &self.elements)
            .field("physical", &self.physical)
            .field(
                "lease_id",
                &self
                    .lease
                    .as_ref()
                    .map(crate::resident_leases::ResidentLease::id),
            )
            .finish()
    }
}
impl DeviceBuffer {
    /// Existing four-byte ABI (including opaque original token/control words).
    /// Logical activation precision never selects physical storage here.
    pub fn new(handle: Handle, elements: usize) -> Self {
        Self {
            handle,
            elements,
            physical: ResidentBufferLayout::legacy_f32(elements),
            lease: None,
        }
    }
    pub(crate) fn from_lease(
        lease: crate::resident_leases::ResidentLease,
        physical: ResidentBufferLayout,
        role: zerogravity_core::device_events::BufferRole,
        access: zerogravity_core::device_events::DeviceAccess,
    ) -> Result<Self> {
        let value = Self {
            handle: lease.handle(),
            elements: physical.elements(),
            physical,
            lease: Some(lease),
        };
        value.validate_lease(value.lease.as_ref().unwrap(), role, access)?;
        Ok(value)
    }
    #[cfg(all(
        feature = "qwen-source-qualification",
        feature = "hardware-abort-probe",
        target_os = "linux"
    ))]
    pub(crate) fn final_owner_original_lease(
        &self,
    ) -> Option<&crate::resident_leases::ResidentLease> {
        self.lease.as_ref()
    }
    /// Original admitted lease only; public handle metadata cannot retarget it.
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn original_lease(&self) -> Result<&crate::resident_leases::ResidentLease> {
        let lease = self.lease.as_ref().ok_or_else(|| {
            Error::State("typed resident value has no original lease owner".into())
        })?;
        self.validate_lease(
            lease,
            lease.role(),
            zerogravity_core::device_events::DeviceAccess::Read,
        )?;
        Ok(lease)
    }
    pub(crate) fn require_layout(&self, expected: ResidentBufferLayout) -> Result<()> {
        self.physical.require(expected)?;
        if self.elements != self.physical.elements() {
            return Err(Error::State(
                "resident physical descriptor element count changed".into(),
            ));
        }
        Ok(())
    }
    fn validate_lease(
        &self,
        lease: &crate::resident_leases::ResidentLease,
        role: zerogravity_core::device_events::BufferRole,
        access: zerogravity_core::device_events::DeviceAccess,
    ) -> Result<()> {
        self.physical.validate_role(role, access)?;
        let expected = BufferRange::new(&lease.handle())?;
        let actual = BufferRange::new(&self.handle)?;
        if lease.role() != role
            || lease.bytes() != self.physical.bytes()?
            || expected.service != actual.service
            || expected.memory != actual.memory
            || expected.bytes != actual.bytes
        {
            return Err(Error::State(
                "resident typed original lease/source/role/range differs".into(),
            ));
        }
        self.physical
            .validate_region(actual.bytes.start, actual.bytes.end - actual.bytes.start)?;
        Ok(())
    }
    /// Raw descriptor only. Actual completion/readback ownership stays external.
    pub(crate) fn readback(
        &self,
        lease: &crate::resident_leases::ResidentLease,
        expected: ResidentBufferLayout,
    ) -> Result<crate::resident_leases::ResidentReadback> {
        use zerogravity_core::device_events::DeviceAccess;
        self.require_layout(expected)?;
        self.validate_lease(lease, lease.role(), DeviceAccess::Read)?;
        if let Some(original) = &self.lease
            && (original.id() != lease.id() || !Arc::ptr_eq(original.service(), lease.service()))
        {
            return Err(Error::State(
                "resident typed readback original lease differs".into(),
            ));
        }
        lease.readback(0, self.physical.bytes()?)
    }
    /// Existing shaders/state/control consumers retain exact original F32-word
    /// minimum-range validation; a BF16 tag can never enter that ABI.
    fn check(&self, client: &Client, expected: usize) -> Result<BufferRange> {
        self.require_layout(ResidentBufferLayout::legacy_f32(expected))?;
        let range = BufferRange::new(&self.handle)?;
        if self.elements != expected
            || range.service != client.service_id()
            || range.bytes.end - range.bytes.start < self.physical.bytes()? as u64
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
struct NativeExecution<'a> {
    #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
    qwen: Option<&'a dyn qwen_source::Execution>,
    #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
    gemma: Option<&'a dyn gemma_text::Execution>,
    cancelled: Option<&'a std::sync::atomic::AtomicBool>,
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    phase: Option<&'a crate::resident_inference::native_fragment::PrefixPhaseEnqueue<'a>>,
}
impl<'a> NativeExecution<'a> {
    fn new(cancelled: Option<&'a std::sync::atomic::AtomicBool>) -> Self {
        Self {
            cancelled,
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            phase: None,
            #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
            gemma: None,
            #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
            qwen: None,
        }
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    fn with_phase(
        mut self,
        phase: &'a crate::resident_inference::native_fragment::PrefixPhaseEnqueue<'a>,
    ) -> Self {
        self.phase = Some(phase);
        self
    }
    #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
    fn with_gemma(mut self, execution: &'a dyn gemma_text::Execution) -> Self {
        self.gemma = Some(execution);
        self
    }
    #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
    fn with_qwen(mut self, execution: &'a dyn qwen_source::Execution) -> Self {
        self.qwen = Some(execution);
        self
    }
}
enum CurrentExecution<'a> {
    Single {
        parameters: RuntimeParameters,
        states: &'a [StateBinding],
    },
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    Cohort {
        permits: &'a [crate::resident_inference::native_fragment::PrefixPhaseEnqueue<'a>],
    },
}
impl CurrentExecution<'_> {
    fn single(&self) -> Result<(RuntimeParameters, &[StateBinding])> {
        match self {
            Self::Single { parameters, states } => Ok((*parameters, states)),
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            Self::Cohort { .. } => Err(Error::State(
                "cohort cannot borrow first-row runtime/state authority".into(),
            )),
        }
    }
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
#[cfg(any(
    feature = "vllm-cuda-qualification",
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
impl StateLayout {
    /// Actual plane count and F32 word extent per plane. A KV domain is two
    /// planes of one checked `slab*capacity` extent; total requirement bytes/4
    /// covers both and is never a plane extent.
    pub(crate) fn planes(&self) -> Result<(usize, usize)> {
        match *self {
            Self::Vector { elements } => Ok((1usize, elements)),
            Self::Kv { slab, capacity } => Ok((
                2usize,
                slab.checked_mul(capacity)
                    .ok_or_else(|| Error::Limit("resident state KV plane extent".into()))?,
            )),
        }
    }
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

#[derive(Clone)]
enum ResidentProgramSource {
    Owned(Arc<dyn TensorSource>),
    #[cfg(all(feature = "qualification", target_os = "linux"))]
    TypedWindows(zerogravity_artifacts::source_weights::VerifiedTensorSource),
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    Windows(Arc<zerogravity_model_qwen35::original_binding::OriginalWindowModelSource>),
}
impl ResidentProgramSource {
    fn source_identity(&self) -> SourceIdentity {
        match self {
            Self::Owned(source) => source.source_identity(),
            #[cfg(all(feature = "qualification", target_os = "linux"))]
            Self::TypedWindows(source) => source.source_identity(),
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            Self::Windows(source) => source.source_identity(),
        }
    }
}
struct LegacyProgram {
    tensors: BTreeMap<String, ResidentTensor>,
    constants: Vec<Handle>,
    scratch: [Handle; SCRATCH_SLOTS],
    node_status: Vec<NodeStatus>,
    projections: Vec<Option<Box<DeviceActivation>>>,
    token: Handle,
    exp_tables: Handle,
    trig_tables: Handle,
}
enum ResidentProgramBody {
    #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
    QwenSource(Arc<qwen_source::QwenSourceBody>),
    Legacy(Box<LegacyProgram>),
    #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
    GemmaText(Arc<gemma_text::GemmaTextBody>),
    #[cfg(all(feature = "qualification", target_os = "linux"))]
    TypedSlice(Arc<typed_slice::TypedBody>),
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    Native(Arc<native::NativeProgram>),
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    NativeFull(Arc<native_full::FullModelProgram>),
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    NativeCohort(Arc<native_cohort::FullModelCohort>),
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    Qwen2bComponent(Arc<qwen2b_component::ComponentProgram>),
}
pub(crate) struct ResidentProgram {
    _source_owner: ResidentProgramSource,
    plan: ForwardPlan,
    preflight: ResidentPreflight,
    values: Vec<DeviceBuffer>,
    status: Handle,
    owned_ranges: OwnedRanges,
    body: ResidentProgramBody,
}
#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod canonical47_binding;
#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod native;
#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod native_cohort;
#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod native_full;
#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod natural_binding;
#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod qwen2b_component;

#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    target_os = "linux"
))]
pub(crate) mod bf16_projection;
#[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
pub(crate) mod gemma_text;

#[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
pub(crate) mod qwen_source;

#[derive(Clone, Debug)]
pub(crate) struct ResidentSubmission {
    pub logits: Option<DeviceBuffer>,
    pub proposals: Vec<(StateVersion, StateBuffers)>,
    /// Present only for sealed real request cohorts; never projected through one plan.
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub cohort_proposals: Option<Vec<Vec<(StateVersion, StateBuffers)>>>,
    pub status: Handle,
    /// Every dynamic-quant status pair, in plan order. Zero is success.
    pub quant_status: Vec<(Handle, usize, Handle, usize)>,
    pub embedding_status: Vec<(Handle, usize)>,
    pub dispatches: usize,
    /// Conservative subset: actual native multi-row projection launches.
    pub fused_dispatches: usize,
}

impl ResidentProgram {
    pub(crate) fn is_gemma_text(&self) -> bool {
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        {
            matches!(self.body, ResidentProgramBody::GemmaText(_))
        }
        #[cfg(not(all(feature = "gemma-bf16-qualification", target_os = "linux")))]
        {
            false
        }
    }
    fn is_typed_slice(&self) -> bool {
        #[cfg(all(feature = "qualification", target_os = "linux"))]
        {
            matches!(self.body, ResidentProgramBody::TypedSlice(_))
        }
        #[cfg(not(all(feature = "qualification", target_os = "linux")))]
        {
            false
        }
    }
    fn is_source_window_body(&self) -> bool {
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if self.is_qwen_source() {
            return true;
        }
        self.is_gemma_text()
    }
    fn legacy(&self) -> Result<&LegacyProgram> {
        match &self.body {
            ResidentProgramBody::Legacy(body) => Ok(body),
            #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
            ResidentProgramBody::QwenSource(_) => Err(Error::Unsupported(
                "source native body cannot enter legacy execution".into(),
            )),
            #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
            ResidentProgramBody::GemmaText(_) => Err(Error::Unsupported(
                "Gemma text cannot enter legacy execution".into(),
            )),
            #[cfg(all(feature = "qualification", target_os = "linux"))]
            ResidentProgramBody::TypedSlice(_) => Err(Error::Unsupported(
                "typed slice cannot enter legacy execution".into(),
            )),
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            ResidentProgramBody::Native(_)
            | ResidentProgramBody::NativeFull(_)
            | ResidentProgramBody::NativeCohort(_)
            | ResidentProgramBody::Qwen2bComponent(_) => Err(Error::Unsupported(
                "native fragment cannot use legacy F32 program".into(),
            )),
        }
    }
    fn legacy_mut(&mut self) -> Result<&mut LegacyProgram> {
        match &mut self.body {
            ResidentProgramBody::Legacy(body) => Ok(body),
            #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
            ResidentProgramBody::QwenSource(_) => Err(Error::Unsupported(
                "source native body cannot enter legacy execution".into(),
            )),
            #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
            ResidentProgramBody::GemmaText(_) => Err(Error::Unsupported(
                "Gemma text cannot enter legacy execution".into(),
            )),
            #[cfg(all(feature = "qualification", target_os = "linux"))]
            ResidentProgramBody::TypedSlice(_) => Err(Error::Unsupported(
                "typed slice cannot enter legacy execution".into(),
            )),
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            ResidentProgramBody::Native(_)
            | ResidentProgramBody::NativeFull(_)
            | ResidentProgramBody::NativeCohort(_)
            | ResidentProgramBody::Qwen2bComponent(_) => Err(Error::Unsupported(
                "native fragment cannot use legacy F32 program".into(),
            )),
        }
    }
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
                    PlanOperation::TextMoe { .. }
                        | PlanOperation::TextElementwise(_)
                        | PlanOperation::TextRotary { .. }
                        | PlanOperation::TextWindowQk { .. }
                        | PlanOperation::TypedBulkF32Input { .. }
                        | PlanOperation::TypedBulkF32Matmul { .. }
                        | PlanOperation::TypedZenSsmOutInput { .. }
                        | PlanOperation::TypedZenSsmOutProjection { .. }
                        | PlanOperation::TypedGgufPrefixEmbedding { .. }
                        | PlanOperation::TypedGgufPrefixRms { .. }
                        | PlanOperation::TypedGgufGdnInput { .. }
                        | PlanOperation::TypedGgufGdnProjection { .. }
                        | PlanOperation::TypedGgufGdnConvSilu { .. }
                        | PlanOperation::TypedGgufGdnQk { .. }
                        | PlanOperation::TypedEmbed { .. }
                        | PlanOperation::TypedProject { .. }
                        | PlanOperation::TypedOrdinaryRms { .. }
                        | PlanOperation::TypedVllmBf16Embedding { .. }
                        | PlanOperation::TypedVllmInputRms { .. }
                        | PlanOperation::TypedVllmGdnConv { .. }
                        | PlanOperation::TypedVllmGdnRecurrent { .. }
                        | PlanOperation::TypedVllmStateCarry
                        | PlanOperation::TypedVllm08 { .. }
                        | PlanOperation::TypedVllmGdnProjection { .. }
                        | PlanOperation::TypedVllmSiluAndMul { .. }
                        | PlanOperation::TypedResidualRms { .. }
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
            || !supported_prefix_capacity(plan.prefix_capacity())
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
                .map(|(s, _)| {
                    s.capacity_elements()
                        .and_then(|n| ResidentBufferLayout::legacy_f32(n).bytes())
                })
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
                allocation.empty_layout(
                    ResidentBufferRole::Value,
                    ResidentBufferLayout::legacy_f32(n),
                )?
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
            _source_owner: ResidentProgramSource::Owned(model),
            plan,
            preflight,
            values,
            status: allocation.empty(ResidentBufferRole::Status, 4)?,
            owned_ranges: OwnedRanges::new(client.service_id()),
            body: ResidentProgramBody::Legacy(Box::new(LegacyProgram {
                tensors,
                constants,
                scratch,
                node_status,
                projections: Vec::new(),
                token: allocation.empty(ResidentBufferRole::Parameter, 4)?,
                exp_tables: allocation.upload(
                    ResidentBufferRole::Source,
                    f64::as_bytes(&crate::source_transcendentals::TABLES),
                )?,
                trig_tables: allocation.upload(
                    ResidentBufferRole::Source,
                    f64::as_bytes(&crate::source_trig::TRIG_TABLES),
                )?,
            })),
        };
        let projections = program
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
        program.legacy_mut()?.projections = projections;
        #[cfg(feature = "cuda")]
        program.preflight_silu(client, 1)?;
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
        let legacy = self.legacy()?;
        let t = &legacy.tensors[&tensor.name];
        let layout = t
            .layout
            .ok_or_else(|| Error::State("resident quant projection layout absent".into()))?;
        let flags = &legacy.node_status[index];
        let input = self.plan.nodes()[index].inputs[0];
        DeviceActivation::from_bound(
            client,
            &layout,
            self.values[input.0 as usize].handle.clone(),
            t.scales.clone(),
            t.biases.clone(),
            [
                legacy.scratch[0].clone(),
                legacy.scratch[1].clone(),
                legacy.scratch[2].clone(),
                legacy.scratch[3].clone(),
                legacy.scratch[4].clone(),
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
        self._source_owner.source_identity()
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
        #[cfg(all(feature = "qualification", target_os = "linux"))]
        if let ResidentProgramBody::TypedSlice(body) = &self.body {
            return body.accesses().iter().map(|(l, _)| l.handle()).collect();
        }
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        if let ResidentProgramBody::GemmaText(body) = &self.body {
            return body.accesses().iter().map(|(l, _)| l.handle()).collect();
        }
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        match &self.body {
            ResidentProgramBody::Native(native) => {
                return native.handles(&self.values, &self.status);
            }
            ResidentProgramBody::NativeFull(native) => {
                return native.handles().expect("sealed full24 phase accesses");
            }
            ResidentProgramBody::NativeCohort(body) => {
                return body
                    .accesses()
                    .iter()
                    .map(|(lease, _)| lease.handle())
                    .collect();
            }
            ResidentProgramBody::Qwen2bComponent(body) => {
                return body.handles().expect("sealed source2B component accesses");
            }
            _ => {}
        }
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if let ResidentProgramBody::QwenSource(body) = &self.body {
            return body.accesses().iter().map(|(l, _)| l.handle()).collect();
        }
        let legacy = self.legacy().expect("legacy program body checked");
        let nodes = output.nodes(&self.plan);
        let mut handles = vec![
            self.status.clone(),
            legacy.token.clone(),
            legacy.exp_tables.clone(),
            legacy.trig_tables.clone(),
        ];
        for t in legacy.tensors.values() {
            handles.extend([t.packed.clone(), t.scales.clone(), t.biases.clone()]);
        }
        handles.extend(legacy.constants.iter().cloned());
        handles.extend(legacy.scratch.iter().cloned());
        // SSA aliases retain the same original handle. The enclosing owner
        // selects each admitted lease once; all alias producers precede it in
        // this same closed prefix, including during state-only prefill.
        handles.extend(
            nodes
                .iter()
                .flat_map(|n| &n.outputs)
                .map(|id| self.values[id.0 as usize].handle.clone()),
        );
        for s in &legacy.node_status[..nodes.len()] {
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
        // Native BF16 bodies retain their own source/phase checks. Ordered
        // source SiLU belongs only to the original legacy F32 body.
        #[cfg(feature = "cuda")]
        if matches!(&self.body, ResidentProgramBody::Legacy(_)) {
            self.preflight_silu(client, 1)?;
        }
        #[cfg(all(feature = "qualification", target_os = "linux"))]
        if let ResidentProgramBody::TypedSlice(body) = &self.body {
            return body.validate_runtime(client, p, bindings);
        }
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        if let ResidentProgramBody::GemmaText(body) = &self.body {
            body.validate_runtime(client, p)?;
        }
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if let ResidentProgramBody::QwenSource(body) = &self.body {
            body.validate_runtime(client, p)?;
        }
        self.validate_state_runtime(client, p, bindings)
    }
    fn validate_state_runtime(
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
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        let native = match &self.body {
            ResidentProgramBody::Native(native) => {
                native.validate_runtime(p)?;
                true
            }
            ResidentProgramBody::NativeFull(native) => {
                native.validate_runtime(p)?;
                true
            }
            ResidentProgramBody::Qwen2bComponent(body) => {
                body.validate_runtime(p)?;
                true
            }
            _ => false,
        };
        #[cfg(not(all(feature = "vllm-cuda-qualification", target_os = "linux")))]
        let native = false;
        if !native
            && (p.valid_tokens
                != p.position
                    .checked_add(1)
                    .ok_or_else(|| Error::Limit("resident position overflow".into()))?
                || p.valid_tokens > self.plan.prefix_capacity()
                || p.position > 32768)
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
        self.legacy()?;
        self.launch_impl(
            client,
            CurrentExecution::Single {
                parameters: p,
                states: bindings,
            },
            None,
            output,
            NativeExecution::new(None),
        )
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
        self.legacy()?;
        token.check(client, 1)?;
        self.launch_impl(
            client,
            CurrentExecution::Single {
                parameters: p,
                states: bindings,
            },
            Some(token),
            output,
            NativeExecution::new(None),
        )
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn launch_fragment(
        &mut self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<ResidentSubmission> {
        if !matches!(self.body, ResidentProgramBody::Native(_))
            || self.plan.vllm_gdn_fragment().is_none()
        {
            return Err(Error::Unsupported(
                "not an unpublishable native fragment".into(),
            ));
        }
        self.launch_impl(
            client,
            CurrentExecution::Single {
                parameters: p,
                states: bindings,
            },
            None,
            ResidentOutput::StateOnly,
            NativeExecution::new(cancelled),
        )
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn launch_attempt_fragment(
        &mut self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        permit: &crate::resident_inference::native_fragment::PrefixPhaseEnqueue<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<ResidentSubmission> {
        match &self.body {
            ResidentProgramBody::Native(body) => body.require_attempt(permit)?,
            ResidentProgramBody::NativeFull(body) => body.require_attempt(permit)?,
            ResidentProgramBody::Qwen2bComponent(body) => body.require_attempt(permit)?,
            _ => return Err(Error::Unsupported("original native body required".into())),
        }
        self.launch_impl(
            client,
            CurrentExecution::Single {
                parameters: p,
                states: bindings,
            },
            None,
            ResidentOutput::StateOnly,
            NativeExecution::new(cancelled).with_phase(permit),
        )
    }
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn launch_cohort<'a>(
        &mut self,
        client: &Client,
        permits: &'a [crate::resident_inference::native_fragment::PrefixPhaseEnqueue<'a>],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<ResidentSubmission> {
        self.launch_impl(
            client,
            CurrentExecution::Cohort { permits },
            None,
            ResidentOutput::StateOnly,
            NativeExecution::new(cancelled),
        )
    }
    fn launch_impl(
        &mut self,
        client: &Client,
        current: CurrentExecution<'_>,
        token: Option<&DeviceBuffer>,
        output: ResidentOutput,
        execution: NativeExecution<'_>,
    ) -> Result<ResidentSubmission> {
        let cancelled = execution.cancelled;
        match &current {
            CurrentExecution::Single { parameters, states } => {
                #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
                if matches!(self.body, ResidentProgramBody::NativeCohort(_)) {
                    return Err(Error::State(
                        "cohort requires every original row permit".into(),
                    ));
                }
                self.validate_runtime(client, *parameters, states)?;
            }
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            CurrentExecution::Cohort { permits } => {
                let ResidentProgramBody::NativeCohort(body) = &self.body else {
                    return Err(Error::State(
                        "cohort current execution requires sealed cohort body".into(),
                    ));
                };
                if body.status().service().client().service_id() != client.service_id() {
                    return Err(Error::State("cohort physical client differs".into()));
                }
                body.begin_launch(permits, cancelled)?;
            }
        }
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        match &self.body {
            ResidentProgramBody::Native(native) => {
                native.begin_launch(cancelled, execution.phase)?
            }
            ResidentProgramBody::NativeFull(native) => {
                native.begin_launch(cancelled, execution.phase)?
            }
            ResidentProgramBody::Qwen2bComponent(body) => {
                body.begin_launch(cancelled, execution.phase)?
            }
            _ => {}
        }
        #[cfg(not(all(feature = "vllm-cuda-qualification", target_os = "linux")))]
        let _ = cancelled;
        #[cfg(all(feature = "qualification", target_os = "linux"))]
        if let ResidentProgramBody::TypedSlice(body) = &self.body {
            body.begin()?;
        }
        let mut dispatches = 0;
        unsafe {
            crate::program_kernels::clear_step_status::launch_unchecked(
                client,
                grid(1),
                threads(),
                arg(&self.status, 1),
            );
            if self.plan.vllm_gdn_fragment().is_none()
                && !self.is_typed_slice()
                && !self.is_source_window_body()
            {
                let legacy = self.legacy()?;
                if let Some(token) = token {
                    self.copy::<false>(
                        client,
                        &token.handle,
                        1,
                        &legacy.token,
                        1,
                        TensorView::vector(0, 1),
                        0,
                    );
                } else {
                    crate::resident_bindings::write_token::launch_unchecked(
                        client,
                        grid(1),
                        threads(),
                        arg(&legacy.token, 1),
                        current.single()?.0.token,
                    );
                }
            }
        }
        dispatches += 1 + usize::from(
            self.plan.vllm_gdn_fragment().is_none()
                && !self.is_typed_slice()
                && !self.is_source_window_body(),
        );
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        let nodes = match &self.body {
            ResidentProgramBody::Qwen2bComponent(body) => body.recipe().nodes(),
            ResidentProgramBody::NativeFull(body) => body.execution_nodes()?,
            ResidentProgramBody::NativeCohort(body) => body.nodes(),
            #[cfg(all(feature = "qualification", target_os = "linux"))]
            ResidentProgramBody::TypedSlice(_) => self.plan.nodes(),
            _ => output.nodes(&self.plan),
        };
        #[cfg(not(all(feature = "vllm-cuda-qualification", target_os = "linux")))]
        let nodes = if self.is_typed_slice() {
            self.plan.nodes()
        } else {
            output.nodes(&self.plan)
        };
        for (index, node) in nodes.iter().enumerate() {
            dispatches += match &current {
                CurrentExecution::Single { parameters, states } => {
                    self.launch_node(client, index, node, *parameters, states, &execution)?
                }
                #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
                CurrentExecution::Cohort { permits } => {
                    let ResidentProgramBody::NativeCohort(body) = &self.body else {
                        unreachable!("validated cohort body")
                    };
                    body.enqueue(index, node, permits, cancelled)?
                }
            };
            // A preflight-validated View fully writes its dense destination and
            // certifies every copied word in the same kernel. All other output
            // certificates (including Contiguous aliases) remain separate.
            if matches!(node.operation, PlanOperation::View(_)) {
                #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
                if let ResidentProgramBody::QwenSource(body) = &self.body {
                    body.finish_node(index, node)?;
                }
                #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
                match &self.body {
                    ResidentProgramBody::NativeFull(body) => {
                        body.finish_value_storage_node(index)?
                    }
                    ResidentProgramBody::NativeCohort(body) => {
                        body.finish_value_storage_node(index)?
                    }
                    _ => {}
                }
                #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
                if let ResidentProgramBody::GemmaText(body) = &self.body {
                    body.finish_node(index, node)?;
                }
                continue;
            }
            for (out_index, id) in node.outputs.iter().enumerate() {
                let value = &self.values[id.0 as usize];
                if let PlanOperation::AttentionQk { geometry, .. }
                | PlanOperation::TextWindowQk { geometry, .. } = &node.operation
                    && out_index == 1
                {
                    unsafe {
                        crate::resident_bindings::certify_causal::launch_unchecked(
                            client,
                            grid(value.elements),
                            threads(),
                            arg(&value.handle, value.elements),
                            arg(&self.status, 1),
                            current.single()?.0.valid_tokens as u32,
                            geometry.query_heads,
                            self.plan.prefix_capacity(),
                        );
                    }
                } else {
                    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
                    match &self.body {
                        ResidentProgramBody::Native(native) => {
                            native.certify(value, &self.status)?
                        }
                        ResidentProgramBody::NativeFull(native) => {
                            // Counts actual launches; retained finite cohorts
                            // run at the terminal original-flight boundary.
                            let n = native.certify(value, &self.status)?;
                            dispatches += n;
                            continue;
                        }
                        ResidentProgramBody::NativeCohort(body) => {
                            dispatches += body.certify(value, &self.status)?;
                            continue;
                        }
                        ResidentProgramBody::Qwen2bComponent(body) => {
                            body.certify(value, &self.status)?
                        }
                        _ => self.certify(client, &value.handle, value.elements),
                    }
                    #[cfg(not(all(feature = "vllm-cuda-qualification", target_os = "linux")))]
                    self.certify(client, &value.handle, value.elements);
                }
                dispatches += 1;
            }
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            match &self.body {
                ResidentProgramBody::NativeFull(body) => body.finish_value_storage_node(index)?,
                ResidentProgramBody::NativeCohort(body) => body.finish_value_storage_node(index)?,
                _ => {}
            }
            #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
            if let ResidentProgramBody::GemmaText(body) = &self.body {
                body.finish_node(index, node)?;
            }
            #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
            if let ResidentProgramBody::QwenSource(body) = &self.body {
                body.finish_node(index, node)?;
            }
        }
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        {
            if let ResidentProgramBody::NativeFull(body) = &self.body {
                dispatches += body.flush_certificates(cancelled)?;
            }
            let carries = match &self.body {
                ResidentProgramBody::Qwen2bComponent(body) => {
                    body.finish_nodes()?;
                    body.recipe().carries().collect::<Vec<_>>()
                }
                ResidentProgramBody::NativeFull(body) => body.canonical47_prefix_carries()?,
                _ => Vec::new(),
            };
            // Same original canonical copies for every untouched domain.
            for slot in carries {
                let binding = current
                    .single()?
                    .1
                    .iter()
                    .find(|b| b.slot == slot)
                    .ok_or_else(|| {
                        Error::State("original finite prefix carry binding absent".into())
                    })?;
                for (prior, candidate) in binding
                    .prior
                    .buffers()
                    .into_iter()
                    .zip(binding.candidate.buffers())
                {
                    prior.check(client, prior.elements)?;
                    candidate.check(client, prior.elements)?;
                    self.copy::<false>(
                        client,
                        &prior.handle,
                        prior.elements,
                        &candidate.handle,
                        candidate.elements,
                        TensorView::vector(0, prior.elements),
                        0,
                    );
                    dispatches += 1;
                }
            }
        }
        #[cfg(all(feature = "qualification", target_os = "linux"))]
        if let ResidentProgramBody::TypedSlice(body) = &self.body {
            body.require_finished()?;
        }
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        if let ResidentProgramBody::GemmaText(body) = &self.body {
            body.require_finished(output)?;
        }
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if let ResidentProgramBody::QwenSource(body) = &self.body {
            body.require_finished()?;
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
        let candidate_rows: Vec<&[StateBinding]> = match &current {
            CurrentExecution::Single { states, .. } => vec![states],
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            CurrentExecution::Cohort { .. } => {
                let ResidentProgramBody::NativeCohort(body) = &self.body else {
                    unreachable!("validated cohort body")
                };
                body.state_bindings().collect()
            }
        };
        for row in candidate_rows {
            for binding in row {
                for value in binding.candidate.buffers() {
                    self.certify(client, &value.handle, value.elements);
                    dispatches += 1;
                }
            }
        }
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        if let ResidentProgramBody::NativeFull(native) = &self.body {
            dispatches += native.enqueue_output(cancelled)?;
        }
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        if let ResidentProgramBody::NativeCohort(body) = &self.body {
            dispatches += body.enqueue_output(cancelled)?;
        }
        let proposals = match &current {
            CurrentExecution::Single { states, .. } => {
                let mut proposals = Vec::new();
                for version in self.plan.proposed_domains() {
                    let binding = states
                        .iter()
                        .find(|binding| binding.slot == version.slot)
                        .expect("validated state binding");
                    proposals.push((*version, binding.candidate.clone()));
                }
                proposals
            }
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            CurrentExecution::Cohort { .. } => Vec::new(),
        };
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        let cohort_proposals = match &current {
            CurrentExecution::Single { .. } => None,
            CurrentExecution::Cohort { .. } => {
                let ResidentProgramBody::NativeCohort(body) = &self.body else {
                    unreachable!("validated cohort body")
                };
                Some(body.proposals()?)
            }
        };
        let mut quant_status = Vec::new();
        let mut embedding_status = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            if self.is_source_window_body() {
                break;
            }
            match &node.operation {
                PlanOperation::Project { tensor } => {
                    let legacy = self.legacy()?;
                    if let Some(l) = legacy.tensors[&tensor.name].layout {
                        quant_status.push((
                            self.legacy()?.node_status[i]
                                .block
                                .as_ref()
                                .expect("allocated block status")
                                .clone(),
                            l.blocks,
                            self.legacy()?.node_status[i]
                                .row
                                .as_ref()
                                .expect("allocated row status")
                                .clone(),
                            l.rows,
                        ));
                    }
                }
                PlanOperation::Embed { tensor, .. } => embedding_status.push((
                    self.legacy()?.node_status[i]
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
            #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
            cohort_proposals,
            status: self.status.clone(),
            quant_status,
            embedding_status,
            dispatches,
            fused_dispatches: {
                #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
                {
                    match &self.body {
                        ResidentProgramBody::Native(native) => native.native_launches(),
                        ResidentProgramBody::NativeFull(native) => native.native_launches(),
                        ResidentProgramBody::NativeCohort(body) => body.native_launches(),
                        ResidentProgramBody::Qwen2bComponent(body) => body.native_launches(),
                        _ => 0,
                    }
                }
                #[cfg(not(all(feature = "vllm-cuda-qualification", target_os = "linux")))]
                {
                    0
                }
            },
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
        execution: &NativeExecution<'_>,
    ) -> Result<usize> {
        let cancelled = execution.cancelled;
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if let ResidentProgramBody::QwenSource(body) = &self.body {
            if cancelled.is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire)) {
                return Err(Error::State("source native call cancelled".into()));
            }
            return body.enqueue(
                index,
                node,
                p,
                bindings,
                execution
                    .qwen
                    .ok_or_else(|| Error::State("source original armed execution absent".into()))?,
            );
        }
        #[cfg(all(feature = "qualification", target_os = "linux"))]
        if let ResidentProgramBody::TypedSlice(body) = &self.body {
            return body.enqueue(index, node, cancelled);
        }
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        if let ResidentProgramBody::GemmaText(body) = &self.body {
            return body.enqueue(
                index,
                node,
                p,
                bindings,
                execution.gemma.ok_or_else(|| {
                    Error::State("Gemma original armed execution context absent".into())
                })?,
                cancelled,
            );
        }
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State("resident native fragment cancelled".into()));
        }
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        if let ResidentProgramBody::NativeFull(native) = &self.body {
            return native.enqueue(index, node, cancelled, execution.phase);
        }
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        if let ResidentProgramBody::Qwen2bComponent(body) = &self.body {
            return body.enqueue(index, node, cancelled, execution.phase);
        }
        #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
        if let ResidentProgramBody::Native(native) = &self.body {
            if matches!(node.operation, PlanOperation::TypedVllmStateCarry) {
                let effect = node
                    .state
                    .ok_or_else(|| Error::State("native carry state effect absent".into()))?;
                let binding = bindings
                    .iter()
                    .find(|b| b.slot == effect.read.slot)
                    .ok_or_else(|| Error::State("native carry original state absent".into()))?;
                let mut count = 0;
                for (prior, candidate) in binding
                    .prior
                    .buffers()
                    .into_iter()
                    .zip(binding.candidate.buffers())
                {
                    prior.check(client, prior.elements)?;
                    candidate.check(client, prior.elements)?;
                    self.copy::<false>(
                        client,
                        &prior.handle,
                        prior.elements,
                        &candidate.handle,
                        candidate.elements,
                        TensorView::vector(0, prior.elements),
                        0,
                    );
                    count += 1;
                }
                return Ok(count);
            }
            return native.enqueue(index, node, cancelled, execution.phase);
        }
        let legacy = self.legacy()?;
        #[cfg(not(all(feature = "vllm-cuda-qualification", target_os = "linux")))]
        let _ = cancelled;
        let input = |i: usize| &self.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.values[node.outputs[i].0 as usize];
        let s = |i: usize| &legacy.scratch[i];
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
                PlanOperation::TypedSourceStateRead { .. }
                | PlanOperation::TypedSourceStateWrite { .. }
                | PlanOperation::TypedTorchGdn { .. }
                | PlanOperation::TypedTorchFullConvolution
                | PlanOperation::TypedTorchAttention { .. }
                | PlanOperation::TextMoe { .. }
                | PlanOperation::TextElementwise(_)
                | PlanOperation::TextRotary { .. }
                | PlanOperation::TextWindowQk { .. }
                | PlanOperation::TypedBulkF32Input { .. }
                | PlanOperation::TypedBulkF32Matmul { .. }
                | PlanOperation::TypedZenSsmOutInput { .. }
                | PlanOperation::TypedZenSsmOutProjection { .. }
                | PlanOperation::TypedGgufPrefixEmbedding { .. }
                | PlanOperation::TypedGgufPrefixRms { .. }
                | PlanOperation::TypedGgufGdnInput { .. }
                | PlanOperation::TypedGgufGdnProjection { .. }
                | PlanOperation::TypedGgufGdnConvSilu { .. }
                | PlanOperation::TypedGgufGdnQk { .. }
                | PlanOperation::TypedEmbed { .. }
                | PlanOperation::TypedProject { .. }
                | PlanOperation::TypedOrdinaryRms { .. }
                | PlanOperation::TypedVllmBf16Embedding { .. }
                | PlanOperation::TypedVllmInputRms { .. }
                | PlanOperation::TypedVllmGdnConv { .. }
                | PlanOperation::TypedVllmGdnRecurrent { .. }
                | PlanOperation::TypedVllmStateCarry
                | PlanOperation::TypedVllm08 { .. }
                | PlanOperation::TypedVllmGdnProjection { .. }
                | PlanOperation::TypedVllmSiluAndMul { .. }
                | PlanOperation::TypedResidualRms { .. }
                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                    "resident typed operation has no qualified lowering".into(),
                )),
                PlanOperation::Embed { tensor, .. } => {
                    let t = &legacy.tensors[&tensor.name];
                    let status = legacy.node_status[index]
                        .block
                        .as_ref()
                        .expect("embedding status");
                    crate::embedding::gather_quantized::launch_unchecked(
                        client,
                        grid(tensor.columns),
                        threads(),
                        arg(&t.packed, t.elements),
                        arg(&legacy.token, 1),
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
                    let t = &legacy.tensors[&tensor.name];
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
                    let dynamic = legacy.projections[index]
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
                        arg(&legacy.constants[gamma.0 as usize], width),
                        arg(s(1), groups * 2),
                        arg(&output(0).handle, n),
                        width,
                        *groups,
                    );
                    Ok(4)
                }
                PlanOperation::Silu { .. } => self.launch_silu(client, node, 1),
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
                                &legacy.exp_tables,
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
                        arg(&legacy.constants[weights.0 as usize], n * 4),
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
                        arg(&legacy.constants[dt.0 as usize], h),
                        arg(&legacy.constants[a.0 as usize], h),
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
                                &legacy.exp_tables,
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
                        arg(&legacy.trig_tables, crate::source_trig::TRIG_TABLES.len()),
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
        let legacy = self.legacy()?;
        let s = |i: usize| &legacy.scratch[i];
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
    fn empty_layout(
        &mut self,
        role: ResidentBufferRole,
        layout: ResidentBufferLayout,
    ) -> Result<DeviceBuffer> {
        use zerogravity_core::device_events::{BufferRole, DeviceAccess};
        let semantic = match role {
            ResidentBufferRole::Value => BufferRole::Activation,
            ResidentBufferRole::Workspace => BufferRole::Scratch,
            _ => {
                return Err(Error::Unsupported(
                    "typed physical allocation requires Value/Workspace role".into(),
                ));
            }
        };
        layout.validate_role(semantic, DeviceAccess::Write)?;
        let handle = self.empty(role, layout.bytes()?)?;
        if layout.kind() == ResidentPhysicalType::Bf16 {
            let range = BufferRange::new(&handle)?;
            layout.validate_region(range.bytes.start, range.bytes.end - range.bytes.start)?;
        }
        Ok(DeviceBuffer {
            handle,
            elements: layout.elements(),
            physical: layout,
            lease: None,
        })
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
    fn only_original_and_opt_in_diagnostic_context_capacities() {
        for capacity in [256, 512] {
            assert!(supported_prefix_capacity(capacity));
        }
        for capacity in [768, 1024] {
            assert_eq!(
                supported_prefix_capacity(capacity),
                cfg!(all(feature = "qualification", feature = "cuda"))
            );
        }
        for capacity in [
            0,
            1,
            255,
            257,
            511,
            513,
            767,
            769,
            1023,
            1025,
            2048,
            4096,
            8192,
            usize::MAX,
        ] {
            assert!(!supported_prefix_capacity(capacity));
        }
    }

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
            for cap in [1, 255, 257] {
                let plan = model.forward_plan(cap, true, false).unwrap();
                assert!(ResidentProgram::preflight(model.as_ref(), &plan).is_err());
            }
            for cap in [768, 1024] {
                let plan = model.forward_plan(cap, true, false).unwrap();
                let result = ResidentProgram::preflight(model.as_ref(), &plan);
                assert_eq!(
                    result.is_ok(),
                    cfg!(all(feature = "qualification", feature = "cuda")),
                    "capacity={cap}: {:?}",
                    result.as_ref().err()
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "resident_program/typed_refusal_tests.rs"]
mod typed_refusal_tests;

#[cfg(test)]
#[path = "resident_program/bulk_f32_tests.rs"]
mod bulk_f32_tests;
#[cfg(test)]
#[path = "resident_program/vllm_native_tests.rs"]
mod vllm_native_tests;

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod bf16_greedy;

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
pub(crate) mod text_positions;

mod silu;
