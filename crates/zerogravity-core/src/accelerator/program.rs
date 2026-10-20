//! Typed forward-program seam. No device runtime or publication owner.
//!
//! The model's ONE mathematical schedule can evaluate host values or record
//! this bounded straight-line plan. Runtime parameters/state generations are
//! bound after compilation; the plan never contains a session or GPU handle.

use super::SourceIdentity;
use crate::quant::QuantKind;
use crate::source::{SourceTensorRef, TensorSource};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod gdn_conv;
pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
mod typed;
pub use typed::{
    ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
    ProjectionContract, SourceParameter,
};

static NEXT_PLAN_BUILDER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorView {
    pub offset: usize,
    pub rows: usize,
    pub width: usize,
    pub row_stride: usize,
}
impl TensorView {
    pub fn vector(offset: usize, elements: usize) -> Self {
        Self {
            offset,
            rows: 1,
            width: elements,
            row_stride: elements,
        }
    }
    pub fn elements(self) -> Result<usize> {
        if self.rows == 0 || self.width == 0 {
            return Err(Error::Malformed("empty mathematical tensor view".into()));
        }
        self.rows
            .checked_mul(self.width)
            .ok_or_else(|| Error::Limit("tensor view count overflow".into()))
    }
    pub fn validate(self, source_elements: usize) -> Result<()> {
        self.elements()?;
        if self.row_stride < self.width {
            return Err(Error::Malformed(
                "overlapping mathematical tensor rows".into(),
            ));
        }
        let end = (self.rows - 1)
            .checked_mul(self.row_stride)
            .and_then(|n| n.checked_add(self.offset))
            .and_then(|n| n.checked_add(self.width))
            .ok_or_else(|| Error::Limit("tensor view range overflow".into()))?;
        if end > source_elements {
            return Err(Error::Malformed(
                "mathematical tensor view exceeds input".into(),
            ));
        }
        Ok(())
    }
    pub fn is_contiguous(self) -> bool {
        self.rows == 1 || self.row_stride == self.width
    }
}

/// Raw matrix header order; packed output rows are never transposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixAxes {
    /// GGUF: [input width, output rows].
    InputOutput,
    /// Hugging Face: [output rows, input width].
    OutputInput,
}
impl MatrixAxes {
    fn shape(self, columns: usize, rows: usize) -> [u64; 2] {
        match self {
            Self::InputOutput => [columns as u64, rows as u64],
            Self::OutputInput => [rows as u64, columns as u64],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceTensor {
    pub source: SourceIdentity,
    pub name: String,
    pub kind: QuantKind,
    pub axes: MatrixAxes,
    pub columns: usize,
    pub rows: usize,
}
impl SourceTensor {
    /// Bind actual source metadata/bytes without copying or changing raw axes.
    /// This descriptor grants no operator or trained-model eligibility.
    pub fn from_source(source: &dyn TensorSource, name: &str, axes: MatrixAxes) -> Result<Self> {
        let identity = source.source_identity();
        if !identity.is_bound() || name.is_empty() {
            return Err(Error::State("unbound forward source tensor".into()));
        }
        let actual = source.tensor(name)?;
        actual.validate()?;
        let [first, second] = actual.shape else {
            return Err(Error::Malformed("forward source is not a matrix".into()));
        };
        let (columns, rows) = match axes {
            MatrixAxes::InputOutput => (*first, *second),
            MatrixAxes::OutputInput => (*second, *first),
        };
        let tensor = Self {
            source: identity,
            name: name.into(),
            kind: actual.kind,
            axes,
            columns: usize::try_from(columns)
                .map_err(|_| Error::Limit("forward matrix columns overflow".into()))?,
            rows: usize::try_from(rows)
                .map_err(|_| Error::Limit("forward matrix rows overflow".into()))?,
        };
        tensor.validate_ref(identity, actual)?;
        Ok(tensor)
    }

    fn packed_bytes(&self) -> Result<u64> {
        if !self.source.is_bound() || self.name.is_empty() || self.columns == 0 || self.rows == 0 {
            return Err(Error::State("invalid forward source descriptor".into()));
        }
        self.kind
            .byte_len(self.columns as u64)?
            .checked_mul(self.rows as u64)
            .ok_or_else(|| Error::Limit("forward matrix byte size overflow".into()))
    }

    fn validate_ref(&self, identity: SourceIdentity, actual: SourceTensorRef<'_>) -> Result<()> {
        let bytes = self.packed_bytes()?;
        actual.validate()?;
        if self.source != identity
            || actual.kind != self.kind
            || actual.shape != self.axes.shape(self.columns, self.rows)
            || actual.packed_bytes.len() as u64 != bytes
        {
            return Err(Error::State(
                "forward tensor differs from immutable source".into(),
            ));
        }
        Ok(())
    }

    /// Revalidate a plan descriptor against the same immutable source owner.
    pub fn validate_source(&self, source: &dyn TensorSource) -> Result<()> {
        self.packed_bytes()?;
        let identity = source.source_identity();
        if self.source != identity {
            return Err(Error::State("forward tensor source mismatch".into()));
        }
        self.validate_ref(identity, source.tensor(&self.name)?)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SourceVector<'a> {
    pub source: SourceIdentity,
    pub name: &'a str,
    pub values: &'a [f32],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConvDomain(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecurrentDomain(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KvDomain(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateSlot {
    Conv(ConvDomain),
    Recurrent(RecurrentDomain),
    Kv(KvDomain),
}

#[derive(Clone, Copy, Debug)]
pub struct ForwardInputs<I> {
    pub token: I,
    pub position: I,
    pub valid_tokens: I,
}
#[derive(Clone, Copy, Debug)]
pub struct AttentionGeometry {
    pub width: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
}
impl AttentionGeometry {
    fn validate(self) -> Result<(usize, usize)> {
        if self.width == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
        {
            return Err(Error::Malformed(
                "invalid forward attention geometry".into(),
            ));
        }
        let q = self
            .width
            .checked_mul(self.query_heads)
            .ok_or_else(|| Error::Limit("forward query count overflow".into()))?;
        let kv = self
            .width
            .checked_mul(self.kv_heads)
            .ok_or_else(|| Error::Limit("forward KV count overflow".into()))?;
        Ok((q, kv))
    }
}
#[derive(Clone, Copy, Debug)]
pub struct RopeGeometry {
    pub attention: AttentionGeometry,
    pub rotate: usize,
    pub theta_scale: f32,
}
pub struct CoefficientValues<V> {
    pub packed: V,
    pub softplus: V,
    pub gate_raw: V,
}
pub struct RopeValues<V> {
    pub query: V,
    pub key: V,
    pub value: V,
}
pub struct QkValues<V> {
    pub dots: V,
    pub scores: V,
}

/// Value operations plus typed PRIVATE state effects. Layer scheduling lives
/// only in qwen35::forward, never in a backend implementation of this trait.
#[allow(clippy::too_many_arguments)] // Source closed-op inputs stay explicit.
pub trait MathematicalBackend {
    type Value: Clone;
    type Index: Copy;
    fn elements(&self, value: &Self::Value) -> usize;
    fn precision(&self, _value: &Self::Value) -> ActivationPrecision {
        ActivationPrecision::F32
    }
    fn embed_profiled(
        &mut self,
        tensor: SourceTensor,
        token: Self::Index,
        contract: EmbeddingContract,
    ) -> Result<Self::Value> {
        match contract {
            EmbeddingContract::Source => self.embed(tensor, token),
            _ => Err(Error::Unsupported(
                "backend does not implement explicit embedding contract".into(),
            )),
        }
    }
    fn project_profiled(
        &mut self,
        tensor: SourceTensor,
        input: &Self::Value,
        contract: ProjectionContract,
    ) -> Result<Self::Value> {
        match contract {
            ProjectionContract::Source => self.project(tensor, input),
            _ => Err(Error::Unsupported(
                "backend does not implement explicit projection contract".into(),
            )),
        }
    }
    fn ordinary_rms_profiled(
        &mut self,
        _parameter: SourceParameter,
        _input: &Self::Value,
        _contract: OrdinaryRmsContract,
    ) -> Result<Self::Value> {
        Err(Error::Unsupported(
            "backend does not implement explicit ordinary RMS contract".into(),
        ))
    }
    /// Pure original convolution window; no state owner or implicit fallback.
    fn causal_conv_silu_profiled(
        &mut self,
        _parameter: SourceParameter,
        _input: &Self::Value,
        _history: &Self::Value,
        _contract: CausalConvSiluContract,
    ) -> Result<CausalConvValues<Self::Value>> {
        Err(Error::Unsupported(
            "backend does not implement explicit causal convolution SiLU contract".into(),
        ))
    }
    fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
    fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
    fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
    fn contiguous(&mut self, input: &Self::Value) -> Result<Self::Value>;
    fn rms(
        &mut self,
        name: &str,
        input: &Self::Value,
        gamma: SourceVector<'_>,
        eps: f32,
    ) -> Result<Self::Value>;
    fn grouped_rms(
        &mut self,
        name: &str,
        input: &Self::Value,
        gamma: SourceVector<'_>,
        groups: usize,
        eps: f32,
    ) -> Result<Self::Value>;
    fn silu(
        &mut self,
        name: &str,
        input: &Self::Value,
        multiplier: Option<&Self::Value>,
    ) -> Result<Self::Value>;
    fn add(&mut self, name: &str, left: &Self::Value, right: &Self::Value) -> Result<Self::Value>;
    fn sigmoid_mul(
        &mut self,
        name: &str,
        value: &Self::Value,
        gate: &Self::Value,
    ) -> Result<Self::Value>;
    fn conv_step(
        &mut self,
        name: &str,
        domain: ConvDomain,
        input: &Self::Value,
        weights: SourceVector<'_>,
    ) -> Result<Self::Value>;
    fn coefficients(
        &mut self,
        name: &str,
        alpha: &Self::Value,
        beta: &Self::Value,
        dt: SourceVector<'_>,
        a: SourceVector<'_>,
        scale: f32,
    ) -> Result<CoefficientValues<Self::Value>>;
    fn recurrent_step(
        &mut self,
        name: &str,
        domain: RecurrentDomain,
        q: &Self::Value,
        k: &Self::Value,
        v: &Self::Value,
        coefficients: &Self::Value,
        width: usize,
    ) -> Result<Self::Value>;
    fn validate_kv_prefix(&mut self, domain: KvDomain, position: Self::Index) -> Result<()>;
    fn rope(
        &mut self,
        name: &str,
        q: &Self::Value,
        k: &Self::Value,
        v: &Self::Value,
        position: Self::Index,
        geometry: RopeGeometry,
    ) -> Result<RopeValues<Self::Value>>;
    fn append_kv(&mut self, domain: KvDomain, k: &Self::Value, v: &Self::Value) -> Result<()>;
    fn attention_qk(
        &mut self,
        name: &str,
        domain: KvDomain,
        query: &Self::Value,
        valid_tokens: Self::Index,
        geometry: AttentionGeometry,
        scale: f32,
    ) -> Result<QkValues<Self::Value>>;
    fn attention_softmax(
        &mut self,
        name: &str,
        scores: &Self::Value,
        heads: usize,
        valid_tokens: Self::Index,
    ) -> Result<Self::Value>;
    fn attention_value(
        &mut self,
        name: &str,
        domain: KvDomain,
        probabilities: &Self::Value,
        valid_tokens: Self::Index,
        geometry: AttentionGeometry,
    ) -> Result<Self::Value>;
    fn assert_finite(&mut self, layer: usize, value: &Self::Value) -> Result<()>;
    fn observe(&mut self, layer: usize, stage: &'static str, value: &Self::Value) -> Result<()>;
    fn observe_state(&mut self, layer: usize, stage: &'static str, domain: StateSlot)
    -> Result<()>;
    /// Advance PRIVATE proposed metadata. This never publishes a session root.
    fn advance_frontier(&mut self) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterSlot {
    Token,
    Position,
    ValidTokens,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterRef(pub ParameterSlot);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueShape {
    Vector(usize),
    CausalRows { heads: usize, capacity: usize },
}
impl ValueShape {
    pub fn capacity_elements(self) -> Result<usize> {
        match self {
            Self::Vector(n) if n > 0 => Ok(n),
            Self::CausalRows { heads, capacity } if heads > 0 && capacity > 0 => heads
                .checked_mul(capacity)
                .ok_or_else(|| Error::Limit("causal value capacity overflow".into())),
            _ => Err(Error::Malformed("empty forward-plan value".into())),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct PlanValue {
    pub id: ValueId,
    pub shape: ValueShape,
    owner: u64,
    elements: usize,
    contiguous: bool,
    precision: ActivationPrecision,
}
impl PlanValue {
    pub fn precision(self) -> ActivationPrecision {
        self.precision
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticVectorId(pub u32);
#[derive(Clone, Debug)]
pub struct StaticVector {
    pub source: SourceIdentity,
    pub name: String,
    pub values: Arc<[f32]>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateVersion {
    pub slot: StateSlot,
    pub ordinal: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct StateEffect {
    pub read: StateVersion,
    pub write: StateVersion,
}

/// Per-op source semantics, NOT backend launches or a second Qwen schedule.
/// Lowering may add explicit Pack/Copy instructions for qualified kernel ABIs.
#[derive(Clone, Debug)]
pub enum PlanOperation {
    TypedEmbed {
        tensor: SourceTensor,
        token: ParameterRef,
        contract: EmbeddingContract,
    },
    TypedProject {
        tensor: SourceTensor,
        contract: ProjectionContract,
    },
    TypedCausalConvSilu {
        parameter: SourceParameter,
        contract: CausalConvSiluContract,
    },
    TypedOrdinaryRms {
        parameter: SourceParameter,
        contract: OrdinaryRmsContract,
    },
    Embed {
        tensor: SourceTensor,
        token: ParameterRef,
    },
    Project {
        tensor: SourceTensor,
    },
    View(TensorView),
    Contiguous,
    Rms {
        gamma: StaticVectorId,
        eps: f32,
        groups: usize,
    },
    Silu {
        multiplied: bool,
    },
    Add,
    SigmoidGate,
    Conv {
        weights: StaticVectorId,
    },
    Coefficients {
        dt: StaticVectorId,
        a: StaticVectorId,
        scale: f32,
    },
    Recurrent {
        width: usize,
    },
    KvPrefixGuard {
        state: StateVersion,
        position: ParameterRef,
    },
    Rope {
        geometry: RopeGeometry,
        position: ParameterRef,
    },
    AppendKv,
    AttentionQk {
        state: StateVersion,
        geometry: AttentionGeometry,
        valid_tokens: ParameterRef,
        scale: f32,
    },
    AttentionSoftmax {
        heads: usize,
        valid_tokens: ParameterRef,
    },
    AttentionValue {
        state: StateVersion,
        geometry: AttentionGeometry,
        valid_tokens: ParameterRef,
        padding: usize,
    },
    CheckFinite {
        layer: usize,
    },
    Observe {
        layer: usize,
        stage: &'static str,
    },
    ObserveState {
        layer: usize,
        stage: &'static str,
        state: StateVersion,
    },
}
#[derive(Clone, Debug)]
pub struct PlanNode {
    pub name: String,
    pub operation: PlanOperation,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub state: Option<StateEffect>,
}
#[derive(Clone, Debug)]
pub struct ForwardPlan {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    operator_slice: bool,
    precisions: Vec<ActivationPrecision>,
    prefix_capacity: usize,
    nodes: Vec<PlanNode>,
    values: Vec<ValueShape>,
    constants: Vec<StaticVector>,
    required_domains: Vec<StateSlot>,
    proposed_domains: Vec<StateVersion>,
    logits: Option<ValueId>,
    frontier_delta: usize,
    state_node_count: usize,
}
/// State-free fragment recorded by the same builder; no token/root advance.
#[derive(Clone, Debug)]
pub struct RecordedOperatorSlice {
    plan: ForwardPlan,
    output: ValueId,
}
impl RecordedOperatorSlice {
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn output(&self) -> ValueId {
        self.output
    }
}
impl ForwardPlan {
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn is_operator_slice(&self) -> bool {
        self.operator_slice
    }
    pub fn precisions(&self) -> &[ActivationPrecision] {
        &self.precisions
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    pub fn prefix_capacity(&self) -> usize {
        self.prefix_capacity
    }
    pub fn nodes(&self) -> &[PlanNode] {
        &self.nodes
    }
    /// Complete state transition, including its numerical checks and observations.
    /// The schedule's private frontier advance closes this prefix. Any remaining
    /// nodes compute optional outputs and cannot access state.
    pub fn state_nodes(&self) -> &[PlanNode] {
        &self.nodes[..self.state_node_count]
    }
    pub fn values(&self) -> &[ValueShape] {
        &self.values
    }
    pub fn constants(&self) -> &[StaticVector] {
        &self.constants
    }
    pub fn required_domains(&self) -> &[StateSlot] {
        &self.required_domains
    }
    pub fn proposed_domains(&self) -> &[StateVersion] {
        &self.proposed_domains
    }
    pub fn logits(&self) -> Option<ValueId> {
        self.logits
    }
    pub fn frontier_delta(&self) -> usize {
        self.frontier_delta
    }
}

pub struct PlanBuilder {
    owner: u64,
    profile: ProgramNumericalProfile,
    operator_slice: bool,
    precisions: Vec<ActivationPrecision>,
    source: SourceIdentity,
    prefix_capacity: usize,
    nodes: Vec<PlanNode>,
    values: Vec<ValueShape>,
    constants: Vec<StaticVector>,
    constant_names: BTreeMap<String, StaticVectorId>,
    domains: BTreeMap<StateSlot, StateVersion>,
    required: BTreeSet<StateSlot>,
    capture: bool,
    advanced: Option<usize>,
}
impl PlanBuilder {
    pub fn new(
        source: SourceIdentity,
        prefix_capacity: usize,
        required: Vec<StateSlot>,
        capture: bool,
    ) -> Result<Self> {
        Self::new_inner(
            source,
            prefix_capacity,
            required,
            capture,
            ProgramNumericalProfile::GgufSourceV1,
            false,
        )
    }
    pub fn new_with_profile(
        source: SourceIdentity,
        prefix_capacity: usize,
        required: Vec<StateSlot>,
        capture: bool,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        Self::new_inner(source, prefix_capacity, required, capture, profile, false)
    }
    pub fn new_operator_slice(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        Self::new_inner(source, 1, Vec::new(), false, profile, true)
    }
    fn new_inner(
        source: SourceIdentity,
        prefix_capacity: usize,
        required: Vec<StateSlot>,
        capture: bool,
        profile: ProgramNumericalProfile,
        operator_slice: bool,
    ) -> Result<Self> {
        let expected_count = required.len();
        let required: BTreeSet<_> = required.into_iter().collect();
        if !source.is_bound()
            || prefix_capacity == 0
            || required.is_empty() != operator_slice
            || required.len() != expected_count
        {
            return Err(Error::Malformed("invalid source-bound forward plan".into()));
        }
        let owner = NEXT_PLAN_BUILDER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit("forward plan builder identity exhausted".into()))?;
        Ok(Self {
            owner,
            profile,
            operator_slice,
            precisions: Vec::new(),
            source,
            prefix_capacity,
            nodes: Vec::new(),
            values: Vec::new(),
            constants: Vec::new(),
            constant_names: BTreeMap::new(),
            domains: BTreeMap::new(),
            required,
            capture,
            advanced: None,
        })
    }
    pub fn inputs() -> ForwardInputs<ParameterRef> {
        ForwardInputs {
            token: ParameterRef(ParameterSlot::Token),
            position: ParameterRef(ParameterSlot::Position),
            valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
        }
    }
    fn source_tensor(&self, tensor: &SourceTensor) -> Result<()> {
        tensor.packed_bytes()?;
        if tensor.source != self.source {
            return Err(Error::State(
                "forward plan changed source tensor identity".into(),
            ));
        }
        Ok(())
    }
    fn constant(&mut self, vector: SourceVector<'_>) -> Result<StaticVectorId> {
        if self.profile != ProgramNumericalProfile::GgufSourceV1 {
            return Err(Error::Unsupported(
                "profiled operators require raw source parameter descriptors".into(),
            ));
        }
        if vector.source != self.source
            || vector.name.is_empty()
            || vector.values.is_empty()
            || !vector.values.iter().all(|v| v.is_finite())
        {
            return Err(Error::Malformed(
                "invalid forward-plan static vector".into(),
            ));
        }
        if let Some(id) = self.constant_names.get(vector.name).copied() {
            let old = &self.constants[id.0 as usize];
            if old.values.len() != vector.values.len()
                || old
                    .values
                    .iter()
                    .zip(vector.values)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(Error::State(
                    "source vector role changed within one plan".into(),
                ));
            }
            return Ok(id);
        }
        let id = StaticVectorId(
            u32::try_from(self.constants.len())
                .map_err(|_| Error::Limit("static vector slot overflow".into()))?,
        );
        self.constants.push(StaticVector {
            source: vector.source,
            name: vector.name.into(),
            values: vector.values.into(),
        });
        self.constant_names.insert(vector.name.into(), id);
        Ok(id)
    }
    fn result_precision(
        &self,
        operation: &PlanOperation,
        inputs: &[PlanValue],
    ) -> Result<ActivationPrecision> {
        match operation {
            PlanOperation::TypedEmbed {
                tensor,
                contract: EmbeddingContract::Bf16OutputInput,
                ..
            } => {
                typed::validate_bf16_matrix(tensor)?;
                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1 {
                    return Err(Error::Unsupported(
                        "typed embedding numerical profile".into(),
                    ));
                }
                Ok(ActivationPrecision::Bf16WidenedF32)
            }
            PlanOperation::TypedProject {
                tensor,
                contract: ProjectionContract::TorchCpuBf16Avx2,
            } => {
                typed::validate_bf16_matrix(tensor)?;
                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
                    || inputs.len() != 1
                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
                {
                    return Err(Error::Unsupported(
                        "typed projection numerical profile/precision".into(),
                    ));
                }
                Ok(ActivationPrecision::Bf16WidenedF32)
            }
            PlanOperation::TypedOrdinaryRms {
                parameter,
                contract,
            } => {
                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
                    || parameter.source != self.source
                    || inputs.len() != 1
                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
                {
                    return Err(Error::Unsupported(
                        "ordinary RMS numerical profile/source/precision".into(),
                    ));
                }
                contract.validate(parameter, inputs[0].elements)?;
                Ok(ActivationPrecision::Bf16WidenedF32)
            }
            PlanOperation::TypedCausalConvSilu {
                parameter,
                contract,
            } => gdn_conv::result_precision(self, parameter, *contract, inputs),
            PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                .first()
                .map(|value| value.precision)
                .ok_or_else(|| Error::State("missing typed view input".into())),
            PlanOperation::Observe { .. } | PlanOperation::CheckFinite { .. } => {
                Ok(ActivationPrecision::F32)
            }
            _ if self.profile == ProgramNumericalProfile::GgufSourceV1
                && inputs
                    .iter()
                    .all(|input| input.precision == ActivationPrecision::F32)
                && !matches!(
                    operation,
                    PlanOperation::TypedEmbed { .. }
                        | PlanOperation::TypedProject { .. }
                        | PlanOperation::TypedOrdinaryRms { .. }
                        | PlanOperation::TypedCausalConvSilu { .. }
                ) =>
            {
                Ok(ActivationPrecision::F32)
            }
            _ => Err(Error::Unsupported(
                "operator lacks explicit numerical/activation contract".into(),
            )),
        }
    }
    fn record(
        &mut self,
        name: &str,
        operation: PlanOperation,
        inputs: &[PlanValue],
        shapes: &[ValueShape],
        state: Option<StateEffect>,
    ) -> Result<Vec<PlanValue>> {
        for input in inputs {
            if input.owner != self.owner
                || self.values.get(input.id.0 as usize) != Some(&input.shape)
                || input.elements != input.shape.capacity_elements()?
                || self.precisions.get(input.id.0 as usize) != Some(&input.precision)
            {
                return Err(Error::State(
                    "foreign or malformed forward-plan value".into(),
                ));
            }
            if !input.contiguous && !matches!(&operation, PlanOperation::Contiguous) {
                return Err(Error::State(
                    "plan operator requires explicit contiguous input".into(),
                ));
            }
        }
        let precision = self.result_precision(&operation, inputs)?;
        let elements: Vec<_> = shapes
            .iter()
            .map(|shape| shape.capacity_elements())
            .collect::<Result<_>>()?;
        let next_count = self
            .values
            .len()
            .checked_add(shapes.len())
            .ok_or_else(|| Error::Limit("forward value count overflow".into()))?;
        if next_count > u32::MAX as usize {
            return Err(Error::Limit("forward value slot overflow".into()));
        }
        if let Some(effect) = state
            && (self.current(effect.read.slot)? != effect.read
                || effect.write.slot != effect.read.slot
                || effect.read.ordinal.checked_add(1) != Some(effect.write.ordinal))
        {
            return Err(Error::State("stale forward-plan state effect".into()));
        }
        let mut outputs = Vec::with_capacity(shapes.len());
        for (&shape, elements) in shapes.iter().zip(elements) {
            let id = ValueId(self.values.len() as u32);
            self.values.push(shape);
            self.precisions.push(precision);
            outputs.push(PlanValue {
                id,
                shape,
                owner: self.owner,
                elements,
                contiguous: true,
                precision,
            });
        }
        self.nodes.push(PlanNode {
            name: name.into(),
            operation,
            inputs: inputs.iter().map(|v| v.id).collect(),
            outputs: outputs.iter().map(|v| v.id).collect(),
            state,
        });
        if let Some(effect) = state {
            self.domains.insert(effect.write.slot, effect.write);
        }
        Ok(outputs)
    }
    fn one(
        &mut self,
        name: &str,
        operation: PlanOperation,
        inputs: &[PlanValue],
        shape: ValueShape,
        state: Option<StateEffect>,
    ) -> Result<PlanValue> {
        Ok(self
            .record(name, operation, inputs, &[shape], state)?
            .remove(0))
    }
    fn current(&self, slot: StateSlot) -> Result<StateVersion> {
        if self.advanced.is_some() {
            return Err(Error::State(
                "state access after forward-plan frontier advance".into(),
            ));
        }
        if !self.required.contains(&slot) {
            return Err(Error::State("unregistered forward state domain".into()));
        }
        Ok(self
            .domains
            .get(&slot)
            .copied()
            .unwrap_or(StateVersion { slot, ordinal: 0 }))
    }
    fn effect(&self, slot: StateSlot) -> Result<StateEffect> {
        let read = self.current(slot)?;
        if read.ordinal != 0 {
            return Err(Error::State(
                "state domain written twice in one forward step".into(),
            ));
        }
        let write = StateVersion {
            slot,
            ordinal: read
                .ordinal
                .checked_add(1)
                .ok_or_else(|| Error::Limit("state proposal ordinal overflow".into()))?,
        };
        Ok(StateEffect { read, write })
    }
    pub fn finish_operator_slice(self, output: PlanValue) -> Result<RecordedOperatorSlice> {
        if !self.operator_slice
            || !self.required.is_empty()
            || !self.domains.is_empty()
            || self.advanced.is_some()
            || output.owner != self.owner
            || self.values.get(output.id.0 as usize) != Some(&output.shape)
            || self.precisions.get(output.id.0 as usize) != Some(&output.precision)
        {
            return Err(Error::State("invalid state-free operator slice".into()));
        }
        Ok(RecordedOperatorSlice {
            output: output.id,
            plan: ForwardPlan {
                source: self.source,
                profile: self.profile,
                operator_slice: true,
                precisions: self.precisions,
                prefix_capacity: self.prefix_capacity,
                nodes: self.nodes,
                values: self.values,
                constants: self.constants,
                required_domains: Vec::new(),
                proposed_domains: Vec::new(),
                logits: None,
                frontier_delta: 0,
                state_node_count: 0,
            },
        })
    }
    pub fn finish(self, logits: Option<PlanValue>) -> Result<ForwardPlan> {
        if self.operator_slice
            || self.advanced.is_none()
            || self.domains.len() != self.required.len()
            || self
                .required
                .iter()
                .any(|slot| self.domains.get(slot).is_none_or(|v| v.ordinal != 1))
        {
            return Err(Error::State(
                "incomplete forward-plan state proposal".into(),
            ));
        }
        if let Some(value) = logits
            && (value.owner != self.owner
                || self.values.get(value.id.0 as usize) != Some(&value.shape)
                || self.precisions.get(value.id.0 as usize) != Some(&value.precision))
        {
            return Err(Error::State("foreign forward-plan logits".into()));
        }
        Ok(ForwardPlan {
            source: self.source,
            profile: self.profile,
            operator_slice: false,
            precisions: self.precisions,
            prefix_capacity: self.prefix_capacity,
            nodes: self.nodes,
            values: self.values,
            constants: self.constants,
            required_domains: self.required.into_iter().collect(),
            proposed_domains: self.domains.into_values().collect(),
            logits: logits.map(|v| v.id),
            frontier_delta: 1,
            state_node_count: self.advanced.expect("checked frontier advance"),
        })
    }
}

#[allow(clippy::too_many_arguments)] // Source closed-op inputs stay explicit.
impl MathematicalBackend for PlanBuilder {
    type Value = PlanValue;
    type Index = ParameterRef;
    fn elements(&self, value: &PlanValue) -> usize {
        value.elements
    }
    fn precision(&self, value: &PlanValue) -> ActivationPrecision {
        value.precision
    }
    fn embed_profiled(
        &mut self,
        tensor: SourceTensor,
        token: ParameterRef,
        contract: EmbeddingContract,
    ) -> Result<PlanValue> {
        if contract == EmbeddingContract::Source {
            return self.embed(tensor, token);
        }
        self.source_tensor(&tensor)?;
        if token.0 != ParameterSlot::Token {
            return Err(Error::Malformed("typed embedding token slot".into()));
        }
        let columns = tensor.columns;
        self.one(
            "embedding",
            PlanOperation::TypedEmbed {
                tensor,
                token,
                contract,
            },
            &[],
            ValueShape::Vector(columns),
            None,
        )
    }
    fn project_profiled(
        &mut self,
        tensor: SourceTensor,
        input: &PlanValue,
        contract: ProjectionContract,
    ) -> Result<PlanValue> {
        if contract == ProjectionContract::Source {
            return self.project(tensor, input);
        }
        self.source_tensor(&tensor)?;
        if input.elements != tensor.columns {
            return Err(Error::Malformed("typed projection width".into()));
        }
        let rows = tensor.rows;
        let name = tensor.name.clone();
        self.one(
            &name,
            PlanOperation::TypedProject { tensor, contract },
            &[*input],
            ValueShape::Vector(rows),
            None,
        )
    }
    fn ordinary_rms_profiled(
        &mut self,
        parameter: SourceParameter,
        input: &PlanValue,
        contract: OrdinaryRmsContract,
    ) -> Result<PlanValue> {
        let name = parameter.name.clone();
        self.one(
            &name,
            PlanOperation::TypedOrdinaryRms {
                parameter,
                contract,
            },
            &[*input],
            input.shape,
            None,
        )
    }
    fn causal_conv_silu_profiled(
        &mut self,
        parameter: SourceParameter,
        input: &PlanValue,
        history: &PlanValue,
        contract: CausalConvSiluContract,
    ) -> Result<CausalConvValues<PlanValue>> {
        gdn_conv::record(self, parameter, input, history, contract)
    }
    fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
        self.source_tensor(&tensor)?;
        let columns = tensor.columns;
        self.one(
            "embedding",
            PlanOperation::Embed { tensor, token },
            &[],
            ValueShape::Vector(columns),
            None,
        )
    }
    fn project(&mut self, tensor: SourceTensor, input: &PlanValue) -> Result<PlanValue> {
        self.source_tensor(&tensor)?;
        if input.elements != tensor.columns {
            return Err(Error::Malformed("plan projection width mismatch".into()));
        }
        let rows = tensor.rows;
        let name = tensor.name.clone();
        self.one(
            &name,
            PlanOperation::Project { tensor },
            &[*input],
            ValueShape::Vector(rows),
            None,
        )
    }
    fn view(&mut self, input: &PlanValue, view: TensorView) -> Result<PlanValue> {
        if !matches!(input.shape, ValueShape::Vector(_)) {
            return Err(Error::Unsupported(
                "causal rows require a bounded runtime view".into(),
            ));
        }
        if !input.contiguous {
            return Err(Error::State(
                "nested strided plan view requires packing".into(),
            ));
        }
        view.validate(input.elements)?;
        let mut output = self.one(
            "view",
            PlanOperation::View(view),
            &[*input],
            ValueShape::Vector(view.elements()?),
            None,
        )?;
        output.contiguous = view.is_contiguous();
        Ok(output)
    }
    fn contiguous(&mut self, input: &PlanValue) -> Result<PlanValue> {
        self.one(
            "contiguous",
            PlanOperation::Contiguous,
            &[*input],
            input.shape,
            None,
        )
    }
    fn rms(
        &mut self,
        name: &str,
        input: &PlanValue,
        gamma: SourceVector<'_>,
        eps: f32,
    ) -> Result<PlanValue> {
        self.grouped_rms(name, input, gamma, 1, eps)
    }
    fn grouped_rms(
        &mut self,
        name: &str,
        input: &PlanValue,
        gamma: SourceVector<'_>,
        groups: usize,
        eps: f32,
    ) -> Result<PlanValue> {
        if groups == 0
            || gamma.values.len().checked_mul(groups) != Some(input.elements)
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(Error::Malformed(
                "plan normalization geometry invalid".into(),
            ));
        }
        let gamma = self.constant(gamma)?;
        self.one(
            name,
            PlanOperation::Rms { gamma, eps, groups },
            &[*input],
            input.shape,
            None,
        )
    }
    fn silu(
        &mut self,
        name: &str,
        input: &PlanValue,
        multiplier: Option<&PlanValue>,
    ) -> Result<PlanValue> {
        let mut inputs = vec![*input];
        if let Some(m) = multiplier {
            if m.shape != input.shape {
                return Err(Error::Malformed("plan SiLU multiplier shape".into()));
            }
            inputs.push(*m);
        }
        self.one(
            name,
            PlanOperation::Silu {
                multiplied: multiplier.is_some(),
            },
            &inputs,
            input.shape,
            None,
        )
    }
    fn add(&mut self, name: &str, left: &PlanValue, right: &PlanValue) -> Result<PlanValue> {
        if left.shape != right.shape {
            return Err(Error::Malformed("plan binary shape".into()));
        }
        self.one(name, PlanOperation::Add, &[*left, *right], left.shape, None)
    }
    fn sigmoid_mul(
        &mut self,
        name: &str,
        value: &PlanValue,
        gate: &PlanValue,
    ) -> Result<PlanValue> {
        if value.shape != gate.shape {
            return Err(Error::Malformed("plan gate shape".into()));
        }
        self.one(
            name,
            PlanOperation::SigmoidGate,
            &[*value, *gate],
            value.shape,
            None,
        )
    }
    fn conv_step(
        &mut self,
        name: &str,
        domain: ConvDomain,
        input: &PlanValue,
        weights: SourceVector<'_>,
    ) -> Result<PlanValue> {
        if weights.values.len()
            != input
                .elements
                .checked_mul(4)
                .ok_or_else(|| Error::Limit("plan convolution shape overflow".into()))?
        {
            return Err(Error::Malformed("plan convolution weights".into()));
        }
        let weights = self.constant(weights)?;
        let effect = self.effect(StateSlot::Conv(domain))?;
        self.one(
            name,
            PlanOperation::Conv { weights },
            &[*input],
            input.shape,
            Some(effect),
        )
    }
    fn coefficients(
        &mut self,
        name: &str,
        alpha: &PlanValue,
        beta: &PlanValue,
        dt: SourceVector<'_>,
        a: SourceVector<'_>,
        scale: f32,
    ) -> Result<CoefficientValues<PlanValue>> {
        if alpha.shape != beta.shape
            || alpha.elements != dt.values.len()
            || alpha.elements != a.values.len()
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(Error::Malformed("plan GDN coefficient geometry".into()));
        }
        let heads = alpha.elements;
        let dt = self.constant(dt)?;
        let a = self.constant(a)?;
        let packed = heads
            .checked_mul(3)
            .ok_or_else(|| Error::Limit("plan coefficient count overflow".into()))?;
        let output = self.record(
            name,
            PlanOperation::Coefficients { dt, a, scale },
            &[*alpha, *beta],
            &[
                ValueShape::Vector(packed),
                ValueShape::Vector(heads),
                ValueShape::Vector(heads),
            ],
            None,
        )?;
        Ok(CoefficientValues {
            packed: output[0],
            softplus: output[1],
            gate_raw: output[2],
        })
    }
    fn recurrent_step(
        &mut self,
        name: &str,
        domain: RecurrentDomain,
        q: &PlanValue,
        k: &PlanValue,
        v: &PlanValue,
        coefficients: &PlanValue,
        width: usize,
    ) -> Result<PlanValue> {
        if width == 0
            || q.shape != k.shape
            || q.shape != v.shape
            || !q.elements.is_multiple_of(width)
            || (q.elements / width).checked_mul(3) != Some(coefficients.elements)
        {
            return Err(Error::Malformed("plan recurrent geometry".into()));
        }
        let effect = self.effect(StateSlot::Recurrent(domain))?;
        self.one(
            name,
            PlanOperation::Recurrent { width },
            &[*q, *k, *v, *coefficients],
            q.shape,
            Some(effect),
        )
    }
    fn validate_kv_prefix(&mut self, domain: KvDomain, position: ParameterRef) -> Result<()> {
        let state = self.current(StateSlot::Kv(domain))?;
        if state.ordinal != 0 {
            return Err(Error::State("KV prefix guard must precede append".into()));
        }
        self.record(
            "kv-prefix",
            PlanOperation::KvPrefixGuard { state, position },
            &[],
            &[],
            None,
        )?;
        Ok(())
    }
    fn rope(
        &mut self,
        name: &str,
        q: &PlanValue,
        k: &PlanValue,
        v: &PlanValue,
        position: ParameterRef,
        geometry: RopeGeometry,
    ) -> Result<RopeValues<PlanValue>> {
        let (queries, keys) = geometry.attention.validate()?;
        if q.elements != queries
            || k.elements != keys
            || v.elements != keys
            || geometry.rotate == 0
            || geometry.rotate > geometry.attention.width
            || !geometry.rotate.is_multiple_of(2)
            || !geometry.theta_scale.is_finite()
            || geometry.theta_scale <= 0.0
            || geometry.theta_scale > 1.0
        {
            return Err(Error::Malformed("plan source RoPE geometry".into()));
        }
        let output = self.record(
            name,
            PlanOperation::Rope { geometry, position },
            &[*q, *k, *v],
            &[q.shape, k.shape, v.shape],
            None,
        )?;
        Ok(RopeValues {
            query: output[0],
            key: output[1],
            value: output[2],
        })
    }
    fn append_kv(&mut self, domain: KvDomain, k: &PlanValue, v: &PlanValue) -> Result<()> {
        if k.shape != v.shape {
            return Err(Error::Malformed("plan KV slab shape".into()));
        }
        let effect = self.effect(StateSlot::Kv(domain))?;
        self.record(
            "kv-append",
            PlanOperation::AppendKv,
            &[*k, *v],
            &[],
            Some(effect),
        )?;
        Ok(())
    }
    fn attention_qk(
        &mut self,
        name: &str,
        domain: KvDomain,
        query: &PlanValue,
        valid_tokens: ParameterRef,
        geometry: AttentionGeometry,
        scale: f32,
    ) -> Result<QkValues<PlanValue>> {
        let (queries, _) = geometry.validate()?;
        if query.elements != queries || !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Malformed("plan source QK geometry".into()));
        }
        let state = self.current(StateSlot::Kv(domain))?;
        if state.ordinal != 1 {
            return Err(Error::State(
                "QK requires the private appended KV version".into(),
            ));
        }
        let shape = ValueShape::CausalRows {
            heads: geometry.query_heads,
            capacity: self.prefix_capacity,
        };
        let output = self.record(
            name,
            PlanOperation::AttentionQk {
                state,
                geometry,
                valid_tokens,
                scale,
            },
            &[*query],
            &[shape, shape],
            None,
        )?;
        Ok(QkValues {
            dots: output[0],
            scores: output[1],
        })
    }
    fn attention_softmax(
        &mut self,
        name: &str,
        scores: &PlanValue,
        heads: usize,
        valid_tokens: ParameterRef,
    ) -> Result<PlanValue> {
        if scores.shape
            != (ValueShape::CausalRows {
                heads,
                capacity: self.prefix_capacity,
            })
        {
            return Err(Error::Malformed("plan softmax shape".into()));
        }
        self.one(
            name,
            PlanOperation::AttentionSoftmax {
                heads,
                valid_tokens,
            },
            &[*scores],
            scores.shape,
            None,
        )
    }
    fn attention_value(
        &mut self,
        name: &str,
        domain: KvDomain,
        probabilities: &PlanValue,
        valid_tokens: ParameterRef,
        geometry: AttentionGeometry,
    ) -> Result<PlanValue> {
        geometry.validate()?;
        let state = self.current(StateSlot::Kv(domain))?;
        if state.ordinal != 1 {
            return Err(Error::State(
                "value matmul requires private appended KV".into(),
            ));
        }
        if probabilities.shape
            != (ValueShape::CausalRows {
                heads: geometry.query_heads,
                capacity: self.prefix_capacity,
            })
        {
            return Err(Error::Malformed("plan attention value shape".into()));
        }
        let count = geometry
            .query_heads
            .checked_mul(geometry.width)
            .ok_or_else(|| Error::Limit("plan attention output overflow".into()))?;
        self.one(
            name,
            PlanOperation::AttentionValue {
                state,
                geometry,
                valid_tokens,
                padding: 256,
            },
            &[*probabilities],
            ValueShape::Vector(count),
            None,
        )
    }
    fn assert_finite(&mut self, layer: usize, value: &PlanValue) -> Result<()> {
        self.record(
            "finite",
            PlanOperation::CheckFinite { layer },
            &[*value],
            &[],
            None,
        )?;
        Ok(())
    }
    fn observe(&mut self, layer: usize, stage: &'static str, value: &PlanValue) -> Result<()> {
        if self.capture {
            self.record(
                stage,
                PlanOperation::Observe { layer, stage },
                &[*value],
                &[],
                None,
            )?;
        }
        Ok(())
    }
    fn observe_state(
        &mut self,
        layer: usize,
        stage: &'static str,
        domain: StateSlot,
    ) -> Result<()> {
        if self.capture {
            let state = self.current(domain)?;
            self.record(
                stage,
                PlanOperation::ObserveState {
                    layer,
                    stage,
                    state,
                },
                &[],
                &[],
                None,
            )?;
        }
        Ok(())
    }
    fn advance_frontier(&mut self) -> Result<()> {
        if self.operator_slice {
            return Err(Error::State(
                "operator slice cannot advance a session frontier".into(),
            ));
        }
        if self.advanced.is_some() {
            return Err(Error::State(
                "duplicate forward-plan frontier advance".into(),
            ));
        }
        if self.domains.len() != self.required.len()
            || self
                .required
                .iter()
                .any(|slot| self.domains.get(slot).is_none_or(|v| v.ordinal != 1))
        {
            return Err(Error::State(
                "frontier advance before complete state transition".into(),
            ));
        }
        self.advanced = Some(self.nodes.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MatrixSource {
        identity: SourceIdentity,
        kind: QuantKind,
        shape: Vec<u64>,
        packed: Vec<u8>,
    }
    impl TensorSource for MatrixSource {
        fn source_identity(&self) -> SourceIdentity {
            self.identity
        }
        fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
            if name != "matrix" {
                return Err(Error::Malformed("unknown fixture tensor".into()));
            }
            Ok(SourceTensorRef {
                kind: self.kind,
                shape: &self.shape,
                packed_bytes: &self.packed,
            })
        }
    }
    fn matrix_source(shape: &[u64]) -> MatrixSource {
        MatrixSource {
            identity: source(),
            kind: QuantKind::F32,
            shape: shape.into(),
            packed: [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        }
    }

    #[test]
    fn explicit_matrix_axes_preserve_original_packed_rows_and_plan_shapes() {
        for (axes, shape) in [
            (MatrixAxes::InputOutput, [3, 2]),
            (MatrixAxes::OutputInput, [2, 3]),
        ] {
            let original = matrix_source(&shape);
            let before = original.packed.clone();
            let tensor = SourceTensor::from_source(&original, "matrix", axes).unwrap();
            assert_eq!((tensor.columns, tensor.rows, tensor.axes), (3, 2, axes));
            tensor.validate_source(&original).unwrap();
            assert_eq!(original.packed, before);
            assert_eq!(original.shape, shape);
            let mut plan = builder(vec![StateSlot::Kv(KvDomain(0))]);
            let x = plan
                .embed(tensor.clone(), ParameterRef(ParameterSlot::Token))
                .unwrap();
            assert_eq!(plan.elements(&x), 3);
            let y = plan.project(tensor.clone(), &x).unwrap();
            assert_eq!(plan.elements(&y), 2);
            assert!(matches!(
                &plan.nodes[1].operation,
                PlanOperation::Project { tensor: recorded } if recorded == &tensor
            ));
        }
    }

    #[test]
    fn source_matrix_binding_rejects_foreign_identity_axes_kind_and_bytes() {
        let original = matrix_source(&[2, 3]);
        let tensor =
            SourceTensor::from_source(&original, "matrix", MatrixAxes::OutputInput).unwrap();
        for changed in [
            SourceTensor {
                source: SourceIdentity::unbound(),
                ..tensor.clone()
            },
            SourceTensor {
                source: SourceIdentity::from_issued_id(114),
                ..tensor.clone()
            },
            SourceTensor {
                axes: MatrixAxes::InputOutput,
                ..tensor.clone()
            },
            SourceTensor {
                kind: QuantKind::F16,
                ..tensor.clone()
            },
            SourceTensor {
                columns: 2,
                rows: 3,
                ..tensor.clone()
            },
            SourceTensor {
                name: "missing".into(),
                ..tensor.clone()
            },
            SourceTensor {
                rows: 0,
                ..tensor.clone()
            },
            SourceTensor {
                columns: usize::MAX,
                rows: 2,
                ..tensor.clone()
            },
        ] {
            assert!(changed.validate_source(&original).is_err());
        }
        let mut foreign = matrix_source(&[2, 3]);
        foreign.identity = SourceIdentity::from_issued_id(114);
        assert!(tensor.validate_source(&foreign).is_err());
        for count in [23, 25] {
            let mut damaged = matrix_source(&[2, 3]);
            damaged.packed.resize(count, 0);
            assert!(tensor.validate_source(&damaged).is_err());
            assert!(
                SourceTensor::from_source(&damaged, "matrix", MatrixAxes::OutputInput).is_err()
            );
        }
    }

    #[test]
    fn source_matrix_constructor_refuses_bad_rank_overflow_and_cross_row_quant_blocks() {
        for shape in [
            vec![],
            vec![6],
            vec![1, 2, 3],
            vec![0, 6],
            vec![u64::MAX, 2],
        ] {
            let original = matrix_source(&shape);
            assert!(
                SourceTensor::from_source(&original, "matrix", MatrixAxes::OutputInput).is_err()
            );
        }
        let mut original = matrix_source(&[16, 2]);
        original.kind = QuantKind::Q8_0;
        original.packed = vec![0; 34];
        // Total source elements form one block, but output rows do not.
        assert!(original.tensor("matrix").unwrap().validate().is_ok());
        assert!(SourceTensor::from_source(&original, "matrix", MatrixAxes::InputOutput).is_err());
        let mut unbound = matrix_source(&[3, 2]);
        unbound.identity = SourceIdentity::unbound();
        assert!(SourceTensor::from_source(&unbound, "matrix", MatrixAxes::InputOutput).is_err());
        assert!(
            SourceTensor::from_source(&matrix_source(&[3, 2]), "", MatrixAxes::InputOutput)
                .is_err()
        );
    }

    fn source() -> SourceIdentity {
        SourceIdentity::from_issued_id(113)
    }
    fn source_tensor(columns: usize) -> SourceTensor {
        SourceTensor {
            source: source(),
            name: "token_embd.weight".into(),
            kind: QuantKind::F32,
            axes: MatrixAxes::InputOutput,
            columns,
            rows: 4,
        }
    }
    fn builder(domains: Vec<StateSlot>) -> PlanBuilder {
        PlanBuilder::new(source(), 257, domains, true).unwrap()
    }
    fn input(builder: &mut PlanBuilder, columns: usize) -> PlanValue {
        builder
            .embed(source_tensor(columns), ParameterRef(ParameterSlot::Token))
            .unwrap()
    }

    #[test]
    fn tensor_view_bounds_overflow_and_strided_packing_are_explicit() {
        assert!(
            TensorView {
                offset: 1,
                rows: 2,
                width: 1,
                row_stride: usize::MAX
            }
            .validate(usize::MAX)
            .is_err()
        );
        assert!(
            TensorView {
                offset: 0,
                rows: 0,
                width: 1,
                row_stride: 1
            }
            .validate(1)
            .is_err()
        );
        assert!(
            TensorView {
                offset: 0,
                rows: 2,
                width: 2,
                row_stride: 1
            }
            .validate(4)
            .is_err()
        );
        assert!(TensorView::vector(1, 4).validate(4).is_err());
        let mut b = builder(vec![StateSlot::Kv(KvDomain(0))]);
        let x = input(&mut b, 32);
        let layout = TensorView {
            offset: 1,
            rows: 4,
            width: 4,
            row_stride: 8,
        };
        let strided = b.view(&x, layout).unwrap();
        assert_eq!(b.elements(&strided), 16);
        assert!(b.view(&strided, TensorView::vector(0, 16)).is_err());
        assert!(b.add("illegal-strided-add", &strided, &strided).is_err());
        let packed = b.contiguous(&strided).unwrap();
        assert!(b.view(&packed, TensorView::vector(0, 16)).is_ok());
        assert!(
            b.nodes
                .iter()
                .any(|n| matches!(n.operation, PlanOperation::View(v) if v == layout))
        );
        assert!(
            b.nodes
                .iter()
                .any(|n| matches!(n.operation, PlanOperation::Contiguous))
        );
    }

    #[test]
    fn builder_source_value_and_constant_identities_fail_closed() {
        assert!(
            PlanBuilder::new(
                SourceIdentity::unbound(),
                1,
                vec![StateSlot::Kv(KvDomain(0))],
                false
            )
            .is_err()
        );
        assert!(PlanBuilder::new(source(), 0, vec![StateSlot::Kv(KvDomain(0))], false).is_err());
        assert!(
            PlanBuilder::new(
                source(),
                1,
                vec![StateSlot::Kv(KvDomain(0)), StateSlot::Kv(KvDomain(0))],
                false
            )
            .is_err()
        );
        let mut a = builder(vec![StateSlot::Kv(KvDomain(0))]);
        let mut b = builder(vec![StateSlot::Kv(KvDomain(0))]);
        let x = input(&mut a, 32);
        let foreign = input(&mut b, 32);
        assert_eq!(x.id, foreign.id);
        assert!(a.add("foreign", &x, &foreign).is_err());
        let mut wrong = source_tensor(32);
        wrong.source = SourceIdentity::from_issued_id(114);
        assert!(a.embed(wrong, ParameterRef(ParameterSlot::Token)).is_err());
        let gamma = [1.0f32; 32];
        a.rms(
            "rms",
            &x,
            SourceVector {
                source: source(),
                name: "gamma",
                values: &gamma,
            },
            1e-6,
        )
        .unwrap();
        let mut changed = gamma;
        changed[0] = -0.0;
        assert!(
            a.rms(
                "rms",
                &x,
                SourceVector {
                    source: source(),
                    name: "gamma",
                    values: &changed
                },
                1e-6
            )
            .is_err()
        );
        let mut invalid = gamma;
        invalid[0] = f32::NAN;
        assert!(
            a.rms(
                "rms",
                &x,
                SourceVector {
                    source: source(),
                    name: "nan-gamma",
                    values: &invalid
                },
                1e-6
            )
            .is_err()
        );
        assert!(
            a.rms(
                "rms",
                &x,
                SourceVector {
                    source: source(),
                    name: "gamma",
                    values: &gamma
                },
                f32::NAN
            )
            .is_err()
        );
        assert!(
            a.grouped_rms(
                "rms",
                &x,
                SourceVector {
                    source: source(),
                    name: "gamma",
                    values: &gamma
                },
                2,
                1e-6
            )
            .is_err()
        );
    }

    #[test]
    fn state_coverage_duplicate_effects_and_frontier_are_checked() {
        let conv = StateSlot::Conv(ConvDomain(0));
        let recurrent = StateSlot::Recurrent(RecurrentDomain(0));
        let weights = [1.0f32; 128];
        let mut incomplete = builder(vec![conv, recurrent]);
        let x = input(&mut incomplete, 32);
        incomplete
            .conv_step(
                "conv",
                ConvDomain(0),
                &x,
                SourceVector {
                    source: source(),
                    name: "conv.weight",
                    values: &weights,
                },
            )
            .unwrap();
        assert!(incomplete.advance_frontier().is_err());
        assert!(incomplete.finish(None).is_err());

        let mut missing_frontier = builder(vec![conv]);
        let x = input(&mut missing_frontier, 32);
        missing_frontier
            .conv_step(
                "conv",
                ConvDomain(0),
                &x,
                SourceVector {
                    source: source(),
                    name: "conv.weight",
                    values: &weights,
                },
            )
            .unwrap();
        assert!(missing_frontier.finish(None).is_err());

        let mut complete = builder(vec![conv]);
        let x = input(&mut complete, 32);
        let y = complete
            .conv_step(
                "conv",
                ConvDomain(0),
                &x,
                SourceVector {
                    source: source(),
                    name: "conv.weight",
                    values: &weights,
                },
            )
            .unwrap();
        assert!(
            complete
                .conv_step(
                    "conv-again",
                    ConvDomain(0),
                    &x,
                    SourceVector {
                        source: source(),
                        name: "conv.weight",
                        values: &weights
                    }
                )
                .is_err()
        );
        assert!(
            complete
                .conv_step(
                    "foreign-domain",
                    ConvDomain(2),
                    &x,
                    SourceVector {
                        source: source(),
                        name: "conv.weight",
                        values: &weights
                    }
                )
                .is_err()
        );
        complete.advance_frontier().unwrap();
        assert!(complete.advance_frontier().is_err());
        let plan = complete.finish(Some(y)).unwrap();
        assert_eq!(plan.frontier_delta(), 1);
        assert_eq!(
            plan.proposed_domains(),
            &[StateVersion {
                slot: conv,
                ordinal: 1
            }]
        );
        let effects: Vec<_> = plan.nodes().iter().filter_map(|node| node.state).collect();
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].read.ordinal, 0);
        assert_eq!(effects[0].write.ordinal, 1);
    }

    #[test]
    fn causal_plan_uses_symbolic_prefix_and_appended_private_version() {
        let slot = StateSlot::Kv(KvDomain(0));
        let mut b = builder(vec![slot]);
        let q = input(&mut b, 64);
        let k = b.view(&q, TensorView::vector(0, 32)).unwrap();
        let parameters = PlanBuilder::inputs();
        let geometry = AttentionGeometry {
            width: 32,
            query_heads: 2,
            kv_heads: 1,
        };
        assert!(
            b.attention_qk(
                "premature",
                KvDomain(0),
                &q,
                parameters.valid_tokens,
                geometry,
                0.125
            )
            .is_err()
        );
        b.validate_kv_prefix(KvDomain(0), parameters.position)
            .unwrap();
        let rope = b
            .rope(
                "rope",
                &q,
                &k,
                &k,
                parameters.position,
                RopeGeometry {
                    attention: geometry,
                    rotate: 2,
                    theta_scale: 0.5,
                },
            )
            .unwrap();
        b.append_kv(KvDomain(0), &rope.key, &rope.value).unwrap();
        assert!(
            b.validate_kv_prefix(KvDomain(0), parameters.position)
                .is_err()
        );
        assert!(b.append_kv(KvDomain(0), &rope.key, &rope.value).is_err());
        let qk = b
            .attention_qk(
                "qk",
                KvDomain(0),
                &rope.query,
                parameters.valid_tokens,
                geometry,
                0.125,
            )
            .unwrap();
        assert_eq!(
            qk.scores.shape,
            ValueShape::CausalRows {
                heads: 2,
                capacity: 257
            }
        );
        assert!(b.view(&qk.scores, TensorView::vector(0, 2)).is_err());
        let probabilities = b
            .attention_softmax("softmax", &qk.scores, 2, parameters.valid_tokens)
            .unwrap();
        let result = b
            .attention_value(
                "value",
                KvDomain(0),
                &probabilities,
                parameters.valid_tokens,
                geometry,
            )
            .unwrap();
        b.advance_frontier().unwrap();
        let node_count = b.nodes.len();
        assert!(
            b.validate_kv_prefix(KvDomain(0), parameters.position)
                .is_err()
        );
        assert!(
            b.attention_qk(
                "late-qk",
                KvDomain(0),
                &q,
                parameters.valid_tokens,
                geometry,
                0.125
            )
            .is_err()
        );
        assert!(
            b.attention_value(
                "late-value",
                KvDomain(0),
                &probabilities,
                parameters.valid_tokens,
                geometry
            )
            .is_err()
        );
        assert!(b.observe_state(0, "late-state", slot).is_err());
        assert_eq!(b.nodes.len(), node_count);
        let plan = b.finish(Some(result)).unwrap();
        for node in plan.nodes() {
            match node.operation {
                PlanOperation::KvPrefixGuard { state, position } => {
                    assert_eq!(state.ordinal, 0);
                    assert_eq!(position, parameters.position);
                }
                PlanOperation::AttentionQk {
                    state,
                    valid_tokens,
                    ..
                }
                | PlanOperation::AttentionValue {
                    state,
                    valid_tokens,
                    ..
                } => {
                    assert_eq!(state.ordinal, 1);
                    assert_eq!(valid_tokens, parameters.valid_tokens);
                }
                _ => {}
            }
        }
        assert_eq!(
            plan.values()[plan.logits().unwrap().0 as usize],
            ValueShape::Vector(64)
        );
    }

    #[test]
    fn state_prefix_keeps_checks_and_excludes_optional_output_math() {
        let mut b = builder(vec![StateSlot::Conv(ConvDomain(0))]);
        let x = input(&mut b, 32);
        b.conv_step(
            "conv",
            ConvDomain(0),
            &x,
            SourceVector {
                source: source(),
                name: "conv.weight",
                values: &[1.0; 128],
            },
        )
        .unwrap();
        b.assert_finite(0, &x).unwrap();
        b.observe(0, "state-stage-output", &x).unwrap();
        b.advance_frontier().unwrap();
        let count = b.nodes.len();
        let output = b.project(source_tensor(32), &x).unwrap();
        let plan = b.finish(Some(output)).unwrap();
        assert_eq!(plan.state_nodes().len(), count);
        assert!(matches!(
            plan.state_nodes()[count - 2].operation,
            PlanOperation::CheckFinite { .. }
        ));
        assert!(matches!(
            plan.state_nodes()[count - 1].operation,
            PlanOperation::Observe { .. }
        ));
        assert_eq!(plan.nodes().len(), count + 1);
        assert_eq!(plan.proposed_domains().len(), 1);
    }

    #[test]
    fn attention_geometry_and_coefficient_layout_refuse_overflow() {
        assert!(
            AttentionGeometry {
                width: usize::MAX,
                query_heads: 2,
                kv_heads: 1
            }
            .validate()
            .is_err()
        );
        assert!(
            AttentionGeometry {
                width: 32,
                query_heads: 3,
                kv_heads: 2
            }
            .validate()
            .is_err()
        );
        let mut b = builder(vec![StateSlot::Kv(KvDomain(0))]);
        let q = input(&mut b, 64);
        let k = b.view(&q, TensorView::vector(0, 32)).unwrap();
        let geometry = AttentionGeometry {
            width: 32,
            query_heads: 2,
            kv_heads: 1,
        };
        let position = ParameterRef(ParameterSlot::Position);
        assert!(
            b.rope(
                "odd-rope",
                &q,
                &k,
                &k,
                position,
                RopeGeometry {
                    attention: geometry,
                    rotate: 3,
                    theta_scale: 0.5
                }
            )
            .is_err()
        );
        assert!(
            b.rope(
                "nan-rope",
                &q,
                &k,
                &k,
                position,
                RopeGeometry {
                    attention: geometry,
                    rotate: 2,
                    theta_scale: f32::NAN
                }
            )
            .is_err()
        );
        let constants = [1.0f32; 64];
        assert!(
            b.coefficients(
                "wrong-heads",
                &q,
                &k,
                SourceVector {
                    source: source(),
                    name: "dt",
                    values: &constants
                },
                SourceVector {
                    source: source(),
                    name: "a",
                    values: &constants
                },
                0.125
            )
            .is_err()
        );
    }

    #[test]
    fn public_value_retargeting_cannot_reuse_cached_capacity() {
        let mut plan = builder(vec![StateSlot::Conv(ConvDomain(0))]);
        let mut narrow_tensor = source_tensor(32);
        narrow_tensor.name = "narrow_embedding".into();
        let narrow = plan
            .embed(narrow_tensor, ParameterRef(ParameterSlot::Token))
            .unwrap();
        let mut wide_tensor = source_tensor(256);
        wide_tensor.name = "wide_embedding".into();
        let wide = plan
            .embed(wide_tensor, ParameterRef(ParameterSlot::Token))
            .unwrap();
        let mut projection = source_tensor(256);
        projection.name = "projection".into();
        plan.project(projection.clone(), &wide).unwrap();

        // Both public fields now name the honest narrow value, but copying
        // the wide value retained its private 256-element capacity cache.
        let mut retargeted = wide;
        retargeted.id = narrow.id;
        retargeted.shape = narrow.shape;
        assert_eq!(plan.elements(&retargeted), 256);
        let values_before = plan.values.clone();
        let nodes_before = plan.nodes.len();
        let domains_before = plan.domains.clone();
        assert!(matches!(
            plan.project(projection, &retargeted),
            Err(Error::State(_))
        ));
        assert_eq!(plan.values, values_before);
        assert_eq!(plan.nodes.len(), nodes_before);
        assert_eq!(plan.domains, domains_before);

        // Honest descriptors remain usable after the refusal.
        let logits = plan.project(source_tensor(32), &narrow).unwrap();
        assert_eq!(plan.elements(&logits), 4);
        assert_eq!(plan.nodes.len(), nodes_before + 1);
        assert_eq!(plan.domains, domains_before);
    }

    #[test]
    fn failed_record_never_advances_state_or_leaves_partial_outputs() {
        let slot = StateSlot::Conv(ConvDomain(0));
        let mut a = builder(vec![slot]);
        let mut b = builder(vec![slot]);
        let x = input(&mut a, 32);
        let foreign = input(&mut b, 32);
        let weights = [1.0f32; 128];
        assert!(
            a.conv_step(
                "foreign-conv",
                ConvDomain(0),
                &foreign,
                SourceVector {
                    source: source(),
                    name: "weights",
                    values: &weights
                }
            )
            .is_err()
        );
        assert_eq!(a.current(slot).unwrap().ordinal, 0);
        let previous_values = a.values.len();
        assert!(
            a.record(
                "partial-output",
                PlanOperation::Contiguous,
                &[x],
                &[ValueShape::Vector(32), ValueShape::Vector(0)],
                None
            )
            .is_err()
        );
        assert_eq!(a.values.len(), previous_values);
        let output = a
            .conv_step(
                "conv",
                ConvDomain(0),
                &x,
                SourceVector {
                    source: source(),
                    name: "weights",
                    values: &weights,
                },
            )
            .unwrap();
        assert_eq!(a.current(slot).unwrap().ordinal, 1);
        let stale = StateEffect {
            read: StateVersion { slot, ordinal: 0 },
            write: StateVersion { slot, ordinal: 1 },
        };
        assert!(
            a.record(
                "stale-effect",
                PlanOperation::Contiguous,
                &[x],
                &[x.shape],
                Some(stale)
            )
            .is_err()
        );
        a.advance_frontier().unwrap();
        let plan = a.finish(Some(output)).unwrap();
        assert_eq!(
            plan.nodes()
                .iter()
                .filter(|node| node.state.is_some())
                .count(),
            1
        );
    }
}

/// Source-numeric diagnostic record shared by family and prepared execution.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PlanObservation {
    pub position: usize,
    pub layer: usize,
    pub stage: &'static str,
    pub values: Vec<f32>,
}
impl std::fmt::Debug for PlanObservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerObservation")
            .field("position", &self.position)
            .field("layer", &self.layer)
            .field("stage", &self.stage)
            .field("values", &self.values)
            .finish()
    }
}

#[cfg(test)]
#[path = "program/typed_tests.rs"]
mod typed_tests;
