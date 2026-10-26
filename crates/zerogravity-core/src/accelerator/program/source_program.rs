//! Source-bound mathematical operators; no model recipe, root or native permit.
use super::*;
mod quantized;

impl PlanValue {
    pub fn is_contiguous(self) -> bool {
        self.contiguous
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceCallMode {
    Prefill,
    Decode,
}

/// Immutable row issuer. Equal IDs in a separately constructed call are distinct.
#[derive(Clone, Debug)]
pub struct SourceRows(Arc<SourceRowsData>);
#[derive(Debug)]
struct SourceRowsData {
    geometry: TextCallGeometry,
    base: usize,
    ids: Arc<[u32]>,
    teacher: Option<TeacherSpan>,
}
impl PartialEq for SourceRows {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || match (&self.0.teacher, &other.0.teacher) {
                (Some(a), Some(b)) => a == b && self.0.geometry == other.0.geometry,
                _ => false,
            }
    }
}
impl Eq for SourceRows {}
impl SourceRows {
    /// Metadata only; the source owner must authenticate actual bytes and call.
    pub fn new(geometry: TextCallGeometry, base: usize, ids: Arc<[u32]>) -> Result<Self> {
        if ids.is_empty()
            || base
                .checked_add(ids.len())
                .is_none_or(|end| end > geometry.capacity())
            || ids.iter().any(|id| *id as usize >= geometry.vocabulary())
        {
            return Err(Error::Malformed("source rows original extent/IDs".into()));
        }
        Ok(Self(Arc::new(SourceRowsData {
            geometry,
            base,
            ids,
            teacher: None,
        })))
    }
    /// Preserve the existing immutable generic teacher issuer; no reissuance.
    pub fn from_teacher_span(span: TeacherSpan, vocabulary: usize) -> Result<Self> {
        let geometry = TextCallGeometry::new(span.source(), vocabulary, span.capacity())?;
        if span.ids().iter().any(|id| *id as usize >= vocabulary) {
            return Err(Error::State("source teacher vocabulary differs".into()));
        }
        Ok(Self(Arc::new(SourceRowsData {
            geometry,
            base: span.base(),
            ids: Arc::from(span.ids()),
            teacher: Some(span),
        })))
    }
    pub fn source(&self) -> SourceIdentity {
        self.0.geometry.source()
    }
    pub fn base(&self) -> usize {
        self.0.base
    }
    pub fn capacity(&self) -> usize {
        self.0.geometry.capacity()
    }
    pub fn vocabulary(&self) -> usize {
        self.0.geometry.vocabulary()
    }
    pub fn ids(&self) -> &[u32] {
        &self.0.ids
    }
    pub fn len(&self) -> usize {
        self.0.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn frontier(&self) -> usize {
        self.base() + self.len()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceCall {
    rows: SourceRows,
    mode: SourceCallMode,
}
impl SourceCall {
    pub fn new(rows: SourceRows, mode: SourceCallMode) -> Result<Self> {
        if mode == SourceCallMode::Decode && (rows.base() == 0 || rows.len() != 1) {
            return Err(Error::Malformed("source decode actual cached row".into()));
        }
        Ok(Self { rows, mode })
    }
    pub fn rows(&self) -> &SourceRows {
        &self.rows
    }
    pub fn mode(&self) -> SourceCallMode {
        self.mode
    }
    pub fn source(&self) -> SourceIdentity {
        self.rows.source()
    }
    fn same_layout(&self, other: &Self) -> bool {
        self.source() == other.source()
            && self.mode == other.mode
            && self.rows.base() == other.rows.base()
            && self.rows.len() == other.rows.len()
            && self.rows.capacity() == other.rows.capacity()
            && self.rows.vocabulary() == other.rows.vocabulary()
    }
}

/// Exact historical arithmetic identity, independent of a model's source roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceNumericalContract {
    // Prior arithmetic name: Vllm031CudaSm86Bf16V1.
    ReferenceCudaSm86Bf16V1,
    GgmlSourceF32V1,
    GgmlQ4Repack8x8F32V1,
    OpenBlas033Zen16Q8F32V1,
}
impl SourceNumericalContract {
    pub fn is_f32(self) -> bool {
        self != Self::ReferenceCudaSm86Bf16V1
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceRetainedOrigin {
    Normalized { epsilon_bits: u32 },
    Gated,
}
/// Input teacher-major Q/K/V; output all Q rows, then K rows, then unchanged V rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceGroupedQkGeometry {
    pub heads: usize,
    pub width: usize,
    pub epsilon: f32,
    pub scale: f32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceGamma {
    Direct,
    OnePlus,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceRmsGeometry {
    pub width: usize,
    pub epsilon: f32,
    pub gamma: SourceGamma,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceConvGeometry {
    pub qkv_channels: usize,
    pub z_channels: usize,
    pub coefficient_channels: usize,
    pub kernel: usize,
}
/// L2 Q/K normalization, recurrent FP32 accumulation, source SiLU output gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceGdnContract {
    // Prior arithmetic name: Vllm031CudaSm86L2SiluBf16V1.
    ReferenceCudaSm86L2SiluBf16V1,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceGdnGeometry {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub epsilon: f32,
    pub contract: SourceGdnContract,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceQkRotaryGeometry {
    pub attention: AttentionGeometry,
    pub rotary_dimensions: usize,
    pub theta: f32,
    pub epsilon: f32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceAttentionMask {
    Causal,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourceOperatorKind {
    /// Metadata import only; actual retained backing is authenticated by its owner.
    RetainedRows {
        width: usize,
        origin: SourceRetainedOrigin,
    },
    /// Pure finite window with zero prior history; no persistent state effect.
    CausalConvSiluZero {
        channels: usize,
        kernel: usize,
    },
    GroupedQkNormalize(SourceGroupedQkGeometry),
    EmbeddingRows,
    /// Matrices remain separate, in source concatenation order. Each original
    /// member's rows/axes/codec are explicit; no synthetic packed source name.
    Projection {
        selected_row: Option<usize>,
    },
    Rms(SourceRmsGeometry),
    ResidualRms(SourceRmsGeometry),
    CausalConvSplit {
        domain: ConvDomain,
        geometry: SourceConvGeometry,
    },
    GatedDelta {
        domain: RecurrentDomain,
        geometry: SourceGdnGeometry,
    },
    /// Packed query/gate then key then value; V remains the original strided tail.
    QkRmsRotaryGated(SourceQkRotaryGeometry),
    KvAppend {
        domain: KvDomain,
        key_width: usize,
        packed_value: TensorView,
    },
    FlashAttention {
        domain: KvDomain,
        geometry: AttentionGeometry,
        scale: f32,
        mask: SourceAttentionMask,
    },
    Sigmoid {
        width: usize,
    },
    Multiply {
        width: usize,
    },
    SiluAndMul {
        width: usize,
    },
    StateCarry {
        slot: StateSlot,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceStateAccess {
    None,
    Write(StateSlot),
    Read(StateSlot),
}
#[derive(Clone, Debug, PartialEq)]
pub struct SourceOperator {
    call: SourceCall,
    semantics: Arc<SourceOperatorSemantics>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct SourceOperatorSemantics {
    kind: SourceOperatorKind,
    numerical: SourceNumericalContract,
    matrices: Vec<SourceTensor>,
    parameters: Vec<SourceParameter>,
}
impl SourceOperator {
    pub fn new(
        call: SourceCall,
        kind: SourceOperatorKind,
        numerical: SourceNumericalContract,
        matrices: Vec<SourceTensor>,
        parameters: Vec<SourceParameter>,
    ) -> Result<Self> {
        let value = Self {
            call,
            semantics: Arc::new(SourceOperatorSemantics {
                kind,
                numerical,
                matrices,
                parameters,
            }),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn call(&self) -> &SourceCall {
        &self.call
    }
    pub fn kind(&self) -> SourceOperatorKind {
        self.semantics.kind
    }
    pub fn numerical_contract(&self) -> SourceNumericalContract {
        self.semantics.numerical
    }
    pub fn matrices(&self) -> &[SourceTensor] {
        &self.semantics.matrices
    }
    pub fn parameters(&self) -> &[SourceParameter] {
        &self.semantics.parameters
    }
    pub fn shares_semantics(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.semantics, &other.semantics)
    }
    pub fn state_access(&self) -> SourceStateAccess {
        use SourceOperatorKind::*;
        match self.kind() {
            CausalConvSplit { domain, .. } => SourceStateAccess::Write(StateSlot::Conv(domain)),
            GatedDelta { domain, .. } => SourceStateAccess::Write(StateSlot::Recurrent(domain)),
            KvAppend { domain, .. } => SourceStateAccess::Write(StateSlot::Kv(domain)),
            FlashAttention { domain, .. } => SourceStateAccess::Read(StateSlot::Kv(domain)),
            StateCarry { slot } => SourceStateAccess::Write(slot),
            _ => SourceStateAccess::None,
        }
    }
    fn product(a: usize, b: usize) -> Result<usize> {
        a.checked_mul(b)
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::Limit("source operator positive product".into()))
    }
    fn sum(a: usize, b: usize) -> Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| Error::Limit("source operator sum".into()))
    }
    fn projection_extent(&self) -> Result<(usize, usize)> {
        let first = self
            .matrices()
            .first()
            .ok_or_else(|| Error::Malformed("source projection members absent".into()))?;
        let mut rows = 0usize;
        for matrix in self.matrices() {
            if matrix.columns != first.columns {
                return Err(Error::Malformed(
                    "source concatenation input columns differ".into(),
                ));
            }
            rows = Self::sum(rows, matrix.rows)?;
        }
        Ok((first.columns, rows))
    }
    fn parameter(&self, index: usize, kinds: &[QuantKind], shape: &[usize]) -> Result<()> {
        let p = self
            .parameters()
            .get(index)
            .ok_or_else(|| Error::Malformed("source operator parameter absent".into()))?;
        let count = shape
            .iter()
            .try_fold(1usize, |n, width| Self::product(n, *width))?;
        let element_bytes = match p.kind {
            QuantKind::Bf16 => 2,
            QuantKind::F32 => 4,
            _ => {
                return Err(Error::Unsupported(
                    "source operator original parameter codec".into(),
                ));
            }
        };
        if !kinds.contains(&p.kind)
            || p.shape.iter().copied().ne(shape.iter().map(|n| *n as u64))
            || p.bytes != Self::product(count, element_bytes)?
        {
            return Err(Error::Malformed(
                "source parameter exact codec/shape/bytes".into(),
            ));
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        use SourceOperatorKind::*;
        if self.numerical_contract().is_f32() {
            return quantized::validate(self);
        }
        if matches!(
            self.kind(),
            RetainedRows { .. } | CausalConvSiluZero { .. } | GroupedQkNormalize(_)
        ) || matches!(self.kind(), Rms(g) | ResidualRms(g) if g.gamma != SourceGamma::OnePlus)
        {
            return Err(Error::Unsupported("BF16 source kind/gamma contract".into()));
        }
        if self.matrices().len() > 3 || self.parameters().len() > 3 {
            return Err(Error::Limit(
                "source operator bounded ordered members".into(),
            ));
        }
        for matrix in self.matrices() {
            if matrix.source != self.call.source() {
                return Err(Error::State("source operator matrix owner differs".into()));
            }
            typed::validate_bf16_matrix(matrix)?;
            matrix.packed_bytes()?;
        }
        for parameter in self.parameters() {
            if parameter.source != self.call.source() || parameter.name.is_empty() {
                return Err(Error::State(
                    "source operator parameter owner/name differs".into(),
                ));
            }
        }
        let (matrices, parameters) = match self.kind() {
            RetainedRows { .. } | CausalConvSiluZero { .. } | GroupedQkNormalize(_) => {
                unreachable!("separate F32 contract")
            }
            EmbeddingRows => {
                let matrix = self
                    .matrices()
                    .first()
                    .ok_or_else(|| Error::Malformed("source embedding absent".into()))?;
                if matrix.rows != self.call.rows.vocabulary() {
                    return Err(Error::State("source embedding actual vocabulary".into()));
                }
                (1, 0)
            }
            Projection { selected_row } => {
                self.projection_extent()?;
                if selected_row.is_some_and(|row| row >= self.call.rows.len()) {
                    return Err(Error::Malformed(
                        "source projection selected actual row".into(),
                    ));
                }
                (self.matrices().len(), 0)
            }
            Rms(geometry) | ResidualRms(geometry) => {
                if !geometry.epsilon.is_finite() || geometry.epsilon <= 0.0 {
                    return Err(Error::Malformed("source RMS epsilon".into()));
                }
                self.parameter(0, &[QuantKind::Bf16, QuantKind::F32], &[geometry.width])?;
                (0, 1)
            }
            CausalConvSplit { geometry, .. } => {
                Self::product(geometry.z_channels, self.call.rows.len())?;
                Self::product(geometry.coefficient_channels, self.call.rows.len())?;
                self.parameter(
                    0,
                    &[QuantKind::Bf16],
                    &[geometry.qkv_channels, 1, geometry.kernel],
                )?;
                (0, 1)
            }
            GatedDelta { geometry, .. } => {
                if geometry.key_heads == 0
                    || geometry.value_heads == 0
                    || !geometry.value_heads.is_multiple_of(geometry.key_heads)
                    || !geometry.epsilon.is_finite()
                    || geometry.epsilon <= 0.0
                {
                    return Err(Error::Malformed("source gated delta geometry".into()));
                }
                Self::product(geometry.key_heads, geometry.key_width)?;
                Self::product(geometry.value_heads, geometry.value_width)?;
                self.parameter(0, &[QuantKind::F32], &[geometry.value_heads])?;
                self.parameter(1, &[QuantKind::Bf16], &[geometry.value_heads])?;
                self.parameter(2, &[QuantKind::F32], &[geometry.value_width])?;
                (0, 3)
            }
            QkRmsRotaryGated(geometry) => {
                geometry.attention.validate()?;
                if geometry.rotary_dimensions == 0
                    || !geometry.rotary_dimensions.is_multiple_of(2)
                    || geometry.rotary_dimensions > geometry.attention.width
                    || !geometry.theta.is_finite()
                    || geometry.theta <= 1.0
                    || !geometry.epsilon.is_finite()
                    || geometry.epsilon <= 0.0
                {
                    return Err(Error::Malformed("source QK RMS/rotary geometry".into()));
                }
                for index in 0..2 {
                    self.parameter(index, &[QuantKind::Bf16], &[geometry.attention.width])?;
                }
                (0, 2)
            }
            KvAppend {
                key_width,
                packed_value,
                ..
            } => {
                if packed_value.rows != self.call.rows.len() || packed_value.width != key_width {
                    return Err(Error::Malformed(
                        "source KV original packed value rows/width".into(),
                    ));
                }
                packed_value.validate(Self::product(
                    self.call.rows.len(),
                    packed_value.row_stride,
                )?)?;
                (0, 0)
            }
            FlashAttention {
                geometry, scale, ..
            } => {
                geometry.validate()?;
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(Error::Malformed("source attention scale".into()));
                }
                (0, 0)
            }
            Sigmoid { width } | Multiply { width } | SiluAndMul { width } => {
                Self::product(self.call.rows.len(), width)?;
                (0, 0)
            }
            StateCarry { .. } => (0, 0),
        };
        if self.matrices().len() != matrices || self.parameters().len() != parameters {
            return Err(Error::Malformed(
                "source operator exact member arity".into(),
            ));
        }
        self.input_elements()?;
        self.output_elements()?;
        Ok(())
    }
    pub fn input_elements(&self) -> Result<Vec<usize>> {
        use SourceOperatorKind::*;
        let n = self.call.rows.len();
        let p = Self::product;
        let values = match self.kind() {
            EmbeddingRows | StateCarry { .. } | RetainedRows { .. } => vec![],
            CausalConvSiluZero { channels, .. } => vec![p(n, channels)?],
            GroupedQkNormalize(g) => vec![p(n, p(3, p(g.heads, g.width)?)?)?],
            Projection { .. } => vec![p(n, self.projection_extent()?.0)?],
            Rms(g) => vec![p(n, g.width)?],
            ResidualRms(g) => vec![p(n, g.width)?; 2],
            CausalConvSplit { geometry: g, .. } => vec![
                p(n, Self::sum(g.qkv_channels, g.z_channels)?)?,
                p(n, g.coefficient_channels)?,
            ],
            GatedDelta { geometry: g, .. } => {
                let key = p(g.key_heads, g.key_width)?;
                let value = p(g.value_heads, g.value_width)?;
                vec![
                    p(n, Self::sum(p(2, key)?, value)?)?,
                    p(n, value)?,
                    p(n, p(2, g.value_heads)?)?,
                ]
            }
            QkRmsRotaryGated(g) => {
                let (query, kv) = g.attention.validate()?;
                vec![p(n, p(2, Self::sum(query, kv)?)?)?]
            }
            KvAppend {
                key_width,
                packed_value,
                ..
            } => vec![p(n, key_width)?, p(n, packed_value.row_stride)?],
            FlashAttention { geometry, .. } => vec![p(n, geometry.validate()?.0)?],
            Sigmoid { width } | SiluAndMul { width } => vec![p(
                n,
                if matches!(self.kind(), SiluAndMul { .. }) {
                    p(2, width)?
                } else {
                    width
                },
            )?],
            Multiply { width } => vec![p(n, width)?; 2],
        };
        Ok(values)
    }
    pub fn output_elements(&self) -> Result<Vec<usize>> {
        use SourceOperatorKind::*;
        let n = self.call.rows.len();
        let p = Self::product;
        let values = match self.kind() {
            RetainedRows { width, .. } => vec![p(n, width)?],
            CausalConvSiluZero { channels, .. } => vec![p(n, channels)?],
            GroupedQkNormalize(g) => vec![p(n, p(3, p(g.heads, g.width)?)?)?],
            EmbeddingRows => vec![p(
                n,
                self.matrices()
                    .first()
                    .ok_or_else(|| Error::Malformed("source embedding absent".into()))?
                    .columns,
            )?],
            Projection { selected_row } => vec![p(
                if selected_row.is_some() { 1 } else { n },
                self.projection_extent()?.1,
            )?],
            Rms(g) => vec![p(n, g.width)?],
            ResidualRms(g) => vec![p(n, g.width)?; 2],
            CausalConvSplit { geometry: g, .. } => vec![p(n, g.qkv_channels)?, p(n, g.z_channels)?],
            GatedDelta { geometry: g, .. } => vec![p(n, p(g.value_heads, g.value_width)?)?],
            QkRmsRotaryGated(g) => {
                let (query, kv) = g.attention.validate()?;
                vec![p(n, query)?, p(n, kv)?, p(n, query)?]
            }
            KvAppend { .. } | StateCarry { .. } => vec![],
            FlashAttention { geometry, .. } => vec![p(n, geometry.validate()?.0)?],
            Sigmoid { width } | Multiply { width } | SiluAndMul { width } => vec![p(n, width)?],
        };
        Ok(values)
    }
}

/// Immutable producer minted by the single builder, independent of family roles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceValueProvenance {
    call: SourceCall,
    node_index: usize,
    output_index: usize,
    inputs: Arc<[ValueId]>,
}
impl SourceValueProvenance {
    pub fn call(&self) -> &SourceCall {
        &self.call
    }
    pub fn node_index(&self) -> usize {
        self.node_index
    }
    pub fn output_index(&self) -> usize {
        self.output_index
    }
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }
}

impl PlanBuilder {
    /// Complete source arithmetic without a frontier/logits publication capability.
    pub fn new_source_program(call: SourceCall, required: Vec<StateSlot>) -> Result<Self> {
        let mut builder = Self::new_inner(
            call.source(),
            call.rows.capacity(),
            required,
            false,
            ProgramNumericalProfile::SourceCudaSm86Bf16V1,
            false,
        )?;
        builder.source_call = Some(call);
        Ok(builder)
    }
    pub fn record_source_operator(
        &mut self,
        name: &str,
        descriptor: &SourceOperator,
        inputs: &[PlanValue],
    ) -> Result<Vec<PlanValue>> {
        record_named(self, name, descriptor, inputs)
    }
    pub fn source_call(&self) -> Option<&SourceCall> {
        self.source_call.as_ref()
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn nodes(&self) -> &[PlanNode] {
        &self.nodes
    }
    pub fn values(&self) -> &[ValueShape] {
        &self.values
    }
    pub fn precisions(&self) -> &[ActivationPrecision] {
        &self.precisions
    }
    pub fn constants(&self) -> &[StaticVector] {
        &self.constants
    }
    pub fn required_domains(&self) -> &BTreeSet<StateSlot> {
        &self.required
    }
    pub fn proposed_domains(&self) -> &BTreeMap<StateSlot, StateVersion> {
        &self.domains
    }
    pub fn has_advanced(&self) -> bool {
        self.advanced.is_some()
    }
    pub fn is_operator_slice(&self) -> bool {
        self.operator_slice
    }
    pub fn current_state(&self, slot: StateSlot) -> Result<StateVersion> {
        self.current(slot)
    }
    pub fn validate_value(&self, value: &PlanValue) -> Result<()> {
        if value.owner != self.owner
            || self.values.get(value.id.0 as usize) != Some(&value.shape)
            || value.elements != value.shape.capacity_elements()?
            || self.precisions.get(value.id.0 as usize) != Some(&value.precision)
        {
            return Err(Error::State(
                "foreign or malformed source program value".into(),
            ));
        }
        Ok(())
    }
    pub fn source_value_provenance(
        &self,
        value: &PlanValue,
    ) -> Result<Option<&SourceValueProvenance>> {
        self.validate_value(value)?;
        self.source_provenance
            .get(value.id.0 as usize)
            .map(Option::as_ref)
            .ok_or_else(|| Error::State("source program producer coverage".into()))
    }
    pub fn finish_source_program(self) -> Result<ForwardPlan> {
        if self.source_call.is_none()
            || self.advanced.is_some()
            || self.operator_slice
            || self.required.len() != self.domains.len()
            || self
                .required
                .iter()
                .any(|slot| self.domains.get(slot).is_none_or(|v| v.ordinal != 1))
        {
            return Err(Error::State(
                "incomplete source arithmetic state proposal".into(),
            ));
        }
        let state_node_count = self.nodes.len();
        Ok(ForwardPlan {
            binding_metadata: None,
            source_call: self.source_call,
            source_provenance: self.source_provenance,
            nodes: self.nodes,
            semantics: Arc::new(ForwardPlanSemantics {
                metadata: None,
                source: self.source,
                profile: self.profile,
                standalone_exp: self.standalone_exp,
                bf16_text: self.bf16_text,
                operator_slice: false,
                precisions: self.precisions,
                prefix_capacity: self.prefix_capacity,
                values: self.values,
                constants: self.constants,
                required_domains: self.required.into_iter().collect(),
                proposed_domains: self.domains.into_values().collect(),
                logits: None,
                frontier_delta: 0,
                state_node_count,
            }),
        })
    }
}
impl ForwardPlan {
    pub fn source_call(&self) -> Option<&SourceCall> {
        self.source_call.as_ref()
    }
    pub fn source_value_provenance(&self, id: ValueId) -> Result<Option<&SourceValueProvenance>> {
        if self.values.get(id.0 as usize).is_none() {
            return Err(Error::State("foreign source program value ID".into()));
        }
        self.source_provenance
            .get(id.0 as usize)
            .map(Option::as_ref)
            .ok_or_else(|| Error::State("source program producer coverage".into()))
    }
    /// Same mathematical layout only. The model owner authenticates its finite
    /// workload/source and actual per-row root before this metadata operation.
    pub fn rebind_source_call(&self, call: SourceCall) -> Result<Self> {
        let prior = self
            .source_call
            .as_ref()
            .ok_or_else(|| Error::State("not source arithmetic record".into()))?;
        if !prior.same_layout(&call) {
            return Err(Error::State("source call layout differs".into()));
        }
        let mut bound = self.clone();
        bound.binding_metadata = None;
        for node in &mut bound.nodes {
            let PlanOperation::SourceOperator { descriptor, .. } = &mut node.operation else {
                return Err(Error::State(
                    "source rebind contains foreign operation".into(),
                ));
            };
            if descriptor.call != *prior {
                return Err(Error::State("source rebind node issuer differs".into()));
            }
            descriptor.call = call.clone();
        }
        for tag in &mut bound.source_provenance {
            let tag = tag
                .as_mut()
                .ok_or_else(|| Error::State("source rebind producer absent".into()))?;
            if tag.call != *prior {
                return Err(Error::State("source rebind producer issuer differs".into()));
            }
            tag.call = call.clone();
        }
        bound.source_call = Some(call);
        Ok(bound)
    }
    pub fn shares_semantics(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.semantics, &other.semantics)
    }
    /// Retain an existing source-owned host charge; this grants no eligibility.
    pub fn retain_prepared_metadata(
        mut self,
        metadata: crate::resource_arbiter::HostWorkspace,
    ) -> Result<Self> {
        if metadata.source_identity() != self.source() || metadata.bytes() == 0 {
            return Err(Error::State(
                "prepared source plan metadata owner/extent".into(),
            ));
        }
        let semantics = Arc::get_mut(&mut self.semantics)
            .ok_or_else(|| Error::State("prepared metadata must precede sharing".into()))?;
        if semantics.metadata.is_some() {
            return Err(Error::State("prepared metadata already retained".into()));
        }
        semantics.metadata = Some(Arc::new(PreparedPlanMetadata(metadata)));
        Ok(self)
    }
    /// Existing source-owned charge, shared with the immutable plan semantics.
    pub fn prepared_metadata_bytes(&self) -> Option<usize> {
        self.semantics
            .metadata
            .as_ref()
            .map(|metadata| metadata.0.bytes())
    }
    /// Charge invocation-local nodes/provenance separately from shared semantics.
    /// Replacement releases only the previous local charge; no eligibility grant.
    pub fn retain_binding_metadata(
        mut self,
        metadata: crate::resource_arbiter::HostWorkspace,
    ) -> Result<Self> {
        if metadata.source_identity() != self.source()
            || metadata.bytes() < self.source_binding_metadata_bytes()?
        {
            return Err(Error::State("source binding metadata owner/extent".into()));
        }
        self.binding_metadata = Some(Arc::new(PreparedPlanMetadata(metadata)));
        Ok(self)
    }
    pub fn source_binding_metadata_bytes(&self) -> Result<usize> {
        let overflow = || Error::Limit("source binding metadata extent".into());
        let boxes = self.nodes.iter().try_fold(0usize, |sum, node| {
            let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
                return Err(Error::State("source binding foreign operation".into()));
            };
            sum.checked_add(std::mem::size_of_val(&**descriptor))
                .ok_or_else(overflow)
        })?;
        let nodes = self
            .nodes
            .capacity()
            .checked_mul(std::mem::size_of::<PlanNode>())
            .ok_or_else(overflow)?;
        let tags = self
            .source_provenance
            .capacity()
            .checked_mul(std::mem::size_of::<Option<SourceValueProvenance>>())
            .ok_or_else(overflow)?;
        boxes
            .checked_add(nodes)
            .and_then(|v| v.checked_add(tags))
            .ok_or_else(overflow)
    }
}
pub(super) fn precision(
    builder: &PlanBuilder,
    descriptor: &SourceOperator,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    descriptor.validate()?;
    if descriptor.numerical_contract().is_f32() {
        return quantized::precision(builder, descriptor, inputs);
    }
    if builder.profile != ProgramNumericalProfile::SourceCudaSm86Bf16V1
        || !(builder.operator_slice || builder.source_call.is_some())
        || builder.source != descriptor.call.source()
        || builder
            .source_call
            .as_ref()
            .is_some_and(|call| *call != descriptor.call)
        || inputs.len() != descriptor.input_elements()?.len()
    {
        return Err(Error::Unsupported(
            "source operator numerical/call/source/arity contract".into(),
        ));
    }
    for (input, elements) in inputs.iter().zip(descriptor.input_elements()?) {
        builder.validate_value(input)?;
        if input.precision != ActivationPrecision::Bf16WidenedF32
            || input.shape != ValueShape::Vector(elements)
            || input.elements != elements
            || builder
                .source_value_provenance(input)?
                .is_none_or(|tag| tag.call != descriptor.call)
        {
            return Err(Error::State(
                "source operator same-call dense BF16 producer".into(),
            ));
        }
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn provenance(
    builder: &PlanBuilder,
    operation: &PlanOperation,
    inputs: &[PlanValue],
    output_index: usize,
) -> Result<Option<SourceValueProvenance>> {
    let PlanOperation::SourceOperator { descriptor, .. } = operation else {
        return Ok(None);
    };
    if output_index >= descriptor.output_elements()?.len() {
        return Err(Error::State("source producer output index".into()));
    }
    Ok(Some(SourceValueProvenance {
        call: descriptor.call.clone(),
        node_index: builder.nodes.len(),
        output_index,
        inputs: inputs.iter().map(|value| value.id).collect(),
    }))
}
pub(super) fn record(
    builder: &mut PlanBuilder,
    descriptor: &SourceOperator,
    inputs: &[PlanValue],
) -> Result<Vec<PlanValue>> {
    record_named(builder, "source.operator", descriptor, inputs)
}
fn record_named(
    builder: &mut PlanBuilder,
    name: &str,
    descriptor: &SourceOperator,
    inputs: &[PlanValue],
) -> Result<Vec<PlanValue>> {
    precision(builder, descriptor, inputs)?;
    let (effect, state) = match descriptor.state_access() {
        SourceStateAccess::None => (None, None),
        SourceStateAccess::Write(slot) => (Some(builder.effect(slot)?), None),
        SourceStateAccess::Read(slot) => {
            let state = builder.current(slot)?;
            if state.ordinal != 1 {
                return Err(Error::State(
                    "source attention requires private appended KV version".into(),
                ));
            }
            (None, Some(state))
        }
    };
    builder.record(
        name,
        PlanOperation::SourceOperator {
            descriptor: Box::new(descriptor.clone()),
            state,
        },
        inputs,
        &descriptor
            .output_elements()?
            .into_iter()
            .map(ValueShape::Vector)
            .collect::<Vec<_>>(),
        effect,
    )
}

#[cfg(test)]
mod tests;
