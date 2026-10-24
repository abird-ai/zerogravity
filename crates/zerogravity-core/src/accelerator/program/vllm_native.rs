//! Source-member/span metadata; finite native coverage is a separate binding.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VllmBf16Mode {
    Prefill,
    Decode,
}

fn legacy_phase_for(mode: VllmBf16Mode, base: usize, rows: usize) -> Option<VllmBf16Phase> {
    match (mode, base, rows) {
        (VllmBf16Mode::Prefill, 0, 1) => Some(VllmBf16Phase::InitialOne),
        (VllmBf16Mode::Prefill, 0, 64) => Some(VllmBf16Phase::Initial64),
        (VllmBf16Mode::Prefill, 64, 2) => Some(VllmBf16Phase::ContinuedTwo),
        (VllmBf16Mode::Decode, 1 | 66, 1) => Some(VllmBf16Phase::CachedOne),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VllmBf16Phase {
    InitialOne,
    CachedOne,
    Initial64,
    ContinuedTwo,
}
impl VllmBf16Phase {
    pub fn mode(self) -> VllmBf16Mode {
        match self {
            Self::CachedOne => VllmBf16Mode::Decode,
            _ => VllmBf16Mode::Prefill,
        }
    }
    pub fn rows(self) -> usize {
        match self {
            Self::InitialOne | Self::CachedOne => 1,
            Self::Initial64 => 64,
            Self::ContinuedTwo => 2,
        }
    }
    fn validate(self, teachers: &VllmBf16Teachers) -> Result<()> {
        let base = teachers.base();
        if teachers.short_hf_span().is_some()
            || teachers.len() != self.rows()
            || !match self {
                Self::InitialOne | Self::Initial64 => base == 0,
                Self::CachedOne => matches!(base, 1 | 66),
                Self::ContinuedTwo => base == 64,
            }
        {
            return Err(Error::Unsupported(
                "closed vLLM genuine phase/span1/64/2".into(),
            ));
        }
        Ok(())
    }
}

/// Share this descriptor's teacher issuer through the SAME plan producer chain.
/// The physical lowerer must also validate its model/config/current invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmBf16Invocation {
    layer: GdnPrefixLayer,
    mode: VllmBf16Mode,
    teachers: VllmBf16Teachers,
}
impl VllmBf16Invocation {
    pub(super) fn from_short_hf_span(span: ShortHfSpan) -> Result<Self> {
        span.validate()?;
        let value = Self {
            layer: GdnPrefixLayer::new(span.source(), 0)?,
            mode: span.mode(),
            teachers: VllmBf16Teachers::from_short_hf_span(span),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn new(
        layer: GdnPrefixLayer,
        phase: VllmBf16Phase,
        teachers: VllmBf16Teachers,
    ) -> Result<Self> {
        if layer.source() != teachers.source() {
            return Err(Error::State("vLLM invocation source/teacher issuer".into()));
        }
        phase.validate(&teachers)?;
        Self::from_span(layer, phase.mode(), teachers)
    }
    /// Semantic metadata only. Existing backends still require finite coverage.
    pub fn from_span(
        layer: GdnPrefixLayer,
        mode: VllmBf16Mode,
        teachers: VllmBf16Teachers,
    ) -> Result<Self> {
        if teachers.short_hf_span().is_some() {
            return Err(Error::Unsupported(
                "short-HF issuer requires its source-profile call".into(),
            ));
        }
        let value = Self {
            layer,
            mode,
            teachers,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn layer(&self) -> GdnPrefixLayer {
        self.layer
    }
    pub fn mode(&self) -> VllmBf16Mode {
        self.mode
    }
    /// Exact old finite metadata only; never infer coverage for another span.
    pub fn legacy_phase(&self) -> Result<VllmBf16Phase> {
        self.validate()?;
        if self.teachers.short_hf_span().is_some() {
            return Err(Error::Unsupported(
                "short-HF span has no legacy vLLM native phase".into(),
            ));
        }
        legacy_phase_for(self.mode, self.teachers.base(), self.rows())
            .ok_or_else(|| Error::Unsupported("span has no legacy vLLM native phase".into()))
    }
    pub fn teachers(&self) -> &VllmBf16Teachers {
        &self.teachers
    }
    pub fn rows(&self) -> usize {
        self.teachers.len()
    }
    pub fn validate(&self) -> Result<()> {
        if self.layer.source() != self.teachers.source() {
            return Err(Error::State("vLLM invocation source changed".into()));
        }
        let t = &self.teachers;
        if let Some(span) = t.short_hf_span() {
            span.validate()?;
            if self.layer.index() != 0 || self.mode != span.mode() {
                return Err(Error::State("short-HF invocation scope changed".into()));
            }
            return Ok(());
        }
        if !(1..=64).contains(&t.len())
            || t.capacity() == 0
            || t.capacity() > 8448
            || t.base()
                .checked_add(t.len())
                .is_none_or(|end| end > t.capacity())
            || t.ids().iter().any(|&id| id >= 248_320)
            || self.mode == VllmBf16Mode::Decode && (t.len() != 1 || t.base() == 0)
        {
            return Err(Error::Malformed("vLLM semantic source/span/mode".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VllmNativeProjectionContract {
    Vllm031CudaSm86Bf16V1,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VllmSiluAndMulContract {
    Vllm031CudaSm86Bf16V1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VllmGdnProjectionKind {
    Qkvz,
    Ba,
    GdnOut,
    GateUp,
    Down,
}
impl VllmGdnProjectionKind {
    pub fn input_width(self) -> usize {
        match self {
            Self::Qkvz | Self::Ba | Self::GateUp => 1024,
            Self::GdnOut => 2048,
            Self::Down => 3584,
        }
    }
    pub fn output_width(self) -> usize {
        match self {
            Self::Qkvz => 8192,
            Self::Ba => 32,
            Self::GdnOut | Self::Down => 1024,
            Self::GateUp => 7168,
        }
    }
    fn members(self) -> &'static [(&'static str, usize)] {
        match self {
            Self::Qkvz => &[
                ("linear_attn.in_proj_qkv.weight", 6144),
                ("linear_attn.in_proj_z.weight", 2048),
            ],
            Self::Ba => &[
                ("linear_attn.in_proj_b.weight", 16),
                ("linear_attn.in_proj_a.weight", 16),
            ],
            Self::GdnOut => &[("linear_attn.out_proj.weight", 1024)],
            Self::GateUp => &[("mlp.gate_proj.weight", 3584), ("mlp.up_proj.weight", 3584)],
            Self::Down => &[("mlp.down_proj.weight", 1024)],
        }
    }
}

/// Actual original matrices in original row order; no packed-source tensor name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmGdnProjection {
    invocation: VllmBf16Invocation,
    kind: VllmGdnProjectionKind,
    members: Vec<SourceTensor>,
}
impl VllmGdnProjection {
    pub fn new(
        invocation: VllmBf16Invocation,
        kind: VllmGdnProjectionKind,
        members: Vec<SourceTensor>,
    ) -> Result<Self> {
        let value = Self {
            invocation,
            kind,
            members,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn invocation(&self) -> &VllmBf16Invocation {
        &self.invocation
    }
    pub fn kind(&self) -> VllmGdnProjectionKind {
        self.kind
    }
    pub fn members(&self) -> &[SourceTensor] {
        &self.members
    }
    pub fn contract(&self) -> VllmNativeProjectionContract {
        VllmNativeProjectionContract::Vllm031CudaSm86Bf16V1
    }
    pub fn input_elements(&self) -> usize {
        self.invocation.rows() * self.kind.input_width()
    }
    pub fn output_elements(&self) -> usize {
        self.invocation.rows() * self.kind.output_width()
    }
    pub fn validate(&self) -> Result<()> {
        self.invocation.validate()?;
        let expected = self.kind.members();
        if self.members.len() != expected.len() || self.members.capacity() > 2 {
            return Err(Error::Limit(
                "vLLM exact ordered source member count/capacity".into(),
            ));
        }
        for (member, &(suffix, rows)) in self.members.iter().zip(expected) {
            let name = format!(
                "model.language_model.layers.{}.{suffix}",
                self.invocation.layer.index()
            );
            if member.source != self.invocation.layer.source() || member.name != name {
                return Err(Error::State(
                    "vLLM exact ordered original source roles".into(),
                ));
            }
            if member.kind != QuantKind::Bf16
                || member.axes != MatrixAxes::OutputInput
                || member.columns != self.kind.input_width()
                || member.rows != rows
            {
                return Err(Error::Unsupported(
                    "vLLM original BF16 matrix axes/shape".into(),
                ));
            }
            member.packed_bytes()?;
        }
        Ok(())
    }
    pub fn validate_source(&self, source: &dyn TensorSource) -> Result<()> {
        self.validate()?;
        if source.source_identity() != self.invocation.layer.source() {
            return Err(Error::State(
                "vLLM packed join immutable source issuer".into(),
            ));
        }
        for member in &self.members {
            member.validate_source(source)?;
        }
        Ok(())
    }
}

/// SiLU rounds to BF16 before the widened multiply and final BF16 store.
/// No Rust/CPU replacement equation or generic Silu fallback is defined here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmSiluAndMul {
    invocation: VllmBf16Invocation,
}
impl VllmSiluAndMul {
    pub fn new(invocation: VllmBf16Invocation) -> Result<Self> {
        invocation.validate()?;
        Ok(Self { invocation })
    }
    pub fn invocation(&self) -> &VllmBf16Invocation {
        &self.invocation
    }
    pub fn contract(&self) -> VllmSiluAndMulContract {
        VllmSiluAndMulContract::Vllm031CudaSm86Bf16V1
    }
    pub fn input_elements(&self) -> usize {
        self.invocation.rows() * 7168
    }
    pub fn output_elements(&self) -> usize {
        self.invocation.rows() * 3584
    }
    pub fn validate(&self) -> Result<()> {
        self.invocation.validate()
    }
}

/// Authoritative producer role, minted only while recording a real node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VllmValueOrigin {
    FullModel(Vllm08Origin),
    SourceRows,
    InputNormalized {
        layer: GdnPrefixLayer,
    },
    ResidualNormalized {
        binding: GdnPrefixRmsBinding,
    },
    CarriedResidual {
        binding: GdnPrefixRmsBinding,
    },
    LegacyResidualNormalized,
    LegacyCarriedResidual,
    Projection {
        layer: GdnPrefixLayer,
        kind: VllmGdnProjectionKind,
    },
    Activation {
        layer: GdnPrefixLayer,
    },
    ConvActivated {
        layer: GdnPrefixLayer,
        node_index: usize,
        qkvz: ValueId,
        ba: ValueId,
    },
    ConvZ {
        layer: GdnPrefixLayer,
        node_index: usize,
        qkvz: ValueId,
        ba: ValueId,
    },
    RecurrentGated {
        layer: GdnPrefixLayer,
        conv_node_index: usize,
    },
}
/// Immutable producer span. Optional legacy phase is metadata, not coverage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmValueProvenance {
    teachers: VllmBf16Teachers,
    phase: Option<VllmBf16Phase>,
    mode: Option<VllmBf16Mode>,
    origin: VllmValueOrigin,
}
impl VllmValueProvenance {
    pub(super) fn bind_short_hf_call(&mut self, call: &Vllm08Call) -> Result<()> {
        self.teachers = call.request().teachers().clone();
        self.phase = call.request().legacy_phase().ok();
        self.mode = Some(call.request().mode());
        match &mut self.origin {
            VllmValueOrigin::SourceRows => {}
            VllmValueOrigin::FullModel(origin) => origin.bind_call(call.clone()),
            _ => {
                return Err(Error::State(
                    "prepared full-model producer origin differs".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn with_origin(mut self, origin: VllmValueOrigin) -> Self {
        self.origin = origin;
        self
    }

    fn embedding(builder: &PlanBuilder, teachers: VllmBf16Teachers) -> Result<Self> {
        if let Some(call) = &builder.vllm_model {
            if call.request().teachers() != &teachers || call.source() != teachers.source() {
                return Err(Error::State(
                    "full-model embedding differs from builder span issuer".into(),
                ));
            }
            let phase = call.request().legacy_phase().ok();
            return Ok(Self {
                teachers,
                phase,
                mode: Some(call.request().mode()),
                origin: VllmValueOrigin::SourceRows,
            });
        }
        let phase = match (teachers.base(), teachers.len()) {
            (0, 1) => Some(VllmBf16Phase::InitialOne),
            (1 | 66, 1) => Some(VllmBf16Phase::CachedOne),
            (0, 64) => Some(VllmBf16Phase::Initial64),
            (64, 2) => Some(VllmBf16Phase::ContinuedTwo),
            _ => None,
        };
        Ok(Self {
            teachers,
            phase,
            mode: phase.map(VllmBf16Phase::mode),
            origin: VllmValueOrigin::SourceRows,
        })
    }
    pub fn origin(&self) -> &VllmValueOrigin {
        &self.origin
    }
    pub fn teachers(&self) -> &VllmBf16Teachers {
        &self.teachers
    }
    pub fn phase(&self) -> Option<VllmBf16Phase> {
        self.phase
    }
    pub fn mode(&self) -> Option<VllmBf16Mode> {
        self.mode
    }
    pub fn require_invocation(&self, invocation: &VllmBf16Invocation) -> Result<()> {
        invocation.validate()?;
        if self.teachers != invocation.teachers || self.mode != Some(invocation.mode) {
            return Err(Error::State(
                "vLLM actual producer invocation differs from descriptor".into(),
            ));
        }
        Ok(())
    }
}

/// Shared with upcoming typed GDN nodes; only validated existing values can
/// supply a tag. No graph walk, caller tag setter or source-identity minting.
pub(super) fn same_input_provenance(
    builder: &PlanBuilder,
    inputs: &[PlanValue],
) -> Result<VllmValueProvenance> {
    let first = inputs
        .first()
        .ok_or_else(|| Error::Malformed("vLLM provenance inputs empty".into()))?;
    let expected = builder
        .vllm_value_provenance(first)?
        .ok_or_else(|| Error::State("vLLM producer teacher provenance absent".into()))?;
    if expected.teachers.source() != builder.source {
        return Err(Error::State("vLLM producer source issuer differs".into()));
    }
    for input in inputs {
        let actual = builder
            .vllm_value_provenance(input)?
            .ok_or_else(|| Error::State("vLLM producer teacher provenance absent".into()))?;
        if actual.teachers != expected.teachers
            || actual.phase != expected.phase
            || actual.mode != expected.mode
        {
            return Err(Error::State(
                "vLLM producer teacher issuer/span/phase differs".into(),
            ));
        }
    }
    Ok(expected.clone())
}
pub(super) fn require_invocation(
    builder: &PlanBuilder,
    invocation: &VllmBf16Invocation,
    inputs: &[PlanValue],
) -> Result<VllmValueProvenance> {
    invocation.validate()?;
    let tag = same_input_provenance(builder, inputs)?;
    tag.require_invocation(invocation)?;
    Ok(tag)
}
pub(super) fn result_provenance(
    builder: &PlanBuilder,
    operation: &PlanOperation,
    inputs: &[PlanValue],
    output_index: usize,
) -> Result<Option<VllmValueProvenance>> {
    let tag_origin = |mut tag: VllmValueProvenance, origin| {
        tag.origin = origin;
        Some(tag)
    };
    Ok(match operation {
        PlanOperation::TypedVllm08 { descriptor, .. } => Some(vllm_model::provenance(
            builder,
            descriptor,
            inputs,
            output_index,
        )?),
        PlanOperation::TypedVllmBf16Embedding { descriptor } => Some(
            VllmValueProvenance::embedding(builder, descriptor.teachers().clone())?,
        ),
        PlanOperation::TypedVllmInputRms { descriptor } => tag_origin(
            require_invocation(builder, descriptor.invocation(), inputs)?,
            VllmValueOrigin::InputNormalized {
                layer: descriptor.invocation().layer(),
            },
        ),
        PlanOperation::TypedResidualRms { contract, .. } => {
            let tag = same_input_provenance(builder, inputs)?;
            let origin = match (*contract, output_index) {
                (ResidualRmsContract::Vllm031CudaGdnPrefix(binding), 0) => {
                    VllmValueOrigin::ResidualNormalized { binding }
                }
                (ResidualRmsContract::Vllm031CudaGdnPrefix(binding), 1) => {
                    VllmValueOrigin::CarriedResidual { binding }
                }
                (_, 0) => VllmValueOrigin::LegacyResidualNormalized,
                (_, 1) => VllmValueOrigin::LegacyCarriedResidual,
                _ => return Err(Error::State("dual RMS output role".into())),
            };
            tag_origin(tag, origin)
        }
        PlanOperation::TypedVllmGdnProjection { descriptor } => tag_origin(
            require_invocation(builder, descriptor.invocation(), inputs)?,
            VllmValueOrigin::Projection {
                layer: descriptor.invocation().layer(),
                kind: descriptor.kind(),
            },
        ),
        PlanOperation::TypedVllmSiluAndMul { descriptor } => tag_origin(
            require_invocation(builder, descriptor.invocation(), inputs)?,
            VllmValueOrigin::Activation {
                layer: descriptor.invocation().layer(),
            },
        ),
        PlanOperation::TypedVllmGdnConv { descriptor } => {
            let tag = require_invocation(builder, descriptor.invocation(), inputs)?;
            let [qkvz, ba] = inputs else {
                return Err(Error::State("Conv origin arity".into()));
            };
            let origin = match output_index {
                0 => VllmValueOrigin::ConvActivated {
                    layer: descriptor.invocation().layer(),
                    node_index: builder.nodes.len(),
                    qkvz: qkvz.id,
                    ba: ba.id,
                },
                1 => VllmValueOrigin::ConvZ {
                    layer: descriptor.invocation().layer(),
                    node_index: builder.nodes.len(),
                    qkvz: qkvz.id,
                    ba: ba.id,
                },
                _ => return Err(Error::State("Conv output role".into())),
            };
            tag_origin(tag, origin)
        }
        PlanOperation::TypedVllmGdnRecurrent { descriptor } => {
            let tag = require_invocation(builder, descriptor.invocation(), inputs)?;
            let VllmValueOrigin::ConvActivated { node_index, .. } = builder
                .vllm_value_provenance(&inputs[0])?
                .ok_or_else(|| Error::State("Conv origin absent".into()))?
                .origin()
            else {
                return Err(Error::State("actual Conv origin required".into()));
            };
            tag_origin(
                tag,
                VllmValueOrigin::RecurrentGated {
                    layer: descriptor.invocation().layer(),
                    conv_node_index: *node_index,
                },
            )
        }
        _ => None,
    })
}

fn input_precision(
    builder: &PlanBuilder,
    invocation: &VllmBf16Invocation,
    input: &[PlanValue],
    elements: usize,
) -> Result<ActivationPrecision> {
    invocation.validate()?;
    if !(builder.operator_slice || builder.vllm_fragment.is_some())
        || builder.profile != ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1
        || builder.source != invocation.layer.source()
        || input.len() != 1
        || input[0].precision != ActivationPrecision::Bf16WidenedF32
        || input[0].shape != ValueShape::Vector(elements)
        || input[0].elements != elements
    {
        return Err(Error::Unsupported(
            "vLLM native profile/source/BF16/dense input".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn projection_precision(
    builder: &PlanBuilder,
    descriptor: &VllmGdnProjection,
    input: &[PlanValue],
) -> Result<ActivationPrecision> {
    descriptor.validate()?;
    input_precision(
        builder,
        descriptor.invocation(),
        input,
        descriptor.input_elements(),
    )
}
pub(super) fn activation_precision(
    builder: &PlanBuilder,
    descriptor: &VllmSiluAndMul,
    input: &[PlanValue],
) -> Result<ActivationPrecision> {
    descriptor.validate()?;
    input_precision(
        builder,
        descriptor.invocation(),
        input,
        descriptor.input_elements(),
    )
}
pub(super) fn record_projection(
    builder: &mut PlanBuilder,
    descriptor: &VllmGdnProjection,
    input: &PlanValue,
) -> Result<PlanValue> {
    descriptor.validate()?;
    for member in descriptor.members() {
        builder.source_tensor(member)?;
    }
    builder.one(
        "vllm.original.ordered_projection",
        PlanOperation::TypedVllmGdnProjection {
            descriptor: Box::new(descriptor.clone()),
        },
        &[*input],
        ValueShape::Vector(descriptor.output_elements()),
        None,
    )
}
pub(super) fn record_activation(
    builder: &mut PlanBuilder,
    descriptor: &VllmSiluAndMul,
    input: &PlanValue,
) -> Result<PlanValue> {
    descriptor.validate()?;
    builder.one(
        "vllm.original.silu_and_mul",
        PlanOperation::TypedVllmSiluAndMul {
            descriptor: Box::new(descriptor.clone()),
        },
        &[*input],
        ValueShape::Vector(descriptor.output_elements()),
        None,
    )
}

#[cfg(test)]
mod tests;
