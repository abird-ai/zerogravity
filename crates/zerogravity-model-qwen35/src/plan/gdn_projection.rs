//! Source-member/span metadata; finite native coverage is a separate binding.
use super::*;

fn reference_phase_for(mode: SourceCallMode, base: usize, rows: usize) -> Option<ReferencePhase> {
    match (mode, base, rows) {
        (SourceCallMode::Prefill, 0, 1) => Some(ReferencePhase::InitialOne),
        (SourceCallMode::Prefill, 0, 64) => Some(ReferencePhase::Initial64),
        (SourceCallMode::Prefill, 64, 2) => Some(ReferencePhase::ContinuedTwo),
        (SourceCallMode::Decode, 1 | 66, 1) => Some(ReferencePhase::CachedOne),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferencePhase {
    InitialOne,
    CachedOne,
    Initial64,
    ContinuedTwo,
}
impl ReferencePhase {
    pub fn mode(self) -> SourceCallMode {
        match self {
            Self::CachedOne => SourceCallMode::Decode,
            _ => SourceCallMode::Prefill,
        }
    }
    pub fn rows(self) -> usize {
        match self {
            Self::InitialOne | Self::CachedOne => 1,
            Self::Initial64 => 64,
            Self::ContinuedTwo => 2,
        }
    }
    fn validate(self, teachers: &QwenInputRows) -> Result<()> {
        let base = teachers.base();
        if teachers.prompt_span().is_some()
            || teachers.len() != self.rows()
            || !match self {
                Self::InitialOne | Self::Initial64 => base == 0,
                Self::CachedOne => matches!(base, 1 | 66),
                Self::ContinuedTwo => base == 64,
            }
        {
            return Err(Error::Unsupported(
                "closed reference CUDA genuine phase/span1/64/2".into(),
            ));
        }
        Ok(())
    }
}

/// Share this descriptor's teacher issuer through the SAME plan producer chain.
/// The physical lowerer must also validate its model/config/current invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenGdnCall {
    layer: QwenPrefixLayer,
    mode: SourceCallMode,
    teachers: QwenInputRows,
}
impl QwenGdnCall {
    pub(super) fn from_prompt_span(span: PromptSpan) -> Result<Self> {
        span.validate()?;
        let value = Self {
            layer: QwenPrefixLayer::new(span.source(), 0)?,
            mode: span.mode(),
            teachers: QwenInputRows::from_prompt_span(span)?,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn new(
        layer: QwenPrefixLayer,
        phase: ReferencePhase,
        teachers: QwenInputRows,
    ) -> Result<Self> {
        if layer.source() != teachers.source() {
            return Err(Error::State(
                "reference CUDA invocation source/teacher issuer".into(),
            ));
        }
        phase.validate(&teachers)?;
        Self::from_span(layer, phase.mode(), teachers)
    }
    /// Semantic metadata only. Existing backends still require finite coverage.
    pub fn from_span(
        layer: QwenPrefixLayer,
        mode: SourceCallMode,
        teachers: QwenInputRows,
    ) -> Result<Self> {
        if teachers.prompt_span().is_some() {
            return Err(Error::Unsupported(
                "bounded prompt issuer requires its source-profile call".into(),
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
    pub fn source_call(&self) -> Result<SourceCall> {
        self.teachers.source_call(self.mode)
    }
    /// Finite reference layout only; actual producer/root issuers stay separate.
    pub fn require_reference_source_call(&self, actual: &SourceCall) -> Result<ReferencePhase> {
        self.reference_phase()?;
        if actual.source() != self.layer.source()
            || actual.rows().capacity() != self.teachers.capacity()
            || actual.rows().vocabulary() != 248_320
        {
            return Err(Error::State(
                "reference source call source/capacity/vocabulary differs".into(),
            ));
        }
        reference_phase_for(actual.mode(), actual.rows().base(), actual.rows().len())
            .ok_or_else(|| Error::Unsupported("source call has no finite reference phase".into()))
    }
    pub fn layer(&self) -> QwenPrefixLayer {
        self.layer
    }
    pub fn mode(&self) -> SourceCallMode {
        self.mode
    }
    /// Exact old finite metadata only; never infer coverage for another span.
    pub fn reference_phase(&self) -> Result<ReferencePhase> {
        self.validate()?;
        if self.teachers.prompt_span().is_some() {
            return Err(Error::Unsupported(
                "bounded prompt span has no legacy reference CUDA native phase".into(),
            ));
        }
        reference_phase_for(self.mode, self.teachers.base(), self.rows()).ok_or_else(|| {
            Error::Unsupported("span has no legacy reference CUDA native phase".into())
        })
    }
    pub fn teachers(&self) -> &QwenInputRows {
        &self.teachers
    }
    pub fn rows(&self) -> usize {
        self.teachers.len()
    }
    pub fn validate(&self) -> Result<()> {
        if self.layer.source() != self.teachers.source() {
            return Err(Error::State(
                "reference CUDA invocation source changed".into(),
            ));
        }
        let t = &self.teachers;
        if let Some(span) = t.prompt_span() {
            span.validate()?;
            if self.layer.index() != 0 || self.mode != span.mode() {
                return Err(Error::State(
                    "bounded prompt invocation scope changed".into(),
                ));
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
            || self.mode == SourceCallMode::Decode && (t.len() != 1 || t.base() == 0)
        {
            return Err(Error::Malformed(
                "reference CUDA semantic source/span/mode".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenProjectionContract {
    ReferenceCudaSm86Bf16V1,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenSiluContract {
    ReferenceCudaSm86Bf16V1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnProjection {
    Qkvz,
    Ba,
    GdnOut,
    GateUp,
    Down,
}
impl GdnProjection {
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
pub struct QwenGdnProjection {
    invocation: QwenGdnCall,
    kind: GdnProjection,
    members: Vec<SourceTensor>,
}
impl QwenGdnProjection {
    pub fn new(
        invocation: QwenGdnCall,
        kind: GdnProjection,
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
    pub fn invocation(&self) -> &QwenGdnCall {
        &self.invocation
    }
    pub fn kind(&self) -> GdnProjection {
        self.kind
    }
    pub fn members(&self) -> &[SourceTensor] {
        &self.members
    }
    pub fn contract(&self) -> QwenProjectionContract {
        QwenProjectionContract::ReferenceCudaSm86Bf16V1
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
                "reference CUDA exact ordered source member count/capacity".into(),
            ));
        }
        for (member, &(suffix, rows)) in self.members.iter().zip(expected) {
            let name = format!(
                "model.language_model.layers.{}.{suffix}",
                self.invocation.layer.index()
            );
            if member.source != self.invocation.layer.source() || member.name != name {
                return Err(Error::State(
                    "reference CUDA exact ordered original source roles".into(),
                ));
            }
            if member.kind != QuantKind::Bf16
                || member.axes != MatrixAxes::OutputInput
                || member.columns != self.kind.input_width()
                || member.rows != rows
            {
                return Err(Error::Unsupported(
                    "reference CUDA original BF16 matrix axes/shape".into(),
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
                "reference CUDA packed join immutable source issuer".into(),
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
pub struct QwenSiluAndMul {
    invocation: QwenGdnCall,
}
impl QwenSiluAndMul {
    pub fn new(invocation: QwenGdnCall) -> Result<Self> {
        invocation.validate()?;
        Ok(Self { invocation })
    }
    pub fn invocation(&self) -> &QwenGdnCall {
        &self.invocation
    }
    pub fn contract(&self) -> QwenSiluContract {
        QwenSiluContract::ReferenceCudaSm86Bf16V1
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
pub enum QwenValueOrigin {
    FullModel(QwenProducer),
    SourceRows,
    InputNormalized {
        layer: QwenPrefixLayer,
    },
    ResidualNormalized {
        binding: QwenPrefixRmsBinding,
    },
    CarriedResidual {
        binding: QwenPrefixRmsBinding,
    },
    LegacyResidualNormalized,
    LegacyCarriedResidual,
    Projection {
        layer: QwenPrefixLayer,
        kind: GdnProjection,
    },
    Activation {
        layer: QwenPrefixLayer,
    },
    ConvActivated {
        layer: QwenPrefixLayer,
        node_index: usize,
        qkvz: ValueId,
        ba: ValueId,
    },
    ConvZ {
        layer: QwenPrefixLayer,
        node_index: usize,
        qkvz: ValueId,
        ba: ValueId,
    },
    RecurrentGated {
        layer: QwenPrefixLayer,
        conv_node_index: usize,
    },
}
/// Immutable producer span. Optional legacy phase is metadata, not coverage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenValueProvenance {
    teachers: QwenInputRows,
    phase: Option<ReferencePhase>,
    mode: Option<SourceCallMode>,
    origin: QwenValueOrigin,
}
impl QwenValueProvenance {
    pub(super) fn bind_prompt_call(&mut self, call: &QwenSourceCall) -> Result<()> {
        self.teachers = call.request().teachers().clone();
        self.phase = call.request().reference_phase().ok();
        self.mode = Some(call.request().mode());
        match &mut self.origin {
            QwenValueOrigin::SourceRows => {}
            QwenValueOrigin::FullModel(origin) => origin.bind_call(call.clone()),
            _ => {
                return Err(Error::State(
                    "prepared full-model producer origin differs".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn with_origin(mut self, origin: QwenValueOrigin) -> Self {
        self.origin = origin;
        self
    }

    pub(super) fn embedding(builder: &QwenPlanBuilder, teachers: QwenInputRows) -> Result<Self> {
        if let Some(call) = &builder.model_call {
            if call.request().teachers() != &teachers || call.source() != teachers.source() {
                return Err(Error::State(
                    "full-model embedding differs from builder span issuer".into(),
                ));
            }
            let phase = call.request().reference_phase().ok();
            return Ok(Self {
                teachers,
                phase,
                mode: Some(call.request().mode()),
                origin: QwenValueOrigin::SourceRows,
            });
        }
        let phase = match (teachers.base(), teachers.len()) {
            (0, 1) => Some(ReferencePhase::InitialOne),
            (1 | 66, 1) => Some(ReferencePhase::CachedOne),
            (0, 64) => Some(ReferencePhase::Initial64),
            (64, 2) => Some(ReferencePhase::ContinuedTwo),
            _ => None,
        };
        Ok(Self {
            teachers,
            phase,
            mode: phase.map(ReferencePhase::mode),
            origin: QwenValueOrigin::SourceRows,
        })
    }
    pub fn origin(&self) -> &QwenValueOrigin {
        &self.origin
    }
    pub fn teachers(&self) -> &QwenInputRows {
        &self.teachers
    }
    pub fn phase(&self) -> Option<ReferencePhase> {
        self.phase
    }
    pub fn mode(&self) -> Option<SourceCallMode> {
        self.mode
    }
    pub fn require_invocation(&self, invocation: &QwenGdnCall) -> Result<()> {
        invocation.validate()?;
        if self.teachers != invocation.teachers || self.mode != Some(invocation.mode) {
            return Err(Error::State(
                "reference CUDA actual producer invocation differs from descriptor".into(),
            ));
        }
        Ok(())
    }
}

/// Shared with upcoming typed GDN nodes; only validated existing values can
/// supply a tag. No graph walk, caller tag setter or source-identity minting.
pub(super) fn same_input_provenance(
    builder: &QwenPlanBuilder,
    inputs: &[PlanValue],
) -> Result<QwenValueProvenance> {
    let first = inputs
        .first()
        .ok_or_else(|| Error::Malformed("reference CUDA provenance inputs empty".into()))?;
    let expected = builder
        .value_provenance(first)?
        .ok_or_else(|| Error::State("reference CUDA producer teacher provenance absent".into()))?;
    if expected.teachers.source() != builder.source {
        return Err(Error::State(
            "reference CUDA producer source issuer differs".into(),
        ));
    }
    for input in inputs {
        let actual = builder.value_provenance(input)?.ok_or_else(|| {
            Error::State("reference CUDA producer teacher provenance absent".into())
        })?;
        if actual.teachers != expected.teachers
            || actual.phase != expected.phase
            || actual.mode != expected.mode
        {
            return Err(Error::State(
                "reference CUDA producer teacher issuer/span/phase differs".into(),
            ));
        }
    }
    Ok(expected.clone())
}
pub(super) fn require_invocation(
    builder: &QwenPlanBuilder,
    invocation: &QwenGdnCall,
    inputs: &[PlanValue],
) -> Result<QwenValueProvenance> {
    invocation.validate()?;
    let tag = same_input_provenance(builder, inputs)?;
    tag.require_invocation(invocation)?;
    Ok(tag)
}
fn input_precision(
    builder: &QwenPlanBuilder,
    invocation: &QwenGdnCall,
    input: &[PlanValue],
    elements: usize,
) -> Result<ActivationPrecision> {
    invocation.validate()?;
    if !(builder.operator_slice || builder.fragment_call.is_some())
        || builder.profile != ProgramNumericalProfile::SourceCudaSm86Bf16V1
        || builder.source != invocation.layer.source()
        || input.len() != 1
        || input[0].precision() != ActivationPrecision::Bf16WidenedF32
        || input[0].shape != ValueShape::Vector(elements)
        || input[0].elements() != elements
    {
        return Err(Error::Unsupported(
            "reference CUDA native profile/source/BF16/dense input".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn projection_precision(
    builder: &QwenPlanBuilder,
    descriptor: &QwenGdnProjection,
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
    builder: &QwenPlanBuilder,
    descriptor: &QwenSiluAndMul,
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
    builder: &mut QwenPlanBuilder,
    descriptor: &QwenGdnProjection,
    input: &PlanValue,
) -> Result<PlanValue> {
    projection_precision(builder, descriptor, &[*input])?;
    let tag = require_invocation(builder, descriptor.invocation(), &[*input])?.with_origin(
        QwenValueOrigin::Projection {
            layer: descriptor.invocation().layer(),
            kind: descriptor.kind(),
        },
    );
    QwenPlanBuilder::one_output(builder.record_source(
        "qwen.source.ordered_projection",
        builder.projection_operator(descriptor)?,
        &[*input],
        vec![tag],
    )?)
}
pub(super) fn record_activation(
    builder: &mut QwenPlanBuilder,
    descriptor: &QwenSiluAndMul,
    input: &PlanValue,
) -> Result<PlanValue> {
    activation_precision(builder, descriptor, &[*input])?;
    let tag = require_invocation(builder, descriptor.invocation(), &[*input])?.with_origin(
        QwenValueOrigin::Activation {
            layer: descriptor.invocation().layer(),
        },
    );
    QwenPlanBuilder::one_output(builder.record_source(
        "qwen.source.silu_and_mul",
        builder.activation_operator(descriptor)?,
        &[*input],
        vec![tag],
    )?)
}

#[cfg(test)]
#[path = "gdn_projection/tests.rs"]
mod tests;
