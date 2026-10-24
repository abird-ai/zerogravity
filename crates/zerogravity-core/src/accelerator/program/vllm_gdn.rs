//! Closed GDN source records. State authority belongs to the original session.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmInputRms {
    invocation: VllmBf16Invocation,
    parameter: SourceParameter,
}
impl VllmInputRms {
    pub fn new(invocation: VllmBf16Invocation, parameter: SourceParameter) -> Result<Self> {
        let value = Self {
            invocation,
            parameter,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn invocation(&self) -> &VllmBf16Invocation {
        &self.invocation
    }
    pub fn parameter(&self) -> &SourceParameter {
        &self.parameter
    }
    pub fn validate(&self) -> Result<()> {
        self.invocation.validate()?;
        if self.invocation.layer().index() != 0 {
            return Err(Error::Unsupported(
                "ordinary source input RMS is layer0; later layers require pending dual RMS".into(),
            ));
        }
        let bytes = match self.parameter.kind {
            QuantKind::Bf16 => 2048,
            QuantKind::F32 => 4096,
            _ => {
                return Err(Error::Unsupported(
                    "vLLM input RMS raw BF16/F32 gamma".into(),
                ));
            }
        };
        parameter(
            &self.invocation,
            &self.parameter,
            "input_layernorm.weight",
            self.parameter.kind,
            &[1024],
            bytes,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmGdnConv {
    invocation: VllmBf16Invocation,
    parameter: SourceParameter,
}
impl VllmGdnConv {
    pub fn new(invocation: VllmBf16Invocation, parameter: SourceParameter) -> Result<Self> {
        let value = Self {
            invocation,
            parameter,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn invocation(&self) -> &VllmBf16Invocation {
        &self.invocation
    }
    pub fn parameter(&self) -> &SourceParameter {
        &self.parameter
    }
    pub fn validate(&self) -> Result<()> {
        self.invocation.validate()?;
        parameter(
            &self.invocation,
            &self.parameter,
            "linear_attn.conv1d.weight",
            QuantKind::Bf16,
            &[6144, 1, 4],
            49_152,
        )
    }
}
#[derive(Clone)]
pub struct VllmGdnConvValues<V> {
    pub activated: V,
    pub z: V,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VllmGdnRecurrent {
    invocation: VllmBf16Invocation,
    parameters: [SourceParameter; 3],
}
impl VllmGdnRecurrent {
    pub fn new(invocation: VllmBf16Invocation, parameters: [SourceParameter; 3]) -> Result<Self> {
        let value = Self {
            invocation,
            parameters,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn invocation(&self) -> &VllmBf16Invocation {
        &self.invocation
    }
    /// Original A_log F32, dt_bias BF16 and raw norm.weight F32, in that order.
    pub fn parameters(&self) -> &[SourceParameter; 3] {
        &self.parameters
    }
    pub fn validate(&self) -> Result<()> {
        self.invocation.validate()?;
        for (p, (suffix, kind, shape, bytes)) in self.parameters.iter().zip([
            ("linear_attn.A_log", QuantKind::F32, 16, 64),
            ("linear_attn.dt_bias", QuantKind::Bf16, 16, 32),
            ("linear_attn.norm.weight", QuantKind::F32, 128, 512),
        ]) {
            parameter(&self.invocation, p, suffix, kind, &[shape], bytes)?;
        }
        Ok(())
    }
}
fn parameter(
    invocation: &VllmBf16Invocation,
    p: &SourceParameter,
    suffix: &str,
    kind: QuantKind,
    shape: &[u64],
    bytes: usize,
) -> Result<()> {
    if p.source != invocation.layer().source()
        || p.name
            != format!(
                "model.language_model.layers.{}.{suffix}",
                invocation.layer().index()
            )
    {
        return Err(Error::State(
            "GDN exact original parameter source/layer/role".into(),
        ));
    }
    if p.kind != kind || p.shape != shape || p.bytes != bytes {
        return Err(Error::Unsupported(
            "GDN raw original parameter dtype/shape/extent".into(),
        ));
    }
    Ok(())
}
fn require_dense(
    builder: &PlanBuilder,
    invocation: &VllmBf16Invocation,
    inputs: &[PlanValue],
    widths: &[usize],
) -> Result<()> {
    invocation.validate()?;
    if builder.profile != ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1
        || builder.source != invocation.layer().source()
        || !(builder.operator_slice || builder.vllm_fragment.is_some())
        || inputs.len() != widths.len()
    {
        return Err(Error::Unsupported(
            "GDN closed BF16 source/profile/scope/arity".into(),
        ));
    }
    for (input, width) in inputs.iter().zip(widths) {
        let elements = invocation.rows() * width;
        if input.precision != ActivationPrecision::Bf16WidenedF32
            || input.shape != ValueShape::Vector(elements)
            || input.elements != elements
        {
            return Err(Error::Unsupported(
                "GDN exact dense BF16 operand extent".into(),
            ));
        }
    }
    vllm_native::require_invocation(builder, invocation, inputs)?;
    Ok(())
}
pub(super) fn input_rms_precision(
    builder: &PlanBuilder,
    d: &VllmInputRms,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    d.validate()?;
    require_dense(builder, d.invocation(), inputs, &[1024])?;
    if builder
        .vllm_value_provenance(&inputs[0])?
        .map(VllmValueProvenance::origin)
        != Some(&VllmValueOrigin::SourceRows)
    {
        return Err(Error::State(
            "input RMS must consume original raw source rows".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
fn fragment_invocation(builder: &PlanBuilder, invocation: &VllmBf16Invocation) -> Result<bool> {
    let Some(base) = builder.vllm_fragment.as_ref() else {
        return Ok(false);
    };
    Ok(base.teachers() == invocation.teachers()
        && base.legacy_phase()? == invocation.legacy_phase()?
        && base.layer().source() == invocation.layer().source()
        && invocation.layer().index() <= 2)
}
pub(super) fn conv_precision(
    builder: &PlanBuilder,
    d: &VllmGdnConv,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    d.validate()?;
    if !fragment_invocation(builder, d.invocation())? {
        return Err(Error::Unsupported(
            "stateful GDN requires closed original-root fragment".into(),
        ));
    }
    require_dense(builder, d.invocation(), inputs, &[8192, 32])?;
    for (input, kind) in inputs
        .iter()
        .zip([VllmGdnProjectionKind::Qkvz, VllmGdnProjectionKind::Ba])
    {
        if builder
            .vllm_value_provenance(input)?
            .map(VllmValueProvenance::origin)
            != Some(&VllmValueOrigin::Projection {
                layer: d.invocation().layer(),
                kind,
            })
        {
            return Err(Error::State(
                "GDN must consume exact original same-layer QKVZ/BA producer roles".into(),
            ));
        }
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn recurrent_precision(
    builder: &PlanBuilder,
    d: &VllmGdnRecurrent,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    d.validate()?;
    if !fragment_invocation(builder, d.invocation())? {
        return Err(Error::Unsupported(
            "stateful GDN requires closed original-root fragment".into(),
        ));
    }
    require_dense(builder, d.invocation(), inputs, &[6144, 2048, 32])?;
    let tags = inputs
        .iter()
        .map(|v| {
            builder
                .vllm_value_provenance(v)?
                .ok_or_else(|| Error::State("GDN producer tag absent".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let (
        VllmValueOrigin::ConvActivated {
            layer,
            node_index,
            qkvz,
            ba,
        },
        VllmValueOrigin::ConvZ {
            layer: z_layer,
            node_index: z_node,
            qkvz: z_qkvz,
            ba: z_ba,
        },
    ) = (tags[0].origin(), tags[1].origin())
    else {
        return Err(Error::State(
            "GDN recurrent requires actual Conv activated/Z outputs".into(),
        ));
    };
    if *layer != d.invocation().layer()
        || layer != z_layer
        || node_index != z_node
        || qkvz != z_qkvz
        || ba != z_ba
        || *ba != inputs[2].id
        || tags[2].origin()
            != &(VllmValueOrigin::Projection {
                layer: *layer,
                kind: VllmGdnProjectionKind::Ba,
            })
    {
        return Err(Error::State(
            "GDN exact Conv cohort and original BA ValueID".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn record_input_rms(
    builder: &mut PlanBuilder,
    d: &VllmInputRms,
    input: &PlanValue,
) -> Result<PlanValue> {
    d.validate()?;
    builder.one(
        "vllm.original.input_rms",
        PlanOperation::TypedVllmInputRms {
            descriptor: Box::new(d.clone()),
        },
        &[*input],
        input.shape,
        None,
    )
}
pub(super) fn record_conv(
    builder: &mut PlanBuilder,
    d: &VllmGdnConv,
    qkvz: &PlanValue,
    ba: &PlanValue,
) -> Result<VllmGdnConvValues<PlanValue>> {
    d.validate()?;
    let effect = builder.effect(StateSlot::Conv(ConvDomain(d.invocation().layer().index())))?;
    let values = builder.record(
        "vllm.original.gdn.conv",
        PlanOperation::TypedVllmGdnConv {
            descriptor: Box::new(d.clone()),
        },
        &[*qkvz, *ba],
        &[
            ValueShape::Vector(d.invocation().rows() * 6144),
            ValueShape::Vector(d.invocation().rows() * 2048),
        ],
        Some(effect),
    )?;
    let [activated, z]: [PlanValue; 2] = values
        .try_into()
        .map_err(|_| Error::State("GDN Conv output arity".into()))?;
    Ok(VllmGdnConvValues { activated, z })
}
pub(super) fn record_recurrent(
    builder: &mut PlanBuilder,
    d: &VllmGdnRecurrent,
    activated: &PlanValue,
    z: &PlanValue,
    ba: &PlanValue,
) -> Result<PlanValue> {
    d.validate()?;
    let effect = builder.effect(StateSlot::Recurrent(RecurrentDomain(
        d.invocation().layer().index(),
    )))?;
    builder.one(
        "vllm.original.gdn.recurrent_gated",
        PlanOperation::TypedVllmGdnRecurrent {
            descriptor: Box::new(d.clone()),
        },
        &[*activated, *z, *ba],
        ValueShape::Vector(d.invocation().rows() * 2048),
        Some(effect),
    )
}

/// Unpublishable physical fragment: no logits, frontier advance or model Ready.
#[derive(Clone, Debug)]
pub struct RecordedVllmGdnFragment {
    plan: ForwardPlan,
    output: ValueId,
    residual_output: Option<ValueId>,
    completed_layers: usize,
}
impl RecordedVllmGdnFragment {
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn output(&self) -> ValueId {
        self.output
    }
    /// Zero is the original gated-only cut; one/three are complete MLP cuts.
    pub fn completed_layers(&self) -> usize {
        self.completed_layers
    }
    pub fn residual_output(&self) -> Option<ValueId> {
        self.residual_output
    }
    pub fn into_parts(self) -> (ForwardPlan, ValueId) {
        (self.plan, self.output)
    }
}
fn all_domains() -> Result<Vec<StateSlot>> {
    hybrid_state_domains((0..24).map(|index| index % 4 == 3))
}

impl PlanBuilder {
    /// Metadata recipe only. Real lowerer must bind the SAME original all42 root.
    pub fn new_vllm_gdn_fragment(invocation: VllmBf16Invocation) -> Result<Self> {
        invocation.validate()?;
        if invocation.layer().index() != 0
            || !matches!(
                invocation.legacy_phase()?,
                VllmBf16Phase::InitialOne | VllmBf16Phase::Initial64
            )
            || invocation.teachers().capacity() != 512
        {
            return Err(Error::Unsupported(
                "initial layer0 original-root fragment only; no cached/root promotion".into(),
            ));
        }
        Self::new_vllm_fragment_inner(invocation)
    }
    /// Metadata only for a private phase of ONE original unpublished attempt.
    /// Existing initial and full-model execution admission remain separate.
    pub fn new_vllm_gdn_attempt_phase(invocation: VllmBf16Invocation) -> Result<Self> {
        invocation.validate()?;
        invocation.legacy_phase()?;
        if invocation.layer().index() != 0 || invocation.teachers().capacity() != 512 {
            return Err(Error::Unsupported(
                "closed layer0 request phase metadata only".into(),
            ));
        }
        Self::new_vllm_fragment_inner(invocation)
    }
    fn new_vllm_fragment_inner(invocation: VllmBf16Invocation) -> Result<Self> {
        let mut builder = Self::new_inner(
            invocation.layer().source(),
            512,
            all_domains()?,
            false,
            ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
            false,
        )?;
        builder.vllm_fragment = Some(invocation);
        Ok(builder)
    }
    pub fn finish_vllm_gdn_fragment(self, output: PlanValue) -> Result<RecordedVllmGdnFragment> {
        let invocation = self
            .vllm_fragment
            .clone()
            .ok_or_else(|| Error::State("not an original-root GDN fragment".into()))?;
        self.vllm_value_provenance(&output)?
            .ok_or_else(|| Error::State("fragment output provenance absent".into()))?
            .require_invocation(&invocation)?;
        if self.advanced.is_some()
            || self.operator_slice
            || self.required != all_domains()?.into_iter().collect()
            || self.domains.len() != 2
            || self.current(StateSlot::Conv(ConvDomain(0)))?.ordinal != 1
            || self
                .current(StateSlot::Recurrent(RecurrentDomain(0)))?
                .ordinal
                != 1
            || !matches!(self.vllm_value_provenance(&output)?.map(VllmValueProvenance::origin), Some(VllmValueOrigin::RecurrentGated {layer, ..}) if *layer == invocation.layer())
        {
            return Err(Error::State(
                "fragment requires exact Conv/recurrent pair and actual Gated output".into(),
            ));
        }
        self.finish_fragment(output, None, 0)
    }
    /// SAME all42 authority, complete initial layer0 or initial GDN prefix0..2.
    /// Pending MLP branch and carried BF16 residual remain distinct outputs.
    pub fn finish_vllm_gdn_mlp_fragment(
        self,
        output: PlanValue,
        residual: PlanValue,
        completed_layers: usize,
    ) -> Result<RecordedVllmGdnFragment> {
        let invocation = self
            .vllm_fragment
            .as_ref()
            .ok_or_else(|| Error::State("not an original-root GDN fragment".into()))?;
        if !matches!(completed_layers, 1 | 3)
            || output.id == residual.id
            || self.operator_slice
            || self.advanced.is_some()
            || self.required != all_domains()?.into_iter().collect()
            || self.domains.len() != completed_layers * 2
        {
            return Err(Error::Unsupported(
                "closed complete initial GDN layer0/prefix0..2".into(),
            ));
        }
        let layer = GdnPrefixLayer::new(invocation.layer().source(), completed_layers - 1)?;
        for index in 0..completed_layers {
            for slot in [
                StateSlot::Conv(ConvDomain(index)),
                StateSlot::Recurrent(RecurrentDomain(index)),
            ] {
                if self.current(slot)?.ordinal != 1 {
                    return Err(Error::State(
                        "one real Conv/recurrent effect for every prefix layer".into(),
                    ));
                }
            }
        }
        let output_tag = self
            .vllm_value_provenance(&output)?
            .ok_or_else(|| Error::State("pending MLP branch provenance absent".into()))?;
        let residual_tag = self
            .vllm_value_provenance(&residual)?
            .ok_or_else(|| Error::State("carried residual provenance absent".into()))?;
        output_tag.require_invocation(invocation)?;
        residual_tag.require_invocation(invocation)?;
        if output_tag.origin()
            != &(VllmValueOrigin::Projection {
                layer,
                kind: VllmGdnProjectionKind::Down,
            })
            || residual_tag.origin()
                != &(VllmValueOrigin::CarriedResidual {
                    binding: layer.rms(GdnPrefixRmsSite::PostAttention)?,
                })
        {
            return Err(Error::State(
                "actual last-layer Down branch and separate post-attention residual".into(),
            ));
        }
        self.finish_fragment(output, Some(residual.id), completed_layers)
    }
    fn finish_fragment(
        mut self,
        output: PlanValue,
        residual_output: Option<ValueId>,
        completed_layers: usize,
    ) -> Result<RecordedVllmGdnFragment> {
        let untouched: Vec<_> = self
            .required
            .iter()
            .copied()
            .filter(|slot| !self.domains.contains_key(slot))
            .collect();
        for slot in untouched {
            let effect = self.effect(slot)?;
            self.record(
                "vllm.original.untouched_state_copy",
                PlanOperation::TypedVllmStateCarry,
                &[],
                &[],
                Some(effect),
            )?;
        }
        let state_node_count = self.nodes.len();
        Ok(RecordedVllmGdnFragment {
            output: output.id,
            residual_output,
            completed_layers,
            plan: ForwardPlan {
                vllm_fragment: self.vllm_fragment,
                vllm_provenance: self.vllm_provenance,
                nodes: self.nodes,
                semantics: Arc::new(ForwardPlanSemantics {
                    metadata: None,
                    source: self.source,
                    profile: self.profile,
                    standalone_exp: None,
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
            },
        })
    }
}
#[cfg(test)]
#[path = "vllm_gdn/tests.rs"]
mod tests;
