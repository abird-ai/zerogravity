//! Closed GDN source records. State authority belongs to the original session.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenInputRms {
    invocation: QwenGdnCall,
    parameter: SourceParameter,
}
impl QwenInputRms {
    pub fn new(invocation: QwenGdnCall, parameter: SourceParameter) -> Result<Self> {
        let value = Self {
            invocation,
            parameter,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn invocation(&self) -> &QwenGdnCall {
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
                    "reference CUDA input RMS raw BF16/F32 gamma".into(),
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
pub struct QwenGdnConv {
    invocation: QwenGdnCall,
    parameter: SourceParameter,
}
impl QwenGdnConv {
    pub fn new(invocation: QwenGdnCall, parameter: SourceParameter) -> Result<Self> {
        let value = Self {
            invocation,
            parameter,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn invocation(&self) -> &QwenGdnCall {
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
pub struct GdnConvValues<V> {
    pub activated: V,
    pub z: V,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenGdnRecurrent {
    invocation: QwenGdnCall,
    parameters: [SourceParameter; 3],
}
impl QwenGdnRecurrent {
    pub fn new(invocation: QwenGdnCall, parameters: [SourceParameter; 3]) -> Result<Self> {
        let value = Self {
            invocation,
            parameters,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn invocation(&self) -> &QwenGdnCall {
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
    invocation: &QwenGdnCall,
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
    builder: &QwenPlanBuilder,
    invocation: &QwenGdnCall,
    inputs: &[PlanValue],
    widths: &[usize],
) -> Result<()> {
    invocation.validate()?;
    if builder.profile != ProgramNumericalProfile::SourceCudaSm86Bf16V1
        || builder.source != invocation.layer().source()
        || !(builder.operator_slice || builder.fragment_call.is_some())
        || inputs.len() != widths.len()
    {
        return Err(Error::Unsupported(
            "GDN closed BF16 source/profile/scope/arity".into(),
        ));
    }
    for (input, width) in inputs.iter().zip(widths) {
        let elements = invocation.rows() * width;
        if input.precision() != ActivationPrecision::Bf16WidenedF32
            || input.shape != ValueShape::Vector(elements)
            || input.elements() != elements
        {
            return Err(Error::Unsupported(
                "GDN exact dense BF16 operand extent".into(),
            ));
        }
    }
    gdn_projection::require_invocation(builder, invocation, inputs)?;
    Ok(())
}
pub(super) fn input_rms_precision(
    builder: &QwenPlanBuilder,
    d: &QwenInputRms,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    d.validate()?;
    require_dense(builder, d.invocation(), inputs, &[1024])?;
    if builder
        .value_provenance(&inputs[0])?
        .map(QwenValueProvenance::origin)
        != Some(&QwenValueOrigin::SourceRows)
    {
        return Err(Error::State(
            "input RMS must consume original raw source rows".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
fn fragment_invocation(builder: &QwenPlanBuilder, invocation: &QwenGdnCall) -> Result<bool> {
    let Some(base) = builder.fragment_call.as_ref() else {
        return Ok(false);
    };
    Ok(base.teachers() == invocation.teachers()
        && base.reference_phase()? == invocation.reference_phase()?
        && base.layer().source() == invocation.layer().source()
        && invocation.layer().index() <= 2)
}
pub(super) fn conv_precision(
    builder: &QwenPlanBuilder,
    d: &QwenGdnConv,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    d.validate()?;
    if !fragment_invocation(builder, d.invocation())? {
        return Err(Error::Unsupported(
            "stateful GDN requires closed original-root fragment".into(),
        ));
    }
    require_dense(builder, d.invocation(), inputs, &[8192, 32])?;
    for (input, kind) in inputs.iter().zip([GdnProjection::Qkvz, GdnProjection::Ba]) {
        if builder
            .value_provenance(input)?
            .map(QwenValueProvenance::origin)
            != Some(&QwenValueOrigin::Projection {
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
    builder: &QwenPlanBuilder,
    d: &QwenGdnRecurrent,
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
                .value_provenance(v)?
                .ok_or_else(|| Error::State("GDN producer tag absent".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let (
        QwenValueOrigin::ConvActivated {
            layer,
            node_index,
            qkvz,
            ba,
        },
        QwenValueOrigin::ConvZ {
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
            != &(QwenValueOrigin::Projection {
                layer: *layer,
                kind: GdnProjection::Ba,
            })
    {
        return Err(Error::State(
            "GDN exact Conv cohort and original BA ValueID".into(),
        ));
    }
    Ok(ActivationPrecision::Bf16WidenedF32)
}
pub(super) fn record_input_rms(
    builder: &mut QwenPlanBuilder,
    d: &QwenInputRms,
    input: &PlanValue,
) -> Result<PlanValue> {
    input_rms_precision(builder, d, &[*input])?;
    let tag = gdn_projection::require_invocation(builder, d.invocation(), &[*input])?.with_origin(
        QwenValueOrigin::InputNormalized {
            layer: d.invocation().layer(),
        },
    );
    QwenPlanBuilder::one_output(builder.record_source(
        "qwen.source.input_rms",
        builder.input_rms_operator(d)?,
        &[*input],
        vec![tag],
    )?)
}
pub(super) fn record_conv(
    builder: &mut QwenPlanBuilder,
    d: &QwenGdnConv,
    qkvz: &PlanValue,
    ba: &PlanValue,
) -> Result<GdnConvValues<PlanValue>> {
    let inputs = [*qkvz, *ba];
    conv_precision(builder, d, &inputs)?;
    let tag = gdn_projection::require_invocation(builder, d.invocation(), &inputs)?;
    let node_index = builder.core.nodes().len();
    let layer = d.invocation().layer();
    let tags = vec![
        tag.clone().with_origin(QwenValueOrigin::ConvActivated {
            layer,
            node_index,
            qkvz: qkvz.id,
            ba: ba.id,
        }),
        tag.with_origin(QwenValueOrigin::ConvZ {
            layer,
            node_index,
            qkvz: qkvz.id,
            ba: ba.id,
        }),
    ];
    let values = builder.record_source(
        "qwen.source.gdn.conv",
        builder.conv_operator(d)?,
        &inputs,
        tags,
    )?;
    let [activated, z] = values
        .try_into()
        .map_err(|_| Error::State("Qwen conv dual output arity".into()))?;
    Ok(GdnConvValues { activated, z })
}
pub(super) fn record_recurrent(
    builder: &mut QwenPlanBuilder,
    d: &QwenGdnRecurrent,
    activated: &PlanValue,
    z: &PlanValue,
    ba: &PlanValue,
) -> Result<PlanValue> {
    let inputs = [*activated, *z, *ba];
    recurrent_precision(builder, d, &inputs)?;
    let tag = gdn_projection::require_invocation(builder, d.invocation(), &inputs)?;
    let QwenValueOrigin::ConvActivated { node_index, .. } = builder
        .value_provenance(activated)?
        .ok_or_else(|| Error::State("Qwen conv producer absent".into()))?
        .origin()
    else {
        return Err(Error::State("Qwen actual conv producer required".into()));
    };
    let tag = tag.with_origin(QwenValueOrigin::RecurrentGated {
        layer: d.invocation().layer(),
        conv_node_index: *node_index,
    });
    QwenPlanBuilder::one_output(builder.record_source(
        "qwen.source.gdn.recurrent_gated",
        builder.recurrent_operator(d)?,
        &inputs,
        vec![tag],
    )?)
}

/// Unpublishable physical fragment: one shared-engine plan plus family metadata.
#[derive(Clone, Debug)]
pub struct QwenFragmentPlan {
    plan: ForwardPlan,
    metadata: QwenFragmentMetadata,
}
/// Existing family producer data; no second graph or execution permission.
#[derive(Clone, Debug)]
pub struct QwenFragmentMetadata {
    invocation: QwenGdnCall,
    provenance: Vec<Option<QwenValueProvenance>>,
    output: ValueId,
    residual_output: Option<ValueId>,
    completed_layers: usize,
}
#[derive(Clone, Copy)]
pub struct QwenFragmentView<'a> {
    plan: &'a ForwardPlan,
    metadata: &'a QwenFragmentMetadata,
}
impl QwenFragmentPlan {
    pub fn invocation(&self) -> &QwenGdnCall {
        self.metadata.invocation()
    }
    pub fn value_provenance(&self, id: ValueId) -> Result<Option<&QwenValueProvenance>> {
        self.plan.source_value_provenance(id)?;
        Ok(self
            .metadata
            .provenance
            .get(id.0 as usize)
            .and_then(Option::as_ref))
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn output(&self) -> ValueId {
        self.metadata.output()
    }
    pub fn completed_layers(&self) -> usize {
        self.metadata.completed_layers()
    }
    pub fn residual_output(&self) -> Option<ValueId> {
        self.metadata.residual_output()
    }
    pub fn view(&self) -> Result<QwenFragmentView<'_>> {
        self.metadata.view(&self.plan)
    }
    pub fn into_parts(self) -> (ForwardPlan, QwenFragmentMetadata) {
        (self.plan, self.metadata)
    }
}
impl QwenFragmentMetadata {
    pub fn invocation(&self) -> &QwenGdnCall {
        &self.invocation
    }
    pub fn output(&self) -> ValueId {
        self.output
    }
    pub fn completed_layers(&self) -> usize {
        self.completed_layers
    }
    pub fn residual_output(&self) -> Option<ValueId> {
        self.residual_output
    }
    pub fn binding_metadata_bytes(&self) -> Result<usize> {
        self.provenance
            .capacity()
            .checked_mul(std::mem::size_of::<Option<QwenValueProvenance>>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()))
            .ok_or_else(|| Error::Limit("fragment family metadata extent".into()))
    }
    /// Metadata association only. Consumers still authenticate actual recipe/source/SSA.
    pub fn view<'a>(&'a self, plan: &'a ForwardPlan) -> Result<QwenFragmentView<'a>> {
        self.invocation.validate()?;
        let call = self.invocation.source_call()?;
        if plan.source() != self.invocation.layer().source()
            || plan.numerical_profile() != ProgramNumericalProfile::SourceCudaSm86Bf16V1
            || plan.source_call() != Some(&call)
            || plan.is_operator_slice()
            || plan.values().len() != self.provenance.len()
            || plan.values().get(self.output.0 as usize).is_none()
            || self
                .residual_output
                .is_some_and(|id| plan.values().get(id.0 as usize).is_none())
        {
            return Err(Error::State(
                "fragment actual plan/call/producer coverage differs".into(),
            ));
        }
        for (index, family) in self.provenance.iter().enumerate() {
            let family = family
                .as_ref()
                .ok_or_else(|| Error::State("fragment family producer absent".into()))?;
            family.require_invocation(&self.invocation)?;
            let common = plan
                .source_value_provenance(ValueId(
                    u32::try_from(index)
                        .map_err(|_| Error::Limit("fragment producer index".into()))?,
                ))?
                .ok_or_else(|| Error::State("fragment mathematical producer absent".into()))?;
            if common.call() != &call {
                return Err(Error::State(
                    "fragment mathematical producer call differs".into(),
                ));
            }
        }
        Ok(QwenFragmentView {
            plan,
            metadata: self,
        })
    }
}
impl<'a> QwenFragmentView<'a> {
    pub fn plan(self) -> &'a ForwardPlan {
        self.plan
    }
    pub fn invocation(self) -> &'a QwenGdnCall {
        self.metadata.invocation()
    }
    pub fn output(self) -> ValueId {
        self.metadata.output()
    }
    pub fn completed_layers(self) -> usize {
        self.metadata.completed_layers()
    }
    pub fn residual_output(self) -> Option<ValueId> {
        self.metadata.residual_output()
    }
    pub fn value_provenance(self, id: ValueId) -> Result<Option<&'a QwenValueProvenance>> {
        self.plan.source_value_provenance(id)?;
        Ok(self
            .metadata
            .provenance
            .get(id.0 as usize)
            .and_then(Option::as_ref))
    }
}
fn all_domains() -> Result<Vec<StateSlot>> {
    hybrid_state_domains((0..24).map(|index| index % 4 == 3))
}

impl QwenPlanBuilder {
    /// Metadata recipe only. Real lowerer must bind the SAME original all42 root.
    pub fn new_gdn_fragment(invocation: QwenGdnCall) -> Result<Self> {
        invocation.validate()?;
        if invocation.layer().index() != 0
            || !matches!(
                invocation.reference_phase()?,
                ReferencePhase::InitialOne | ReferencePhase::Initial64
            )
            || invocation.teachers().capacity() != 512
        {
            return Err(Error::Unsupported(
                "initial layer0 original-root fragment only; no cached/root promotion".into(),
            ));
        }
        Self::new_fragment_inner(invocation)
    }
    /// Metadata only for a private phase of ONE original unpublished attempt.
    /// Existing initial and full-model execution admission remain separate.
    pub fn new_gdn_attempt_phase(invocation: QwenGdnCall) -> Result<Self> {
        invocation.validate()?;
        invocation.reference_phase()?;
        if invocation.layer().index() != 0 || invocation.teachers().capacity() != 512 {
            return Err(Error::Unsupported(
                "closed layer0 request phase metadata only".into(),
            ));
        }
        Self::new_fragment_inner(invocation)
    }
    fn new_fragment_inner(invocation: QwenGdnCall) -> Result<Self> {
        let mut builder = Self::from_core(CorePlanBuilder::new_source_program(
            invocation.source_call()?,
            all_domains()?,
        )?);
        builder.fragment_call = Some(invocation);
        Ok(builder)
    }
    pub fn finish_gdn_fragment(self, output: PlanValue) -> Result<QwenFragmentPlan> {
        let invocation = self
            .fragment_call
            .clone()
            .ok_or_else(|| Error::State("not an original-root GDN fragment".into()))?;
        self.value_provenance(&output)?
            .ok_or_else(|| Error::State("fragment output provenance absent".into()))?
            .require_invocation(&invocation)?;
        if self.core.has_advanced()
            || self.operator_slice
            || *self.core.required_domains() != all_domains()?.into_iter().collect()
            || self.core.proposed_domains().len() != 2
            || self
                .core
                .current_state(StateSlot::Conv(ConvDomain(0)))?
                .ordinal
                != 1
            || self
                .current_state(StateSlot::Recurrent(RecurrentDomain(0)))?
                .ordinal
                != 1
            || !matches!(self.value_provenance(&output)?.map(QwenValueProvenance::origin), Some(QwenValueOrigin::RecurrentGated {layer, ..}) if *layer == invocation.layer())
        {
            return Err(Error::State(
                "fragment requires exact Conv/recurrent pair and actual Gated output".into(),
            ));
        }
        self.finish_fragment(output, None, 0)
    }
    /// SAME all42 authority, complete initial layer0 or initial GDN prefix0..2.
    /// Pending MLP branch and carried BF16 residual remain distinct outputs.
    pub fn finish_gdn_mlp_fragment(
        self,
        output: PlanValue,
        residual: PlanValue,
        completed_layers: usize,
    ) -> Result<QwenFragmentPlan> {
        let invocation = self
            .fragment_call
            .as_ref()
            .ok_or_else(|| Error::State("not an original-root GDN fragment".into()))?;
        if !matches!(completed_layers, 1 | 3)
            || output.id == residual.id
            || self.operator_slice
            || self.core.has_advanced()
            || *self.core.required_domains() != all_domains()?.into_iter().collect()
            || self.core.proposed_domains().len() != completed_layers * 2
        {
            return Err(Error::Unsupported(
                "closed complete initial GDN layer0/prefix0..2".into(),
            ));
        }
        let layer = QwenPrefixLayer::new(invocation.layer().source(), completed_layers - 1)?;
        for index in 0..completed_layers {
            for slot in [
                StateSlot::Conv(ConvDomain(index)),
                StateSlot::Recurrent(RecurrentDomain(index)),
            ] {
                if self.core.current_state(slot)?.ordinal != 1 {
                    return Err(Error::State(
                        "one real Conv/recurrent effect for every prefix layer".into(),
                    ));
                }
            }
        }
        let output_tag = self
            .value_provenance(&output)?
            .ok_or_else(|| Error::State("pending MLP branch provenance absent".into()))?;
        let residual_tag = self
            .value_provenance(&residual)?
            .ok_or_else(|| Error::State("carried residual provenance absent".into()))?;
        output_tag.require_invocation(invocation)?;
        residual_tag.require_invocation(invocation)?;
        if output_tag.origin()
            != &(QwenValueOrigin::Projection {
                layer,
                kind: GdnProjection::Down,
            })
            || residual_tag.origin()
                != &(QwenValueOrigin::CarriedResidual {
                    binding: layer.rms(QwenPrefixRmsSite::PostAttention)?,
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
    ) -> Result<QwenFragmentPlan> {
        let call = self
            .fragment_call
            .as_ref()
            .ok_or_else(|| Error::State("Qwen fragment original call absent".into()))?
            .source_call()?;
        let untouched = self
            .core
            .required_domains()
            .iter()
            .copied()
            .filter(|slot| !self.core.proposed_domains().contains_key(slot))
            .collect::<Vec<_>>();
        for slot in untouched {
            let descriptor = SourceOperator::new(
                call.clone(),
                SourceOperatorKind::StateCarry { slot },
                SourceNumericalContract::ReferenceCudaSm86Bf16V1,
                Vec::new(),
                Vec::new(),
            )?;
            self.record_source(
                "qwen.source.untouched_state_copy",
                descriptor,
                &[],
                Vec::new(),
            )?;
        }
        Ok(QwenFragmentPlan {
            metadata: QwenFragmentMetadata {
                output: output.id,
                residual_output,
                completed_layers,
                invocation: self
                    .fragment_call
                    .ok_or_else(|| Error::State("Qwen fragment call absent".into()))?,
                provenance: self.provenance,
            },
            plan: self.core.finish_source_program()?,
        })
    }
}
#[cfg(test)]
#[path = "gdn/tests.rs"]
mod tests;
