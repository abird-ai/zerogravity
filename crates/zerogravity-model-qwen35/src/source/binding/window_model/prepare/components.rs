//! Model-owned initial GDN/post-attention/MLP eligibility. Prior GPU ComponentRecipe.
//! Captures select reference implementations; actual native bytes/ABI remain backend-owned.
use super::*;
use crate::plan::{QwenValueOrigin, ReferencePhase};
use crate::source::binding::{QwenMatrixRole as M, QwenParameterRole as P};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentOutput {
    Gated,
    PostAttention,
    Mlp,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentImplementation {
    Reference,
    OwnedSolve4Recompute3,
    OwnedFixedPromptSolve4,
    OriginalPlainSolve2,
}
impl ComponentImplementation {
    fn require_owned(self) -> Result<()> {
        if self != Self::OwnedFixedPromptSolve4 {
            return Err(Error::Unsupported(
                "connected fixed prompt original PLAIN tuple is not implemented".into(),
            ));
        }
        Ok(())
    }
    fn require_initial_owned(self, input_tokens: usize) -> Result<()> {
        if self != Self::OwnedSolve4Recompute3 || !matches!(input_tokens, 1 | 64) {
            return Err(Error::Unsupported(
                "owned2B explicit initial1/64 tuple only".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentKernelCapture {
    recipe_sha256: &'static str,
    explicit_owned: bool,
}
impl ComponentKernelCapture {
    pub fn recipe_sha256(self) -> &'static str {
        self.recipe_sha256
    }
    pub fn explicit_owned(self) -> bool {
        self.explicit_owned
    }
}
pub struct PreparedComponent {
    prepared: PreparedPlan,
    output: ComponentOutput,
    capture: ComponentKernelCapture,
    node_end: usize,
    value_end: usize,
    provider_bytes: usize,
    record_admission: Option<Arc<WindowWorkspace>>,
}
impl PreparedComponent {
    pub fn prepared(&self) -> &PreparedPlan {
        &self.prepared
    }
    pub fn output(&self) -> ComponentOutput {
        self.output
    }
    pub fn capture(&self) -> ComponentKernelCapture {
        self.capture
    }
    pub fn node_end(&self) -> usize {
        self.node_end
    }
    pub fn value_end(&self) -> usize {
        self.value_end
    }
    pub fn provider_bytes(&self) -> usize {
        self.provider_bytes
    }
    pub fn record_admission(&self) -> Option<&Arc<WindowWorkspace>> {
        self.record_admission.as_ref()
    }
    pub fn into_parts(
        self,
    ) -> (
        PreparedPlan,
        ComponentKernelCapture,
        Option<Arc<WindowWorkspace>>,
    ) {
        (self.prepared, self.capture, self.record_admission)
    }
}
const NODE_END: usize = 6;
const VALUE_END: usize = 7;
const REFERENCE_2B_CAPTURE: &str =
    "356a1149294dca85172679bb48d3070777f77d3e98eba6071ed5408a79e647cd";
const FIXED_PROMPT_CAPTURE: &str =
    "dd8c09131ee5e88efca478044a9a41869532453a9c93f972b2861022740bb2dd";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ModelComponentKind {
    Qwen2b,
    FixedPrompt(ComponentImplementation),
}
impl ModelComponentKind {
    pub(super) fn profile(self) -> QwenSourceProfile {
        match self {
            Self::Qwen2b => QwenSourceProfile::Qwen2B,
            Self::FixedPrompt(_) => QwenSourceProfile::Qwen08,
        }
    }
    pub(super) fn columns(self) -> usize {
        match self {
            Self::Qwen2b => 2048,
            Self::FixedPrompt(_) => 1024,
        }
    }
    pub(super) fn require_request(self, request: &crate::plan::QwenGdnCall) -> Result<()> {
        request.validate()?;
        if request.layer().index() != 0
            || request.teachers().prompt_span().is_some()
            || request.teachers().base() != 0
            || request.teachers().capacity() != 512
        {
            return Err(Error::Unsupported(
                "initial source component request only".into(),
            ));
        }
        match self {
            Self::Qwen2b
                if matches!(
                    request.reference_phase()?,
                    ReferencePhase::InitialOne | ReferencePhase::Initial64
                ) =>
            {
                Ok(())
            }
            Self::FixedPrompt(arithmetic)
                if request.mode() == SourceCallMode::Prefill
                    && request.rows() == 47
                    && request.reference_phase().is_err()
                    && request.teachers().ids() == FIXED_PROMPT_IDS =>
            {
                arithmetic.require_owned()
            }
            _ => Err(Error::Unsupported(
                "closed source component geometry/prompt/arithmetic".into(),
            )),
        }
    }
}
pub(super) const FIXED_PROMPT_IDS: [u32; 47] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ModelComponentCut {
    Gated,
    PostAttention,
    Mlp,
    Qwen2bMlp,
}
impl ModelComponentCut {
    pub(super) fn includes_post_attention(self) -> bool {
        matches!(self, Self::PostAttention | Self::Mlp | Self::Qwen2bMlp)
    }
    pub(super) fn includes_mlp(self) -> bool {
        matches!(self, Self::Mlp | Self::Qwen2bMlp)
    }
    fn require_kind(self, kind: ModelComponentKind) -> Result<()> {
        if self == Self::Mlp && !matches!(kind, ModelComponentKind::FixedPrompt(_)) {
            return Err(Error::Unsupported(
                "MLP component is canonical47 only".into(),
            ));
        }
        if self == Self::Qwen2bMlp && kind != ModelComponentKind::Qwen2b {
            return Err(Error::Unsupported(
                "explicit2B MLP source family only".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn node_end(self) -> usize {
        match self {
            Self::Gated => NODE_END,
            Self::PostAttention => 8,
            Self::Mlp | Self::Qwen2bMlp => 11,
        }
    }
    pub(super) fn value_end(self) -> usize {
        match self {
            Self::Gated => VALUE_END,
            Self::PostAttention => 10,
            Self::Mlp | Self::Qwen2bMlp => 13,
        }
    }
    pub(super) fn provider_bytes(self) -> usize {
        match self {
            Self::Gated => 192 * 1024 * 1024,
            Self::PostAttention | Self::Mlp => 224 * 1024 * 1024,
            Self::Qwen2bMlp => 320 * 1024 * 1024,
        }
    }
}
fn require_qwen2b_arithmetic(
    kind: ModelComponentKind,
    cut: ModelComponentCut,
    rows: usize,
    arithmetic: ComponentImplementation,
) -> Result<()> {
    if kind != ModelComponentKind::Qwen2b
        || !matches!(
            cut,
            ModelComponentCut::Gated
                | ModelComponentCut::PostAttention
                | ModelComponentCut::Qwen2bMlp
        )
    {
        return Err(Error::Unsupported(
            "explicit2B tuple only on exact finite initial route".into(),
        ));
    }
    arithmetic.require_initial_owned(match rows {
        1 => 1,
        64 => 64,
        _ => {
            return Err(Error::Unsupported(
                "explicit2B initial1/64 arithmetic".into(),
            ));
        }
    })
}

fn require_weight_request(
    profile: QwenSourceProfile,
    request: &crate::plan::QwenGdnCall,
) -> Result<ReferencePhase> {
    request.validate()?;
    if profile != QwenSourceProfile::Qwen2B
        || request.layer().index() != 0
        || request.mode() != SourceCallMode::Prefill
        || request.teachers().base() != 0
        || request.teachers().capacity() != 512
    {
        return Err(Error::Unsupported(
            "finite original2B initial MLP W source span".into(),
        ));
    }
    match request.reference_phase()? {
        phase @ (ReferencePhase::InitialOne | ReferencePhase::Initial64) => Ok(phase),
        _ => Err(Error::Unsupported(
            "original2B MLP weights initial1/64 only".into(),
        )),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelProjectionRole {
    Qkvz,
    Ba,
}
fn require_component_projection(operator: &QwenOperator) -> Result<ModelProjectionRole> {
    operator.validate()?;
    if operator.layer().map(|l| l.index()) != Some(0) {
        return Err(Error::Unsupported(
            "exact source layer0 learned component producer".into(),
        ));
    }
    match operator.call().model_profile() {
        QwenSourceProfile::Qwen2B => {
            operator.call().request().reference_phase()?;
        }
        QwenSourceProfile::Qwen08 => {
            ModelComponentKind::FixedPrompt(ComponentImplementation::OwnedFixedPromptSolve4)
                .require_request(operator.call().request())?
        }
    }
    match operator.kind() {
        QwenOperation::Projection(QwenProjection::Qkvz) => Ok(ModelProjectionRole::Qkvz),
        QwenOperation::Projection(QwenProjection::Ba) => Ok(ModelProjectionRole::Ba),
        _ => Err(Error::Unsupported("source QKVZ/BA component only".into())),
    }
}
fn state_effect(node: &PlanNode, slot: StateSlot) -> Result<StateEffect> {
    let effect = node
        .state
        .ok_or_else(|| Error::State("source2B actual state effect absent".into()))?;
    if effect.read.slot != slot
        || effect.write.slot != slot
        || effect.read.ordinal != 0
        || effect.write.ordinal != 1
    {
        return Err(Error::State("source2B one original Conv/H proposal".into()));
    }
    Ok(effect)
}

impl BoundFullModelSource {
    pub fn prepare_component(
        &self,
        record: QwenPlan,
        output: ComponentOutput,
        implementation: ComponentImplementation,
        record_admission: Option<Arc<WindowWorkspace>>,
    ) -> Result<PreparedComponent> {
        let source = self;
        let (kind, qwen2b_arithmetic) = match (self.model_profile(), implementation) {
            (QwenSourceProfile::Qwen2B, ComponentImplementation::Reference) => {
                (ModelComponentKind::Qwen2b, None)
            }
            (QwenSourceProfile::Qwen2B, ComponentImplementation::OwnedSolve4Recompute3) => {
                (ModelComponentKind::Qwen2b, Some(implementation))
            }
            (
                QwenSourceProfile::Qwen08,
                ComponentImplementation::OwnedFixedPromptSolve4
                | ComponentImplementation::OriginalPlainSolve2,
            ) => (ModelComponentKind::FixedPrompt(implementation), None),
            _ => {
                return Err(Error::Unsupported(
                    "component source/profile/reference implementation differs".into(),
                ));
            }
        };
        let cut = match (output, kind) {
            (ComponentOutput::Gated, _) => ModelComponentCut::Gated,
            (ComponentOutput::PostAttention, _) => ModelComponentCut::PostAttention,
            (ComponentOutput::Mlp, ModelComponentKind::Qwen2b) => ModelComponentCut::Qwen2bMlp,
            (ComponentOutput::Mlp, ModelComponentKind::FixedPrompt(_)) => ModelComponentCut::Mlp,
        };
        cut.require_kind(kind)?;
        if cut == ModelComponentCut::Qwen2bMlp {
            require_weight_request(source.model_profile(), record.call().request())?;
        }
        match (cut, record_admission.as_ref()) {
            (ModelComponentCut::Qwen2bMlp, Some(ticket))
                if ticket.bytes() == 32 * 1024 * 1024
                    && ticket.source().same_issuer(source.source().source()) => {}
            (ModelComponentCut::Qwen2bMlp, _) => {
                return Err(Error::State(
                    "same real2B record32MiB admission required".into(),
                ));
            }
            (_, None) => {}
            (_, Some(_)) => {
                return Err(Error::Unsupported(
                    "old component cannot borrow new2B record admission".into(),
                ));
            }
        }
        if cut == ModelComponentCut::Qwen2bMlp && qwen2b_arithmetic.is_none() {
            return Err(Error::Unsupported(
                "new2B MLP requires explicit owned arithmetic".into(),
            ));
        }
        kind.require_request(record.call().request())?;
        if let Some(arithmetic) = qwen2b_arithmetic {
            require_qwen2b_arithmetic(kind, cut, record.call().rows(), arithmetic)?;
        }
        let metadata = source.source().source().reserve_workspace(METADATA_BYTES)?;
        require_ordinary_source(source.source(), kind)?;
        let call = record.call();
        let plan = record.plan();
        if source.model_profile() != kind.profile()
            || call.model_profile() != kind.profile()
            || call.source() != source.source().source_identity()
            || call.request().teachers().base() != 0
            || call.request().teachers().capacity() != 512
            || plan.nodes().len() != 255
            || plan.values().len() != 327
            || plan.required_domains().len() != 42
            || plan.proposed_domains().len() != 42
            || plan.frontier_delta() != 0
            || plan.logits().is_some()
            || plan.is_operator_slice()
            || plan.source() != call.source()
            || plan.source_call() != Some(&call.request().source_call()?)
        {
            return Err(Error::Unsupported(
                "authentic initial2B six-node source prefix only".into(),
            ));
        }
        let rows = call.rows();
        let nodes = &plan.nodes()[..cut.node_end()];
        record.operator(0)?;
        let PlanOperation::SourceOperator {
            descriptor,
            state: None,
        } = &nodes[0].operation
        else {
            return Err(Error::State(
                "source2B original embedding node absent".into(),
            ));
        };
        if descriptor.kind()
            != zerogravity_core::accelerator::program::SourceOperatorKind::EmbeddingRows
            || descriptor.call() != &call.request().source_call()?
            || descriptor.matrices().len() != 1
            || !nodes[0].inputs.is_empty()
            || nodes[0].outputs != [ValueId(0)]
            || nodes[0].state.is_some()
        {
            return Err(Error::State(
                "source2B original embedding producer/issuer".into(),
            ));
        }
        source
            .source()
            .validate_matrix(M::Embedding, &descriptor.matrices()[0])?;
        let mut expected = vec![
            (QwenOperation::InputRms, vec![ValueId(0)], vec![ValueId(1)]),
            (
                QwenOperation::Projection(QwenProjection::Qkvz),
                vec![ValueId(1)],
                vec![ValueId(2)],
            ),
            (
                QwenOperation::Projection(QwenProjection::Ba),
                vec![ValueId(1)],
                vec![ValueId(3)],
            ),
            (
                QwenOperation::GdnConv,
                vec![ValueId(2), ValueId(3)],
                vec![ValueId(4), ValueId(5)],
            ),
            (
                QwenOperation::GdnRecurrent,
                vec![ValueId(4), ValueId(5), ValueId(3)],
                vec![ValueId(6)],
            ),
        ];
        if cut.includes_post_attention() {
            expected.extend([
                (
                    QwenOperation::Projection(QwenProjection::GdnOut),
                    vec![ValueId(6)],
                    vec![ValueId(7)],
                ),
                (
                    QwenOperation::ResidualRms(QwenRmsSite::PostAttention),
                    vec![ValueId(7), ValueId(0)],
                    vec![ValueId(8), ValueId(9)],
                ),
            ]);
        }
        if cut.includes_mlp() {
            expected.extend([
                (
                    QwenOperation::Projection(QwenProjection::GateUp),
                    vec![ValueId(8)],
                    vec![ValueId(10)],
                ),
                (
                    QwenOperation::SiluAndMul,
                    vec![ValueId(10)],
                    vec![ValueId(11)],
                ),
                (
                    QwenOperation::Projection(QwenProjection::Down),
                    vec![ValueId(11)],
                    vec![ValueId(12)],
                ),
            ]);
        }
        for (index, (node, (kind, inputs, outputs))) in nodes[1..].iter().zip(expected).enumerate()
        {
            let descriptor = record
                .operator(index + 1)?
                .ok_or_else(|| Error::State("actual source2B prefix node absent".into()))?;
            let PlanOperation::SourceOperator { state, .. } = &node.operation else {
                return Err(Error::State(
                    "actual source2B mathematical prefix absent".into(),
                ));
            };
            descriptor.validate()?;
            if matches!(index + 1, 2 | 3) {
                require_component_projection(&descriptor)?;
            }
            if descriptor.call() != call
                || descriptor.kind() != kind
                || descriptor.layer().map(|x| x.index()) != Some(0)
                || node.inputs != inputs
                || node.outputs != outputs
                || state.is_some()
                || !matches!(kind, QwenOperation::GdnConv | QwenOperation::GdnRecurrent)
                    && node.state.is_some()
            {
                return Err(Error::State(
                    "source2B exact ordered first-six SSA/effect recipe".into(),
                ));
            }
            let (matrices, parameters): (&[M], &[P]) = match kind {
                QwenOperation::InputRms => (&[], &[P::InputNorm(0)]),
                QwenOperation::Projection(QwenProjection::Qkvz) => {
                    (&[M::GdnQkv(0), M::GdnGate(0)], &[])
                }
                QwenOperation::Projection(QwenProjection::Ba) => {
                    (&[M::GdnBeta(0), M::GdnAlpha(0)], &[])
                }
                QwenOperation::GdnConv => (&[], &[P::GdnConvolution(0)]),
                QwenOperation::GdnRecurrent => (&[], &[P::GdnA(0), P::GdnDt(0), P::GdnNorm(0)]),
                QwenOperation::Projection(QwenProjection::GdnOut) => (&[M::GdnOutput(0)], &[]),
                QwenOperation::ResidualRms(QwenRmsSite::PostAttention) => (&[], &[P::FfnNorm(0)]),
                QwenOperation::Projection(QwenProjection::GateUp) => {
                    (&[M::FfnGate(0), M::FfnUp(0)], &[])
                }
                QwenOperation::SiluAndMul => (&[], &[]),
                QwenOperation::Projection(QwenProjection::Down) => (&[M::FfnDown(0)], &[]),
                _ => {
                    return Err(Error::State(
                        "source component prefix operator absent".into(),
                    ));
                }
            };
            if descriptor.matrices().len() != matrices.len()
                || descriptor.parameters().len() != parameters.len()
            {
                return Err(Error::State("source2B actual source-role arity".into()));
            }
            for (actual, role) in descriptor.matrices().iter().zip(matrices) {
                source.source().validate_matrix(*role, actual)?;
            }
            for (actual, role) in descriptor.parameters().iter().zip(parameters) {
                source.source().validate_parameter(*role, actual)?;
            }
            for (out, id) in node.outputs.iter().enumerate() {
                let tag = record
                    .value_provenance(*id)?
                    .ok_or_else(|| Error::State("source2B actual output tag absent".into()))?;
                tag.require_invocation(call.request())?;
                let QwenValueOrigin::FullModel(origin) = tag.origin() else {
                    return Err(Error::State(
                        "source2B original output origin absent".into(),
                    ));
                };
                if origin.call() != call
                    || origin.kind() != kind
                    || origin.node_index() != index + 1
                    || origin.output_index() != out
                    || origin.layer().map(|x| x.index()) != Some(0)
                {
                    return Err(Error::State(
                        "source2B exact producer node/output/layer origin".into(),
                    ));
                }
            }
        }
        for index in [1, 7].into_iter().filter(|&i| i < cut.node_end()) {
            let PlanOperation::SourceOperator {
                descriptor,
                state: None,
            } = &plan.nodes()[index].operation
            else {
                return Err(Error::State(
                    "component actual RMS mathematical producer absent".into(),
                ));
            };
            let geometry = match (index, descriptor.kind()) {
                (1, SourceOperatorKind::Rms(g)) | (7, SourceOperatorKind::ResidualRms(g)) => g,
                _ => {
                    return Err(Error::State(
                        "component ordinary/post RMS kind differs".into(),
                    ));
                }
            };
            if geometry.width != kind.columns()
                || geometry.epsilon != 1e-6
                || geometry.gamma != SourceGamma::OnePlus
                || descriptor.numerical_contract()
                    != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            {
                return Err(Error::Unsupported(
                    "component exact RMS epsilon/gamma/width/arithmetic".into(),
                ));
            }
        }
        let mut widths = vec![kind.columns(), kind.columns(), 8192, 32, 6144, 2048, 2048];
        if cut.includes_post_attention() {
            widths.extend([kind.columns(); 3]);
        }
        if cut.includes_mlp() {
            widths.extend(match cut {
                ModelComponentCut::Mlp => [7168, 3584, 1024],
                ModelComponentCut::Qwen2bMlp => [12288, 6144, 2048],
                _ => unreachable!(),
            });
        }
        for (id, width) in widths.into_iter().enumerate() {
            if plan.values()[id] != ValueShape::Vector(rows * width)
                || plan.precisions()[id] != ActivationPrecision::Bf16WidenedF32
            {
                return Err(Error::State(
                    "source component exact dense BF16 layouts".into(),
                ));
            }
        }
        state_effect(&nodes[4], StateSlot::Conv(ConvDomain(0)))?;
        state_effect(&nodes[5], StateSlot::Recurrent(RecurrentDomain(0)))?;
        let next = &plan.nodes()[cut.node_end()];
        let following = record
            .operator(cut.node_end())?
            .ok_or_else(|| Error::State("component next source producer absent".into()))?;
        let stop = if cut.includes_mlp() {
            following.call() == call
                && following.layer().map(|v| v.index()) == Some(1)
                && following.kind() == QwenOperation::ResidualRms(QwenRmsSite::Input)
                && matches!(
                    &next.operation,
                    PlanOperation::SourceOperator { state: None, .. }
                )
                && next.inputs == [ValueId(12), ValueId(9)]
                && next.outputs == [ValueId(13), ValueId(14)]
                && next.state.is_none()
        } else {
            following.kind()
                == QwenOperation::Projection(match cut {
                    ModelComponentCut::Gated => QwenProjection::GdnOut,
                    ModelComponentCut::PostAttention => QwenProjection::GateUp,
                    _ => unreachable!(),
                })
        };
        if !stop {
            return Err(Error::State(
                "source component exact stop before next original producer".into(),
            ));
        }
        // Charge the moved graph and immutable selectors before dropping family metadata.
        let gdn_parameters = vec![PreparedGdnParameters::prepare(
            source.source(),
            &record
                .operator(4)?
                .ok_or_else(|| Error::State("component convolution absent".into()))?,
            &record
                .operator(5)?
                .ok_or_else(|| Error::State("component recurrence absent".into()))?,
            [4, 5],
            record.plan(),
        )?]
        .into_boxed_slice();
        let bytes = record
            .binding_metadata_bytes()?
            .checked_add(
                std::mem::size_of::<PreparedPlan>() + std::mem::size_of::<PreparedComponent>(),
            )
            .and_then(|n| n.checked_add(gdn::metadata_bytes(&gdn_parameters).ok()?))
            .ok_or_else(|| Error::Limit("component model preparation metadata extent".into()))?;
        if bytes > metadata.bytes() {
            return Err(Error::Limit(
                "component model preparation metadata bound".into(),
            ));
        }
        let selected = match output {
            ComponentOutput::Gated => ValueId(6),
            ComponentOutput::PostAttention => ValueId(8),
            ComponentOutput::Mlp => ValueId(12),
        };
        let residual_output = cut.includes_post_attention().then_some(ValueId(9));
        let prepared = PreparedPlan {
            source: source.source().source().clone(),
            plan: record
                .into_plan()
                .retain_prepared_metadata(metadata.into_host_workspace()?)?,
            output: selected,
            residual_output,
            normalized: None,
            boundaries: Box::new([]),
            selected_row: None,
            completed_layers: 0,
            first_mlp_down_node: cut.includes_mlp().then_some(10),
            gdn_parameters,
        };
        let capture = ComponentKernelCapture {
            recipe_sha256: match kind {
                ModelComponentKind::Qwen2b => REFERENCE_2B_CAPTURE,
                ModelComponentKind::FixedPrompt(_) => FIXED_PROMPT_CAPTURE,
            },
            explicit_owned: qwen2b_arithmetic.is_some(),
        };
        Ok(PreparedComponent {
            prepared,
            output,
            capture,
            node_end: cut.node_end(),
            value_end: cut.value_end(),
            provider_bytes: cut.provider_bytes(),
            record_admission,
        })
    }
}

// Ordinary RMS had these gates independently of the component post-RMS binder.
fn require_ordinary_source(
    model: &OriginalWindowModelSource,
    kind: ModelComponentKind,
) -> Result<()> {
    use zerogravity_core::quant::QuantKind;
    let hidden = kind.columns();
    match kind {
        ModelComponentKind::Qwen2b => {
            let c = model.owner().config();
            let g = model.geometry();
            if model.owner().config_sha256()
                != "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
                || model.owner().verified_index_sha256()
                    != "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
                || model.source().source_bytes() != 4_548_221_488
                || model.source().staging_chunk_bytes() != 1024 * 1024
                || g.family != crate::config::Family::QwenHybrid
                || g.hidden_size != 2048
                || g.vocab_size != 248320
                || g.layers.len() != 24
                || g.head_dim != 256
                || g.num_attention_heads != 8
                || g.num_key_value_heads != 2
                || g.recurrent_key_heads != Some(16)
                || g.recurrent_value_heads != Some(16)
                || c.intermediate_size() != 6144
                || c.linear_key_head_dim() != 128
                || c.linear_value_head_dim() != 128
                || c.linear_conv_kernel_dim() != 4
                || c.rms_norm_eps().to_bits() != 1e-6f64.to_bits()
                || !c.tie_word_embeddings()
                || model.owner().text_tensor_count() != 320
            {
                return Err(Error::Unsupported(
                    "closed original2B RMS config/geometry/source".into(),
                ));
            }
        }
        ModelComponentKind::FixedPrompt(_) => {
            model.require_small_model_source()?;
            if model.source().source_bytes() != 1_746_942_600
                || model.source().staging_chunk_bytes() != 1024 * 1024
                || model.geometry().hidden_size != 1024
                || model.geometry().vocab_size != 248320
                || model.geometry().layers.len() != 24
                || model.owner().config().rms_norm_eps() != 1e-6
            {
                return Err(Error::Unsupported(
                    "closed fixed prompt RMS config/source extent".into(),
                ));
            }
        }
    }
    let embedding = model.matrix_descriptor(M::Embedding)?;
    model.validate_matrix(M::Embedding, &embedding)?;
    if embedding.kind != QuantKind::Bf16 || embedding.rows != 248320 || embedding.columns != hidden
    {
        return Err(Error::Unsupported(
            "component original BF16 embedding geometry".into(),
        ));
    }
    for role in [P::InputNorm(0), P::FinalNorm] {
        let p = model.parameter_descriptor(role)?;
        model.validate_parameter(role, &p)?;
        if p.kind != QuantKind::Bf16 || p.shape != [hidden as u64] || p.bytes != hidden * 2 {
            return Err(Error::Unsupported(
                "component original raw gamma role/shape".into(),
            ));
        }
    }
    if model.owner().config().rms_norm_eps() != 1e-6 {
        return Err(Error::Unsupported("component RMS exact epsilon".into()));
    }
    Ok(())
}

#[cfg(test)]
#[path = "components/tests.rs"]
mod tests;
