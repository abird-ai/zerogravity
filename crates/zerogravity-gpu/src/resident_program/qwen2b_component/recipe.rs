//! Checked view of the SAME recorded SSA; no component IR or model frontier.
use std::sync::Arc;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    ActivationPrecision, ConvDomain, FullModelProfile, PlanNode, PlanOperation,
    RecordedVllm08Model, RecurrentDomain, StateEffect, StateSlot, ValueId, ValueShape, Vllm08Kind,
    Vllm08Operator, Vllm08Projection, Vllm08RmsSite, VllmBf16Mode, VllmBf16Phase, VllmValueOrigin,
};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::{
    BoundFullModelSource, QwenMatrixRole as M, QwenParameterRole as P,
};

pub(super) const NODE_END: usize = 6;
pub(super) const VALUE_END: usize = 7;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComponentKind {
    Qwen2b,
    Canonical47(crate::vllm_cuda_gdn::Canonical47GdnArithmetic),
}
impl ComponentKind {
    pub(crate) fn profile(self) -> FullModelProfile {
        match self {
            Self::Qwen2b => FullModelProfile::Qwen2B,
            Self::Canonical47(_) => FullModelProfile::Qwen08,
        }
    }
    pub(crate) fn columns(self) -> usize {
        match self {
            Self::Qwen2b => 2048,
            Self::Canonical47(_) => 1024,
        }
    }
    pub(crate) fn require_request(
        self,
        request: &zerogravity_core::accelerator::program::VllmBf16Invocation,
    ) -> Result<()> {
        request.validate()?;
        if request.layer().index() != 0
            || request.teachers().short_hf_span().is_some()
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
                    request.legacy_phase()?,
                    VllmBf16Phase::InitialOne | VllmBf16Phase::Initial64
                ) =>
            {
                Ok(())
            }
            Self::Canonical47(arithmetic)
                if request.mode() == VllmBf16Mode::Prefill
                    && request.rows() == 47
                    && request.legacy_phase().is_err()
                    && request.teachers().ids() == CANONICAL47_PROMPT =>
            {
                arithmetic.require_owned()
            }
            _ => Err(Error::Unsupported(
                "closed source component geometry/prompt/arithmetic".into(),
            )),
        }
    }
}
pub(crate) const CANONICAL47_PROMPT: [u32; 47] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComponentCut {
    Gated,
    PostAttention,
    Mlp,
    Qwen2bMlp,
}
impl ComponentCut {
    pub(crate) fn includes_post_attention(self) -> bool {
        matches!(self, Self::PostAttention | Self::Mlp | Self::Qwen2bMlp)
    }
    pub(crate) fn includes_mlp(self) -> bool {
        matches!(self, Self::Mlp | Self::Qwen2bMlp)
    }
    fn require_kind(self, kind: ComponentKind) -> Result<()> {
        if self == Self::Mlp && !matches!(kind, ComponentKind::Canonical47(_)) {
            return Err(Error::Unsupported(
                "MLP component is canonical47 only".into(),
            ));
        }
        if self == Self::Qwen2bMlp && kind != ComponentKind::Qwen2b {
            return Err(Error::Unsupported(
                "explicit2B MLP source family only".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn node_end(self) -> usize {
        match self {
            Self::Gated => NODE_END,
            Self::PostAttention => 8,
            Self::Mlp | Self::Qwen2bMlp => 11,
        }
    }
    pub(crate) fn value_end(self) -> usize {
        match self {
            Self::Gated => VALUE_END,
            Self::PostAttention => 10,
            Self::Mlp | Self::Qwen2bMlp => 13,
        }
    }
    pub(crate) fn provider_bytes(self) -> usize {
        match self {
            Self::Gated => super::PROVIDER_BYTES,
            Self::PostAttention | Self::Mlp => super::POST_ATTENTION_PROVIDER_BYTES,
            Self::Qwen2bMlp => super::QWEN2B_MLP_PROVIDER_BYTES,
        }
    }
}
pub(crate) struct ComponentRecipe {
    cut: ComponentCut,
    kind: ComponentKind,
    qwen2b_arithmetic: Option<crate::vllm_cuda_gdn::Qwen2bGdnArithmetic>,
    source: BoundFullModelSource,
    record: Arc<RecordedVllm08Model>,
    record_admission: Option<Arc<WindowWorkspace>>,
    _metadata: WindowWorkspace,
}
fn require_qwen2b_arithmetic(
    kind: ComponentKind,
    cut: ComponentCut,
    rows: usize,
    arithmetic: crate::vllm_cuda_gdn::Qwen2bGdnArithmetic,
) -> Result<()> {
    if kind != ComponentKind::Qwen2b
        || !matches!(
            cut,
            ComponentCut::Gated | ComponentCut::PostAttention | ComponentCut::Qwen2bMlp
        )
    {
        return Err(Error::Unsupported(
            "explicit2B tuple only on exact finite initial route".into(),
        ));
    }
    arithmetic.require_owned(match rows {
        1 => crate::vllm_cuda_gdn::VllmGdnPhase::InitialOne,
        64 => crate::vllm_cuda_gdn::VllmGdnPhase::Initial64,
        _ => {
            return Err(Error::Unsupported(
                "explicit2B initial1/64 arithmetic".into(),
            ));
        }
    })
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
impl ComponentRecipe {
    pub(super) fn bind(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
    ) -> Result<Arc<Self>> {
        Self::bind_kind(source, record, ComponentKind::Qwen2b)
    }
    pub(super) fn bind_qwen2b_arithmetic(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        arithmetic: crate::vllm_cuda_gdn::Qwen2bGdnArithmetic,
    ) -> Result<Arc<Self>> {
        Self::bind_cut_arithmetic(
            source,
            record,
            ComponentKind::Qwen2b,
            ComponentCut::Gated,
            Some(arithmetic),
            None,
        )
    }
    pub(crate) fn bind_qwen2b_post_attention_arithmetic(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        arithmetic: crate::vllm_cuda_gdn::Qwen2bGdnArithmetic,
    ) -> Result<Arc<Self>> {
        Self::bind_cut_arithmetic(
            source,
            record,
            ComponentKind::Qwen2b,
            ComponentCut::PostAttention,
            Some(arithmetic),
            None,
        )
    }
    pub(crate) fn bind_qwen2b_mlp_arithmetic(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        arithmetic: crate::vllm_cuda_gdn::Qwen2bGdnArithmetic,
        record_admission: Arc<WindowWorkspace>,
    ) -> Result<Arc<Self>> {
        Self::bind_cut_arithmetic(
            source,
            record,
            ComponentKind::Qwen2b,
            ComponentCut::Qwen2bMlp,
            Some(arithmetic),
            Some(record_admission),
        )
    }
    pub(crate) fn qwen2b_mlp_record_admission_retained(&self) -> bool {
        self.cut == ComponentCut::Qwen2bMlp
            && self.record_admission.as_ref().is_some_and(|ticket| {
                ticket.bytes() == 32 * 1024 * 1024
                    && ticket.source().same_issuer(self.source.source().source())
            })
    }
    pub(crate) fn qwen2b_arithmetic(&self) -> Option<crate::vllm_cuda_gdn::Qwen2bGdnArithmetic> {
        self.qwen2b_arithmetic
    }
    pub(super) fn bind_canonical47(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        arithmetic: crate::vllm_cuda_gdn::Canonical47GdnArithmetic,
    ) -> Result<Arc<Self>> {
        Self::bind_kind(source, record, ComponentKind::Canonical47(arithmetic))
    }
    fn bind_kind(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        kind: ComponentKind,
    ) -> Result<Arc<Self>> {
        Self::bind_cut(source, record, kind, ComponentCut::Gated)
    }
    pub(crate) fn bind_post_attention(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        kind: ComponentKind,
    ) -> Result<Arc<Self>> {
        Self::bind_cut(source, record, kind, ComponentCut::PostAttention)
    }
    pub(crate) fn bind_mlp(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        arithmetic: crate::vllm_cuda_gdn::Canonical47GdnArithmetic,
    ) -> Result<Arc<Self>> {
        Self::bind_cut(
            source,
            record,
            ComponentKind::Canonical47(arithmetic),
            ComponentCut::Mlp,
        )
    }
    fn bind_cut(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        kind: ComponentKind,
        cut: ComponentCut,
    ) -> Result<Arc<Self>> {
        Self::bind_cut_arithmetic(source, record, kind, cut, None, None)
    }
    fn bind_cut_arithmetic(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        kind: ComponentKind,
        cut: ComponentCut,
        qwen2b_arithmetic: Option<crate::vllm_cuda_gdn::Qwen2bGdnArithmetic>,
        record_admission: Option<Arc<WindowWorkspace>>,
    ) -> Result<Arc<Self>> {
        cut.require_kind(kind)?;
        match (cut, record_admission.as_ref()) {
            (ComponentCut::Qwen2bMlp, Some(ticket))
                if ticket.bytes() == 32 * 1024 * 1024
                    && ticket.source().same_issuer(source.source().source()) => {}
            (ComponentCut::Qwen2bMlp, _) => {
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
        if cut == ComponentCut::Qwen2bMlp && qwen2b_arithmetic.is_none() {
            return Err(Error::Unsupported(
                "new2B MLP requires explicit owned arithmetic".into(),
            ));
        }
        kind.require_request(record.call().request())?;
        if let Some(arithmetic) = qwen2b_arithmetic {
            require_qwen2b_arithmetic(kind, cut, record.call().rows(), arithmetic)?;
        }
        let metadata = source.source().source().reserve_workspace(64 * 1024)?;
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
            || plan.vllm_gdn_fragment() != Some(call.request())
        {
            return Err(Error::Unsupported(
                "authentic initial2B six-node source prefix only".into(),
            ));
        }
        let rows = call.rows();
        let nodes = &plan.nodes()[..cut.node_end()];
        let PlanOperation::TypedVllmBf16Embedding { descriptor } = &nodes[0].operation else {
            return Err(Error::State(
                "source2B original embedding node absent".into(),
            ));
        };
        source
            .source()
            .validate_matrix(M::Embedding, descriptor.tensor())?;
        if descriptor.model_profile() != kind.profile()
            || descriptor.teachers() != call.request().teachers()
            || !nodes[0].inputs.is_empty()
            || nodes[0].outputs != [ValueId(0)]
            || nodes[0].state.is_some()
        {
            return Err(Error::State(
                "source2B original embedding producer/teacher issuer".into(),
            ));
        }
        let mut expected = vec![
            (Vllm08Kind::InputRms, vec![ValueId(0)], vec![ValueId(1)]),
            (
                Vllm08Kind::Projection(Vllm08Projection::Qkvz),
                vec![ValueId(1)],
                vec![ValueId(2)],
            ),
            (
                Vllm08Kind::Projection(Vllm08Projection::Ba),
                vec![ValueId(1)],
                vec![ValueId(3)],
            ),
            (
                Vllm08Kind::GdnConv,
                vec![ValueId(2), ValueId(3)],
                vec![ValueId(4), ValueId(5)],
            ),
            (
                Vllm08Kind::GdnRecurrent,
                vec![ValueId(4), ValueId(5), ValueId(3)],
                vec![ValueId(6)],
            ),
        ];
        if cut.includes_post_attention() {
            expected.extend([
                (
                    Vllm08Kind::Projection(Vllm08Projection::GdnOut),
                    vec![ValueId(6)],
                    vec![ValueId(7)],
                ),
                (
                    Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention),
                    vec![ValueId(7), ValueId(0)],
                    vec![ValueId(8), ValueId(9)],
                ),
            ]);
        }
        if cut.includes_mlp() {
            expected.extend([
                (
                    Vllm08Kind::Projection(Vllm08Projection::GateUp),
                    vec![ValueId(8)],
                    vec![ValueId(10)],
                ),
                (Vllm08Kind::SiluAndMul, vec![ValueId(10)], vec![ValueId(11)]),
                (
                    Vllm08Kind::Projection(Vllm08Projection::Down),
                    vec![ValueId(11)],
                    vec![ValueId(12)],
                ),
            ]);
        }
        for (index, (node, (kind, inputs, outputs))) in nodes[1..].iter().zip(expected).enumerate()
        {
            let PlanOperation::TypedVllm08 { descriptor, state } = &node.operation else {
                return Err(Error::State(
                    "actual source2B typed prefix node absent".into(),
                ));
            };
            descriptor.validate()?;
            if descriptor.call() != call
                || descriptor.kind() != kind
                || descriptor.layer().map(|x| x.index()) != Some(0)
                || node.inputs != inputs
                || node.outputs != outputs
                || state.is_some()
                || !matches!(kind, Vllm08Kind::GdnConv | Vllm08Kind::GdnRecurrent)
                    && node.state.is_some()
            {
                return Err(Error::State(
                    "source2B exact ordered first-six SSA/effect recipe".into(),
                ));
            }
            let (matrices, parameters): (&[M], &[P]) = match kind {
                Vllm08Kind::InputRms => (&[], &[P::InputNorm(0)]),
                Vllm08Kind::Projection(Vllm08Projection::Qkvz) => {
                    (&[M::GdnQkv(0), M::GdnGate(0)], &[])
                }
                Vllm08Kind::Projection(Vllm08Projection::Ba) => {
                    (&[M::GdnBeta(0), M::GdnAlpha(0)], &[])
                }
                Vllm08Kind::GdnConv => (&[], &[P::GdnConvolution(0)]),
                Vllm08Kind::GdnRecurrent => (&[], &[P::GdnA(0), P::GdnDt(0), P::GdnNorm(0)]),
                Vllm08Kind::Projection(Vllm08Projection::GdnOut) => (&[M::GdnOutput(0)], &[]),
                Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention) => (&[], &[P::FfnNorm(0)]),
                Vllm08Kind::Projection(Vllm08Projection::GateUp) => {
                    (&[M::FfnGate(0), M::FfnUp(0)], &[])
                }
                Vllm08Kind::SiluAndMul => (&[], &[]),
                Vllm08Kind::Projection(Vllm08Projection::Down) => (&[M::FfnDown(0)], &[]),
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
                let tag = plan
                    .vllm_value_provenance(*id)?
                    .ok_or_else(|| Error::State("source2B actual output tag absent".into()))?;
                tag.require_invocation(call.request())?;
                let VllmValueOrigin::FullModel(origin) = tag.origin() else {
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
        let mut widths = vec![kind.columns(), kind.columns(), 8192, 32, 6144, 2048, 2048];
        if cut.includes_post_attention() {
            widths.extend([kind.columns(); 3]);
        }
        if cut.includes_mlp() {
            widths.extend(match cut {
                ComponentCut::Mlp => [7168, 3584, 1024],
                ComponentCut::Qwen2bMlp => [12288, 6144, 2048],
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
        let stop = if cut.includes_mlp() {
            matches!(&next.operation, PlanOperation::TypedVllm08 { descriptor, state: None }
                if descriptor.call() == call && descriptor.layer().map(|v|v.index()) == Some(1)
                    && descriptor.kind() == Vllm08Kind::ResidualRms(Vllm08RmsSite::Input))
                && next.inputs == [ValueId(12), ValueId(9)]
                && next.outputs == [ValueId(13), ValueId(14)]
                && next.state.is_none()
        } else {
            matches!(&next.operation,PlanOperation::TypedVllm08 { descriptor,.. }
            if descriptor.kind() == Vllm08Kind::Projection(match cut {
                ComponentCut::Gated=>Vllm08Projection::GdnOut,
                ComponentCut::PostAttention=>Vllm08Projection::GateUp,
                ComponentCut::Mlp | ComponentCut::Qwen2bMlp=>unreachable!(),
            }))
        };
        if !stop {
            return Err(Error::State(
                "source component exact stop before next original producer".into(),
            ));
        }
        Ok(Arc::new(Self {
            cut,
            kind,
            qwen2b_arithmetic,
            source,
            record,
            record_admission,
            _metadata: metadata,
        }))
    }
    pub(crate) fn cut(&self) -> ComponentCut {
        self.cut
    }
    pub(crate) fn node_end(&self) -> usize {
        self.cut.node_end()
    }
    pub(crate) fn value_end(&self) -> usize {
        self.cut.value_end()
    }
    pub(crate) fn provider_bytes(&self) -> usize {
        self.cut.provider_bytes()
    }
    pub(crate) fn kind(&self) -> ComponentKind {
        self.kind
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    pub(crate) fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    pub(crate) fn nodes(&self) -> &[PlanNode] {
        &self.record.plan().nodes()[..self.node_end()]
    }
    pub(crate) fn operator(&self, index: usize) -> Result<&Vllm08Operator> {
        match self.nodes().get(index).map(|n| &n.operation) {
            Some(PlanOperation::TypedVllm08 { descriptor, .. }) => Ok(descriptor),
            _ => Err(Error::State(
                "not an original source2B prefix operator".into(),
            )),
        }
    }
    pub(crate) fn effects(&self) -> Result<[StateEffect; 2]> {
        Ok([
            state_effect(&self.nodes()[4], StateSlot::Conv(ConvDomain(0)))?,
            state_effect(&self.nodes()[5], StateSlot::Recurrent(RecurrentDomain(0)))?,
        ])
    }
    pub(crate) fn carries(&self) -> impl Iterator<Item = StateSlot> + '_ {
        self.record
            .plan()
            .required_domains()
            .iter()
            .copied()
            .filter(|slot| {
                !matches!(
                    slot,
                    StateSlot::Conv(ConvDomain(0)) | StateSlot::Recurrent(RecurrentDomain(0))
                )
            })
    }
}

#[cfg(test)]
#[path = "canonical47_tests.rs"]
mod canonical47_tests;

#[cfg(test)]
mod mlp_cut_tests {
    use super::*;
    #[test]
    fn mlp_is_additive47_only_with_distinct_stop_and_value_extent() {
        assert_eq!(
            (ComponentCut::Mlp.node_end(), ComponentCut::Mlp.value_end()),
            (11, 13)
        );
        assert_eq!(ComponentCut::Mlp.provider_bytes(), 224 * 1024 * 1024);
        assert!(
            ComponentCut::Mlp
                .require_kind(ComponentKind::Qwen2b)
                .is_err()
        );
        assert!(
            ComponentCut::Mlp
                .require_kind(ComponentKind::Canonical47(
                    crate::vllm_cuda_gdn::Canonical47GdnArithmetic::OwnedCanonical47Solve4
                ))
                .is_ok()
        );
        assert_eq!(
            (
                ComponentCut::Gated.node_end(),
                ComponentCut::Gated.value_end()
            ),
            (6, 7)
        );
        assert_eq!(
            (
                ComponentCut::PostAttention.node_end(),
                ComponentCut::PostAttention.value_end()
            ),
            (8, 10)
        );
    }
}

#[cfg(test)]
#[path = "explicit2b_post_attention_tests.rs"]
mod explicit2b_post_attention_tests;

#[cfg(test)]
mod qwen2b_mlp_scope_tests {
    use super::*;
    #[test]
    fn new_mlp_cut_keeps_old47_and_six_eight_node_guards() {
        let cut = ComponentCut::Qwen2bMlp;
        assert_eq!(
            (cut.node_end(), cut.value_end(), cut.provider_bytes()),
            (11, 13, 320 * 1024 * 1024)
        );
        assert!(cut.includes_mlp() && cut.includes_post_attention());
        assert!(cut.require_kind(ComponentKind::Qwen2b).is_ok());
        let old = ComponentKind::Canonical47(
            crate::vllm_cuda_gdn::Canonical47GdnArithmetic::OwnedCanonical47Solve4,
        );
        assert!(cut.require_kind(old).is_err());
        assert!(
            ComponentCut::Mlp
                .require_kind(ComponentKind::Qwen2b)
                .is_err()
        );
        assert_eq!(ComponentCut::Mlp.provider_bytes(), 224 * 1024 * 1024);
        assert_eq!(
            (
                ComponentCut::Gated.node_end(),
                ComponentCut::Gated.value_end()
            ),
            (6, 7)
        );
        assert_eq!(
            (
                ComponentCut::PostAttention.node_end(),
                ComponentCut::PostAttention.value_end()
            ),
            (8, 10)
        );
    }
    #[test]
    fn arithmetic_seal_does_not_retag_old47_or_accept_new_rows() {
        let a = crate::vllm_cuda_gdn::Qwen2bGdnArithmetic::OwnedSolve4Recompute3;
        for rows in [1, 64] {
            assert!(
                require_qwen2b_arithmetic(ComponentKind::Qwen2b, ComponentCut::Qwen2bMlp, rows, a)
                    .is_ok()
            );
        }
        for rows in [0, 2, 47, 65] {
            assert!(
                require_qwen2b_arithmetic(ComponentKind::Qwen2b, ComponentCut::Qwen2bMlp, rows, a)
                    .is_err()
            );
        }
        assert!(require_qwen2b_arithmetic(ComponentKind::Qwen2b, ComponentCut::Mlp, 1, a).is_err());
    }
}
