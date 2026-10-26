//! Actual shared graph/source/SSA and native-capture joins; no model eligibility.
//! Prior source-family policy: qwen2b_component::recipe; now model prepare/components.
use crate::resident_program::source_bindings::SourceBindings;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::*;
use zerogravity_core::{Error, Result};
pub(super) const NODE_END: usize = 6;
pub(super) const VALUE_END: usize = 7;
// Explicit physical catalogue identity. Never selected from matching geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComponentKind {
    Qwen2b,
    FixedPrompt(crate::cuda::gdn::FixedPromptGdnArithmetic),
}
impl ComponentKind {
    pub(crate) fn columns(self) -> usize {
        match self {
            Self::Qwen2b => 2048,
            Self::FixedPrompt(_) => 1024,
        }
    }
    pub(crate) fn require_call(self, call: &SourceCall) -> Result<()> {
        let valid = match self {
            Self::Qwen2b => matches!(call.rows().len(), 1 | 64),
            Self::FixedPrompt(arithmetic) => {
                arithmetic.require_owned()?;
                call.rows().len() == 47
            }
        };
        if !valid
            || call.mode() != SourceCallMode::Prefill
            || call.rows().base() != 0
            || call.rows().capacity() != 512
        {
            return Err(Error::Unsupported(
                "component native capture initial call/extent differs".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentOutput {
    Gated,
    PostAttention,
    Mlp,
}
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
pub struct ComponentRecipe {
    cut: ComponentCut,
    kind: ComponentKind,
    qwen2b_arithmetic: Option<crate::cuda::gdn::Qwen2bGdnArithmetic>,
    bindings: Arc<SourceBindings>,
    record_admission: Option<Arc<WindowWorkspace>>,
    _metadata: WindowWorkspace,
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
    /// Cold composition consumes explicit model-selected pins and the actual graph.
    /// Matching pins alone authenticate neither native bytes nor source-family eligibility.
    pub fn bind(
        bindings: Arc<SourceBindings>,
        output: ComponentOutput,
        recipe_sha256: &str,
        explicit_owned: bool,
        record_admission: Option<Arc<WindowWorkspace>>,
    ) -> Result<Arc<Self>> {
        let (kind, qwen2b_arithmetic) = match recipe_sha256 {
            REFERENCE_2B_CAPTURE => (
                ComponentKind::Qwen2b,
                explicit_owned
                    .then_some(crate::cuda::gdn::Qwen2bGdnArithmetic::OwnedSolve4Recompute3),
            ),
            FIXED_PROMPT_CAPTURE if !explicit_owned => (
                ComponentKind::FixedPrompt(
                    crate::cuda::gdn::FixedPromptGdnArithmetic::OwnedFixedPromptSolve4,
                ),
                None,
            ),
            _ => {
                return Err(Error::Unsupported(
                    "unknown component reference capture/owned selection".into(),
                ));
            }
        };
        let cut = match (output, kind) {
            (ComponentOutput::Gated, _) => ComponentCut::Gated,
            (ComponentOutput::PostAttention, _) => ComponentCut::PostAttention,
            (ComponentOutput::Mlp, ComponentKind::Qwen2b) => ComponentCut::Qwen2bMlp,
            (ComponentOutput::Mlp, ComponentKind::FixedPrompt(_)) => ComponentCut::Mlp,
        };
        kind.require_call(bindings.call())?;
        if let Some(arithmetic) = qwen2b_arithmetic {
            arithmetic.require_owned(match bindings.call().rows().len() {
                1 => crate::cuda::gdn::GdnPhase::InitialOne,
                64 => crate::cuda::gdn::GdnPhase::Initial64,
                _ => {
                    return Err(Error::Unsupported(
                        "owned component exact initial call".into(),
                    ));
                }
            })?;
        }
        match (cut, record_admission.as_ref()) {
            (ComponentCut::Qwen2bMlp, Some(ticket))
                if qwen2b_arithmetic.is_some()
                    && ticket.bytes() == 32 * 1024 * 1024
                    && ticket.source().same_issuer(bindings.source())
                    && bindings
                        .source_record_admission()
                        .is_some_and(|retained| Arc::ptr_eq(ticket, retained)) => {}
            (ComponentCut::Qwen2bMlp, _) => {
                return Err(Error::State(
                    "same owned component32MiB record admission required".into(),
                ));
            }
            (_, None) => {}
            (_, Some(_)) => {
                return Err(Error::Unsupported(
                    "component cannot borrow separate MLP record admission".into(),
                ));
            }
        }
        let source = bindings.source();
        let plan = bindings.plan();
        let call = bindings.call();
        let metadata = source.reserve_workspace(64 * 1024)?;
        if plan.nodes().len() != 255
            || plan.values().len() != 327
            || plan.required_domains().len() != 42
            || plan.proposed_domains().len() != 42
            || plan.frontier_delta() != 0
            || plan.logits().is_some()
            || plan.is_operator_slice()
            || plan.source() != call.source()
            || source.source_identity() != call.source()
            || plan.source_call() != Some(call)
        {
            return Err(Error::Unsupported(
                "component actual full graph/source/call shape differs".into(),
            ));
        }
        let nodes = &plan.nodes()[..cut.node_end()];
        for (index, node) in nodes.iter().enumerate() {
            let PlanOperation::SourceOperator {
                descriptor,
                state: None,
            } = &node.operation
            else {
                return Err(Error::State(
                    "component actual source operator absent".into(),
                ));
            };
            descriptor.validate()?;
            if descriptor.call() != call
                || descriptor.numerical_contract()
                    != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            {
                return Err(Error::State(
                    "component foreign actual operator/call/contract".into(),
                ));
            }
            let kind_matches = match (index, descriptor.kind()) {
                (0, SourceOperatorKind::EmbeddingRows) => true,
                (1, SourceOperatorKind::Rms(g)) | (7, SourceOperatorKind::ResidualRms(g)) => {
                    g.width == kind.columns()
                        && g.epsilon == 1e-6
                        && g.gamma == SourceGamma::OnePlus
                }
                (2 | 3 | 6 | 8 | 10, SourceOperatorKind::Projection { selected_row: None }) => true,
                (4, SourceOperatorKind::CausalConvSplit { .. })
                | (5, SourceOperatorKind::GatedDelta { .. })
                | (9, SourceOperatorKind::SiluAndMul { .. }) => true,
                _ => false,
            };
            if !kind_matches || (!matches!(index, 4 | 5) && node.state.is_some()) {
                return Err(Error::State(
                    "component actual native operator/state kind differs".into(),
                ));
            }
            for (output_index, &id) in node.outputs.iter().enumerate() {
                let tag = plan
                    .source_value_provenance(id)?
                    .ok_or_else(|| Error::State("component actual SSA producer absent".into()))?;
                if tag.call() != call
                    || tag.node_index() != index
                    || tag.output_index() != output_index
                    || tag.inputs() != node.inputs
                {
                    return Err(Error::State(
                        "component exact actual SSA source producer differs".into(),
                    ));
                }
            }
        }
        let expected: &[(&[ValueId], &[ValueId])] = &[
            (&[], &[ValueId(0)]),
            (&[ValueId(0)], &[ValueId(1)]),
            (&[ValueId(1)], &[ValueId(2)]),
            (&[ValueId(1)], &[ValueId(3)]),
            (&[ValueId(2), ValueId(3)], &[ValueId(4), ValueId(5)]),
            (&[ValueId(4), ValueId(5), ValueId(3)], &[ValueId(6)]),
            (&[ValueId(6)], &[ValueId(7)]),
            (&[ValueId(7), ValueId(0)], &[ValueId(8), ValueId(9)]),
            (&[ValueId(8)], &[ValueId(10)]),
            (&[ValueId(10)], &[ValueId(11)]),
            (&[ValueId(11)], &[ValueId(12)]),
        ];
        for (node, (inputs, outputs)) in nodes.iter().zip(expected) {
            if node.inputs != *inputs || node.outputs != *outputs {
                return Err(Error::State(
                    "component exact native SSA layout differs".into(),
                ));
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
                _ => return Err(Error::State("component MLP extent".into())),
            });
        }
        for (index, width) in widths.into_iter().enumerate() {
            if plan.values()[index] != ValueShape::Vector(call.rows().len() * width)
                || plan.precisions()[index] != ActivationPrecision::Bf16WidenedF32
            {
                return Err(Error::State(
                    "component native exact BF16 dense layout".into(),
                ));
            }
        }
        state_effect(&nodes[4], StateSlot::Conv(ConvDomain(0)))?;
        state_effect(&nodes[5], StateSlot::Recurrent(RecurrentDomain(0)))?;
        Ok(Arc::new(Self {
            cut,
            kind,
            qwen2b_arithmetic,
            bindings,
            record_admission,
            _metadata: metadata,
        }))
    }
    pub(crate) fn qwen2b_mlp_record_admission_retained(&self) -> bool {
        self.cut == ComponentCut::Qwen2bMlp
            && self.record_admission.as_ref().is_some_and(|ticket| {
                ticket.bytes() == 32 * 1024 * 1024
                    && ticket.source().same_issuer(self.source())
                    && self
                        .bindings
                        .source_record_admission()
                        .is_some_and(|retained| Arc::ptr_eq(ticket, retained))
            })
    }
    pub(crate) fn qwen2b_arithmetic(&self) -> Option<crate::cuda::gdn::Qwen2bGdnArithmetic> {
        self.qwen2b_arithmetic
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
    pub(crate) fn bindings(&self) -> &Arc<SourceBindings> {
        &self.bindings
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        self.bindings.source()
    }
    pub(crate) fn nodes(&self) -> &[PlanNode] {
        &self.bindings.plan().nodes()[..self.node_end()]
    }
    pub(crate) fn operator(&self, index: usize) -> Result<&SourceOperator> {
        match self.nodes().get(index).map(|node| &node.operation) {
            Some(PlanOperation::SourceOperator { descriptor, .. }) => Ok(descriptor),
            _ => Err(Error::State(
                "component mathematical operator absent".into(),
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
        self.bindings
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
const REFERENCE_2B_CAPTURE: &str =
    "356a1149294dca85172679bb48d3070777f77d3e98eba6071ed5408a79e647cd";
const FIXED_PROMPT_CAPTURE: &str =
    "dd8c09131ee5e88efca478044a9a41869532453a9c93f972b2861022740bb2dd";

#[cfg(test)]
pub(crate) const FIXED_PROMPT_IDS: [u32; 47] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];
