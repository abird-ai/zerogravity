//! Model-owned source/recipe preparation; one shared plan, no device/root permit.
//! Prior binders: short_hf_model, canonical47_model, resident_program::native.
use super::*;
use crate::plan::{
    GdnProjection, PromptCall, QwenFragmentPlan, QwenFragmentView, QwenGdnCall, QwenGdnConv,
    QwenGdnRecurrent, QwenOperation, QwenOperator, QwenPlan, QwenProjection, QwenRmsSite,
    QwenSourceProfile,
};
use std::collections::{BTreeMap, BTreeSet};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::*;

/// Family-authenticated preparation. The backend still joins actual source,
/// mathematical/SSA/ABI data, physical storage and the original session attempt.
pub struct PreparedPlan {
    source: VerifiedTensorSource,
    plan: ForwardPlan,
    output: ValueId,
    residual_output: Option<ValueId>,
    normalized: Option<ValueId>,
    boundaries: Box<[[ValueId; 2]]>,
    selected_row: Option<usize>,
    completed_layers: usize,
    first_mlp_down_node: Option<usize>,
    gdn_parameters: Box<[PreparedGdnParameters]>,
}
impl PreparedPlan {
    pub fn gdn_parameters(&self) -> &[PreparedGdnParameters] {
        &self.gdn_parameters
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn output(&self) -> ValueId {
        self.output
    }
    pub fn residual_output(&self) -> Option<ValueId> {
        self.residual_output
    }
    pub fn normalized(&self) -> Option<ValueId> {
        self.normalized
    }
    pub fn boundaries(&self) -> &[[ValueId; 2]] {
        &self.boundaries
    }
    pub fn selected_row(&self) -> Option<usize> {
        self.selected_row
    }
    pub fn completed_layers(&self) -> usize {
        self.completed_layers
    }
    pub fn first_mlp_down_node(&self) -> Option<usize> {
        self.first_mlp_down_node
    }
    /// Move the one graph. Read required observation selectors first; selected
    /// values alone confer no native or publication authority.
    pub fn into_parts(self) -> (VerifiedTensorSource, ForwardPlan, ValueId, Option<ValueId>) {
        (self.source, self.plan, self.output, self.residual_output)
    }
}
const METADATA_BYTES: usize = 512 * 1024;
mod gdn;
pub use gdn::PreparedGdnParameters;
mod native;
mod positions;
mod qualification;
pub use qualification::PreparedQualificationSource;
mod rms;
pub use native::ReferenceKernelCapture;
pub use rms::{PreparedFinalRmsSource, PreparedRmsSource};
mod components;
mod fixed_prompt;
pub use components::{
    ComponentImplementation, ComponentKernelCapture, ComponentOutput, PreparedComponent,
};

impl BoundPromptCall {
    /// Exact supported text workload and source, before device/service preparation.
    pub fn prepare_program(self, record: QwenPlan) -> Result<PreparedPlan> {
        self.call().validate()?;
        let source = self.source().source();
        if self.source().model_profile() != self.call().model_profile()
            || source.source_identity() != record.call().source()
            || source.source().staging_chunk_bytes() != 1024 * 1024
            || source.owner().config().rms_norm_eps() != 1e-6
        {
            return Err(Error::State(
                "Qwen text authenticated source/profile/chunk/RMS".into(),
            ));
        }
        let metadata = source.source().reserve_workspace(METADATA_BYTES)?;
        let analysis =
            analyze_text_record(self.call(), &record, |op| source.validate_operator(op))?;
        let indices = analysis
            .rms
            .capacity()
            .checked_add(analysis.projections.capacity())
            .and_then(|n| n.checked_mul(std::mem::size_of::<usize>()))
            .ok_or_else(|| Error::Limit("Qwen preparation indices".into()))?;
        let bytes = record
            .binding_metadata_bytes()?
            .checked_add(indices)
            .and_then(|n| n.checked_add(256 * 1024 + std::mem::size_of::<PreparedPlan>()))
            .ok_or_else(|| Error::Limit("Qwen preparation metadata extent".into()))?;
        if bytes > metadata.bytes() {
            return Err(Error::Limit("Qwen preparation metadata bound".into()));
        }
        require_analysis_values(&record, &analysis)?;
        require_embedding(source, record.plan())?;
        require_state_layout(source, record.plan())?;
        {
            let pairs = gdn::for_record(source, &record)?;
            prepare_whole(source.source().clone(), record, metadata, pairs)
        }
    }
}
impl OriginalWindowModelSource {
    /// Exact original fixed prompt/cached positions; physical coverage stays separate.
    pub fn prepare_fixed_prompt_program(&self, record: QwenPlan) -> Result<PreparedPlan> {
        fixed_prompt::prepare(self, record)
    }
    /// Preserve the existing .8B source and finite reference-call gate.
    pub fn prepare_reference_program(&self, record: QwenPlan) -> Result<PreparedPlan> {
        record.call().request().reference_phase()?;
        require_reference_record(self, &record)?;
        let metadata = self.source().reserve_workspace(METADATA_BYTES)?;
        require_embedding(self, record.plan())?;
        for index in 0..record.plan().nodes().len() {
            if let Some(operator) = record.operator(index)? {
                self.validate_operator(&operator)?;
            }
        }
        require_state_layout(self, record.plan())?;
        {
            let pairs = gdn::for_record(self, &record)?;
            prepare_whole(self.source().clone(), record, metadata, pairs)
        }
    }
    /// Existing initial GDN/MLP cuts only; original root remains unpublished.
    pub fn prepare_fragment(&self, record: QwenFragmentPlan) -> Result<PreparedPlan> {
        require_fragment_config(self)?;
        let recipe = require_fragment_recipe(self, record.view()?)?;
        let metadata = self.source().reserve_workspace(METADATA_BYTES)?;
        require_state_layout(self, record.plan())?;
        let output = record.output();
        let residual_output = record.residual_output();
        require_selected_value(record.plan(), output)?;
        if let Some(id) = residual_output {
            require_selected_value(record.plan(), id)?;
        }
        let completed_layers = record.completed_layers();
        let first_mlp_down_node = recipe.first_down;
        let site_bytes = recipe.gdns.capacity() * std::mem::size_of::<[usize; 2]>()
            + recipe.fused.capacity() * std::mem::size_of::<usize>();
        let gdn_parameters = gdn::for_fragment(self, record.view()?, &recipe.gdns)?;
        let gdn_bytes = gdn::metadata_bytes(&gdn_parameters)?;
        let (plan, family) = record.into_parts();
        let bytes = family
            .binding_metadata_bytes()?
            .checked_add(plan.source_binding_metadata_bytes()?)
            .and_then(|n| n.checked_add(std::mem::size_of::<PreparedPlan>()))
            .and_then(|n| n.checked_add(site_bytes))
            .and_then(|n| n.checked_add(gdn_bytes))
            .ok_or_else(|| Error::Limit("Qwen fragment metadata extent".into()))?;
        if bytes > metadata.bytes() {
            return Err(Error::Limit("Qwen fragment metadata bound".into()));
        }
        drop(family);
        drop(recipe);
        Ok(PreparedPlan {
            source: self.source().clone(),
            plan: retain_preparation_metadata(plan, metadata)?,
            output,
            residual_output,
            normalized: None,
            boundaries: Box::new([]),
            selected_row: None,
            completed_layers,
            first_mlp_down_node,
            gdn_parameters,
        })
    }
}
fn prepare_whole(
    source: VerifiedTensorSource,
    record: QwenPlan,
    metadata: WindowWorkspace,
    gdn_parameters: Box<[PreparedGdnParameters]>,
) -> Result<PreparedPlan> {
    let gdn_bytes = gdn::metadata_bytes(&gdn_parameters)?;
    let bytes = record
        .binding_metadata_bytes()?
        .checked_add(std::mem::size_of::<PreparedPlan>() + 24 * std::mem::size_of::<[ValueId; 2]>())
        .and_then(|n| n.checked_add(gdn_bytes))
        .ok_or_else(|| Error::Limit("Qwen whole preparation metadata extent".into()))?;
    if source.source_identity() != record.plan().source() || bytes > metadata.bytes() {
        return Err(Error::State(
            "Qwen whole preparation source/metadata differs".into(),
        ));
    }
    let output = record.logits();
    let normalized = record.normalized();
    require_selected_value(record.plan(), output)?;
    require_selected_value(record.plan(), normalized)?;
    for pair in record.boundaries() {
        if pair[0] == pair[1] {
            return Err(Error::State("Qwen distinct boundary outputs".into()));
        }
        for &id in pair {
            require_selected_value(record.plan(), id)?;
        }
    }
    let selected_row = record.selected_row();
    let boundaries = record.boundaries().to_vec().into_boxed_slice();
    let plan = retain_preparation_metadata(record.into_plan(), metadata)?;
    Ok(PreparedPlan {
        source,
        plan,
        output,
        residual_output: None,
        normalized: Some(normalized),
        boundaries,
        selected_row: Some(selected_row),
        completed_layers: 24,
        first_mlp_down_node: None,
        gdn_parameters,
    })
}
fn retain_preparation_metadata(
    plan: ForwardPlan,
    metadata: WindowWorkspace,
) -> Result<ForwardPlan> {
    let metadata = metadata.into_host_workspace()?;
    if plan.prepared_metadata_bytes().is_some() {
        plan.retain_binding_metadata(metadata)
    } else {
        plan.retain_prepared_metadata(metadata)
    }
}
fn require_analysis_values(record: &QwenPlan, analysis: &WholeAnalysis) -> Result<()> {
    let plan = record.plan();
    for site in analysis.gdn.values() {
        for &index in &site.nodes {
            source_state_descriptor(&plan.nodes()[index])?;
        }
        for &value in &site.operands {
            require_selected_value(plan, value)?;
        }
    }
    for site in analysis.attention.values() {
        for &index in &site.nodes {
            source_state_descriptor(&plan.nodes()[index])?;
        }
        for &value in &site.operands {
            require_selected_value(plan, value)?;
        }
    }
    let head = analysis.head;
    let input_bytes = plan.values()[record.normalized().0 as usize]
        .capacity_elements()?
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("Qwen head input extent".into()))?;
    let output_bytes = plan.values()[record.logits().0 as usize]
        .capacity_elements()?
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("Qwen head output extent".into()))?;
    if input_bytes != head.full_bytes
        || output_bytes != head.output_bytes
        || head.offset.checked_add(head.selected_bytes) != Some(input_bytes)
    {
        return Err(Error::State(
            "Qwen selected head view differs from actual values".into(),
        ));
    }
    Ok(())
}
fn require_selected_value(plan: &ForwardPlan, id: ValueId) -> Result<()> {
    let tag = plan
        .source_value_provenance(id)?
        .ok_or_else(|| Error::State("Qwen selected output has no source producer".into()))?;
    let node = plan
        .nodes()
        .get(tag.node_index())
        .ok_or_else(|| Error::State("Qwen selected producer outside plan".into()))?;
    if Some(tag.call()) != plan.source_call()
        || node.outputs.get(tag.output_index()) != Some(&id)
        || tag.inputs() != node.inputs
    {
        return Err(Error::State(
            "Qwen selected output/producer/input issuer differs".into(),
        ));
    }
    Ok(())
}
fn require_embedding(source: &OriginalWindowModelSource, plan: &ForwardPlan) -> Result<()> {
    let node = plan
        .nodes()
        .first()
        .ok_or_else(|| Error::State("Qwen embedding source node absent".into()))?;
    let PlanOperation::SourceOperator {
        descriptor,
        state: None,
    } = &node.operation
    else {
        return Err(Error::State(
            "Qwen original embedding descriptor absent".into(),
        ));
    };
    if descriptor.kind() != SourceOperatorKind::EmbeddingRows
        || descriptor.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
        || descriptor.matrices().len() != 1
        || !node.inputs.is_empty()
        || node.outputs.len() != 1
        || Some(descriptor.call()) != plan.source_call()
        || descriptor.matrices()[0] != source.tied_output_descriptor()?
    {
        return Err(Error::State(
            "Qwen exact tied embedding source/issuer".into(),
        ));
    }
    source.validate_matrix(QwenMatrixRole::Embedding, &descriptor.matrices()[0])
}
fn require_state_layout(source: &OriginalWindowModelSource, plan: &ForwardPlan) -> Result<()> {
    if source.geometry().layers.len() != 24
        || source.state_specs().len() != 24
        || plan.prefix_capacity() != 512
    {
        return Err(Error::Unsupported("Qwen all42 source state layout".into()));
    }
    let mut slots = Vec::with_capacity(42);
    for (layer, spec) in source.state_specs().iter().enumerate() {
        match *spec {
            crate::StateSpec::Delta { conv, state }
                if layer % 4 != 3 && conv == 6144 * 3 && state == 16 * 128 * 128 =>
            {
                slots.push(StateSlot::Conv(ConvDomain(layer)));
                slots.push(StateSlot::Recurrent(RecurrentDomain(layer)));
            }
            crate::StateSpec::Full { kv_width: 512 } if layer % 4 == 3 => {
                slots.push(StateSlot::Kv(KvDomain(layer)));
            }
            _ => {
                return Err(Error::Unsupported(
                    "Qwen source canonical state geometry/layer".into(),
                ));
            }
        }
    }
    slots.sort();
    if slots != plan.required_domains() {
        return Err(Error::State(
            "Qwen source all42 state slots differ from plan".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct GdnSite {
    pub(crate) nodes: [usize; 2],
    /// QKVZ, BA, actual Conv output, actual Z copy, recurrent gated output.
    pub(crate) operands: [ValueId; 5],
}
#[derive(Clone, Debug)]
pub(crate) struct AttentionSite {
    /// QKV, QK, KV append, FA, sigmoid, multiply, Out.
    pub(crate) nodes: [usize; 7],
    /// Normalized input, QKV, Q, K, gate, FA, sigmoid, multiplied, Out.
    pub(crate) operands: [ValueId; 9],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadView {
    pub(crate) full_bytes: usize,
    pub(crate) offset: usize,
    pub(crate) selected_bytes: usize,
    pub(crate) output_bytes: usize,
}
impl HeadView {
    fn for_call(call: &PromptCall) -> Result<Self> {
        call.validate()?;
        let row = call
            .model_profile()
            .hidden_size()
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("Qwen text program head row bytes".into()))?;
        let rows = call.span().len();
        let full = rows
            .checked_mul(row)
            .ok_or_else(|| Error::Limit("Qwen text program full head input".into()))?;
        let offset = (rows - 1)
            .checked_mul(row)
            .ok_or_else(|| Error::Limit("Qwen text program last row".into()))?;
        if offset.checked_add(row) != Some(full) {
            return Err(Error::State(
                "Qwen text program exact final selected row".into(),
            ));
        }
        Ok(Self {
            full_bytes: full,
            offset,
            selected_bytes: row,
            output_bytes: 248320 * 2,
        })
    }
}

struct WholeAnalysis {
    rms: Vec<usize>,
    projections: Vec<usize>,
    gdn: BTreeMap<usize, GdnSite>,
    attention: BTreeMap<usize, AttentionSite>,
    head: HeadView,
}

fn require_state_read(node: &PlanNode, descriptor: &QwenOperator) -> Result<()> {
    let PlanOperation::SourceOperator {
        descriptor: actual,
        state,
    } = &node.operation
    else {
        return Err(Error::State(
            "Qwen text program typed source state-read operation absent".into(),
        ));
    };
    descriptor.require_mathematical_operator(actual)?;
    let valid = match (descriptor.kind(), state) {
        (QwenOperation::FlashAttention, Some(version)) => {
            node.state.is_none()
                && descriptor.layer().is_some_and(|layer| {
                    layer.is_attention()
                        && version.slot == StateSlot::Kv(KvDomain(layer.index()))
                        && version.ordinal == 1
                })
        }
        (QwenOperation::FlashAttention, None) => false,
        (_, None) => true,
        (_, Some(_)) => false,
    };
    if !valid {
        return Err(Error::State(
            "Qwen text program FA reads exact appended KV ordinal1, never a write effect".into(),
        ));
    }
    Ok(())
}
fn typed(record: &QwenPlan, index: usize) -> Result<QwenOperator> {
    record
        .operator(index)?
        .ok_or_else(|| Error::State("Qwen text program actual typed source node absent".into()))
}
fn require_text_record(call: &PromptCall, record: &QwenPlan) -> Result<()> {
    call.validate()?;
    let plan = record.plan();
    if record.call() != call.as_source_call()
        || plan.source() != call.span().source()
        || plan.nodes().len() != 255
        || plan.state_nodes().len() != 255
        || plan.values().len() != 327
        || plan.required_domains().len() != 42
        || plan.frontier_delta() != 0
        || plan.logits().is_some()
        || record.boundaries().len() != 24
        || record.normalized() != ValueId(324)
        || record.logits() != ValueId(326)
        || record.selected_row() != call.span().len() - 1
    {
        return Err(Error::State(
            "Qwen text program SAME source-issued255/327 unpublished whole record".into(),
        ));
    }
    for (index, node) in plan.nodes().iter().enumerate() {
        if let Some(descriptor) = record.operator(index)?
            && descriptor.call() != record.call()
        {
            return Err(Error::State(
                "Qwen text program foreign operator call issuer".into(),
            ));
        }
        for id in node.inputs.iter().chain(&node.outputs) {
            record
                .value_provenance(*id)?
                .ok_or_else(|| {
                    Error::State("Qwen text program absent source value provenance".into())
                })?
                .require_invocation(call.request())?;
        }
    }
    Ok(())
}
fn analyze_text_record(
    call: &PromptCall,
    record: &QwenPlan,
    mut validate_member: impl FnMut(&QwenOperator) -> Result<()>,
) -> Result<WholeAnalysis> {
    require_text_record(call, record)?;
    for (index, node) in record.plan().nodes().iter().enumerate() {
        if let Some(descriptor) = &record.operator(index)? {
            descriptor.validate()?;
            require_state_read(node, descriptor)?;
        }
    }
    let mut rms = Vec::with_capacity(49);
    let mut projections = Vec::with_capacity(114);
    let mut gdn = BTreeMap::new();
    let mut attention = BTreeMap::new();
    let mut matrices = BTreeSet::new();
    let mut parameters = BTreeSet::new();
    for (index, node) in record.plan().nodes().iter().enumerate() {
        let Some(descriptor) = &record.operator(index)? else {
            continue;
        };
        validate_member(descriptor)?;
        matrices.extend(descriptor.matrices().iter().map(|m| m.name.clone()));
        parameters.extend(descriptor.parameters().iter().map(|p| p.name.clone()));
        match descriptor.kind() {
            QwenOperation::InputRms | QwenOperation::ResidualRms(_) => rms.push(index),
            QwenOperation::Projection(p) if p != QwenProjection::Head => projections.push(index),
            QwenOperation::GdnConv => {
                let layer = descriptor
                    .layer()
                    .ok_or_else(|| Error::State("Qwen text program GDN layer absent".into()))?
                    .index();
                let next = &record.plan().nodes()[index + 1];
                let rd = typed(record, index + 1)?;
                let ([qkvz, ba], [activated, z], [ra, rz, rb], [gated]) = (
                    node.inputs.as_slice(),
                    node.outputs.as_slice(),
                    next.inputs.as_slice(),
                    next.outputs.as_slice(),
                ) else {
                    return Err(Error::State(
                        "Qwen text program actual Conv/recurrent arities".into(),
                    ));
                };
                if rd.kind() != QwenOperation::GdnRecurrent
                    || rd.layer() != descriptor.layer()
                    || activated != ra
                    || z != rz
                    || ba != rb
                    || node.state.is_none()
                    || next.state.is_none()
                    || gdn
                        .insert(
                            layer,
                            GdnSite {
                                nodes: [index, index + 1],
                                operands: [*qkvz, *ba, *activated, *z, *gated],
                            },
                        )
                        .is_some()
                {
                    return Err(Error::State(
                        "Qwen text program SAME source Conv/Z/BA/recurrent producer pair".into(),
                    ));
                }
            }
            _ => {}
        }
    }
    // Projection membership above includes all attention QKV sites. Derive
    // their true SSA from each source descriptor, not a layer0 retag.
    for &index in &projections {
        let descriptor = typed(record, index)?;
        if descriptor.kind() != QwenOperation::Projection(QwenProjection::AttentionQkv) {
            continue;
        }
        let layer = descriptor.layer().unwrap().index();
        let ns: [usize; 7] = std::array::from_fn(|i| index + i);
        let nodes = &record.plan().nodes();
        let expected = [
            QwenOperation::Projection(QwenProjection::AttentionQkv),
            QwenOperation::AttentionQk,
            QwenOperation::KvAppend,
            QwenOperation::FlashAttention,
            QwenOperation::Sigmoid,
            QwenOperation::Multiply,
            QwenOperation::Projection(QwenProjection::AttentionOut),
        ];
        for (&n, kind) in ns.iter().zip(expected) {
            let op = typed(record, n)?;
            if op.kind() != kind || op.layer() != descriptor.layer() {
                return Err(Error::State(
                    "Qwen text program actual attention operator order/layer".into(),
                ));
            }
        }
        let (
            [x],
            [qkv],
            [pqkv],
            [q, k, gate],
            [pk, pv],
            [],
            [fa_q],
            [o],
            [sig_in],
            [sig],
            [mul_o, mul_s],
            [gated],
            [out_in],
            [y],
        ) = (
            nodes[ns[0]].inputs.as_slice(),
            nodes[ns[0]].outputs.as_slice(),
            nodes[ns[1]].inputs.as_slice(),
            nodes[ns[1]].outputs.as_slice(),
            nodes[ns[2]].inputs.as_slice(),
            nodes[ns[2]].outputs.as_slice(),
            nodes[ns[3]].inputs.as_slice(),
            nodes[ns[3]].outputs.as_slice(),
            nodes[ns[4]].inputs.as_slice(),
            nodes[ns[4]].outputs.as_slice(),
            nodes[ns[5]].inputs.as_slice(),
            nodes[ns[5]].outputs.as_slice(),
            nodes[ns[6]].inputs.as_slice(),
            nodes[ns[6]].outputs.as_slice(),
        )
        else {
            return Err(Error::State(
                "Qwen text program actual QKV/QK/KV/FA/gates/Out arity".into(),
            ));
        };
        if pqkv != qkv
            || pk != k
            || pv != qkv
            || fa_q != q
            || sig_in != gate
            || mul_o != o
            || mul_s != sig
            || out_in != gated
            || nodes[ns[2]].state.is_none()
            || nodes[ns[3]].state.is_some()
            || attention
                .insert(
                    layer,
                    AttentionSite {
                        nodes: ns,
                        operands: [*x, *qkv, *q, *k, *gate, *o, *sig, *gated, *y],
                    },
                )
                .is_some()
        {
            return Err(Error::State(
                "Qwen text program exact packed-V/current-KV/FA/gate producer chain".into(),
            ));
        }
    }
    if rms.len() != 49
        || rms.first() != Some(&1)
        || rms.last() != Some(&253)
        || projections.len() != 114
        || matrices.len() != 187
        || parameters.len() != 133
        || !matrices.is_disjoint(&parameters)
        || gdn.keys().copied().collect::<Vec<_>>()
            != (0..24).filter(|i| i % 4 != 3).collect::<Vec<_>>()
        || attention.keys().copied().collect::<Vec<_>>() != (3..24).step_by(4).collect::<Vec<_>>()
    {
        return Err(Error::State(
            "Qwen text program complete320 roles/49 RMS/18 GDN/6attention/114 groups".into(),
        ));
    }
    let head = HeadView::for_call(call)?;
    let final_node = &record.plan().nodes()[253];
    let head_node = &record.plan().nodes()[254];
    if typed(record, 253)?.kind() != QwenOperation::ResidualRms(QwenRmsSite::Final)
        || typed(record, 254)?.kind() != QwenOperation::Projection(QwenProjection::Head)
        || final_node.outputs != [ValueId(324), ValueId(325)]
        || head_node.inputs != [ValueId(324)]
        || head_node.outputs != [ValueId(326)]
        || typed(record, 254)?.selected_row() != Some(record.selected_row())
    {
        return Err(Error::State(
            "Qwen text program original Final253/Head254 producer/selected-row view".into(),
        ));
    }
    Ok(WholeAnalysis {
        rms,
        projections,
        gdn,
        attention,
        head,
    })
}

fn require_reference_record(source: &OriginalWindowModelSource, record: &QwenPlan) -> Result<()> {
    positions::require_reference_positions(source)?;
    source.require_small_model_source()?;
    record.call().validate()?;
    if record.call().model_profile() != QwenSourceProfile::Qwen08 {
        return Err(Error::Unsupported(
            "native full24 source geometry remains .8B only".into(),
        ));
    }
    // Private recorded type is minted only by the checked family recipe.
    let plan = record.plan();
    if plan.source() != source.source_identity()
        || plan.nodes().len() != 255
        || plan.state_nodes().len() != 255
        || plan.values().len() != 327
        || plan.required_domains().len() != 42
        || plan.frontier_delta() != 0
        || plan.logits().is_some()
        || record.boundaries().len() != 24
    {
        return Err(Error::Unsupported(
            "closed actual original full24 unpublished source recipe".into(),
        ));
    }
    for (index, node) in plan.nodes().iter().enumerate() {
        record.operator(index)?;
        source_state_descriptor(node)?;
        for id in node.inputs.iter().chain(&node.outputs) {
            record
                .value_provenance(*id)?
                .ok_or_else(|| Error::State("full24 actual recipe provenance absent".into()))?
                .require_invocation(record.call().request())?;
        }
    }
    Ok(())
}

struct Recipe {
    gdns: Vec<[usize; 2]>,
    fused: Vec<usize>,
    first_down: Option<usize>,
    _metadata: WindowWorkspace,
}
fn only_output(node: &PlanNode) -> Result<zerogravity_core::accelerator::program::ValueId> {
    match node.outputs.as_slice() {
        [value] => Ok(*value),
        _ => Err(Error::State("one actual typed output".into())),
    }
}
fn fragment_rms() -> SourceRmsGeometry {
    SourceRmsGeometry {
        width: 1024,
        epsilon: 1e-6,
        gamma: SourceGamma::OnePlus,
    }
}
fn source_descriptor<'a>(node: &'a PlanNode, call: &QwenGdnCall) -> Result<&'a SourceOperator> {
    match &node.operation {
        PlanOperation::SourceOperator {
            descriptor,
            state: None,
        } if descriptor.call() == &call.source_call()?
            && descriptor.numerical_contract()
                == SourceNumericalContract::ReferenceCudaSm86Bf16V1 =>
        {
            Ok(descriptor)
        }
        _ => Err(Error::State(
            "native fragment actual mathematical call/contract differs".into(),
        )),
    }
}
fn require_fragment_recipe(
    source: &OriginalWindowModelSource,
    record: QwenFragmentView<'_>,
) -> Result<Recipe> {
    use crate::plan::{QwenPrefixLayer, QwenPrefixRmsSite};
    use crate::source::binding::{QwenMatrixRole as M, QwenParameterRole as P};
    let metadata = source.source().reserve_workspace(64 * 1024)?;
    let plan = record.plan();
    let base = record.invocation();
    if plan.source_call() != Some(&base.source_call()?) {
        return Err(Error::State(
            "native fragment common call issuer differs".into(),
        ));
    }
    if source.source_identity() != plan.source()
        || base.layer().index() != 0
        || plan.frontier_delta() != 0
        || plan.logits().is_some()
        || plan.is_operator_slice()
        || !matches!(record.completed_layers(), 0 | 1 | 3)
    {
        return Err(Error::Unsupported(
            "unpublishable exact initial source GDN cut".into(),
        ));
    }
    let full = record.completed_layers() != 0;
    let layers = record.completed_layers().max(1);
    let node = |index: usize| {
        plan.nodes()
            .get(index)
            .ok_or_else(|| Error::State("source recipe node absent".into()))
    };
    let embedding = node(0)?;
    let raw = only_output(embedding)?;
    let descriptor = source_descriptor(embedding, base)?;
    if descriptor.kind() != SourceOperatorKind::EmbeddingRows
        || descriptor.matrices().len() != 1
        || !embedding.inputs.is_empty()
    {
        return Err(Error::State("original source teacher issuer".into()));
    }
    source.validate_matrix(M::Embedding, &descriptor.matrices()[0])?;
    let norm = node(1)?;
    let first_normal = only_output(norm)?;
    let descriptor = source_descriptor(norm, base)?;
    if descriptor.kind() != SourceOperatorKind::Rms(fragment_rms())
        || norm.inputs != [raw]
        || descriptor.parameters().len() != 1
    {
        return Err(Error::State("ordinary RMS consumes actual raw rows".into()));
    }
    source.validate_parameter(P::InputNorm(0), &descriptor.parameters()[0])?;
    let mut recipe = Recipe {
        first_down: None,
        gdns: Vec::with_capacity(layers),
        fused: Vec::with_capacity(2 * layers - 1),
        _metadata: metadata,
    };
    let mut cursor = 2;
    let mut branch = raw;
    let mut residual = raw;
    let projection = |index: usize,
                      call: &QwenGdnCall,
                      kind: GdnProjection,
                      input: zerogravity_core::accelerator::program::ValueId,
                      roles: &[M]|
     -> Result<zerogravity_core::accelerator::program::ValueId> {
        let n = node(index)?;
        let descriptor = source_descriptor(n, call)?;
        if descriptor.kind() != (SourceOperatorKind::Projection { selected_row: None })
            || descriptor.input_elements()? != [call.rows() * kind.input_width()]
            || descriptor.output_elements()? != [call.rows() * kind.output_width()]
            || n.inputs != [input]
            || descriptor.matrices().len() != roles.len()
        {
            return Err(Error::State(
                "source ordered projection/producer recipe".into(),
            ));
        }
        for (member, role) in descriptor.matrices().iter().zip(roles) {
            source.validate_matrix(*role, member)?;
        }
        only_output(n)
    };
    let fused = |index: usize,
                 layer: QwenPrefixLayer,
                 site: QwenPrefixRmsSite,
                 left: zerogravity_core::accelerator::program::ValueId,
                 right: zerogravity_core::accelerator::program::ValueId,
                 role: P|
     -> Result<(
        zerogravity_core::accelerator::program::ValueId,
        zerogravity_core::accelerator::program::ValueId,
    )> {
        let n = node(index)?;
        let descriptor = source_descriptor(n, base)?;
        layer.rms(site)?;
        if descriptor.kind() != SourceOperatorKind::ResidualRms(fragment_rms())
            || n.inputs != [left, right]
            || n.outputs.len() != 2
            || descriptor.parameters().len() != 1
        {
            return Err(Error::State(
                "actual branch/carried residual/site recipe".into(),
            ));
        }
        source.validate_parameter(role, &descriptor.parameters()[0])?;
        Ok((n.outputs[0], n.outputs[1]))
    };
    for i in 0..layers {
        let layer = QwenPrefixLayer::new(plan.source(), i)?;
        let call = QwenGdnCall::new(layer, base.reference_phase()?, base.teachers().clone())?;
        let (normal, carried) = if i == 0 {
            (first_normal, raw)
        } else {
            let pair = fused(
                cursor,
                layer,
                QwenPrefixRmsSite::InputResidual,
                branch,
                residual,
                P::InputNorm(i),
            )?;
            recipe.fused.push(cursor);
            cursor += 1;
            pair
        };
        let q = projection(
            cursor,
            &call,
            GdnProjection::Qkvz,
            normal,
            &[M::GdnQkv(i), M::GdnGate(i)],
        )?;
        cursor += 1;
        let b = projection(
            cursor,
            &call,
            GdnProjection::Ba,
            normal,
            &[M::GdnBeta(i), M::GdnAlpha(i)],
        )?;
        cursor += 1;
        let conv_index = cursor;
        let conv = node(cursor)?;
        cursor += 1;
        let rec_index = cursor;
        let rec = node(cursor)?;
        cursor += 1;
        let c = source_descriptor(conv, &call)?;
        let r = source_descriptor(rec, &call)?;
        if conv.inputs != [q, b]
            || conv.outputs.len() != 2
            || rec.inputs != [conv.outputs[0], conv.outputs[1], b]
            || c.parameters().len() != 1
            || r.parameters().len() != 3
        {
            return Err(Error::State("exact source/layer/cohort GDN recipe".into()));
        }
        QwenGdnConv::new(call.clone(), c.parameters()[0].clone())?
            .require_mathematical_operator(c)?;
        let parameters = r
            .parameters()
            .to_vec()
            .try_into()
            .map_err(|_| Error::State("native exact recurrent parameter arity".into()))?;
        QwenGdnRecurrent::new(call.clone(), parameters)?.require_mathematical_operator(r)?;
        source.validate_parameter(P::GdnConvolution(i), &c.parameters()[0])?;
        for (parameter, role) in r
            .parameters()
            .iter()
            .zip([P::GdnA(i), P::GdnDt(i), P::GdnNorm(i)])
        {
            source.validate_parameter(role, parameter)?;
        }
        let gated = only_output(rec)?;
        recipe.gdns.push([conv_index, rec_index]);
        if full {
            let out = projection(
                cursor,
                &call,
                GdnProjection::GdnOut,
                gated,
                &[M::GdnOutput(i)],
            )?;
            cursor += 1;
            let (post, post_residual) = fused(
                cursor,
                layer,
                QwenPrefixRmsSite::PostAttention,
                out,
                carried,
                P::FfnNorm(i),
            )?;
            recipe.fused.push(cursor);
            cursor += 1;
            let gate_up = projection(
                cursor,
                &call,
                GdnProjection::GateUp,
                post,
                &[M::FfnGate(i), M::FfnUp(i)],
            )?;
            cursor += 1;
            let act = node(cursor)?;
            cursor += 1;
            let descriptor = source_descriptor(act, &call)?;
            if descriptor.kind() != (SourceOperatorKind::SiluAndMul { width: 3584 })
                || act.inputs != [gate_up]
            {
                return Err(Error::State(
                    "actual computed GateUp activation recipe".into(),
                ));
            }
            if i == 0 {
                recipe.first_down = Some(cursor);
            }
            branch = projection(
                cursor,
                &call,
                GdnProjection::Down,
                only_output(act)?,
                &[M::FfnDown(i)],
            )?;
            cursor += 1;
            residual = post_residual;
        } else {
            branch = gated;
        }
    }
    let rest = &plan.nodes()[cursor..];
    if record.output() != branch
        || record.residual_output() != full.then_some(residual)
        || rest.len() != 42 - 2 * layers
        || rest.iter().any(|n| {
            !matches!(&n.operation, PlanOperation::SourceOperator { descriptor, state: None } if matches!(descriptor.kind(), SourceOperatorKind::StateCarry { slot } if n.state.is_some_and(|e| e.read.slot==slot && e.write.slot==slot)))
                || !n.inputs.is_empty()
                || !n.outputs.is_empty()
        })
    {
        return Err(Error::State(
            "exact source initial cut output/canonical remaining-domain copies".into(),
        ));
    }
    Ok(recipe)
}

fn require_fragment_config(source: &OriginalWindowModelSource) -> Result<()> {
    use crate::config::{Family, LayerKind};
    use crate::plan::{QwenPrefixLayer, QwenPrefixRmsSite};
    use crate::source::safetensors::GdnOutputGate;
    let layer = 0;
    let site = QwenPrefixRmsSite::PostAttention;
    let config = source.owner().config();
    let g = config.geometry();
    if g.family != Family::QwenHybrid
        || g.hidden_size != 1024
        || g.vocab_size != 248320
        || g.layers.len() != 24
        || g.head_dim != 256
        || g.num_attention_heads != 8
        || g.num_key_value_heads != 2
        || g.recurrent_key_heads != Some(16)
        || g.recurrent_value_heads != Some(16)
        || config.linear_key_head_dim() != 128
        || config.linear_value_head_dim() != 128
        || config.linear_conv_kernel_dim() != 4
        || config.intermediate_size() != 3584
        || config.gdn_output_gate() != GdnOutputGate::Silu
        || config.rms_norm_eps() != 1e-6
        || !config.attention_output_gate()
        || g.layers.iter().enumerate().any(|(i, kind)| {
            *kind
                != if i % 4 == 3 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::LinearAttention
                }
        })
        || layer > 2
        || g.layers.get(layer) != Some(&LayerKind::LinearAttention)
    {
        return Err(Error::Unsupported(
            "original .8B24/GDN16x128 prefix config".into(),
        ));
    }
    let key = QwenPrefixLayer::new(source.source_identity(), layer)?.rms(site)?;
    source.owner().text_tensor(&key.gamma_name())?;
    let actual = source.source().parameter(&key.gamma_name())?;
    key.validate(&actual.source_parameter(), 1024)?;
    Ok(())
}

fn source_state_descriptor(node: &PlanNode) -> Result<&SourceOperator> {
    let PlanOperation::SourceOperator { descriptor, state } = &node.operation else {
        return Err(Error::State(
            "full24 actual typed node descriptor absent".into(),
        ));
    };
    let valid = match (descriptor.kind(), state) {
        (SourceOperatorKind::FlashAttention { domain, .. }, Some(version)) => {
            node.state.is_none() && version.slot == StateSlot::Kv(domain) && version.ordinal == 1
        }
        (SourceOperatorKind::FlashAttention { .. }, None) => false,
        (_, None) => true,
        (_, Some(_)) => false,
    };
    if !valid {
        return Err(Error::State(
            "full24 state read must be exact private appended FA KV; non-FA has no state read"
                .into(),
        ));
    }
    Ok(descriptor)
}

#[cfg(test)]
mod tests;
