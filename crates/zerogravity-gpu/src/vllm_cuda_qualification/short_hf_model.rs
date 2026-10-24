//! Exact source-issued whole-call binding. The original loop/root owns execution.
use super::{
    canonical47_model::validate_source_members, model_values::ModelValues,
    model_weights::ModelWeights,
};
use crate::resident_leases::{ModelWeight, ResidentLease};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        KvDomain, PlanNode, PlanOperation, RecordedVllm08Model, ShortHfCall, StateSlot, ValueId,
        Vllm08Kind, Vllm08Operator, Vllm08Projection, Vllm08RmsSite,
    },
};
use zerogravity_model_qwen35::original_binding::{BoundFullModelSource, BoundShortHfCall};

#[derive(Clone, Debug)]
pub(crate) struct ShortHfGdnSite {
    pub(crate) nodes: [usize; 2],
    /// QKVZ, BA, actual Conv output, actual Z copy, recurrent gated output.
    pub(crate) operands: [ValueId; 5],
}
#[derive(Clone, Debug)]
pub(crate) struct ShortHfAttentionSite {
    /// QKV, QK, KV append, FA, sigmoid, multiply, Out.
    pub(crate) nodes: [usize; 7],
    /// Normalized input, QKV, Q, K, gate, FA, sigmoid, multiplied, Out.
    pub(crate) operands: [ValueId; 9],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShortHfHeadView {
    pub(crate) full_bytes: usize,
    pub(crate) offset: usize,
    pub(crate) selected_bytes: usize,
    pub(crate) output_bytes: usize,
}
impl ShortHfHeadView {
    fn for_call(call: &ShortHfCall) -> Result<Self> {
        call.validate()?;
        let row = call
            .model_profile()
            .hidden_size()
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("short-HF head row bytes".into()))?;
        let rows = call.span().len();
        let full = rows
            .checked_mul(row)
            .ok_or_else(|| Error::Limit("short-HF full head input".into()))?;
        let offset = (rows - 1)
            .checked_mul(row)
            .ok_or_else(|| Error::Limit("short-HF last row".into()))?;
        if offset.checked_add(row) != Some(full) {
            return Err(Error::State("short-HF exact final selected row".into()));
        }
        Ok(Self {
            full_bytes: full,
            offset,
            selected_bytes: row,
            output_bytes: 248320 * 2,
        })
    }
}

pub(crate) struct ShortHfModelBinding {
    source: BoundShortHfCall,
    record: Arc<RecordedVllm08Model>,
    prepared: Arc<PreparedShortHfModel>,
    _metadata: WindowWorkspace,
}
struct ShortHfAnalysis {
    rms: Vec<usize>,
    projections: Vec<usize>,
    gdn: BTreeMap<usize, ShortHfGdnSite>,
    attention: BTreeMap<usize, ShortHfAttentionSite>,
    head: ShortHfHeadView,
}
struct PreparedShortHfModel {
    record: Arc<RecordedVllm08Model>,
    analysis: ShortHfAnalysis,
    _metadata: zerogravity_core::resource_arbiter::HostWorkspace,
}
/// Concrete source preparation owner; contains no session, roots or leases.
pub(crate) struct ShortHfModelBindings {
    source: BoundFullModelSource,
    prepared: std::sync::Mutex<Vec<Arc<PreparedShortHfModel>>>,
    _metadata: WindowWorkspace,
}

fn require_state_read(node: &PlanNode, descriptor: &Vllm08Operator) -> Result<()> {
    let PlanOperation::TypedVllm08 { state, .. } = &node.operation else {
        return Err(Error::State(
            "short-HF typed source state-read operation absent".into(),
        ));
    };
    let valid = match (descriptor.kind(), state) {
        (Vllm08Kind::FlashAttention, Some(version)) => {
            node.state.is_none()
                && descriptor.layer().is_some_and(|layer| {
                    layer.is_attention()
                        && version.slot == StateSlot::Kv(KvDomain(layer.index()))
                        && version.ordinal == 1
                })
        }
        (Vllm08Kind::FlashAttention, None) => false,
        (_, None) => true,
        (_, Some(_)) => false,
    };
    if !valid {
        return Err(Error::State(
            "short-HF FA reads exact appended KV ordinal1, never a write effect".into(),
        ));
    }
    Ok(())
}
fn typed(record: &RecordedVllm08Model, index: usize) -> Result<&Vllm08Operator> {
    match record.plan().nodes().get(index).map(|n| &n.operation) {
        Some(PlanOperation::TypedVllm08 { descriptor, .. }) => Ok(descriptor),
        _ => Err(Error::State(
            "short-HF actual typed source node absent".into(),
        )),
    }
}
fn require_invocation_record(call: &ShortHfCall, record: &RecordedVllm08Model) -> Result<()> {
    call.validate()?;
    let plan = record.plan();
    if record.call() != call.as_vllm_call()
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
            "short-HF SAME source-issued255/327 unpublished whole record".into(),
        ));
    }
    for node in plan.nodes() {
        if let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation
            && descriptor.call() != record.call()
        {
            return Err(Error::State("short-HF foreign operator call issuer".into()));
        }
        for id in node.inputs.iter().chain(&node.outputs) {
            plan.vllm_value_provenance(*id)?
                .ok_or_else(|| Error::State("short-HF absent source value provenance".into()))?
                .require_invocation(call.request())?;
        }
    }
    Ok(())
}
fn analyze_record(
    call: &ShortHfCall,
    record: &RecordedVllm08Model,
    mut validate_member: impl FnMut(&Vllm08Operator) -> Result<()>,
) -> Result<ShortHfAnalysis> {
    require_invocation_record(call, record)?;
    for node in record.plan().nodes() {
        if let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation {
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
        let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
            continue;
        };
        validate_member(descriptor)?;
        matrices.extend(descriptor.matrices().iter().map(|m| m.name.clone()));
        parameters.extend(descriptor.parameters().iter().map(|p| p.name.clone()));
        match descriptor.kind() {
            Vllm08Kind::InputRms | Vllm08Kind::ResidualRms(_) => rms.push(index),
            Vllm08Kind::Projection(p) if p != Vllm08Projection::Head => projections.push(index),
            Vllm08Kind::GdnConv => {
                let layer = descriptor
                    .layer()
                    .ok_or_else(|| Error::State("short-HF GDN layer absent".into()))?
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
                        "short-HF actual Conv/recurrent arities".into(),
                    ));
                };
                if rd.kind() != Vllm08Kind::GdnRecurrent
                    || rd.layer() != descriptor.layer()
                    || activated != ra
                    || z != rz
                    || ba != rb
                    || node.state.is_none()
                    || next.state.is_none()
                    || gdn
                        .insert(
                            layer,
                            ShortHfGdnSite {
                                nodes: [index, index + 1],
                                operands: [*qkvz, *ba, *activated, *z, *gated],
                            },
                        )
                        .is_some()
                {
                    return Err(Error::State(
                        "short-HF SAME source Conv/Z/BA/recurrent producer pair".into(),
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
        if descriptor.kind() != Vllm08Kind::Projection(Vllm08Projection::AttentionQkv) {
            continue;
        }
        let layer = descriptor.layer().unwrap().index();
        let ns: [usize; 7] = std::array::from_fn(|i| index + i);
        let nodes = &record.plan().nodes();
        let expected = [
            Vllm08Kind::Projection(Vllm08Projection::AttentionQkv),
            Vllm08Kind::AttentionQk,
            Vllm08Kind::KvAppend,
            Vllm08Kind::FlashAttention,
            Vllm08Kind::Sigmoid,
            Vllm08Kind::Multiply,
            Vllm08Kind::Projection(Vllm08Projection::AttentionOut),
        ];
        for (&n, kind) in ns.iter().zip(expected) {
            let op = typed(record, n)?;
            if op.kind() != kind || op.layer() != descriptor.layer() {
                return Err(Error::State(
                    "short-HF actual attention operator order/layer".into(),
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
                "short-HF actual QKV/QK/KV/FA/gates/Out arity".into(),
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
                    ShortHfAttentionSite {
                        nodes: ns,
                        operands: [*x, *qkv, *q, *k, *gate, *o, *sig, *gated, *y],
                    },
                )
                .is_some()
        {
            return Err(Error::State(
                "short-HF exact packed-V/current-KV/FA/gate producer chain".into(),
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
            "short-HF complete320 roles/49 RMS/18 GDN/6attention/114 groups".into(),
        ));
    }
    let head = ShortHfHeadView::for_call(call)?;
    let final_node = &record.plan().nodes()[253];
    let head_node = &record.plan().nodes()[254];
    if typed(record, 253)?.kind() != Vllm08Kind::ResidualRms(Vllm08RmsSite::Final)
        || typed(record, 254)?.kind() != Vllm08Kind::Projection(Vllm08Projection::Head)
        || final_node.outputs != [ValueId(324), ValueId(325)]
        || head_node.inputs != [ValueId(324)]
        || head_node.outputs != [ValueId(326)]
        || typed(record, 254)?.selected_row() != Some(record.selected_row())
    {
        return Err(Error::State(
            "short-HF original Final253/Head254 producer/selected-row view".into(),
        ));
    }
    Ok(ShortHfAnalysis {
        rms,
        projections,
        gdn,
        attention,
        head,
    })
}
fn require_prepared_record(
    prepared: &RecordedVllm08Model,
    current: &RecordedVllm08Model,
) -> Result<()> {
    if !prepared.shares_prepared_semantics(current) {
        return Err(Error::State(
            "short-HF immutable prepared source/SSA/numerical identity differs".into(),
        ));
    }
    Ok(())
}
impl PreparedShortHfModel {
    fn metadata_bytes(record: &RecordedVllm08Model, analysis: &ShortHfAnalysis) -> Result<usize> {
        // Two finite site maps ≤64KiB each; transient320 canonical role names/sets
        // ≤128KiB. Include the retained first record's local Vec/Box witness too.
        let overflow = || Error::Limit("short-HF prepared binder metadata extent".into());
        let indices = analysis
            .rms
            .capacity()
            .checked_add(analysis.projections.capacity())
            .and_then(|n| n.checked_mul(std::mem::size_of::<usize>()))
            .ok_or_else(overflow)?;
        [
            std::mem::size_of::<Self>(),
            2 * std::mem::size_of::<usize>(),
            indices,
            256 * 1024,
        ]
        .into_iter()
        .try_fold(record.binding_metadata_bytes()?, |n, bytes| {
            n.checked_add(bytes).ok_or_else(overflow)
        })
    }
    fn new(
        record: Arc<RecordedVllm08Model>,
        analysis: ShortHfAnalysis,
        metadata: zerogravity_core::resource_arbiter::HostWorkspace,
    ) -> Result<Self> {
        if metadata.source_identity() != record.call().source()
            || Self::metadata_bytes(&record, &analysis)? > metadata.bytes()
        {
            return Err(Error::Limit(
                "short-HF original prepared binder/witness metadata bound".into(),
            ));
        }
        Ok(Self {
            record,
            analysis,
            _metadata: metadata,
        })
    }
    fn require_call(&self, call: &ShortHfCall, record: &RecordedVllm08Model) -> Result<()> {
        require_invocation_record(call, record)?;
        require_prepared_record(&self.record, record)
    }
}
impl ShortHfModelBindings {
    const MAX_SHAPES: usize = 40;
    pub(crate) const PREPARED_METADATA_BYTES: usize = Self::MAX_SHAPES * 512 * 1024 + 64 * 1024;
    pub(crate) fn new(source: BoundFullModelSource) -> Result<Arc<Self>> {
        let metadata = source.source().source().reserve_workspace(64 * 1024)?;
        Ok(Arc::new(Self {
            source,
            prepared: std::sync::Mutex::new(Vec::new()),
            _metadata: metadata,
        }))
    }
    pub(crate) fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    pub(crate) fn bind(
        &self,
        source: BoundShortHfCall,
        record: Arc<RecordedVllm08Model>,
    ) -> Result<Arc<ShortHfModelBinding>> {
        source.call().validate()?;
        let bound = source.source();
        if !Arc::ptr_eq(bound.source(), self.source.source())
            || bound.model_profile() != source.call().model_profile()
            || bound.source().source_identity() != record.call().source()
            || bound.source().source().staging_chunk_bytes() != 1024 * 1024
            || bound.source().owner().config().rms_norm_eps() != 1e-6
        {
            return Err(Error::State(
                "short-HF exact authenticated source/profile/chunk/RMS".into(),
            ));
        }
        let prepared = {
            let mut cache = self
                .prepared
                .lock()
                .map_err(|_| Error::State("short-HF prepared binding cache poisoned".into()))?;
            if let Some(prepared) = cache
                .iter()
                .find(|p| p.record.shares_prepared_semantics(&record))
            {
                prepared.clone()
            } else {
                if cache.len() >= Self::MAX_SHAPES {
                    return Err(Error::Limit(
                        "short-HF finite prepared binding shapes".into(),
                    ));
                }
                let metadata = bound.source().source().reserve_workspace(512 * 1024)?;
                let analysis = analyze_record(source.call(), &record, |op| {
                    validate_source_members(bound, op)
                })?;
                let prepared = Arc::new(PreparedShortHfModel::new(
                    record.clone(),
                    analysis,
                    metadata.into_host_workspace()?,
                )?);
                cache.push(prepared.clone());
                prepared
            }
        };
        prepared.require_call(source.call(), &record)?;
        let metadata = bound.source().source().reserve_workspace(512 * 1024)?;
        Ok(Arc::new(ShortHfModelBinding {
            source,
            record,
            prepared,
            _metadata: metadata,
        }))
    }
}
impl ShortHfModelBinding {
    // One-off diagnostic/inventory callers retain their original full validation.
    pub(crate) fn bind(
        source: BoundShortHfCall,
        record: Arc<RecordedVllm08Model>,
    ) -> Result<Arc<Self>> {
        ShortHfModelBindings::new(source.source().clone())?.bind(source, record)
    }
    pub(crate) fn bound_call(&self) -> &BoundShortHfCall {
        &self.source
    }
    pub(crate) fn source(&self) -> &BoundFullModelSource {
        self.source.source()
    }
    pub(crate) fn call(&self) -> &ShortHfCall {
        self.source.call()
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    /// Reuse this complete typed source-family proof; legacy .8B binders stay closed.
    pub(crate) fn require_record(
        &self,
        source: &BoundFullModelSource,
        record: &Arc<RecordedVllm08Model>,
    ) -> Result<()> {
        if !Arc::ptr_eq(source.source(), self.source().source())
            || source.model_profile() != self.call().model_profile()
            || !Arc::ptr_eq(record, &self.record)
        {
            return Err(Error::State(
                "short-HF exact source/profile/current record owner".into(),
            ));
        }
        self.prepared.require_call(self.call(), record)
    }
    pub(crate) fn rms_nodes(&self) -> &[usize] {
        &self.prepared.analysis.rms
    }
    pub(crate) fn projection_nodes(&self) -> &[usize] {
        &self.prepared.analysis.projections
    }
    pub(crate) fn head_view(&self) -> ShortHfHeadView {
        self.prepared.analysis.head
    }
    pub(crate) fn gdn_site(&self, layer: usize) -> Result<&ShortHfGdnSite> {
        self.prepared
            .analysis
            .gdn
            .get(&layer)
            .ok_or_else(|| Error::State("short-HF actual GDN source layer".into()))
    }
    pub(crate) fn attention_site(&self, layer: usize) -> Result<&ShortHfAttentionSite> {
        self.prepared
            .analysis
            .attention
            .get(&layer)
            .ok_or_else(|| Error::State("short-HF actual attention source layer".into()))
    }
    pub(crate) fn require_values(&self, values: &Arc<ModelValues>) -> Result<()> {
        if !Arc::ptr_eq(values.record(), &self.record)
            || values.service().source_identity() != self.source.source().source().source_identity()
        {
            return Err(Error::State(
                "short-HF SAME source/record/value issuer".into(),
            ));
        }
        Ok(())
    }
    /// The complete immutable cohort is checked once for this fresh record;
    /// each node still binds its own actual source operator and W/X/Y owners.
    pub(crate) fn bind_projections(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
        weights: &Arc<ModelWeights>,
    ) -> Result<BTreeMap<usize, ShortHfProjectionBinding>> {
        self.require_values(&values)?;
        weights.require_short_hf(self)?;
        self.prepared
            .analysis
            .projections
            .iter()
            .map(|&node| Ok((node, self.bind_projection(values.clone(), weights, node)?)))
            .collect()
    }
    fn bind_projection(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
        weights: &Arc<ModelWeights>,
        node: usize,
    ) -> Result<ShortHfProjectionBinding> {
        self.require_values(&values)?;
        if !self.prepared.analysis.projections.contains(&node)
            || !Arc::ptr_eq(weights.source(), self.source.source().source())
        {
            return Err(Error::State(
                "short-HF source-owned learned node/weights".into(),
            ));
        }
        let op = typed(&self.record, node)?;
        let n = &self.record.plan().nodes()[node];
        let ([x], [y]) = (n.inputs.as_slice(), n.outputs.as_slice()) else {
            return Err(Error::State("short-HF unary projection SSA".into()));
        };
        let weight = weights.require_operator(op)?.clone();
        let input = values.lease(*x)?.clone();
        let output = values.lease(*y)?.clone();
        if input.id() == output.id()
            || [input.id(), output.id()].contains(&weight.lease().id())
            || input.bytes() != op.input_elements()[0] * 2
            || output.bytes() != op.output_elements()[0] * 2
            || !Arc::ptr_eq(weight.lease().service(), values.service())
            || weight.lease().service_generation() != values.service().generation()
        {
            return Err(Error::State(
                "short-HF W/X/Y source extent/generation/nonalias".into(),
            ));
        }
        Ok(ShortHfProjectionBinding {
            scope: self.clone(),
            values,
            node,
            weight,
            input,
            output,
        })
    }
}
pub(crate) struct ShortHfProjectionBinding {
    scope: Arc<ShortHfModelBinding>,
    values: Arc<ModelValues>,
    node: usize,
    weight: Arc<ModelWeight>,
    input: ResidentLease,
    output: ResidentLease,
}
impl ShortHfProjectionBinding {
    pub(crate) fn weight(&self) -> &Arc<ModelWeight> {
        &self.weight
    }
    pub(crate) fn input(&self) -> &ResidentLease {
        &self.input
    }
    pub(crate) fn output(&self) -> &ResidentLease {
        &self.output
    }
    pub(crate) fn require(
        &self,
        values: &Arc<ModelValues>,
        op: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<()> {
        self.scope.require_values(values)?;
        self.weight
            .require_operator(self.scope.source().source(), op)?;
        if !Arc::ptr_eq(values, &self.values)
            || typed(&self.scope.record, self.node)? != op
            || input.id() != self.input.id()
            || output.id() != self.output.id()
            || !Arc::ptr_eq(input.service(), values.service())
            || !Arc::ptr_eq(output.service(), values.service())
        {
            return Err(Error::State(
                "short-HF sealed actual node/source/lease changed".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "short_hf_model/prepared_tests.rs"]
mod prepared_tests;
