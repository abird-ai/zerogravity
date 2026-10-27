//! Actual source/SSA bindings; model factories own family recipes and eligibility.
use super::{model_values::ModelValues, model_weights::ModelWeights};
use crate::cuda::gdn::TextGdnImplementation;
use crate::cuda::native::text_native::TextKernelCapture;
use crate::resident_leases::{ModelWeight, ResidentLease};
use crate::resident_program::source_bindings::{SourceBindings, SourceSelection};
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    ForwardPlan, PlanNode, PlanOperation, SourceCall, SourceCallMode, SourceOperator,
    SourceOperatorKind, StateSlot, ValueId,
};
use zerogravity_core::{Error, Result};

#[derive(Clone, Debug)]
pub(crate) struct TextGdnSite {
    pub(crate) nodes: [usize; 2],
    pub(crate) operands: [ValueId; 5],
}
#[derive(Clone, Debug)]
pub(crate) struct TextAttentionSite {
    pub(crate) nodes: [usize; 7],
    pub(crate) operands: [ValueId; 9],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TextHeadView {
    pub(crate) full_bytes: usize,
    pub(crate) offset: usize,
    pub(crate) selected_bytes: usize,
    pub(crate) output_bytes: usize,
}
impl TextHeadView {
    fn new(rows: usize, columns: usize, outputs: usize, selected_row: usize) -> Result<Self> {
        if rows == 0 || columns == 0 || outputs == 0 || selected_row >= rows {
            return Err(Error::State(
                "source head actual selected row/extent".into(),
            ));
        }
        let checked = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| Error::Limit("source head byte extent".into()))
        };
        let row = checked(columns, 2)?;
        Ok(Self {
            full_bytes: checked(rows, row)?,
            offset: checked(selected_row, row)?,
            selected_bytes: row,
            output_bytes: checked(outputs, 2)?,
        })
    }
}
pub(crate) struct TextModelBinding {
    record: Arc<SourceBindings>,
    prepared: Arc<PreparedTextModel>,
    gdn_implementation: TextGdnImplementation,
    native_capture: TextKernelCapture,
    _metadata: WindowWorkspace,
}
pub(crate) struct TextAnalysis {
    rms: Vec<usize>,
    projections: Vec<usize>,
    gdn: BTreeMap<usize, TextGdnSite>,
    pub(crate) attention: BTreeMap<usize, TextAttentionSite>,
    head: TextHeadView,
}
struct PreparedTextModel {
    record: Arc<SourceBindings>,
    analysis: TextAnalysis,
    gdn_implementation: TextGdnImplementation,
    native_capture: TextKernelCapture,
    _metadata: zerogravity_core::resource_arbiter::HostWorkspace,
}
pub(crate) struct TextModelBindings {
    source: VerifiedTensorSource,
    capture: TextKernelCapture,
    prepared: std::sync::Mutex<Vec<Arc<PreparedTextModel>>>,
    _metadata: WindowWorkspace,
}
fn require_state_read(node: &PlanNode, descriptor: &SourceOperator) -> Result<()> {
    let PlanOperation::SourceOperator {
        descriptor: actual,
        state,
    } = &node.operation
    else {
        return Err(Error::State("source state-read operation absent".into()));
    };
    if actual.as_ref() != descriptor {
        return Err(Error::State(
            "source actual mathematical state-read descriptor differs".into(),
        ));
    }
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
            "source FA reads actual appended KV ordinal1, never a write".into(),
        ));
    }
    Ok(())
}
fn operator(plan: &ForwardPlan, index: usize) -> Result<&SourceOperator> {
    let node = plan
        .nodes()
        .get(index)
        .ok_or_else(|| Error::State("source operator node outside actual graph".into()))?;
    let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
        return Err(Error::State("source mathematical operator absent".into()));
    };
    Ok(descriptor)
}
fn typed(record: &SourceBindings, index: usize) -> Result<&SourceOperator> {
    operator(record.plan(), index)
}
pub(crate) fn analyze_record(record: &SourceBindings) -> Result<TextAnalysis> {
    analyze_plan(record.plan(), record.selection())
}
fn analyze_plan(plan: &ForwardPlan, selection: &SourceSelection) -> Result<TextAnalysis> {
    let output = selection.output();
    let normalized = selection
        .normalized()
        .ok_or_else(|| Error::State("source head normalized value selection absent".into()))?;
    let head_index = plan
        .source_value_provenance(output)?
        .ok_or_else(|| Error::State("source head output producer absent".into()))?
        .node_index();
    let head_operator = operator(plan, head_index)?;
    let SourceOperatorKind::Projection {
        selected_row: Some(selected_row),
    } = head_operator.kind()
    else {
        return Err(Error::State(
            "source head actual selected projection absent".into(),
        ));
    };
    let [matrix] = head_operator.matrices() else {
        return Err(Error::State("source head actual tied matrix arity".into()));
    };
    let head_node = &plan.nodes()[head_index];
    if selection.selected_row() != Some(selected_row)
        || head_node.inputs != [normalized]
        || head_node.outputs != [output]
    {
        return Err(Error::State(
            "source head selected row/input/output SSA differs".into(),
        ));
    }
    let head = TextHeadView::new(
        head_operator.call().rows().len(),
        matrix.columns,
        matrix.rows,
        selected_row,
    )?;
    let mut rms = Vec::new();
    let mut projections = Vec::new();
    let mut gdn = BTreeMap::new();
    let mut attention = BTreeMap::new();
    for (index, node) in plan.nodes().iter().enumerate() {
        let descriptor = operator(plan, index)?;
        require_state_read(node, descriptor)?;
        match descriptor.kind() {
            SourceOperatorKind::Rms(_) | SourceOperatorKind::ResidualRms(_) => rms.push(index),
            SourceOperatorKind::Projection { selected_row: None } => projections.push(index),
            SourceOperatorKind::CausalConvSplit { domain, .. } => {
                let next = plan
                    .nodes()
                    .get(index + 1)
                    .ok_or_else(|| Error::State("source recurrence absent".into()))?;
                let recurrent = operator(plan, index + 1)?;
                let ([qkvz, ba], [activated, z], [ra, rz, rb], [gated]) = (
                    node.inputs.as_slice(),
                    node.outputs.as_slice(),
                    next.inputs.as_slice(),
                    next.outputs.as_slice(),
                ) else {
                    return Err(Error::State("source GDN pair actual arities".into()));
                };
                if !matches!(recurrent.kind(), SourceOperatorKind::GatedDelta { domain: rd, .. }
                    if rd.0 == domain.0)
                    || recurrent.call() != descriptor.call()
                    || activated != ra
                    || z != rz
                    || ba != rb
                    || node.state.is_none()
                    || next.state.is_none()
                    || gdn
                        .insert(
                            domain.0,
                            TextGdnSite {
                                nodes: [index, index + 1],
                                operands: [*qkvz, *ba, *activated, *z, *gated],
                            },
                        )
                        .is_some()
                {
                    return Err(Error::State(
                        "source GDN actual call/domain/producer pair".into(),
                    ));
                }
            }
            SourceOperatorKind::QkRmsRotaryGated(_) => {
                let first = index
                    .checked_sub(1)
                    .ok_or_else(|| Error::State("source attention QKV producer absent".into()))?;
                let ns: [usize; 7] = std::array::from_fn(|offset| first + offset);
                let ops = ns
                    .map(|n| operator(plan, n))
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
                let domain = match ops[2].kind() {
                    SourceOperatorKind::KvAppend { domain, .. } => domain,
                    _ => return Err(Error::State("source attention KV append absent".into())),
                };
                if !matches!(
                    ops[0].kind(),
                    SourceOperatorKind::Projection { selected_row: None }
                ) || !matches!(ops[3].kind(), SourceOperatorKind::FlashAttention { domain: d, .. } if d == domain)
                    || !matches!(ops[4].kind(), SourceOperatorKind::Sigmoid { .. })
                    || !matches!(ops[5].kind(), SourceOperatorKind::Multiply { .. })
                    || !matches!(
                        ops[6].kind(),
                        SourceOperatorKind::Projection { selected_row: None }
                    )
                    || ops.iter().any(|op| op.call() != descriptor.call())
                {
                    return Err(Error::State(
                        "source attention actual mathematical order/domain".into(),
                    ));
                }
                let nodes = plan.nodes();
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
                        "source attention actual producer arities".into(),
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
                            domain.0,
                            TextAttentionSite {
                                nodes: ns,
                                operands: [*x, *qkv, *q, *k, *gate, *o, *sig, *gated, *y],
                            },
                        )
                        .is_some()
                {
                    return Err(Error::State(
                        "source attention actual packed/current KV/producer chain".into(),
                    ));
                }
            }
            _ => {}
        }
    }
    if rms.len() > 49 || projections.len() > 114 || gdn.len() > 18 || attention.len() > 6 {
        return Err(Error::Limit(
            "source prepared site metadata capacity".into(),
        ));
    }
    let bytes = |id: ValueId| {
        plan.values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("source head actual value absent".into()))?
            .capacity_elements()?
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("source head actual value bytes".into()))
    };
    if bytes(normalized)? != head.full_bytes || bytes(output)? != head.output_bytes {
        return Err(Error::State(
            "source head actual value extent differs".into(),
        ));
    }
    Ok(TextAnalysis {
        rms,
        projections,
        gdn,
        attention,
        head,
    })
}
fn require_prepared_record(prepared: &SourceBindings, current: &SourceBindings) -> Result<()> {
    if !prepared.shares_prepared_semantics(current) {
        return Err(Error::State(
            "source immutable prepared math/SSA identity differs".into(),
        ));
    }
    Ok(())
}
impl PreparedTextModel {
    fn metadata_bytes(analysis: &TextAnalysis) -> Result<usize> {
        analysis
            .rms
            .capacity()
            .checked_add(analysis.projections.capacity())
            .and_then(|n| n.checked_mul(std::mem::size_of::<usize>()))
            .and_then(|n| {
                n.checked_add(
                    std::mem::size_of::<Self>() + 256 * 1024 + 2 * std::mem::size_of::<usize>(),
                )
            })
            .ok_or_else(|| Error::Limit("source prepared site metadata extent".into()))
    }
    fn new(
        record: Arc<SourceBindings>,
        analysis: TextAnalysis,
        gdn_implementation: TextGdnImplementation,
        native_capture: TextKernelCapture,
        metadata: zerogravity_core::resource_arbiter::HostWorkspace,
    ) -> Result<Self> {
        if metadata.source_identity() != record.call().source()
            || Self::metadata_bytes(&analysis)? > metadata.bytes()
        {
            return Err(Error::Limit("source prepared site metadata bound".into()));
        }
        Ok(Self {
            record,
            analysis,
            gdn_implementation,
            native_capture,
            _metadata: metadata,
        })
    }
    fn require_call(
        &self,
        record: &SourceBindings,
        implementation: TextGdnImplementation,
        capture: TextKernelCapture,
    ) -> Result<()> {
        if self.gdn_implementation != implementation || self.native_capture != capture {
            return Err(Error::State(
                "source prepared GDN implementation differs".into(),
            ));
        }
        require_prepared_record(&self.record, record)
    }
}
impl TextModelBindings {
    pub(crate) fn prepared_metadata_bytes(capture: TextKernelCapture) -> Result<usize> {
        capture
            .prepared_call_count()?
            .checked_mul(512 * 1024)
            .and_then(|n| n.checked_add(64 * 1024))
            .ok_or_else(|| Error::Limit("source prepared binding metadata extent".into()))
    }
    pub(crate) fn new(
        source: VerifiedTensorSource,
        capture: TextKernelCapture,
    ) -> Result<Arc<Self>> {
        let metadata = source.reserve_workspace(64 * 1024)?;
        Ok(Arc::new(Self {
            source,
            capture,
            prepared: std::sync::Mutex::new(Vec::new()),
            _metadata: metadata,
        }))
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn bind(
        &self,
        record: Arc<SourceBindings>,
        gdn_implementation: TextGdnImplementation,
        native_capture: TextKernelCapture,
    ) -> Result<Arc<TextModelBinding>> {
        if native_capture != self.capture
            || !self.source.same_issuer(record.source())
            || gdn_implementation.input_tokens() != record.call().rows().len()
            || gdn_implementation.is_decode() != (record.call().mode() == SourceCallMode::Decode)
        {
            return Err(Error::State(
                "source actual source/call/GDN implementation differs".into(),
            ));
        }
        let prepared = {
            let mut cache = self
                .prepared
                .lock()
                .map_err(|_| Error::State("source prepared binding cache poisoned".into()))?;
            if let Some(prepared) = cache.iter().find(|p| {
                p.record.shares_prepared_semantics(&record)
                    && p.gdn_implementation == gdn_implementation
                    && p.native_capture == native_capture
            }) {
                prepared.clone()
            } else {
                if cache.len() >= self.capture.prepared_call_count()? {
                    return Err(Error::Limit(
                        "source finite prepared binding capacity".into(),
                    ));
                }
                native_capture.require_bindings(&record)?;
                let metadata = self.source.reserve_workspace(512 * 1024)?;
                let analysis = analyze_record(&record)?;
                let prepared = Arc::new(PreparedTextModel::new(
                    record.clone(),
                    analysis,
                    gdn_implementation,
                    native_capture,
                    metadata.into_host_workspace()?,
                )?);
                cache.push(prepared.clone());
                prepared
            }
        };
        prepared.require_call(&record, gdn_implementation, native_capture)?;
        let metadata = self.source.reserve_workspace(512 * 1024)?;
        Ok(Arc::new(TextModelBinding {
            record,
            prepared,
            gdn_implementation,
            native_capture,
            _metadata: metadata,
        }))
    }
}
impl TextModelBinding {
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        self.record.source()
    }
    pub(crate) fn call(&self) -> &SourceCall {
        self.record.call()
    }
    pub(crate) fn record(&self) -> &Arc<SourceBindings> {
        &self.record
    }
    pub(crate) fn native_capture(&self) -> TextKernelCapture {
        self.native_capture
    }
    pub(crate) fn hidden_size(&self) -> usize {
        self.prepared.analysis.head.selected_bytes / 2
    }
    pub(crate) fn gdn_implementation(&self) -> TextGdnImplementation {
        self.gdn_implementation
    }
    pub(crate) fn require_record(
        &self,
        source: &VerifiedTensorSource,
        record: &Arc<SourceBindings>,
    ) -> Result<()> {
        if !source.same_issuer(self.source()) || !Arc::ptr_eq(record, &self.record) {
            return Err(Error::State(
                "source exact current source/graph owner differs".into(),
            ));
        }
        self.prepared
            .require_call(record, self.gdn_implementation, self.native_capture)
    }
    pub(crate) fn rms_nodes(&self) -> &[usize] {
        &self.prepared.analysis.rms
    }
    pub(crate) fn projection_nodes(&self) -> &[usize] {
        &self.prepared.analysis.projections
    }
    pub(crate) fn head_view(&self) -> TextHeadView {
        self.prepared.analysis.head
    }
    pub(crate) fn gdn_sites(&self) -> impl ExactSizeIterator<Item = (usize, &TextGdnSite)> {
        self.prepared
            .analysis
            .gdn
            .iter()
            .map(|(&site, value)| (site, value))
    }
    pub(crate) fn attention_sites(
        &self,
    ) -> impl ExactSizeIterator<Item = (usize, &TextAttentionSite)> {
        self.prepared
            .analysis
            .attention
            .iter()
            .map(|(&site, value)| (site, value))
    }
    pub(crate) fn gdn_site(&self, layer: usize) -> Result<&TextGdnSite> {
        self.prepared
            .analysis
            .gdn
            .get(&layer)
            .ok_or_else(|| Error::State("source actual GDN domain site absent".into()))
    }
    pub(crate) fn attention_site(&self, layer: usize) -> Result<&TextAttentionSite> {
        self.prepared
            .analysis
            .attention
            .get(&layer)
            .ok_or_else(|| Error::State("source actual attention domain site absent".into()))
    }
    pub(crate) fn require_values(&self, values: &Arc<ModelValues>) -> Result<()> {
        if !Arc::ptr_eq(values.record(), &self.record)
            || values.service().source_identity() != self.source().source_identity()
        {
            return Err(Error::State("source same actual graph/value issuer".into()));
        }
        Ok(())
    }
    /// The complete immutable cohort is checked once for this fresh record;
    /// each node still binds its own actual source operator and W/X/Y owners.
    pub(crate) fn bind_projections(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
        weights: &Arc<ModelWeights>,
    ) -> Result<BTreeMap<usize, TextProjectionBinding>> {
        self.require_values(&values)?;
        weights.require_text(self)?;
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
    ) -> Result<TextProjectionBinding> {
        self.require_values(&values)?;
        if !self.prepared.analysis.projections.contains(&node)
            || !weights.source().same_issuer(self.source())
        {
            return Err(Error::State(
                "text source-owned learned node/weights".into(),
            ));
        }
        let op = typed(&self.record, node)?;
        let n = &self.record.plan().nodes()[node];
        let ([x], [y]) = (n.inputs.as_slice(), n.outputs.as_slice()) else {
            return Err(Error::State("text unary projection SSA".into()));
        };
        let weight = weights.require_operator(op)?.clone();
        let input = values.lease(*x)?.clone();
        let output = values.lease(*y)?.clone();
        if input.id() == output.id()
            || [input.id(), output.id()].contains(&weight.lease().id())
            || input.bytes() != op.input_elements()?[0] * 2
            || output.bytes() != op.output_elements()?[0] * 2
            || !Arc::ptr_eq(weight.lease().service(), values.service())
            || weight.lease().service_generation() != values.service().generation()
        {
            return Err(Error::State(
                "text W/X/Y source extent/generation/nonalias".into(),
            ));
        }
        Ok(TextProjectionBinding {
            scope: self.clone(),
            values,
            node,
            weight,
            input,
            output,
        })
    }
}
pub(crate) struct TextProjectionBinding {
    scope: Arc<TextModelBinding>,
    values: Arc<ModelValues>,
    node: usize,
    weight: Arc<ModelWeight>,
    input: ResidentLease,
    output: ResidentLease,
}
impl TextProjectionBinding {
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
        op: &zerogravity_core::accelerator::program::SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<()> {
        self.scope.require_values(values)?;
        self.weight
            .require_source_operator(self.scope.source(), op)?;
        if !Arc::ptr_eq(values, &self.values)
            || !matches!(&self.scope.record.plan().nodes()[self.node].operation,
                PlanOperation::SourceOperator { descriptor, .. } if descriptor.as_ref() == op)
            || input.id() != self.input.id()
            || output.id() != self.output.id()
            || !Arc::ptr_eq(input.service(), values.service())
            || !Arc::ptr_eq(output.service(), values.service())
        {
            return Err(Error::State(
                "text sealed actual node/source/lease changed".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "text_model/prepared_tests.rs"]
mod prepared_tests;
