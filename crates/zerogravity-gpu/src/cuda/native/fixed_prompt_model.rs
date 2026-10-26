//! Full-record source/producer seals only. No attention/head/Root/native permit.
use super::model_values::ModelValues;
use super::model_weights::ModelWeights;
use crate::cuda::gdn::{FixedPromptGdnArithmetic, GdnArithmetic};
use crate::cuda::rms::BoundModelRmsSource;
use crate::resident_leases::{ModelWeight, ResidentLease};
use crate::resident_program::native_full::FullModelProgram;
use crate::resident_program::source_bindings::SourceBindings;
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::accelerator::program::{PlanOperation, SourceCallMode};
use zerogravity_core::accelerator::program::{SourceCall, SourceOperator, SourceOperatorKind};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FixedPromptCallKind {
    Prefill,
    Decode { base: usize },
}
impl FixedPromptCallKind {
    pub(crate) fn of(request: &SourceCall) -> Result<Self> {
        if request.rows().capacity() != 512 || request.rows().vocabulary() != 248320 {
            return Err(Error::Unsupported(
                "fixed prompt physical capacity512 only".into(),
            ));
        }
        match (request.mode(), request.rows().base(), request.rows().len()) {
            (SourceCallMode::Prefill, 0, 47) => Ok(Self::Prefill),
            (SourceCallMode::Decode, base @ 47..=53, 1) => Ok(Self::Decode { base }),
            _ => Err(Error::Unsupported(
                "fixed prompt physical Prefill47/Decode47..53".into(),
            )),
        }
    }
    pub(crate) fn require_prefill_gdn(self, arithmetic: FixedPromptGdnArithmetic) -> Result<()> {
        if self != Self::Prefill {
            return Err(Error::Unsupported(
                "fixed_prompt cached GDN exact code/control/actual-prior coverage remains closed"
                    .into(),
            ));
        }
        arithmetic.require_owned()
    }
}
#[derive(Clone)]
pub(crate) struct FixedPromptGdnSite {
    layer: usize,
    nodes: [usize; 2],
    qkvz: ValueId,
    ba: ValueId,
    activated: ValueId,
    z: ValueId,
    gated: ValueId,
}
pub(crate) struct FixedPromptModelBinding {
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    kind: FixedPromptCallKind,
    rms: BTreeMap<usize, BoundModelRmsSource>,
    gdn: BTreeMap<usize, FixedPromptGdnSite>,
    projections: Vec<usize>,
    _metadata: WindowWorkspace,
}
impl FixedPromptModelBinding {
    /// Actual original recorder/source/teacher issuer, never a new plan or root.
    pub(crate) fn bind(record: Arc<SourceBindings>) -> Result<Arc<Self>> {
        FullModelProgram::require_record(&record)?;
        let source = record.source().clone();
        let kind = FixedPromptCallKind::of(record.call())?;
        let metadata = source.reserve_workspace(512 * 1024)?;
        let mut rms = BTreeMap::new();
        let mut gdn = BTreeMap::new();
        let mut projections = Vec::with_capacity(114);
        let head = record
            .plan()
            .source_value_provenance(record.selection().output())?
            .map(|p| p.node_index());
        for (index, node) in record.plan().nodes().iter().enumerate() {
            let Some(descriptor) = record.operator(index)? else {
                continue;
            };
            if descriptor.call() != record.call() {
                return Err(Error::State("fixed prompt foreign actual call".into()));
            }
            match descriptor.kind() {
                SourceOperatorKind::Rms(_) | SourceOperatorKind::ResidualRms(_) => {
                    rms.insert(
                        index,
                        BoundModelRmsSource::bind(source.clone(), descriptor.clone())?,
                    );
                }
                SourceOperatorKind::CausalConvSplit { domain, .. } => {
                    let layer = domain.0;
                    let next = record
                        .plan()
                        .nodes()
                        .get(index + 1)
                        .ok_or_else(|| Error::State("fixed prompt recurrence absent".into()))?;
                    let recurrent = record.operator(index + 1)?.ok_or_else(|| {
                        Error::State("fixed prompt recurrence descriptor absent".into())
                    })?;
                    let ([qkvz, ba], [activated, z], [ra, rz, rb], [gated]) = (
                        node.inputs.as_slice(),
                        node.outputs.as_slice(),
                        next.inputs.as_slice(),
                        next.outputs.as_slice(),
                    ) else {
                        return Err(Error::State(
                            "fixed prompt five GDN producer arities".into(),
                        ));
                    };
                    if !matches!(recurrent.kind(),SourceOperatorKind::GatedDelta {domain, ..} if domain.0==layer)
                        || recurrent.call() != record.call()
                        || activated != ra
                        || z != rz
                        || ba != rb
                        || gdn.contains_key(&layer)
                    {
                        return Err(Error::State(
                            "fixed prompt actual GDN pair/call/SSA differs".into(),
                        ));
                    }
                    gdn.insert(
                        layer,
                        FixedPromptGdnSite {
                            layer,
                            nodes: [index, index + 1],
                            qkvz: *qkvz,
                            ba: *ba,
                            activated: *activated,
                            z: *z,
                            gated: *gated,
                        },
                    );
                }
                SourceOperatorKind::Projection { selected_row: None } if Some(index) != head => {
                    projections.push(index);
                }
                _ => {}
            }
        }
        if rms.is_empty()
            || rms.len() > 49
            || gdn.len() > 18
            || projections.len() > 114
            || !rms.contains_key(&1)
        {
            return Err(Error::Unsupported(
                "fixed prompt native site storage capacity".into(),
            ));
        }
        Ok(Arc::new(Self {
            source,
            record,
            kind,
            rms,
            gdn,
            projections,
            _metadata: metadata,
        }))
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn record(&self) -> &Arc<SourceBindings> {
        &self.record
    }
    pub(crate) fn kind(&self) -> FixedPromptCallKind {
        self.kind
    }
    pub(crate) fn rms(&self, node: usize) -> Result<&BoundModelRmsSource> {
        self.rms
            .get(&node)
            .ok_or_else(|| Error::State("fixed_prompt exact original RMS node absent".into()))
    }
    pub(crate) fn require_values(&self, values: &Arc<ModelValues>) -> Result<()> {
        if !Arc::ptr_eq(values.record(), &self.record)
            || values.service().source_identity() != self.source.source_identity()
        {
            return Err(Error::State(
                "fixed_prompt actual complete value/source/record issuer differs".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn gdn_site(&self, layer: usize) -> Result<&FixedPromptGdnSite> {
        self.gdn
            .get(&layer)
            .ok_or_else(|| Error::Unsupported("fixed_prompt actual GDN layer only".into()))
    }
    /// Existing physical ModelGdnBinding must still seal all real leases/domains.
    pub(crate) fn require_gdn_arithmetic(
        &self,
        layer: usize,
        actual: &GdnArithmetic,
        declared: FixedPromptGdnArithmetic,
    ) -> Result<()> {
        self.gdn_site(layer)?;
        self.kind.require_prefill_gdn(declared)?;
        declared.require_prepared(actual)
    }
    pub(crate) fn bind_projection(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
        weights: &Arc<ModelWeights>,
        node: usize,
    ) -> Result<FixedPromptProjectionBinding> {
        self.require_values(&values)?;
        if !self.projections.contains(&node) || !weights.source().same_issuer(&self.source) {
            return Err(Error::State(
                "fixed_prompt exact source projection node/weight issuer".into(),
            ));
        }
        let item = &self.record.plan().nodes()[node];
        let Some(descriptor) = self.record.operator(node)? else {
            return Err(Error::State(
                "fixed_prompt actual projection descriptor".into(),
            ));
        };
        let ([x], [y]) = (item.inputs.as_slice(), item.outputs.as_slice()) else {
            return Err(Error::State(
                "fixed_prompt actual unary projection producer".into(),
            ));
        };
        let weight = weights.require_operator(descriptor)?.clone();
        let input = values.lease(*x)?.clone();
        let output = values.lease(*y)?.clone();
        if input.id() == output.id()
            || [input.id(), output.id()].contains(&weight.lease().id())
            || !Arc::ptr_eq(weight.lease().service(), values.service())
            || weight.lease().service_generation() != values.service().generation()
        {
            return Err(Error::State(
                "fixed_prompt actual W/producer/output nonalias".into(),
            ));
        }
        Ok(FixedPromptProjectionBinding {
            scope: self.clone(),
            values,
            node,
            weight,
            input,
            output,
        })
    }
    /// Cold explicit blocker; no attention/head proof or current-root schedule exists yet.
    pub(crate) fn require_native_activation(&self) -> Result<()> {
        Err(Error::Unsupported("fixed_prompt all24 native activation closed: exact layer/call projection and GDN coverage, genuine current-attempt/cache, attention47, head47 and budgets required".into()))
    }
}
pub(crate) struct FixedPromptProjectionBinding {
    scope: Arc<FixedPromptModelBinding>,
    values: Arc<ModelValues>,
    node: usize,
    weight: Arc<ModelWeight>,
    input: ResidentLease,
    output: ResidentLease,
}
impl FixedPromptProjectionBinding {
    pub(crate) fn weight(&self) -> &ResidentLease {
        self.weight.lease()
    }
    /// Selected call0 QKV packet scope only; no current-attempt/cursor or Ready authority.
    pub(crate) fn require_fixed_prompt_attention_qkv(&self) -> Result<()> {
        let node = &self.scope.record.plan().nodes()[self.node];
        let Some(descriptor) = self.scope.record.operator(self.node)? else {
            return Err(Error::State(
                "fixed_prompt actual attention QKV node".into(),
            ));
        };
        self.require(&self.values, descriptor, &self.input, &self.output)?;
        self.weight.require_initialized()?;
        if self.node != 32
            || node.inputs != [ValueId(39)]
            || node.outputs != [ValueId(41)]
            || FixedPromptCallKind::of(descriptor.call())? != FixedPromptCallKind::Prefill
        {
            return Err(Error::Unsupported(
                "fixed prompt physical QKV32/SSA only".into(),
            ));
        }
        if self.scope.kind != FixedPromptCallKind::Prefill
            || !matches!(
                descriptor.kind(),
                SourceOperatorKind::Projection { selected_row: None }
            )
            || descriptor.call() != self.scope.record.call()
            || descriptor.matrices().len() != 3
            || !descriptor.parameters().is_empty()
            || descriptor.input_elements()?.as_slice() != [47 * 1024]
            || descriptor.output_elements()?.as_slice() != [47 * 5120]
            || node.state.is_some()
            || self.input.bytes() != 96256
            || self.output.bytes() != 481280
            || self.weight.lease().bytes() != 10485760
        {
            return Err(Error::State(
                "fixed_prompt exact layer3 QKV source/shape/weight".into(),
            ));
        }
        let normal = &self.scope.record.plan().nodes()[31];
        let Some(producer) = self.scope.record.operator(31)? else {
            return Err(Error::State(
                "fixed_prompt QKV actual dual producer absent".into(),
            ));
        };
        if producer.call() != descriptor.call()
            || !matches!(producer.kind(), SourceOperatorKind::ResidualRms(_))
            || normal.inputs.as_slice() != [ValueId(38), ValueId(35)]
            || normal.outputs.as_slice() != [ValueId(39), ValueId(40)]
            || normal.state.is_some()
        {
            return Err(Error::State(
                "fixed_prompt QKV same original computed dual input".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require(
        &self,
        values: &Arc<ModelValues>,
        operator: &SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<()> {
        self.scope.require_values(values)?;
        let PlanOperation::SourceOperator { descriptor, .. } =
            &self.scope.record.plan().nodes()[self.node].operation
        else {
            return Err(Error::State(
                "fixed_prompt original projection node changed".into(),
            ));
        };
        if operator != descriptor.as_ref() {
            return Err(Error::State(
                "fixed prompt actual mathematical operator changed".into(),
            ));
        }
        self.weight
            .require_source_operator(&self.scope.source, operator)?;
        if !Arc::ptr_eq(values, &self.values)
            || input.id() != self.input.id()
            || output.id() != self.output.id()
            || !Arc::ptr_eq(input.service(), self.values.service())
            || !Arc::ptr_eq(output.service(), self.values.service())
        {
            return Err(Error::State(
                "fixed_prompt sealed actual projection source/SSA/lease binding changed".into(),
            ));
        }
        Ok(())
    }
}
impl FixedPromptGdnSite {
    pub(crate) fn layer(&self) -> usize {
        self.layer
    }
    pub(crate) fn nodes(&self) -> [usize; 2] {
        self.nodes
    }
    pub(crate) fn operands(&self) -> [ValueId; 5] {
        [self.qkvz, self.ba, self.activated, self.z, self.gated]
    }
}
#[cfg(test)]
mod tests;
