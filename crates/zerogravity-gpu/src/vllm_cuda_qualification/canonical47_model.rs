//! Full-record source/producer seals only. No attention/head/Root/native permit.
use super::model_values::ModelValues;
use super::model_weights::ModelWeights;
use crate::resident_leases::{ModelWeight, ResidentLease};
use crate::resident_program::native_full::FullModelProgram;
use crate::resident_program::qwen2b_component::CANONICAL47_PROMPT;
use crate::vllm_cuda_gdn::{Canonical47GdnArithmetic, GdnArithmetic};
use crate::vllm_cuda_input_rms::BoundModelRmsSource;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    FullModelProfile, PlanOperation, RecordedVllm08Model, ValueId, Vllm08Kind, Vllm08Operator,
    Vllm08Projection, Vllm08RmsSite, VllmBf16Invocation, VllmBf16Mode,
};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::{
    BoundFullModelSource, QwenMatrixRole, QwenParameterRole,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Canonical47CallKind {
    Prefill,
    Decode { base: usize },
}
impl Canonical47CallKind {
    pub(crate) fn of(request: &VllmBf16Invocation) -> Result<Self> {
        request.validate()?;
        if request.layer().index() != 0
            || request.teachers().short_hf_span().is_some()
            || request.teachers().capacity() != 512
            || request.legacy_phase().is_ok()
        {
            return Err(Error::Unsupported(
                "canonical47 full-record source span only".into(),
            ));
        }
        match (request.mode(), request.teachers().base(), request.rows()) {
            (VllmBf16Mode::Prefill, 0, 47) if request.teachers().ids() == CANONICAL47_PROMPT => {
                Ok(Self::Prefill)
            }
            (VllmBf16Mode::Decode, base @ 47..=53, 1) => Ok(Self::Decode { base }),
            _ => Err(Error::Unsupported(
                "canonical47 Prefill47/Decode47..53 source span".into(),
            )),
        }
    }
    pub(crate) fn require_prefill_gdn(self, arithmetic: Canonical47GdnArithmetic) -> Result<()> {
        if self != Self::Prefill {
            return Err(Error::Unsupported(
                "canonical47 cached GDN exact code/control/actual-prior coverage remains closed"
                    .into(),
            ));
        }
        arithmetic.require_owned()
    }
}
#[derive(Clone)]
pub(crate) struct Canonical47GdnSite {
    layer: usize,
    nodes: [usize; 2],
    qkvz: ValueId,
    ba: ValueId,
    activated: ValueId,
    z: ValueId,
    gated: ValueId,
}
pub(crate) struct Canonical47ModelBinding {
    source: BoundFullModelSource,
    record: Arc<RecordedVllm08Model>,
    kind: Canonical47CallKind,
    rms: BTreeMap<usize, BoundModelRmsSource>,
    gdn: BTreeMap<usize, Canonical47GdnSite>,
    projections: Vec<usize>,
    _metadata: WindowWorkspace,
}
impl Canonical47ModelBinding {
    /// Actual original recorder/source/teacher issuer, never a new plan or root.
    pub(crate) fn bind(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
    ) -> Result<Arc<Self>> {
        if source.model_profile() != FullModelProfile::Qwen08
            || record.call().model_profile() != FullModelProfile::Qwen08
            || source.source().source().source_bytes() != 1_746_942_600
            || source.source().source().staging_chunk_bytes() != 1024 * 1024
            || source.source().owner().config().rms_norm_eps() != 1e-6
        {
            return Err(Error::Unsupported(
                "canonical47 original .8B source/config/1MiB staging".into(),
            ));
        }
        FullModelProgram::require_record(source.source(), &record)?;
        let kind = Canonical47CallKind::of(record.call().request())?;
        let metadata = source.source().source().reserve_workspace(512 * 1024)?;
        source.source().validate_matrix(
            QwenMatrixRole::Embedding,
            &source
                .source()
                .matrix_descriptor(QwenMatrixRole::Embedding)?,
        )?;
        let mut rms = BTreeMap::new();
        let mut gdn = BTreeMap::new();
        let mut projections = Vec::with_capacity(114);
        let mut matrix_names = BTreeSet::new();
        let mut parameter_names = BTreeSet::new();
        for (index, node) in record.plan().nodes().iter().enumerate() {
            let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                continue;
            };
            if descriptor.call() != record.call() {
                return Err(Error::State(
                    "canonical47 full-record foreign source/teacher issuer".into(),
                ));
            }
            validate_source_members(&source, descriptor)?;
            matrix_names.extend(descriptor.matrices().iter().map(|m| m.name.clone()));
            parameter_names.extend(descriptor.parameters().iter().map(|p| p.name.clone()));
            match descriptor.kind() {
                Vllm08Kind::InputRms | Vllm08Kind::ResidualRms(_) => {
                    rms.insert(
                        index,
                        BoundModelRmsSource::bind(
                            source.source().clone(),
                            descriptor.as_ref().clone(),
                        )?,
                    );
                }
                Vllm08Kind::GdnConv => {
                    let layer = descriptor
                        .layer()
                        .ok_or_else(|| Error::State("canonical47 Conv layer absent".into()))?
                        .index();
                    let next = record
                        .plan()
                        .nodes()
                        .get(index + 1)
                        .ok_or_else(|| Error::State("canonical47 recurrence absent".into()))?;
                    let PlanOperation::TypedVllm08 {
                        descriptor: recurrent,
                        ..
                    } = &next.operation
                    else {
                        return Err(Error::State("canonical47 typed recurrence absent".into()));
                    };
                    let ([qkvz, ba], [activated, z], [ra, rz, rb], [gated]) = (
                        node.inputs.as_slice(),
                        node.outputs.as_slice(),
                        next.inputs.as_slice(),
                        next.outputs.as_slice(),
                    ) else {
                        return Err(Error::State(
                            "canonical47 actual five GDN producer arities".into(),
                        ));
                    };
                    if recurrent.kind() != Vllm08Kind::GdnRecurrent
                        || recurrent.call() != record.call()
                        || recurrent.layer() != descriptor.layer()
                        || activated != ra
                        || z != rz
                        || ba != rb
                        || gdn.contains_key(&layer)
                    {
                        return Err(Error::State(
                            "canonical47 exact per-layer Conv/recurrent SSA/source pair".into(),
                        ));
                    }
                    gdn.insert(
                        layer,
                        Canonical47GdnSite {
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
                Vllm08Kind::Projection(p) if p != Vllm08Projection::Head => projections.push(index),
                _ => {}
            }
        }
        if rms.len() != 49
            || gdn.len() != 18
            || projections.len() != 114
            || matrix_names.len() != 187
            || parameter_names.len() != 133
            || !matrix_names.is_disjoint(&parameter_names)
            || !rms.contains_key(&1)
            || gdn.keys().copied().collect::<Vec<_>>()
                != (0..24).filter(|i| i % 4 != 3).collect::<Vec<_>>()
        {
            return Err(Error::State(
                "canonical47 all24 source coverage:49 RMS/18 GDN/114 learned groups".into(),
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
    pub(crate) fn source(&self) -> &BoundFullModelSource {
        &self.source
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    pub(crate) fn kind(&self) -> Canonical47CallKind {
        self.kind
    }
    pub(crate) fn rms(&self, node: usize) -> Result<&BoundModelRmsSource> {
        self.rms
            .get(&node)
            .ok_or_else(|| Error::State("canonical47 exact original RMS node absent".into()))
    }
    pub(crate) fn require_values(&self, values: &Arc<ModelValues>) -> Result<()> {
        if !Arc::ptr_eq(values.record(), &self.record)
            || values.service().source_identity() != self.source.source().source_identity()
        {
            return Err(Error::State(
                "canonical47 actual complete value/source/record issuer differs".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn gdn_site(&self, layer: usize) -> Result<&Canonical47GdnSite> {
        self.gdn
            .get(&layer)
            .ok_or_else(|| Error::Unsupported("canonical47 actual GDN layer only".into()))
    }
    /// Existing physical ModelGdnBinding must still seal all real leases/domains.
    pub(crate) fn require_gdn_arithmetic(
        &self,
        layer: usize,
        actual: &GdnArithmetic,
        declared: Canonical47GdnArithmetic,
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
    ) -> Result<Canonical47ProjectionBinding> {
        self.require_values(&values)?;
        if !self.projections.contains(&node) || !Arc::ptr_eq(weights.source(), self.source.source())
        {
            return Err(Error::State(
                "canonical47 exact source projection node/weight issuer".into(),
            ));
        }
        let item = &self.record.plan().nodes()[node];
        let PlanOperation::TypedVllm08 { descriptor, .. } = &item.operation else {
            return Err(Error::State(
                "canonical47 actual projection descriptor".into(),
            ));
        };
        let ([x], [y]) = (item.inputs.as_slice(), item.outputs.as_slice()) else {
            return Err(Error::State(
                "canonical47 actual unary projection producer".into(),
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
                "canonical47 actual W/producer/output nonalias".into(),
            ));
        }
        Ok(Canonical47ProjectionBinding {
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
        Err(Error::Unsupported("canonical47 all24 native activation closed: exact layer/call projection and GDN coverage, genuine current-attempt/cache, attention47, head47 and budgets required".into()))
    }
}
pub(crate) struct Canonical47ProjectionBinding {
    scope: Arc<Canonical47ModelBinding>,
    values: Arc<ModelValues>,
    node: usize,
    weight: Arc<ModelWeight>,
    input: ResidentLease,
    output: ResidentLease,
}
impl Canonical47ProjectionBinding {
    pub(crate) fn weight(&self) -> &ResidentLease {
        self.weight.lease()
    }
    /// Selected call0 QKV packet scope only; no current-attempt/cursor or Ready authority.
    pub(crate) fn require_initial47_attention_qkv(&self) -> Result<()> {
        let node = &self.scope.record.plan().nodes()[self.node];
        let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
            return Err(Error::State("canonical47 actual attention QKV node".into()));
        };
        self.require(&self.values, descriptor, &self.input, &self.output)?;
        self.weight.require_initialized()?;
        let layer = descriptor
            .layer()
            .ok_or_else(|| Error::State("canonical47 actual QKV layer absent".into()))?;
        super::canonical47_attention_projection::require_site(
            descriptor.call().request(),
            layer.index(),
            self.node,
            &node.inputs,
            &node.outputs,
        )?;
        if self.scope.kind != Canonical47CallKind::Prefill
            || descriptor.kind() != Vllm08Kind::Projection(Vllm08Projection::AttentionQkv)
            || descriptor.call().model_profile() != FullModelProfile::Qwen08
            || descriptor.call() != self.scope.record.call()
            || descriptor.matrices().len() != 3
            || !descriptor.parameters().is_empty()
            || descriptor.input_elements().as_slice() != [47 * 1024]
            || descriptor.output_elements().as_slice() != [47 * 5120]
            || node.state.is_some()
            || self.input.bytes() != 96256
            || self.output.bytes() != 481280
            || self.weight.lease().bytes() != 10485760
        {
            return Err(Error::State(
                "canonical47 exact layer3 QKV source/shape/weight".into(),
            ));
        }
        let normal = &self.scope.record.plan().nodes()[31];
        let PlanOperation::TypedVllm08 {
            descriptor: producer,
            ..
        } = &normal.operation
        else {
            return Err(Error::State(
                "canonical47 QKV actual dual producer absent".into(),
            ));
        };
        if producer.call() != descriptor.call()
            || producer.layer() != Some(layer)
            || producer.kind() != Vllm08Kind::ResidualRms(Vllm08RmsSite::Input)
            || normal.inputs.as_slice() != [ValueId(38), ValueId(35)]
            || normal.outputs.as_slice() != [ValueId(39), ValueId(40)]
            || normal.state.is_some()
        {
            return Err(Error::State(
                "canonical47 QKV same original computed dual input".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require(
        &self,
        values: &Arc<ModelValues>,
        operator: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<()> {
        self.scope.require_values(values)?;
        let PlanOperation::TypedVllm08 { descriptor, .. } =
            &self.scope.record.plan().nodes()[self.node].operation
        else {
            return Err(Error::State(
                "canonical47 original projection node changed".into(),
            ));
        };
        self.weight
            .require_operator(self.scope.source.source(), operator)?;
        if !Arc::ptr_eq(values, &self.values)
            || descriptor.as_ref() != operator
            || input.id() != self.input.id()
            || output.id() != self.output.id()
            || !Arc::ptr_eq(input.service(), self.values.service())
            || !Arc::ptr_eq(output.service(), self.values.service())
        {
            return Err(Error::State(
                "canonical47 sealed actual projection source/SSA/lease binding changed".into(),
            ));
        }
        Ok(())
    }
}
impl Canonical47GdnSite {
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
pub(super) fn validate_source_members(
    source: &BoundFullModelSource,
    op: &Vllm08Operator,
) -> Result<()> {
    use QwenMatrixRole as M;
    use QwenParameterRole as P;
    use Vllm08Kind as K;
    use Vllm08Projection as X;
    let layer = op.layer().map(|l| l.index());
    let i = layer.unwrap_or(0);
    let matrices = match op.kind() {
        K::Projection(X::Qkvz) => vec![M::GdnQkv(i), M::GdnGate(i)],
        K::Projection(X::Ba) => vec![M::GdnBeta(i), M::GdnAlpha(i)],
        K::Projection(X::GdnOut) => vec![M::GdnOutput(i)],
        K::Projection(X::AttentionQkv) => vec![M::Query(i), M::Key(i), M::Value(i)],
        K::Projection(X::AttentionOut) => vec![M::AttentionOutput(i)],
        K::Projection(X::GateUp) => vec![M::FfnGate(i), M::FfnUp(i)],
        K::Projection(X::Down) => vec![M::FfnDown(i)],
        K::Projection(X::Head) => vec![M::Output],
        _ => Vec::new(),
    };
    let parameters = match op.kind() {
        K::InputRms | K::ResidualRms(Vllm08RmsSite::Input) => vec![P::InputNorm(i)],
        K::ResidualRms(Vllm08RmsSite::PostAttention) => vec![P::FfnNorm(i)],
        K::ResidualRms(Vllm08RmsSite::Final) => vec![P::FinalNorm],
        K::GdnConv => vec![P::GdnConvolution(i)],
        K::GdnRecurrent => vec![P::GdnA(i), P::GdnDt(i), P::GdnNorm(i)],
        K::AttentionQk => vec![P::QueryNorm(i), P::KeyNorm(i)],
        _ => Vec::new(),
    };
    if matrices.len() != op.matrices().len() || parameters.len() != op.parameters().len() {
        return Err(Error::State(
            "canonical47 exact ordered original source-member roles".into(),
        ));
    }
    for (role, descriptor) in matrices.into_iter().zip(op.matrices()) {
        source.source().validate_matrix(role, descriptor)?;
    }
    for (role, descriptor) in parameters.into_iter().zip(op.parameters()) {
        source.source().validate_parameter(role, descriptor)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
