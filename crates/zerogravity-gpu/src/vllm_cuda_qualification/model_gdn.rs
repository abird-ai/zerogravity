//! Full-layer source/producer binding for the SAME original GDN instructions.
use super::model_cache::ModelCache;
use super::model_values::ModelValues;
use super::{finite_bf16, loaded_gamma};
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState,
};
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentReadback, ResidentService,
};
use crate::vllm_cuda_gdn::{
    GdnArithmetic, GdnBuffer, GdnCacheSlot, GdnInputs, GdnPrepared, GdnTypedOutputs,
    ModelGdnAdapter, ModelGdnConvEnqueued, VllmGdnPhase,
};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{ParameterWindow, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    PlanOperation, SourceParameter, Vllm08Kind, Vllm08Layer, Vllm08Operator, VllmBf16Phase,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

/// Immutable learned source payload only, shared by genuine request phases.
struct ParameterPayload {
    source: Arc<OriginalWindowModelSource>,
    layer: Vllm08Layer,
    short_hf_profile: Option<zerogravity_core::accelerator::program::FullModelProfile>,
    descriptors: [SourceParameter; 4],
    windows: Vec<ParameterWindow>,
    bytes: [Vec<u8>; 4],
    _host: WindowWorkspace,
    _metadata: WindowWorkspace,
}
pub(crate) struct ModelGdnParameters {
    payload: Arc<ParameterPayload>,
    leases: [ResidentLease; 4],
}
impl ModelGdnParameters {
    pub(crate) fn prepare(
        source: Arc<OriginalWindowModelSource>,
        service: &Arc<ResidentService>,
        conv: &Vllm08Operator,
        recurrent: &Vllm08Operator,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        source.require_vllm08_full_source()?;
        Self::prepare_source(source, service, conv, recurrent, init, None, cancelled)
    }
    fn prepare_source(
        source: Arc<OriginalWindowModelSource>,
        service: &Arc<ResidentService>,
        conv: &Vllm08Operator,
        recurrent: &Vllm08Operator,
        init: &mut ResidentInitialization<'_>,
        short_hf_profile: Option<zerogravity_core::accelerator::program::FullModelProfile>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::projection_cubin::not_cancelled(cancelled)?;
        init.require_service(service)?;
        if service.source_identity() != source.source_identity() {
            return Err(Error::State(
                "full24 original GDN source/service before upload".into(),
            ));
        }
        conv.validate()?;
        recurrent.validate()?;
        let layer = conv
            .layer()
            .ok_or_else(|| Error::State("full24 GDN source layer absent".into()))?;
        if layer.is_attention()
            || conv.kind() != Vllm08Kind::GdnConv
            || recurrent.kind() != Vllm08Kind::GdnRecurrent
            || conv.call() != recurrent.call()
            || recurrent.layer() != Some(layer)
            || conv.call().source() != source.source_identity()
        {
            return Err(Error::State(
                "full24 actual paired GDN source descriptors".into(),
            ));
        }
        let metadata = source.source().reserve_workspace(64 * 1024)?;
        let host = source.source().reserve_workspace(256 * 1024)?;
        let descriptors = [
            conv.parameters()[0].clone(),
            recurrent.parameters()[0].clone(),
            recurrent.parameters()[1].clone(),
            recurrent.parameters()[2].clone(),
        ];
        let allocation = parameter_allocation_layout(&descriptors)?;
        let i = layer.index();
        let roles = zerogravity_model_qwen35::hybrid_decoder::ParameterGroup::GdnAll.roles(i);
        let mut windows = Vec::with_capacity(4);
        let mut bytes: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
        for (index, (descriptor, role)) in descriptors.iter().zip(roles).enumerate() {
            source.validate_parameter(role, descriptor)?;
            let window = source
                .source()
                .read_declared_parameter(descriptor, cancelled)?;
            bytes[index] = if index == 3 {
                loaded_gamma(&window)?
            } else {
                window.as_slice().to_vec()
            };
            if bytes[index].len() != allocation[index] {
                return Err(Error::State(
                    "original GDN source/loaded device extent differs".into(),
                ));
            }
            if !(if index == 1 {
                super::layer_common::finite_f32(&bytes[index])
            } else {
                finite_bf16(&bytes[index])
            }) {
                return Err(Error::Malformed(
                    "full24 finite loaded original GDN parameter".into(),
                ));
            }
            windows.push(window);
        }
        let payload = Arc::new(ParameterPayload {
            source,
            layer,
            short_hf_profile,
            descriptors,
            windows,
            bytes,
            _host: host,
            _metadata: metadata,
        });
        // Retain original windows, converted payloads and host charges BEFORE
        // the first real write, including any later partial-upload failure.
        init.retain_completion_owner(payload.clone())?;
        let leases = payload
            .bytes
            .iter()
            .map(|raw| init.upload_bytes(raw, BufferRole::ImmutableWeights))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("full24 fixed four source GDN parameter leases".into()))?;
        let parameters = Arc::new(Self { payload, leases });
        init.retain_completion_owner(parameters.clone())?;
        Ok(parameters)
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.leases
            .iter()
            .cloned()
            .zip(self.payload.bytes.iter().map(Vec::as_slice))
            .collect()
    }
    fn require(&self, conv: &Vllm08Operator, recurrent: &Vllm08Operator) -> Result<()> {
        if let Some(profile) = self.payload.short_hf_profile {
            if conv.call().model_profile() != profile
                || conv.call().request().teachers().short_hf_span().is_none()
                || recurrent
                    .call()
                    .request()
                    .teachers()
                    .short_hf_span()
                    .is_none()
            {
                return Err(Error::State(
                    "short-HF GDN genuine source/profile/call issuer".into(),
                ));
            }
        } else {
            self.payload.source.require_vllm08_full_source()?;
        }
        conv.validate()?;
        recurrent.validate()?;
        if conv.layer() != Some(self.payload.layer)
            || recurrent.layer() != Some(self.payload.layer)
            || conv.kind() != Vllm08Kind::GdnConv
            || recurrent.kind() != Vllm08Kind::GdnRecurrent
            || conv.call() != recurrent.call()
            || conv.call().source() != self.payload.source.source_identity()
            || conv.parameters() != &self.payload.descriptors[..1]
            || recurrent.parameters() != &self.payload.descriptors[1..]
        {
            return Err(Error::State(
                "full24 exact shared GDN source-role parameters".into(),
            ));
        }
        for ((lease, window), descriptor) in self
            .leases
            .iter()
            .zip(&self.payload.windows)
            .zip(&self.payload.descriptors)
        {
            if !lease.is_initialized()
                || lease.source_identity() != self.payload.source.source_identity()
                || window.parameter().source_parameter() != *descriptor
            {
                return Err(Error::State(
                    "full24 original GDN parameter initialization/owner".into(),
                ));
            }
        }
        Ok(())
    }
}
pub(crate) struct ModelGdnBinding {
    arithmetic: Arc<GdnArithmetic>,
    values: Arc<ModelValues>,
    parameters: Arc<ModelGdnParameters>,
    nodes: [usize; 2],
    operands: [ResidentLease; 5],
}
impl ModelGdnBinding {
    pub(crate) fn source(&self) -> &Arc<OriginalWindowModelSource> {
        &self.parameters.payload.source
    }
    pub(crate) fn layer(&self) -> Vllm08Layer {
        self.parameters.payload.layer
    }
    pub(crate) fn call(&self) -> &zerogravity_core::accelerator::program::Vllm08Call {
        self.values.record().call()
    }
    pub(crate) fn require_prepared(
        &self,
        prepared: &GdnPrepared,
        domains: &BoundModelState,
    ) -> Result<()> {
        self.arithmetic.require_same(prepared.arithmetic())?;
        if domains.call() != self.call() || domains.layer() != self.layer() {
            return Err(Error::State(
                "full24 exact source layer/request/domain binding".into(),
            ));
        }
        for (role, lease) in [
            GdnBuffer::PackedQkvz,
            GdnBuffer::PackedBa,
            GdnBuffer::ConvOutput,
            GdnBuffer::ZCompact,
            GdnBuffer::Gated,
        ]
        .into_iter()
        .zip(&self.operands)
        .chain(
            [
                GdnBuffer::ConvWeights,
                GdnBuffer::ALog,
                GdnBuffer::DtBias,
                GdnBuffer::GatedGamma,
            ]
            .into_iter()
            .zip(&self.parameters.leases),
        ) {
            if prepared.lease(role)?.id() != lease.id()
                || !Arc::ptr_eq(prepared.lease(role)?.service(), self.values.service())
            {
                return Err(Error::State(
                    "full24 exact original GDN producer/parameter lease seal".into(),
                ));
            }
        }
        Ok(())
    }
}
#[derive(Clone)]
struct RetainedArena {
    _legacy: Option<Arc<ModelCache>>,
    _shared: Option<Arc<super::model_cache::SharedModelCache>>,
}
pub(crate) struct ModelGdnPrepared {
    binding: Arc<ModelGdnBinding>,
    prepared: Arc<GdnPrepared>,
    adapter: Arc<ModelGdnAdapter>,
    domains: Arc<BoundModelState>,
    _arena: RetainedArena,
    pending: Mutex<Option<ModelGdnConvEnqueued>>,
    _metadata: WindowWorkspace,
    canonical47: Option<Arc<super::canonical47_model::Canonical47ModelBinding>>,
    short_hf: Option<Arc<super::short_hf_model::ShortHfModelBinding>>,
}
impl ModelGdnPrepared {
    pub(crate) async fn prepare(
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        domains: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        nodes: [usize; 2],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_impl(values, parameters, domains, arena, nodes, None, cancelled).await
    }
    pub(crate) async fn prepare_canonical47_prefix(
        scope: Arc<super::canonical47_model::Canonical47ModelBinding>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        domains: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        declared: crate::vllm_cuda_gdn::Canonical47GdnArithmetic,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::projection_cubin::not_cancelled(cancelled)?;
        scope.require_values(&values)?;
        scope.kind().require_prefill_gdn(declared)?;
        domains.require_canonical47_model_prefix(&scope)?;
        let nodes = scope.gdn_site(domains.layer().index())?.nodes();
        let result = Self::prepare_impl(
            values,
            parameters,
            domains,
            arena,
            nodes,
            Some(scope.clone()),
            cancelled,
        )
        .await?;
        result.require_canonical47_binding(&scope, declared)?;
        Ok(result)
    }
    async fn prepare_impl(
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        domains: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        nodes: [usize; 2],
        canonical47: Option<Arc<super::canonical47_model::Canonical47ModelBinding>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::projection_cubin::not_cancelled(cancelled)?;
        arena.require_binding(&domains)?;
        let cache = arena.lease().clone();
        let record = values.record();
        let conv = record
            .plan()
            .nodes()
            .get(nodes[0])
            .ok_or_else(|| Error::State("full24 Conv node absent".into()))?;
        let recurrent = record
            .plan()
            .nodes()
            .get(nodes[1])
            .ok_or_else(|| Error::State("full24 recurrent node absent".into()))?;
        let (
            PlanOperation::TypedVllm08 { descriptor: cd, .. },
            PlanOperation::TypedVllm08 { descriptor: rd, .. },
        ) = (&conv.operation, &recurrent.operation)
        else {
            return Err(Error::State("full24 actual typed GDN nodes".into()));
        };
        parameters.require(cd, rd)?;
        let ([qkvz, ba], [activated, z], [ra, rz, rb], [gated]) = (
            conv.inputs.as_slice(),
            conv.outputs.as_slice(),
            recurrent.inputs.as_slice(),
            recurrent.outputs.as_slice(),
        ) else {
            return Err(Error::State(
                "full24 exact GDN producer/output arity".into(),
            ));
        };
        if nodes[0] >= nodes[1]
            || activated != ra
            || z != rz
            || ba != rb
            || cd.call() != record.call()
            || rd.call() != record.call()
        {
            return Err(Error::State(
                "full24 actual same-call Conv/Z/BA SSA cohort".into(),
            ));
        }
        domains.require_effect(
            conv.state
                .ok_or_else(|| Error::State("full24 Conv effect absent".into()))?,
            false,
        )?;
        domains.require_effect(
            recurrent
                .state
                .ok_or_else(|| Error::State("full24 recurrent effect absent".into()))?,
            true,
        )?;
        let operands: [ResidentLease; 5] = [*qkvz, *ba, *activated, *z, *gated]
            .into_iter()
            .map(|id| values.lease(id).cloned())
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("full24 exact five GDN values".into()))?;
        let phase = if let Some(scope) = &canonical47 {
            scope.require_values(&values)?;
            domains.require_canonical47_gdn_scope(scope)?;
            scope.kind().require_prefill_gdn(
                crate::vllm_cuda_gdn::Canonical47GdnArithmetic::OwnedCanonical47Solve4,
            )?;
            VllmGdnPhase::Initial47
        } else {
            match record.call().request().legacy_phase()? {
                VllmBf16Phase::InitialOne => VllmGdnPhase::InitialOne,
                VllmBf16Phase::Initial64 => VllmGdnPhase::Initial64,
                VllmBf16Phase::ContinuedTwo => VllmGdnPhase::ContinuedTwo,
                VllmBf16Phase::CachedOne => VllmGdnPhase::CachedDecode,
            }
        };
        let slot = if matches!(phase, VllmGdnPhase::InitialOne | VllmGdnPhase::Initial47)
            || phase == VllmGdnPhase::CachedDecode && record.call().request().teachers().base() == 1
        {
            GdnCacheSlot::One
        } else {
            GdnCacheSlot::Four
        };
        let inputs = GdnInputs {
            qkvz: operands[0].clone(),
            ba: operands[1].clone(),
            cache: cache.clone(),
            conv_weights: parameters.leases[0].clone(),
            a_log: parameters.leases[1].clone(),
            dt_bias: parameters.leases[2].clone(),
            gamma: parameters.leases[3].clone(),
        };
        let outputs = GdnTypedOutputs {
            activated: operands[2].clone(),
            z: operands[3].clone(),
            gated: operands[4].clone(),
        };
        let metadata = parameters
            .payload
            .source
            .source()
            .reserve_workspace(crate::vllm_cuda_gdn::model_metadata_bound(phase)?)?;
        let prepared = GdnPrepared::prepare_attempt_outputs(
            values.service().clone(),
            phase,
            slot,
            inputs,
            outputs,
            cancelled,
        )
        .await?;
        let binding = Arc::new(ModelGdnBinding {
            arithmetic: prepared.arithmetic().clone(),
            values,
            parameters,
            nodes,
            operands,
        });
        binding.require_prepared(&prepared, &domains)?;
        let adapter = ModelGdnAdapter::prepare(
            binding.clone(),
            domains.clone(),
            prepared.clone(),
            cache,
            phase,
            slot,
            cancelled,
        )
        .await?;
        Ok(Arc::new(Self {
            binding,
            prepared,
            adapter,
            domains,
            _arena: RetainedArena {
                _legacy: Some(arena),
                _shared: None,
            },
            pending: Mutex::new(None),
            _metadata: metadata,
            canonical47,
            short_hf: None,
        }))
    }
    pub(crate) fn require_value_state(
        &self,
        values: &Arc<ModelValues>,
        domains: &Arc<BoundModelState>,
    ) -> Result<()> {
        if !Arc::ptr_eq(values, &self.binding.values) || !Arc::ptr_eq(domains, &self.domains) {
            return Err(Error::State(
                "full24 GDN body differs from original sealed value/state map".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn arithmetic(&self) -> &Arc<GdnArithmetic> {
        &self.binding.arithmetic
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut result = self.prepared.accesses();
        for (lease, access) in self.adapter.accesses() {
            match result
                .iter_mut()
                .find(|(other, _)| other.id() == lease.id())
            {
                Some((_, old)) if access == DeviceAccess::Write => *old = access,
                Some(_) => {}
                None => result.push((lease, access)),
            }
        }
        result
    }
    pub(crate) fn status_readback(&self) -> Result<ResidentReadback> {
        self.adapter.status_readback()
    }
    pub(crate) fn enqueue(
        &self,
        node: usize,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        super::projection_cubin::not_cancelled(cancelled)?;
        self.domains
            .require_enqueue(self.binding.values.service(), permit)?;
        if self.short_hf.is_some() {
            // SAME active registered original flight/node protects shared private GDN storage.
            self.binding.values.require_storage_node(node)?;
        }
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| Error::State("full24 GDN split guard poisoned".into()))?;
        if node == self.binding.nodes[0] {
            if pending.is_some() {
                return Err(Error::State("full24 GDN Conv already enqueued".into()));
            }
            let token = self.prepared.launch_model_convolution(
                self.adapter.clone(),
                self.domains.effect(false)?,
                permit,
                cancelled,
            )?;
            let count = token.convolution_dispatches();
            *pending = Some(token);
            Ok(count)
        } else if node == self.binding.nodes[1] {
            pending
                .take()
                .ok_or_else(|| {
                    Error::State("full24 recurrence lacks actual same-flight Conv".into())
                })?
                .launch_recurrent(self.domains.effect(true)?, permit, cancelled)
        } else {
            Err(Error::State("not a full24 bound GDN node".into()))
        }
    }
}

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod natural;

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod canonical47;

mod whole;

mod canonical_natural;
pub(crate) use canonical_natural::ColdCanonicalGdn;

mod short_hf;
pub(crate) use short_hf::ColdShortHfGdn;

fn parameter_allocation_layout(
    parameters: &[zerogravity_core::accelerator::program::SourceParameter; 4],
) -> Result<[usize; 4]> {
    let p = &parameters[3];
    let elements = p.shape.iter().try_fold(1usize, |n, &v| {
        usize::try_from(v)
            .ok()
            .and_then(|v| n.checked_mul(v))
            .ok_or_else(|| Error::Limit("source GDN gamma shape extent".into()))
    })?;
    let gamma = elements
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("loaded GDN gamma extent".into()))?;
    Ok([
        parameters[0].bytes,
        parameters[1].bytes,
        parameters[2].bytes,
        gamma,
    ])
}
