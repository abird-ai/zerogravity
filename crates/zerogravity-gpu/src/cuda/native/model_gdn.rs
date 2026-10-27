//! Full-layer source/producer binding for the SAME original GDN instructions.
use super::model_cache::ModelCache;
use super::model_values::ModelValues;
use super::{finite_bf16, loaded_gamma};
use crate::cuda::gdn::{
    GdnArithmetic, GdnBuffer, GdnCacheSlot, GdnFamilyIdentity, GdnInputs, GdnPhase, GdnPrepared,
    GdnTypedOutputs, ModelGdnAdapter, ModelGdnConvEnqueued,
};
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState,
};
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentReadback, ResidentService,
};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{
    ParameterWindow, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::accelerator::program::{
    ConvDomain, RecurrentDomain, SourceCall, SourceConvGeometry, SourceGdnContract,
    SourceGdnGeometry, SourceNumericalContract, SourceOperator, SourceOperatorKind,
    SourceParameter,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

fn require_native_gdn_pair(
    conv: &SourceOperator,
    recurrent: &SourceOperator,
) -> Result<(ConvDomain, RecurrentDomain)> {
    conv.validate()?;
    recurrent.validate()?;
    let (
        SourceOperatorKind::CausalConvSplit {
            domain: cd,
            geometry: cg,
        },
        SourceOperatorKind::GatedDelta {
            domain: rd,
            geometry: rg,
        },
    ) = (conv.kind(), recurrent.kind())
    else {
        return Err(Error::State(
            "full24 actual paired GDN source descriptors".into(),
        ));
    };
    if conv.call() != recurrent.call()
        || conv.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
        || recurrent.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
        || cg
            != (SourceConvGeometry {
                qkv_channels: 6144,
                z_channels: 2048,
                coefficient_channels: 32,
                kernel: 4,
            })
        || rg
            != (SourceGdnGeometry {
                key_heads: 16,
                value_heads: 16,
                key_width: 128,
                value_width: 128,
                epsilon: 1e-6,
                contract: SourceGdnContract::ReferenceCudaSm86L2SiluBf16V1,
            })
        || conv.parameters().len() != 1
        || recurrent.parameters().len() != 3
    {
        return Err(Error::Unsupported(
            "exact native GDN mathematical/source ABI".into(),
        ));
    }
    Ok((cd, rd))
}

/// Immutable learned source payload only, shared by genuine request phases.
struct ParameterPayload {
    source: VerifiedTensorSource,
    site: usize,
    text_implementation: Option<GdnFamilyIdentity>,
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
        source: VerifiedTensorSource,
        service: &Arc<ResidentService>,
        conv: &SourceOperator,
        recurrent: &SourceOperator,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_source(source, service, conv, recurrent, init, None, cancelled)
    }
    fn prepare_source(
        source: VerifiedTensorSource,
        service: &Arc<ResidentService>,
        conv: &SourceOperator,
        recurrent: &SourceOperator,
        init: &mut ResidentInitialization<'_>,
        text_implementation: Option<GdnFamilyIdentity>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::projection_cubin::not_cancelled(cancelled)?;
        init.require_service(service)?;
        if service.source_identity() != source.source_identity() {
            return Err(Error::State(
                "full24 original GDN source/service before upload".into(),
            ));
        }
        let (conv_domain, recurrent_domain) = require_native_gdn_pair(conv, recurrent)?;
        let site = conv_domain.0;
        if recurrent_domain.0 != site || conv.call().source() != source.source_identity() {
            return Err(Error::State(
                "full24 actual paired GDN source descriptors".into(),
            ));
        }
        let metadata = source.reserve_workspace(64 * 1024)?;
        let host = source.reserve_workspace(256 * 1024)?;
        let descriptors = [
            conv.parameters()[0].clone(),
            recurrent.parameters()[0].clone(),
            recurrent.parameters()[1].clone(),
            recurrent.parameters()[2].clone(),
        ];
        let allocation = parameter_allocation_layout(&descriptors)?;
        let mut windows = Vec::with_capacity(4);
        let mut bytes: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
        for (index, descriptor) in descriptors.iter().enumerate() {
            let window = source.read_declared_parameter(descriptor, cancelled)?;
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
            site,
            text_implementation,
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
    fn require(&self, conv: &SourceOperator, recurrent: &SourceOperator) -> Result<()> {
        let (conv_domain, recurrent_domain) = require_native_gdn_pair(conv, recurrent)?;
        if conv_domain.0 != self.payload.site
            || recurrent_domain.0 != self.payload.site
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
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.parameters.payload.source
    }
    pub(crate) fn layer(&self) -> usize {
        self.parameters.payload.site
    }
    pub(crate) fn call(&self) -> &SourceCall {
        self.values.record().call()
    }
    pub(crate) fn require_prepared(
        &self,
        prepared: &GdnPrepared,
        domains: &BoundModelState,
    ) -> Result<()> {
        self.arithmetic.require_same(prepared.arithmetic())?;
        if domains.call() != self.call()
            || domains.layer() != self.layer()
            || domains.conv_domain()? != ConvDomain(self.layer())
            || domains.recurrent_domain()? != RecurrentDomain(self.layer())
        {
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
    fixed_prompt: Option<Arc<super::fixed_prompt_model::FixedPromptModelBinding>>,
    text: Option<Arc<super::text_model::TextModelBinding>>,
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
    pub(crate) async fn prepare_fixed_prompt_prefix(
        scope: Arc<super::fixed_prompt_model::FixedPromptModelBinding>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        domains: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        declared: crate::cuda::gdn::FixedPromptGdnArithmetic,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::projection_cubin::not_cancelled(cancelled)?;
        scope.require_values(&values)?;
        scope.kind().require_prefill_gdn(declared)?;
        domains.require_fixed_prompt_model_prefix(&scope)?;
        let nodes = scope.gdn_site(domains.layer())?.nodes();
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
        result.require_fixed_prompt_binding(&scope, declared)?;
        Ok(result)
    }
    async fn prepare_impl(
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        domains: Arc<BoundModelState>,
        arena: Arc<ModelCache>,
        nodes: [usize; 2],
        fixed_prompt: Option<Arc<super::fixed_prompt_model::FixedPromptModelBinding>>,
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
        let (Some(cd), Some(rd)) = (&record.operator(nodes[0])?, &record.operator(nodes[1])?)
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
        let phase = if let Some(scope) = &fixed_prompt {
            scope.require_values(&values)?;
            domains.require_fixed_prompt_gdn_scope(scope)?;
            scope.kind().require_prefill_gdn(
                crate::cuda::gdn::FixedPromptGdnArithmetic::OwnedFixedPromptSolve4,
            )?;
            GdnPhase::FixedPrompt
        } else {
            GdnPhase::for_reference_call(record.call())?
        };
        let slot = if matches!(phase, GdnPhase::InitialOne | GdnPhase::FixedPrompt)
            || phase == GdnPhase::CachedDecode && record.call().rows().base() == 1
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
            .reserve_workspace(crate::cuda::gdn::model_metadata_bound(phase)?)?;
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
            fixed_prompt,
            text: None,
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
        if self.text.is_some() {
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

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod natural;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod fixed_prompt;

mod whole;

mod fixed_prompt_natural;
pub(crate) use fixed_prompt_natural::ColdFixedPromptGdn;

mod text;
pub(crate) use text::ColdTextGdn;

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
