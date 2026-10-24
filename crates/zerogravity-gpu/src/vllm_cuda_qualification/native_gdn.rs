//! Sealed original-source parameter and real plan producer binding for GDN.
use super::*;
use crate::resident_inference::gdn_state::BoundGdnDomains;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_program::{DeviceBuffer, ResidentBufferLayout, ResidentPhysicalType};
use crate::vllm_cuda_gdn::{
    GdnCacheSlot, GdnConvEnqueued, GdnInputs, GdnPrepared, GdnStateAdapter, GdnTypedOutputs,
};
use crate::vllm_cuda_input_rms::BoundRmsSource;
use std::sync::{Mutex, atomic::AtomicBool};
use zerogravity_artifacts::model::qwen_owned::QwenOriginalWindowText;
use zerogravity_core::accelerator::program::{
    ForwardPlan, GdnPrefixRmsSite, PlanOperation, SourceParameter, ValueId, VllmBf16Invocation,
    VllmBf16Phase, VllmGdnConv, VllmGdnRecurrent, VllmValueOrigin,
};

struct ParameterPayload {
    windows: Vec<ParameterWindow>,
    bytes: [Vec<u8>; 4],
    cache_zero: Vec<u8>,
    source: Arc<QwenOriginalWindowText>,
    // Actual raw/converted host payloads precede their SAME source admissions.
    _host: WindowWorkspace,
    _metadata: WindowWorkspace,
}
/// No public constructor: only this actual source/window/upload preparation.
pub(crate) struct NativeGdnBinding {
    payload: Arc<ParameterPayload>,
    source: BoundRmsSource,
    invocation: VllmBf16Invocation,
    nodes: [usize; 2],
    value_ids: [ValueId; 5],
    values: [ResidentLease; 5],
    parameters: [ResidentLease; 4],
    descriptors: [SourceParameter; 4],
}
impl NativeGdnBinding {
    pub(crate) fn source_binding(&self) -> &BoundRmsSource {
        &self.source
    }
    pub(crate) fn invocation(&self) -> &VllmBf16Invocation {
        &self.invocation
    }
    pub(crate) fn require_prepared(
        &self,
        prepared: &GdnPrepared,
        domains: &BoundGdnDomains,
    ) -> Result<()> {
        self.source.require_source(self.payload.source.source())?;
        if domains.invocation() != &self.invocation || domains.layer() != self.source.key().layer()
        {
            return Err(Error::State(
                "GDN same actual invocation/layer/root parameter owner".into(),
            ));
        }
        for ((window, descriptor), (role, lease)) in
            self.payload.windows.iter().zip(&self.descriptors).zip(
                [
                    GdnBuffer::ConvWeights,
                    GdnBuffer::ALog,
                    GdnBuffer::DtBias,
                    GdnBuffer::GatedGamma,
                ]
                .into_iter()
                .zip(&self.parameters),
            )
        {
            if !window.is_complete()
                || window.parameter().source_parameter() != *descriptor
                || prepared.lease(role)?.id() != lease.id()
            {
                return Err(Error::State(
                    "GDN exact original role/window/uploaded lease seal".into(),
                ));
            }
        }
        for (role, lease) in [
            GdnBuffer::PackedQkvz,
            GdnBuffer::PackedBa,
            GdnBuffer::ConvOutput,
            GdnBuffer::ZCompact,
            GdnBuffer::Gated,
        ]
        .into_iter()
        .zip(&self.values)
        {
            if prepared.lease(role)?.id() != lease.id()
                || !Arc::ptr_eq(prepared.lease(role)?.service(), lease.service())
            {
                return Err(Error::State(
                    "GDN exact original plan producer/output lease seal".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn require_record(&self, plan: &ForwardPlan) -> Result<()> {
        let conv = plan
            .nodes()
            .get(self.nodes[0])
            .ok_or_else(|| Error::State("GDN original Conv record absent".into()))?;
        let recur = plan
            .nodes()
            .get(self.nodes[1])
            .ok_or_else(|| Error::State("GDN original recurrent record absent".into()))?;
        let (
            PlanOperation::TypedVllmGdnConv { descriptor: cd },
            PlanOperation::TypedVllmGdnRecurrent { descriptor: rd },
        ) = (&conv.operation, &recur.operation)
        else {
            return Err(Error::State("GDN exact actual typed records".into()));
        };
        if cd.invocation() != &self.invocation
            || rd.invocation() != &self.invocation
            || cd.parameter() != &self.descriptors[0]
            || rd.parameters() != &self.descriptors[1..]
            || conv.inputs != self.value_ids[..2]
            || conv.outputs != self.value_ids[2..4]
            || recur.inputs != [self.value_ids[2], self.value_ids[3], self.value_ids[1]]
            || recur.outputs != [self.value_ids[4]]
        {
            return Err(Error::State(
                "GDN source parameter and producer ValueID cohort changed".into(),
            ));
        }
        Ok(())
    }
}
pub(crate) struct NativeGdnContext {
    pub service: Arc<ResidentService>,
    pub source: Arc<QwenOriginalWindowText>,
    pub domains: Arc<BoundGdnDomains>,
}
pub(crate) struct NativeGdnPrepared {
    binding: Arc<NativeGdnBinding>,
    prepared: Arc<GdnPrepared>,
    adapter: Arc<GdnStateAdapter>,
    pending: Mutex<Option<GdnConvEnqueued>>,
}
fn operands(
    plan: &ForwardPlan,
    nodes: [usize; 2],
) -> Result<(VllmGdnConv, VllmGdnRecurrent, [ValueId; 5])> {
    let conv = plan
        .nodes()
        .get(nodes[0])
        .ok_or_else(|| Error::State("typed Conv absent".into()))?;
    let recurrent = plan
        .nodes()
        .get(nodes[1])
        .ok_or_else(|| Error::State("typed recurrent absent".into()))?;
    let (
        PlanOperation::TypedVllmGdnConv {
            descriptor: conv_desc,
        },
        PlanOperation::TypedVllmGdnRecurrent {
            descriptor: recurrent_desc,
        },
    ) = (&conv.operation, &recurrent.operation)
    else {
        return Err(Error::State("typed GDN producer records".into()));
    };
    conv_desc.validate()?;
    recurrent_desc.validate()?;
    let ([qkvz, ba], [activated, z], [r_activated, r_z, r_ba], [gated]) = (
        conv.inputs.as_slice(),
        conv.outputs.as_slice(),
        recurrent.inputs.as_slice(),
        recurrent.outputs.as_slice(),
    ) else {
        return Err(Error::State("typed GDN exact arity".into()));
    };
    let layer = conv_desc.invocation().layer();
    if conv_desc.invocation() != recurrent_desc.invocation()
        || activated != r_activated
        || z != r_z
        || ba != r_ba
        || nodes[0] >= nodes[1]
        || plan.vllm_value_provenance(*activated)?.map(|p| p.origin())
            != Some(&VllmValueOrigin::ConvActivated {
                layer,
                node_index: nodes[0],
                qkvz: *qkvz,
                ba: *ba,
            })
        || plan.vllm_value_provenance(*z)?.map(|p| p.origin())
            != Some(&VllmValueOrigin::ConvZ {
                layer,
                node_index: nodes[0],
                qkvz: *qkvz,
                ba: *ba,
            })
    {
        return Err(Error::State(
            "typed GDN exact layer/Conv output cohort".into(),
        ));
    }
    for id in [*qkvz, *ba, *activated, *z, *gated] {
        plan.vllm_value_provenance(id)?
            .ok_or_else(|| Error::State("typed GDN provenance absent".into()))?
            .require_invocation(conv_desc.invocation())?;
    }
    Ok((
        *conv_desc.clone(),
        *recurrent_desc.clone(),
        [*qkvz, *ba, *activated, *z, *gated],
    ))
}
impl NativeGdnPrepared {
    /// Guard-only preparation. Caller owns the SAME original preparation slot.
    pub(crate) async fn prepare(
        context: NativeGdnContext,
        plan: &ForwardPlan,
        values: &[DeviceBuffer],
        nodes: [usize; 2],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::projection_cubin::not_cancelled(cancelled)?;
        let (conv, recurrent, ids) = operands(plan, nodes)?;
        let invocation = conv.invocation().clone();
        invocation.legacy_phase()?;
        if context.domains.invocation() != &invocation
            || !plan.vllm_gdn_fragment().is_some_and(|base| {
                base.teachers() == invocation.teachers()
                    && base.mode() == invocation.mode()
                    && base.layer().source() == invocation.layer().source()
            })
            || (!context.domains.is_attempt_local()
                && !matches!(
                    invocation.legacy_phase()?,
                    VllmBf16Phase::InitialOne | VllmBf16Phase::Initial64
                ))
            || invocation.layer().source() != context.source.source_identity()
            || context.service.source_identity() != context.source.source_identity()
        {
            return Err(Error::Unsupported(
                "initial genuine original source/root GDN fragment only".into(),
            ));
        }
        let metadata = context.source.source().reserve_workspace(2 * 1024 * 1024)?;
        let host = context.source.source().reserve_workspace(8 * 1024 * 1024)?;
        let descriptors = [
            conv.parameter().clone(),
            recurrent.parameters()[0].clone(),
            recurrent.parameters()[1].clone(),
            recurrent.parameters()[2].clone(),
        ];
        let mut windows = Vec::with_capacity(4);
        let mut bytes: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
        for (index, descriptor) in descriptors.iter().enumerate() {
            context.source.text_tensor(&descriptor.name)?;
            let p = if index == 0 {
                context.source.source().conv4_parameter(&descriptor.name)?
            } else {
                context.source.source().parameter(&descriptor.name)?
            };
            if p.source_parameter() != *descriptor {
                return Err(Error::State(
                    "authentic original parameter role/raw header differs".into(),
                ));
            }
            let window = if let Some(flag) = cancelled {
                context
                    .source
                    .source()
                    .read_parameter_cancellable(&p, 0..p.elements(), flag)?
            } else {
                context
                    .source
                    .source()
                    .read_parameter(&p, 0..p.elements())?
            };
            if !window.is_complete() {
                return Err(Error::State("complete original learned parameter".into()));
            }
            bytes[index] = if index == 3 {
                loaded_gamma(&window)?
            } else {
                window.as_slice().to_vec()
            };
            if !if index == 1 {
                super::layer_common::finite_f32(&bytes[index])
            } else {
                finite_bf16(&bytes[index])
            } {
                return Err(Error::Malformed(
                    "finite original GDN parameter payload".into(),
                ));
            }
            windows.push(window);
        }
        let phase = match invocation.legacy_phase()? {
            VllmBf16Phase::InitialOne => VllmGdnPhase::InitialOne,
            VllmBf16Phase::Initial64 => VllmGdnPhase::Initial64,
            VllmBf16Phase::ContinuedTwo => VllmGdnPhase::ContinuedTwo,
            VllmBf16Phase::CachedOne => VllmGdnPhase::CachedDecode,
        };
        let slot = if matches!(phase, VllmGdnPhase::InitialOne)
            || (phase == VllmGdnPhase::CachedDecode && invocation.teachers().base() == 1)
        {
            GdnCacheSlot::One
        } else {
            GdnCacheSlot::Four
        };
        let payload = Arc::new(ParameterPayload {
            windows,
            bytes,
            cache_zero: vec![0; GdnBuffer::CacheArena.bytes(phase)],
            source: context.source.clone(),
            _host: host,
            _metadata: metadata,
        });
        let mut original_values = Vec::with_capacity(5);
        for (id, width) in ids.into_iter().zip([8192, 32, 6144, 2048, 2048]) {
            let value = values
                .get(id.0 as usize)
                .ok_or_else(|| Error::State("typed GDN value map absent".into()))?;
            value.require_layout(ResidentBufferLayout::new(
                ResidentPhysicalType::Bf16,
                invocation.rows() * width,
            )?)?;
            let lease = value.original_lease()?;
            if !Arc::ptr_eq(lease.service(), &context.service) {
                return Err(Error::State(
                    "same original GDN source/service/value lease".into(),
                ));
            }
            if original_values
                .iter()
                .any(|other: &ResidentLease| other.id() == lease.id())
            {
                return Err(Error::State(
                    "GDN distinct original producer/output values".into(),
                ));
            }
            original_values.push(lease.clone());
        }
        let original_values: [ResidentLease; 5] = original_values
            .try_into()
            .map_err(|_| Error::State("GDN fixed value map".into()))?;
        let mut init = context
            .service
            .begin_initialization(context.domains.base().root())?;
        init.retain_completion_owner(payload.clone())?;
        let parameters = payload
            .bytes
            .iter()
            .map(|raw| init.upload_bytes(raw, BufferRole::ImmutableWeights))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("GDN fixed source parameter cohort".into()))?;
        let cache = init.upload_bytes(&payload.cache_zero, BufferRole::RecurrentState)?;
        drop(init.complete_sync().await?);
        let source = BoundRmsSource::bind(
            context.source,
            invocation.layer().index(),
            if invocation.layer().index() == 0 {
                GdnPrefixRmsSite::PostAttention
            } else {
                GdnPrefixRmsSite::InputResidual
            },
        )?;
        let binding = Arc::new(NativeGdnBinding {
            payload,
            source,
            invocation,
            nodes,
            value_ids: ids,
            values: original_values,
            parameters,
            descriptors,
        });
        binding.require_record(plan)?;
        let inputs = GdnInputs {
            qkvz: binding.values[0].clone(),
            ba: binding.values[1].clone(),
            cache: cache.clone(),
            conv_weights: binding.parameters[0].clone(),
            a_log: binding.parameters[1].clone(),
            dt_bias: binding.parameters[2].clone(),
            gamma: binding.parameters[3].clone(),
        };
        let outputs = GdnTypedOutputs {
            activated: binding.values[2].clone(),
            z: binding.values[3].clone(),
            gated: binding.values[4].clone(),
        };
        let prepared = if context.domains.is_attempt_local() {
            GdnPrepared::prepare_attempt_outputs(
                context.service.clone(),
                phase,
                slot,
                inputs,
                outputs,
                cancelled,
            )
            .await?
        } else {
            GdnPrepared::prepare_initial_outputs(
                context.service.clone(),
                phase,
                inputs,
                outputs,
                cancelled,
            )
            .await?
        };
        let adapter = GdnStateAdapter::prepare(
            context.service,
            binding.clone(),
            context.domains,
            cache,
            slot,
            cancelled,
        )
        .await?;
        binding.require_prepared(&prepared, adapter.domains())?;
        Ok(Arc::new(Self {
            binding,
            prepared,
            adapter,
            pending: Mutex::new(None),
        }))
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut result = self.prepared.accesses();
        for (lease, access) in self.adapter.accesses() {
            if let Some((_, prior)) = result
                .iter_mut()
                .find(|(other, _)| other.id() == lease.id())
            {
                if access == DeviceAccess::Write {
                    *prior = DeviceAccess::Write;
                }
            } else {
                result.push((lease, access));
            }
        }
        result
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.binding
            .parameters
            .iter()
            .cloned()
            .zip(self.binding.payload.bytes.iter().map(Vec::as_slice))
            .collect()
    }
    pub(crate) fn status_readback(&self) -> Result<crate::resident_leases::ResidentReadback> {
        self.adapter.status_readback()
    }
    pub(crate) fn requires_attempt_permit(&self) -> bool {
        self.adapter.domains().is_attempt_local()
    }
    pub(crate) fn enqueue(
        &self,
        node_index: usize,
        cancelled: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<usize> {
        super::projection_cubin::not_cancelled(cancelled)?;
        self.adapter
            .domains()
            .require_enqueue(self.binding.values[0].service(), permit)?;
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| Error::State("GDN split enqueue guard poisoned".into()))?;
        if node_index == self.binding.nodes[0] {
            if pending.is_some() {
                return Err(Error::State("GDN Conv already enqueued".into()));
            }
            let token = self.prepared.launch_root_convolution(
                self.adapter.clone(),
                self.adapter.domains().effect(true),
                cancelled,
                permit,
            )?;
            let count = token.convolution_dispatches();
            *pending = Some(token);
            Ok(count)
        } else if node_index == self.binding.nodes[1] {
            pending
                .take()
                .ok_or_else(|| {
                    Error::State("GDN recurrence requires actual same-flight Conv".into())
                })?
                .launch_recurrent(self.adapter.domains().effect(false), cancelled, permit)
        } else {
            Err(Error::Unsupported("not a bound GDN plan node".into()))
        }
    }
}
