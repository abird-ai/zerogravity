//! Sealed original-source parameter and real plan producer binding for GDN.
use super::*;
use crate::cuda::gdn::{
    GdnCacheSlot, GdnConvEnqueued, GdnInputs, GdnPrepared, GdnStateAdapter, GdnTypedOutputs,
};
use crate::resident_inference::gdn_state::BoundGdnDomains;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_program::source_bindings::SourceBindings;
use crate::resident_program::{DeviceBuffer, ResidentBufferLayout, ResidentPhysicalType};
use std::sync::{Mutex, atomic::AtomicBool};
use zerogravity_core::accelerator::program::ForwardPlan;
use zerogravity_core::accelerator::program::PlanOperation;
use zerogravity_core::accelerator::program::SourceParameter;
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::accelerator::program::{
    SourceCall, SourceCallMode, SourceGdnContract, SourceOperator, SourceOperatorKind,
};

struct ParameterPayload {
    windows: Vec<ParameterWindow>,
    bytes: [Vec<u8>; 4],
    cache_zero: Vec<u8>,
    source: VerifiedTensorSource,
    // Actual raw/converted host payloads precede their SAME source admissions.
    _host: WindowWorkspace,
    _metadata: WindowWorkspace,
}
/// No public constructor: only this actual source/window/upload preparation.
pub(crate) struct NativeGdnBinding {
    payload: Arc<ParameterPayload>,
    source: Arc<SourceBindings>,
    invocation: SourceCall,
    nodes: [usize; 2],
    value_ids: [ValueId; 5],
    values: [ResidentLease; 5],
    parameters: [ResidentLease; 4],
    descriptors: [SourceParameter; 4],
    operators: [zerogravity_core::accelerator::program::SourceOperator; 2],
}
impl NativeGdnBinding {
    pub(crate) fn source_binding(&self) -> &SourceBindings {
        &self.source
    }
    pub(crate) fn operators(&self) -> &[SourceOperator; 2] {
        &self.operators
    }
    pub(crate) fn nodes(&self) -> [usize; 2] {
        self.nodes
    }
    pub(crate) fn value_ids(&self) -> [ValueId; 5] {
        self.value_ids
    }
    pub(crate) fn invocation(&self) -> &SourceCall {
        &self.invocation
    }
    pub(crate) fn require_prepared(
        &self,
        prepared: &GdnPrepared,
        domains: &BoundGdnDomains,
    ) -> Result<()> {
        if !self.source.source().same_issuer(&self.payload.source) {
            return Err(Error::State(
                "GDN actual original source issuer differs".into(),
            ));
        }
        if domains.invocation() != &self.invocation
            || !matches!(self.operators[0].kind(), SourceOperatorKind::CausalConvSplit { domain, .. }
                if domain == domains.conv_domain()?)
            || !matches!(self.operators[1].kind(), SourceOperatorKind::GatedDelta { domain, .. }
                if domain == domains.recurrent_domain()?)
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
            PlanOperation::SourceOperator { descriptor: cd, .. },
            PlanOperation::SourceOperator { descriptor: rd, .. },
        ) = (&conv.operation, &recur.operation)
        else {
            return Err(Error::State("GDN exact actual typed records".into()));
        };
        if cd.as_ref() != &self.operators[0]
            || rd.as_ref() != &self.operators[1]
            || cd.call() != &self.invocation
            || rd.call() != &self.invocation
            || cd.parameters() != &self.descriptors[..1]
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
    pub bindings: Arc<SourceBindings>,
    pub domains: Arc<BoundGdnDomains>,
}
pub(crate) struct NativeGdnPrepared {
    binding: Arc<NativeGdnBinding>,
    prepared: Arc<GdnPrepared>,
    adapter: Arc<GdnStateAdapter>,
    pending: Mutex<Option<GdnConvEnqueued>>,
}
pub(crate) fn phase(call: &SourceCall) -> Result<GdnPhase> {
    if call.rows().capacity() != 512 || call.rows().vocabulary() != 248320 {
        return Err(Error::Unsupported(
            "native GDN finite source call geometry".into(),
        ));
    }
    match (call.mode(), call.rows().base(), call.rows().len()) {
        (SourceCallMode::Prefill, 0, 1) => Ok(GdnPhase::InitialOne),
        (SourceCallMode::Prefill, 0, 64) => Ok(GdnPhase::Initial64),
        (SourceCallMode::Prefill, 64, 2) => Ok(GdnPhase::ContinuedTwo),
        (SourceCallMode::Decode, 1 | 66, 1) => Ok(GdnPhase::CachedDecode),
        _ => Err(Error::Unsupported("native GDN finite source phase".into())),
    }
}
pub(crate) fn operands(
    bindings: &SourceBindings,
    nodes: [usize; 2],
) -> Result<([SourceOperator; 2], [ValueId; 5])> {
    let plan = bindings.plan();
    let get = |index| {
        plan.nodes()
            .get(index)
            .ok_or_else(|| Error::State("GDN actual source node absent".into()))
    };
    let conv = get(nodes[0])?;
    let recurrent = get(nodes[1])?;
    let (
        PlanOperation::SourceOperator {
            descriptor: cd,
            state: None,
        },
        PlanOperation::SourceOperator {
            descriptor: rd,
            state: None,
        },
    ) = (&conv.operation, &recurrent.operation)
    else {
        return Err(Error::State("GDN actual mathematical pair absent".into()));
    };
    let (
        SourceOperatorKind::CausalConvSplit {
            domain: c,
            geometry: cg,
        },
        SourceOperatorKind::GatedDelta {
            domain: r,
            geometry: rg,
        },
    ) = (cd.kind(), rd.kind())
    else {
        return Err(Error::State(
            "GDN actual convolution/recurrent kinds differ".into(),
        ));
    };
    if c.0 != r.0
        || cg.qkv_channels != 6144
        || cg.z_channels != 2048
        || cg.coefficient_channels != 32
        || cg.kernel != 4
        || rg.key_heads != 16
        || rg.value_heads != 16
        || rg.key_width != 128
        || rg.value_width != 128
        || rg.epsilon != 1e-6
        || rg.contract != SourceGdnContract::ReferenceCudaSm86L2SiluBf16V1
        || cd.call() != bindings.call()
        || rd.call() != bindings.call()
        || nodes[0] >= nodes[1]
    {
        return Err(Error::Unsupported(
            "GDN actual supported native math/domain/call".into(),
        ));
    }
    let ([qkvz, ba], [activated, z], [ra, rz, rb], [gated]) = (
        conv.inputs.as_slice(),
        conv.outputs.as_slice(),
        recurrent.inputs.as_slice(),
        recurrent.outputs.as_slice(),
    ) else {
        return Err(Error::State("GDN actual producer pair arities".into()));
    };
    if activated != ra || z != rz || ba != rb {
        return Err(Error::State(
            "GDN actual convolution/Z/BA producer chain differs".into(),
        ));
    }
    // Both projections must consume the SAME genuine normalized producer.
    let input = |id: ValueId| -> Result<ValueId> {
        let tag = plan
            .source_value_provenance(id)?
            .ok_or_else(|| Error::State("GDN projection source producer absent".into()))?;
        let node = get(tag.node_index())?;
        let PlanOperation::SourceOperator {
            descriptor,
            state: None,
        } = &node.operation
        else {
            return Err(Error::State("GDN actual input projection absent".into()));
        };
        let [normalized] = node.inputs.as_slice() else {
            return Err(Error::State("GDN actual normalized input arity".into()));
        };
        if descriptor.kind() != (SourceOperatorKind::Projection { selected_row: None })
            || descriptor.call() != bindings.call()
            || node.outputs != [id]
            || tag.output_index() != 0
        {
            return Err(Error::State(
                "GDN actual normalized projection producer differs".into(),
            ));
        }
        Ok(*normalized)
    };
    let normalized = input(*qkvz)?;
    if input(*ba)? != normalized {
        return Err(Error::State(
            "GDN QKVZ/BA normalized producers differ".into(),
        ));
    }
    let tag = plan
        .source_value_provenance(normalized)?
        .ok_or_else(|| Error::State("GDN normalized source producer absent".into()))?;
    let node = get(tag.node_index())?;
    let PlanOperation::SourceOperator {
        descriptor,
        state: None,
    } = &node.operation
    else {
        return Err(Error::State(
            "GDN normalized mathematical producer absent".into(),
        ));
    };
    if !matches!(
        descriptor.kind(),
        SourceOperatorKind::Rms(_) | SourceOperatorKind::ResidualRms(_)
    ) || descriptor.call() != bindings.call()
        || tag.output_index() != 0
        || node.outputs.first() != Some(&normalized)
    {
        return Err(Error::State(
            "GDN normalized source/SSA producer differs".into(),
        ));
    }
    let ids = [*qkvz, *ba, *activated, *z, *gated];
    for id in ids {
        let tag = plan
            .source_value_provenance(id)?
            .ok_or_else(|| Error::State("GDN actual value provenance absent".into()))?;
        let producer = get(tag.node_index())?;
        if tag.call() != bindings.call()
            || producer.outputs.get(tag.output_index()) != Some(&id)
            || tag.inputs() != producer.inputs
        {
            return Err(Error::State(
                "GDN actual value producer/call differs".into(),
            ));
        }
    }
    Ok(([cd.as_ref().clone(), rd.as_ref().clone()], ids))
}
impl NativeGdnPrepared {
    /// Guard-only preparation. Caller owns the SAME original preparation slot.
    pub(crate) async fn prepare(
        context: NativeGdnContext,
        values: &[DeviceBuffer],
        nodes: [usize; 2],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::projection_cubin::not_cancelled(cancelled)?;
        let plan = context.bindings.plan();
        let (operators, ids) = operands(&context.bindings, nodes)?;
        let conv = &operators[0];
        let recurrent = &operators[1];
        let invocation = conv.call().clone();
        let phase = phase(&invocation)?;
        if context.domains.invocation() != &invocation
            || context.domains.phase() != phase
            || (!context.domains.is_attempt_local()
                && !matches!(phase, GdnPhase::InitialOne | GdnPhase::Initial64))
            || context.service.source_identity() != context.bindings.source().source_identity()
        {
            return Err(Error::Unsupported(
                "GDN original source/call/root phase differs".into(),
            ));
        }
        let metadata = context
            .bindings
            .source()
            .reserve_workspace(2 * 1024 * 1024)?;
        let host = context
            .bindings
            .source()
            .reserve_workspace(8 * 1024 * 1024)?;
        let descriptors = [
            conv.parameters()[0].clone(),
            recurrent.parameters()[0].clone(),
            recurrent.parameters()[1].clone(),
            recurrent.parameters()[2].clone(),
        ];
        let mut windows = Vec::with_capacity(4);
        let mut bytes: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
        for (index, descriptor) in descriptors.iter().enumerate() {
            let p = if index == 0 {
                context
                    .bindings
                    .source()
                    .conv4_parameter(&descriptor.name)?
            } else {
                context.bindings.source().parameter(&descriptor.name)?
            };
            if p.source_parameter() != *descriptor {
                return Err(Error::State(
                    "authentic original parameter role/raw header differs".into(),
                ));
            }
            let window = if let Some(flag) = cancelled {
                context
                    .bindings
                    .source()
                    .read_parameter_cancellable(&p, 0..p.elements(), flag)?
            } else {
                context
                    .bindings
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
        let slot = if matches!(phase, GdnPhase::InitialOne)
            || (phase == GdnPhase::CachedDecode && invocation.rows().base() == 1)
        {
            GdnCacheSlot::One
        } else {
            GdnCacheSlot::Four
        };
        let payload = Arc::new(ParameterPayload {
            windows,
            bytes,
            cache_zero: vec![0; GdnBuffer::CacheArena.bytes(phase)],
            source: context.bindings.source().clone(),
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
                invocation.rows().len() * width,
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
        let source = context.bindings.clone();
        let binding = Arc::new(NativeGdnBinding {
            payload,
            source,
            invocation,
            nodes,
            value_ids: ids,
            values: original_values,
            parameters,
            descriptors,
            operators,
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
