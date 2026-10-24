//! Full-source attention operands sealed from the actual SAME full24 value map.
use super::super::{
    model_cache::ModelCache, model_kv_transport::ModelKvTransport, model_rotary::ModelRotary,
    model_values::ModelValues, model_weights::ModelWeights,
};
use super::{
    descriptor::{AttentionBindings, AttentionDescriptor, AttentionScope},
    manifests::*,
};
use crate::resident_inference::{
    ResidentSession,
    native_fragment::{PrefixPhaseEnqueue, model_state::BoundModelState},
};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use crate::resident_program::text_positions::TextPositionsPrepared;
use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::{ParameterWindow, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    PlanOperation, Vllm08Kind, Vllm08Layer, Vllm08Projection, VllmBf16Phase,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::{OriginalWindowModelSource, QwenParameterRole};

struct ParameterPayload {
    _source: Arc<OriginalWindowModelSource>,
    windows: [ParameterWindow; 2],
    _host: WindowWorkspace,
}
pub(crate) struct ModelAttentionParameters {
    layer: Vllm08Layer,
    source: Arc<OriginalWindowModelSource>,
    leases: [ResidentLease; 2],
    payload: Arc<ParameterPayload>,
}
impl ModelAttentionParameters {
    pub(crate) fn prepare(
        source: Arc<OriginalWindowModelSource>,
        layer: Vllm08Layer,
        service: &Arc<crate::resident_leases::ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::super::projection_cubin::not_cancelled(cancelled)?;
        source.require_vllm08_full_source()?;
        init.require_service(service)?;
        if !layer.is_attention()
            || layer.source() != source.source_identity()
            || service.source_identity() != source.source_identity()
        {
            return Err(Error::State(
                "full24 original attention gamma source/layer/service".into(),
            ));
        }
        let host = source.source().reserve_workspace(64 * 1024)?;
        let mut windows = Vec::with_capacity(2);
        for role in [
            QwenParameterRole::QueryNorm(layer.index()),
            QwenParameterRole::KeyNorm(layer.index()),
        ] {
            let descriptor = source.parameter_descriptor(role)?;
            source.validate_parameter(role, &descriptor)?;
            let parameter = source.source().parameter(&descriptor.name)?;
            if parameter.source_parameter() != descriptor || descriptor.bytes != 512 {
                return Err(Error::State(
                    "full24 exact original attention gamma256".into(),
                ));
            }
            let window = match cancelled {
                Some(flag) => {
                    source
                        .source()
                        .read_parameter_cancellable(&parameter, 0..256, flag)?
                }
                None => source.source().read_parameter(&parameter, 0..256)?,
            };
            if !window.is_complete() || !super::super::finite_bf16(window.as_slice()) {
                return Err(Error::Malformed(
                    "full24 finite raw BF16 attention gamma".into(),
                ));
            }
            windows.push(window);
        }
        let windows = windows.try_into().map_err(|_| {
            Error::State("full24 fixed two original attention gamma windows".into())
        })?;
        let payload = Arc::new(ParameterPayload {
            _source: source.clone(),
            windows,
            _host: host,
        });
        init.retain_completion_owner(payload.clone())?;
        let leases = payload
            .windows
            .iter()
            .map(|window| init.upload_bytes(window.as_slice(), BufferRole::ImmutableWeights))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("full24 two original attention gamma leases".into()))?;
        let prepared = Arc::new(Self {
            source,
            layer,
            leases,
            payload,
        });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.leases
            .iter()
            .cloned()
            .zip(self.payload.windows.iter().map(|w| w.as_slice()))
            .collect()
    }
}
pub(crate) struct ModelAttentionBinding {
    values: Arc<ModelValues>,
    layer: Vllm08Layer,
    case: AttentionCase,
    nodes: BTreeMap<usize, AttentionStage>,
    native: Arc<AttentionBindings>,
    positions: Arc<TextPositionsPrepared>,
    rotary: Arc<ModelRotary>,
    kv: Arc<ModelKvTransport>,
    domains: Arc<BoundModelState>,
    source: Arc<OriginalWindowModelSource>,
    _weights: Arc<ModelWeights>,
    _parameters: Arc<ModelAttentionParameters>,
    _cache: Arc<ModelCache>,
    _controls: Arc<ControlPayload>,
}
struct ControlPayload {
    bytes: Vec<Vec<u8>>,
    _host: WindowWorkspace,
}
pub(crate) struct ModelAttentionInputs {
    pub(crate) values: Arc<ModelValues>,
    pub(crate) weights: Arc<ModelWeights>,
    pub(crate) parameters: Arc<ModelAttentionParameters>,
    pub(crate) positions: Arc<TextPositionsPrepared>,
    pub(crate) rotary: Arc<ModelRotary>,
    pub(crate) domains: Arc<BoundModelState>,
    pub(crate) cache: Arc<ModelCache>,
    pub(crate) workspace: ResidentLease,
}
impl ModelAttentionBinding {
    pub(crate) fn prepare(
        inputs: ModelAttentionInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let ModelAttentionInputs {
            values,
            weights,
            parameters,
            positions,
            rotary,
            domains,
            cache,
            workspace,
        } = inputs;
        super::super::projection_cubin::not_cancelled(cancelled)?;
        let record = values.record();
        let call = record.call();
        init.require_service(values.service())?;
        let layer = domains.layer();
        let source = weights.source().clone();
        source.require_vllm08_full_source()?;
        cache.require_binding(&domains)?;
        positions.require_invocation(call.request())?;
        if !layer.is_attention()
            || domains.call() != call
            || parameters.layer != layer
            || !Arc::ptr_eq(&source, &parameters.source)
        {
            return Err(Error::State(
                "full24 exact attention source/layer/teacher owners".into(),
            ));
        }
        let case = match (
            call.request().legacy_phase()?,
            call.request().teachers().base(),
        ) {
            (VllmBf16Phase::InitialOne, 0) => AttentionCase::InitialOne,
            (VllmBf16Phase::CachedOne, 1) => AttentionCase::CachedOneAt1,
            (VllmBf16Phase::Initial64, 0) => AttentionCase::Initial64,
            (VllmBf16Phase::ContinuedTwo, 64) => AttentionCase::ContinuedTwoAt64,
            (VllmBf16Phase::CachedOne, 66) => AttentionCase::CachedOneAt66,
            _ => {
                return Err(Error::Unsupported(
                    "full24 exact original attention phase ABI".into(),
                ));
            }
        };
        let descriptor =
            AttentionDescriptor::with_scope(call.source(), case, AttentionScope::ComputedGate)?;
        let mut nodes = BTreeMap::new();
        let mut stages = BTreeMap::new();
        for (index, node) in record.plan().nodes().iter().enumerate() {
            let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                continue;
            };
            if descriptor.layer() != Some(layer) {
                continue;
            }
            let stage = match descriptor.kind() {
                Vllm08Kind::Projection(Vllm08Projection::AttentionQkv) => AttentionStage::Qkv,
                Vllm08Kind::AttentionQk => AttentionStage::Qk,
                Vllm08Kind::KvAppend => AttentionStage::Kv,
                Vllm08Kind::FlashAttention => AttentionStage::Fa,
                Vllm08Kind::Sigmoid => AttentionStage::Sigmoid,
                Vllm08Kind::Multiply => AttentionStage::Multiply,
                Vllm08Kind::Projection(Vllm08Projection::AttentionOut) => AttentionStage::Out,
                _ => continue,
            };
            descriptor.validate()?;
            if descriptor.call() != call || stages.insert(stage, index).is_some() {
                return Err(Error::State(
                    "full24 unique original attention node/teacher".into(),
                ));
            }
            nodes.insert(index, stage);
        }
        if nodes.len() != 7 {
            return Err(Error::State(
                "full24 complete seven attention/gate nodes".into(),
            ));
        }
        let get =
            |stage: AttentionStage| -> Result<&zerogravity_core::accelerator::program::PlanNode> {
                stages
                    .get(&stage)
                    .map(|index| &record.plan().nodes()[*index])
                    .ok_or_else(|| Error::State("full24 original attention stage absent".into()))
            };
        let qkv = get(AttentionStage::Qkv)?;
        let qk = get(AttentionStage::Qk)?;
        let fa = get(AttentionStage::Fa)?;
        let sigmoid = get(AttentionStage::Sigmoid)?;
        let multiply = get(AttentionStage::Multiply)?;
        let out = get(AttentionStage::Out)?;
        let learned =
            |node: &zerogravity_core::accelerator::program::PlanNode| -> Result<ResidentLease> {
                let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                    unreachable!()
                };
                Ok(weights.require_operator(descriptor)?.lease().clone())
            };
        let mut leases = BTreeMap::from([
            (AttentionRole::QkvW, learned(qkv)?),
            (AttentionRole::OutW, learned(out)?),
            (
                AttentionRole::QkvInput,
                values.lease(qkv.inputs[0])?.clone(),
            ),
            (
                AttentionRole::PackedQkv,
                values.lease(qkv.outputs[0])?.clone(),
            ),
            (AttentionRole::Q, values.lease(qk.outputs[0])?.clone()),
            (AttentionRole::K, values.lease(qk.outputs[1])?.clone()),
            (AttentionRole::Gate, values.lease(qk.outputs[2])?.clone()),
            (
                AttentionRole::AttentionOutput,
                values.lease(fa.outputs[0])?.clone(),
            ),
            (
                AttentionRole::SigmoidGate,
                values.lease(sigmoid.outputs[0])?.clone(),
            ),
            (
                AttentionRole::OutInput,
                values.lease(multiply.outputs[0])?.clone(),
            ),
            (
                AttentionRole::Projected,
                values.lease(out.outputs[0])?.clone(),
            ),
            (AttentionRole::QGamma, parameters.leases[0].clone()),
            (AttentionRole::KGamma, parameters.leases[1].clone()),
            (AttentionRole::OriginalKvState, cache.lease().clone()),
            (
                AttentionRole::PositionsI64Pitch65,
                positions.prepared_lease().clone(),
            ),
            (AttentionRole::CosSinSourcePrefix67, rotary.lease().clone()),
            (AttentionRole::Workspace, workspace),
        ]);
        let host = source.source().reserve_workspace(64 * 1024)?;
        let controls = descriptor
            .required_roles()?
            .keys()
            .filter_map(|role| {
                super::qualify::control_bytes(&descriptor, *role).map(|bytes| (*role, bytes))
            })
            .filter(|(role, _)| *role != AttentionRole::PositionsI64Pitch65)
            .collect::<Vec<_>>();
        let control_roles = controls.iter().map(|(role, _)| *role).collect::<Vec<_>>();
        let controls = Arc::new(ControlPayload {
            bytes: controls.into_iter().map(|(_, bytes)| bytes).collect(),
            _host: host,
        });
        init.retain_completion_owner(controls.clone())?;
        for (role, bytes) in control_roles.into_iter().zip(&controls.bytes) {
            if leases
                .insert(role, init.upload_bytes(bytes, BufferRole::Activation)?)
                .is_some()
            {
                return Err(Error::State(
                    "full24 derived attention control role duplicates producer".into(),
                ));
            }
        }
        // Only actual native scratch remains; no observed activation is loaded.
        let mut exact = BTreeMap::new();
        for (role, (bytes, _)) in descriptor.required_roles()? {
            let lease = match leases.remove(&role) {
                Some(lease) => lease,
                None => init.allocate_empty(bytes, BufferRole::Scratch)?,
            };
            exact.insert(role, lease);
        }
        if leases.keys().any(|role| *role != AttentionRole::Workspace) {
            return Err(Error::State(
                "full24 native attention silently unused producer role".into(),
            ));
        }
        // Original source provenance graph is retained by both native bindings
        // and the returned wrapper, without retaining a body that owns them.
        let native = unsafe {
            AttentionBindings::new(descriptor, values.service().clone(), source.clone(), exact)
        }?;
        let kv = ModelKvTransport::prepare(
            source.clone(),
            domains.clone(),
            cache.clone(),
            init,
            cancelled,
        )?;
        let binding = Arc::new(Self {
            values,
            layer,
            case,
            nodes,
            native,
            positions,
            rotary,
            kv,
            domains,
            source,
            _weights: weights,
            _parameters: parameters,
            _cache: cache,
            _controls: controls,
        });
        init.retain_completion_owner(binding.clone())?;
        Ok(binding)
    }
    pub(crate) fn require_value_state(
        &self,
        values: &Arc<ModelValues>,
        domains: &Arc<BoundModelState>,
    ) -> Result<()> {
        if !Arc::ptr_eq(values, &self.values) || !Arc::ptr_eq(domains, &self.domains) {
            return Err(Error::State(
                "full24 attention body differs from original sealed value/state map".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn case(&self) -> AttentionCase {
        self.case
    }
    pub(crate) fn layer(&self) -> usize {
        self.layer.index()
    }
    pub(super) fn native(&self) -> &Arc<AttentionBindings> {
        &self.native
    }
    pub(crate) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut accesses = self.native.accesses()?;
        for (lease, access) in self.kv.accesses() {
            match accesses.iter_mut().find(|(old, _)| old.id() == lease.id()) {
                Some((_, old)) if access == DeviceAccess::Write => *old = access,
                Some(_) => {}
                None => accesses.push((lease, access)),
            }
        }
        Ok(accesses)
    }
    pub(super) fn stage(&self, node: usize) -> Result<AttentionStage> {
        self.nodes
            .get(&node)
            .copied()
            .ok_or_else(|| Error::State("full24 unbound attention node".into()))
    }
    pub(crate) fn require_enqueue(
        &self,
        session: &ResidentSession,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        self.domains
            .require_enqueue(self.values.service(), permit)?;
        self.positions
            .require_enqueue(session, Some(permit), cancelled)?;
        self.rotary.require_call(
            &self.source,
            self.values.record().call(),
            self.values.service(),
        )
    }
    pub(super) fn before_kv(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.kv
            .before_append(permit, self.domains.effect(false)?, cancelled)
    }
    pub(super) fn after_kv(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.kv
            .after_append(permit, self.domains.effect(false)?, cancelled)
    }
    pub(crate) fn kv_status_readback(&self) -> Result<crate::resident_leases::ResidentReadback> {
        self.kv.status_readback()
    }
}

/// Closed actual manifest/control scratch union; public/source/state roles are
/// charged by the original full-body allocator, never counted as new buffers.
pub(crate) fn full_model_private_bound(
    source: zerogravity_core::accelerator::SourceIdentity,
) -> Result<usize> {
    let mut bound = 0usize;
    for case in AttentionCase::ALL {
        let descriptor =
            AttentionDescriptor::with_scope(source, case, AttentionScope::ComputedGate)?;
        for (role, (bytes, _)) in descriptor.required_roles()? {
            if matches!(
                role,
                AttentionRole::QkvW
                    | AttentionRole::OutW
                    | AttentionRole::QkvInput
                    | AttentionRole::PackedQkv
                    | AttentionRole::Q
                    | AttentionRole::K
                    | AttentionRole::Gate
                    | AttentionRole::AttentionOutput
                    | AttentionRole::SigmoidGate
                    | AttentionRole::OutInput
                    | AttentionRole::Projected
                    | AttentionRole::QGamma
                    | AttentionRole::KGamma
                    | AttentionRole::OriginalKvState
                    | AttentionRole::PositionsI64Pitch65
                    | AttentionRole::CosSinSourcePrefix67
                    | AttentionRole::Workspace
            ) {
                continue;
            }
            bound =
                bound
                    .checked_add(bytes.checked_mul(6).ok_or_else(|| {
                        Error::Limit("full24 attention private role bytes".into())
                    })?)
                    .ok_or_else(|| Error::Limit("full24 attention private union".into()))?;
        }
    }
    Ok(bound)
}

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod natural;

mod canonical47;

mod whole;

mod canonical_natural;
