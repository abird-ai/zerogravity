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
use crate::resident_program::source_bindings::SourceBindings;
use crate::resident_program::text_positions::TextPositionsPrepared;
use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::{ParameterWindow, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    SourceCallMode, SourceNumericalContract, SourceOperator, SourceOperatorKind,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

struct ParameterPayload {
    _source: VerifiedTensorSource,
    windows: [ParameterWindow; 2],
    _host: WindowWorkspace,
}
pub(crate) struct ModelAttentionParameters {
    operator: SourceOperator,
    source: VerifiedTensorSource,
    leases: [ResidentLease; 2],
    payload: Arc<ParameterPayload>,
}
impl ModelAttentionParameters {
    pub(crate) fn prepare(
        source: VerifiedTensorSource,
        operator: &SourceOperator,
        service: &Arc<crate::resident_leases::ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::super::projection_cubin::not_cancelled(cancelled)?;
        init.require_service(service)?;
        Self::require_qk_operator(operator)?;
        if operator.call().source() != source.source_identity()
            || service.source_identity() != source.source_identity()
        {
            return Err(Error::State(
                "attention gamma actual source/call/service differs".into(),
            ));
        }
        let host = source.reserve_workspace(64 * 1024)?;
        let mut windows = Vec::with_capacity(2);
        for descriptor in operator.parameters() {
            let parameter = source.parameter(&descriptor.name)?;
            parameter.validate_source_parameter(descriptor)?;
            if descriptor.bytes != 512
                || descriptor.kind != zerogravity_core::quant::QuantKind::Bf16
                || descriptor.shape.as_slice() != [256]
            {
                return Err(Error::State("exact original attention gamma256".into()));
            }
            let window = match cancelled {
                Some(flag) => source.read_parameter_cancellable(&parameter, 0..256, flag)?,
                None => source.read_parameter(&parameter, 0..256)?,
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
            operator: operator.clone(),
            leases,
            payload,
        });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
    fn require_qk_operator(operator: &SourceOperator) -> Result<()> {
        operator.validate()?;
        if !matches!(operator.kind(), SourceOperatorKind::QkRmsRotaryGated(g)
            if g.attention.width == 256 && g.attention.query_heads == 8
                && g.attention.kv_heads == 2 && g.rotary_dimensions == 64
                && g.theta == 10_000_000.0 && g.epsilon == 1e-6)
            || operator.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            || operator.call().rows().capacity() != 512
            || operator.call().rows().vocabulary() != 248320
            || operator.parameters().iter().any(|p| {
                p.kind != zerogravity_core::quant::QuantKind::Bf16
                    || p.shape.as_slice() != [256]
                    || p.bytes != 512
            })
            || operator.parameters().len() != 2
            || !operator.matrices().is_empty()
        {
            return Err(Error::State(
                "attention gamma physical QK mathematics/headers differ".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn nodes(record: &SourceBindings, layer: usize) -> Result<[usize; 7]> {
        Ok(
            super::super::fixed_prompt_model_prefix::AttentionSite::bind_nodes(
                record.call(),
                record.plan().nodes(),
                layer,
            )?
            .nodes(),
        )
    }
    fn require_operator(
        &self,
        actual: &SourceOperator,
        service: &Arc<crate::resident_leases::ResidentService>,
    ) -> Result<[ResidentLease; 2]> {
        Self::require_qk_operator(actual)?;
        if actual.kind() != self.operator.kind()
            || actual.parameters() != self.operator.parameters()
            || actual.call().source() != self.source.source_identity()
        {
            return Err(Error::State(
                "attention original QK source/member order differs".into(),
            ));
        }
        for (i, parameter) in actual.parameters().iter().enumerate() {
            let lease = &self.leases[i];
            if self.payload.windows[i].parameter().source_parameter() != *parameter
                || lease.bytes() != 512
                || lease.role() != BufferRole::ImmutableWeights
                || !lease.is_initialized()
                || !Arc::ptr_eq(lease.service(), service)
                || lease.service_generation() != service.generation()
            {
                return Err(Error::State(
                    "attention QK source/generation/extent differs".into(),
                ));
            }
        }
        Ok(self.leases.clone())
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
    layer: usize,
    case: AttentionCase,
    nodes: BTreeMap<usize, AttentionStage>,
    native: Arc<AttentionBindings>,
    positions: Arc<TextPositionsPrepared>,
    rotary: Arc<ModelRotary>,
    kv: Arc<ModelKvTransport>,
    domains: Arc<BoundModelState>,
    source: VerifiedTensorSource,
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
        let source = weights.source().clone();
        cache.require_binding(&domains)?;
        positions.require_invocation(call)?;
        let layer = domains.kv_domain()?.0;
        if domains.call() != call || !source.same_issuer(&parameters.source) {
            return Err(Error::State(
                "attention exact actual source/domain/call owners differ".into(),
            ));
        }
        let case = match (call.mode(), call.rows().base(), call.rows().len()) {
            (SourceCallMode::Prefill, 0, 1) => AttentionCase::InitialOne,
            (SourceCallMode::Decode, 1, 1) => AttentionCase::CachedOneAt1,
            (SourceCallMode::Prefill, 0, 64) => AttentionCase::Initial64,
            (SourceCallMode::Prefill, 64, 2) => AttentionCase::ContinuedTwoAt64,
            (SourceCallMode::Decode, 66, 1) => AttentionCase::CachedOneAt66,
            _ => {
                return Err(Error::Unsupported(
                    "exact original attention phase ABI".into(),
                ));
            }
        };
        let descriptor =
            AttentionDescriptor::with_scope(call.source(), case, AttentionScope::ComputedGate)?;
        let indices = ModelAttentionParameters::nodes(record, layer)?;
        parameters.require_operator(
            record
                .operator(indices[1])?
                .ok_or_else(|| Error::State("actual QK operator absent".into()))?,
            values.service(),
        )?;
        let stages = [
            AttentionStage::Qkv,
            AttentionStage::Qk,
            AttentionStage::Kv,
            AttentionStage::Fa,
            AttentionStage::Sigmoid,
            AttentionStage::Multiply,
            AttentionStage::Out,
        ]
        .into_iter()
        .zip(indices)
        .collect::<BTreeMap<_, _>>();
        let nodes = stages
            .iter()
            .map(|(&stage, &index)| (index, stage))
            .collect();
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
        let learned = |stage: AttentionStage| -> Result<ResidentLease> {
            let Some(descriptor) = record.operator(stages[&stage])? else {
                unreachable!()
            };
            Ok(weights.require_operator(descriptor)?.lease().clone())
        };
        let mut leases = BTreeMap::from([
            (AttentionRole::QkvW, learned(AttentionStage::Qkv)?),
            (AttentionRole::OutW, learned(AttentionStage::Out)?),
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
        let host = source.reserve_workspace(64 * 1024)?;
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
            AttentionBindings::new(
                descriptor,
                values.service().clone(),
                Arc::new(source.clone()),
                exact,
            )
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
        self.layer
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

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod natural;

mod fixed_prompt;

mod whole;

mod fixed_prompt_natural;

#[cfg(test)]
mod source_operator_tests {
    use super::*;
    use zerogravity_core::accelerator::{
        SourceIdentity,
        program::{
            AttentionGeometry, SourceCall, SourceParameter, SourceQkRotaryGeometry, SourceRows,
            TeacherSpan,
        },
    };

    fn qk(theta: f32, query_heads: usize) -> Result<SourceOperator> {
        let source = SourceIdentity::for_new_allocation()?;
        let call = SourceCall::new(
            SourceRows::from_teacher_span(TeacherSpan::new(source, 0, 512, 248320, &[1])?, 248320)?,
            SourceCallMode::Prefill,
        )?;
        let parameters = ["q", "k"]
            .into_iter()
            .map(|name| SourceParameter {
                source,
                name: name.into(),
                kind: zerogravity_core::quant::QuantKind::Bf16,
                shape: vec![256],
                bytes: 512,
            })
            .collect();
        SourceOperator::new(
            call,
            SourceOperatorKind::QkRmsRotaryGated(SourceQkRotaryGeometry {
                attention: AttentionGeometry {
                    width: 256,
                    query_heads,
                    kv_heads: 2,
                },
                rotary_dimensions: 64,
                theta,
                epsilon: 1e-6,
            }),
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            vec![],
            parameters,
        )
    }
    #[test]
    fn actual_qk_math_must_match_native_gamma_capture() -> Result<()> {
        ModelAttentionParameters::require_qk_operator(&qk(10_000_000.0, 8)?)?;
        assert!(ModelAttentionParameters::require_qk_operator(&qk(1_000_000.0, 8)?).is_err());
        assert!(ModelAttentionParameters::require_qk_operator(&qk(10_000_000.0, 16)?).is_err());
        Ok(())
    }
}
