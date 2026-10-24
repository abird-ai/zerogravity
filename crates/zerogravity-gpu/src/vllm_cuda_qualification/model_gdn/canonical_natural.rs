//! Root-free source/math preparation, then actual current-root binding per call.
use super::*;
use crate::resident_inference::native_fragment::canonical47_natural::CanonicalCallInput;
use crate::vllm_cuda_gdn::{GdnFamilyIdentity, ModelGdnTransportStorage};
use crate::vllm_cuda_qualification::{
    canonical47_model::{Canonical47CallKind, Canonical47ModelBinding},
    model_cache::SharedModelCache,
};

pub(crate) struct ColdCanonicalGdn {
    scope: Arc<Canonical47ModelBinding>,
    binding: Arc<ModelGdnBinding>,
    prepared: Arc<GdnPrepared>,
    transport: Arc<ModelGdnTransportStorage>,
    arena: Arc<SharedModelCache>,
    phase: VllmGdnPhase,
    _metadata: WindowWorkspace,
}

fn phase(scope: &Canonical47ModelBinding) -> VllmGdnPhase {
    match scope.kind() {
        Canonical47CallKind::Prefill => VllmGdnPhase::Initial47,
        Canonical47CallKind::Decode { .. } => VllmGdnPhase::CachedDecode,
    }
}

fn require_arithmetic(
    scope: &Canonical47ModelBinding,
    layer: usize,
    actual: &GdnArithmetic,
) -> Result<()> {
    let phase = phase(scope);
    actual.require_instance(phase, GdnCacheSlot::One, true)?;
    actual
        .family()
        .require_same(&GdnFamilyIdentity::original08())?;
    if scope.kind() == Canonical47CallKind::Prefill {
        scope.require_gdn_arithmetic(
            layer,
            actual,
            crate::vllm_cuda_gdn::Canonical47GdnArithmetic::OwnedCanonical47Solve4,
        )?;
    }
    // CachedDecode is the exact3-stage assets factory. It has no Solve stage;
    // do not attach the Initial47 Solve4 declaration to a decoded invocation.
    Ok(())
}

fn binding(
    scope: &Arc<Canonical47ModelBinding>,
    values: Arc<ModelValues>,
    parameters: Arc<ModelGdnParameters>,
    nodes: [usize; 2],
    arithmetic: Arc<GdnArithmetic>,
) -> Result<Arc<ModelGdnBinding>> {
    scope.require_values(&values)?;
    let record = values.record();
    let conv = record
        .plan()
        .nodes()
        .get(nodes[0])
        .ok_or_else(|| Error::State("canonical actual Conv node absent".into()))?;
    let recurrent = record
        .plan()
        .nodes()
        .get(nodes[1])
        .ok_or_else(|| Error::State("canonical actual recurrent node absent".into()))?;
    let (
        PlanOperation::TypedVllm08 { descriptor: cd, .. },
        PlanOperation::TypedVllm08 { descriptor: rd, .. },
    ) = (&conv.operation, &recurrent.operation)
    else {
        return Err(Error::State(
            "canonical actual paired typed GDN operators".into(),
        ));
    };
    parameters.require(cd, rd)?;
    let layer = parameters.payload.layer.index();
    let site = scope.gdn_site(layer)?;
    let ([qkvz, ba], [activated, z], [ra, rz, rb], [gated]) = (
        conv.inputs.as_slice(),
        conv.outputs.as_slice(),
        recurrent.inputs.as_slice(),
        recurrent.outputs.as_slice(),
    ) else {
        return Err(Error::State(
            "canonical exact source GDN SSA arities".into(),
        ));
    };
    if cd.call() != record.call()
        || rd.call() != record.call()
        || cd.layer() != Some(parameters.payload.layer)
        || rd.layer() != cd.layer()
        || nodes != site.nodes()
        || activated != ra
        || z != rz
        || ba != rb
        || [*qkvz, *ba, *activated, *z, *gated] != site.operands()
    {
        return Err(Error::State(
            "canonical same original source/layer/GDN producer nodes".into(),
        ));
    }
    let operands = [*qkvz, *ba, *activated, *z, *gated]
        .into_iter()
        .map(|id| values.lease(id).cloned())
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| Error::State("canonical exact five producer operands".into()))?;
    Ok(Arc::new(ModelGdnBinding {
        arithmetic,
        values,
        parameters,
        nodes,
        operands,
    }))
}

impl ColdCanonicalGdn {
    /// No current-root/state adapter is created here. All genuine source/math
    /// buffers and status H2D complete before the shared native module set loads.
    pub(crate) async fn prepare(
        scope: Arc<Canonical47ModelBinding>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        arena: Arc<SharedModelCache>,
        layer: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::super::projection_cubin::not_cancelled(cancelled)?;
        scope.require_values(&values)?;
        if layer % 4 == 3
            || parameters.payload.layer != scope.record().call().layer(layer)?
            || !Arc::ptr_eq(&parameters.payload.source, scope.source().source())
        {
            return Err(Error::State(
                "canonical cold actual GDN layer/source".into(),
            ));
        }
        let site = scope.gdn_site(layer)?;
        let operands = site
            .operands()
            .into_iter()
            .map(|id| values.lease(id).cloned())
            .collect::<Result<Vec<_>>>()?;
        let phase = phase(&scope);
        let metadata = scope
            .source()
            .source()
            .source()
            .reserve_workspace(crate::vllm_cuda_gdn::model_metadata_bound(phase)?)?;
        let inputs = GdnInputs {
            qkvz: operands[0].clone(),
            ba: operands[1].clone(),
            cache: arena.lease().clone(),
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
        let prepared = GdnPrepared::prepare_attempt_outputs(
            values.service().clone(),
            phase,
            GdnCacheSlot::One,
            inputs,
            outputs,
            cancelled,
        )
        .await?;
        require_arithmetic(&scope, layer, prepared.arithmetic())?;
        let binding = binding(
            &scope,
            values,
            parameters,
            site.nodes(),
            prepared.arithmetic().clone(),
        )?;
        let mut init = binding.values.service().begin_initialization(0)?;
        init.retain_completion_owner(scope.clone())?;
        init.retain_completion_owner(binding.clone())?;
        init.retain_completion_owner(prepared.clone())?;
        init.retain_completion_owner(arena.clone())?;
        let transport =
            ModelGdnTransportStorage::prepare(&prepared, &mut init, binding.source().source())?;
        drop(init.complete_sync().await?);
        Ok(Arc::new(Self {
            scope,
            binding,
            prepared,
            transport,
            arena,
            phase,
            _metadata: metadata,
        }))
    }

    pub(crate) fn bind_call(
        &self,
        input: &Arc<CanonicalCallInput>,
        values: Arc<ModelValues>,
        domains: Arc<BoundModelState>,
    ) -> Result<Arc<ModelGdnPrepared>> {
        let scope = input.scope();
        scope.require_values(&values)?;
        domains.require_canonical47_published(scope)?;
        self.arena.require_binding(scope, &domains)?;
        let layer = domains.layer().index();
        if phase(scope) != self.phase
            || self.binding.layer() != domains.layer()
            || self.binding.call().source() != input.call().source()
            || !Arc::ptr_eq(self.scope.source().source(), scope.source().source())
        {
            return Err(Error::State(
                "canonical cold geometry differs from actual current GDN call".into(),
            ));
        }
        require_arithmetic(scope, layer, self.prepared.arithmetic())?;
        let binding = binding(
            scope,
            values,
            self.binding.parameters.clone(),
            scope.gdn_site(layer)?.nodes(),
            self.prepared.arithmetic().clone(),
        )?;
        for (old, new) in self.binding.operands.iter().zip(&binding.operands) {
            crate::resident_program::natural_binding::same_lease(old, new)?;
        }
        let nodes = input.record().plan().nodes();
        domains.require_effect(
            nodes[binding.nodes[0]]
                .state
                .ok_or_else(|| Error::State("canonical Conv effect absent".into()))?,
            false,
        )?;
        domains.require_effect(
            nodes[binding.nodes[1]]
                .state
                .ok_or_else(|| Error::State("canonical recurrence effect absent".into()))?,
            true,
        )?;
        binding.require_prepared(&self.prepared, &domains)?;
        let metadata = input
            .source()
            .source()
            .source()
            .reserve_workspace(64 * 1024)?;
        let adapter = self
            .transport
            .bind(binding.clone(), domains.clone(), &self.prepared)?;
        Ok(Arc::new(ModelGdnPrepared {
            binding,
            prepared: self.prepared.clone(),
            adapter,
            domains,
            _arena: RetainedArena {
                _legacy: None,
                _shared: Some(self.arena.clone()),
            },
            pending: Mutex::new(None),
            _metadata: metadata,
            canonical47: Some(scope.clone()),
            short_hf: None,
        }))
    }
}

impl ColdCanonicalGdn {
    pub(crate) fn storage_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut all = self.prepared.accesses();
        all.extend(
            self.transport
                .leases()
                .into_iter()
                .map(|l| (l, DeviceAccess::Write)),
        );
        all
    }
}
