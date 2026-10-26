//! Root-free source/math preparation, then actual current-root binding per call.
use super::*;
use crate::cuda::gdn::{GdnFamilyIdentity, ModelGdnTransportStorage};
use crate::cuda::native::{
    fixed_prompt_model::{FixedPromptCallKind, FixedPromptModelBinding},
    model_cache::SharedModelCache,
};
use crate::resident_inference::native_fragment::fixed_prompt_natural::FixedPromptCallInput;

pub(crate) struct ColdFixedPromptGdn {
    scope: Arc<FixedPromptModelBinding>,
    binding: Arc<ModelGdnBinding>,
    prepared: Arc<GdnPrepared>,
    transport: Arc<ModelGdnTransportStorage>,
    arena: Arc<SharedModelCache>,
    phase: GdnPhase,
    _metadata: WindowWorkspace,
}

fn phase(scope: &FixedPromptModelBinding) -> GdnPhase {
    match scope.kind() {
        FixedPromptCallKind::Prefill => GdnPhase::FixedPrompt,
        FixedPromptCallKind::Decode { .. } => GdnPhase::CachedDecode,
    }
}

fn require_arithmetic(
    scope: &FixedPromptModelBinding,
    layer: usize,
    actual: &GdnArithmetic,
) -> Result<()> {
    let phase = phase(scope);
    actual.require_instance(phase, GdnCacheSlot::One, true)?;
    actual
        .family()
        .require_same(&GdnFamilyIdentity::original08())?;
    if scope.kind() == FixedPromptCallKind::Prefill {
        scope.require_gdn_arithmetic(
            layer,
            actual,
            crate::cuda::gdn::FixedPromptGdnArithmetic::OwnedFixedPromptSolve4,
        )?;
    }
    // CachedDecode is the exact3-stage assets factory. It has no Solve stage;
    // do not attach the FixedPrompt Solve4 declaration to a decoded invocation.
    Ok(())
}

fn binding(
    scope: &Arc<FixedPromptModelBinding>,
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
    let (Some(cd), Some(rd)) = (&record.operator(nodes[0])?, &record.operator(nodes[1])?) else {
        return Err(Error::State(
            "canonical actual paired typed GDN operators".into(),
        ));
    };
    parameters.require(cd, rd)?;
    let layer = parameters.payload.site;
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
        || !matches!(cd.kind(), SourceOperatorKind::CausalConvSplit { domain, .. } if domain.0 == parameters.payload.site)
        || !matches!(rd.kind(), SourceOperatorKind::GatedDelta { domain, .. } if domain.0 == parameters.payload.site)
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

impl ColdFixedPromptGdn {
    /// No current-root/state adapter is created here. All genuine source/math
    /// buffers and status H2D complete before the shared native module set loads.
    pub(crate) async fn prepare(
        scope: Arc<FixedPromptModelBinding>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        arena: Arc<SharedModelCache>,
        layer: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::super::projection_cubin::not_cancelled(cancelled)?;
        scope.require_values(&values)?;
        if parameters.payload.site != layer
            || !parameters.payload.source.same_issuer(scope.source())
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
            .reserve_workspace(crate::cuda::gdn::model_metadata_bound(phase)?)?;
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
        let transport = ModelGdnTransportStorage::prepare(&prepared, &mut init, binding.source())?;
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
        input: &Arc<FixedPromptCallInput>,
        values: Arc<ModelValues>,
        domains: Arc<BoundModelState>,
    ) -> Result<Arc<ModelGdnPrepared>> {
        let scope = input.scope();
        scope.require_values(&values)?;
        domains.require_fixed_prompt_published(scope)?;
        self.arena.require_binding(scope, &domains)?;
        let layer = domains.layer();
        if phase(scope) != self.phase
            || self.binding.layer() != domains.layer()
            || self.binding.call().source() != input.call().source()
            || !self.scope.source().same_issuer(scope.source())
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
        let metadata = input.source().reserve_workspace(64 * 1024)?;
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
            fixed_prompt: Some(scope.clone()),
            text: None,
        }))
    }
}

impl ColdFixedPromptGdn {
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
