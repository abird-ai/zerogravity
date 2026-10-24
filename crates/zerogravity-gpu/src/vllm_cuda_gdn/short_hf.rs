//! Source-owned whole-call selection; missing actual tuples fail before H2D.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::vllm_cuda_qualification::{
    model_gdn::ModelGdnParameters, model_values::ModelValues, short_hf_model::ShortHfModelBinding,
};
use zerogravity_core::accelerator::program::{FullModelProfile, ShortHfCall, VllmBf16Mode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShortHfGdnGeometry {
    pub(crate) phase: VllmGdnPhase,
    pub(crate) slot: GdnCacheSlot,
    pub(crate) profile: FullModelProfile,
}
impl ShortHfGdnGeometry {
    pub(crate) fn for_call(call: &ShortHfCall) -> Result<Self> {
        call.validate()?;
        Ok(Self {
            phase: match call.span().mode() {
                VllmBf16Mode::Prefill => VllmGdnPhase::ShortHfPrefill(call.span().workload()),
                VllmBf16Mode::Decode => VllmGdnPhase::CachedDecode,
            },
            slot: GdnCacheSlot::One,
            profile: call.model_profile(),
        })
    }
    pub(crate) fn require_catalog(self) -> Result<()> {
        self.slot.validate(self.phase)?;
        for &stage in self.phase.stages() {
            select(self.profile, self.phase, stage)?;
        }
        Ok(())
    }
    pub(crate) fn metadata_bound(self) -> Result<usize> {
        self.require_catalog()?;
        self.phase.stages().iter().try_fold(
            256usize * 1024 + 6 * PinnedKernel::KEY_METADATA_BYTES,
            |sum, &stage| {
                let source = select(self.profile, self.phase, stage)?.source;
                sum.checked_add(
                    source
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| Error::Limit("short-HF GDN code extent".into()))?,
                )
                .and_then(|n| n.checked_add(4096 + PinnedKernel::KEY_METADATA_BYTES))
                .ok_or_else(|| Error::Limit("short-HF GDN source metadata bound".into()))
            },
        )
    }
}

/// Source/config/scalars are from the actual observed original per-cell run.
/// Missing tuples refuse; accepted owned legacy profiles keep their old selectors.
pub(super) fn select(
    profile: FullModelProfile,
    phase: VllmGdnPhase,
    stage: GdnStage,
) -> Result<assets::Asset> {
    observed::select(profile, phase, stage)
}

mod observed;
pub(super) fn observed_catalogue() -> &'static [&'static str] {
    observed::CATALOGUE
}

/// Holds the genuine model/role/producer owners before private storage begins.
/// It has no StateRoot, expected tensor, completion or publication authority.
pub(crate) struct ShortHfGdnCode {
    scope: Arc<ShortHfModelBinding>,
    values: Arc<ModelValues>,
    parameters: Arc<ModelGdnParameters>,
    layer: usize,
    geometry: ShortHfGdnGeometry,
    workspace: Arc<ShortHfGdnWorkspace>,
    operands: [ResidentLease; 5],
    _metadata: zerogravity_artifacts::source_weights::WindowWorkspace,
}
impl ShortHfGdnCode {
    pub(super) fn require_same_recipe(&self, old: &Self) -> Result<()> {
        if self.geometry != old.geometry || self.layer != old.layer {
            return Err(Error::State(
                "GDN immutable kernel geometry/site changed".into(),
            ));
        }
        // Both recipes use this same closed selector. No per-call/source-root
        // operand enters kernel construction; those remain fresh binding checks.
        self.geometry.require_catalog()
    }
    pub(crate) fn bind(
        scope: Arc<ShortHfModelBinding>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        layer: usize,
        workspace: Arc<ShortHfGdnWorkspace>,
    ) -> Result<Arc<Self>> {
        scope.require_values(&values)?;
        let geometry = ShortHfGdnGeometry::for_call(scope.call())?;
        // Complete tuple membership before any private allocation or upload.
        geometry.require_catalog()?;
        workspace.require_scope(&scope, &values)?;
        let site = scope.gdn_site(layer)?;
        parameters.require_short_hf(&scope, layer)?;
        let operands = site
            .operands
            .into_iter()
            .map(|id| values.lease(id).cloned())
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("short-HF exact five GDN operands".into()))?;
        let metadata = scope
            .source()
            .source()
            .source()
            .reserve_workspace(64 * 1024)?;
        Ok(Arc::new(Self {
            scope,
            values,
            parameters,
            layer,
            geometry,
            workspace,
            operands,
            _metadata: metadata,
        }))
    }
    pub(crate) fn bind_call(
        &self,
        scope: Arc<ShortHfModelBinding>,
        values: Arc<ModelValues>,
    ) -> Result<Arc<Self>> {
        let workspace = self.workspace.bind_call(scope.clone(), values.clone())?;
        Self::bind(
            scope,
            values,
            self.parameters.clone(),
            self.layer,
            workspace,
        )
    }
    pub(crate) fn geometry(&self) -> ShortHfGdnGeometry {
        self.geometry
    }
    pub(super) fn workspace(&self) -> &Arc<ShortHfGdnWorkspace> {
        &self.workspace
    }
    pub(crate) fn scope(&self) -> &Arc<ShortHfModelBinding> {
        &self.scope
    }
    pub(crate) fn values(&self) -> &Arc<ModelValues> {
        &self.values
    }
    pub(crate) fn parameters(&self) -> &Arc<ModelGdnParameters> {
        &self.parameters
    }
    pub(crate) fn layer(&self) -> usize {
        self.layer
    }
    pub(crate) fn operands(&self) -> &[ResidentLease; 5] {
        &self.operands
    }
    pub(super) fn family(&self) -> GdnFamilyIdentity {
        GdnFamilyIdentity::short_hf(self.geometry.profile)
    }
    pub(super) fn require(
        &self,
        service: &Arc<ResidentService>,
        phase: VllmGdnPhase,
    ) -> Result<()> {
        self.scope.require_values(&self.values)?;
        self.workspace.require_scope(&self.scope, &self.values)?;
        self.parameters.require_short_hf(&self.scope, self.layer)?;
        if !Arc::ptr_eq(service, self.values.service()) || phase != self.geometry.phase {
            return Err(Error::State(
                "short-HF GDN same source/geometry/service".into(),
            ));
        }
        self.geometry.require_catalog()
    }
    pub(super) fn kernel(&self, phase: VllmGdnPhase, stage: GdnStage) -> Result<LaunchSpec> {
        if phase != self.geometry.phase {
            return Err(Error::State("short-HF GDN phase changed".into()));
        }
        kernel_from_asset(
            phase,
            stage,
            true,
            select(self.geometry.profile, phase, stage)?,
            arithmetic::GdnArithmeticFamily::ShortHf(self.geometry.profile),
        )
    }
    pub(crate) fn inputs(&self, cache: ResidentLease) -> GdnInputs {
        let p = self.parameters.short_hf_leases();
        GdnInputs {
            qkvz: self.operands[0].clone(),
            ba: self.operands[1].clone(),
            cache,
            conv_weights: p[0].clone(),
            a_log: p[1].clone(),
            dt_bias: p[2].clone(),
            gamma: p[3].clone(),
        }
    }
    pub(crate) fn outputs(&self) -> GdnTypedOutputs {
        GdnTypedOutputs {
            activated: self.operands[2].clone(),
            z: self.operands[3].clone(),
            gated: self.operands[4].clone(),
        }
    }
    pub(super) fn require_inputs(
        &self,
        service: &Arc<ResidentService>,
        i: &GdnInputs,
        o: &GdnTypedOutputs,
    ) -> Result<()> {
        self.require(service, self.geometry.phase)?;
        let p = self.parameters.short_hf_leases();
        for (actual, expected) in [
            &i.qkvz,
            &i.ba,
            &o.activated,
            &o.z,
            &o.gated,
            &i.conv_weights,
            &i.a_log,
            &i.dt_bias,
            &i.gamma,
        ]
        .into_iter()
        .zip(self.operands.iter().chain(p.iter()))
        {
            crate::resident_program::natural_binding::same_lease(actual, expected)?;
        }
        if !Arc::ptr_eq(i.cache.service(), service)
            || i.cache.source_identity() != service.source_identity()
            || i.cache.service_generation() != service.generation()
            || i.cache.bytes() != GdnBuffer::CacheArena.bytes(self.geometry.phase)
            || !i.cache.is_initialized()
            || i.cache.id() != self.workspace.lease(GdnBuffer::CacheArena)?.id()
            || self
                .operands
                .iter()
                .chain(p.iter())
                .any(|l| l.id() == i.cache.id())
        {
            return Err(Error::State(
                "short-HF original initialized cache/source/nonalias".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require_arithmetic(&self, actual: &GdnArithmetic) -> Result<()> {
        actual.require_instance(self.geometry.phase, self.geometry.slot, true)?;
        self.family().require_same(actual.family())
    }
}

#[cfg(test)]
mod tests;

mod workspace;
pub(crate) use workspace::ShortHfGdnWorkspace;

/// Persistent public/source/control/Rstd storage only; pooled private values are
/// not a per-layer late observation even though their physical owner is alive.
pub(super) fn retained_history(role: GdnBuffer) -> bool {
    workspace::is_external(role)
        || matches!(
            role,
            GdnBuffer::Rstd
                | GdnBuffer::CacheIndices
                | GdnBuffer::HasInitial
                | GdnBuffer::QueryStarts
                | GdnBuffer::ConvBatch
                | GdnBuffer::ConvChunkOffsets
                | GdnBuffer::ChunkIndices
                | GdnBuffer::ChunkOffsets
        )
}
