//! Source-owned whole-call selection; missing actual tuples fail before H2D.
use super::*;
use crate::cuda::native::{
    model_gdn::ModelGdnParameters, model_values::ModelValues, text_model::TextModelBinding,
};
use crate::resident_leases::{ResidentLease, ResidentService};
pub use observed::GdnKernelRecipe;
use zerogravity_core::accelerator::program::{
    PlanOperation, SourceCall, SourceCallMode, SourceConvGeometry, SourceOperatorKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TextGdnGeometry {
    pub(crate) phase: GdnPhase,
    pub(crate) slot: GdnCacheSlot,
    pub(crate) recipe: GdnKernelRecipe,
}
impl TextGdnGeometry {
    pub(crate) fn for_call(call: &SourceCall, recipe: GdnKernelRecipe) -> Result<Self> {
        if call.rows().len() != recipe.input_tokens()
            || (call.mode() == SourceCallMode::Decode) != recipe.is_decode()
        {
            return Err(Error::State(
                "actual GDN source call versus exact reference recipe".into(),
            ));
        }
        Ok(Self {
            phase: match call.mode() {
                SourceCallMode::Prefill => GdnPhase::TextPrefill(call.rows().len()),
                SourceCallMode::Decode => GdnPhase::CachedDecode,
            },
            slot: GdnCacheSlot::One,
            recipe,
        })
    }
    pub(crate) fn require_catalog(self) -> Result<()> {
        self.slot.validate(self.phase)?;
        for &stage in self.phase.stages() {
            select(self.recipe, self.phase, stage)?;
        }
        Ok(())
    }
    pub(crate) fn metadata_bound(self) -> Result<usize> {
        self.require_catalog()?;
        self.phase.stages().iter().try_fold(
            256usize * 1024 + 6 * PinnedKernel::KEY_METADATA_BYTES,
            |sum, &stage| {
                let source = select(self.recipe, self.phase, stage)?.source;
                sum.checked_add(
                    source
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| Error::Limit("text GDN code extent".into()))?,
                )
                .and_then(|n| n.checked_add(4096 + PinnedKernel::KEY_METADATA_BYTES))
                .ok_or_else(|| Error::Limit("text GDN source metadata bound".into()))
            },
        )
    }
}

/// Prefill Conv is one checked parameterized body. Remaining finite stages keep
/// their observed numerical configurations until separately parameterized.
pub(super) fn select(
    recipe: GdnKernelRecipe,
    phase: GdnPhase,
    stage: GdnStage,
) -> Result<assets::Asset> {
    if stage == GdnStage::Conv && matches!(phase, GdnPhase::TextPrefill(_)) {
        if recipe.input_tokens() != phase.tokens() || recipe.is_decode() {
            return Err(Error::Unsupported(
                "exact recorded GDN recipe/phase/count".into(),
            ));
        }
        return Ok(parameterized_conv::prepare(
            phase,
            GdnCacheSlot::One,
            SourceConvGeometry {
                qkv_channels: 6144,
                z_channels: 2048,
                coefficient_channels: 32,
                kernel: 4,
            },
        )?
        .asset);
    }
    observed::select(recipe, phase, stage)
}

mod observed;
mod parameterized_conv;
pub(super) fn observed_catalogue() -> &'static [&'static str] {
    observed::CATALOGUE
}

/// Holds the genuine model/role/producer owners before private storage begins.
/// It has no StateRoot, expected tensor, completion or publication authority.
pub(crate) struct TextGdnCode {
    scope: Arc<TextModelBinding>,
    values: Arc<ModelValues>,
    parameters: Arc<ModelGdnParameters>,
    layer: usize,
    geometry: TextGdnGeometry,
    conv_geometry: SourceConvGeometry,
    workspace: Arc<TextGdnWorkspace>,
    operands: [ResidentLease; 5],
    _metadata: zerogravity_artifacts::source_weights::WindowWorkspace,
}
impl TextGdnCode {
    pub(super) fn require_same_recipe(&self, old: &Self) -> Result<()> {
        if self.geometry != old.geometry
            || self.conv_geometry != old.conv_geometry
            || self.layer != old.layer
        {
            return Err(Error::State(
                "GDN immutable kernel geometry/site changed".into(),
            ));
        }
        // Both recipes use this same closed selector. No per-call/source-root
        // operand enters kernel construction; those remain fresh binding checks.
        self.geometry.require_catalog()
    }
    pub(crate) fn bind(
        scope: Arc<TextModelBinding>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        layer: usize,
        workspace: Arc<TextGdnWorkspace>,
    ) -> Result<Arc<Self>> {
        scope.require_values(&values)?;
        let geometry = TextGdnGeometry::for_call(scope.call(), *scope.gdn_recipe())?;
        // Complete tuple membership before any private allocation or upload.
        geometry.require_catalog()?;
        workspace.require_scope(&scope, &values)?;
        let site = scope.gdn_site(layer)?;
        let node = scope
            .record()
            .plan()
            .nodes()
            .get(site.nodes[0])
            .ok_or_else(|| Error::State("text GDN convolution source node absent".into()))?;
        let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
            return Err(Error::State(
                "text GDN convolution source operator absent".into(),
            ));
        };
        let SourceOperatorKind::CausalConvSplit {
            geometry: conv_geometry,
            ..
        } = descriptor.kind()
        else {
            return Err(Error::State(
                "text GDN convolution source geometry absent".into(),
            ));
        };
        if descriptor.call() != scope.call() {
            return Err(Error::State(
                "text GDN convolution source call differs".into(),
            ));
        }
        parameters.require_text(&scope, layer)?;
        let operands = site
            .operands
            .into_iter()
            .map(|id| values.lease(id).cloned())
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("text exact five GDN operands".into()))?;
        let metadata = scope.source().reserve_workspace(64 * 1024)?;
        Ok(Arc::new(Self {
            scope,
            values,
            parameters,
            layer,
            geometry,
            conv_geometry,
            workspace,
            operands,
            _metadata: metadata,
        }))
    }
    pub(crate) fn bind_call(
        &self,
        scope: Arc<TextModelBinding>,
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
    pub(crate) fn geometry(&self) -> TextGdnGeometry {
        self.geometry
    }
    pub(super) fn workspace(&self) -> &Arc<TextGdnWorkspace> {
        &self.workspace
    }
    pub(crate) fn scope(&self) -> &Arc<TextModelBinding> {
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
        GdnFamilyIdentity::text(self.geometry.recipe)
    }
    pub(super) fn require(&self, service: &Arc<ResidentService>, phase: GdnPhase) -> Result<()> {
        self.scope.require_values(&self.values)?;
        self.workspace.require_scope(&self.scope, &self.values)?;
        self.parameters.require_text(&self.scope, self.layer)?;
        if !Arc::ptr_eq(service, self.values.service()) || phase != self.geometry.phase {
            return Err(Error::State("text GDN same source/geometry/service".into()));
        }
        self.geometry.require_catalog()
    }
    pub(super) fn kernel(&self, phase: GdnPhase, stage: GdnStage) -> Result<LaunchSpec> {
        if phase != self.geometry.phase {
            return Err(Error::State("text GDN phase changed".into()));
        }
        let family = arithmetic::GdnArithmeticFamily::Text(self.geometry.recipe);
        if stage == GdnStage::Conv && matches!(phase, GdnPhase::TextPrefill(_)) {
            let definition =
                parameterized_conv::prepare(phase, self.geometry.slot, self.conv_geometry)?;
            kernel_from_asset_with_scalars(
                phase,
                stage,
                true,
                definition.asset,
                family,
                definition.scalars,
            )
        } else {
            kernel_from_asset(
                phase,
                stage,
                true,
                select(self.geometry.recipe, phase, stage)?,
                family,
            )
        }
    }
    pub(crate) fn inputs(&self, cache: ResidentLease) -> GdnInputs {
        let p = self.parameters.text_leases();
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
        let p = self.parameters.text_leases();
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
                "text original initialized cache/source/nonalias".into(),
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
pub(crate) use workspace::TextGdnWorkspace;

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
