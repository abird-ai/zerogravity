//! Source-owned whole-call selection; missing actual tuples fail before H2D.
use super::*;
use crate::cuda::native::{
    model_gdn::ModelGdnParameters, model_values::ModelValues, text_model::TextModelBinding,
};
use crate::resident_leases::{ResidentLease, ResidentService};
pub use config::{
    GdnKernelConfig, GdnKktKernel, GdnOutputKernel, GdnRecomputeKernel, GdnStateKernel,
    TextGdnImplementation,
};
#[cfg(test)]
pub(crate) use observed::GdnKernelRecipe;
use zerogravity_core::accelerator::program::{
    PlanOperation, SourceCall, SourceCallMode, SourceConvGeometry, SourceOperatorKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TextGdnGeometry {
    pub(crate) phase: GdnPhase,
    pub(crate) slot: GdnCacheSlot,
    pub(crate) implementation: TextGdnImplementation,
}
impl TextGdnGeometry {
    pub(crate) fn for_call(
        call: &SourceCall,
        implementation: TextGdnImplementation,
    ) -> Result<Self> {
        if call.rows().len() != implementation.input_tokens()
            || (call.mode() == SourceCallMode::Decode) != implementation.is_decode()
        {
            return Err(Error::State(
                "actual GDN source call versus explicit implementation".into(),
            ));
        }
        let phase = match call.mode() {
            SourceCallMode::Prefill => GdnPhase::TextPrefill(call.rows().len()),
            SourceCallMode::Decode => GdnPhase::CachedDecode,
        };
        implementation.require_phase(phase)?;
        Ok(Self {
            phase,
            slot: GdnCacheSlot::One,
            implementation,
        })
    }
    pub(crate) fn require_catalog(self) -> Result<()> {
        self.slot.validate(self.phase)?;
        for &stage in self.phase.stages() {
            select(self.implementation.config(), self.phase, stage)?;
        }
        Ok(())
    }
    pub(crate) fn metadata_bound(self) -> Result<usize> {
        self.require_catalog()?;
        self.phase.stages().iter().try_fold(
            256usize * 1024 + 6 * PinnedKernel::KEY_METADATA_BYTES,
            |sum, &stage| {
                let source = select(self.implementation.config(), self.phase, stage)?.source;
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

/// Checked parameterized stages are prepared from an explicit coherent config.
/// The exact historical recipe remains only the whole-call admission boundary;
/// no executable prefill stage reaches the retained fixture selector.
pub(super) fn select(
    config: Option<GdnKernelConfig>,
    phase: GdnPhase,
    stage: GdnStage,
) -> Result<assets::Asset> {
    if matches!(phase, GdnPhase::TextPrefill(_)) {
        let config = GdnKernelConfig::require(config)?;
        return match stage {
            GdnStage::Conv => Ok(parameterized_conv::prepare(
                phase,
                GdnCacheSlot::One,
                SourceConvGeometry {
                    qkv_channels: 6144,
                    z_channels: 2048,
                    coefficient_channels: 32,
                    kernel: 4,
                },
            )?
            .asset),
            GdnStage::Gated => Ok(parameterized_gated::prepare(phase, GdnCacheSlot::One)?.asset),
            GdnStage::Prep | GdnStage::Cumsum | GdnStage::Solve => {
                Ok(parameterized_prefill::prepare(phase, GdnCacheSlot::One, stage)?.asset)
            }
            GdnStage::Kkt | GdnStage::Recompute | GdnStage::H | GdnStage::Output => Ok(
                parameterized_prefill_variants::prepare(config, phase, GdnCacheSlot::One, stage)?
                    .asset,
            ),
            _ => Err(Error::Unsupported("configured prefill GDN stage".into())),
        };
    }
    if phase == GdnPhase::CachedDecode {
        if config.is_some() {
            return Err(Error::Unsupported(
                "cached GDN cannot carry a prefill config".into(),
            ));
        }
        return match stage {
            GdnStage::Conv | GdnStage::Decode => {
                Ok(parameterized_cached::prepare(phase, GdnCacheSlot::One, stage)?.asset)
            }
            GdnStage::Gated => Ok(parameterized_gated::prepare(phase, GdnCacheSlot::One)?.asset),
            _ => Err(Error::Unsupported("original cached GDN stage".into())),
        };
    }
    Err(Error::Unsupported("configured text GDN stage".into()))
}

mod config;
mod observed;
mod parameterized_cached;
mod parameterized_conv;
mod parameterized_gated;
mod parameterized_prefill;
mod parameterized_prefill_variants;
#[cfg(test)]
pub(super) fn fixture_implementation(
    catalogue: usize,
    input_tokens: usize,
    is_decode: bool,
) -> Result<TextGdnImplementation> {
    use GdnKktKernel::{Bk32W4S3, Bk32W8S3, Bk64W8S4};
    use GdnOutputKernel::{Bk32Bv32W4S2, Bk32Bv32W4S3};
    use GdnRecomputeKernel::{Bk64Bv64W8S3, Bk64Bv64W8S4};
    use GdnStateKernel::{Bv32W4S2, Bv64W4S2};

    let recipe = GdnKernelRecipe::fixture(catalogue, input_tokens, is_decode)?;
    if is_decode {
        return TextGdnImplementation::cached(recipe.recipe_sha256());
    }
    let stages = match (catalogue, input_tokens) {
        (0, 47) | (1, 57) => (Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S2),
        (0, 57 | 62) => (Bk32W8S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S3),
        (0, 67) => (Bk64W8S4, Bk64Bv64W8S3, Bv32W4S2, Bk32Bv32W4S3),
        (0, 72) | (1, 62 | 67) => (Bk32W8S3, Bk64Bv64W8S4, Bv32W4S2, Bk32Bv32W4S2),
        (1, 47) => (Bk32W4S3, Bk64Bv64W8S4, Bv64W4S2, Bk32Bv32W4S3),
        (1, 72) => (Bk32W4S3, Bk64Bv64W8S4, Bv32W4S2, Bk32Bv32W4S3),
        _ => return Err(Error::Unsupported("test GDN implementation cell".into())),
    };
    TextGdnImplementation::prefill(
        recipe.recipe_sha256(),
        input_tokens,
        GdnKernelConfig::from_stages(stages.0, stages.1, stages.2, stages.3)?,
    )
}

pub(super) fn text_source_contract(
    config: Option<GdnKernelConfig>,
) -> (
    Option<GdnKernelConfig>,
    Option<[&'static str; 4]>,
    &'static [&'static str],
) {
    const PREFILL_COMMON: &[&str] = &[
        parameterized_conv::SOURCE,
        parameterized_prefill::PREP_SOURCE,
        parameterized_prefill::CUMSUM_SOURCE,
        parameterized_prefill::SOLVE_SOURCE,
        parameterized_gated::PREFILL_SOURCE,
    ];
    const CACHED_COMMON: &[&str] = &[
        parameterized_cached::CONV_SOURCE,
        parameterized_cached::DECODE_SOURCE,
        parameterized_gated::CACHED_SOURCE,
    ];
    match config {
        Some(config) => (
            Some(config),
            Some(parameterized_prefill_variants::source_contract(config)),
            PREFILL_COMMON,
        ),
        None => (None, None, CACHED_COMMON),
    }
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
        let geometry = TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())?;
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
        GdnFamilyIdentity::text(self.geometry.implementation.config())
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
        let family = arithmetic::GdnArithmeticFamily::Text(self.geometry.implementation.config());
        let parameterized = matches!(phase, GdnPhase::TextPrefill(_))
            && matches!(
                stage,
                GdnStage::Conv
                    | GdnStage::Prep
                    | GdnStage::Cumsum
                    | GdnStage::Kkt
                    | GdnStage::Solve
                    | GdnStage::Recompute
                    | GdnStage::H
                    | GdnStage::Output
                    | GdnStage::Gated
            )
            || phase == GdnPhase::CachedDecode;
        if !parameterized {
            return kernel_from_asset(
                phase,
                stage,
                true,
                select(self.geometry.implementation.config(), phase, stage)?,
                family,
            );
        }
        let (asset, scalars) = match (phase, stage) {
            (GdnPhase::TextPrefill(_), GdnStage::Conv) => {
                let definition =
                    parameterized_conv::prepare(phase, self.geometry.slot, self.conv_geometry)?;
                (definition.asset, definition.scalars)
            }
            (GdnPhase::TextPrefill(_), GdnStage::Gated)
            | (GdnPhase::CachedDecode, GdnStage::Gated) => {
                let definition = parameterized_gated::prepare(phase, self.geometry.slot)?;
                (definition.asset, definition.scalars)
            }
            (GdnPhase::TextPrefill(_), _) => {
                let definition = match stage {
                    GdnStage::Prep | GdnStage::Cumsum | GdnStage::Solve => {
                        parameterized_prefill::prepare(phase, self.geometry.slot, stage)?
                    }
                    GdnStage::Kkt | GdnStage::Recompute | GdnStage::H | GdnStage::Output => {
                        parameterized_prefill_variants::prepare(
                            GdnKernelConfig::require(self.geometry.implementation.config())?,
                            phase,
                            self.geometry.slot,
                            stage,
                        )?
                    }
                    _ => return Err(Error::Unsupported("configured prefill GDN stage".into())),
                };
                (definition.asset, definition.scalars)
            }
            (GdnPhase::CachedDecode, GdnStage::Conv | GdnStage::Decode) => {
                let definition = parameterized_cached::prepare(phase, self.geometry.slot, stage)?;
                (definition.asset, definition.scalars)
            }
            _ => return Err(Error::Unsupported("parameterized text GDN stage".into())),
        };
        let spec = kernel_from_asset_with_scalars(phase, stage, true, asset, family, scalars)?;
        match stage {
            GdnStage::Gated => {
                parameterized_gated::require_bindings(phase, scalars, &spec.bindings)?
            }
            GdnStage::Conv | GdnStage::Decode if phase == GdnPhase::CachedDecode => {
                parameterized_cached::require_bindings(phase, stage, scalars, &spec.bindings)?
            }
            _ => {}
        }
        Ok(spec)
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
