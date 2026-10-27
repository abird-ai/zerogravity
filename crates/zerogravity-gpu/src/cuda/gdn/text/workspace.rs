//! One geometry's private GDN intermediates, shared across source layers.
//! Actual ModelValues run/cursor and original flight serialize their writes.
use super::*;
use crate::resident_leases::ResidentInitialization;
use std::collections::{BTreeMap, BTreeSet};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::SourceCall;
use zerogravity_core::device_events::BufferRole;

pub(crate) fn is_external(role: GdnBuffer) -> bool {
    matches!(
        role,
        GdnBuffer::PackedQkvz
            | GdnBuffer::PackedBa
            | GdnBuffer::ConvOutput
            | GdnBuffer::ZCompact
            | GdnBuffer::Gated
            | GdnBuffer::ConvWeights
            | GdnBuffer::ALog
            | GdnBuffer::DtBias
            | GdnBuffer::GatedGamma
    )
}
fn layout(geometry: TextGdnGeometry) -> Result<BTreeMap<GdnBuffer, usize>> {
    geometry.require_implementation()?;
    let mut roles = BTreeSet::from([GdnBuffer::CacheArena]);
    for &stage in geometry.phase.stages() {
        let spec = kernel_from_asset(
            geometry.phase,
            stage,
            true,
            select(geometry.implementation.config(), geometry.phase, stage)?,
            arithmetic::GdnArithmeticFamily::Text(geometry.implementation.config()),
        )?;
        roles.extend(spec.bindings.into_iter().map(|b| b.role));
    }
    // Packing destinations are part of the actual Conv subpipeline too.
    if geometry.phase.tokens() > 1 {
        roles.extend([
            GdnBuffer::AContiguous,
            GdnBuffer::BContiguous,
            GdnBuffer::ZCompact,
        ]);
    }
    Ok(roles
        .into_iter()
        .filter(|r| !is_external(*r) && *r != GdnBuffer::Rstd)
        .map(|r| (r, r.bytes(geometry.phase)))
        .collect())
}
struct CacheZero {
    bytes: Vec<u8>,
    _host: WindowWorkspace,
}
pub(crate) struct TextGdnWorkspace {
    scope: Arc<TextModelBinding>,
    values: Arc<ModelValues>,
    geometry: TextGdnGeometry,
    controls: BTreeMap<GdnBuffer, Vec<u8>>,
    leases: BTreeMap<GdnBuffer, ResidentLease>,
    _metadata: Arc<WindowWorkspace>,
    _binding_metadata: Option<WindowWorkspace>,
}
impl TextGdnWorkspace {
    pub(crate) fn allocate(
        scope: Arc<TextModelBinding>,
        values: Arc<ModelValues>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        scope.require_values(&values)?;
        init.require_service(values.service())?;
        let geometry = TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())?;
        geometry.require_implementation()?; // Invalid implementations refuse before H2D/allocation.
        let source = scope.source();
        let metadata = Arc::new(
            source.reserve_workspace(
                geometry
                    .metadata_bound()?
                    .checked_add(256 * 1024)
                    .ok_or_else(|| Error::Limit("text GDN workspace metadata".into()))?,
            )?,
        );
        let roles = layout(geometry)?;
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(scope.clone())?;
        init.retain_completion_owner(values.clone())?;
        let controls = metadata_controls(geometry)?;
        let zero_host = source
            .reserve_workspace(GdnBuffer::CacheArena.bytes(geometry.phase) * 3 + 64 * 1024)?;
        let zero = Arc::new(CacheZero {
            bytes: vec![0; GdnBuffer::CacheArena.bytes(geometry.phase)],
            _host: zero_host,
        });
        init.retain_completion_owner(zero.clone())?;
        let mut leases = BTreeMap::new();
        for (role, bytes) in roles {
            let lease = if role == GdnBuffer::CacheArena {
                init.upload_bytes(&zero.bytes, BufferRole::RecurrentState)?
            } else if let Some(control) = controls.get(&role) {
                init.upload_bytes(control, BufferRole::ImmutableWeights)?
            } else {
                init.allocate_empty(
                    bytes,
                    if matches!(role, GdnBuffer::InitialState | GdnBuffer::FinalState) {
                        BufferRole::RecurrentState
                    } else {
                        BufferRole::Scratch
                    },
                )?
            };
            leases.insert(role, lease);
        }
        if controls.keys().any(|role| !leases.contains_key(role)) {
            return Err(Error::State("text GDN unexpected control role".into()));
        }
        let result = Arc::new(Self {
            scope,
            values,
            geometry,
            controls,
            leases,
            _metadata: metadata,
            _binding_metadata: None,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    /// Rebind the SAME physical private roles to the fresh original record.
    /// The matching actual ValueStorageRun/cursor is still required per partition.
    pub(crate) fn bind_call(
        &self,
        scope: Arc<TextModelBinding>,
        values: Arc<ModelValues>,
    ) -> Result<Arc<Self>> {
        scope.require_values(&values)?;
        if Arc::ptr_eq(&scope, &self.scope)
            || Arc::ptr_eq(&values, &self.values)
            || !scope.source().same_issuer(self.scope.source())
            || !Arc::ptr_eq(values.service(), self.values.service())
            || TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())? != self.geometry
        {
            return Err(Error::State(
                "text GDN fresh record/same actual physical geometry".into(),
            ));
        }
        let metadata = scope.source().reserve_workspace(64 * 1024)?;
        let result = Arc::new(Self {
            scope,
            values,
            geometry: self.geometry,
            controls: self.controls.clone(),
            leases: self.leases.clone(),
            _metadata: self._metadata.clone(),
            _binding_metadata: Some(metadata),
        });
        result.require_scope(&result.scope, &result.values)?;
        Ok(result)
    }
    pub(crate) fn cache(&self) -> Result<&ResidentLease> {
        self.leases
            .get(&GdnBuffer::CacheArena)
            .ok_or_else(|| Error::State("text private GDN cache absent".into()))
    }
    pub(super) fn require_scope(
        &self,
        scope: &Arc<TextModelBinding>,
        values: &Arc<ModelValues>,
    ) -> Result<()> {
        scope.require_values(values)?;
        if !Arc::ptr_eq(scope, &self.scope)
            || !Arc::ptr_eq(values, &self.values)
            || TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())? != self.geometry
        {
            return Err(Error::State(
                "text private GDN same physical geometry/source scope".into(),
            ));
        }
        for (&role, lease) in &self.leases {
            if lease.bytes() != role.bytes(self.geometry.phase)
                || !Arc::ptr_eq(lease.service(), values.service())
                || lease.source_identity() != scope.record().call().source()
                || lease.service_generation() != values.service().generation()
            {
                return Err(Error::State(
                    "text private GDN actual lease/source/extent".into(),
                ));
            }
        }
        Ok(())
    }
    pub(in crate::cuda::gdn) fn control(
        &self,
        role: GdnBuffer,
        bytes: &[u8],
    ) -> Result<ResidentLease> {
        if self.controls.get(&role).map(Vec::as_slice) != Some(bytes) {
            return Err(Error::State(
                "text GDN exact source control bytes changed".into(),
            ));
        }
        let lease = self.lease(role)?;
        if !lease.is_initialized() || lease.role() != BufferRole::ImmutableWeights {
            return Err(Error::State(
                "text original GDN control initialization".into(),
            ));
        }
        Ok(lease)
    }
    pub(in crate::cuda::gdn) fn lease(&self, role: GdnBuffer) -> Result<ResidentLease> {
        self.leases
            .get(&role)
            .cloned()
            .ok_or_else(|| Error::State("text shared private GDN role absent".into()))
    }
    pub(in crate::cuda::gdn) fn optional_private(
        &self,
        role: GdnBuffer,
    ) -> Result<Option<ResidentLease>> {
        // Keep genuine per-layer Rstd outputs separate. Their original allocation
        // supplies a nonempty original init; they remain available for diagnostics.
        if role == GdnBuffer::Rstd {
            return Ok(None);
        }
        if is_external(role) {
            return Err(Error::State(
                "text external role cannot alias private GDN workspace".into(),
            ));
        }
        self.lease(role).map(Some)
    }
}
fn metadata_controls(geometry: TextGdnGeometry) -> Result<BTreeMap<GdnBuffer, Vec<u8>>> {
    metadata(geometry.phase, geometry.slot).map(|values| values.into_iter().collect())
}

impl TextGdnWorkspace {
    pub(crate) fn immutable_controls(&self) -> Vec<(ResidentLease, &[u8])> {
        self.controls
            .iter()
            .map(|(r, b)| (self.leases[r].clone(), b.as_slice()))
            .collect()
    }
    pub(crate) fn storage_accesses(
        &self,
    ) -> Vec<(ResidentLease, zerogravity_core::device_events::DeviceAccess)> {
        self.leases
            .values()
            .map(|l| {
                (
                    l.clone(),
                    zerogravity_core::device_events::DeviceAccess::Write,
                )
            })
            .collect()
    }
}

impl TextGdnWorkspace {
    pub(crate) fn allocation_layout(
        call: &SourceCall,
        implementation: TextGdnImplementation,
    ) -> Result<Vec<usize>> {
        Ok(layout(TextGdnGeometry::for_call(call, implementation)?)?
            .into_values()
            .collect())
    }
    pub(crate) fn host_layout(
        call: &SourceCall,
        implementation: TextGdnImplementation,
    ) -> Result<(usize, usize)> {
        let geometry = TextGdnGeometry::for_call(call, implementation)?;
        geometry.require_implementation()?;
        let retained = geometry
            .metadata_bound()?
            .checked_add(256 * 1024)
            .ok_or_else(|| Error::Limit("text private GDN metadata overflow".into()))?;
        let initial = GdnBuffer::CacheArena
            .bytes(geometry.phase)
            .checked_mul(3)
            .and_then(|n| n.checked_add(64 * 1024))
            .ok_or_else(|| Error::Limit("text actual cache init staging overflow".into()))?;
        Ok((retained, initial))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_private_overwritten_intermediates_can_enter_shared_workspace() {
        for role in [
            GdnBuffer::PackedQkvz,
            GdnBuffer::PackedBa,
            GdnBuffer::ConvOutput,
            GdnBuffer::ZCompact,
            GdnBuffer::Gated,
            GdnBuffer::ConvWeights,
            GdnBuffer::ALog,
            GdnBuffer::DtBias,
            GdnBuffer::GatedGamma,
        ] {
            assert!(is_external(role));
        }
        for role in [
            GdnBuffer::Q,
            GdnBuffer::K,
            GdnBuffer::V,
            GdnBuffer::GRaw,
            GdnBuffer::Kkt,
            GdnBuffer::Inverse,
            GdnBuffer::W,
            GdnBuffer::U,
            GdnBuffer::HChunks,
            GdnBuffer::Core,
            GdnBuffer::InitialState,
            GdnBuffer::FinalState,
            GdnBuffer::CacheArena,
        ] {
            assert!(!is_external(role));
        }
    }
    #[test]
    fn selected_geometry_never_allocates_source_or_public_roles_in_private_pool() -> Result<()> {
        for prompt_tokens in [47, 57, 62, 67, 72] {
            let g = TextGdnGeometry {
                implementation: fixture_implementation(0, prompt_tokens, false)?,
                phase: GdnPhase::TextPrefill(prompt_tokens),
                slot: GdnCacheSlot::One,
            };
            let actual = layout(g)?;
            assert!(
                actual
                    .keys()
                    .all(|r| !is_external(*r) && *r != GdnBuffer::Rstd)
            );
            assert_eq!(actual[&GdnBuffer::CacheArena], 5570560);
            assert_eq!(actual[&GdnBuffer::Core], prompt_tokens * 4096);
            assert_eq!(
                actual[&GdnBuffer::HChunks],
                prompt_tokens.div_ceil(64) * 524288
            );
        }
        Ok(())
    }
}
