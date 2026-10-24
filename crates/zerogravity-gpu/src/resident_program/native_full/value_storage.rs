//! Source-issued full graph, original actors and original dispatch cursor.
use super::*;
use crate::vllm_cuda_input_rms::FullModelRmsScratch;
use crate::vllm_cuda_qualification::{
    model_values::bounded::{StorageBuilder, StorageMode, StorageRecipe},
    short_hf_model::ShortHfModelBinding,
};
use zerogravity_core::accelerator::program::ValueStorageLimits;

/// One pre-admitted maximum actual diagnostic cohort. All six cold geometries
/// run under the SAME original workspace lock; returned Pending drops raw D2H.
/// No separate dispatcher, uncharged capacity or public input authority.
pub(crate) struct ShortHfReadbackAdmission {
    source: Arc<OriginalWindowModelSource>,
    service: Arc<ResidentService>,
    profile: zerogravity_core::accelerator::program::FullModelProfile,
    maximum: usize,
    tickets: Arc<Vec<WindowWorkspace>>,
}
enum StorageReadback<'a> {
    Shared(Arc<ShortHfReadbackAdmission>),
    SharedViews(Arc<ShortHfReadbackAdmission>, Arc<crate::vllm_cuda_qualification::model_values::bounded::workspace::ValueWorkspaceGeometry>),
    Existing(
        &'a Arc<FullModelProgram>,
        &'a ResidentPreflight,
        Option<&'a Arc<crate::resident_leases::TiedWeight>>,
    ),
}
fn diagnostic_bytes(public: usize, states: usize, hidden: usize, rows: usize) -> Result<usize> {
    let row = rows
        .checked_mul(hidden)
        .ok_or_else(|| Error::Limit("short-HF readback row overflow".into()))?;
    let dual = checked_sum([
        hidden
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("short-HF gamma extent".into()))?,
        row.checked_mul(8)
            .ok_or_else(|| Error::Limit("short-HF dual readback overflow".into()))?,
    ])?;
    // 279 public values already include ordinary Embedding/output. Each actual
    // dual still reads GammaRaw/SumF32/normalized/residual, including duplicates.
    checked_sum([
        public,
        states
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("short-HF both roots overflow".into()))?,
        dual.checked_mul(48)
            .ok_or_else(|| Error::Limit("short-HF48 readback overflow".into()))?,
        hidden * 2,
        19 * 4,
        3 * 513 * 8,
        6 * 4,
        12 + 4,
    ])
}
pub(super) fn actual_diagnostic_bytes(scope: &ShortHfModelBinding) -> Result<usize> {
    let record = scope.record();
    let mut public = 0usize;
    let mut retained = 0usize;
    for (index, value) in record.plan().values().iter().enumerate() {
        let overwritten=record.plan().vllm_value_provenance(ValueId(u32::try_from(index)
            .map_err(|_|Error::Limit("short-HF actual ValueId extent".into()))?))?
            .is_some_and(|tag|matches!(tag.origin(),
                zerogravity_core::accelerator::program::VllmValueOrigin::FullModel(op)
                if matches!(op.kind(),Vllm08Kind::Projection(Vllm08Projection::GateUp)|Vllm08Kind::SiluAndMul)));
        if overwritten {
            continue;
        }
        retained += 1;
        public = checked_sum([
            public,
            value
                .capacity_elements()?
                .checked_mul(2)
                .ok_or_else(|| Error::Limit("short-HF retained readback extent".into()))?,
        ])?;
    }
    if retained != 279 || record.plan().values().len() != 327 || scope.rms_nodes().len() != 49 {
        return Err(Error::State(
            "short-HF exact retained279 and49 actual actors".into(),
        ));
    }
    let states = state_requirements(scope.source().source(), record.plan())?;
    let states = checked_sum(states.iter().map(|s| s.bytes))?;
    diagnostic_bytes(
        public,
        states,
        scope.call().model_profile().hidden_size(),
        scope.call().span().len(),
    )
}
pub(super) fn readback_chunks(maximum: usize) -> Result<usize> {
    let chunks = maximum.div_ceil(32 * 1024 * 1024);
    if !(3..=8).contains(&chunks) {
        return Err(Error::Limit(
            "original bounded full-model readback admission".into(),
        ));
    }
    Ok(chunks)
}
pub(super) fn require_geometry_cohort(rows: &[usize]) -> Result<()> {
    if rows.len() != 6
        || rows
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            != [1, 47, 57, 62, 67, 72].into_iter().collect()
    {
        return Err(Error::State(
            "short-HF one complete six-geometry readback admission".into(),
        ));
    }
    Ok(())
}
impl ShortHfReadbackAdmission {
    fn require(&self, scope: &ShortHfModelBinding, service: &Arc<ResidentService>) -> Result<()> {
        if !Arc::ptr_eq(&self.source, scope.source().source())
            || !Arc::ptr_eq(&self.service, service)
            || self.profile != scope.call().model_profile()
            || self.source.source_identity() != service.source_identity()
            || !matches!(scope.call().span().len(), 1 | 47 | 57 | 62 | 67 | 72)
            || actual_diagnostic_bytes(scope)? > self.maximum
            || self.tickets.len() != readback_chunks(self.maximum)?
        {
            return Err(Error::State(
                "short-HF source/service/profile-bound shared readback owner differs".into(),
            ));
        }
        Ok(())
    }
}

impl FullModelProgram {
    pub(crate) fn admit_short_hf_readbacks(
        scopes: &[Arc<ShortHfModelBinding>],
        service: &Arc<ResidentService>,
    ) -> Result<Arc<ShortHfReadbackAdmission>> {
        if scopes.len() != 6 {
            return Err(Error::State("short-HF exact six actual scopes".into()));
        }
        require_geometry_cohort(
            &scopes
                .iter()
                .map(|s| s.call().span().len())
                .collect::<Vec<_>>(),
        )?;
        service.ensure_usable()?;
        let first = &scopes[0];
        let source = first.source().source().clone();
        let profile = first.call().model_profile();
        let mut maximum = 0;
        for scope in scopes {
            if !Arc::ptr_eq(&source, scope.source().source())
                || scope.call().model_profile() != profile
                || source.source_identity() != service.source_identity()
            {
                return Err(Error::State(
                    "short-HF all six authentic source/profile scopes".into(),
                ));
            }
            maximum = maximum.max(actual_diagnostic_bytes(scope)?);
        }
        let chunks = readback_chunks(maximum)?;
        let tickets = Arc::new(
            (0..chunks)
                .map(|_| source.source().reserve_workspace(32 * 1024 * 1024))
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(Arc::new(ShortHfReadbackAdmission {
            source,
            service: service.clone(),
            profile,
            maximum,
            tickets,
        }))
    }
    /// Preparation supplies storage/actors only. The new source-owned native
    /// binder must seal ALL255 actual lowerer envelopes and the original attempt.
    pub(crate) async fn prepare_short_hf_values(
        scope: Arc<ShortHfModelBinding>,
        rms_scope: Arc<FullModelRmsScratch>,
        gdn_family: GdnFamilyIdentity,
        mode: StorageMode,
        limits: ValueStorageLimits,
        readback: Arc<ShortHfReadbackAdmission>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        Self::prepare_short_hf_values_inner(
            scope,
            rms_scope,
            gdn_family,
            mode,
            limits,
            StorageReadback::Shared(readback),
            cancel,
        )
        .await
    }

    pub(crate) async fn prepare_short_hf_shared_values(
        scope: Arc<ShortHfModelBinding>,
        rms_scope: Arc<FullModelRmsScratch>,
        readback: Arc<ShortHfReadbackAdmission>,
        geometry: Arc<crate::vllm_cuda_qualification::model_values::bounded::workspace::ValueWorkspaceGeometry>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        geometry.require_scope(&scope)?;
        Self::prepare_short_hf_values_inner(
            scope,
            rms_scope.clone(),
            GdnFamilyIdentity::short_hf(rms_scope.record().call().model_profile()),
            StorageMode::ReuseMlp,
            short_hf::limits(),
            StorageReadback::SharedViews(readback, geometry),
            cancel,
        )
        .await
    }

    pub(crate) async fn bind_short_hf_values(
        scope: Arc<ShortHfModelBinding>,
        rms_scope: Arc<FullModelRmsScratch>,
        gdn_family: GdnFamilyIdentity,
        mode: StorageMode,
        limits: ValueStorageLimits,
        cold: (
            &Arc<FullModelProgram>,
            &ResidentPreflight,
            Option<&Arc<crate::resident_leases::TiedWeight>>,
        ),
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        Self::prepare_short_hf_values_inner(
            scope,
            rms_scope,
            gdn_family,
            mode,
            limits,
            StorageReadback::Existing(cold.0, cold.1, cold.2),
            cancel,
        )
        .await
    }

    async fn prepare_short_hf_values_inner(
        scope: Arc<ShortHfModelBinding>,
        rms_scope: Arc<FullModelRmsScratch>,
        gdn_family: GdnFamilyIdentity,
        mode: StorageMode,
        limits: ValueStorageLimits,
        readback: StorageReadback<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        cancelled(cancel)?;
        let record = scope.record().clone();
        let source = scope.source().source().clone();
        let service = rms_scope.service().clone();
        if !Arc::ptr_eq(&record, rms_scope.record())
            || service.source_identity() != source.source_identity()
            || record.call() != scope.call().as_vllm_call()
        {
            return Err(Error::State(
                "short-HF SAME source/current record/private RMS storage".into(),
            ));
        }
        let metadata = Arc::new(source.source().reserve_workspace(4 * 1024 * 1024)?);
        let (template, shared, cold_preflight, tied, geometry) = match readback {
            StorageReadback::Shared(admission) => (None, Some(admission), None, None, None),
            StorageReadback::SharedViews(admission, geometry) => {
                (None, Some(admission), None, None, Some(geometry))
            }
            StorageReadback::Existing(cold, preflight, tied) => {
                (Some(cold), None, Some(preflight), tied, None)
            }
        };
        let recipe = match (template, geometry.as_ref()) {
            (Some(cold), None) => {
                StorageRecipe::bind_call(scope.clone(), mode, limits, cold.values())?
            }
            (None, Some(geometry)) => geometry.recipe.clone(),
            (None, None) => StorageRecipe::new(scope.clone(), mode, limits)?,
            _ => {
                return Err(Error::State(
                    "short-HF storage preparation mode differs".into(),
                ));
            }
        };
        let readback = if let Some(cold) = template {
            if !Arc::ptr_eq(&source, &cold.source)
                || !Arc::ptr_eq(&service, cold.values.service())
                || cold._readback.is_empty()
            {
                return Err(Error::State(
                    "short-HF original geometry/readback admission differs".into(),
                ));
            }
            cold._readback.clone()
        } else {
            let shared = shared
                .ok_or_else(|| Error::State("short-HF shared readback admission missing".into()))?;
            shared.require(&scope, &service)?;
            shared.tickets.clone()
        };
        let mut existing = template
            .map(|cold| StorageBuilder::bind_call(recipe.clone(), cold.values()))
            .transpose()?;
        let ordinary_binding = rms_scope.ordinary_binding()?;
        let rms = match template {
            Some(cold) => cold.rms.bind_short_hf_call(rms_scope.clone(), tied)?,
            None => RmsPrepared::prepare_short_hf_pooled(rms_scope.clone(), 0, cancel).await?,
        };
        let mut init = if template.is_none() {
            Some(service.begin_initialization(0)?)
        } else {
            None
        };
        let status = if let Some(init) = &mut init {
            // Retain authentic owners/admissions before any new allocation/upload.
            init.retain_completion_owner(scope.clone())?;
            init.retain_completion_owner(metadata.clone())?;
            init.retain_completion_owner(readback.clone())?;
            init.retain_completion_owner(rms_scope.clone())?;
            init.retain_completion_owner(rms.completion_owner())?;
            init.upload_bytes(&[0; 4], BufferRole::Scratch)?
        } else {
            // launch_impl already clears this exact status on the original
            // stream after begin_launch, under the armed Write flight. All
            // previous reads finish under the workspace lock before Pending
            // escapes. Unknown work keeps the service/pool fenced; no extra
            // initialization flight, synchronization or status clear is needed.
            let status = template.unwrap().status();
            if !status.is_initialized() {
                return Err(Error::State(
                    "short-HF cold status lacks initial completion".into(),
                ));
            }
            status.clone()
        };
        let mut storage = if let Some(existing) = existing.take() {
            existing
        } else if let Some(geometry) = geometry {
            StorageBuilder::from_views(geometry, service.clone(), init.as_mut().unwrap())?
        } else {
            StorageBuilder::allocate(recipe.clone(), service.clone(), init.as_mut().unwrap())?
        };
        storage.bind_ordinary(&rms, &rms_scope)?;
        if let Some(init) = init {
            drop(init.complete_sync().await?);
        }
        let mut fused = BTreeMap::new();
        for &node_index in scope.rms_nodes().iter().skip(1) {
            cancelled(cancel)?;
            let node = &record.plan().nodes()[node_index];
            let [left, right] = node.inputs.as_slice() else {
                return Err(Error::State(
                    "short-HF actual two-input residual RMS".into(),
                ));
            };
            let actor = match template {
                Some(cold) => cold
                    .fused
                    .get(&node_index)
                    .ok_or_else(|| Error::State("short-HF cold dual actor absent".into()))?
                    .bind_short_hf_call(
                        rms_scope.clone(),
                        node_index,
                        storage.lease(*left)?,
                        storage.lease(*right)?,
                    )?,
                None => {
                    FusedRmsPrepared::prepare_short_hf_pooled(
                        rms_scope.clone(),
                        node_index,
                        storage.lease(*left)?.clone(),
                        storage.lease(*right)?.clone(),
                        0,
                        cancel,
                    )
                    .await?
                }
            };
            storage.bind_dual(node_index, &actor, &rms_scope)?;
            fused.insert(node_index, actor);
        }
        if fused.len() != 48 {
            return Err(Error::State("short-HF exact48 original dual actors".into()));
        }
        let (values, buffers) = storage.finish(status)?;
        scope.require_values(&values)?;
        let inventory = values.storage_inventory()?;
        let certificates = if let Some(cold) = template {
            // StorageBuilder checked exact cold slots. RMS rebinds must also
            // preserve their dedicated physical owners before certificate reuse.
            for (index, buffer) in buffers.iter().enumerate() {
                let old = cold.values.lease(ValueId(index as u32))?;
                let current = buffer.original_lease()?;
                if old.id() != current.id()
                    || old.allocation_generation() != current.allocation_generation()
                {
                    return Err(Error::State(
                        "short-HF cold certificate physical owner differs".into(),
                    ));
                }
            }
            if values.status().id() != cold.status().id()
                || values.status().allocation_generation() != cold.status().allocation_generation()
            {
                return Err(Error::State(
                    "short-HF cold certificate status differs".into(),
                ));
            }
            cold.certificates.clone()
        } else {
            Arc::new(prepare_certificates(&service, &buffers, values.status())?)
        };
        let preflight = if let Some(cold) = cold_preflight {
            cold.clone()
        } else {
            let states = state_requirements(&source, record.plan())?;
            let state_bytes = checked_sum(states.iter().map(|r| r.bytes))?;
            ResidentPreflight {
                source: source.source_identity(),
                source_bytes: 0,
                value_bytes: inventory.aligned_slot_bytes,
                workspace_bytes: 0,
                status_bytes: 4,
                parameter_bytes: 0,
                prior_state_bytes: state_bytes,
                candidate_state_bytes: state_bytes,
                states,
                scratch: [0; SCRATCH_SLOTS],
            }
        };
        let body = Arc::new(Self {
            gdn_family,
            canonical47: None,
            source,
            record,
            values,
            rms,
            _ordinary_binding: ordinary_binding,
            fused,
            certificates,
            bound: OnceLock::new(),
            greedy: OnceLock::new(),
            launched: AtomicBool::new(false),
            next_node: AtomicUsize::new(0),
            output_submitted: AtomicBool::new(false),
            fault: AtomicUsize::new(0),
            fault_cut: AtomicBool::new(false),
            convolution_dispatches: AtomicUsize::new(0),
            internal_cancel: AtomicBool::new(false),
            native_launches: AtomicUsize::new(0),
            workspace_clears: AtomicUsize::new(0),
            device_copies: AtomicUsize::new(0),
            _readback: readback,
            _metadata: metadata,
            natural: None,
        });
        body.require_rms_scratch(&rms_scope)?;
        Ok((body, buffers, preflight))
    }

    pub(crate) fn finish_value_storage_node(&self, index: usize) -> Result<()> {
        self.values
            .finish_storage_node(index, self.next_node.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod shared_readback_tests {
    use super::*;
    #[test]
    fn wide72_actual_retained_cohort_fits_original_eight_ticket_limit() {
        // Independent closed recipe:146 H columns +18 GDN/6 attention fixed
        // columns427584; omit24 actual GateUp2I and ACTI output histories.
        let retained = (146 * 2048 + 427584) * 72 * 2 + 496640;
        assert_eq!(retained, 105_125_888);
        let actual = diagnostic_bytes(retained, 32_784_384, 2048, 72).unwrap();
        assert_eq!(actual, 227_530_892);
        assert_eq!(readback_chunks(actual).unwrap(), 7);
        let all327 = retained + 24 * 3 * 6144 * 72 * 2;
        assert!(readback_chunks(diagnostic_bytes(all327, 32_784_384, 2048, 72).unwrap()).is_err());
    }
    #[test]
    fn six_geometries_share_one_maximum_not_six_ticket_sets() {
        let expected = [(1024, 177_590_412, 6), (2048, 227_530_892, 7)];
        for (hidden, bytes, chunks) in expected {
            let actual = [1, 47, 57, 62, 67, 72]
                .into_iter()
                .map(|rows| {
                    let public = (146 * hidden + 427584) * rows * 2 + 496640;
                    diagnostic_bytes(public, 32_784_384, hidden, rows).unwrap()
                })
                .max()
                .unwrap();
            assert_eq!(actual, bytes);
            assert_eq!(readback_chunks(actual).unwrap(), chunks);
        }
    }
    #[test]
    fn missing_duplicate_extra_and_unsupported_geometry_refuse() {
        require_geometry_cohort(&[47, 57, 62, 67, 72, 1]).unwrap();
        require_geometry_cohort(&[1, 72, 67, 62, 57, 47]).unwrap();
        for rows in [
            vec![47, 57, 62, 67, 72],
            vec![47, 57, 62, 67, 72, 72],
            vec![47, 57, 62, 67, 72, 2],
            vec![47, 57, 62, 67, 72, 1, 1],
            Vec::new(),
        ] {
            assert!(require_geometry_cohort(&rows).is_err());
        }
    }
    #[test]
    fn diagnostic_overflow_and_original_capacity_refuse() {
        assert!(diagnostic_bytes(usize::MAX, 32_784_384, 2048, 72).is_err());
        assert!(diagnostic_bytes(1, usize::MAX, 2048, 72).is_err());
        assert!(diagnostic_bytes(1, 1, usize::MAX, 72).is_err());
        assert!(readback_chunks(2 * 32 * 1024 * 1024).is_err());
        assert!(readback_chunks(8 * 32 * 1024 * 1024 + 1).is_err());
    }
}
