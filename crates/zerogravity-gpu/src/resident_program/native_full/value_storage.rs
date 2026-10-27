//! Source-issued full graph, original actors and original dispatch cursor.
use super::*;
use crate::cuda::native::{
    model_values::bounded::{StorageBuilder, StorageMode, StorageRecipe},
    text_model::TextModelBinding,
    text_native::TextKernelCapture,
};
use crate::cuda::rms::FullModelRmsScratch;
use zerogravity_core::accelerator::program::ValueStorageLimits;

/// One pre-admitted maximum actual diagnostic cohort. All six cold geometries
/// run under the SAME original workspace lock; returned Pending drops raw D2H.
/// No separate dispatcher, uncharged capacity or public input authority.
pub(crate) struct TextReadbackAdmission {
    source: VerifiedTensorSource,
    capture: TextKernelCapture,
    service: Arc<ResidentService>,
    width: usize,
    maximum: usize,
    tickets: Arc<Vec<WindowWorkspace>>,
}
enum StorageReadback<'a> {
    Shared(Arc<TextReadbackAdmission>),
    SharedViews(
        Arc<TextReadbackAdmission>,
        Arc<crate::cuda::native::model_values::bounded::workspace::ValueWorkspaceGeometry>,
    ),
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
pub(super) fn is_mlp_temporary(record: &SourceBindings, id: ValueId) -> Result<bool> {
    let producer = record
        .plan()
        .source_value_provenance(id)?
        .ok_or_else(|| Error::State("actual public producer absent".into()))?;
    let index = producer.node_index();
    let descriptor = record
        .operator(index)?
        .ok_or_else(|| Error::State("actual public operator absent".into()))?;
    if matches!(descriptor.kind(), SourceOperatorKind::SiluAndMul { .. }) {
        return Ok(true);
    }
    if matches!(
        descriptor.kind(),
        SourceOperatorKind::Projection { selected_row: None }
    ) && let Some(next) = record.plan().nodes().get(index + 1)
        && let PlanOperation::SourceOperator {
            descriptor: activation,
            ..
        } = &next.operation
        && matches!(activation.kind(), SourceOperatorKind::SiluAndMul { .. })
        && activation.call() == record.call()
        && next.inputs == [id]
    {
        return Ok(true);
    }
    Ok(false)
}
pub(super) fn actual_diagnostic_bytes(scope: &TextModelBinding) -> Result<usize> {
    let record = scope.record();
    let mut public = 0usize;
    let mut retained = 0usize;
    for (index, value) in record.plan().values().iter().enumerate() {
        let overwritten = is_mlp_temporary(
            record,
            ValueId(
                u32::try_from(index).map_err(|_| Error::Limit("actual ValueId extent".into()))?,
            ),
        )?;
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
    let states = state_requirements(record)?;
    let states = checked_sum(states.iter().map(|s| s.bytes))?;
    diagnostic_bytes(
        public,
        states,
        scope.hidden_size(),
        scope.call().rows().len(),
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
pub(super) fn require_geometry_cohort(capture: TextKernelCapture, rows: &[usize]) -> Result<()> {
    let expected = if capture.is_fresh_prefill() {
        capture.captured_prefill_rows().to_vec()
    } else {
        capture
            .captured_prefill_rows()
            .iter()
            .copied()
            .chain(std::iter::once(1))
            .collect()
    };
    if rows.len() != capture.cold_scope_count()?
        || rows
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            != expected.into_iter().collect()
    {
        return Err(Error::State(
            "short-HF complete capture-geometry readback admission".into(),
        ));
    }
    Ok(())
}
impl TextReadbackAdmission {
    fn require(&self, scope: &TextModelBinding, service: &Arc<ResidentService>) -> Result<()> {
        if !self.source.same_issuer(scope.source())
            || !Arc::ptr_eq(&self.service, service)
            || self.width != scope.hidden_size()
            || self.source.source_identity() != service.source_identity()
            || scope.native_capture() != self.capture
            || self
                .capture
                .cold_scope_index(scope.call().mode(), scope.call().rows().len())
                .is_err()
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
    pub(crate) fn admit_text_readbacks(
        scopes: &[Arc<TextModelBinding>],
        service: &Arc<ResidentService>,
    ) -> Result<Arc<TextReadbackAdmission>> {
        let capture = scopes
            .first()
            .ok_or_else(|| Error::State("short-HF no actual scopes".into()))?
            .native_capture();
        if scopes.len() != capture.cold_scope_count()?
            || scopes.iter().any(|scope| scope.native_capture() != capture)
        {
            return Err(Error::State("short-HF exact captured scopes".into()));
        }
        require_geometry_cohort(
            capture,
            &scopes
                .iter()
                .map(|s| s.call().rows().len())
                .collect::<Vec<_>>(),
        )?;
        service.ensure_usable()?;
        let first = &scopes[0];
        let source = first.source().clone();
        let width = first.hidden_size();
        let mut maximum = 0;
        for scope in scopes {
            if !source.same_issuer(scope.source())
                || scope.hidden_size() != width
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
                .map(|_| source.reserve_workspace(32 * 1024 * 1024))
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(Arc::new(TextReadbackAdmission {
            source,
            capture,
            service: service.clone(),
            width,
            maximum,
            tickets,
        }))
    }
    /// Preparation supplies storage/actors only. The new source-owned native
    /// binder must seal ALL255 actual lowerer envelopes and the original attempt.
    pub(crate) async fn prepare_text_values(
        scope: Arc<TextModelBinding>,
        rms_scope: Arc<FullModelRmsScratch>,
        gdn_family: GdnFamilyIdentity,
        mode: StorageMode,
        limits: ValueStorageLimits,
        readback: Arc<TextReadbackAdmission>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        Self::prepare_text_values_inner(
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

    pub(crate) async fn prepare_text_shared_values(
        scope: Arc<TextModelBinding>,
        rms_scope: Arc<FullModelRmsScratch>,
        readback: Arc<TextReadbackAdmission>,
        geometry: Arc<
            crate::cuda::native::model_values::bounded::workspace::ValueWorkspaceGeometry,
        >,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        geometry.require_scope(&scope)?;
        let family = GdnFamilyIdentity::text(scope.gdn_implementation().config());
        Self::prepare_text_values_inner(
            scope,
            rms_scope.clone(),
            family,
            StorageMode::ReuseMlp,
            text::limits(),
            StorageReadback::SharedViews(readback, geometry),
            cancel,
        )
        .await
    }

    pub(crate) async fn bind_text_values(
        scope: Arc<TextModelBinding>,
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
        Self::prepare_text_values_inner(
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

    async fn prepare_text_values_inner(
        scope: Arc<TextModelBinding>,
        rms_scope: Arc<FullModelRmsScratch>,
        gdn_family: GdnFamilyIdentity,
        mode: StorageMode,
        limits: ValueStorageLimits,
        readback: StorageReadback<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        cancelled(cancel)?;
        let record = scope.record().clone();
        let source = scope.source().clone();
        let service = rms_scope.service().clone();
        if !Arc::ptr_eq(&record, rms_scope.record())
            || service.source_identity() != source.source_identity()
            || record.call() != scope.call()
        {
            return Err(Error::State(
                "short-HF SAME source/current record/private RMS storage".into(),
            ));
        }
        let metadata = Arc::new(source.reserve_workspace(4 * 1024 * 1024)?);
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
            if !source.same_issuer(&cold.source)
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
            Some(cold) => cold.rms.bind_prompt_call(rms_scope.clone(), tied)?,
            None => RmsPrepared::prepare_text_pooled(rms_scope.clone(), 0, cancel).await?,
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
                    .bind_prompt_call(
                        rms_scope.clone(),
                        node_index,
                        storage.lease(*left)?,
                        storage.lease(*right)?,
                    )?,
                None => {
                    FusedRmsPrepared::prepare_text_pooled(
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
            let states = state_requirements(&record)?;
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
            attention_sites: attention_sites(&record)?,
            gdn_family,
            fixed_prompt: None,
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
        let historical = TextKernelCapture::bind(
            "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d",
        )
        .unwrap();
        require_geometry_cohort(historical, &[47, 57, 62, 67, 72, 1]).unwrap();
        require_geometry_cohort(historical, &[1, 72, 67, 62, 57, 47]).unwrap();
        let fresh = TextKernelCapture::bind_fresh_prefill(historical.recipe_sha256()).unwrap();
        require_geometry_cohort(fresh, &[47, 48, 49, 50]).unwrap();
        for rows in [
            vec![47, 57, 62, 67, 72],
            vec![47, 57, 62, 67, 72, 72],
            vec![47, 57, 62, 67, 72, 2],
            vec![47, 57, 62, 67, 72, 1, 1],
            Vec::new(),
        ] {
            assert!(require_geometry_cohort(historical, &rows).is_err());
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
