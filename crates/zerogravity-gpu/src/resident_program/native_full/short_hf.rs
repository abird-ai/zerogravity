//! Fresh source records over bounded cold storage, in the original255-node loop.
use super::*;
use crate::resident_inference::native_fragment::short_hf::{ShortHfCallInput, ShortHfCallState};
use crate::resident_program::text_positions::short_hf::ShortHfPositions;
use crate::vllm_cuda_input_rms::FullModelRmsScratch;
use crate::vllm_cuda_qualification::{
    model_values::bounded::{NodeStorageAccess, StorageMode},
    short_hf_model::ShortHfModelBinding,
    short_hf_native::ShortHfPrograms,
};
use zerogravity_core::accelerator::program::{ValueStorageLimits, ValueStorageReusePolicy};

pub(crate) struct ShortHfColdValues {
    pub(crate) body: Arc<FullModelProgram>,
    pub(crate) buffers: Vec<DeviceBuffer>,
    pub(crate) preflight: ResidentPreflight,
    pub(crate) rms_scope: Arc<FullModelRmsScratch>,
    greedy: Arc<PreparedBf16Greedy>,
}
pub(super) struct CurrentCall {
    pub(super) input: Arc<ShortHfCallInput>,
    pub(super) state: Arc<ShortHfCallState>,
    pub(super) native: Arc<ShortHfPrograms>,
    pub(super) positions: Arc<ShortHfPositions>,
    pub(super) rms_scope: Arc<FullModelRmsScratch>,
    pub(super) certification: super::super::native::CertificationRun,
}
pub(crate) fn limits() -> ValueStorageLimits {
    ValueStorageLimits {
        max_nodes: 255,
        max_values: 327,
        max_slots: 327,
        max_slot_bytes: 16 * 1024 * 1024,
        max_total_bytes: 256 * 1024 * 1024,
        alignment: 512,
        reuse_policy: ValueStorageReusePolicy::ExactAlignedSize,
    }
}
impl FullModelProgram {
    pub(crate) async fn prepare_short_hf_shared_cold(
        scopes: &[Arc<ShortHfModelBinding>],
        service: Arc<ResidentService>,
        readback: Arc<ShortHfReadbackAdmission>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<ShortHfColdValues>> {
        let workspace = crate::vllm_cuda_qualification::model_values::bounded::workspace::SharedValueWorkspace::prepare(scopes, service.clone(), limits())?;
        let mut cold = Vec::with_capacity(6);
        for (index, scope) in scopes.iter().enumerate() {
            cold.push(
                Self::prepare_short_hf_cold_inner(
                    scope.clone(),
                    service.clone(),
                    readback.clone(),
                    Some(workspace.geometry(index)?),
                    cancel,
                )
                .await?,
            );
        }
        Ok(cold)
    }
    async fn prepare_short_hf_cold_inner(
        scope: Arc<ShortHfModelBinding>,
        service: Arc<ResidentService>,
        readback: Arc<ShortHfReadbackAdmission>,
        geometry: Option<Arc<crate::vllm_cuda_qualification::model_values::bounded::workspace::ValueWorkspaceGeometry>>,
        cancel: Option<&AtomicBool>,
    ) -> Result<ShortHfColdValues> {
        cancelled(cancel)?;
        let rms_scope = if let Some(geometry) = &geometry {
            geometry.require_scope(&scope)?;
            FullModelRmsScratch::bind_views(
                scope.bound_call().clone(),
                scope.record().clone(),
                service.clone(),
                geometry.rms_views()?,
                cancel,
            )?
        } else {
            let mut init = service.begin_initialization(0)?;
            let scope = FullModelRmsScratch::allocate(
                scope.bound_call().clone(),
                scope.record().clone(),
                service.clone(),
                &mut init,
                cancel,
            )?;
            drop(init.complete_sync().await?);
            scope
        };
        let (body, buffers, preflight) = if let Some(geometry) = geometry {
            Self::prepare_short_hf_shared_values(
                scope,
                rms_scope.clone(),
                readback,
                geometry,
                cancel,
            )
            .await?
        } else {
            Self::prepare_short_hf_values(
                scope,
                rms_scope.clone(),
                GdnFamilyIdentity::short_hf(rms_scope.record().call().model_profile()),
                StorageMode::ReuseMlp,
                limits(),
                readback,
                cancel,
            )
            .await?
        };
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(body.clone())?;
        let greedy = PreparedBf16Greedy::prepare(
            body.source.clone(),
            body.values.clone(),
            &mut init,
            cancel,
        )?;
        init.retain_completion_owner(greedy.clone())?;
        drop(init.complete_sync().await?);
        Ok(ShortHfColdValues {
            body,
            buffers,
            preflight,
            rms_scope,
            greedy,
        })
    }
    pub(crate) fn bind_short_hf_call(
        &self,
        input: Arc<ShortHfCallInput>,
        state: Arc<ShortHfCallState>,
        native: Arc<ShortHfPrograms>,
        positions: Arc<ShortHfPositions>,
        rms_scope: Arc<FullModelRmsScratch>,
        gdn: BTreeMap<usize, Arc<ModelGdnPrepared>>,
    ) -> Result<()> {
        state.require_submission_claimed()?;
        if self.bound.get().is_some()
            || self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != 0
            || !Arc::ptr_eq(state.input(), &input)
            || !Arc::ptr_eq(&self.record, input.record())
            || self.greedy.get().is_none()
            || self.natural.is_some()
            || self.fused.len() != 48
            || gdn.len() != 18
            || state.states().len() != 24
            || !positions.retained(&input, self.values.service())
            || !Arc::ptr_eq(self.values.short_hf_storage_scope()?, input.scope())
            || !Arc::ptr_eq(rms_scope.record(), input.record())
        {
            return Err(Error::State(
                "short-HF once-only SAME original source/current record/actors".into(),
            ));
        }
        native.require_values(&input, &self.values)?;
        self.require_rms_scratch(&rms_scope)?;
        let mut all = BTreeMap::new();
        for (l, a) in self
            .values
            .accesses()
            .into_iter()
            .chain(self.rms.accesses().iter().cloned())
        {
            insert_access(&mut all, l, a);
        }
        for f in self.fused.values() {
            for (l, a) in f.accesses() {
                insert_access(&mut all, l.clone(), *a);
            }
        }
        for (layer, s) in state.states().iter().enumerate() {
            s.require_short_hf(input.scope())?;
            if s.layer().index() != layer {
                return Err(Error::State("short-HF exact24 bound state order".into()));
            }
            for (l, a) in s.leases() {
                insert_access(&mut all, l, a);
            }
            if layer % 4 != 3 {
                gdn.get(&layer)
                    .ok_or_else(|| Error::State("short-HF source GDN missing".into()))?
                    .require_value_state(&self.values, s)?;
            }
        }
        let arithmetic = gdn
            .values()
            .next()
            .ok_or_else(|| Error::State("short-HF arithmetic absent".into()))?
            .arithmetic()
            .clone();
        self.gdn_family.require_same(arithmetic.family())?;
        for p in gdn.values() {
            p.require_short_hf_binding(input.scope())?;
            arithmetic.require_same(p.arithmetic())?;
            for (l, a) in p.accesses() {
                insert_access(&mut all, l, a);
            }
        }
        for (l, a) in native
            .accesses()
            .iter()
            .chain(self.greedy.get().unwrap().accesses())
        {
            insert_access(&mut all, l.clone(), *a);
        }
        for (l, a) in positions.accesses() {
            insert_access(&mut all, l, a);
        }
        // Concrete whole-operation unions, never PlanNode-only access guesses.
        let mut envelopes = Vec::with_capacity(255);
        for (node_index, node) in self.record.plan().nodes().iter().enumerate() {
            let actual = if node_index == 0 {
                self.rms
                    .short_hf_embedding_accesses()?
                    .into_iter()
                    .chain(positions.accesses())
                    .collect()
            } else {
                let op = descriptor(node)?;
                match op.kind() {
                    Vllm08Kind::InputRms => self.rms.short_hf_rms_accesses()?,
                    Vllm08Kind::ResidualRms(_) => self
                        .fused
                        .get(&node_index)
                        .ok_or_else(|| {
                            Error::State("short-HF concrete dual envelope absent".into())
                        })?
                        .short_hf_storage_accesses()?,
                    Vllm08Kind::GdnConv | Vllm08Kind::GdnRecurrent => gdn
                        .get(
                            &op.layer()
                                .ok_or_else(|| {
                                    Error::State("short-HF GDN envelope layer absent".into())
                                })?
                                .index(),
                        )
                        .ok_or_else(|| Error::State("short-HF GDN envelope absent".into()))?
                        .short_hf_storage_accesses(node_index)?,
                    _ => native.storage_accesses(node_index)?,
                }
            };
            envelopes.push(NodeStorageAccess::bind(&self.values, node_index, &actual)?);
        }
        self.values.seal_storage_accesses(envelopes)?;
        let certification = super::super::native::CertificationRun::bind(
            self.certificates.clone(),
            self.values.clone(),
        )?;
        self.bound
            .set(BoundPhase {
                gdn_arithmetic: arithmetic,
                phase: 0,
                library: None,
                canonical47_prefix: None,
                canonical_call: None,
                states: state.states().to_vec(),
                gdn,
                attention: BTreeMap::new(),
                accesses: all.into_values().collect(),
                short_hf: Some(CurrentCall {
                    input,
                    state,
                    native,
                    positions,
                    rms_scope,
                    certification,
                }),
            })
            .map_err(|_| Error::State("short-HF current body already bound".into()))
    }
    pub(crate) fn register_short_hf_head(
        &self,
        registration: &mut crate::resident_inference::RmsScratchRegistration<'_, '_>,
    ) -> Result<()> {
        if let Some(current) = self.bound.get().and_then(|b| b.short_hf.as_ref()) {
            current.native.head().register(registration)?;
        }
        Ok(())
    }
    pub(crate) fn short_hf_enqueue_placement(
        &self,
    ) -> Result<crate::resident_leases::EnqueuePlacement> {
        Ok(self.short_hf_current()?.native.head().enqueue_placement())
    }
    pub(crate) fn short_hf_rms_scope(&self) -> Result<&Arc<FullModelRmsScratch>> {
        Ok(&self.short_hf_current()?.rms_scope)
    }
    fn short_hf_current(&self) -> Result<&CurrentCall> {
        self.bound
            .get()
            .and_then(|b| b.short_hf.as_ref())
            .ok_or_else(|| Error::State("short-HF current body absent".into()))
    }
    pub(crate) fn short_hf_extra_reads(&self) -> Result<Vec<ResidentReadback>> {
        self.short_hf_current()?.native.extra_readbacks()
    }
    pub(crate) fn validate_short_hf_extra(&self, raw: &[cubecl::bytes::Bytes]) -> Result<()> {
        self.short_hf_current()?.native.validate_extra(raw)
    }
    pub(crate) fn arm_short_hf_fault(&self, mode: usize) -> Result<()> {
        let c = self.short_hf_current()?;
        if c.input.rows() != 1
            || c.input.call_ordinal() != 1
            || !matches!(mode, 1 | 2 | 19 | 20 | 21)
            || self.launched.load(Ordering::Acquire)
            || self.fault.load(Ordering::Acquire) != 0
        {
            return Err(Error::Unsupported(
                "short-HF cached-first actual fault only".into(),
            ));
        }
        if mode == 20 || mode == 21 {
            c.native.head().arm_fault(if mode==20 {
            crate::vllm_cuda_qualification::short_hf_head::StreamedHeadFault::UnknownAfterNative
        } else {crate::vllm_cuda_qualification::short_hf_head::StreamedHeadFault::CancelBeforeOverwrite})?;
        }
        self.fault.store(mode, Ordering::Release);
        Ok(())
    }
    pub(crate) fn short_hf_source_graph_retained(&self) -> Result<bool> {
        let c = self.short_hf_current()?;
        Ok(Arc::ptr_eq(&self.record, c.input.record())
            && Arc::ptr_eq(c.state.input(), &c.input)
            && Arc::ptr_eq(self.values.short_hf_storage_scope()?, c.input.scope())
            && c.positions.retained(&c.input, self.values.service())
            && c.native.enqueued_records() == c.native.submission_counts().native
            && c.native.retained_values()?)
    }
}
impl ShortHfColdValues {
    pub(crate) async fn bind_values(
        &self,
        input: &Arc<ShortHfCallInput>,
        state: &Arc<ShortHfCallState>,
        tied: Option<&Arc<crate::resident_leases::TiedWeight>>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(
        Arc<FullModelProgram>,
        Vec<DeviceBuffer>,
        ResidentPreflight,
        Arc<FullModelRmsScratch>,
    )> {
        state.require_submission_claimed()?;
        if self.body.bound.get().is_some() || self.body.launched.load(Ordering::Acquire) {
            return Err(Error::State(
                "short-HF cold prototype cannot be dispatched".into(),
            ));
        }
        let scope = self.rms_scope.bind_call(
            input.scope().bound_call().clone(),
            input.record().clone(),
            cancel,
        )?;
        let (body, buffers, preflight) = FullModelProgram::bind_short_hf_values(
            input.scope().clone(),
            scope.clone(),
            GdnFamilyIdentity::short_hf(input.scope().call().model_profile()),
            StorageMode::ReuseMlp,
            limits(),
            (&self.body, &self.preflight, tied),
            cancel,
        )
        .await?;
        let greedy = self.greedy.bind_short_hf_call(body.values.clone())?;
        body.greedy
            .set(greedy)
            .map_err(|_| Error::State("short-HF fresh sampler already bound".into()))?;
        Ok((body, buffers, preflight, scope))
    }
}

pub(crate) struct ReadbackLayout {
    pub(crate) reads: Vec<ResidentReadback>,
    pub(crate) values: BTreeMap<u32, usize>,
    pub(crate) sums: BTreeMap<usize, usize>,
    pub(crate) packet: usize,
    pub(crate) count: usize,
    ordinary: std::ops::Range<usize>,
    duals: BTreeMap<usize, std::ops::Range<usize>>,
    status: usize,
}
impl FullModelProgram {
    pub(crate) fn short_hf_readbacks(&self, diagnostic: bool) -> Result<ReadbackLayout> {
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("short-HF readbacks before binding".into()))?;
        self.short_hf_current()?;
        let mut reads = vec![self.status().readback(0, 4)?];
        for p in bound.gdn.values() {
            reads.push(p.status_readback()?);
        }
        if reads.len() != 19 {
            return Err(Error::State(
                "short-HF original19 transport statuses".into(),
            ));
        }
        let mut layout = ReadbackLayout {
            reads,
            values: BTreeMap::new(),
            sums: BTreeMap::new(),
            packet: 0,
            count: 0,
            ordinary: 0..0,
            duals: BTreeMap::new(),
            status: 19,
        };
        if diagnostic {
            let first = layout.reads.len();
            layout.reads.extend(self.rms.readbacks()?);
            layout.ordinary = first..layout.reads.len();
            // Ordinary source gamma, embedding and normalized are actual retained actors.
            if layout.ordinary.len() != 3 {
                return Err(Error::State("short-HF ordinary exact3 reads".into()));
            }
            layout.values.insert(0, first + 1);
            layout.values.insert(1, first + 2);
            for id in 2..327 {
                let id = ValueId(id);
                let origin = self
                    .record
                    .plan()
                    .vllm_value_provenance(id)?
                    .ok_or_else(|| {
                        Error::State("short-HF original public producer absent".into())
                    })?;
                if let zerogravity_core::accelerator::program::VllmValueOrigin::FullModel(op) =
                    origin.origin()
                    && matches!(
                        op.kind(),
                        Vllm08Kind::Projection(Vllm08Projection::GateUp) | Vllm08Kind::SiluAndMul
                    )
                {
                    continue;
                }
                let r = self.values.retained_value_readback(id)?;
                layout.values.insert(id.0, layout.reads.len());
                layout.reads.push(r);
            }
            for (&node, fused) in &self.fused {
                let start = layout.reads.len();
                layout.reads.extend(fused.readbacks()?);
                if layout.reads.len() != start + 4 {
                    return Err(Error::State("short-HF actual dual4 reads".into()));
                }
                layout.duals.insert(node, start..start + 4);
                layout.sums.insert(node, start + 1);
            }
            if layout.values.len() != 279 || layout.duals.len() != 48 {
                return Err(Error::State(
                    "short-HF retained279 public/48 actual duals".into(),
                ));
            }
        }
        if !diagnostic {
            let extras = self.short_hf_extra_reads()?;
            if extras.len() != 7 {
                return Err(Error::State(
                    "short-HF actual6 KV statuses and positions".into(),
                ));
            }
            layout.reads.extend(extras.into_iter().skip(1));
        }
        layout.packet = layout.reads.len();
        layout.reads.extend(
            self.greedy
                .get()
                .ok_or_else(|| Error::State("short-HF actual sampler absent".into()))?
                .readbacks()?,
        );
        layout.count = layout.reads.len();
        Ok(layout)
    }
    pub(crate) fn short_hf_transfer_retirement(&self) -> Result<[usize; 5]> {
        self.short_hf_current()?.native.head().retirement_counts()
    }
    pub(crate) fn validate_short_hf_readbacks(
        &self,
        layout: &ReadbackLayout,
        raw: &[cubecl::bytes::Bytes],
    ) -> Result<(u32, u32)> {
        self.require_publication_finished()?;
        self.short_hf_current()?
            .native
            .head()
            .retire_transfers_after_completion()?;
        if raw.len() != layout.count
            || layout.status != 19
            || raw[..19].iter().any(|b| b.as_ref() != [0; 4])
        {
            return Err(Error::State("short-HF actual body/status cohort".into()));
        }
        if !layout.ordinary.is_empty() {
            self.rms.validate_readback(&raw[layout.ordinary.clone()])?;
            for (&node, range) in &layout.duals {
                self.fused
                    .get(&node)
                    .ok_or_else(|| Error::State("short-HF source dual absent".into()))?
                    .validate_readback(&raw[range.clone()])?;
            }
        }
        if layout.ordinary.is_empty() {
            if layout.packet != 25 {
                return Err(Error::State(
                    "short-HF production27 descriptor layout".into(),
                ));
            }
            for status in &raw[19..25] {
                crate::vllm_cuda_qualification::model_kv_transport::ModelKvTransport::validate_status(status.as_ref())?;
            }
        }
        self.greedy
            .get()
            .unwrap()
            .validate_readback(&raw[layout.packet..layout.packet + 2])
    }
}

impl ShortHfColdValues {
    pub(crate) fn storage_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut all = BTreeMap::new();
        for (l, a) in self
            .body
            .values
            .accesses()
            .into_iter()
            .chain(self.body.rms.accesses().iter().cloned())
            .chain(self.greedy.accesses().iter().cloned())
        {
            insert_access(&mut all, l, a);
        }
        for f in self.body.fused.values() {
            for (l, a) in f.accesses() {
                insert_access(&mut all, l.clone(), *a);
            }
        }
        all.into_values().collect()
    }
    pub(crate) fn immutable_rms(&self) -> Result<Vec<(ResidentLease, &[u8])>> {
        let mut all = vec![self.body.rms.short_hf_immutable_gamma()?];
        all.extend(self.body.fused.values().map(|f| f.immutable_gamma()));
        Ok(all)
    }
}
impl FullModelProgram {
    pub(crate) fn short_hf_submission_accesses(&self) -> Result<&[(ResidentLease, DeviceAccess)]> {
        // Metadata-only quarantine inspection. Normal accesses() still proves
        // the complete current run; this getter never consults service health.
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("short-HF current binding absent".into()))?;
        if bound.short_hf.is_none() {
            return Err(Error::Unsupported("not a short-HF current body".into()));
        }
        Ok(&bound.accesses)
    }
}

impl FullModelProgram {
    pub(crate) fn short_hf_head_retention(
        &self,
    ) -> Result<crate::vllm_cuda_qualification::short_hf_head::StreamedHeadRetained> {
        self.short_hf_current()?
            .native
            .head()
            .retention_witness()?
            .inspect()
    }
}
