//! Finite original recorded nodes in the SAME ResidentProgram loop; no publisher.
use super::native::{
    CertificateSet, cancelled as not_cancelled, certify_original, prepare_certificates,
    state_requirements,
};
use super::*;
use crate::cuda::native::fixed_prompt_mlp::{
    FixedPromptMlpQualificationInputs, FixedPromptMlpWeights,
};
use crate::cuda::native::qwen2b_projection::ProjectionPrograms;
use crate::cuda::native::{CubinQualificationInputs, qwen2b_component::ComponentGdnPrepared};
use crate::cuda::rms::{FusedRmsPrepared, RmsPrepared};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_inference::{gdn_state::BoundGdnDomains, native_fragment::PrefixPhaseEnqueue};
use crate::resident_leases::{
    Qwen2bMlpSourceVerification, Qwen2bMlpWeightCohort, Qwen2bMlpWeights, ResidentInitialization,
    ResidentLease, ResidentReadback, ResidentService,
};
use cubecl::bytes::Bytes;
use std::{
    collections::BTreeSet,
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

mod cursor;
mod projections;
mod recipe;
#[cfg(test)]
pub(crate) use recipe::FIXED_PROMPT_IDS;
mod values;
use cursor::ComponentCursor;
use projections::ComponentProjections;
pub(crate) use recipe::{ComponentCut, ComponentKind};
pub use recipe::{ComponentOutput, ComponentRecipe};
pub(crate) use values::ComponentValues;

const MIB: usize = 1024 * 1024;
/// Independent diagnostic envelope, derived from all real managed/native owners.
/// Existing128MiB leaf and all model envelopes remain unchanged.
pub(crate) const PROVIDER_BYTES: usize = 192 * MIB;
pub(crate) const POST_ATTENTION_PROVIDER_BYTES: usize = 224 * MIB;
pub(crate) const QWEN2B_MLP_PROVIDER_BYTES: usize = 320 * MIB;
pub(crate) fn validate_component(recipe: &Arc<ComponentRecipe>) -> Result<()> {
    state_requirements(recipe.bindings())?;
    Ok(())
}
struct ComponentProducer {
    rms: Arc<RmsPrepared>,
    post_attention: Option<Arc<FusedRmsPrepared>>,
    values: Arc<ComponentValues>,
    workspace: ResidentLease,
    projections: OnceLock<Arc<ComponentProjections>>,
    mlp_weights: OnceLock<Arc<FixedPromptMlpWeights>>,
    qwen2b_mlp_weights: OnceLock<Arc<Qwen2bMlpWeights>>,
}
pub(crate) struct ComponentProgram {
    // Certificate handles retire before the original value/status leases.
    certificates: CertificateSet,
    recipe: Arc<ComponentRecipe>,
    producer: Arc<ComponentProducer>,
    cursor: ComponentCursor,
    gdn: OnceLock<Arc<ComponentGdnPrepared>>,
    native_launches: AtomicUsize,
    fault: AtomicUsize,
    cut: AtomicBool,
    internal_cancel: AtomicBool,
    _readback_host: Vec<WindowWorkspace>,
    _metadata: WindowWorkspace,
}
fn merge(
    target: &mut Vec<(ResidentLease, DeviceAccess)>,
    more: impl IntoIterator<Item = (ResidentLease, DeviceAccess)>,
) {
    for (lease, access) in more {
        if let Some((_, old)) = target
            .iter_mut()
            .find(|(other, _)| other.id() == lease.id())
        {
            if access == DeviceAccess::Write {
                *old = access;
            }
        } else {
            target.push((lease, access));
        }
    }
}
impl ComponentProgram {
    pub(crate) async fn prepare_values(
        service: Arc<ResidentService>,
        recipe: Arc<ComponentRecipe>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        not_cancelled(cancelled)?;
        if recipe.source().staging_chunk_bytes() != MIB {
            return Err(Error::Unsupported(
                "source2B component original1MiB authenticated chunks".into(),
            ));
        }
        let source = recipe.source();
        let metadata = source.reserve_workspace(4 * MIB)?;
        // Canonical prior+candidate65.6MB +sourceW33.7MB +actual values/RMS/gamma.
        let readback_host = (0..4)
            .map(|_| source.reserve_workspace(32 * MIB))
            .collect::<Result<Vec<_>>>()?;
        let states = state_requirements(recipe.bindings())?;
        let state_bytes = states.iter().try_fold(0usize, |n, s| {
            n.checked_add(s.bytes)
                .ok_or_else(|| Error::Limit("source2B canonical root extent".into()))
        })?;
        let physical_rows = match recipe.kind() {
            ComponentKind::FixedPrompt(_) => crate::cuda::rms::Rows::FixedPrompt,
            ComponentKind::Qwen2b => {
                crate::cuda::rms::Rows::new(recipe.bindings().call().rows().len())?
            }
        };
        let rms = RmsPrepared::prepare_rows_bound(
            service.clone(),
            recipe.bindings().clone(),
            physical_rows,
            0,
            cancelled,
        )
        .await?;
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(recipe.clone())?;
        init.retain_completion_owner(rms.completion_owner())?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let workspace = init.allocate_empty(8_519_680, BufferRole::Scratch)?;
        let mut values = Vec::with_capacity(recipe.value_end());
        let allocated_end = if recipe.cut().includes_post_attention() {
            8
        } else {
            recipe.value_end()
        };
        for (index, shape) in recipe.bindings().plan().values()[..allocated_end]
            .iter()
            .enumerate()
        {
            let layout =
                ResidentBufferLayout::new(ResidentPhysicalType::Bf16, shape.capacity_elements()?)?;
            let lease = match index {
                0 => rms.embedding().clone(),
                1 => rms.normalized().clone(),
                _ => init.allocate_empty(layout.bytes()?, BufferRole::Activation)?,
            };
            values.push(DeviceBuffer::from_lease(
                lease.clone(),
                layout,
                lease.role(),
                DeviceAccess::Write,
            )?);
        }
        drop(init.complete_sync().await?);
        let post_attention = if recipe.cut().includes_post_attention() {
            let binding = crate::cuda::rms::BoundModelRmsSource::bind_for_plan(
                recipe.bindings().clone(),
                recipe.operator(7)?.clone(),
            )?;
            let prepared = FusedRmsPrepared::prepare_model_rows(
                service.clone(),
                binding,
                physical_rows,
                values[7].original_lease()?.clone(),
                values[0].original_lease()?.clone(),
                0,
                cancelled,
            )
            .await?;
            for lease in [prepared.normalized(), prepared.residual()] {
                let layout = ResidentBufferLayout::new(
                    ResidentPhysicalType::Bf16,
                    recipe.bindings().call().rows().len() * recipe.kind().columns(),
                )?;
                values.push(DeviceBuffer::from_lease(
                    lease.clone(),
                    layout,
                    lease.role(),
                    DeviceAccess::Write,
                )?);
            }
            Some(prepared)
        } else {
            None
        };
        if recipe.cut().includes_mlp() {
            let mut init = service.begin_initialization(0)?;
            init.retain_completion_owner(recipe.clone())?;
            init.retain_completion_owner(rms.completion_owner())?;
            init.retain_completion_owner(
                post_attention
                    .as_ref()
                    .ok_or_else(|| Error::State("MLP actual dual producer absent".into()))?
                    .completion_owner(),
            )?;
            for shape in &recipe.bindings().plan().values()[10..13] {
                let layout = ResidentBufferLayout::new(
                    ResidentPhysicalType::Bf16,
                    shape.capacity_elements()?,
                )?;
                let lease = init.allocate_empty(layout.bytes()?, BufferRole::Activation)?;
                values.push(DeviceBuffer::from_lease(
                    lease.clone(),
                    layout,
                    lease.role(),
                    DeviceAccess::Write,
                )?);
            }
            drop(init.complete_sync().await?);
        }
        let value_map =
            ComponentValues::bind(recipe.clone(), &values, status.clone(), service.clone())?;
        let certificates = prepare_certificates(&service, &values, &status)?;
        let preflight = ResidentPreflight {
            source: service.source_identity(),
            source_bytes: 0,
            value_bytes: values.iter().try_fold(0usize, |n, v| {
                n.checked_add(v.physical.bytes()?)
                    .ok_or_else(|| Error::Limit("source2B seven values bytes".into()))
            })?,
            workspace_bytes: 0,
            status_bytes: 4,
            parameter_bytes: 0,
            prior_state_bytes: state_bytes,
            candidate_state_bytes: state_bytes,
            states,
            scratch: [0; SCRATCH_SLOTS],
        };
        let body = Arc::new(Self {
            cursor: ComponentCursor::new(recipe.clone()),
            recipe,
            producer: Arc::new(ComponentProducer {
                rms,
                post_attention,
                values: value_map,
                workspace,
                projections: OnceLock::new(),
                mlp_weights: OnceLock::new(),
                qwen2b_mlp_weights: OnceLock::new(),
            }),
            certificates,
            gdn: OnceLock::new(),
            native_launches: AtomicUsize::new(0),
            fault: AtomicUsize::new(0),
            cut: AtomicBool::new(false),
            internal_cancel: AtomicBool::new(false),
            _readback_host: readback_host,
            _metadata: metadata,
        });
        Ok((body, values, preflight))
    }
    pub(crate) fn recipe(&self) -> &Arc<ComponentRecipe> {
        &self.recipe
    }
    pub(crate) fn status(&self) -> &ResidentLease {
        self.producer.values.status()
    }
    pub(crate) fn completion_owner(self: &Arc<Self>) -> CompletionOwner {
        self.clone()
    }
    pub(crate) async fn prepare_gdn(
        &self,
        domains: Arc<BoundGdnDomains>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        if self.gdn.get().is_some() {
            return Err(Error::State("source2B GDN prepared twice".into()));
        }
        let upstream: CompletionOwner = self.producer.clone();
        let prepared = ComponentGdnPrepared::prepare(
            self.producer.values.clone(),
            domains,
            upstream,
            cancelled,
        )
        .await?;
        self.gdn
            .set(prepared)
            .map_err(|_| Error::State("source2B GDN owner installed twice".into()))
    }
    /// Final initializer already contains the actual original two W uploads.
    pub(crate) fn prepare_native(
        self: &Arc<Self>,
        inputs: CubinQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        not_cancelled(cancelled)?;
        if self.recipe.cut().includes_mlp()
            || self.gdn.get().is_none()
            || self.producer.projections.get().is_some()
        {
            return Err(Error::State(
                "source2B all GDN/managed owners precede native activation".into(),
            ));
        }
        init.retain_completion_owner(self.completion_owner())?;
        let programs = ComponentProjections::prepare(
            self.producer.values.clone(),
            self.producer.values.service(),
            inputs,
            init,
            cancelled,
        )?;
        self.producer
            .projections
            .set(programs)
            .map_err(|_| Error::State("source2B native factory installed twice".into()))
    }
    pub(crate) async fn prepare_mlp_weights(
        self: &Arc<Self>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        if self.recipe.cut() != ComponentCut::Mlp
            || self.gdn.get().is_none()
            || self.producer.projections.get().is_some()
            || self.producer.mlp_weights.get().is_some()
        {
            return Err(Error::State(
                "MLP weights require same fully prepared managed graph before native activation"
                    .into(),
            ));
        }
        let weights = FixedPromptMlpWeights::upload(
            self.recipe.clone(),
            self.producer.values.service().clone(),
            self.completion_owner(),
            cancelled,
        )
        .await?;
        self.producer
            .mlp_weights
            .set(weights)
            .map_err(|_| Error::State("MLP source weights installed twice".into()))
    }
    pub(crate) fn prepare_mlp_native(
        self: &Arc<Self>,
        inputs: FixedPromptMlpQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        not_cancelled(cancelled)?;
        if self.recipe.cut() != ComponentCut::Mlp
            || self.gdn.get().is_none()
            || self.producer.projections.get().is_some()
        {
            return Err(Error::State(
                "same eleven-node managed graph required before native activation".into(),
            ));
        }
        let weights = self.mlp_weights()?.clone();
        init.retain_completion_owner(self.completion_owner())?;
        let programs = ComponentProjections::prepare_mlp(
            self.producer.values.clone(),
            weights,
            inputs,
            init,
            cancelled,
        )?;
        self.producer
            .projections
            .set(programs)
            .map_err(|_| Error::State("MLP native factory installed twice".into()))
    }
    pub(crate) async fn prepare_qwen2b_mlp_weights(
        self: &Arc<Self>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        if self.recipe.cut() != ComponentCut::Qwen2bMlp
            || self.gdn.get().is_none()
            || self.producer.projections.get().is_some()
            || self.producer.qwen2b_mlp_weights.get().is_some()
        {
            return Err(Error::State(
                "new2B W requires same managed graph before native activation".into(),
            ));
        }
        let cohort = Qwen2bMlpWeightCohort::bind(&self.recipe)?;
        let weights = Qwen2bMlpWeights::upload(
            cohort,
            self.producer.values.service().clone(),
            self.completion_owner(),
            cancelled,
        )
        .await?;
        weights.require_bindings(self.recipe.bindings(), self.producer.values.service())?;
        self.producer
            .qwen2b_mlp_weights
            .set(weights)
            .map_err(|_| Error::State("new2B W installed twice".into()))
    }
    fn qwen2b_mlp_weights(&self) -> Result<&Arc<Qwen2bMlpWeights>> {
        if self.recipe.cut() != ComponentCut::Qwen2bMlp {
            return Err(Error::Unsupported("new2B MLP W cut only".into()));
        }
        self.producer
            .qwen2b_mlp_weights
            .get()
            .ok_or_else(|| Error::State("new2B initialized W absent".into()))
    }
    pub(crate) fn qwen2b_mlp_verification_plan(
        self: &Arc<Self>,
        cut: Option<crate::resident_leases::Qwen2bMlpComputeReadCut>,
    ) -> Result<Arc<Qwen2bMlpSourceVerification>> {
        self.qwen2b_mlp_weights()?
            .compute_verification_plan(self.completion_owner(), cut)
    }
    pub(crate) fn qwen2b_mlp_retirement_plan(
        self: &Arc<Self>,
    ) -> Result<Arc<Qwen2bMlpSourceVerification>> {
        self.qwen2b_mlp_weights()?
            .verification_plan(self.completion_owner())
    }
    pub(crate) fn qwen2b_mlp_weight_bytes(&self) -> Result<usize> {
        Ok(self.qwen2b_mlp_weights()?.immutable_bytes())
    }
    pub(crate) fn qwen2b_mlp_weight_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        Ok(self.qwen2b_mlp_weights()?.accesses())
    }
    pub(crate) fn qwen2b_mlp_plan_retained(
        self: &Arc<Self>,
        plan: &Arc<Qwen2bMlpSourceVerification>,
    ) -> Result<bool> {
        Ok(
            plan.retains_compute_graph(self.qwen2b_mlp_weights()?, &self.completion_owner())
                && self.recipe.qwen2b_mlp_record_admission_retained(),
        )
    }
    pub(crate) fn qwen2b_mlp_graph_retained(&self) -> Result<bool> {
        Ok(self.recipe.cut() == ComponentCut::Qwen2bMlp
            && self.producer.qwen2b_mlp_weights.get().is_some()
            && self.post_attention_graph_retained()?)
    }
    pub(crate) fn prepare_qwen2b_mlp_native(
        self: &Arc<Self>,
        inputs: crate::cuda::native::qwen2b_mlp::Qwen2bMlpQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        not_cancelled(cancelled)?;
        if self.recipe.cut() != ComponentCut::Qwen2bMlp
            || self.gdn.get().is_none()
            || self.producer.projections.get().is_some()
        {
            return Err(Error::State(
                "same new2B eleven-node managed graph before native activation".into(),
            ));
        }
        let weights = self.qwen2b_mlp_weights()?.clone();
        weights.require_bindings(self.recipe.bindings(), self.producer.values.service())?;
        init.retain_completion_owner(self.completion_owner())?;
        let programs = ComponentProjections::prepare_qwen2b_mlp(
            self.producer.values.clone(),
            weights,
            inputs,
            init,
            cancelled,
        )?;
        self.producer
            .projections
            .set(programs)
            .map_err(|_| Error::State("new2B native factory installed twice".into()))
    }
    fn mlp_weights(&self) -> Result<&Arc<FixedPromptMlpWeights>> {
        self.producer
            .mlp_weights
            .get()
            .ok_or_else(|| Error::State("MLP original initialized source weights absent".into()))
    }
    pub(crate) fn append_mlp_weight_readbacks(
        &self,
        reads: &mut Vec<ResidentReadback>,
    ) -> Result<std::ops::Range<usize>> {
        if self.recipe.cut() != ComponentCut::Mlp {
            return Err(Error::Unsupported(
                "new MLP W cohort unavailable on old cut".into(),
            ));
        }
        self.mlp_weights()?.append_readbacks(reads)
    }
    pub(crate) fn validate_mlp_weight_readbacks(
        &self,
        raw: &[Bytes],
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        if self.recipe.cut() != ComponentCut::Mlp {
            return Err(Error::Unsupported(
                "new MLP W verification unavailable on old cut".into(),
            ));
        }
        self.mlp_weights()?.validate_completed(raw, cancelled)
    }
    pub(crate) fn mlp_weight_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        Ok(self.mlp_weights()?.accesses())
    }
    pub(crate) fn mlp_weight_bytes(&self) -> Result<usize> {
        Ok(self.mlp_weights()?.immutable_bytes())
    }
    pub(crate) fn mlp_graph_retained(&self) -> Result<bool> {
        Ok(self.recipe.cut() == ComponentCut::Mlp
            && self.producer.mlp_weights.get().is_some()
            && self.post_attention_graph_retained()?)
    }
    pub(crate) fn require_attempt(&self, permit: &PrefixPhaseEnqueue<'_>) -> Result<()> {
        permit.require_invocation(
            self.producer.values.service(),
            self.recipe.bindings().call(),
            0,
            1,
        )
    }
    pub(crate) fn begin_launch(
        &self,
        cancelled: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<()> {
        not_cancelled(cancelled)?;
        if self.gdn.get().is_none() || self.producer.projections.get().is_none() {
            return Err(Error::State(
                "source2B complete authentic component preparation required".into(),
            ));
        }
        self.cursor.begin(
            self.producer.values.service(),
            permit
                .ok_or_else(|| Error::State("original borrowed component attempt absent".into()))?,
        )
    }
    pub(crate) fn validate_runtime(&self, p: RuntimeParameters) -> Result<()> {
        if p.position != 0
            || p.valid_tokens != self.recipe.bindings().call().rows().len()
            || self.recipe.bindings().call().rows().ids().first().copied() != Some(p.token)
        {
            return Err(Error::State(
                "source2B exact original initial teacher span".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn enqueue(
        &self,
        index: usize,
        node: &PlanNode,
        cancelled: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        let permit = permit
            .ok_or_else(|| Error::State("source2B original attempt absent at node".into()))?;
        self.require_attempt(permit)?;
        self.cursor.take(index, node)?;
        match index {
            0 => Ok(0), // Actual authenticated source rows already uploaded, never normalized aliases.
            1 => {
                self.producer.rms.enqueue()?;
                Ok(10)
            }
            2 | 3 => {
                let count = self
                    .producer
                    .projections
                    .get()
                    .ok_or_else(|| Error::State("source2B native factory absent".into()))?
                    .enqueue(
                        self.recipe.operator(index)?,
                        self.producer.values.lease(ValueId(1))?,
                        self.producer
                            .values
                            .lease(ValueId(if index == 2 { 2 } else { 3 }))?,
                        &self.producer.workspace,
                        cancelled,
                    )?;
                self.native_launches.fetch_add(count, Ordering::AcqRel);
                Ok(count)
            }
            4 | 5 => {
                let gdn = self
                    .gdn
                    .get()
                    .ok_or_else(|| Error::State("source2B actual GDN absent".into()))?;
                self.cursor.require_stateful_node(
                    index,
                    gdn.domains(),
                    self.producer.values.service(),
                    permit,
                )?;
                let count = gdn.enqueue(
                    index,
                    permit,
                    if index == 5 && self.fault.load(Ordering::Acquire) == 2 {
                        self.internal_cancel.store(true, Ordering::Release);
                        Some(&self.internal_cancel)
                    } else {
                        cancelled
                    },
                )?;
                if index == 4 && matches!(self.fault.load(Ordering::Acquire), 1 | 2) {
                    self.cut.store(true, Ordering::Release);
                    if self.fault.load(Ordering::Acquire) == 1 {
                        return Err(Error::State(
                            match self.recipe.kind() {
                                ComponentKind::Qwen2b => {
                                    "deliberate source2B unknown after actual Conv"
                                }
                                ComponentKind::FixedPrompt(_) => {
                                    "deliberate canonical47 unknown after actual Conv"
                                }
                            }
                            .into(),
                        ));
                    }
                }
                Ok(count)
            }
            6 if self.recipe.cut().includes_post_attention() => {
                let count = self
                    .producer
                    .projections
                    .get()
                    .ok_or_else(|| Error::State("GdnOut original program absent".into()))?
                    .enqueue(
                        self.recipe.operator(6)?,
                        self.producer.values.lease(ValueId(6))?,
                        self.producer.values.lease(ValueId(7))?,
                        &self.producer.workspace,
                        cancelled,
                    )?;
                self.native_launches.fetch_add(count, Ordering::AcqRel);
                if matches!(self.fault.load(Ordering::Acquire), 3 | 4) {
                    self.cut.store(true, Ordering::Release);
                    if self.fault.load(Ordering::Acquire) == 3 {
                        return Err(Error::State(
                            "deliberate source component unknown after actual GdnOut".into(),
                        ));
                    }
                }
                Ok(count)
            }
            7 if self.recipe.cut().includes_post_attention() => {
                let rms = self.producer.post_attention.as_ref().ok_or_else(|| {
                    Error::State("postAttention actual dual producer absent".into())
                })?;
                rms.require_component_values(&self.producer.values)?;
                let cancel = if self.fault.load(Ordering::Acquire) == 4 {
                    self.internal_cancel.store(true, Ordering::Release);
                    Some(&self.internal_cancel)
                } else {
                    cancelled
                };
                rms.enqueue(cancel)?;
                Ok(10)
            }
            8..=10 if self.recipe.cut().includes_mlp() => {
                let (input, output) = match index {
                    8 => (8, 10),
                    9 => (10, 11),
                    _ => (11, 12),
                };
                if index == 9 && self.fault.load(Ordering::Acquire) == 6 {
                    self.internal_cancel.store(true, Ordering::Release);
                    not_cancelled(Some(&self.internal_cancel))?;
                }
                let count = self
                    .producer
                    .projections
                    .get()
                    .ok_or_else(|| Error::State("MLP actual native graph absent".into()))?
                    .enqueue(
                        self.recipe.operator(index)?,
                        self.producer.values.lease(ValueId(input))?,
                        self.producer.values.lease(ValueId(output))?,
                        &self.producer.workspace,
                        cancelled,
                    )?;
                self.native_launches.fetch_add(count, Ordering::AcqRel);
                if index == 8 && matches!(self.fault.load(Ordering::Acquire), 5 | 6) {
                    self.cut.store(true, Ordering::Release);
                    if self.fault.load(Ordering::Acquire) == 5 {
                        return Err(Error::State(
                            if self.recipe.cut() == ComponentCut::Qwen2bMlp {
                                "deliberate qwen2b MLP unknown after actual GateUp"
                            } else {
                                "deliberate canonical47 MLP unknown after actual GateUp"
                            }
                            .into(),
                        ));
                    }
                }
                Ok(count)
            }
            _ => Err(Error::Unsupported(
                "source2B no GdnOut or later node".into(),
            )),
        }
    }
    pub(crate) fn finish_nodes(&self) -> Result<()> {
        self.cursor.finished()
    }
    pub(crate) fn certify(&self, value: &DeviceBuffer, status: &Handle) -> Result<()> {
        certify_original(&self.certificates, value, status, self.status())
    }
    pub(crate) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut accesses = self.producer.rms.accesses().to_vec();
        if let Some(post) = &self.producer.post_attention {
            merge(&mut accesses, post.accesses().iter().cloned());
        }
        merge(&mut accesses, self.producer.values.accesses());
        merge(
            &mut accesses,
            self.gdn
                .get()
                .ok_or_else(|| Error::State("source2B GDN absent".into()))?
                .accesses(),
        );
        let programs = self
            .producer
            .projections
            .get()
            .ok_or_else(|| Error::State("source2B programs absent".into()))?;
        for (node, out) in [(2, 2), (3, 3)] {
            merge(
                &mut accesses,
                programs.accesses(
                    self.recipe.operator(node)?,
                    self.producer.values.lease(ValueId(1))?,
                    self.producer.values.lease(ValueId(out))?,
                    &self.producer.workspace,
                )?,
            );
        }
        if self.recipe.cut().includes_post_attention() {
            merge(
                &mut accesses,
                programs.accesses(
                    self.recipe.operator(6)?,
                    self.producer.values.lease(ValueId(6))?,
                    self.producer.values.lease(ValueId(7))?,
                    &self.producer.workspace,
                )?,
            );
        }
        if self.recipe.cut().includes_mlp() {
            for (node, input, output) in [(8, 8, 10), (9, 10, 11), (10, 11, 12)] {
                merge(
                    &mut accesses,
                    programs.accesses(
                        self.recipe.operator(node)?,
                        self.producer.values.lease(ValueId(input))?,
                        self.producer.values.lease(ValueId(output))?,
                        &self.producer.workspace,
                    )?,
                );
            }
        }
        Ok(accesses)
    }
    /// Actual owners only; no service usability/readiness query on uncertain work.
    pub(crate) fn retained_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut accesses = self.producer.rms.accesses().to_vec();
        if let Some(post) = &self.producer.post_attention {
            merge(&mut accesses, post.accesses().iter().cloned());
        }
        merge(&mut accesses, self.producer.values.accesses());
        merge(
            &mut accesses,
            self.gdn
                .get()
                .ok_or_else(|| Error::State("source2B GDN retained owner absent".into()))?
                .accesses(),
        );
        let programs = self
            .producer
            .projections
            .get()
            .ok_or_else(|| Error::State("source2B retained native owner absent".into()))?;
        merge(
            &mut accesses,
            programs
                .immutable_weights()
                .into_iter()
                .map(|(lease, _)| (lease, DeviceAccess::Read)),
        );
        merge(
            &mut accesses,
            std::iter::once((self.producer.workspace.clone(), DeviceAccess::Write)),
        );
        if self.recipe.cut() == ComponentCut::Mlp {
            merge(&mut accesses, self.mlp_weights()?.accesses());
        }
        if self.recipe.cut() == ComponentCut::Qwen2bMlp {
            merge(&mut accesses, self.qwen2b_mlp_weights()?.accesses());
        }
        Ok(accesses)
    }
    pub(crate) fn post_attention_graph_retained(&self) -> Result<bool> {
        if !self.recipe.cut().includes_post_attention() {
            return Ok(false);
        }
        let Some(post) = &self.producer.post_attention else {
            return Ok(false);
        };
        post.require_component_values(&self.producer.values)?;
        Ok(self
            .producer
            .projections
            .get()
            .is_some_and(|p| p.retained_post_attention_values(&self.producer.values)))
    }
    pub(crate) fn outputs_unpromoted(&self) -> Result<bool> {
        (1..self.recipe.value_end()).try_fold(true, |okay, id| {
            Ok(okay
                && !self
                    .producer
                    .values
                    .lease(ValueId(id as u32))?
                    .is_initialized())
        })
    }
    pub(crate) fn handles(&self) -> Result<Vec<Handle>> {
        Ok(self
            .accesses()?
            .into_iter()
            .map(|(l, _)| l.handle())
            .collect())
    }
    pub(crate) fn readbacks(&self) -> Result<Vec<ResidentReadback>> {
        let mut reads = vec![self.status().readback(0, 4)?];
        reads.extend(self.producer.rms.readbacks()?);
        for id in 2..self.recipe.value_end() {
            let lease = self.producer.values.lease(ValueId(id as u32))?;
            reads.push(lease.readback(0, lease.bytes())?);
        }
        if let Some(post) = &self.producer.post_attention {
            let extra = post.readbacks()?;
            reads.extend(extra.into_iter().take(2));
        }
        Ok(reads)
    }
    pub(crate) fn append_fixed_prompt_diagnostics(
        &self,
        reads: &mut Vec<ResidentReadback>,
    ) -> Result<Vec<(crate::cuda::gdn::FixedPromptGdnIntermediate, usize)>> {
        let more = self
            .gdn
            .get()
            .ok_or_else(|| Error::State("47 actual GDN diagnostic owner absent".into()))?
            .fixed_prompt_diagnostic_readbacks()?;
        let mut map = Vec::with_capacity(more.len());
        for (key, read) in more {
            map.push((key, reads.len()));
            reads.push(read);
        }
        Ok(map)
    }
    pub(crate) fn validate_readbacks(&self, raw: &[Bytes]) -> Result<()> {
        if raw.len() < 2 + self.recipe.value_end() || raw[0][..] != [0; 4] {
            return Err(Error::State(
                "source2B actual finite transport/program status".into(),
            ));
        }
        self.producer.rms.validate_readback(&raw[1..4])?;
        for bytes in &raw[4..2 + self.recipe.value_end()] {
            if !crate::cuda::native::qwen2b_component::finite_component_words(bytes.as_ref()) {
                return Err(Error::Malformed(
                    "source2B actual finite computed prefix output".into(),
                ));
            }
        }
        if let Some(post) = &self.producer.post_attention {
            let gamma = 2 + self.recipe.value_end();
            let sum = gamma + 1;
            if raw.len() <= sum {
                return Err(Error::State(
                    "postAttention complete sum/gamma readbacks absent".into(),
                ));
            }
            post.validate_readback_words([
                &raw[gamma][..],
                &raw[sum][..],
                &raw[10][..],
                &raw[11][..],
            ])?;
        }
        Ok(())
    }
    pub(crate) fn immutable_weights(&self) -> Result<Vec<(ResidentLease, &[u8])>> {
        let mut weights = self
            .producer
            .projections
            .get()
            .ok_or_else(|| Error::State("source2B W absent".into()))?
            .immutable_weights();
        weights.extend(
            self.gdn
                .get()
                .ok_or_else(|| Error::State("source2B GDN W absent".into()))?
                .immutable_weights(),
        );
        if let Some(post) = &self.producer.post_attention {
            weights.push(post.immutable_gamma());
        }
        Ok(weights)
    }
    pub(crate) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        unsafe {
            self.producer
                .projections
                .get()
                .ok_or_else(|| Error::State("source2B retirement programs absent".into()))?
                .retire_after_physical_completion()
        }
    }
    pub(crate) fn native_launches(&self) -> usize {
        self.native_launches.load(Ordering::Acquire)
    }
    pub(crate) fn qwen2b_native_launches(&self) -> Result<usize> {
        self.producer
            .projections
            .get()
            .ok_or_else(|| Error::State("2B actual native graph absent".into()))?
            .qwen2b_native_launches()
    }
    pub(crate) fn qwen2b_down1_enqueues(&self) -> Result<usize> {
        if self.recipe.cut() != ComponentCut::Qwen2bMlp
            || self.recipe.bindings().call().rows().len() != 1
        {
            return Err(Error::Unsupported(
                "one-row2B Down record metadata only".into(),
            ));
        }
        self.producer
            .projections
            .get()
            .ok_or_else(|| Error::State("2B actual native graph absent".into()))?
            .down1_enqueues()
    }
    pub(crate) fn qwen2b_down1_output_unpromoted(&self) -> Result<bool> {
        self.producer
            .projections
            .get()
            .ok_or_else(|| Error::State("2B native programs absent".into()))?
            .down1_output_unpromoted()
    }
    pub(crate) fn arm_fault(&self, value: usize) -> Result<()> {
        if value > 8
            || value > 6
                && (self.recipe.cut() != ComponentCut::Qwen2bMlp
                    || self.recipe.bindings().call().rows().len() != 1)
            || value > 4 && !self.recipe.cut().includes_mlp()
            || value > 2 && !self.recipe.cut().includes_post_attention()
            || self
                .fault
                .compare_exchange(0, value, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(Error::State("source2B closed initial fault cut".into()));
        }
        if matches!(value, 7 | 8) {
            self.producer
                .projections
                .get()
                .ok_or_else(|| Error::State("2B Down native programs absent".into()))?
                .arm_qwen2b_down1_fault(value == 8)?;
        }
        Ok(())
    }
    pub(crate) fn cut_reached(&self) -> bool {
        self.cut.load(Ordering::Acquire)
    }
}
impl ResidentProgram {
    pub(crate) async fn prepare_component(
        service: Arc<ResidentService>,
        recipe: Arc<ComponentRecipe>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        let (body, values, preflight) =
            ComponentProgram::prepare_values(service.clone(), recipe.clone(), cancelled).await?;
        Ok(Self {
            _source_owner: ResidentProgramSource::Windows(recipe.source().clone()),
            plan: recipe.bindings().shared_plan(),
            preflight,
            values,
            status: body.status().handle(),
            owned_ranges: OwnedRanges::new(service.client().service_id()),
            body: ResidentProgramBody::Qwen2bComponent(body),
        })
    }
    pub(crate) fn qwen2b_component(&self) -> Result<&Arc<ComponentProgram>> {
        match &self.body {
            ResidentProgramBody::Qwen2bComponent(body) => Ok(body),
            _ => Err(Error::Unsupported(
                "not a source2B component program".into(),
            )),
        }
    }
    pub(crate) fn finalize_qwen2b_component(
        &mut self,
        state_ids: &BTreeSet<zerogravity_core::device_events::BufferLeaseId>,
    ) -> Result<()> {
        let owned = self
            .qwen2b_component()?
            .retained_accesses()?
            .into_iter()
            .filter(|(l, _)| !state_ids.contains(&l.id()))
            .collect::<Vec<_>>();
        let value_ids = self
            .values
            .iter()
            .map(|v| v.original_lease().map(|l| l.id()))
            .collect::<Result<BTreeSet<_>>>()?;
        let status_id = self.qwen2b_component()?.status().id();
        let (mut source, mut values, mut scratch, mut status) = (0usize, 0usize, 64 * MIB, 0usize);
        for (lease, _) in owned {
            if lease.id() == status_id {
                status += lease.bytes();
            } else if value_ids.contains(&lease.id()) {
                values += lease.bytes();
            } else if lease.role() == BufferRole::ImmutableWeights {
                source += lease.bytes();
            } else {
                scratch += lease.bytes();
            }
            self.owned_ranges
                .insert(BufferRange::new(&lease.handle())?)?;
        }
        self.preflight.source_bytes = source;
        self.preflight.value_bytes = values;
        self.preflight.workspace_bytes = scratch;
        self.preflight.status_bytes = status;
        if self.preflight.total_with_state()? > self.qwen2b_component()?.recipe.provider_bytes() {
            return Err(Error::Limit(
                "source2B actual component exceeds closed envelope".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "qwen2b_component/post_attention_tests.rs"]
mod post_attention_tests;
