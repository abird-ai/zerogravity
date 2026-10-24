//! Unpublished full24 body inside the ORIGINAL ResidentProgram node loop.
use super::bf16_greedy::PreparedBf16Greedy;
use super::native::{
    CertificateSet, cancelled, certify_original, prepare_certificates, state_requirements,
};
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState,
};
use crate::resident_leases::{ResidentLease, ResidentReadback, ResidentService};
use crate::vllm_cuda_gdn::{GdnArithmetic, GdnFamilyIdentity};
use crate::vllm_cuda_input_rms::{BoundModelRmsSource, FusedRmsPrepared, RmsPrepared};
use crate::vllm_cuda_qualification::{
    attention_cubin::{ModelAttentionBinding, ModelAttentionPrograms},
    model_gdn::ModelGdnPrepared,
    model_native::ModelNativePrograms,
    model_values::ModelValues,
    vocabulary_head::ModelHeadPrograms,
};
use std::collections::BTreeMap;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    ActivationPrecision, PlanOperation, RecordedVllm08Model, ValueId, Vllm08Kind, Vllm08Projection,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

/// All-phase native records own source/lease identities only. No body, phase
/// binding, previous Ready, borrowed Session or state publisher is stored here.
pub(crate) struct FullModelLibrary {
    pub(crate) source: Arc<OriginalWindowModelSource>,
    pub(crate) projections: Arc<ModelNativePrograms>,
    pub(crate) attention: Arc<ModelAttentionPrograms>,
    pub(crate) head: Arc<ModelHeadPrograms>,
    pub(crate) images: Arc<crate::vllm_cuda_qualification::native_plan::PreparedImages>,
}
struct BoundPhase {
    gdn_arithmetic: Arc<GdnArithmetic>,
    phase: usize,
    library: Option<Arc<FullModelLibrary>>,
    canonical47_prefix: Option<canonical47_prefix::PrefixPhase>,
    canonical_call: Option<canonical47_natural::CurrentCall>,
    short_hf: Option<short_hf::CurrentCall>,
    states: Vec<Arc<BoundModelState>>,
    gdn: BTreeMap<usize, Arc<ModelGdnPrepared>>,
    attention: BTreeMap<usize, Arc<ModelAttentionBinding>>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
}
pub(crate) struct FullModelProgram {
    // Certificate handles retire before the original value/status leases.
    certificates: Arc<CertificateSet>,
    gdn_family: GdnFamilyIdentity,
    canonical47:
        Option<Arc<crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding>>,
    source: Arc<OriginalWindowModelSource>,
    record: Arc<RecordedVllm08Model>,
    values: Arc<ModelValues>,
    rms: Arc<RmsPrepared>,
    _ordinary_binding: BoundModelRmsSource,
    fused: BTreeMap<usize, Arc<FusedRmsPrepared>>,
    bound: OnceLock<BoundPhase>,
    greedy: OnceLock<Arc<PreparedBf16Greedy>>,
    launched: AtomicBool,
    next_node: AtomicUsize,
    output_submitted: AtomicBool,
    fault: AtomicUsize,
    fault_cut: AtomicBool,
    convolution_dispatches: AtomicUsize,
    internal_cancel: AtomicBool,
    native_launches: AtomicUsize,
    workspace_clears: AtomicUsize,
    device_copies: AtomicUsize,
    _readback: Arc<Vec<WindowWorkspace>>,
    _metadata: Arc<WindowWorkspace>,
    natural: Option<
        Arc<crate::resident_inference::native_fragment::natural_publication::NaturalCallState>,
    >,
}

pub(crate) mod canonical47_natural;
mod canonical47_prefix;
mod canonical47_whole;
pub(crate) mod short_hf;
mod state_read;
use state_read::descriptor;

fn insert_access(
    all: &mut BTreeMap<
        zerogravity_core::device_events::BufferLeaseId,
        (ResidentLease, DeviceAccess),
    >,
    lease: ResidentLease,
    access: DeviceAccess,
) {
    let entry = all.entry(lease.id()).or_insert((lease, access));
    if access == DeviceAccess::Write {
        entry.1 = access;
    }
}
impl FullModelProgram {
    /// Pure source/recipe preflight. No allocator/client/module is consulted.
    pub(crate) fn require_record(
        source: &OriginalWindowModelSource,
        record: &RecordedVllm08Model,
    ) -> Result<()> {
        source.require_vllm08_full_source()?;
        record.call().validate()?;
        if record.call().model_profile()
            != zerogravity_core::accelerator::program::FullModelProfile::Qwen08
        {
            return Err(Error::Unsupported(
                "native full24 source geometry remains .8B only".into(),
            ));
        }
        // Private recorded type is minted only by the checked family recipe.
        let plan = record.plan();
        if plan.source() != source.source_identity()
            || plan.nodes().len() != 255
            || plan.state_nodes().len() != 255
            || plan.values().len() != 327
            || plan.required_domains().len() != 42
            || plan.frontier_delta() != 0
            || plan.logits().is_some()
            || record.boundaries().len() != 24
        {
            return Err(Error::Unsupported(
                "closed actual original full24 unpublished source recipe".into(),
            ));
        }
        for node in plan.nodes() {
            if let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation {
                descriptor.validate()?;
                if descriptor.call() != record.call() {
                    return Err(Error::State("full24 recipe contains a foreign call".into()));
                }
            }
            for id in node.inputs.iter().chain(&node.outputs) {
                plan.vllm_value_provenance(*id)?
                    .ok_or_else(|| Error::State("full24 actual recipe provenance absent".into()))?
                    .require_invocation(record.call().request())?;
            }
        }
        Ok(())
    }
    /// Caller holds the original fair preparation slot. All managed resources
    /// exist before any shared cubin record is loaded. Gamma/embedding bytes
    /// come from original source windows, never reference archives.
    pub(crate) async fn prepare_values(
        service: Arc<ResidentService>,
        source: Arc<OriginalWindowModelSource>,
        record: Arc<RecordedVllm08Model>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        cancelled(cancel)?;
        // Preserve the existing legacy entry and its finite geometry permit.
        record.call().request().legacy_phase()?;
        Self::prepare_values_impl(service, source, record, None, None, cancel).await
    }
    pub(crate) async fn prepare_canonical47_prefix_values(
        service: Arc<ResidentService>,
        cut: Arc<crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        cancelled(cancel)?;
        Self::prepare_values_impl(
            service,
            cut.scope().source().source().clone(),
            cut.scope().record().clone(),
            Some(cut.scope().clone()),
            Some(cut),
            cancel,
        )
        .await
    }
    async fn prepare_values_impl(
        service: Arc<ResidentService>,
        source: Arc<OriginalWindowModelSource>,
        record: Arc<RecordedVllm08Model>,
        canonical47: Option<
            Arc<crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding>,
        >,
        cut: Option<
            Arc<crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut>,
        >,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        cancelled(cancel)?;
        Self::require_record(&source, &record)?;
        if service.source_identity() != source.source_identity() {
            return Err(Error::State(
                "full24 source/service before original upload".into(),
            ));
        }
        let metadata = source.source().reserve_workspace(4 * 1024 * 1024)?;
        let plan = record.plan();
        let node_end = cut.as_ref().map_or(255, |c| c.nodes().len());
        let value_end = cut.as_ref().map_or(327, |c| c.values().len());
        let ordinary = descriptor(&plan.nodes()[1])?;
        if ordinary.kind() != Vllm08Kind::InputRms
            || plan.nodes()[1].inputs != [ValueId(0)]
            || plan.nodes()[1].outputs != [ValueId(1)]
        {
            return Err(Error::State(
                "full24 genuine ordinary input RMS producer".into(),
            ));
        }
        let ordinary_binding = if let Some(scope) = &canonical47 {
            scope.rms(1)?.clone()
        } else {
            BoundModelRmsSource::bind(source.clone(), ordinary.clone())?
        };
        let states = state_requirements(&source, plan)?;
        let state_bytes = checked_sum(states.iter().map(|r| r.bytes))?;
        let public_bytes = checked_sum(
            plan.values()[..value_end]
                .iter()
                .map(|v| {
                    v.capacity_elements()?
                        .checked_mul(2)
                        .ok_or_else(|| Error::Limit("full24 readback extent".into()))
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        // All public values + ordinary/dual RMS source/sums + both canonical
        // roots + adapter/status metadata; full W uses bounded window readback.
        let readback_bytes = checked_sum([
            public_bytes,
            2 * state_bytes,
            48 * (2048 + record.call().rows() * 8192),
            2048 + record.call().rows() * 4096,
            1024 * 1024,
        ])?;
        let chunks = readback_bytes.div_ceil(32 * 1024 * 1024);
        if !(3..=8).contains(&chunks) {
            return Err(Error::Limit("closed full24 per-phase D2H admission".into()));
        }
        let readback = (0..chunks)
            .map(|_| source.source().reserve_workspace(32 * 1024 * 1024))
            .collect::<Result<Vec<_>>>()?;
        let rms = if let Some(scope) = &canonical47 {
            RmsPrepared::prepare_canonical47_model(service.clone(), scope.clone(), 0, cancel)
                .await?
        } else {
            RmsPrepared::prepare(
                service.clone(),
                source.source().clone(),
                record.call().request().teachers().ids(),
                0,
            )
            .await?
        };
        cancelled(cancel)?;
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(source.clone())?;
        init.retain_completion_owner(rms.completion_owner())?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let fused_outputs = plan.nodes()[..node_end]
            .iter()
            .filter(|n| {
                matches!(&n.operation, PlanOperation::TypedVllm08 { descriptor, .. }
                if matches!(descriptor.kind(), Vllm08Kind::ResidualRms(_)))
            })
            .flat_map(|n| n.outputs.iter().map(|id| id.0))
            .collect::<BTreeSet<_>>();
        let mut buffers: Vec<Option<DeviceBuffer>> = vec![None; value_end];
        for (index, shape) in plan.values()[..value_end].iter().enumerate() {
            if plan.precisions()[index] != ActivationPrecision::Bf16WidenedF32 {
                return Err(Error::Unsupported(
                    "full24 actual BF16 physical values".into(),
                ));
            }
            if fused_outputs.contains(&(index as u32)) {
                continue;
            }
            let layout =
                ResidentBufferLayout::new(ResidentPhysicalType::Bf16, shape.capacity_elements()?)?;
            let lease = match index {
                0 => rms.embedding().clone(),
                1 => rms.normalized().clone(),
                _ => init.allocate_empty(layout.bytes()?, BufferRole::Activation)?,
            };
            buffers[index] = Some(DeviceBuffer::from_lease(
                lease.clone(),
                layout,
                lease.role(),
                DeviceAccess::Write,
            )?);
        }
        drop(init.complete_sync().await?);
        let mut fused = BTreeMap::new();
        for (index, node) in plan.nodes()[..node_end].iter().enumerate() {
            let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                continue;
            };
            if !matches!(descriptor.kind(), Vllm08Kind::ResidualRms(_)) {
                continue;
            }
            let [left, right] = node.inputs.as_slice() else {
                return Err(Error::State("full24 dualRMS input arity".into()));
            };
            let input = |id: ValueId| {
                buffers[id.0 as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        Error::State("full24 actual earlier RMS producer absent".into())
                    })?
                    .original_lease()
                    .cloned()
            };
            let prepared = if let Some(scope) = &canonical47 {
                FusedRmsPrepared::prepare_canonical47_model(
                    service.clone(),
                    scope,
                    index,
                    input(*left)?,
                    input(*right)?,
                    0,
                    cancel,
                )
                .await?
            } else {
                FusedRmsPrepared::prepare_model(
                    service.clone(),
                    BoundModelRmsSource::bind(source.clone(), descriptor.as_ref().clone())?,
                    input(*left)?,
                    input(*right)?,
                    0,
                    cancel,
                )
                .await?
            };
            prepared.require_full_model_operator(descriptor)?;
            for (id, lease) in node
                .outputs
                .iter()
                .zip([prepared.normalized(), prepared.residual()])
            {
                let layout = ResidentBufferLayout::new(
                    ResidentPhysicalType::Bf16,
                    record.call().rows() * 1024,
                )?;
                if buffers[id.0 as usize].is_some() {
                    return Err(Error::State("full24 duplicate RMS output binding".into()));
                }
                buffers[id.0 as usize] = Some(DeviceBuffer::from_lease(
                    lease.clone(),
                    layout,
                    lease.role(),
                    DeviceAccess::Write,
                )?);
            }
            fused.insert(index, prepared);
        }
        if fused.len() != cut.as_ref().map_or(48, |c| c.duals()) {
            return Err(Error::State("full24 exact48 dualRMS producers".into()));
        }
        let buffers = buffers
            .into_iter()
            .map(|v| v.ok_or_else(|| Error::State("full24 unbound original ValueID".into())))
            .collect::<Result<Vec<_>>>()?;
        let values = if let Some(cut) = &cut {
            ModelValues::bind_canonical47_prefix(cut, &buffers, service.clone(), status.clone())?
        } else {
            ModelValues::bind(record.clone(), &buffers, service.clone(), status.clone())?
        };
        let certificates = prepare_certificates(&service, &buffers, &status)?;
        let preflight = ResidentPreflight {
            source: source.source_identity(),
            source_bytes: 0,
            value_bytes: public_bytes,
            workspace_bytes: 0,
            status_bytes: 4,
            parameter_bytes: 0,
            prior_state_bytes: state_bytes,
            candidate_state_bytes: state_bytes,
            states,
            scratch: [0; SCRATCH_SLOTS],
        };
        Ok((
            Arc::new(Self {
                gdn_family: GdnFamilyIdentity::original08(),
                canonical47,
                source,
                record,
                values,
                rms,
                _ordinary_binding: ordinary_binding,
                fused,
                certificates: Arc::new(certificates),
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
                _readback: Arc::new(readback),
                _metadata: Arc::new(metadata),
                natural: None,
            }),
            buffers,
            preflight,
        ))
    }
    pub(crate) fn prepare_greedy(
        &self,
        init: &mut crate::resident_leases::ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        if let Some(scope) = &self.canonical47 {
            scope.require_native_activation()?;
        }
        if self.greedy.get().is_some() || self.bound.get().is_some() {
            return Err(Error::State(
                "full24 output consumer prepared once before native modules".into(),
            ));
        }
        let prepared =
            PreparedBf16Greedy::prepare(self.source.clone(), self.values.clone(), init, cancel)?;
        self.greedy
            .set(prepared)
            .map_err(|_| Error::State("full24 output consumer already bound".into()))
    }
    /// Per-phase state ownership; shared library deliberately has no reverse
    /// edge into these phase/prior-Ready bindings.
    pub(crate) fn bind_phase(
        &self,
        phase: usize,
        library: Arc<FullModelLibrary>,
        states: Vec<Arc<BoundModelState>>,
        gdn: BTreeMap<usize, Arc<ModelGdnPrepared>>,
        attention: BTreeMap<usize, Arc<ModelAttentionBinding>>,
    ) -> Result<()> {
        if let Some(scope) = &self.canonical47 {
            scope.require_native_activation()?;
        }
        if self.bound.get().is_some()
            || self.launched.load(Ordering::Acquire)
            || states.len() != 24
            || gdn.len() != 18
            || attention.len() != 6
            || phase >= 5
        {
            return Err(Error::State(
                "full24 exact once-bound indexed phase resources".into(),
            ));
        }
        if !Arc::ptr_eq(&library.source, &self.source) {
            return Err(Error::State(
                "full24 native library immutable source owner differs".into(),
            ));
        }
        library.projections.require_phase(phase, &self.values)?;
        library.head.require_phase(phase, &self.values)?;
        library.attention.require_phase(phase, &attention)?;
        let mut layers = BTreeSet::new();
        for state in &states {
            if state.call() != self.record.call() || !layers.insert(state.layer().index()) {
                return Err(Error::State(
                    "full24 bound actual indexed source-state cohort".into(),
                ));
            }
        }
        if layers != (0..24).collect() {
            return Err(Error::State(
                "full24 all original indexed layers required".into(),
            ));
        }
        for state in &states {
            if state.layer().is_attention() {
                attention
                    .get(&state.layer().index())
                    .ok_or_else(|| Error::State("full24 attention layer-key mismatch".into()))?
                    .require_value_state(&self.values, state)?;
            } else {
                gdn.get(&state.layer().index())
                    .ok_or_else(|| Error::State("full24 GDN layer-key mismatch".into()))?
                    .require_value_state(&self.values, state)?;
            }
        }
        let gdn_arithmetic = gdn
            .values()
            .next()
            .ok_or_else(|| Error::State("full24 missing GDN arithmetic".into()))?
            .arithmetic()
            .clone();
        self.gdn_family.require_same(gdn_arithmetic.family())?;
        for prepared in gdn.values() {
            gdn_arithmetic.require_same(prepared.arithmetic())?;
        }
        let mut accesses = BTreeMap::new();
        for (lease, access) in self
            .values
            .accesses()
            .into_iter()
            .chain(self.rms.accesses().iter().cloned())
        {
            insert_access(&mut accesses, lease, access);
        }
        for fused in self.fused.values() {
            for (lease, access) in fused.accesses() {
                insert_access(&mut accesses, lease.clone(), *access);
            }
        }
        for state in &states {
            for (lease, access) in state.leases() {
                insert_access(&mut accesses, lease, access);
            }
        }
        for prepared in gdn.values() {
            for (lease, access) in prepared.accesses() {
                insert_access(&mut accesses, lease, access);
            }
        }
        for binding in attention.values() {
            for (lease, access) in binding.accesses()? {
                insert_access(&mut accesses, lease, access);
            }
        }
        for (lease, access) in library
            .projections
            .accesses(phase)?
            .iter()
            .cloned()
            .chain(library.head.accesses(phase)?)
        {
            insert_access(&mut accesses, lease, access);
        }
        // Attention factory accesses include only this phase's leases. It owns
        // native bindings without original BoundModelState/previous Ready.
        for (lease, access) in library.attention.accesses(phase)? {
            insert_access(&mut accesses, lease, access);
        }
        for (lease, access) in self
            .greedy
            .get()
            .ok_or_else(|| Error::State("full24 genuine Head output consumer not prepared".into()))?
            .accesses()
        {
            insert_access(&mut accesses, lease.clone(), *access);
        }
        self.bound
            .set(BoundPhase {
                gdn_arithmetic,
                phase,
                library: Some(library),
                canonical47_prefix: None,
                canonical_call: None,
                short_hf: None,
                states,
                gdn,
                attention,
                accesses: accesses.into_values().collect(),
            })
            .map_err(|_| Error::State("full24 phase already bound".into()))
    }
    pub(crate) fn values(&self) -> &Arc<ModelValues> {
        &self.values
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    pub(crate) fn status(&self) -> &ResidentLease {
        self.values.status()
    }
    /// Immutable source gamma owners; no bound phase, root or value reuse.
    pub(crate) fn cohort_gammas(
        &self,
    ) -> Result<BTreeMap<usize, crate::vllm_cuda_input_rms::cohort::CohortGamma>> {
        let mut gammas = BTreeMap::new();
        gammas.insert(1, self.rms.cohort_gamma(&self._ordinary_binding)?);
        for (&node, rms) in &self.fused {
            gammas.insert(node, rms.cohort_gamma()?);
        }
        if gammas.len() != 49 {
            return Err(Error::State(
                "cohort all49 original source gamma owners".into(),
            ));
        }
        Ok(gammas)
    }
    pub(crate) fn fused(&self) -> &BTreeMap<usize, Arc<FusedRmsPrepared>> {
        &self.fused
    }
    pub(crate) fn final_normalizer(&self) -> Result<Arc<FusedRmsPrepared>> {
        self.fused
            .iter()
            .next_back()
            .map(|(_, v)| v.clone())
            .ok_or_else(|| Error::State("full24 final normalizer absent".into()))
    }
    pub(crate) fn accesses(&self) -> Result<&[(ResidentLease, DeviceAccess)]> {
        let accesses = &self
            .bound
            .get()
            .ok_or_else(|| Error::State("full24 original phase not bound".into()))?
            .accesses;
        self.values.require_storage_flight(accesses)?;
        Ok(accesses)
    }
    pub(crate) fn completion_owner(self: &Arc<Self>) -> CompletionOwner {
        self.clone()
    }
    pub(crate) fn handles(&self) -> Result<Vec<Handle>> {
        Ok(self.accesses()?.iter().map(|(l, _)| l.handle()).collect())
    }
    pub(crate) fn require_attempt(&self, permit: &PrefixPhaseEnqueue<'_>) -> Result<()> {
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("original model phase absent".into()))?;
        permit.require_invocation(
            self.values.service(),
            self.record.call().request(),
            if bound
                .canonical47_prefix
                .as_ref()
                .is_some_and(|p| !p.cut.is_whole())
            {
                3
            } else {
                24
            },
        )?;
        for state in &self
            .bound
            .get()
            .ok_or_else(|| Error::State("full24 state preparation absent".into()))?
            .states
        {
            state.require_enqueue(self.values.service(), permit)?;
        }
        Ok(())
    }
    pub(crate) fn begin_launch(
        &self,
        cancel: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<()> {
        cancelled(cancel)?;
        self.require_attempt(permit.ok_or_else(|| {
            Error::State("full24 genuine original attempt permit required before ANY launch".into())
        })?)?;
        self.accesses()?;
        if self.launched.swap(true, Ordering::AcqRel) {
            return Err(Error::State("full24 original phase executes once".into()));
        }
        self.values.begin_storage()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, p: RuntimeParameters) -> Result<()> {
        let request = self.record.call().request();
        if p.position != request.teachers().base()
            || p.valid_tokens != self.record.call().rows()
            || Some(p.token) != request.teachers().ids().first().copied()
        {
            return Err(Error::State(
                "full24 runtime differs from original source teacher/phase".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn enqueue(
        &self,
        index: usize,
        node: &PlanNode,
        cancel: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<usize> {
        if !self.launched.load(Ordering::Acquire) || self.next_node.load(Ordering::Acquire) != index
        {
            return Err(Error::State(
                "full24 producer must follow ORIGINAL node traversal in order".into(),
            ));
        }
        let expected = self
            .execution_nodes()?
            .get(index)
            .ok_or_else(|| Error::State("full24 original node index absent".into()))?;
        if node.inputs != expected.inputs
            || node.outputs != expected.outputs
            || node.state != expected.state
            || index > 0 && descriptor(node)? != descriptor(expected)?
        {
            return Err(Error::State(
                "full24 caller node differs from sealed original producer SSA".into(),
            ));
        }
        self.values.begin_storage_node(index, node)?;
        let n = self.enqueue_inner(index, node, cancel, permit)?;
        self.next_node
            .compare_exchange(index, index + 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("full24 concurrent/reordered producer submission".into()))?;
        Ok(n)
    }
    fn enqueue_inner(
        &self,
        index: usize,
        node: &PlanNode,
        cancel: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<usize> {
        cancelled(cancel)?;
        let permit =
            permit.ok_or_else(|| Error::State("full24 original borrowed permit absent".into()))?;
        if let Some(current) = self.bound.get().and_then(|bound| bound.short_hf.as_ref()) {
            // begin_launch checked the complete immutable prior cohort before
            // node0. This borrowed permit prevents restore/publication; priors
            // stay retained Reads. Recheck current authority and physical fence
            // at every original node, without rewalking all24 states.
            self.values.service().ensure_usable()?;
            current
                .input
                .require_permit(permit, self.values.service())?;
        } else {
            self.require_attempt(permit)?;
        }
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("full24 state/library preparation absent".into()))?;
        if matches!(node.operation, PlanOperation::TypedVllmBf16Embedding { .. }) {
            if index != 0 {
                return Err(Error::State("full24 source embedding index differs".into()));
            }
            if let Some(current) = &bound.short_hf {
                self.values.require_storage_node(index)?;
                current
                    .input
                    .require_permit(permit, self.values.service())?;
                let rows = self.rms.enqueue_short_hf_embedding(permit, cancel)?;
                let positions = current
                    .positions
                    .enqueue(permit.session(), permit, cancel)?;
                let count = rows + positions;
                self.device_copies.fetch_add(count, Ordering::AcqRel);
                return Ok(count);
            }
            let mut copies = self.rms.enqueue_embedding(permit.session(), cancel)?;
            if let Some(current) = &bound.canonical_call {
                copies += current.positions.enqueue(permit.session(), cancel)?;
            }
            self.device_copies.fetch_add(copies, Ordering::AcqRel);
            return Ok(copies);
        }
        let op = descriptor(node)?;
        if op.call() != self.record.call() {
            return Err(Error::State(
                "full24 original node invocation differs".into(),
            ));
        }
        if let Some(current) = &bound.short_hf
            && !matches!(
                op.kind(),
                Vllm08Kind::InputRms
                    | Vllm08Kind::ResidualRms(_)
                    | Vllm08Kind::GdnConv
                    | Vllm08Kind::GdnRecurrent
            )
        {
            current
                .input
                .require_permit(permit, self.values.service())?;
            let before = current.native.submission_counts();
            let result = current
                .native
                .enqueue(index, permit.session(), permit, cancel);
            let after = current.native.submission_counts();
            let native = after
                .native
                .checked_sub(before.native)
                .ok_or_else(|| Error::State("short-HF native counter regressed".into()))?;
            let clears = after
                .clears
                .checked_sub(before.clears)
                .ok_or_else(|| Error::State("short-HF clear counter regressed".into()))?;
            let copies = after
                .copies
                .checked_sub(before.copies)
                .ok_or_else(|| Error::State("short-HF copy counter regressed".into()))?;
            self.native_launches.fetch_add(native, Ordering::AcqRel);
            self.workspace_clears.fetch_add(clears, Ordering::AcqRel);
            self.device_copies.fetch_add(copies, Ordering::AcqRel);
            if result.is_err()
                && index == 254
                && matches!(self.fault.load(Ordering::Acquire), 20 | 21)
            {
                let n = current.native.head().counts()?;
                let deliberate = matches!(&result,Err(Error::State(message)) if
                    message=="streamed head actual native cut before checkpoint"
                    || message=="streamed head actual cancellation before next overwrite"
                    || message=="resident head actual native cut before original completion"
                    || message=="resident head actual cancellation after native enqueue");
                if deliberate && n[1] > 0 {
                    self.fault_cut.store(true, Ordering::Release);
                }
            }
            let submitted = result?;
            if (submitted.native, submitted.clears, submitted.copies) != (native, clears, copies) {
                return Err(Error::State(
                    "short-HF concrete lowerer command deltas differ".into(),
                ));
            }
            if self.fault.load(Ordering::Acquire) == 19
                && op.kind() == Vllm08Kind::FlashAttention
                && op.layer().is_some_and(|l| l.index() == 23)
            {
                self.fault_cut.store(true, Ordering::Release);
                return Err(Error::State(
                    "deliberate short-HF unknown after actual last FA".into(),
                ));
            }
            return Ok(submitted.native + submitted.clears + submitted.copies);
        }
        if let Some(current) = &bound.canonical_call
            && !matches!(
                op.kind(),
                Vllm08Kind::InputRms
                    | Vllm08Kind::ResidualRms(_)
                    | Vllm08Kind::GdnConv
                    | Vllm08Kind::GdnRecurrent
            )
        {
            current
                .input
                .require_permit(permit, self.values.service())?;
            let submitted = current
                .native
                .enqueue(index, permit.session(), permit, cancel)?;
            self.native_launches
                .fetch_add(submitted.native, Ordering::AcqRel);
            self.workspace_clears
                .fetch_add(submitted.clears, Ordering::AcqRel);
            self.device_copies
                .fetch_add(submitted.copies, Ordering::AcqRel);
            if self.fault.load(Ordering::Acquire) == 19
                && op.kind() == Vllm08Kind::FlashAttention
                && op.layer().is_some_and(|l| l.index() == 23)
            {
                self.fault_cut.store(true, Ordering::Release);
                return Err(Error::State(
                    "deliberate canonical cached unknown after actual last FA and output copy"
                        .into(),
                ));
            }
            return Ok(submitted.native + submitted.clears + submitted.copies);
        }
        if let Some(prefix) = &bound.canonical47_prefix
            && prefix.cut.is_whole()
            && !matches!(
                op.kind(),
                Vllm08Kind::InputRms
                    | Vllm08Kind::ResidualRms(_)
                    | Vllm08Kind::GdnConv
                    | Vllm08Kind::GdnRecurrent
            )
        {
            let before = prefix.native.enqueued_records();
            let mode = self.fault.load(Ordering::Acquire);
            let result = prefix.native.enqueue_whole(index, permit, cancel, mode);
            let after = prefix.native.enqueued_records();
            self.native_launches.fetch_add(
                after.checked_sub(before).ok_or_else(|| {
                    Error::State("wholeInitial47 native counter regressed".into())
                })?,
                Ordering::AcqRel,
            );
            if result.is_err() && prefix.native.whole_outputs_unpromoted()? {
                let counts = prefix.native.whole_counts()?;
                let reached = match mode {
                    19 => counts[5] == [1, 1, 1, 1] && prefix.native.head()?.enqueues() == 0,
                    20 => index == 254 && prefix.native.head()?.enqueues() == 0,
                    21 => index == 254 && prefix.native.head()?.enqueues() == 1,
                    _ => false,
                };
                if reached {
                    self.fault_cut.store(true, Ordering::Release);
                }
            }
            return result;
        }
        match op.kind() {
            Vllm08Kind::InputRms => {
                if index != 1 {
                    return Err(Error::State("full24 ordinary RMS index differs".into()));
                }
                self.rms.enqueue()?;
                Ok(10)
            }
            Vllm08Kind::ResidualRms(_) => {
                let rms = self
                    .fused
                    .get(&index)
                    .ok_or_else(|| Error::State("full24 bound dualRMS node absent".into()))?;
                rms.require_full_model_operator(op)?;
                rms.enqueue(cancel)?;
                Ok(10)
            }
            Vllm08Kind::GdnConv | Vllm08Kind::GdnRecurrent => {
                let layer = op
                    .layer()
                    .ok_or_else(|| Error::State("full24 GDN layer absent".into()))?
                    .index();
                let prepared = bound
                    .gdn
                    .get(&layer)
                    .ok_or_else(|| Error::State("full24 GDN indexed binding absent".into()))?;
                let late_cancel = layer == 0
                    && op.kind() == Vllm08Kind::GdnRecurrent
                    && (self.fault.load(Ordering::Acquire) == 2
                        || bound
                            .canonical47_prefix
                            .as_ref()
                            .is_some_and(|p| p.cut.is_whole())
                            && self.fault.load(Ordering::Acquire) == 18);
                if late_cancel {
                    self.internal_cancel.store(true, Ordering::Release);
                    self.fault_cut.store(true, Ordering::Release);
                }
                let count = prepared.enqueue(
                    index,
                    permit,
                    if late_cancel {
                        Some(&self.internal_cancel)
                    } else {
                        cancel
                    },
                )?;
                if op.kind() == Vllm08Kind::GdnConv {
                    self.convolution_dispatches
                        .fetch_add(count, Ordering::AcqRel);
                    if layer == 0
                        && (self.fault.load(Ordering::Acquire) == 1
                            || bound
                                .canonical47_prefix
                                .as_ref()
                                .is_some_and(|p| p.cut.is_whole())
                                && self.fault.load(Ordering::Acquire) == 17)
                    {
                        self.fault_cut.store(true, Ordering::Release);
                        return Err(Error::State(
                            "deliberate full24 unknown after actual source Conv and canonical copy"
                                .into(),
                        ));
                    }
                }
                Ok(count)
            }
            Vllm08Kind::Projection(Vllm08Projection::Head) => {
                let n = bound
                    .full_library()?
                    .head
                    .enqueue(bound.phase, index, cancel)?;
                self.native_launches.fetch_add(n, Ordering::AcqRel);
                Ok(n)
            }
            Vllm08Kind::Projection(Vllm08Projection::AttentionQkv)
                if bound
                    .canonical47_prefix
                    .as_ref()
                    .is_some_and(|p| p.cut.includes_qkv()) =>
            {
                let prefix = bound
                    .canonical47_prefix
                    .as_ref()
                    .ok_or_else(|| Error::State("canonical47 QKV cut absent".into()))?;
                // One original armed flight covers producer and reducer. No
                // initialization certificate or cursor advancement between them.
                let mode = self.fault.load(Ordering::Acquire);
                let before = prefix.native.enqueued_records();
                let result = prefix.native.enqueue_qkv(
                    index,
                    cancel,
                    if matches!(mode, 3 | 4) { mode } else { 0 },
                );
                let enqueued = prefix
                    .native
                    .enqueued_records()
                    .checked_sub(before)
                    .ok_or_else(|| {
                        Error::State("canonical47 native enqueue counter regressed".into())
                    })?;
                self.native_launches.fetch_add(enqueued, Ordering::AcqRel);
                if matches!(mode, 3 | 4)
                    && result.is_err()
                    && enqueued == 1
                    && prefix.native.qkv_enqueues() == 1
                    && prefix.native.qkv_output_unpromoted()?
                {
                    self.fault_cut.store(true, Ordering::Release);
                }
                result
            }
            Vllm08Kind::AttentionQk | Vllm08Kind::KvAppend
                if bound
                    .canonical47_prefix
                    .as_ref()
                    .is_some_and(|p| p.cut.includes_kv()) =>
            {
                let prefix = bound
                    .canonical47_prefix
                    .as_ref()
                    .ok_or_else(|| Error::State("canonical47 QK/KV cut absent".into()))?;
                let mode = self.fault.load(Ordering::Acquire);
                let before = prefix.native.attention_counts();
                let result = prefix.native.enqueue_attention(
                    index,
                    permit,
                    cancel,
                    if matches!(mode, 5..=8) { mode } else { 0 },
                );
                let after = prefix.native.attention_counts();
                let enqueued = (after.0 + after.1)
                    .checked_sub(before.0 + before.1)
                    .ok_or_else(|| {
                        Error::State("canonical47 QK/KV enqueue counter regressed".into())
                    })?;
                self.native_launches.fetch_add(enqueued, Ordering::AcqRel);
                let reached = match mode {
                    5 | 6 => after.0 == 1 && after.1 == 0,
                    7 | 8 => after.0 == 1 && after.1 == 1 && after.2 == 0,
                    _ => false,
                };
                if reached && result.is_err() {
                    self.fault_cut.store(true, Ordering::Release);
                }
                result
            }
            Vllm08Kind::FlashAttention
                if bound
                    .canonical47_prefix
                    .as_ref()
                    .is_some_and(|p| p.cut.includes_fa()) =>
            {
                let prefix = bound
                    .canonical47_prefix
                    .as_ref()
                    .ok_or_else(|| Error::State("canonical47 FA cut absent".into()))?;
                let mode = self.fault.load(Ordering::Acquire);
                let before = prefix.native.fa_enqueues();
                let result = prefix.native.enqueue_attention(
                    index,
                    permit,
                    cancel,
                    if matches!(mode, 9 | 10) { mode } else { 0 },
                );
                let after = prefix.native.fa_enqueues();
                let delta = after
                    .checked_sub(before)
                    .ok_or_else(|| Error::State("canonical47 FA counter regressed".into()))?;
                self.native_launches.fetch_add(delta, Ordering::AcqRel);
                if matches!(mode, 9 | 10)
                    && result.is_err()
                    && prefix.native.attention_counts() == (1, 1, 1)
                    && after == usize::from(mode == 9)
                    && prefix.native.fa_output_unpromoted()?
                {
                    self.fault_cut.store(true, Ordering::Release);
                }
                result
            }
            Vllm08Kind::Sigmoid
            | Vllm08Kind::Multiply
            | Vllm08Kind::Projection(
                Vllm08Projection::AttentionOut | Vllm08Projection::GateUp | Vllm08Projection::Down,
            )
            | Vllm08Kind::SiluAndMul
                if bound
                    .canonical47_prefix
                    .as_ref()
                    .is_some_and(|p| p.cut.includes_layer3())
                    && matches!(index, 36 | 37 | 38 | 40 | 41 | 42) =>
            {
                let prefix = bound
                    .canonical47_prefix
                    .as_ref()
                    .ok_or_else(|| Error::State("canonical47 full layer3 binding absent".into()))?;
                let mode = self.fault.load(Ordering::Acquire);
                let before = prefix.native.enqueued_records();
                let result = prefix.native.enqueue_layer3(
                    index,
                    cancel,
                    if matches!(mode, 11..=16) { mode } else { 0 },
                );
                let after = prefix.native.enqueued_records();
                self.native_launches.fetch_add(
                    after
                        .checked_sub(before)
                        .ok_or_else(|| Error::State("canonical47 tail counter regressed".into()))?,
                    Ordering::AcqRel,
                );
                let expected = match mode {
                    11 | 12 => Some([1, 0, 0, 0, 0, 0]),
                    13 => Some([1, 1, 1, 0, 0, 0]),
                    14 => Some([1, 1, 1, 1, 0, 0]),
                    15 | 16 => Some([1, 1, 1, 1, 1, 1]),
                    _ => None,
                };
                if let Some(expected) = expected
                    && result.is_err()
                    && prefix.native.tail_native_enqueues() == expected
                    && prefix.native.attention_counts() == (1, 1, 1)
                    && prefix.native.fa_enqueues() == 1
                    && prefix.native.tail_outputs_unpromoted()?
                {
                    self.fault_cut.store(true, Ordering::Release);
                }
                result
            }
            Vllm08Kind::Projection(
                Vllm08Projection::AttentionQkv | Vllm08Projection::AttentionOut,
            )
            | Vllm08Kind::AttentionQk
            | Vllm08Kind::KvAppend
            | Vllm08Kind::FlashAttention
            | Vllm08Kind::Sigmoid
            | Vllm08Kind::Multiply => {
                let layer = op
                    .layer()
                    .ok_or_else(|| Error::State("full24 attention layer absent".into()))?
                    .index();
                let binding = bound
                    .attention
                    .get(&layer)
                    .ok_or_else(|| Error::State("full24 actual attention binding absent".into()))?;
                let submitted = bound.full_library()?.attention.enqueue(
                    bound.phase,
                    layer,
                    index,
                    binding,
                    permit,
                    cancel,
                )?;
                self.native_launches
                    .fetch_add(submitted.native_launches, Ordering::AcqRel);
                self.device_copies
                    .fetch_add(submitted.device_copies, Ordering::AcqRel);
                Ok(submitted.native_launches + submitted.device_copies + submitted.transports)
            }
            Vllm08Kind::Projection(_) | Vllm08Kind::SiluAndMul => {
                if let Some(prefix) = &bound.canonical47_prefix {
                    let count = prefix.native.enqueue(index, cancel)?;
                    self.native_launches.fetch_add(count, Ordering::AcqRel);
                    return Ok(count);
                }
                let submitted =
                    bound
                        .full_library()?
                        .projections
                        .enqueue(bound.phase, index, cancel)?;
                self.native_launches
                    .fetch_add(submitted.dispatches, Ordering::AcqRel);
                self.workspace_clears
                    .fetch_add(submitted.workspace_clears, Ordering::AcqRel);
                Ok(submitted.dispatches + submitted.workspace_clears)
            }
        }
    }
    pub(crate) fn arm_fault(&self, mode: usize) -> Result<()> {
        if self
            .bound
            .get()
            .and_then(|b| b.canonical47_prefix.as_ref())
            .is_some_and(|p| p.cut.is_whole())
        {
            return Err(Error::Unsupported(
                "wholeInitial47 requires its distinct fault armer".into(),
            ));
        }
        let qkv = self
            .bound
            .get()
            .and_then(|b| b.canonical47_prefix.as_ref())
            .is_some_and(|p| p.cut.includes_qkv());
        let kv = self
            .bound
            .get()
            .and_then(|b| b.canonical47_prefix.as_ref())
            .is_some_and(|p| p.cut.includes_kv());
        let fa = self
            .bound
            .get()
            .and_then(|b| b.canonical47_prefix.as_ref())
            .is_some_and(|p| p.cut.includes_fa());
        let layer3 = self
            .bound
            .get()
            .and_then(|b| b.canonical47_prefix.as_ref())
            .is_some_and(|p| p.cut.includes_layer3());
        if !(matches!(mode, 0..=2)
            || (qkv && matches!(mode, 3 | 4))
            || (kv && matches!(mode, 5..=8))
            || (fa && matches!(mode, 9 | 10))
            || (layer3 && matches!(mode, 11..=16)))
            || self.launched.load(Ordering::Acquire)
        {
            return Err(Error::State(
                "full24 fault before original launch only".into(),
            ));
        }
        self.fault.store(mode, Ordering::Release);
        Ok(())
    }
    pub(crate) fn fault_cut_observed(&self) -> bool {
        self.fault_cut.load(Ordering::Acquire)
    }
    pub(crate) fn convolution_dispatches(&self) -> usize {
        self.convolution_dispatches.load(Ordering::Acquire)
    }
    pub(crate) fn native_launches(&self) -> usize {
        self.native_launches.load(Ordering::Acquire)
    }
    pub(crate) fn workspace_clears(&self) -> usize {
        self.workspace_clears.load(Ordering::Acquire)
    }
    pub(crate) fn device_copies(&self) -> usize {
        self.device_copies.load(Ordering::Acquire)
    }
    pub(crate) fn enqueue_output(&self, cancel: Option<&AtomicBool>) -> Result<usize> {
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("original model output before binding".into()))?;
        if let Some(current) = &bound.short_hf {
            self.values.require_finished_storage()?;
            current.certification.require_flushed()?;
        } else {
            self.values.finish_storage()?;
        }
        if bound.canonical47_prefix.is_some() {
            cancelled(cancel)?;
            self.finish_canonical47_prefix_nodes()?;
            if self.greedy.get().is_some() || self.output_submitted.swap(true, Ordering::AcqRel) {
                return Err(Error::State(
                    "canonical47 finite prefix has no Head/output consumer; finish once".into(),
                ));
            }
            return Ok(0);
        }
        if !self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != self.record.plan().nodes().len()
            || self.output_submitted.load(Ordering::Acquire)
        {
            return Err(Error::State("full24 output requires actual successful whole traversal/Head enqueue exactly once".into()));
        }
        let n = self
            .greedy
            .get()
            .ok_or_else(|| Error::State("full24 same-map output consumer absent".into()))?
            .enqueue(cancel)?;
        self.output_submitted.store(true, Ordering::Release);
        Ok(n)
    }
    pub(crate) fn certify(&self, value: &DeviceBuffer, status: &Handle) -> Result<usize> {
        if let Some(current) = self.bound.get().and_then(|b| b.short_hf.as_ref()) {
            let node = self
                .next_node
                .load(Ordering::Acquire)
                .checked_sub(1)
                .ok_or_else(|| Error::State("finite certificate before actual producer".into()))?;
            current.certification.certify(node, value, status)
        } else {
            certify_original(&self.certificates, value, status, self.status())?;
            Ok(1)
        }
    }
    /// Before private proposal certification, sampler and any status readback.
    pub(crate) fn flush_certificates(&self, cancel: Option<&AtomicBool>) -> Result<usize> {
        if let Some(current) = self.bound.get().and_then(|b| b.short_hf.as_ref()) {
            if !self.launched.load(Ordering::Acquire)
                || self.next_node.load(Ordering::Acquire) != self.record.plan().nodes().len()
            {
                return Err(Error::State(
                    "finite cohorts require whole actual traversal".into(),
                ));
            }
            self.values.finish_storage()?;
            current.certification.flush(cancel)
        } else {
            Ok(0)
        }
    }
    pub(crate) fn finite_certification_metrics(&self) -> [usize; 5] {
        self.bound
            .get()
            .and_then(|b| b.short_hf.as_ref())
            .map(|c| c.certification.metrics())
            .unwrap_or([0; 5])
    }
    pub(crate) fn cohort_readback_owner(&self) -> Arc<Vec<WindowWorkspace>> {
        self._readback.clone()
    }
    pub(crate) fn require_readback_capacity(&self, bytes: usize) -> Result<()> {
        if bytes
            > self
                ._readback
                .iter()
                .map(WindowWorkspace::bytes)
                .sum::<usize>()
        {
            return Err(Error::Limit(
                "full24 actual D2H cohort exceeds pre-admission".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn readbacks(&self) -> Result<Vec<ResidentReadback>> {
        self.values.require_all_value_readbacks()?;
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("full24 readback before bound preparation".into()))?;
        let mut reads = vec![self.status().readback(0, 4)?];
        for prepared in bound.gdn.values() {
            reads.push(prepared.status_readback()?);
        }
        for prepared in bound.attention.values() {
            reads.push(prepared.kv_status_readback()?);
        }
        reads.extend(self.rms.readbacks()?);
        for index in 0..self.execution_value_count()? {
            let lease = self.values.lease(ValueId(index))?;
            reads.push(lease.readback(0, lease.bytes())?);
        }
        for fused in self.fused.values() {
            reads.extend(fused.readbacks()?);
        }
        if bound.canonical47_prefix.is_some() {
            return Ok(reads);
        }
        reads.extend(
            self.greedy
                .get()
                .ok_or_else(|| Error::State("full24 greedy readback before prepare".into()))?
                .readbacks()?,
        );
        Ok(reads)
    }
    pub(crate) fn validate_readbacks(&self, raw: &[cubecl::bytes::Bytes]) -> Result<()> {
        if self
            .bound
            .get()
            .is_some_and(|b| b.canonical47_prefix.is_some())
        {
            return self.validate_canonical47_prefix_readbacks(raw);
        }
        if self.bound.get().is_some_and(|b| b.canonical_call.is_some()) {
            return self.validate_canonical47_call_readbacks(raw);
        }
        const STATUS: usize = 1 + 18 + 6;
        if raw.len() != STATUS + 3 + 327 + 48 * 4 + 2 || raw[0][..] != [0; 4] {
            return Err(Error::State(
                "full24 original status/readback coverage".into(),
            ));
        }
        for status in &raw[1..19] {
            if status[..] != [0; 4] {
                return Err(Error::State(
                    "full24 canonical state transport status".into(),
                ));
            }
        }
        for status in &raw[19..STATUS] {
            crate::vllm_cuda_qualification::model_kv_transport::ModelKvTransport::validate_status(
                &status[..],
            )?;
        }
        self.rms.validate_readback(&raw[STATUS..STATUS + 3])?;
        for (index, fused) in self.fused.values().enumerate() {
            let first = STATUS + 3 + 327 + index * 4;
            fused.validate_readback(&raw[first..first + 4])?;
        }
        self.greedy
            .get()
            .ok_or_else(|| Error::State("full24 greedy validation before prepare".into()))?
            .validate_readback(&raw[raw.len() - 2..])?;
        Ok(())
    }
}
impl ResidentProgram {
    pub(crate) async fn prepare_full_model_values(
        service: Arc<ResidentService>,
        source: Arc<OriginalWindowModelSource>,
        record: Arc<RecordedVllm08Model>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self> {
        let (body, values, preflight) = FullModelProgram::prepare_values(
            service.clone(),
            source.clone(),
            record.clone(),
            cancel,
        )
        .await?;
        Ok(Self {
            _source_owner: ResidentProgramSource::Windows(source),
            plan: record.plan().clone(),
            preflight,
            values,
            status: body.status().handle(),
            owned_ranges: OwnedRanges::new(service.client().service_id()),
            body: ResidentProgramBody::NativeFull(body),
        })
    }
    pub(crate) fn finalize_full_model_preflight(
        &mut self,
        state_ids: &BTreeSet<zerogravity_core::device_events::BufferLeaseId>,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let owned = self
            .full_model_body()?
            .accesses()?
            .iter()
            .filter(|(lease, _)| !state_ids.contains(&lease.id()))
            .cloned()
            .collect::<Vec<_>>();
        let values = self
            .values
            .iter()
            .map(|v| v.original_lease().map(|l| l.id()))
            .collect::<Result<BTreeSet<_>>>()?;
        let status_id = self.full_model_body()?.status().id();
        let mut source = 0usize;
        let mut value = 0usize;
        let mut scratch = if let Some(current) = self
            .full_model_body()?
            .bound
            .get()
            .and_then(|bound| bound.short_hf.as_ref())
        {
            current.native.native_provider_bytes()?
        } else {
            64 * 1024 * 1024
        };
        let mut status = 0usize;
        for (lease, _) in &owned {
            if lease.id() == status_id {
                status += lease.bytes();
            } else if values.contains(&lease.id()) {
                value += lease.bytes();
            } else if lease.role() == BufferRole::ImmutableWeights {
                source += lease.bytes();
            } else {
                scratch += lease.bytes();
            }
            self.owned_ranges
                .insert(BufferRange::new(&lease.handle())?)?;
        }
        self.preflight.source_bytes = source;
        self.preflight.value_bytes = value;
        self.preflight.workspace_bytes = scratch;
        self.preflight.status_bytes = status;
        Ok(owned)
    }
    pub(crate) fn full_model_body(&self) -> Result<&Arc<FullModelProgram>> {
        match &self.body {
            ResidentProgramBody::NativeFull(body) => Ok(body),
            _ => Err(Error::Unsupported("not original full24 body".into())),
        }
    }
}

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod natural;

mod publication;

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod rms_scratch;

mod value_storage;
pub(crate) use value_storage::ShortHfReadbackAdmission;

mod short_hf_inventory;

impl FullModelProgram {
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
        let individual = self.certificates.prepare_code(queue)?;
        let candidates = self.values.finite_code_candidates()?;
        for sizes in &candidates {
            super::native::certification::definition(sizes)?
                .0
                .prepare_code(queue)?;
        }
        Ok(individual + candidates.len())
    }
}
