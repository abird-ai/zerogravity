//! Native source/body inside the SAME resident value map and dispatch loop.
use super::source_bindings::SourceBindings;
use super::*;
use crate::cuda::kernel::PinnedKernel;
use crate::cuda::native::native_gdn::{NativeGdnContext, NativeGdnPrepared};
use crate::cuda::native::native_plan::{
    NativePlanContext, NativePlanPhase, NativePlanPrograms, PreparedImages,
};
use crate::cuda::rms::{BoundRmsSource, FusedRmsPrepared, RmsPrepared};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_inference::gdn_state::BoundGdnDomains;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_leases::{ResidentLease, ResidentReadback, ResidentService};
use cubecl::bytes::Bytes;
use cubecl_runtime::kernel::BufferIOAttr;
use std::collections::BTreeMap;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::SourceCall;
use zerogravity_core::accelerator::program::{
    ActivationPrecision, SourceGamma, SourceNumericalContract, SourceOperator, SourceOperatorKind,
    SourceRmsGeometry,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
pub(super) mod certification;
pub(super) use certification::CertificationRun;
pub(crate) use certification::qualify as qualify_finite_cohorts;

pub(crate) struct NativeProgram {
    // Certificate handles retire before the original value/status leases.
    certificates: CertificateSet,
    source: VerifiedTensorSource,
    bindings: Arc<SourceBindings>,
    invocation: SourceCall,
    completed_layers: usize,
    rms: Arc<RmsPrepared>,
    status: ResidentLease,
    gdn_nodes: Vec<[usize; 2]>,
    post_attention_fault_node: Option<usize>,
    down_fault_node: Option<usize>,
    fused: BTreeMap<usize, Arc<FusedRmsPrepared>>,
    gdn: OnceLock<Vec<Arc<NativeGdnPrepared>>>,
    projections: OnceLock<Arc<NativePlanPrograms>>,
    images: OnceLock<Arc<PreparedImages>>,
    launched: AtomicBool,
    native_dispatches: AtomicUsize,
    rms_dispatches: AtomicUsize,
    fault: AtomicUsize,
    cut: AtomicBool,
    conv_dispatches: AtomicUsize,
    internal_cancel: AtomicBool,
    _readback: Vec<WindowWorkspace>,
    _metadata: WindowWorkspace,
}
struct Certificate {
    launch: crate::cuda::kernel::PreparedPinnedLaunch,
}
/// Exact immutable physical bindings, shared by every native body adapter.
pub(super) struct CertificateSet {
    entries: Vec<Certificate>,
    by_lease: BTreeMap<zerogravity_core::device_events::BufferLeaseId, usize>,
    cohorts: OnceLock<Arc<certification::CohortSet>>,
}
pub(super) fn cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|v| v.load(Ordering::Acquire)) {
        return Err(Error::State("native plan cancelled".into()));
    }
    Ok(())
}
/// Exact source metadata and source layer indices, also used by Session roots.
pub(super) fn state_requirements(bindings: &SourceBindings) -> Result<Vec<StateRequirement>> {
    Ok(bindings.state_requirements().to_vec())
}
struct Recipe {
    gdns: Vec<[usize; 2]>,
    fused: Vec<usize>,
    first_down: Option<usize>,
    _metadata: WindowWorkspace,
}
fn fragment_rms() -> SourceRmsGeometry {
    SourceRmsGeometry {
        width: 1024,
        epsilon: 1e-6,
        gamma: SourceGamma::OnePlus,
    }
}
fn source_descriptor<'a>(node: &'a PlanNode, call: &SourceCall) -> Result<&'a SourceOperator> {
    match &node.operation {
        PlanOperation::SourceOperator {
            descriptor,
            state: None,
        } if descriptor.call() == call
            && descriptor.numerical_contract()
                == SourceNumericalContract::ReferenceCudaSm86Bf16V1 =>
        {
            Ok(descriptor)
        }
        _ => Err(Error::State(
            "native actual source mathematical call/contract differs".into(),
        )),
    }
}
fn require_recipe(bindings: &SourceBindings) -> Result<Recipe> {
    crate::cuda::native::native_gdn::phase(bindings.call())?;
    let metadata = bindings.source().reserve_workspace(64 * 1024)?;
    let plan = bindings.plan();
    let completed = bindings.selection().completed_layers();
    if plan.is_operator_slice()
        || plan.frontier_delta() != 0
        || plan.logits().is_some()
        || !matches!(completed, 0 | 1 | 3)
    {
        return Err(Error::Unsupported(
            "native unpublished finite fragment extent".into(),
        ));
    }
    let mut recipe = Recipe {
        gdns: Vec::with_capacity(completed.max(1)),
        fused: Vec::with_capacity(2 * completed.max(1) - 1),
        first_down: bindings.selection().first_mlp_down_node(),
        _metadata: metadata,
    };
    for (index, node) in plan.nodes().iter().enumerate() {
        let descriptor = source_descriptor(node, bindings.call())?;
        match descriptor.kind() {
            SourceOperatorKind::EmbeddingRows
                if index == 0 && node.inputs.is_empty() && node.outputs.len() == 1 => {}
            SourceOperatorKind::Rms(geometry)
                if index == 1
                    && geometry == fragment_rms()
                    && node.inputs.len() == 1
                    && node.outputs.len() == 1 => {}
            SourceOperatorKind::ResidualRms(geometry)
                if geometry == fragment_rms()
                    && node.inputs.len() == 2
                    && node.outputs.len() == 2
                    && node.state.is_none() =>
            {
                recipe.fused.push(index)
            }
            SourceOperatorKind::CausalConvSplit { .. } => {
                let pair = [index, index + 1];
                crate::cuda::native::native_gdn::operands(bindings, pair)?;
                recipe.gdns.push(pair);
            }
            SourceOperatorKind::GatedDelta { .. }
                if recipe.gdns.last().is_some_and(|p| p[1] == index) => {}
            SourceOperatorKind::Projection { selected_row: None }
                if node.inputs.len() == 1 && node.outputs.len() == 1 && node.state.is_none() => {}
            SourceOperatorKind::SiluAndMul { width: 3584 }
                if node.inputs.len() == 1 && node.outputs.len() == 1 && node.state.is_none() => {}
            SourceOperatorKind::StateCarry { slot }
                if node.inputs.is_empty()
                    && node.outputs.is_empty()
                    && node.state.is_some_and(|effect| {
                        effect.read.slot == slot
                            && effect.write.slot == slot
                            && effect.read.ordinal == 0
                            && effect.write.ordinal == 1
                    }) => {}
            _ => {
                return Err(Error::Unsupported(
                    "operation outside actual native fragment math".into(),
                ));
            }
        }
    }
    if recipe.gdns.len() != completed.max(1)
        || recipe.fused.len() != if completed == 0 { 0 } else { 2 * completed - 1 }
    {
        return Err(Error::State(
            "native actual GDN/RMS preparation site coverage".into(),
        ));
    }
    Ok(recipe)
}
pub(crate) fn capacity_bound(bindings: &SourceBindings) -> Result<usize> {
    require_recipe(bindings)?;
    let state_bytes = checked_sum(bindings.state_requirements().iter().map(|r| r.bytes))?;
    let rows = bindings.call().rows().len();
    let layers = bindings.selection().completed_layers().max(1);
    let fused = if bindings.selection().completed_layers() == 0 {
        0
    } else {
        layers * 2 - 1
    };
    let full = bindings.selection().completed_layers() != 0;
    // Source-derived initial scope, separate from old component envelopes.
    let per_gdn = 49_504
        + 5_570_560
        + 16_421
        + 2_097_152
        + 524_288
        + rows * (35_072 + if rows > 1 { 64 } else { 0 });
    let weights = if full { 43_057_152 } else { 16_842_752 };
    let extra_values = if full {
        rows * (2_048 + 14_336 + 7_168 + 2_048)
    } else {
        0
    };
    let body_bound = 64 * 1024 * 1024
        + layers * weights
        + 8_519_680
        + (10_240 + rows * 20_492)
        + layers * (rows * 36_928 + per_gdn + extra_values)
        + fused * (10_240 + rows * 20_492)
        + 4 * (layers + 1);
    let bound = state_bytes
        .checked_mul(2)
        .and_then(|n| n.checked_add(body_bound))
        .ok_or_else(|| Error::Limit("native state/source capacity bound overflow".into()))?;
    bound
        .checked_add(32 * 1024 * 1024 - 1)
        .map(|n| n / (32 * 1024 * 1024) * (32 * 1024 * 1024))
        .ok_or_else(|| Error::Limit("native fragment capacity bound overflow".into()))
}
impl NativeProgram {
    pub(super) fn require_plan(&self, plan: &ForwardPlan) -> Result<()> {
        if !std::ptr::eq(plan, self.bindings.plan()) || plan.source_call() != Some(&self.invocation)
        {
            return Err(Error::State(
                "native actual shared graph/call owner differs".into(),
            ));
        }
        Ok(())
    }
    /// Cold pre-binding stage. No image/module or StateRoot is manufactured.
    pub(crate) async fn prepare_values(
        service: Arc<ResidentService>,
        bindings: Arc<SourceBindings>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        cancelled(cancel)?;
        let recipe = require_recipe(&bindings)?;
        let plan = bindings.plan();
        let invocation = bindings.call().clone();
        let source = bindings.source().clone();
        let metadata = source.reserve_workspace(4 * 1024 * 1024)?;
        let states = state_requirements(&bindings)?;
        let layers = bindings.selection().completed_layers().max(1);
        let packed_weights = if bindings.selection().completed_layers() == 0 {
            16_842_752
        } else {
            43_057_152
        };
        let public_bytes = checked_sum(
            plan.values()
                .iter()
                .skip(2)
                .map(|shape| {
                    shape.capacity_elements().and_then(|n| {
                        n.checked_mul(2)
                            .ok_or_else(|| Error::Limit("BF16 readback overflow".into()))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        let max_readback = 2 * 32_784_384
            + layers * (packed_weights + 49_504)
            + 4 * (1 + layers)
            + 2048
            + invocation.rows().len() * 4096
            + public_bytes
            + recipe.fused.len() * (2048 + invocation.rows().len() * 8192);
        let readback_chunks = max_readback.div_ceil(32 * 1024 * 1024);
        if !(3..=8).contains(&readback_chunks) {
            return Err(Error::Limit(
                "closed initial fragment host readback span".into(),
            ));
        }
        let readback = (0..readback_chunks)
            .map(|_| source.reserve_workspace(32 * 1024 * 1024))
            .collect::<Result<Vec<_>>>()?;
        let rms = RmsPrepared::prepare_bound(service.clone(), bindings.clone(), 0, cancel).await?;
        cancelled(cancel)?;
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(bindings.clone())?;
        init.retain_completion_owner(rms.completion_owner())?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let mut values: Vec<Option<DeviceBuffer>> = vec![None; plan.values().len()];
        let fused_outputs = recipe
            .fused
            .iter()
            .flat_map(|index| plan.nodes()[*index].outputs.iter().map(|id| id.0))
            .collect::<std::collections::BTreeSet<_>>();
        for (index, (shape, precision)) in plan.values().iter().zip(plan.precisions()).enumerate() {
            if *precision != ActivationPrecision::Bf16WidenedF32 {
                return Err(Error::Unsupported(
                    "native exact BF16 physical values".into(),
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
            values[index] = Some(DeviceBuffer::from_lease(
                lease.clone(),
                layout,
                lease.role(),
                DeviceAccess::Write,
            )?);
        }
        drop(init.complete_sync().await?);
        let mut fused = BTreeMap::new();
        // Guard-only prepared consumers; no nested initialization guard/fair ticket.
        for &index in &recipe.fused {
            let node = &plan.nodes()[index];
            let PlanOperation::SourceOperator {
                descriptor,
                state: None,
            } = &node.operation
            else {
                return Err(Error::State("bound RMS mathematical recipe changed".into()));
            };
            let [normalized, residual] = node.outputs.as_slice() else {
                return Err(Error::State("bound RMS actual output arity".into()));
            };
            let [parameter] = descriptor.parameters() else {
                return Err(Error::State("bound RMS actual gamma arity".into()));
            };
            if descriptor.kind() != SourceOperatorKind::ResidualRms(fragment_rms())
                || descriptor.call() != &invocation
                || node.state.is_some()
                || node.inputs.len() != 2
            {
                return Err(Error::State(
                    "bound RMS actual source/call/SSA differs".into(),
                ));
            }
            for (ordinal, id) in [*normalized, *residual].into_iter().enumerate() {
                let tag = plan.source_value_provenance(id)?.ok_or_else(|| {
                    Error::State("bound RMS actual output provenance absent".into())
                })?;
                if tag.node_index() != index
                    || tag.output_index() != ordinal
                    || tag.call() != &invocation
                    || tag.inputs() != node.inputs
                {
                    return Err(Error::State(
                        "bound RMS actual dual producer differs".into(),
                    ));
                }
            }
            let source_binding = BoundRmsSource::bind(source.clone(), descriptor.as_ref().clone())?;
            source
                .parameter(&parameter.name)?
                .validate_source_parameter(parameter)?;
            let input =
                |id: zerogravity_core::accelerator::program::ValueId| -> Result<ResidentLease> {
                    values
                        .get(id.0 as usize)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| Error::State("actual fused producer binding absent".into()))?
                        .original_lease()
                        .cloned()
                };
            let prepared = FusedRmsPrepared::prepare_bound(
                service.clone(),
                source_binding,
                invocation.rows().len(),
                input(node.inputs[0])?,
                input(node.inputs[1])?,
                0,
                cancel,
            )
            .await?;
            for (id, lease) in node
                .outputs
                .iter()
                .zip([prepared.normalized(), prepared.residual()])
            {
                let layout = ResidentBufferLayout::new(
                    ResidentPhysicalType::Bf16,
                    invocation.rows().len() * 1024,
                )?;
                if values[id.0 as usize].is_some() {
                    return Err(Error::State("fused typed output already bound".into()));
                }
                values[id.0 as usize] = Some(DeviceBuffer::from_lease(
                    lease.clone(),
                    layout,
                    lease.role(),
                    DeviceAccess::Write,
                )?);
            }
            fused.insert(index, prepared);
        }
        let values = values
            .into_iter()
            .map(|value| value.ok_or_else(|| Error::State("unbound original plan value".into())))
            .collect::<Result<Vec<_>>>()?;
        let certificates = prepare_certificates(&service, &values, &status)?;
        let state_bytes = checked_sum(states.iter().map(|r| r.bytes))?;
        // Final exact owned-byte accounting is sealed after bound native prep.
        let preflight = ResidentPreflight {
            source: source.source_identity(),
            source_bytes: 0,
            value_bytes: checked_sum(
                values
                    .iter()
                    .map(|v| v.physical.bytes())
                    .collect::<Result<Vec<_>>>()?,
            )?,
            workspace_bytes: 0,
            status_bytes: 4,
            parameter_bytes: 0,
            prior_state_bytes: state_bytes,
            candidate_state_bytes: state_bytes,
            states,
            scratch: [0; SCRATCH_SLOTS],
        };
        let body = Arc::new(Self {
            source,
            bindings: bindings.clone(),
            invocation,
            completed_layers: bindings.selection().completed_layers(),
            rms,
            status,
            certificates,
            post_attention_fault_node: recipe.fused.first().copied(),
            down_fault_node: recipe.first_down,
            gdn_nodes: recipe.gdns,
            fused,
            gdn: OnceLock::new(),
            projections: OnceLock::new(),
            images: OnceLock::new(),
            launched: AtomicBool::new(false),
            native_dispatches: AtomicUsize::new(0),
            rms_dispatches: AtomicUsize::new(0),
            fault: AtomicUsize::new(0),
            cut: AtomicBool::new(false),
            conv_dispatches: AtomicUsize::new(0),
            internal_cancel: AtomicBool::new(false),
            _readback: readback,
            _metadata: metadata,
        });
        Ok((body, values, preflight))
    }
    pub(crate) async fn prepare_gdn(
        self: &Arc<Self>,
        service: Arc<ResidentService>,
        plan: &ForwardPlan,
        values: &[DeviceBuffer],
        domains: Vec<Arc<BoundGdnDomains>>,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        self.require_plan(plan)?;
        if self.gdn.get().is_some() || self.images.get().is_some() {
            return Err(Error::State(
                "native source/root body prepared twice".into(),
            ));
        }
        if domains.len() != self.gdn_nodes.len() {
            return Err(Error::State(
                "one original domain binding per GDN layer".into(),
            ));
        }
        let mut gdn = Vec::with_capacity(domains.len());
        for (domains, nodes) in domains.into_iter().zip(&self.gdn_nodes) {
            gdn.push(
                NativeGdnPrepared::prepare(
                    NativeGdnContext {
                        service: service.clone(),
                        bindings: self.bindings.clone(),
                        domains,
                    },
                    values,
                    *nodes,
                    cancel,
                )
                .await?,
            );
        }
        self.gdn
            .set(gdn)
            .map_err(|_| Error::State("native GDN bind raced".into()))?;
        Ok(())
    }
    pub(crate) fn prepare_native(
        self: &Arc<Self>,
        service: Arc<ResidentService>,
        plan: &ForwardPlan,
        values: &[DeviceBuffer],
        images: Arc<PreparedImages>,
        init: &mut crate::resident_leases::ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        self.require_plan(plan)?;
        if self.gdn.get().is_none() || self.projections.get().is_some() {
            return Err(Error::State("native final preparation order".into()));
        }
        init.retain_completion_owner(self.clone())?;
        self.images
            .set(images.clone())
            .map_err(|_| Error::State("native images bound twice".into()))?;
        let programs = NativePlanPrograms::prepare(
            NativePlanContext {
                images,
                bindings: self.bindings.clone(),
                service,
            },
            plan,
            values,
            init,
            cancel,
        )?;
        self.projections
            .set(programs)
            .map_err(|_| Error::State("native programs bound twice".into()))?;
        Ok(())
    }
    pub(crate) fn prepare_native_phases(
        phases: &[(&Arc<Self>, &ForwardPlan, &[DeviceBuffer])],
        service: Arc<ResidentService>,
        images: Arc<PreparedImages>,
        init: &mut crate::resident_leases::ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        if !matches!(phases.len(), 2 | 3) {
            return Err(Error::Unsupported("closed native request phases".into()));
        }
        let mut bindings = Vec::with_capacity(phases.len());
        for &(body, plan, values) in phases {
            body.require_plan(plan)?;
            if body.gdn.get().is_none()
                || body.projections.get().is_some()
                || body.images.get().is_some()
            {
                return Err(Error::State("native phase final preparation order".into()));
            }
            init.retain_completion_owner(body.clone())?;
            body.images
                .set(images.clone())
                .map_err(|_| Error::State("native phase images bound twice".into()))?;
            bindings.push(NativePlanPhase {
                context: NativePlanContext {
                    images: images.clone(),
                    bindings: body.bindings.clone(),
                    service: service.clone(),
                },
                plan,
                values,
            });
        }
        let programs = NativePlanPrograms::prepare_phases(bindings, init, cancel)?;
        for ((body, _, _), programs) in phases.iter().zip(programs) {
            body.projections
                .set(programs)
                .map_err(|_| Error::State("native phase programs bound twice".into()))?;
        }
        Ok(())
    }
    pub(crate) fn require_attempt(&self, permit: &PrefixPhaseEnqueue<'_>) -> Result<()> {
        let layers = self.gdn_nodes.len();
        if !matches!(layers, 1 | 3) || self.fused.len() != 2 * layers - 1 {
            return Err(Error::Unsupported(
                "complete source layer0/prefix attempt only".into(),
            ));
        }
        permit.require_invocation(self.rms.normalized().service(), &self.invocation, 0, layers)
    }
    pub(crate) fn completion_owner(self: &Arc<Self>) -> CompletionOwner {
        self.clone()
    }
    pub(crate) fn begin_launch(
        &self,
        cancel: Option<&AtomicBool>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<()> {
        cancelled(cancel)?;
        if let Some(permit) = permit {
            self.require_attempt(permit)?;
        }
        let gdn = self
            .gdn
            .get()
            .ok_or_else(|| Error::State("native GDN cold preparation absent".into()))?;
        if gdn
            .iter()
            .any(|g| g.requires_attempt_permit() != permit.is_some())
        {
            return Err(Error::State(
                "matching original attempt permit required before ANY enqueue".into(),
            ));
        }
        if self.gdn.get().is_none()
            || self.projections.get().is_none()
            || self.images.get().is_none()
            || self.launched.swap(true, Ordering::AcqRel)
        {
            return Err(Error::State(
                "native fragment requires complete cold preparation and executes once".into(),
            ));
        }
        self.native_dispatches.store(0, Ordering::Release);
        self.rms_dispatches.store(0, Ordering::Release);
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, p: RuntimeParameters) -> Result<()> {
        if p.position != 0
            || p.valid_tokens != self.invocation.rows().len()
            || self.invocation.rows().ids().first().copied() != Some(p.token)
        {
            return Err(Error::State(
                "native runtime differs from recorded genuine initial teacher span".into(),
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
        cancelled(cancel)?;
        if let Some(permit) = permit {
            self.require_attempt(permit)?;
        }
        match source_descriptor(node, &self.invocation)?.kind() {
            SourceOperatorKind::EmbeddingRows => Ok(0), // Actual uploaded source rows, separate from RMS output.
            SourceOperatorKind::Rms(_) => {
                self.rms.enqueue()?;
                self.rms_dispatches.fetch_add(10, Ordering::AcqRel);
                Ok(10)
            }
            SourceOperatorKind::ResidualRms(_) => {
                self.fused
                    .get(&index)
                    .ok_or_else(|| Error::State("actual bound RMS record absent".into()))?
                    .enqueue(cancel)?;
                self.rms_dispatches.fetch_add(10, Ordering::AcqRel);
                if self.fault.load(Ordering::Acquire) == 3
                    && self.post_attention_fault_node == Some(index)
                {
                    self.cut.store(true, Ordering::Release);
                    return Err(Error::State(
                        "deliberate unknown after actual post-attention dual RMS".into(),
                    ));
                }
                Ok(10)
            }
            SourceOperatorKind::Projection { .. } | SourceOperatorKind::SiluAndMul { .. } => {
                let cancel_down =
                    self.fault.load(Ordering::Acquire) == 4 && self.down_fault_node == Some(index);
                if cancel_down {
                    self.internal_cancel.store(true, Ordering::Release);
                    self.cut.store(true, Ordering::Release);
                }
                let result = self
                    .projections
                    .get()
                    .ok_or_else(|| {
                        Error::State("native projection cold preparation absent".into())
                    })?
                    .enqueue(
                        index,
                        if cancel_down {
                            Some(&self.internal_cancel)
                        } else {
                            cancel
                        },
                    )?;
                self.native_dispatches
                    .fetch_add(result.native_launches, Ordering::AcqRel);
                Ok(result.dispatches)
            }
            SourceOperatorKind::CausalConvSplit { .. } => {
                let count = self
                    .gdn
                    .get()
                    .and_then(|all| {
                        self.gdn_nodes
                            .iter()
                            .position(|nodes| nodes[0] == index)
                            .map(|n| &all[n])
                    })
                    .ok_or_else(|| Error::State("native exact Conv root bind absent".into()))?
                    .enqueue(index, cancel, permit)?;
                self.conv_dispatches.fetch_add(count, Ordering::AcqRel);
                if self.fault.load(Ordering::Acquire) == 1 {
                    self.cut.store(true, Ordering::Release);
                    return Err(Error::State(
                        "deliberate unknown after actual root Conv".into(),
                    ));
                }
                Ok(count)
            }
            SourceOperatorKind::GatedDelta { .. } => {
                if self.fault.load(Ordering::Acquire) == 2 {
                    self.internal_cancel.store(true, Ordering::Release);
                    self.cut.store(true, Ordering::Release);
                }
                self.gdn
                    .get()
                    .and_then(|all| {
                        self.gdn_nodes
                            .iter()
                            .position(|nodes| nodes[1] == index)
                            .map(|n| &all[n])
                    })
                    .ok_or_else(|| Error::State("native exact recurrent root bind absent".into()))?
                    .enqueue(
                        index,
                        if self.fault.load(Ordering::Acquire) == 2 {
                            Some(&self.internal_cancel)
                        } else {
                            cancel
                        },
                        permit,
                    )
            }
            _ => Err(Error::Unsupported(
                "operation outside native source fragment body".into(),
            )),
        }
    }
    pub(crate) fn certify(&self, value: &DeviceBuffer, status: &Handle) -> Result<()> {
        certify_original(&self.certificates, value, status, &self.status)
    }
    pub(crate) fn arm_fault(
        &self,
        fault: Option<crate::resident_inference::native_fragment::GdnFragmentFault>,
    ) -> Result<()> {
        if self.launched.load(Ordering::Acquire) {
            return Err(Error::State(
                "cannot change native fault after enqueue".into(),
            ));
        }
        use crate::resident_inference::native_fragment::GdnFragmentFault::*;
        let mode = match fault {
            None => 0,
            Some(UnknownAfterConvolution) => 1,
            Some(CancelBeforeRecurrent) => 2,
            Some(UnknownAfterPostAttentionRms) => 3,
            Some(CancelBeforeMlpDown) => 4,
            Some(AdmissionPressure) => {
                return Err(Error::State(
                    "pressure must precede native allocation".into(),
                ));
            }
        };
        self.fault.store(mode, Ordering::Release);
        Ok(())
    }
    pub(crate) fn fault_cut_observed(&self) -> bool {
        self.cut.load(Ordering::Acquire)
    }
    pub(crate) fn convolution_dispatches(&self) -> usize {
        self.conv_dispatches.load(Ordering::Acquire)
    }
    pub(crate) fn retained_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.accesses(&[])
    }
    pub(crate) fn rms_stages(&self) -> usize {
        self.rms_dispatches.load(Ordering::Acquire)
    }
    pub(crate) fn native_launches(&self) -> usize {
        self.native_dispatches.load(Ordering::Acquire)
    }
    pub(crate) fn accesses(
        &self,
        values: &[DeviceBuffer],
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut all = self.rms.accesses().to_vec();
        all.push((self.status.clone(), DeviceAccess::Write));
        for fused in self.fused.values() {
            all.extend(fused.accesses().iter().cloned());
        }
        for (index, value) in values.iter().enumerate() {
            all.push((
                value.original_lease()?.clone(),
                if index == 0 {
                    DeviceAccess::Read
                } else {
                    DeviceAccess::Write
                },
            ));
        }
        if let Some(gdn) = self.gdn.get() {
            for layer in gdn {
                all.extend(layer.accesses());
            }
        }
        if let Some(programs) = self.projections.get() {
            all.extend(programs.accesses());
        }
        let mut unique: Vec<(ResidentLease, DeviceAccess)> = Vec::new();
        for (lease, access) in all {
            if let Some((_, old)) = unique
                .iter_mut()
                .find(|(other, _)| other.id() == lease.id())
            {
                if access == DeviceAccess::Write {
                    *old = access;
                }
            } else {
                unique.push((lease, access));
            }
        }
        Ok(unique)
    }
    pub(crate) fn handles(&self, values: &[DeviceBuffer], status: &Handle) -> Vec<Handle> {
        let mut handles = self
            .accesses(values)
            .expect("sealed native original value map")
            .into_iter()
            .map(|(lease, _)| lease.handle())
            .collect::<Vec<_>>();
        if !handles.iter().any(|h| same_range(h, status)) {
            handles.push(status.clone());
        }
        handles
    }
    pub(crate) fn status(&self) -> &ResidentLease {
        &self.status
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn gdn_nodes(&self) -> &[[usize; 2]] {
        &self.gdn_nodes
    }
    pub(crate) fn fused_nodes(&self) -> Vec<usize> {
        self.fused.keys().copied().collect()
    }
    pub(crate) fn completed_layers(&self) -> usize {
        self.completed_layers
    }
    pub(crate) fn invocation_for_layer(&self, index: usize) -> Result<SourceCall> {
        if !self.gdn_nodes.iter().any(|nodes| matches!(
            &self.bindings.plan().nodes()[nodes[0]].operation,
            PlanOperation::SourceOperator { descriptor,.. }
                if matches!(descriptor.kind(),SourceOperatorKind::CausalConvSplit { domain,.. } if domain.0==index))) {
            return Err(Error::State("native actual GDN domain site absent".into()));
        }
        Ok(self.invocation.clone())
    }
    pub(crate) fn require_readback_capacity(&self, bytes: usize) -> Result<()> {
        let admitted = self
            ._readback
            .iter()
            .map(WindowWorkspace::bytes)
            .sum::<usize>();
        if bytes > admitted {
            return Err(Error::Limit(
                "actual physical D2H cohort exceeds original host admission".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn readbacks(&self, values: &[DeviceBuffer]) -> Result<Vec<ResidentReadback>> {
        let mut result = vec![self.status.readback(0, 4)?];
        for gdn in self
            .gdn
            .get()
            .ok_or_else(|| Error::State("native GDN absent".into()))?
        {
            result.push(gdn.status_readback()?);
        }
        result.extend(self.rms.readbacks()?);
        for value in values.iter().skip(2) {
            result.push(value.readback(value.original_lease()?, value.physical)?);
        }
        for fused in self.fused.values() {
            result.extend(fused.readbacks()?);
        }
        Ok(result)
    }
    pub(crate) fn validate_readbacks(&self, raw: &[Bytes], value_count: usize) -> Result<()> {
        let rms = 1 + self.gdn_nodes.len();
        if raw.len() < rms + 3 || raw[0].as_ref() != [0; 4] {
            return Err(Error::State("native actual program status".into()));
        }
        for status in &raw[1..rms] {
            crate::cuda::gdn::GdnStateAdapter::validate_status(status.as_ref())?;
        }
        self.rms.validate_readback(&raw[rms..rms + 3])?;
        let first_fused = rms + 3 + value_count - 2;
        for (i, fused) in self.fused.values().enumerate() {
            let start = first_fused + 4 * i;
            fused.validate_readback(
                raw.get(start..start + 4)
                    .ok_or_else(|| Error::State("actual fused readback cohort absent".into()))?,
            )?;
        }
        Ok(())
    }
    pub(crate) fn immutable_weights(&self) -> Result<Vec<(ResidentLease, &[u8])>> {
        let mut weights = self
            .projections
            .get()
            .ok_or_else(|| Error::State("native projections absent".into()))?
            .immutable_weights();
        for gdn in self
            .gdn
            .get()
            .ok_or_else(|| Error::State("native GDN absent".into()))?
        {
            weights.extend(gdn.immutable_weights());
        }
        Ok(weights)
    }
    /// Caller holds an initialized-W original flight after genuine compute Ready.
    pub(crate) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        let programs = self
            .projections
            .get()
            .ok_or_else(|| Error::State("native programs absent".into()))?;
        // SAFETY: enclosing owner proves real completion and armed W retention.
        unsafe { programs.retire_kernels_after_physical_completion() }
            .map_err(|e| Error::State(format!("native kernel retirement: {e}")))?;
        unsafe {
            crate::cuda::native::native_plan::retire_native_set_after_physical_completion(
                self.images
                    .get()
                    .ok_or_else(|| Error::State("native set absent".into()))?,
            )
        }
        .map_err(|e| Error::State(format!("native set retirement: {e}")))
    }
}
pub(super) fn finite_definition(n: usize) -> Result<PinnedKernel> {
    if n == 0 || n > u32::MAX as usize / 2 {
        return Err(Error::Limit("finite definition extent".into()));
    }
    let source:Arc<str>=format!("// cubecl.cuda.dynamic_shared_bytes=0\nextern \"C\" __global__ void zg_native_bf16_finite(const unsigned short* x,unsigned int* status){{unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<{n}u && (x[i]&0x7f80u)==0x7f80u)atomicOr(status,1u);}}\n").into();
    PinnedKernel::new(
        "zg_native_bf16_finite",
        source,
        128,
        Arc::from([
            (n * 2, BufferIOAttr::ReadOnly),
            (4, BufferIOAttr::ReadWrite),
        ]),
    )
}
impl CertificateSet {
    pub(super) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
        for entry in &self.entries {
            entry.launch.prepare_code(queue)?;
        }
        Ok(self.entries.len())
    }
}
pub(super) fn prepare_certificates(
    service: &Arc<ResidentService>,
    values: &[DeviceBuffer],
    status: &ResidentLease,
) -> Result<CertificateSet> {
    let mut certificates = Vec::with_capacity(values.len());
    let mut by_lease = BTreeMap::new();
    for value in values {
        let n = value.elements;
        let kernel = finite_definition(n)?;
        let handles = vec![value.original_lease()?.handle(), status.handle()];
        let grid = [(n as u32).div_ceil(128), 1, 1];
        let launch = kernel.prepare(service.client(), grid, handles)?;
        by_lease
            .entry(value.original_lease()?.id())
            .or_insert(certificates.len());
        certificates.push(Certificate { launch });
    }
    Ok(CertificateSet {
        entries: certificates,
        by_lease,
        cohorts: OnceLock::new(),
    })
}
/// B4 cold-path certificate binder: same per-value immediate launches as
/// `prepare_certificates`, consuming the immutable cold definitions instead
/// of rebuilding 327 `finite_definition` keys per call. A cold/current extent
/// mismatch is refused by the launch binding, never silently re-keyed;
/// `prepare_certificates` itself stays byte-identical for the N1 paths.
pub(super) fn prepare_certificates_cold(
    service: &Arc<ResidentService>,
    values: &[DeviceBuffer],
    status: &ResidentLease,
    definitions: &[PinnedKernel],
) -> Result<CertificateSet> {
    if definitions.len() != values.len() {
        return Err(Error::State("B4 cold finite value population".into()));
    }
    let mut certificates = Vec::with_capacity(values.len());
    let mut by_lease = BTreeMap::new();
    for (value, kernel) in values.iter().zip(definitions) {
        let n = value.elements;
        let handles = vec![value.original_lease()?.handle(), status.handle()];
        let grid = [(n as u32).div_ceil(128), 1, 1];
        let launch = kernel.prepare(service.client(), grid, handles)?;
        by_lease
            .entry(value.original_lease()?.id())
            .or_insert(certificates.len());
        certificates.push(Certificate { launch });
    }
    Ok(CertificateSet {
        entries: certificates,
        by_lease,
        cohorts: OnceLock::new(),
    })
}
pub(super) fn certify_original(
    certificates: &CertificateSet,
    value: &DeviceBuffer,
    status: &Handle,
    original_status: &ResidentLease,
) -> Result<()> {
    let index = *certificates
        .by_lease
        .get(&value.original_lease()?.id())
        .ok_or_else(|| Error::State("native certificate original value absent".into()))?;
    certify_at(certificates, index, value, status, original_status)
}
/// The current body may index an immutable cold certificate, but must still
/// prove both exact physical ranges before the original finite kernel launch.
pub(super) fn certify_at(
    certificates: &CertificateSet,
    index: usize,
    value: &DeviceBuffer,
    status: &Handle,
    original_status: &ResidentLease,
) -> Result<()> {
    require_certificate(certificates, index, value, status, original_status)?.launch()
}
fn require_certificate<'a>(
    certificates: &'a CertificateSet,
    index: usize,
    value: &DeviceBuffer,
    status: &Handle,
    original_status: &ResidentLease,
) -> Result<&'a crate::cuda::kernel::PreparedPinnedLaunch> {
    require_certificate_range(certificates, index, &value.handle, status, original_status)
}
fn require_certificate_range<'a>(
    certificates: &'a CertificateSet,
    index: usize,
    value: &Handle,
    status: &Handle,
    original_status: &ResidentLease,
) -> Result<&'a crate::cuda::kernel::PreparedPinnedLaunch> {
    let c = certificates
        .entries
        .get(index)
        .ok_or_else(|| Error::State("native certificate original index absent".into()))?;
    let handles = c.launch.handles();
    if !same_range(&handles[0], value) {
        return Err(Error::State(
            "native certificate original value differs".into(),
        ));
    }
    if !same_range(&handles[1], status) || !same_range(status, &original_status.handle()) {
        return Err(Error::State("native original status binding".into()));
    }
    original_status.service().ensure_usable()?;
    Ok(&c.launch)
}
fn same_range(a: &Handle, b: &Handle) -> bool {
    a.service == b.service
        && a.clone().binding().memory.id() == b.clone().binding().memory.id()
        && a.offset_start == b.offset_start
        && a.offset_end == b.offset_end
        && a.size() == b.size()
}
impl ResidentProgram {
    pub(crate) async fn prepare_fragment_values(
        service: Arc<ResidentService>,
        bindings: Arc<SourceBindings>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self> {
        let (native, values, preflight) =
            NativeProgram::prepare_values(service.clone(), bindings.clone(), cancel).await?;
        Ok(Self {
            _source_owner: ResidentProgramSource::Windows(bindings.source().clone()),
            plan: bindings.shared_plan(),
            preflight,
            values,
            status: native.status().handle(),
            owned_ranges: OwnedRanges::new(service.client().service_id()),
            body: ResidentProgramBody::Native(native),
        })
    }
    pub(crate) fn native_body(&self) -> Result<&Arc<NativeProgram>> {
        match &self.body {
            ResidentProgramBody::Native(body) => Ok(body),
            _ => Err(Error::Unsupported(
                "not a native source fragment body".into(),
            )),
        }
    }
    pub(crate) fn finalize_fragment_preflight(
        &mut self,
        state_ids: &BTreeSet<zerogravity_core::device_events::BufferLeaseId>,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let all = self.native_body()?.accesses(&self.values)?;
        let owned = all
            .into_iter()
            .filter(|(lease, _)| !state_ids.contains(&lease.id()))
            .collect::<Vec<_>>();
        let values = self
            .values
            .iter()
            .map(|v| v.original_lease().map(|l| l.id()))
            .collect::<Result<BTreeSet<_>>>()?;
        let mut source = 0usize;
        let mut value = 0usize;
        let mut scratch = 64 * 1024 * 1024;
        let mut status = 0usize;
        for (lease, _) in &owned {
            if lease.id() == self.native_body()?.status().id() {
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
}
