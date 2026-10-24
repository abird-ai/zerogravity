//! Native source/body inside the SAME resident value map and dispatch loop.
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_inference::gdn_state::BoundGdnDomains;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_leases::{ResidentLease, ResidentReadback, ResidentService};
use crate::vllm_cuda_input_rms::{BoundRmsSource, FusedRmsPrepared, RmsPrepared};
use crate::vllm_cuda_qualification::native_gdn::{NativeGdnContext, NativeGdnPrepared};
use crate::vllm_cuda_qualification::native_plan::{
    NativePlanContext, NativePlanPhase, NativePlanPrograms, PreparedImages,
};
use crate::vllm_ptx_inline::PinnedKernel;
use cubecl::bytes::Bytes;
use cubecl_runtime::kernel::BufferIOAttr;
use std::collections::BTreeMap;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    ActivationPrecision, ConvDomain, KvDomain, RecordedVllmGdnFragment, RecurrentDomain,
    VllmBf16Invocation, VllmGdnProjectionKind,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_model_qwen35::{StateSpec, original_binding::OriginalWindowModelSource};
pub(super) mod certification;
pub(super) use certification::CertificationRun;
pub(crate) use certification::qualify as qualify_finite_cohorts;

pub(crate) struct NativeProgram {
    // Certificate handles retire before the original value/status leases.
    certificates: CertificateSet,
    source: Arc<OriginalWindowModelSource>,
    invocation: VllmBf16Invocation,
    completed_layers: usize,
    rms: Arc<RmsPrepared>,
    status: ResidentLease,
    gdn_nodes: Vec<[usize; 2]>,
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
    launch: crate::vllm_ptx_inline::PreparedPinnedLaunch,
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
pub(super) fn state_requirements(
    source: &OriginalWindowModelSource,
    plan: &ForwardPlan,
) -> Result<Vec<StateRequirement>> {
    if source.geometry().layers.len() != 24
        || source.state_specs().len() != 24
        || plan.prefix_capacity() != 512
    {
        return Err(Error::Unsupported(
            "initial .8B all42 source state layout".into(),
        ));
    }
    let mut requirements = Vec::with_capacity(42);
    for (layer, spec) in source.state_specs().iter().enumerate() {
        match *spec {
            StateSpec::Delta { conv, state }
                if layer % 4 != 3 && conv == 6144 * 3 && state == 16 * 128 * 128 =>
            {
                requirements.push(StateRequirement {
                    slot: StateSlot::Conv(ConvDomain(layer)),
                    layout: StateLayout::Vector { elements: conv },
                    bytes: conv * 4,
                });
                requirements.push(StateRequirement {
                    slot: StateSlot::Recurrent(RecurrentDomain(layer)),
                    layout: StateLayout::Vector { elements: state },
                    bytes: state * 4,
                });
            }
            StateSpec::Full { kv_width: 512 } if layer % 4 == 3 => {
                requirements.push(StateRequirement {
                    slot: StateSlot::Kv(KvDomain(layer)),
                    layout: StateLayout::Kv {
                        slab: 512,
                        capacity: 512,
                    },
                    bytes: 2 * 512 * 512 * 4,
                })
            }
            _ => {
                return Err(Error::Unsupported(
                    "actual source canonical state geometry/layer indices".into(),
                ));
            }
        }
    }
    requirements.sort_by_key(|r| r.slot);
    if requirements.iter().map(|r| r.slot).collect::<Vec<_>>() != plan.required_domains() {
        return Err(Error::State(
            "same original source all42 state slots differ from fragment".into(),
        ));
    }
    Ok(requirements)
}
struct Recipe {
    gdns: Vec<[usize; 2]>,
    fused: Vec<usize>,
    _metadata: WindowWorkspace,
}
fn only_output(node: &PlanNode) -> Result<zerogravity_core::accelerator::program::ValueId> {
    match node.outputs.as_slice() {
        [value] => Ok(*value),
        _ => Err(Error::State("one actual typed output".into())),
    }
}
fn require_recipe(
    source: &OriginalWindowModelSource,
    record: &RecordedVllmGdnFragment,
) -> Result<Recipe> {
    use zerogravity_core::accelerator::program::{
        GdnPrefixLayer, GdnPrefixRmsSite, ResidualRmsContract,
    };
    use zerogravity_model_qwen35::original_binding::{QwenMatrixRole as M, QwenParameterRole as P};
    let metadata = source.source().reserve_workspace(64 * 1024)?;
    let plan = record.plan();
    let base = plan
        .vllm_gdn_fragment()
        .ok_or_else(|| Error::Unsupported("not native original-root fragment".into()))?;
    if source.source_identity() != plan.source()
        || base.layer().index() != 0
        || plan.frontier_delta() != 0
        || plan.logits().is_some()
        || plan.is_operator_slice()
        || !matches!(record.completed_layers(), 0 | 1 | 3)
    {
        return Err(Error::Unsupported(
            "unpublishable exact initial source GDN cut".into(),
        ));
    }
    let full = record.completed_layers() != 0;
    let layers = record.completed_layers().max(1);
    let node = |index: usize| {
        plan.nodes()
            .get(index)
            .ok_or_else(|| Error::State("source recipe node absent".into()))
    };
    let embedding = node(0)?;
    let raw = only_output(embedding)?;
    let PlanOperation::TypedVllmBf16Embedding { descriptor } = &embedding.operation else {
        return Err(Error::State("original source rows first".into()));
    };
    if descriptor.teachers() != base.teachers() || !embedding.inputs.is_empty() {
        return Err(Error::State("original source teacher issuer".into()));
    }
    source.validate_matrix(M::Embedding, descriptor.tensor())?;
    let norm = node(1)?;
    let first_normal = only_output(norm)?;
    let PlanOperation::TypedVllmInputRms { descriptor } = &norm.operation else {
        return Err(Error::State("explicit layer0 input RMS".into()));
    };
    if descriptor.invocation() != base || norm.inputs != [raw] {
        return Err(Error::State("ordinary RMS consumes actual raw rows".into()));
    }
    source.validate_parameter(P::InputNorm(0), descriptor.parameter())?;
    let mut recipe = Recipe {
        gdns: Vec::with_capacity(layers),
        fused: Vec::with_capacity(2 * layers - 1),
        _metadata: metadata,
    };
    let mut cursor = 2;
    let mut branch = raw;
    let mut residual = raw;
    let projection = |index: usize,
                      call: &VllmBf16Invocation,
                      kind: VllmGdnProjectionKind,
                      input: zerogravity_core::accelerator::program::ValueId,
                      roles: &[M]|
     -> Result<zerogravity_core::accelerator::program::ValueId> {
        let n = node(index)?;
        let PlanOperation::TypedVllmGdnProjection { descriptor } = &n.operation else {
            return Err(Error::State("source projection record".into()));
        };
        if descriptor.invocation() != call
            || descriptor.kind() != kind
            || n.inputs != [input]
            || descriptor.members().len() != roles.len()
        {
            return Err(Error::State(
                "source ordered projection/producer recipe".into(),
            ));
        }
        for (member, role) in descriptor.members().iter().zip(roles) {
            source.validate_matrix(*role, member)?;
        }
        only_output(n)
    };
    let fused = |index: usize,
                 layer: GdnPrefixLayer,
                 site: GdnPrefixRmsSite,
                 left: zerogravity_core::accelerator::program::ValueId,
                 right: zerogravity_core::accelerator::program::ValueId,
                 role: P|
     -> Result<(
        zerogravity_core::accelerator::program::ValueId,
        zerogravity_core::accelerator::program::ValueId,
    )> {
        let n = node(index)?;
        let PlanOperation::TypedResidualRms {
            parameter,
            contract,
        } = &n.operation
        else {
            return Err(Error::State("bound dual residual/RMS record".into()));
        };
        if *contract != ResidualRmsContract::Vllm031CudaGdnPrefix(layer.rms(site)?)
            || n.inputs != [left, right]
            || n.outputs.len() != 2
        {
            return Err(Error::State(
                "actual branch/carried residual/site recipe".into(),
            ));
        }
        source.validate_parameter(role, parameter)?;
        Ok((n.outputs[0], n.outputs[1]))
    };
    for i in 0..layers {
        let layer = GdnPrefixLayer::new(plan.source(), i)?;
        let call = VllmBf16Invocation::new(layer, base.legacy_phase()?, base.teachers().clone())?;
        let (normal, carried) = if i == 0 {
            (first_normal, raw)
        } else {
            let pair = fused(
                cursor,
                layer,
                GdnPrefixRmsSite::InputResidual,
                branch,
                residual,
                P::InputNorm(i),
            )?;
            recipe.fused.push(cursor);
            cursor += 1;
            pair
        };
        let q = projection(
            cursor,
            &call,
            VllmGdnProjectionKind::Qkvz,
            normal,
            &[M::GdnQkv(i), M::GdnGate(i)],
        )?;
        cursor += 1;
        let b = projection(
            cursor,
            &call,
            VllmGdnProjectionKind::Ba,
            normal,
            &[M::GdnBeta(i), M::GdnAlpha(i)],
        )?;
        cursor += 1;
        let conv_index = cursor;
        let conv = node(cursor)?;
        cursor += 1;
        let rec_index = cursor;
        let rec = node(cursor)?;
        cursor += 1;
        let (
            PlanOperation::TypedVllmGdnConv { descriptor: c },
            PlanOperation::TypedVllmGdnRecurrent { descriptor: r },
        ) = (&conv.operation, &rec.operation)
        else {
            return Err(Error::State("actual Conv/recurrent pair".into()));
        };
        if c.invocation() != &call
            || r.invocation() != &call
            || conv.inputs != [q, b]
            || conv.outputs.len() != 2
            || rec.inputs != [conv.outputs[0], conv.outputs[1], b]
        {
            return Err(Error::State("exact source/layer/cohort GDN recipe".into()));
        }
        source.validate_parameter(P::GdnConvolution(i), c.parameter())?;
        for (parameter, role) in r
            .parameters()
            .iter()
            .zip([P::GdnA(i), P::GdnDt(i), P::GdnNorm(i)])
        {
            source.validate_parameter(role, parameter)?;
        }
        let gated = only_output(rec)?;
        recipe.gdns.push([conv_index, rec_index]);
        if full {
            let out = projection(
                cursor,
                &call,
                VllmGdnProjectionKind::GdnOut,
                gated,
                &[M::GdnOutput(i)],
            )?;
            cursor += 1;
            let (post, post_residual) = fused(
                cursor,
                layer,
                GdnPrefixRmsSite::PostAttention,
                out,
                carried,
                P::FfnNorm(i),
            )?;
            recipe.fused.push(cursor);
            cursor += 1;
            let gate_up = projection(
                cursor,
                &call,
                VllmGdnProjectionKind::GateUp,
                post,
                &[M::FfnGate(i), M::FfnUp(i)],
            )?;
            cursor += 1;
            let act = node(cursor)?;
            cursor += 1;
            let PlanOperation::TypedVllmSiluAndMul { descriptor } = &act.operation else {
                return Err(Error::State("actual BF16 SiluAndMul record".into()));
            };
            if descriptor.invocation() != &call || act.inputs != [gate_up] {
                return Err(Error::State(
                    "actual computed GateUp activation recipe".into(),
                ));
            }
            branch = projection(
                cursor,
                &call,
                VllmGdnProjectionKind::Down,
                only_output(act)?,
                &[M::FfnDown(i)],
            )?;
            cursor += 1;
            residual = post_residual;
        } else {
            branch = gated;
        }
    }
    let rest = &plan.nodes()[cursor..];
    if record.output() != branch
        || record.residual_output() != full.then_some(residual)
        || rest.len() != 42 - 2 * layers
        || rest.iter().any(|n| {
            !matches!(n.operation, PlanOperation::TypedVllmStateCarry)
                || !n.inputs.is_empty()
                || !n.outputs.is_empty()
        })
    {
        return Err(Error::State(
            "exact source initial cut output/canonical remaining-domain copies".into(),
        ));
    }
    Ok(recipe)
}
pub(crate) fn capacity_bound(
    source: &OriginalWindowModelSource,
    record: &RecordedVllmGdnFragment,
) -> Result<usize> {
    // Closed actual config/epsilon guard BEFORE any Device/client/admission.
    let _config = crate::vllm_cuda_input_rms::BoundRmsSource::bind(
        source.owner().clone(),
        0,
        zerogravity_core::accelerator::program::GdnPrefixRmsSite::PostAttention,
    )?;
    require_recipe(source, record)?;
    let plan = record.plan();
    state_requirements(source, plan)?;
    let rows = plan.vllm_gdn_fragment().unwrap().rows();
    let layers = record.completed_layers().max(1);
    let fused = if record.completed_layers() == 0 {
        0
    } else {
        layers * 2 - 1
    };
    let full = record.completed_layers() != 0;
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
    let bound = 2 * 32_784_384
        + 64 * 1024 * 1024
        + layers * weights
        + 8_519_680
        + (10_240 + rows * 20_492)
        + layers * (rows * 36_928 + per_gdn + extra_values)
        + fused * (10_240 + rows * 20_492)
        + 4 * (layers + 1);
    bound
        .checked_add(32 * 1024 * 1024 - 1)
        .map(|n| n / (32 * 1024 * 1024) * (32 * 1024 * 1024))
        .ok_or_else(|| Error::Limit("native fragment capacity bound overflow".into()))
}
impl NativeProgram {
    /// Cold pre-binding stage. No image/module or StateRoot is manufactured.
    pub(crate) async fn prepare_values(
        service: Arc<ResidentService>,
        source: Arc<OriginalWindowModelSource>,
        record: &RecordedVllmGdnFragment,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Vec<DeviceBuffer>, ResidentPreflight)> {
        cancelled(cancel)?;
        let recipe = require_recipe(&source, record)?;
        let plan = record.plan();
        let invocation = plan.vllm_gdn_fragment().unwrap().clone();
        let states = state_requirements(&source, plan)?;
        let metadata = source.source().reserve_workspace(4 * 1024 * 1024)?;
        let layers = record.completed_layers().max(1);
        let packed_weights = if record.completed_layers() == 0 {
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
            + invocation.rows() * 4096
            + public_bytes
            + recipe.fused.len() * (2048 + invocation.rows() * 8192);
        let readback_chunks = max_readback.div_ceil(32 * 1024 * 1024);
        if !(3..=8).contains(&readback_chunks) {
            return Err(Error::Limit(
                "closed initial fragment host readback span".into(),
            ));
        }
        let readback = (0..readback_chunks)
            .map(|_| source.source().reserve_workspace(32 * 1024 * 1024))
            .collect::<Result<Vec<_>>>()?;
        let rms = RmsPrepared::prepare(
            service.clone(),
            source.source().clone(),
            invocation.teachers().ids(),
            0,
        )
        .await?;
        cancelled(cancel)?;
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(source.clone())?;
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
            let PlanOperation::TypedResidualRms {
                parameter,
                contract,
            } = &node.operation
            else {
                return Err(Error::State("bound RMS recipe changed".into()));
            };
            let zerogravity_core::accelerator::program::ResidualRmsContract::Vllm031CudaGdnPrefix(
                key,
            ) = *contract
            else {
                return Err(Error::Unsupported("typed bound RMS contract".into()));
            };
            let source_binding =
                BoundRmsSource::bind(source.owner().clone(), key.layer().index(), key.site())?;
            if source
                .source()
                .parameter(&key.gamma_name())?
                .source_parameter()
                != *parameter
            {
                return Err(Error::State(
                    "actual gamma header differs from typed descriptor".into(),
                ));
            }
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
                invocation.rows(),
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
                    invocation.rows() * 1024,
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
            invocation,
            completed_layers: record.completed_layers(),
            rms,
            status,
            certificates,
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
                        source: self.source.owner().clone(),
                        domains,
                    },
                    plan,
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
                source_owner: self.source.owner().clone(),
                invocation: self.invocation.clone(),
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
                    source_owner: body.source.owner().clone(),
                    invocation: body.invocation.clone(),
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
        permit.require_invocation(self.rms.normalized().service(), &self.invocation, layers)
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
            || p.valid_tokens != self.invocation.rows()
            || self.invocation.teachers().ids().first().copied() != Some(p.token)
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
        match &node.operation {
            PlanOperation::TypedVllmBf16Embedding { .. } => Ok(0), // Actual uploaded source rows, separate from RMS output.
            PlanOperation::TypedVllmInputRms { .. } => {
                self.rms.enqueue()?;
                self.rms_dispatches.fetch_add(10, Ordering::AcqRel);
                Ok(10)
            }
            PlanOperation::TypedResidualRms { contract, .. } => {
                self.fused
                    .get(&index)
                    .ok_or_else(|| Error::State("actual bound RMS record absent".into()))?
                    .enqueue(cancel)?;
                self.rms_dispatches.fetch_add(10, Ordering::AcqRel);
                if self.fault.load(Ordering::Acquire) == 3
                    && matches!(contract,
                    zerogravity_core::accelerator::program::ResidualRmsContract::Vllm031CudaGdnPrefix(key)
                    if key.layer().index()==0 && key.site()==zerogravity_core::accelerator::program::GdnPrefixRmsSite::PostAttention)
                {
                    self.cut.store(true, Ordering::Release);
                    return Err(Error::State(
                        "deliberate unknown after actual post-attention dual RMS".into(),
                    ));
                }
                Ok(10)
            }
            PlanOperation::TypedVllmGdnProjection { .. }
            | PlanOperation::TypedVllmSiluAndMul { .. } => {
                let cancel_down = self.fault.load(Ordering::Acquire) == 4
                    && matches!(&node.operation,
                    PlanOperation::TypedVllmGdnProjection{descriptor} if descriptor.kind()==VllmGdnProjectionKind::Down && descriptor.invocation().layer().index()==0);
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
            PlanOperation::TypedVllmGdnConv { .. } => {
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
            PlanOperation::TypedVllmGdnRecurrent { .. } => {
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
        fault: Option<crate::resident_inference::native_fragment::VllmGdnFragmentFault>,
    ) -> Result<()> {
        if self.launched.load(Ordering::Acquire) {
            return Err(Error::State(
                "cannot change native fault after enqueue".into(),
            ));
        }
        use crate::resident_inference::native_fragment::VllmGdnFragmentFault::*;
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
    pub(crate) fn source(&self) -> &Arc<OriginalWindowModelSource> {
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
    pub(crate) fn invocation_for_layer(&self, index: usize) -> Result<VllmBf16Invocation> {
        VllmBf16Invocation::new(
            zerogravity_core::accelerator::program::GdnPrefixLayer::new(
                self.invocation.layer().source(),
                index,
            )?,
            self.invocation.legacy_phase()?,
            self.invocation.teachers().clone(),
        )
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
            crate::vllm_cuda_gdn::GdnStateAdapter::validate_status(status.as_ref())?;
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
        unsafe{crate::vllm_cuda_qualification::native_plan::retire_native_set_after_physical_completion(self.images.get().ok_or_else(||Error::State("native set absent".into()))?)}.map_err(|e|Error::State(format!("native set retirement: {e}")))
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
) -> Result<&'a crate::vllm_ptx_inline::PreparedPinnedLaunch> {
    require_certificate_range(certificates, index, &value.handle, status, original_status)
}
fn require_certificate_range<'a>(
    certificates: &'a CertificateSet,
    index: usize,
    value: &Handle,
    status: &Handle,
    original_status: &ResidentLease,
) -> Result<&'a crate::vllm_ptx_inline::PreparedPinnedLaunch> {
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
        source: Arc<OriginalWindowModelSource>,
        record: RecordedVllmGdnFragment,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self> {
        let (native, values, preflight) =
            NativeProgram::prepare_values(service.clone(), source.clone(), &record, cancel).await?;
        let (plan, _) = record.into_parts();
        Ok(Self {
            _source_owner: ResidentProgramSource::Windows(source),
            plan,
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
