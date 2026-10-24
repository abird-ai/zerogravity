//! SAME-plan projection/ACT records; no traversal, completion or root publisher.
use super::activation_cubin::resident::ActivationPrograms;
use super::activation_manifest::Activation;
use super::cubin_manifests::{Projection, ProjectionBinding, ProjectionImage, WORKSPACE_BYTES};
use super::mlp_cubin::resident::{MlpBindings, MlpPrograms};
use super::mlp_manifests::{MlpOperation, MlpProjection};
pub(crate) use super::projection_cubin::resident::PreparedImages;
use super::projection_cubin::resident::{AdmittedImages, ProjectionBindings, ProjectionPrograms};
use super::projection_cubin::{
    CubinQualificationInputs, native_error, not_cancelled, validate_image_set,
};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use crate::resident_program::{DeviceBuffer, ResidentBufferLayout, ResidentPhysicalType};
use crate::vllm_cuda_input_rms::BoundRmsSource;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::model::qwen_owned::QwenOriginalWindowText;
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    ActivationPrecision, ForwardPlan, GdnPrefixRmsSite, PlanNode, PlanOperation,
    ProgramNumericalProfile, SourceTensor, ValueId, ValueShape, VllmBf16Invocation,
    VllmGdnProjection, VllmGdnProjectionKind,
};
use zerogravity_core::device_events::{BufferLeaseId, BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::ResourceArbiter;
use zerogravity_core::{Error, Result};

const METADATA_BYTES: usize = 2 * 1024 * 1024;
const WINDOW_BYTES: usize = 1024 * 1024;
const MAX_NATIVE_NODES: usize = 18;

struct NativeImageOwner {
    _source_owner: CompletionOwner,
    _metadata: Arc<WindowWorkspace>,
}

fn projection_images(plan: &ForwardPlan) -> Result<BTreeSet<ProjectionImage>> {
    projection_images_inner(plan, false)
}
fn projection_images_inner(
    plan: &ForwardPlan,
    connected: bool,
) -> Result<BTreeSet<ProjectionImage>> {
    if plan.numerical_profile() != ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1 {
        return Err(Error::Unsupported(
            "native image factory exact typed profile".into(),
        ));
    }
    let mut required = BTreeSet::new();
    let mut selected = 0usize;
    for node in plan.nodes() {
        match &node.operation {
            PlanOperation::TypedVllmGdnProjection { descriptor } => {
                let operation = projection_operation(descriptor)?;
                let (input, output) = node_arity(node)?;
                require_value(
                    plan,
                    input,
                    descriptor.invocation(),
                    descriptor.input_elements(),
                )?;
                require_value(
                    plan,
                    output,
                    descriptor.invocation(),
                    descriptor.output_elements(),
                )?;
                match operation {
                    NativeOperation::Projection(p) => {
                        required.extend(p.launches().into_iter().map(|launch| launch.image()));
                    }
                    NativeOperation::Mlp(p) => {
                        required.extend(p.operations().into_iter().filter_map(|operation| {
                            match operation {
                                MlpOperation::Kernel(launch) => Some(launch.image()),
                                MlpOperation::Clear(_) => None,
                            }
                        }));
                    }
                    NativeOperation::Activation(_) => unreachable!(),
                }
                selected += 1;
            }
            PlanOperation::TypedVllmSiluAndMul { descriptor } => {
                if !connected {
                    return Err(Error::Unsupported(
                        "typed activation image requires separately typed connected input".into(),
                    ));
                }
                let (input, output) = node_arity(node)?;
                require_value(
                    plan,
                    input,
                    descriptor.invocation(),
                    descriptor.input_elements(),
                )?;
                require_value(
                    plan,
                    output,
                    descriptor.invocation(),
                    descriptor.output_elements(),
                )?;
                selected += 1;
            }
            _ => {}
        }
    }
    if selected == 0 || selected > MAX_NATIVE_NODES || required.is_empty() || required.len() > 6 {
        return Err(Error::Unsupported(
            "bounded native projection image requirements".into(),
        ));
    }
    Ok(required)
}

/// Source-only image/DSO verification and the unchanged default6/8/19 set.
/// No module is loaded until the operation factories run. This initial bridge
/// cannot admit activation or select the separate connected-set policy.
pub(crate) fn prepare_native_images(
    service: &Arc<ResidentService>,
    source_owner: CompletionOwner,
    source: &VerifiedTensorSource,
    plan: &ForwardPlan,
    inputs: CubinQualificationInputs,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<PreparedImages>> {
    not_cancelled(cancelled)?;
    service.ensure_usable()?;
    init.require_service(service)?;
    if source.source_identity() != service.source_identity()
        || source.source_identity() != plan.source()
        || inputs.images.capacity() > 8
        || ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.device_cap
    {
        return Err(Error::State(
            "native image original source/service/capacity".into(),
        ));
    }
    let metadata = Arc::new(source.reserve_workspace(METADATA_BYTES)?);
    let source_owner: CompletionOwner = Arc::new(NativeImageOwner {
        _source_owner: source_owner,
        _metadata: metadata,
    });
    init.retain_completion_owner(source_owner.clone())?;
    let required = projection_images(plan)?;
    validate_image_set(source, &required, &inputs.images)?;
    not_cancelled(cancelled)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    not_cancelled(cancelled)?;
    let admitted = AdmittedImages::reserve(service, source_owner, inputs.images)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let images = admitted.prepare_images(sources, init, cancelled)?;
    images.require_service(service)?;
    Ok(images)
}

/// Separate closed7/12/21 admission; default6/8/19 bridge stays unchanged.
pub(crate) fn prepare_connected_native_images(
    service: &Arc<ResidentService>,
    source_owner: CompletionOwner,
    source: &VerifiedTensorSource,
    plan: &ForwardPlan,
    inputs: super::LayerMlpCubinInputs,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<PreparedImages>> {
    use super::projection_cubin::validate_activation_image;
    not_cancelled(cancelled)?;
    service.ensure_usable()?;
    init.require_service(service)?;
    if source.source_identity() != service.source_identity()
        || source.source_identity() != plan.source()
        || inputs.projections.images.capacity() > 8
        || ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.projections.device_cap
    {
        return Err(Error::State(
            "connected native original source/service/device capacity".into(),
        ));
    }
    let metadata = Arc::new(source.reserve_workspace(METADATA_BYTES)?);
    let source_owner: CompletionOwner = Arc::new(NativeImageOwner {
        _source_owner: source_owner,
        _metadata: metadata,
    });
    init.retain_completion_owner(source_owner.clone())?;
    let selected = projection_images_inner(plan, true)?;
    if !plan
        .nodes()
        .iter()
        .any(|node| matches!(node.operation, PlanOperation::TypedVllmSiluAndMul { .. }))
    {
        return Err(Error::Unsupported(
            "connected native image policy requires actual typed activation consumer".into(),
        ));
    }
    let all = BTreeSet::from([
        ProjectionImage::Bf16Gemv2TSm86,
        ProjectionImage::Bf16GemvxSm86,
        ProjectionImage::CutlassWmmaSm80,
        ProjectionImage::Bf16SplitKReduceSm86,
        ProjectionImage::AmpereBf16Gemm128x64Sm80,
        ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
    ]);
    if !selected.is_subset(&all) {
        return Err(Error::Unsupported("unknown connected native image".into()));
    }
    validate_image_set(source, &all, &inputs.projections.images)?;
    validate_activation_image(source, &inputs.activation)?;
    not_cancelled(cancelled)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.projections.cublas, &inputs.projections.cublas_lt)
        .map_err(native_error)?;
    let admitted = AdmittedImages::reserve_with_activation(
        service,
        source_owner,
        inputs.projections.images,
        Some(inputs.activation),
    )?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let images = admitted.prepare_connected_images(sources, init, cancelled)?;
    images.require_service(service)?;
    Ok(images)
}

/// Caller proved every original submission complete and retired every shared
/// kernel record. The SAME armed immutable-W flight retains all source/images
/// through this one set unload; errors follow its original quarantine path.
pub(crate) unsafe fn retire_native_set_after_physical_completion(
    images: &PreparedImages,
) -> std::result::Result<(), cubecl::server::ServerError> {
    unsafe { images.set().retire_after_physical_completion() }
}

/// SAME source owner, original queue/library and actual request issuer.
pub(crate) struct NativePlanContext {
    pub images: Arc<PreparedImages>,
    pub source_owner: Arc<QwenOriginalWindowText>,
    pub invocation: VllmBf16Invocation,
    pub service: Arc<ResidentService>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeSubmitted {
    pub dispatches: usize,
    pub native_launches: usize,
    pub workspace_clears: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct WeightKey {
    layer: usize,
    kind: u8,
}
impl WeightKey {
    fn of(descriptor: &VllmGdnProjection) -> Self {
        let kind = match descriptor.kind() {
            VllmGdnProjectionKind::Qkvz => 0,
            VllmGdnProjectionKind::Ba => 1,
            VllmGdnProjectionKind::GdnOut => 2,
            VllmGdnProjectionKind::GateUp => 3,
            VllmGdnProjectionKind::Down => 4,
        };
        Self {
            layer: descriptor.invocation().layer().index(),
            kind,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeOperation {
    Projection(Projection),
    Mlp(MlpProjection),
    Activation(Activation),
}
fn projection_operation(descriptor: &VllmGdnProjection) -> Result<NativeOperation> {
    descriptor.validate()?;
    use VllmGdnProjectionKind::{Ba, Down, GateUp, GdnOut, Qkvz};
    Ok(match (descriptor.invocation().rows(), descriptor.kind()) {
        (1, Qkvz) => NativeOperation::Projection(Projection::Qkvz1),
        (1, Ba) => NativeOperation::Projection(Projection::Ba1),
        (1, GdnOut) => NativeOperation::Projection(Projection::Out1),
        (64, Qkvz) => NativeOperation::Projection(Projection::Qkvz64),
        (64, Ba) => NativeOperation::Projection(Projection::Ba64),
        (64, GdnOut) => NativeOperation::Projection(Projection::Out64),
        (2, Qkvz) => NativeOperation::Projection(Projection::Qkvz2),
        (2, Ba) => NativeOperation::Projection(Projection::Ba2),
        (2, GdnOut) => NativeOperation::Projection(Projection::Out2),
        (1, GateUp) => NativeOperation::Mlp(MlpProjection::GateUp1),
        (1, Down) => NativeOperation::Mlp(MlpProjection::Down1),
        (64, GateUp) => NativeOperation::Mlp(MlpProjection::GateUp64),
        (64, Down) => NativeOperation::Mlp(MlpProjection::Down64),
        (2, GateUp) => NativeOperation::Mlp(MlpProjection::GateUp2),
        (2, Down) => NativeOperation::Mlp(MlpProjection::Down2),
        _ => return Err(Error::Unsupported("closed native projection rows".into())),
    })
}
fn activation_operation(rows: usize) -> Result<NativeOperation> {
    Ok(NativeOperation::Activation(match rows {
        1 => Activation::Rows1,
        64 => Activation::Rows64,
        2 => Activation::Rows2,
        _ => return Err(Error::Unsupported("closed native activation rows".into())),
    }))
}
impl NativeOperation {
    fn require_images(self, images: &PreparedImages) -> Result<()> {
        match self {
            Self::Projection(projection) => {
                for launch in projection.launches() {
                    images.image(launch.image())?;
                }
            }
            Self::Mlp(projection) => {
                for operation in projection.operations() {
                    if let MlpOperation::Kernel(launch) = operation {
                        images.image(launch.image())?;
                    }
                }
            }
            Self::Activation(_) => {
                images.activation()?;
            }
        }
        Ok(())
    }
    fn needs_workspace(self) -> bool {
        match self {
            Self::Projection(projection) => projection
                .launches()
                .iter()
                .any(|launch| launch.bindings().contains(&ProjectionBinding::Workspace)),
            Self::Mlp(projection) => {
                projection
                    .operations()
                    .iter()
                    .any(|operation| match operation {
                        MlpOperation::Clear(_) => true,
                        MlpOperation::Kernel(launch) => {
                            launch.bindings().contains(&ProjectionBinding::Workspace)
                        }
                    })
            }
            Self::Activation(_) => false,
        }
    }
}

/// Per-node operands sealed from the original immutable value map, not supplied
/// again at dispatch. W remains the actual ordered source member join.
struct NativeNode {
    operation: NativeOperation,
    weight: Option<WeightKey>,
    input: ResidentLease,
    output: ResidentLease,
}
struct WeightPayload {
    descriptor: VllmGdnProjection,
    windows: Vec<Arc<TensorWindow>>,
    packed: Vec<u8>,
    source: Arc<QwenOriginalWindowText>,
    // Packed payload + original upload copy admitted before either allocation.
    _payload: WindowWorkspace,
}
#[derive(Clone)]
struct PreparedWeight {
    lease: ResidentLease,
    _payload: Arc<WeightPayload>,
}
impl WeightPayload {
    fn validate(&self, context: &NativePlanContext) -> Result<()> {
        self.descriptor.validate()?;
        same_request(self.descriptor.invocation(), &context.invocation)?;
        if !Arc::ptr_eq(&self.source, &context.source_owner) {
            return Err(Error::State("native W original model owner differs".into()));
        }
        let mut windows = self.windows.iter();
        let mut bytes = 0usize;
        for member in self.descriptor.members() {
            validate_member(&self.source, member)?;
            let mut next_row = 0;
            while next_row < member.rows {
                let window = windows
                    .next()
                    .ok_or_else(|| Error::State("native W source window missing".into()))?;
                let rows = window.row_range();
                let matrix = window.matrix();
                if matrix.source_identity() != member.source
                    || matrix.tensor().name != member.name
                    || matrix.tensor().kind != member.kind
                    || matrix.axes() != member.axes
                    || matrix.rows() != member.rows
                    || matrix.columns() != member.columns
                    || rows.start != next_row
                    || rows.end <= rows.start
                    || rows.end > member.rows
                    || window.as_slice().len() != (rows.end - rows.start) * member.columns * 2
                    || window.as_slice().len() > WINDOW_BYTES
                {
                    return Err(Error::State(
                        "native W original ordered window coverage".into(),
                    ));
                }
                bytes = bytes
                    .checked_add(window.as_slice().len())
                    .ok_or_else(|| Error::Limit("native W source coverage bytes".into()))?;
                next_row = rows.end;
            }
        }
        if windows.next().is_some() || bytes != self.packed.len() {
            return Err(Error::State(
                "native W packed source coverage differs".into(),
            ));
        }
        Ok(())
    }
}

/// One library across layers0..2; factory records deduplicate by actual ABI,
/// original W by exact source layer/roles. Caller owns the SAME armed flight.
pub(crate) struct NativePlanPrograms {
    nodes: BTreeMap<usize, NativeNode>,
    weights: BTreeMap<WeightKey, PreparedWeight>,
    workspace: Option<ResidentLease>,
    projections: Option<Arc<ProjectionPrograms>>,
    mlp: Option<Arc<MlpPrograms>>,
    activations: Option<Arc<ActivationPrograms>>,
    context: NativePlanContext,
    // All private metadata drops before its original source admission.
    _metadata: Arc<WindowWorkspace>,
}

fn same_request(actual: &VllmBf16Invocation, expected: &VllmBf16Invocation) -> Result<()> {
    actual.validate()?;
    expected.validate()?;
    if actual.layer().source() != expected.layer().source()
        || actual.teachers() != expected.teachers()
        || actual.legacy_phase()? != expected.legacy_phase()?
    {
        return Err(Error::State(
            "native plan actual request issuer/span/phase".into(),
        ));
    }
    Ok(())
}
fn node_arity(node: &PlanNode) -> Result<(ValueId, ValueId)> {
    let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
        return Err(Error::State("native plan unary input/output arity".into()));
    };
    if node.state.is_some() || input == output {
        return Err(Error::State(
            "native projection/ACT has state or aliases value".into(),
        ));
    }
    Ok((*input, *output))
}
fn bind_value_identity(
    originals: &mut BTreeMap<BufferLeaseId, ValueId>,
    value: ValueId,
    original: BufferLeaseId,
) -> Result<()> {
    if originals
        .get(&original)
        .is_some_and(|&prior| prior != value)
    {
        return Err(Error::State(
            "distinct native plan values alias one original lease".into(),
        ));
    }
    originals.insert(original, value);
    Ok(())
}
fn require_value(
    plan: &ForwardPlan,
    id: ValueId,
    invocation: &VllmBf16Invocation,
    elements: usize,
) -> Result<()> {
    if plan.values().get(id.0 as usize) != Some(&ValueShape::Vector(elements))
        || plan.precisions().get(id.0 as usize) != Some(&ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State("native plan exact dense BF16 value".into()));
    }
    plan.vllm_value_provenance(id)?
        .ok_or_else(|| Error::State("native plan producer provenance absent".into()))?
        .require_invocation(invocation)
}
fn original_value(
    values: &[DeviceBuffer],
    id: ValueId,
    elements: usize,
    service: &Arc<ResidentService>,
) -> Result<ResidentLease> {
    let buffer = values
        .get(id.0 as usize)
        .ok_or_else(|| Error::State("native original plan value absent".into()))?;
    buffer.require_layout(ResidentBufferLayout::new(
        ResidentPhysicalType::Bf16,
        elements,
    )?)?;
    let lease = buffer.original_lease()?;
    if !Arc::ptr_eq(lease.service(), service)
        || lease.source_identity() != service.source_identity()
        || lease.service_generation() != service.generation()
        || lease.bytes()
            != elements
                .checked_mul(2)
                .ok_or_else(|| Error::Limit("native BF16 physical bytes".into()))?
        || !matches!(
            lease.role(),
            BufferRole::Activation | BufferRole::Output | BufferRole::Scratch
        )
    {
        return Err(Error::State(
            "native value original service/source/role/extent".into(),
        ));
    }
    Ok(lease.clone())
}
fn validate_member(owner: &QwenOriginalWindowText, member: &SourceTensor) -> Result<()> {
    owner.text_tensor(&member.name)?;
    let actual = owner.source().matrix(&member.name, member.axes)?;
    if member.source != owner.source_identity()
        || member.kind != actual.tensor().kind
        || member.rows != actual.rows()
        || member.columns != actual.columns()
        || member.axes != actual.axes()
    {
        return Err(Error::State(
            "native ordered original source matrix differs".into(),
        ));
    }
    Ok(())
}
fn prepare_weight(
    context: &NativePlanContext,
    descriptor: &VllmGdnProjection,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<PreparedWeight> {
    not_cancelled(cancelled)?;
    let source = context.source_owner.source();
    let bytes = descriptor
        .members()
        .iter()
        .try_fold(0usize, |total, member| {
            let member_bytes = u64::try_from(member.rows)
                .ok()
                .zip(u64::try_from(member.columns).ok())
                .and_then(|(rows, columns)| rows.checked_mul(columns))
                .and_then(|elements| elements.checked_mul(2))
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or_else(|| Error::Limit("native BF16 source W byte extent".into()))?;
            total
                .checked_add(member_bytes)
                .ok_or_else(|| Error::Limit("native ordered W bytes".into()))
        })?;
    let payload = source.reserve_operator_buffers(bytes, bytes)?;
    let mut packed = Vec::with_capacity(bytes);
    let mut windows = Vec::new();
    for member in descriptor.members() {
        not_cancelled(cancelled)?;
        validate_member(&context.source_owner, member)?;
        let matrix = source.matrix(&member.name, member.axes)?;
        let row_bytes = member
            .columns
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("native original W row bytes".into()))?;
        let rows_per_window = WINDOW_BYTES
            .checked_div(row_bytes)
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::Limit("native original W row/window bound".into()))?;
        for first in (0..member.rows).step_by(rows_per_window) {
            let end = first.saturating_add(rows_per_window).min(member.rows);
            let window = Arc::new(match cancelled {
                Some(flag) => source.read_rows_cancellable(&matrix, first..end, flag)?,
                None => source.read_rows(&matrix, first..end)?,
            });
            if window.as_slice().len() != (end - first) * row_bytes {
                return Err(Error::State("native complete ordered W window".into()));
            }
            packed.extend_from_slice(window.as_slice());
            windows.push(window);
        }
    }
    if packed.len() != bytes || packed.capacity() != bytes {
        return Err(Error::State("native exact packed W allocation".into()));
    }
    let owner = Arc::new(WeightPayload {
        descriptor: descriptor.clone(),
        windows,
        packed,
        source: context.source_owner.clone(),
        _payload: payload,
    });
    owner.validate(context)?;
    // Attach actual source/window/payload owners BEFORE original H2D.
    init.retain_completion_owner(owner.clone())?;
    let lease = init.upload_bytes(&owner.packed, BufferRole::ImmutableWeights)?;
    if !Arc::ptr_eq(lease.service(), &context.service)
        || lease.source_identity() != context.source_owner.source_identity()
        || lease.bytes() != bytes
    {
        return Err(Error::State(
            "native W original initialization/service".into(),
        ));
    }
    Ok(PreparedWeight {
        lease,
        _payload: owner,
    })
}

struct NativePlanBindings {
    nodes: BTreeMap<usize, NativeNode>,
    descriptors: BTreeMap<WeightKey, VllmGdnProjection>,
    projection_records: BTreeSet<Projection>,
    mlp_records: BTreeSet<MlpProjection>,
    activation_records: BTreeSet<Activation>,
    needs_workspace: bool,
    context: NativePlanContext,
    metadata: Arc<WindowWorkspace>,
}
fn prepare_bindings(
    context: NativePlanContext,
    plan: &ForwardPlan,
    values: &[DeviceBuffer],
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<NativePlanBindings> {
    not_cancelled(cancelled)?;
    context.service.ensure_usable()?;
    init.require_service(&context.service)?;
    context.images.require_service(&context.service)?;
    context.invocation.validate()?;
    if plan.source() != context.source_owner.source_identity()
        || plan.source() != context.service.source_identity()
        || plan.source() != context.invocation.layer().source()
        || plan.numerical_profile() != ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1
        || values.len() != plan.values().len()
    {
        return Err(Error::Unsupported(
            "native same-plan/source/profile/value coverage".into(),
        ));
    }
    let metadata = Arc::new(
        context
            .source_owner
            .source()
            .reserve_workspace(METADATA_BYTES)?,
    );
    init.retain_completion_owner(metadata.clone())?;
    init.retain_completion_owner(context.source_owner.clone())?;
    // Reuse the existing exact .8B config/source guard; no target retag.
    BoundRmsSource::bind(
        context.source_owner.clone(),
        context.invocation.layer().index(),
        GdnPrefixRmsSite::PostAttention,
    )?;
    let mut nodes = BTreeMap::new();
    let mut descriptors: BTreeMap<WeightKey, VllmGdnProjection> = BTreeMap::new();
    let mut projection_records = BTreeSet::new();
    let mut mlp_records = BTreeSet::new();
    let mut activation_records = BTreeSet::new();
    let mut originals = BTreeMap::new();
    let mut needs_workspace = false;
    for (index, node) in plan.nodes().iter().enumerate() {
        let (operation, invocation, input_elements, output_elements, weight) = match &node.operation
        {
            PlanOperation::TypedVllmGdnProjection { descriptor } => {
                descriptor.validate()?;
                same_request(descriptor.invocation(), &context.invocation)?;
                for member in descriptor.members() {
                    validate_member(&context.source_owner, member)?;
                }
                let key = WeightKey::of(descriptor);
                if let Some(prior) = descriptors.get(&key) {
                    if prior != descriptor.as_ref() {
                        return Err(Error::State(
                            "native repeated W role/descriptor differs".into(),
                        ));
                    }
                } else {
                    descriptors.insert(key, descriptor.as_ref().clone());
                }
                (
                    projection_operation(descriptor)?,
                    descriptor.invocation(),
                    descriptor.input_elements(),
                    descriptor.output_elements(),
                    Some(key),
                )
            }
            PlanOperation::TypedVllmSiluAndMul { descriptor } => {
                descriptor.validate()?;
                same_request(descriptor.invocation(), &context.invocation)?;
                (
                    activation_operation(descriptor.invocation().rows())?,
                    descriptor.invocation(),
                    descriptor.input_elements(),
                    descriptor.output_elements(),
                    None,
                )
            }
            _ => continue,
        };
        let (input_id, output_id) = node_arity(node)?;
        require_value(plan, input_id, invocation, input_elements)?;
        require_value(plan, output_id, invocation, output_elements)?;
        let input = original_value(values, input_id, input_elements, &context.service)?;
        let output = original_value(values, output_id, output_elements, &context.service)?;
        bind_value_identity(&mut originals, input_id, input.id())?;
        bind_value_identity(&mut originals, output_id, output.id())?;
        if input.id() == output.id() {
            return Err(Error::State(
                "native independent input/output leases alias".into(),
            ));
        }
        match operation {
            NativeOperation::Projection(p) => {
                projection_records.insert(p);
            }
            NativeOperation::Mlp(p) => {
                mlp_records.insert(p);
            }
            NativeOperation::Activation(a) => {
                activation_records.insert(a);
            }
        }
        needs_workspace |= operation.needs_workspace();
        operation.require_images(&context.images)?;
        nodes.insert(
            index,
            NativeNode {
                operation,
                weight,
                input,
                output,
            },
        );
        if nodes.len() > MAX_NATIVE_NODES {
            return Err(Error::Limit(
                "closed native layers0..2 node metadata".into(),
            ));
        }
    }
    if nodes.is_empty() {
        return Err(Error::Unsupported(
            "same plan has no native projection/ACT nodes".into(),
        ));
    }
    Ok(NativePlanBindings {
        nodes,
        descriptors,
        projection_records,
        mlp_records,
        activation_records,
        needs_workspace,
        context,
        metadata,
    })
}

impl NativePlanPrograms {
    pub(crate) fn prepare(
        context: NativePlanContext,
        plan: &ForwardPlan,
        values: &[DeviceBuffer],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let NativePlanBindings {
            nodes,
            descriptors,
            projection_records,
            mlp_records,
            activation_records,
            needs_workspace,
            context,
            metadata,
        } = prepare_bindings(context, plan, values, init, cancelled)?;
        // All source/plan/physical joins above precede new W staging/H2D.
        let mut weights = BTreeMap::new();
        for (key, descriptor) in descriptors {
            weights.insert(key, prepare_weight(&context, &descriptor, init, cancelled)?);
        }
        let workspace = if needs_workspace {
            Some(init.allocate_empty(WORKSPACE_BYTES, BufferRole::Scratch)?)
        } else {
            None
        };
        // No managed allocation/upload follows the first native module prepare.
        let projections = if projection_records.is_empty() {
            None
        } else {
            Some(ProjectionPrograms::prepare(
                context.images.clone(),
                &projection_records.into_iter().collect::<Vec<_>>(),
                init,
                cancelled,
            )?)
        };
        let mlp = if mlp_records.is_empty() {
            None
        } else {
            Some(MlpPrograms::prepare(
                context.images.clone(),
                &mlp_records.into_iter().collect::<Vec<_>>(),
                init,
                cancelled,
            )?)
        };
        let activations = if activation_records.is_empty() {
            None
        } else {
            Some(ActivationPrograms::prepare(
                context.images.clone(),
                &activation_records.into_iter().collect::<Vec<_>>(),
                init,
                cancelled,
            )?)
        };
        let prepared = Arc::new(Self {
            nodes,
            weights,
            workspace,
            projections,
            mlp,
            activations,
            context,
            _metadata: metadata,
        });
        init.retain_completion_owner(prepared.clone())?;
        not_cancelled(cancelled)?;
        Ok(prepared)
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.weights
            .values()
            .map(|weight| (weight.lease.clone(), weight._payload.packed.as_slice()))
            .collect()
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut values: BTreeMap<BufferLeaseId, (ResidentLease, DeviceAccess)> = BTreeMap::new();
        let mut add = |lease: &ResidentLease, access| {
            let prior = values
                .entry(lease.id())
                .or_insert_with(|| (lease.clone(), access));
            if access == DeviceAccess::Write {
                prior.1 = DeviceAccess::Write;
            }
        };
        for weight in self.weights.values() {
            add(&weight.lease, DeviceAccess::Read);
        }
        for node in self.nodes.values() {
            add(&node.input, DeviceAccess::Read);
            add(&node.output, DeviceAccess::Write);
        }
        if let Some(workspace) = &self.workspace {
            add(workspace, DeviceAccess::Write);
        }
        values.into_values().collect()
    }
    pub(crate) fn enqueue(
        &self,
        node_index: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<NativeSubmitted> {
        not_cancelled(cancelled)?;
        self.context.service.ensure_usable()?;
        let node = self
            .nodes
            .get(&node_index)
            .ok_or_else(|| Error::State("unprepared native plan node index".into()))?;
        let weight = node
            .weight
            .map(|key| {
                self.weights
                    .get(&key)
                    .ok_or_else(|| Error::State("native exact source W absent".into()))
            })
            .transpose()?;
        if let Some(weight) = weight
            && (!Arc::ptr_eq(weight.lease.service(), &self.context.service)
                || weight.lease.role() != BufferRole::ImmutableWeights
                || !weight.lease.is_initialized())
        {
            return Err(Error::State(
                "native W source upload is not completed".into(),
            ));
        }
        let (native_launches, workspace_clears) = match node.operation {
            NativeOperation::Projection(projection) => {
                let library = self
                    .projections
                    .as_ref()
                    .ok_or_else(|| Error::State("native projection library absent".into()))?;
                let weight =
                    weight.ok_or_else(|| Error::State("native projection W absent".into()))?;
                (
                    library.enqueue(
                        projection,
                        ProjectionBindings {
                            weights: &weight.lease,
                            input: &node.input,
                            output: &node.output,
                            workspace: self.workspace.as_ref(),
                        },
                        cancelled,
                    )?,
                    0,
                )
            }
            NativeOperation::Mlp(projection) => {
                let library = self
                    .mlp
                    .as_ref()
                    .ok_or_else(|| Error::State("native MLP library absent".into()))?;
                let weight = weight.ok_or_else(|| Error::State("native MLP W absent".into()))?;
                let actual = library.enqueue(
                    projection,
                    MlpBindings {
                        weights: &weight.lease,
                        input: &node.input,
                        output: &node.output,
                        workspace: self.workspace.as_ref(),
                    },
                    cancelled,
                )?;
                (actual.kernels, actual.clears)
            }
            NativeOperation::Activation(activation) => {
                let library = self
                    .activations
                    .as_ref()
                    .ok_or_else(|| Error::State("native ACT library absent".into()))?;
                (
                    library.enqueue(activation, &node.input, &node.output, cancelled)?,
                    0,
                )
            }
        };
        Ok(NativeSubmitted {
            dispatches: native_launches + workspace_clears,
            native_launches,
            workspace_clears,
        })
    }
    /// Caller proves completion and retains this owner in the original armed
    /// immutable-W flight. The body retires the shared set exactly once later.
    pub(crate) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        if let Some(p) = &self.projections {
            unsafe { p.retire_kernels_after_physical_completion() }?;
        }
        if let Some(p) = &self.mlp {
            unsafe { p.retire_kernels_after_physical_completion() }?;
        }
        if let Some(p) = &self.activations {
            unsafe { p.retire_kernels_after_physical_completion() }?;
        }
        Ok(())
    }
}

#[path = "native_plan/phases.rs"]
mod phases;
pub(crate) use phases::NativePlanPhase;

#[cfg(test)]
#[path = "native_plan/tests.rs"]
mod tests;
