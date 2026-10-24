//! Explicit projection leaf inputs; never operands for the connected layer.
use super::cubin_manifests::{Projection, ProjectionImage, ProjectionImageSource, WORKSPACE_BYTES};
use super::layer_common::projection_weight;
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::ResidentReadback;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::TensorWindow;

mod faults;
pub(super) mod resident;
pub use faults::{
    CubinProjectionFault, CubinProjectionFaultOutcome, qualify_projection_cubin_fault,
};
use resident::{AdmittedImages, ProjectionBindings};
#[cfg(test)]
mod tests;

const HOST_BYTES: usize = 32 * 1024 * 1024;
const FOREIGN_BYTES: usize = 64 * 1024 * 1024;

/// Explicit authentic observed input, only for this separately labelled leaf.
pub struct CubinProjectionFixture {
    pub projection: Projection,
    pub input: Vec<u8>,
    pub expected: Vec<u8>,
}

/// Original source must admit FULL Vec capacity before its allocation.
pub struct CubinImage {
    pub kind: ProjectionImage,
    pub bytes: Vec<u8>,
    pub admission: WindowWorkspace,
}

/// Separately labelled selected activation resource; no loader-input claim.
pub struct CubinActivationImage {
    pub bytes: Vec<u8>,
    pub admission: WindowWorkspace,
}

pub(super) fn validate_activation_image(
    source: &VerifiedTensorSource,
    image: &CubinActivationImage,
) -> Result<()> {
    let record = cubecl::cuda::PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86.capture();
    if record.image_bytes != super::activation_manifest::IMAGE_BYTES
        || record.image_sha256 != super::activation_manifest::IMAGE_SHA256
        || image.bytes.get(..64) != Some(super::activation_manifest::IMAGE_HEADER.as_slice())
        || !image.admission.source().same_issuer(source)
        || image.bytes.len() != record.image_bytes
        || image.bytes.capacity() > HOST_BYTES
        || image.admission.bytes() < image.bytes.capacity()
    {
        return Err(Error::Malformed(
            "closed activation resource issuer/full capacity admission".into(),
        ));
    }
    Ok(())
}

pub struct CubinQualificationInputs {
    pub images: Vec<CubinImage>,
    pub cublas: PathBuf,
    pub cublas_lt: PathBuf,
    pub device_cap: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CubinProjectionComparison {
    pub calls: usize,
    pub native_launches: usize,
    pub bf16_words: usize,
}

struct LeafOwner {
    _source: Arc<VerifiedTensorSource>,
    _windows: Vec<Arc<TensorWindow>>,
    weights: BTreeMap<usize, Vec<u8>>,
    fixtures: Vec<CubinProjectionFixture>,
    // Actual payload allocations drop before these original host charges.
    _host: [WindowWorkspace; 4],
}
struct Job {
    projection: Projection,
    input: ResidentLease,
    output: ResidentLease,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    reads: Vec<ResidentReadback>,
}

pub(super) fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "closed original cubin qualification cancelled".into(),
        ));
    }
    Ok(())
}

pub(super) fn native_error(error: impl std::fmt::Display) -> Error {
    Error::State(format!("closed original projection cubin: {error}"))
}

pub(super) fn required_images(projections: &[Projection]) -> Result<BTreeSet<ProjectionImage>> {
    let required = projections
        .iter()
        .flat_map(|p| p.launches())
        .map(|launch| launch.image())
        .collect::<BTreeSet<_>>();
    require_image_sources(&required)?;
    Ok(required)
}

fn require_image_sources(required: &BTreeSet<ProjectionImage>) -> Result<()> {
    if required.is_empty() || required.len() > 6 {
        return Err(Error::Malformed("exact closed projection image set".into()));
    }
    require_registered_image_sources(required)
}

fn require_registered_image_sources(required: &BTreeSet<ProjectionImage>) -> Result<()> {
    // The original source category is closed; captured code never claims
    // retained-DSO membership or a reproducible provider derivation.
    for image in required {
        let (bytes, sha256, binary_sm) = match image.provenance() {
            ProjectionImageSource::Derived(pinned) => {
                let approved = pinned.derivation();
                (
                    approved.image_bytes,
                    approved.image_sha256,
                    approved.binary_sm,
                )
            }
            ProjectionImageSource::Captured(pinned) => {
                let approved = pinned.capture();
                (
                    approved.image_bytes,
                    approved.image_sha256,
                    approved.binary_sm,
                )
            }
        };
        if image.bytes() != bytes || image.sha256() != sha256 || image.binary_sm() != binary_sm {
            return Err(Error::State(
                "closed observed image differs from original source registry".into(),
            ));
        }
    }
    Ok(())
}

fn require_canonical47_qkkv_image_sources(required: &BTreeSet<ProjectionImage>) -> Result<()> {
    let mut exact = super::canonical47_mlp::required_images();
    exact.extend([
        ProjectionImage::AttentionAmpere256x128Sm80,
        ProjectionImage::AttentionQkSm86,
        ProjectionImage::AttentionKvSm86,
    ]);
    if *required != exact || required.len() != 7 {
        return Err(Error::Malformed(
            "exact canonical47 seven-image QK/KV union".into(),
        ));
    }
    require_registered_image_sources(required)
}

pub(super) fn validate_canonical47_qkkv_image_set(
    source: &VerifiedTensorSource,
    required: &BTreeSet<ProjectionImage>,
    images: &[CubinImage],
) -> Result<()> {
    require_canonical47_qkkv_image_sources(required)?;
    validate_image_payloads(source, required, images)
}

fn require_canonical47_fa_image_sources(required: &BTreeSet<ProjectionImage>) -> Result<()> {
    let mut exact = super::canonical47_mlp::required_images();
    exact.extend([
        ProjectionImage::AttentionAmpere256x128Sm80,
        ProjectionImage::AttentionQkSm86,
        ProjectionImage::AttentionKvSm86,
        ProjectionImage::AttentionFaCausalCombineSm80,
    ]);
    if *required != exact || required.len() != 8 {
        return Err(Error::Malformed(
            "exact canonical47 eight-image FA union".into(),
        ));
    }
    require_registered_image_sources(required)
}
pub(super) fn validate_canonical47_fa_image_set(
    source: &VerifiedTensorSource,
    required: &BTreeSet<ProjectionImage>,
    images: &[CubinImage],
) -> Result<()> {
    require_canonical47_fa_image_sources(required)?;
    validate_image_payloads(source, required, images)
}

fn require_canonical47_layer3_image_sources(required: &BTreeSet<ProjectionImage>) -> Result<()> {
    let mut exact = super::canonical47_mlp::required_images();
    exact.extend([
        ProjectionImage::AttentionAmpere256x128Sm80,
        ProjectionImage::AttentionQkSm86,
        ProjectionImage::AttentionKvSm86,
        ProjectionImage::AttentionFaCausalCombineSm80,
        ProjectionImage::AttentionSigmoidSm86,
        ProjectionImage::AttentionMultiplySm86,
    ]);
    if *required != exact || required.len() != 10 {
        return Err(Error::Malformed(
            "exact canonical47 ten-image layer3 union".into(),
        ));
    }
    require_registered_image_sources(required)
}
pub(super) fn validate_canonical47_layer3_image_set(
    source: &VerifiedTensorSource,
    required: &BTreeSet<ProjectionImage>,
    images: &[CubinImage],
) -> Result<()> {
    require_canonical47_layer3_image_sources(required)?;
    validate_image_payloads(source, required, images)
}
pub(super) fn validate_canonical47_full_image_set(
    source: &VerifiedTensorSource,
    required: &BTreeSet<ProjectionImage>,
    images: &[CubinImage],
) -> Result<()> {
    let mut tail = required.clone();
    if !tail.remove(&ProjectionImage::Bf16Gemv2TSm86) || required.len() != 11 {
        return Err(Error::Malformed(
            "exact canonical47 eleven-image full union".into(),
        ));
    }
    require_canonical47_layer3_image_sources(&tail)?;
    require_registered_image_sources(required)?;
    validate_image_payloads(source, required, images)
}

pub(super) fn validate_images(
    source: &VerifiedTensorSource,
    projections: &[Projection],
    images: &[CubinImage],
) -> Result<()> {
    let required = required_images(projections)?;
    validate_image_set(source, &required, images)
}

pub(super) fn validate_image_set(
    source: &VerifiedTensorSource,
    required: &BTreeSet<ProjectionImage>,
    images: &[CubinImage],
) -> Result<()> {
    require_image_sources(required)?;
    validate_image_payloads(source, required, images)
}

pub(super) fn validate_short_hf_image_set(
    source: &VerifiedTensorSource,
    profile: zerogravity_core::accelerator::program::FullModelProfile,
    cohort_b4: bool,
    images: &[CubinImage],
) -> Result<()> {
    let required = super::short_hf_native::short_hf_required_images_for_cohort(profile, cohort_b4)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    require_registered_image_sources(&required)?;
    validate_image_payloads(source, &required, images)
}

fn validate_image_payloads(
    source: &VerifiedTensorSource,
    required: &BTreeSet<ProjectionImage>,
    images: &[CubinImage],
) -> Result<()> {
    if images.len() != required.len() {
        return Err(Error::Malformed("exact closed projection image set".into()));
    }
    let mut actual = BTreeSet::new();
    let mut total = 0usize;
    for image in images {
        if !actual.insert(image.kind)
            || !image.admission.source().same_issuer(source)
            || image.bytes.len() != image.kind.bytes()
            || image.bytes.capacity() > HOST_BYTES
            || image.admission.bytes() < image.bytes.capacity()
        {
            return Err(Error::Malformed(
                "original cubin unique issuer/full capacity admission".into(),
            ));
        }
        total = total
            .checked_add(image.bytes.capacity())
            .ok_or_else(|| Error::Limit("closed cubin image capacity overflow".into()))?;
    }
    if actual != *required || total > 128 * 1024 * 1024 {
        return Err(Error::Limit(
            "closed original cubin total image capacity".into(),
        ));
    }
    Ok(())
}

fn validate_fixtures(fixtures: &[CubinProjectionFixture]) -> Result<()> {
    if fixtures.is_empty() || fixtures.len() > 15 {
        return Err(Error::Limit(
            "closed original projection fixture count1..15".into(),
        ));
    }
    let mut retained = 0usize;
    for fixture in fixtures {
        if fixture.input.len() != fixture.projection.input_bytes()
            || fixture.expected.len() != fixture.projection.output_bytes()
            || !finite_bf16(&fixture.input)
            || !finite_bf16(&fixture.expected)
        {
            return Err(Error::Malformed(
                "complete finite original projection leaf operands".into(),
            ));
        }
        retained = retained
            .checked_add(fixture.input.capacity())
            .and_then(|n| n.checked_add(fixture.expected.capacity()))
            .ok_or_else(|| Error::Limit("closed projection fixture capacity overflow".into()))?;
    }
    if retained > HOST_BYTES {
        return Err(Error::Limit(
            "closed projection fixture capacity exceeds original host admission".into(),
        ));
    }
    Ok(())
}

/// Independently qualify any bounded subset of the closed observed operations.
/// Inputs are explicit observed leaf operands; expectations are postflight only.
/// Unknown-image operations refuse before host workspace/device/native work.
pub async fn qualify_projection_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinProjectionFixture>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinProjectionComparison> {
    match qualify_projection_inner(device, source, fixtures, inputs, Mode::Compare, cancelled)
        .await?
    {
        Outcome::Compared(compared) => Ok(compared),
        Outcome::Fault(_) => Err(Error::State(
            "projection comparison returned fault outcome".into(),
        )),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Compare,
    Fault(CubinProjectionFault),
}
enum Outcome {
    Compared(CubinProjectionComparison),
    Fault(CubinProjectionFaultOutcome),
}

async fn qualify_projection_inner(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinProjectionFixture>,
    inputs: CubinQualificationInputs,
    mode: Mode,
    cancelled: Option<&AtomicBool>,
) -> Result<Outcome> {
    // FIRST operation: cancellation precedes malformed fixtures/workspaces/CUDA.
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for native projection leaf".into(),
        ));
    }
    validate_fixtures(&fixtures)?;
    if matches!(mode, Mode::Fault(_)) {
        faults::validate_fault_fixtures(&fixtures)?;
    }
    if fixtures.capacity() > 16 || inputs.images.capacity() > 8 {
        return Err(Error::Limit(
            "closed original fixture/image metadata capacity".into(),
        ));
    }
    let projections = fixtures
        .iter()
        .map(|f| f.projection)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    validate_images(&source, &projections, &inputs.images)?;
    let host = [
        source.reserve_workspace(HOST_BYTES)?,
        source.reserve_workspace(HOST_BYTES)?,
        source.reserve_workspace(HOST_BYTES)?,
        source.reserve_workspace(HOST_BYTES)?,
    ];
    let mut windows = Vec::new();
    let mut weights = BTreeMap::new();
    for index in projections
        .iter()
        .map(|p| p.weight_index())
        .collect::<BTreeSet<_>>()
    {
        not_cancelled(cancelled)?;
        let (source_windows, bytes) = projection_weight(&source, index)?;
        windows.extend(source_windows);
        weights.insert(index, bytes);
    }
    let source = Arc::new(source);
    let owner = Arc::new(LeafOwner {
        _source: source.clone(),
        _windows: windows,
        weights,
        fixtures,
        _host: host,
    });
    not_cancelled(cancelled)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("vllm08-cuda-cubin", "original-projection-leaf")?,
        PROVIDER_CAP,
        inputs.device_cap,
    )?;
    let key = service.physical_key();
    if mode == Mode::Fault(CubinProjectionFault::AdmissionPressure) {
        let witness = faults::admission_pressure(&source, &owner, &service)?;
        drop((inputs, owner, source, service));
        return faults::pressure_retired(witness).map(Outcome::Fault);
    }
    let queue = service.pinned_cuda_queue()?.clone();
    let sources = queue
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    let admitted = AdmittedImages::reserve(&service, owner.clone(), inputs.images)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: FOREIGN_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let mut init = service.begin_initialization(100)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let mut weights = BTreeMap::new();
    for (&index, raw) in &owner.weights {
        not_cancelled(cancelled)?;
        weights.insert(index, init.upload_bytes(raw, BufferRole::ImmutableWeights)?);
    }
    let uses_workspace = projections.iter().any(|p| {
        p.launches().iter().any(|launch| {
            launch
                .bindings()
                .contains(&super::cubin_manifests::ProjectionBinding::Workspace)
        })
    });
    let workspace = if uses_workspace {
        Some(init.allocate_empty(WORKSPACE_BYTES, BufferRole::Scratch)?)
    } else {
        None
    };
    let mut jobs = Vec::with_capacity(owner.fixtures.len());
    for fixture in &owner.fixtures {
        not_cancelled(cancelled)?;
        let input = init.upload_bytes(&fixture.input, BufferRole::Activation)?;
        let output = init.allocate_empty(fixture.projection.output_bytes(), BufferRole::Output)?;
        let mut accesses = vec![
            (
                weights[&fixture.projection.weight_index()].clone(),
                DeviceAccess::Read,
            ),
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ];
        if fixture.projection.launches().iter().any(|launch| {
            launch
                .bindings()
                .contains(&super::cubin_manifests::ProjectionBinding::Workspace)
        }) {
            accesses.push((
                workspace
                    .as_ref()
                    .ok_or_else(|| Error::State("closed splitK original workspace absent".into()))?
                    .clone(),
                DeviceAccess::Write,
            ));
        }
        let reads = if mode == Mode::Fault(CubinProjectionFault::UnknownAfterNative) {
            Vec::new()
        } else {
            vec![
                input.readback(0, input.bytes())?,
                output.readback(0, output.bytes())?,
            ]
        };
        let job = Arc::new(Job {
            projection: fixture.projection,
            input,
            output,
            accesses,
            reads,
        });
        init.retain_completion_owner(job.clone())?;
        jobs.push(job);
    }
    let programs = admitted.prepare(sources.clone(), &projections, &mut init, cancelled)?;
    drop(init.complete_sync().await?);
    slot.finish()?;
    let mut compared = CubinProjectionComparison::default();
    let mut index = 0;
    while let Some(job) = jobs.get(index).cloned() {
        let abandoned = if mode == Mode::Fault(CubinProjectionFault::UnknownAfterNative) {
            Some(faults::UnknownWitness::new(
                &source, &owner, &job, &programs, &service,
            )?)
        } else {
            None
        };
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: FOREIGN_BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(200 + index as u64, &job.accesses)?;
        flight.retain_completion_owner(programs.clone())?;
        flight.retain_completion_owner(job.clone())?;
        flight.retain_completion_owner(owner.clone())?;
        not_cancelled(cancelled)?;
        let launches = programs.enqueue(
            job.projection,
            ProjectionBindings {
                weights: &weights[&job.projection.weight_index()],
                input: &job.input,
                output: &job.output,
                workspace: workspace.as_ref(),
            },
            cancelled,
        )?;
        if let Some(witness) = abandoned {
            // No producer comparison/Ready/barrier/native retirement. Original
            // armed drops retain the actual submitted owners until process exit.
            drop(flight);
            drop(slot);
            drop((
                job, jobs, weights, workspace, programs, sources, queue, owner, source, service,
            ));
            return witness.inspect(launches).map(Outcome::Fault);
        }
        let (ready, raw) = flight
            .complete_with_readback(std::slice::from_ref(&job.output), &job.reads, |raw| {
                let fixture = &owner.fixtures[index];
                if raw.len() != 2
                    || &raw[0][..] != fixture.input.as_slice()
                    || &raw[1][..] != fixture.expected.as_slice()
                    || !finite_bf16(&raw[1])
                {
                    return Err(Error::State(format!(
                        "original projection leaf{index} {:?} input/output differs",
                        job.projection
                    )));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        compared.calls += 1;
        compared.native_launches += launches;
        compared.bf16_words += job.output.bytes() / 2;
        drop((ready, raw));
        index += 1;
    }
    let weight_accesses = weights
        .values()
        .map(|w| (w.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let reads = weights
        .values()
        .map(|w| w.readback(0, w.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: FOREIGN_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(999, &weight_accesses)?;
    flight.retain_completion_owner(programs.clone())?;
    not_cancelled(cancelled)?;
    // SAFETY: all original launches physically completed. This same original
    // read-only flight retains every admission through unload/error quarantine.
    unsafe { programs.retire_after_physical_completion() }.map_err(native_error)?;
    let raw = flight
        .complete_read_only_readback(&reads, |raw| {
            if raw.len() != owner.weights.len()
                || raw
                    .iter()
                    .zip(owner.weights.values())
                    .any(|(a, b)| &a[..] != b.as_slice())
            {
                return Err(Error::State("original projection weights changed".into()));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop((
        raw,
        reads,
        weight_accesses,
        jobs,
        weights,
        workspace,
        programs,
        sources,
        queue,
        owner,
        source,
    ));
    service.ensure_usable()?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "original projection buffers/events/native admission not retired".into(),
        ));
    }
    drop(service);
    let retired = ResourceArbiter::process_shared().snapshot(&key)?;
    if retired.reserved != 0 || retired.fenced {
        return Err(Error::State(
            "original projection envelope not retired".into(),
        ));
    }
    Ok(Outcome::Compared(compared))
}

pub(super) fn validate_canonical47_natural_image_set(
    source: &VerifiedTensorSource,
    required: &BTreeSet<ProjectionImage>,
    images: &[CubinImage],
) -> Result<()> {
    let mut exact = super::canonical47_model_native::head::required_images();
    exact.extend([
        ProjectionImage::CutlassWmmaSm80,
        ProjectionImage::Bf16GemvxSm86,
        ProjectionImage::AttentionFaSplitSm80,
    ]);
    if required != &exact || required.len() != 14 {
        return Err(Error::Malformed(
            "canonical exact fourteen source images plus activation".into(),
        ));
    }
    require_registered_image_sources(required)?;
    validate_image_payloads(source, required, images)
}
