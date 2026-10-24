//! Captured MLP leaf operands only; connected callers use computed leases.
use super::cubin_manifests::{ProjectionBinding, ProjectionImage, WORKSPACE_BYTES};
use super::mlp_manifests::{MlpOperation, MlpProjection, MlpRole};
use super::projection_cubin::resident::AdmittedImages;
use super::projection_cubin::{
    CubinQualificationInputs, native_error, not_cancelled, validate_image_set,
};
use super::*;
use crate::resident_leases::ResidentReadback;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::source_weights::TensorWindow;
use zerogravity_core::accelerator::program::MatrixAxes;

mod faults;
pub use faults::{CubinMlpFault, CubinMlpFaultOutcome, qualify_mlp_cubin_fault};
pub(super) mod resident;
use resident::{MlpBindings, MlpPrograms};
#[cfg(test)]
mod tests;

const HOST_BYTES: usize = 32 * 1024 * 1024;
const FOREIGN_BYTES: usize = 64 * 1024 * 1024;

/// Explicit observed operands grant no connected producer or model authority.
pub struct CubinMlpFixture {
    pub projection: MlpProjection,
    pub input: Vec<u8>,
    pub expected: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CubinMlpComparison {
    pub calls: usize,
    pub native_launches: usize,
    pub workspace_clears: usize,
    pub bf16_words: usize,
}

struct LeafOwner {
    _source: Arc<VerifiedTensorSource>,
    _windows: Vec<Arc<TensorWindow>>,
    weights: BTreeMap<usize, Vec<u8>>,
    fixtures: Vec<CubinMlpFixture>,
    _host: [WindowWorkspace; 4],
}
struct Job {
    projection: MlpProjection,
    input: ResidentLease,
    output: ResidentLease,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    reads: [ResidentReadback; 2],
}

fn weight_parts(role: MlpRole) -> &'static [(&'static str, usize, usize)] {
    match role {
        MlpRole::GateUp => &[("gate_proj", 3584, 1024), ("up_proj", 3584, 1024)],
        MlpRole::Down => &[("down_proj", 1024, 3584)],
    }
}

/// Original TP1 row concatenation, with each complete original window retained.
pub(super) fn mlp_weight(
    source: &VerifiedTensorSource,
    projection: MlpProjection,
) -> Result<(Vec<Arc<TensorWindow>>, Vec<u8>)> {
    let parts = weight_parts(projection.role());
    let mut windows = Vec::with_capacity(parts.len());
    let mut packed = Vec::with_capacity(projection.weight_bytes());
    for &(suffix, rows, columns) in parts {
        let name = format!("model.language_model.layers.0.mlp.{suffix}.weight");
        let matrix = source.matrix(&name, MatrixAxes::OutputInput)?;
        if matrix.rows() != rows
            || matrix.columns() != columns
            || matrix.tensor().kind != QuantKind::Bf16
        {
            return Err(Error::Malformed("original MLP BF16 matrix geometry".into()));
        }
        let window = Arc::new(source.read_rows(&matrix, 0..rows)?);
        if window.as_slice().len() != rows * columns * 2 || !finite_bf16(window.as_slice()) {
            return Err(Error::Malformed("complete finite original MLP rows".into()));
        }
        packed.extend_from_slice(window.as_slice());
        windows.push(window);
    }
    if packed.len() != projection.weight_bytes() {
        return Err(Error::State("complete original MLP packed matrix".into()));
    }
    Ok((windows, packed))
}

fn required_images(projections: &[MlpProjection]) -> BTreeSet<ProjectionImage> {
    projections
        .iter()
        .flat_map(|projection| projection.operations())
        .filter_map(|operation| match operation {
            MlpOperation::Kernel(launch) => Some(launch.image()),
            MlpOperation::Clear(_) => None,
        })
        .collect()
}

fn uses_workspace(projection: MlpProjection) -> bool {
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

fn validate_fixtures(fixtures: &[CubinMlpFixture]) -> Result<()> {
    if fixtures.is_empty() || fixtures.len() > 10 {
        return Err(Error::Limit(
            "closed original MLP fixture count1..10".into(),
        ));
    }
    let mut capacity = 0usize;
    for fixture in fixtures {
        let band = match fixture.projection.rows() {
            1 => MlpProjection::for_phase(VllmGdnPhase::InitialOne)?,
            2 => MlpProjection::for_phase(VllmGdnPhase::ContinuedTwo)?,
            64 => MlpProjection::for_phase(VllmGdnPhase::Initial64)?,
            _ => return Err(Error::Unsupported("closed MLP row band".into())),
        };
        if !band.contains(&fixture.projection)
            || fixture.input.len() != fixture.projection.input_bytes()
            || fixture.expected.len() != fixture.projection.output_bytes()
            || !finite_bf16(&fixture.input)
            || !finite_bf16(&fixture.expected)
        {
            return Err(Error::Malformed("complete finite MLP leaf operands".into()));
        }
        capacity = capacity
            .checked_add(fixture.input.capacity())
            .and_then(|bytes| bytes.checked_add(fixture.expected.capacity()))
            .ok_or_else(|| Error::Limit("MLP fixture capacity overflow".into()))?;
    }
    if capacity > HOST_BYTES {
        return Err(Error::Limit(
            "MLP fixture capacity exceeds host admission".into(),
        ));
    }
    Ok(())
}

/// Explicit leaf comparisons only; expected bytes are examined after arithmetic.
pub async fn qualify_mlp_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinMlpFixture>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinMlpComparison> {
    match qualify_mlp_inner(device, source, fixtures, inputs, Mode::Compare, cancelled).await? {
        Outcome::Compared(compared) => Ok(compared),
        Outcome::Fault(_) => Err(Error::State("MLP comparison returned fault outcome".into())),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Compare,
    Fault(CubinMlpFault),
}
enum Outcome {
    Compared(CubinMlpComparison),
    Fault(Box<CubinMlpFaultOutcome>),
}

async fn qualify_mlp_inner(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinMlpFixture>,
    inputs: CubinQualificationInputs,
    mode: Mode,
    cancelled: Option<&AtomicBool>,
) -> Result<Outcome> {
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for MLP leaf".into(),
        ));
    }
    validate_fixtures(&fixtures)?;
    if matches!(mode, Mode::Fault(_)) {
        faults::validate_scope(&fixtures)?;
    }
    if fixtures.capacity() > 16 || inputs.images.capacity() > 8 {
        return Err(Error::Limit(
            "closed MLP fixture/image metadata capacity".into(),
        ));
    }
    let projections = fixtures
        .iter()
        .map(|fixture| fixture.projection)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    validate_image_set(&source, &required_images(&projections), &inputs.images)?;
    let host = [
        source.reserve_workspace(HOST_BYTES)?,
        source.reserve_workspace(HOST_BYTES)?,
        source.reserve_workspace(HOST_BYTES)?,
        source.reserve_workspace(HOST_BYTES)?,
    ];
    let mut windows = Vec::new();
    let mut weights = BTreeMap::new();
    for &projection in &projections {
        if weights.contains_key(&projection.weight_index()) {
            continue;
        }
        not_cancelled(cancelled)?;
        let (original, packed) = mlp_weight(&source, projection)?;
        windows.extend(original);
        weights.insert(projection.weight_index(), packed);
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
        ResourceOwner::new("vllm08-cuda-cubin", "original-MLP-leaf")?,
        PROVIDER_CAP,
        inputs.device_cap,
    )?;
    let key = service.physical_key();
    if mode == Mode::Fault(CubinMlpFault::Pressure) {
        let witness = faults::admission_pressure(&source, &owner, &service)?;
        drop((inputs, owner, source, service));
        return faults::pressure_retired(witness).map(|outcome| Outcome::Fault(Box::new(outcome)));
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
    let workspace = if projections.iter().copied().any(uses_workspace) {
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
        if uses_workspace(fixture.projection) {
            accesses.push((
                workspace
                    .as_ref()
                    .ok_or_else(|| Error::State("original MLP workspace absent".into()))?
                    .clone(),
                DeviceAccess::Write,
            ));
        }
        let reads = [
            input.readback(0, input.bytes())?,
            output.readback(0, output.bytes())?,
        ];
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
    let images = admitted.prepare_images(sources.clone(), &mut init, cancelled)?;
    let programs = MlpPrograms::prepare(images, &projections, &mut init, cancelled)?;
    drop(init.complete_sync().await?);
    slot.finish()?;
    let mut compared = CubinMlpComparison::default();
    let mut index = 0;
    while let Some(job) = jobs.get(index).cloned() {
        let abandoned = if mode == Mode::Fault(CubinMlpFault::Unknown) {
            Some(faults::UnknownWitness::new(
                &source,
                &owner,
                &job,
                workspace
                    .as_ref()
                    .ok_or_else(|| Error::State("MLP fault workspace absent".into()))?,
                &programs,
                &service,
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
        let submitted = programs.enqueue(
            job.projection,
            MlpBindings {
                weights: &weights[&job.projection.weight_index()],
                input: &job.input,
                output: &job.output,
                workspace: workspace.as_ref(),
            },
            cancelled,
        )?;
        if let Some(witness) = abandoned {
            // The real Down64 prefix and kernel were enqueued; no readback,
            // comparison, Ready, barrier or native retirement follows.
            drop(flight);
            drop(slot);
            drop((
                job, jobs, weights, workspace, programs, sources, queue, owner, source, service,
            ));
            return witness
                .inspect(submitted)
                .map(|outcome| Outcome::Fault(Box::new(outcome)));
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
                        "original MLP leaf{index} {:?} input/output differs",
                        job.projection
                    )));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        compared.calls += 1;
        compared.native_launches += submitted.kernels;
        compared.workspace_clears += submitted.clears;
        compared.bf16_words += job.output.bytes() / 2;
        drop((ready, raw));
        index += 1;
    }
    let accesses = weights
        .values()
        .map(|lease| (lease.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let reads = weights
        .values()
        .map(|lease| lease.readback(0, lease.bytes()))
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
    let mut flight = service.submit(999, &accesses)?;
    flight.retain_completion_owner(programs.clone())?;
    not_cancelled(cancelled)?;
    // SAFETY: every producer completed; original armed W flight retains all
    // native/source admission through unload and uncertain-error quarantine.
    unsafe { programs.retire_after_physical_completion() }.map_err(native_error)?;
    let raw = flight
        .complete_read_only_readback(&reads, |raw| {
            if raw.len() != owner.weights.len()
                || raw
                    .iter()
                    .zip(owner.weights.values())
                    .any(|(actual, original)| &actual[..] != original.as_slice())
            {
                return Err(Error::State("original MLP weights changed".into()));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop((
        raw, reads, accesses, jobs, weights, workspace, programs, sources, queue, owner, source,
    ));
    service.ensure_usable()?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "original MLP buffers/events/native charge not retired".into(),
        ));
    }
    drop(service);
    let retired = ResourceArbiter::process_shared().snapshot(&key)?;
    if retired.reserved != 0 || retired.fenced {
        return Err(Error::State("original MLP envelope not retired".into()));
    }
    Ok(Outcome::Compared(compared))
}
