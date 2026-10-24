//! Explicit observed activation leaves, never connected-model operands.
use super::activation_manifest::Activation;
use super::projection_cubin::resident::AdmittedImages;
use super::projection_cubin::{
    CubinActivationImage, native_error, not_cancelled, validate_activation_image,
};
use super::*;
use crate::resident_leases::ResidentReadback;
use std::{collections::BTreeSet, path::PathBuf, sync::atomic::AtomicBool};

mod faults;
pub use faults::{
    CubinActivationFault, CubinActivationFaultOutcome, qualify_activation_cubin_fault,
};
pub(super) mod resident;
use resident::ActivationPrograms;

pub struct CubinActivationFixture {
    pub activation: Activation,
    pub input: Vec<u8>,
    pub expected: Vec<u8>,
}
pub struct CubinActivationInputs {
    pub image: CubinActivationImage,
    pub cublas: PathBuf,
    pub cublas_lt: PathBuf,
    pub device_cap: u64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct CubinActivationComparison {
    pub calls: usize,
    pub native_launches: usize,
    pub bf16_words: usize,
}
struct ActivationOwner {
    _source: Arc<VerifiedTensorSource>,
    fixtures: Vec<CubinActivationFixture>,
    _host: WindowWorkspace,
}
struct Job {
    activation: Activation,
    input: ResidentLease,
    output: ResidentLease,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    reads: Vec<ResidentReadback>,
}
fn validate_fixtures(fixtures: &[CubinActivationFixture]) -> Result<()> {
    if fixtures.len() != 5 {
        return Err(Error::Limit("closed activation leaf count/capacity".into()));
    }
    let mut bytes = 0usize;
    let phases = [
        VllmGdnPhase::InitialOne,
        VllmGdnPhase::CachedDecode,
        VllmGdnPhase::Initial64,
        VllmGdnPhase::ContinuedTwo,
        VllmGdnPhase::CachedDecode,
    ];
    for (fixture, phase) in fixtures.iter().zip(phases) {
        let activation = Activation::for_phase(phase)?;
        if fixture.activation != activation
            || fixture.input.len() != activation.input_bytes()
            || fixture.expected.len() != activation.output_bytes()
            || !finite_bf16(&fixture.input)
            || !finite_bf16(&fixture.expected)
        {
            return Err(Error::Malformed(
                "complete finite activation leaf operands".into(),
            ));
        }
        bytes = bytes
            .checked_add(fixture.input.capacity())
            .and_then(|v| v.checked_add(fixture.expected.capacity()))
            .ok_or_else(|| Error::Limit("activation fixture capacity overflow".into()))?;
    }
    if bytes > 8 * 1024 * 1024 {
        return Err(Error::Limit(
            "activation leaf original host envelope".into(),
        ));
    }
    Ok(())
}

/// Actual captured operands qualify only this independent activation leaf.
pub async fn qualify_activation_cubins(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinActivationFixture>,
    inputs: CubinActivationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinActivationComparison> {
    match qualify_activation_inner(device, source, fixtures, inputs, Mode::Compare, cancelled)
        .await?
    {
        Outcome::Compared(compared) => Ok(compared),
        Outcome::Fault(_) => Err(Error::State(
            "activation comparison returned fault outcome".into(),
        )),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Compare,
    Fault(CubinActivationFault),
}
enum Outcome {
    Compared(CubinActivationComparison),
    Fault(Box<CubinActivationFaultOutcome>),
}

async fn qualify_activation_inner(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinActivationFixture>,
    inputs: CubinActivationInputs,
    mode: Mode,
    cancelled: Option<&AtomicBool>,
) -> Result<Outcome> {
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA activation leaf required".into(),
        ));
    }
    if fixtures.capacity() > 8 {
        return Err(Error::Limit(
            "closed activation fixture metadata capacity".into(),
        ));
    }
    validate_fixtures(&fixtures)?;
    validate_activation_image(&source, &inputs.image)?;
    let activations = fixtures
        .iter()
        .map(|f| f.activation)
        .collect::<BTreeSet<_>>();
    let source = Arc::new(source);
    let owner = Arc::new(ActivationOwner {
        _source: source.clone(),
        fixtures,
        _host: source.reserve_workspace(32 * 1024 * 1024)?,
    });
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("vllm08-cuda-cubin", "original-activation-leaf")?,
        PROVIDER_CAP,
        inputs.device_cap,
    )?;
    let key = service.physical_key();
    if mode == Mode::Fault(CubinActivationFault::Pressure) {
        let witness = faults::admission_pressure(&source, &owner, &service)?;
        drop((inputs, owner, source, service));
        return faults::pressure_retired(witness).map(|outcome| Outcome::Fault(Box::new(outcome)));
    }
    let queue = service.pinned_cuda_queue()?.clone();
    let sources = queue
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    let admitted = AdmittedImages::reserve_with_activation(
        &service,
        owner.clone(),
        Vec::new(),
        Some(inputs.image),
    )?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: 64 * 1024 * 1024,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let mut init = service.begin_initialization(100)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let mut jobs = Vec::with_capacity(owner.fixtures.len());
    for fixture in &owner.fixtures {
        not_cancelled(cancelled)?;
        let input = init.upload_bytes(&fixture.input, BufferRole::Activation)?;
        let activation = fixture.activation;
        let output = init.allocate_empty(activation.output_bytes(), BufferRole::Output)?;
        let job = Arc::new(Job {
            activation,
            accesses: vec![
                (input.clone(), DeviceAccess::Read),
                (output.clone(), DeviceAccess::Write),
            ],
            reads: vec![
                input.readback(0, input.bytes())?,
                output.readback(0, output.bytes())?,
            ],
            input,
            output,
        });
        init.retain_completion_owner(job.clone())?;
        jobs.push(job);
    }
    let images = admitted.prepare_images(sources.clone(), &mut init, cancelled)?;
    let programs = ActivationPrograms::prepare(
        images.clone(),
        &activations.into_iter().collect::<Vec<_>>(),
        &mut init,
        cancelled,
    )?;
    drop(init.complete_sync().await?);
    slot.finish()?;
    let mut compared = CubinActivationComparison::default();
    let mut index = 0;
    while let Some(job) = jobs.get(index).cloned() {
        let abandoned = if mode == Mode::Fault(CubinActivationFault::Unknown) {
            Some(faults::UnknownWitness::new(
                &source, &owner, &jobs, &programs, &service,
            )?)
        } else {
            None
        };
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: 64 * 1024 * 1024,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(200 + index as u64, &job.accesses)?;
        flight.retain_completion_owner(programs.clone())?;
        flight.retain_completion_owner(job.clone())?;
        flight.retain_completion_owner(owner.clone())?;
        let launches = programs.enqueue(job.activation, &job.input, &job.output, cancelled)?;
        if let Some(witness) = abandoned {
            // No readback, comparison, Ready, barrier or native retirement.
            drop(flight);
            drop(slot);
            drop((
                job, jobs, programs, images, sources, queue, owner, source, service,
            ));
            return witness
                .inspect(launches)
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
                        "original activation leaf{index} differs"
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
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: 64 * 1024 * 1024,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let job = &jobs[0];
    let mut flight = service.submit(999, &[(job.input.clone(), DeviceAccess::Read)])?;
    flight.retain_completion_owner(programs.clone())?;
    flight.retain_completion_owner(owner.clone())?;
    not_cancelled(cancelled)?;
    // SAFETY: every original kernel physically completed; this armed flight
    // retains the same module/image/admission graph through unload or error.
    unsafe { programs.retire_after_physical_completion() }.map_err(native_error)?;
    let read = job.input.readback(0, job.input.bytes())?;
    let raw = flight
        .complete_read_only_readback(&[read], |raw| {
            if raw.len() != 1 || &raw[0][..] != owner.fixtures[0].input.as_slice() {
                return Err(Error::State("activation retirement input changed".into()));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop((raw, jobs, programs, images, sources, queue, owner, source));
    service.ensure_usable()?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "activation original buffers/events/native owners not retired".into(),
        ));
    }
    drop(service);
    let retired = ResourceArbiter::process_shared().snapshot(&key)?;
    if retired.reserved != 0 || retired.fenced {
        return Err(Error::State(
            "activation original physical envelope not retired".into(),
        ));
    }
    Ok(Outcome::Compared(compared))
}
