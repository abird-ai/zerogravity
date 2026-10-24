//! Explicit authentic normalized input leaf; no actual model producer claim.
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::ResidentReadback;
use cubecl::Device;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

const MIB: usize = 1024 * 1024;
pub struct Qwen2bProjectionFixture {
    pub record: Arc<RecordedVllm08Model>,
    pub normalized: Vec<u8>,
    /// SAME source: admit before constructing this recipe/Vec payload cohort.
    pub admission: WindowWorkspace,
}
/// Borrowed only within the original trusted actual-completion callback.
pub struct Qwen2bProjectionReadback<'a> {
    pub case: usize,
    pub record: &'a Arc<RecordedVllm08Model>,
    pub normalized: &'a [u8],
    pub qkvz: &'a [u8],
    pub ba: &'a [u8],
}
#[derive(Debug, Default)]
pub struct Qwen2bProjectionReceipt {
    pub calls: usize,
    pub native_launches: usize,
    pub bf16_words: usize,
    pub immutable_weight_bytes: usize,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
    pub model_state_or_producer_qualified: bool,
}
pub(super) struct SourceOwner {
    records: Vec<Arc<RecordedVllm08Model>>,
    fixtures: Vec<Qwen2bProjectionFixture>,
    pub(super) bound: BoundFullModelSource,
    // Each source TensorWindow has its own actual complete-chunk ticket.
    _fixture_host: WindowWorkspace,
    _readback_host: [WindowWorkspace; 2],
}
pub(super) struct Job {
    operators: [Vllm08Operator; 2],
    pub(super) input: ResidentLease,
    pub(super) outputs: [ResidentLease; 2],
    pub(super) accesses: Vec<(ResidentLease, DeviceAccess)>,
    reads: Vec<ResidentReadback>,
}
pub(super) struct PreparedJobs {
    pub(super) jobs: Vec<Arc<Job>>,
    pub(super) workspace: ResidentLease,
}
fn validate(bound: &BoundFullModelSource, fixtures: &[Qwen2bProjectionFixture]) -> Result<()> {
    if bound.model_profile() != FullModelProfile::Qwen2B
        || fixtures.len() != 5
        || bound.source().source().staging_chunk_bytes() != MIB
    {
        return Err(Error::Unsupported(
            "closed original2B five-call projection leaf".into(),
        ));
    }
    use zerogravity_core::accelerator::program::VllmBf16Phase as P;
    let mut retained = 0usize;
    for (fixture, (phase, base, rows)) in fixtures.iter().zip([
        (P::InitialOne, 0, 1),
        (P::CachedOne, 1, 1),
        (P::Initial64, 0, 64),
        (P::ContinuedTwo, 64, 2),
        (P::CachedOne, 66, 1),
    ]) {
        let call = fixture.record.call();
        let payload_capacity = fixture.normalized.capacity();
        if !fixture
            .admission
            .source()
            .same_issuer(bound.source().source())
            || fixture.admission.bytes() < 4 * MIB + payload_capacity
            || fixture.admission.bytes() > 6 * MIB
        {
            return Err(Error::State(
                "original2B admitted source recipe/payload owner".into(),
            ));
        }
        if call.source() != bound.source().source_identity()
            || call.request().legacy_phase()? != phase
            || call.request().teachers().base() != base
            || call.rows() != rows
        {
            return Err(Error::State(
                "exact original2B leaf source/phase/span schedule".into(),
            ));
        }
        for operator in descriptors(&fixture.record)? {
            require_source(bound, &operator)?;
        }
        {
            let (raw, bytes) = (&fixture.normalized, rows * 4096);
            if raw.len() != bytes || raw.capacity() > 2 * MIB || !super::super::finite_bf16(raw) {
                return Err(Error::Malformed(
                    "complete finite explicit2B projection leaf bytes/capacity".into(),
                ));
            }
            retained = retained
                .checked_add(raw.capacity())
                .ok_or_else(|| Error::Limit("original2B retained leaf capacity overflow".into()))?;
        }
    }
    if retained > 2 * MIB {
        return Err(Error::Limit("original2B aggregate leaf capacity".into()));
    }
    Ok(())
}
/// Callers admit fixture/image capacities before constructing payload Vecs.
/// Normalized inputs are explicitly authentic captured leaf operands; connected
/// execution instead obtains them from source RMS in the original node loop.
/// The callback may read independent expected bytes only after real completion.
pub async fn qualify_qwen2b_projections<F>(
    device: Device,
    bound: BoundFullModelSource,
    fixtures: Vec<Qwen2bProjectionFixture>,
    inputs: CubinQualificationInputs,
    validate_actual: F,
    cancelled: Option<&AtomicBool>,
) -> Result<Qwen2bProjectionReceipt>
where
    F: FnMut(Qwen2bProjectionReadback<'_>) -> Result<()>,
{
    let mut probe = None;
    run(
        device,
        bound,
        fixtures,
        inputs,
        cancelled,
        RunHooks {
            fault: None,
            probe: &mut probe,
            validate_actual,
        },
    )
    .await
}
pub(super) struct RunHooks<'a, F> {
    pub(super) fault: Option<super::faults::Qwen2bProjectionFault>,
    pub(super) probe: &'a mut Option<super::faults::Probe>,
    pub(super) validate_actual: F,
}
pub(super) async fn run<F>(
    device: Device,
    bound: BoundFullModelSource,
    fixtures: Vec<Qwen2bProjectionFixture>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    hooks: RunHooks<'_, F>,
) -> Result<Qwen2bProjectionReceipt>
where
    F: FnMut(Qwen2bProjectionReadback<'_>) -> Result<()>,
{
    let RunHooks {
        fault,
        probe,
        mut validate_actual,
    } = hooks;
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA2B projection leaf".into(),
        ));
    }
    validate(&bound, &fixtures)?;
    if fixtures.capacity() > 8 || inputs.images.capacity() > 8 {
        return Err(Error::Limit(
            "original2B fixture/image control capacity".into(),
        ));
    }
    let fixture_host = bound.source().source().reserve_workspace(6 * MIB)?;
    let readback_host = [
        bound.source().source().reserve_workspace(32 * MIB)?,
        bound.source().source().reserve_workspace(32 * MIB)?,
    ];
    let records = fixtures
        .iter()
        .map(|fixture| fixture.record.clone())
        .collect::<Vec<_>>();
    let required = required_images(&records)?;
    validate_image_set(bound.source().source(), &required, &inputs.images)?;
    let owner = Arc::new(SourceOwner {
        records,
        fixtures,
        bound,
        _fixture_host: fixture_host,
        _readback_host: readback_host,
    });
    let service = ResidentService::new_pinned_cuda(
        device,
        owner.bound.source().source_identity(),
        ResourceOwner::new("vllm2b-cuda-cubin", "original-QKVZ-BA-leaf")?,
        128 * MIB,
        inputs.device_cap,
    )?;
    let key = service.physical_key();
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
            bytes: 64 * MIB as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(100)?;
    init.retain_completion_owner(owner.clone())?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let pending = PendingProjectionWeights::upload(
        owner.bound.clone(),
        &owner.records[0],
        &service,
        &mut init,
        cancelled,
    )?;
    drop(init.complete_sync().await?);
    let weights = pending.bind_initialized()?;
    // Real input uploads keep the final native-preparation initializer armed.
    let mut init = service.begin_initialization(101)?;
    init.retain_completion_owner(owner.clone())?;
    init.retain_completion_owner(weights.clone())?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let workspace = init.allocate_empty(WORKSPACE_BYTES, BufferRole::Scratch)?;
    let mut jobs = Vec::with_capacity(5);
    for fixture in &owner.fixtures {
        not_cancelled(cancelled)?;
        let operators = descriptors(&fixture.record)?;
        let input = init.upload_bytes(&fixture.normalized, BufferRole::Activation)?;
        let outputs = [
            init.allocate_empty(fixture.record.call().rows() * 16384, BufferRole::Output)?,
            init.allocate_empty(fixture.record.call().rows() * 64, BufferRole::Output)?,
        ];
        let mut accesses = weights.accesses();
        accesses.extend([
            (input.clone(), DeviceAccess::Read),
            (outputs[0].clone(), DeviceAccess::Write),
            (outputs[1].clone(), DeviceAccess::Write),
        ]);
        if fixture.record.call().rows() == 64 {
            accesses.push((workspace.clone(), DeviceAccess::Write));
        }
        let reads = [&input, &outputs[0], &outputs[1]]
            .into_iter()
            .map(|lease| lease.readback(0, lease.bytes()))
            .collect::<Result<Vec<_>>>()?;
        let job = Arc::new(Job {
            operators,
            input,
            outputs,
            accesses,
            reads,
        });
        init.retain_completion_owner(job.clone())?;
        jobs.push(job);
    }
    let jobs = Arc::new(PreparedJobs {
        jobs,
        workspace: workspace.clone(),
    });
    init.retain_completion_owner(jobs.clone())?;
    // ALL managed allocations precede the first native module load.
    let images = admitted.prepare_images(sources.clone(), &mut init, cancelled)?;
    let programs = ProjectionPrograms::prepare(
        images.clone(),
        weights.clone(),
        &owner.records,
        &mut init,
        cancelled,
    )?;
    drop(init.complete_sync().await?);
    slot.finish()?;
    let mut receipt = Qwen2bProjectionReceipt::default();
    for (index, job) in jobs.jobs.iter().enumerate() {
        not_cancelled(cancelled)?;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: 64 * MIB as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(200 + index as u64, &job.accesses)?;
        flight.retain_completion_owner(owner.clone())?;
        flight.retain_completion_owner(jobs.clone())?;
        flight.retain_completion_owner(programs.clone())?;
        if fault.is_some() {
            *probe = Some(super::faults::Probe::new(
                &owner, &jobs, &programs, &weights, &service,
            )?);
        }
        let mut native = programs.enqueue(
            &job.operators[0],
            &job.input,
            &job.outputs[0],
            &workspace,
            cancelled,
        )?;
        if let Some(fault) = fault {
            probe
                .as_mut()
                .ok_or_else(|| Error::State("actual2B projection witness absent".into()))?
                .native = native;
            return match fault {
                super::faults::Qwen2bProjectionFault::UnknownAfterQkvz => {
                    Err(Error::State(super::faults::UNKNOWN.into()))
                }
                super::faults::Qwen2bProjectionFault::CancelBeforeBa => {
                    let flag = AtomicBool::new(true);
                    programs.enqueue(
                        &job.operators[1],
                        &job.input,
                        &job.outputs[1],
                        &workspace,
                        Some(&flag),
                    )?;
                    Err(Error::State(
                        "original2B late cancel failed to refuse BA".into(),
                    ))
                }
                super::faults::Qwen2bProjectionFault::Pressure => {
                    Err(Error::State("2B pressure reached native work".into()))
                }
            };
        }
        native += programs.enqueue(
            &job.operators[1],
            &job.input,
            &job.outputs[1],
            &workspace,
            cancelled,
        )?;
        let (ready, raw) = flight
            .complete_with_readback(&job.outputs, &job.reads, |raw| {
                let fixture = &owner.fixtures[index];
                if raw.len() != 3
                    || &raw[0][..] != fixture.normalized.as_slice()
                    || raw[1].len() != fixture.record.call().rows() * 16384
                    || raw[2].len() != fixture.record.call().rows() * 64
                    || !super::super::finite_bf16(&raw[1])
                    || !super::super::finite_bf16(&raw[2])
                {
                    return Err(Error::State(format!(
                        "original2B QKVZ/BA leaf{index} actual bytes differ"
                    )));
                }
                validate_actual(Qwen2bProjectionReadback {
                    case: index,
                    record: &fixture.record,
                    normalized: &raw[0],
                    qkvz: &raw[1],
                    ba: &raw[2],
                })
            })
            .await?;
        slot.finish()?;
        receipt.calls += 2;
        receipt.native_launches += native;
        receipt.bf16_words += job
            .outputs
            .iter()
            .map(|lease| lease.bytes() / 2)
            .sum::<usize>();
        drop((ready, raw));
    }
    let accesses = weights.accesses();
    let reads = weights.reads()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: 64 * MIB as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(999, &accesses)?;
    let retained: CompletionOwner = Arc::new((owner.clone(), programs.clone()));
    flight.retain_completion_owner(retained)?;
    not_cancelled(cancelled)?;
    // SAFETY: every native flight physically completed; original read-only W
    // guard retains exact images/charge/source on partial unload/readback error.
    unsafe { programs.retire_after_physical_completion() }?;
    let raw = flight
        .complete_read_only_readback(&reads, |raw| weights.validate(raw))
        .await?;
    slot.finish()?;
    receipt.immutable_weight_bytes = Role::Qkvz.bytes() + Role::Ba.bytes();
    drop((
        raw, reads, accesses, jobs, workspace, programs, images, weights, sources, queue, owner,
    ));
    service.ensure_usable()?;
    receipt.final_resident_bytes = service.resident_bytes()?;
    if receipt.final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "original2B projection buffers/native/event owners remain".into(),
        ));
    }
    drop(service);
    let retired = ResourceArbiter::process_shared().snapshot(&key)?;
    receipt.final_physical_reserved_bytes = retired.reserved;
    if retired.reserved != 0 || retired.fenced {
        return Err(Error::State(
            "original2B projection envelope not retired".into(),
        ));
    }
    Ok(receipt)
}
