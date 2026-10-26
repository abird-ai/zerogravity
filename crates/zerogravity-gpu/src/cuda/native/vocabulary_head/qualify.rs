//! Same original preparation/flight/readback loop for both closed head scopes.
use super::super::projection_cubin::{resident::AdmittedImages, validate_image_set};
use super::*;

pub async fn qualify_vocabulary_head_cubins(
    device: Device,
    source: Arc<RetainedSource>,
    fixtures: VocabularyHeadFixtures,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    prepare_tied: impl FnOnce() -> Result<Arc<TiedWeightSource>>,
) -> Result<VocabularyHeadComparison> {
    match inner(
        device,
        source,
        fixtures,
        inputs,
        Scope::CapturedSelected,
        Mode::Compare,
        cancelled,
        || Err(Error::State("captured head has no final RMS source".into())),
        prepare_tied,
    )
    .await?
    {
        Outcome::Compared(result) => Ok(result),
        Outcome::Fault(_) => Err(Error::State("positive captured head returned fault".into())),
    }
}
pub async fn qualify_final_norm_head_cubins(
    device: Device,
    source: Arc<RetainedSource>,
    fixtures: VocabularyHeadFixtures,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    prepare_final: impl FnOnce() -> Result<BoundFinalRmsSource>,
    prepare_tied: impl FnOnce() -> Result<Arc<TiedWeightSource>>,
) -> Result<VocabularyHeadComparison> {
    match inner(
        device,
        source,
        fixtures,
        inputs,
        Scope::ComputedFinalNorm,
        Mode::Compare,
        cancelled,
        prepare_final,
        prepare_tied,
    )
    .await?
    {
        Outcome::Compared(result) => Ok(result),
        Outcome::Fault(_) => Err(Error::State(
            "positive final norm/head returned fault".into(),
        )),
    }
}

fn prepare_accesses(
    weight: &ResidentLease,
    input: &HeadInputView,
    output: &ResidentLease,
    fused: Option<&Arc<FusedRmsPrepared>>,
) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
    let mut all = BTreeMap::new();
    if let Some(fused) = fused {
        for (lease, access) in fused.accesses() {
            insert_access(&mut all, lease.clone(), *access)?;
        }
    }
    for (lease, access) in [
        (weight, DeviceAccess::Read),
        (input.lease(), DeviceAccess::Read),
        (output, DeviceAccess::Write),
    ] {
        insert_access(&mut all, lease.clone(), access)?;
    }
    Ok(all.into_values().collect())
}
fn validate_comparison(job: &Job, fixture: &VocabularyHeadFixture, raw: &[Bytes]) -> Result<()> {
    if job.case != fixture.case {
        return Err(Error::State("head physical original phase changed".into()));
    }
    if let Some(fused) = &job.fused {
        let original = fixture
            .final_norm
            .as_ref()
            .ok_or_else(|| Error::State("computed head dual input missing".into()))?;
        if raw.len() != 8
            || &raw[0][..] != original.branch.as_slice()
            || &raw[1][..] != original.residual.as_slice()
        {
            return Err(Error::State(
                "computed final norm original inputs changed".into(),
            ));
        }
        fused.validate_readback(&raw[2..6])?;
        for (expected, index) in fixture.expected.iter().zip([3, 4, 5, 6, 7]) {
            if &raw[index][..] != expected.bytes.as_slice() {
                return Err(Error::State(format!(
                    "final norm/head {:?} {:?} differs",
                    fixture.case, expected.output
                )));
            }
        }
        if !finite_f32(&raw[3]) || [4, 5, 6, 7].into_iter().any(|i| !finite_bf16(&raw[i])) {
            return Err(Error::State(
                "nonfinite complete computed final norm/head".into(),
            ));
        }
    } else if raw.len() != 2
        || &raw[0][..] != fixture.selected.as_slice()
        || &raw[1][..] != fixture.expected[0].bytes.as_slice()
        || !finite_bf16(&raw[1])
    {
        return Err(Error::State(format!(
            "original captured selected head {:?} differs",
            fixture.case
        )));
    }
    Ok(())
}
#[expect(clippy::too_many_arguments)]
pub(super) async fn inner(
    device: Device,
    source: Arc<RetainedSource>,
    fixtures: VocabularyHeadFixtures,
    inputs: CubinQualificationInputs,
    scope: Scope,
    mode: Mode,
    cancelled: Option<&AtomicBool>,
    prepare_final: impl FnOnce() -> Result<BoundFinalRmsSource>,
    prepare_tied: impl FnOnce() -> Result<Arc<TiedWeightSource>>,
) -> Result<Outcome> {
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA native head qualification".into(),
        ));
    }
    let metadata = source.source().reserve_workspace(128 * 1024)?;
    validate_fixtures(&source, &fixtures, scope, mode)?;
    validate_image_set(
        source.source(),
        &BTreeSet::from([ProjectionImage::Bf16Gemv2TSm86]),
        &inputs.images,
    )?;
    // Actual final source/config binding refuses before any CUDA service.
    let final_binding = if scope == Scope::ComputedFinalNorm {
        let binding = prepare_final()?;
        binding.require_source(source.source())?;
        Some(binding)
    } else {
        drop(prepare_final);
        None
    };
    let owner = Arc::new(HeadOwner {
        cohort: fixtures,
        source,
        _metadata: metadata,
    });
    let service = ResidentService::new_pinned_cuda(
        device,
        owner.source.source_identity(),
        ResourceOwner::new("qwen08-tied-vocabulary-head", "closed-original-five-phases")?,
        HEAD_CAP,
        inputs.device_cap,
    )?;
    let key = service.physical_key();
    if mode == Mode::Fault(VocabularyHeadFault::Pressure) {
        let witness = faults::pressure(&service)?;
        drop((prepare_tied, final_binding, owner, inputs, service));
        return faults::pressure_retired(witness, scope).map(Outcome::Fault);
    }
    let queue = service.pinned_cuda_queue()?.clone();
    let sources = queue
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: (TIED_BYTES + NATIVE_BYTES) as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    // Complete genuine base uploads first. Existing fused preparation opens
    // its own original initialization; none may nest an armed submission.
    let mut base_init = service.begin_initialization(1000)?;
    base_init.retain_completion_owner(owner.clone())?;
    let mut bases = Vec::with_capacity(owner.cohort.fixtures.len());
    for fixture in &owner.cohort.fixtures {
        not_cancelled(cancelled)?;
        let output = base_init.allocate_empty(496_640, BufferRole::Output)?;
        let input = match scope {
            Scope::CapturedSelected => BaseInput::Captured(
                base_init.upload_bytes(&fixture.selected, BufferRole::Activation)?,
            ),
            Scope::ComputedFinalNorm => {
                let original = fixture
                    .final_norm
                    .as_ref()
                    .ok_or_else(|| Error::State("final norm authentic inputs absent".into()))?;
                BaseInput::Final {
                    branch: base_init.upload_bytes(&original.branch, BufferRole::Activation)?,
                    prior: base_init.upload_bytes(&original.residual, BufferRole::Activation)?,
                }
            }
        };
        bases.push(BasePhase {
            case: fixture.case,
            input,
            output,
        });
    }
    drop(base_init.complete_sync().await?);
    let mut phases = Vec::with_capacity(bases.len());
    for (index, base) in bases.into_iter().enumerate() {
        not_cancelled(cancelled)?;
        let (input, fused, mut reads) = match base.input {
            BaseInput::Captured(lease) => {
                let input =
                    HeadInputView::captured(owner.source.source().clone(), base.case, lease)?;
                let reads = vec![input.readback()?];
                (input, None, reads)
            }
            BaseInput::Final { branch, prior } => {
                let mut reads = vec![
                    branch.readback(0, branch.bytes())?,
                    prior.readback(0, prior.bytes())?,
                ];
                let binding = final_binding
                    .as_ref()
                    .ok_or_else(|| Error::State("closed final source binding absent".into()))?
                    .clone();
                let fused = FusedRmsPrepared::prepare_final(
                    service.clone(),
                    binding,
                    base.case.phase_rows(),
                    branch,
                    prior,
                    1100 + index as u64,
                    cancelled,
                )
                .await?;
                reads.extend(Vec::from(fused.readbacks()?));
                let input = HeadInputView::computed(
                    owner.source.source().clone(),
                    base.case,
                    fused.clone(),
                )?;
                reads.push(input.readback()?);
                (input, Some(fused), reads)
            }
        };
        reads.push(base.output.readback(0, base.output.bytes())?);
        phases.push(Arc::new(PhaseJob {
            case: base.case,
            input,
            output: base.output,
            fused,
            reads,
        }));
    }
    // Real W H2D is LAST. Retain every already prepared phase/source before
    // its first window upload and before partial native image/module loading.
    let mut init = service.begin_initialization(1500)?;
    init.retain_completion_owner(owner.clone())?;
    for phase in &phases {
        init.retain_completion_owner(phase.clone())?;
    }
    // Family tied-role authentication remains after all original phase setup.
    let tied = prepare_tied()?;
    if !tied.source().same_issuer(owner.source.source())
        || !Arc::ptr_eq(tied.owner(), owner.source.owner())
    {
        return Err(Error::State(
            "head factory changed original source owner".into(),
        ));
    }
    let head_output = tied.output().clone();
    drop(tied);
    let pending = init
        .upload_tied_bf16(
            owner.source.source().clone(),
            head_output.clone(),
            cancelled,
        )
        .await?;
    let mut jobs = Vec::with_capacity(phases.len());
    for phase in &phases {
        let accesses = prepare_accesses(
            pending.lease(),
            &phase.input,
            &phase.output,
            phase.fused.as_ref(),
        )?;
        let outputs = accesses
            .iter()
            .filter(|(_, a)| *a == DeviceAccess::Write)
            .map(|(l, _)| l.clone())
            .collect();
        let job = Arc::new(Job {
            phase: phase.clone(),
            accesses,
            outputs,
        });
        init.retain_completion_owner(job.clone())?;
        jobs.push(job);
    }
    // All original managed W/phase/RMS/Y buffers exist before any native load.
    let admitted = AdmittedImages::reserve(&service, owner.clone(), inputs.images)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let images = admitted.prepare_vocabulary_head_images(sources.clone(), &mut init, cancelled)?;
    let programs = HeadPrograms::prepare(images.clone(), &mut init, cancelled)?;
    drop(init.complete_sync().await?);
    slot.finish()?;
    let weight = pending.bind_initialized()?;
    // Only actual Jobs retain phases from here; unknown inspection has no
    // external strong phase or source-binding witness.
    drop((phases, final_binding));
    let (copies, checkpoints) = weight.copy_counts()?;
    let mut comparison = VocabularyHeadComparison {
        weight_copies: copies,
        weight_checkpoints: checkpoints,
        ..Default::default()
    };
    let mut index = 0;
    while let Some(job) = jobs.get(index).cloned() {
        let witness = if let Mode::Fault(fault) = mode {
            Some(faults::UnknownWitness::new(
                &owner, &job, &programs, &weight, &service, scope, fault,
            )?)
        } else {
            None
        };
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: NATIVE_BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        not_cancelled(cancelled)?;
        slot.begin_submission()?;
        let mut flight = service.submit(2000 + index as u64, &job.accesses)?;
        flight.retain_completion_owner(owner.clone())?;
        flight.retain_completion_owner(weight.clone())?;
        flight.retain_completion_owner(programs.clone())?;
        flight.retain_completion_owner(job.clone())?;
        if let Some(fused) = &job.fused {
            fused.enqueue(cancelled)?;
        }
        let launches = programs.enqueue(&weight, &job.input, &job.output, cancelled)?;
        // SAME natural post-enqueue cancellation boundary in positive/fault
        // execution. Any cancellation here drops the original armed flight.
        let late_flag =
            AtomicBool::new(mode == Mode::Fault(VocabularyHeadFault::LateCancelAfterHead));
        let boundary = not_cancelled(
            if mode == Mode::Fault(VocabularyHeadFault::LateCancelAfterHead) {
                Some(&late_flag)
            } else {
                cancelled
            },
        );
        if let Some(witness) = witness {
            if mode == Mode::Fault(VocabularyHeadFault::LateCancelAfterHead) {
                if boundary.is_ok() {
                    return Err(Error::State(
                        "actual head late cancellation did not refuse".into(),
                    ));
                }
            } else {
                boundary?;
            }
            drop(flight);
            drop(slot);
            drop((
                job, jobs, weight, programs, images, sources, queue, owner, service,
            ));
            return witness.inspect(launches).map(Outcome::Fault);
        }
        boundary?;
        let (ready, raw) = flight
            .complete_with_readback(&job.outputs, &job.reads, |raw| {
                validate_comparison(&job, &owner.cohort.fixtures[index], raw)
            })
            .await?;
        slot.finish()?;
        comparison.calls += 1;
        comparison.native_launches += launches;
        comparison.producer_flights += 1;
        comparison.vocabulary_bf16_words += 248_320;
        if job.fused.is_some() {
            comparison.rms_stages += 10;
            comparison.sum_f32_words += job.case.phase_rows() * 1024;
            comparison.normalized_bf16_words += job.case.phase_rows() * 1024;
            comparison.residual_bf16_words += job.case.phase_rows() * 1024;
            comparison.selected_bf16_words += 1024;
        }
        drop((ready, raw));
        index += 1;
    }
    // Actual full unchanged-W comparisons remain finite. Retire/unload under
    // the LAST original initialized-W Read guard, never an empty fake guard.
    let readback_admission = Arc::new(owner.source.source().reserve_workspace(32 * 1024 * 1024)?);
    let descriptor = head_output.clone();
    let matrix = owner
        .source
        .source()
        .matrix(&descriptor.name, descriptor.axes)?;
    for index in 0..TIED_WINDOWS {
        not_cancelled(cancelled)?;
        let first = index * 512;
        let window = Arc::new(match cancelled {
            Some(flag) => {
                owner
                    .source
                    .source()
                    .read_rows_cancellable(&matrix, first..first + 512, flag)?
            }
            None => owner
                .source
                .source()
                .read_rows(&matrix, first..first + 512)?,
        });
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: NATIVE_BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        not_cancelled(cancelled)?;
        slot.begin_submission()?;
        let mut flight = service.submit(
            3000 + index as u64,
            &[(weight.lease().clone(), DeviceAccess::Read)],
        )?;
        flight.retain_completion_owner(owner.clone())?;
        flight.retain_completion_owner(weight.clone())?;
        flight.retain_completion_owner(programs.clone())?;
        flight.retain_completion_owner(readback_admission.clone())?;
        flight.retain_completion_owner(window.clone())?;
        if index == TIED_WINDOWS - 1 {
            // SAFETY: all original producer flights completed. This exact Read
            // flight retains real initialized W/image/module/admission through
            // every unload failure and its final source readback.
            unsafe { programs.retire() }.map_err(native_error)?;
        }
        let raw = flight
            .complete_read_only_readback(
                &[weight
                    .lease()
                    .readback(index * TIED_WINDOW_BYTES, TIED_WINDOW_BYTES)?],
                |raw| {
                    if raw.len() != 1 || &raw[0][..] != window.as_slice() {
                        return Err(Error::State("native head original full W changed".into()));
                    }
                    Ok(())
                },
            )
            .await?;
        drop(raw);
        slot.finish()?;
        comparison.weight_bytes_compared += TIED_WINDOW_BYTES;
        comparison.weight_readback_flights += 1;
    }
    if comparison.weight_bytes_compared != TIED_BYTES || comparison.calls != 5 {
        return Err(Error::State(
            "full original head coverage incomplete".into(),
        ));
    }
    drop((
        readback_admission,
        jobs,
        weight,
        programs,
        images,
        sources,
        queue,
        owner,
    ));
    service.ensure_usable()?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "native head original buffers/events/charge not retired".into(),
        ));
    }
    drop(service);
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.reserved != 0 || physical.fenced || physical.reservations != 0 {
        return Err(Error::State(
            "native head original envelope not retired".into(),
        ));
    }
    Ok(Outcome::Compared(comparison))
}
