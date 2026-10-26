//! Closed physical phases in ONE unpublished original session attempt.
//! Same recorded model recipe, workspace and ResidentProgram dispatch loop.
use super::continuation::{
    AttemptPhaseBinding, CompletedPrefixPhase, GdnAttemptSchedule, PlannedPrefixPrior,
    PrefixPhaseEnqueue,
};
use super::*;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::native::NativeProgram;

mod sequence_faults;
pub use sequence_faults::{GdnSequenceFault, GdnSequenceFaultOutcome, qualify_gdn_sequence_fault};
use sequence_faults::{SequenceMode, SequenceOutcome};

#[derive(Clone, Debug)]
pub struct GdnSequenceReceipt {
    pub phases: Vec<GdnFragmentReceipt>,
    /// Proposed original attempt span; never a published model frontier.
    pub attempted_tokens: u64,
    pub original_root: RootBinding,
    pub model_root_published: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct GdnSequenceRestoreReceipt {
    pub completed_first_phase: GdnFragmentReceipt,
    pub original_root: RootBinding,
    pub restored_original_zero_root: RootBinding,
    pub continuation_refused_before_enqueue: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}

struct PhaseReadbacks {
    cohort: Arc<Vec<ResidentReadback>>,
    prior: Vec<(StateSlot, std::ops::Range<usize>)>,
    candidate: Vec<(StateSlot, std::ops::Range<usize>)>,
    fused_nodes: Vec<usize>,
    value_count: usize,
    gdn_count: usize,
    weight_start: usize,
    bytes: usize,
}
/// Actual future producer/cohort/state ownership, retained before each enqueue.
/// No body points back to this owner; preceding completions form an acyclic chain.
struct SequenceOwners {
    frames: Vec<PhaseReadbacks>,
    candidates: Vec<Vec<Domain>>,
    initial: Vec<Domain>,
    bodies: Vec<Arc<NativeProgram>>,
    _source: VerifiedTensorSource,
}

/// Expected records are opened by the callback only after genuine arithmetic.
/// Later `state(slot, false)` is the actual preceding phase's canonical output.
pub async fn qualify_gdn_sequence<F>(
    device: Device,
    records: Vec<Arc<SourceBindings>>,
    inputs: LayerMlpCubinInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<GdnSequenceReceipt>
where
    F: FnMut(usize, GdnFragmentReadback<'_>) -> Result<()>,
{
    match qualify_sequence_inner(
        device,
        records,
        inputs,
        cancelled,
        validate,
        SequenceMode::Compare,
    )
    .await?
    {
        SequenceOutcome::Compared(value) => Ok(value),
        SequenceOutcome::Fault(_) | SequenceOutcome::Restored(_) => {
            Err(Error::State("positive sequence returned other mode".into()))
        }
    }
}

/// Revoke an actual prior physical completion using the original all42 zero
/// snapshot. No prefix candidate is published as a model root.
pub async fn qualify_gdn_sequence_restore_refusal<F>(
    device: Device,
    records: Vec<Arc<SourceBindings>>,
    inputs: LayerMlpCubinInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<GdnSequenceRestoreReceipt>
where
    F: FnMut(usize, GdnFragmentReadback<'_>) -> Result<()>,
{
    match qualify_sequence_inner(
        device,
        records,
        inputs,
        cancelled,
        validate,
        SequenceMode::RestoreBeforeContinuation,
    )
    .await?
    {
        SequenceOutcome::Restored(value) => Ok(value),
        _ => Err(Error::State(
            "restore subgate did not prove original refusal".into(),
        )),
    }
}

/// Separate fixed diagnostic envelopes; existing component/initial caps stay put.
fn sequence_capacity(
    source: &VerifiedTensorSource,
    records: &[Arc<SourceBindings>],
) -> Result<(usize, usize, Vec<SourceCall>)> {
    let layers = records
        .first()
        .map_or(0, |record| record.selection().completed_layers());
    if !matches!(layers, 1 | 3) || !matches!(records.len(), 2 | 3) {
        return Err(Error::Unsupported(
            "closed complete layer0/prefix request scope".into(),
        ));
    }
    let mut invocations = Vec::with_capacity(records.len());
    for record in records {
        // Actual config/epsilon/source headers and exact model recipe BEFORE CUDA.
        capacity_bound(record)?;
        if record.selection().completed_layers() != layers {
            return Err(Error::State(
                "request changes completed source layers".into(),
            ));
        }
        invocations.push(record.call().clone());
    }
    let expected: &[(SourceCallMode, usize, usize)] = if records.len() == 2 {
        &[
            (SourceCallMode::Prefill, 0, 1),
            (SourceCallMode::Decode, 1, 1),
        ]
    } else {
        &[
            (SourceCallMode::Prefill, 0, 64),
            (SourceCallMode::Prefill, 64, 2),
            (SourceCallMode::Decode, 66, 1),
        ]
    };
    for ((record, invocation), &(mode, base, rows)) in
        records.iter().zip(&invocations).zip(expected)
    {
        if !record.source().same_issuer(source)
            || invocation.source() != source.source_identity()
            || invocation.mode() != mode
            || invocation.rows().base() != base
            || invocation.rows().len() != rows
            || invocation.rows().capacity() != 512
            || invocation.rows().vocabulary() != 248320
        {
            return Err(Error::State(
                "exact source-owned two request schedules".into(),
            ));
        }
    }
    let cap = device_bound(layers, records.len())?;
    Ok((cap, layers, invocations))
}
fn device_bound(layers: usize, phases: usize) -> Result<usize> {
    // All (P+1) canonical states; shared source-W/workspace/native charge once;
    // every phase's actual RMS/GDN/value/fused/control allocation. GDN scratch
    // keeps the conservative prefill upper bound for the cached phase too.
    let rows: &[usize] = match phases {
        2 => &[1, 1],
        3 => &[64, 2, 1],
        _ => return Err(Error::Unsupported("closed phase count".into())),
    };
    if !matches!(layers, 1 | 3) {
        return Err(Error::Unsupported("closed completed layer count".into()));
    }
    let mut bound = (phases + 1) * 32_784_384 + 64 * 1024 * 1024 + layers * 43_057_152 + 8_519_680;
    for &rows in rows {
        let per_gdn = 49_504
            + 5_570_560
            + 16_421
            + 2_097_152
            + 524_288
            + rows * (35_072 + if rows > 1 { 64 } else { 0 });
        bound += 10_240
            + rows * 20_492
            + layers * (rows * 36_928 + per_gdn + rows * (2_048 + 14_336 + 7_168 + 2_048))
            + (2 * layers - 1) * (10_240 + rows * 20_492)
            + 4 * (layers + 1);
    }
    Ok(bound.div_ceil(32 * 1024 * 1024) * (32 * 1024 * 1024))
}
fn native_phase(invocation: &SourceCall) -> Result<crate::cuda::gdn::GdnPhase> {
    use crate::cuda::gdn::GdnPhase as P;
    match (
        invocation.mode(),
        invocation.rows().base(),
        invocation.rows().len(),
    ) {
        (SourceCallMode::Prefill, 0, 1) => Ok(P::InitialOne),
        (SourceCallMode::Prefill, 0, 64) => Ok(P::Initial64),
        (SourceCallMode::Prefill, 64, 2) => Ok(P::ContinuedTwo),
        (SourceCallMode::Decode, 1 | 66, 1) => Ok(P::CachedDecode),
        _ => Err(Error::Unsupported(
            "closed original GDN source phase".into(),
        )),
    }
}

fn touched(slot: StateSlot, layers: usize) -> bool {
    use zerogravity_core::accelerator::program::{ConvDomain, RecurrentDomain};
    matches!(slot, StateSlot::Conv(ConvDomain(i)) | StateSlot::Recurrent(RecurrentDomain(i)) if i < layers)
}

async fn qualify_sequence_inner<F>(
    device: Device,
    records: Vec<Arc<SourceBindings>>,
    inputs: LayerMlpCubinInputs,
    cancelled: Option<&AtomicBool>,
    mut validate: F,
    mode: SequenceMode,
) -> Result<SequenceOutcome>
where
    F: FnMut(usize, GdnFragmentReadback<'_>) -> Result<()>,
{
    not_cancelled(cancelled)?;
    let source = records
        .first()
        .ok_or_else(|| Error::Unsupported("closed phase count".into()))?
        .source()
        .clone();
    let (cap, layers, invocations) = sequence_capacity(&source, &records)?;
    let image_owner: CompletionOwner = records[0].clone();
    validate_provider(&device)?;
    let phases = records.len();
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner {
            model: format!("original-hf-native-{:?}", source.source_identity()),
            session: "unpublished-original-prefix-sequence".into(),
        },
        cap,
        inputs.projections.device_cap,
    )?;
    let physical_key = service.physical_key();
    let arbiter = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared();
    let initial = arbiter.snapshot(&physical_key)?;
    if initial.fenced || initial.reserved != cap as u64 {
        return Err(Error::State(
            "sequence requires original solitary envelope".into(),
        ));
    }
    if mode.is_pressure() {
        let pressure = faults::pressure(service, GdnFragmentFault::AdmissionPressure)?;
        return Ok(SequenceOutcome::Fault(GdnSequenceFaultOutcome {
            fault: GdnSequenceFault::AdmissionPressure,
            completed_phases: 0,
            attempted_phases: phases,
            original_previous_ready_retained: false,
            all_prepared_phases_retained: false,
            prepared_state_buffers: 0,
            active: pressure,
        }));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: cap as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut programs = Vec::with_capacity(phases);
    for record in records {
        programs.push(Some(
            ResidentProgram::prepare_fragment_values(service.clone(), record, cancelled).await?,
        ));
    }
    slot.finish()?;
    let first = programs[0].take().unwrap();
    let states = first.budget().states.clone();
    let state_bytes = first.budget().prior_state_bytes;
    let model = Arc::new(ResidentModel {
        service: service.clone(),
        workspace: Mutex::new(Workspace {
            program: first,
            greedy: None,
            leases: Vec::new(),
            state_accesses: Vec::new(),
            logit_accesses: Vec::new(),
            batch: None,
        }),
        allocation: Mutex::new(()),
        states,
        state_bytes,
        capacity: 512,
        vocab: 248320,
        state_pool: std::sync::OnceLock::new(),
        native_batch: false,
        #[cfg(feature = "hardware-abort-probe")]
        abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
        #[cfg(feature = "hardware-abort-probe")]
        launch_seen: std::sync::atomic::AtomicUsize::new(0),
    });
    let mut session = model.session().await?;
    let base = session.binding();
    let original_snapshot = mode.is_restore().then(|| session.snapshot());
    let last = invocations.last().unwrap();
    let attempted_tokens = (last.rows().base() + last.rows().len()) as u64;
    let attempt = session.authority.begin_advance(attempted_tokens)?;
    let schedule = GdnAttemptSchedule::new(base, attempt.clone(), invocations, layers)?;
    let mut candidates = Vec::with_capacity(phases);
    for _ in 0..phases {
        candidates.push(model.allocate_state(false).await?);
    }
    let mut planned = Vec::with_capacity(phases);
    let mut phase_bindings = Vec::with_capacity(phases);
    for index in 0..phases {
        let prior = if index == 0 {
            None
        } else {
            Some(PlannedPrefixPrior::new(
                &session,
                schedule.clone(),
                index,
                candidates[index - 1].clone(),
            )?)
        };
        phase_bindings.push(AttemptPhaseBinding::new(
            schedule.clone(),
            index,
            prior.clone(),
        )?);
        planned.push(prior);
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: cap as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut workspace = model.workspace.lock().await;
    let mut bodies = Vec::with_capacity(phases);
    for index in 0..phases {
        let program = if index == 0 {
            &workspace.program
        } else {
            programs[index].as_ref().unwrap()
        };
        let body = program.native_body()?.clone();
        let mut domains = Vec::with_capacity(layers);
        for (layer, nodes) in body.gdn_nodes().iter().enumerate() {
            let invocation = body.invocation_for_layer(layer)?;
            domains.push(session.bind_gdn_attempt_phase_state(
                &attempt,
                super::super::gdn_state::GdnPhasePlan {
                    candidate: &candidates[index],
                    plan: program.plan(),
                    invocation: &invocation,
                    phase: native_phase(&invocation)?,
                    nodes: *nodes,
                },
                phase_bindings[index].clone(),
            )?);
        }
        body.prepare_gdn(
            service.clone(),
            program.plan(),
            program.values(),
            domains,
            cancelled,
        )
        .await?;
        bodies.push(body);
    }
    not_cancelled(cancelled)?;
    let mut init = service.begin_initialization(base.root())?;
    for body in &bodies {
        init.retain_completion_owner(body.completion_owner())?;
    }
    // Immutable source/model owner, never a body that later owns this image set.
    let images = prepare_connected_native_images(
        &service,
        image_owner,
        &source,
        workspace.program.plan(),
        inputs,
        &mut init,
        cancelled,
    )?;
    let mut native_phases = Vec::with_capacity(phases);
    for index in 0..phases {
        let program = if index == 0 {
            &workspace.program
        } else {
            programs[index].as_ref().unwrap()
        };
        native_phases.push((&bodies[index], program.plan(), program.values()));
    }
    NativeProgram::prepare_native_phases(
        &native_phases,
        service.clone(),
        images,
        &mut init,
        cancelled,
    )?;
    drop(native_phases);
    drop(init.complete_sync().await?);
    let state_ids = session
        .domains
        .iter()
        .chain(candidates.iter().flatten())
        .flat_map(|domain| domain.leases.iter().map(ResidentLease::id))
        .collect::<BTreeSet<_>>();
    let mut owned = Vec::new();
    let mut frames = Vec::with_capacity(phases);
    for index in 0..phases {
        let program = if index == 0 {
            &mut workspace.program
        } else {
            programs[index].as_mut().unwrap()
        };
        merge_accesses(&mut owned, program.finalize_fragment_preflight(&state_ids)?);
        let body = &bodies[index];
        let prior = if index == 0 {
            &session.domains
        } else {
            &candidates[index - 1]
        };
        let mut cohort = body.readbacks(program.values())?;
        let prior_keys = domain_readbacks(&mut cohort, prior)?;
        let candidate_keys = domain_readbacks(&mut cohort, &candidates[index])?;
        let weights = body.immutable_weights()?;
        let weight_start = cohort.len();
        for (lease, _) in &weights {
            cohort.push(lease.readback(0, lease.bytes())?);
        }
        let bytes = cohort.iter().map(ResidentReadback::bytes).sum();
        body.require_readback_capacity(bytes)?;
        frames.push(PhaseReadbacks {
            cohort: Arc::new(cohort),
            prior: prior_keys,
            candidate: candidate_keys,
            fused_nodes: body.fused_nodes(),
            value_count: program.values().len(),
            gdn_count: layers,
            weight_start,
            bytes,
        });
    }
    if service.resident_bytes()? > cap {
        return Err(Error::Limit(
            "actual aggregate sequence exceeds fixed envelope".into(),
        ));
    }
    workspace.leases = owned.clone();
    workspace.state_accesses = owned;
    workspace.logit_accesses = Vec::new();
    let owners = Arc::new(SequenceOwners {
        frames,
        candidates,
        initial: session.domains.clone(),
        bodies,
        _source: source.clone(),
    });
    slot.finish()?;
    let mut previous: Option<Arc<CompletedPrefixPhase>> = None;
    let mut receipts = Vec::with_capacity(phases);
    let mut restored_root = None;
    for index in 0..phases {
        if index != 0 {
            planned[index]
                .as_ref()
                .unwrap()
                .bind_actual_completion(&session, previous.as_ref().unwrap().clone())?;
            let next = programs[index].take().unwrap();
            programs[index - 1] = Some(std::mem::replace(&mut workspace.program, next));
        }
        let invocation = schedule.invocation(index)?;
        let body = &owners.bodies[index];
        let frame = &owners.frames[index];
        let prior = if index == 0 {
            &owners.initial
        } else {
            &owners.candidates[index - 1]
        };
        let candidate = &owners.candidates[index];
        let bindings = bind_domains(prior, candidate, 0)?;
        let parameters = RuntimeParameters {
            token: invocation.rows().ids()[0],
            position: 0,
            valid_tokens: invocation.rows().len(),
        };
        workspace
            .program
            .validate_runtime(service.client(), parameters, &bindings)?;
        let permit = PrefixPhaseEnqueue::new(&session, &schedule, index, previous.as_ref())?;
        let mut accesses = body.accesses(workspace.program.values())?;
        merge_accesses(
            &mut accesses,
            prior
                .iter()
                .flat_map(|d| d.leases.iter().cloned().map(|l| (l, DeviceAccess::Read))),
        );
        merge_accesses(
            &mut accesses,
            candidate
                .iter()
                .flat_map(|d| d.leases.iter().cloned().map(|l| (l, DeviceAccess::Write))),
        );
        let fault = mode.active_fault(index, phases);
        body.arm_fault(fault)?;
        if fault.is_some() {
            mode.arm(sequence_faults::Arm {
                service: &service,
                source: &source,
                owners: &owners,
                index,
                previous: previous.as_ref(),
                accesses: &accesses,
                base,
            })?;
        }
        not_cancelled(cancelled)?;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(base.root(), &accesses)?;
        flight.retain_completion_owner(owners.clone())?;
        if let Some(previous) = &previous {
            flight.retain_completion_owner(previous.clone())?;
        }
        let submitted = workspace.program.launch_attempt_fragment(
            service.client(),
            parameters,
            &bindings,
            &permit,
            cancelled,
        )?;
        if submitted.logits.is_some()
            || submitted.proposals.len() != 42
            || submitted
                .proposals
                .iter()
                .zip(workspace.program.plan().proposed_domains())
                .any(|((version, _), expected)| version != expected)
        {
            return Err(Error::State(
                "sequence phase cannot publish incomplete model root/logits".into(),
            ));
        }
        let promoted = accesses
            .iter()
            .filter(|(_, access)| *access == DeviceAccess::Write)
            .map(|(lease, _)| lease.clone())
            .collect::<Vec<_>>();
        let weights = body.immutable_weights()?;
        let (ready, raw) = flight
            .complete_with_readback(&promoted, &frame.cohort, |raw| {
                not_cancelled(cancelled)?;
                body.validate_readbacks(raw, frame.value_count)?;
                for (slot, range) in &frame.prior {
                    if (index == 0 || !touched(*slot, layers))
                        && raw[range.clone()].iter().any(|b| b.iter().any(|v| *v != 0))
                    {
                        return Err(Error::State(
                            "actual original/untouched prior changed".into(),
                        ));
                    }
                }
                for (slot, range) in &frame.candidate {
                    if !touched(*slot, layers)
                        && raw[range.clone()].iter().any(|b| b.iter().any(|v| *v != 0))
                    {
                        return Err(Error::State(
                            "untouched canonical domains not carried exactly".into(),
                        ));
                    }
                }
                for (actual, (_, original)) in raw[frame.weight_start..].iter().zip(&weights) {
                    let actual: &[u8] = actual.as_ref();
                    if actual != *original {
                        return Err(Error::State("immutable sequence source W changed".into()));
                    }
                }
                validate(
                    index,
                    GdnFragmentReadback {
                        invocation,
                        raw,
                        value_count: frame.value_count,
                        gdn_count: frame.gdn_count,
                        fused_nodes: &frame.fused_nodes,
                        prior: &frame.prior,
                        candidate: &frame.candidate,
                    },
                )
            })
            .await?;
        drop(raw);
        // Keep the original session borrow through actual physical completion.
        permit.require_invocation(&service, invocation, 0, layers)?;
        slot.finish()?;
        previous = Some(CompletedPrefixPhase::after_completion(
            &session,
            schedule.clone(),
            index,
            candidate.clone(),
            ready,
            body.clone(),
            previous.take(),
        )?);
        receipts.push(GdnFragmentReceipt {
            rows: invocation.rows().len(),
            dispatches: submitted.dispatches,
            native_launches: submitted.fused_dispatches,
            readback_bytes: frame.bytes,
            carried_domains: 42 - 2 * layers,
            prior_root: base,
            model_root_published: false,
            // Filled only after all phase/module/source owners physically retire.
            final_resident_bytes: 0,
            final_physical_reserved_bytes: 0,
        });
        drop(submitted);
        if let Some(snapshot) = &original_snapshot {
            session.restore(snapshot)?;
            let restored = session.binding();
            if index != 0
                || restored.frontier() != 0
                || restored.session_epoch() == base.session_epoch()
                || PrefixPhaseEnqueue::new(&session, &schedule, 1, previous.as_ref()).is_ok()
                || owners.bodies.iter().skip(1).any(|b| {
                    b.rms_stages() != 0
                        || b.native_launches() != 0
                        || b.convolution_dispatches() != 0
                })
            {
                return Err(Error::State(
                    "original restore did not revoke continuation before any enqueue".into(),
                ));
            }
            restored_root = Some(restored);
            break;
        }
    }
    if (restored_root.is_none() && session.binding() != base) || attempt.base() != base {
        return Err(Error::State(
            "unpublished request changed original logical root".into(),
        ));
    }
    // All source/parameter copies are checked again during ONE shared retirement.
    let mut weights = Vec::new();
    let mut seen = BTreeSet::new();
    for body in &owners.bodies {
        for weight in body.immutable_weights()? {
            if seen.insert(weight.0.id()) {
                weights.push(weight);
            }
        }
    }
    let retire_accesses = weights
        .iter()
        .map(|(lease, _)| (lease.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let reads = weights
        .iter()
        .map(|(lease, _)| lease.readback(0, lease.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: 1,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut retirement = service.submit(session.binding().root(), &retire_accesses)?;
    retirement.retain_completion_owner(owners.clone())?;
    if let Some(previous) = &previous {
        retirement.retain_completion_owner(previous.clone())?;
    }
    // All programs share this one union of records and this one module set.
    unsafe {
        owners.bodies[0].retire_after_physical_completion()?;
    }
    let raw = retirement
        .complete_read_only_readback(&reads, |raw| {
            for (actual, (_, expected)) in raw.iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "immutable source changed during shared retirement".into(),
                    ));
                }
            }
            Ok(())
        })
        .await?;
    drop(raw);
    slot.finish()?;
    drop(reads);
    drop(retire_accesses);
    drop(weights);
    drop(previous);
    drop(owners);
    drop(phase_bindings);
    drop(planned);
    drop(programs);
    drop(workspace);
    drop(session);
    drop(original_snapshot);
    drop(model);
    drop(schedule);
    drop(attempt);
    drop(source);
    let final_resident_bytes = service.resident_bytes()?;
    if final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "sequence actual original ledger not zero".into(),
        ));
    }
    drop(service);
    let physical = arbiter.snapshot(&physical_key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "sequence original physical envelope not retired".into(),
        ));
    }
    if let Some(restored) = restored_root {
        return Ok(SequenceOutcome::Restored(GdnSequenceRestoreReceipt {
            completed_first_phase: receipts.remove(0),
            original_root: base,
            restored_original_zero_root: restored,
            continuation_refused_before_enqueue: true,
            final_resident_bytes,
            final_physical_reserved_bytes: physical.reserved,
        }));
    }
    Ok(SequenceOutcome::Compared(GdnSequenceReceipt {
        phases: receipts,
        attempted_tokens,
        original_root: base,
        model_root_published: false,
        final_resident_bytes,
        final_physical_reserved_bytes: physical.reserved,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aggregate_envelopes_cover_every_phase_and_preserve_old_caps() {
        for (layers, phases, mib) in [(1, 2, 224), (1, 3, 288), (3, 2, 352), (3, 3, 448)] {
            assert_eq!(device_bound(layers, phases).unwrap(), mib * 1024 * 1024);
        }
        for (layers, phases) in [(0, 2), (2, 2), (3, 1), (1, 4)] {
            assert!(device_bound(layers, phases).is_err());
        }
    }
    #[test]
    fn actual_state_keys_are_layer_indexes_not_gdn_ordinals() {
        use zerogravity_core::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain};
        for i in 0..3 {
            assert!(touched(StateSlot::Conv(ConvDomain(i)), 3));
            assert!(touched(StateSlot::Recurrent(RecurrentDomain(i)), 3));
        }
        assert!(!touched(StateSlot::Kv(KvDomain(3)), 3));
        assert!(!touched(StateSlot::Conv(ConvDomain(4)), 3));
        assert!(!touched(StateSlot::Recurrent(RecurrentDomain(1)), 1));
    }
}
