//! Source-owned eleven-node cut on the original model/session/attempt/node loop.
//! Genuine all42 candidates are diagnostic; no RootReady/commit/model route.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule};
use super::mlp_component::MlpComponentReadback;
use super::*;
use crate::resident_inference::gdn_state::GdnPhasePlan;
use crate::resident_leases::Qwen2bMlpSourceVerification;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::qwen2b_component::{
    ComponentProgram, ComponentRecipe, QWEN2B_MLP_PROVIDER_BYTES as PROVIDER_BYTES,
    validate_qwen2b_mlp_component_source,
};
use crate::vllm_cuda_gdn::Qwen2bGdnArithmetic;
use crate::vllm_cuda_qualification::qwen2b_mlp::{Qwen2bMlpQualificationInputs, validate_inputs};
use zerogravity_artifacts::source_weights::WindowWorkspace;

use zerogravity_core::accelerator::program::RecordedVllm08Model;
use zerogravity_core::device_events::BufferRole;
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;

/// Preexisting source/record/admission. Only the controlled helper establishes reserve-before-record chronology.
pub struct Qwen2bMlpComponentSource {
    record: Arc<RecordedVllm08Model>,
    bound: BoundFullModelSource,
    record_admission: Arc<WindowWorkspace>,
}
impl Qwen2bMlpComponentSource {
    pub fn new(
        bound: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        record_admission: WindowWorkspace,
    ) -> Result<Self> {
        if record_admission.bytes() != 32 * 1024 * 1024
            || !record_admission
                .source()
                .same_issuer(bound.source().source())
            || record.call().source() != bound.source().source_identity()
            || record.call().model_profile()
                != zerogravity_core::accelerator::program::FullModelProfile::Qwen2B
            || bound.model_profile() != record.call().model_profile()
        {
            return Err(Error::State(
                "same immutable2B record/source and actual32MiB ticket".into(),
            ));
        }
        Ok(Self {
            record,
            bound,
            record_admission: Arc::new(record_admission),
        })
    }
}

#[path = "qwen2b_mlp_component/faults.rs"]
mod faults;
pub use faults::{
    Qwen2bMlpComponentFault, Qwen2bMlpComponentFaultOutcome, qualify_qwen2b_mlp_component_fault,
};
#[derive(Clone, Debug)]
pub struct Qwen2bMlpComponentReceipt {
    pub arithmetic: Qwen2bGdnArithmetic,
    pub expected_trajectory: &'static str,
    pub original_selected_ptx_equals_owned_tuple: Option<bool>,
    pub parent_model_accepted: bool,
    pub rows: usize,
    pub executed_nodes: usize,
    pub native_launches: usize,
    pub down1_enqueues: Option<usize>,
    pub dispatches: usize,
    pub readback_bytes: usize,
    pub immutable_weight_bytes: usize,
    pub streamed_weight_windows: usize,
    pub retirement_streamed_weight_windows: usize,
    pub carried_domains: usize,
    pub original_root: RootBinding,
    pub model_root_published: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}
struct RunMode<'a> {
    arithmetic: Qwen2bGdnArithmetic,
    fault: Option<Qwen2bMlpComponentFault>,
    probe: Option<&'a faults::Probe>,
}
enum Outcome {
    Compared(Qwen2bMlpComponentReceipt),
    Pressure(Qwen2bMlpComponentFaultOutcome),
}
/// Expected outputs are available ONLY inside original trusted completion.
pub async fn qualify_qwen2b_mlp_component<F>(
    device: Device,
    source: Qwen2bMlpComponentSource,
    arithmetic: Qwen2bGdnArithmetic,
    inputs: Qwen2bMlpQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<Qwen2bMlpComponentReceipt>
where
    F: FnOnce(MlpComponentReadback<'_>) -> Result<()>,
{
    match run(
        device,
        source,
        inputs,
        cancelled,
        validate,
        RunMode {
            arithmetic,
            fault: None,
            probe: None,
        },
    )
    .await?
    {
        Outcome::Compared(receipt) => Ok(receipt),
        Outcome::Pressure(_) => Err(Error::State("positive component returned pressure".into())),
    }
}
async fn run<F>(
    device: Device,
    source: Qwen2bMlpComponentSource,
    inputs: Qwen2bMlpQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
    mode: RunMode<'_>,
) -> Result<Outcome>
where
    F: FnOnce(MlpComponentReadback<'_>) -> Result<()>,
{
    let RunMode {
        arithmetic,
        fault,
        probe,
    } = mode;
    not_cancelled(cancelled)?;
    let Qwen2bMlpComponentSource {
        record,
        bound: source,
        record_admission,
    } = source;
    let phase = match record.call().rows() {
        1 => crate::vllm_cuda_gdn::VllmGdnPhase::InitialOne,
        64 => crate::vllm_cuda_gdn::VllmGdnPhase::Initial64,
        _ => return Err(Error::Unsupported("explicit2B MLP initial1/64 only".into())),
    };
    arithmetic.require_owned(phase)?;
    if record.call().rows() != 1
        && matches!(
            fault,
            Some(
                Qwen2bMlpComponentFault::UnknownAfterDownMain
                    | Qwen2bMlpComponentFault::CancelBeforeDownReducer
            )
        )
    {
        return Err(Error::Unsupported(
            "Down main/reducer cut is one-row only".into(),
        ));
    }
    validate_qwen2b_mlp_component_source(&source, &record, arithmetic, record_admission.clone())?;
    if fault != Some(Qwen2bMlpComponentFault::AdmissionPressure) {
        let recipe = ComponentRecipe::bind_qwen2b_mlp_arithmetic(
            source.clone(),
            record.clone(),
            arithmetic,
            record_admission.clone(),
        )?;
        validate_inputs(&recipe, &inputs)?;
    }
    validate_provider(&device)?;
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported("release MLP component required".into()));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source().source_identity(),
        ResourceOwner::new(
            "source-qwen2b-owned-mlp",
            "unpublished-eleven-node-component",
        )?,
        PROVIDER_BYTES,
        inputs.projections.device_cap,
    )?;
    let physical_key = service.physical_key();
    if service.client().properties().memory.alignment != 512 {
        return Err(Error::Unsupported(
            "explicit2B MLP actual512B allocation alignment".into(),
        ));
    }
    let initial = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if initial.fenced || initial.reserved != PROVIDER_BYTES as u64 {
        return Err(Error::State(
            "MLP component requires solitary original envelope".into(),
        ));
    }
    if fault == Some(Qwen2bMlpComponentFault::AdmissionPressure) {
        return faults::pressure(service, arithmetic).map(Outcome::Pressure);
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: PROVIDER_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let program = ResidentProgram::prepare_qwen2b_mlp_component_arithmetic(
        service.clone(),
        source.clone(),
        record.clone(),
        arithmetic,
        record_admission.clone(),
        cancelled,
    )
    .await?;
    slot.finish()?;
    let states = program.budget().states.clone();
    let state_bytes = program.budget().prior_state_bytes;
    let model = Arc::new(ResidentModel {
        service: service.clone(),
        workspace: Mutex::new(Workspace {
            program,
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
    let prepared_owner = {
        let workspace = model.workspace.lock().await;
        workspace.program.qwen2b_component()?.completion_owner()
    };
    let session = model.session_retaining(prepared_owner).await?;
    let base = session.binding();
    if base.frontier() != 0 {
        return Err(Error::State(
            "MLP component genuine initial root required".into(),
        ));
    }
    not_cancelled(cancelled)?;
    let candidate = model.allocate_state(false).await?;
    let invocation = record.call().request();
    let attempt = session.authority.begin_advance(invocation.rows() as u64)?;
    let schedule = GdnAttemptSchedule::new_initial_qwen2b_component(
        &source,
        attempt.clone(),
        invocation.clone(),
    )?;
    let phase_binding = AttemptPhaseBinding::new(schedule.clone(), 0, None)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: PROVIDER_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut workspace = model.workspace.lock().await;
    let body = workspace.program.qwen2b_component()?.clone();
    let domains = session.bind_gdn_attempt_phase_state(
        &attempt,
        GdnPhasePlan {
            candidate: &candidate,
            plan: workspace.program.plan(),
            invocation,
            phase,
            nodes: [4, 5],
        },
        phase_binding.clone(),
    )?;
    body.prepare_gdn(domains.clone(), cancelled).await?;
    body.prepare_qwen2b_mlp_weights(cancelled).await?;
    let source_plan =
        body.qwen2b_mlp_verification_plan(fault.and_then(Qwen2bMlpComponentFault::read_cut))?;
    // Last real W upload is the native preparation guard. No managed owner is
    // allocated after these native records activate their original pool.
    let mut init = service.begin_initialization(base.root())?;
    init.retain_completion_owner(body.completion_owner())?;
    body.prepare_qwen2b_mlp_native(inputs, &mut init, cancelled)?;
    drop(init.complete_sync().await?);
    let state_ids = session
        .domains
        .iter()
        .chain(&candidate)
        .flat_map(|d| d.leases.iter().map(ResidentLease::id))
        .collect::<BTreeSet<_>>();
    workspace.program.finalize_qwen2b_component(&state_ids)?;
    slot.finish()?;
    let bindings = bind_domains(&session.domains, &candidate, 0)?;
    let mut accesses = body.accesses()?;
    merge_accesses(
        &mut accesses,
        session
            .domains
            .iter()
            .flat_map(|d| d.leases.iter().cloned().map(|l| (l, DeviceAccess::Read))),
    );
    merge_accesses(
        &mut accesses,
        candidate
            .iter()
            .flat_map(|d| d.leases.iter().cloned().map(|l| (l, DeviceAccess::Write))),
    );
    let parameters = RuntimeParameters {
        token: invocation.teachers().ids()[0],
        position: 0,
        valid_tokens: invocation.rows(),
    };
    workspace
        .program
        .validate_runtime(service.client(), parameters, &bindings)?;
    let mut reads = body.readbacks()?;
    let prior = domain_readbacks(&mut reads, &session.domains)?;
    let proposed = domain_readbacks(&mut reads, &candidate)?;
    let weights = body.immutable_weights()?;
    let immutable_weight_bytes = weights
        .iter()
        .map(|(lease, _)| lease.bytes())
        .sum::<usize>()
        + body.qwen2b_mlp_weight_bytes()?;
    let weight_start = reads.len();
    for (lease, _) in &weights {
        reads.push(lease.readback(0, lease.bytes())?);
    }
    let readback_bytes = reads.iter().map(ResidentReadback::bytes).sum();
    if readback_bytes > 128 * 1024 * 1024 {
        return Err(Error::Limit(
            "MLP original admitted128MiB readback cohort".into(),
        ));
    }
    let cohort = Arc::new(reads);
    body.arm_fault(match fault {
        Some(Qwen2bMlpComponentFault::UnknownAfterGateUp) => 5,
        Some(Qwen2bMlpComponentFault::CancelBeforeActivation) => 6,
        Some(Qwen2bMlpComponentFault::UnknownAfterDownMain) => 7,
        Some(Qwen2bMlpComponentFault::CancelBeforeDownReducer) => 8,
        _ => 0,
    })?;
    if let Some(probe) = probe {
        probe.arm(&service, &body, &cohort, &source_plan, &accesses, base)?;
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
    let permit = PrefixPhaseEnqueue::new(&session, &schedule, 0, None)?;
    let mut flight = service.submit(base.root(), &accesses)?;
    flight.retain_completion_owner(body.completion_owner())?;
    flight.retain_completion_owner(cohort.clone())?;
    source_plan.retain_before_enqueue(&mut flight)?;
    let submitted = workspace.program.launch_attempt_fragment(
        service.client(),
        parameters,
        &bindings,
        &permit,
        cancelled,
    )?;
    if submitted.logits.is_some() || submitted.proposals.len() != 42 {
        return Err(Error::State(
            "MLP cut cannot publish logits/root or omit canonical proposals".into(),
        ));
    }
    let promoted = accesses
        .iter()
        .filter(|(_, access)| *access == DeviceAccess::Write)
        .map(|(l, _)| l.clone())
        .collect::<Vec<_>>();
    let (ready, actual) = source_plan
        .complete_compute(flight, &promoted, &cohort, cancelled, |raw| {
            source_plan.enter_compute_validator()?;
            if fault == Some(Qwen2bMlpComponentFault::ValidatorRefusal) {
                return Err(Error::State(
                    "deliberate qwen2b MLP trusted validator refusal".into(),
                ));
            }
            not_cancelled(cancelled)?;
            if raw.len() != cohort.len() {
                return Err(Error::State("MLP exact completed readback cohort".into()));
            }
            body.validate_readbacks(raw)?;
            for (slot, candidate_span) in &proposed {
                let (_, prior_span) = prior
                    .iter()
                    .find(|(key, _)| key == slot)
                    .ok_or_else(|| Error::State("MLP carry prior absent".into()))?;
                let changed = matches!(
                    slot,
                    StateSlot::Conv(zerogravity_core::accelerator::program::ConvDomain(0))
                        | StateSlot::Recurrent(
                            zerogravity_core::accelerator::program::RecurrentDomain(0)
                        )
                );
                if !changed
                    && (candidate_span.len() != prior_span.len()
                        || raw[candidate_span.clone()]
                            .iter()
                            .zip(&raw[prior_span.clone()])
                            .any(|(a, b)| a[..] != b[..]))
                {
                    return Err(Error::State(
                        "MLP remaining40 canonical domains not exact prior copies".into(),
                    ));
                }
            }
            for (actual, (_, expected)) in raw[weight_start..].iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "MLP immutable source W/parameters changed".into(),
                    ));
                }
            }
            validate(MlpComponentReadback::from_qwen2b_completed(
                &record, raw, &prior, &proposed,
            )?)
        })
        .await?;
    drop(actual);
    drop(ready);
    let progress = source_plan.compute_progress()?;
    if progress.completed_reads != 73
        || progress.compared_source_bytes != 75_497_472
        || !progress.trusted_validator_entered
    {
        return Err(Error::State(
            "actual73 complete source windows before trusted compute Ready".into(),
        ));
    }
    let streamed_weight_windows = progress.completed_reads;
    // Fresh retirement plan has no compute probe; no reset/reuse of fault metadata.
    drop(source_plan);
    slot.finish()?;
    if session.binding() != base || attempt.base() != base {
        return Err(Error::State("MLP diagnostic advanced original root".into()));
    }
    let mut retire_accesses = weights
        .iter()
        .map(|(l, _)| (l.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    merge_accesses(&mut retire_accesses, body.qwen2b_mlp_weight_accesses()?);
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
    let retirement_plan = body.qwen2b_mlp_retirement_plan()?;
    let mut guard = service.submit(base.root(), &retire_accesses)?;
    guard.retain_completion_owner(body.completion_owner())?;
    retirement_plan.retain_before_enqueue(&mut guard)?;
    unsafe {
        body.retire_after_physical_completion()?;
    }
    let rereads = weights
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let actual = retirement_plan
        .complete_retirement(guard, &rereads, cancelled, |raw| {
            if raw.len() != weights.len() {
                return Err(Error::State(
                    "MLP exact immutable retirement readback arity".into(),
                ));
            }
            for (actual, (_, expected)) in raw[..].iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "MLP source W changed during guarded retirement".into(),
                    ));
                }
            }
            Ok(())
        })
        .await?;
    drop(actual);
    // The sealed uninstrumented itinerary refuses unless actual73 source comparisons complete.
    let retirement_streamed_weight_windows = 73;
    drop(retirement_plan);
    slot.finish()?;
    let (dispatches, native_launches, rows) = (
        submitted.dispatches,
        submitted.fused_dispatches,
        invocation.rows(),
    );
    let down1_enqueues = if rows == 1 {
        Some(body.qwen2b_down1_enqueues()?)
    } else {
        None
    };
    drop(submitted);
    drop(rereads);
    drop(retire_accesses);
    drop(weights);
    drop(cohort);
    drop(promoted);
    drop(accesses);
    drop(bindings);
    drop(domains);
    drop(phase_binding);
    drop(schedule);
    drop(body);
    drop(workspace);
    drop(candidate);
    drop(attempt);
    drop(session);
    drop(model);
    drop(record);
    drop(source);
    drop(record_admission);
    let final_resident_bytes = service.resident_bytes()?;
    if final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "MLP original ledger did not retire to zero".into(),
        ));
    }
    drop(service);
    let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "MLP original physical envelope did not retire".into(),
        ));
    }
    Ok(Outcome::Compared(Qwen2bMlpComponentReceipt {
        arithmetic,
        expected_trajectory: "CallerValidatedCoherentOriginal2B",
        original_selected_ptx_equals_owned_tuple: None,
        parent_model_accepted: false,
        rows,
        executed_nodes: 11,
        native_launches,
        down1_enqueues,
        dispatches,
        readback_bytes,
        immutable_weight_bytes,
        streamed_weight_windows,
        retirement_streamed_weight_windows,
        carried_domains: 40,
        original_root: base,
        model_root_published: false,
        final_resident_bytes,
        final_physical_reserved_bytes: physical.reserved,
    }))
}
