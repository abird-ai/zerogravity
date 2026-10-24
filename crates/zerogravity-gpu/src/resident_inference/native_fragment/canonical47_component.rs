//! Source-owned six-node cut on the original model/session/attempt/node loop.
//! Genuine all42 candidates are diagnostic; no RootReady/commit/model route.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule};
use super::*;
use crate::resident_inference::gdn_state::GdnPhasePlan;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::qwen2b_component::{
    ComponentProgram, PROVIDER_BYTES, validate_canonical47_component_source,
};
use crate::vllm_cuda_gdn::{Canonical47GdnArithmetic, Canonical47GdnIntermediate};
use zerogravity_core::accelerator::program::RecordedVllm08Model;
use zerogravity_core::device_events::BufferRole;
use zerogravity_model_qwen35::original_binding::BoundFullModelSource;

#[path = "canonical47_component/faults.rs"]
mod faults;
pub use faults::{
    Canonical47GdnComponentFault, Canonical47GdnComponentFaultOutcome,
    qualify_canonical47_gdn_component_fault,
};
#[derive(Clone, Debug)]
pub struct Canonical47GdnComponentReceipt {
    pub arithmetic: Canonical47GdnArithmetic,
    pub expected_trajectory: &'static str,
    pub original_selected_ptx_equals_owned_tuple: bool,
    pub parent_model_accepted: bool,
    pub rows: usize,
    pub executed_nodes: usize,
    pub native_launches: usize,
    pub dispatches: usize,
    pub readback_bytes: usize,
    pub immutable_weight_bytes: usize,
    pub carried_domains: usize,
    pub original_root: RootBinding,
    pub model_root_published: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}
pub struct Canonical47GdnComponentReadback<'a> {
    arithmetic: Canonical47GdnArithmetic,
    record: &'a Arc<RecordedVllm08Model>,
    raw: &'a [Bytes],
    prior: &'a [(StateSlot, std::ops::Range<usize>)],
    candidate: &'a [(StateSlot, std::ops::Range<usize>)],
    intermediates: &'a [(Canonical47GdnIntermediate, usize)],
}
impl Canonical47GdnComponentReadback<'_> {
    pub fn arithmetic(&self) -> Canonical47GdnArithmetic {
        self.arithmetic
    }
    pub fn intermediate(&self, key: Canonical47GdnIntermediate) -> Result<&[u8]> {
        let (_, index) = self
            .intermediates
            .iter()
            .find(|(actual, _)| *actual == key)
            .ok_or_else(|| Error::State("47 actual intermediate descriptor absent".into()))?;
        let bytes = self
            .raw
            .get(*index)
            .ok_or_else(|| Error::State("47 actual intermediate readback absent".into()))?;
        if bytes.len() != key.bytes() {
            return Err(Error::State(
                "47 completed intermediate physical bytes differ".into(),
            ));
        }
        Ok(&bytes[..])
    }
    pub fn record(&self) -> &Arc<RecordedVllm08Model> {
        self.record
    }
    pub fn value(&self, id: ValueId) -> Result<&[u8]> {
        if id.0 >= 7 {
            return Err(Error::Unsupported(
                "canonical47 cut has no value after Gated".into(),
            ));
        }
        self.raw
            .get(2 + id.0 as usize)
            .map(|v| v.as_ref())
            .ok_or_else(|| Error::State("canonical47 actual value absent".into()))
    }
    pub fn state(&self, slot: StateSlot, candidate: bool) -> Result<(&[u8], Option<&[u8]>)> {
        let keys = if candidate {
            self.candidate
        } else {
            self.prior
        };
        let (_, span) = keys
            .iter()
            .find(|(key, _)| *key == slot)
            .ok_or_else(|| Error::State("canonical47 actual canonical domain absent".into()))?;
        Ok((
            self.raw[span.start].as_ref(),
            if span.len() == 2 {
                Some(self.raw[span.start + 1].as_ref())
            } else {
                None
            },
        ))
    }
}
struct RunMode<'a> {
    arithmetic: Canonical47GdnArithmetic,
    fault: Option<Canonical47GdnComponentFault>,
    probe: Option<&'a faults::Probe>,
}
enum Outcome {
    Compared(Canonical47GdnComponentReceipt),
    Pressure(Canonical47GdnComponentFaultOutcome),
}
/// Expected outputs are available ONLY inside original trusted completion.
pub async fn qualify_canonical47_gdn_component<F>(
    device: Device,
    source: BoundFullModelSource,
    record: Arc<RecordedVllm08Model>,
    arithmetic: Canonical47GdnArithmetic,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<Canonical47GdnComponentReceipt>
where
    F: FnOnce(Canonical47GdnComponentReadback<'_>) -> Result<()>,
{
    match run(
        device,
        source,
        record,
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
    source: BoundFullModelSource,
    record: Arc<RecordedVllm08Model>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
    mode: RunMode<'_>,
) -> Result<Outcome>
where
    F: FnOnce(Canonical47GdnComponentReadback<'_>) -> Result<()>,
{
    let RunMode {
        arithmetic,
        fault,
        probe,
    } = mode;
    not_cancelled(cancelled)?;
    validate_canonical47_component_source(&source, &record, arithmetic)?;
    if fault != Some(Canonical47GdnComponentFault::AdmissionPressure) {
        crate::vllm_cuda_qualification::canonical47_projection::component::validate_component_images(
            &source, &record, &inputs,
        )?;
    }
    validate_provider(&device)?;
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "release canonical47 component required".into(),
        ));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source().source_identity(),
        ResourceOwner::new("original47-source-prefix", "unpublished-six-node-component")?,
        PROVIDER_BYTES,
        inputs.device_cap,
    )?;
    let physical_key = service.physical_key();
    let initial = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if initial.fenced || initial.reserved != PROVIDER_BYTES as u64 {
        return Err(Error::State(
            "canonical47 component requires solitary original envelope".into(),
        ));
    }
    if fault == Some(Canonical47GdnComponentFault::AdmissionPressure) {
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
    let program = ResidentProgram::prepare_canonical47_component(
        service.clone(),
        source.clone(),
        record.clone(),
        arithmetic,
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
    let session = model.session().await?;
    let base = session.binding();
    if base.frontier() != 0 {
        return Err(Error::State(
            "canonical47 component genuine initial root required".into(),
        ));
    }
    not_cancelled(cancelled)?;
    let candidate = model.allocate_state(false).await?;
    let invocation = record.call().request();
    let attempt = session.authority.begin_advance(invocation.rows() as u64)?;
    let schedule = GdnAttemptSchedule::new_initial_canonical47_component(
        &source,
        attempt.clone(),
        invocation.clone(),
        arithmetic,
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
            phase: crate::vllm_cuda_gdn::VllmGdnPhase::Initial47,
            nodes: [4, 5],
        },
        phase_binding.clone(),
    )?;
    body.prepare_gdn(domains.clone(), cancelled).await?;
    // Last real W upload is the native preparation guard. No managed owner is
    // allocated after these native records activate their original pool.
    let mut init = service.begin_initialization(base.root())?;
    init.retain_completion_owner(body.completion_owner())?;
    body.prepare_native(inputs, &mut init, cancelled)?;
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
    let intermediates = body.append_canonical47_diagnostics(&mut reads)?;
    let prior = domain_readbacks(&mut reads, &session.domains)?;
    let proposed = domain_readbacks(&mut reads, &candidate)?;
    let weights = body.immutable_weights()?;
    let immutable_weight_bytes = weights.iter().map(|(lease, _)| lease.bytes()).sum();
    let weight_start = reads.len();
    for (lease, _) in &weights {
        reads.push(lease.readback(0, lease.bytes())?);
    }
    let readback_bytes = reads.iter().map(ResidentReadback::bytes).sum();
    if readback_bytes > 128 * 1024 * 1024 {
        return Err(Error::Limit(
            "canonical47 original admitted128MiB readback cohort".into(),
        ));
    }
    let cohort = Arc::new(reads);
    body.arm_fault(match fault {
        Some(Canonical47GdnComponentFault::UnknownAfterConvolution) => 1,
        Some(Canonical47GdnComponentFault::CancelBeforeRecurrent) => 2,
        _ => 0,
    })?;
    if let Some(probe) = probe {
        probe.arm(&service, &body, &cohort, &accesses, base)?;
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
    let submitted = workspace.program.launch_attempt_fragment(
        service.client(),
        parameters,
        &bindings,
        &permit,
        cancelled,
    )?;
    if submitted.logits.is_some() || submitted.proposals.len() != 42 {
        return Err(Error::State(
            "canonical47 cut cannot publish logits/root or omit canonical proposals".into(),
        ));
    }
    let promoted = accesses
        .iter()
        .filter(|(_, access)| *access == DeviceAccess::Write)
        .map(|(l, _)| l.clone())
        .collect::<Vec<_>>();
    let (ready, actual) = flight
        .complete_with_readback(&promoted, &cohort, |raw| {
            not_cancelled(cancelled)?;
            body.validate_readbacks(raw)?;
            for (slot, candidate_span) in &proposed {
                let (_, prior_span) = prior
                    .iter()
                    .find(|(key, _)| key == slot)
                    .ok_or_else(|| Error::State("canonical47 carry prior absent".into()))?;
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
                        "canonical47 remaining40 canonical domains not exact prior copies".into(),
                    ));
                }
            }
            for (actual, (_, expected)) in raw[weight_start..].iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "canonical47 immutable source W/parameters changed".into(),
                    ));
                }
            }
            validate(Canonical47GdnComponentReadback {
                arithmetic,
                record: &record,
                raw,
                prior: &prior,
                candidate: &proposed,
                intermediates: &intermediates,
            })
        })
        .await?;
    drop(actual);
    drop(ready);
    slot.finish()?;
    if session.binding() != base || attempt.base() != base {
        return Err(Error::State(
            "canonical47 diagnostic advanced original root".into(),
        ));
    }
    let retire_accesses = weights
        .iter()
        .map(|(l, _)| (l.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
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
    let mut guard = service.submit(base.root(), &retire_accesses)?;
    guard.retain_completion_owner(body.completion_owner())?;
    unsafe {
        body.retire_after_physical_completion()?;
    }
    let rereads = weights
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let actual = guard
        .complete_read_only_readback(&rereads, |raw| {
            for (actual, (_, expected)) in raw.iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "canonical47 source W changed during guarded retirement".into(),
                    ));
                }
            }
            Ok(())
        })
        .await?;
    drop(actual);
    slot.finish()?;
    let (dispatches, native_launches, rows) = (
        submitted.dispatches,
        submitted.fused_dispatches,
        invocation.rows(),
    );
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
    let final_resident_bytes = service.resident_bytes()?;
    if final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "canonical47 original ledger did not retire to zero".into(),
        ));
    }
    drop(service);
    let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "canonical47 original physical envelope did not retire".into(),
        ));
    }
    Ok(Outcome::Compared(Canonical47GdnComponentReceipt {
        arithmetic,
        expected_trajectory: "ObservedOriginal47",
        original_selected_ptx_equals_owned_tuple: false,
        parent_model_accepted: false,
        rows,
        executed_nodes: 6,
        native_launches,
        dispatches,
        readback_bytes,
        immutable_weight_bytes,
        carried_domains: 40,
        original_root: base,
        model_root_published: false,
        final_resident_bytes,
        final_physical_reserved_bytes: physical.reserved,
    }))
}
