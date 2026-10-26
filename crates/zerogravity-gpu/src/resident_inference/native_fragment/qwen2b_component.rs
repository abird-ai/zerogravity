//! Source-owned six-node cut on the original model/session/attempt/node loop.
//! Genuine all42 candidates are diagnostic; no RootReady/commit/model route.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule};
use super::*;
use crate::resident_inference::gdn_state::GdnPhasePlan;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::qwen2b_component::{
    ComponentCut, ComponentKind, ComponentProgram, ComponentRecipe, PROVIDER_BYTES,
    validate_component,
};
use crate::resident_program::source_bindings::SourceBindings;

#[path = "qwen2b_component/faults.rs"]
mod faults;
pub use faults::{
    Qwen2bGdnComponentFault, Qwen2bGdnComponentFaultOutcome, qualify_qwen2b_gdn_component_fault,
};
#[derive(Clone, Debug)]
pub struct Qwen2bGdnComponentReceipt {
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
pub struct Qwen2bGdnComponentReadback<'a> {
    record: &'a Arc<SourceBindings>,
    raw: &'a [Bytes],
    prior: &'a [(StateSlot, std::ops::Range<usize>)],
    candidate: &'a [(StateSlot, std::ops::Range<usize>)],
    arithmetic: Option<crate::cuda::gdn::Qwen2bGdnArithmetic>,
}
impl Qwen2bGdnComponentReadback<'_> {
    pub fn arithmetic(&self) -> Option<crate::cuda::gdn::Qwen2bGdnArithmetic> {
        self.arithmetic
    }
    pub fn record(&self) -> &Arc<SourceBindings> {
        self.record
    }
    pub fn value(&self, id: ValueId) -> Result<&[u8]> {
        if id.0 >= 7 {
            return Err(Error::Unsupported(
                "source2B cut has no value after Gated".into(),
            ));
        }
        self.raw
            .get(2 + id.0 as usize)
            .map(|v| v.as_ref())
            .ok_or_else(|| Error::State("source2B actual value absent".into()))
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
            .ok_or_else(|| Error::State("source2B actual canonical domain absent".into()))?;
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
    fault: Option<Qwen2bGdnComponentFault>,
    probe: Option<&'a faults::Probe>,
}
enum Outcome {
    Compared(Qwen2bGdnComponentReceipt),
    Pressure(Qwen2bGdnComponentFaultOutcome),
}
/// Expected outputs are available ONLY inside original trusted completion.
pub async fn qualify_qwen2b_gdn_component<F>(
    device: Device,
    recipe: Arc<ComponentRecipe>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<Qwen2bGdnComponentReceipt>
where
    F: FnOnce(Qwen2bGdnComponentReadback<'_>) -> Result<()>,
{
    match run(
        device,
        recipe,
        inputs,
        cancelled,
        validate,
        RunMode {
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
/// Additive explicit arithmetic route; legacy entrypoints keep their original binding.
pub async fn qualify_qwen2b_gdn_component_arithmetic<F>(
    device: Device,
    recipe: Arc<ComponentRecipe>,
    arithmetic: crate::cuda::gdn::Qwen2bGdnArithmetic,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<Qwen2bGdnComponentReceipt>
where
    F: FnOnce(Qwen2bGdnComponentReadback<'_>) -> Result<()>,
{
    match run_with_arithmetic(
        device,
        recipe,
        ArithmeticInputs {
            arithmetic: Some(arithmetic),
            inputs,
        },
        cancelled,
        validate,
        RunMode {
            fault: None,
            probe: None,
        },
    )
    .await?
    {
        Outcome::Compared(receipt) => Ok(receipt),
        Outcome::Pressure(_) => Err(Error::State(
            "positive explicit2B tuple returned pressure".into(),
        )),
    }
}
async fn run<F>(
    device: Device,
    recipe: Arc<ComponentRecipe>,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
    mode: RunMode<'_>,
) -> Result<Outcome>
where
    F: FnOnce(Qwen2bGdnComponentReadback<'_>) -> Result<()>,
{
    run_with_arithmetic(
        device,
        recipe,
        ArithmeticInputs {
            arithmetic: None,
            inputs,
        },
        cancelled,
        validate,
        mode,
    )
    .await
}
struct ArithmeticInputs {
    arithmetic: Option<crate::cuda::gdn::Qwen2bGdnArithmetic>,
    inputs: CubinQualificationInputs,
}
async fn run_with_arithmetic<F>(
    device: Device,
    recipe: Arc<ComponentRecipe>,
    inputs: ArithmeticInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
    mode: RunMode<'_>,
) -> Result<Outcome>
where
    F: FnOnce(Qwen2bGdnComponentReadback<'_>) -> Result<()>,
{
    let RunMode { fault, probe } = mode;
    let ArithmeticInputs { arithmetic, inputs } = inputs;
    not_cancelled(cancelled)?;
    validate_component(&recipe)?;
    if recipe.cut() != ComponentCut::Gated
        || recipe.kind() != ComponentKind::Qwen2b
        || recipe.qwen2b_arithmetic() != arithmetic
    {
        return Err(Error::State(
            "source2B actual component cut/arithmetic differs".into(),
        ));
    }
    let source = recipe.source().clone();
    let record = recipe.bindings().clone();
    if let Some(choice) = arithmetic {
        choice.require_owned(if record.call().rows().len() == 1 {
            crate::cuda::gdn::GdnPhase::InitialOne
        } else {
            crate::cuda::gdn::GdnPhase::Initial64
        })?;
    }
    if fault != Some(Qwen2bGdnComponentFault::AdmissionPressure) {
        crate::cuda::native::qwen2b_component::validate_component_images(
            &source, &record, &inputs,
        )?;
    }
    validate_provider(&device)?;
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "release source2B component required".into(),
        ));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("original2B-source-prefix", "unpublished-six-node-component")?,
        PROVIDER_BYTES,
        inputs.device_cap,
    )?;
    let physical_key = service.physical_key();
    let initial = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if initial.fenced || initial.reserved != PROVIDER_BYTES as u64 {
        return Err(Error::State(
            "source2B component requires solitary original envelope".into(),
        ));
    }
    if fault == Some(Qwen2bGdnComponentFault::AdmissionPressure) {
        return faults::pressure(service).map(Outcome::Pressure);
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
    let program =
        ResidentProgram::prepare_component(service.clone(), recipe.clone(), cancelled).await?;
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
            "source2B component genuine initial root required".into(),
        ));
    }
    not_cancelled(cancelled)?;
    let candidate = model.allocate_state(false).await?;
    let invocation = record.call();
    let attempt = session
        .authority
        .begin_advance(invocation.rows().len() as u64)?;
    let schedule = GdnAttemptSchedule::new_initial_component(&recipe, attempt.clone())?;
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
            phase: if invocation.rows().len() == 1 {
                crate::cuda::gdn::GdnPhase::InitialOne
            } else {
                crate::cuda::gdn::GdnPhase::Initial64
            },
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
        token: invocation.rows().ids()[0],
        position: 0,
        valid_tokens: invocation.rows().len(),
    };
    workspace
        .program
        .validate_runtime(service.client(), parameters, &bindings)?;
    let mut reads = body.readbacks()?;
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
            "source2B original admitted128MiB readback cohort".into(),
        ));
    }
    let cohort = Arc::new(reads);
    body.arm_fault(match fault {
        Some(Qwen2bGdnComponentFault::UnknownAfterConvolution) => 1,
        Some(Qwen2bGdnComponentFault::CancelBeforeRecurrent) => 2,
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
            "source2B cut cannot publish logits/root or omit canonical proposals".into(),
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
                    .ok_or_else(|| Error::State("source2B carry prior absent".into()))?;
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
                        "source2B remaining40 canonical domains not exact prior copies".into(),
                    ));
                }
            }
            for (actual, (_, expected)) in raw[weight_start..].iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "source2B immutable source W/parameters changed".into(),
                    ));
                }
            }
            validate(Qwen2bGdnComponentReadback {
                record: &record,
                raw,
                prior: &prior,
                candidate: &proposed,
                arithmetic,
            })
        })
        .await?;
    drop(actual);
    drop(ready);
    slot.finish()?;
    if session.binding() != base || attempt.base() != base {
        return Err(Error::State(
            "source2B diagnostic advanced original root".into(),
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
                        "source2B source W changed during guarded retirement".into(),
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
        invocation.rows().len(),
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
            "source2B original ledger did not retire to zero".into(),
        ));
    }
    drop(service);
    let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "source2B original physical envelope did not retire".into(),
        ));
    }
    Ok(Outcome::Compared(Qwen2bGdnComponentReceipt {
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
