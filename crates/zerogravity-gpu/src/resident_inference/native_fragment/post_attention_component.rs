//! Source-owned eight-node cut on the original model/session/attempt/node loop.
//! Genuine all42 candidates are diagnostic; no RootReady/commit/model route.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule};
use super::*;
use crate::cuda::gdn::{FixedPromptGdnArithmetic, FixedPromptGdnIntermediate};
use crate::cuda::native::post_attention_projection::validate_images;
use crate::resident_inference::gdn_state::GdnPhasePlan;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::qwen2b_component::{
    ComponentCut, ComponentKind, ComponentProgram, ComponentRecipe,
    POST_ATTENTION_PROVIDER_BYTES as PROVIDER_BYTES, validate_component,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostAttentionArithmetic {
    Qwen2bOwned,
    Qwen2bExplicit(crate::cuda::gdn::Qwen2bGdnArithmetic),
    FixedPrompt(FixedPromptGdnArithmetic),
}
impl PostAttentionArithmetic {
    fn kind(self) -> ComponentKind {
        match self {
            Self::Qwen2bOwned | Self::Qwen2bExplicit(_) => ComponentKind::Qwen2b,
            Self::FixedPrompt(a) => ComponentKind::FixedPrompt(a),
        }
    }
    fn explicit_qwen2b(self) -> Option<crate::cuda::gdn::Qwen2bGdnArithmetic> {
        match self {
            Self::Qwen2bExplicit(arithmetic) => Some(arithmetic),
            Self::Qwen2bOwned | Self::FixedPrompt(_) => None,
        }
    }
    fn require_binding_parts(
        self,
        kind: ComponentKind,
        cut: ComponentCut,
        bound: Option<crate::cuda::gdn::Qwen2bGdnArithmetic>,
    ) -> Result<()> {
        if kind != self.kind()
            || cut != ComponentCut::PostAttention
            || bound != self.explicit_qwen2b()
        {
            return Err(Error::State(
                "postAttention actual source recipe/arithmetic differs".into(),
            ));
        }
        Ok(())
    }
    fn require_binding(self, recipe: &ComponentRecipe) -> Result<()> {
        self.require_binding_parts(recipe.kind(), recipe.cut(), recipe.qwen2b_arithmetic())
    }
}

use crate::resident_program::source_bindings::SourceBindings;

#[path = "post_attention_component/faults.rs"]
mod faults;
pub use faults::{
    PostAttentionComponentFault, PostAttentionComponentFaultOutcome,
    qualify_post_attention_component_fault,
};
#[derive(Clone, Debug)]
pub struct PostAttentionComponentReceipt {
    pub arithmetic: PostAttentionArithmetic,
    pub expected_trajectory: &'static str,
    pub original_selected_ptx_equals_owned_tuple: Option<bool>,
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
pub struct PostAttentionComponentReadback<'a> {
    arithmetic: PostAttentionArithmetic,
    record: &'a Arc<SourceBindings>,
    raw: &'a [Bytes],
    prior: &'a [(StateSlot, std::ops::Range<usize>)],
    candidate: &'a [(StateSlot, std::ops::Range<usize>)],
    intermediates: &'a [(FixedPromptGdnIntermediate, usize)],
}
impl PostAttentionComponentReadback<'_> {
    pub fn arithmetic(&self) -> PostAttentionArithmetic {
        self.arithmetic
    }
    pub fn intermediate(&self, key: FixedPromptGdnIntermediate) -> Result<&[u8]> {
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
    pub fn record(&self) -> &Arc<SourceBindings> {
        self.record
    }
    pub fn value(&self, id: ValueId) -> Result<&[u8]> {
        if id.0 >= 10 {
            return Err(Error::Unsupported(
                "postAttention cut has no value after postAttention".into(),
            ));
        }
        self.raw
            .get(2 + id.0 as usize)
            .map(|v| v.as_ref())
            .ok_or_else(|| Error::State("postAttention actual value absent".into()))
    }
    pub fn dual_sum_f32(&self) -> Result<&[u8]> {
        let bytes = self
            .raw
            .get(13)
            .ok_or_else(|| Error::State("completed unrounded dual sum absent".into()))?;
        if bytes.len() != self.record.call().rows().len() * self.arithmetic.kind().columns() * 4 {
            return Err(Error::State("unrounded dual sum dense F32 extent".into()));
        }
        Ok(&bytes[..])
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
            .ok_or_else(|| Error::State("postAttention actual canonical domain absent".into()))?;
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
    arithmetic: PostAttentionArithmetic,
    fault: Option<PostAttentionComponentFault>,
    probe: Option<&'a faults::Probe>,
}
enum Outcome {
    Compared(PostAttentionComponentReceipt),
    Pressure(PostAttentionComponentFaultOutcome),
}
/// Expected outputs are available ONLY inside original trusted completion.
pub async fn qualify_post_attention_component<F>(
    device: Device,
    recipe: Arc<ComponentRecipe>,
    arithmetic: PostAttentionArithmetic,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<PostAttentionComponentReceipt>
where
    F: FnOnce(PostAttentionComponentReadback<'_>) -> Result<()>,
{
    match run(
        device,
        recipe,
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
/// Additive typed owned2B path; expected data enters only the trusted callback.
/// The caller must independently close its original reference; no pin is created.
pub async fn qualify_qwen2b_post_attention_component_arithmetic<F>(
    device: Device,
    recipe: Arc<ComponentRecipe>,
    arithmetic: crate::cuda::gdn::Qwen2bGdnArithmetic,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<PostAttentionComponentReceipt>
where
    F: FnOnce(PostAttentionComponentReadback<'_>) -> Result<()>,
{
    qualify_post_attention_component(
        device,
        recipe,
        PostAttentionArithmetic::Qwen2bExplicit(arithmetic),
        inputs,
        cancelled,
        validate,
    )
    .await
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
    F: FnOnce(PostAttentionComponentReadback<'_>) -> Result<()>,
{
    let RunMode {
        arithmetic,
        fault,
        probe,
    } = mode;
    not_cancelled(cancelled)?;
    validate_component(&recipe)?;
    arithmetic.require_binding(&recipe)?;
    let source = recipe.source().clone();
    let record = recipe.bindings().clone();
    if fault != Some(PostAttentionComponentFault::AdmissionPressure) {
        validate_images(&recipe, &inputs)?;
    }
    validate_provider(&device)?;
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "release postAttention component required".into(),
        ));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("source-post-attention", "unpublished-eight-node-component")?,
        PROVIDER_BYTES,
        inputs.device_cap,
    )?;
    let physical_key = service.physical_key();
    let initial = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if initial.fenced || initial.reserved != PROVIDER_BYTES as u64 {
        return Err(Error::State(
            "postAttention component requires solitary original envelope".into(),
        ));
    }
    if fault == Some(PostAttentionComponentFault::AdmissionPressure) {
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
            "postAttention component genuine initial root required".into(),
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
    arithmetic.require_binding(body.recipe())?;
    let domains = session.bind_gdn_attempt_phase_state(
        &attempt,
        GdnPhasePlan {
            candidate: &candidate,
            plan: workspace.program.plan(),
            invocation,
            phase: match (arithmetic.kind(), invocation.rows().len()) {
                (ComponentKind::Qwen2b, 1) => crate::cuda::gdn::GdnPhase::InitialOne,
                (ComponentKind::Qwen2b, 64) => crate::cuda::gdn::GdnPhase::Initial64,
                _ => crate::cuda::gdn::GdnPhase::FixedPrompt,
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
    let intermediates = if matches!(arithmetic, PostAttentionArithmetic::FixedPrompt(_)) {
        body.append_fixed_prompt_diagnostics(&mut reads)?
    } else {
        Vec::new()
    };
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
            "postAttention original admitted128MiB readback cohort".into(),
        ));
    }
    let cohort = Arc::new(reads);
    body.arm_fault(match fault {
        Some(PostAttentionComponentFault::UnknownAfterOut) => 3,
        Some(PostAttentionComponentFault::CancelBeforePostAttentionRms) => 4,
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
            "postAttention cut cannot publish logits/root or omit canonical proposals".into(),
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
            if raw.len() != cohort.len() || raw.len() - weight_start != weights.len() {
                return Err(Error::State(
                    "postAttention exact completed readback cohort".into(),
                ));
            }
            arithmetic.require_binding(body.recipe())?;
            body.validate_readbacks(raw)?;
            for (slot, candidate_span) in &proposed {
                let (_, prior_span) = prior
                    .iter()
                    .find(|(key, _)| key == slot)
                    .ok_or_else(|| Error::State("postAttention carry prior absent".into()))?;
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
                        "postAttention remaining40 canonical domains not exact prior copies".into(),
                    ));
                }
            }
            for (actual, (_, expected)) in raw[weight_start..].iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "postAttention immutable source W/parameters changed".into(),
                    ));
                }
            }
            validate(PostAttentionComponentReadback {
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
            "postAttention diagnostic advanced original root".into(),
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
            if raw.len() != weights.len() {
                return Err(Error::State(
                    "postAttention exact immutable retirement readback arity".into(),
                ));
            }
            for (actual, (_, expected)) in raw.iter().zip(&weights) {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "postAttention source W changed during guarded retirement".into(),
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
            "postAttention original ledger did not retire to zero".into(),
        ));
    }
    drop(service);
    let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "postAttention original physical envelope did not retire".into(),
        ));
    }
    Ok(Outcome::Compared(PostAttentionComponentReceipt {
        arithmetic,
        expected_trajectory: match arithmetic {
            PostAttentionArithmetic::Qwen2bOwned | PostAttentionArithmetic::Qwen2bExplicit(_) => {
                "Original2B"
            }
            PostAttentionArithmetic::FixedPrompt(_) => "ObservedOriginal47",
        },
        original_selected_ptx_equals_owned_tuple: match arithmetic {
            PostAttentionArithmetic::Qwen2bOwned | PostAttentionArithmetic::Qwen2bExplicit(_) => {
                None
            }
            PostAttentionArithmetic::FixedPrompt(_) => Some(false),
        },
        parent_model_accepted: false,
        rows,
        executed_nodes: 8,
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

#[cfg(test)]
#[path = "post_attention_component/explicit2b_tests.rs"]
mod explicit2b_post_attention_tests;
