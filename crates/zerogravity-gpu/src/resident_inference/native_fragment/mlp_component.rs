//! Source-owned eleven-node cut on the original model/session/attempt/node loop.
//! Genuine all42 candidates are diagnostic; no RootReady/commit/model route.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule};
use super::*;
use crate::cuda::gdn::FixedPromptGdnIntermediate;
use crate::cuda::native::fixed_prompt_mlp::{FixedPromptMlpQualificationInputs, validate_inputs};
use crate::resident_inference::gdn_state::GdnPhasePlan;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::qwen2b_component::{
    ComponentCut, ComponentKind, ComponentProgram, ComponentRecipe,
    POST_ATTENTION_PROVIDER_BYTES as PROVIDER_BYTES, validate_component,
};

use super::post_attention_component::PostAttentionArithmetic;

use crate::resident_program::source_bindings::SourceBindings;

#[path = "mlp_component/faults.rs"]
mod faults;
pub use faults::{MlpComponentFault, MlpComponentFaultOutcome, qualify_mlp_component_fault};
#[derive(Clone, Debug)]
pub struct MlpComponentReceipt {
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
    pub streamed_weight_windows: usize,
    pub retirement_streamed_weight_windows: usize,
    pub carried_domains: usize,
    pub original_root: RootBinding,
    pub model_root_published: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}
pub struct MlpComponentReadback<'a> {
    kind: ComponentKind,
    arithmetic: PostAttentionArithmetic,
    record: &'a Arc<SourceBindings>,
    raw: &'a [Bytes],
    prior: &'a [(StateSlot, std::ops::Range<usize>)],
    candidate: &'a [(StateSlot, std::ops::Range<usize>)],
    intermediates: &'a [(FixedPromptGdnIntermediate, usize)],
}
fn mlp_readback_columns(
    kind: ComponentKind,
    arithmetic: PostAttentionArithmetic,
    rows: usize,
) -> Result<usize> {
    match (kind, arithmetic, rows) {
        (ComponentKind::FixedPrompt(bound), PostAttentionArithmetic::FixedPrompt(actual), 47)
            if bound == actual =>
        {
            Ok(1024)
        }
        (
            ComponentKind::Qwen2b,
            PostAttentionArithmetic::Qwen2bExplicit(
                crate::cuda::gdn::Qwen2bGdnArithmetic::OwnedSolve4Recompute3,
            ),
            1 | 64,
        ) => Ok(2048),
        _ => Err(Error::Unsupported(
            "finite actual MLP readback source/arithmetic/geometry".into(),
        )),
    }
}
impl<'a> MlpComponentReadback<'a> {
    pub(super) fn from_qwen2b_completed(
        record: &'a Arc<SourceBindings>,
        kind: ComponentKind,
        raw: &'a [Bytes],
        prior: &'a [(StateSlot, std::ops::Range<usize>)],
        candidate: &'a [(StateSlot, std::ops::Range<usize>)],
    ) -> Result<Self> {
        let value = Self {
            kind,
            arithmetic: PostAttentionArithmetic::Qwen2bExplicit(
                crate::cuda::gdn::Qwen2bGdnArithmetic::OwnedSolve4Recompute3,
            ),
            record,
            raw,
            prior,
            candidate,
            intermediates: &[],
        };
        value.readback_columns()?;
        Ok(value)
    }
    fn readback_columns(&self) -> Result<usize> {
        mlp_readback_columns(self.kind, self.arithmetic, self.record.call().rows().len())
    }
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
        if id.0 >= 13 {
            return Err(Error::Unsupported("MLP cut has no value after MLP".into()));
        }
        self.raw
            .get(2 + id.0 as usize)
            .map(|v| v.as_ref())
            .ok_or_else(|| Error::State("MLP actual value absent".into()))
    }
    pub fn dual_sum_f32(&self) -> Result<&[u8]> {
        let bytes = self
            .raw
            .get(16)
            .ok_or_else(|| Error::State("completed unrounded dual sum absent".into()))?;
        if bytes.len() != self.record.call().rows().len() * self.readback_columns()? * 4 {
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
            .ok_or_else(|| Error::State("MLP actual canonical domain absent".into()))?;
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
    fault: Option<MlpComponentFault>,
    probe: Option<&'a faults::Probe>,
}
enum Outcome {
    Compared(MlpComponentReceipt),
    Pressure(MlpComponentFaultOutcome),
}
/// Expected outputs are available ONLY inside original trusted completion.
pub async fn qualify_mlp_component<F>(
    device: Device,
    recipe: Arc<ComponentRecipe>,
    arithmetic: PostAttentionArithmetic,
    inputs: FixedPromptMlpQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<MlpComponentReceipt>
where
    F: FnOnce(MlpComponentReadback<'_>) -> Result<()>,
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
async fn run<F>(
    device: Device,
    recipe: Arc<ComponentRecipe>,
    inputs: FixedPromptMlpQualificationInputs,
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
    let owned = match arithmetic {
        PostAttentionArithmetic::FixedPrompt(a) => a,
        PostAttentionArithmetic::Qwen2bOwned | PostAttentionArithmetic::Qwen2bExplicit(_) => {
            return Err(Error::Unsupported(
                "MLP eleven-node component is canonical47 only".into(),
            ));
        }
    };
    validate_component(&recipe)?;
    if recipe.cut() != ComponentCut::Mlp
        || recipe.kind() != ComponentKind::FixedPrompt(owned)
        || recipe.qwen2b_arithmetic().is_some()
    {
        return Err(Error::State(
            "fixed prompt actual MLP cut/arithmetic differs".into(),
        ));
    }
    let source = recipe.source().clone();
    let record = recipe.bindings().clone();
    if fault != Some(MlpComponentFault::AdmissionPressure) {
        validate_inputs(&recipe, &inputs)?;
    }
    validate_provider(&device)?;
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported("release MLP component required".into()));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new(
            "source-canonical47-mlp",
            "unpublished-eleven-node-component",
        )?,
        PROVIDER_BYTES,
        inputs.projections.device_cap,
    )?;
    let physical_key = service.physical_key();
    let initial = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&physical_key)?;
    if initial.fenced || initial.reserved != PROVIDER_BYTES as u64 {
        return Err(Error::State(
            "MLP component requires solitary original envelope".into(),
        ));
    }
    if fault == Some(MlpComponentFault::AdmissionPressure) {
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
            "MLP component genuine initial root required".into(),
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
            phase: crate::cuda::gdn::GdnPhase::FixedPrompt,
            nodes: [4, 5],
        },
        phase_binding.clone(),
    )?;
    body.prepare_gdn(domains.clone(), cancelled).await?;
    body.prepare_mlp_weights(cancelled).await?;
    // Last real W upload is the native preparation guard. No managed owner is
    // allocated after these native records activate their original pool.
    let mut init = service.begin_initialization(base.root())?;
    init.retain_completion_owner(body.completion_owner())?;
    body.prepare_mlp_native(inputs, &mut init, cancelled)?;
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
    let immutable_weight_bytes = weights
        .iter()
        .map(|(lease, _)| lease.bytes())
        .sum::<usize>()
        + body.mlp_weight_bytes()?;
    let weight_start = reads.len();
    for (lease, _) in &weights {
        reads.push(lease.readback(0, lease.bytes())?);
    }
    let streamed_weight_reads = body.append_mlp_weight_readbacks(&mut reads)?;
    if streamed_weight_reads.len() != 2
        || streamed_weight_reads.start != weight_start + weights.len()
    {
        return Err(Error::State(
            "MLP exact two streamed W cohort descriptors".into(),
        ));
    }
    let readback_bytes = reads.iter().map(ResidentReadback::bytes).sum();
    if readback_bytes > 128 * 1024 * 1024 {
        return Err(Error::Limit(
            "MLP original admitted128MiB readback cohort".into(),
        ));
    }
    let cohort = Arc::new(reads);
    body.arm_fault(match fault {
        Some(MlpComponentFault::UnknownAfterGateUp) => 5,
        Some(MlpComponentFault::CancelBeforeActivation) => 6,
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
            "MLP cut cannot publish logits/root or omit canonical proposals".into(),
        ));
    }
    let promoted = accesses
        .iter()
        .filter(|(_, access)| *access == DeviceAccess::Write)
        .map(|(l, _)| l.clone())
        .collect::<Vec<_>>();
    let mut streamed_weight_windows = 0;
    let (ready, actual) = flight
        .complete_with_readback(&promoted, &cohort, |raw| {
            not_cancelled(cancelled)?;
            if raw.len() != cohort.len() || streamed_weight_reads.end != raw.len() {
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
            for (actual, (_, expected)) in raw[weight_start..streamed_weight_reads.start]
                .iter()
                .zip(&weights)
            {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "MLP immutable source W/parameters changed".into(),
                    ));
                }
            }
            streamed_weight_windows =
                body.validate_mlp_weight_readbacks(&raw[streamed_weight_reads.clone()], cancelled)?;
            if streamed_weight_windows != 22 {
                return Err(Error::State(
                    "MLP complete fourteen/eight fresh source windows required".into(),
                ));
            }
            validate(MlpComponentReadback {
                kind: recipe.kind(),
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
        return Err(Error::State("MLP diagnostic advanced original root".into()));
    }
    let mut retire_accesses = weights
        .iter()
        .map(|(l, _)| (l.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    merge_accesses(&mut retire_accesses, body.mlp_weight_accesses()?);
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
    let mut rereads = weights
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let retirement_streamed_reads = body.append_mlp_weight_readbacks(&mut rereads)?;
    if retirement_streamed_reads.start != weights.len() || retirement_streamed_reads.len() != 2 {
        return Err(Error::State(
            "MLP retirement exact two source W descriptors".into(),
        ));
    }
    let mut retirement_streamed_weight_windows = 0;
    let actual = guard
        .complete_read_only_readback(&rereads, |raw| {
            if raw.len() != retirement_streamed_reads.end {
                return Err(Error::State(
                    "MLP exact immutable retirement readback arity".into(),
                ));
            }
            for (actual, (_, expected)) in
                raw[..retirement_streamed_reads.start].iter().zip(&weights)
            {
                let actual: &[u8] = actual.as_ref();
                if actual != *expected {
                    return Err(Error::State(
                        "MLP source W changed during guarded retirement".into(),
                    ));
                }
            }
            retirement_streamed_weight_windows = body.validate_mlp_weight_readbacks(
                &raw[retirement_streamed_reads.clone()],
                cancelled,
            )?;
            if retirement_streamed_weight_windows != 22 {
                return Err(Error::State(
                    "MLP retirement full original source window coverage".into(),
                ));
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
    Ok(Outcome::Compared(MlpComponentReceipt {
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
        executed_nodes: 11,
        native_launches,
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

#[cfg(test)]
mod qwen2b_readback_scope_tests {
    use super::*;
    #[test]
    fn readback_width_requires_whole_explicit_source_family_and_geometry() {
        let owned = PostAttentionArithmetic::Qwen2bExplicit(
            crate::cuda::gdn::Qwen2bGdnArithmetic::OwnedSolve4Recompute3,
        );
        for rows in [1, 64] {
            assert_eq!(
                mlp_readback_columns(ComponentKind::Qwen2b, owned, rows).unwrap(),
                2048
            );
        }
        for rows in [0, 2, 47, 65] {
            assert!(mlp_readback_columns(ComponentKind::Qwen2b, owned, rows).is_err());
        }
        assert!(
            mlp_readback_columns(
                ComponentKind::FixedPrompt(
                    crate::cuda::gdn::FixedPromptGdnArithmetic::OwnedFixedPromptSolve4
                ),
                owned,
                1
            )
            .is_err()
        );
        assert!(
            mlp_readback_columns(
                ComponentKind::Qwen2b,
                PostAttentionArithmetic::Qwen2bOwned,
                1
            )
            .is_err()
        );
        let legacy = PostAttentionArithmetic::FixedPrompt(
            crate::cuda::gdn::FixedPromptGdnArithmetic::OwnedFixedPromptSolve4,
        );
        assert_eq!(
            mlp_readback_columns(
                ComponentKind::FixedPrompt(
                    crate::cuda::gdn::FixedPromptGdnArithmetic::OwnedFixedPromptSolve4
                ),
                legacy,
                47
            )
            .unwrap(),
            1024
        );
        assert!(mlp_readback_columns(ComponentKind::Qwen2b, legacy, 47).is_err());
    }
}
