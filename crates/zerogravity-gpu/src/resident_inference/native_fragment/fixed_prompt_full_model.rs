//! One unpublished Initial47 full record through genuine final RMS/full vocabulary.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule, PrefixPhaseEnqueue};
use super::fixed_prompt_model_prefix::FixedPromptQkKvQualificationInputs;
use super::*;
use crate::cuda::gdn::FixedPromptGdnArithmetic;
use crate::cuda::native::{
    attention_cubin::ModelAttentionParameters,
    fixed_prompt_model::FixedPromptModelBinding,
    fixed_prompt_model_native::{
        FixedPromptModelNativePrograms, attention,
        head::{self, FixedPromptHeadPrograms},
    },
    fixed_prompt_model_prefix::{FixedPromptPrefixCut, FixedPromptPrefixKind},
    model_cache::ModelCache,
    model_gdn::{ModelGdnParameters, ModelGdnPrepared},
    model_rotary::ModelRotary,
    model_weights::{ModelWeights, PendingModelWeights},
};
use crate::resident_leases::{
    ResidentReadback, TIED_BYTES, TIED_WINDOW_BYTES, TIED_WINDOWS, TiedWeight,
};
use crate::resident_program::source_bindings::SourceBindings;
use std::{collections::BTreeMap, ops::Range};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;
mod budget;
mod faults;
mod readback;
pub use faults::{
    FixedPromptFullModelFault, FixedPromptFullModelFaultOutcome,
    qualify_fixed_prompt_full_model_fault,
};
use faults::{Mode, Outcome};
pub use readback::FixedPromptFullModelReadback;
#[derive(Debug)]
pub struct FixedPromptFullModelReceipt {
    pub arithmetic: FixedPromptGdnArithmetic,
    pub executed_nodes: usize,
    pub computed_values_read_back: usize,
    pub native_launches: usize,
    pub dispatches: usize,
    pub readback_bytes: usize,
    pub readback_descriptors: usize,
    pub attention_native_enqueues: [[usize; 4]; 6],
    pub head_native_enqueues: usize,
    pub source_weight_windows: usize,
    pub retirement_source_weight_windows: usize,
    pub tied_weight_windows: usize,
    pub retirement_tied_weight_windows: usize,
    pub carried_domains: usize,
    pub original_root: RootBinding,
    pub model_root_published: bool,
    pub parent_model_accepted: bool,
    pub all327_values_numerically_compared: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}
// This complete owner belongs ONLY to original initialization/flight. No native/body backedge.
struct ColdOwners {
    body: Arc<crate::resident_program::native_full::FullModelProgram>,
    initial: Vec<Domain>,
    candidate: Vec<Domain>,
    _domains: Vec<Arc<super::model_state::BoundModelState>>,
    _gdn: BTreeMap<usize, Arc<ModelGdnPrepared>>,
    _caches: Vec<Arc<ModelCache>>,
    _parameters: BTreeMap<usize, Arc<ModelGdnParameters>>,
    _attention_parameters: BTreeMap<usize, Arc<ModelAttentionParameters>>,
    _positions: Arc<crate::resident_program::text_positions::TextPositionsPrepared>,
    _rotary: Arc<ModelRotary>,
    weights: Arc<ModelWeights>,
    tied: Arc<TiedWeight>,
    _workspace: ResidentLease,
}
struct PreparedOwners {
    base: RootBinding,
    cold: Arc<ColdOwners>,
    cut: Arc<FixedPromptPrefixCut>,
    native: Arc<FixedPromptModelNativePrograms>,
    attention: BTreeMap<usize, Arc<attention::Binding>>,
    head: Arc<FixedPromptHeadPrograms>,
    cohort: Arc<Vec<ResidentReadback>>,
    _metadata: WindowWorkspace,
}
pub async fn qualify_fixed_prompt_full_model<F>(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<FixedPromptFullModelReceipt>
where
    F: FnOnce(FixedPromptFullModelReadback<'_>) -> Result<()>,
{
    match qualify_inner(
        device,
        source,
        record,
        arithmetic,
        inputs,
        cancelled,
        Mode::Compare,
        validate,
    )
    .await?
    {
        Outcome::Compared(value) => Ok(*value),
        Outcome::Fault(_) => Err(Error::State("whole positive returned fault outcome".into())),
    }
}
#[expect(
    clippy::too_many_arguments,
    reason = "one original source/current-attempt/qualification flight"
)]
async fn qualify_inner<F>(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    cancelled: Option<&AtomicBool>,
    mode: Mode,
    validate: F,
) -> Result<Outcome>
where
    F: FnOnce(FixedPromptFullModelReadback<'_>) -> Result<()>,
{
    not_cancelled(cancelled)?;
    arithmetic.require_owned()?;
    if !source.same_issuer(record.source()) {
        return Err(Error::State(
            "fixed prompt actual source/graph owner differs".into(),
        ));
    }
    let scope = FixedPromptModelBinding::bind(record.clone())?;
    let cut = FixedPromptPrefixCut::bind_kind(scope.clone(), FixedPromptPrefixKind::WholePrefill)?;
    let provider = budget::require(&cut)?;
    validate_provider(&device)?;
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "release canonical47 whole model required".into(),
        ));
    }
    let FixedPromptQkKvQualificationInputs {
        native: inputs,
        rotary,
    } = inputs;
    let prefix = if mode.is_pressure() {
        None
    } else {
        head::validate_inputs_whole(&cut, &inputs)?;
        let prefix = rotary
            .ok_or_else(|| {
                Error::State("whole genuine completed rotary constructor absent".into())
            })?
            .into_fixed_prefix()?;
        prefix.require_source(&source)?;
        prefix.require_span(0, 47)?;
        Some(prefix)
    };
    let metadata = source.reserve_workspace(4 * 1024 * 1024)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new(
            "source-canonical47-full24",
            "unpublished-original-initial47",
        )?,
        provider,
        inputs.projections.device_cap,
    )?;
    if service.client().properties().memory.alignment != 512 {
        return Err(Error::Unsupported(
            "whole exact512-byte managed alignment".into(),
        ));
    }
    let key = service.physical_key();
    let arbiter = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared();
    let physical = arbiter.snapshot(&key)?;
    if physical.fenced || physical.reserved != provider as u64 {
        return Err(Error::State(
            "whole solitary original physical envelope".into(),
        ));
    }
    if mode.is_pressure() {
        return faults::pressure(service).map(|value| Outcome::Fault(Box::new(value)));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: provider as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let program = ResidentProgram::prepare_fixed_prompt_model_prefix_values(
        service.clone(),
        cut.clone(),
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
    let candidate = model.allocate_state(false).await?;
    let attempt = session.authority.begin_advance(47)?;
    let schedule = GdnAttemptSchedule::new_fixed_prompt_whole(&cut, attempt.clone())?;
    let phase = AttemptPhaseBinding::new(schedule.clone(), 0, None)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: provider as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut workspace = model.workspace.lock().await;
    let body = workspace.program.full_model_body()?.clone();
    let mut domains = Vec::with_capacity(24);
    for layer in 0..24 {
        domains.push(session.bind_fixed_prompt_whole_state(
            &attempt,
            &cut,
            layer,
            &candidate,
            phase.clone(),
        )?);
    }
    let mut caches = Vec::with_capacity(24);
    for domain in &domains {
        caches.push(
            ModelCache::prepare_fixed_prompt_whole(
                &cut,
                service.clone(),
                &attempt,
                domain.clone(),
                cancelled,
            )
            .await?,
        );
    }
    let mut init = service.begin_initialization(base.root())?;
    init.retain_completion_owner(Arc::new((
        body.clone(),
        session.domains.clone(),
        candidate.clone(),
        domains.clone(),
        caches.clone(),
        cut.clone(),
    )))?;
    let pending_tied = init
        .upload_tied_bf16(
            source.clone(),
            super::fixed_prompt_natural::tied_member(&record)?,
            cancelled,
        )
        .await?;
    let pending =
        PendingModelWeights::upload_fixed_prompt_prefix(&cut, &mut init, cancelled).await?;
    let mut parameters = BTreeMap::new();
    let mut attention_parameters = BTreeMap::new();
    for layer in 0..24 {
        if record.state_requirements().iter().any(|r| {
            r.slot == StateSlot::Kv(zerogravity_core::accelerator::program::KvDomain(layer))
        }) {
            attention_parameters.insert(
                layer,
                ModelAttentionParameters::prepare(
                    source.clone(),
                    record
                        .operator(ModelAttentionParameters::nodes(&record, layer)?[1])?
                        .ok_or_else(|| {
                            Error::State("whole actual QK source operator absent".into())
                        })?,
                    &service,
                    &mut init,
                    cancelled,
                )?,
            );
        } else {
            let pair = scope.gdn_site(layer)?.nodes();
            let op = |i: usize| {
                record
                    .operator(i)?
                    .ok_or_else(|| Error::State("whole actual GDN source descriptor".into()))
            };
            parameters.insert(
                layer,
                ModelGdnParameters::prepare(
                    source.clone(),
                    &service,
                    op(pair[0])?,
                    op(pair[1])?,
                    &mut init,
                    cancelled,
                )?,
            );
        }
    }
    let position_binding =
        session.bind_fixed_prompt_whole_text_positions(&cut, &attempt, phase.clone(), cancelled)?;
    let positions = crate::resident_program::text_positions::TextPositionsPrepared::prepare(
        position_binding,
        &mut init,
        cancelled,
    )?;
    let rotary = ModelRotary::prepare(
        source.clone(),
        service.clone(),
        prefix.ok_or_else(|| Error::State("whole rotary prefix absent".into()))?,
        &mut init,
    )?;
    drop(init.complete_sync().await?);
    let weights = pending.bind_initialized()?;
    let tied = pending_tied.bind_initialized()?;
    let mut gdn = BTreeMap::new();
    for (&layer, parameter) in &parameters {
        gdn.insert(
            layer,
            ModelGdnPrepared::prepare_fixed_prompt_whole(
                &cut,
                body.values().clone(),
                parameter.clone(),
                domains[layer].clone(),
                caches[layer].clone(),
                arithmetic,
                cancelled,
            )
            .await?,
        );
    }
    let mut init = service.begin_initialization(base.root())?;
    let native_workspace = init.allocate_empty(8_519_680, BufferRole::Scratch)?;
    let cold = Arc::new(ColdOwners {
        body: body.clone(),
        initial: session.domains.clone(),
        candidate: candidate.clone(),
        _domains: domains.clone(),
        _gdn: gdn.clone(),
        _caches: caches.clone(),
        _parameters: parameters.clone(),
        _attention_parameters: attention_parameters.clone(),
        _positions: positions.clone(),
        _rotary: rotary.clone(),
        weights: weights.clone(),
        tied: tied.clone(),
        _workspace: native_workspace.clone(),
    });
    init.retain_completion_owner(cold.clone())?;
    let mut attention = BTreeMap::new();
    for (&layer, parameter) in &attention_parameters {
        attention.insert(
            layer,
            attention::Binding::prepare_whole(
                attention::Inputs {
                    cut: cut.clone(),
                    values: body.values().clone(),
                    parameters: parameter.clone(),
                    positions: positions.clone(),
                    rotary: rotary.clone(),
                    domains: domains[layer].clone(),
                    cache: caches[layer].clone(),
                },
                layer,
                &mut init,
                cancelled,
            )?,
        );
    }
    let images = head::prepare_images_whole(&cut, &service, inputs, &mut init, cancelled)?;
    let head = FixedPromptHeadPrograms::prepare(
        cut.clone(),
        body.values().clone(),
        tied.clone(),
        body.final_normalizer()?,
        images.clone(),
        &mut init,
        cancelled,
    )?;
    let native = FixedPromptModelNativePrograms::prepare_whole(
        cut.clone(),
        body.values().clone(),
        weights.clone(),
        native_workspace.clone(),
        images.clone(),
        attention.clone(),
        head.clone(),
        &mut init,
        cancelled,
    )?;
    body.bind_fixed_prompt_whole(
        cut.clone(),
        native.clone(),
        domains.clone(),
        gdn.clone(),
        arithmetic,
    )?;
    drop(init.complete_sync().await?);
    let state_ids = session
        .domains
        .iter()
        .chain(&candidate)
        .flat_map(|d| d.leases.iter().map(ResidentLease::id))
        .collect::<BTreeSet<_>>();
    workspace.leases = workspace
        .program
        .finalize_full_model_preflight(&state_ids)?;
    slot.finish()?;
    let bindings = bind_domains(&session.domains, &candidate, 0)?;
    let mut accesses = body.accesses()?.to_vec();
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
    let runtime = RuntimeParameters {
        token: record.call().rows().ids()[0],
        position: 0,
        valid_tokens: 47,
    };
    workspace
        .program
        .validate_runtime(service.client(), runtime, &bindings)?;
    let mut reads = body.readbacks()?;
    let dual_nodes = body.fused().keys().copied().collect::<Vec<_>>();
    let layout = readback::Layout::new(reads.len(), gdn.len(), cut.values().len(), &dual_nodes)?;
    reads.push(positions.readback()?);
    for binding in attention.values() {
        let mut extra = binding.extra_readbacks()?;
        if extra.len() != 2 {
            return Err(Error::State("whole exact per-layer KV status".into()));
        }
        reads.push(
            extra
                .pop()
                .ok_or_else(|| Error::State("whole KV status missing".into()))?,
        );
    }
    let prior = domain_readbacks(&mut reads, &session.domains)?;
    let proposed = domain_readbacks(&mut reads, &candidate)?;
    let parameter_start = reads.len();
    let mut immutable = BTreeMap::new();
    for parameter in parameters.values() {
        for (lease, bytes) in parameter.immutable_weights() {
            immutable.insert(lease.id(), (lease, bytes));
        }
    }
    for binding in attention.values() {
        for (lease, bytes) in binding.immutable_weights() {
            if let Some((old, previous)) = immutable.get(&lease.id()) {
                if old.allocation_generation() != lease.allocation_generation()
                    || *previous != bytes
                {
                    return Err(Error::State("whole immutable shared owner differs".into()));
                }
            } else {
                immutable.insert(lease.id(), (lease, bytes));
            }
        }
    }
    let immutable = immutable.into_values().collect::<Vec<_>>();
    for (lease, _) in &immutable {
        reads.push(lease.readback(0, lease.bytes())?);
    }
    let readback_bytes = reads.iter().map(ResidentReadback::bytes).sum::<usize>();
    body.require_readback_capacity(readback_bytes)?;
    if readback_bytes != budget::MATH_BYTES
        || reads.len() != budget::DESCRIPTORS
        || service.resident_bytes()? > provider
    {
        return Err(Error::Limit(
            "whole actual descriptor/device/D2H source bound differs".into(),
        ));
    }
    let cohort = Arc::new(reads);
    let owners = Arc::new(PreparedOwners {
        base,
        cold,
        cut: cut.clone(),
        native: native.clone(),
        attention: attention.clone(),
        head: head.clone(),
        cohort: cohort.clone(),
        _metadata: metadata,
    });
    mode.arm(&owners, &service, &accesses, base)?;
    body.arm_fixed_prompt_whole_fault(mode.code())?;
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
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let permit = PrefixPhaseEnqueue::new(&session, &schedule, 0, None)?;
    let mut flight = service.submit(base.root(), &accesses)?;
    flight.retain_completion_owner(owners.clone())?;
    let submitted = workspace.program.launch_attempt_fragment(
        service.client(),
        runtime,
        &bindings,
        &permit,
        cancelled,
    )?;
    if !matches!(mode, Mode::Compare) {
        return Err(Error::State("whole intended fault cut not reached".into()));
    }
    if submitted.logits.is_some() || submitted.proposals.len() != 42 {
        return Err(Error::State(
            "whole unpublished original complete42 proposals".into(),
        ));
    }
    let promoted = accesses
        .iter()
        .filter(|(_, a)| *a == DeviceAccess::Write)
        .map(|(l, _)| l.clone())
        .collect::<Vec<_>>();
    let (ready, actual) = flight
        .complete_with_readback(&promoted, &cohort, |raw| {
            not_cancelled(cancelled)?;
            if raw.len() != cohort.len() {
                return Err(Error::State("whole complete actual cohort".into()));
            }
            body.validate_readbacks(&raw[..layout.body])?;
            positions.validate_readback(&raw[layout.positions..layout.positions + 1])?;
            for (_, index) in &layout.kv {
                crate::cuda::native::model_kv_transport::ModelKvTransport::validate_status(
                    &raw[*index],
                )?;
            }
            for (_, span) in &prior {
                if raw[span.clone()].iter().any(|v| v.iter().any(|b| *b != 0)) {
                    return Err(Error::State(
                        "whole genuine all42 fresh priors not zero".into(),
                    ));
                }
            }
            for ((slot, a), (other, b)) in prior.iter().zip(&proposed) {
                if slot != other {
                    return Err(Error::State("whole actual canonical domain order".into()));
                }
                if matches!(slot, StateSlot::Kv(_)) {
                    for (p, c) in raw[a.clone()].iter().zip(&raw[b.clone()]) {
                        if p.len() != 512 * 512 * 4
                            || c.len() != p.len()
                            || p[47 * 512 * 4..] != c[47 * 512 * 4..]
                        {
                            return Err(Error::State(
                                "whole canonical future KV rows changed".into(),
                            ));
                        }
                    }
                }
            }
            if raw[parameter_start..].len() != immutable.len()
                || raw[parameter_start..]
                    .iter()
                    .zip(&immutable)
                    .any(|(actual, (_, expected))| &actual[..] != *expected)
            {
                return Err(Error::State(
                    "whole source parameters/controls changed".into(),
                ));
            }
            validate(FixedPromptFullModelReadback {
                record: &record,
                raw,
                prior: &prior,
                candidate: &proposed,
                layout: &layout,
                dual_nodes: &dual_nodes,
            })
        })
        .await?;
    drop(actual);
    drop(ready);
    slot.finish()?;
    if session.binding() != base {
        return Err(Error::State(
            "whole diagnostic published original root".into(),
        ));
    }
    let source_weight_windows = weights
        .verify_source_windows(owners.clone(), cancelled)
        .await?
        .0;
    let tied_weight_windows = verify_tied_windows(&service, &owners, cancelled).await?;
    let mut retire_accesses = weights.accesses();
    retire_accesses.push((tied.lease().clone(), DeviceAccess::Read));
    merge_accesses(
        &mut retire_accesses,
        immutable
            .iter()
            .map(|(lease, _)| (lease.clone(), DeviceAccess::Read)),
    );
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
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut guard = service.submit(base.root(), &retire_accesses)?;
    guard.retain_completion_owner(owners.clone())?;
    // SAFETY: same completed initialized W/source guard owns the entire native graph.
    unsafe {
        native.retire_after_physical_completion()?;
    }
    let rereads = immutable
        .iter()
        .map(|(lease, _)| lease.readback(0, lease.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let raw = guard
        .complete_read_only_readback(&rereads, |raw| {
            if raw.len() != immutable.len()
                || raw.iter().zip(&immutable).any(|(a, (_, b))| &a[..] != *b)
            {
                return Err(Error::State("whole source immutability at unload".into()));
            }
            Ok(())
        })
        .await?;
    drop(raw);
    slot.finish()?;
    let retirement_source_weight_windows = weights
        .verify_source_windows(owners.clone(), cancelled)
        .await?
        .0;
    let retirement_tied_weight_windows = verify_tied_windows(&service, &owners, cancelled).await?;
    let (dispatches, native_launches) = (submitted.dispatches, body.native_launches());
    let attention_native_enqueues = native.whole_counts()?;
    let head_native_enqueues = head.enqueues();
    drop(submitted);
    drop(promoted);
    drop(cohort);
    drop(rereads);
    drop(retire_accesses);
    drop(immutable);
    drop(accesses);
    drop(bindings);
    drop(attention);
    drop(head);
    drop(native);
    drop(images);
    drop(native_workspace);
    drop(tied);
    drop(weights);
    drop(gdn);
    drop(parameters);
    drop(attention_parameters);
    drop(caches);
    drop(domains);
    drop(positions);
    drop(rotary);
    drop(phase);
    drop(schedule);
    drop(body);
    drop(workspace);
    drop(candidate);
    drop(attempt);
    drop(session);
    drop(model);
    drop(cut);
    drop(scope);
    drop(record);
    drop(source);
    drop(owners);
    let final_resident_bytes = service.resident_bytes()?;
    if final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "whole actual original ledger not retired".into(),
        ));
    }
    drop(service);
    let physical = arbiter.snapshot(&key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "whole actual original physical envelope not retired".into(),
        ));
    }
    Ok(Outcome::Compared(Box::new(FixedPromptFullModelReceipt {
        arithmetic,
        executed_nodes: 255,
        computed_values_read_back: 327,
        native_launches,
        dispatches,
        readback_bytes,
        readback_descriptors: budget::DESCRIPTORS,
        attention_native_enqueues,
        head_native_enqueues,
        source_weight_windows,
        retirement_source_weight_windows,
        tied_weight_windows,
        retirement_tied_weight_windows,
        carried_domains: 0,
        original_root: base,
        model_root_published: false,
        parent_model_accepted: false,
        all327_values_numerically_compared: false,
        final_resident_bytes,
        final_physical_reserved_bytes: physical.reserved,
    })))
}
async fn verify_tied_windows(
    service: &Arc<ResidentService>,
    owners: &Arc<PreparedOwners>,
    cancelled: Option<&AtomicBool>,
) -> Result<usize> {
    let tied = &owners.cold.tied;
    let source = owners.cut.scope().source();
    if !tied.source().same_issuer(source)
        || tied.copy_counts()? != (TIED_WINDOWS, TIED_WINDOWS)
        || tied.lease().bytes() != TIED_BYTES
    {
        return Err(Error::State(
            "whole original tied upload coverage/source owner".into(),
        ));
    }
    let descriptor = super::fixed_prompt_natural::tied_member(owners.cut.scope().record())?;
    let matrix = source.matrix(&descriptor.name, descriptor.axes)?;
    tied.matrix().require_matrix(source, service, &matrix)?;
    if matrix.rows() != 248320 || matrix.columns() != 1024 {
        return Err(Error::State("whole original tied table geometry".into()));
    }
    let admission = Arc::new(source.reserve_workspace(32 * 1024 * 1024)?);
    for index in 0..TIED_WINDOWS {
        not_cancelled(cancelled)?;
        let first = index * 512;
        let window = Arc::new(match cancelled {
            Some(flag) => source.read_rows_cancellable(&matrix, first..first + 512, flag)?,
            None => source.read_rows(&matrix, first..first + 512)?,
        });
        if window.as_slice().len() != TIED_WINDOW_BYTES
            || window.row_range() != (first..first + 512)
        {
            return Err(Error::State(
                "whole exact independent tied source window".into(),
            ));
        }
        let accesses = [(tied.lease().clone(), DeviceAccess::Read)];
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: 32 * 1024 * 1024,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        not_cancelled(cancelled)?;
        slot.begin_submission()?;
        let mut flight = service.submit(owners.base.root(), &accesses)?;
        flight.retain_completion_owner(owners.clone())?;
        flight.retain_completion_owner(admission.clone())?;
        flight.retain_completion_owner(window.clone())?;
        let reads = [tied
            .lease()
            .readback(index * TIED_WINDOW_BYTES, TIED_WINDOW_BYTES)?];
        let raw = flight
            .complete_read_only_readback(&reads, |raw| {
                if raw.len() != 1 || &raw[0][..] != window.as_slice() {
                    return Err(Error::State("whole complete tied table changed".into()));
                }
                Ok(())
            })
            .await?;
        drop(raw);
        slot.finish()?;
    }
    Ok(TIED_WINDOWS)
}
