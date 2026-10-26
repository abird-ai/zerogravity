//! Five source-produced full24 phases, through the original session/workspace.
//! Diagnostic completion only: no Ready publisher/frontier/serving capability.
use super::continuation::{
    AttemptPhaseBinding, CompletedPrefixPhase, GdnAttemptSchedule, PlannedPrefixPrior,
    PrefixPhaseEnqueue,
};
use super::*;
use crate::cuda::native::{
    CubinActivationImage, CubinQualificationInputs,
    attention_cubin::{ModelAttentionBinding, ModelAttentionInputs, ModelAttentionParameters},
    full_model_budget::provider_bound,
    full_model_images::prepare_full_model_images,
    model_cache::ModelCache,
    model_gdn::{ModelGdnParameters, ModelGdnPrepared},
    model_library::FullModelFactoryInputs,
    model_rotary::ModelRotary,
    model_weights::{ModelWeights, PendingModelWeights},
    source_rotary::RotaryOutcome,
};
use crate::resident_leases::{ResidentReadback, TiedWeight};
use crate::resident_program::native_full::{FullModelLibrary, FullModelProgram};
use crate::resident_program::text_positions::TextPositionsPrepared;
use std::collections::BTreeMap;

use crate::resident_program::source_bindings::SourceBindings;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::{KvDomain, SourceOperatorKind, SourceTensor};
mod scopes;
pub use scopes::SourceModelScopes;

mod faults;
use faults::{Mode, Outcome};
pub use faults::{SourceModelFault, SourceModelFaultOutcome, qualify_source_model_fault};
pub(crate) mod natural_cold;
struct Execution<'a> {
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&'a AtomicBool>,
    mode: Mode,
    natural: bool,
    reusable: bool,
}
pub struct SourceModelInputs {
    pub projections: CubinQualificationInputs,
    pub activation: CubinActivationImage,
}
pub use super::source_readback::{SourceDomainReadback, SourceModelReadback};
#[derive(Debug)]
pub struct SourceModelReceipt {
    pub phases: usize,
    pub native_launches: usize,
    pub workspace_clears: usize,
    pub device_copies: usize,
    pub dispatches: usize,
    pub source_weight_windows: usize,
    pub tied_weight_windows: usize,
    pub original_roots: [RootBinding; 2],
    pub model_root_published: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}
struct Frame {
    reads: Arc<Vec<ResidentReadback>>,
    prior: Vec<(StateSlot, std::ops::Range<usize>)>,
    candidate: Vec<(StateSlot, std::ops::Range<usize>)>,
    rms_nodes: Vec<usize>,
}
/// Positive and unknown submissions use this SAME real source/prepared graph.
/// No body/library/previous completion points back to this collection owner.
struct ColdOwners {
    _bodies: Vec<Arc<FullModelProgram>>,
    _states: Vec<Vec<Arc<super::model_state::BoundModelState>>>,
    _gdn: Vec<BTreeMap<usize, Arc<ModelGdnPrepared>>>,
    _attention: Vec<BTreeMap<usize, Arc<ModelAttentionBinding>>>,
    _caches: Vec<Vec<Arc<ModelCache>>>,
    _initial: [Vec<Domain>; 2],
    _candidates: Vec<Vec<Domain>>,
    _parameters: Vec<Arc<ModelGdnParameters>>,
    _attention_parameters: Vec<Arc<ModelAttentionParameters>>,
    _positions: Vec<Arc<TextPositionsPrepared>>,
    _rotary: Arc<ModelRotary>,
    _weights: Arc<ModelWeights>,
    _tied: Arc<TiedWeight>,
    _metadata: zerogravity_artifacts::source_weights::WindowWorkspace,
}
struct FullOwners {
    bodies: Vec<Arc<FullModelProgram>>,
    frames: Vec<Frame>,
    candidates: Vec<Vec<Domain>>,
    initial: [Vec<Domain>; 2],
    _source: VerifiedTensorSource,
    _parameters: Vec<Arc<ModelGdnParameters>>,
    _attention_parameters: Vec<Arc<ModelAttentionParameters>>,
    _positions: Vec<Arc<TextPositionsPrepared>>,
    _rotary: Arc<ModelRotary>,
    library: Arc<FullModelLibrary>,
    weights: Arc<ModelWeights>,
    tied: Arc<TiedWeight>,
    // Original control/metadata admission drops after every retained graph.
    _cold: Arc<ColdOwners>,
}
fn history(phase: usize) -> usize {
    usize::from(phase >= 2)
}
fn local_phase(phase: usize) -> usize {
    if phase < 2 { phase } else { phase - 2 }
}
fn node_pair(record: &SourceBindings, layer: usize) -> Result<[usize; 2]> {
    let mut pair = [None, None];
    for (index, _node) in record.plan().nodes().iter().enumerate() {
        let Some(descriptor) = &record.operator(index)? else {
            continue;
        };
        let slot = match descriptor.kind() {
            SourceOperatorKind::CausalConvSplit { domain, .. } if domain.0 == layer => 0,
            SourceOperatorKind::GatedDelta { domain, .. } if domain.0 == layer => 1,
            _ => continue,
        };
        if pair[slot].replace(index).is_some() {
            return Err(Error::State(
                "full24 one indexed Conv/recurrent pair".into(),
            ));
        }
    }
    Ok([
        pair[0].ok_or_else(|| Error::State("full24 indexed Conv absent".into()))?,
        pair[1].ok_or_else(|| Error::State("full24 indexed recurrent absent".into()))?,
    ])
}

fn attention_layer(record: &SourceBindings, layer: usize) -> bool {
    record
        .state_requirements()
        .iter()
        .any(|state| state.slot == StateSlot::Kv(KvDomain(layer)))
}

fn head_member(record: &SourceBindings) -> Result<&SourceTensor> {
    let at = record
        .plan()
        .source_value_provenance(record.selection().output())?
        .ok_or_else(|| Error::State("full24 actual head producer absent".into()))?
        .node_index();
    let op = record
        .operator(at)?
        .ok_or_else(|| Error::State("full24 actual head source absent".into()))?;
    let [matrix] = op.matrices() else {
        return Err(Error::State("full24 exact head member arity".into()));
    };
    if !matches!(
        op.kind(),
        SourceOperatorKind::Projection {
            selected_row: Some(_)
        }
    ) {
        return Err(Error::State(
            "full24 selected head projection absent".into(),
        ));
    }
    Ok(matrix)
}
fn qk_operator(
    record: &SourceBindings,
    layer: usize,
) -> Result<&zerogravity_core::accelerator::program::SourceOperator> {
    for (index, node) in record.plan().nodes().iter().enumerate() {
        let Some(op) = record.operator(index)? else {
            continue;
        };
        if matches!(op.kind(),SourceOperatorKind::KvAppend{domain,..} if domain==KvDomain(layer)) {
            let query = *node
                .inputs
                .first()
                .ok_or_else(|| Error::State("full24 KV query SSA absent".into()))?;
            let producer = record
                .plan()
                .source_value_provenance(query)?
                .ok_or_else(|| Error::State("full24 actual QK producer absent".into()))?
                .node_index();
            let op = record
                .operator(producer)?
                .ok_or_else(|| Error::State("full24 QK descriptor absent".into()))?;
            if !matches!(op.kind(), SourceOperatorKind::QkRmsRotaryGated(_)) {
                return Err(Error::State("full24 exact QK producer".into()));
            }
            return Ok(op);
        }
    }
    Err(Error::State("full24 attention domain absent".into()))
}

/// Expected archives may be opened only by `validate`, invoked inside the
/// original trusted completion AFTER real producer arithmetic and actual D2H.
/// `rotary` is a genuine completed original constructor result, not user bytes.
pub async fn qualify_source_model<F>(
    device: Device,
    scopes: SourceModelScopes,
    inputs: SourceModelInputs,
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<SourceModelReceipt>
where
    F: FnMut(SourceModelReadback<'_>) -> Result<()>,
{
    let (source, records) = scopes.into_parts();
    match qualify_inner(
        device,
        source,
        records,
        inputs,
        Execution {
            rotary,
            cancelled,
            mode: Mode::Compare,
            natural: false,
            reusable: false,
        },
        validate,
    )
    .await?
    {
        Outcome::Compared(value) => Ok(value),
        Outcome::Natural(_) => Err(Error::State(
            "forced model qualifier returned natural preparation".into(),
        )),
        Outcome::Fault(_) => Err(Error::State(
            "full24 positive returned diagnostic fault".into(),
        )),
    }
}
async fn qualify_inner<F>(
    device: Device,
    source: VerifiedTensorSource,
    records: [Arc<SourceBindings>; 5],
    inputs: SourceModelInputs,
    execution: Execution<'_>,
    mut validate: F,
) -> Result<Outcome>
where
    F: FnMut(SourceModelReadback<'_>) -> Result<()>,
{
    let Execution {
        rotary,
        cancelled,
        mode,
        natural,
        reusable,
    } = execution;
    not_cancelled(cancelled)?;
    SourceModelScopes::require(&source, &records)?;
    let natural_metadata = if natural {
        Some(source.reserve_workspace(1024 * 1024)?)
    } else {
        None
    };
    let metadata = source.reserve_workspace(4 * 1024 * 1024)?;
    let records = records.into_iter().collect::<Vec<_>>();
    let cap = if reusable {
        if !natural {
            return Err(Error::State(
                "reusable preparation requires genuine natural cold mode".into(),
            ));
        }
        crate::cuda::native::full_model_budget::reusable_provider_bound(&source, &records)?
    } else {
        provider_bound(&source, &records)?
    };
    validate_provider(&device)?;
    let prefix = if mode.is_pressure() {
        None
    } else {
        let prefix = rotary
            .ok_or_else(|| {
                Error::Unsupported(
                    "full24 requires genuinely completed original rotary producer".into(),
                )
            })?
            .into_prefix()?;
        prefix.require_source(&source)?;
        Some(prefix)
    };
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner {
            model: format!("original-hf-full24-{:?}", source.source_identity()),
            session: "unpublished-forced-five-phase-full24".into(),
        },
        cap,
        inputs.projections.device_cap,
    )?;
    if reusable && service.client().properties().memory.alignment != 512 {
        return Err(Error::Unsupported(
            "reusable packet bound requires original512B allocation alignment".into(),
        ));
    }
    let key = service.physical_key();
    let arbiter = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared();
    let first_snapshot = arbiter.snapshot(&key)?;
    if first_snapshot.fenced || first_snapshot.reserved != cap as u64 {
        return Err(Error::State(
            "full24 original solitary physical envelope".into(),
        ));
    }
    if mode.is_pressure() {
        return faults::pressure(service).map(Outcome::Fault);
    }
    let prefix =
        prefix.ok_or_else(|| Error::State("full24 positive rotary prefix absent".into()))?;
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
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    // Actual original cold cleanup after retired rotary, before learned/model
    // allocations. Logical producer0 never substitutes for allocator drainage.
    #[cfg(feature = "qualification")]
    crate::gpu_telemetry::allocator_report(service.client(), "full24_before_cold_cleanup");
    let cleanup_work = service.budget().begin(service.client())?;
    if let Err(error) = service.client().memory_cleanup() {
        drop(cleanup_work);
        service.fence_and_retain_owners(&[], Vec::new());
        return Err(Error::State(format!(
            "full24 original cold allocator cleanup: {error}"
        )));
    }
    cleanup_work.complete()?;
    #[cfg(feature = "qualification")]
    crate::gpu_telemetry::allocator_report(service.client(), "full24_after_cold_cleanup");
    let mut programs = Vec::with_capacity(5);
    for record in &records {
        programs.push(Some(
            ResidentProgram::prepare_full_model_values(service.clone(), record.clone(), cancelled)
                .await?,
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
    let sessions = [model.session().await?, model.session().await?];
    let bases = [sessions[0].binding(), sessions[1].binding()];
    let attempts = [
        sessions[0].authority.begin_advance(2)?,
        sessions[1].authority.begin_advance(67)?,
    ];
    let schedules = [
        GdnAttemptSchedule::new_model(
            &source,
            bases[0],
            attempts[0].clone(),
            records[..2].iter().map(|r| r.call().clone()).collect(),
        )?,
        GdnAttemptSchedule::new_model(
            &source,
            bases[1],
            attempts[1].clone(),
            records[2..].iter().map(|r| r.call().clone()).collect(),
        )?,
    ];
    let mut candidates = Vec::with_capacity(5);
    for _ in 0..5 {
        candidates.push(model.allocate_state(false).await?);
    }
    let mut planned = Vec::with_capacity(5);
    let mut phases = Vec::with_capacity(5);
    for index in 0..5 {
        let h = history(index);
        let local = local_phase(index);
        let prior = if local == 0 {
            None
        } else {
            Some(PlannedPrefixPrior::new(
                &sessions[h],
                schedules[h].clone(),
                local,
                candidates[index - 1].clone(),
            )?)
        };
        phases.push(AttemptPhaseBinding::new(
            schedules[h].clone(),
            local,
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
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut workspace = model.workspace.lock().await;
    let mut bodies = Vec::with_capacity(5);
    for (index, program) in programs.iter().enumerate() {
        bodies.push(
            if index == 0 {
                &workspace.program
            } else {
                program.as_ref().unwrap()
            }
            .full_model_body()?
            .clone(),
        );
    }
    let mut domains = Vec::with_capacity(5);
    for index in 0..5 {
        let h = history(index);
        let mut bound = Vec::with_capacity(24);
        for layer in 0..24 {
            bound.push(sessions[h].bind_full_model_state(
                &attempts[h],
                &records[index],
                layer,
                &candidates[index],
                phases[index].clone(),
            )?);
        }
        domains.push(bound);
    }
    // At most one bounded zero/upload staging payload survives each actual
    // completion; no native module is active during this managed preparation.
    let mut caches = Vec::with_capacity(2);
    for first in [0, 2] {
        let h = history(first);
        let mut per_layer = Vec::with_capacity(24);
        for state in &domains[first] {
            per_layer.push(
                ModelCache::prepare(
                    source.clone(),
                    service.clone(),
                    &attempts[h],
                    state.clone(),
                    cancelled,
                )
                .await?,
            );
        }
        caches.push(per_layer);
    }
    let mut init = service.begin_initialization(0)?;
    for body in &bodies {
        init.retain_completion_owner(body.completion_owner())?;
    }
    let pending_tied = init
        .upload_tied_bf16(source.clone(), head_member(&records[0])?.clone(), cancelled)
        .await?;
    let pending_weights = PendingModelWeights::upload(&records[0], &mut init, cancelled).await?;
    let shared_workspace = init.allocate_empty(8_519_680, BufferRole::Scratch)?;
    let rotary = ModelRotary::prepare(source.clone(), service.clone(), prefix, &mut init)?;
    let mut parameters = Vec::with_capacity(18);
    let mut parameter_map = BTreeMap::new();
    let mut attention_parameters = Vec::with_capacity(6);
    let mut attention_parameter_map = BTreeMap::new();
    for layer in 0..24 {
        if attention_layer(&records[0], layer) {
            let parameter = ModelAttentionParameters::prepare(
                source.clone(),
                qk_operator(&records[0], layer)?,
                &service,
                &mut init,
                cancelled,
            )?;
            attention_parameter_map.insert(layer, parameter.clone());
            attention_parameters.push(parameter);
        } else {
            let pair = node_pair(&records[0], layer)?;
            let op = |index: usize| {
                records[0]
                    .operator(index)?
                    .ok_or_else(|| Error::State("full24 original GDN parameter descriptor".into()))
            };
            let parameter = ModelGdnParameters::prepare(
                source.clone(),
                &service,
                op(pair[0])?,
                op(pair[1])?,
                &mut init,
                cancelled,
            )?;
            parameter_map.insert(layer, parameter.clone());
            parameters.push(parameter);
        }
    }
    let mut positions = Vec::with_capacity(5);
    for index in 0..5 {
        let h = history(index);
        let binding = sessions[h].bind_text_positions(
            records[index].clone(),
            &attempts[h],
            Some(phases[index].clone()),
            cancelled,
        )?;
        positions.push(TextPositionsPrepared::prepare(
            binding, &mut init, cancelled,
        )?);
    }
    drop(init.complete_sync().await?);
    let tied = pending_tied.bind_initialized()?;
    let weights = pending_weights.bind_initialized()?;
    let mut gdn = Vec::with_capacity(5);
    for index in 0..5 {
        let h = history(index);
        let mut phase = BTreeMap::new();
        for layer in 0..24 {
            if !attention_layer(&records[index], layer) {
                phase.insert(
                    layer,
                    ModelGdnPrepared::prepare(
                        bodies[index].values().clone(),
                        parameter_map[&layer].clone(),
                        domains[index][layer].clone(),
                        caches[h][layer].clone(),
                        node_pair(&records[index], layer)?,
                        cancelled,
                    )
                    .await?,
                );
            }
        }
        gdn.push(phase);
    }
    let mut init = service.begin_initialization(0)?;
    for body in &bodies {
        init.retain_completion_owner(body.completion_owner())?;
    }
    let mut attention = Vec::with_capacity(5);
    let mut all_attention = Vec::with_capacity(30);
    for index in 0..5 {
        let h = history(index);
        let mut phase = BTreeMap::new();
        for layer in (0..24).filter(|layer| attention_layer(&records[index], *layer)) {
            let binding = ModelAttentionBinding::prepare(
                ModelAttentionInputs {
                    values: bodies[index].values().clone(),
                    weights: weights.clone(),
                    parameters: attention_parameter_map[&layer].clone(),
                    positions: positions[index].clone(),
                    rotary: rotary.clone(),
                    domains: domains[index][layer].clone(),
                    cache: caches[h][layer].clone(),
                    workspace: shared_workspace.clone(),
                },
                &mut init,
                cancelled,
            )?;
            all_attention.push(binding.clone());
            phase.insert(layer, binding);
        }
        attention.push(phase);
    }
    // Integer controls/gamma are actual original H2D; module loading begins
    // only after this completion and every other managed allocation above.
    drop(init.complete_sync().await?);
    let mut init = service.begin_initialization(0)?;
    for body in &bodies {
        init.retain_completion_owner(body.completion_owner())?;
    }
    let cold = Arc::new(ColdOwners {
        _bodies: bodies.clone(),
        _states: domains.clone(),
        _gdn: gdn.clone(),
        _attention: attention.clone(),
        _caches: caches.clone(),
        _initial: [sessions[0].domains.clone(), sessions[1].domains.clone()],
        _candidates: candidates.clone(),
        _parameters: parameters.clone(),
        _attention_parameters: attention_parameters.clone(),
        _positions: positions.clone(),
        _rotary: rotary.clone(),
        _weights: weights.clone(),
        _tied: tied.clone(),
        _metadata: metadata,
    });
    // Original init owns the complete graph through partial module preparation.
    // Never use this owner as native/image admission owner: bodies later own
    // that library, which would create a reverse retention cycle.
    init.retain_completion_owner(cold.clone())?;
    for body in &bodies {
        body.prepare_greedy(&mut init, cancelled)?;
    }
    let images = prepare_full_model_images(
        source.clone(),
        &service,
        inputs.projections,
        inputs.activation,
        &mut init,
        cancelled,
    )?;
    let library = FullModelLibrary::prepare(
        FullModelFactoryInputs {
            images,
            weights: weights.clone(),
            tied: tied.clone(),
            values: bodies.iter().map(|b| b.values().clone()).collect(),
            final_normalizers: bodies
                .iter()
                .map(|b| b.final_normalizer())
                .collect::<Result<Vec<_>>>()?,
            attention: all_attention,
            workspace: shared_workspace,
        },
        &mut init,
        cancelled,
    )?;
    for (index, ((state, gdn), attention)) in
        domains.into_iter().zip(gdn).zip(attention).enumerate()
    {
        bodies[index].bind_phase(index, library.clone(), state, gdn, attention)?;
    }
    drop(init.complete_sync().await?);
    let state_ids = sessions
        .iter()
        .flat_map(|s| s.domains.iter())
        .chain(candidates.iter().flatten())
        .flat_map(|d| d.leases.iter().map(ResidentLease::id))
        .collect::<BTreeSet<_>>();
    let mut frames = Vec::with_capacity(5);
    let mut all_owned = Vec::new();
    for index in 0..5 {
        let program = if index == 0 {
            &mut workspace.program
        } else {
            programs[index].as_mut().unwrap()
        };
        merge_accesses(
            &mut all_owned,
            program.finalize_full_model_preflight(&state_ids)?,
        );
        let prior = if local_phase(index) == 0 {
            &sessions[history(index)].domains
        } else {
            &candidates[index - 1]
        };
        let mut reads = bodies[index].readbacks()?;
        let prior = domain_readbacks(&mut reads, prior)?;
        let candidate = domain_readbacks(&mut reads, &candidates[index])?;
        reads.push(positions[index].readback()?);
        let bytes = reads.iter().map(ResidentReadback::bytes).sum();
        bodies[index].require_readback_capacity(bytes)?;
        frames.push(Frame {
            reads: Arc::new(reads),
            prior,
            candidate,
            rms_nodes: bodies[index].fused().keys().copied().collect(),
        });
    }
    if service.resident_bytes()? > cap {
        return Err(Error::Limit(
            "full24 actual prepared graph exceeds closed source bound".into(),
        ));
    }
    // Aggregate ownership may retain future buffers, but NEVER use this Write
    // union for any flight/Ready: every enqueue below uses its own phase only.
    workspace.leases = all_owned;
    workspace.state_accesses = Vec::new();
    workspace.logit_accesses = Vec::new();
    let owners = Arc::new(FullOwners {
        _cold: cold,
        bodies,
        frames,
        candidates,
        initial: [sessions[0].domains.clone(), sessions[1].domains.clone()],
        _source: source.clone(),
        _parameters: parameters,
        _attention_parameters: attention_parameters,
        _positions: positions,
        _rotary: rotary,
        library,
        weights,
        tied,
    });
    slot.finish()?;
    if natural {
        let mut templates = Vec::with_capacity(5);
        for (index, prepared) in programs.iter().enumerate() {
            let program = if index == 0 {
                &workspace.program
            } else {
                prepared.as_ref().unwrap()
            };
            templates.push(program.natural_cold_slot()?);
        }
        let state_pool = super::natural_publication::NaturalStatePool::from_prepared(
            &source,
            &model,
            owners.candidates.clone(),
        )?;
        let slots = natural_cold::ColdModelSlots::new(
            &model,
            &sessions,
            templates,
            owners,
            natural_metadata
                .ok_or_else(|| Error::State("natural cold metadata was not admitted".into()))?,
        )?;
        drop(workspace);
        return Ok(Outcome::Natural(natural_cold::PreparedNaturalCold {
            model,
            sessions,
            slots,
            state_pool,
        }));
    }
    let mut previous: [Option<Arc<CompletedPrefixPhase>>; 2] = [None, None];
    let mut current = 0usize;
    let mut dispatches = 0usize;
    for index in 0..5 {
        let h = history(index);
        let local = local_phase(index);
        let session = &sessions[h];
        if let Some(prior) = &planned[index] {
            prior.bind_actual_completion(
                session,
                previous[h]
                    .as_ref()
                    .ok_or_else(|| Error::State("full24 previous actual completion absent".into()))?
                    .clone(),
            )?;
        }
        if index != 0 {
            let next = programs[index].take().unwrap();
            programs[current] = Some(std::mem::replace(&mut workspace.program, next));
            current = index;
        }
        let body = &owners.bodies[index];
        let frame = &owners.frames[index];
        let prior = if local == 0 {
            &owners.initial[h]
        } else {
            &owners.candidates[index - 1]
        };
        let candidate = &owners.candidates[index];
        let call = records[index].call();
        let bindings = bind_domains(prior, candidate, call.rows().base())?;
        let runtime = RuntimeParameters {
            token: call.rows().ids()[0],
            position: call.rows().base(),
            valid_tokens: call.rows().len(),
        };
        let permit = PrefixPhaseEnqueue::new(session, &schedules[h], local, previous[h].as_ref())?;
        let accesses = body.accesses()?.to_vec();
        if index == 0 {
            mode.arm(&owners, &service, &accesses)?;
            body.arm_fault(mode.fault_mode())?;
        }
        let outputs = accesses
            .iter()
            .filter(|(_, a)| *a == DeviceAccess::Write)
            .map(|(l, _)| l.clone())
            .collect::<Vec<_>>();
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
        let mut flight = service.submit(bases[h].root(), &accesses)?;
        flight.retain_completion_owner(owners.clone())?;
        if let Some(previous) = &previous[h] {
            flight.retain_completion_owner(previous.clone())?;
        }
        let submitted = workspace.program.launch_attempt_fragment(
            service.client(),
            runtime,
            &bindings,
            &permit,
            cancelled,
        )?;
        if submitted.logits.is_some() || submitted.proposals.len() != 42 {
            return Err(Error::State(
                "full24 remains an unpublished complete private state transition".into(),
            ));
        }
        let (ready,raw)=flight.complete_with_readback(&outputs,&frame.reads,|raw|{
            body.validate_readbacks(&raw[..1+18+6+3+327+48*4+2])?;
            owners._positions[index].validate_readback(&raw[raw.len()-1..])?;
            if local==0 && frame.prior.iter().any(|(_,range)|raw[range.clone()].iter().any(|b|b.iter().any(|v|*v!=0))){
                return Err(Error::State("full24 original all42 initial prior differs from zero".into()));
            }
            let prior_domains=frame.prior.iter().map(|(slot,range)|SourceDomainReadback{slot:*slot,buffers:&raw[range.clone()]}).collect::<Vec<_>>();
            let domains=frame.candidate.iter().map(|(slot,range)|SourceDomainReadback{slot:*slot,buffers:&raw[range.clone()]}).collect::<Vec<_>>();
            let end=call.rows().base()+call.rows().len();
            for (prior,candidate) in prior_domains.iter().zip(&domains){
                if prior.slot!=candidate.slot{return Err(Error::State("full24 exact original domain order".into()));}
                if matches!(prior.slot,StateSlot::Kv(_)) {
                    for (p,c) in prior.buffers.iter().zip(candidate.buffers){
                        if p.len()!=512*512*4 || c.len()!=p.len() || p[end*512*4..]!=c[end*512*4..] {
                            return Err(Error::State("full24 canonical future KV padding not carried bitwise from actual prior".into()));
                        }
                    }
                }
            }
            const VALUES:usize=1+18+6+3;
            let rms=frame.rms_nodes.iter().enumerate().map(|(i,n)|(*n,&raw[VALUES+327+i*4..VALUES+327+(i+1)*4])).collect::<Vec<_>>();
            validate(SourceModelReadback{phase:index,record:&records[index],values:&raw[VALUES..VALUES+327],
                input_rms:&raw[25..28],rms:&rms,prior_domains:&prior_domains,domains:&domains,
                greedy:&raw[VALUES+327+48*4..VALUES+327+48*4+2]})
        }).await?;
        drop(raw);
        permit.require_invocation(&service, call, 0, 24)?;
        slot.finish()?;
        previous[h] = Some(CompletedPrefixPhase::after_full_completion(
            session,
            schedules[h].clone(),
            local,
            candidate.clone(),
            ready,
            body.clone(),
            previous[h].take(),
        )?);
        dispatches += submitted.dispatches;
        drop(submitted);
        if session.binding() != bases[h] {
            return Err(Error::State(
                "full24 diagnostic completion published an original model root".into(),
            ));
        }
    }
    let source_weight_windows = owners
        .weights
        .verify_source_windows(owners.clone(), cancelled)
        .await?
        .0;
    let tied_weight_windows = verify_tied_and_retire(&service, &owners, cancelled).await?;
    let native_launches = owners.bodies.iter().map(|b| b.native_launches()).sum();
    let workspace_clears = owners.bodies.iter().map(|b| b.workspace_clears()).sum();
    let device_copies = owners.bodies.iter().map(|b| b.device_copies()).sum();
    drop(previous);
    drop(planned);
    drop(phases);
    drop(parameter_map);
    drop(attention_parameter_map);
    drop(caches);
    drop(programs);
    drop(workspace);
    drop(sessions);
    drop(model);
    drop(schedules);
    drop(attempts);
    drop(records);
    drop(owners);
    drop(source);
    let final_resident_bytes = service.resident_bytes()?;
    if final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "full24 actual original owner/event ledger did not retire".into(),
        ));
    }
    drop(service);
    let physical = arbiter.snapshot(&key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "full24 original physical envelope not retired".into(),
        ));
    }
    Ok(Outcome::Compared(SourceModelReceipt {
        phases: 5,
        native_launches,
        workspace_clears,
        device_copies,
        dispatches,
        source_weight_windows,
        tied_weight_windows,
        original_roots: bases,
        model_root_published: false,
        final_resident_bytes,
        final_physical_reserved_bytes: physical.reserved,
    }))
}

/// Independent source-table windows and final exact parameter/rotary checks.
/// Last flight is the ORIGINAL initialized-W Read guard for the one set unload.
async fn verify_tied_and_retire(
    service: &Arc<ResidentService>,
    owners: &Arc<FullOwners>,
    cancelled: Option<&AtomicBool>,
) -> Result<usize> {
    const WINDOWS: usize = 485;
    const BYTES: usize = 1024 * 1024;
    let weight = &owners.tied;
    let source = &owners._source;
    if !weight.source().same_issuer(source) {
        return Err(Error::State(
            "full24 original tied owner differs before independent readback".into(),
        ));
    }
    let (copies, checkpoints) = weight.copy_counts()?;
    if copies != WINDOWS || checkpoints != WINDOWS {
        return Err(Error::State(
            "full24 actual tied source H2D/checkpoint coverage".into(),
        ));
    }
    let table = head_member(
        owners
            .bodies
            .first()
            .ok_or_else(|| Error::State("full24 retained body absent".into()))?
            .record(),
    )?;
    let matrix = source.matrix(&table.name, table.axes)?;
    if matrix.rows() != 248320
        || matrix.columns() != 1024
        || weight.lease().bytes() != WINDOWS * BYTES
    {
        return Err(Error::State(
            "full24 original full table source geometry/extent".into(),
        ));
    }
    let admission = Arc::new(source.reserve_workspace(32 * BYTES)?);
    let mut parameters = Vec::new();
    for parameter in &owners._parameters {
        parameters.extend(parameter.immutable_weights());
    }
    for parameter in &owners._attention_parameters {
        parameters.extend(parameter.immutable_weights());
    }
    let mut all_weights = owners.weights.accesses();
    all_weights.push((weight.lease().clone(), DeviceAccess::Read));
    all_weights.extend(
        parameters
            .iter()
            .map(|(lease, _)| (lease.clone(), DeviceAccess::Read)),
    );
    all_weights.push((owners._rotary.lease().clone(), DeviceAccess::Read));
    for index in 0..WINDOWS {
        not_cancelled(cancelled)?;
        let first = index * 512;
        let window = Arc::new(match cancelled {
            Some(flag) => source.read_rows_cancellable(&matrix, first..first + 512, flag)?,
            None => source.read_rows(&matrix, first..first + 512)?,
        });
        if window.as_slice().len() != BYTES || window.row_range() != (first..first + 512) {
            return Err(Error::State(
                "full24 exact independent tied source window".into(),
            ));
        }
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: 32 * BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        not_cancelled(cancelled)?;
        slot.begin_submission()?;
        let mut flight = service.submit(0, &all_weights)?;
        flight.retain_completion_owner(owners.clone())?;
        flight.retain_completion_owner(window.clone())?;
        flight.retain_completion_owner(admission.clone())?;
        let mut reads = vec![weight.lease().readback(index * BYTES, BYTES)?];
        if index == WINDOWS - 1 {
            reads.extend(
                parameters
                    .iter()
                    .map(|(lease, _)| lease.readback(0, lease.bytes()))
                    .collect::<Result<Vec<_>>>()?,
            );
            reads.push(
                owners
                    ._rotary
                    .lease()
                    .readback(0, owners._rotary.lease().bytes())?,
            );
            // All original compute/earlier readback completions have succeeded.
            // This same real W/source flight owns every partial unload failure.
            unsafe {
                owners.library.retire_after_physical_completion()?;
            }
        }
        let raw = flight
            .complete_read_only_readback(&reads, |raw| {
                if raw.len() != reads.len() || &raw[0][..] != window.as_slice() {
                    return Err(Error::State(
                        "full24 original complete tied W changed".into(),
                    ));
                }
                if index == WINDOWS - 1 {
                    for (actual, (_, expected)) in
                        raw[1..1 + parameters.len()].iter().zip(&parameters)
                    {
                        if &actual[..] != *expected {
                            return Err(Error::State(
                                "full24 original source GDN/QK parameter changed".into(),
                            ));
                        }
                    }
                    if &raw[raw.len() - 1][..] != owners._rotary.source_bytes() {
                        return Err(Error::State(
                            "full24 genuine computed rotary prefix changed".into(),
                        ));
                    }
                }
                Ok(())
            })
            .await?;
        drop(raw);
        slot.finish()?;
    }
    Ok(WINDOWS)
}
