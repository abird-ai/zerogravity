//! Genuine original first3 source execution; diagnostic completion, no publisher.
use super::continuation::{AttemptPhaseBinding, GdnAttemptSchedule, PrefixPhaseEnqueue};
use super::*;
use crate::cuda::gdn::FixedPromptGdnArithmetic;
use crate::cuda::native::{
    fixed_prompt_mlp::FixedPromptMlpQualificationInputs,
    fixed_prompt_model::FixedPromptModelBinding,
    fixed_prompt_model_native::{self, FixedPromptModelNativePrograms},
    fixed_prompt_model_prefix::{FixedPromptPrefixCut, FixedPromptPrefixKind},
    model_cache::ModelCache,
    model_gdn::{ModelGdnParameters, ModelGdnPrepared},
    model_weights::{ModelWeights, PendingModelWeights},
};
use crate::resident_leases::ResidentReadback;
use crate::resident_program::source_bindings::SourceBindings;
use std::{collections::BTreeMap, ops::Range};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::PlanOperation;

// Source-derived narrow39/six RMS/three native GDN/two canonical roots, including
// original64MiB native charge. Exact512-byte alignment required before allocation.
const PROVIDER_BYTES: usize = 320 * 1024 * 1024;
// New cut requires independent per-lease/native/physical accounting. No inherited
// first3 envelope or guessed increase; ROOT materializes after source review.
const QK_KV_PROVIDER_BYTES: Option<usize> = Some(320 * 1024 * 1024);
const QKV_PROVIDER_BYTES: Option<usize> = Some(320 * 1024 * 1024);
const FA_PROVIDER_BYTES: Option<usize> = Some(320 * 1024 * 1024);
const FA_HOST_BOUND: Option<usize> = Some(733_214_412);
const LAYER3_PROVIDER_BYTES: Option<usize> = Some(384 * 1024 * 1024);
const LAYER3_HOST_BOUND: Option<usize> = Some(750_271_896);
fn provider_bytes(kind: FixedPromptPrefixKind) -> Result<usize> {
    match kind {
        FixedPromptPrefixKind::WholePrefill => Err(Error::Unsupported(
            "wholeInitial47 requires separate full-model qualifier".into(),
        )),
        FixedPromptPrefixKind::FirstThree => Ok(PROVIDER_BYTES),
        FixedPromptPrefixKind::ThroughLayer3 => {
            LAYER3_HOST_BOUND.ok_or_else(|| {
                Error::Unsupported("canonical47 layer3 host bound unqualified".into())
            })?;
            LAYER3_PROVIDER_BYTES.ok_or_else(|| {
                Error::Unsupported("canonical47 layer3 provider envelope unqualified".into())
            })
        }
        FixedPromptPrefixKind::ThroughAttentionFa => {
            FA_HOST_BOUND.ok_or_else(|| {
                Error::Unsupported("canonical47 FA host bound unqualified".into())
            })?;
            FA_PROVIDER_BYTES.ok_or_else(|| {
                Error::Unsupported("canonical47 FA provider envelope unqualified".into())
            })
        }
        FixedPromptPrefixKind::ThroughAttentionKv => QK_KV_PROVIDER_BYTES.ok_or_else(|| {
            Error::Unsupported("canonical47 QK/KV provider envelope unqualified".into())
        }),
        FixedPromptPrefixKind::ThroughAttentionQkv => QKV_PROVIDER_BYTES.ok_or_else(|| {
            Error::Unsupported("canonical47 QKV provider envelope unqualified".into())
        }),
    }
}
fn rms_sum_index(value_count: usize, dual_nodes: &[usize], node: usize) -> Result<usize> {
    let index = dual_nodes
        .iter()
        .position(|id| *id == node)
        .ok_or_else(|| Error::Unsupported("canonical47 computed dual RMS node absent".into()))?;
    Ok(7 + value_count + index * 4 + 1)
}

mod faults;
pub use faults::{
    FixedPromptModelPrefixFault, FixedPromptModelPrefixFaultOutcome,
    qualify_fixed_prompt_model_fa_prefix_fault, qualify_fixed_prompt_model_kv_prefix_fault,
    qualify_fixed_prompt_model_layer3_prefix_fault, qualify_fixed_prompt_model_prefix_fault,
    qualify_fixed_prompt_model_qkv_prefix_fault,
};
use faults::{Mode, Outcome};

struct Execution<'a> {
    cancelled: Option<&'a AtomicBool>,
    mode: Mode,
    kind: FixedPromptPrefixKind,
    rotary: Option<crate::cuda::native::source_rotary::RotaryOutcome>,
}
pub struct FixedPromptModelPrefixReadback<'a> {
    record: &'a Arc<SourceBindings>,
    raw: &'a [Bytes],
    prior: &'a [(StateSlot, Range<usize>)],
    candidate: &'a [(StateSlot, Range<usize>)],
    value_count: usize,
    dual_nodes: &'a [usize],
    positions: Option<usize>,
}
impl FixedPromptModelPrefixReadback<'_> {
    /// Same source-produced control, available only inside actual completion.
    pub fn positions(&self) -> Result<&[u8]> {
        let index = self.positions.ok_or_else(|| {
            Error::Unsupported("canonical47 source positions not in this cut".into())
        })?;
        let bytes = self
            .raw
            .get(index)
            .ok_or_else(|| Error::State("canonical47 actual positions descriptor absent".into()))?;
        if bytes.len() != 1560 {
            return Err(Error::State(
                "canonical47 exact pitch65 positions extent".into(),
            ));
        }
        Ok(&bytes[..])
    }
    /// Original outgoing bit-transport status; never a synthetic completion.
    pub fn kv_transport_status(&self) -> Result<&[u8]> {
        let index = self
            .positions
            .ok_or_else(|| Error::Unsupported("canonical47 KV status not in this cut".into()))?
            + 1;
        let bytes = self
            .raw
            .get(index)
            .ok_or_else(|| Error::State("canonical47 actual KV status descriptor absent".into()))?;
        if bytes.len() != 4 {
            return Err(Error::State("canonical47 exact four-byte KV status".into()));
        }
        Ok(&bytes[..])
    }
    pub fn record(&self) -> &Arc<SourceBindings> {
        self.record
    }
    /// All computed39/42 values are available for diagnostics. Expected coverage
    /// is only the independently captured boundaries named by the caller.
    pub fn value(&self, id: ValueId) -> Result<&[u8]> {
        if id.0 as usize >= self.value_count {
            return Err(Error::Unsupported(
                "canonical47 prefix has no future value".into(),
            ));
        }
        Ok(self.raw[7 + id.0 as usize].as_ref())
    }
    /// Actual F32 sum readback of the indexed original dual RMS actor.
    pub fn rms_sum(&self, node: usize) -> Result<&[u8]> {
        let index = rms_sum_index(self.value_count, self.dual_nodes, node)?;
        let bytes = self
            .raw
            .get(index)
            .ok_or_else(|| Error::State("canonical47 dual sum descriptor absent".into()))?;
        if bytes.len() != 47 * 1024 * 4 {
            return Err(Error::State(
                "canonical47 exact47x1024 F32 dual sum extent".into(),
            ));
        }
        Ok(bytes.as_ref())
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
            .ok_or_else(|| Error::State("canonical47 actual canonical state absent".into()))?;
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
#[derive(Debug)]
pub struct FixedPromptModelPrefixReceipt {
    pub arithmetic: FixedPromptGdnArithmetic,
    pub executed_nodes: usize,
    pub computed_values_read_back: usize,
    pub native_launches: usize,
    pub dispatches: usize,
    pub readback_bytes: usize,
    pub source_weight_windows: usize,
    pub retirement_source_weight_windows: usize,
    pub carried_domains: usize,
    pub original_root: RootBinding,
    pub model_root_published: bool,
    pub parent_model_accepted: bool,
    pub all39_values_numerically_compared: bool,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}
// Original init/flight graph anchor. Never retained by images/native/Body.
struct ColdOwners {
    _body: Arc<crate::resident_program::native_full::FullModelProgram>,
    _initial: Vec<Domain>,
    _candidate: Vec<Domain>,
    _domains: Vec<Arc<super::model_state::BoundModelState>>,
    _gdn: BTreeMap<usize, Arc<ModelGdnPrepared>>,
    _caches: Vec<Arc<ModelCache>>,
    _parameters: Vec<Arc<ModelGdnParameters>>,
    _weights: Arc<ModelWeights>,
    _workspace: ResidentLease,
}
/// Caller pins ONE coherent original observed trajectory, then declares exact
/// compared boundaries/states separately. No expected value enters native inputs.
/// Existing source tuple is OwnedCanonical47Solve4; no original-PTX tuple equality.
pub async fn qualify_fixed_prompt_model_prefix<F>(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptMlpQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<FixedPromptModelPrefixReceipt>
where
    F: FnOnce(FixedPromptModelPrefixReadback<'_>) -> Result<()>,
{
    match qualify_inner(
        device,
        source,
        record,
        arithmetic,
        inputs,
        Execution {
            cancelled,
            mode: Mode::Compare,
            rotary: None,
            kind: FixedPromptPrefixKind::FirstThree,
        },
        validate,
    )
    .await?
    {
        Outcome::Compared(value) => Ok(value),
        Outcome::Fault(_) => Err(Error::State(
            "canonical47 first3 positive returned fault outcome".into(),
        )),
    }
}

/// Separate original33-node cut through actual layer3 QKV, before QK/KV/FA.
/// Source stage cold-refuses until new budget and complete ABI pins are accepted.
pub async fn qualify_fixed_prompt_model_qkv_prefix<F>(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptMlpQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<FixedPromptModelPrefixReceipt>
where
    F: FnOnce(FixedPromptModelPrefixReadback<'_>) -> Result<()>,
{
    match qualify_inner(
        device,
        source,
        record,
        arithmetic,
        inputs,
        Execution {
            cancelled,
            mode: Mode::Compare,
            rotary: None,
            kind: FixedPromptPrefixKind::ThroughAttentionQkv,
        },
        validate,
    )
    .await?
    {
        Outcome::Compared(value) => Ok(value),
        Outcome::Fault(_) => Err(Error::State(
            "canonical47 QKV positive returned fault outcome".into(),
        )),
    }
}

/// Native image inputs plus genuine completed same-engine rotary constructor.
/// No bytes constructor, captured control operand or legacy phase substitution.
pub struct FixedPromptQkKvQualificationInputs {
    pub native: FixedPromptMlpQualificationInputs,
    pub rotary: Option<crate::cuda::native::source_rotary::RotaryOutcome>,
}
pub async fn qualify_fixed_prompt_model_kv_prefix<F>(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<FixedPromptModelPrefixReceipt>
where
    F: FnOnce(FixedPromptModelPrefixReadback<'_>) -> Result<()>,
{
    let FixedPromptQkKvQualificationInputs { native, rotary } = inputs;
    match qualify_inner(
        device,
        source,
        record,
        arithmetic,
        native,
        Execution {
            cancelled,
            mode: Mode::Compare,
            kind: FixedPromptPrefixKind::ThroughAttentionKv,
            rotary,
        },
        validate,
    )
    .await?
    {
        Outcome::Compared(value) => Ok(value),
        Outcome::Fault(_) => Err(Error::State(
            "canonical47 QK/KV positive returned fault outcome".into(),
        )),
    }
}

/// Finite36-node FA continuation; independent source/proof/budget gates remain closed.
pub async fn qualify_fixed_prompt_model_fa_prefix<F>(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<FixedPromptModelPrefixReceipt>
where
    F: FnOnce(FixedPromptModelPrefixReadback<'_>) -> Result<()>,
{
    let FixedPromptQkKvQualificationInputs { native, rotary } = inputs;
    match qualify_inner(
        device,
        source,
        record,
        arithmetic,
        native,
        Execution {
            cancelled,
            mode: Mode::Compare,
            kind: FixedPromptPrefixKind::ThroughAttentionFa,
            rotary,
        },
        validate,
    )
    .await?
    {
        Outcome::Compared(value) => Ok(value),
        Outcome::Fault(_) => Err(Error::State(
            "canonical47 FA positive returned fault outcome".into(),
        )),
    }
}

/// Same original full layer3 source traversal; new qualification gates remain closed.
pub async fn qualify_fixed_prompt_model_layer3_prefix<F>(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<FixedPromptModelPrefixReceipt>
where
    F: FnOnce(FixedPromptModelPrefixReadback<'_>) -> Result<()>,
{
    let FixedPromptQkKvQualificationInputs { native, rotary } = inputs;
    match qualify_inner(
        device,
        source,
        record,
        arithmetic,
        native,
        Execution {
            cancelled,
            mode: Mode::Compare,
            kind: FixedPromptPrefixKind::ThroughLayer3,
            rotary,
        },
        validate,
    )
    .await?
    {
        Outcome::Compared(value) => Ok(value),
        Outcome::Fault(_) => Err(Error::State(
            "canonical47 layer3 positive returned fault outcome".into(),
        )),
    }
}

async fn qualify_inner<F>(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptMlpQualificationInputs,
    execution: Execution<'_>,
    validate: F,
) -> Result<Outcome>
where
    F: FnOnce(FixedPromptModelPrefixReadback<'_>) -> Result<()>,
{
    let Execution {
        cancelled,
        mode,
        kind,
        rotary,
    } = execution;
    not_cancelled(cancelled)?;
    arithmetic.require_owned()?;
    if !source.same_issuer(record.source()) {
        return Err(Error::State(
            "fixed prompt actual source/graph owner differs".into(),
        ));
    }
    let scope = FixedPromptModelBinding::bind(record.clone())?;
    let cut = match kind {
        FixedPromptPrefixKind::WholePrefill => {
            return Err(Error::Unsupported(
                "wholeInitial47 cannot enter a legacy prefix qualifier".into(),
            ));
        }
        FixedPromptPrefixKind::FirstThree => FixedPromptPrefixCut::bind(scope.clone())?,
        FixedPromptPrefixKind::ThroughAttentionQkv
        | FixedPromptPrefixKind::ThroughAttentionKv
        | FixedPromptPrefixKind::ThroughAttentionFa
        | FixedPromptPrefixKind::ThroughLayer3 => {
            FixedPromptPrefixCut::bind_kind(scope.clone(), kind)?
        }
    };
    let rotary_prefix = if cut.includes_kv() && !mode.is_pressure() {
        crate::cuda::native::attention_cubin::fixed_prompt_packets::require_memory_qualification()?;
        let prefix = rotary
            .ok_or_else(|| {
                Error::State("canonical47 genuine completed rotary constructor required".into())
            })?
            .into_prefix()?;
        prefix.require_source(&source)?;
        prefix.require_span(0, 47)?;
        Some(prefix)
    } else {
        if rotary.is_some() {
            return Err(Error::State(
                "legacy canonical47 cut cannot accept rotary owner".into(),
            ));
        }
        None
    };
    let provider_bytes = provider_bytes(kind)?;
    let body_reads = 4 + 3 + cut.values().len() + cut.duals() * 4;
    let executed_nodes = cut.nodes().len();
    let computed_values_read_back = cut.values().len();
    let dual_nodes = cut.nodes().iter().enumerate().filter_map(|(index,node)|
        matches!(&node.operation, PlanOperation::SourceOperator{descriptor,..}
            if matches!(descriptor.kind(), zerogravity_core::accelerator::program::SourceOperatorKind::ResidualRms(_)))
        .then_some(index)).collect::<Vec<_>>();
    if !mode.is_pressure() {
        fixed_prompt_model_native::validate_inputs(&cut, &inputs)?;
    }
    validate_provider(&device)?;
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "release canonical47 model prefix required".into(),
        ));
    }
    let metadata = source.reserve_workspace(4 * 1024 * 1024)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new(
            "source-canonical47-first3",
            "unpublished-original-model-cut",
        )?,
        provider_bytes,
        inputs.projections.device_cap,
    )?;
    if service.client().properties().memory.alignment != 512 {
        return Err(Error::Unsupported(
            "canonical47 first3 exact512-byte managed alignment".into(),
        ));
    }
    let key = service.physical_key();
    let physical =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.fenced || physical.reserved != provider_bytes as u64 {
        return Err(Error::State(
            "canonical47 first3 solitary original physical envelope".into(),
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
            bytes: provider_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
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
    let schedule = GdnAttemptSchedule::new_fixed_prompt_model_prefix(&scope, attempt.clone())?;
    let phase = AttemptPhaseBinding::new(schedule.clone(), 0, None)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: provider_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut workspace = model.workspace.lock().await;
    let body = workspace.program.full_model_body()?.clone();
    let mut domains = Vec::new();
    for layer in 0..3 {
        domains.push(session.bind_fixed_prompt_model_prefix_state(
            &attempt,
            &scope,
            layer,
            &candidate,
            phase.clone(),
        )?);
    }
    if cut.includes_kv() {
        domains.push(session.bind_fixed_prompt_attention_kv_state(
            &attempt,
            &cut,
            &candidate,
            phase.clone(),
        )?);
    }
    let mut caches = Vec::new();
    for domain in &domains {
        if domain.kv_domain().is_ok() {
            caches.push(
                ModelCache::prepare_fixed_prompt_attention_kv(
                    &cut,
                    service.clone(),
                    &attempt,
                    domain.clone(),
                    cancelled,
                )
                .await?,
            );
            continue;
        }
        caches.push(
            ModelCache::prepare_fixed_prompt_prefix(
                scope.clone(),
                service.clone(),
                &attempt,
                domain.clone(),
                cancelled,
            )
            .await?,
        );
    }
    let mut init = service.begin_initialization(base.root())?;
    init.retain_completion_owner(body.completion_owner())?;
    if cut.includes_kv() {
        // New route: all already-prepared roots/cache/state metadata survive
        // uncertainty on the first genuine learned-source upload as well.
        init.retain_completion_owner(Arc::new((
            session.domains.clone(),
            candidate.clone(),
            domains.clone(),
            caches.clone(),
        )))?;
    }
    let pending =
        PendingModelWeights::upload_fixed_prompt_prefix(&cut, &mut init, cancelled).await?;
    let mut parameters = Vec::new();
    for layer in 0..3 {
        let pair = scope.gdn_site(layer)?.nodes();
        let operator = |i: usize| {
            record.operator(i)?.ok_or_else(|| {
                Error::State("canonical47 actual model GDN parameter descriptor".into())
            })
        };
        parameters.push(ModelGdnParameters::prepare(
            source.clone(),
            &service,
            operator(pair[0])?,
            operator(pair[1])?,
            &mut init,
            cancelled,
        )?);
    }
    let attention_sources = if let Some(prefix) = rotary_prefix {
        use crate::cuda::native::{
            attention_cubin::ModelAttentionParameters, model_rotary::ModelRotary,
        };
        let parameters = ModelAttentionParameters::prepare(
            source.clone(),
            record
                .operator(ModelAttentionParameters::nodes(&record, 3)?[1])?
                .ok_or_else(|| {
                    Error::State("fixed prefix actual QK source operator absent".into())
                })?,
            &service,
            &mut init,
            cancelled,
        )?;
        let position_binding =
            session.bind_fixed_prompt_text_positions(&cut, &attempt, phase.clone(), cancelled)?;
        let positions = crate::resident_program::text_positions::TextPositionsPrepared::prepare(
            position_binding,
            &mut init,
            cancelled,
        )?;
        let rotary = ModelRotary::prepare(source.clone(), service.clone(), prefix, &mut init)?;
        Some((parameters, positions, rotary))
    } else {
        None
    };
    drop(init.complete_sync().await?);
    let weights = pending.bind_initialized()?;
    let mut gdn = BTreeMap::new();
    for layer in 0..3 {
        gdn.insert(
            layer,
            ModelGdnPrepared::prepare_fixed_prompt_prefix(
                scope.clone(),
                body.values().clone(),
                parameters[layer].clone(),
                domains[layer].clone(),
                caches[layer].clone(),
                arithmetic,
                cancelled,
            )
            .await?,
        );
    }
    // All managed actors/controls/source uploads precede the original native pool.
    let mut init = service.begin_initialization(base.root())?;
    let native_workspace = init.allocate_empty(8_519_680, BufferRole::Scratch)?;
    let cold = Arc::new(ColdOwners {
        _body: body.clone(),
        _initial: session.domains.clone(),
        _candidate: candidate.clone(),
        _domains: domains.clone(),
        _gdn: gdn.clone(),
        _caches: caches.clone(),
        _parameters: parameters.clone(),
        _weights: weights.clone(),
        _workspace: native_workspace.clone(),
    });
    init.retain_completion_owner(cold.clone())?;
    let attention = attention_sources
        .map(|(parameters, positions, rotary)| {
            fixed_prompt_model_native::attention::Binding::prepare(
                fixed_prompt_model_native::attention::Inputs {
                    cut: cut.clone(),
                    values: body.values().clone(),
                    parameters,
                    positions,
                    rotary,
                    domains: domains[3].clone(),
                    cache: caches[3].clone(),
                },
                &mut init,
                cancelled,
            )
        })
        .transpose()?;
    let images =
        fixed_prompt_model_native::prepare_images(&cut, &service, inputs, &mut init, cancelled)?;
    let native = if cut.includes_kv() {
        FixedPromptModelNativePrograms::prepare_with_attention(
            cut.clone(),
            body.values().clone(),
            weights.clone(),
            native_workspace.clone(),
            images.clone(),
            attention.clone(),
            &mut init,
            cancelled,
        )?
    } else {
        FixedPromptModelNativePrograms::prepare(
            cut.clone(),
            body.values().clone(),
            weights.clone(),
            native_workspace.clone(),
            images.clone(),
            &mut init,
            cancelled,
        )?
    };
    body.bind_fixed_prompt_prefix(
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
    if reads.len() != body_reads {
        return Err(Error::State(
            "canonical47 exact narrow body readback count".into(),
        ));
    }
    let positions_index = attention.as_ref().map(|_| reads.len());
    if let Some(binding) = &attention {
        reads.extend(binding.extra_readbacks()?);
    }
    let prior = domain_readbacks(&mut reads, &session.domains)?;
    let proposed = domain_readbacks(&mut reads, &candidate)?;
    let parameter_start = reads.len();
    let mut immutable = parameters
        .iter()
        .flat_map(|p| p.immutable_weights())
        .collect::<Vec<_>>();
    if let Some(binding) = &attention {
        immutable.extend(binding.immutable_weights());
    }
    for (lease, _) in &immutable {
        reads.push(lease.readback(0, lease.bytes())?);
    }
    let readback_bytes = reads.iter().map(ResidentReadback::bytes).sum::<usize>();
    body.require_readback_capacity(readback_bytes)?;
    if readback_bytes > 128 * 1024 * 1024 || service.resident_bytes()? > provider_bytes {
        return Err(Error::Limit(
            "canonical47 original D2H cohort/provider bound exceeded".into(),
        ));
    }
    let cohort = Arc::new(reads);
    let owners = Arc::new(faults::PreparedOwners {
        cold,
        cut: cut.clone(),
        native: native.clone(),
        cohort: cohort.clone(),
        _metadata: metadata,
    });
    mode.arm(&owners, &service, &accesses, base)?;
    body.arm_fault(mode.fault_mode())?;
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
    flight.retain_completion_owner(owners.clone())?;
    let submitted = workspace.program.launch_attempt_fragment(
        service.client(),
        runtime,
        &bindings,
        &permit,
        cancelled,
    )?;
    if !matches!(&mode, Mode::Compare) {
        return Err(Error::State(
            "canonical47 first3 intended fault cut was not observed".into(),
        ));
    }
    if submitted.logits.is_some() || submitted.proposals.len() != 42 {
        return Err(Error::State(
            "canonical47 first3 cannot publish logits/root or omit canonical proposals".into(),
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
                return Err(Error::State(
                    "canonical47 complete exact math/state cohort".into(),
                ));
            }
            body.validate_readbacks(&raw[..body_reads])?;
            if let Some(binding) = &attention {
                binding.validate_extra(&raw[body_reads..body_reads + 2])?;
            }
            for (_, span) in &prior {
                if raw[span.clone()].iter().any(|v| v.iter().any(|b| *b != 0)) {
                    return Err(Error::State(
                        "canonical47 all42 genuine fresh prior domains differ from zero".into(),
                    ));
                }
            }
            for slot in cut.carries() {
                let (_, a) = prior
                    .iter()
                    .find(|(s, _)| s == slot)
                    .ok_or_else(|| Error::State("canonical47 carry prior absent".into()))?;
                let (_, b) = proposed
                    .iter()
                    .find(|(s, _)| s == slot)
                    .ok_or_else(|| Error::State("canonical47 carry candidate absent".into()))?;
                if a.len() != b.len()
                    || raw[a.clone()]
                        .iter()
                        .zip(&raw[b.clone()])
                        .any(|(x, y)| x[..] != y[..])
                {
                    return Err(Error::State(
                        "canonical47 remaining36 domains not exact original prior copies".into(),
                    ));
                }
            }
            for (actual, (_, expected)) in raw[parameter_start..].iter().zip(&immutable) {
                if &actual[..] != *expected {
                    return Err(Error::State(
                        "canonical47 original GDN parameters changed".into(),
                    ));
                }
            }
            validate(FixedPromptModelPrefixReadback {
                record: &record,
                raw,
                prior: &prior,
                candidate: &proposed,
                value_count: computed_values_read_back,
                dual_nodes: &dual_nodes,
                positions: positions_index,
            })
        })
        .await?;
    drop(actual);
    drop(ready);
    slot.finish()?;
    if session.binding() != base {
        return Err(Error::State(
            "canonical47 finite diagnostic published original root".into(),
        ));
    }
    // After diagnostic Ready/callback: serial original W verifier flights.
    // W packs never join the128MiB math cohort; final publication stays closed.
    let source_weight_windows = native
        .weights()
        .verify_source_windows(body.completion_owner(), cancelled)
        .await?
        .0;
    let mut retire_accesses = weights.accesses();
    merge_accesses(
        &mut retire_accesses,
        immutable
            .iter()
            .map(|(l, _)| (l.clone(), DeviceAccess::Read)),
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
    slot.begin_submission()?;
    let mut guard = service.submit(base.root(), &retire_accesses)?;
    guard.retain_completion_owner(body.completion_owner())?;
    unsafe {
        native.retire_after_physical_completion()?;
    }
    let rereads = immutable
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let raw = guard
        .complete_read_only_readback(&rereads, |raw| {
            if raw.len() != immutable.len()
                || raw.iter().zip(&immutable).any(|(a, (_, b))| &a[..] != *b)
            {
                return Err(Error::State(
                    "canonical47 immutable GDN parameters changed at retirement".into(),
                ));
            }
            Ok(())
        })
        .await?;
    drop(raw);
    slot.finish()?;
    let retirement_source_weight_windows = native
        .weights()
        .verify_source_windows(body.completion_owner(), cancelled)
        .await?
        .0;
    let carried_domains = cut.carries().len();
    let (dispatches, native_launches) = (submitted.dispatches, body.native_launches());
    drop(submitted);
    drop(promoted);
    drop(cohort);
    drop(rereads);
    drop(retire_accesses);
    drop(immutable);
    drop(accesses);
    drop(bindings);
    drop(attention);
    drop(native);
    drop(images);
    drop(native_workspace);
    drop(weights);
    drop(gdn);
    drop(parameters);
    drop(caches);
    drop(domains);
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
            "canonical47 original managed ledger did not retire".into(),
        ));
    }
    drop(service);
    let physical =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "canonical47 physical envelope did not retire".into(),
        ));
    }
    Ok(Outcome::Compared(FixedPromptModelPrefixReceipt {
        arithmetic,
        executed_nodes,
        computed_values_read_back,
        native_launches,
        dispatches,
        readback_bytes,
        source_weight_windows,
        retirement_source_weight_windows,
        carried_domains,
        original_root: base,
        model_root_published: false,
        parent_model_accepted: false,
        all39_values_numerically_compared: false,
        final_resident_bytes,
        final_physical_reserved_bytes: physical.reserved,
    }))
}

#[cfg(test)]
mod tests;
