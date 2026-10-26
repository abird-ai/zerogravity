//! One loaded source/native set, original independent Sessions and real commits.
use super::completion::PreparedCall;
use super::*;
use crate::cuda::native::{
    attention_cubin::ModelAttentionParameters,
    fixed_prompt_model_native::natural::{AttentionStorage, FixedPromptNativeStorage},
    model_cache::SharedModelCache,
    model_gdn::{ColdFixedPromptGdn, ModelGdnParameters},
    model_rotary::ModelRotary,
    model_weights::PendingModelWeights,
};
use crate::resident_inference::native_fragment::fixed_prompt_model_prefix::FixedPromptQkKvQualificationInputs;
use crate::resident_leases::TiedWeight;
use crate::resident_program::text_positions::fixed_prompt_natural::FixedPromptPositionsStorage;
use crate::resident_program::{
    ResidentProgram,
    native_full::{FullModelProgram, fixed_prompt_natural::CanonicalColdValues},
};
use std::{collections::BTreeMap, sync::atomic::AtomicUsize};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};

mod capacity;
pub use capacity::FixedPromptLifecycleProfile;
mod faults;
mod source_check;
pub(super) use faults::ProbeRecord;
pub use faults::{FixedPromptNaturalFault, FixedPromptNaturalFaultOutcome};

pub(super) struct ColdModel {
    source: VerifiedTensorSource,
    values: [CanonicalColdValues; 2],
    gdn: [BTreeMap<usize, Arc<ColdFixedPromptGdn>>; 2],
    native: Arc<FixedPromptNativeStorage>,
    positions: Arc<FixedPromptPositionsStorage>,
    tied: Arc<TiedWeight>,
    parameters: BTreeMap<usize, Arc<ModelGdnParameters>>,
    attention_parameters: BTreeMap<usize, Arc<ModelAttentionParameters>>,
    rotary: Arc<ModelRotary>,
    _arenas: Vec<Arc<SharedModelCache>>,
    _metadata: Arc<WindowWorkspace>,
    _capacity_metadata: Option<Arc<WindowWorkspace>>,
}
impl ColdModel {
    pub(super) fn bind_program(
        &self,
        input: Arc<FixedPromptCallInput>,
        state: Arc<FixedPromptCallState>,
    ) -> Result<ResidentProgram> {
        let which = usize::from(input.rows() == 1);
        let cold = &self.values[which];
        let body = cold.bind_values(&input, &state, self.tied.clone())?;
        let positions = self.positions.bind(input.clone())?;
        let native = self.native.bind(
            input.clone(),
            state.clone(),
            body.values().clone(),
            positions.clone(),
            self.tied.clone(),
            body.final_normalizer()?,
        )?;
        let mut gdn = BTreeMap::new();
        for (layer, cold) in &self.gdn[which] {
            gdn.insert(
                *layer,
                cold.bind_call(
                    &input,
                    body.values().clone(),
                    state.states()[*layer].clone(),
                )?,
            );
        }
        body.bind_fixed_prompt_call(input.clone(), state, native, positions, gdn)?;
        ResidentProgram::from_natural_body(
            input.source().clone(),
            body,
            cold.buffers.clone(),
            cold.preflight.clone(),
        )
    }
}
pub struct FixedPromptLoaded {
    profile: FixedPromptLifecycleProfile,
    admission: std::sync::Mutex<()>,
    model: Arc<ResidentModel>,
    cold: Arc<ColdModel>,
    active: Arc<AtomicUsize>,
    pending: Arc<AtomicUsize>,
    snapshots: Arc<AtomicUsize>,
    closed: AtomicBool,
    closing: AtomicBool,
}
pub struct FixedPromptSession {
    pub(super) original: ResidentSession,
    loaded: Arc<FixedPromptLoaded>,
    pub(super) feedback: Option<Arc<CommittedPacket>>,
    _reservation: CountReservation,
}
pub struct FixedPromptPending {
    prepared: Option<PreparedCall>,
    model: Weak<ResidentModel>,
    _reservation: CountReservation,
    _capacity_metadata: Option<Arc<WindowWorkspace>>,
}
struct CountReservation(Arc<AtomicUsize>);
impl CountReservation {
    fn new(count: Arc<AtomicUsize>, max: usize) -> Result<Self> {
        count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < max).then_some(n + 1)
            })
            .map_err(|_| Error::Limit("canonical live owner capacity".into()))?;
        Ok(Self(count))
    }
}
impl Drop for CountReservation {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
struct CloseReservation<'a>(&'a AtomicBool);
impl Drop for CloseReservation<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

pub struct FixedPromptEmptySnapshot {
    original: ResidentSnapshot,
    model: Weak<ResidentModel>,
    _reservation: CountReservation,
    _metadata: WindowWorkspace,
    _capacity_metadata: Option<Arc<WindowWorkspace>>,
}
#[derive(Clone, Debug)]
pub struct FixedPromptCallReceipt {
    pub root: RootBinding,
    pub selected_id: u32,
    pub max_bits: u32,
    pub executed_nodes: usize,
    pub native_launches: usize,
    pub dispatches: usize,
    pub readback_bytes: usize,
    pub consumed_tokens: usize,
}
#[derive(Clone, Debug)]
pub struct FixedPromptCloseReceipt {
    pub source_weight_windows: usize,
    pub tied_weight_windows: usize,
    pub retirement_source_weight_windows: usize,
    pub retirement_tied_weight_windows: usize,
}

/// Retains only the original service for post-retirement ledger observation.
pub struct FixedPromptLoadedObservation {
    service: Arc<ResidentService>,
}
#[derive(Clone, Debug)]
pub struct FixedPromptLoadedSnapshot {
    pub physical_key: zerogravity_core::resource_arbiter::PhysicalDeviceKey,
    pub provider_capacity_bytes: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub usable: bool,
}
impl FixedPromptLoadedObservation {
    pub fn snapshot(&self) -> Result<FixedPromptLoadedSnapshot> {
        Ok(FixedPromptLoadedSnapshot {
            physical_key: self.service.physical_key(),
            provider_capacity_bytes: self.service.capacity(),
            resident_bytes: self.service.resident_bytes()?,
            pending_events: self.service.pending_events()?,
            uploaded_value_bytes: self.service.uploaded_value_bytes(),
            usable: self.service.ensure_usable().is_ok(),
        })
    }
}

#[cfg(test)]
fn validate_prompt_contract(ids: &[u32], max_tokens: usize) -> Result<()> {
    if ids != crate::resident_program::qwen2b_component::FIXED_PROMPT_IDS
        || !(1..=8).contains(&max_tokens)
    {
        return Err(Error::Unsupported(
            "canonical exact47 prompt and1..8 output tokens only".into(),
        ));
    }
    Ok(())
}
impl FixedPromptLoaded {
    pub fn lifecycle_profile(&self) -> FixedPromptLifecycleProfile {
        self.profile
    }
    pub fn observation(&self) -> FixedPromptLoadedObservation {
        FixedPromptLoadedObservation {
            service: self.model.service.clone(),
        }
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.cold.source
    }
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.model.service
    }
    pub async fn session(
        self: &Arc<Self>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<FixedPromptSession> {
        not_cancelled(cancelled)?;
        self.service().ensure_usable()?;
        let reservation = {
            let _gate = self
                .admission
                .lock()
                .map_err(|_| Error::State("canonical admission gate poisoned".into()))?;
            if self.closed.load(Ordering::Acquire) || self.closing.load(Ordering::Acquire) {
                return Err(Error::State(
                    "canonical loaded source retired/closing".into(),
                ));
            }
            CountReservation::new(self.active.clone(), self.profile.session_limit())?
        };
        let original = match self.profile {
            FixedPromptLifecycleProfile::FiniteTwo => self.model.session().await?,
            FixedPromptLifecycleProfile::BenchmarkEight => {
                self.model.session_retaining(self.cold.clone()).await?
            }
        };
        capacity::require_session_metadata(self.profile, &original)?;
        not_cancelled(cancelled)?;
        Ok(FixedPromptSession {
            original,
            loaded: self.clone(),
            feedback: None,
            _reservation: reservation,
        })
    }
    pub async fn close(
        self: Arc<Self>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<FixedPromptCloseReceipt> {
        not_cancelled(cancelled)?;
        {
            let _gate = self
                .admission
                .lock()
                .map_err(|_| Error::State("canonical admission gate poisoned".into()))?;
            if self.active.load(Ordering::Acquire) != 0
                || self.pending.load(Ordering::Acquire) != 0
                || self.snapshots.load(Ordering::Acquire) != 0
            {
                return Err(Error::State(
                    "canonical close requires all sessions/private proposals/snapshots dropped"
                        .into(),
                ));
            }
            if self.closing.swap(true, Ordering::AcqRel) || self.closed.load(Ordering::Acquire) {
                return Err(Error::State(
                    "canonical source already closing/retired".into(),
                ));
            }
        }
        let _closing = CloseReservation(&self.closing);
        // No new session can enter; original fair ticket and workspace cover any
        // last borrower. Source verifier takes its own original fair tickets.
        let service = self.service();
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 1,
            }],
            None,
        )?;
        let slot = ticket.wait().await?;
        let workspace = self.model.workspace.lock().await;
        service.ensure_usable()?;
        if self.active.load(Ordering::Acquire) != 0
            || self.pending.load(Ordering::Acquire) != 0
            || self.snapshots.load(Ordering::Acquire) != 0
        {
            return Err(Error::State(
                "canonical concurrent session entered close".into(),
            ));
        }
        // Acquired-only quiescence barrier; no device command was submitted.
        drop(workspace);
        drop(slot);
        let first = source_check::verify(self.cold.clone(), cancelled).await?;
        self.closed.store(true, Ordering::Release);
        source_check::retire(self.cold.clone(), cancelled).await?;
        let second = source_check::verify(self.cold.clone(), cancelled).await?;
        self.closed.store(true, Ordering::Release);
        Ok(FixedPromptCloseReceipt {
            source_weight_windows: first.0,
            tied_weight_windows: first.1,
            retirement_source_weight_windows: second.0,
            retirement_tied_weight_windows: second.1,
        })
    }
}
impl FixedPromptSession {
    pub fn binding(&self) -> RootBinding {
        self.original.binding()
    }
    pub fn snapshot_empty(&self) -> Result<FixedPromptEmptySnapshot> {
        if self.binding().frontier() != 0 || self.feedback.is_some() {
            return Err(Error::Unsupported(
                "canonical empty-frontier checkpoint only".into(),
            ));
        }
        let reservation = CountReservation::new(self.loaded.snapshots.clone(), 2)?;
        let metadata = self.loaded.source().reserve_workspace(64 * 1024)?;
        Ok(FixedPromptEmptySnapshot {
            _capacity_metadata: self.loaded.cold._capacity_metadata.clone(),
            original: self.original.snapshot(),
            model: Arc::downgrade(&self.original.model),
            _reservation: reservation,
            _metadata: metadata,
        })
    }
    pub fn restore_empty(
        &mut self,
        snapshot: &FixedPromptEmptySnapshot,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        not_cancelled(cancelled)?;
        if !snapshot
            .model
            .upgrade()
            .is_some_and(|m| Arc::ptr_eq(&m, &self.original.model))
        {
            return Err(Error::State("canonical foreign empty snapshot".into()));
        }
        self.original.restore(&snapshot.original)?;
        self.feedback = None;
        if self.binding().frontier() != 0 {
            return Err(Error::State(
                "canonical restored snapshot was not empty".into(),
            ));
        }
        Ok(())
    }
    async fn prepare_inner(
        &self,
        input: Arc<FixedPromptCallInput>,
        validate: Option<NaturalValidation<'_>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<FixedPromptPending> {
        not_cancelled(cancelled)?;
        if self.loaded.closed.load(Ordering::Acquire) || self.loaded.closing.load(Ordering::Acquire)
        {
            return Err(Error::State("canonical loaded source closing".into()));
        }
        // At most two completed proposals can coexist per loaded model. This
        // also bounds retained immutable packets/candidate roots on stale cuts.
        let reservation = CountReservation::new(self.loaded.pending.clone(), capacity::PENDING)?;
        let state = self
            .original
            .bind_fresh_fixed_prompt_state(input.clone(), cancelled)
            .await?;
        let prepared = self
            .original
            .prepare_canonical_call(
                self.loaded.cold.clone(),
                input,
                state,
                None,
                validate,
                cancelled,
            )
            .await?;
        Ok(FixedPromptPending {
            _capacity_metadata: self.loaded.cold._capacity_metadata.clone(),
            prepared: Some(prepared),
            model: Arc::downgrade(&self.original.model),
            _reservation: reservation,
        })
    }
    pub fn feedback_request(&self) -> Result<FixedPromptFeedbackRequest<'_>> {
        self.loaded.service().ensure_usable()?;
        if self.loaded.closed.load(Ordering::Acquire) || self.loaded.closing.load(Ordering::Acquire)
        {
            return Err(Error::State("fixed prompt loaded source closing".into()));
        }
        let packet = self
            .feedback
            .as_ref()
            .ok_or_else(|| Error::State("fixed prompt no original committed feedback".into()))?;
        packet.require_current(&self.original, self.loaded.source())?;
        let base = usize::try_from(self.binding().frontier())
            .map_err(|_| Error::Limit("fixed prompt current root frontier".into()))?;
        Ok(FixedPromptFeedbackRequest {
            session: self,
            packet,
            base,
        })
    }
    pub async fn prepare_prompt_with<F>(
        &self,
        record: Arc<SourceBindings>,
        cancelled: Option<&AtomicBool>,
        mut validate: F,
    ) -> Result<FixedPromptPending>
    where
        F: for<'a> FnMut(FixedPromptNaturalReadback<'a>) -> Result<()>,
    {
        if self.feedback.is_some() || self.binding().frontier() != 0 {
            return Err(Error::State(
                "fixed prompt requires empty root/no feedback".into(),
            ));
        }
        let input = self.original.bind_fixed_prompt_prompt(
            self.loaded.cold.source.clone(),
            record,
            cancelled,
        )?;
        self.prepare_inner(input, Some(&mut validate), cancelled)
            .await
    }
    pub fn commit(
        &mut self,
        mut pending: FixedPromptPending,
        cancelled: Option<&AtomicBool>,
    ) -> Result<FixedPromptCallReceipt> {
        if !pending
            .model
            .upgrade()
            .is_some_and(|m| Arc::ptr_eq(&m, &self.original.model))
        {
            return Err(Error::State(
                "canonical proposal belongs to another loaded model".into(),
            ));
        }
        let prepared = pending
            .prepared
            .take()
            .ok_or_else(|| Error::State("canonical proposal consumed".into()))?;
        let (receipt, feedback) = self.original.commit_canonical_call(prepared, cancelled)?;
        self.feedback = Some(feedback);
        Ok(receipt)
    }
    pub async fn prepare_next_with<F>(
        &self,
        request: FixedPromptFeedbackRequest<'_>,
        record: Arc<SourceBindings>,
        cancelled: Option<&AtomicBool>,
        mut validate: F,
    ) -> Result<FixedPromptPending>
    where
        F: for<'a> FnMut(FixedPromptNaturalReadback<'a>) -> Result<()>,
    {
        let packet = request.require(self, self.loaded.source())?.clone();
        let input = self.original.bind_fixed_prompt_feedback(
            self.loaded.cold.source.clone(),
            record,
            packet,
            cancelled,
        )?;
        self.prepare_inner(input, Some(&mut validate), cancelled)
            .await
    }
    pub async fn prepare_prompt(
        &self,
        record: Arc<SourceBindings>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<FixedPromptPending> {
        if self.feedback.is_some() || self.binding().frontier() != 0 {
            return Err(Error::State(
                "fixed prompt requires empty root/no feedback".into(),
            ));
        }
        let input = self.original.bind_fixed_prompt_prompt(
            self.loaded.cold.source.clone(),
            record,
            cancelled,
        )?;
        self.prepare_inner(input, None, cancelled).await
    }
    pub async fn prepare_next(
        &self,
        request: FixedPromptFeedbackRequest<'_>,
        record: Arc<SourceBindings>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<FixedPromptPending> {
        let packet = request.require(self, self.loaded.source())?.clone();
        let input = self.original.bind_fixed_prompt_feedback(
            self.loaded.cold.source.clone(),
            record,
            packet,
            cancelled,
        )?;
        self.prepare_inner(input, None, cancelled).await
    }
}

pub async fn prepare_fixed_prompt_natural(
    device: Device,
    source: VerifiedTensorSource,
    records: [Arc<SourceBindings>; 2],
    inputs: FixedPromptQkKvQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<FixedPromptLoaded>> {
    prepare_fixed_prompt_natural_with_profile(
        device,
        source,
        records,
        inputs,
        FixedPromptLifecycleProfile::FiniteTwo,
        cancelled,
    )
    .await
}

/// Same source, kernels and ordered workspace; only the checked owner envelope differs.
pub async fn prepare_fixed_prompt_natural_with_profile(
    device: Device,
    source: VerifiedTensorSource,
    records: [Arc<SourceBindings>; 2],
    inputs: FixedPromptQkKvQualificationInputs,
    profile: FixedPromptLifecycleProfile,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<FixedPromptLoaded>> {
    not_cancelled(cancelled)?;
    validate_provider(&device)?;
    let (provider, _host) = capacity::require_profile(&source, profile)?;
    profile.require_device_cap(inputs.native.projections.device_cap)?;
    let FixedPromptQkKvQualificationInputs {
        native: inputs,
        rotary,
    } = inputs;
    let prefix = rotary
        .ok_or_else(|| Error::State("canonical actual source rotary producer absent".into()))?
        .into_prefix()?;
    prefix.require_source(&source)?;
    prefix.require_span(0, 54)?;
    let metadata = Arc::new(source.reserve_workspace(8 * 1024 * 1024)?);
    // Shared charge survives a completed proposal/snapshot after caller/model drops.
    let capacity_metadata = if profile.metadata_bytes() == 0 {
        None
    } else {
        Some(Arc::new(
            source.reserve_workspace(profile.metadata_bytes())?,
        ))
    };
    let scopes = [
        FixedPromptModelBinding::bind(records[0].clone())?,
        FixedPromptModelBinding::bind(records[1].clone())?,
    ];
    if scopes[0].kind() != FixedPromptCallKind::Prefill
        || scopes[1].kind() != (FixedPromptCallKind::Decode { base: 47 })
        || scopes
            .iter()
            .any(|scope| !source.same_issuer(scope.source()))
    {
        return Err(Error::State(
            "fixed prompt exact cold source/call scopes differ".into(),
        ));
    }
    // Cold records are source/shape metadata and source-owned private initial
    // values only. No Session/RootAttempt/commit is manufactured for Decode47.
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("canonical47-bf16", "shared-native-two-geometry")?,
        provider,
        inputs.projections.device_cap,
    )?;
    if service.client().properties().memory.alignment != 512 {
        return Err(Error::Unsupported(
            "canonical exact512 managed alignment".into(),
        ));
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
    let first = FullModelProgram::prepare_fixed_prompt_unbound_values(
        service.clone(),
        scopes[0].clone(),
        cancelled,
    )
    .await?;
    let second = FullModelProgram::prepare_fixed_prompt_unbound_values(
        service.clone(),
        scopes[1].clone(),
        cancelled,
    )
    .await?;
    let program = ResidentProgram::from_natural_body(
        source.clone(),
        first.body.clone(),
        first.buffers.clone(),
        first.preflight.clone(),
    )?;
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
        abort_after_launch: AtomicUsize::new(0),
        #[cfg(feature = "hardware-abort-probe")]
        launch_seen: AtomicUsize::new(0),
    });
    capacity::require_model_metadata(profile, &model)?;
    let mut arenas = Vec::with_capacity(24);
    for layer in 0..24 {
        arenas.push(SharedModelCache::prepare(source.clone(), &service, layer, cancelled).await?);
    }
    let mut init = service.begin_initialization(0)?;
    init.retain_completion_owner(metadata.clone())?;
    init.retain_completion_owner(first.body.clone())?;
    init.retain_completion_owner(second.body.clone())?;
    let pending_tied = init
        .upload_tied_bf16(source.clone(), tied_member(scopes[0].record())?, cancelled)
        .await?;
    let pending = PendingModelWeights::upload(scopes[0].record(), &mut init, cancelled).await?;
    let mut parameters = BTreeMap::new();
    let mut attention_parameters = BTreeMap::new();
    for layer in 0..24 {
        if scopes[0].record().state_requirements().iter().any(|r| {
            r.slot == StateSlot::Kv(zerogravity_core::accelerator::program::KvDomain(layer))
        }) {
            attention_parameters.insert(
                layer,
                ModelAttentionParameters::prepare(
                    source.clone(),
                    scopes[0]
                        .record()
                        .operator(ModelAttentionParameters::nodes(scopes[0].record(), layer)?[1])?
                        .ok_or_else(|| {
                            Error::State("fixed prompt actual QK source operator absent".into())
                        })?,
                    &service,
                    &mut init,
                    cancelled,
                )?,
            );
        } else {
            let [c, r] = scopes[0].gdn_site(layer)?.nodes();
            let op = |i: usize| {
                scopes[0]
                    .record()
                    .operator(i)?
                    .ok_or_else(|| Error::State("canonical cold GDN descriptor".into()))
            };
            parameters.insert(
                layer,
                ModelGdnParameters::prepare(
                    source.clone(),
                    &service,
                    op(c)?,
                    op(r)?,
                    &mut init,
                    cancelled,
                )?,
            );
        }
    }
    let positions = FixedPromptPositionsStorage::prepare(&mut init)?;
    let rotary = ModelRotary::prepare(source.clone(), service.clone(), prefix, &mut init)?;
    let native_workspace = init.allocate_empty(8_519_680, BufferRole::Scratch)?;
    drop(init.complete_sync().await?);
    let tied = pending_tied.bind_initialized()?;
    let weights = pending.bind_initialized()?;
    let values = [first, second];
    let mut gdn = [BTreeMap::new(), BTreeMap::new()];
    for which in 0..2 {
        for (layer, p) in &parameters {
            gdn[which].insert(
                *layer,
                ColdFixedPromptGdn::prepare(
                    scopes[which].clone(),
                    values[which].body.values().clone(),
                    p.clone(),
                    arenas[*layer].clone(),
                    *layer,
                    cancelled,
                )
                .await?,
            );
        }
    }
    let mut init = service.begin_initialization(0)?;
    init.retain_completion_owner(Arc::new((
        values[0].body.clone(),
        values[1].body.clone(),
        gdn.clone(),
        arenas.clone(),
        weights.clone(),
        tied.clone(),
    )))?;
    let buffers=crate::cuda::native::fixed_prompt_model_native::natural::attention::AttentionBuffers::prepare(&source,attention_parameters.clone(),[3,7,11,15,19,23].into_iter().map(|l|(l,arenas[l].clone())).collect(),positions.clone(),rotary.clone(),&mut init,cancelled)?;
    let images = crate::cuda::native::fixed_prompt_model_native::natural::images::prepare(
        &scopes, &service, inputs, &mut init, cancelled,
    )?;
    let attention = AttentionStorage::prepare(buffers, images.clone(), &mut init, cancelled)?;
    let native = FixedPromptNativeStorage::prepare(
        scopes,
        [
            values[0].body.values().clone(),
            values[1].body.values().clone(),
        ],
        weights,
        native_workspace,
        images,
        attention,
        &mut init,
        cancelled,
    )?;
    let cold = Arc::new(ColdModel {
        source,
        values,
        gdn,
        native,
        positions,
        tied,
        parameters,
        attention_parameters,
        rotary,
        _arenas: arenas,
        _metadata: metadata,
        _capacity_metadata: capacity_metadata,
    });
    init.retain_completion_owner(cold.clone())?;
    drop(init.complete_sync().await?);
    slot.finish()?;
    Ok(Arc::new(FixedPromptLoaded {
        profile,
        admission: std::sync::Mutex::new(()),
        model,
        cold,
        active: Arc::new(AtomicUsize::new(0)),
        pending: Arc::new(AtomicUsize::new(0)),
        snapshots: Arc::new(AtomicUsize::new(0)),
        closed: AtomicBool::new(false),
        closing: AtomicBool::new(false),
    }))
}

pub fn fixed_prompt_natural_host_bound() -> Result<usize> {
    capacity::host()
}
#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    #[test]
    fn prompt_admission_is_exact_before_session_allocation() {
        let prompt = crate::resident_program::qwen2b_component::FIXED_PROMPT_IDS;
        for n in 1..=8 {
            validate_prompt_contract(&prompt, n).unwrap();
        }
        for n in [0, 9, usize::MAX] {
            assert!(validate_prompt_contract(&prompt, n).is_err());
        }
        for i in 0..prompt.len() {
            let mut changed = prompt;
            changed[i] ^= 1;
            assert!(validate_prompt_contract(&changed, 8).is_err());
        }
        assert!(validate_prompt_contract(&prompt[..46], 8).is_err());
        let mut extended = prompt.to_vec();
        extended.push(0);
        assert!(validate_prompt_contract(&extended, 8).is_err());
    }
    #[test]
    fn owner_capacity_refusal_does_not_consume_another_reservation() {
        let count = Arc::new(AtomicUsize::new(0));
        let first = CountReservation::new(count.clone(), 2).unwrap();
        let second = CountReservation::new(count.clone(), 2).unwrap();
        assert!(CountReservation::new(count.clone(), 2).is_err());
        assert_eq!(count.load(Ordering::Acquire), 2);
        drop(first);
        assert_eq!(count.load(Ordering::Acquire), 1);
        drop(second);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }
    #[test]
    fn eight_session_reservations_do_not_widen_pending_or_snapshot_capacity() {
        let profile = FixedPromptLifecycleProfile::BenchmarkEight;
        let count = Arc::new(AtomicUsize::new(0));
        let mut owners = Vec::new();
        for n in 1..=8 {
            owners.push(CountReservation::new(count.clone(), profile.session_limit()).unwrap());
            assert_eq!(count.load(Ordering::Acquire), n);
        }
        assert!(CountReservation::new(count.clone(), profile.session_limit()).is_err());
        assert_eq!(count.load(Ordering::Acquire), 8);
        drop(owners.pop());
        owners.push(CountReservation::new(count.clone(), profile.session_limit()).unwrap());
        drop(owners);
        assert_eq!(count.load(Ordering::Acquire), 0);
        assert_eq!(capacity::PENDING, 2);
        let snapshots = Arc::new(AtomicUsize::new(0));
        let a = CountReservation::new(snapshots.clone(), 2).unwrap();
        let b = CountReservation::new(snapshots.clone(), 2).unwrap();
        assert!(CountReservation::new(snapshots, 2).is_err());
        drop((a, b));
    }
    #[test]
    fn aborted_async_setup_releases_its_owner_reservation() {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};
        let count = Arc::new(AtomicUsize::new(0));
        let observed = count.clone();
        let mut future = Box::pin(async move {
            let _reservation = CountReservation::new(count, 2).unwrap();
            std::future::pending::<()>().await;
        });
        let mut context = Context::from_waker(Waker::noop());
        assert_eq!(future.as_mut().poll(&mut context), Poll::Pending);
        assert_eq!(observed.load(Ordering::Acquire), 1);
        drop(future);
        assert_eq!(observed.load(Ordering::Acquire), 0);
    }
}
