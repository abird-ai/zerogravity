//! One loaded source/native set, original independent Sessions and real commits.
use super::completion::PreparedCall;
use super::*;
use crate::resident_inference::native_fragment::canonical47_model_prefix::Canonical47QkKvQualificationInputs;
use crate::resident_leases::TiedWeight;
use crate::resident_program::text_positions::canonical_natural::CanonicalPositionsStorage;
use crate::resident_program::{
    ResidentProgram,
    native_full::{FullModelProgram, canonical47_natural::CanonicalColdValues},
};
use crate::vllm_cuda_qualification::{
    attention_cubin::ModelAttentionParameters,
    canonical47_model_native::natural::{AttentionStorage, CanonicalNativeStorage},
    model_cache::SharedModelCache,
    model_gdn::{ColdCanonicalGdn, ModelGdnParameters},
    model_rotary::ModelRotary,
    model_weights::PendingModelWeights,
};
use std::{collections::BTreeMap, sync::atomic::AtomicUsize};
use zerogravity_core::accelerator::program::PlanOperation;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};
mod capacity;
pub use capacity::Canonical47LifecycleProfile;
mod faults;
mod source_check;
pub(super) use faults::ProbeRecord;
pub use faults::{Canonical47NaturalFault, Canonical47NaturalFaultOutcome};

pub(super) struct ColdModel {
    source: BoundFullModelSource,
    values: [CanonicalColdValues; 2],
    gdn: [BTreeMap<usize, Arc<ColdCanonicalGdn>>; 2],
    native: Arc<CanonicalNativeStorage>,
    positions: Arc<CanonicalPositionsStorage>,
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
        input: Arc<CanonicalCallInput>,
        state: Arc<CanonicalCallState>,
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
        body.bind_canonical47_call(input.clone(), state, native, positions, gdn)?;
        ResidentProgram::from_natural_body(
            input.source().source().clone(),
            body,
            cold.buffers.clone(),
            cold.preflight.clone(),
        )
    }
}
pub struct Canonical47Loaded {
    profile: Canonical47LifecycleProfile,
    admission: std::sync::Mutex<()>,
    model: Arc<ResidentModel>,
    cold: Arc<ColdModel>,
    active: Arc<AtomicUsize>,
    pending: Arc<AtomicUsize>,
    snapshots: Arc<AtomicUsize>,
    closed: AtomicBool,
    closing: AtomicBool,
}
pub struct Canonical47Session {
    original: ResidentSession,
    loaded: Arc<Canonical47Loaded>,
    feedback: Option<Arc<CommittedPacket>>,
    _reservation: CountReservation,
}
pub struct Canonical47Pending {
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

pub struct Canonical47EmptySnapshot {
    original: ResidentSnapshot,
    model: Weak<ResidentModel>,
    _reservation: CountReservation,
    _metadata: WindowWorkspace,
    _capacity_metadata: Option<Arc<WindowWorkspace>>,
}
#[derive(Clone, Debug)]
pub struct Canonical47CallReceipt {
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
pub struct Canonical47CloseReceipt {
    pub source_weight_windows: usize,
    pub tied_weight_windows: usize,
    pub retirement_source_weight_windows: usize,
    pub retirement_tied_weight_windows: usize,
}

/// Retains only the original service for post-retirement ledger observation.
pub struct Canonical47LoadedObservation {
    service: Arc<ResidentService>,
}
#[derive(Clone, Debug)]
pub struct Canonical47LoadedSnapshot {
    pub physical_key: zerogravity_core::resource_arbiter::PhysicalDeviceKey,
    pub provider_capacity_bytes: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub usable: bool,
}
impl Canonical47LoadedObservation {
    pub fn snapshot(&self) -> Result<Canonical47LoadedSnapshot> {
        Ok(Canonical47LoadedSnapshot {
            physical_key: self.service.physical_key(),
            provider_capacity_bytes: self.service.capacity(),
            resident_bytes: self.service.resident_bytes()?,
            pending_events: self.service.pending_events()?,
            uploaded_value_bytes: self.service.uploaded_value_bytes(),
            usable: self.service.ensure_usable().is_ok(),
        })
    }
}

fn validate_prompt_contract(ids: &[u32], max_tokens: usize) -> Result<()> {
    if ids != crate::resident_program::qwen2b_component::CANONICAL47_PROMPT
        || !(1..=8).contains(&max_tokens)
    {
        return Err(Error::Unsupported(
            "canonical exact47 prompt and1..8 output tokens only".into(),
        ));
    }
    Ok(())
}
impl Canonical47Loaded {
    pub fn lifecycle_profile(&self) -> Canonical47LifecycleProfile {
        self.profile
    }
    pub fn observation(&self) -> Canonical47LoadedObservation {
        Canonical47LoadedObservation {
            service: self.model.service.clone(),
        }
    }
    pub fn validate_prompt(&self, ids: &[u32], max_tokens: usize) -> Result<()> {
        validate_prompt_contract(ids, max_tokens)
    }
    pub fn source(&self) -> &BoundFullModelSource {
        &self.cold.source
    }
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.model.service
    }
    pub async fn session(
        self: &Arc<Self>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Canonical47Session> {
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
            Canonical47LifecycleProfile::FiniteTwo => self.model.session().await?,
            Canonical47LifecycleProfile::BenchmarkEight => {
                self.model.session_retaining(self.cold.clone()).await?
            }
        };
        capacity::require_session_metadata(self.profile, &original)?;
        not_cancelled(cancelled)?;
        Ok(Canonical47Session {
            original,
            loaded: self.clone(),
            feedback: None,
            _reservation: reservation,
        })
    }
    pub async fn close(
        self: Arc<Self>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Canonical47CloseReceipt> {
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
        Ok(Canonical47CloseReceipt {
            source_weight_windows: first.0,
            tied_weight_windows: first.1,
            retirement_source_weight_windows: second.0,
            retirement_tied_weight_windows: second.1,
        })
    }
}
impl Canonical47Session {
    pub fn binding(&self) -> RootBinding {
        self.original.binding()
    }
    pub fn snapshot_empty(&self) -> Result<Canonical47EmptySnapshot> {
        if self.binding().frontier() != 0 || self.feedback.is_some() {
            return Err(Error::Unsupported(
                "canonical empty-frontier checkpoint only".into(),
            ));
        }
        let reservation = CountReservation::new(self.loaded.snapshots.clone(), 2)?;
        let metadata = self
            .loaded
            .source()
            .source()
            .source()
            .reserve_workspace(64 * 1024)?;
        Ok(Canonical47EmptySnapshot {
            _capacity_metadata: self.loaded.cold._capacity_metadata.clone(),
            original: self.original.snapshot(),
            model: Arc::downgrade(&self.original.model),
            _reservation: reservation,
            _metadata: metadata,
        })
    }
    pub fn restore_empty(
        &mut self,
        snapshot: &Canonical47EmptySnapshot,
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
        input: Arc<CanonicalCallInput>,
        validate: Option<NaturalValidation<'_>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Canonical47Pending> {
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
            .bind_fresh_canonical47_state(input.clone(), cancelled)
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
        Ok(Canonical47Pending {
            _capacity_metadata: self.loaded.cold._capacity_metadata.clone(),
            prepared: Some(prepared),
            model: Arc::downgrade(&self.original.model),
            _reservation: reservation,
        })
    }
    pub async fn prepare_prompt_with<F>(
        &self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
        mut validate: F,
    ) -> Result<Canonical47Pending>
    where
        F: for<'a> FnMut(Canonical47NaturalReadback<'a>) -> Result<()>,
    {
        let input = self.original.bind_canonical47_prompt(
            self.loaded.cold.source.clone(),
            ids,
            cancelled,
        )?;
        self.prepare_inner(input, Some(&mut validate), cancelled)
            .await
    }
    pub fn commit(
        &mut self,
        mut pending: Canonical47Pending,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Canonical47CallReceipt> {
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
    pub async fn infer_prompt_with<F>(
        &mut self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<Canonical47CallReceipt>
    where
        F: for<'a> FnMut(Canonical47NaturalReadback<'a>) -> Result<()>,
    {
        let p = self.prepare_prompt_with(ids, cancelled, validate).await?;
        self.commit(p, cancelled)
    }
    pub async fn infer_next_with<F>(
        &mut self,
        cancelled: Option<&AtomicBool>,
        mut validate: F,
    ) -> Result<Canonical47CallReceipt>
    where
        F: for<'a> FnMut(Canonical47NaturalReadback<'a>) -> Result<()>,
    {
        let feedback = self
            .feedback
            .clone()
            .ok_or_else(|| Error::State("canonical no original committed feedback".into()))?;
        let input = self.original.bind_canonical47_feedback(
            self.loaded.cold.source.clone(),
            feedback,
            cancelled,
        )?;
        let p = self
            .prepare_inner(input, Some(&mut validate), cancelled)
            .await?;
        self.commit(p, cancelled)
    }
    pub async fn infer_prompt(
        &mut self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Canonical47CallReceipt> {
        let input = self.original.bind_canonical47_prompt(
            self.loaded.cold.source.clone(),
            ids,
            cancelled,
        )?;
        let p = self.prepare_inner(input, None, cancelled).await?;
        self.commit(p, cancelled)
    }
    pub async fn infer_next(
        &mut self,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Canonical47CallReceipt> {
        let feedback = self
            .feedback
            .clone()
            .ok_or_else(|| Error::State("canonical no original committed feedback".into()))?;
        let input = self.original.bind_canonical47_feedback(
            self.loaded.cold.source.clone(),
            feedback,
            cancelled,
        )?;
        let p = self.prepare_inner(input, None, cancelled).await?;
        self.commit(p, cancelled)
    }
}

pub async fn prepare_canonical47_natural(
    device: Device,
    source: BoundFullModelSource,
    inputs: Canonical47QkKvQualificationInputs,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<Canonical47Loaded>> {
    prepare_canonical47_natural_with_profile(
        device,
        source,
        inputs,
        Canonical47LifecycleProfile::FiniteTwo,
        cancelled,
    )
    .await
}

/// Same source, kernels and ordered workspace; only the checked owner envelope differs.
pub async fn prepare_canonical47_natural_with_profile(
    device: Device,
    source: BoundFullModelSource,
    inputs: Canonical47QkKvQualificationInputs,
    profile: Canonical47LifecycleProfile,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<Canonical47Loaded>> {
    not_cancelled(cancelled)?;
    source.source().require_vllm08_full_source()?;
    validate_provider(&device)?;
    let (provider, _host) = capacity::require_profile(source.source(), profile)?;
    profile.require_device_cap(inputs.native.projections.device_cap)?;
    let Canonical47QkKvQualificationInputs {
        native: inputs,
        rotary,
    } = inputs;
    let prefix = rotary
        .ok_or_else(|| Error::State("canonical actual source rotary producer absent".into()))?
        .into_prefix()?;
    prefix.require_source(source.source())?;
    prefix.require_span(0, 54)?;
    let metadata = Arc::new(
        source
            .source()
            .source()
            .reserve_workspace(8 * 1024 * 1024)?,
    );
    // Shared charge survives a completed proposal/snapshot after caller/model drops.
    let capacity_metadata = if profile.metadata_bytes() == 0 {
        None
    } else {
        Some(Arc::new(
            source
                .source()
                .source()
                .reserve_workspace(profile.metadata_bytes())?,
        ))
    };
    let record = |mode, base, ids: &[u32]| -> Result<Arc<Canonical47ModelBinding>> {
        let teachers = if ids.len() == 1 {
            VllmBf16Teachers::new(source.source().source_identity(), base, 512, ids)?
        } else {
            VllmBf16Teachers::from_span(TeacherSpan::new(
                source.source().source_identity(),
                base,
                512,
                248320,
                ids,
            )?)?
        };
        let request = VllmBf16Invocation::from_span(
            GdnPrefixLayer::new(source.source().source_identity(), 0)?,
            mode,
            teachers,
        )?;
        Canonical47ModelBinding::bind(
            source.clone(),
            Arc::new(
                source
                    .source()
                    .record_vllm08_model(Vllm08Call::from_span(request)?)?,
            ),
        )
    };
    let scopes = [
        record(
            VllmBf16Mode::Prefill,
            0,
            &crate::resident_program::qwen2b_component::CANONICAL47_PROMPT,
        )?,
        record(VllmBf16Mode::Decode, 47, &[0])?,
    ];
    // Cold records are source/shape metadata and source-owned private initial
    // values only. No Session/RootAttempt/commit is manufactured for Decode47.
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source().source_identity(),
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
    let first = FullModelProgram::prepare_canonical47_unbound_values(
        service.clone(),
        scopes[0].clone(),
        cancelled,
    )
    .await?;
    let second = FullModelProgram::prepare_canonical47_unbound_values(
        service.clone(),
        scopes[1].clone(),
        cancelled,
    )
    .await?;
    let program = ResidentProgram::from_natural_body(
        source.source().clone(),
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
        arenas.push(
            SharedModelCache::prepare(
                source.source().clone(),
                &service,
                scopes[0].record().call().layer(layer)?,
                cancelled,
            )
            .await?,
        );
    }
    let mut init = service.begin_initialization(0)?;
    init.retain_completion_owner(metadata.clone())?;
    init.retain_completion_owner(first.body.clone())?;
    init.retain_completion_owner(second.body.clone())?;
    let pending_tied = init
        .upload_original_tied_bf16(source.source().clone(), cancelled)
        .await?;
    let pending = PendingModelWeights::upload(
        source.source().clone(),
        scopes[0].record(),
        &mut init,
        cancelled,
    )
    .await?;
    let mut parameters = BTreeMap::new();
    let mut attention_parameters = BTreeMap::new();
    for layer in 0..24 {
        if layer % 4 == 3 {
            attention_parameters.insert(
                layer,
                ModelAttentionParameters::prepare(
                    source.source().clone(),
                    scopes[0].record().call().layer(layer)?,
                    &service,
                    &mut init,
                    cancelled,
                )?,
            );
        } else {
            let [c, r] = scopes[0].gdn_site(layer)?.nodes();
            let plan = scopes[0].record().plan();
            let op = |i: usize| match &plan.nodes()[i].operation {
                PlanOperation::TypedVllm08 { descriptor, .. } => Ok(descriptor.as_ref()),
                _ => Err(Error::State("canonical cold GDN descriptor".into())),
            };
            parameters.insert(
                layer,
                ModelGdnParameters::prepare(
                    source.source().clone(),
                    &service,
                    op(c)?,
                    op(r)?,
                    &mut init,
                    cancelled,
                )?,
            );
        }
    }
    let positions = CanonicalPositionsStorage::prepare(&mut init)?;
    let rotary = ModelRotary::prepare(source.source().clone(), service.clone(), prefix, &mut init)?;
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
                ColdCanonicalGdn::prepare(
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
    let buffers=crate::vllm_cuda_qualification::canonical47_model_native::natural::attention::AttentionBuffers::prepare(source.source(),attention_parameters.clone(),[3,7,11,15,19,23].into_iter().map(|l|(l,arenas[l].clone())).collect(),positions.clone(),rotary.clone(),&mut init,cancelled)?;
    let images =
        crate::vllm_cuda_qualification::canonical47_model_native::natural::images::prepare(
            &scopes, &service, inputs, &mut init, cancelled,
        )?;
    let attention = AttentionStorage::prepare(buffers, images.clone(), &mut init, cancelled)?;
    let native = CanonicalNativeStorage::prepare(
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
    Ok(Arc::new(Canonical47Loaded {
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

pub fn canonical47_natural_host_bound() -> Result<usize> {
    capacity::host()
}
#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    #[test]
    fn prompt_admission_is_exact_before_session_allocation() {
        let prompt = crate::resident_program::qwen2b_component::CANONICAL47_PROMPT;
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
        let profile = Canonical47LifecycleProfile::BenchmarkEight;
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
