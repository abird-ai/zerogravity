//! One loaded immutable source; original independent roots and committed feedback.
use super::completion::PreparedCall;
use super::*;
use crate::cuda::gdn::TextGdnWorkspace;
use crate::cuda::native::{
    FixedPromptMlpQualificationInputs,
    model_gdn::{ColdTextGdn, ModelGdnParameters},
    model_rotary::ModelRotary,
    model_weights::PendingModelWeights,
    source_rotary::RotaryOutcome,
    text_native::{TextNativeGeometry, TextNativePreparation, TextNativeStorage},
};
use crate::resident_leases::backing_pool::ResidentPool;
use crate::resident_program::text_positions::text::TextPositionsStorage;
use crate::resident_program::{
    ResidentProgram,
    native_full::{FullModelProgram, text::TextColdValues},
};
use std::{collections::BTreeMap, sync::atomic::AtomicUsize};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};
mod faults;
pub(super) use faults::ProbeRecord;
pub use faults::{TextFault, TextFaultOutcome};
pub(super) mod performance;
mod source_check;
mod workspace;
const MIB: usize = 1024 * 1024;
pub(crate) mod cohort;
mod inventory;
pub use cohort::{
    TextCohortFault, TextCohortFaultOutcome, TextCohortFaultRetention, TextCohortPreparation,
    TextCohortReadback, TextCohortReceipt, TextCohortRowReceipt, TextPendingCohort,
};
pub use inventory::TextResourceInventory;

pub struct TextQualificationInputs {
    pub native: FixedPromptMlpQualificationInputs,
    pub rotary: Option<RotaryOutcome>,
}
pub(super) struct ColdModel {
    source: VerifiedTensorSource,
    capture: TextKernelCapture,
    bindings: Arc<TextModelBindings>,
    values: Vec<TextColdValues>,
    gdn: Vec<BTreeMap<usize, Arc<ColdTextGdn>>>,
    native: Arc<TextNativeStorage>,
    positions: Arc<TextPositionsStorage>,
    parameters: BTreeMap<usize, Arc<ModelGdnParameters>>,
    rotary: Arc<ModelRotary>,
    private: Vec<Arc<TextGdnWorkspace>>,
    _metadata: Arc<WindowWorkspace>,
    _owner_metadata: Arc<WindowWorkspace>,
    resources: TextResourceInventory,
}
impl ColdModel {
    pub(super) async fn bind_program(
        &self,
        input: Arc<TextCallInput>,
        state: Arc<TextCallState>,
        cancel: Option<&AtomicBool>,
    ) -> Result<ResidentProgram> {
        let which = self
            .capture
            .cold_scope_index(input.call().mode(), input.rows())?;
        let cold = self
            .values
            .get(which)
            .ok_or_else(|| Error::State("short HF captured cold geometry absent".into()))?;
        let (body, buffers, preflight, rms_scope) = cold
            .bind_values(&input, &state, self.native.head().resident_weight(), cancel)
            .await?;
        let positions = self.positions.bind(input.clone())?;
        let native = self.native.bind(
            input.clone(),
            state.clone(),
            body.values().clone(),
            rms_scope.clone(),
            body.final_normalizer()?,
            positions.clone(),
        )?;
        let mut gdn = BTreeMap::new();
        for (&layer, cold) in &self.gdn[which] {
            gdn.insert(
                layer,
                cold.bind_call(
                    input.scope().clone(),
                    body.values().clone(),
                    state.states()[layer].clone(),
                )?,
            );
        }
        body.bind_prompt_call(input.clone(), state, native, positions, rms_scope, gdn)?;
        ResidentProgram::from_natural_body(input.source().clone(), body, buffers, preflight)
    }
}
/// Bounded N1 coverage derived from the source-issued cold scopes actually prepared.
#[derive(Clone, Debug)]
struct TextColdGeometry {
    prompt_rows: &'static [usize],
    cached_rows: usize,
}
impl TextColdGeometry {
    fn from_scopes(capture: TextKernelCapture, scopes: &[Arc<TextModelBinding>]) -> Result<Self> {
        capture.require_cold_scopes(scopes.iter().map(Arc::as_ref))?;
        Ok(Self {
            prompt_rows: capture.captured_prefill_rows(),
            cached_rows: 1,
        })
    }
    fn prompt_rows(&self) -> &[usize] {
        self.prompt_rows
    }
    fn cached_rows(&self) -> usize {
        self.cached_rows
    }
}
fn request_not_cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("short HF request cancelled".into()));
    }
    Ok(())
}
fn normalize_cancel(error: Error, flag: Option<&AtomicBool>, service: &ResidentService) -> Error {
    if flag.is_some_and(|f| f.load(Ordering::Acquire))
        && service.ensure_usable().is_ok()
        && matches!(&error,Error::State(s) if matches!(s.as_str(),
            "native fragment cancelled"|"native plan cancelled"|"post-attention RMS cancelled"|
            "BF16 greedy cancelled before enqueue"|"closed vLLM GDN phase cancelled"|
            "short-HF RMS pool cancelled"|"short HF request cancelled"|
            "closed original cubin qualification cancelled"))
    {
        Error::State("short HF request cancelled".into())
    } else {
        error
    }
}
pub struct TextLoaded {
    admission: std::sync::Mutex<()>,
    model: Arc<ResidentModel>,
    root_pool: Arc<ResidentPool>,
    cold: Arc<ColdModel>,
    active: Arc<AtomicUsize>,
    pending: Arc<AtomicUsize>,
    snapshots: Arc<AtomicUsize>,
    closed: AtomicBool,
    closing: AtomicBool,
    cohort: Option<Arc<cohort::ColdCohort>>,
    readiness_geometry: TextColdGeometry,
    readiness_candidates: usize,
    readiness_usage: cubecl::cuda::CompilerContextUsage,
}
pub struct TextSession {
    pub(super) original: ResidentSession,
    loaded: Arc<TextLoaded>,
    pub(super) feedback: Option<Arc<CommittedTextPacket>>,
    cohort_root: Option<RootBinding>,
    _reservation: CountReservation,
}
pub struct TextPending {
    prepared: Option<PreparedCall>,
    model: Weak<ResidentModel>,
    _reservation: CountReservation,
}
struct CountReservation(Arc<AtomicUsize>);
impl CountReservation {
    fn new(count: Arc<AtomicUsize>, max: usize) -> Result<Self> {
        count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < max).then_some(n + 1)
            })
            .map_err(|_| Error::Limit("short HF live owner capacity".into()))?;
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

pub struct TextEmptySnapshot {
    original: ResidentSnapshot,
    model: Weak<ResidentModel>,
    _reservation: CountReservation,
    _metadata: WindowWorkspace,
}
#[derive(Clone, Debug)]
pub struct TextCallReceipt {
    pub root: RootBinding,
    pub selected_id: u32,
    pub max_bits: u32,
    pub executed_nodes: usize,
    pub native_launches: usize,
    pub dispatches: usize,
    /// Deferred ranges, enabled cohorts, immediate/cohort enqueues, cold metadata upper.
    pub finite_certification: [usize; 5],
    /// Actual source retirement: issued, retired, queries, waits, pending.
    pub transfer_retirement: [usize; 5],
    pub readback_bytes: usize,
    pub consumed_tokens: usize,
}
#[derive(Clone, Debug)]
pub struct TextCloseReceipt {
    pub source_weight_windows: usize,
    pub streamed_head_windows: usize,
    pub retirement_source_weight_windows: usize,
    pub retirement_streamed_head_windows: usize,
    pub resident_head_windows: usize,
    pub retirement_resident_head_windows: usize,
}

/// Retains only the original service for post-retirement ledger observation.
pub struct TextLoadedObservation {
    service: Arc<ResidentService>,
    root_pool: Weak<ResidentPool>,
}
#[derive(Clone, Debug)]
pub struct TextLoadedSnapshot {
    pub physical_key: zerogravity_core::resource_arbiter::PhysicalDeviceKey,
    pub provider_capacity_bytes: usize,
    pub backend_host_reserved_bytes: u64,
    pub backend_device_reserved_bytes: u64,
    pub physical_reserved_bytes: u64,
    pub compiler_cumulative_usage: cubecl::cuda::CompilerContextUsage,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub usable: bool,
    pub root_pool_allocated_sets: usize,
    pub root_pool_idle_sets: usize,
    pub root_pool_backing_bytes: usize,
}
impl TextLoadedObservation {
    pub fn snapshot(&self) -> Result<TextLoadedSnapshot> {
        let pool = self
            .root_pool
            .upgrade()
            .map(|pool| pool.snapshot())
            .transpose()?
            .unwrap_or_default();
        let (backend_host_reserved_bytes, backend_device_reserved_bytes, compiler_cumulative_usage) =
            self.service.observe_cuda_compiler_context()?;
        Ok(TextLoadedSnapshot {
            physical_key: self.service.physical_key(),
            provider_capacity_bytes: self.service.capacity(),
            backend_host_reserved_bytes,
            backend_device_reserved_bytes,
            compiler_cumulative_usage,
            physical_reserved_bytes:
                zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
                    .snapshot(&self.service.physical_key())?
                    .reserved,
            resident_bytes: self.service.resident_bytes()?,
            pending_events: self.service.pending_events()?,
            uploaded_value_bytes: self.service.uploaded_value_bytes(),
            usable: self.service.ensure_usable().is_ok(),
            root_pool_allocated_sets: pool.allocated_sets,
            root_pool_idle_sets: pool.idle_sets,
            root_pool_backing_bytes: pool.backing_bytes,
        })
    }
}

impl TextLoaded {
    pub fn require_open(&self) -> Result<()> {
        self.service().ensure_usable()?;
        if self.closed.load(Ordering::Acquire) || self.closing.load(Ordering::Acquire) {
            return Err(Error::State("text loaded source closing".into()));
        }
        Ok(())
    }
    pub fn readiness_usage(&self) -> cubecl::cuda::CompilerContextUsage {
        self.readiness_usage
    }
    /// Constructor calls, including already loaded keys, not a unique-key count.
    pub fn readiness_candidates(&self) -> usize {
        self.readiness_candidates
    }
    /// Source-issued N1 prompt rows prepared by this successful load.
    pub fn readiness_n1_prompt_rows(&self) -> &[usize] {
        self.readiness_geometry.prompt_rows()
    }
    /// Source-issued N1 cached (decode) geometries prepared by this successful load.
    pub fn readiness_n1_cached_rows(&self) -> usize {
        self.readiness_geometry.cached_rows()
    }
    /// All immutable finite groups and individual fallbacks are prepared.
    /// Current actual retained/write proof still decides which groups may run.
    pub fn readiness_finite_grouping_complete(&self) -> bool {
        true
    }
    pub fn observation(&self) -> TextLoadedObservation {
        TextLoadedObservation {
            service: self.model.service.clone(),
            root_pool: Arc::downgrade(&self.root_pool),
        }
    }
    pub fn resources(&self) -> &TextResourceInventory {
        &self.cold.resources
    }
    pub fn source(&self) -> &VerifiedTensorSource {
        &self.cold.source
    }
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.model.service
    }
    pub async fn session(self: &Arc<Self>, cancelled: Option<&AtomicBool>) -> Result<TextSession> {
        request_not_cancelled(cancelled)?;
        self.service().ensure_usable()?;
        let reservation = {
            let _gate = self
                .admission
                .lock()
                .map_err(|_| Error::State("short HF admission gate poisoned".into()))?;
            if self.closed.load(Ordering::Acquire) || self.closing.load(Ordering::Acquire) {
                return Err(Error::State(
                    "short HF loaded source retired/closing".into(),
                ));
            }
            CountReservation::new(self.active.clone(), 8)?
        };
        // Inside the ORIGINAL allocation lock, before first root H2D; retain
        // actual host staging admission through physical completion/unknown.
        let authority = SessionAuthority::new(self.service().source_identity())?;
        let domains = super::state::allocate_pooled_state(
            &self.model,
            &self.root_pool,
            true,
            cancelled,
            |init| {
                let host = inventory::root_host_admission(
                    &self.cold.source,
                    self.cold.resources.root_initialization_host_bytes,
                )?;
                init.retain_completion_owner(Arc::new((self.cold.clone(), host)))
            },
        )
        .await?;
        let original = ResidentSession {
            model: self.model.clone(),
            authority,
            domains,
        };
        request_not_cancelled(cancelled)?;
        Ok(TextSession {
            original,
            loaded: self.clone(),
            feedback: None,
            cohort_root: None,
            _reservation: reservation,
        })
    }
    pub async fn close(
        self: Arc<Self>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<TextCloseReceipt> {
        request_not_cancelled(cancelled)?;
        {
            let _gate = self
                .admission
                .lock()
                .map_err(|_| Error::State("short HF admission gate poisoned".into()))?;
            if self.active.load(Ordering::Acquire) != 0
                || self.pending.load(Ordering::Acquire) != 0
                || self.snapshots.load(Ordering::Acquire) != 0
                || self
                    .cohort
                    .as_ref()
                    .is_some_and(|cohort| cohort.capacity.pending() != 0)
            {
                return Err(Error::State(
                    "short HF close requires all sessions/private proposals/snapshots dropped"
                        .into(),
                ));
            }
            if self.closing.swap(true, Ordering::AcqRel) || self.closed.load(Ordering::Acquire) {
                return Err(Error::State(
                    "short HF source already closing/retired".into(),
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
            || self
                .cohort
                .as_ref()
                .is_some_and(|cohort| cohort.capacity.pending() != 0)
        {
            return Err(Error::State(
                "short HF concurrent session entered close".into(),
            ));
        }
        // Acquired-only quiescence barrier; no device command was submitted.
        drop(workspace);
        drop(slot);
        let first = source_check::verify(self.cold.clone(), cancelled).await?;
        self.closed.store(true, Ordering::Release);
        source_check::retire(self.cold.clone(), self.cohort.clone(), cancelled).await?;
        let second = source_check::verify(self.cold.clone(), cancelled).await?;
        self.closed.store(true, Ordering::Release);
        Ok(TextCloseReceipt {
            source_weight_windows: first.0,
            streamed_head_windows: first.1,
            retirement_source_weight_windows: second.0,
            retirement_streamed_head_windows: second.1,
            resident_head_windows: first.2,
            retirement_resident_head_windows: second.2,
        })
    }
}
impl TextSession {
    pub fn binding(&self) -> RootBinding {
        self.original.binding()
    }
    pub fn snapshot_empty(&self) -> Result<TextEmptySnapshot> {
        if self.binding().frontier() != 0 || self.feedback.is_some() {
            return Err(Error::Unsupported(
                "short HF empty-frontier checkpoint only".into(),
            ));
        }
        let reservation = CountReservation::new(self.loaded.snapshots.clone(), 2)?;
        let metadata = self.loaded.source().reserve_workspace(64 * 1024)?;
        Ok(TextEmptySnapshot {
            original: self.original.snapshot(),
            model: Arc::downgrade(&self.original.model),
            _reservation: reservation,
            _metadata: metadata,
        })
    }
    pub fn restore_empty(
        &mut self,
        snapshot: &TextEmptySnapshot,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        request_not_cancelled(cancelled)?;
        if !snapshot
            .model
            .upgrade()
            .is_some_and(|m| Arc::ptr_eq(&m, &self.original.model))
        {
            return Err(Error::State("short HF foreign empty snapshot".into()));
        }
        self.original.restore(&snapshot.original)?;
        self.feedback = None;
        self.cohort_root = None;
        if self.binding().frontier() != 0 {
            return Err(Error::State(
                "short HF restored snapshot was not empty".into(),
            ));
        }
        Ok(())
    }
    async fn prepare_inner(
        &self,
        input: Arc<TextCallInput>,
        validate: Option<TextValidation<'_>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<TextPending> {
        let mut wall = performance::Phases::new("prepare", input.base() as u64, input.rows());
        request_not_cancelled(cancelled)?;
        if self.loaded.closed.load(Ordering::Acquire) || self.loaded.closing.load(Ordering::Acquire)
        {
            return Err(Error::State("short HF loaded source closing".into()));
        }
        // At most two completed proposals can coexist per loaded model. This
        // also bounds retained immutable packets/candidate roots on stale cuts.
        let reservation = CountReservation::new(self.loaded.pending.clone(), 2)?;
        wall.mark("admission");
        let state = self
            .original
            .bind_fresh_text_state(input.clone(), &self.loaded.root_pool, cancelled)
            .await?;
        wall.mark("fresh_state");
        let prepared = self
            .original
            .prepare_text_call(
                self.loaded.cold.clone(),
                input,
                state,
                None,
                validate,
                cancelled,
            )
            .await
            .map_err(|e| normalize_cancel(e, cancelled, self.loaded.service()))?;
        wall.mark("prepare_call");
        wall.succeed();
        Ok(TextPending {
            prepared: Some(prepared),
            model: Arc::downgrade(&self.original.model),
            _reservation: reservation,
        })
    }
    pub fn feedback_request(&self) -> Result<TextFeedbackRequest<'_>> {
        self.loaded.service().ensure_usable()?;
        if self.loaded.closed.load(Ordering::Acquire) || self.loaded.closing.load(Ordering::Acquire)
        {
            return Err(Error::State("text loaded source closing".into()));
        }
        let packet = self
            .feedback
            .as_ref()
            .ok_or_else(|| Error::State("text no original committed feedback".into()))?;
        packet.require_current(&self.original, self.loaded.source())?;
        let base = usize::try_from(self.binding().frontier())
            .map_err(|_| Error::Limit("text current root frontier".into()))?;
        Ok(TextFeedbackRequest {
            session: self,
            packet,
            base,
        })
    }
    pub async fn prepare_prompt_with<F>(
        &self,
        record: Arc<SourceBindings>,
        implementation: TextGdnImplementation,
        cancelled: Option<&AtomicBool>,
        mut validate: F,
    ) -> Result<TextPending>
    where
        F: for<'a> FnMut(TextReadback<'a>) -> Result<()>,
    {
        request_not_cancelled(cancelled)?;
        if self.feedback.is_some() || self.binding().frontier() != 0 {
            return Err(Error::State(
                "text prompt requires empty root/no feedback".into(),
            ));
        }
        let input = self.original.bind_text_prompt(
            self.loaded.cold.bindings.clone(),
            record,
            implementation,
            self.loaded.cold.capture,
            cancelled,
        )?;
        self.prepare_inner(input, Some(&mut validate), cancelled)
            .await
    }
    pub fn commit(
        &mut self,
        mut pending: TextPending,
        cancelled: Option<&AtomicBool>,
    ) -> Result<TextCallReceipt> {
        let mut wall = performance::Phases::new("commit", self.binding().frontier(), 0);
        if !pending
            .model
            .upgrade()
            .is_some_and(|m| Arc::ptr_eq(&m, &self.original.model))
        {
            return Err(Error::State(
                "short HF proposal belongs to another loaded model".into(),
            ));
        }
        let prepared = pending
            .prepared
            .take()
            .ok_or_else(|| Error::State("short HF proposal consumed".into()))?;
        let (receipt, feedback) = self
            .original
            .commit_text_call(prepared, cancelled)
            .map_err(|e| normalize_cancel(e, cancelled, self.loaded.service()))?;
        self.feedback = Some(feedback);
        self.cohort_root = None;
        wall.mark("commit");
        wall.succeed();
        Ok(receipt)
    }
    pub async fn prepare_next_with<F>(
        &self,
        request: TextFeedbackRequest<'_>,
        record: Arc<SourceBindings>,
        implementation: TextGdnImplementation,
        cancelled: Option<&AtomicBool>,
        mut validate: F,
    ) -> Result<TextPending>
    where
        F: for<'a> FnMut(TextReadback<'a>) -> Result<()>,
    {
        request_not_cancelled(cancelled)?;
        let packet = request.require(self, self.loaded.source())?.clone();
        let input = self.original.bind_text_feedback(
            self.loaded.cold.bindings.clone(),
            record,
            implementation,
            self.loaded.cold.capture,
            packet,
            cancelled,
        )?;
        self.prepare_inner(input, Some(&mut validate), cancelled)
            .await
    }
    pub async fn prepare_prompt(
        &self,
        record: Arc<SourceBindings>,
        implementation: TextGdnImplementation,
        cancelled: Option<&AtomicBool>,
    ) -> Result<TextPending> {
        request_not_cancelled(cancelled)?;
        if self.feedback.is_some() || self.binding().frontier() != 0 {
            return Err(Error::State(
                "text prompt requires empty root/no feedback".into(),
            ));
        }
        let input = self.original.bind_text_prompt(
            self.loaded.cold.bindings.clone(),
            record,
            implementation,
            self.loaded.cold.capture,
            cancelled,
        )?;
        self.prepare_inner(input, None, cancelled).await
    }
    pub async fn prepare_next(
        &self,
        request: TextFeedbackRequest<'_>,
        record: Arc<SourceBindings>,
        implementation: TextGdnImplementation,
        cancelled: Option<&AtomicBool>,
    ) -> Result<TextPending> {
        request_not_cancelled(cancelled)?;
        let packet = request.require(self, self.loaded.source())?.clone();
        let input = self.original.bind_text_feedback(
            self.loaded.cold.bindings.clone(),
            record,
            implementation,
            self.loaded.cold.capture,
            packet,
            cancelled,
        )?;
        self.prepare_inner(input, None, cancelled).await
    }
}

pub async fn prepare_text(
    device: Device,
    source: VerifiedTensorSource,
    capture: TextKernelCapture,
    scopes: Vec<(Arc<SourceBindings>, TextGdnImplementation)>,
    inputs: TextQualificationInputs,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<TextLoaded>> {
    prepare_text_profile(
        false,
        device,
        source,
        capture,
        scopes,
        inputs,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}
/// Explicit source/shape admission; numerical qualification is a separate gate.
pub async fn prepare_text_b4(
    device: Device,
    source: VerifiedTensorSource,
    capture: TextKernelCapture,
    scopes: Vec<(Arc<SourceBindings>, TextGdnImplementation)>,
    inputs: TextQualificationInputs,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<TextLoaded>> {
    prepare_text_profile(
        true,
        device,
        source,
        capture,
        scopes,
        inputs,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}
/// Consume source-authenticated scopes; no model/graph issuance occurs in GPU.
fn cold_source_scopes(
    source: &VerifiedTensorSource,
    capture: TextKernelCapture,
    calls: &[(Arc<SourceBindings>, TextGdnImplementation)],
) -> Result<(Arc<TextModelBindings>, Vec<Arc<TextModelBinding>>)> {
    if calls.len() != capture.cold_scope_count()? {
        return Err(Error::State(
            "text complete captured cold scopes required".into(),
        ));
    }
    let bindings = TextModelBindings::new(source.clone(), capture)?;
    let mut scopes = Vec::with_capacity(calls.len());
    for (index, (record, implementation)) in calls.iter().enumerate() {
        let call = record.call();
        capture.cold_scope_role(index)?.require_call(call)?;
        if !source.same_issuer(record.source())
            || call.rows().capacity() != 512
            || call.rows().vocabulary() != 248320
        {
            return Err(Error::State(
                "text actual source/cold call geometry differs".into(),
            ));
        }
        capture.require_bindings(record)?;
        scopes.push(bindings.bind(record.clone(), *implementation, capture)?);
    }
    Ok((bindings, scopes))
}
#[expect(
    clippy::too_many_arguments,
    reason = "capture/call provenance, qualifier inputs, physical capacity and cancellation are independent admission inputs"
)]
async fn prepare_text_profile(
    cohort_b4: bool,
    device: Device,
    source: VerifiedTensorSource,
    capture: TextKernelCapture,
    calls: Vec<(Arc<SourceBindings>, TextGdnImplementation)>,
    inputs: TextQualificationInputs,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<TextLoaded>> {
    request_not_cancelled(cancelled)?;
    validate_provider(&device)?;
    if source.staging_chunk_bytes() != MIB
        || inputs.native.projections.device_cap != 5120 * MIB as u64
    {
        return Err(Error::Limit(
            "text original1MiB source /5GiB device bounds".into(),
        ));
    }
    let TextQualificationInputs {
        native: inputs,
        rotary,
    } = inputs;
    let prefix = rotary
        .ok_or_else(|| Error::State("short HF genuine source rotary missing".into()))?
        .into_text_prefix()?;
    prefix.require_source(&source)?;
    let rotary_positions = prefix.positions();
    prefix.require_span(0, rotary_positions)?;
    let rotary_retained_host_bytes = prefix.retained_host_bytes()?;
    let metadata = Arc::new(source.reserve_workspace(8 * MIB)?);
    let owner_metadata = Arc::new(source.reserve_workspace(2 * MIB)?);
    let (bindings, scopes) = cold_source_scopes(&source, capture, &calls)?;
    let readiness_geometry = TextColdGeometry::from_scopes(capture, &scopes)?;
    let preference = crate::cuda::native::text_head::HeadMode::environment_preference()?;
    if provider_capacity_bytes == 0
        || provider_capacity_bytes as u64 > inputs.projections.device_cap
    {
        return Err(Error::Limit(
            "short-HF provider allowance exceeds physical envelope".into(),
        ));
    }
    let physical = crate::resident_leases::ResidentPhysical::new_pinned_cuda(
        device,
        provider_capacity_bytes,
        inputs.projections.device_cap,
    )?;
    let service = physical.attach_model(
        crate::resident_leases::ResidentSource::Windows(source.clone()),
        ResourceOwner::new("bf16-text-source", "shared-six-geometry-source")?,
        provider_capacity_bytes,
    )?;
    let compiler_context = service.admit_cuda_compiler_context(&source)?;
    let memory = zerogravity_core::optimization::MemoryFootprint {
        host_bytes: source.workspace_limit_bytes()?,
        device_bytes: provider_capacity_bytes as u64,
    };
    let source_admission = inventory::TextSourceAdmission {
        rotary_positions,
        rotary_retained_host_bytes,
        compiler_context: Some(&compiler_context),
    };
    let (head_mode, resources) = if cohort_b4 {
        TextResourceInventory::select_profile(
            true,
            &source,
            capture,
            &scopes,
            preference,
            memory,
            source_admission,
        )?
    } else {
        TextResourceInventory::select_profile(
            false,
            &source,
            capture,
            &scopes,
            preference,
            memory,
            source_admission,
        )?
    };
    if service.client().properties().memory.alignment != 512 {
        return Err(Error::Unsupported(
            "short HF actual512-byte managed alignment".into(),
        ));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: provider_capacity_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    request_not_cancelled(cancelled)?;
    slot.begin_submission()?;
    // One original workspace serializes the captured geometries, including D2H.
    // Price the exact maximum retained diagnostic cohort once.
    let readback = FullModelProgram::admit_text_readbacks(&scopes, &service)?;
    let values =
        FullModelProgram::prepare_text_shared_cold(&scopes, service.clone(), readback, cancelled)
            .await?;
    let program = ResidentProgram::from_natural_body(
        source.clone(),
        values[0].body.clone(),
        values[0].buffers.clone(),
        values[0].preflight.clone(),
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
    let root_layout = super::state::root_pool_layout(&model)?;
    let pool_metadata = source
        .reserve_workspace(ResidentPool::metadata_bytes(
            root_layout.len(),
            resources.root_count,
        )?)?
        .into_host_workspace()?;
    let root_pool = ResidentPool::prepare(
        service.clone(),
        pool_metadata,
        &root_layout,
        resources.root_count,
    )?;
    let mut init = service.begin_initialization(0)?;
    init.retain_completion_owner(metadata.clone())?;
    init.retain_completion_owner(owner_metadata.clone())?;
    for v in &values {
        init.retain_completion_owner(v.body.clone())?;
    }
    let pending = PendingModelWeights::upload_text(&scopes[0], &mut init, cancelled).await?;
    let pending_tied = if head_mode.resident() {
        Some(
            init.upload_tied_bf16(
                source.clone(),
                head_member(scopes[0].record())?.clone(),
                cancelled,
            )
            .await?,
        )
    } else {
        None
    };
    let mut parameters = BTreeMap::new();
    for (layer, _) in scopes[0].gdn_sites() {
        parameters.insert(
            layer,
            ModelGdnParameters::prepare_text(&scopes[0], layer, &service, &mut init, cancelled)?,
        );
    }
    let positions = TextPositionsStorage::prepare(&mut init)?;
    let rotary = ModelRotary::prepare_text(
        source.clone(),
        service.clone(),
        prefix,
        resources.rotary_uploaded_bytes,
        &mut init,
    )?;
    let mut private = Vec::with_capacity(scopes.len());
    for (scope, value) in scopes.iter().zip(&values) {
        private.push(TextGdnWorkspace::allocate(
            scope.clone(),
            value.body.values().clone(),
            &mut init,
        )?);
    }
    drop(init.complete_sync().await?);
    // Readiness follows actual upload completion; source checks precede uploads.
    for (&layer, parameter) in &parameters {
        parameter.require_text(&scopes[0], layer)?;
    }
    let weights = pending.bind_initialized()?;
    let tied_head = pending_tied.map(|p| p.bind_initialized()).transpose()?;
    let mut gdn = Vec::with_capacity(scopes.len());
    for ((scope, value), workspace) in scopes.iter().zip(&values).zip(&private) {
        let mut cohort = BTreeMap::new();
        for (&layer, p) in &parameters {
            cohort.insert(
                layer,
                ColdTextGdn::prepare(
                    scope.clone(),
                    value.body.values().clone(),
                    p.clone(),
                    workspace.clone(),
                    layer,
                    cancelled,
                )
                .await?,
            );
        }
        gdn.push(cohort);
    }
    let mut geometries = Vec::with_capacity(scopes.len());
    for (scope, value) in scopes.iter().zip(&values) {
        geometries.push(TextNativeGeometry {
            scope: scope.clone(),
            values: value.body.values().clone(),
            rms_scope: value.rms_scope.clone(),
            final_rms: value.body.final_normalizer()?,
        });
    }
    let mut init = service.begin_initialization(0)?;
    // Acyclic complete cold aggregate precedes actual module preparation. The
    // real workspace allocation keeps this original initialization nonempty.
    init.retain_completion_owner(Arc::new((
        values.iter().map(|v| v.body.clone()).collect::<Vec<_>>(),
        gdn.clone(),
        private.clone(),
        weights.clone(),
        parameters.clone(),
        rotary.clone(),
        positions.clone(),
    )))?;
    let workspace = init.allocate_empty(8_519_680, BufferRole::Scratch)?;
    let native = TextNativeStorage::prepare(
        TextNativePreparation {
            source: source.clone(),
            capture,
            geometries,
            weights,
            workspace,
            rotary: rotary.clone(),
            positions: positions.clone(),
            inputs,
            audit_head: false,
            cohort_b4,
            head_mode,
            tied_head,
        },
        &mut init,
        cancelled,
    )?;
    let cold = Arc::new(ColdModel {
        source,
        capture,
        bindings,
        values,
        gdn,
        native,
        positions,
        parameters,
        rotary,
        private,
        _metadata: metadata,
        _owner_metadata: owner_metadata,
        resources,
    });
    init.retain_completion_owner(cold.clone())?;
    drop(init.complete_sync().await?);
    let cohort = if cohort_b4 {
        Some(cohort::ColdCohort::prepare(&cold, &scopes, service.clone(), cancelled).await?)
    } else {
        None
    };
    let readiness_metadata = Arc::new(cold.source.reserve_workspace(64 * 1024)?);
    let readiness = compiler_context.begin_preparation(
        service.budget(),
        &cold.source,
        vec![readiness_metadata, Arc::new((cold.clone(), cohort.clone()))],
    )?;
    let queue = service.pinned_cuda_queue()?;
    let prepared = (|| -> Result<usize> {
        let mut count = 0;
        for which in 0..cold.values.len() {
            request_not_cancelled(cancelled)?;
            count += cold.values[which].rms_scope.prepare_code(queue)?;
            count += cold.values[which].body.prepare_code(queue)?;
            for gdn in cold.gdn[which].values() {
                request_not_cancelled(cancelled)?;
                count += gdn.prepare_code(queue)?;
            }
            // Same source-issued state requirements as the actual candidate
            // rows; the runtime certifies every plane via `ResidentProgram::certify`.
            count += crate::program_kernels::prepare_state_certificates(
                service.client(),
                queue,
                &cold.values[which].preflight.states,
            )?;
        }
        count += cold.native.prepare_finite_code(queue)?;
        count += crate::cuda::gdn::prepare_state_code(queue)?;
        count += crate::program_kernels::prepare_text_code(
            service.client(),
            queue,
            cohort_b4,
            cold.capture,
        )?;
        count += crate::resident_program::bf16_greedy::prepare_text_code(
            service.client(),
            queue,
            cohort_b4,
        )?;
        if let Some(cohort) = &cohort {
            count += cohort.prepare_code(&cold.source, queue)?;
        }
        Ok(count)
    })();
    let readiness_candidates = match prepared {
        Ok(count) => count,
        Err(error) => {
            compiler_context.fence_unknown();
            return Err(error);
        }
    };
    let readiness_usage = queue
        .compiler_context_usage()
        .map_err(|e| Error::State(format!("original compiler usage: {e}")))?
        .ok_or_else(|| Error::State("original compiler context absent after preparation".into()))?;
    readiness.complete()?;
    slot.finish()?;
    Ok(Arc::new(TextLoaded {
        admission: std::sync::Mutex::new(()),
        model,
        root_pool,
        cold,
        active: Arc::new(AtomicUsize::new(0)),
        pending: Arc::new(AtomicUsize::new(0)),
        snapshots: Arc::new(AtomicUsize::new(0)),
        closed: AtomicBool::new(false),
        closing: AtomicBool::new(false),
        cohort,
        readiness_geometry,
        readiness_candidates,
        readiness_usage,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_session_pending_snapshot_counts_release_on_abort() {
        let sessions = Arc::new(AtomicUsize::new(0));
        let pending = Arc::new(AtomicUsize::new(0));
        let mut admitted = Vec::new();
        for _ in 0..8 {
            admitted.push(CountReservation::new(sessions.clone(), 8).unwrap());
        }
        assert!(CountReservation::new(sessions.clone(), 8).is_err());
        let one = CountReservation::new(pending.clone(), 2).unwrap();
        let two = CountReservation::new(pending.clone(), 2).unwrap();
        assert!(CountReservation::new(pending.clone(), 2).is_err());
        assert_eq!(sessions.load(Ordering::Acquire), 8);
        drop(one);
        assert_eq!(pending.load(Ordering::Acquire), 1);
        drop(two);
        drop(admitted);
        assert_eq!(
            (
                sessions.load(Ordering::Acquire),
                pending.load(Ordering::Acquire)
            ),
            (0, 0)
        );
    }
    #[test]
    fn cancelled_preflight_is_not_a_device_or_publication_fact() {
        let flag = AtomicBool::new(true);
        assert!(
            matches!(request_not_cancelled(Some(&flag)),Err(Error::State(s)) if s=="short HF request cancelled")
        );
        flag.store(false, Ordering::Release);
        request_not_cancelled(Some(&flag)).unwrap();
        request_not_cancelled(None).unwrap();
    }
}
