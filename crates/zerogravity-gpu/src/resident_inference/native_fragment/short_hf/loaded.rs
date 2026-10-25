//! One loaded immutable source; original independent roots and committed feedback.
use super::completion::PreparedCall;
use super::*;
use crate::resident_leases::backing_pool::ResidentPool;
use crate::resident_program::text_positions::short_hf::ShortHfPositionsStorage;
use crate::resident_program::{
    ResidentProgram,
    native_full::{FullModelProgram, short_hf::ShortHfColdValues},
};
use crate::vllm_cuda_gdn::ShortHfGdnWorkspace;
use crate::vllm_cuda_qualification::{
    Mlp47QualificationInputs,
    model_gdn::{ColdShortHfGdn, ModelGdnParameters},
    model_rotary::ModelRotary,
    model_weights::PendingModelWeights,
    short_hf_native::{ShortHfNativeGeometry, ShortHfNativePreparation, ShortHfNativeStorage},
    source_rotary::RotaryOutcome,
};
use std::{collections::BTreeMap, sync::atomic::AtomicUsize};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};
mod faults;
pub(super) use faults::ProbeRecord;
pub use faults::{ShortHfFault, ShortHfFaultOutcome};
pub(super) mod performance;
mod source_check;
mod workspace;
const MIB: usize = 1024 * 1024;
/// Explicit short-HF HOST policy. The source's own workspace limit remains
/// authoritative; CPU and other models keep their existing limits.
fn supported_host_limit(bytes: u64) -> bool {
    bytes == 4096 * MIB as u64 || bytes == 8192 * MIB as u64
}
const N1_PROMPT_GEOMETRIES: usize = 5;
pub(crate) mod cohort;
mod inventory;
pub use cohort::{
    ShortHfCohortFault, ShortHfCohortFaultOutcome, ShortHfCohortFaultRetention,
    ShortHfCohortPreparation, ShortHfCohortReadback, ShortHfCohortReceipt, ShortHfCohortRowReceipt,
    ShortHfPendingCohort,
};
pub use inventory::ShortHfResourceInventory;

pub struct ShortHfQualificationInputs {
    pub native: Mlp47QualificationInputs,
    pub rotary: Option<RotaryOutcome>,
}
pub(super) struct ColdModel {
    source: BoundFullModelSource,
    bindings: Arc<ShortHfModelBindings>,
    values: Vec<ShortHfColdValues>,
    gdn: Vec<BTreeMap<usize, Arc<ColdShortHfGdn>>>,
    native: Arc<ShortHfNativeStorage>,
    positions: Arc<ShortHfPositionsStorage>,
    parameters: BTreeMap<usize, Arc<ModelGdnParameters>>,
    rotary: Arc<ModelRotary>,
    private: Vec<Arc<ShortHfGdnWorkspace>>,
    _metadata: Arc<WindowWorkspace>,
    _owner_metadata: Arc<WindowWorkspace>,
    resources: ShortHfResourceInventory,
}
impl ColdModel {
    pub(super) async fn bind_program(
        &self,
        input: Arc<ShortHfCallInput>,
        state: Arc<ShortHfCallState>,
        cancel: Option<&AtomicBool>,
    ) -> Result<ResidentProgram> {
        let which = if input.rows() == 1 {
            5
        } else {
            [47, 57, 62, 67, 72]
                .iter()
                .position(|&r| r == input.rows())
                .ok_or_else(|| Error::Unsupported("short HF cold geometry absent".into()))?
        };
        let cold = &self.values[which];
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
        body.bind_short_hf_call(input.clone(), state, native, positions, rms_scope, gdn)?;
        ResidentProgram::from_natural_body(
            input.source().source().clone(),
            body,
            buffers,
            preflight,
        )
    }
}
/// Bounded N1 coverage derived from the source-issued cold scopes actually prepared.
#[derive(Clone, Debug)]
struct ShortHfColdGeometry {
    prompt_rows: [usize; N1_PROMPT_GEOMETRIES],
    cached_rows: usize,
}
impl ShortHfColdGeometry {
    fn from_scopes(scopes: &[Arc<ShortHfModelBinding>]) -> Result<Self> {
        let mut prompt_rows = [0; N1_PROMPT_GEOMETRIES];
        let mut prompts = 0;
        let mut cached_rows = 0;
        for scope in scopes {
            let span = scope.call().span();
            match span.mode() {
                VllmBf16Mode::Prefill => {
                    let slot = prompt_rows.get_mut(prompts).ok_or_else(|| {
                        Error::State("short HF prepared N1 prompt geometry overflow".into())
                    })?;
                    *slot = span.workload().prompt_rows();
                    prompts += 1;
                }
                VllmBf16Mode::Decode => cached_rows += 1,
            }
        }
        if prompts != N1_PROMPT_GEOMETRIES || cached_rows == 0 {
            return Err(Error::State(
                "short HF prepared N1 geometry coverage incomplete".into(),
            ));
        }
        Ok(Self {
            prompt_rows,
            cached_rows,
        })
    }
    fn prompt_rows(&self) -> &[usize] {
        &self.prompt_rows
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
pub struct ShortHfLoaded {
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
    readiness_geometry: ShortHfColdGeometry,
    readiness_candidates: usize,
    readiness_usage: cubecl::cuda::CompilerContextUsage,
}
pub struct ShortHfSession {
    original: ResidentSession,
    loaded: Arc<ShortHfLoaded>,
    feedback: Option<Arc<CommittedShortHfPacket>>,
    cohort_root: Option<RootBinding>,
    _reservation: CountReservation,
}
pub struct ShortHfPending {
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

pub struct ShortHfEmptySnapshot {
    original: ResidentSnapshot,
    model: Weak<ResidentModel>,
    _reservation: CountReservation,
    _metadata: WindowWorkspace,
}
#[derive(Clone, Debug)]
pub struct ShortHfCallReceipt {
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
pub struct ShortHfCloseReceipt {
    pub source_weight_windows: usize,
    pub streamed_head_windows: usize,
    pub retirement_source_weight_windows: usize,
    pub retirement_streamed_head_windows: usize,
    pub resident_head_windows: usize,
    pub retirement_resident_head_windows: usize,
}

/// Retains only the original service for post-retirement ledger observation.
pub struct ShortHfLoadedObservation {
    service: Arc<ResidentService>,
    root_pool: Weak<ResidentPool>,
}
#[derive(Clone, Debug)]
pub struct ShortHfLoadedSnapshot {
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
impl ShortHfLoadedObservation {
    pub fn snapshot(&self) -> Result<ShortHfLoadedSnapshot> {
        let pool = self
            .root_pool
            .upgrade()
            .map(|pool| pool.snapshot())
            .transpose()?
            .unwrap_or_default();
        let (backend_host_reserved_bytes, backend_device_reserved_bytes, compiler_cumulative_usage) =
            self.service.observe_cuda_compiler_context()?;
        Ok(ShortHfLoadedSnapshot {
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

fn validate_prompt_contract(ids: &[u32], max_tokens: usize) -> Result<()> {
    if !matches!(ids.len(), 47 | 57 | 62 | 67 | 72)
        || ids.iter().any(|&id| id >= 248320)
        || !(1..=8).contains(&max_tokens)
    {
        return Err(Error::Unsupported(
            "short HF finite prompt rows47/57/62/67/72 and1..8 outputs only".into(),
        ));
    }
    Ok(())
}
impl ShortHfLoaded {
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
    pub fn observation(&self) -> ShortHfLoadedObservation {
        ShortHfLoadedObservation {
            service: self.model.service.clone(),
            root_pool: Arc::downgrade(&self.root_pool),
        }
    }
    pub fn validate_prompt(&self, ids: &[u32], max_tokens: usize) -> Result<()> {
        self.service().ensure_usable()?;
        if self.closed.load(Ordering::Acquire) || self.closing.load(Ordering::Acquire) {
            return Err(Error::State("short HF loaded source closing".into()));
        }
        validate_prompt_contract(ids, max_tokens)
    }
    pub fn resources(&self) -> &ShortHfResourceInventory {
        &self.cold.resources
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
    ) -> Result<ShortHfSession> {
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
        Ok(ShortHfSession {
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
    ) -> Result<ShortHfCloseReceipt> {
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
        Ok(ShortHfCloseReceipt {
            source_weight_windows: first.0,
            streamed_head_windows: first.1,
            retirement_source_weight_windows: second.0,
            retirement_streamed_head_windows: second.1,
            resident_head_windows: first.2,
            retirement_resident_head_windows: second.2,
        })
    }
}
impl ShortHfSession {
    pub fn binding(&self) -> RootBinding {
        self.original.binding()
    }
    pub fn snapshot_empty(&self) -> Result<ShortHfEmptySnapshot> {
        if self.binding().frontier() != 0 || self.feedback.is_some() {
            return Err(Error::Unsupported(
                "short HF empty-frontier checkpoint only".into(),
            ));
        }
        let reservation = CountReservation::new(self.loaded.snapshots.clone(), 2)?;
        let metadata = self
            .loaded
            .source()
            .source()
            .source()
            .reserve_workspace(64 * 1024)?;
        Ok(ShortHfEmptySnapshot {
            original: self.original.snapshot(),
            model: Arc::downgrade(&self.original.model),
            _reservation: reservation,
            _metadata: metadata,
        })
    }
    pub fn restore_empty(
        &mut self,
        snapshot: &ShortHfEmptySnapshot,
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
        input: Arc<ShortHfCallInput>,
        validate: Option<ShortHfValidation<'_>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ShortHfPending> {
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
            .bind_fresh_short_hf_state(input.clone(), &self.loaded.root_pool, cancelled)
            .await?;
        wall.mark("fresh_state");
        let prepared = self
            .original
            .prepare_short_hf_call(
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
        Ok(ShortHfPending {
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
    ) -> Result<ShortHfPending>
    where
        F: for<'a> FnMut(ShortHfReadback<'a>) -> Result<()>,
    {
        request_not_cancelled(cancelled)?;
        let input = self.original.bind_short_hf_prompt(
            self.loaded.cold.bindings.clone(),
            ids,
            cancelled,
        )?;
        self.prepare_inner(input, Some(&mut validate), cancelled)
            .await
    }
    pub fn commit(
        &mut self,
        mut pending: ShortHfPending,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ShortHfCallReceipt> {
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
            .commit_short_hf_call(prepared, cancelled)
            .map_err(|e| normalize_cancel(e, cancelled, self.loaded.service()))?;
        self.feedback = Some(feedback);
        self.cohort_root = None;
        wall.mark("commit");
        wall.succeed();
        Ok(receipt)
    }
    pub async fn infer_prompt_with<F>(
        &mut self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<ShortHfCallReceipt>
    where
        F: for<'a> FnMut(ShortHfReadback<'a>) -> Result<()>,
    {
        let p = self.prepare_prompt_with(ids, cancelled, validate).await?;
        self.commit(p, cancelled)
    }
    pub async fn prepare_next_with<F>(
        &self,
        cancelled: Option<&AtomicBool>,
        mut validate: F,
    ) -> Result<ShortHfPending>
    where
        F: for<'a> FnMut(ShortHfReadback<'a>) -> Result<()>,
    {
        request_not_cancelled(cancelled)?;
        let feedback = self
            .feedback
            .clone()
            .ok_or_else(|| Error::State("short HF no original committed feedback".into()))?;
        let input = self.original.bind_short_hf_feedback(
            self.loaded.cold.bindings.clone(),
            feedback,
            cancelled,
        )?;
        self.prepare_inner(input, Some(&mut validate), cancelled)
            .await
    }
    pub async fn infer_next_with<F>(
        &mut self,
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<ShortHfCallReceipt>
    where
        F: for<'a> FnMut(ShortHfReadback<'a>) -> Result<()>,
    {
        let pending = self.prepare_next_with(cancelled, validate).await?;
        self.commit(pending, cancelled)
    }
    pub async fn infer_prompt(
        &mut self,
        ids: &[u32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<ShortHfCallReceipt> {
        request_not_cancelled(cancelled)?;
        let input = self.original.bind_short_hf_prompt(
            self.loaded.cold.bindings.clone(),
            ids,
            cancelled,
        )?;
        let p = self.prepare_inner(input, None, cancelled).await?;
        self.commit(p, cancelled)
    }
    pub async fn infer_next(
        &mut self,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ShortHfCallReceipt> {
        request_not_cancelled(cancelled)?;
        let feedback = self
            .feedback
            .clone()
            .ok_or_else(|| Error::State("short HF no original committed feedback".into()))?;
        let input = self.original.bind_short_hf_feedback(
            self.loaded.cold.bindings.clone(),
            feedback,
            cancelled,
        )?;
        let p = self.prepare_inner(input, None, cancelled).await?;
        self.commit(p, cancelled)
    }
}

pub async fn prepare_short_hf(
    device: Device,
    source: BoundFullModelSource,
    inputs: ShortHfQualificationInputs,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<ShortHfLoaded>> {
    prepare_short_hf_profile(
        false,
        device,
        source,
        inputs,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}
/// Explicit source/shape admission; numerical qualification is a separate gate.
pub async fn prepare_short_hf_b4(
    device: Device,
    source: BoundFullModelSource,
    inputs: ShortHfQualificationInputs,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<ShortHfLoaded>> {
    prepare_short_hf_profile(
        true,
        device,
        source,
        inputs,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}
/// Same source-owned cold six-shape issuer for loading and cost inspection.
fn cold_source_scopes(
    source: &BoundFullModelSource,
) -> Result<(Arc<ShortHfModelBindings>, Vec<Arc<ShortHfModelBinding>>)> {
    let identity = source.source().source_identity();
    // Cold issuer IDs supply source-owned shape/storage only. These bodies are
    // never dispatched or committed; each call binds fresh actual IDs/record.
    let bindings = ShortHfModelBindings::new(source.clone())?;
    let mut scopes = Vec::with_capacity(6);
    for rows in [47, 57, 62, 67, 72, 1] {
        let workload = ShortHfWorkload::from_prompt_rows(if rows == 1 { 47 } else { rows })?;
        let span = if rows == 1 {
            ShortHfSpan::cached(identity, workload, 47, 0)?
        } else {
            ShortHfSpan::prefill(identity, workload, &vec![0; rows])?
        };
        let call = source.bind_short_hf_span(span)?;
        let record = Arc::new(call.record_prepared()?);
        scopes.push(bindings.bind(call, record)?);
    }
    Ok((bindings, scopes))
}
async fn prepare_short_hf_profile(
    cohort_b4: bool,
    device: Device,
    source: BoundFullModelSource,
    inputs: ShortHfQualificationInputs,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<ShortHfLoaded>> {
    request_not_cancelled(cancelled)?;
    validate_provider(&device)?;
    let source_host_limit = source.source().source().workspace_limit_bytes()?;
    if source.source().source().staging_chunk_bytes() != MIB
        || !supported_host_limit(source_host_limit)
        || inputs.native.projections.device_cap != 5120 * MIB as u64
    {
        return Err(Error::Limit(
            "short HF original1MiB source /supported4or8GiB host /5GiB device bounds".into(),
        ));
    }
    let ShortHfQualificationInputs {
        native: inputs,
        rotary,
    } = inputs;
    let prefix = rotary
        .ok_or_else(|| Error::State("short HF genuine source rotary missing".into()))?
        .into_prefix()?;
    prefix.require_source(source.source())?;
    prefix.require_span(0, 79)?;
    let metadata = Arc::new(source.source().source().reserve_workspace(8 * MIB)?);
    let owner_metadata = Arc::new(source.source().source().reserve_workspace(2 * MIB)?);
    let (bindings, scopes) = cold_source_scopes(&source)?;
    let readiness_geometry = ShortHfColdGeometry::from_scopes(&scopes)?;
    let preference =
        crate::vllm_cuda_qualification::short_hf_head::HeadMode::environment_preference()?;
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
        crate::resident_leases::ResidentSource::Windows(source.source().source().clone()),
        ResourceOwner::new("short-hf-bf16", "shared-six-geometry-original")?,
        provider_capacity_bytes,
    )?;
    let compiler_context = service.admit_cuda_compiler_context(source.source().source())?;
    let memory = zerogravity_core::optimization::MemoryFootprint {
        host_bytes: source.source().source().workspace_limit_bytes()?,
        device_bytes: provider_capacity_bytes as u64,
    };
    let (head_mode, resources) = if cohort_b4 {
        ShortHfResourceInventory::select_profile(
            true,
            &source,
            &scopes,
            preference,
            memory,
            Some(&compiler_context),
        )?
    } else {
        ShortHfResourceInventory::select_profile(
            false,
            &source,
            &scopes,
            preference,
            memory,
            Some(&compiler_context),
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
    // One original workspace serializes all six geometries, including D2H.
    // Price the exact maximum retained diagnostic cohort once, not six times.
    let readback = FullModelProgram::admit_short_hf_readbacks(&scopes, &service)?;
    let values = FullModelProgram::prepare_short_hf_shared_cold(
        &scopes,
        service.clone(),
        readback,
        cancelled,
    )
    .await?;
    let program = ResidentProgram::from_natural_body(
        source.source().clone(),
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
        .source()
        .source()
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
    let pending = PendingModelWeights::upload_short_hf(&scopes[0], &mut init, cancelled).await?;
    let pending_tied = if head_mode.resident() {
        Some(
            init.upload_original_tied_bf16(source.source().clone(), cancelled)
                .await?,
        )
    } else {
        None
    };
    let mut parameters = BTreeMap::new();
    for layer in 0..24 {
        if layer % 4 != 3 {
            parameters.insert(
                layer,
                ModelGdnParameters::prepare_short_hf(
                    &scopes[0], layer, &service, &mut init, cancelled,
                )?,
            );
        }
    }
    let positions = ShortHfPositionsStorage::prepare(&mut init)?;
    let rotary =
        ModelRotary::prepare_short_hf(source.source().clone(), service.clone(), prefix, &mut init)?;
    let mut private = Vec::with_capacity(6);
    for (scope, value) in scopes.iter().zip(&values) {
        private.push(ShortHfGdnWorkspace::allocate(
            scope.clone(),
            value.body.values().clone(),
            &mut init,
        )?);
    }
    drop(init.complete_sync().await?);
    // Readiness follows actual upload completion; source checks precede uploads.
    for (&layer, parameter) in &parameters {
        parameter.require_short_hf(&scopes[0], layer)?;
    }
    let weights = pending.bind_initialized()?;
    let tied_head = pending_tied.map(|p| p.bind_initialized()).transpose()?;
    let mut gdn = Vec::with_capacity(6);
    for ((scope, value), workspace) in scopes.iter().zip(&values).zip(&private) {
        let mut cohort = BTreeMap::new();
        for (&layer, p) in &parameters {
            cohort.insert(
                layer,
                ColdShortHfGdn::prepare(
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
    let mut geometries = Vec::with_capacity(6);
    for (scope, value) in scopes.iter().zip(&values) {
        geometries.push(ShortHfNativeGeometry {
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
    let native = ShortHfNativeStorage::prepare(
        ShortHfNativePreparation {
            source: source.clone(),
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
    let readiness_metadata = Arc::new(cold.source.source().source().reserve_workspace(64 * 1024)?);
    let readiness = compiler_context.begin_preparation(
        service.budget(),
        cold.source.source().source(),
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
        count += crate::vllm_cuda_gdn::prepare_state_code(queue)?;
        count += crate::program_kernels::prepare_short_hf_code(
            service.client(),
            queue,
            cohort_b4,
            cold.source.model_profile(),
        )?;
        count += crate::resident_program::bf16_greedy::prepare_short_hf_code(
            service.client(),
            queue,
            cohort_b4,
        )?;
        if let Some(cohort) = &cohort {
            count += cohort.prepare_code(cold.source.source().source(), queue)?;
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
    Ok(Arc::new(ShortHfLoaded {
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
    fn exact_five_geometry_and_output_bounds_precede_model_admission() {
        for n in [47, 57, 62, 67, 72] {
            let mut ids = vec![248319; n];
            for max in 1..=8 {
                validate_prompt_contract(&ids, max).unwrap();
            }
            ids[n - 1] = 248320;
            assert!(validate_prompt_contract(&ids, 8).is_err());
            for max in [0, 9, usize::MAX] {
                assert!(validate_prompt_contract(&vec![0; n], max).is_err());
            }
        }
        for n in [0, 1, 2, 46, 48, 56, 58, 61, 63, 64, 66, 68, 71, 73, 512] {
            assert!(validate_prompt_contract(&vec![0; n], 8).is_err());
        }
    }
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
