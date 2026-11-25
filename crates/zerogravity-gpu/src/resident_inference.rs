//! Physical execution of the engine's one recorded schedule and immutable roots.
//! Arithmetic and layer ordering belong to ForwardPlan; this owner only binds
//! leases, private state, actual completion and the common logical authority.
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) mod native_fragment;
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    any(feature = "hip", feature = "vulkan")
))]
#[path = "resident_inference/portable_qualification.rs"]
mod portable_qualification;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    GdnFragmentFault, GdnFragmentFaultOutcome, GdnFragmentReadback, GdnFragmentReceipt,
    GdnSequenceFault, GdnSequenceFaultOutcome, GdnSequenceReceipt, GdnSequenceRestoreReceipt,
    qualify_gdn_fragment, qualify_gdn_fragment_fault, qualify_gdn_mlp_fragment,
    qualify_gdn_mlp_fragment_fault, qualify_gdn_sequence, qualify_gdn_sequence_fault,
    qualify_gdn_sequence_restore_refusal,
};
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    feature = "hip",
    feature = "vulkan",
    feature = "hardware-abort-probe"
))]
pub use portable_qualification::{SamePciPeerReceipt, same_pci_peer};
#[cfg(all(test, feature = "qualification", feature = "cuda"))]
#[path = "resident_inference/admission_qualification.rs"]
mod admission_qualification;
#[cfg(feature = "qualification")]
#[path = "resident_alias_qualification.rs"]
mod alias_qualification;
#[cfg(feature = "portable-checkpoint")]
#[path = "resident_checkpoint.rs"]
mod checkpoint;
#[cfg(all(
    feature = "portable-checkpoint",
    feature = "qualification",
    feature = "hardware-abort-probe"
))]
#[path = "resident_checkpoint_abort.rs"]
mod checkpoint_abort;
mod fused;
#[cfg(feature = "qualification")]
#[path = "resident_fused_greedy_qualification.rs"]
mod fused_greedy_qualification;
#[cfg(feature = "qualification")]
#[path = "resident_fused_qualification.rs"]
mod fused_qualification;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) mod gdn_state;
#[cfg(all(
    feature = "cuda",
    any(feature = "qualification", feature = "cuda-native")
))]
mod graph;
#[allow(
    dead_code,
    reason = "paged physical adapter is retained for the next resident KV integration slice"
)]
mod paged_state;

#[cfg(all(
    test,
    feature = "qualification",
    feature = "portable-checkpoint",
    feature = "cuda"
))]
#[path = "resident_inference/ragged_qualification.rs"]
mod ragged_qualification;

#[cfg(feature = "qualification")]
#[path = "resident_prefill_qualification.rs"]
mod prefill_qualification;

#[cfg(feature = "qualification")]
#[path = "resident_range_qualification.rs"]
mod range_qualification;
#[cfg(feature = "qualification")]
#[path = "resident_view_qualification.rs"]
mod view_qualification;
use crate::backend::validate_provider;
use crate::resident_batch::ResidentBatchRows;
use crate::resident_derived_state::{DerivedStateCache, DerivedStateUse};
use crate::resident_leases::{
    ResidentLease, ResidentPhysical, ResidentReady, ResidentService, ResidentSource,
};
#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
use crate::resident_program::cuda_forward;
use crate::resident_program::{
    AllocationInit, BatchProgram, DeviceBuffer, ResidentBufferLayout, ResidentBufferRole,
    ResidentOutput, ResidentPhysicalType, ResidentPreflight, ResidentProgram, RuntimeParameters,
    StateBinding, StateBuffers, StateLayout, StateRequirement,
};
use async_lock::Mutex;
use cubecl::{Device, bytes::Bytes, server::Handle};
#[cfg(all(test, feature = "qualification", feature = "cuda"))]
use std::path::Path;
use std::sync::Arc;
#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::program::{
    ForwardPlan, ParameterSlot, PlanObservation as LayerObservation, ValueShape,
};
use zerogravity_core::accelerator::program::{PlanOperation, StateSlot};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
use zerogravity_core::resource_arbiter::PhysicalDeviceKey;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};
use zerogravity_core::session_root::{RootAttempt, RootBinding, SessionAuthority};
use zerogravity_core::source::TensorSource;
#[cfg(all(test, feature = "qualification", feature = "cuda"))]
use zerogravity_model_qwen35::LoadedQwenModel;

mod state_pool;
struct Workspace {
    greedy: Option<[Handle; 2]>,
    batch: Option<BatchWorkspace>,
    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    forward_cohort_values: Option<Arc<cuda_forward::CudaForwardCohortValues>>,
    program: ResidentProgram,
    // Raw aliases/programs drop before final leases trigger source cleanup.
    leases: Vec<(ResidentLease, DeviceAccess)>,
    state_accesses: Vec<(ResidentLease, DeviceAccess)>,
    logit_accesses: Vec<(ResidentLease, DeviceAccess)>,
}
struct BatchWorkspace {
    #[cfg(all(
        feature = "cuda",
        any(feature = "qualification", feature = "cuda-native")
    ))]
    graph_cache: graph::ProductionGraphCache,
    program: BatchProgram,
    leases: Vec<(ResidentLease, DeviceAccess)>,
    rows: usize,
}

#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaForwardWeightStorage {
    Resident,
    Streamed,
}

/// Exact host-only result of the same cold-fit calculation used by forward
/// CUDA model loading. No device/client/resource owner is created.
#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaForwardSourceInventory {
    pub provider_capacity_bytes: usize,
    pub fit_capacity_bytes: usize,
    pub context_capacity_tokens: usize,
    pub root_count: usize,
    pub cohort_width: usize,
    pub weight_storage: CudaForwardWeightStorage,
    pub source_weight_bytes: usize,
    pub resident_weight_bytes: usize,
    pub value_bytes: usize,
    pub workspace_bytes: usize,
    pub parameter_bytes: usize,
    /// Full contiguous bytes of one logical root; actual pool reserve below.
    pub root_bytes: usize,
    pub state_reserve_bytes: usize,
    pub runtime_reserve_bytes: usize,
    pub model_admission_bytes: usize,
    pub total_admission_bytes: usize,
}

/// Shared immutable source and serialized physical workspace, independent roots.
pub struct ResidentModel {
    service: Arc<ResidentService>,
    // Pool/raw handles retire before the prepared program/native admission.
    state_pool: std::sync::OnceLock<state_pool::PooledStates>,
    workspace: Mutex<Workspace>,
    allocation: Mutex<()>,
    states: Vec<StateRequirement>,
    state_bytes: usize,
    capacity: usize,
    vocab: u64,
    native_batch: bool,
    #[cfg(feature = "hardware-abort-probe")]
    abort_after_launch: std::sync::atomic::AtomicUsize,
    #[cfg(feature = "hardware-abort-probe")]
    launch_seen: std::sync::atomic::AtomicUsize,
}
#[derive(Clone)]
struct Domain {
    slot: StateSlot,
    buffers: StateBuffers,
    paged_kv: Option<paged_state::ResidentPagedKvDomain>,
    leases: Vec<ResidentLease>,
    generations: Vec<u64>,
}
impl Domain {
    #[allow(
        dead_code,
        reason = "canonical paged KV Domain construction is selected by the next forward-session slice"
    )]
    fn from_paged_kv(paged_kv: paged_state::ResidentPagedKvDomain) -> Result<Self> {
        let slot = paged_kv.slot();
        let leases = paged_kv.leases()?;
        let generations = leases
            .iter()
            .map(ResidentLease::allocation_generation)
            .collect();
        Ok(Self {
            slot,
            buffers: StateBuffers::PagedKv,
            paged_kv: Some(paged_kv),
            leases,
            generations,
        })
    }

    fn paged_kv(&self) -> Result<&paged_state::ResidentPagedKvDomain> {
        self.paged_kv
            .as_ref()
            .filter(|paged| {
                self.slot == paged.slot() && matches!(self.buffers, StateBuffers::PagedKv)
            })
            .ok_or_else(|| Error::State("resident domain is not canonical paged KV".into()))
    }
}
#[derive(Clone)]
pub struct ResidentSnapshot {
    binding: RootBinding,
    service_generation: u64,
    capacity: usize,
    domains: Vec<Domain>,
}
pub struct ResidentSession {
    authority: SessionAuthority,
    domains: Vec<Domain>,
    derived: DerivedStateCache,
    // Last model/native owner must outlive this session's actual root handles.
    model: Arc<ResidentModel>,
}
#[derive(Default)]
pub struct InferenceReadback {
    pub logits: Vec<f32>,
    pub domains: Vec<(StateSlot, Vec<f32>)>,
    pub observations: Vec<LayerObservation>,
    pub dispatches: usize,
    pub bytes: usize,
    pub chunk_bytes: usize,
    pub state_only_steps: usize,
    pub logit_steps: usize,
    pub device_token: Option<ResidentToken>,
}
#[derive(Clone)]
pub struct ResidentToken {
    lease: ResidentLease,
    producer: Option<RootBinding>,
    pub id: u32,
    pub max_bits: u32,
}
#[derive(Clone, Copy)]
pub enum Readout {
    None,
    Logits,
    Greedy,
}
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    target_os = "linux"
))]
struct SourceSteps<'a, 'o> {
    _observer_lifetime: std::marker::PhantomData<&'o ()>,
    #[cfg(feature = "gemma-bf16-qualification")]
    bf16_text: Option<bf16_text::TextStep<'a, 'o>>,
    #[cfg(feature = "qwen-source-qualification")]
    bf16_text_source: Option<bf16_text_source::TextSourceStep<'a, 'o>>,
}
struct StepInput<'a> {
    token: u32,
    device: Option<&'a ResidentToken>,
    readout: Readout,
    diagnostic: bool,
}
pub struct PreparedInference {
    attempt: RootAttempt,
    domains: Vec<Domain>,
    derived: Vec<DerivedStateUse>,
    ready: ResidentReady,
    readback: InferenceReadback,
}

/// Source-identical private candidates under one logical publication barrier.
/// Serial and native-batch preparation retain distinct launch receipts.
/// A failed preparation/changed root cannot publish any sibling row.
#[must_use = "dropping a resident cohort discards every uncommitted root"]
pub struct PreparedResidentCohort {
    model: Arc<ResidentModel>,
    roots: ResidentBatchRows,
    prepared: Vec<PreparedInference>,
    dispatches: usize,
    fused_dispatches: usize,
    native_greedy_dispatches: usize,
}
impl PreparedResidentCohort {
    /// Actual physical launches for the complete cohort, counted once.
    pub fn dispatches(&self) -> usize {
        self.dispatches
    }
    /// Native projection-dot launches; excludes activation quantization,
    /// parameter staging and certification. `dispatches` includes those stages.
    pub fn fused_dispatches(&self) -> usize {
        self.fused_dispatches
    }
    /// Actual native multi-row sampler stages; excludes private packet copies.
    pub fn native_greedy_dispatches(&self) -> usize {
        self.native_greedy_dispatches
    }
    pub fn len(&self) -> usize {
        self.prepared.len()
    }
    pub fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }
    pub fn source(&self) -> zerogravity_core::accelerator::SourceIdentity {
        self.roots.source()
    }
    /// Common prior frontier, when every row has the same position.
    pub fn prior_frontier(&self) -> Option<u64> {
        self.roots.frontier()
    }
    pub fn prior_frontiers(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.roots.frontiers()
    }

    /// After every per-row physical completion and cohort preflight,
    /// acquire ONE original device fence/publication barrier. All
    /// RootTransitions are reserved BEFORE any state buffer swap.
    /// No new model/session authority, invented Ready or shadow token loop.
    pub fn commit(
        mut self,
        sessions: &mut [&mut ResidentSession],
    ) -> Result<Vec<InferenceReadback>> {
        publish_prepared_cohort(&self.model, &self.roots, sessions, &mut self.prepared)?;
        Ok(self.prepared.into_iter().map(|row| row.readback).collect())
    }
}

// Codec-independent publication only. PreparedInference still carries the
// genuine original Ready/state/attempt; no batch math or allocation authority.
fn publish_prepared_cohort(
    model: &Arc<ResidentModel>,
    roots: &ResidentBatchRows,
    sessions: &mut [&mut ResidentSession],
    prepared: &mut [PreparedInference],
) -> Result<()> {
    if sessions.len() != prepared.len() {
        return Err(Error::Malformed("resident GPU cohort session count".into()));
    }
    let bindings = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
    roots.require_current(&bindings)?;
    // Refuse foreign resident models even if a source identity was
    // erroneously reused by some future model binder.
    for (session, prepared) in sessions.iter().zip(prepared.iter()) {
        if !Arc::ptr_eq(&session.model, model) {
            return Err(Error::State(
                "foreign physical resident model in batch".into(),
            ));
        }
        session.validate_prepared(prepared)?;
    }
    let service = Arc::clone(&model.service);
    let published = service.publish_if_usable(|| {
        // Split these disjoint struct fields so all mutable logical
        // authority guards can exist simultaneously with state slabs.
        let (mut authorities, mut states): (Vec<_>, Vec<_>) = sessions
            .iter_mut()
            .map(|s| (&mut s.authority, &mut s.domains))
            .unzip();
        let transitions = authorities
            .iter_mut()
            .zip(prepared.iter())
            .map(|(authority, prepared)| authority.prepare_publish(prepared.attempt.clone()))
            .collect::<Result<Vec<_>>>()?;
        // Everything above is fallible; everything below is infallible.
        // Existing ResourceArbiter's with_unfenced keeps the atomic
        // logical cohort publication fenced with respect to GPU poison.
        let mut published = Vec::with_capacity(prepared.len());
        for ((state, prepared), guard) in
            states.iter_mut().zip(prepared.iter_mut()).zip(transitions)
        {
            std::mem::swap(*state, &mut prepared.domains);
            let root = guard.commit();
            if let Some(token) = prepared.readback.device_token.as_mut() {
                token.producer = Some(root);
            }
            published.push(root);
        }
        Ok(published)
    })?;
    // Derived representations are optimization-only. Canonical publication is
    // already complete, so a failed cache retag can only turn into a miss.
    for (prepared, root) in prepared.iter_mut().zip(published) {
        for derived in &mut prepared.derived {
            let _ = derived.retag_after_publish(root);
        }
    }
    Ok(())
}

fn publish_domain_swap(
    service: &Arc<ResidentService>,
    authority: &mut SessionAuthority,
    current: &mut Vec<Domain>,
    attempt: RootAttempt,
    candidate: &mut Vec<Domain>,
) -> Result<RootBinding> {
    service.publish_if_usable(|| {
        let transition = authority.prepare_publish(attempt)?;
        std::mem::swap(current, candidate);
        Ok(transition.commit())
    })
}

fn canonical_write_outputs(current: &[Domain], candidate: &[Domain]) -> Result<Vec<ResidentLease>> {
    if current.len() != candidate.len() {
        return Err(Error::State(
            "resident prepared/current domain coverage differs".into(),
        ));
    }
    let mut outputs = Vec::new();
    for (current, candidate) in current.iter().zip(candidate) {
        match (
            matches!(current.buffers, StateBuffers::PagedKv),
            matches!(candidate.buffers, StateBuffers::PagedKv),
        ) {
            (false, false) => outputs.extend(candidate.leases.iter().cloned()),
            (true, true) => outputs.extend(
                candidate
                    .paged_kv()?
                    .required_write_leases_from(current.paged_kv()?)?,
            ),
            _ => {
                return Err(Error::State(
                    "resident prepared state representation changed before publication".into(),
                ));
            }
        }
    }
    Ok(outputs)
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
fn forward_cancel_requested(cancelled: Option<&std::sync::atomic::AtomicBool>) -> bool {
    cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
fn forward_not_cancelled(cancelled: Option<&std::sync::atomic::AtomicBool>) -> Result<()> {
    if forward_cancel_requested(cancelled) {
        return Err(Error::State("CUDA forward request cancelled".into()));
    }
    Ok(())
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
async fn retire_forward_cancelled(
    flight: crate::resident_leases::ResidentSubmission<'_>,
    accesses: &[(ResidentLease, DeviceAccess)],
    status: &Handle,
) -> Result<()> {
    let status = lease_for(accesses, status)?.readback(0, 4)?;
    let read = flight.retire_cancelled_with_readback(&[status]).await?;
    if read.len() != 1 || !all_zero(&read[0]) {
        return Err(Error::State(
            "cancelled forward source work reported a nonzero status".into(),
        ));
    }
    Ok(())
}

fn into_unique_owner<T>(owner: Arc<T>, label: &str) -> Result<T> {
    Arc::try_unwrap(owner)
        .map_err(|_| Error::State(format!("{label} still has live session/model owners")))
}

fn words(bytes: &Bytes) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::State("resident f32 readback alignment".into()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_ne_bytes(b.try_into().unwrap())))
        .collect())
}
fn resident_plane_finite(bytes: &Bytes, layout: ResidentBufferLayout) -> Result<bool> {
    if bytes.len() != layout.bytes()? {
        return Err(Error::State(
            "resident typed snapshot physical byte extent differs".into(),
        ));
    }
    Ok(match layout.kind() {
        ResidentPhysicalType::F32 => bytes
            .chunks_exact(4)
            .all(|word| f32::from_bits(u32::from_ne_bytes(word.try_into().unwrap())).is_finite()),
        ResidentPhysicalType::Bf16 => bytes
            .chunks_exact(2)
            .all(|word| (u16::from_ne_bytes(word.try_into().unwrap()) & 0x7f80) != 0x7f80),
    })
}

fn resident_plane_values(
    bytes: &Bytes,
    layout: ResidentBufferLayout,
    elements: usize,
) -> Result<Vec<f32>> {
    if elements > layout.elements() || bytes.len() != layout.bytes()? {
        return Err(Error::State(
            "resident typed snapshot active element extent differs".into(),
        ));
    }
    let end = elements
        .checked_mul(layout.kind().byte_width())
        .ok_or_else(|| Error::Limit("resident typed snapshot active bytes".into()))?;
    let raw = &bytes[..end];
    Ok(match layout.kind() {
        ResidentPhysicalType::F32 => raw
            .chunks_exact(4)
            .map(|word| f32::from_bits(u32::from_ne_bytes(word.try_into().unwrap())))
            .collect(),
        ResidentPhysicalType::Bf16 => raw
            .chunks_exact(2)
            .map(|word| {
                f32::from_bits(u32::from(u16::from_ne_bytes(word.try_into().unwrap())) << 16)
            })
            .collect(),
    })
}

fn resident_plane_tail_zero(
    bytes: &Bytes,
    layout: ResidentBufferLayout,
    active_elements: usize,
) -> Result<bool> {
    if active_elements > layout.elements() || bytes.len() != layout.bytes()? {
        return Err(Error::State(
            "resident typed snapshot tail extent differs".into(),
        ));
    }
    let start = active_elements
        .checked_mul(layout.kind().byte_width())
        .ok_or_else(|| Error::Limit("resident typed snapshot tail byte offset".into()))?;
    Ok(bytes[start..].iter().all(|byte| *byte == 0))
}

fn all_zero(bytes: &Bytes) -> bool {
    bytes.iter().all(|b| *b == 0)
}
fn same_handle(a: &Handle, b: &Handle) -> bool {
    let a = a.clone().binding();
    let b = b.clone().binding();
    a.service == b.service && a.memory.id() == b.memory.id() && a.range() == b.range()
}
fn lease_for<'a>(
    leases: &'a [(ResidentLease, DeviceAccess)],
    handle: &Handle,
) -> Result<&'a ResidentLease> {
    leases
        .iter()
        .find(|(l, _)| same_handle(&l.handle(), handle))
        .map(|(l, _)| l)
        .ok_or_else(|| Error::State("unadmitted resident graph handle".into()))
}

impl zerogravity_core::provider::PreparedExecutable for ResidentModel {
    type Source = Arc<dyn TensorSource>;
    async fn source(&self) -> Result<Arc<dyn TensorSource>> {
        self.workspace.lock().await.program.owned_source()
    }
    async fn program(&self) -> Result<Arc<ForwardPlan>> {
        Ok(self.workspace.lock().await.program.shared_plan())
    }
}

#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
fn cuda_forward_source_admission(
    plan: Arc<ForwardPlan>,
    provider_cap: usize,
    roots: usize,
    cohort_width: usize,
    state_reserve_bytes: Option<usize>,
) -> Result<(
    cuda_forward::CudaForwardAdmission,
    CudaForwardSourceInventory,
)> {
    let admission = cuda_forward::CudaForwardAdmission::analyze(plan)?;
    let admission = match state_reserve_bytes {
        Some(bytes) => admission.with_state_reserve(roots, bytes)?,
        None => admission.with_roots(roots)?,
    };
    let source_weight_bytes = admission.preflight().source_bytes;
    let greedy_workspace = crate::resident_greedy::workspace_bytes(admission.vocabulary())?
        .into_iter()
        .try_fold(0usize, |sum, bytes| {
            sum.checked_add(bytes)
                .ok_or_else(|| Error::Limit("CUDA forward greedy workspace overflow".into()))
        })?;
    let cohort_value_reserve = admission.cohort_value_bytes(cohort_width)?;
    let greedy_reserve = greedy_workspace
        .checked_add(
            roots
                .checked_mul(12)
                .ok_or_else(|| Error::Limit("CUDA forward greedy packet reserve".into()))?,
        )
        .ok_or_else(|| Error::Limit("CUDA forward greedy reserve overflow".into()))?;
    let runtime_reserve = greedy_reserve
        .checked_add(cohort_value_reserve)
        .ok_or_else(|| Error::Limit("CUDA forward runtime reserve overflow".into()))?;
    let fit_cap = provider_cap.checked_sub(runtime_reserve).ok_or_else(|| {
        Error::resource(
            "CUDA provider memory / VRAM budget",
            "CUDA forward runtime reserve admission",
            runtime_reserve as u64,
            provider_cap as u64,
        )
    })?;
    let admission = admission.fit(fit_cap).map_err(|error| match error {
        // The payload fit excludes the already-priced runtime reserve.
        // Compare/report all candidate capacity misses in the whole-model envelope.
        Error::Resource(miss) => Error::resource(
            "CUDA provider memory / VRAM budget",
            "CUDA forward cold/root/runtime admission",
            miss.requested.saturating_add(runtime_reserve as u64),
            provider_cap as u64,
        ),
        other => other,
    })?;
    let model_admission_bytes = admission.total_bytes()?;
    let total = model_admission_bytes
        .checked_add(runtime_reserve)
        .ok_or_else(|| Error::Limit("CUDA forward total admission overflow".into()))?;
    if total > provider_cap {
        return Err(Error::resource(
            "CUDA provider memory / VRAM budget",
            "CUDA forward cold/root/runtime admission",
            total as u64,
            provider_cap as u64,
        ));
    }
    let preflight = admission.preflight();
    let resident_weight_bytes = preflight.source_bytes;
    if resident_weight_bytes != 0 && resident_weight_bytes != source_weight_bytes {
        return Err(Error::State(
            "CUDA forward fitted source residency is neither full nor streamed".into(),
        ));
    }
    let inventory = CudaForwardSourceInventory {
        provider_capacity_bytes: provider_cap,
        fit_capacity_bytes: fit_cap,
        context_capacity_tokens: admission.capacity(),
        root_count: roots,
        cohort_width,
        weight_storage: if resident_weight_bytes == 0 {
            CudaForwardWeightStorage::Streamed
        } else {
            CudaForwardWeightStorage::Resident
        },
        source_weight_bytes,
        resident_weight_bytes,
        value_bytes: preflight.value_bytes,
        workspace_bytes: preflight.workspace_bytes,
        parameter_bytes: preflight.parameter_bytes,
        root_bytes: preflight.prior_state_bytes,
        state_reserve_bytes: admission.state_reserve_bytes(),
        runtime_reserve_bytes: runtime_reserve,
        model_admission_bytes,
        total_admission_bytes: total,
    };
    Ok((admission, inventory))
}

#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
fn observe_cuda_source_device(
    device: &Device,
    provider_cap: usize,
    device_cap: u64,
) -> Result<PhysicalDeviceKey> {
    let Device::Cuda(selected) = device else {
        return Err(Error::Unsupported(
            "CUDA forward resident model requires CUDA".into(),
        ));
    };
    crate::physical_service::CudaCompilerContextBudget::require_device_envelope(
        provider_cap as u64,
        device_cap,
    )?;
    let devices = cubecl::cuda::observe_cuda_devices()
        .map_err(|error| Error::Unsupported(format!("CUDA source device observation: {error}")))?;
    let actual = devices
        .iter()
        .find(|view| view.ordinal == selected.index)
        .ok_or_else(|| Error::Unsupported("selected CUDA source device unavailable".into()))?;
    if actual.compute_capability != (8, 6) {
        return Err(Error::Unsupported(
            "CUDA source requires qualified SM86 device".into(),
        ));
    }
    let pci = |view: &cubecl::cuda::CudaDeviceMetadata| -> Result<cubecl::ir::PciAddress> {
        view.pci_bus_id
            .parse()
            .map_err(|error| Error::State(format!("CUDA source physical PCI identity: {error}")))
    };
    let key = pci(actual)?;
    for other in &devices {
        if other.ordinal != actual.ordinal && pci(other)? == key {
            return Err(Error::Unsupported(
                "CUDA source physical PCI has ambiguous device views".into(),
            ));
        }
    }
    crate::physical_service::checked_caps(
        Some(actual.total_bytes),
        device_cap,
        provider_cap as u64,
    )?;
    PhysicalDeviceKey::new("pci", &key.to_string())
}

impl ResidentModel {
    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub fn cuda_forward_source_preflight(
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        cohort_width: usize,
        provider_cap: usize,
    ) -> Result<CudaForwardSourceInventory> {
        if !(1..=64).contains(&max_sessions)
            || !matches!(cohort_width, 0 | 4 | 8)
            || (cohort_width != 0 && max_sessions < cohort_width)
        {
            return Err(Error::Limit("resident-text session capacity".into()));
        }
        let roots = max_sessions
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| Error::Limit("resident-text root count overflow".into()))?;
        cuda_forward_source_admission(plan, provider_cap, roots, cohort_width, None)
            .map(|(_, inventory)| inventory)
    }

    pub async fn load(
        device: Device,
        model: Arc<dyn TensorSource>,
        plan: impl Into<Arc<ForwardPlan>>,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<Arc<Self>> {
        Self::load_inner(device, model, plan, provider_cap, device_cap, false).await
    }

    async fn load_inner(
        device: Device,
        model: Arc<dyn TensorSource>,
        plan: impl Into<Arc<ForwardPlan>>,
        provider_cap: usize,
        device_cap: u64,
        pinned_graph: bool,
    ) -> Result<Arc<Self>> {
        Self::load_attached(
            device,
            model,
            plan,
            provider_cap,
            device_cap,
            pinned_graph,
            None,
        )
        .await
    }

    /// Load into the original physical envelope with an explicit model quota.
    pub async fn load_shared(
        physical: Arc<ResidentPhysical>,
        model: Arc<dyn TensorSource>,
        plan: ForwardPlan,
        model_cap: usize,
    ) -> Result<Arc<Self>> {
        let device = physical.device();
        Self::load_attached(device, model, plan, model_cap, 0, false, Some(physical)).await
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    #[allow(clippy::too_many_arguments)]
    pub async fn load_cuda_forward(
        device: Device,
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        provider_cap: usize,
        device_cap: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::load_cuda_forward_roots(
            device,
            source,
            plan,
            provider_cap,
            device_cap,
            2,
            0,
            None,
            None,
            cancelled,
        )
        .await
        .map(|(model, _)| model)
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    #[allow(clippy::too_many_arguments)]
    async fn load_cuda_forward_roots(
        device: Device,
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        provider_cap: usize,
        device_cap: u64,
        roots: usize,
        cohort_width: usize,
        state_reserve_bytes: Option<usize>,
        source_owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(Arc<Self>, CudaForwardSourceInventory)> {
        validate_provider(&device)?;
        let observed_key = observe_cuda_source_device(&device, provider_cap, device_cap)?;

        // Validate the configured source/plan envelope before touching physical
        // CUDA. A live shared-card observation below may only LOWER this cap;
        // when it does, rerun this exact same placement helper rather than
        // introducing a second estimator.
        let (configured_admission, configured_inventory) = cuda_forward_source_admission(
            plan.clone(),
            provider_cap,
            roots,
            cohort_width,
            state_reserve_bytes,
        )?;
        let source_identity = configured_admission.source();
        if source.source_identity() != source_identity {
            return Err(Error::State(
                "CUDA forward verified source differs from plan".into(),
            ));
        }

        let physical = ResidentPhysical::new_pinned_cuda(device, provider_cap, device_cap)?;
        if physical.physical_key() != observed_key {
            return Err(Error::State(
                "CUDA source actor physical identity differs from selected device".into(),
            ));
        }
        let effective_cap = physical.available_model_capacity()?;
        let (admission, inventory) = if effective_cap == provider_cap {
            (configured_admission, configured_inventory)
        } else {
            cuda_forward_source_admission(
                plan,
                effective_cap,
                roots,
                cohort_width,
                state_reserve_bytes,
            )?
        };
        let capacity = admission.capacity();
        let vocab = admission.vocabulary() as u64;
        let expected_preflight = admission.preflight().clone();
        let owner = ResourceOwner {
            model: format!("forward-qwen-source-{source_identity:?}"),
            session: "resident-program".into(),
        };
        let service = physical.attach_model(
            ResidentSource::Windows(source.clone()),
            owner,
            effective_cap,
        )?;
        let prepared = cuda_forward::prepare_program(
            service.clone(),
            admission,
            source,
            source_owner,
            cohort_width,
            cancelled,
        )
        .await?;
        let program = prepared.program;
        if program.source_identity() != source_identity
            || program.plan().source() != source_identity
            || program.plan().prefix_capacity() != capacity
            || program.budget().owned_bytes()?
                != prepared
                    .owned_accesses
                    .iter()
                    .try_fold(0usize, |sum, (lease, _)| {
                        sum.checked_add(lease.bytes()).ok_or_else(|| {
                            Error::Limit("CUDA forward owned access byte sum".into())
                        })
                    })?
            || program.budget().prior_state_bytes != expected_preflight.prior_state_bytes
            || program.budget().candidate_state_bytes != expected_preflight.candidate_state_bytes
        {
            return Err(Error::State(
                "CUDA forward bound resident model/preflight differs".into(),
            ));
        }
        let states = program.budget().states.clone();
        let state_bytes = program.budget().prior_state_bytes;
        let greedy = prepared.greedy;
        let forward_cohort_values = prepared.cohort_values;
        let mut leases = prepared.owned_accesses;
        leases.extend(prepared.cohort_value_accesses);
        leases.extend(prepared.greedy_accesses);

        Ok((
            Arc::new(Self {
                service,
                states,
                state_bytes,
                capacity,
                vocab,
                native_batch: false,
                state_pool: std::sync::OnceLock::new(),
                #[cfg(feature = "hardware-abort-probe")]
                abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
                #[cfg(feature = "hardware-abort-probe")]
                launch_seen: std::sync::atomic::AtomicUsize::new(0),
                workspace: Mutex::new(Workspace {
                    program,
                    greedy: Some(greedy),
                    forward_cohort_values,
                    leases,
                    state_accesses: prepared.state_accesses,
                    logit_accesses: prepared.logit_accesses,
                    batch: None,
                }),
                allocation: Mutex::new(()),
            }),
            inventory,
        ))
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn shutdown_cuda_forward(self: Arc<Self>) -> Result<()> {
        let model = into_unique_owner(self, "CUDA forward resident model")?;
        let service = model.service.clone();
        {
            let workspace = model.workspace.lock().await;
            if !workspace.program.is_cuda_forward() {
                return Err(Error::State(
                    "CUDA forward shutdown requested from another resident model".into(),
                ));
            }
            workspace.program.retire_cuda_forward().await?;
        }

        // No sessions can remain after the unique-owner check. Dropping the
        // model retires all cold/value/state-pool leases while this service Arc
        // keeps the original namespace available for exact ledger validation.
        drop(model);
        service
            .client()
            .memory_cleanup()
            .map_err(|error| Error::State(format!("CUDA forward shutdown cleanup: {error}")))?;
        if service.resident_bytes()? != 0 || service.model_events()? != 0 {
            return Err(Error::State(
                "CUDA forward shutdown left resident bytes or events".into(),
            ));
        }
        service.close_model_namespace()
    }

    /// Explicit retirement of an ordinary borrowed-tensor executable.
    /// No session can enter after exclusive ownership is acquired. The
    /// existing native source adapters retain their separate module teardown.
    pub async fn shutdown_tensor_source(self: Arc<Self>) -> Result<()> {
        let model = into_unique_owner(self, "resident tensor-source model")?;
        model.workspace.lock().await.program.owned_source()?;
        let service = model.service.clone();
        service.ensure_usable()?;
        #[cfg(all(
            feature = "cuda",
            any(feature = "qualification", feature = "cuda-native")
        ))]
        model.retire_production_graph_cached().await?;
        // The shared actor cleanup can wait on its streams. Serialize it
        // against other original models' commands using the existing queue,
        // after any graph-specific retirement (which owns its own slots).
        let bytes = service.resident_bytes()?;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        service.ensure_usable()?;
        slot.begin_submission()?;
        // Drop original handles while the service still owns the namespace.
        drop(model);
        service.client().memory_cleanup().map_err(|error| {
            Error::State(format!("resident tensor-source shutdown cleanup: {error}"))
        })?;
        // Known actor cleanup is complete. Outstanding external readers
        // still refuse namespace close without inventing unknown work.
        slot.finish()?;
        service.close_model_namespace()
    }

    /// Cold description of the actual ordinary tensor-source program. No
    /// graph/template capability is inferred from the backing device.
    pub async fn tensor_source_description(
        &self,
    ) -> Result<zerogravity_core::provider::ExecutionDescription> {
        use zerogravity_core::provider::{ExecutionDescription, ExecutionMode, SelectedMode};
        let workspace = self.workspace.lock().await;
        workspace.program.owned_source()?;
        let provider = crate::backend::Provider::selected(&self.service.physical_owner().device())?;
        Ok(ExecutionDescription {
            selected: SelectedMode {
                provider: provider.kind(),
                mode: ExecutionMode::Native,
            },
            numerical_profile: Some(workspace.program.plan().numerical_profile()),
            text_calls: None,
        })
    }
    /// Read the existing model quota/leases, with no new allocation or actor IO.
    pub fn memory_snapshot(&self) -> Result<zerogravity_core::provider::ResidentModelMemory> {
        let service = &self.service;
        Ok(zerogravity_core::provider::ResidentModelMemory {
            device: service.physical_key(),
            model_capacity_bytes: service.budget().provider_capacity(),
            physical_limit_bytes: service.budget().device_capacity(),
            resident_bytes: service.resident_bytes()?,
            root_bytes: self.state_bytes,
        })
    }

    pub fn physical_owner(&self) -> Arc<ResidentPhysical> {
        self.service.physical_owner().clone()
    }

    #[allow(clippy::too_many_arguments)]
    async fn load_attached(
        device: Device,
        model: Arc<dyn TensorSource>,
        plan: impl Into<Arc<ForwardPlan>>,
        provider_cap: usize,
        device_cap: u64,
        pinned_graph: bool,
        physical: Option<Arc<ResidentPhysical>>,
    ) -> Result<Arc<Self>> {
        validate_provider(&device)?;
        let native_batch = matches!(device, Device::Cuda(_));
        let plan = plan.into();
        let (capacity, vocab, preflight) = preflight_profile(model.as_ref(), &plan)?;
        if preflight.source != model.source_identity() {
            return Err(Error::State("resident preflight source mismatch".into()));
        }
        let greedy_sizes = crate::resident_greedy::workspace_bytes(vocab as usize)?;
        let greedy_bytes = greedy_sizes.iter().sum::<usize>();
        // Require source/workspace and two private states before any GPU allocation.
        let graph_bytes = preflight
            .total_with_state()?
            .checked_add(greedy_bytes + 12)
            .ok_or_else(|| Error::Limit("complete resident graph admission overflow".into()))?;
        if graph_bytes > provider_cap {
            return Err(Error::resource(
                "GPU provider memory / VRAM budget",
                "complete resident graph admission",
                graph_bytes as u64,
                provider_cap as u64,
            ));
        }
        if !pinned_graph {
            ResidentProgram::ensure_capability(&device.client())?;
        }
        let owner = ResourceOwner {
            model: format!("qwen-source-{:?}", model.source_identity()),
            session: "resident-program".into(),
        };
        let physical = if let Some(physical) = physical {
            physical
        } else if pinned_graph {
            #[cfg(all(feature = "qualification", feature = "cuda"))]
            {
                ResidentPhysical::new_pinned_cuda(device, provider_cap, device_cap)?
            }
            #[cfg(not(all(feature = "qualification", feature = "cuda")))]
            {
                return Err(Error::Unsupported(
                    "graph qualifier requires CUDA/qualification".into(),
                ));
            }
        } else {
            ResidentPhysical::new(device, provider_cap, device_cap)?
        };
        let service =
            physical.attach_model(ResidentSource::Tensor(model.clone()), owner, provider_cap)?;
        if pinned_graph {
            ResidentProgram::ensure_capability(service.client())?;
        }
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: preflight.owned_bytes()? as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut initialization = service.begin_initialization(0)?;
        // Cold model admission may follow retired models on other streams.
        // Drain completed deferred frees before allocating this envelope.
        // Live/graph-pinned handles remain owned; errors use the same guard.
        #[cfg(feature = "qualification")]
        crate::gpu_telemetry::allocator_report(service.client(), "before_cold_cleanup");
        service
            .client()
            .memory_cleanup()
            .map_err(|e| Error::State(format!("resident cold allocator cleanup failed: {e}")))?;
        #[cfg(feature = "qualification")]
        crate::gpu_telemetry::allocator_report(service.client(), "after_cold_cleanup");
        let mut leases = Vec::new();
        let program = ResidentProgram::prepare_with_allocator(
            service.client(),
            model,
            plan,
            |role, bytes, data| {
                let (role, access) = match role {
                    ResidentBufferRole::Source => {
                        (BufferRole::ImmutableWeights, DeviceAccess::Read)
                    }
                    ResidentBufferRole::Value => (BufferRole::Activation, DeviceAccess::Write),
                    ResidentBufferRole::Workspace | ResidentBufferRole::Status => {
                        (BufferRole::Scratch, DeviceAccess::Write)
                    }
                    ResidentBufferRole::Parameter => (BufferRole::Activation, DeviceAccess::Write),
                };
                let lease = match data {
                    AllocationInit::Empty => initialization.allocate_empty(bytes, role)?,
                    AllocationInit::Upload(data) => {
                        if data.len() != bytes {
                            return Err(Error::State("resident initializer byte mismatch".into()));
                        }
                        initialization.upload_bytes(data, role)?
                    }
                };
                let handle = lease.handle();
                leases.push((lease, access));
                Ok(handle)
            },
        )?;
        for handle in program.owned_handles() {
            lease_for(&leases, &handle)?;
        }
        let mut greedy_handles = Vec::new();
        for bytes in greedy_sizes {
            let lease = initialization.allocate_empty(bytes, BufferRole::Scratch)?;
            greedy_handles.push(lease.handle());
            leases.push((lease, DeviceAccess::Write));
        }
        let greedy: [Handle; 2] = greedy_handles
            .try_into()
            .map_err(|_| Error::State("greedy workspace coverage".into()))?;
        if program.source_identity() != service.source_identity()
            || leases.iter().map(|(l, _)| l.bytes()).sum::<usize>()
                != program.budget().owned_bytes()? + greedy_bytes
        {
            return Err(Error::State(
                "resident exact allocation preflight mismatch".into(),
            ));
        }
        let select = |output| {
            let handles = program.execution_handles(output);
            leases
                .iter()
                .filter(|(lease, _)| handles.iter().any(|h| same_handle(h, &lease.handle())))
                .cloned()
                .collect()
        };
        let state_accesses = select(ResidentOutput::StateOnly);
        let logit_accesses = select(ResidentOutput::Logits);
        let ready = initialization.complete_sync().await?;
        if ready.outputs().len() != leases.len() {
            return Err(Error::State("incomplete source preparation".into()));
        }
        slot.finish()?;
        Ok(Arc::new(Self {
            service,
            states: preflight.states,
            state_bytes: preflight.prior_state_bytes,
            capacity,
            vocab,
            native_batch,
            state_pool: std::sync::OnceLock::new(),
            #[cfg(feature = "hardware-abort-probe")]
            abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "hardware-abort-probe")]
            launch_seen: std::sync::atomic::AtomicUsize::new(0),
            workspace: Mutex::new(Workspace {
                program,
                greedy: Some(greedy),
                #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
                forward_cohort_values: None,
                leases,
                state_accesses,
                logit_accesses,
                batch: None,
            }),
            allocation: Mutex::new(()),
        }))
    }
    pub async fn session(self: &Arc<Self>) -> Result<ResidentSession> {
        self.session_with(|_| Ok(())).await
    }
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    async fn session_retaining(
        self: &Arc<Self>,
        owner: crate::physical_quarantine::CompletionOwner,
    ) -> Result<ResidentSession> {
        self.session_with(|initialization| initialization.retain_completion_owner(owner))
            .await
    }
    async fn session_with<F>(self: &Arc<Self>, retain: F) -> Result<ResidentSession>
    where
        F: FnOnce(&mut crate::resident_leases::ResidentInitialization<'_>) -> Result<()>,
    {
        let authority = SessionAuthority::new(self.service.source_identity())?;
        let domains = self.allocate_state_with(true, retain).await?;
        Ok(ResidentSession::new(self.clone(), authority, domains))
    }
    /// Prepare a set of SAME-source, SAME-frontier sessions. One immutable
    /// original ForwardPlan, weight owner and queue are reused; private
    /// per-row state never aliases. Each GPU proposal is completed before
    /// returning; this stage does NOT fuse their physical operator graphs.
    /// Caller can drop the cohort for an all-row logical rollback.
    pub async fn prepare_greedy_cohort(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
    ) -> Result<PreparedResidentCohort> {
        self.prepare_cohort_mode(rows, Readout::Greedy).await
    }

    /// Diagnostic full-vocabulary cohort of the same original physical
    /// execution. This uses the same source, session and commit barrier as
    /// greedy serving, with explicit GPU logits only for qualification.
    pub async fn prepare_logits_cohort(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
    ) -> Result<PreparedResidentCohort> {
        self.prepare_cohort_mode(rows, Readout::Logits).await
    }

    async fn prepare_cohort_mode(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        readout: Readout,
    ) -> Result<PreparedResidentCohort> {
        let bindings = rows
            .iter()
            .map(|(session, _)| session.binding())
            .collect::<Vec<_>>();
        let roots = ResidentBatchRows::new(&bindings, 1)?;
        if roots.source() != self.service.source_identity() {
            return Err(Error::State("GPU cohort model source changed".into()));
        }
        for (session, token) in rows {
            if !Arc::ptr_eq(&session.model, self)
                || u64::from(*token) >= self.vocab
                || session.binding().frontier() >= self.capacity as u64
            {
                return Err(Error::State(
                    "foreign or exhausted resident GPU batch row".into(),
                ));
            }
        }
        let mut prepared = Vec::with_capacity(rows.len());
        for (session, token) in rows {
            prepared.push(session.prepare_mode(&[*token], false, readout).await?);
        }
        // Do not publish ANY row if the requested greedy feedback cohort
        // contains a missing/foreign physical output packet.
        if matches!(readout, Readout::Greedy)
            && prepared.iter().any(|p| p.readback.device_token.is_none())
        {
            return Err(Error::State("incomplete GPU cohort greedy packets".into()));
        }
        Ok(PreparedResidentCohort {
            model: Arc::clone(self),
            roots,
            dispatches: prepared.iter().map(|p| p.readback.dispatches).sum(),
            fused_dispatches: 0,
            native_greedy_dispatches: 0,
            prepared,
        })
    }

    async fn allocate_state(&self, zero: bool) -> Result<Vec<Domain>> {
        self.allocate_state_with(zero, |_| Ok(())).await
    }
    async fn allocate_state_with<F>(&self, zero: bool, retain: F) -> Result<Vec<Domain>>
    where
        F: FnOnce(&mut crate::resident_leases::ResidentInitialization<'_>) -> Result<()>,
    {
        if let Some(storage) = self.state_pool.get() {
            #[cfg(all(
                any(
                    feature = "qualification",
                    feature = "hip-source",
                    feature = "cuda-cublas"
                ),
                target_os = "linux"
            ))]
            let host = if zero {
                let bytes = self
                    .state_bytes
                    .checked_mul(3)
                    .ok_or_else(|| Error::Limit("resident root initialization host peak".into()))?;
                storage
                    .source
                    .as_ref()
                    .map(|source| source.reserve_backend_workspace(bytes).map(Arc::new))
                    .transpose()?
            } else {
                None
            };
            return match &storage.storage {
                state_pool::PooledStateStorage::Contiguous(pool) => {
                    state_pool::allocate_pooled_state(self, pool, zero, None, |init| {
                        #[cfg(all(
                            any(
                                feature = "qualification",
                                feature = "hip-source",
                                feature = "cuda-cublas"
                            ),
                            target_os = "linux"
                        ))]
                        if let Some(host) = host {
                            init.retain_completion_owner(host)?;
                        }
                        retain(init)
                    })
                    .await
                }
                state_pool::PooledStateStorage::Hybrid(pools) => {
                    if !zero {
                        return Err(Error::Unsupported(
                            "resident hybrid candidate allocation requires current paged root"
                                .into(),
                        ));
                    }
                    state_pool::allocate_hybrid_initial_state(self, pools, None, |init| {
                        #[cfg(all(
                            any(
                                feature = "qualification",
                                feature = "hip-source",
                                feature = "cuda-cublas"
                            ),
                            target_os = "linux"
                        ))]
                        if let Some(host) = host {
                            init.retain_completion_owner(host)?;
                        }
                        retain(init)
                    })
                    .await
                }
            };
        }
        let _admission = self.allocation.lock().await;
        self.service.ensure_usable()?;
        let available = self.service.remaining_capacity()?;
        if self.state_bytes > available {
            return Err(Error::resource(
                "GPU provider memory / VRAM budget",
                "resident state admission",
                self.state_bytes as u64,
                available as u64,
            ));
        }
        let mut output = Vec::new();
        if zero {
            let ticket = self.service.queue().submit(
                self.service.budget(),
                self.service.owner().clone(),
                &[AllocationRequest {
                    kind: AllocationClass::RecurrentState,
                    bytes: self.state_bytes as u64,
                }],
                None,
            )?;
            let mut slot = ticket.wait().await?;
            slot.begin_submission()?;
            let mut initialization = self.service.begin_initialization(0)?;
            retain(&mut initialization)?;
            for r in &self.states {
                let mut allocate = |layout: ResidentBufferLayout| {
                    initialization.upload_zeroed(layout.bytes()?, BufferRole::RecurrentState)
                };
                output.push(make_domain(r, &mut allocate)?);
            }
            drop(initialization.complete_sync().await?);
            slot.finish()?;
        } else {
            output = self.allocate_uninitialized_state()?;
        }
        Ok(output)
    }
    /// Caller holds the original allocation admission lock; no GPU command.
    fn allocate_uninitialized_state(&self) -> Result<Vec<Domain>> {
        if let Some(storage) = self.state_pool.get() {
            let state_pool::PooledStateStorage::Contiguous(pool) = &storage.storage else {
                return Err(Error::Unsupported(
                    "resident hybrid candidate allocation requires current paged root".into(),
                ));
            };
            let leases = pool.checkout()?;
            let mut next = leases.iter();
            let domains = self
                .states
                .iter()
                .map(|requirement| {
                    make_domain(requirement, &mut |layout| {
                        let lease = next
                            .next()
                            .ok_or_else(|| Error::State("resident root plane missing".into()))?;
                        if lease.bytes() != layout.bytes()?
                            || lease.is_initialized()
                            || lease.role() != BufferRole::RecurrentState
                            || !Arc::ptr_eq(lease.service(), &self.service)
                        {
                            return Err(Error::State(
                                "resident root original plane/source differs".into(),
                            ));
                        }
                        Ok(lease.clone())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if next.next().is_some() {
                return Err(Error::State("resident root extra plane".into()));
            }
            self.validate_domains(&domains, false)?;
            return Ok(domains);
        }
        self.service.ensure_usable()?;
        let available = self.service.remaining_capacity()?;
        if self.state_bytes > available {
            return Err(Error::resource(
                "GPU provider memory / VRAM budget",
                "resident state admission",
                self.state_bytes as u64,
                available as u64,
            ));
        }
        self.states
            .iter()
            .map(|r| {
                let mut allocate = |layout: ResidentBufferLayout| {
                    self.service
                        .allocate_uninitialized(layout.bytes()?, BufferRole::RecurrentState)
                };
                make_domain(r, &mut allocate)
            })
            .collect()
    }
    fn validate_domains(&self, domains: &[Domain], initialized: bool) -> Result<()> {
        if domains.len() != self.states.len() {
            return Err(Error::State("resident domain coverage".into()));
        }
        for (domain, requirement) in domains.iter().zip(&self.states) {
            if domain.slot != requirement.slot || domain.generations.len() != domain.leases.len() {
                return Err(Error::State(
                    "resident domain slot/generation coverage".into(),
                ));
            }
            if matches!(domain.buffers, StateBuffers::PagedKv) {
                let paged = domain.paged_kv()?;
                paged.require_requirement(requirement)?;
                let page_leases = paged.leases()?;
                if page_leases.len() != domain.leases.len()
                    || page_leases
                        .iter()
                        .zip(&domain.leases)
                        .any(|(actual, retained)| {
                            actual.id() != retained.id()
                                || actual.allocation_generation()
                                    != retained.allocation_generation()
                        })
                {
                    return Err(Error::State(
                        "resident paged KV lease coverage differs".into(),
                    ));
                }
                for (lease, expected) in domain.leases.iter().zip(&domain.generations) {
                    if !Arc::ptr_eq(lease.service(), &self.service)
                        || lease.service_generation() != self.service.generation()
                        || lease.source_identity() != self.service.source_identity()
                        || lease.allocation_generation() != *expected
                        || (initialized && !lease.is_initialized())
                    {
                        return Err(Error::State(
                            "resident paged KV allocation/service generation mismatch".into(),
                        ));
                    }
                }
                continue;
            } else if domain.paged_kv.is_some() {
                return Err(Error::State(
                    "resident contiguous domain retains paged KV authority".into(),
                ));
            }
            let handles = domain.buffers.handles();
            if handles.len() != domain.leases.len() {
                return Err(Error::State("resident physical domain coverage".into()));
            }
            for ((lease, expected), handle) in
                domain.leases.iter().zip(&domain.generations).zip(handles)
            {
                if !Arc::ptr_eq(lease.service(), &self.service)
                    || lease.service_generation() != self.service.generation()
                    || lease.source_identity() != self.service.source_identity()
                    || lease.allocation_generation() != *expected
                    || !same_handle(&lease.handle(), &handle)
                    || (initialized && !lease.is_initialized())
                {
                    return Err(Error::State(
                        "resident domain allocation/service generation mismatch".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    async fn execute(
        &self,
        base: RootBinding,
        position: usize,
        prior: &[Domain],
        candidate: Vec<Domain>,
        input: StepInput<'_>,
    ) -> Result<(Vec<Domain>, ResidentReady, InferenceReadback)> {
        self.execute_inner(
            base,
            position,
            prior,
            candidate,
            input,
            #[cfg(all(
                any(
                    feature = "gemma-bf16-qualification",
                    feature = "qwen-source-qualification"
                ),
                target_os = "linux"
            ))]
            SourceSteps {
                _observer_lifetime: std::marker::PhantomData,
                #[cfg(feature = "gemma-bf16-qualification")]
                bf16_text: None,
                #[cfg(feature = "qwen-source-qualification")]
                bf16_text_source: None,
            },
        )
        .await
    }
    async fn execute_inner(
        &self,
        base: RootBinding,
        position: usize,
        prior: &[Domain],
        candidate: Vec<Domain>,
        input: StepInput<'_>,
        #[cfg(all(
            any(
                feature = "gemma-bf16-qualification",
                feature = "qwen-source-qualification"
            ),
            target_os = "linux"
        ))]
        sources: SourceSteps<'_, '_>,
    ) -> Result<(Vec<Domain>, ResidentReady, InferenceReadback)> {
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        let bf16_text = sources.bf16_text;
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        let mut bf16_text_source = sources.bf16_text_source;
        self.service.ensure_usable()?;
        if base.source() != self.service.source_identity() || position >= self.capacity {
            return Err(Error::State("resident source/frontier mismatch".into()));
        }
        let StepInput {
            token,
            device,
            readout,
            diagnostic,
        } = input;
        if u64::from(token) >= self.vocab {
            return Err(Error::Malformed(
                "resident token exceeds original vocabulary".into(),
            ));
        }
        self.validate_domains(prior, true)?;
        self.validate_domains(&candidate, false)?;
        let bindings = bind_domains(prior, &candidate, position)?;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            ResourceOwner {
                model: self.service.owner().model.clone(),
                session: format!("session-{}", base.session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: self.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        let mut workspace = self.workspace.lock().await;
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if let Some(step) = &bf16_text_source {
            if device.is_some() || matches!(readout, Readout::None) || step.call.rows() != 1 {
                return Err(Error::Unsupported(
                    "source-bound one-row logits/greedy call".into(),
                ));
            }
            // Immutable source records share this serialized workspace/queue.
            // No new actor or family traversal is selected here.
            workspace.program = step.body.bind()?;
            workspace.state_accesses = step.body.accesses().to_vec();
            workspace.logit_accesses = workspace.state_accesses.clone();
        }
        let parameters = RuntimeParameters {
            token,
            position,
            valid_tokens: position + 1,
        };
        workspace
            .program
            .validate_runtime(self.service.client(), parameters, &bindings)?;
        let output = if matches!(readout, Readout::None) {
            ResidentOutput::StateOnly
        } else {
            ResidentOutput::Logits
        };
        let mut accesses = match output {
            ResidentOutput::StateOnly => workspace.state_accesses.clone(),
            ResidentOutput::Logits => workspace.logit_accesses.clone(),
        };
        if let Some(token) = device {
            if token.producer != Some(base) || !Arc::ptr_eq(token.lease.service(), &self.service) {
                return Err(Error::State("stale/foreign device token".into()));
            }
            accesses.push((token.lease.clone(), DeviceAccess::Read));
        }
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if let Some(feedback) = bf16_text_source.as_ref().and_then(|step| step.feedback) {
            if feedback.producer != Some(base)
                || feedback.id != token
                || !Arc::ptr_eq(feedback.lease.service(), &self.service)
                || !feedback.lease.is_initialized()
            {
                return Err(Error::State(
                    "Qwen original current feedback/input mismatch".into(),
                ));
            }
            accesses.push((feedback.lease.clone(), DeviceAccess::Read));
        }
        let packet = if matches!(readout, Readout::Greedy) {
            for handle in workspace.greedy.as_ref().ok_or_else(|| {
                Error::Unsupported("native fragment has no model greedy workspace".into())
            })? {
                accesses.push((
                    lease_for(&workspace.leases, handle)?.clone(),
                    DeviceAccess::Write,
                ));
            }
            let lease = self
                .service
                .allocate_uninitialized(12, BufferRole::Output)?;
            accesses.push((lease.clone(), DeviceAccess::Write));
            Some(lease)
        } else {
            None
        };
        for domain in prior {
            accesses.extend(
                domain
                    .leases
                    .iter()
                    .cloned()
                    .map(|l| (l, DeviceAccess::Read)),
            );
        }
        for domain in &candidate {
            accesses.extend(
                domain
                    .leases
                    .iter()
                    .cloned()
                    .map(|l| (l, DeviceAccess::Write)),
            );
        }
        slot.begin_submission()?;
        #[allow(unused_mut)] // Gemma checkpoints borrow this original armed flight.
        let mut flight = self.service.submit(base.root(), &accesses)?;
        #[allow(unused_mut)] // Source adapters are feature-gated; both use this flight.
        let mut source_submitted: Option<crate::resident_program::ResidentSubmission> = None;
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        if let Some(step) = bf16_text.as_ref() {
            source_submitted = Some(workspace.program.launch_bf16_text(
                &mut flight,
                step.call,
                base,
                step.row,
                &bindings,
                step.observer,
                step.cancelled,
            )?);
        }
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if let Some(step) = bf16_text_source.as_mut() {
            if source_submitted.is_some() {
                return Err(Error::State("two source execution adapters".into()));
            }
            source_submitted = Some(workspace.program.launch_bf16_text_source(
                &mut flight,
                step.call,
                base,
                &bindings,
                step.observer,
                step.cancelled,
            )?);
        }
        let mut submitted = if let Some(source) = source_submitted {
            source
        } else if let Some(token) = device {
            workspace.program.launch_device_token(
                self.service.client(),
                parameters,
                &bindings,
                &DeviceBuffer::new(token.lease.handle(), 3),
                output,
            )?
        } else {
            workspace
                .program
                .launch(self.service.client(), parameters, &bindings, output)?
        };
        let expected = workspace.program.plan().proposed_domains();
        if submitted.proposals.len() != expected.len() {
            return Err(Error::State("incomplete GPU state effect receipt".into()));
        }
        for ((version, buffers), expected) in submitted.proposals.iter().zip(expected) {
            let domain = candidate
                .iter()
                .find(|d| d.slot == version.slot)
                .ok_or_else(|| Error::State("foreign GPU state effect receipt".into()))?;
            let actual = buffers.handles();
            let target = domain.buffers.handles();
            if version.slot != expected.slot
                || version.ordinal != expected.ordinal
                || actual.len() != target.len()
                || actual.iter().zip(target).any(|(a, b)| !same_handle(a, &b))
            {
                return Err(Error::State(
                    "GPU state effect version/allocation mismatch".into(),
                ));
            }
        }
        if let Some(packet) = &packet {
            let logits = submitted
                .logits
                .as_ref()
                .ok_or_else(|| Error::State("missing greedy logits".into()))?;
            let numeric_zero_ties = false;
            #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
            let numeric_zero_ties = numeric_zero_ties || workspace.program.is_bf16_text();
            #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
            let numeric_zero_ties = numeric_zero_ties || workspace.program.is_bf16_text_source();
            let greedy = workspace
                .greedy
                .as_ref()
                .ok_or_else(|| Error::Unsupported("source has no model greedy workspace".into()))?;
            #[cfg(all(
                any(
                    feature = "gemma-bf16-qualification",
                    feature = "qwen-source-qualification"
                ),
                target_os = "linux"
            ))]
            let source_dispatches = if numeric_zero_ties {
                Some(crate::resident_greedy::launch_numeric_zero(
                    self.service.client(),
                    &logits.handle,
                    self.vocab as usize,
                    &greedy[0],
                    &greedy[1],
                    &packet.handle(),
                    &submitted.status,
                )?)
            } else {
                None
            };
            #[cfg(not(all(
                any(
                    feature = "gemma-bf16-qualification",
                    feature = "qwen-source-qualification"
                ),
                target_os = "linux"
            )))]
            let source_dispatches: Option<usize> = {
                let _ = numeric_zero_ties;
                None
            };
            submitted.dispatches += if let Some(n) = source_dispatches {
                n
            } else {
                crate::resident_greedy::launch(
                    self.service.client(),
                    &logits.handle,
                    self.vocab as usize,
                    &greedy[0],
                    &greedy[1],
                    &packet.handle(),
                    &submitted.status,
                )?
            };
        }

        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        if let Some(step) = bf16_text.as_ref() {
            let row = step.call.row(step.row)?;
            let observer = step.observer.borrow_mut().take();
            if let Some(observer) = observer {
                let result: Result<()> = async {
                    for (is_prior, domains, active) in [
                        (true, prior, base.frontier() as usize),
                        (false, candidate.as_slice(), row.valid_tokens()),
                    ] {
                        if is_prior && (step.row != 0 || active == 0)
                            || !is_prior && !row.needs_logits()
                        {
                            continue;
                        }
                        for domain in domains {
                            let StateSlot::Kv(slot) = domain.slot else {
                                return Err(Error::State(
                                    "Gemma actual state has a foreign domain".into(),
                                ));
                            };
                            for (plane, lease) in domain.leases.iter().enumerate() {
                                workspace
                                    .program
                                    .checkpoint_bf16_text_kv(
                                        &mut flight,
                                        lease,
                                        is_prior,
                                        slot,
                                        plane == 1,
                                        active,
                                        &mut *observer,
                                    )
                                    .await?;
                            }
                        }
                    }
                    Ok(())
                }
                .await;
                *step.observer.borrow_mut() = Some(observer);
                result?;
            }
            bf16_text::check_cancel(step.cancelled)?;
        }
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        if let Some(step) = bf16_text_source.as_ref() {
            let observer = step.observer.borrow_mut().take();
            if let Some(observer) = observer {
                let result: Result<()> = async {
                    for (prior, domains, active) in [
                        (true, prior, position),
                        (false, candidate.as_slice(), position + 1),
                    ] {
                        for domain in domains {
                            for (plane, lease) in domain.leases.iter().enumerate() {
                                workspace
                                    .program
                                    .checkpoint_bf16_text_source_state(
                                        &mut flight,
                                        lease,
                                        position,
                                        domain.slot,
                                        plane,
                                        prior,
                                        active,
                                        &mut *observer,
                                    )
                                    .await?;
                            }
                        }
                    }
                    Ok(())
                }
                .await;
                *step.observer.borrow_mut() = Some(observer);
                result?;
            }
            bf16_text_source::check_cancel(step.cancelled)?;
        }
        #[cfg(feature = "hardware-abort-probe")]
        self.qualification_after_launch(flight.handles().len())
            .await;
        let mut cohort = Vec::new();
        let add = |cohort: &mut Vec<_>, handle: &Handle, count: usize| -> Result<()> {
            let lease = lease_for(&accesses, handle)?;
            cohort.push(lease.readback(0, count * 4)?);
            Ok(())
        };
        add(&mut cohort, &submitted.status, 1)?;
        if diagnostic {
            for (block, blocks, row, rows) in &submitted.quant_status {
                add(&mut cohort, block, *blocks)?;
                add(&mut cohort, row, *rows)?;
            }
            for (handle, n) in &submitted.embedding_status {
                add(&mut cohort, handle, *n)?;
            }
        }
        let statuses = cohort.len();
        let observe_bf16_text_source_logits = false;
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        let observe_bf16_text_source_logits = observe_bf16_text_source_logits
            || bf16_text_source.as_ref().is_some_and(|step| {
                matches!(readout, Readout::Greedy) && step.observer.borrow().is_some()
            });
        match readout {
            Readout::Logits => {
                let logits = submitted
                    .logits
                    .as_ref()
                    .ok_or_else(|| Error::State("missing resident logits".into()))?;
                let lease = lease_for(&accesses, &logits.handle)?;
                cohort.push(logits.readback(
                    lease,
                    ResidentBufferLayout::new(ResidentPhysicalType::F32, logits.elements)?,
                )?);
            }
            Readout::Greedy => {
                let p = packet.as_ref().unwrap();
                cohort.push(p.readback(0, 12)?);
                if observe_bf16_text_source_logits {
                    let logits = submitted
                        .logits
                        .as_ref()
                        .ok_or_else(|| Error::State("observed source logits absent".into()))?;
                    let lease = lease_for(&accesses, &logits.handle)?;
                    cohort.push(logits.readback(
                        lease,
                        ResidentBufferLayout::new(ResidentPhysicalType::F32, logits.elements)?,
                    )?);
                }
            }
            Readout::None => {
                if submitted.logits.is_some() {
                    return Err(Error::State("state-only graph returned logits".into()));
                }
            }
        }
        let domain_start = cohort.len();
        if diagnostic {
            for domain in &candidate {
                for l in &domain.leases {
                    cohort.push(l.readback(0, l.bytes())?);
                }
            }
        }
        let observation_start = cohort.len();
        let mut observation_keys = Vec::new();
        if diagnostic {
            let nodes = match output {
                ResidentOutput::StateOnly => workspace.program.plan().state_nodes(),
                ResidentOutput::Logits => workspace.program.plan().nodes(),
            };
            for node in nodes {
                match &node.operation {
                    PlanOperation::Observe { layer, stage } => {
                        let value = &workspace.program.values()[node.inputs[0].0 as usize];
                        value.require_layout(ResidentBufferLayout::new(
                            ResidentPhysicalType::F32,
                            value.elements,
                        )?)?;
                        add(&mut cohort, &value.handle, value.elements)?;
                        observation_keys.push((*layer, *stage, value.elements));
                    }
                    PlanOperation::ObserveState {
                        layer,
                        stage,
                        state,
                    } => {
                        let domain = candidate
                            .iter()
                            .find(|d| d.slot == state.slot)
                            .ok_or_else(|| Error::State("missing observed domain".into()))?;
                        let l = &domain.leases[0];
                        cohort.push(l.readback(0, l.bytes())?);
                        observation_keys.push((*layer, *stage, l.bytes() / 4));
                    }
                    _ => {}
                }
            }
        }
        let mut promoted = candidate
            .iter()
            .flat_map(|d| d.leases.iter().cloned())
            .collect::<Vec<_>>();
        promoted.extend(packet.iter().cloned());
        let expected_readback_bytes = cohort.iter().map(|item| item.bytes()).sum();
        let (ready, read) = flight
            .complete_with_readback(&promoted, &cohort, |values| {
                if values[..statuses].iter().any(|b| !all_zero(b)) {
                    return Err(Error::State(
                        "resident GPU status rejected proposed graph".into(),
                    ));
                }
                if matches!(readout, Readout::Logits)
                    && !words(&values[statuses])?.iter().all(|x| x.is_finite())
                {
                    return Err(Error::State("nonfinite resident logits".into()));
                }
                if observe_bf16_text_source_logits
                    && !words(&values[statuses + 1])?.iter().all(|x| x.is_finite())
                {
                    return Err(Error::State("nonfinite observed source logits".into()));
                }
                if matches!(readout, Readout::Greedy) {
                    let packet = &values[statuses];
                    let id = u32::from_ne_bytes(packet[0..4].try_into().unwrap());
                    let flags = u32::from_ne_bytes(packet[8..12].try_into().unwrap());
                    if u64::from(id) >= self.vocab || flags != 0 {
                        return Err(Error::State("invalid device greedy packet".into()));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let mut result = InferenceReadback {
            logits: if matches!(readout, Readout::Logits) {
                words(&read[statuses])?
            } else if observe_bf16_text_source_logits {
                words(&read[statuses + 1])?
            } else {
                Vec::new()
            },
            dispatches: submitted.dispatches,
            bytes: expected_readback_bytes,
            chunk_bytes: expected_readback_bytes,
            state_only_steps: usize::from(output == ResidentOutput::StateOnly),
            logit_steps: usize::from(output == ResidentOutput::Logits),
            ..Default::default()
        };
        if let Some(lease) = packet {
            result.device_token = Some(ResidentToken {
                lease,
                producer: None,
                id: u32::from_ne_bytes(read[statuses][0..4].try_into().unwrap()),
                max_bits: u32::from_ne_bytes(read[statuses][4..8].try_into().unwrap()),
            });
        }
        if diagnostic {
            let mut cursor = domain_start;
            for domain in &candidate {
                let mut flat = Vec::new();
                for l in &domain.leases {
                    let data = words(&read[cursor])?;
                    match domain.buffers {
                        StateBuffers::Vector(_) => flat.extend(data),
                        StateBuffers::Kv { .. } => {
                            let active = l.bytes() / 4 / self.capacity * (position + 1);
                            if data[active..].iter().any(|x| x.to_bits() != 0) {
                                return Err(Error::State(
                                    "resident canonical KV tail is not +0".into(),
                                ));
                            }
                            flat.extend_from_slice(&data[..active]);
                        }
                        StateBuffers::PagedKv => {
                            return Err(Error::State(
                                "resident diagnostic readback requires explicit paged KV reader"
                                    .into(),
                            ));
                        }
                    }
                    cursor += 1;
                }
                result.domains.push((domain.slot, flat));
            }
            for ((layer, stage, _), data) in
                observation_keys.into_iter().zip(&read[observation_start..])
            {
                let mut values = words(data)?;
                if matches!(stage, "full_qk_dots" | "full_scores" | "full_probs") {
                    let heads = values.len() / self.capacity;
                    values = (0..heads)
                        .flat_map(|h| {
                            values[h * self.capacity..h * self.capacity + position + 1]
                                .iter()
                                .copied()
                        })
                        .collect();
                }
                result.observations.push(LayerObservation {
                    position,
                    layer,
                    stage,
                    values,
                });
            }
        }
        Ok((candidate, ready, result))
    }
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
enum ForwardCandidateState {
    Contiguous {
        domains: Vec<Domain>,
        bindings: Vec<StateBinding>,
    },
    #[cfg(feature = "cuda-cublas")]
    Hybrid(state_pool::HybridCandidateBundle),
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
impl ForwardCandidateState {
    fn bindings(&self) -> &[StateBinding] {
        match self {
            Self::Contiguous { bindings, .. } => bindings,
            #[cfg(feature = "cuda-cublas")]
            Self::Hybrid(bundle) => bundle.bindings(),
        }
    }

    fn state_accesses(&self, prior: &[Domain]) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        match self {
            Self::Contiguous { domains, .. } => {
                let mut accesses = Vec::new();
                for domain in prior {
                    accesses.extend(
                        domain
                            .leases
                            .iter()
                            .cloned()
                            .map(|lease| (lease, DeviceAccess::Read)),
                    );
                }
                for domain in domains {
                    accesses.extend(
                        domain
                            .leases
                            .iter()
                            .cloned()
                            .map(|lease| (lease, DeviceAccess::Write)),
                    );
                }
                Ok(accesses)
            }
            #[cfg(feature = "cuda-cublas")]
            Self::Hybrid(bundle) => bundle.accesses(),
        }
    }

    fn write_outputs(&self) -> Result<Vec<ResidentLease>> {
        match self {
            Self::Contiguous { domains, .. } => Ok(domains
                .iter()
                .flat_map(|domain| domain.leases.iter().cloned())
                .collect()),
            #[cfg(feature = "cuda-cublas")]
            Self::Hybrid(bundle) => bundle.write_outputs(),
        }
    }

    #[cfg(feature = "cuda-cublas")]
    fn enqueue_gather_cuda(
        &self,
        flight: &crate::resident_leases::ResidentSubmission<'_>,
    ) -> Result<usize> {
        match self {
            Self::Contiguous { .. } => Ok(0),
            Self::Hybrid(bundle) => bundle.enqueue_gather_cuda(flight),
        }
    }

    #[cfg(feature = "cuda-cublas")]
    fn enqueue_scatter_cuda(
        &self,
        flight: &crate::resident_leases::ResidentSubmission<'_>,
    ) -> Result<usize> {
        match self {
            Self::Contiguous { .. } => Ok(0),
            Self::Hybrid(bundle) => bundle.enqueue_scatter_cuda(flight),
        }
    }

    fn into_domains(self) -> Result<Vec<Domain>> {
        match self {
            Self::Contiguous { domains, .. } => Ok(domains),
            #[cfg(feature = "cuda-cublas")]
            Self::Hybrid(bundle) => bundle.into_domains(),
        }
    }
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
fn dedupe_resident_accesses(
    accesses: Vec<(ResidentLease, DeviceAccess)>,
) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
    let mut unique = std::collections::BTreeMap::new();
    for (lease, access) in accesses {
        match unique.entry(lease.id()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((lease, access));
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let (existing, existing_access) = entry.get_mut();
                if existing.allocation_generation() != lease.allocation_generation()
                    || existing.bytes() != lease.bytes()
                    || existing.role() != lease.role()
                    || !Arc::ptr_eq(existing.service(), lease.service())
                {
                    return Err(Error::State(
                        "forward resident access identity aliases".into(),
                    ));
                }
                if access == DeviceAccess::Write {
                    *existing_access = DeviceAccess::Write;
                }
            }
        }
    }
    Ok(unique.into_values().collect())
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
impl ResidentModel {
    async fn forward_candidate_state(
        &self,
        prior: &[Domain],
        tokens: usize,
        position: usize,
    ) -> Result<ForwardCandidateState> {
        if let Some(storage) = self.state_pool.get()
            && let state_pool::PooledStateStorage::Hybrid(pools) = &storage.storage
        {
            #[cfg(feature = "cuda-cublas")]
            {
                return Ok(ForwardCandidateState::Hybrid(pools.candidate(
                    &self.states,
                    prior,
                    tokens,
                    position,
                )?));
            }
            #[cfg(not(feature = "cuda-cublas"))]
            {
                let _ = (pools, tokens);
                return Err(Error::Unsupported(
                    "hybrid paged state requires CUDA forward".into(),
                ));
            }
        }
        let domains = self.allocate_state(false).await?;
        self.validate_domains(&domains, false)?;
        let bindings = bind_domains(prior, &domains, position)?;
        Ok(ForwardCandidateState::Contiguous { domains, bindings })
    }

    async fn execute_forward_source(
        &self,
        base: RootBinding,
        prior: &[Domain],
        candidate: ForwardCandidateState,
        request: &ForwardPlan,
        options: ForwardSourceOptions<'_, '_>,
    ) -> Result<(Vec<Domain>, ResidentReady, InferenceReadback)> {
        let ForwardSourceOptions {
            output,
            readout,
            device_token,
            cancelled,
            observer,
            completion_owner,
            submitted: submitted_flag,
        } = options;
        forward_not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        if !matches!(
            (output, readout),
            (ResidentOutput::StateOnly, Readout::None)
                | (ResidentOutput::Logits, Readout::Logits)
                | (ResidentOutput::Logits, Readout::Greedy)
        ) {
            return Err(Error::State(
                "CUDA forward execution/readout policy differs".into(),
            ));
        }
        let call = request
            .text_invocation()
            .ok_or_else(|| Error::State("CUDA forward materialized call absent".into()))?;
        if base.source() != self.service.source_identity()
            || call.source() != base.source()
            || call.tokens().base() as u64 != base.frontier()
            || call.tokens().capacity() != self.capacity
            || call.tokens().vocabulary() as u64 != self.vocab
            || call.tokens().frontier() > self.capacity
        {
            return Err(Error::State(
                "CUDA forward session root/call geometry differs".into(),
            ));
        }
        if let Some(token) = &device_token
            && (!matches!(readout, Readout::Greedy)
                || call.tokens().len() != 1
                || token.producer != Some(base)
                || token.id != call.tokens().ids()[0]
                || token.lease.bytes() != 12
                || !Arc::ptr_eq(token.lease.service(), &self.service))
        {
            return Err(Error::State(
                "CUDA forward device token differs from request root/input".into(),
            ));
        }
        self.validate_domains(prior, true)?;
        let bindings = candidate.bindings();

        forward_not_cancelled(cancelled)?;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            ResourceOwner {
                model: self.service.owner().model.clone(),
                session: format!("forward-session-{}", base.session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: self.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        forward_not_cancelled(cancelled)?;
        let workspace = self.workspace.lock().await;
        if !workspace.program.is_forward_source() {
            return Err(Error::State(
                "CUDA forward request reached another resident model".into(),
            ));
        }
        workspace.program.validate_forward_source_request(request)?;
        workspace
            .program
            .validate_text_execution_states(request, bindings)?;
        // Completion owners are retained by the armed physical flight and follow
        // it into quarantine on uncertainty. Only backend page semantics may
        // require waiting through cancellation.
        let wait_on_cancel = workspace.program.text_execution_waits_on_cancel();
        let mut accesses = match output {
            ResidentOutput::StateOnly => workspace.state_accesses.clone(),
            ResidentOutput::Logits => workspace.logit_accesses.clone(),
        };
        let packet = if matches!(readout, Readout::Greedy) {
            let greedy = workspace.greedy.as_ref().ok_or_else(|| {
                Error::Unsupported("CUDA forward model has no greedy workspace".into())
            })?;
            for handle in greedy {
                accesses.push((
                    lease_for(&workspace.leases, handle)?.clone(),
                    DeviceAccess::Write,
                ));
            }
            let packet = self
                .service
                .allocate_uninitialized(12, BufferRole::Output)?;
            accesses.push((packet.clone(), DeviceAccess::Write));
            Some(packet)
        } else {
            None
        };
        accesses.extend(candidate.state_accesses(prior)?);
        if let Some(token) = &device_token {
            accesses.push((token.lease.clone(), DeviceAccess::Read));
        }
        let accesses = dedupe_resident_accesses(accesses)?;
        forward_not_cancelled(cancelled)?;
        slot.begin_submission()?;
        let mut flight = self.service.submit(base.root(), &accesses)?;
        if let Some(owner) = completion_owner {
            flight.retain_completion_owner(owner)?;
        }
        if let Some(flag) = submitted_flag {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        #[cfg(feature = "cuda-cublas")]
        let gather_dispatches = candidate.enqueue_gather_cuda(&flight)?;
        let submitted = workspace
            .program
            .launch_forward_source(
                &mut flight,
                request,
                bindings,
                output,
                matches!(readout, Readout::Greedy),
                device_token
                    .as_ref()
                    .map(|token| token.lease.handle().offset_end(8)),
                if wait_on_cancel { None } else { cancelled },
                observer,
            )
            .await?;
        #[cfg(feature = "cuda-cublas")]
        let mut submitted = {
            let mut submitted = submitted;
            let scatter_dispatches = candidate.enqueue_scatter_cuda(&flight)?;
            submitted.dispatches = submitted
                .dispatches
                .checked_add(gather_dispatches)
                .and_then(|n| n.checked_add(scatter_dispatches))
                .ok_or_else(|| Error::Limit("forward hybrid state dispatch count".into()))?;
            submitted
        };
        #[cfg(not(feature = "cuda-cublas"))]
        let submitted = submitted;
        if let Some(packet) = &packet {
            let logits = submitted
                .logits
                .as_ref()
                .ok_or_else(|| Error::State("CUDA forward greedy logits absent".into()))?;
            let greedy = workspace.greedy.as_ref().ok_or_else(|| {
                Error::Unsupported("CUDA forward model has no greedy workspace".into())
            })?;
            #[cfg(feature = "cuda-cublas")]
            {
                logits.require_layout(ResidentBufferLayout::new(
                    ResidentPhysicalType::Bf16,
                    self.vocab as usize,
                )?)?;
                submitted.dispatches += crate::resident_greedy::launch_bf16_numeric_zero(
                    self.service.client(),
                    &logits.handle,
                    self.vocab as usize,
                    &greedy[0],
                    &greedy[1],
                    &packet.handle(),
                    &submitted.status,
                )?;
            }
            #[cfg(not(feature = "cuda-cublas"))]
            {
                let _ = (logits, greedy, packet);
                return Err(Error::Unsupported(
                    "forward source device greedy requires CUDA".into(),
                ));
            }
        }
        #[cfg(feature = "hardware-abort-probe")]
        self.qualification_after_launch(flight.handles().len())
            .await;
        if !wait_on_cancel && forward_cancel_requested(cancelled) {
            retire_forward_cancelled(flight, &accesses, &submitted.status).await?;
            slot.finish()?;
            return Err(Error::State("CUDA forward request cancelled".into()));
        }

        let expected = workspace.program.plan().proposed_domains();
        if submitted.proposals.len() != expected.len() {
            return Err(Error::State(
                "CUDA forward incomplete state effect receipt".into(),
            ));
        }
        for ((version, buffers), expected) in submitted.proposals.iter().zip(expected) {
            let binding = bindings
                .iter()
                .find(|binding| binding.slot == version.slot)
                .ok_or_else(|| Error::State("CUDA forward foreign state proposal".into()))?;
            let actual = buffers.handles();
            let target = binding.candidate.handles();
            if version.slot != expected.slot
                || version.ordinal != expected.ordinal
                || actual.len() != target.len()
                || actual.iter().zip(target).any(|(a, b)| !same_handle(a, &b))
            {
                return Err(Error::State(
                    "CUDA forward state effect version/allocation mismatch".into(),
                ));
            }
        }

        let mut cohort = Vec::with_capacity(2);
        match readout {
            Readout::Logits => {
                cohort.push(lease_for(&accesses, &submitted.status)?.readback(0, 4)?);
                let logits = submitted
                    .logits
                    .as_ref()
                    .ok_or_else(|| Error::State("CUDA forward logits receipt absent".into()))?;
                let lease = lease_for(&accesses, &logits.handle)?;
                cohort.push(logits.readback(
                    lease,
                    ResidentBufferLayout::new(ResidentPhysicalType::F32, logits.elements)?,
                )?);
            }
            Readout::Greedy => {
                // finish_argmax folds the shared status into packet[2] after
                // every model/argmax launch, so a second status readback is redundant.
                cohort.push(
                    packet
                        .as_ref()
                        .ok_or_else(|| Error::State("CUDA forward greedy packet absent".into()))?
                        .readback(0, 12)?,
                );
            }
            Readout::None => {
                if submitted.logits.is_some() {
                    return Err(Error::State(
                        "CUDA forward state-only execution returned logits".into(),
                    ));
                }
                cohort.push(lease_for(&accesses, &submitted.status)?.readback(0, 4)?);
            }
        }

        let mut promoted = candidate.write_outputs()?;
        promoted.extend(packet.iter().cloned());
        let expected_readback_bytes = cohort.iter().map(|item| item.bytes()).sum();
        if !wait_on_cancel && forward_cancel_requested(cancelled) {
            retire_forward_cancelled(flight, &accesses, &submitted.status).await?;
            slot.finish()?;
            return Err(Error::State("CUDA forward request cancelled".into()));
        }
        let (ready, read) = flight
            .complete_with_readback(&promoted, &cohort, |values| {
                match readout {
                    Readout::Logits => {
                        if !all_zero(&values[0]) {
                            return Err(Error::State(
                                "CUDA forward shared status rejected proposed graph".into(),
                            ));
                        }
                        if !words(&values[1])?.iter().all(|value| value.is_finite()) {
                            return Err(Error::State("CUDA forward nonfinite F32 logits".into()));
                        }
                    }
                    Readout::Greedy => {
                        let packet = &values[0];
                        if packet.len() != 12 {
                            return Err(Error::State("CUDA forward greedy packet extent".into()));
                        }
                        let id = u32::from_ne_bytes(packet[0..4].try_into().unwrap());
                        let bits = u32::from_ne_bytes(packet[4..8].try_into().unwrap());
                        let flags = u32::from_ne_bytes(packet[8..12].try_into().unwrap());
                        if u64::from(id) >= self.vocab
                            || !f32::from_bits(bits).is_finite()
                            || flags != 0
                        {
                            return Err(Error::State("CUDA forward invalid greedy packet".into()));
                        }
                    }
                    Readout::None => {
                        if !all_zero(&values[0]) {
                            return Err(Error::State(
                                "CUDA forward shared status rejected proposed graph".into(),
                            ));
                        }
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        if !wait_on_cancel {
            forward_not_cancelled(cancelled)?;
        }
        let candidate = candidate.into_domains()?;
        self.validate_domains(&candidate, true)?;

        let device_token = if matches!(readout, Readout::Greedy) {
            Some(ResidentToken {
                lease: packet.ok_or_else(|| {
                    Error::State("CUDA forward greedy packet absent after completion".into())
                })?,
                producer: None,
                id: u32::from_ne_bytes(read[0][0..4].try_into().unwrap()),
                max_bits: u32::from_ne_bytes(read[0][4..8].try_into().unwrap()),
            })
        } else {
            None
        };
        Ok((
            candidate,
            ready,
            InferenceReadback {
                logits: if matches!(readout, Readout::Logits) {
                    words(&read[1])?
                } else {
                    Vec::new()
                },
                device_token,
                dispatches: submitted.dispatches,
                bytes: expected_readback_bytes,
                chunk_bytes: expected_readback_bytes,
                state_only_steps: usize::from(output == ResidentOutput::StateOnly),
                logit_steps: usize::from(output == ResidentOutput::Logits),
                ..Default::default()
            },
        ))
    }
}

fn make_domain(
    r: &StateRequirement,
    allocate: &mut impl FnMut(ResidentBufferLayout) -> Result<ResidentLease>,
) -> Result<Domain> {
    let (planes, physical) = r.plane_layout()?;
    let (leases, buffers) = match r.layout {
        StateLayout::Vector { .. } => {
            if planes != 1 {
                return Err(Error::State("resident vector state plane count".into()));
            }
            let l = allocate(physical)?;
            let b = DeviceBuffer::from_lease(
                l.clone(),
                physical,
                BufferRole::RecurrentState,
                DeviceAccess::Write,
            )?;
            (vec![l], StateBuffers::Vector(b))
        }
        StateLayout::Kv { .. } => {
            if planes != 2 {
                return Err(Error::State("resident KV state plane count".into()));
            }
            let k = allocate(physical)?;
            let v = allocate(physical)?;
            let b = StateBuffers::Kv {
                keys: DeviceBuffer::from_lease(
                    k.clone(),
                    physical,
                    BufferRole::RecurrentState,
                    DeviceAccess::Write,
                )?,
                values: DeviceBuffer::from_lease(
                    v.clone(),
                    physical,
                    BufferRole::RecurrentState,
                    DeviceAccess::Write,
                )?,
            };
            (vec![k, v], b)
        }
    };
    let generations = leases
        .iter()
        .map(ResidentLease::allocation_generation)
        .collect();
    Ok(Domain {
        slot: r.slot,
        buffers,
        paged_kv: None,
        leases,
        generations,
    })
}
fn bind_domains(
    prior: &[Domain],
    candidate: &[Domain],
    position: usize,
) -> Result<Vec<StateBinding>> {
    if prior.len() != candidate.len() {
        return Err(Error::State("resident state domain count".into()));
    }
    prior
        .iter()
        .zip(candidate)
        .map(|(p, c)| {
            if p.slot != c.slot {
                return Err(Error::State("resident state domain order".into()));
            }
            Ok(StateBinding {
                slot: p.slot,
                prior: p.buffers.clone(),
                candidate: c.buffers.clone(),
                prior_tokens: position,
                mode: crate::resident_program::StateBindingMode::Distinct,
            })
        })
        .collect()
}
impl ResidentSnapshot {
    pub async fn read_domains(&self) -> Result<Vec<(StateSlot, Vec<f32>)>> {
        self.read_domains_retaining(None).await
    }
    async fn read_domains_retaining(
        &self,
        owner: Option<crate::physical_quarantine::CompletionOwner>,
    ) -> Result<Vec<(StateSlot, Vec<f32>)>> {
        #[derive(Clone)]
        struct SnapshotPlane {
            lease: ResidentLease,
            layout: ResidentBufferLayout,
            active_elements: usize,
            zero_tail: bool,
        }

        let frontier = usize::try_from(self.binding.frontier())
            .map_err(|_| Error::Limit("resident snapshot frontier".into()))?;
        let domain_planes = self
            .domains
            .iter()
            .map(|domain| {
                if matches!(domain.buffers, StateBuffers::PagedKv) {
                    let paged = domain.paged_kv()?;
                    if paged.tokens()? != frontier {
                        return Err(Error::State(
                            "resident paged snapshot frontier differs".into(),
                        ));
                    }
                    return paged
                        .snapshot_planes()?
                        .into_iter()
                        .map(|(lease, layout, active_elements)| {
                            if active_elements == 0 || active_elements > layout.elements() {
                                return Err(Error::State(
                                    "resident paged snapshot active extent differs".into(),
                                ));
                            }
                            Ok(SnapshotPlane {
                                lease,
                                layout,
                                active_elements,
                                zero_tail: active_elements < layout.elements(),
                            })
                        })
                        .collect::<Result<Vec<_>>>();
                }

                let buffers = domain.buffers.buffers();
                if buffers.len() != domain.leases.len() {
                    return Err(Error::State(
                        "resident snapshot typed domain plane coverage differs".into(),
                    ));
                }
                domain
                    .leases
                    .iter()
                    .zip(buffers)
                    .map(|(lease, buffer)| {
                        let layout = buffer.physical_layout();
                        if lease.bytes() != layout.bytes()? {
                            return Err(Error::State(
                                "resident snapshot typed lease/layout differs".into(),
                            ));
                        }
                        let active_elements = if matches!(&domain.buffers, StateBuffers::Kv { .. })
                        {
                            if !layout.elements().is_multiple_of(self.capacity) {
                                return Err(Error::State(
                                    "resident snapshot KV plane/capacity differs".into(),
                                ));
                            }
                            let slab = layout.elements() / self.capacity;
                            slab.checked_mul(frontier)
                                .filter(|active| *active <= layout.elements())
                                .ok_or_else(|| {
                                    Error::Limit("resident snapshot active KV extent".into())
                                })?
                        } else {
                            layout.elements()
                        };
                        Ok(SnapshotPlane {
                            lease: lease.clone(),
                            layout,
                            active_elements,
                            zero_tail: matches!(&domain.buffers, StateBuffers::Kv { .. })
                                && active_elements < layout.elements(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let planes = domain_planes.iter().flatten().cloned().collect::<Vec<_>>();
        let service = self
            .domains
            .iter()
            .flat_map(|domain| domain.leases.iter())
            .next()
            .ok_or_else(|| Error::State("empty resident snapshot".into()))?
            .service()
            .clone();
        service.ensure_usable()?;
        if service.generation() != self.service_generation
            || service.source_identity() != self.binding.source()
        {
            return Err(Error::State("snapshot service generation mismatch".into()));
        }
        let accesses = planes
            .iter()
            .map(|plane| (plane.lease.clone(), DeviceAccess::Read))
            .collect::<Vec<_>>();
        let cohort = planes
            .iter()
            .map(|plane| plane.lease.readback(0, plane.lease.bytes()))
            .collect::<Result<Vec<_>>>()?;
        let ticket = service.queue().submit(
            service.budget(),
            ResourceOwner {
                model: service.owner().model.clone(),
                session: format!("snapshot-{}", self.binding.session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: planes.iter().map(|plane| plane.lease.bytes() as u64).sum(),
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(self.binding.root(), &accesses)?;
        if let Some(owner) = owner {
            flight.retain_completion_owner(owner)?;
        }
        let read = flight
            .complete_read_only_readback(&cohort, |data| {
                if data.len() != planes.len() {
                    return Err(Error::State(
                        "resident snapshot typed readback coverage differs".into(),
                    ));
                }
                for (bytes, plane) in data.iter().zip(&planes) {
                    if !resident_plane_finite(bytes, plane.layout)? {
                        return Err(Error::State("nonfinite committed resident snapshot".into()));
                    }
                    if plane.zero_tail
                        && !resident_plane_tail_zero(bytes, plane.layout, plane.active_elements)?
                    {
                        return Err(Error::State(
                            "snapshot noncanonical inactive state tail".into(),
                        ));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let mut cursor = 0;
        let mut output = Vec::new();
        for (domain, planes) in self.domains.iter().zip(&domain_planes) {
            let mut flat = Vec::new();
            for plane in planes {
                flat.extend(resident_plane_values(
                    &read[cursor],
                    plane.layout,
                    plane.active_elements,
                )?);
                cursor += 1;
            }
            output.push((domain.slot, flat));
        }
        if cursor != read.len() {
            return Err(Error::State(
                "resident snapshot readback cursor coverage differs".into(),
            ));
        }
        Ok(output)
    }
}

impl ResidentSession {
    fn new(model: Arc<ResidentModel>, authority: SessionAuthority, domains: Vec<Domain>) -> Self {
        Self {
            model,
            authority,
            domains,
            derived: DerivedStateCache::default(),
        }
    }

    /// Original session's physical model owner, not a new model lookup.
    pub fn resident_model(&self) -> &Arc<ResidentModel> {
        &self.model
    }
    pub fn binding(&self) -> RootBinding {
        self.authority.binding()
    }
    /// Variable backing copied by a complete-domain snapshot (including
    /// page tables and lease/generation vectors). Excludes fixed owner
    /// controls, allocator rounding and the shared physical allocations.
    pub fn snapshot_backing_bytes(&self) -> Result<usize> {
        let mut bytes = std::mem::size_of_val(self.domains.as_slice());
        for domain in &self.domains {
            let pages = domain
                .paged_kv
                .as_ref()
                .map(|kv| kv.cloned_page_bytes())
                .transpose()?
                .unwrap_or(0);
            bytes = bytes
                .checked_add(std::mem::size_of_val(domain.leases.as_slice()))
                .and_then(|n| n.checked_add(std::mem::size_of_val(domain.generations.as_slice())))
                .and_then(|n| n.checked_add(pages))
                .ok_or_else(|| Error::Limit("resident snapshot backing overflow".into()))?;
        }
        Ok(bytes)
    }
    pub fn snapshot(&self) -> ResidentSnapshot {
        ResidentSnapshot {
            binding: self.binding(),
            service_generation: self.model.service.generation(),
            capacity: self.model.capacity,
            domains: self.domains.clone(),
        }
    }
    pub async fn prepare(&self, tokens: &[u32], diagnostic: bool) -> Result<PreparedInference> {
        self.prepare_mode(tokens, diagnostic, Readout::Logits).await
    }
    async fn prepare_mode(
        &self,
        tokens: &[u32],
        diagnostic: bool,
        readout: Readout,
    ) -> Result<PreparedInference> {
        self.prepare_mode_inner(
            tokens,
            diagnostic,
            readout,
            #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
            None,
            #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
            None,
        )
        .await
    }
    // `'o` exists only for the Linux source-option adapters. Keep it when
    // either qualified parameter is compiled; otherwise narrow-clippy-allow
    // the genuinely absent use instead of changing the signature or body.
    #[cfg_attr(
        not(all(
            target_os = "linux",
            any(
                feature = "gemma-bf16-qualification",
                feature = "qwen-source-qualification"
            )
        )),
        allow(clippy::extra_unused_lifetimes)
    )]
    async fn prepare_mode_inner<'o>(
        &self,
        tokens: &[u32],
        diagnostic: bool,
        readout: Readout,
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))] bf16_text: Option<
            bf16_text::TextOptions<'_, 'o>,
        >,
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        bf16_text_source: Option<bf16_text_source::TextSourceOptions<'_, 'o>>,
    ) -> Result<PreparedInference> {
        let count =
            u64::try_from(tokens.len()).map_err(|_| Error::Limit("resident token count".into()))?;
        let attempt = self.authority.begin_advance(count)?;
        let base = self.binding();
        if base
            .frontier()
            .checked_add(count)
            .is_none_or(|n| n > self.model.capacity as u64)
        {
            return Err(Error::Limit("resident context capacity exhausted".into()));
        }
        if tokens
            .iter()
            .any(|token| u64::from(*token) >= self.model.vocab)
        {
            return Err(Error::Malformed(
                "resident chunk contains invalid source token".into(),
            ));
        }
        #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
        let (_call_admission, call) = if let Some(options) = &bf16_text {
            bf16_text::check_cancel(options.cancelled)?;
            if options.source.source_identity() != base.source()
                || options.bindings.source_identity() != base.source()
                || options.bindings.call().geometry().capacity() != self.model.capacity
                || options.bindings.call().geometry().vocabulary() as u64 != self.model.vocab
            {
                return Err(Error::State(
                    "text original call source/geometry differs".into(),
                ));
            }
            (
                Some(options.source.reserve_workspace(65536)?),
                Some(
                    options
                        .bindings
                        .bind_call(attempt.clone(), Arc::from(tokens))?,
                ),
            )
        } else {
            (None, None)
        };
        #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
        let (_bf16_text_text_invocation_admission, bf16_text_text_invocation) =
            if let Some(options) = &bf16_text_source {
                bf16_text_source::check_cancel(options.cancelled)?;
                if tokens.len() != 1 || base.frontier() as usize >= options.bodies.len() {
                    return Err(Error::Unsupported(
                        "source single-row recorded frontier".into(),
                    ));
                }
                let body = &options.bodies[base.frontier() as usize];
                if body.recorded().plan().source() != base.source() {
                    return Err(Error::State("resident-text model owner differs".into()));
                }
                (
                    Some(body.source().reserve_workspace(16 * 1024 * 1024)?),
                    Some(body.recorded().bind_call(attempt.clone(), tokens[0])?),
                )
            } else {
                (None, None)
            };
        // Reserve the complete private ping-pong peak before the first launch.
        let mut pool = vec![self.model.allocate_state(false).await?];
        if tokens.len() > 1 {
            pool.push(self.model.allocate_state(false).await?);
        }
        let mut domains = self.domains.clone();
        let mut last = None;
        let mut dispatches = 0;
        let mut chunk_bytes = 0;
        let mut state_only_steps = 0;
        let mut logit_steps = 0;
        for (i, token) in tokens.iter().enumerate() {
            let (next, ready, readback) = self
                .model
                .execute_inner(
                    base,
                    base.frontier() as usize + i,
                    &domains,
                    pool[i % pool.len()].clone(),
                    StepInput {
                        token: *token,
                        device: None,
                        readout: if i + 1 != tokens.len() {
                            Readout::None
                        } else {
                            readout
                        },
                        diagnostic: diagnostic && i + 1 == tokens.len(),
                    },
                    #[cfg(all(
                        any(
                            feature = "gemma-bf16-qualification",
                            feature = "qwen-source-qualification"
                        ),
                        target_os = "linux"
                    ))]
                    SourceSteps {
                        _observer_lifetime: std::marker::PhantomData,
                        #[cfg(feature = "gemma-bf16-qualification")]
                        bf16_text: bf16_text.as_ref().map(|options| bf16_text::TextStep {
                            call: call.as_ref().unwrap(),
                            row: i,
                            observer: options.observer,
                            cancelled: options.cancelled,
                        }),
                        #[cfg(feature = "qwen-source-qualification")]
                        bf16_text_source: bf16_text_source.as_ref().map(|options| {
                            bf16_text_source::TextSourceStep {
                                body: &options.bodies[base.frontier() as usize],
                                call: bf16_text_text_invocation.as_ref().unwrap(),
                                observer: options.observer,
                                feedback: options.feedback,
                                cancelled: options.cancelled,
                            }
                        }),
                    },
                )
                .await?;
            dispatches += readback.dispatches;
            chunk_bytes += readback.chunk_bytes;
            state_only_steps += readback.state_only_steps;
            logit_steps += readback.logit_steps;
            domains = next;
            last = Some((ready, readback));
        }
        let (ready, mut readback) =
            last.ok_or_else(|| Error::State("empty resident transaction".into()))?;
        readback.dispatches = dispatches;
        readback.chunk_bytes = chunk_bytes;
        readback.state_only_steps = state_only_steps;
        readback.logit_steps = logit_steps;
        Ok(PreparedInference {
            attempt,
            domains,
            derived: Vec::new(),
            ready,
            readback,
        })
    }
    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn prepare_cuda_forward(&self, request: ForwardPlan) -> Result<PreparedInference> {
        self.prepare_forward_source_mode(
            request,
            zerogravity_core::accelerator::program::NumericalContract::CudaSm86Bf16ReferenceV1,
            ForwardSourceOptions {
                output: ResidentOutput::Logits,
                readout: Readout::Logits,
                device_token: None,
                cancelled: None,
                observer: None,
                completion_owner: None,
                submitted: None,
            },
        )
        .await
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn prepare_cuda_forward_cancellable(
        &self,
        request: ForwardPlan,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<PreparedInference> {
        self.prepare_forward_source_mode(
            request,
            zerogravity_core::accelerator::program::NumericalContract::CudaSm86Bf16ReferenceV1,
            ForwardSourceOptions {
                output: ResidentOutput::Logits,
                readout: Readout::Logits,
                device_token: None,
                cancelled: Some(cancelled),
                observer: None,
                completion_owner: None,
                submitted: None,
            },
        )
        .await
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn prepare_cuda_forward_state_only(
        &self,
        request: ForwardPlan,
    ) -> Result<PreparedInference> {
        self.prepare_forward_source_mode(
            request,
            zerogravity_core::accelerator::program::NumericalContract::CudaSm86Bf16ReferenceV1,
            ForwardSourceOptions {
                output: ResidentOutput::StateOnly,
                readout: Readout::None,
                device_token: None,
                cancelled: None,
                observer: None,
                completion_owner: None,
                submitted: None,
            },
        )
        .await
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn prepare_cuda_forward_state_only_cancellable(
        &self,
        request: ForwardPlan,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<PreparedInference> {
        self.prepare_forward_source_mode(
            request,
            zerogravity_core::accelerator::program::NumericalContract::CudaSm86Bf16ReferenceV1,
            ForwardSourceOptions {
                output: ResidentOutput::StateOnly,
                readout: Readout::None,
                device_token: None,
                cancelled: Some(cancelled),
                observer: None,
                completion_owner: None,
                submitted: None,
            },
        )
        .await
    }

    #[cfg(all(
        target_os = "linux",
        any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
    ))]
    async fn prepare_forward_source_mode(
        &self,
        request: ForwardPlan,
        contract: zerogravity_core::accelerator::program::NumericalContract,
        options: ForwardSourceOptions<'_, '_>,
    ) -> Result<PreparedInference> {
        let cancelled = options.cancelled;
        forward_not_cancelled(cancelled)?;
        if request.numerical_profile() != contract.program_profile() {
            return Err(Error::Unsupported(
                "forward source request numerical contract differs".into(),
            ));
        }
        let call = request
            .text_invocation()
            .ok_or_else(|| Error::State("CUDA forward request call absent".into()))?
            .clone();
        let base = self.binding();
        if call.source() != base.source()
            || call.tokens().base() as u64 != base.frontier()
            || call.tokens().capacity() != self.model.capacity
            || call.tokens().vocabulary() as u64 != self.model.vocab
            || call.tokens().frontier() > self.model.capacity
        {
            return Err(Error::State(
                "CUDA forward request differs from current session root".into(),
            ));
        }

        // Validate source semantics/topology before candidate-root allocation.
        {
            let workspace = self.model.workspace.lock().await;
            if !workspace.program.is_forward_source() {
                return Err(Error::State(
                    "CUDA forward request belongs to another resident model".into(),
                ));
            }
            workspace
                .program
                .validate_forward_source_request(&request)?;
        }

        forward_not_cancelled(cancelled)?;
        let count = u64::try_from(call.tokens().len())
            .map_err(|_| Error::Limit("CUDA forward request row count".into()))?;
        let attempt = self.authority.begin_advance(count)?;
        call.require_attempt(base, &attempt)?;
        forward_not_cancelled(cancelled)?;
        let candidate = self
            .model
            .forward_candidate_state(&self.domains, call.tokens().len(), call.tokens().base())
            .await?;
        forward_not_cancelled(cancelled)?;
        let retained_call = options.completion_owner.is_some();
        let (domains, ready, readback) = self
            .model
            .execute_forward_source(base, &self.domains, candidate, &request, options)
            .await?;
        if !retained_call {
            forward_not_cancelled(cancelled)?;
        }
        Ok(PreparedInference {
            attempt,
            domains,
            derived: Vec::new(),
            ready,
            readback,
        })
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn forward_cuda(&mut self, request: ForwardPlan) -> Result<InferenceReadback> {
        let prepared = self.prepare_cuda_forward(request).await?;
        self.commit(prepared)
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn forward_cuda_cancellable(
        &mut self,
        request: ForwardPlan,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<InferenceReadback> {
        let prepared = self
            .prepare_cuda_forward_cancellable(request, cancelled)
            .await?;
        forward_not_cancelled(Some(cancelled))?;
        self.commit(prepared)
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn forward_cuda_state_only(
        &mut self,
        request: ForwardPlan,
    ) -> Result<InferenceReadback> {
        let prepared = self.prepare_cuda_forward_state_only(request).await?;
        self.commit(prepared)
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn forward_cuda_state_only_cancellable(
        &mut self,
        request: ForwardPlan,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<InferenceReadback> {
        let prepared = self
            .prepare_cuda_forward_state_only_cancellable(request, cancelled)
            .await?;
        forward_not_cancelled(Some(cancelled))?;
        self.commit(prepared)
    }

    /// Full already-Ready typed state/lease preflight shared by B1 and B>1
    /// logical commit. Must not alter state or issue a new event.
    fn validate_prepared(&self, prepared: &PreparedInference) -> Result<()> {
        self.model.service.ensure_usable()?;
        if self.binding() != prepared.attempt.base() {
            return Err(Error::State(
                "resident root changed before state publication".into(),
            ));
        }
        if prepared.ready.expected_root() != prepared.attempt.base().root()
            || prepared
                .ready
                .root_binding()
                .is_some_and(|binding| binding != prepared.attempt.base())
            || !Arc::ptr_eq(prepared.ready.service(), &self.model.service)
            || prepared.domains.len() != self.model.states.len()
            || prepared.ready.event_ids().len() != 1
        {
            return Err(Error::State(
                "foreign/incomplete resident Ready proposal".into(),
            ));
        }
        for derived in &prepared.derived {
            if derived.base() != prepared.attempt.base()
                || !Arc::ptr_eq(derived.lease().service(), &self.model.service)
                || derived.lease().role() != BufferRole::RecurrentState
                || derived.lease().bytes() != derived.layout().bytes()?
                || !derived.lease().is_initialized()
            {
                return Err(Error::State(
                    "derived resident state lacks exact-root physical ownership".into(),
                ));
            }
        }
        self.model.validate_domains(&prepared.domains, true)?;
        let mut canonical_outputs = canonical_write_outputs(&self.domains, &prepared.domains)?;
        canonical_outputs.extend(
            prepared
                .readback
                .device_token
                .iter()
                .map(|t| t.lease.clone()),
        );
        let ready_outputs = prepared.ready.outputs();
        let canonical_len = canonical_outputs.len();
        if ready_outputs.len() != canonical_len + prepared.derived.len()
            || canonical_outputs
                .iter()
                .zip(&ready_outputs[..canonical_len])
                .any(|(a, b)| {
                    a.id() != b.id()
                        || a.allocation_generation() != b.allocation_generation()
                        || !Arc::ptr_eq(a.service(), b.service())
                })
            || prepared
                .derived
                .iter()
                .zip(&ready_outputs[canonical_len..])
                .any(|(derived, output)| derived.lease().id() != output.id())
        {
            return Err(Error::State(
                "resident promoted canonical/derived output cohort mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn commit(&mut self, mut prepared: PreparedInference) -> Result<InferenceReadback> {
        self.validate_prepared(&prepared)?;
        let mut displaced = prepared.domains;
        let published = publish_domain_swap(
            &self.model.service,
            &mut self.authority,
            &mut self.domains,
            prepared.attempt,
            &mut displaced,
        )?;
        drop(displaced);
        for derived in &mut prepared.derived {
            let _ = derived.retag_after_publish(published);
        }
        if let Some(token) = prepared.readback.device_token.as_mut() {
            token.producer = Some(published);
        }
        Ok(prepared.readback)
    }
    pub async fn forward(&mut self, tokens: &[u32], diagnostic: bool) -> Result<InferenceReadback> {
        let prepared = self.prepare(tokens, diagnostic).await?;
        self.commit(prepared)
    }
    pub async fn forward_greedy(&mut self, tokens: &[u32]) -> Result<InferenceReadback> {
        let prepared = self.prepare_mode(tokens, false, Readout::Greedy).await?;
        self.commit(prepared)
    }
    /// Publish a checked prompt prefix without running an output-only tail.
    pub async fn prefill_without_logits(
        &mut self,
        tokens: &[u32],
        diagnostic: bool,
    ) -> Result<InferenceReadback> {
        let prepared = self.prepare_mode(tokens, diagnostic, Readout::None).await?;
        self.commit(prepared)
    }
    pub async fn forward_device_token(
        &mut self,
        token: &ResidentToken,
    ) -> Result<InferenceReadback> {
        let base = self.binding();
        if token.producer != Some(base) {
            return Err(Error::State("device token belongs to another root".into()));
        }
        if base.frontier() >= self.model.capacity as u64 {
            return Err(Error::Limit("resident context capacity exhausted".into()));
        }
        let attempt = self.authority.begin_advance(1)?;
        let candidate = self.model.allocate_state(false).await?;
        let (domains, ready, readback) = self
            .model
            .execute(
                base,
                base.frontier() as usize,
                &self.domains,
                candidate,
                StepInput {
                    token: 0,
                    device: Some(token),
                    readout: Readout::Greedy,
                    diagnostic: false,
                },
            )
            .await?;
        self.commit(PreparedInference {
            attempt,
            domains,
            derived: Vec::new(),
            ready,
            readback,
        })
    }
    fn validate_snapshot(&self, snapshot: &ResidentSnapshot) -> Result<()> {
        if snapshot.service_generation != self.model.service.generation()
            || snapshot.capacity != self.model.capacity
            || snapshot.binding.source() != self.model.service.source_identity()
            || snapshot.binding.frontier() > self.model.capacity as u64
        {
            return Err(Error::State(
                "foreign resident snapshot source/service/capacity".into(),
            ));
        }
        self.model.service.ensure_usable()?;
        self.model.validate_domains(&snapshot.domains, true)?;
        for domain in &snapshot.domains {
            if matches!(domain.buffers, StateBuffers::PagedKv)
                && domain.paged_kv()?.tokens()? as u64 != snapshot.binding.frontier()
            {
                return Err(Error::State(
                    "resident snapshot paged domain frontier differs".into(),
                ));
            }
        }
        Ok(())
    }

    fn install_snapshot(
        &mut self,
        snapshot: &ResidentSnapshot,
        attempt: RootAttempt,
    ) -> Result<()> {
        let mut displaced = snapshot.domains.clone();
        let _ = publish_domain_swap(
            &self.model.service,
            &mut self.authority,
            &mut self.domains,
            attempt,
            &mut displaced,
        )?;
        self.derived.invalidate();
        drop(displaced);
        Ok(())
    }

    pub fn restore(&mut self, snapshot: &ResidentSnapshot) -> Result<()> {
        self.validate_snapshot(snapshot)?;
        let attempt = self.authority.begin_restore(snapshot.binding)?;
        self.install_snapshot(snapshot, attempt)
    }
    /// Seed a fresh session from an immutable complete-domain snapshot while
    /// preserving the destination session as the sole root authority.
    pub fn fork_shared_prefix(&mut self, snapshot: &ResidentSnapshot) -> Result<()> {
        let current = self.binding();
        if current.root() != 0
            || current.session_epoch() != 0
            || current.frontier() != 0
            || snapshot.binding.frontier() == 0
        {
            return Err(Error::State(
                "resident shared prefix destination/snapshot differs".into(),
            ));
        }
        self.validate_snapshot(snapshot)?;
        let attempt = self
            .authority
            .begin_validated_replacement(snapshot.binding.frontier())?;
        self.install_snapshot(snapshot, attempt)
    }
    pub async fn reset(&mut self) -> Result<()> {
        self.reset_retaining(|_| Ok(())).await
    }
    async fn reset_retaining<F>(&mut self, retain: F) -> Result<()>
    where
        F: FnOnce(&mut crate::resident_leases::ResidentInitialization<'_>) -> Result<()>,
    {
        let attempt = self.authority.begin_reset()?;
        let mut displaced = self.model.allocate_state_with(true, retain).await?;
        self.model.service.publish_if_usable(|| {
            let transition = self.authority.prepare_publish(attempt)?;
            std::mem::swap(&mut self.domains, &mut displaced);
            transition.commit();
            Ok(())
        })?;
        self.derived.invalidate();
        drop(displaced);
        Ok(())
    }
}

#[cfg(feature = "qualification")]
#[path = "resident_inference/qualification.rs"]
mod qualification;
#[cfg(feature = "qualification")]
use qualification::{compare_domains, exact};

/// Cold token-consuming profile geometry. No device/client/admission is touched.
fn validate_profile_plan(plan: &ForwardPlan) -> Result<(usize, u64)> {
    let id = plan
        .logits()
        .ok_or_else(|| Error::Unsupported("resident profile requires logits".into()))?;
    let vocab = profile_vocabulary(plan.values().get(id.0 as usize).copied())?;
    let mut token_embed = false;
    for node in plan.nodes() {
        if let PlanOperation::Embed { tensor, token } = &node.operation
            && token.0 == ParameterSlot::Token
        {
            token_embed = true;
            if tensor.rows != vocab {
                return Err(Error::State(
                    "resident token embedding and logits vocabulary differ".into(),
                ));
            }
        }
    }
    if !token_embed {
        return Err(Error::Unsupported(
            "resident token profile requires a token embedding".into(),
        ));
    }
    Ok((plan.prefix_capacity(), vocab as u64))
}

fn profile_vocabulary(shape: Option<ValueShape>) -> Result<usize> {
    match shape {
        Some(ValueShape::Vector(vocab)) if vocab > 0 && u32::try_from(vocab).is_ok() => Ok(vocab),
        _ => Err(Error::Unsupported(
            "resident logits require a nonempty u32 token vector".into(),
        )),
    }
}

fn preflight_profile(
    source: &dyn TensorSource,
    plan: &ForwardPlan,
) -> Result<(usize, u64, ResidentPreflight)> {
    let (capacity, vocab) = validate_profile_plan(plan)?;
    let budget = ResidentProgram::preflight(source, plan)?;
    Ok((capacity, vocab, budget))
}

#[cfg(test)]
#[path = "resident_inference/cold_tests.rs"]
mod cold_tests;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    SourceDomainReadback, SourceModelFault, SourceModelFaultOutcome, SourceModelInputs,
    SourceModelReadback, SourceModelReceipt, SourceModelScopes, qualify_source_model,
    qualify_source_model_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) mod text_positions;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    AutoregressiveFault, AutoregressiveFaultExecution, AutoregressiveFaultOutcome,
    AutoregressiveReceipt, AutoregressiveReuseFault, AutoregressiveReuseFaultExecution,
    AutoregressiveReuseFaultOutcome, AutoregressiveReuseReceipt, AutoregressiveReuseRound,
    qualify_autoregressive_fault, qualify_autoregressive_reuse_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    AutoregressiveCommit, AutoregressiveFeedback, AutoregressiveObserver,
    AutoregressiveReuseStorage, AutoregressiveSlot, AutoregressiveStatePool, AutoregressiveStep,
    ColdModelSlots, CommittedBf16Feedback, PreparedAutoregressiveCold, PreparedAutoregressiveReuse,
    prepare_autoregressive_cold, prepare_autoregressive_reuse,
    qualify_autoregressive_reuse_pressure,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    Qwen2BGdnComponentFault, Qwen2BGdnComponentFaultOutcome, Qwen2BGdnComponentReadback,
    Qwen2BGdnComponentReceipt, qualify_qwen2b_gdn_component,
    qualify_qwen2b_gdn_component_arithmetic, qualify_qwen2b_gdn_component_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    FixedPromptGdnComponentFault, FixedPromptGdnComponentFaultOutcome,
    FixedPromptGdnComponentReadback, FixedPromptGdnComponentReceipt,
    qualify_fixed_prompt_gdn_component, qualify_fixed_prompt_gdn_component_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    PostAttentionArithmetic, PostAttentionComponentFault, PostAttentionComponentFaultOutcome,
    PostAttentionComponentReadback, PostAttentionComponentReceipt,
    qualify_post_attention_component, qualify_post_attention_component_fault,
    qualify_qwen2b_post_attention_component_arithmetic,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    MlpComponentFault, MlpComponentFaultOutcome, MlpComponentReadback, MlpComponentReceipt,
    qualify_mlp_component, qualify_mlp_component_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    FixedPromptModelPrefixFault, FixedPromptModelPrefixFaultOutcome,
    FixedPromptModelPrefixReadback, FixedPromptModelPrefixReceipt,
    FixedPromptQkKvQualificationInputs, qualify_fixed_prompt_model_fa_prefix,
    qualify_fixed_prompt_model_fa_prefix_fault, qualify_fixed_prompt_model_kv_prefix,
    qualify_fixed_prompt_model_kv_prefix_fault, qualify_fixed_prompt_model_layer3_prefix,
    qualify_fixed_prompt_model_layer3_prefix_fault, qualify_fixed_prompt_model_prefix,
    qualify_fixed_prompt_model_prefix_fault, qualify_fixed_prompt_model_qkv_prefix,
    qualify_fixed_prompt_model_qkv_prefix_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    FixedPromptFullModelFault, FixedPromptFullModelFaultOutcome, FixedPromptFullModelReadback,
    FixedPromptFullModelReceipt, qualify_fixed_prompt_full_model,
    qualify_fixed_prompt_full_model_fault,
};
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    Qwen2BMlpComponentFault, Qwen2BMlpComponentFaultOutcome, Qwen2BMlpComponentReceipt,
    Qwen2BMlpComponentSource, qualify_qwen2b_mlp_component, qualify_qwen2b_mlp_component_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    FixedPromptAutoregressiveFault, FixedPromptAutoregressiveFaultOutcome,
    FixedPromptAutoregressiveReadback, FixedPromptCloseReceipt, FixedPromptEmptySnapshot,
    FixedPromptFeedbackRequest, FixedPromptInvocationReceipt, FixedPromptLifecycleProfile,
    FixedPromptLoaded, FixedPromptLoadedObservation, FixedPromptLoadedSnapshot, FixedPromptPending,
    FixedPromptSession, fixed_prompt_autoregressive_host_bound,
    prepare_fixed_prompt_autoregressive, prepare_fixed_prompt_autoregressive_with_profile,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod rms_scratch;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use rms_scratch::RmsScratchRegistration;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    TextCloseReceipt, TextCohortFault, TextCohortFaultOutcome, TextCohortFaultRetention,
    TextCohortPreparation, TextCohortReadback, TextCohortReceipt, TextCohortRowReceipt,
    TextEmptySnapshot, TextFault, TextFaultOutcome, TextFeedbackRequest, TextInvocationReceipt,
    TextLoaded, TextLoadedObservation, TextLoadedSnapshot, TextPending, TextPendingCohort,
    TextQualificationInputs, TextReadback, TextResourceInventory, TextSession, prepare_text,
    prepare_text_b4,
};

#[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
pub(crate) mod bf16_text;

#[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
pub(crate) mod bf16_text_source;
#[cfg(all(
    feature = "qwen-source-qualification",
    feature = "hip",
    target_os = "linux"
))]
pub mod source_text;

#[cfg(feature = "qualification")]
pub mod diagnostics;

#[cfg(all(test, feature = "qualification", feature = "cuda"))]
async fn load_qualified(
    device: Device,
    model: Arc<LoadedQwenModel>,
    capacity: usize,
    provider_cap: usize,
    device_cap: u64,
) -> Result<Arc<ResidentModel>> {
    let plan = model.forward_plan(capacity, true, true)?;
    ResidentModel::load(device, model, plan, provider_cap, device_cap).await
}
#[cfg(all(test, feature = "portable-checkpoint"))]
trait CheckpointFixture {
    async fn import_checkpoint(
        &mut self,
        checkpoint: &zerogravity_model_qwen35::QwenCheckpoint,
    ) -> Result<()>;
}
#[cfg(all(test, feature = "portable-checkpoint"))]
impl CheckpointFixture for ResidentSession {
    async fn import_checkpoint(
        &mut self,
        checkpoint: &zerogravity_model_qwen35::QwenCheckpoint,
    ) -> Result<()> {
        self.import_state_domains(
            checkpoint.source_identity(),
            checkpoint.position(),
            &checkpoint.domain_values(),
        )
        .await
    }
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
struct ForwardSourceOptions<'a, 'b> {
    output: ResidentOutput,
    readout: Readout,
    device_token: Option<ResidentToken>,
    cancelled: Option<&'a std::sync::atomic::AtomicBool>,
    observer: Option<&'b mut dyn crate::resident_program::text_observation::TextObserver>,
    completion_owner: Option<crate::physical_quarantine::CompletionOwner>,
    submitted: Option<&'a std::sync::atomic::AtomicBool>,
}

#[cfg(test)]
mod forward_shutdown_owner_tests {
    use super::into_unique_owner;
    use std::sync::Arc;

    #[test]
    fn shutdown_requires_unique_model_owner() {
        let owner = Arc::new(17usize);
        let sibling = owner.clone();
        assert!(into_unique_owner(owner, "CUDA forward test model").is_err());
        drop(sibling);

        let owner = Arc::new(23usize);
        assert_eq!(
            into_unique_owner(owner, "CUDA forward test model").unwrap(),
            23
        );
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    #[test]
    fn forward_cancellation_is_explicit_and_fail_closed() {
        let flag = std::sync::atomic::AtomicBool::new(false);
        assert!(super::forward_not_cancelled(Some(&flag)).is_ok());
        flag.store(true, std::sync::atomic::Ordering::Release);
        assert!(super::forward_not_cancelled(Some(&flag)).is_err());
        assert!(super::forward_not_cancelled(None).is_ok());
    }

    #[test]
    fn typed_snapshot_decode_preserves_f32_and_bf16_and_rejects_nonfinite() {
        let f32_layout =
            super::ResidentBufferLayout::new(super::ResidentPhysicalType::F32, 3).unwrap();
        let f32_raw = cubecl::bytes::Bytes::from_bytes_vec(
            [0.0f32, -1.5, 3.25]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
        );
        assert!(super::resident_plane_finite(&f32_raw, f32_layout).unwrap());
        assert_eq!(
            super::resident_plane_values(&f32_raw, f32_layout, 3).unwrap(),
            [0.0, -1.5, 3.25]
        );

        let bf16_layout =
            super::ResidentBufferLayout::new(super::ResidentPhysicalType::Bf16, 4).unwrap();
        let bf16_raw = cubecl::bytes::Bytes::from_bytes_vec(
            [0x0000u16, 0x8000, 0x3f80, 0xbf80]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect(),
        );
        assert!(super::resident_plane_finite(&bf16_raw, bf16_layout).unwrap());
        let actual = super::resident_plane_values(&bf16_raw, bf16_layout, 4).unwrap();
        assert_eq!(actual[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(actual[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(actual[2], 1.0);
        assert_eq!(actual[3], -1.0);

        let nonfinite = cubecl::bytes::Bytes::from_bytes_vec(
            [0x7f80u16, 0x0000, 0x0000, 0x0000]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect(),
        );
        assert!(!super::resident_plane_finite(&nonfinite, bf16_layout).unwrap());
    }

    #[test]
    fn typed_snapshot_kv_tail_is_raw_zero_and_active_extent_is_element_based() {
        let layout =
            super::ResidentBufferLayout::new(super::ResidentPhysicalType::Bf16, 8).unwrap();
        let mut raw = vec![0u8; 16];
        for (index, bits) in [0x3f80u16, 0x4000, 0x4040, 0x4080].into_iter().enumerate() {
            raw[index * 2..index * 2 + 2].copy_from_slice(&bits.to_ne_bytes());
        }
        let bytes = cubecl::bytes::Bytes::from_bytes_vec(raw.clone());
        assert!(super::resident_plane_tail_zero(&bytes, layout, 4).unwrap());
        assert_eq!(
            super::resident_plane_values(&bytes, layout, 4)
                .unwrap()
                .len(),
            4
        );
        raw[14] = 1;
        let dirty = cubecl::bytes::Bytes::from_bytes_vec(raw);
        assert!(!super::resident_plane_tail_zero(&dirty, layout, 4).unwrap());
    }
}

#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
mod resident_text;
#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
pub use resident_text::CudaSourceStatePolicy;
#[cfg(all(
    feature = "cuda-cublas",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use resident_text::ResidentTextPagedPoolSnapshot;
#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
pub use resident_text::{
    ResidentTextAdmission, ResidentTextModel, ResidentTextPending, ResidentTextPreparation,
    ResidentTextPublication, ResidentTextReadback, ResidentTextSession, ResidentTextSnapshot,
    ResidentTextStateIdentity,
};
#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip"))
))]
pub use resident_text::{ResidentTextCohortPending, ResidentTextCohortPreparation};

#[cfg(all(
    feature = "hip-source",
    feature = "hip",
    feature = "hardware-abort-probe",
    feature = "qwen-source-qualification",
    target_os = "linux"
))]
pub use resident_text::{ResidentTextFinalReader, ResidentTextReaderObservation};

#[cfg(all(
    feature = "hip-source",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub(crate) mod source_native_probe;

#[cfg(all(feature = "hip-source", feature = "hip", target_os = "linux"))]
pub use resident_text::ResidentTextInventory;
