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
#[cfg(all(feature = "qualification", feature = "cuda"))]
mod graph;

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
use crate::resident_leases::{
    ResidentLease, ResidentPhysical, ResidentReady, ResidentService, ResidentSource,
};
#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
use crate::resident_program::ordinary_cuda;
use crate::resident_program::{
    AllocationInit, BatchProgram, DeviceBuffer, ResidentBufferLayout, ResidentBufferRole,
    ResidentOutput, ResidentPhysicalType, ResidentPreflight, ResidentProgram, RuntimeParameters,
    StateBinding, StateBuffers, StateLayout, StateRequirement,
};
use async_lock::Mutex;
use cubecl::{Device, bytes::Bytes, server::Handle};
#[cfg(all(test, feature = "qualification"))]
use std::path::Path;
use std::sync::Arc;
#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::program::{
    ForwardPlan, ParameterSlot, PlanObservation as LayerObservation, ValueShape,
};
use zerogravity_core::accelerator::program::{PlanOperation, StateSlot};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};
use zerogravity_core::session_root::{RootAttempt, RootBinding, SessionAuthority};
use zerogravity_core::source::TensorSource;
#[cfg(all(test, feature = "qualification"))]
use zerogravity_model_qwen35::LoadedQwenModel;

mod state_pool;
struct Workspace {
    greedy: Option<[Handle; 2]>,
    batch: Option<BatchWorkspace>,
    program: ResidentProgram,
    // Raw aliases/programs drop before final leases trigger source cleanup.
    leases: Vec<(ResidentLease, DeviceAccess)>,
    state_accesses: Vec<(ResidentLease, DeviceAccess)>,
    logit_accesses: Vec<(ResidentLease, DeviceAccess)>,
}
struct BatchWorkspace {
    program: BatchProgram,
    leases: Vec<(ResidentLease, DeviceAccess)>,
    rows: usize,
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
    leases: Vec<ResidentLease>,
    generations: Vec<u64>,
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
    service.publish_if_usable(|| {
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
        for ((state, prepared), guard) in
            states.iter_mut().zip(prepared.iter_mut()).zip(transitions)
        {
            std::mem::swap(*state, &mut prepared.domains);
            let root = guard.commit();
            if let Some(token) = prepared.readback.device_token.as_mut() {
                token.producer = Some(root);
            }
        }
        Ok(())
    })?;
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

impl ResidentModel {
    pub async fn load(
        device: Device,
        model: Arc<dyn TensorSource>,
        plan: ForwardPlan,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<Arc<Self>> {
        Self::load_inner(device, model, plan, provider_cap, device_cap, false).await
    }

    async fn load_inner(
        device: Device,
        model: Arc<dyn TensorSource>,
        plan: ForwardPlan,
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
    pub async fn load_ordinary_cuda(
        device: Device,
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        providers: crate::OrdinaryCudaProviderPaths,
        provider_cap: usize,
        device_cap: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        validate_provider(&device)?;
        if !matches!(device, Device::Cuda(_)) {
            return Err(Error::Unsupported(
                "ordinary CUDA resident model requires CUDA".into(),
            ));
        }

        // Pure source-plan/resource analysis. This MUST precede physical owner
        // creation, source attachment, cleanup, queue work, or allocation.
        let admission = ordinary_cuda::OrdinaryCudaAdmission::analyze(plan)?;
        let total = admission.total_bytes()?;
        if total > provider_cap {
            return Err(Error::Limit(format!(
                "ordinary CUDA exact cold plus two-root plan does not fit provider cap: {total} > {provider_cap}"
            )));
        }
        let source_identity = admission.source();
        if source.source_identity() != source_identity {
            return Err(Error::State(
                "ordinary CUDA verified source differs from plan".into(),
            ));
        }
        let capacity = admission.capacity();
        let vocab = admission.vocabulary() as u64;
        let expected_preflight = admission.preflight().clone();

        let physical = ResidentPhysical::new_pinned_cuda(device, provider_cap, device_cap)?;
        let owner = ResourceOwner {
            model: format!("ordinary-qwen-source-{source_identity:?}"),
            session: "resident-program".into(),
        };
        let service =
            physical.attach_model(ResidentSource::Windows(source.clone()), owner, provider_cap)?;
        let prepared = ordinary_cuda::prepare_program(
            service.clone(),
            admission,
            source,
            &providers,
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
                            Error::Limit("ordinary CUDA owned access byte sum".into())
                        })
                    })?
            || program.budget().prior_state_bytes != expected_preflight.prior_state_bytes
            || program.budget().candidate_state_bytes != expected_preflight.candidate_state_bytes
        {
            return Err(Error::State(
                "ordinary CUDA bound resident model/preflight differs".into(),
            ));
        }
        let states = program.budget().states.clone();
        let state_bytes = program.budget().prior_state_bytes;

        Ok(Arc::new(Self {
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
                greedy: None,
                leases: prepared.owned_accesses,
                state_accesses: prepared.state_accesses,
                logit_accesses: prepared.logit_accesses,
                batch: None,
            }),
            allocation: Mutex::new(()),
        }))
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn shutdown_ordinary_cuda(self: Arc<Self>) -> Result<()> {
        let model = into_unique_owner(self, "ordinary CUDA resident model")?;
        let service = model.service.clone();
        {
            let workspace = model.workspace.lock().await;
            if !workspace.program.is_ordinary_cuda() {
                return Err(Error::State(
                    "ordinary CUDA shutdown requested from another resident model".into(),
                ));
            }
            workspace.program.retire_ordinary_cuda().await?;
        }

        // No sessions can remain after the unique-owner check. Dropping the
        // model retires all cold/value/state-pool leases while this service Arc
        // keeps the original namespace available for exact ledger validation.
        drop(model);
        service
            .client()
            .memory_cleanup()
            .map_err(|error| Error::State(format!("ordinary CUDA shutdown cleanup: {error}")))?;
        if service.resident_bytes()? != 0 || service.model_events()? != 0 {
            return Err(Error::State(
                "ordinary CUDA shutdown left resident bytes or events".into(),
            ));
        }
        service.close_model_namespace()
    }

    pub fn physical_owner(&self) -> Arc<ResidentPhysical> {
        self.service.physical_owner().clone()
    }

    #[allow(clippy::too_many_arguments)]
    async fn load_attached(
        device: Device,
        model: Arc<dyn TensorSource>,
        plan: ForwardPlan,
        provider_cap: usize,
        device_cap: u64,
        pinned_graph: bool,
        physical: Option<Arc<ResidentPhysical>>,
    ) -> Result<Arc<Self>> {
        validate_provider(&device)?;
        let native_batch = matches!(device, Device::Cuda(_));
        let (capacity, vocab, preflight) = preflight_profile(model.as_ref(), &plan)?;
        if preflight.source != model.source_identity() {
            return Err(Error::State("resident preflight source mismatch".into()));
        }
        let greedy_sizes = crate::resident_greedy::workspace_bytes(vocab as usize)?;
        let greedy_bytes = greedy_sizes.iter().sum::<usize>();
        // Require source/workspace and two private states before any GPU allocation.
        if preflight
            .total_with_state()?
            .checked_add(greedy_bytes + 12)
            .is_none_or(|n| n > provider_cap)
        {
            return Err(Error::Limit(
                "complete resident graph does not fit provider cap".into(),
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
                leases,
                state_accesses,
                logit_accesses,
                batch: None,
            }),
            allocation: Mutex::new(()),
        }))
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.service
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
        Ok(ResidentSession {
            model: self.clone(),
            authority,
            domains,
        })
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
            #[cfg(all(feature = "qualification", target_os = "linux"))]
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
            return state_pool::allocate_pooled_state(self, &storage.pool, zero, None, |init| {
                #[cfg(all(feature = "qualification", target_os = "linux"))]
                if let Some(host) = host {
                    init.retain_completion_owner(host)?;
                }
                retain(init)
            })
            .await;
        }
        let _admission = self.allocation.lock().await;
        self.service.ensure_usable()?;
        if self.state_bytes > self.service.remaining_capacity()? {
            return Err(Error::Limit("resident state admission exhausted".into()));
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
            let leases = storage.pool.checkout()?;
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
        if self.state_bytes > self.service.remaining_capacity()? {
            return Err(Error::Limit("resident state admission exhausted".into()));
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
            // Two immutable semantic phase records share this existing serialized
            // workspace/queue. No new actor or family traversal is selected here.
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
                &DeviceBuffer::new(token.lease.handle(), 1),
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
        if self
            .abort_after_launch
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |remaining| remaining.checked_sub(1),
            )
            .is_ok_and(|remaining| remaining == 1)
        {
            self.launch_seen
                .store(flight.handles().len(), std::sync::atomic::Ordering::Release);
            // Owned probes may resume by clearing the observed launch count.
            // Terminal unknown probes leave it set and drop this actual flight.
            std::future::poll_fn(|_| {
                if self.launch_seen.load(std::sync::atomic::Ordering::Acquire) == 0 {
                    std::task::Poll::Ready(())
                } else {
                    std::task::Poll::Pending
                }
            })
            .await;
        }
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

#[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
impl ResidentModel {
    async fn execute_ordinary_cuda(
        &self,
        base: RootBinding,
        prior: &[Domain],
        candidate: Vec<Domain>,
        request: &ForwardPlan,
        output: ResidentOutput,
    ) -> Result<(Vec<Domain>, ResidentReady, InferenceReadback)> {
        self.service.ensure_usable()?;
        let call = request
            .source_call()
            .ok_or_else(|| Error::State("ordinary CUDA materialized call absent".into()))?;
        if base.source() != self.service.source_identity()
            || call.source() != base.source()
            || call.rows().base() as u64 != base.frontier()
            || call.rows().capacity() != self.capacity
            || call.rows().vocabulary() as u64 != self.vocab
            || call.rows().frontier() > self.capacity
        {
            return Err(Error::State(
                "ordinary CUDA session root/call geometry differs".into(),
            ));
        }
        self.validate_domains(prior, true)?;
        self.validate_domains(&candidate, false)?;
        let bindings = bind_domains(prior, &candidate, call.rows().base())?;

        let ticket = self.service.queue().submit(
            self.service.budget(),
            ResourceOwner {
                model: self.service.owner().model.clone(),
                session: format!("ordinary-session-{}", base.session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: self.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        let workspace = self.workspace.lock().await;
        if !workspace.program.is_ordinary_cuda() {
            return Err(Error::State(
                "ordinary CUDA request reached another resident model".into(),
            ));
        }
        workspace.program.validate_ordinary_cuda_request(request)?;

        let mut accesses = match output {
            ResidentOutput::StateOnly => workspace.state_accesses.clone(),
            ResidentOutput::Logits => workspace.logit_accesses.clone(),
        };
        for domain in prior {
            accesses.extend(
                domain
                    .leases
                    .iter()
                    .cloned()
                    .map(|lease| (lease, DeviceAccess::Read)),
            );
        }
        for domain in &candidate {
            accesses.extend(
                domain
                    .leases
                    .iter()
                    .cloned()
                    .map(|lease| (lease, DeviceAccess::Write)),
            );
        }
        slot.begin_submission()?;
        let mut flight = self.service.submit(base.root(), &accesses)?;
        let submitted =
            workspace
                .program
                .launch_ordinary_cuda(&mut flight, request, &bindings, output)?;

        let expected = workspace.program.plan().proposed_domains();
        if submitted.proposals.len() != expected.len() {
            return Err(Error::State(
                "ordinary CUDA incomplete state effect receipt".into(),
            ));
        }
        for ((version, buffers), expected) in submitted.proposals.iter().zip(expected) {
            let domain = candidate
                .iter()
                .find(|domain| domain.slot == version.slot)
                .ok_or_else(|| Error::State("ordinary CUDA foreign state proposal".into()))?;
            let actual = buffers.handles();
            let target = domain.buffers.handles();
            if version.slot != expected.slot
                || version.ordinal != expected.ordinal
                || actual.len() != target.len()
                || actual.iter().zip(target).any(|(a, b)| !same_handle(a, &b))
            {
                return Err(Error::State(
                    "ordinary CUDA state effect version/allocation mismatch".into(),
                ));
            }
        }

        let status_lease = lease_for(&accesses, &submitted.status)?;
        let mut cohort = vec![status_lease.readback(0, 4)?];
        if output == ResidentOutput::Logits {
            let logits = submitted
                .logits
                .as_ref()
                .ok_or_else(|| Error::State("ordinary CUDA logits receipt absent".into()))?;
            let lease = lease_for(&accesses, &logits.handle)?;
            cohort.push(logits.readback(
                lease,
                ResidentBufferLayout::new(ResidentPhysicalType::F32, logits.elements)?,
            )?);
        } else if submitted.logits.is_some() {
            return Err(Error::State(
                "ordinary CUDA state-only execution returned logits".into(),
            ));
        }

        let promoted = candidate
            .iter()
            .flat_map(|domain| domain.leases.iter().cloned())
            .collect::<Vec<_>>();
        let expected_readback_bytes = cohort.iter().map(|item| item.bytes()).sum();
        let (ready, read) = flight
            .complete_with_readback(&promoted, &cohort, |values| {
                if !all_zero(&values[0]) {
                    return Err(Error::State(
                        "ordinary CUDA shared status rejected proposed graph".into(),
                    ));
                }
                if output == ResidentOutput::Logits
                    && !words(&values[1])?.iter().all(|value| value.is_finite())
                {
                    return Err(Error::State("ordinary CUDA nonfinite F32 logits".into()));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;

        Ok((
            candidate,
            ready,
            InferenceReadback {
                logits: if output == ResidentOutput::Logits {
                    words(&read[1])?
                } else {
                    Vec::new()
                },
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
            })
        })
        .collect()
}
impl ResidentSnapshot {
    pub async fn read_domains(&self) -> Result<Vec<(StateSlot, Vec<f32>)>> {
        let service = self
            .domains
            .first()
            .and_then(|d| d.leases.first())
            .ok_or_else(|| Error::State("empty resident snapshot".into()))?
            .service()
            .clone();
        service.ensure_usable()?;
        if service.generation() != self.service_generation
            || service.source_identity() != self.binding.source()
        {
            return Err(Error::State("snapshot service generation mismatch".into()));
        }
        let leases = self
            .domains
            .iter()
            .flat_map(|d| d.leases.iter().cloned())
            .collect::<Vec<_>>();
        let accesses = leases
            .iter()
            .cloned()
            .map(|l| (l, DeviceAccess::Read))
            .collect::<Vec<_>>();
        let cohort = leases
            .iter()
            .map(|l| l.readback(0, l.bytes()))
            .collect::<Result<Vec<_>>>()?;
        let ticket = service.queue().submit(
            service.budget(),
            ResourceOwner {
                model: service.owner().model.clone(),
                session: format!("snapshot-{}", self.binding.session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: leases.iter().map(|l| l.bytes() as u64).sum(),
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let flight = service.submit(self.binding.root(), &accesses)?;
        let read = flight
            .complete_read_only_readback(&cohort, |data| {
                for b in data {
                    if !words(b)?.iter().all(|v| v.is_finite()) {
                        return Err(Error::State("nonfinite committed resident snapshot".into()));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        let mut cursor = 0;
        let mut output = Vec::new();
        for d in &self.domains {
            let mut flat = Vec::new();
            for l in &d.leases {
                let values = words(&read[cursor])?;
                if matches!(&d.buffers, StateBuffers::Kv { .. }) {
                    let active = l.bytes() / 4 / self.capacity * self.binding.frontier() as usize;
                    if values[active..].iter().any(|x| x.to_bits() != 0) {
                        return Err(Error::State("snapshot noncanonical KV tail".into()));
                    }
                    flat.extend_from_slice(&values[..active]);
                } else {
                    flat.extend(values);
                }
                cursor += 1;
            }
            output.push((d.slot, flat));
        }
        Ok(output)
    }
}

impl ResidentSession {
    pub fn binding(&self) -> RootBinding {
        self.authority.binding()
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
        let (_bf16_text_source_call_admission, bf16_text_source_call) =
            if let Some(options) = &bf16_text_source {
                bf16_text_source::check_cancel(options.cancelled)?;
                if tokens.len() != 1 || base.frontier() > 1 {
                    return Err(Error::Unsupported("finite source teacher row".into()));
                }
                let body = &options.bodies[base.frontier() as usize];
                if body.recorded().plan().source() != base.source() {
                    return Err(Error::State("source call owner differs".into()));
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
                                call: bf16_text_source_call.as_ref().unwrap(),
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
            ready,
            readback,
        })
    }
    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn prepare_ordinary_cuda(&self, request: ForwardPlan) -> Result<PreparedInference> {
        self.prepare_ordinary_cuda_mode(request, ResidentOutput::Logits)
            .await
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn prepare_ordinary_cuda_state_only(
        &self,
        request: ForwardPlan,
    ) -> Result<PreparedInference> {
        self.prepare_ordinary_cuda_mode(request, ResidentOutput::StateOnly)
            .await
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    async fn prepare_ordinary_cuda_mode(
        &self,
        request: ForwardPlan,
        output: ResidentOutput,
    ) -> Result<PreparedInference> {
        let call = request
            .source_call()
            .ok_or_else(|| Error::State("ordinary CUDA request call absent".into()))?
            .clone();
        let base = self.binding();
        if call.source() != base.source()
            || call.rows().base() as u64 != base.frontier()
            || call.rows().capacity() != self.model.capacity
            || call.rows().vocabulary() as u64 != self.model.vocab
            || call.rows().frontier() > self.model.capacity
        {
            return Err(Error::State(
                "ordinary CUDA request differs from current session root".into(),
            ));
        }

        // Validate source semantics/topology before candidate-root allocation.
        {
            let workspace = self.model.workspace.lock().await;
            if !workspace.program.is_ordinary_cuda() {
                return Err(Error::State(
                    "ordinary CUDA request belongs to another resident model".into(),
                ));
            }
            workspace.program.validate_ordinary_cuda_request(&request)?;
        }

        let count = u64::try_from(call.rows().len())
            .map_err(|_| Error::Limit("ordinary CUDA request row count".into()))?;
        let attempt = self.authority.begin_advance(count)?;
        let candidate = self.model.allocate_state(false).await?;
        let (domains, ready, readback) = self
            .model
            .execute_ordinary_cuda(base, &self.domains, candidate, &request, output)
            .await?;
        Ok(PreparedInference {
            attempt,
            domains,
            ready,
            readback,
        })
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn forward_ordinary_cuda(
        &mut self,
        request: ForwardPlan,
    ) -> Result<InferenceReadback> {
        let prepared = self.prepare_ordinary_cuda(request).await?;
        self.commit(prepared)
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    pub async fn forward_ordinary_cuda_state_only(
        &mut self,
        request: ForwardPlan,
    ) -> Result<InferenceReadback> {
        let prepared = self.prepare_ordinary_cuda_state_only(request).await?;
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
        self.model.validate_domains(&prepared.domains, true)?;
        let mut outputs = prepared
            .domains
            .iter()
            .flat_map(|d| d.leases.iter())
            .collect::<Vec<_>>();
        outputs.extend(prepared.readback.device_token.iter().map(|t| &t.lease));
        if outputs.len() != prepared.ready.outputs().len()
            || outputs
                .iter()
                .zip(prepared.ready.outputs())
                .any(|(a, b)| a.id() != b.id())
        {
            return Err(Error::State(
                "resident promoted domain cohort mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn commit(&mut self, mut prepared: PreparedInference) -> Result<InferenceReadback> {
        self.validate_prepared(&prepared)?;
        let mut displaced = prepared.domains;
        let published = self.model.service.publish_if_usable(|| {
            let transition = self.authority.prepare_publish(prepared.attempt)?;
            std::mem::swap(&mut self.domains, &mut displaced);
            Ok(transition.commit())
        })?;
        drop(displaced);
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
            ready,
            readback,
        })
    }
    pub fn restore(&mut self, snapshot: &ResidentSnapshot) -> Result<()> {
        if snapshot.service_generation != self.model.service.generation()
            || snapshot.domains.len() != self.model.states.len()
        {
            return Err(Error::State(
                "foreign resident snapshot service/domains".into(),
            ));
        }
        self.model.service.ensure_usable()?;
        self.model.validate_domains(&snapshot.domains, true)?;
        let attempt = self.authority.begin_restore(snapshot.binding)?;
        let mut displaced = snapshot.domains.clone();
        self.model.service.publish_if_usable(|| {
            let transition = self.authority.prepare_publish(attempt)?;
            std::mem::swap(&mut self.domains, &mut displaced);
            transition.commit();
            Ok(())
        })?;
        drop(displaced);
        Ok(())
    }
    pub async fn reset(&mut self) -> Result<()> {
        let attempt = self.authority.begin_reset()?;
        let mut displaced = self.model.allocate_state(true).await?;
        self.model.service.publish_if_usable(|| {
            let transition = self.authority.prepare_publish(attempt)?;
            std::mem::swap(&mut self.domains, &mut displaced);
            transition.commit();
            Ok(())
        })?;
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
    NaturalFault, NaturalFaultExecution, NaturalFaultOutcome, NaturalReceipt, NaturalReuseFault,
    NaturalReuseFaultExecution, NaturalReuseFaultOutcome, NaturalReuseReceipt, NaturalReuseRound,
    qualify_natural_fault, qualify_natural_reuse_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    ColdModelSlots, CommittedBf16Feedback, NaturalCommitted, NaturalFeedbackRequest,
    NaturalObserver, NaturalSlot, NaturalStatePool, NaturalStep, PreparedNaturalCold,
    PreparedReusableNatural, ReusableNaturalStorage, prepare_natural_cold,
    prepare_reusable_natural_cold, qualify_natural_reuse_pressure,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    Qwen2bGdnComponentFault, Qwen2bGdnComponentFaultOutcome, Qwen2bGdnComponentReadback,
    Qwen2bGdnComponentReceipt, qualify_qwen2b_gdn_component,
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
    Qwen2bMlpComponentFault, Qwen2bMlpComponentFaultOutcome, Qwen2bMlpComponentReceipt,
    Qwen2bMlpComponentSource, qualify_qwen2b_mlp_component, qualify_qwen2b_mlp_component_fault,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    FixedPromptCallReceipt, FixedPromptCloseReceipt, FixedPromptEmptySnapshot,
    FixedPromptFeedbackRequest, FixedPromptLifecycleProfile, FixedPromptLoaded,
    FixedPromptLoadedObservation, FixedPromptLoadedSnapshot, FixedPromptNaturalFault,
    FixedPromptNaturalFaultOutcome, FixedPromptNaturalReadback, FixedPromptPending,
    FixedPromptSession, fixed_prompt_natural_host_bound, prepare_fixed_prompt_natural,
    prepare_fixed_prompt_natural_with_profile,
};

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod rms_scratch;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use rms_scratch::RmsScratchRegistration;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use native_fragment::{
    TextCallReceipt, TextCloseReceipt, TextCohortFault, TextCohortFaultOutcome,
    TextCohortFaultRetention, TextCohortPreparation, TextCohortReadback, TextCohortReceipt,
    TextCohortRowReceipt, TextEmptySnapshot, TextFault, TextFaultOutcome, TextFeedbackRequest,
    TextLoaded, TextLoadedObservation, TextLoadedSnapshot, TextPending, TextPendingCohort,
    TextQualificationInputs, TextReadback, TextResourceInventory, TextSession, prepare_text,
    prepare_text_b4,
};

#[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
pub(crate) mod bf16_text;

#[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
pub(crate) mod bf16_text_source;

#[cfg(feature = "qualification")]
pub mod diagnostics;

#[cfg(all(test, feature = "qualification"))]
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

#[cfg(test)]
mod ordinary_shutdown_owner_tests {
    use super::into_unique_owner;
    use std::sync::Arc;

    #[test]
    fn shutdown_requires_unique_model_owner() {
        let owner = Arc::new(17usize);
        let sibling = owner.clone();
        assert!(into_unique_owner(owner, "ordinary CUDA test model").is_err());
        drop(sibling);

        let owner = Arc::new(23usize);
        assert_eq!(
            into_unique_owner(owner, "ordinary CUDA test model").unwrap(),
            23
        );
    }
}
