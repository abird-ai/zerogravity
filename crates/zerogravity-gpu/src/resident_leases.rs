//! Persistent physical leases. Ready retires an event, never its live outputs.
//! The engine/binder owns session epochs, immutable roots and fair worker slots.
//! Every raw handle must stay paired with its owning lease while it is used.
use crate::physical_queue::{PhysicalQueue, queue_for};
use crate::physical_service::{PhysicalServiceBudget, PhysicalWork};
use cubecl::{Device, bytes::Bytes, client::Client, server::Handle};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::device_events::{
    BufferLeaseId, BufferRole, DeviceAccess, DeviceEventId, DeviceEventLedger,
};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceOwner};
use zerogravity_core::session_root::RootBinding;

pub(crate) mod backing_pool;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod enqueue;
#[cfg(all(
    any(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    ),
    feature = "hardware-abort-probe"
))]
pub(crate) mod final_owner_probe;
mod foreign_provider;
#[cfg(any(
    test,
    feature = "cuda-native-qualification",
    all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    )
))]
mod native_cache;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use enqueue::{
    ENQUEUE_PROFILE_ADMISSION_BYTES, EnqueuePlacement, enqueue_profile_enabled,
    profile_head_enqueue,
};
#[cfg(all(
    target_os = "linux",
    any(feature = "hip", feature = "vulkan"),
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification",
        all(test, feature = "qualification", feature = "hip")
    )
))]
pub(crate) use native_cache::NativeCacheBudget;
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    target_os = "linux"
))]
pub(crate) mod frame_checkpoint;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod model_upload;
#[cfg(all(feature = "qualification", target_os = "linux"))]
pub(crate) mod resident_matrix;
#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    target_os = "linux"
))]
pub(crate) mod source_page;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod source_upload;
#[cfg(any(
    feature = "cuda-native-qualification",
    all(
        target_os = "linux",
        feature = "hip",
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification")
        )
    )
))]
pub(crate) use foreign_provider::ForeignProviderCharge;
use foreign_provider::ResidentLedger;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use model_upload::{
    ModelWeight, PendingModelWeight, Qwen2bMlpComputeReadCut, Qwen2bMlpSourceVerification,
    Qwen2bMlpWeightCohort, Qwen2bMlpWeights,
};
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub use model_upload::{Qwen2bMlpWeightMode, Qwen2bMlpWeightOutcome, qualify_qwen2b_mlp_weights};
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use resident_matrix::ResidentMatrix;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use source_upload::{
    TIED_BYTES, TIED_ROW_BYTES, TIED_WINDOW_BYTES, TIED_WINDOWS, TiedUploadAttempt, TiedUploadCut,
    TiedWeight,
};

static NEXT_SERVICE_GENERATION: AtomicU64 = AtomicU64::new(1);
#[cfg(all(
    target_os = "linux",
    any(feature = "hip", feature = "vulkan"),
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification",
        all(test, feature = "qualification", feature = "hip")
    )
))]
static NATIVE_CONTEXT_CONSTRUCTED: AtomicBool = AtomicBool::new(false);

fn access_positions(
    ids: impl IntoIterator<Item = BufferLeaseId>,
) -> Result<BTreeMap<BufferLeaseId, usize>> {
    let mut positions = BTreeMap::new();
    for (index, id) in ids.into_iter().enumerate() {
        if positions.insert(id, index).is_some() {
            return Err(Error::State("duplicate resident access identity".into()));
        }
    }
    Ok(positions)
}

fn issue(counter: &AtomicU64) -> Result<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::Limit("resident generation exhausted".into()))
}

mod physical;
use physical::ModelAdmission;
pub use physical::ResidentPhysical;
pub(crate) use physical::ResidentSource;
/// Source-specific model admission over one original physical owner.
pub(crate) struct ResidentService {
    physical: Arc<ResidentPhysical>,
    // Source view of the SAME actor; internal field accesses never fall through
    // Deref to the neutral physical control Client.
    client: Arc<Client>,
    budget: PhysicalServiceBudget,
    admission: Arc<ModelAdmission>,
    source: SourceIdentity,
    owner: ResourceOwner,
    generation: u64,
    uploaded_value_bytes: AtomicU64,
    cap: usize,
    retained_forever: AtomicBool,
}
impl Drop for ResidentService {
    fn drop(&mut self) {
        // Final source reader only; never synchronize per hot logical lease.
        // Both original admission and physical envelope remain fields until
        // the actual backend drop queue/unused-page cleanup has completed.
        let known = self.ensure_usable().is_ok()
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.client().memory_cleanup()
            }))
            .is_ok_and(|r| r.is_ok());
        if !known {
            self.poisoned.store(true, Ordering::Release);
            self.budget.fence_flag().store(true, Ordering::Release);
            self.budget.retained_envelope().fence_unknown();
            crate::physical_quarantine::quarantine_uncertain_with_owners(
                self.client().clone(),
                Vec::new(),
                vec![self.physical.clone(), self.admission.clone()],
            );
        }
    }
}
impl std::ops::Deref for ResidentService {
    type Target = ResidentPhysical;
    fn deref(&self) -> &Self::Target {
        &self.physical
    }
}

impl ResidentService {
    #[cfg(any(test, feature = "qualification"))]
    pub fn new(
        device: Device,
        source: SourceIdentity,
        owner: ResourceOwner,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<Arc<Self>> {
        Self::validate_owner(source, &owner)?;
        ResidentPhysical::new(device, provider_cap, device_cap)?
            .attach_inner(source, owner, provider_cap, None)
            .map(Arc::new)
    }
    #[cfg(all(test, feature = "qualification", feature = "hip", target_os = "linux"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_native_context(
        device: Device,
        source: SourceIdentity,
        owner: ResourceOwner,
        provider_cap: usize,
        device_cap: u64,
        policy: Arc<str>,
        context_bytes: u64,
    ) -> Result<(Arc<Self>, Arc<crate::physical_service::NativeContextBudget>)> {
        Self::validate_owner(source, &owner)?;
        let (physical, context) = ResidentPhysical::new_with_native_context(
            device,
            provider_cap,
            device_cap,
            policy,
            context_bytes,
        )?;
        let cap = physical.available_model_capacity()?;
        Ok((
            Arc::new(physical.attach_inner(source, owner, cap, None)?),
            context,
        ))
    }

    fn validate_owner(source: SourceIdentity, owner: &ResourceOwner) -> Result<()> {
        if !source.is_bound() || owner.model.is_empty() || owner.session.is_empty() {
            return Err(Error::State(
                "resident service requires bound source/owner".into(),
            ));
        }
        Ok(())
    }

    /// Qualifier-only cold constructor. Pin before any own clone, allocation,
    /// transfer or submission. Global capture still requires process isolation.
    #[cfg(all(feature = "qualification", feature = "cuda"))]
    pub fn new_pinned_cuda(
        device: Device,
        source: SourceIdentity,
        owner: ResourceOwner,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<Arc<Self>> {
        Self::validate_owner(source, &owner)?;
        ResidentPhysical::new_pinned_cuda(device, provider_cap, device_cap)?
            .attach_inner(source, owner, provider_cap, None)
            .map(Arc::new)
    }
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    pub(crate) fn admit_cuda_compiler_context(
        &self,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    ) -> Result<Arc<crate::physical_service::CudaCompilerContextBudget>> {
        use crate::physical_service::CudaCompilerContextBudget;
        let queue = self.pinned_cuda_queue()?;
        let actor_error = |e| Error::State(format!("original CUDA compiler admission: {e}"));
        if let Some(owner) = queue
            .compiler_context_owner::<CudaCompilerContextBudget>()
            .map_err(actor_error)?
        {
            owner.require(self.budget())?;
            owner.require_host(source)?;
            return Ok(owner);
        }
        let owner = CudaCompilerContextBudget::admit(self.budget(), source)?;
        // The local Arc survives submission/result loss. A known pre-effect
        // refusal releases; adoption or an unknown actor outcome retains/fences.
        match unsafe { queue.install_compiler_context(owner.clone(), owner.adoption_flag()) } {
            Ok(()) => {
                owner.mark_adopted();
                Ok(owner)
            }
            Err(error) => {
                match queue.compiler_context_owner::<CudaCompilerContextBudget>() {
                    Ok(Some(installed)) if Arc::ptr_eq(&owner, &installed) => owner.mark_adopted(),
                    Ok(None) => {}
                    _ => owner.fence_unknown(),
                }
                Err(actor_error(error))
            }
        }
    }
    /// Read-only original actor/lease observation, also valid after model close
    /// or a physical fence. It grants no work or healthy completion authority.
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    pub(crate) fn observe_cuda_compiler_context(
        &self,
    ) -> Result<(u64, u64, cubecl::cuda::CompilerContextUsage)> {
        let queue = self
            .pinned_cuda_queue
            .as_ref()
            .ok_or_else(|| Error::State("CUDA compiler observation has no pinned actor".into()))?;
        let owner = queue
            .compiler_context_owner::<crate::physical_service::CudaCompilerContextBudget>()
            .map_err(|e| Error::State(format!("original compiler observation: {e}")))?
            .ok_or_else(|| Error::State("CUDA compiler observation owner absent".into()))?;
        let (host, device) = owner.observed_reservations(self.budget())?;
        let usage = queue
            .compiler_context_usage()
            .map_err(|e| Error::State(format!("original compiler usage: {e}")))?
            .ok_or_else(|| Error::State("CUDA compiler observation usage absent".into()))?;
        Ok((host, device, usage))
    }
    pub(crate) fn physical_owner(&self) -> &Arc<ResidentPhysical> {
        &self.physical
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
    #[cfg(all(feature = "qualification", feature = "cuda"))]
    pub fn pinned_cuda_queue(&self) -> Result<&Arc<cubecl::cuda::OwnedCudaQueue>> {
        self.ensure_usable()?;
        self.pinned_cuda_queue
            .as_ref()
            .ok_or_else(|| Error::State("resident client was not pinned cold".into()))
    }
    pub fn budget(&self) -> &PhysicalServiceBudget {
        &self.budget
    }
    pub fn queue(&self) -> &Arc<PhysicalQueue> {
        &self.queue
    }
    pub fn owner(&self) -> &ResourceOwner {
        &self.owner
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    #[cfg(feature = "qualification")]
    pub fn capacity(&self) -> usize {
        self.cap
    }
    /// Explicit buffer H2D bytes; excludes dispatch/control parameter uniforms.
    #[cfg(feature = "qualification")]
    pub fn uploaded_value_bytes(&self) -> u64 {
        self.uploaded_value_bytes.load(Ordering::Acquire)
    }
    #[cfg(all(feature = "qualification", target_os = "linux"))]
    pub(crate) fn record_value_upload(&self, bytes: usize) {
        self.uploaded_value_bytes
            .fetch_add(bytes as u64, Ordering::AcqRel);
    }
    pub fn physical_key(&self) -> PhysicalDeviceKey {
        self.budget.physical_key()
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub fn physical_remaining(&self) -> Result<u64> {
        self.budget.remaining()
    }

    pub fn resident_bytes(&self) -> Result<usize> {
        self.ledger
            .lock()
            .map_err(|_| Error::State("resident lease ledger poisoned".into()))?
            .model_bytes(self.generation)
    }
    pub fn remaining_capacity(&self) -> Result<usize> {
        self.cap
            .checked_sub(self.resident_bytes()?)
            .ok_or_else(|| Error::State("resident capacity accounting invalid".into()))
    }
    #[cfg(feature = "qualification")]
    pub fn pending_events(&self) -> Result<usize> {
        self.ledger
            .lock()
            .map(|l| l.model_events(self.generation))
            .map_err(|_| Error::State("resident event ledger poisoned".into()))
    }
    pub fn ensure_usable(&self) -> Result<()> {
        self.admission
            .require_binding(self.source, &self.owner, self.generation, self.cap)?;
        if self.poisoned.load(Ordering::Acquire) || self.budget.fence_flag().load(Ordering::Acquire)
        {
            return Err(Error::State("resident service fenced".into()));
        }
        self.budget
            .prepaid_bytes_for(&self.physical_key(), &self.owner)?;
        crate::physical_quarantine::ensure_service_safe(&self.client)
    }

    /// Linearize logical publication against EVERY fence on this physical card.
    /// Caller may only swap prepared payloads/assign root metadata in `publish`.
    /// Displaced leases and captured GPU owners must drop after this returns.
    pub fn publish_if_usable<T>(&self, publish: impl FnOnce() -> Result<T>) -> Result<T> {
        self.ensure_usable()?;
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().with_unfenced(
            &self.physical_key(),
            || {
                self.admission.ensure_open()?;
                if self.poisoned.load(Ordering::Acquire)
                    || self.budget.fence_flag().load(Ordering::Acquire)
                {
                    return Err(Error::State(
                        "resident service fenced before publication".into(),
                    ));
                }
                publish()
            },
        )
    }

    // Keep exactly one service Arc permanently after uncertainty. The existing
    // quarantine owns actual handles; this keepalive owns their coarse charge
    // and poisoned ledger. It introduces no second resource/fence authority.
    fn fence_and_retain(self: &Arc<Self>, leases: &[ResidentLease]) {
        self.fence_and_retain_owners(leases, Vec::new());
    }

    pub(crate) fn fence_and_retain_owners(
        self: &Arc<Self>,
        leases: &[ResidentLease],
        owners: Vec<crate::physical_quarantine::CompletionOwner>,
    ) {
        self.poisoned.store(true, Ordering::Release);
        // If no active PhysicalWork owns the envelope, create/drop one before
        // quarantining handles so an unexpected release failure also fences it.
        if let Ok(work) = self.budget.begin(&self.client) {
            drop(work);
        }
        self.budget.fence_flag().store(true, Ordering::Release);
        let handles = leases
            .iter()
            .filter_map(|lease| {
                if lease.0.quarantined.swap(true, Ordering::AcqRel) {
                    None
                } else {
                    Some(lease.0.handle().clone())
                }
            })
            .collect::<Vec<_>>();
        crate::physical_quarantine::quarantine_uncertain_with_owners(
            self.client().clone(),
            handles,
            owners,
        );
        if !self.retained_forever.swap(true, Ordering::AcqRel) {
            std::mem::forget(self.clone());
        }
    }

    /// Local admission subdivides the existing envelope; no global re-charge.
    /// Empty memory has no valid contents until a successful Write submission.
    pub fn allocate_uninitialized(
        self: &Arc<Self>,
        bytes: usize,
        role: BufferRole,
    ) -> Result<ResidentLease> {
        self.ensure_usable()?;
        let allocation_generation = issue(&self.next_allocation)?;
        let id = self
            .ledger
            .lock()
            .map_err(|_| Error::State("resident lease ledger poisoned".into()))?
            .admit_for(self.generation, bytes, role)?;
        let mut admission = PendingAdmission {
            service: self.clone(),
            id,
            armed: true,
        };
        let handle = self.client.empty(bytes);
        let lease = ResidentLease(Arc::new(LeaseInner {
            service: self.clone(),
            id,
            handle: Some(handle),
            _backing: None,
            pool_use: None,
            workspace: None,
            bytes,
            role,
            allocation_generation,
            initialized: AtomicBool::new(false),
            workspace_written_epoch: AtomicU64::new(0),
            quarantined: AtomicBool::new(false),
        }));
        admission.armed = false;
        Ok(lease)
    }

    /// Start before a prepared-program allocator creates ANY handle. Caller
    /// owns the selected, begun fair worker slot. This guard admits and pins
    /// every allocation before create, and retains it before any H2D write.
    pub fn begin_initialization<'a>(
        self: &'a Arc<Self>,
        expected_root: u64,
    ) -> Result<ResidentInitialization<'a>> {
        self.ensure_usable()?;
        let work = self.budget.begin(&self.client)?;
        Ok(ResidentInitialization {
            service: self,
            expected_root,
            events: Vec::new(),
            leases: Vec::new(),
            uploaded: BTreeSet::new(),
            #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
            source_uploads: Vec::new(),
            #[cfg(all(feature = "qualification", target_os = "linux"))]
            matrix_source_uploads: Vec::new(),
            #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
            model_source_uploads: Vec::new(),
            completion_owners: Vec::new(),
            work: Some(work),
            armed: true,
        })
    }

    fn check_lease(self: &Arc<Self>, lease: &ResidentLease) -> Result<()> {
        if !Arc::ptr_eq(self, &lease.0.service)
            || self.generation != lease.service_generation()
            || self.source != lease.source_identity()
            || lease.0.handle().service != self.client.service_id()
        {
            return Err(Error::State(
                "resident lease source/service/generation mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Caller holds the selected fair worker slot and begins it immediately
    /// before this method. Accesses are whole-lease, unique, and retain every
    /// actual source/input/scratch/candidate handle until physical retirement.
    pub fn submit<'a>(
        self: &'a Arc<Self>,
        expected_root: u64,
        accesses: &[(ResidentLease, DeviceAccess)],
    ) -> Result<ResidentSubmission<'a>> {
        self.submit_bound(expected_root, None, accesses)
    }

    /// One physical event pins every row's actual buffers. Preserve complete
    /// source/session/epoch/root bindings before any kernel can be launched.
    pub fn submit_cohort<'a>(
        self: &'a Arc<Self>,
        roots: &[RootBinding],
        accesses: &[(ResidentLease, DeviceAccess)],
    ) -> Result<ResidentSubmission<'a>> {
        let cohort = ResidentCohortRoots::new(self.source, roots)?;
        self.submit_bound(roots[0].root(), Some(cohort), accesses)
    }

    fn submit_bound<'a>(
        self: &'a Arc<Self>,
        expected_root: u64,
        cohort: Option<ResidentCohortRoots>,
        accesses: &[(ResidentLease, DeviceAccess)],
    ) -> Result<ResidentSubmission<'a>> {
        self.ensure_usable()?;
        let access_index = access_positions(accesses.iter().map(|(lease, _)| lease.id()))?;
        let mut workspace_runs = BTreeMap::<usize, Arc<backing_pool::ResidentWorkspaceRun>>::new();
        for (lease, access) in accesses {
            self.check_lease(lease)?;
            if let Some(workspace) = &lease.0.workspace {
                if let Some(run) = workspace_runs.get(&workspace.key()) {
                    workspace.require_run(run)?;
                } else {
                    workspace_runs.insert(workspace.key(), workspace.active_run()?);
                }
            }
            if *access == DeviceAccess::Read {
                lease.require_completed_read(
                    lease
                        .0
                        .workspace
                        .as_ref()
                        .and_then(|w| workspace_runs.get(&w.key()))
                        .map(Arc::as_ref),
                )?;
            }
            if *access == DeviceAccess::Write
                && lease.0.role == BufferRole::ImmutableWeights
                && lease.is_initialized()
            {
                return Err(Error::State(
                    "duplicate/uninitialized/immutable resident access".into(),
                ));
            }
        }
        let event = self
            .ledger
            .lock()
            .map_err(|_| Error::State("resident event ledger poisoned".into()))?
            .submit(
                expected_root,
                &accesses
                    .iter()
                    .map(|(l, a)| (l.id(), *a))
                    .collect::<Vec<_>>(),
            )?;
        let work = match self.budget.begin(&self.client) {
            Ok(work) => work,
            Err(error) => {
                if let Ok(mut ledger) = self.ledger.lock() {
                    let _ = ledger.report_completion_unknown(event);
                }
                self.fence_and_retain(&accesses.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>());
                return Err(error);
            }
        };
        Ok(ResidentSubmission {
            service: self,
            event,
            expected_root,
            cohort,
            accesses: accesses.to_vec(),
            access_index,
            work: Some(work),
            armed: true,
            completion_owners: workspace_runs
                .values()
                .cloned()
                .map(|run| run as crate::physical_quarantine::CompletionOwner)
                .collect(),
            workspace_runs,
        })
    }
}

/// Immutable metadata for one submitted cohort; never a publication authority.
struct ResidentCohortRoots {
    bindings: Vec<RootBinding>,
}
impl ResidentCohortRoots {
    fn new(source: SourceIdentity, bindings: &[RootBinding]) -> Result<Self> {
        if !source.is_bound() || !matches!(bindings.len(), 2 | 4 | 8) {
            return Err(Error::Limit(
                "resident completion cohort source/size".into(),
            ));
        }
        let mut sessions = BTreeSet::new();
        for binding in bindings {
            if binding.source() != source || !sessions.insert(binding.session()) {
                return Err(Error::State(
                    "resident completion cohort source/session mismatch".into(),
                ));
            }
        }
        Ok(Self {
            bindings: bindings.to_vec(),
        })
    }

    fn require_exact(&self, bindings: impl Iterator<Item = RootBinding>) -> Result<()> {
        if self.bindings.iter().copied().ne(bindings) {
            return Err(Error::State(
                "resident completion cohort roots changed or reordered".into(),
            ));
        }
        Ok(())
    }
}

struct PendingAdmission {
    service: Arc<ResidentService>,
    id: BufferLeaseId,
    armed: bool,
}
impl Drop for PendingAdmission {
    fn drop(&mut self) {
        if self.armed {
            let released = self
                .service
                .ledger
                .lock()
                .ok()
                .is_some_and(|mut l| l.release_lease(self.id).is_ok());
            if !released {
                self.service.fence_and_retain(&[]);
            }
        }
    }
}

struct LeaseInner {
    // Rust drops fields in declaration order after Drop: release the physical
    // handle BEFORE the final service Arc can release its prepaid envelope.
    handle: Option<Handle>,
    _backing: Option<Arc<backing_pool::BackingPlane>>,
    pool_use: Option<Arc<backing_pool::PoolUse>>,
    workspace: Option<backing_pool::WorkspaceMembership>,
    id: BufferLeaseId,
    bytes: usize,
    role: BufferRole,
    allocation_generation: u64,
    initialized: AtomicBool,
    workspace_written_epoch: AtomicU64,
    quarantined: AtomicBool,
    service: Arc<ResidentService>,
}
impl LeaseInner {
    fn handle(&self) -> &Handle {
        self.handle.as_ref().expect("live resident handle")
    }
}
impl Drop for LeaseInner {
    fn drop(&mut self) {
        #[cfg(all(
            any(
                feature = "qwen-source-qualification",
                feature = "gemma-bf16-qualification"
            ),
            feature = "hardware-abort-probe"
        ))]
        let refused = final_owner_probe::refuse_release(self.service.generation, self.id);
        #[cfg(not(all(
            any(
                feature = "qwen-source-qualification",
                feature = "gemma-bf16-qualification"
            ),
            feature = "hardware-abort-probe"
        )))]
        let refused = false;
        let released = if !refused && self.service.ensure_usable().is_ok() {
            self.service.ledger.lock().ok().is_some_and(|mut ledger| {
                if ledger.release_lease(self.id).is_ok() {
                    // Keep admission locked until the original handle has
                    // dropped, preventing a local allocation-overlap gap.
                    drop(self.handle.take());
                    true
                } else {
                    false
                }
            })
        } else {
            false
        };
        if !released {
            if let Some(pool_use) = &self.pool_use {
                pool_use.poison();
            }
            self.service.fence_and_retain(&[]);
            if !self.quarantined.swap(true, Ordering::AcqRel) {
                crate::physical_quarantine::quarantine_uncertain(
                    self.service.client().clone(),
                    vec![self.handle().clone()],
                );
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct ResidentLease(Arc<LeaseInner>);
impl ResidentLease {
    fn require_completed_read(
        &self,
        run: Option<&backing_pool::ResidentWorkspaceRun>,
    ) -> Result<()> {
        if !self.is_initialized() {
            return Err(Error::State(
                "resident read before physical initialization".into(),
            ));
        }
        if let Some(workspace) = &self.0.workspace {
            let run =
                run.ok_or_else(|| Error::State("resident workspace read owner absent".into()))?;
            workspace.require_run(run)?;
            let epoch = run.epoch();
            if self.0.workspace_written_epoch.load(Ordering::Acquire) != epoch {
                return Err(Error::State(
                    "resident workspace read from a prior activation".into(),
                ));
            }
        }
        Ok(())
    }
    fn record_completed_write(
        &self,
        run: Option<&backing_pool::ResidentWorkspaceRun>,
    ) -> Result<()> {
        if let Some(workspace) = &self.0.workspace {
            let run =
                run.ok_or_else(|| Error::State("resident workspace write owner absent".into()))?;
            workspace.require_run(run)?;
            self.0
                .workspace_written_epoch
                .store(run.epoch(), Ordering::Release);
        }
        self.0.initialized.store(true, Ordering::Release);
        Ok(())
    }
    pub fn id(&self) -> BufferLeaseId {
        self.0.id
    }
    pub fn bytes(&self) -> usize {
        self.0.bytes
    }
    pub(crate) fn role(&self) -> BufferRole {
        self.0.role
    }
    pub fn handle(&self) -> Handle {
        self.0.handle().clone()
    }
    pub fn service(&self) -> &Arc<ResidentService> {
        &self.0.service
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.0.service.source
    }
    pub fn service_generation(&self) -> u64 {
        self.0.service.generation
    }
    pub fn allocation_generation(&self) -> u64 {
        self.0.allocation_generation
    }
    pub fn is_initialized(&self) -> bool {
        self.0.initialized.load(Ordering::Acquire)
    }
    pub fn readback(&self, offset: usize, bytes: usize) -> Result<ResidentReadback> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|n| n > self.bytes()) {
            return Err(Error::Malformed(
                "resident readback view outside allocation".into(),
            ));
        }
        Ok(ResidentReadback {
            lease: self.clone(),
            offset,
            bytes,
        })
    }
}

pub(crate) struct ResidentReadback {
    lease: ResidentLease,
    offset: usize,
    bytes: usize,
}
impl ResidentReadback {
    #[cfg(feature = "cuda-native-qualification")]
    pub(crate) fn lease_id(&self) -> BufferLeaseId {
        self.lease.id()
    }
    #[cfg(feature = "cuda-native-qualification")]
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }
    fn handle(&self) -> Handle {
        self.lease
            .handle()
            .offset_start(self.offset as u64)
            .offset_end((self.lease.bytes() - self.offset - self.bytes) as u64)
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
enum SourceCompletionMode<'a> {
    Compute(&'a [ResidentLease]),
    Retirement,
}
#[must_use]
pub(crate) struct ResidentSubmission<'a> {
    service: &'a Arc<ResidentService>,
    event: DeviceEventId,
    expected_root: u64,
    cohort: Option<ResidentCohortRoots>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    // Positions in the original retained cohort; never substitutes its owners.
    access_index: BTreeMap<BufferLeaseId, usize>,
    work: Option<PhysicalWork<'a>>,
    armed: bool,
    completion_owners: Vec<crate::physical_quarantine::CompletionOwner>,
    workspace_runs: BTreeMap<usize, Arc<backing_pool::ResidentWorkspaceRun>>,
}
impl ResidentSubmission<'_> {
    /// Exact current-flight membership. Read accepts a retained Write; Write
    /// requires that same actual allocation's submitted Write authority.
    pub(crate) fn require_accesses<'a>(
        &self,
        required: impl IntoIterator<Item = (&'a ResidentLease, DeviceAccess)>,
    ) -> Result<()> {
        if !self.armed || self.work.is_none() {
            return Err(Error::State(
                "resident access needs original armed flight".into(),
            ));
        }
        self.service.ensure_usable()?;
        for (lease, access) in required {
            self.service.check_lease(lease)?;
            let actual = self
                .access_index
                .get(&lease.id())
                .and_then(|&i| self.accesses.get(i));
            if !actual.is_some_and(|(actual, submitted)| {
                Arc::ptr_eq(&actual.0, &lease.0)
                    && actual.allocation_generation() == lease.allocation_generation()
                    && actual.bytes() == lease.bytes()
                    && (access == DeviceAccess::Read || *submitted == DeviceAccess::Write)
            }) {
                return Err(Error::State(
                    "original flight missing exact resident lease/access".into(),
                ));
            }
        }
        Ok(())
    }
    #[cfg(feature = "qualification")]
    pub(crate) fn retain_completion_owner(
        &mut self,
        owner: crate::physical_quarantine::CompletionOwner,
    ) -> Result<()> {
        // Attach first: even refusal on an already fenced service must not
        // drop an executable/source owner of previously enqueued work.
        self.completion_owners.push(owner);
        self.service.ensure_usable()
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub fn handles(&self) -> Vec<Handle> {
        self.accesses.iter().map(|(l, _)| l.handle()).collect()
    }
    fn check_outputs(&self, outputs: &[ResidentLease]) -> Result<()> {
        if outputs.is_empty() {
            return Err(Error::State("resident Ready requires outputs".into()));
        }
        let mut unique = BTreeSet::new();
        for output in outputs {
            self.service.check_lease(output)?;
            // Scratch views are scoped to the original workspace activation.
            // Exported roots/packets use independent storage, never these views.
            if output.0.workspace.is_some()
                || !unique.insert(output.id())
                || !self.access_index.get(&output.id()).is_some_and(|&index| {
                    let (lease, access) = &self.accesses[index];
                    Arc::ptr_eq(&lease.0, &output.0) && *access == DeviceAccess::Write
                })
            {
                return Err(Error::State(
                    "resident promotion requires unique submitted writes".into(),
                ));
            }
        }
        Ok(())
    }
    fn finish_ready(mut self, outputs: &[ResidentLease]) -> Result<ResidentReady> {
        self.service.ensure_usable()?;
        {
            let mut ledger = self
                .service
                .ledger
                .lock()
                .map_err(|_| Error::State("resident Ready ledger poisoned".into()))?;
            ledger.report_physical_ready(self.event)?;
            ledger.retire_physical_ready(self.event)?; // Retires event pins ONLY.
        }
        self.work
            .take()
            .ok_or_else(|| Error::State("resident physical guard absent".into()))?
            .complete()?;
        for (lease, access) in &self.accesses {
            if *access == DeviceAccess::Write {
                lease.record_completed_write(
                    lease
                        .0
                        .workspace
                        .as_ref()
                        .and_then(|w| self.workspace_runs.get(&w.key()))
                        .map(Arc::as_ref),
                )?;
            }
        }
        self.armed = false;
        Ok(ResidentReady {
            service: self.service.clone(),
            events: vec![self.event],
            expected_root: self.expected_root,
            root_binding: None,
            outputs: outputs.to_vec(),
        })
    }
    /// Actual readback and trusted numerical/status validation occur HERE,
    /// before the private Ready constructor can retire any event pin.
    pub async fn complete_with_readback<F>(
        self,
        outputs: &[ResidentLease],
        cohort: &[ResidentReadback],
        validate: F,
    ) -> Result<(ResidentReady, Vec<Bytes>)>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        if self.cohort.is_some() {
            return Err(Error::State(
                "resident cohort requires all-row completion".into(),
            ));
        }
        self.check_outputs(outputs)?;
        let bytes = self.readback_checked(cohort, validate).await?;
        let ready = self.finish_ready(outputs)?;
        Ok((ready, bytes))
    }

    /// Exact finite source plan installed on THIS armed flight before enqueue.
    /// Serial source verification and the trusted validator precede originalReady.
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    async fn complete_with_readback_and_source_verification<F>(
        self,
        mode: SourceCompletionMode<'_>,
        cohort: &[ResidentReadback],
        plan: &Arc<Qwen2bMlpSourceVerification>,
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<(ResidentReady, Vec<Bytes>)>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        if self.cohort.is_some() {
            return Err(Error::State(
                "finite2B source completion is one original flight".into(),
            ));
        }
        let check_cancel = || -> Result<()> {
            if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                Err(Error::State(
                    "original full-model learned upload cancelled".into(),
                ))
            } else {
                Ok(())
            }
        };
        check_cancel()?;
        let outputs =
            match mode {
                SourceCompletionMode::Compute(outputs) => {
                    self.check_outputs(outputs)?;
                    outputs
                }
                SourceCompletionMode::Retirement => {
                    if self.accesses.iter().any(|(lease, access)| {
                        *access != DeviceAccess::Read || !lease.is_initialized()
                    }) {
                        return Err(Error::State(
                            "source retirement requires existing initialized Reads".into(),
                        ));
                    }
                    &[]
                }
            };
        let total = cohort
            .iter()
            .try_fold(0usize, |sum, read| sum.checked_add(read.bytes()))
            .ok_or_else(|| Error::Limit("finite source-completion cohort overflow".into()))?;
        if total > 128 * 1024 * 1024 {
            return Err(Error::Limit("finite source-completion128MiB cohort".into()));
        }
        plan.require_installed(&self)?;
        let bytes = self.readback_checked(cohort, |_| Ok(())).await?;
        plan.verify(&self, cancelled).await?;
        check_cancel()?;
        validate(&bytes)?;
        check_cancel()?;
        let ready = self.finish_ready(outputs)?;
        Ok((ready, bytes))
    }

    /// Split a single trusted physical completion into exact original row
    /// proofs. No row can promote a foreign, shared or non-written output.
    pub async fn complete_cohort_with_readback<F>(
        self,
        rows: &[(RootBinding, Vec<ResidentLease>)],
        cohort: &[ResidentReadback],
        validate: F,
    ) -> Result<(Vec<ResidentReady>, Vec<Bytes>)>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        self.cohort
            .as_ref()
            .ok_or_else(|| Error::State("resident submission has no bound cohort".into()))?
            .require_exact(rows.iter().map(|(root, _)| *root))?;
        if rows.iter().any(|(_, outputs)| outputs.is_empty()) {
            return Err(Error::State("resident cohort row has no outputs".into()));
        }
        let outputs = rows
            .iter()
            .flat_map(|(_, outputs)| outputs.iter().cloned())
            .collect::<Vec<_>>();
        self.check_outputs(&outputs)?;
        let bytes = self.readback_checked(cohort, validate).await?;
        let completion = self.finish_ready(&outputs)?;
        let ready = rows
            .iter()
            .map(|(binding, outputs)| ResidentReady {
                service: completion.service.clone(),
                events: completion.events.clone(),
                expected_root: binding.root(),
                root_binding: Some(*binding),
                outputs: outputs.clone(),
            })
            .collect();
        Ok((ready, bytes))
    }
    async fn readback_checked<F>(
        &self,
        cohort: &[ResidentReadback],
        validate: F,
    ) -> Result<Vec<Bytes>>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        if cohort.is_empty() {
            return Err(Error::State(
                "resident completion requires physical readback".into(),
            ));
        }
        self.require_accesses(cohort.iter().map(|item| (&item.lease, DeviceAccess::Read)))?;
        let bytes = self
            .service
            .client
            .read_async(cohort.iter().map(|item| item.handle()).collect())
            .await
            .map_err(|e| Error::State(format!("resident physical readback failed: {e}")))?;
        if bytes.len() != cohort.len() || bytes.iter().zip(cohort).any(|(b, c)| b.len() != c.bytes)
        {
            return Err(Error::State(
                "resident physical readback cohort incomplete".into(),
            ));
        }
        validate(&bytes)?;
        Ok(bytes)
    }
    /// Snapshot/export uses only existing initialized Read leases. It never
    /// fabricates a Write or promotes/initializes any new state allocation.
    pub async fn complete_read_only_readback<F>(
        self,
        cohort: &[ResidentReadback],
        validate: F,
    ) -> Result<Vec<Bytes>>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        if self.cohort.is_some()
            || self
                .accesses
                .iter()
                .any(|(lease, access)| *access != DeviceAccess::Read || !lease.is_initialized())
        {
            return Err(Error::State(
                "read-only completion requires initialized Reads".into(),
            ));
        }
        let bytes = self.readback_checked(cohort, validate).await?;
        drop(self.finish_ready(&[])?);
        Ok(bytes)
    }
}
impl Drop for ResidentSubmission<'_> {
    fn drop(&mut self) {
        if self.armed {
            if let Ok(mut ledger) = self.service.ledger.lock() {
                let _ = ledger.report_completion_unknown(self.event);
            }
            self.service.fence_and_retain_owners(
                &self
                    .accesses
                    .iter()
                    .map(|(l, _)| l.clone())
                    .collect::<Vec<_>>(),
                std::mem::take(&mut self.completion_owners),
            );
            drop(self.work.take());
        }
    }
}

/// Persistent physically Ready outputs. Root/epoch/session/domain validation
/// and publication remain with the sole engine coordinator, never this type.
#[must_use]
pub(crate) struct ResidentReady {
    service: Arc<ResidentService>,
    events: Vec<DeviceEventId>,
    expected_root: u64,
    root_binding: Option<RootBinding>,
    outputs: Vec<ResidentLease>,
}
impl ResidentReady {
    pub fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub fn event_ids(&self) -> &[DeviceEventId] {
        &self.events
    }
    pub fn expected_root(&self) -> u64 {
        self.expected_root
    }
    pub fn root_binding(&self) -> Option<RootBinding> {
        self.root_binding
    }
    pub fn outputs(&self) -> &[ResidentLease] {
        &self.outputs
    }
}

/// Allocation callback owner for model-load/compiled-program preparation.
/// Source and scratch allocations share one existing prepaid physical guard.
#[must_use]
pub(crate) struct ResidentInitialization<'a> {
    service: &'a Arc<ResidentService>,
    expected_root: u64,
    events: Vec<DeviceEventId>,
    leases: Vec<ResidentLease>,
    uploaded: BTreeSet<BufferLeaseId>,
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    source_uploads: Vec<Arc<source_upload::UploadOwner>>,
    #[cfg(all(feature = "qualification", target_os = "linux"))]
    matrix_source_uploads: Vec<Arc<resident_matrix::upload::UploadOwner>>,
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    model_source_uploads: Vec<Arc<model_upload::ModelUploadOwner>>,
    completion_owners: Vec<crate::physical_quarantine::CompletionOwner>,
    work: Option<PhysicalWork<'a>>,
    armed: bool,
}
impl ResidentInitialization<'_> {
    #[cfg(feature = "qualification")]
    pub(crate) fn require_service(&self, service: &Arc<ResidentService>) -> Result<()> {
        if !Arc::ptr_eq(self.service, service) {
            return Err(Error::State(
                "native original initialization service differs".into(),
            ));
        }
        service.ensure_usable()
    }
    #[cfg(feature = "qualification")]
    pub(crate) fn retain_completion_owner(
        &mut self,
        owner: crate::physical_quarantine::CompletionOwner,
    ) -> Result<()> {
        // Attach first: refusal must retain owners of previously queued work.
        self.completion_owners.push(owner);
        self.service.ensure_usable()
    }
    pub fn allocate_empty(&mut self, bytes: usize, role: BufferRole) -> Result<ResidentLease> {
        self.service.ensure_usable()?;
        let allocation_generation = issue(&self.service.next_allocation)?;
        let (id, event) = {
            let mut ledger = self
                .service
                .ledger
                .lock()
                .map_err(|_| Error::State("resident initialization ledger poisoned".into()))?;
            let id = ledger.admit_for(self.service.generation, bytes, role)?;
            match ledger.submit(self.expected_root, &[(id, DeviceAccess::Write)]) {
                Ok(event) => (id, event),
                Err(error) => {
                    ledger.release_lease(id)?;
                    return Err(error);
                }
            }
        };
        // Retain the event before create can panic; no write/copy has occurred.
        self.events.push(event);
        let handle = self.service.client.empty(bytes);
        let lease = ResidentLease(Arc::new(LeaseInner {
            handle: Some(handle),
            _backing: None,
            pool_use: None,
            workspace: None,
            id,
            bytes,
            role,
            allocation_generation,
            initialized: AtomicBool::new(false),
            workspace_written_epoch: AtomicU64::new(0),
            quarantined: AtomicBool::new(false),
            service: self.service.clone(),
        }));
        self.leases.push(lease.clone());
        Ok(lease)
    }
    /// Move zeroes directly; backend staging needs no caller slice clone.
    pub(crate) fn upload_zeroed(
        &mut self,
        bytes: usize,
        role: BufferRole,
    ) -> Result<ResidentLease> {
        let lease = self.allocate_empty(bytes, role)?;
        self.upload_zeroed_into(&lease)?;
        Ok(lease)
    }
    /// Fresh leases retain their allocation Write; pooled leases register one.
    pub(crate) fn upload_zeroed_into(&mut self, lease: &ResidentLease) -> Result<()> {
        self.service.ensure_usable()?;
        self.service.check_lease(lease)?;
        if lease.is_initialized() || self.uploaded.contains(&lease.id()) {
            return Err(Error::State(
                "resident zero destination already written".into(),
            ));
        }
        if let Some(held) = self.leases.iter().find(|held| held.id() == lease.id()) {
            if !Arc::ptr_eq(&held.0, &lease.0) {
                return Err(Error::State(
                    "resident zero destination alias differs".into(),
                ));
            }
        } else {
            if let Some(workspace) = &lease.0.workspace {
                self.completion_owners.push(workspace.active_run()?);
            }
            let event = self
                .service
                .ledger
                .lock()
                .map_err(|_| Error::State("resident initialization ledger poisoned".into()))?
                .submit(self.expected_root, &[(lease.id(), DeviceAccess::Write)])?;
            self.events.push(event);
            self.leases.push(lease.clone());
        }
        // Both the exact event and destination owner precede allocation/H2D.
        self.service.client.write(
            &lease.handle(),
            Bytes::from_bytes_vec(vec![0u8; lease.bytes()]),
        );
        self.service
            .uploaded_value_bytes
            .fetch_add(lease.bytes() as u64, Ordering::AcqRel);
        self.uploaded.insert(lease.id());
        Ok(())
    }
    pub fn upload_bytes(&mut self, bytes: &[u8], role: BufferRole) -> Result<ResidentLease> {
        let lease = self.allocate_empty(bytes.len(), role)?;
        // The original handle is already retained by this guard BEFORE H2D.
        self.service
            .client
            .write(&lease.handle(), Bytes::from_bytes_vec(bytes.to_vec()));
        self.service
            .uploaded_value_bytes
            .fetch_add(bytes.len() as u64, Ordering::AcqRel);
        self.uploaded.insert(lease.id());
        Ok(lease)
    }
    pub async fn complete_sync(mut self) -> Result<ResidentReady> {
        #[cfg(all(feature = "qualification", target_os = "linux"))]
        for upload in &self.matrix_source_uploads {
            upload.require_complete()?;
        }
        #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
        for upload in &self.source_uploads {
            upload.require_complete()?;
        }
        #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
        for upload in &self.model_source_uploads {
            upload.require_complete()?;
        }
        if self.events.is_empty() {
            return Err(Error::State("empty resident initialization".into()));
        }
        self.service
            .client
            .sync()
            .await
            .map_err(|e| Error::State(format!("resident preparation barrier failed: {e}")))?;
        self.service.ensure_usable()?;
        {
            let mut ledger =
                self.service.ledger.lock().map_err(|_| {
                    Error::State("resident preparation Ready ledger poisoned".into())
                })?;
            for event in &self.events {
                ledger.report_physical_ready(*event)?;
                ledger.retire_physical_ready(*event)?;
            }
        }
        self.work
            .take()
            .ok_or_else(|| Error::State("resident preparation guard absent".into()))?
            .complete()?;
        for lease in &self.leases {
            if self.uploaded.contains(&lease.id()) {
                let run = lease
                    .0
                    .workspace
                    .as_ref()
                    .map(|w| w.active_run())
                    .transpose()?;
                lease.record_completed_write(run.as_deref())?;
            }
        }
        self.armed = false;
        Ok(ResidentReady {
            service: self.service.clone(),
            events: self.events.clone(),
            expected_root: self.expected_root,
            root_binding: None,
            // Initializers may write private workspace views, but Ready must
            // not export raw cloneable aliases beyond their activation.
            outputs: self
                .leases
                .iter()
                .filter(|l| l.0.workspace.is_none())
                .cloned()
                .collect(),
        })
    }
}
impl Drop for ResidentInitialization<'_> {
    fn drop(&mut self) {
        if self.armed {
            if let Ok(mut ledger) = self.service.ledger.lock() {
                for event in &self.events {
                    let _ = ledger.report_completion_unknown(*event);
                }
            }
            self.service
                .fence_and_retain_owners(&self.leases, std::mem::take(&mut self.completion_owners));
            drop(self.work.take());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::session_root::SessionAuthority;

    #[test]
    fn access_index_keeps_original_cohort_positions_and_refuses_duplicate_ids() {
        let mut ledger = DeviceEventLedger::new(1, 64).unwrap();
        let a = ledger.admit(4, BufferRole::Activation).unwrap();
        let b = ledger.admit(4, BufferRole::Scratch).unwrap();
        let absent = ledger.admit(4, BufferRole::Output).unwrap();
        let positions = access_positions([b, a]).unwrap();
        assert_eq!(positions[&b], 0);
        assert_eq!(positions[&a], 1);
        assert!(!positions.contains_key(&absent));
        assert!(access_positions([a, b, a]).is_err());
    }

    fn sessions(source: SourceIdentity, count: usize) -> Vec<SessionAuthority> {
        (0..count)
            .map(|_| SessionAuthority::new(source).unwrap())
            .collect()
    }

    #[test]
    fn completion_cohort_accepts_only_finite_native_row_counts() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let owners = sessions(source, 9);
        let bindings = owners
            .iter()
            .map(SessionAuthority::binding)
            .collect::<Vec<_>>();
        for count in [2, 4, 8] {
            let cohort = ResidentCohortRoots::new(source, &bindings[..count]).unwrap();
            cohort
                .require_exact(bindings[..count].iter().copied())
                .unwrap();
        }
        for count in [0, 1, 3, 5, 6, 7, 9] {
            assert!(ResidentCohortRoots::new(source, &bindings[..count]).is_err());
        }
        assert!(ResidentCohortRoots::new(SourceIdentity::unbound(), &bindings[..2]).is_err());
    }

    #[test]
    fn completion_cohort_rejects_foreign_duplicate_and_binds_differing_frontiers() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut owners = sessions(source, 2);
        let foreign = SessionAuthority::new(SourceIdentity::for_new_allocation().unwrap()).unwrap();
        let initial = owners[0].binding();
        assert!(ResidentCohortRoots::new(source, &[initial, initial]).is_err());
        assert!(ResidentCohortRoots::new(source, &[initial, foreign.binding()]).is_err());
        let advance = owners[1].begin_advance(1).unwrap();
        owners[1].prepare_publish(advance).unwrap().commit();
        let bindings = [initial, owners[1].binding()];
        let cohort = ResidentCohortRoots::new(source, &bindings).unwrap();
        cohort.require_exact(bindings.into_iter()).unwrap();
        assert!(cohort.require_exact(bindings.into_iter().rev()).is_err());
        assert_eq!(owners[0].binding(), initial);
    }

    #[test]
    fn completion_cohort_preserves_independent_root_epochs_and_exact_order() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut owners = sessions(source, 2);
        let advance = owners[0].begin_advance(1).unwrap();
        owners[0].prepare_publish(advance).unwrap().commit();
        let advance = owners[1].begin_advance(5).unwrap();
        owners[1].prepare_publish(advance).unwrap().commit();
        let reset = owners[1].begin_reset().unwrap();
        owners[1].prepare_publish(reset).unwrap().commit();
        let advance = owners[1].begin_advance(1).unwrap();
        owners[1].prepare_publish(advance).unwrap().commit();
        let bindings = [owners[0].binding(), owners[1].binding()];
        assert_ne!(bindings[0].root(), bindings[1].root());
        assert_ne!(bindings[0].session_epoch(), bindings[1].session_epoch());
        assert_eq!(bindings[0].frontier(), bindings[1].frontier());
        let cohort = ResidentCohortRoots::new(source, &bindings).unwrap();
        cohort.require_exact(bindings.into_iter()).unwrap();
        assert!(cohort.require_exact(bindings.into_iter().rev()).is_err());
        assert!(cohort.require_exact(bindings[..1].iter().copied()).is_err());
        assert!(cohort.require_exact([bindings[0]; 2].into_iter()).is_err());
        let replace = owners[1].begin_validated_replacement(1).unwrap();
        owners[1].prepare_publish(replace).unwrap().commit();
        assert_eq!(owners[1].binding().frontier(), bindings[1].frontier());
        assert!(
            cohort
                .require_exact([bindings[0], owners[1].binding()].into_iter())
                .is_err()
        );
        // Capture and validation never publish any root themselves.
        assert_eq!(owners[0].binding(), bindings[0]);
    }
}

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod qwen2b_projection;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
mod rms_scratch;
