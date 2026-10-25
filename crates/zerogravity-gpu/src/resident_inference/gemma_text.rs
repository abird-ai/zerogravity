//! Candidate binding around the original ResidentModel/SessionAuthority owners.
use super::*;
#[cfg(all(feature = "qualification", any(feature = "hip", feature = "vulkan")))]
use crate::resident_program::gemma_text::{
    GemmaTextBody, GemmaTextHostOwners, GemmaTextNativeOwners, GemmaTextStorage,
};
use crate::resident_program::gemma_text::{GemmaTextObserver, GemmaTextResourceInventory};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::Bf16TextCall;
use zerogravity_model_gemma4::text::OriginalGemma4Text;

#[cfg(all(
    feature = "gemma-bf16-qualification",
    any(feature = "hip", feature = "vulkan"),
    feature = "hardware-abort-probe"
))]
mod final_owner;
#[cfg(all(
    feature = "gemma-bf16-qualification",
    any(feature = "hip", feature = "vulkan"),
    feature = "hardware-abort-probe"
))]
pub use final_owner::GemmaFinalOwnerReceipt;
#[cfg(all(
    feature = "gemma-bf16-qualification",
    feature = "vulkan",
    feature = "hip",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
mod same_pci;
#[cfg(all(
    feature = "gemma-bf16-qualification",
    feature = "vulkan",
    feature = "hip",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
pub use same_pci::{GemmaVulkanNativeRetention, LearnedSamePciPeerReceipt};

#[cfg(all(
    feature = "qualification",
    target_os = "linux",
    any(
        feature = "hip",
        all(feature = "vulkan", feature = "gemma-bf16-qualification")
    )
))]
enum GemmaNativeTarget {
    #[cfg(feature = "hip")]
    Hip(crate::backend::hip_native_policy::HipColdOwner),
    #[cfg(all(
        feature = "vulkan",
        feature = "gemma-bf16-qualification",
        target_os = "linux"
    ))]
    Vulkan(Device),
}
#[cfg(all(
    feature = "qualification",
    target_os = "linux",
    any(
        feature = "hip",
        all(feature = "vulkan", feature = "gemma-bf16-qualification")
    )
))]
impl GemmaNativeTarget {
    fn device(&self) -> Device {
        match self {
            #[cfg(feature = "hip")]
            Self::Hip(owner) => owner.device(),
            #[cfg(all(
                feature = "vulkan",
                feature = "gemma-bf16-qualification",
                target_os = "linux"
            ))]
            Self::Vulkan(device) => device.clone(),
        }
    }
    fn host_bytes(&self) -> u64 {
        match self {
            #[cfg(feature = "hip")]
            Self::Hip(owner) => owner.host_bytes(),
            #[cfg(all(
                feature = "vulkan",
                feature = "gemma-bf16-qualification",
                target_os = "linux"
            ))]
            Self::Vulkan(_) => {
                crate::backend::vulkan_native_policy::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES
            }
        }
    }
}
pub(super) struct GemmaOptions<'a, 'o> {
    pub source: &'a VerifiedTensorSource,
    pub observer: &'a std::cell::RefCell<Option<&'o mut dyn GemmaTextObserver>>,
    pub cancelled: Option<&'a AtomicBool>,
}
pub(super) struct GemmaStep<'a, 'o> {
    pub call: &'a Bf16TextCall,
    pub row: usize,
    pub observer: &'a std::cell::RefCell<Option<&'o mut dyn GemmaTextObserver>>,
    pub cancelled: Option<&'a AtomicBool>,
}
pub(super) fn check_cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("Gemma source-bound GPU call cancelled".into()));
    }
    Ok(())
}

pub struct Gemma4Bf16GpuCandidate {
    model: Arc<ResidentModel>,
    source: Arc<OriginalGemma4Text>,
    _source_owner: Arc<dyn std::any::Any + Send + Sync>,
    inventory: GemmaTextResourceInventory,
    call_lock: Mutex<()>,
    proposals: Arc<async_lock::Semaphore>,
    sessions: Arc<async_lock::Semaphore>,
    snapshots: Arc<async_lock::Semaphore>,
}
pub struct Gemma4Bf16GpuSession {
    inner: ResidentSession,
    prompt: Arc<[u32]>,
    feedback: Option<ResidentToken>,
    // Final actual root/feedback readers precede source/native retirement.
    loaded: Arc<Gemma4Bf16GpuCandidate>,
    limit: usize,
    calls: usize,
    stopped: bool,
    _session: async_lock::SemaphoreGuardArc,
}
pub struct Gemma4Bf16GpuPending {
    inner: PreparedInference,
    loaded: Arc<Gemma4Bf16GpuCandidate>,
    limit: usize,
    _proposal: async_lock::SemaphoreGuardArc,
}
pub struct Gemma4Bf16GpuSnapshot {
    inner: ResidentSnapshot,
    feedback: Option<ResidentToken>,
    // Snapshot domains alone do not retain model/native owners.
    loaded: Arc<Gemma4Bf16GpuCandidate>,
    limit: usize,
    calls: usize,
    stopped: bool,
    _snapshot: async_lock::SemaphoreGuardArc,
}
#[derive(Clone, Copy, Debug)]
pub struct Gemma4Bf16GpuCall {
    pub root: RootBinding,
    pub selected_id: u32,
    pub max_bits: u32,
    pub eos: bool,
}
impl Gemma4Bf16GpuCandidate {
    /// Exact x2 HIP qualification only. Bind the managed source under the
    /// backend's canonical16GiB host envelope before entry; retain its snapshot
    /// lease through model retirement. No trained/serving eligibility follows.
    ///
    /// # Safety
    /// `source_owner` must retain the authentic managed SnapshotLease and source
    /// lifecycle/close guard bound to this exact `source`. That owner must remain
    /// live through every prepared native, initialization, flight and quarantine
    /// owner. An arbitrary Arc is not proof. Use the validated safe facade.
    #[cfg(all(feature = "qualification", feature = "hip", target_os = "linux"))]
    pub async unsafe fn load_hip_qualification(
        device: Device,
        source: Arc<OriginalGemma4Text>,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        ids: Arc<[u32]>,
        host: GemmaTextHostOwners,
        storage: GemmaTextStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Gemma4Bf16GpuSession)> {
        Self::load_owner(
            GemmaNativeTarget::Hip(crate::backend::hip_native_policy::HipColdOwner::Single(
                device,
            )),
            source,
            source_owner,
            ids,
            host,
            storage,
            cancelled,
        )
        .await
    }
    /// # Safety
    /// Same authentic retained source obligations as `load_hip_qualification`.
    #[cfg(all(
        feature = "qualification",
        feature = "hip",
        feature = "qwen-source-qualification",
        target_os = "linux"
    ))]
    pub async unsafe fn load_shared_hip_qualification(
        shared: Arc<crate::backend::hip_native_policy::SharedHipNative>,
        source: Arc<OriginalGemma4Text>,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        ids: Arc<[u32]>,
        host: GemmaTextHostOwners,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Gemma4Bf16GpuSession)> {
        Self::load_owner(
            GemmaNativeTarget::Hip(crate::backend::hip_native_policy::HipColdOwner::Shared(
                shared,
            )),
            source,
            source_owner,
            ids,
            host,
            GemmaTextStorage::ResidentProjections,
            cancelled,
        )
        .await
    }
    /// One source->inventory->initialization->ResidentModel->six-root->session
    /// body for both original native backends. Vulkan chooses only the target;
    /// it never inherits a HIP fingerprint or policy.
    #[cfg(all(
        feature = "qualification",
        target_os = "linux",
        any(
            feature = "hip",
            all(feature = "vulkan", feature = "gemma-bf16-qualification")
        )
    ))]
    #[allow(clippy::too_many_arguments)]
    async fn load_owner(
        target: GemmaNativeTarget,
        source: Arc<OriginalGemma4Text>,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        ids: Arc<[u32]>,
        host: GemmaTextHostOwners,
        storage: GemmaTextStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Gemma4Bf16GpuSession)> {
        let device = target.device();
        check_cancel(cancelled)?;
        validate_provider(&device)?;
        source
            .source()
            .require_process_host_workspace(target.host_bytes())?;
        let authority = SessionAuthority::new(source.source_identity())?;
        let attempt = authority.begin_advance(ids.len() as u64)?;
        let recorded = Arc::new(source.record_bf16_text_call(attempt, ids.clone(), cancelled)?);
        // Checked complete source-shaped peak precedes context/client/compiler or H2D.
        let cold_inventory = match &target {
            #[cfg(feature = "hip")]
            GemmaNativeTarget::Hip(crate::backend::hip_native_policy::HipColdOwner::Single(_)) => {
                GemmaTextBody::cold_resource_inventory(&recorded, &host, storage)?
            }
            #[cfg(all(feature = "hip", feature = "qwen-source-qualification"))]
            GemmaNativeTarget::Hip(crate::backend::hip_native_policy::HipColdOwner::Shared(_)) => {
                GemmaTextBody::scoped_cold_resource_inventory(&recorded, &host)?
            }
            #[cfg(all(
                feature = "vulkan",
                feature = "gemma-bf16-qualification",
                target_os = "linux"
            ))]
            GemmaNativeTarget::Vulkan(_) => {
                GemmaTextBody::cold_resource_inventory(&recorded, &host, storage)?
            }
        };
        eprintln!("GEMMA_COLD_RESOURCE_INVENTORY storage={storage:?} {cold_inventory:?}");
        let (service, native) = match target {
            #[cfg(feature = "hip")]
            GemmaNativeTarget::Hip(owner) => {
                GemmaTextNativeOwners::prepare_hip_qualification(
                    owner,
                    &recorded,
                    source_owner.clone(),
                    storage,
                    &cold_inventory,
                )
                .await?
            }
            #[cfg(all(
                feature = "vulkan",
                feature = "gemma-bf16-qualification",
                target_os = "linux"
            ))]
            GemmaNativeTarget::Vulkan(device) => {
                GemmaTextNativeOwners::prepare_vulkan_qualification(
                    device,
                    &recorded,
                    source_owner.clone(),
                    storage,
                    &cold_inventory,
                )
                .await?
            }
        };
        let inventory = GemmaTextBody::resource_inventory(&recorded, &host, &native, storage)?;
        if inventory.device_with_six_roots_bytes > service.capacity() {
            return Err(Error::Limit(
                "Gemma full six-root+native original remainder admission".into(),
            ));
        }
        GemmaTextBody::require_capability(service.client())?;
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: inventory.owned_device_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut initialization = service.begin_initialization(0)?;
        let pending = GemmaTextBody::prepare(
            recorded,
            service.clone(),
            &mut initialization,
            &host,
            native,
            storage,
            cancelled,
        )
        .await?;
        let mut sampler = Vec::new();
        for bytes in crate::resident_greedy::workspace_bytes(262144)? {
            sampler.push(initialization.allocate_empty(bytes, BufferRole::Scratch)?);
        }
        check_cancel(cancelled)?;
        let ready = initialization.complete_sync().await?;
        let program = pending.bind()?;
        let preflight = program.budget().clone();
        let mut leases = program.gemma_accesses()?;
        let accesses = leases.clone();
        let greedy = [sampler[0].handle(), sampler[1].handle()];
        leases.extend(sampler.into_iter().map(|l| (l, DeviceAccess::Write)));
        if ready.outputs().len() != leases.len() {
            return Err(Error::State(
                "Gemma original initialized owner coverage".into(),
            ));
        }
        slot.finish()?;
        let model = Arc::new(ResidentModel {
            service,
            workspace: Mutex::new(Workspace {
                program,
                greedy: Some(greedy),
                leases,
                state_accesses: accesses.clone(),
                logit_accesses: accesses,
                batch: None,
            }),
            allocation: Mutex::new(()),
            states: preflight.states,
            state_bytes: preflight.prior_state_bytes,
            capacity: 64,
            vocab: 262144,
            state_pool: std::sync::OnceLock::new(),
            native_batch: false,
            #[cfg(feature = "hardware-abort-probe")]
            abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "hardware-abort-probe")]
            launch_seen: std::sync::atomic::AtomicUsize::new(0),
        });
        model.install_window_root_pool(source.source().clone(), 6)?;
        let domains = model.allocate_state(true).await?;
        let inner = ResidentSession {
            model: model.clone(),
            authority,
            domains,
        };
        let loaded = Arc::new(Self {
            model,
            source,
            _source_owner: source_owner,
            inventory,
            call_lock: Mutex::new(()),
            proposals: Arc::new(async_lock::Semaphore::new(1)),
            sessions: Arc::new(async_lock::Semaphore::new(2)),
            snapshots: Arc::new(async_lock::Semaphore::new(2)),
        });
        let session_permit = loaded
            .sessions
            .try_acquire_arc()
            .ok_or_else(|| Error::State("Gemma initial session owner".into()))?;
        let session = Gemma4Bf16GpuSession {
            loaded: loaded.clone(),
            inner,
            prompt: ids,
            feedback: None,
            limit: 0,
            calls: 0,
            stopped: false,
            _session: session_permit,
        };
        Ok((loaded, session))
    }
    #[cfg(all(
        feature = "qualification",
        target_os = "linux",
        any(
            feature = "hip",
            all(feature = "vulkan", feature = "gemma-bf16-qualification")
        )
    ))]
    pub fn qualification_host_bytes() -> u64 {
        #[cfg(feature = "hip")]
        {
            return crate::backend::hip_native_policy::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES;
        }
        #[cfg(not(feature = "hip"))]
        {
            crate::backend::vulkan_native_policy::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES
        }
    }
    pub async fn traffic(&self) -> Result<crate::resident_program::gemma_text::GemmaTextTraffic> {
        self.model.workspace.lock().await.program.gemma_traffic()
    }
    /// Exact x2 Vulkan/SPIR-V qualification only. The source owner obligations
    /// and single-stream process rule are the same as the original HIP route.
    ///
    /// # Safety
    /// `source_owner` must retain the authentic managed SnapshotLease and source
    /// lifecycle/close guard bound to this exact `source`.
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        target_os = "linux"
    ))]
    pub async unsafe fn load_vulkan_qualification(
        device: Device,
        source: Arc<OriginalGemma4Text>,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        ids: Arc<[u32]>,
        host: GemmaTextHostOwners,
        storage: GemmaTextStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Gemma4Bf16GpuSession)> {
        let storage = match storage {
            GemmaTextStorage::ResidentProjections => GemmaTextStorage::ResidentProjectionsVulkan,
            GemmaTextStorage::Streamed => {
                return Err(Error::Unsupported(
                    "Gemma Vulkan shared/multi-set reload requires resident projections".into(),
                ));
            }
            GemmaTextStorage::ResidentProjectionsVulkan => storage,
        };
        Self::load_owner(
            GemmaNativeTarget::Vulkan(device),
            source,
            source_owner,
            ids,
            host,
            storage,
            cancelled,
        )
        .await
    }
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        target_os = "linux"
    ))]
    pub fn vulkan_native_identity(
        &self,
    ) -> Result<crate::backend::vulkan_native_policy::VulkanSourceSetIdentity> {
        self.model
            .workspace
            .try_lock()
            .ok_or_else(|| Error::State("Gemma workspace busy during native observation".into()))?
            .program
            .gemma_vulkan_native_identity(&self.model.service)
    }
    #[cfg(all(feature = "hip", feature = "qwen-source-qualification"))]
    pub fn native_identity(
        &self,
    ) -> Result<crate::backend::hip_native_policy::HipSourceSetIdentity> {
        self.model
            .workspace
            .try_lock()
            .ok_or_else(|| Error::State("Gemma workspace busy during native observation".into()))?
            .program
            .gemma_native_identity(&self.model.service)
    }
    #[cfg(all(feature = "hip", feature = "qwen-source-qualification"))]
    pub(super) fn source_client(&self) -> &cubecl::client::Client {
        self.model.service.client()
    }
    pub fn inventory(&self) -> &GemmaTextResourceInventory {
        &self.inventory
    }
    pub fn trained_model_qualified(&self) -> bool {
        false
    }
    pub async fn session(self: &Arc<Self>, ids: Arc<[u32]>) -> Result<Gemma4Bf16GpuSession> {
        if ids.is_empty() || ids.len() > 64 || ids.iter().any(|id| *id >= 262144) {
            return Err(Error::State("Gemma genuine prompt IDs/context".into()));
        }
        let permit = self
            .sessions
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("Gemma two independent sessions".into()))?;
        let inner = self.model.session().await?;
        Ok(Gemma4Bf16GpuSession {
            loaded: self.clone(),
            inner,
            prompt: ids,
            feedback: None,
            limit: 0,
            calls: 0,
            stopped: false,
            _session: permit,
        })
    }
}
impl Gemma4Bf16GpuSession {
    pub fn binding(&self) -> RootBinding {
        self.inner.binding()
    }
    pub fn snapshot(&self) -> Result<Gemma4Bf16GpuSnapshot> {
        let permit = self
            .loaded
            .snapshots
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("Gemma two retained snapshots".into()))?;
        Ok(Gemma4Bf16GpuSnapshot {
            inner: self.inner.snapshot(),
            loaded: self.loaded.clone(),
            feedback: self.feedback.clone(),
            limit: self.limit,
            calls: self.calls,
            stopped: self.stopped,
            _snapshot: permit,
        })
    }
    pub async fn restore(&mut self, snapshot: &Gemma4Bf16GpuSnapshot) -> Result<()> {
        let loaded = self.loaded.clone();
        let _call = loaded.call_lock.lock().await;
        if !Arc::ptr_eq(&loaded, &snapshot.loaded) {
            return Err(Error::State(
                "Gemma snapshot belongs to another model".into(),
            ));
        }
        if let Some(token) = &snapshot.feedback
            && (token.producer != Some(snapshot.inner.binding)
                || !Arc::ptr_eq(token.lease.service(), &loaded.model.service)
                || !token.lease.is_initialized())
        {
            return Err(Error::State(
                "Gemma snapshot private feedback/source ownership".into(),
            ));
        }
        self.inner.restore(&snapshot.inner)?;
        self.feedback = snapshot.feedback.clone();
        if let Some(token) = &mut self.feedback {
            token.producer = Some(self.inner.binding());
        }
        self.limit = snapshot.limit;
        self.calls = snapshot.calls;
        self.stopped = snapshot.stopped;
        Ok(())
    }
    pub async fn reset(&mut self) -> Result<()> {
        let loaded = self.loaded.clone();
        let _call = loaded.call_lock.lock().await;
        self.inner.reset().await?;
        self.feedback = None;
        self.limit = 0;
        self.calls = 0;
        self.stopped = false;
        Ok(())
    }
    pub async fn prepare_prompt(
        &self,
        max_output: usize,
        observer: Option<&mut dyn GemmaTextObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4Bf16GpuPending> {
        if self.calls != 0
            || self.binding().frontier() != 0
            || self.feedback.is_some()
            || max_output == 0
            || self
                .prompt
                .len()
                .checked_add(max_output - 1)
                .is_none_or(|n| n > 64)
        {
            return Err(Error::State(
                "Gemma genuine initial prompt/output bound".into(),
            ));
        }
        self.prepare_step(&self.prompt, max_output, observer, cancelled)
            .await
    }
    pub async fn prepare_next(
        &self,
        observer: Option<&mut dyn GemmaTextObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4Bf16GpuPending> {
        check_cancel(cancelled)?;
        let feedback = self.feedback.as_ref().ok_or_else(|| {
            Error::State("Gemma natural continuation needs committed private feedback".into())
        })?;
        if self.stopped
            || self.calls >= self.limit
            || feedback.producer != Some(self.binding())
            || !Arc::ptr_eq(feedback.lease.service(), &self.loaded.model.service)
        {
            return Err(Error::State(
                "Gemma stale/foreign/stopped private feedback".into(),
            ));
        }
        self.prepare_step(&[feedback.id], self.limit, observer, cancelled)
            .await
    }
    async fn prepare_step(
        &self,
        ids: &[u32],
        limit: usize,
        observer: Option<&mut dyn GemmaTextObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Gemma4Bf16GpuPending> {
        check_cancel(cancelled)?;
        // One proposal guard permits the original two-state prompt ping-pong;
        // current2 + private2 + snapshots2 is the complete six-root bound.
        let proposal = self.loaded.proposals.acquire_arc().await;
        check_cancel(cancelled)?;
        // FIFO whole-call serialization precedes every physical flight. The
        // original fair queue still rotates one row's ticket at a time.
        let _call = self.loaded.call_lock.lock().await;
        check_cancel(cancelled)?;
        let observer = std::cell::RefCell::new(observer);
        let inner = self
            .inner
            .prepare_mode_inner(
                ids,
                false,
                Readout::Greedy,
                Some(GemmaOptions {
                    source: self.loaded.source.source(),
                    observer: &observer,
                    cancelled,
                }),
                #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
                None,
            )
            .await?;
        Ok(Gemma4Bf16GpuPending {
            inner,
            loaded: self.loaded.clone(),
            limit,
            _proposal: proposal,
        })
    }
    pub fn commit(&mut self, pending: Gemma4Bf16GpuPending) -> Result<Gemma4Bf16GpuCall> {
        if !Arc::ptr_eq(&self.loaded, &pending.loaded) {
            return Err(Error::State(
                "Gemma proposal belongs to another model".into(),
            ));
        }
        self.inner.validate_prepared(&pending.inner)?;
        let mut token = pending
            .inner
            .readback
            .device_token
            .as_ref()
            .ok_or_else(|| Error::State("Gemma actual private greedy packet missing".into()))?
            .clone();
        if token.producer.is_some()
            || token.id >= 262144
            || !f32::from_bits(token.max_bits).is_finite()
            || !Arc::ptr_eq(token.lease.service(), &self.loaded.model.service)
        {
            return Err(Error::State("Gemma fresh Ready feedback packet".into()));
        }
        let calls = self
            .calls
            .checked_add(1)
            .filter(|&n| n <= pending.limit)
            .ok_or_else(|| Error::State("Gemma pending output limit".into()))?;
        let eos = matches!(token.id, 1 | 106 | 50);
        self.inner.commit(pending.inner)?;
        // Original commit publishes only after all Ready/domain checks and
        // stamps this same packet. Nothing after publication may refuse.
        let root = self.binding();
        token.producer = Some(root);
        self.calls = calls;
        self.limit = pending.limit;
        self.stopped = eos || self.calls == self.limit;
        let result = Gemma4Bf16GpuCall {
            root,
            selected_id: token.id,
            max_bits: token.max_bits,
            eos,
        };
        self.feedback = Some(token);
        Ok(result)
    }
    pub fn stopped(&self) -> bool {
        self.stopped
    }
    pub async fn infer_prompt(
        &mut self,
        max_output: usize,
        observer: Option<&mut dyn GemmaTextObserver>,
        cancelled: Option<&AtomicBool>,
        validate_completed: impl FnOnce(&InferenceReadback) -> Result<()>,
    ) -> Result<Gemma4Bf16GpuCall> {
        if self.calls != 0
            || self.binding().frontier() != 0
            || self.feedback.is_some()
            || max_output == 0
            || self
                .prompt
                .len()
                .checked_add(max_output - 1)
                .is_none_or(|n| n > 64)
        {
            return Err(Error::State(
                "Gemma genuine initial prompt/output bound".into(),
            ));
        }
        let ids = self.prompt.clone();
        self.step(&ids, max_output, observer, cancelled, validate_completed)
            .await
    }
    /// Next IDs come only from this session's actual committed private packet.
    pub async fn infer_next(
        &mut self,
        observer: Option<&mut dyn GemmaTextObserver>,
        cancelled: Option<&AtomicBool>,
        validate_completed: impl FnOnce(&InferenceReadback) -> Result<()>,
    ) -> Result<Gemma4Bf16GpuCall> {
        check_cancel(cancelled)?;
        let feedback = self.feedback.as_ref().ok_or_else(|| {
            Error::State("Gemma natural continuation needs committed private feedback".into())
        })?;
        if self.stopped
            || self.calls >= self.limit
            || feedback.producer != Some(self.binding())
            || !Arc::ptr_eq(feedback.lease.service(), &self.loaded.model.service)
        {
            return Err(Error::State(
                "Gemma stale/foreign/stopped private feedback".into(),
            ));
        }
        let id = feedback.id;
        self.step(&[id], self.limit, observer, cancelled, validate_completed)
            .await
    }
    async fn step(
        &mut self,
        ids: &[u32],
        limit: usize,
        observer: Option<&mut dyn GemmaTextObserver>,
        cancelled: Option<&AtomicBool>,
        validate_completed: impl FnOnce(&InferenceReadback) -> Result<()>,
    ) -> Result<Gemma4Bf16GpuCall> {
        let pending = self.prepare_step(ids, limit, observer, cancelled).await?;
        // True original Ready precedes this callback and logical publication.
        validate_completed(&pending.inner.readback)?;
        check_cancel(cancelled)?;
        self.commit(pending)
    }
}

impl Gemma4Bf16GpuPending {
    pub fn base(&self) -> RootBinding {
        self.inner.attempt.base()
    }
    pub fn readback(&self) -> &InferenceReadback {
        &self.inner.readback
    }
}
impl Gemma4Bf16GpuSnapshot {
    /// Explicit diagnostic read; ordinary inference never calls it.
    pub async fn read_domains(&self) -> Result<Vec<(StateSlot, Vec<f32>)>> {
        self.inner.read_domains().await
    }
    pub fn binding(&self) -> RootBinding {
        self.inner.binding
    }
}
