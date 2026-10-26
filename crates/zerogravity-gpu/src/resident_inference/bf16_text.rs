//! Candidate binding around the original ResidentModel/SessionAuthority owners.
// Previous names: resident_inference::gemma_text, Gemma4Bf16Gpu*, GemmaText*.
use super::*;
#[cfg(all(feature = "qualification", any(feature = "hip", feature = "vulkan")))]
use crate::resident_program::bf16_text::{
    Bf16TextBody, TextHostOwners, TextNativeOwners, TextStorage,
};
use crate::resident_program::bf16_text::{TextObserver, TextResourceInventory};
use crate::resident_program::text_source::TextSourcePlan;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::{Bf16TextCall, TextPlanBindings};

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
pub use final_owner::TextFinalOwnerReceipt;
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
pub use same_pci::{LearnedSamePciPeerReceipt, TextVulkanNativeRetention};

#[cfg(all(
    feature = "qualification",
    target_os = "linux",
    any(
        feature = "hip",
        all(feature = "vulkan", feature = "gemma-bf16-qualification")
    )
))]
enum TextNativeTarget {
    #[cfg(feature = "hip")]
    Hip(crate::hip::native::HipColdOwner),
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
impl TextNativeTarget {
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
            Self::Vulkan(_) => crate::vulkan::native::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES,
        }
    }
}
pub(super) struct TextOptions<'a, 'o> {
    pub source: &'a VerifiedTensorSource,
    pub bindings: &'a TextPlanBindings,
    pub observer: &'a std::cell::RefCell<Option<&'o mut dyn TextObserver>>,
    pub cancelled: Option<&'a AtomicBool>,
}
pub(super) struct TextStep<'a, 'o> {
    pub call: &'a Bf16TextCall,
    pub row: usize,
    pub observer: &'a std::cell::RefCell<Option<&'o mut dyn TextObserver>>,
    pub cancelled: Option<&'a AtomicBool>,
}
pub(super) fn check_cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State(
            "BF16 text source-bound GPU call cancelled".into(),
        ));
    }
    Ok(())
}

pub struct Bf16TextCandidate {
    model: Arc<ResidentModel>,
    source: VerifiedTensorSource,
    bindings: Arc<TextPlanBindings>,
    _source_owner: Arc<dyn std::any::Any + Send + Sync>,
    inventory: TextResourceInventory,
    call_lock: Mutex<()>,
    proposals: Arc<async_lock::Semaphore>,
    sessions: Arc<async_lock::Semaphore>,
    snapshots: Arc<async_lock::Semaphore>,
}
pub struct Bf16TextSession {
    inner: ResidentSession,
    prompt: Arc<[u32]>,
    feedback: Option<ResidentToken>,
    // Final actual root/feedback readers precede source/native retirement.
    loaded: Arc<Bf16TextCandidate>,
    limit: usize,
    calls: usize,
    stopped: bool,
    _session: async_lock::SemaphoreGuardArc,
}
pub struct Bf16TextPending {
    inner: PreparedInference,
    loaded: Arc<Bf16TextCandidate>,
    limit: usize,
    _proposal: async_lock::SemaphoreGuardArc,
}
pub struct Bf16TextSnapshot {
    inner: ResidentSnapshot,
    feedback: Option<ResidentToken>,
    // Snapshot domains alone do not retain model/native owners.
    loaded: Arc<Bf16TextCandidate>,
    limit: usize,
    calls: usize,
    stopped: bool,
    _snapshot: async_lock::SemaphoreGuardArc,
}
#[derive(Clone, Copy, Debug)]
pub struct Bf16TextResult {
    pub root: RootBinding,
    pub selected_id: u32,
    pub max_bits: u32,
    pub eos: bool,
}
impl Bf16TextCandidate {
    /// Exact x2 HIP qualification only. Bind the managed source under the
    /// backend's canonical16GiB host envelope before entry; retain its snapshot
    /// lease through model retirement. No trained/serving eligibility follows.
    ///
    /// # Safety
    /// `source_owner` must retain the authentic managed SnapshotLease and source
    /// lifecycle/close guard bound to this exact `source`; `bindings` must come from
    /// the family source factory after its finite policy and raw-word checks.
    /// That owner must remain
    /// live through every prepared native, initialization, flight and quarantine
    /// owner. An arbitrary Arc is not proof. Use the validated safe facade.
    #[cfg(all(feature = "qualification", feature = "hip", target_os = "linux"))]
    pub async unsafe fn load_hip_qualification(
        device: Device,
        source: VerifiedTensorSource,
        bindings: Arc<TextPlanBindings>,
        authority: SessionAuthority,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        host: TextHostOwners,
        storage: TextStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Bf16TextSession)> {
        Self::load_owner(
            TextNativeTarget::Hip(crate::hip::native::HipColdOwner::Single(device)),
            source,
            bindings,
            authority,
            source_owner,
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
        shared: Arc<crate::hip::native::SharedHipNative>,
        source: VerifiedTensorSource,
        bindings: Arc<TextPlanBindings>,
        authority: SessionAuthority,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        host: TextHostOwners,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Bf16TextSession)> {
        Self::load_owner(
            TextNativeTarget::Hip(crate::hip::native::HipColdOwner::Shared(shared)),
            source,
            bindings,
            authority,
            source_owner,
            host,
            TextStorage::ResidentProjections,
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
        target: TextNativeTarget,
        source: VerifiedTensorSource,
        bindings: Arc<TextPlanBindings>,
        authority: SessionAuthority,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        host: TextHostOwners,
        storage: TextStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Bf16TextSession)> {
        let device = target.device();
        check_cancel(cancelled)?;
        bindings.call().require_current(authority.binding())?;
        if authority.binding().source() != source.source_identity()
            || bindings.source_identity() != source.source_identity()
            || authority.binding().frontier() != 0
        {
            return Err(Error::State(
                "text cold authority/source/initial frontier differs".into(),
            ));
        }
        validate_provider(&device)?;
        source.require_process_host_workspace(target.host_bytes())?;
        let ids: Arc<[u32]> = Arc::from(bindings.call().ids());
        let recorded = TextSourcePlan::bind(source.clone(), bindings.clone())?;
        // Checked complete source-shaped peak precedes context/client/compiler or H2D.
        let cold_inventory = match &target {
            #[cfg(feature = "hip")]
            TextNativeTarget::Hip(crate::hip::native::HipColdOwner::Single(_)) => {
                Bf16TextBody::cold_resource_inventory(&recorded, &host, storage)?
            }
            #[cfg(all(feature = "hip", feature = "qwen-source-qualification"))]
            TextNativeTarget::Hip(crate::hip::native::HipColdOwner::Shared(_)) => {
                Bf16TextBody::scoped_cold_resource_inventory(&recorded, &host)?
            }
            #[cfg(all(
                feature = "vulkan",
                feature = "gemma-bf16-qualification",
                target_os = "linux"
            ))]
            TextNativeTarget::Vulkan(_) => {
                Bf16TextBody::cold_resource_inventory(&recorded, &host, storage)?
            }
        };
        eprintln!("GEMMA_COLD_RESOURCE_INVENTORY storage={storage:?} {cold_inventory:?}");
        let (service, native) = match target {
            #[cfg(feature = "hip")]
            TextNativeTarget::Hip(owner) => {
                TextNativeOwners::prepare_hip_qualification(
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
            TextNativeTarget::Vulkan(device) => {
                TextNativeOwners::prepare_vulkan_qualification(
                    device,
                    &recorded,
                    source_owner.clone(),
                    storage,
                    &cold_inventory,
                )
                .await?
            }
        };
        let inventory = Bf16TextBody::resource_inventory(&recorded, &host, &native, storage)?;
        if inventory.device_with_six_roots_bytes > service.capacity() {
            return Err(Error::Limit(
                "BF16 text full six-root+native original remainder admission".into(),
            ));
        }
        Bf16TextBody::require_capability(service.client())?;
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
        let pending = Bf16TextBody::prepare(
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
        let mut leases = program.bf16_text_accesses()?;
        let accesses = leases.clone();
        let greedy = [sampler[0].handle(), sampler[1].handle()];
        leases.extend(sampler.into_iter().map(|l| (l, DeviceAccess::Write)));
        if ready.outputs().len() != leases.len() {
            return Err(Error::State(
                "BF16 text original initialized owner coverage".into(),
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
        model.install_window_root_pool(source.clone(), 6)?;
        let domains = model.allocate_state(true).await?;
        let inner = ResidentSession {
            model: model.clone(),
            authority,
            domains,
        };
        let loaded = Arc::new(Self {
            model,
            source,
            bindings,
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
            .ok_or_else(|| Error::State("BF16 text initial session owner".into()))?;
        let session = Bf16TextSession {
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
            return crate::hip::native::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES;
        }
        #[cfg(not(feature = "hip"))]
        {
            crate::vulkan::native::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES
        }
    }
    pub async fn traffic(&self) -> Result<crate::resident_program::bf16_text::TextTraffic> {
        self.model
            .workspace
            .lock()
            .await
            .program
            .bf16_text_traffic()
    }
    /// Exact x2 Vulkan/SPIR-V qualification only. The source owner obligations
    /// and single-stream process rule are the same as the original HIP route.
    ///
    /// # Safety
    /// `source_owner` must retain the authentic managed SnapshotLease and source
    /// lifecycle/close guard bound to this exact `source`; `bindings` must be the
    /// family source factory output with its finite policy and raw-word checks.
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        target_os = "linux"
    ))]
    pub async unsafe fn load_vulkan_qualification(
        device: Device,
        source: VerifiedTensorSource,
        bindings: Arc<TextPlanBindings>,
        authority: SessionAuthority,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        host: TextHostOwners,
        storage: TextStorage,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<Self>, Bf16TextSession)> {
        let storage = match storage {
            TextStorage::ResidentProjections => TextStorage::ResidentProjectionsVulkan,
            TextStorage::Streamed => {
                return Err(Error::Unsupported(
                    "BF16 text Vulkan shared/multi-set reload requires resident projections".into(),
                ));
            }
            TextStorage::ResidentProjectionsVulkan => storage,
        };
        Self::load_owner(
            TextNativeTarget::Vulkan(device),
            source,
            bindings,
            authority,
            source_owner,
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
    pub fn vulkan_native_identity(&self) -> Result<crate::vulkan::native::VulkanSourceSetIdentity> {
        self.model
            .workspace
            .try_lock()
            .ok_or_else(|| {
                Error::State("BF16 text workspace busy during native observation".into())
            })?
            .program
            .bf16_text_vulkan_native_identity(&self.model.service)
    }
    #[cfg(all(feature = "hip", feature = "qwen-source-qualification"))]
    pub fn native_identity(&self) -> Result<crate::hip::native::HipSourceSetIdentity> {
        self.model
            .workspace
            .try_lock()
            .ok_or_else(|| {
                Error::State("BF16 text workspace busy during native observation".into())
            })?
            .program
            .bf16_text_native_identity(&self.model.service)
    }
    #[cfg(all(feature = "hip", feature = "qwen-source-qualification"))]
    pub(super) fn source_client(&self) -> &cubecl::client::Client {
        self.model.service.client()
    }
    pub fn inventory(&self) -> &TextResourceInventory {
        &self.inventory
    }
    pub fn trained_model_qualified(&self) -> bool {
        false
    }
    pub async fn session(self: &Arc<Self>, ids: Arc<[u32]>) -> Result<Bf16TextSession> {
        if ids.is_empty() || ids.len() > 64 || ids.iter().any(|id| *id >= 262144) {
            return Err(Error::State("BF16 text genuine prompt IDs/context".into()));
        }
        let permit = self
            .sessions
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("BF16 text two independent sessions".into()))?;
        let inner = self.model.session().await?;
        Ok(Bf16TextSession {
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
impl Bf16TextSession {
    pub fn binding(&self) -> RootBinding {
        self.inner.binding()
    }
    pub fn snapshot(&self) -> Result<Bf16TextSnapshot> {
        let permit = self
            .loaded
            .snapshots
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("BF16 text two retained snapshots".into()))?;
        Ok(Bf16TextSnapshot {
            inner: self.inner.snapshot(),
            loaded: self.loaded.clone(),
            feedback: self.feedback.clone(),
            limit: self.limit,
            calls: self.calls,
            stopped: self.stopped,
            _snapshot: permit,
        })
    }
    pub async fn restore(&mut self, snapshot: &Bf16TextSnapshot) -> Result<()> {
        let loaded = self.loaded.clone();
        let _call = loaded.call_lock.lock().await;
        if !Arc::ptr_eq(&loaded, &snapshot.loaded) {
            return Err(Error::State(
                "BF16 text snapshot belongs to another model".into(),
            ));
        }
        if let Some(token) = &snapshot.feedback
            && (token.producer != Some(snapshot.inner.binding)
                || !Arc::ptr_eq(token.lease.service(), &loaded.model.service)
                || !token.lease.is_initialized())
        {
            return Err(Error::State(
                "BF16 text snapshot private feedback/source ownership".into(),
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
        observer: Option<&mut dyn TextObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Bf16TextPending> {
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
                "BF16 text genuine initial prompt/output bound".into(),
            ));
        }
        self.prepare_step(&self.prompt, max_output, observer, cancelled)
            .await
    }
    pub async fn prepare_next(
        &self,
        observer: Option<&mut dyn TextObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Bf16TextPending> {
        check_cancel(cancelled)?;
        let feedback = self.feedback.as_ref().ok_or_else(|| {
            Error::State("BF16 text natural continuation needs committed private feedback".into())
        })?;
        if self.stopped
            || self.calls >= self.limit
            || feedback.producer != Some(self.binding())
            || !Arc::ptr_eq(feedback.lease.service(), &self.loaded.model.service)
        {
            return Err(Error::State(
                "BF16 text stale/foreign/stopped private feedback".into(),
            ));
        }
        self.prepare_step(&[feedback.id], self.limit, observer, cancelled)
            .await
    }
    async fn prepare_step(
        &self,
        ids: &[u32],
        limit: usize,
        observer: Option<&mut dyn TextObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Bf16TextPending> {
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
                Some(TextOptions {
                    source: &self.loaded.source,
                    bindings: &self.loaded.bindings,
                    observer: &observer,
                    cancelled,
                }),
                #[cfg(all(feature = "qwen-source-qualification", target_os = "linux"))]
                None,
            )
            .await?;
        Ok(Bf16TextPending {
            inner,
            loaded: self.loaded.clone(),
            limit,
            _proposal: proposal,
        })
    }
    pub fn commit(&mut self, pending: Bf16TextPending) -> Result<Bf16TextResult> {
        if !Arc::ptr_eq(&self.loaded, &pending.loaded) {
            return Err(Error::State(
                "BF16 text proposal belongs to another model".into(),
            ));
        }
        self.inner.validate_prepared(&pending.inner)?;
        let mut token = pending
            .inner
            .readback
            .device_token
            .as_ref()
            .ok_or_else(|| Error::State("BF16 text actual private greedy packet missing".into()))?
            .clone();
        if token.producer.is_some()
            || token.id >= 262144
            || !f32::from_bits(token.max_bits).is_finite()
            || !Arc::ptr_eq(token.lease.service(), &self.loaded.model.service)
        {
            return Err(Error::State("BF16 text fresh Ready feedback packet".into()));
        }
        let calls = self
            .calls
            .checked_add(1)
            .filter(|&n| n <= pending.limit)
            .ok_or_else(|| Error::State("BF16 text pending output limit".into()))?;
        let eos = self.loaded.bindings.end_tokens().contains(&token.id);
        self.inner.commit(pending.inner)?;
        // Original commit publishes only after all Ready/domain checks and
        // stamps this same packet. Nothing after publication may refuse.
        let root = self.binding();
        token.producer = Some(root);
        self.calls = calls;
        self.limit = pending.limit;
        self.stopped = eos || self.calls == self.limit;
        let result = Bf16TextResult {
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
        observer: Option<&mut dyn TextObserver>,
        cancelled: Option<&AtomicBool>,
        validate_completed: impl FnOnce(&InferenceReadback) -> Result<()>,
    ) -> Result<Bf16TextResult> {
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
                "BF16 text genuine initial prompt/output bound".into(),
            ));
        }
        let ids = self.prompt.clone();
        self.step(&ids, max_output, observer, cancelled, validate_completed)
            .await
    }
    /// Next IDs come only from this session's actual committed private packet.
    pub async fn infer_next(
        &mut self,
        observer: Option<&mut dyn TextObserver>,
        cancelled: Option<&AtomicBool>,
        validate_completed: impl FnOnce(&InferenceReadback) -> Result<()>,
    ) -> Result<Bf16TextResult> {
        check_cancel(cancelled)?;
        let feedback = self.feedback.as_ref().ok_or_else(|| {
            Error::State("BF16 text natural continuation needs committed private feedback".into())
        })?;
        if self.stopped
            || self.calls >= self.limit
            || feedback.producer != Some(self.binding())
            || !Arc::ptr_eq(feedback.lease.service(), &self.loaded.model.service)
        {
            return Err(Error::State(
                "BF16 text stale/foreign/stopped private feedback".into(),
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
        observer: Option<&mut dyn TextObserver>,
        cancelled: Option<&AtomicBool>,
        validate_completed: impl FnOnce(&InferenceReadback) -> Result<()>,
    ) -> Result<Bf16TextResult> {
        let pending = self.prepare_step(ids, limit, observer, cancelled).await?;
        // True original Ready precedes this callback and logical publication.
        validate_completed(&pending.inner.readback)?;
        check_cancel(cancelled)?;
        self.commit(pending)
    }
}

impl Bf16TextPending {
    pub fn base(&self) -> RootBinding {
        self.inner.attempt.base()
    }
    pub fn readback(&self) -> &InferenceReadback {
        &self.inner.readback
    }
}
impl Bf16TextSnapshot {
    /// Explicit diagnostic read; ordinary inference never calls it.
    pub async fn read_domains(&self) -> Result<Vec<(StateSlot, Vec<f32>)>> {
        self.inner.read_domains().await
    }
    pub fn binding(&self) -> RootBinding {
        self.inner.binding
    }
}
