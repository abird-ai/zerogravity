//! Finite source diagnostic on original ResidentModel/SessionAuthority owners.
use super::*;
mod natural;
use crate::resident_program::qwen_source::{
    QwenSourceBody, QwenSourceFrame, QwenSourceFrameKind, QwenSourceObserver, QwenSourceReadout,
    QwenSourceResourceInventory,
};
pub use natural::{Qwen27bNaturalCall, Qwen27bNaturalPending};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::Bf16TextCall;
use zerogravity_model_qwen35::original_binding::{OriginalWindowOperators, RecordedWindowToken};
#[cfg(feature = "hardware-abort-probe")]
mod final_owner;
#[cfg(feature = "hardware-abort-probe")]
pub use final_owner::QwenFinalOwnerReceipt;

pub(super) struct QwenOptions<'a, 'o> {
    pub bodies: &'a [Arc<QwenSourceBody>; 2],
    pub observer: &'a std::cell::RefCell<Option<&'o mut dyn QwenSourceObserver>>,
    pub feedback: Option<&'a ResidentToken>,
    pub cancelled: Option<&'a AtomicBool>,
}
pub(super) struct QwenStep<'a, 'o> {
    pub body: &'a Arc<QwenSourceBody>,
    pub call: &'a Bf16TextCall,
    pub observer: &'a std::cell::RefCell<Option<&'o mut dyn QwenSourceObserver>>,
    pub feedback: Option<&'a ResidentToken>,
    pub cancelled: Option<&'a AtomicBool>,
}
pub(super) fn check_cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("Qwen source call cancelled".into()));
    }
    Ok(())
}
pub struct Qwen27bSourceGpuCandidate {
    model: Arc<ResidentModel>,
    source: Arc<OriginalWindowOperators>,
    bodies: [Arc<QwenSourceBody>; 2],
    _source_owner: crate::physical_quarantine::CompletionOwner,
    inventory: QwenSourceResourceInventory,
    readout: QwenSourceReadout,
    call_lock: Mutex<()>,
    sessions: Arc<async_lock::Semaphore>,
    proposals: Arc<async_lock::Semaphore>,
    snapshots: Arc<async_lock::Semaphore>,
}
pub struct Qwen27bSourceGpuSession {
    inner: ResidentSession,
    feedback: Option<ResidentToken>,
    natural: natural::Progress,
    // All actual root/feedback readers precede native/source retirement.
    loaded: Arc<Qwen27bSourceGpuCandidate>,
    _permit: async_lock::SemaphoreGuardArc,
}
pub struct Qwen27bSourceGpuPending {
    // Actual root readers retire while the original source/native owner is live.
    inner: PreparedInference,
    loaded: Arc<Qwen27bSourceGpuCandidate>,
    _host: WindowWorkspace,
    _permit: async_lock::SemaphoreGuardArc,
}
pub struct Qwen27bSourceGpuOutput {
    readback: InferenceReadback,
    _loaded: Arc<Qwen27bSourceGpuCandidate>,
    _host: WindowWorkspace,
}
pub struct Qwen27bSourceGpuSnapshot {
    // Actual root readers retire while the original source/native owner is live.
    inner: ResidentSnapshot,
    feedback: Option<ResidentToken>,
    natural: natural::Progress,
    loaded: Arc<Qwen27bSourceGpuCandidate>,
    _permit: async_lock::SemaphoreGuardArc,
}
impl Qwen27bSourceGpuOutput {
    pub fn logits(&self) -> &[f32] {
        &self.readback.logits
    }
    pub fn dispatches(&self) -> usize {
        self.readback.dispatches
    }
}
impl Qwen27bSourceGpuPending {
    pub fn logits(&self) -> &[f32] {
        &self.inner.readback.logits
    }
}
impl Qwen27bSourceGpuCandidate {
    pub fn qualification_host_bytes() -> u64 {
        crate::backend::hip_native_policy::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES
    }
    /// One cold original HIP process, teacher ID11 INITIAL1 then ID353 cached1.
    /// No native/natural/serving qualification follows from successful loading.
    ///
    /// # Safety
    /// `source_owner` must retain the actual managed SnapshotLease and lifecycle
    /// close guard bound to this exact `source`, through all native, flight and
    /// unknown/quarantine owners. Arbitrary Any does not prove this obligation.
    pub async unsafe fn load_hip_qualification(
        device: Device,
        source: Arc<OriginalWindowOperators>,
        source_owner: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        // SAFETY: unchanged retained-source contract delegated to the same loader.
        unsafe {
            Self::load_hip_owner(
                crate::backend::hip_native_policy::HipColdOwner::Single(device),
                source,
                source_owner,
                QwenSourceReadout::TeacherLogits,
                cancelled,
            )
            .await
        }
    }
    /// Separately prepared two-position natural candidate, not a trained permit.
    ///
    /// # Safety
    /// Same authentic managed source/lifecycle owner obligation as
    /// `load_hip_qualification`; retain it through all native/unknown readers.
    pub async unsafe fn load_hip_natural_qualification(
        device: Device,
        source: Arc<OriginalWindowOperators>,
        source_owner: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        // SAFETY: caller supplies the same actual retained source contract.
        unsafe {
            Self::load_hip_owner(
                crate::backend::hip_native_policy::HipColdOwner::Single(device),
                source,
                source_owner,
                QwenSourceReadout::NaturalGreedy,
                cancelled,
            )
            .await
        }
    }
    /// # Safety
    /// Same authentic retained source obligations as `load_hip_qualification`.
    #[cfg(feature = "gemma-bf16-qualification")]
    pub async unsafe fn load_shared_hip_qualification(
        shared: Arc<crate::backend::hip_native_policy::SharedHipNative>,
        source: Arc<OriginalWindowOperators>,
        source_owner: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        // SAFETY: the caller retains the same authentic source obligations.
        unsafe {
            Self::load_hip_owner(
                crate::backend::hip_native_policy::HipColdOwner::Shared(shared),
                source,
                source_owner,
                QwenSourceReadout::TeacherLogits,
                cancelled,
            )
            .await
        }
    }
    async unsafe fn load_hip_owner(
        owner: crate::backend::hip_native_policy::HipColdOwner,
        source: Arc<OriginalWindowOperators>,
        source_owner: crate::physical_quarantine::CompletionOwner,
        readout: QwenSourceReadout,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        #[cfg(feature = "gemma-bf16-qualification")]
        if readout.natural()
            && matches!(
                &owner,
                crate::backend::hip_native_policy::HipColdOwner::Shared(_)
            )
        {
            return Err(Error::Unsupported(
                "shared Qwen natural readout is not an admitted route".into(),
            ));
        }
        let device = owner.device();
        check_cancel(cancelled)?;
        validate_provider(&device)?;
        source
            .source()
            .require_process_host_workspace(owner.host_bytes())?;
        let records: [Arc<RecordedWindowToken>; 2] = [
            Arc::new(source.record_full_source_token(0)?),
            Arc::new(source.record_full_source_token(1)?),
        ];
        let upper = match &owner {
            crate::backend::hip_native_policy::HipColdOwner::Single(_) => {
                crate::resident_program::qwen_source::preparation_inventory(&records, readout)?
            }
            #[cfg(feature = "gemma-bf16-qualification")]
            crate::backend::hip_native_policy::HipColdOwner::Shared(_) => {
                crate::resident_program::qwen_source::scoped_preparation_inventory(
                    &records, readout,
                )?
            }
        };
        if upper.complete_device_bytes > 12 * 1024 * 1024 * 1024 {
            return Err(Error::Limit(
                "Qwen complete cold upper exceeds fixed model remainder".into(),
            ));
        }
        let cold = match &owner {
            crate::backend::hip_native_policy::HipColdOwner::Single(_) => {
                crate::backend::hip_native_policy::begin_qwen_qualification(
                    owner.device(),
                    source.source(),
                    source_owner.clone(),
                )?
            }
            #[cfg(feature = "gemma-bf16-qualification")]
            crate::backend::hip_native_policy::HipColdOwner::Shared(shared) => {
                shared.begin_qwen(source.source(), source_owner.clone(), &upper)?
            }
        };
        let catalogue =
            crate::resident_program::qwen_source::catalogue(cold.client(), &records, readout)?;
        let service = cold.service();
        let native = cold.prepare(catalogue.0, catalogue.1).await?;
        let inventory =
            crate::resident_program::qwen_source::inventory(&records, &native, readout)?;
        if inventory.complete_device_bytes > service.capacity() {
            return Err(Error::Limit(
                "Qwen complete paired-program/six-root/native quota".into(),
            ));
        }
        let initialization_host = source.source().reserve_workspace(32 * 1024 * 1024)?;
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
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(Arc::new(initialization_host))?;
        let first = QwenSourceBody::prepare(
            records[0].clone(),
            service.clone(),
            &mut init,
            native.clone(),
        )?;
        let cached =
            QwenSourceBody::prepare(records[1].clone(), service.clone(), &mut init, native)?;
        let mut sampler = Vec::new();
        if readout.natural() {
            for bytes in crate::resident_greedy::workspace_bytes(
                crate::resident_program::qwen_source::vocabulary(&records[0])?,
            )? {
                sampler.push(init.allocate_empty(bytes, BufferRole::Scratch)?);
            }
        }
        check_cancel(cancelled)?;
        let ready = init.complete_sync().await?;
        let (program, first) = first.bind()?;
        let (_, cached) = cached.bind()?;
        let greedy = if readout.natural() {
            Some([sampler[0].handle(), sampler[1].handle()])
        } else {
            None
        };
        let preflight = program.budget().clone();
        let first_accesses = program.qwen_accesses()?;
        let mut leases = first_accesses.clone();
        leases.extend(cached.accesses().iter().cloned());
        leases.extend(
            sampler
                .into_iter()
                .map(|lease| (lease, DeviceAccess::Write)),
        );
        if ready.outputs().len() != leases.len() {
            return Err(Error::State("source initialized owner coverage".into()));
        }
        slot.finish()?;
        let vocab = records[0].plan().values()[records[0]
            .plan()
            .logits()
            .ok_or_else(|| Error::State("source full vocabulary absent".into()))?
            .0 as usize]
            .capacity_elements()?;
        let model = Arc::new(ResidentModel {
            state_pool: std::sync::OnceLock::new(),
            service,
            workspace: Mutex::new(Workspace {
                program,
                greedy,
                leases,
                state_accesses: first_accesses.clone(),
                logit_accesses: first_accesses,
                batch: None,
            }),
            allocation: Mutex::new(()),
            states: preflight.states,
            state_bytes: preflight.prior_state_bytes,
            capacity: 2,
            vocab: vocab as u64,
            native_batch: false,
            #[cfg(feature = "hardware-abort-probe")]
            abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "hardware-abort-probe")]
            launch_seen: std::sync::atomic::AtomicUsize::new(0),
        });
        model.install_window_root_pool(source.source().clone(), 6)?;
        Ok(Arc::new(Self {
            model,
            source,
            bodies: [first, cached],
            _source_owner: source_owner,
            inventory,
            readout,
            call_lock: Mutex::new(()),
            sessions: Arc::new(async_lock::Semaphore::new(2)),
            proposals: Arc::new(async_lock::Semaphore::new(1)),
            snapshots: Arc::new(async_lock::Semaphore::new(2)),
        }))
    }
    #[cfg(feature = "gemma-bf16-qualification")]
    pub fn native_identity(&self) -> crate::backend::hip_native_policy::HipSourceSetIdentity {
        self.bodies[0].native_identity()
    }
    #[cfg(feature = "gemma-bf16-qualification")]
    pub fn peer_native_view_refused(
        &self,
        peer: &super::gemma_text::Gemma4Bf16GpuCandidate,
    ) -> bool {
        self.bodies[0]
            .require_native_view(peer.source_client())
            .is_err()
    }
    pub fn inventory(&self) -> &QwenSourceResourceInventory {
        &self.inventory
    }
    pub fn prepared_natural_available(&self) -> bool {
        self.readout.natural()
    }
    pub fn trained_model_qualified(&self) -> bool {
        false
    }
    pub async fn session(self: &Arc<Self>) -> Result<Qwen27bSourceGpuSession> {
        let permit = self
            .sessions
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("source two independent sessions".into()))?;
        let inner = self.model.session().await?;
        Ok(Qwen27bSourceGpuSession {
            loaded: self.clone(),
            inner,
            feedback: None,
            natural: natural::Progress::default(),
            _permit: permit,
        })
    }
    pub fn resident_bytes(&self) -> Result<usize> {
        self.model.service.resident_bytes()
    }
    pub fn physical_fenced(&self) -> bool {
        self.model.service.ensure_usable().is_err()
    }
}
impl Qwen27bSourceGpuSession {
    pub fn binding(&self) -> RootBinding {
        self.inner.binding()
    }
    pub fn snapshot(&self) -> Result<Qwen27bSourceGpuSnapshot> {
        let permit = self
            .loaded
            .snapshots
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("source two retained snapshots".into()))?;
        Ok(Qwen27bSourceGpuSnapshot {
            loaded: self.loaded.clone(),
            inner: self.inner.snapshot(),
            feedback: self.feedback.clone(),
            natural: self.natural,
            _permit: permit,
        })
    }
    pub async fn restore(&mut self, snapshot: &Qwen27bSourceGpuSnapshot) -> Result<()> {
        let loaded = self.loaded.clone();
        let _call = loaded.call_lock.lock().await;
        if !Arc::ptr_eq(&loaded, &snapshot.loaded) {
            return Err(Error::State("source foreign model snapshot".into()));
        }
        if let Some(token) = &snapshot.feedback
            && (token.producer != Some(snapshot.inner.binding)
                || !Arc::ptr_eq(token.lease.service(), &loaded.model.service)
                || !token.lease.is_initialized())
        {
            return Err(Error::State(
                "Qwen snapshot original feedback owner differs".into(),
            ));
        }
        self.inner.restore(&snapshot.inner)?;
        self.feedback = snapshot.feedback.clone();
        let restored = self.binding();
        if let Some(token) = &mut self.feedback {
            token.producer = Some(restored);
        }
        self.natural = snapshot.natural;
        Ok(())
    }
    pub async fn reset(&mut self) -> Result<()> {
        let loaded = self.loaded.clone();
        let _call = loaded.call_lock.lock().await;
        self.inner.reset().await?;
        self.feedback = None;
        self.natural = natural::Progress::default();
        Ok(())
    }
    /// Fixed explicit qualification domain; tokens are source IDs, not counts.
    pub async fn prepare_teacher(
        &self,
        observer: &mut dyn QwenSourceObserver,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Qwen27bSourceGpuPending> {
        check_cancel(cancelled)?;
        if self.feedback.is_some() || self.natural.calls != 0 {
            return Err(Error::State(
                "teacher input cannot replace committed natural feedback".into(),
            ));
        }
        let permit = self.loaded.proposals.acquire_arc().await;
        let _call = self.loaded.call_lock.lock().await;
        check_cancel(cancelled)?;
        let position = self.binding().frontier();
        let token = match position {
            0 => 11,
            1 => 353,
            _ => return Err(Error::Unsupported("finite teacher source frontier".into())),
        };
        let host = self
            .loaded
            .source
            .source()
            .reserve_workspace(32 * 1024 * 1024)?;
        let observer_cell = std::cell::RefCell::new(Some(&mut *observer));
        let inner = self
            .inner
            .prepare_mode_inner(
                &[token],
                false,
                Readout::Logits,
                #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
                None,
                Some(QwenOptions {
                    bodies: &self.loaded.bodies,
                    observer: &observer_cell,
                    feedback: None,
                    cancelled,
                }),
            )
            .await?;
        // Full-V comparison follows actual physical completion, before publication.
        let bytes = inner
            .readback
            .logits
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        observer_cell
            .borrow_mut()
            .as_deref_mut()
            .ok_or_else(|| Error::State("source teacher observer absent".into()))?
            .completed(
                QwenSourceFrame {
                    position: position as usize,
                    kind: QwenSourceFrameKind::Value {
                        layer: 63,
                        stage: "logits",
                    },
                    offset_words: 0,
                    total_words: inner.readback.logits.len(),
                },
                &bytes,
            )?;
        check_cancel(cancelled)?;
        Ok(Qwen27bSourceGpuPending {
            loaded: self.loaded.clone(),
            inner,
            _host: host,
            _permit: permit,
        })
    }
    pub fn commit(&mut self, pending: Qwen27bSourceGpuPending) -> Result<Qwen27bSourceGpuOutput> {
        if !Arc::ptr_eq(&self.loaded, &pending.loaded) {
            return Err(Error::State("source foreign pending model".into()));
        }
        self.inner.validate_prepared(&pending.inner)?;
        let readback = self.inner.commit(pending.inner)?;
        Ok(Qwen27bSourceGpuOutput {
            _loaded: pending.loaded,
            readback,
            _host: pending._host,
        })
    }
}
