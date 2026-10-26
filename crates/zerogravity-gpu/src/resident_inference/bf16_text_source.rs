//! Finite source diagnostic on original ResidentModel/SessionAuthority owners.
// Previous names: qwen_source, QwenSource*, Qwen27bSourceGpu*.
use super::*;
mod natural;
use crate::resident_program::bf16_text_source::{
    Bf16TextSourceBody, TextSourceFrame, TextSourceFrameKind, TextSourceObserver,
    TextSourceReadout, TextSourceResourceInventory,
};
pub use natural::{Bf16TextSourceGreedyPending, Bf16TextSourceGreedyResult};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{Bf16TextCall, TextStepBindings};
#[cfg(feature = "hardware-abort-probe")]
mod final_owner;
#[cfg(feature = "hardware-abort-probe")]
pub use final_owner::TextSourceFinalOwnerReceipt;

pub(super) struct TextSourceOptions<'a, 'o> {
    pub bodies: &'a [Arc<Bf16TextSourceBody>; 2],
    pub observer: &'a std::cell::RefCell<Option<&'o mut dyn TextSourceObserver>>,
    pub feedback: Option<&'a ResidentToken>,
    pub cancelled: Option<&'a AtomicBool>,
}
pub(super) struct TextSourceStep<'a, 'o> {
    pub body: &'a Arc<Bf16TextSourceBody>,
    pub call: &'a Bf16TextCall,
    pub observer: &'a std::cell::RefCell<Option<&'o mut dyn TextSourceObserver>>,
    pub feedback: Option<&'a ResidentToken>,
    pub cancelled: Option<&'a AtomicBool>,
}
pub(super) fn check_cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("BF16 text source call cancelled".into()));
    }
    Ok(())
}
pub struct Bf16TextSourceCandidate {
    model: Arc<ResidentModel>,
    source: VerifiedTensorSource,
    bodies: [Arc<Bf16TextSourceBody>; 2],
    _source_owner: crate::physical_quarantine::CompletionOwner,
    inventory: TextSourceResourceInventory,
    readout: TextSourceReadout,
    call_lock: Mutex<()>,
    sessions: Arc<async_lock::Semaphore>,
    proposals: Arc<async_lock::Semaphore>,
    snapshots: Arc<async_lock::Semaphore>,
}
pub struct Bf16TextSourceSession {
    inner: ResidentSession,
    feedback: Option<ResidentToken>,
    natural: natural::Progress,
    // All actual root/feedback readers precede native/source retirement.
    loaded: Arc<Bf16TextSourceCandidate>,
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    probe_issued: AtomicBool,
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    probe_armed: AtomicBool,
    _permit: async_lock::SemaphoreGuardArc,
}
pub struct Bf16TextSourcePending {
    // Actual root readers retire while the original source/native owner is live.
    inner: PreparedInference,
    loaded: Arc<Bf16TextSourceCandidate>,
    _host: WindowWorkspace,
    _permit: async_lock::SemaphoreGuardArc,
}
pub struct Bf16TextSourceResult {
    readback: InferenceReadback,
    _loaded: Arc<Bf16TextSourceCandidate>,
    _host: WindowWorkspace,
}
/// Typed final retention disposition of one source output. It is joined to the
/// actual shared-final native observation; an unknown disposition retains the
/// source context/metadata/native inventory and proves no released-HOST extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextSourceRetentionMode {
    /// Published root: the actual final native/source owners retire.
    Known,
    /// Refused publication: the actual native/source owners remain retained.
    Unknown,
}
impl TextSourceRetentionMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Known => "known",
            Self::Unknown => "unknown",
        }
    }
    pub fn retains_source(self) -> bool {
        matches!(self, Self::Unknown)
    }
    /// Join to the actual shared-final observation: retained flag, live module
    /// count and retained module HOST bytes must agree with the typed mode.
    pub fn agrees_with_final(self, retained: bool, modules: usize, host_bytes: usize) -> bool {
        if self.retains_source() {
            retained && modules > 0 && host_bytes > 0
        } else {
            !retained && modules == 0 && host_bytes == 0
        }
    }
}
pub struct Bf16TextSourceSnapshot {
    // Actual root readers retire while the original source/native owner is live.
    inner: ResidentSnapshot,
    feedback: Option<ResidentToken>,
    natural: natural::Progress,
    loaded: Arc<Bf16TextSourceCandidate>,
    _permit: async_lock::SemaphoreGuardArc,
}
/// Weak/value-only original-owner observation: no strong candidate/service
/// keepalive and no second state owner. Equivalent to the final-reader facts
/// for cached-unknown, where no final wrapper exists.
#[cfg(feature = "hardware-abort-probe")]
#[derive(Clone, Copy, Debug)]
pub struct TextSourceRetentionObservation {
    pub native_admission_retained: bool,
    pub native_modules: usize,
    pub native_code_bytes: usize,
    pub native_host_bytes: usize,
    pub native_device_bytes: usize,
    pub original_service_retained: bool,
    pub quarantined_handles: usize,
}
#[cfg(feature = "hardware-abort-probe")]
impl TextSourceRetentionObservation {
    /// Typed join of this actual weak final-owner observation to the source
    /// disposition: `known` must prove no retained native owner, an
    /// unknown/fault publication must prove retained modules and HOST bytes.
    pub fn retention_agrees(&self, known: bool) -> bool {
        let mode = if known {
            TextSourceRetentionMode::Known
        } else {
            TextSourceRetentionMode::Unknown
        };
        mode.agrees_with_final(
            self.native_admission_retained,
            self.native_modules,
            self.native_host_bytes,
        )
    }
}
#[cfg(feature = "hardware-abort-probe")]
pub struct TextSourceRetentionWitness {
    native: crate::hip::native::FinalNativeProbe,
    service: std::sync::Weak<ResidentService>,
}
#[cfg(feature = "hardware-abort-probe")]
impl TextSourceRetentionWitness {
    pub fn observe(&self) -> Result<TextSourceRetentionObservation> {
        let native = self.native.observe()?;
        let (original_service_retained, quarantined_handles) = match self.service.upgrade() {
            Some(service) => (
                true,
                crate::physical_quarantine::quarantined_handles_for(service.client())?,
            ),
            None => (false, 0),
        };
        Ok(TextSourceRetentionObservation {
            native_admission_retained: native.retained,
            native_modules: native.modules,
            native_code_bytes: native.code_bytes,
            native_host_bytes: native.host_bytes,
            native_device_bytes: native.device_bytes,
            original_service_retained,
            quarantined_handles,
        })
    }
}
impl Bf16TextSourceCandidate {
    #[cfg(feature = "hardware-abort-probe")]
    pub fn retention_witness(&self) -> TextSourceRetentionWitness {
        TextSourceRetentionWitness {
            native: self.bodies[0].final_owner_native_probe(),
            service: Arc::downgrade(&self.model.service),
        }
    }
}
impl Bf16TextSourceSnapshot {
    /// Narrow existing qualification reader; no second state owner or reader.
    pub async fn read_domains(&self) -> Result<Vec<(StateSlot, Vec<f32>)>> {
        self.inner.read_domains().await
    }
}
impl Bf16TextSourceResult {
    pub fn logits(&self) -> &[f32] {
        &self.readback.logits
    }
    pub fn dispatches(&self) -> usize {
        self.readback.dispatches
    }
    /// A committed output published a known root; no native retention is claimed.
    pub fn retention(&self) -> &'static str {
        TextSourceRetentionMode::Known.name()
    }
    /// Companion of `retention` without leaking the crate-private typed mode.
    pub fn retains_source(&self) -> bool {
        TextSourceRetentionMode::Known.retains_source()
    }
}
impl Bf16TextSourcePending {
    pub fn logits(&self) -> &[f32] {
        &self.inner.readback.logits
    }
}
impl Bf16TextSourceCandidate {
    pub fn qualification_host_bytes() -> u64 {
        crate::hip::native::SINGLE_SOURCE_QUALIFICATION_HOST_BYTES
    }
    /// One cold original HIP process, teacher ID11 INITIAL1 then ID353 cached1.
    /// No native/natural/serving qualification follows from successful loading.
    ///
    /// # Safety
    /// `source_owner` must retain the actual managed SnapshotLease and lifecycle
    /// close guard bound to this exact `source`, through all native, flight and
    /// unknown/quarantine owners. Arbitrary Any does not prove this obligation.
    /// Both records must come from the family factory for this exact snapshot,
    /// original positions and numerical/source profile; data alone is not proof.
    pub async unsafe fn load_hip_qualification(
        device: Device,
        source: VerifiedTensorSource,
        records: [Arc<TextStepBindings>; 2],
        source_owner: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        // SAFETY: unchanged retained-source contract delegated to the same loader.
        unsafe {
            Self::load_hip_owner(
                crate::hip::native::HipColdOwner::Single(device),
                source,
                records,
                source_owner,
                TextSourceReadout::TeacherLogits,
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
        source: VerifiedTensorSource,
        records: [Arc<TextStepBindings>; 2],
        source_owner: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        // SAFETY: caller supplies the same actual retained source contract.
        unsafe {
            Self::load_hip_owner(
                crate::hip::native::HipColdOwner::Single(device),
                source,
                records,
                source_owner,
                TextSourceReadout::NaturalGreedy,
                cancelled,
            )
            .await
        }
    }
    /// # Safety
    /// Same authentic retained source obligations as `load_hip_qualification`.
    #[cfg(feature = "gemma-bf16-qualification")]
    pub async unsafe fn load_shared_hip_qualification(
        shared: Arc<crate::hip::native::SharedHipNative>,
        source: VerifiedTensorSource,
        records: [Arc<TextStepBindings>; 2],
        source_owner: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        // SAFETY: the caller retains the same authentic source obligations.
        unsafe {
            Self::load_hip_owner(
                crate::hip::native::HipColdOwner::Shared(shared),
                source,
                records,
                source_owner,
                TextSourceReadout::TeacherLogits,
                cancelled,
            )
            .await
        }
    }
    async unsafe fn load_hip_owner(
        owner: crate::hip::native::HipColdOwner,
        source: VerifiedTensorSource,
        records: [Arc<TextStepBindings>; 2],
        source_owner: crate::physical_quarantine::CompletionOwner,
        readout: TextSourceReadout,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        #[cfg(feature = "gemma-bf16-qualification")]
        if readout.natural() && matches!(&owner, crate::hip::native::HipColdOwner::Shared(_)) {
            return Err(Error::Unsupported(
                "shared BF16 text source natural readout is not an admitted route".into(),
            ));
        }
        let device = owner.device();
        check_cancel(cancelled)?;
        crate::resident_program::bf16_text_source::require_step_pair(&source, &records)?;
        validate_provider(&device)?;
        source.require_process_host_workspace(owner.host_bytes())?;
        let upper = match &owner {
            crate::hip::native::HipColdOwner::Single(_) => {
                crate::resident_program::bf16_text_source::preparation_inventory(
                    &source, &records, readout,
                )?
            }
            #[cfg(feature = "gemma-bf16-qualification")]
            crate::hip::native::HipColdOwner::Shared(_) => {
                crate::resident_program::bf16_text_source::scoped_preparation_inventory(
                    &source, &records, readout,
                )?
            }
        };
        if upper.complete_device_bytes > 12 * 1024 * 1024 * 1024 {
            return Err(Error::Limit(
                "BF16 text source complete cold upper exceeds fixed model remainder".into(),
            ));
        }
        let cold = match &owner {
            crate::hip::native::HipColdOwner::Single(_) => {
                crate::hip::native::begin_qwen_qualification(
                    owner.device(),
                    &source,
                    source_owner.clone(),
                )?
            }
            #[cfg(feature = "gemma-bf16-qualification")]
            crate::hip::native::HipColdOwner::Shared(shared) => {
                shared.begin_qwen(&source, source_owner.clone(), &upper)?
            }
        };
        let catalogue = crate::resident_program::bf16_text_source::catalogue(
            cold.client(),
            &source,
            &records,
            readout,
        )?;
        let service = cold.service();
        let native = cold.prepare(catalogue.0, catalogue.1).await?;
        let inventory = crate::resident_program::bf16_text_source::inventory(
            &source, &records, &native, readout,
        )?;
        if inventory.complete_device_bytes > service.capacity() {
            return Err(Error::Limit(
                "BF16 text source complete paired-program/six-root/native quota".into(),
            ));
        }
        let initialization_host = source.reserve_workspace(32 * 1024 * 1024)?;
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
        let first = Bf16TextSourceBody::prepare(
            source.clone(),
            records[0].clone(),
            service.clone(),
            &mut init,
            native.clone(),
        )?;
        let cached = Bf16TextSourceBody::prepare(
            source.clone(),
            records[1].clone(),
            service.clone(),
            &mut init,
            native,
        )?;
        let mut sampler = Vec::new();
        if readout.natural() {
            for bytes in crate::resident_greedy::workspace_bytes(
                crate::resident_program::bf16_text_source::vocabulary(&records[0])?,
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
        let first_accesses = program.bf16_text_source_accesses()?;
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
        model.install_window_root_pool(source.clone(), 6)?;
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
    pub fn native_identity(&self) -> crate::hip::native::HipSourceSetIdentity {
        self.bodies[0].native_identity()
    }
    #[cfg(feature = "gemma-bf16-qualification")]
    pub fn peer_native_view_refused(&self, peer: &super::bf16_text::Bf16TextCandidate) -> bool {
        self.bodies[0]
            .require_native_view(peer.source_client())
            .is_err()
    }
    pub fn inventory(&self) -> &TextSourceResourceInventory {
        &self.inventory
    }
    pub fn prepared_natural_available(&self) -> bool {
        self.readout.natural()
    }
    pub fn trained_model_qualified(&self) -> bool {
        false
    }
    pub async fn session(self: &Arc<Self>) -> Result<Bf16TextSourceSession> {
        let permit = self
            .sessions
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("source two independent sessions".into()))?;
        let inner = self.model.session().await?;
        Ok(Bf16TextSourceSession {
            loaded: self.clone(),
            inner,
            feedback: None,
            natural: natural::Progress::default(),
            #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
            probe_issued: AtomicBool::new(false),
            #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
            probe_armed: AtomicBool::new(false),
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
impl Bf16TextSourceSession {
    pub fn binding(&self) -> RootBinding {
        self.inner.binding()
    }
    /// Arm exactly one probe on the first actual source body at frontier zero.
    /// The existing call lock serializes this with every prepare/commit; a
    /// previously issued session or a second arm is refused.
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    pub async fn arm_source_page_event_probe(
        &mut self,
        mode: cubecl::hip::SourcePageEventProbeMode,
    ) -> Result<crate::resident_leases::source_page::SourcePageProbeParts> {
        let loaded = self.loaded.clone();
        let _call = loaded.call_lock.lock().await;
        if self.probe_armed.load(Ordering::Acquire) {
            return Err(Error::State("source page probe already armed".into()));
        }
        if self.probe_issued.load(Ordering::Acquire)
            || self.feedback.is_some()
            || self.natural.calls != 0
            || self.inner.binding().frontier() != 0
        {
            return Err(Error::State(
                "source page probe session already issued".into(),
            ));
        }
        let parts = loaded.bodies[0].arm_source_page_event_probe(mode)?;
        self.probe_armed.store(true, Ordering::Release);
        Ok(parts)
    }
    pub fn snapshot(&self) -> Result<Bf16TextSourceSnapshot> {
        let permit = self
            .loaded
            .snapshots
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("source two retained snapshots".into()))?;
        Ok(Bf16TextSourceSnapshot {
            loaded: self.loaded.clone(),
            inner: self.inner.snapshot(),
            feedback: self.feedback.clone(),
            natural: self.natural,
            _permit: permit,
        })
    }
    pub async fn restore(&mut self, snapshot: &Bf16TextSourceSnapshot) -> Result<()> {
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
                "BF16 text source snapshot original feedback owner differs".into(),
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
        observer: &mut dyn TextSourceObserver,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Bf16TextSourcePending> {
        #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
        self.probe_issued.store(true, Ordering::Release);
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
        let host = self.loaded.source.reserve_workspace(32 * 1024 * 1024)?;
        let observer_cell = std::cell::RefCell::new(Some(&mut *observer));
        let inner = self
            .inner
            .prepare_mode_inner(
                &[token],
                false,
                Readout::Logits,
                #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
                None,
                Some(TextSourceOptions {
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
                TextSourceFrame {
                    position: position as usize,
                    kind: TextSourceFrameKind::Value {
                        layer: 63,
                        stage: "logits",
                    },
                    offset_words: 0,
                    total_words: inner.readback.logits.len(),
                },
                &bytes,
            )?;
        check_cancel(cancelled)?;
        Ok(Bf16TextSourcePending {
            loaded: self.loaded.clone(),
            inner,
            _host: host,
            _permit: permit,
        })
    }
    pub fn commit(&mut self, pending: Bf16TextSourcePending) -> Result<Bf16TextSourceResult> {
        if !Arc::ptr_eq(&self.loaded, &pending.loaded) {
            return Err(Error::State("source foreign pending model".into()));
        }
        self.inner.validate_prepared(&pending.inner)?;
        let readback = self.inner.commit(pending.inner)?;
        Ok(Bf16TextSourceResult {
            _loaded: pending.loaded,
            readback,
            _host: pending._host,
        })
    }
}
