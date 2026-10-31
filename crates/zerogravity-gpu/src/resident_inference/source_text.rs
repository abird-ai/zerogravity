//! Source-bound single-row model execution under the original root authority.
//! Admission/kernel preparation does not enable a production routing decision.
use super::bf16_text_source::{TextSourceOptions, check_cancel};
use super::*;
pub use crate::resident_program::bf16_text_source::source_set::SourceSetInventory;
use crate::resident_program::bf16_text_source::source_set::SourceStepSet;
use crate::resident_program::bf16_text_source::{Bf16TextSourceBody, TextSourceObserver};
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::TextStepBindings;

pub struct SourceTextCandidate {
    model: Arc<ResidentModel>,
    source: VerifiedTensorSource,
    bodies: Vec<Arc<Bf16TextSourceBody>>,
    inventory: SourceSetInventory,
    #[cfg(feature = "hardware-abort-probe")]
    native_probe: crate::hip::native::FinalNativeProbe,
    _source_owner: crate::physical_quarantine::CompletionOwner,
    sessions: Arc<async_lock::Semaphore>,
    snapshots: Arc<async_lock::Semaphore>,
    proposals: Arc<async_lock::Semaphore>,
    outputs: Arc<async_lock::Semaphore>,
}
#[cfg(feature = "hardware-abort-probe")]
pub struct SourceTextNativeProbe(crate::hip::native::FinalNativeProbe);
#[cfg(feature = "hardware-abort-probe")]
#[derive(Debug)]
pub struct SourceTextNativeState {
    pub retained: bool,
    pub modules: usize,
    pub code_bytes: usize,
    pub host_bytes: usize,
    pub device_bytes: usize,
}
#[cfg(feature = "hardware-abort-probe")]
impl SourceTextNativeProbe {
    pub fn observe(&self) -> Result<SourceTextNativeState> {
        let o = self.0.observe()?;
        Ok(SourceTextNativeState {
            retained: o.retained,
            modules: o.modules,
            code_bytes: o.code_bytes,
            host_bytes: o.host_bytes,
            device_bytes: o.device_bytes,
        })
    }
}
pub struct SourceTextSession {
    inner: ResidentSession,
    loaded: Arc<SourceTextCandidate>,
    _permit: async_lock::SemaphoreGuardArc,
}
pub struct SourceTextSnapshot {
    inner: ResidentSnapshot,
    loaded: Arc<SourceTextCandidate>,
    _permit: async_lock::SemaphoreGuardArc,
}
#[must_use = "dropping a source proposal discards its unpublished root"]
pub struct SourceTextPending {
    inner: PreparedInference,
    loaded: Arc<SourceTextCandidate>,
    host: WindowWorkspace,
    _permit: async_lock::SemaphoreGuardArc,
    output_permit: async_lock::SemaphoreGuardArc,
}
#[cfg(feature = "hardware-abort-probe")]
pub struct SourceTextState {
    domains: Vec<(StateSlot, Vec<f32>)>,
    _host: WindowWorkspace,
    _loaded: Arc<SourceTextCandidate>,
}
#[cfg(feature = "hardware-abort-probe")]
impl SourceTextState {
    pub fn domains(&self) -> &[(StateSlot, Vec<f32>)] {
        &self.domains
    }
}
pub struct SourceTextOutput {
    readback: InferenceReadback,
    _loaded: Arc<SourceTextCandidate>,
    _host: WindowWorkspace,
    _permit: async_lock::SemaphoreGuardArc,
}
impl SourceTextPending {
    pub fn logits(&self) -> &[f32] {
        &self.inner.readback.logits
    }
    pub fn dispatches(&self) -> usize {
        self.inner.readback.dispatches
    }
}
impl SourceTextOutput {
    pub fn logits(&self) -> &[f32] {
        &self.readback.logits
    }
    pub fn dispatches(&self) -> usize {
        self.readback.dispatches
    }
}
impl SourceTextCandidate {
    /// Source/geometry and complete cold pricing only; no native eligibility.
    pub fn preflight(
        source: VerifiedTensorSource,
        records: Vec<Arc<TextStepBindings>>,
        max_sessions: usize,
    ) -> Result<SourceSetInventory> {
        if !(1..=64).contains(&max_sessions) {
            return Err(Error::Limit("source session population".into()));
        }
        Ok(SourceStepSet::new(source, records, 2 * max_sessions + 2)?
            .inventory()
            .clone())
    }
    pub async fn prepare(
        device: cubecl::Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        records: Vec<Arc<TextStepBindings>>,
        max_sessions: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        check_cancel(cancelled)?;
        if !(1..=64).contains(&max_sessions) {
            return Err(Error::Limit("source session population".into()));
        }
        let capacity = records.len();
        let set = SourceStepSet::new(source.clone(), records, 2 * max_sessions + 2)?;
        let inventory = set.inventory().clone();
        let cold = crate::hip::native::begin_resident_source(
            device,
            &source,
            source_owner.clone(),
            inventory.model_device_upper_bytes,
        )?;
        let catalogue = set.catalogue(cold.client())?;
        let service = cold.service();
        let native = cold.prepare(catalogue.0, catalogue.1).await?;
        #[cfg(feature = "hardware-abort-probe")]
        let native_probe = native.final_owner_probe();
        check_cancel(cancelled)?;
        let initialization_host = source.reserve_workspace(32 * 1024 * 1024)?;
        let mut slot = service
            .queue()
            .submit(
                service.budget(),
                service.owner().clone(),
                &[AllocationRequest {
                    kind: AllocationClass::TransferStaging,
                    bytes: (inventory.workspace_bytes + inventory.resident_weight_bytes) as u64,
                }],
                None,
            )?
            .wait()
            .await?;
        slot.begin_submission()?;
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(Arc::new(initialization_host))?;
        let pending = set
            .prepare(service.clone(), &mut init, native, cancelled)
            .await?;
        check_cancel(cancelled)?;
        let ready = init.complete_sync().await?;
        let bodies = pending.bind()?;
        let program = bodies[0].bind()?;
        let accesses = bodies[0].accesses().to_vec();
        if ready.outputs().len() != accesses.len() {
            return Err(Error::State(
                "source shared initialization owner coverage".into(),
            ));
        }
        let preflight = program.budget().clone();
        let vocab = crate::resident_program::bf16_text_source::vocabulary(bodies[0].recorded())?;
        slot.finish()?;
        let model = Arc::new(ResidentModel {
            state_pool: std::sync::OnceLock::new(),
            service,
            workspace: Mutex::new(Workspace {
                program,
                greedy: None,
                leases: accesses.clone(),
                state_accesses: accesses.clone(),
                logit_accesses: accesses,
                batch: None,
            }),
            allocation: Mutex::new(()),
            states: preflight.states,
            state_bytes: preflight.prior_state_bytes,
            capacity,
            vocab: vocab as u64,
            native_batch: false,
            #[cfg(feature = "hardware-abort-probe")]
            abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "hardware-abort-probe")]
            launch_seen: std::sync::atomic::AtomicUsize::new(0),
        });
        model.install_window_root_pool(source.clone(), inventory.root_count)?;
        Ok(Arc::new(Self {
            model,
            source,
            bodies,
            inventory,
            #[cfg(feature = "hardware-abort-probe")]
            native_probe,
            _source_owner: source_owner,
            sessions: Arc::new(async_lock::Semaphore::new(max_sessions)),
            snapshots: Arc::new(async_lock::Semaphore::new(max_sessions)),
            proposals: Arc::new(async_lock::Semaphore::new(1)),
            outputs: Arc::new(async_lock::Semaphore::new(max_sessions)),
        }))
    }
    pub fn resident_bytes(&self) -> Result<usize> {
        self.model.service.resident_bytes()
    }
    pub fn pending_events(&self) -> Result<usize> {
        self.model.service.pending_events()
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub fn native_probe(&self) -> SourceTextNativeProbe {
        SourceTextNativeProbe(self.native_probe.clone())
    }
    /// Close only after every session/snapshot/proposal/output has released ownership.
    pub fn shutdown(self: Arc<Self>) -> Result<()> {
        if Arc::strong_count(&self) != 1 {
            return Err(Error::State("source shutdown has live owners".into()));
        }
        let service = self.model.service.clone();
        service.ensure_usable()?;
        drop(self);
        service.close_model_namespace()
    }
    pub fn inventory(&self) -> &SourceSetInventory {
        &self.inventory
    }
    pub async fn session(self: &Arc<Self>) -> Result<SourceTextSession> {
        let permit = self
            .sessions
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("source session pressure".into()))?;
        let inner = self.model.session().await?;
        Ok(SourceTextSession {
            inner,
            loaded: self.clone(),
            _permit: permit,
        })
    }
}
impl SourceTextSession {
    #[cfg(feature = "hardware-abort-probe")]
    pub async fn diagnostic_state(&self) -> Result<SourceTextState> {
        let bytes = self
            .loaded
            .inventory
            .root_bytes
            .checked_mul(3)
            .and_then(|n| n.checked_add(1024 * 1024))
            .ok_or_else(|| Error::Limit("source root observation host extent".into()))?;
        let host = self.loaded.source.reserve_backend_workspace(bytes)?;
        let domains = self.inner.snapshot().read_domains().await?;
        Ok(SourceTextState {
            domains,
            _host: host,
            _loaded: self.loaded.clone(),
        })
    }
    pub fn binding(&self) -> RootBinding {
        self.inner.binding()
    }
    pub fn snapshot(&self) -> Result<SourceTextSnapshot> {
        let permit = self
            .loaded
            .snapshots
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("source snapshot pressure".into()))?;
        Ok(SourceTextSnapshot {
            inner: self.inner.snapshot(),
            loaded: self.loaded.clone(),
            _permit: permit,
        })
    }
    pub fn restore(&mut self, snapshot: &SourceTextSnapshot) -> Result<()> {
        if !Arc::ptr_eq(&self.loaded, &snapshot.loaded) {
            return Err(Error::State("source foreign snapshot".into()));
        }
        self.inner.restore(&snapshot.inner)
    }
    pub async fn reset(&mut self) -> Result<()> {
        self.inner.reset().await
    }
    pub async fn prepare(
        &self,
        token: u32,
        observer: Option<&mut dyn TextSourceObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<SourceTextPending> {
        check_cancel(cancelled)?;
        let output_permit = self
            .loaded
            .outputs
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("source retained output pressure".into()))?;
        let permit = self.loaded.proposals.acquire_arc().await;
        check_cancel(cancelled)?;
        let host = self.loaded.source.reserve_workspace(32 * 1024 * 1024)?;
        let cell = std::cell::RefCell::new(observer);
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
                    observer: &cell,
                    feedback: None,
                    cancelled,
                }),
            )
            .await?;
        // Model-level full-V observation after actual completion, before publication.
        if let Some(observer) = cell.borrow_mut().as_deref_mut() {
            let bytes = inner
                .readback
                .logits
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            observer.completed(
                crate::resident_program::bf16_text_source::TextSourceFrame {
                    position: self.binding().frontier() as usize,
                    kind: crate::resident_program::bf16_text_source::TextSourceFrameKind::Value {
                        layer: usize::MAX,
                        stage: "logits",
                    },
                    offset_words: 0,
                    total_words: inner.readback.logits.len(),
                },
                &bytes,
            )?;
        }
        check_cancel(cancelled)?;
        Ok(SourceTextPending {
            inner,
            loaded: self.loaded.clone(),
            host,
            _permit: permit,
            output_permit,
        })
    }
    pub fn commit(&mut self, pending: SourceTextPending) -> Result<SourceTextOutput> {
        if !Arc::ptr_eq(&self.loaded, &pending.loaded) {
            return Err(Error::State("source foreign pending owner".into()));
        }
        self.inner.validate_prepared(&pending.inner)?;
        let readback = self.inner.commit(pending.inner)?;
        Ok(SourceTextOutput {
            readback,
            _loaded: pending.loaded,
            _host: pending.host,
            _permit: pending.output_permit,
        })
    }
}
