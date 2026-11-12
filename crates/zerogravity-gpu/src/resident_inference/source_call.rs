//! Source-priced lifetimes around the original resident execution/publication.
use super::*;
#[cfg(all(feature = "hip-source", feature = "hip"))]
use crate::resident_program::source_call as program;
#[cfg(all(feature = "hip-source", feature = "hip"))]
pub use program::SourceCallInventory;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    BoundSourceInvocation, SourceCallMode, SourceNumericalContract,
};

#[cfg(feature = "cuda-cublas")]
mod cohort;
#[cfg(feature = "cuda-cublas")]
pub use cohort::{SourceCallCohortPending, SourceCallCohortPreparation};

#[cfg(all(
    feature = "hardware-abort-probe",
    feature = "qwen-source-qualification",
    feature = "hip-source",
    feature = "hip"
))]
mod probe;
#[cfg(all(
    feature = "hardware-abort-probe",
    feature = "qwen-source-qualification",
    feature = "hip-source",
    feature = "hip"
))]
pub use probe::{SourceCallFinalReader, SourceCallReaderObservation};

const CONTROL_BYTES: usize = 64 * 1024;
const CALL_HOST_BYTES: usize = 32 * 1024 * 1024;

/// No exposed inner model/session: detached readers must retain these owners.
pub struct SourceCallModel {
    inner: Arc<ResidentModel>,
    source: VerifiedTensorSource,
    sessions: Arc<async_lock::Semaphore>,
    _control: WindowWorkspace,
    contract: SourceNumericalContract,
    #[cfg(feature = "cuda-cublas")]
    native_rows: usize,
    #[cfg(feature = "cuda-cublas")]
    prefill_projection_classes:
        Option<crate::resident_program::ordinary_cuda::PrefillProjectionClasses>,
    _source_owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
}
pub struct SourceCallSession {
    inner: ResidentSession,
    _permit: async_lock::SemaphoreGuardArc,
    model: Arc<SourceCallModel>,
    _control: Arc<WindowWorkspace>,
}
struct ReaderScope {
    model: Arc<SourceCallModel>,
    _host: WindowWorkspace,
}
/// Immutable access only: moving a raw Vec out cannot release its credit early.
pub struct SourceCallReadback<T> {
    data: T,
    _scope: Arc<ReaderScope>,
}
impl<T> std::ops::Deref for SourceCallReadback<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.data
    }
}
#[must_use = "dropping a completed proposal discards its private root"]
pub struct SourceCallPending {
    inner: Box<PreparedInference>,
    cancelled: Option<Arc<AtomicBool>>,
    scope: Arc<ReaderScope>,
}
impl SourceCallPending {
    pub fn readback(&self) -> &InferenceReadback {
        &self.inner.readback
    }
}
pub enum SourceCallPreparation {
    /// Proven no submission; the original host/root admission refused.
    Capacity,
    CancelledBeforeLaunch,
    Ready(SourceCallPending),
}
pub enum SourceCallPublication {
    Committed(SourceCallReadback<InferenceReadback>),
    /// Actual completed readback/metrics; no root was published.
    Discarded(SourceCallReadback<InferenceReadback>),
}
struct SnapshotOwner {
    inner: ResidentSnapshot,
    scope: Arc<ReaderScope>,
}
/// Clones share the actual metadata allocation and its one reservation.
#[derive(Clone)]
pub struct SourceCallSnapshot(Arc<SnapshotOwner>);
impl SourceCallSnapshot {
    pub fn binding(&self) -> RootBinding {
        self.0.inner.binding
    }
    pub async fn read_domains(&self) -> Result<SourceCallReadback<Vec<(StateSlot, Vec<f32>)>>> {
        let model = &self.0.scope.model;
        let bytes = model
            .inner
            .state_bytes
            .checked_mul(4)
            .and_then(|n| n.checked_add(CONTROL_BYTES))
            .ok_or_else(|| Error::Limit("source snapshot readback peak".into()))?;
        let scope = model.reader_scope(bytes)?;
        let data = self
            .0
            .inner
            .read_domains_retaining(Some(scope.clone()))
            .await?;
        Ok(SourceCallReadback {
            data,
            _scope: scope,
        })
    }
}

#[cfg(all(feature = "hip-source", feature = "hip"))]
impl ResidentModel {
    pub fn ordinary_cpu_source_preflight(
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
    ) -> Result<SourceCallInventory> {
        Ok(program::Admission::new(source, plan, max_sessions)?
            .inventory()
            .clone())
    }
    pub fn ordinary_cpu_source_preflight_with_budget(
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        budget: crate::hip::native::HipSourceBudget,
    ) -> Result<SourceCallInventory> {
        Ok(
            program::Admission::with_budget(source, plan, max_sessions, budget)?
                .inventory()
                .clone(),
        )
    }
    pub async fn load_ordinary_cpu_source(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<SourceCallModel>> {
        Self::load_ordinary_cpu_source_with_budget(
            device,
            source,
            source_owner,
            plan,
            max_sessions,
            crate::hip::native::HipSourceBudget::default(),
            cancelled,
        )
        .await
    }
    #[allow(clippy::too_many_arguments)]
    pub async fn load_ordinary_cpu_source_with_budget(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        budget: crate::hip::native::HipSourceBudget,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<SourceCallModel>> {
        ordinary_not_cancelled(cancelled)?;
        let control = source.reserve_workspace(CONTROL_BYTES)?;
        let capacity = plan.prefix_capacity();
        let admission =
            program::Admission::with_budget(source.clone(), plan, max_sessions, budget)?;
        let prepared = program::prepare(device, admission, source_owner, cancelled).await?;
        let preflight = prepared.program.budget().clone();
        let vocabulary = prepared.program.plan().values()[prepared
            .program
            .plan()
            .logits()
            .ok_or_else(|| Error::State("source-call readout absent".into()))?
            .0 as usize]
            .capacity_elements()?;
        let model = Arc::new(Self {
            state_pool: std::sync::OnceLock::new(),
            service: prepared.service,
            workspace: Mutex::new(Workspace {
                program: prepared.program,
                greedy: None,
                state_accesses: prepared.accesses.clone(),
                logit_accesses: prepared.accesses.clone(),
                leases: prepared.accesses,
                batch: None,
            }),
            allocation: Mutex::new(()),
            states: preflight.states,
            state_bytes: preflight.prior_state_bytes,
            capacity,
            vocab: vocabulary as u64,
            native_batch: false,
            #[cfg(feature = "hardware-abort-probe")]
            abort_after_launch: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "hardware-abort-probe")]
            launch_seen: std::sync::atomic::AtomicUsize::new(0),
        });
        model.install_window_root_pool(source.clone(), prepared.inventory.root_count)?;
        ordinary_not_cancelled(cancelled)?;
        Ok(Arc::new(SourceCallModel {
            inner: model,
            source,
            sessions: Arc::new(async_lock::Semaphore::new(max_sessions)),
            _control: control,
            contract: SourceNumericalContract::TorchCpuBf16Avx2MixedSourceV1,
            #[cfg(feature = "cuda-cublas")]
            native_rows: 0,
            #[cfg(feature = "cuda-cublas")]
            prefill_projection_classes: None,
            _source_owner: None,
        }))
    }
}
#[cfg(feature = "cuda-cublas")]
impl ResidentModel {
    #[allow(clippy::too_many_arguments)]
    pub async fn load_ordinary_cuda_source(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        native_rows: usize,
        provider_cap: usize,
        device_cap: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<SourceCallModel>> {
        ordinary_not_cancelled(cancelled)?;
        if !(1..=64).contains(&max_sessions)
            || !matches!(native_rows, 0 | 4 | 8)
            || (native_rows != 0 && max_sessions < native_rows)
        {
            return Err(Error::Limit("source-call session capacity".into()));
        }
        let roots = max_sessions * 2 + 2;
        let control = source.reserve_workspace(CONTROL_BYTES)?;
        let prefill_projection_classes = Some(
            crate::resident_program::ordinary_cuda::PrefillProjectionClasses::from_plan(&plan)?,
        );
        let model = Self::load_ordinary_cuda_roots(
            device,
            source.clone(),
            plan,
            provider_cap,
            device_cap,
            roots,
            native_rows,
            Some(source_owner.clone()),
            cancelled,
        )
        .await?;
        model.install_window_root_pool(source.clone(), roots)?;
        ordinary_not_cancelled(cancelled)?;
        Ok(Arc::new(SourceCallModel {
            inner: model,
            source,
            sessions: Arc::new(async_lock::Semaphore::new(max_sessions)),
            _control: control,
            contract: SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            native_rows,
            prefill_projection_classes,
            _source_owner: Some(source_owner),
        }))
    }
}
impl SourceCallModel {
    #[cfg(feature = "cuda-cublas")]
    pub fn prefill_rows_compatible(&self, left: usize, right: usize) -> bool {
        self.native_rows != 0
            && self
                .prefill_projection_classes
                .as_ref()
                .is_some_and(|classes| classes.compatible(left, right))
    }

    fn reader_scope(self: &Arc<Self>, bytes: usize) -> Result<Arc<ReaderScope>> {
        Ok(Arc::new(ReaderScope {
            model: self.clone(),
            _host: self.source.reserve_backend_workspace(bytes)?,
        }))
    }
    pub async fn session(self: &Arc<Self>) -> Result<SourceCallSession> {
        let permit = self
            .sessions
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("source-call session capacity".into()))?;
        let control = Arc::new(self.source.reserve_workspace(CONTROL_BYTES)?);
        let inner = self
            .inner
            .session_with(|init| {
                init.retain_completion_owner(self.clone())?;
                init.retain_completion_owner(control.clone())
            })
            .await?;
        Ok(SourceCallSession {
            inner,
            _permit: permit,
            model: self.clone(),
            _control: control,
        })
    }

    pub async fn shutdown_ordinary_cpu_source(self: Arc<Self>) -> Result<()> {
        self.shutdown().await
    }
    pub async fn shutdown(self: Arc<Self>) -> Result<()> {
        if Arc::strong_count(&self) != 1 || Arc::strong_count(&self.inner) != 1 {
            return Err(Error::State(
                "source-call shutdown has original session/snapshot/result owners".into(),
            ));
        }
        self.inner.service.ensure_usable()?;
        let service = self.inner.service.clone();
        #[cfg(feature = "cuda-cublas")]
        let cuda = {
            let workspace = self.inner.workspace.lock().await;
            let cuda = workspace.program.is_ordinary_cuda();
            if cuda {
                workspace.program.retire_ordinary_cuda().await?;
            }
            cuda
        };
        drop(self);
        #[cfg(feature = "cuda-cublas")]
        if cuda {
            service
                .client()
                .memory_cleanup()
                .map_err(|error| Error::State(format!("source-call CUDA cleanup: {error}")))?;
            if service.resident_bytes()? != 0 || service.model_events()? != 0 {
                return Err(Error::State(
                    "source-call CUDA shutdown left bytes or events".into(),
                ));
            }
        }
        service.close_model_namespace()
    }
    #[cfg(all(
        feature = "hardware-abort-probe",
        feature = "hip-source",
        feature = "hip"
    ))]
    pub async fn ordinary_cpu_source_native_probe(
        &self,
    ) -> Result<super::source_native_probe::SourceTextNativeProbe> {
        let workspace = self.inner.workspace.lock().await;
        Ok(
            super::source_native_probe::SourceTextNativeProbe::from_original(
                workspace.program.source_call_native_probe()?,
                &self.inner.service,
            ),
        )
    }
}
impl SourceCallSession {
    pub fn supports_device_greedy(&self) -> bool {
        #[cfg(feature = "cuda-cublas")]
        {
            matches!(
                self.model.contract,
                SourceNumericalContract::ReferenceCudaSm86Bf16V1
            )
        }
        #[cfg(not(feature = "cuda-cublas"))]
        {
            false
        }
    }
    pub fn binding(&self) -> RootBinding {
        self.inner.binding()
    }
    pub fn snapshot(&self) -> Result<SourceCallSnapshot> {
        let scope = self.model.reader_scope(CONTROL_BYTES)?;
        Ok(SourceCallSnapshot(Arc::new(SnapshotOwner {
            inner: self.inner.snapshot(),
            scope,
        })))
    }
    pub fn restore(&mut self, snapshot: &SourceCallSnapshot) -> Result<()> {
        if !Arc::ptr_eq(&self.model, &snapshot.0.scope.model) {
            return Err(Error::State(
                "source snapshot original model differs".into(),
            ));
        }
        self.inner.restore(&snapshot.0.inner)
    }
    pub async fn reset(&mut self) -> Result<()> {
        // Root initialization already has its own source-priced staging. Keep
        // native/source owners on the actual initialization even on uncertainty.
        self.inner
            .reset_retaining(|init| init.retain_completion_owner(self.model.clone()))
            .await
    }
    pub async fn prepare(
        &self,
        invocation: &BoundSourceInvocation,
        mode: SourceCallMode,
        output: ResidentOutput,
        cancelled: Option<Arc<AtomicBool>>,
    ) -> Result<SourceCallPreparation> {
        let readout = match output {
            ResidentOutput::StateOnly => Readout::None,
            ResidentOutput::Logits => Readout::Logits,
        };
        self.prepare_inner(invocation, mode, output, readout, cancelled, None, None)
            .await
    }
    pub async fn prepare_greedy(
        &self,
        invocation: &BoundSourceInvocation,
        mode: SourceCallMode,
        cancelled: Option<Arc<AtomicBool>>,
    ) -> Result<SourceCallPreparation> {
        if !self.supports_device_greedy() {
            return Err(Error::Unsupported(
                "source-call device greedy requires ordinary CUDA".into(),
            ));
        }
        self.prepare_inner(
            invocation,
            mode,
            ResidentOutput::Logits,
            Readout::Greedy,
            cancelled,
            None,
            None,
        )
        .await
    }
    pub async fn prepare_greedy_device_token(
        &self,
        invocation: &BoundSourceInvocation,
        mode: SourceCallMode,
        token: &ResidentToken,
        cancelled: Option<Arc<AtomicBool>>,
    ) -> Result<SourceCallPreparation> {
        if !self.supports_device_greedy()
            || mode != SourceCallMode::Decode
            || token.producer != Some(self.binding())
            || token.lease.bytes() != 12
            || !Arc::ptr_eq(token.lease.service(), &self.model.inner.service)
            || invocation.invocation().rows() != 1
            || invocation.invocation().ids() != [token.id]
        {
            return Err(Error::State(
                "source-call device token differs from committed decode root/input".into(),
            ));
        }
        self.prepare_inner(
            invocation,
            mode,
            ResidentOutput::Logits,
            Readout::Greedy,
            cancelled,
            Some(token),
            None,
        )
        .await
    }
    pub async fn prepare_observed(
        &self,
        invocation: &BoundSourceInvocation,
        mode: SourceCallMode,
        output: ResidentOutput,
        cancelled: Option<Arc<AtomicBool>>,
        observer: &mut dyn crate::resident_program::source_observation::SourceCallObserver,
    ) -> Result<SourceCallPreparation> {
        let readout = match output {
            ResidentOutput::StateOnly => Readout::None,
            ResidentOutput::Logits => Readout::Logits,
        };
        self.prepare_inner(
            invocation,
            mode,
            output,
            readout,
            cancelled,
            None,
            Some(observer),
        )
        .await
    }
    #[allow(clippy::too_many_arguments)] // Keep publication/readout/token ownership explicit at this boundary.
    async fn prepare_inner(
        &self,
        invocation: &BoundSourceInvocation,
        mode: SourceCallMode,
        output: ResidentOutput,
        readout: Readout,
        cancelled: Option<Arc<AtomicBool>>,
        device_token: Option<&ResidentToken>,
        observer: Option<&mut dyn crate::resident_program::source_observation::SourceCallObserver>,
    ) -> Result<SourceCallPreparation> {
        if cancelled
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Acquire))
        {
            return Ok(SourceCallPreparation::CancelledBeforeLaunch);
        }
        // Credit precedes materialization, access/control containers and readback.
        if invocation.invocation().emit_logits() != (output == ResidentOutput::Logits) {
            return Err(Error::State(
                "source invocation/readout policy differs".into(),
            ));
        }
        let scope = match self.model.reader_scope(CALL_HOST_BYTES) {
            Ok(scope) => scope,
            Err(error) => return preparation_failure(error, false),
        };
        let request = invocation.materialize_source_plan(mode)?;
        let submitted = AtomicBool::new(false);
        let prepared = self
            .inner
            .prepare_ordinary_source_mode(
                request,
                self.model.contract,
                OrdinarySourceOptions {
                    output,
                    readout,
                    device_token: device_token.cloned(),
                    cancelled: cancelled.as_deref(),
                    observer,
                    completion_owner: Some(scope.clone()),
                    submitted: Some(&submitted),
                },
            )
            .await;
        let inner = match prepared {
            Ok(inner) => inner,
            Err(error) => return preparation_failure(error, submitted.load(Ordering::Acquire)),
        };
        Ok(SourceCallPreparation::Ready(SourceCallPending {
            inner: Box::new(inner),
            cancelled,
            scope,
        }))
    }
    pub fn commit(&mut self, pending: SourceCallPending) -> Result<SourceCallPublication> {
        if !Arc::ptr_eq(&self.model, &pending.scope.model) {
            return Err(Error::State(
                "source proposal original model differs".into(),
            ));
        }
        // Fatal/fenced/stale proposals never become a cancellation outcome.
        self.inner.validate_prepared(&pending.inner)?;
        let discarded = pending
            .cancelled
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Acquire));
        let data = if discarded {
            pending.inner.readback
        } else {
            self.inner.commit(*pending.inner)?
        };
        let output = SourceCallReadback {
            data,
            _scope: pending.scope,
        };
        Ok(if discarded {
            SourceCallPublication::Discarded(output)
        } else {
            SourceCallPublication::Committed(output)
        })
    }
}

// Only these original pre-effect refusals permit scheduling outcomes. Unknown
// errors (including cancellation-looking errors after submission) stay fatal.
fn preparation_failure(error: Error, submitted: bool) -> Result<SourceCallPreparation> {
    match error {
        Error::State(reason) if !submitted && reason == "ordinary CUDA request cancelled" => {
            Ok(SourceCallPreparation::CancelledBeforeLaunch)
        }
        Error::Limit(reason)
            if !submitted
                && matches!(
                    reason.as_str(),
                    "physical device memory admission refused"
                        | "resident pool busy"
                        | "resident pool exhausted"
                ) =>
        {
            Ok(SourceCallPreparation::Capacity)
        }
        error => Err(error),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_proven_prelaunch_refusals_become_scheduler_outcomes() {
        assert!(matches!(
            preparation_failure(
                Error::State("ordinary CUDA request cancelled".into()),
                false
            ),
            Ok(SourceCallPreparation::CancelledBeforeLaunch)
        ));
        for reason in [
            "physical device memory admission refused",
            "resident pool busy",
            "resident pool exhausted",
        ] {
            assert!(matches!(
                preparation_failure(Error::Limit(reason.into()), false),
                Ok(SourceCallPreparation::Capacity)
            ));
            assert!(preparation_failure(Error::Limit(reason.into()), true).is_err());
        }
        assert!(
            preparation_failure(Error::State("ordinary CUDA request cancelled".into()), true)
                .is_err()
        );
        for submitted in [false, true] {
            assert!(
                preparation_failure(
                    Error::State("physical device completion uncertain: fenced".into()),
                    submitted
                )
                .is_err()
            );
            assert!(
                preparation_failure(
                    Error::Limit("resource reservation ID exhausted".into()),
                    submitted
                )
                .is_err()
            );
            assert!(
                preparation_failure(Error::Unsupported("source contract".into()), submitted)
                    .is_err()
            );
        }
    }
}
