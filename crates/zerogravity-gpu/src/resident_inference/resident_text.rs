//! Source-priced lifetimes around the original resident execution/publication.
use super::*;
#[cfg(all(feature = "hip-source", feature = "hip"))]
use crate::resident_program::text_execution as program;
#[cfg(all(feature = "hip-source", feature = "hip"))]
pub use program::ResidentTextInventory;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{
    accelerator::program::{
        BoundInvocation, InvocationPhase, NumericalContract, SourcePlanGeometry,
    },
    identity::SourceIdentity,
    model::ModelProgram,
    numerics::NumericalProfile,
    provider::{
        ExecutionMode, ObservationSupport, PreparedTextProgram, ProviderKind, SelectedMode,
    },
    semantic::InvocationPorts,
};

#[cfg(feature = "cuda-cublas")]
mod cohort;
#[cfg(all(feature = "hip-source", feature = "hip"))]
mod hip_cohort;

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
pub use probe::{ResidentTextFinalReader, ResidentTextReaderObservation};

mod admission;
#[cfg(all(
    feature = "state-checkpoint",
    feature = "hip-source",
    target_os = "linux"
))]
mod checkpoint;
pub use zerogravity_core::provider::ResidentTextAdmission;

const CONTROL_BYTES: usize = 64 * 1024;
const CALL_HOST_BYTES: usize = 32 * 1024 * 1024;

use zerogravity_core::provider::ResidentStatePolicy;
#[cfg(feature = "cuda-cublas")]
pub use zerogravity_core::provider::ResidentStatePolicy as CudaSourceStatePolicy;

#[cfg(feature = "cuda-cublas")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentTextPagedPoolSnapshot {
    pub vector_allocated_sets: usize,
    pub vector_idle_sets: usize,
    pub staging_allocated_sets: usize,
    pub staging_idle_sets: usize,
    pub allocated_pages: usize,
    pub idle_pages: usize,
    pub max_pages: usize,
    pub page_bytes: usize,
}

#[cfg(all(feature = "hardware-abort-probe", feature = "cuda-cublas"))]
pub struct ResidentTextPagedPressure {
    _pages: Vec<ResidentLease>,
}

/// Exact process-local compatibility of one resident text state. The original
/// service generation binds the immutable prepared program and physical state
/// schema. The geometry/ports describe actual token IDs and contiguous absolute
/// positions; tokenizer and chat framing precede this token-space boundary.
/// Equality never grants restore, cross-service transfer or publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentTextStateIdentity {
    service_generation: u64,
    geometry: SourcePlanGeometry,
    selected: SelectedMode,
    numerical_profile: NumericalProfile,
    ports: InvocationPorts,
    storage: ResidentStatePolicy,
}
impl ResidentTextStateIdentity {
    pub fn source(self) -> SourceIdentity {
        self.geometry.source()
    }
    pub fn geometry(self) -> SourcePlanGeometry {
        self.geometry
    }
    pub fn selected(self) -> SelectedMode {
        self.selected
    }
    pub fn numerical_profile(self) -> NumericalProfile {
        self.numerical_profile
    }
    pub fn invocation_ports(self) -> InvocationPorts {
        self.ports
    }
    pub fn state_policy(self) -> ResidentStatePolicy {
        self.storage
    }
}

/// No exposed inner model/session: detached readers must retain these owners.
pub struct ResidentTextModel {
    inner: Arc<ResidentModel>,
    prepared: PreparedTextProgram<VerifiedTensorSource>,
    sessions: Arc<async_lock::Semaphore>,
    _control: WindowWorkspace,
    contract: NumericalContract,
    admission: ResidentTextAdmission,
    state_policy: ResidentStatePolicy,
    #[cfg(feature = "cuda-cublas")]
    prefill_projection_classes:
        Option<crate::resident_program::cuda_forward::PrefillProjectionClasses>,
    _source_owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
}
impl zerogravity_core::provider::PreparedExecutable for ResidentTextModel {
    type Source = VerifiedTensorSource;
    async fn source(&self) -> Result<Self::Source> {
        self.inner.workspace.lock().await.program.window_source()
    }
    async fn program(&self) -> Result<Arc<ForwardPlan>> {
        Ok(self.inner.workspace.lock().await.program.shared_plan())
    }
    fn prepared_text(&self) -> Option<&PreparedTextProgram<Self::Source>> {
        Some(&self.prepared)
    }
}

pub struct ResidentTextSession {
    inner: ResidentSession,
    _permit: async_lock::SemaphoreGuardArc,
    model: Arc<ResidentTextModel>,
    _control: Arc<WindowWorkspace>,
}
struct ReaderScope {
    model: Arc<ResidentTextModel>,
    _host: WindowWorkspace,
}

pub enum ResidentTextCohortPreparation {
    Capacity,
    Ready(ResidentTextCohortPending),
}

#[must_use = "dropping a completed source cohort discards every private root"]
pub struct ResidentTextCohortPending {
    inner: PreparedResidentCohort,
    scopes: Vec<Arc<ReaderScope>>,
}

impl ResidentTextCohortPending {
    pub fn dispatches(&self) -> usize {
        self.inner.dispatches()
    }
    pub fn fused_dispatches(&self) -> usize {
        self.inner.fused_dispatches()
    }
    pub fn native_greedy_dispatches(&self) -> usize {
        self.inner.native_greedy_dispatches()
    }
    pub fn commit(
        self,
        sessions: &mut [&mut ResidentTextSession],
    ) -> Result<Vec<ResidentTextReadback<InferenceReadback>>> {
        if sessions.len() != self.scopes.len()
            || sessions
                .iter()
                .zip(&self.scopes)
                .any(|(session, scope)| !Arc::ptr_eq(&session.model, &scope.model))
        {
            return Err(Error::State(
                "source cohort original session/model coverage differs".into(),
            ));
        }
        let mut inner = sessions
            .iter_mut()
            .map(|session| &mut session.inner)
            .collect::<Vec<_>>();
        let readbacks = self.inner.commit(&mut inner)?;
        Ok(readbacks
            .into_iter()
            .zip(self.scopes)
            .map(|(data, scope)| ResidentTextReadback {
                data,
                _scope: scope,
            })
            .collect())
    }
}

/// Immutable access only: moving a raw Vec out cannot release its credit early.
pub struct ResidentTextReadback<T> {
    data: T,
    _scope: Arc<ReaderScope>,
}
impl<T> std::ops::Deref for ResidentTextReadback<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.data
    }
}
#[must_use = "dropping a completed proposal discards its private root"]
pub struct ResidentTextPending {
    inner: Box<PreparedInference>,
    cancelled: Option<Arc<AtomicBool>>,
    scope: Arc<ReaderScope>,
}
impl ResidentTextPending {
    pub fn readback(&self) -> &InferenceReadback {
        &self.inner.readback
    }
}
pub enum ResidentTextPreparation {
    /// Proven no submission; the original host/root admission refused.
    Capacity,
    CancelledBeforeLaunch,
    Ready(ResidentTextPending),
}
pub enum ResidentTextPublication {
    Committed(ResidentTextReadback<InferenceReadback>),
    /// Actual completed readback/metrics; no root was published.
    Discarded(ResidentTextReadback<InferenceReadback>),
}
struct SnapshotOwner {
    inner: ResidentSnapshot,
    scope: Arc<ReaderScope>,
}
/// Clones share the actual metadata allocation and its one reservation.
#[derive(Clone)]
pub struct ResidentTextSnapshot(Arc<SnapshotOwner>);
impl ResidentTextSnapshot {
    pub fn binding(&self) -> RootBinding {
        self.0.inner.binding
    }
    /// Derived from this snapshot's retained prepared owner, not a caller key.
    pub fn state_identity(&self) -> Result<ResidentTextStateIdentity> {
        self.0.scope.model.state_identity()
    }

    pub async fn read_domains(&self) -> Result<ResidentTextReadback<Vec<(StateSlot, Vec<f32>)>>> {
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
        Ok(ResidentTextReadback {
            data,
            _scope: scope,
        })
    }
}

#[cfg(all(feature = "hip-source", feature = "hip"))]
impl ResidentModel {
    pub fn cpu_forward_source_preflight(
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
    ) -> Result<ResidentTextInventory> {
        Ok(program::Admission::new(source, plan, max_sessions)?
            .inventory()
            .clone())
    }
    pub fn cpu_forward_source_preflight_with_budget(
        source: VerifiedTensorSource,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        budget: crate::hip::native::HipSourceBudget,
    ) -> Result<ResidentTextInventory> {
        Ok(
            program::Admission::with_budget(source, plan, max_sessions, budget)?
                .inventory()
                .clone(),
        )
    }
    pub async fn load_cpu_forward_source(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ResidentTextModel>> {
        Self::load_cpu_forward_source_with_budget(
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
    pub async fn load_cpu_forward_source_with_budget(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        budget: crate::hip::native::HipSourceBudget,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ResidentTextModel>> {
        Self::load_cpu_forward_source_with_budget_and_cohort(
            device,
            source,
            source_owner,
            plan,
            max_sessions,
            0,
            budget,
            cancelled,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn load_cpu_forward_source_with_budget_and_cohort(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        cohort_width: usize,
        budget: crate::hip::native::HipSourceBudget,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ResidentTextModel>> {
        if plan.numerical_profile()
            != zerogravity_core::numerics::NumericalProfile::TorchCpuAvx2Bf16MixedV1
        {
            return Err(Error::Unsupported(
                "exact CPU-source HIP profile required".into(),
            ));
        }
        Self::load_hip_forward_source_with_budget_and_cohort(
            device,
            source,
            source_owner,
            plan,
            max_sessions,
            cohort_width,
            budget,
            cancelled,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn load_hip_forward_source_with_budget_and_cohort(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        cohort_width: usize,
        budget: crate::hip::native::HipSourceBudget,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ResidentTextModel>> {
        use zerogravity_core::numerics::NumericalProfile;
        let contract = match plan.numerical_profile() {
            NumericalProfile::TorchCpuAvx2Bf16MixedV1 => NumericalContract::TorchCpuAvx2Bf16MixedV1,
            NumericalProfile::HipRocBlasBf16MixedV1 => NumericalContract::HipRocBlasBf16MixedV1,
            NumericalProfile::HipRocBlasBf16MixedV2 => NumericalContract::HipRocBlasBf16MixedV2,
            _ => return Err(Error::Unsupported("resident HIP numerical profile".into())),
        };
        if !matches!(cohort_width, 0 | 4 | 8) || cohort_width != 0 && max_sessions < cohort_width {
            return Err(Error::Limit("resident-text HIP cohort capacity".into()));
        }
        forward_not_cancelled(cancelled)?;
        let control = source.reserve_workspace(CONTROL_BYTES)?;
        let capacity = plan.prefix_capacity();
        let executable = prepared_text_contract(
            &source,
            &plan,
            ProviderKind::Hip,
            ObservationSupport::AtProducer,
        )?;
        let native_policy = crate::hip::native::source_native_policy(&device)?;
        let admission = program::Admission::with_budget_and_native_policy(
            source.clone(),
            plan,
            max_sessions,
            budget,
            native_policy,
        )?;
        let prepared = program::prepare(device, admission, source_owner, cancelled).await?;
        require_prepared_text(&executable, &prepared.program)?;
        let admission = admission::hip(
            &prepared.service,
            &prepared.inventory,
            max_sessions,
            cohort_width,
        )?;
        let preflight = prepared.program.budget().clone();
        let vocabulary = prepared.program.plan().values()[prepared
            .program
            .plan()
            .logits()
            .ok_or_else(|| Error::State("resident-text readout absent".into()))?
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
                #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
                forward_cohort_values: None,
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
        forward_not_cancelled(cancelled)?;
        Ok(Arc::new(ResidentTextModel {
            inner: model,
            prepared: executable,
            sessions: Arc::new(async_lock::Semaphore::new(max_sessions)),
            _control: control,
            contract,
            admission,
            state_policy: ResidentStatePolicy::Contiguous,
            #[cfg(feature = "cuda-cublas")]
            prefill_projection_classes: None,
            _source_owner: None,
        }))
    }
}
#[cfg(feature = "cuda-cublas")]
impl ResidentModel {
    fn resident_text_root_count(max_sessions: usize) -> Result<usize> {
        max_sessions
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| Error::Limit("resident-text root count overflow".into()))
    }

    fn resident_text_state_reserve(
        plan: &Arc<ForwardPlan>,
        roots: usize,
        max_sessions: usize,
        state_policy: CudaSourceStatePolicy,
    ) -> Result<Option<usize>> {
        match state_policy {
            CudaSourceStatePolicy::Contiguous => Ok(None),
            CudaSourceStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            } => {
                if page_tokens == 0 || max_append_tokens == 0 {
                    return Err(Error::Unsupported(
                        "paged resident-text state requires positive page/append policy".into(),
                    ));
                }
                let admission =
                    crate::resident_program::cuda_forward::CudaForwardAdmission::analyze(
                        plan.clone(),
                    )?;
                let preflight = admission.preflight();
                super::state_pool::hybrid_state_reserve_bytes(
                    &preflight.states,
                    preflight.prior_state_bytes,
                    roots,
                    max_sessions,
                    page_tokens,
                    max_append_tokens,
                )
                .map(Some)
            }
        }
    }

    /// Pure resident-text admission for one explicit state representation.
    /// No device/context/source owner is created. Smart startup selectors use
    /// this once at model load to compare only qualified physical policies.
    pub fn cuda_forward_source_preflight_for_state_policy(
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        cohort_width: usize,
        provider_cap: usize,
        state_policy: CudaSourceStatePolicy,
    ) -> Result<CudaForwardSourceInventory> {
        if !(1..=64).contains(&max_sessions)
            || !matches!(cohort_width, 0 | 4 | 8)
            || (cohort_width != 0 && max_sessions < cohort_width)
        {
            return Err(Error::Limit("resident-text session capacity".into()));
        }
        let roots = Self::resident_text_root_count(max_sessions)?;
        let state_reserve_bytes =
            Self::resident_text_state_reserve(&plan, roots, max_sessions, state_policy)?;
        super::cuda_forward_source_admission(
            plan,
            provider_cap,
            roots,
            cohort_width,
            state_reserve_bytes,
        )
        .map(|(_, inventory)| inventory)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn load_cuda_forward_source(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        cohort_width: usize,
        provider_cap: usize,
        device_cap: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ResidentTextModel>> {
        Self::load_cuda_forward_source_with_state_policy(
            device,
            source,
            source_owner,
            plan,
            max_sessions,
            cohort_width,
            provider_cap,
            device_cap,
            CudaSourceStatePolicy::Contiguous,
            cancelled,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn load_cuda_forward_source_with_state_policy(
        device: Device,
        source: VerifiedTensorSource,
        source_owner: Arc<dyn std::any::Any + Send + Sync>,
        plan: Arc<ForwardPlan>,
        max_sessions: usize,
        cohort_width: usize,
        provider_cap: usize,
        device_cap: u64,
        state_policy: CudaSourceStatePolicy,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ResidentTextModel>> {
        forward_not_cancelled(cancelled)?;
        if !(1..=64).contains(&max_sessions)
            || !matches!(cohort_width, 0 | 4 | 8)
            || (cohort_width != 0 && max_sessions < cohort_width)
        {
            return Err(Error::Limit("resident-text session capacity".into()));
        }
        let roots = Self::resident_text_root_count(max_sessions)?;
        let state_reserve_bytes =
            Self::resident_text_state_reserve(&plan, roots, max_sessions, state_policy)?;
        let control = source.reserve_workspace(CONTROL_BYTES)?;
        let prefill_projection_classes = Some(
            crate::resident_program::cuda_forward::PrefillProjectionClasses::from_plan(&plan)?,
        );
        let executable =
            prepared_text_contract(&source, &plan, ProviderKind::Cuda, ObservationSupport::None)?;
        let (model, inventory) = Self::load_cuda_forward_roots(
            device,
            source.clone(),
            plan,
            provider_cap,
            device_cap,
            roots,
            cohort_width,
            state_reserve_bytes,
            Some(source_owner.clone()),
            cancelled,
        )
        .await?;
        let admission = admission::cuda(&model.service, &inventory, provider_cap, max_sessions)?;
        match state_policy {
            CudaSourceStatePolicy::Contiguous => {
                model.install_window_root_pool(source.clone(), roots)?;
            }
            CudaSourceStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            } => {
                model.install_window_hybrid_state_pools(
                    source.clone(),
                    roots,
                    max_sessions,
                    page_tokens,
                    max_append_tokens,
                    NumericalContract::CudaSm86Bf16ReferenceV1,
                )?;
            }
        }
        forward_not_cancelled(cancelled)?;
        {
            let workspace = model.workspace.lock().await;
            require_prepared_text(&executable, &workspace.program)?;
        }
        Ok(Arc::new(ResidentTextModel {
            inner: model,
            prepared: executable,
            sessions: Arc::new(async_lock::Semaphore::new(max_sessions)),
            _control: control,
            contract: NumericalContract::CudaSm86Bf16ReferenceV1,
            admission,
            state_policy,
            prefill_projection_classes,
            _source_owner: Some(source_owner),
        }))
    }
}
fn prepared_text_contract(
    source: &VerifiedTensorSource,
    plan: &Arc<ForwardPlan>,
    provider: ProviderKind,
    observations: ObservationSupport,
) -> Result<PreparedTextProgram<VerifiedTensorSource>> {
    let max_rows = plan
        .source_plan_geometry()
        .ok_or_else(|| Error::State("resident source template absent".into()))?
        .max_rows();
    PreparedTextProgram::new(
        ModelProgram::from_recorded(source.clone(), plan.clone())?,
        SelectedMode {
            provider,
            mode: ExecutionMode::Native,
        },
        1..=max_rows,
        observations,
    )
}

fn require_prepared_text(
    prepared: &PreparedTextProgram<VerifiedTensorSource>,
    actual: &ResidentProgram,
) -> Result<()> {
    if !Arc::ptr_eq(prepared.model().program(), &actual.shared_plan())
        || !prepared
            .model()
            .source()
            .same_issuer(&actual.window_source()?)
    {
        return Err(Error::State(
            "resident text prepared owner differs from actual body".into(),
        ));
    }
    Ok(())
}

impl ResidentTextModel {
    /// Read-only effective cold fit, from the original loaded service.
    pub fn admission(&self) -> &ResidentTextAdmission {
        &self.admission
    }
    /// Current leases in the original model namespace, not device free memory.
    pub fn resident_bytes(&self) -> Result<usize> {
        self.inner.service.resident_bytes()
    }
    pub fn state_policy(&self) -> ResidentStatePolicy {
        self.state_policy
    }
    /// Selected retained region observation from the original physical owner.
    pub async fn describe_region(
        &self,
        node: usize,
    ) -> Result<Option<zerogravity_core::provider::ExecutionDescription>> {
        self.inner.describe_region(node).await
    }
    /// Allocation-free observation of the original prepared state contract.
    /// The loader proves exact plan/source binding with require_prepared_text.
    pub fn state_identity(&self) -> Result<ResidentTextStateIdentity> {
        let plan = self.prepared.model().program();
        let geometry = plan
            .source_plan_geometry()
            .ok_or_else(|| Error::State("resident text state has no original template".into()))?;
        if geometry.source() != self.inner.service.source_identity()
            || geometry.context_capacity() != self.inner.capacity
            || geometry.vocabulary() as u64 != self.inner.vocab
            || plan.numerical_profile() != self.contract.program_profile()
        {
            return Err(Error::State(
                "resident text state differs from prepared program".into(),
            ));
        }
        Ok(ResidentTextStateIdentity {
            service_generation: self.inner.service.generation(),
            geometry,
            selected: self.prepared.selected(),
            numerical_profile: plan.numerical_profile(),
            ports: self.prepared.invocation_ports(),
            storage: self.state_policy,
        })
    }

    fn bind_invocation<'a>(
        &self,
        invocation: &'a BoundInvocation,
        phase: InvocationPhase,
        observed: bool,
    ) -> Result<zerogravity_core::semantic::BoundTextCall<'a>> {
        self.prepared.bind(invocation, phase, observed)
    }
    #[cfg(feature = "cuda-cublas")]
    pub fn supports_shared_prefix(&self) -> bool {
        matches!(self.state_policy, CudaSourceStatePolicy::Paged { .. })
    }

    #[cfg(any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip")))]
    pub fn prefill_rows_compatible(&self, left: usize, right: usize) -> bool {
        if self.admission.cohort_width == 0 {
            return false;
        }
        #[cfg(all(feature = "hip-source", feature = "hip"))]
        if self.contract.uses_torch_mixed_operators() {
            // A one-token prompt tail uses singleton decode arithmetic.
            return left == right && (2..=64).contains(&left);
        }
        #[cfg(feature = "cuda-cublas")]
        {
            return self
                .prefill_projection_classes
                .as_ref()
                .is_some_and(|classes| classes.compatible(left, right));
        }
        #[allow(unreachable_code)]
        false
    }

    #[cfg(any(feature = "cuda-cublas", all(feature = "hip-source", feature = "hip")))]
    pub async fn prepare_prefill_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentTextSession, &BoundInvocation)],
    ) -> Result<ResidentTextCohortPreparation> {
        #[cfg(all(feature = "hip-source", feature = "hip"))]
        if self.contract.uses_torch_mixed_operators() {
            return self
                .prepare_hip_cohort_admitted(
                    rows,
                    InvocationPhase::Prefill,
                    self.supports_device_greedy(),
                    None,
                )
                .await;
        }
        #[cfg(feature = "cuda-cublas")]
        {
            return self.prepare_cuda_prefill_cohort_admitted(rows).await;
        }
        #[allow(unreachable_code)]
        Err(Error::Unsupported("source prefill cohort backend".into()))
    }

    /// Compact readout and committed-token feedback are part of the selected
    /// original native route, not of a host-side generation loop.
    fn supports_device_greedy(&self) -> bool {
        match self.contract {
            #[cfg(feature = "cuda-cublas")]
            NumericalContract::CudaSm86Bf16ReferenceV1 => true,
            #[cfg(all(feature = "hip-source", feature = "hip"))]
            NumericalContract::HipRocBlasBf16MixedV2 => true,
            _ => false,
        }
    }

    pub async fn prepare_greedy_decode_cohort_admitted(
        self: &Arc<Self>,
        rows: &[(&ResidentTextSession, &BoundInvocation, &ResidentToken)],
    ) -> Result<ResidentTextCohortPreparation> {
        #[cfg(all(feature = "hip-source", feature = "hip"))]
        if self.contract.uses_torch_mixed_operators() {
            let inputs = rows
                .iter()
                .map(|(session, invocation, _)| (*session, *invocation))
                .collect::<Vec<_>>();
            let tokens = rows.iter().map(|(_, _, token)| *token).collect::<Vec<_>>();
            return self
                .prepare_hip_cohort_admitted(&inputs, InvocationPhase::Decode, true, Some(&tokens))
                .await;
        }
        #[cfg(feature = "cuda-cublas")]
        {
            return self.prepare_cuda_greedy_decode_cohort_admitted(rows).await;
        }
        #[allow(unreachable_code)]
        Err(Error::Unsupported(
            "source device-token cohort backend".into(),
        ))
    }

    fn reader_scope(self: &Arc<Self>, bytes: usize) -> Result<Arc<ReaderScope>> {
        Ok(Arc::new(ReaderScope {
            model: self.clone(),
            _host: self
                .prepared
                .model()
                .source()
                .reserve_backend_workspace(bytes)?,
        }))
    }
    pub async fn session(self: &Arc<Self>) -> Result<ResidentTextSession> {
        let permit = self
            .sessions
            .try_acquire_arc()
            .ok_or_else(|| Error::Limit("resident-text session capacity".into()))?;
        let control = Arc::new(
            self.prepared
                .model()
                .source()
                .reserve_workspace(CONTROL_BYTES)?,
        );
        let inner = self
            .inner
            .session_with(|init| {
                init.retain_completion_owner(self.clone())?;
                init.retain_completion_owner(control.clone())
            })
            .await?;
        Ok(ResidentTextSession {
            inner,
            _permit: permit,
            model: self.clone(),
            _control: control,
        })
    }

    pub async fn shutdown_cpu_forward_source(self: Arc<Self>) -> Result<()> {
        self.shutdown().await
    }
    pub async fn shutdown(self: Arc<Self>) -> Result<()> {
        if Arc::strong_count(&self) != 1 || Arc::strong_count(&self.inner) != 1 {
            return Err(Error::State(
                "resident-text shutdown has original session/snapshot/result owners".into(),
            ));
        }
        self.inner.service.ensure_usable()?;
        let service = self.inner.service.clone();
        #[cfg(feature = "cuda-cublas")]
        {
            let workspace = self.inner.workspace.lock().await;
            if workspace.program.is_cuda_forward() {
                workspace.program.retire_cuda_forward().await?;
            }
        }
        drop(self);
        // The last model values may drop after native module retirement.
        // Drain the original allocator before returning the model's quota.
        service.cleanup_and_close_model()
    }

    #[cfg(feature = "hardware-abort-probe")]
    pub fn diagnostic_pause_next_launch(&self) -> Result<()> {
        let _idle = self.inner.workspace.try_lock().ok_or_else(|| {
            Error::State("resident-text launch probe requires idle workspace".into())
        })?;
        self.inner.service.ensure_usable()?;
        if self.inner.qualification_launch_seen() != 0 {
            return Err(Error::State(
                "resident-text launch probe already reached".into(),
            ));
        }
        self.inner
            .abort_after_launch
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("resident-text launch probe already armed".into()))?;
        Ok(())
    }

    #[cfg(feature = "hardware-abort-probe")]
    pub fn diagnostic_launch_handles(&self) -> usize {
        self.inner.qualification_launch_seen()
    }

    #[cfg(feature = "hardware-abort-probe")]
    pub fn diagnostic_resume_launch(&self) {
        self.inner.qualification_reset_launch_seen();
    }

    #[cfg(feature = "hardware-abort-probe")]
    pub fn diagnostic_pending_events(&self) -> Result<usize> {
        self.inner.service.pending_events()
    }

    #[cfg(feature = "hardware-abort-probe")]
    pub fn diagnostic_service_usable(&self) -> bool {
        self.inner.service.ensure_usable().is_ok()
    }

    #[cfg(all(feature = "hardware-abort-probe", feature = "cuda-cublas"))]
    pub fn diagnostic_quarantined_handles(&self) -> Result<usize> {
        crate::physical_quarantine::quarantined_handles_for(self.inner.service.client())
    }

    #[cfg(all(feature = "hardware-abort-probe", feature = "cuda-cublas"))]
    pub fn diagnostic_quarantined_completion_owners(&self) -> Result<usize> {
        crate::physical_quarantine::quarantined_completion_owners_for(self.inner.service.client())
    }

    #[cfg(feature = "cuda-cublas")]
    pub fn paged_pool_snapshot(&self) -> Result<Option<ResidentTextPagedPoolSnapshot>> {
        let states = self
            .inner
            .state_pool
            .get()
            .ok_or_else(|| Error::State("resident-text state pool absent".into()))?;
        let super::state_pool::PooledStateStorage::Hybrid(pools) = &states.storage else {
            return Ok(None);
        };
        let vector = pools.vector_pool.snapshot()?;
        let staging = pools.staging_pool.snapshot()?;
        let pages = pools.page_pool.snapshot()?;
        Ok(Some(ResidentTextPagedPoolSnapshot {
            vector_allocated_sets: vector.allocated_sets,
            vector_idle_sets: vector.idle_sets,
            staging_allocated_sets: staging.allocated_sets,
            staging_idle_sets: staging.idle_sets,
            allocated_pages: pages.allocated_pages,
            idle_pages: pages.idle_pages,
            max_pages: pools.plan.max_pages,
            page_bytes: pools.page_pool.page_bytes(),
        }))
    }

    #[cfg(all(feature = "hardware-abort-probe", feature = "cuda-cublas"))]
    pub fn diagnostic_paged_pool_snapshot(&self) -> Result<ResidentTextPagedPoolSnapshot> {
        self.paged_pool_snapshot()?
            .ok_or_else(|| Error::State("resident-text state pool is not paged".into()))
    }

    #[cfg(all(feature = "hardware-abort-probe", feature = "cuda-cublas"))]
    pub fn diagnostic_hold_paged_pages(&self, count: usize) -> Result<ResidentTextPagedPressure> {
        let states = self
            .inner
            .state_pool
            .get()
            .ok_or_else(|| Error::State("resident-text state pool absent".into()))?;
        let super::state_pool::PooledStateStorage::Hybrid(pools) = &states.storage else {
            return Err(Error::State("resident-text state pool is not paged".into()));
        };
        let mut pages = Vec::with_capacity(count);
        for _ in 0..count {
            pages.push(pools.page_pool.checkout()?);
        }
        Ok(ResidentTextPagedPressure { _pages: pages })
    }

    #[cfg(all(
        feature = "hardware-abort-probe",
        feature = "hip-source",
        feature = "hip"
    ))]
    pub async fn cpu_forward_source_native_probe(
        &self,
    ) -> Result<super::source_native_probe::SourceTextNativeProbe> {
        let workspace = self.inner.workspace.lock().await;
        Ok(
            super::source_native_probe::SourceTextNativeProbe::from_original(
                workspace.program.text_execution_native_probe()?,
                &self.inner.service,
            ),
        )
    }
}
impl ResidentTextSession {
    pub fn supports_device_greedy(&self) -> bool {
        self.model.supports_device_greedy()
    }

    /// HIP's current cohort is aligned; CUDA can accept distinct decode
    /// frontiers. This constraint is independent of the readout format.
    pub fn decode_frontiers_compatible(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.model, &other.model)
            && (!self.model.contract.uses_torch_mixed_operators()
                || self.binding().frontier() == other.binding().frontier())
    }

    fn require_device_feedback(
        &self,
        invocation: &BoundInvocation,
        token: &ResidentToken,
    ) -> Result<()> {
        if !self.supports_device_greedy()
            || token.producer != Some(self.binding())
            || token.lease.bytes() != 12
            || !Arc::ptr_eq(token.lease.service(), &self.model.inner.service)
            || invocation.input().token_count() != 1
            || invocation.input().ids() != [token.id]
        {
            return Err(Error::State(
                "resident-text device token differs from committed decode root/input".into(),
            ));
        }
        Ok(())
    }
    pub fn binding(&self) -> RootBinding {
        self.inner.binding()
    }
    pub fn state_identity(&self) -> Result<ResidentTextStateIdentity> {
        self.model.state_identity()
    }
    #[cfg(feature = "cuda-cublas")]
    pub fn supports_shared_prefix(&self) -> bool {
        self.model.supports_shared_prefix()
    }
    pub fn snapshot(&self) -> Result<ResidentTextSnapshot> {
        // Keep the original control allowance and admit the actual variable
        // backing copied by a complete-domain snapshot. No new device pages.
        let bytes = CONTROL_BYTES
            .checked_add(self.inner.snapshot_backing_bytes()?)
            .ok_or_else(|| Error::Limit("source snapshot metadata overflow".into()))?;
        let scope = self.model.reader_scope(bytes)?;
        Ok(ResidentTextSnapshot(Arc::new(SnapshotOwner {
            inner: self.inner.snapshot(),
            scope,
        })))
    }
    pub fn restore(&mut self, snapshot: &ResidentTextSnapshot) -> Result<()> {
        if !Arc::ptr_eq(&self.model, &snapshot.0.scope.model) {
            return Err(Error::State(
                "source snapshot original model differs".into(),
            ));
        }
        self.inner.restore(&snapshot.0.inner)
    }
    pub fn fork_shared_prefix(&mut self, snapshot: &ResidentTextSnapshot) -> Result<()> {
        if !Arc::ptr_eq(&self.model, &snapshot.0.scope.model) {
            return Err(Error::State(
                "source shared prefix original model differs".into(),
            ));
        }
        self.inner.fork_shared_prefix(&snapshot.0.inner)
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
        invocation: &BoundInvocation,
        mode: InvocationPhase,
        output: ResidentOutput,
        cancelled: Option<Arc<AtomicBool>>,
    ) -> Result<ResidentTextPreparation> {
        let readout = match output {
            ResidentOutput::StateOnly => Readout::None,
            ResidentOutput::Logits => Readout::Logits,
        };
        self.prepare_inner(invocation, mode, output, readout, cancelled, None, None)
            .await
    }
    pub async fn prepare_greedy(
        &self,
        invocation: &BoundInvocation,
        mode: InvocationPhase,
        cancelled: Option<Arc<AtomicBool>>,
    ) -> Result<ResidentTextPreparation> {
        if !self.supports_device_greedy() {
            return Err(Error::Unsupported(
                "resident-text source has no prepared device greedy readout".into(),
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
        invocation: &BoundInvocation,
        mode: InvocationPhase,
        token: &ResidentToken,
        cancelled: Option<Arc<AtomicBool>>,
    ) -> Result<ResidentTextPreparation> {
        if mode != InvocationPhase::Decode {
            return Err(Error::State(
                "source device feedback requires decode".into(),
            ));
        }
        self.require_device_feedback(invocation, token)?;
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
        invocation: &BoundInvocation,
        mode: InvocationPhase,
        output: ResidentOutput,
        cancelled: Option<Arc<AtomicBool>>,
        observer: &mut dyn crate::resident_program::text_observation::TextObserver,
    ) -> Result<ResidentTextPreparation> {
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
        invocation: &BoundInvocation,
        mode: InvocationPhase,
        output: ResidentOutput,
        readout: Readout,
        cancelled: Option<Arc<AtomicBool>>,
        device_token: Option<&ResidentToken>,
        observer: Option<&mut dyn crate::resident_program::text_observation::TextObserver>,
    ) -> Result<ResidentTextPreparation> {
        if cancelled
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Acquire))
        {
            return Ok(ResidentTextPreparation::CancelledBeforeLaunch);
        }
        let call = self
            .model
            .bind_invocation(invocation, mode, observer.is_some())?;
        // Credit precedes materialization, access/control containers and readback.
        if invocation.input().emit_logits() != (output == ResidentOutput::Logits) {
            return Err(Error::State(
                "source invocation/readout policy differs".into(),
            ));
        }
        let scope = match self.model.reader_scope(CALL_HOST_BYTES) {
            Ok(scope) => scope,
            Err(error) => return preparation_failure(error, false),
        };
        let request = call.materialize()?;
        let submitted = AtomicBool::new(false);
        let prepared = self
            .inner
            .prepare_forward_source_mode(
                request,
                self.model.contract,
                ForwardSourceOptions {
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
        Ok(ResidentTextPreparation::Ready(ResidentTextPending {
            inner: Box::new(inner),
            cancelled,
            scope,
        }))
    }
    pub fn commit(&mut self, pending: ResidentTextPending) -> Result<ResidentTextPublication> {
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
        let output = ResidentTextReadback {
            data,
            _scope: pending.scope,
        };
        Ok(if discarded {
            ResidentTextPublication::Discarded(output)
        } else {
            ResidentTextPublication::Committed(output)
        })
    }
}

// Only these original pre-effect refusals permit scheduling outcomes. Unknown
// errors (including cancellation-looking errors after submission) stay fatal.
fn preparation_failure(error: Error, submitted: bool) -> Result<ResidentTextPreparation> {
    match error {
        Error::State(reason) if !submitted && reason == "CUDA forward request cancelled" => {
            Ok(ResidentTextPreparation::CancelledBeforeLaunch)
        }
        Error::Resource(_) if !submitted => Ok(ResidentTextPreparation::Capacity),
        Error::Limit(reason)
            if !submitted
                && matches!(
                    reason.as_str(),
                    "resident pool busy" | "resident pool exhausted"
                ) =>
        {
            Ok(ResidentTextPreparation::Capacity)
        }
        error => Err(error),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    #[test]
    fn paged_2b_32k_prices_reachable_state_and_keeps_weights_resident() {
        const MIB: usize = 1024 * 1024;
        const PROVIDER_CAP: usize = 4608 * MIB;
        let plan = zerogravity_model_qwen35::plan::diagnostic_forward_plan(
            zerogravity_model_qwen35::plan::QwenModelVariant::Qwen2B,
            32_768,
        )
        .unwrap();
        let roots = 4;
        let admission =
            crate::resident_program::cuda_forward::CudaForwardAdmission::analyze(plan.clone())
                .unwrap();
        let preflight = admission.preflight();
        let paged_state = super::super::state_pool::hybrid_state_reserve_bytes(
            &preflight.states,
            preflight.prior_state_bytes,
            roots,
            1,
            64,
            64,
        )
        .unwrap();
        assert!(paged_state < preflight.prior_state_bytes * roots);
        let (_, inventory) = super::super::cuda_forward_source_admission(
            plan,
            PROVIDER_CAP,
            roots,
            0,
            Some(paged_state),
        )
        .unwrap();
        assert_eq!(
            inventory.weight_storage,
            super::super::CudaForwardWeightStorage::Resident
        );
        assert_eq!(
            inventory.resident_weight_bytes,
            inventory.source_weight_bytes
        );
    }

    #[test]
    fn only_proven_prelaunch_refusals_become_scheduler_outcomes() {
        assert!(matches!(
            preparation_failure(Error::State("CUDA forward request cancelled".into()), false),
            Ok(ResidentTextPreparation::CancelledBeforeLaunch)
        ));
        let resource = Error::resource("device memory", "test admission", 2, 1);
        assert!(matches!(
            preparation_failure(resource, false),
            Ok(ResidentTextPreparation::Capacity)
        ));
        assert!(
            preparation_failure(
                Error::resource("device memory", "test admission", 2, 1),
                true
            )
            .is_err()
        );
        for reason in ["resident pool busy", "resident pool exhausted"] {
            assert!(matches!(
                preparation_failure(Error::Limit(reason.into()), false),
                Ok(ResidentTextPreparation::Capacity)
            ));
            assert!(preparation_failure(Error::Limit(reason.into()), true).is_err());
        }
        assert!(
            preparation_failure(Error::State("CUDA forward request cancelled".into()), true)
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
