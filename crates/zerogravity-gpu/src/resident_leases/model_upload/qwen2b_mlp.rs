//! Finite source/weight ownership only. No native packet or model-cut permission.
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::vllm_cuda_gdn::{Qwen2bGdnArithmetic, VllmGdnPhase};
use zerogravity_core::accelerator::program::{
    ActivationPrecision, FullModelProfile, PlanOperation, RecordedVllm08Model, ValueShape,
    Vllm08RmsSite, VllmBf16Invocation, VllmBf16Mode, VllmBf16Phase,
};
use zerogravity_model_qwen35::original_binding::{BoundFullModelSource, QwenMatrixRole};

mod cold_probe;
mod compute_probe;
pub use cold_probe::{Qwen2bMlpWeightMode, Qwen2bMlpWeightOutcome, qualify_qwen2b_mlp_weights};
use compute_probe::ComputeReadProbe;
pub(crate) use compute_probe::{Qwen2bMlpComputeProgress, Qwen2bMlpComputeReadCut};

pub(crate) struct Qwen2bMlpWeightCohort {
    source: BoundFullModelSource,
    record: Arc<RecordedVllm08Model>,
    arithmetic: Qwen2bGdnArithmetic,
    _metadata: WindowWorkspace,
}
fn require_request(
    profile: FullModelProfile,
    request: &VllmBf16Invocation,
) -> Result<VllmGdnPhase> {
    request.validate()?;
    if profile != FullModelProfile::Qwen2B
        || request.layer().index() != 0
        || request.mode() != VllmBf16Mode::Prefill
        || request.teachers().base() != 0
        || request.teachers().capacity() != 512
    {
        return Err(Error::Unsupported(
            "finite original2B initial MLP W source span".into(),
        ));
    }
    match request.legacy_phase()? {
        VllmBf16Phase::InitialOne => Ok(VllmGdnPhase::InitialOne),
        VllmBf16Phase::Initial64 => Ok(VllmGdnPhase::Initial64),
        _ => Err(Error::Unsupported(
            "original2B MLP weights initial1/64 only".into(),
        )),
    }
}
fn require_node(
    record: &RecordedVllm08Model,
    index: usize,
    kind: Vllm08Kind,
    inputs: &[u32],
    outputs: &[u32],
) -> Result<()> {
    let node = record
        .plan()
        .nodes()
        .get(index)
        .ok_or_else(|| Error::State("original2B MLP node absent".into()))?;
    let PlanOperation::TypedVllm08 {
        descriptor,
        state: None,
    } = &node.operation
    else {
        return Err(Error::State(
            "original2B MLP exact typed stateless node".into(),
        ));
    };
    descriptor.validate()?;
    if descriptor.call() != record.call()
        || descriptor.kind() != kind
        || descriptor.layer().map(|v| v.index()) != Some(if index == 11 { 1 } else { 0 })
        || node.state.is_some()
        || node.inputs.iter().map(|v| v.0).collect::<Vec<_>>() != inputs
        || node.outputs.iter().map(|v| v.0).collect::<Vec<_>>() != outputs
    {
        return Err(Error::State(
            "original2B MLP same call/issuer/SSA node".into(),
        ));
    }
    Ok(())
}
impl Qwen2bMlpWeightCohort {
    pub(crate) fn bind(
        source: BoundFullModelSource,
        record: Arc<RecordedVllm08Model>,
        arithmetic: Qwen2bGdnArithmetic,
    ) -> Result<Arc<Self>> {
        let phase = require_request(record.call().model_profile(), record.call().request())?;
        arithmetic.require_owned(phase)?;
        if source.model_profile() != FullModelProfile::Qwen2B
            || source.source().source_identity() != record.call().source()
            || record.plan().vllm_gdn_fragment() != Some(record.call().request())
        {
            return Err(Error::State(
                "same sealed original2B model/record source".into(),
            ));
        }
        require_node(
            &record,
            8,
            Vllm08Kind::Projection(Vllm08Projection::GateUp),
            &[8],
            &[10],
        )?;
        require_node(&record, 9, Vllm08Kind::SiluAndMul, &[10], &[11])?;
        require_node(
            &record,
            10,
            Vllm08Kind::Projection(Vllm08Projection::Down),
            &[11],
            &[12],
        )?;
        require_node(
            &record,
            11,
            Vllm08Kind::ResidualRms(Vllm08RmsSite::Input),
            &[12, 9],
            &[13, 14],
        )?;
        let rows = record.call().rows();
        for (id, width) in [(8, 2048), (9, 2048), (10, 12288), (11, 6144), (12, 2048)] {
            if record.plan().values().get(id) != Some(&ValueShape::Vector(rows * width))
                || record.plan().precisions().get(id) != Some(&ActivationPrecision::Bf16WidenedF32)
            {
                return Err(Error::State(
                    "original2B finite MLP value geometry/precision".into(),
                ));
            }
        }
        for (index, roles) in [
            (
                8,
                &[QwenMatrixRole::FfnGate(0), QwenMatrixRole::FfnUp(0)][..],
            ),
            (10, &[QwenMatrixRole::FfnDown(0)][..]),
        ] {
            let descriptor = match &record.plan().nodes()[index].operation {
                PlanOperation::TypedVllm08 { descriptor, .. } => descriptor,
                _ => return Err(Error::State("original2B weight descriptor absent".into())),
            };
            if !descriptor.parameters().is_empty() || descriptor.matrices().len() != roles.len() {
                return Err(Error::State(
                    "original2B complete ordered MLP learned members".into(),
                ));
            }
            for (&role, member) in roles.iter().zip(descriptor.matrices()) {
                source.source().validate_matrix(role, member)?;
            }
        }
        let metadata = source.source().source().reserve_workspace(64 * 1024)?;
        Ok(Arc::new(Self {
            source,
            record,
            arithmetic,
            _metadata: metadata,
        }))
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    pub(crate) fn arithmetic(&self) -> Qwen2bGdnArithmetic {
        self.arithmetic
    }
    fn operator(&self, index: usize) -> Result<&Vllm08Operator> {
        if !matches!(index, 8 | 10) {
            return Err(Error::Unsupported(
                "two original2B MLP learned nodes only".into(),
            ));
        }
        match &self.record.plan().nodes()[index].operation {
            PlanOperation::TypedVllm08 {
                descriptor,
                state: None,
            } => Ok(descriptor),
            _ => Err(Error::State(
                "original2B source learned descriptor absent".into(),
            )),
        }
    }
    fn require_service(&self, service: &Arc<ResidentService>) -> Result<()> {
        require_request(
            self.record.call().model_profile(),
            self.record.call().request(),
        )?;
        if service.source_identity() != self.source.source().source_identity()
            || self.source.model_profile() != FullModelProfile::Qwen2B
        {
            return Err(Error::State(
                "original2B MLP W actual source/service".into(),
            ));
        }
        Ok(())
    }
}
impl ResidentInitialization<'_> {
    pub(crate) async fn upload_qwen2b_mlp_weight(
        &mut self,
        cohort: &Arc<Qwen2bMlpWeightCohort>,
        index: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PendingModelWeight> {
        not_cancelled(cancelled)?;
        cohort.require_service(self.service)?;
        let descriptor = cohort.operator(index)?.clone();
        if !matches!(
            layout(&descriptor)?,
            Vllm08WeightLayout::Qwen2bMlpGateUp | Vllm08WeightLayout::Qwen2bMlpDown
        ) {
            return Err(Error::State("sealed original2B MLP finite W layout".into()));
        }
        self.upload_model_weight_inner(cohort.source.source().clone(), descriptor, cancelled)
            .await
    }
}

pub(crate) struct Qwen2bMlpWeights {
    cohort: Arc<Qwen2bMlpWeightCohort>,
    service: Arc<ResidentService>,
    generation: u64,
    gate_up: Arc<ModelWeight>,
    down: Arc<ModelWeight>,
}
impl Qwen2bMlpWeights {
    pub(crate) async fn upload(
        cohort: Arc<Qwen2bMlpWeightCohort>,
        service: Arc<ResidentService>,
        completion_owner: CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        cohort.require_service(&service)?;
        let mut init = service.begin_initialization(0)?;
        // Actual enclosing graph retained BEFORE first H2D; never stored in W.
        init.retain_completion_owner(completion_owner)?;
        init.retain_completion_owner(cohort.clone())?;
        let gate = init.upload_qwen2b_mlp_weight(&cohort, 8, cancelled).await?;
        let down = init
            .upload_qwen2b_mlp_weight(&cohort, 10, cancelled)
            .await?;
        drop(init.complete_sync().await?);
        let generation = service.generation();
        let weights = Arc::new(Self {
            cohort,
            service,
            generation,
            gate_up: gate.bind_initialized()?,
            down: down.bind_initialized()?,
        });
        weights.require_initialized()?;
        Ok(weights)
    }
    pub(crate) fn require_record(
        &self,
        record: &Arc<RecordedVllm08Model>,
        service: &Arc<ResidentService>,
    ) -> Result<()> {
        self.require_initialized()?;
        if !Arc::ptr_eq(record, self.cohort.record())
            || !Arc::ptr_eq(service, &self.service)
            || service.generation() != self.generation
        {
            return Err(Error::State(
                "same immutable2B MLP record/service/generation".into(),
            ));
        }
        Ok(())
    }
    /// Ownership witness after taint; no service usability or readiness promotion.
    pub(crate) fn retained_record(
        &self,
        source: &Arc<OriginalWindowModelSource>,
        record: &Arc<RecordedVllm08Model>,
        service: &Arc<ResidentService>,
    ) -> bool {
        if !Arc::ptr_eq(source, self.cohort.source.source())
            || !Arc::ptr_eq(record, self.cohort.record())
            || !Arc::ptr_eq(service, &self.service)
            || self.cohort.source.model_profile() != FullModelProfile::Qwen2B
            || record.call().source() != source.source_identity()
            || service.source_identity() != source.source_identity()
            || service.generation() != self.generation
            || self.gate_up.lease().id() == self.down.lease().id()
        {
            return false;
        }
        [
            (8, &self.gate_up, 50_331_648, 48),
            (10, &self.down, 25_165_824, 25),
        ]
        .into_iter()
        .all(|(index, weight, bytes, windows)| {
            let owner = &weight.owner;
            let Ok(descriptor) = self.cohort.operator(index) else {
                return false;
            };
            if !Arc::ptr_eq(&owner.source, source)
                || owner.descriptor != *descriptor
                || !owner.lease.is_initialized()
                || owner.lease.bytes() != bytes
                || owner.lease.role() != BufferRole::ImmutableWeights
                || owner.lease.source_identity() != source.source_identity()
                || !Arc::ptr_eq(owner.lease.service(), service)
                || owner.lease.service_generation() != self.generation
                || owner.require_complete().is_err()
            {
                return false;
            }
            // Original progress/coverage metadata only; copy_counts requires usability.
            owner
                .progress()
                .is_ok_and(|p| p.enqueued == windows && p.checkpoints == windows)
        })
    }
    fn require_initialized(&self) -> Result<()> {
        self.cohort.require_service(&self.service)?;
        for (index, weight, bytes, windows) in [
            (8, &self.gate_up, 50_331_648, 48),
            (10, &self.down, 25_165_824, 25),
        ] {
            weight.require_operator(self.cohort.source.source(), self.cohort.operator(index)?)?;
            if weight.lease().bytes() != bytes
                || weight.copy_counts()? != (windows, windows)
                || !Arc::ptr_eq(weight.lease().service(), &self.service)
                || weight.lease().service_generation() != self.generation
            {
                return Err(Error::State(
                    "actual initialized complete2B MLP W cohort".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn lease(&self, index: usize) -> Result<&ResidentLease> {
        match index {
            8 => Ok(self.gate_up.lease()),
            10 => Ok(self.down.lease()),
            _ => Err(Error::Unsupported(
                "original2B MLP learned lease index".into(),
            )),
        }
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        [&self.gate_up, &self.down]
            .iter()
            .map(|v| (v.lease().clone(), DeviceAccess::Read))
            .collect()
    }
    pub(crate) fn immutable_bytes(&self) -> usize {
        self.gate_up.lease().bytes() + self.down.lease().bytes()
    }
    pub(crate) fn verification_plan(
        self: &Arc<Self>,
        upstream: CompletionOwner,
    ) -> Result<Arc<Qwen2bMlpSourceVerification>> {
        self.require_initialized()?;
        let host = Arc::new(
            self.cohort
                .source
                .source()
                .source()
                .reserve_workspace(2 * 1024 * 1024)?,
        );
        Ok(Arc::new(Qwen2bMlpSourceVerification {
            weights: self.clone(),
            _upstream: upstream,
            _host: host,
            cold_probe: None,
            compute_probe: None,
        }))
    }
    pub(crate) fn compute_verification_plan(
        self: &Arc<Self>,
        upstream: CompletionOwner,
        cut: Option<Qwen2bMlpComputeReadCut>,
    ) -> Result<Arc<Qwen2bMlpSourceVerification>> {
        let mut plan = self.verification_plan(upstream)?;
        Arc::get_mut(&mut plan)
            .ok_or_else(|| Error::State("fresh exclusive2B compute plan required".into()))?
            .compute_probe = Some(Arc::new(ComputeReadProbe::new(cut)));
        Ok(plan)
    }
}

/// Sealed actual W/source/graph owner, installed BEFORE the first math enqueue.
/// Holds no expected rows, raw-copy authority or native/Root permission.
pub(crate) struct Qwen2bMlpSourceVerification {
    weights: Arc<Qwen2bMlpWeights>,
    _upstream: CompletionOwner,
    _host: Arc<WindowWorkspace>,
    cold_probe: Option<Arc<cold_probe::ReadProbe>>,
    compute_probe: Option<Arc<ComputeReadProbe>>,
}
impl Qwen2bMlpSourceVerification {
    pub(crate) fn retains_compute_graph(
        &self,
        weights: &Arc<Qwen2bMlpWeights>,
        upstream: &CompletionOwner,
    ) -> bool {
        self.compute_probe.is_some()
            && self.cold_probe.is_none()
            && Arc::ptr_eq(&self.weights, weights)
            && Arc::ptr_eq(&self._upstream, upstream)
    }
    pub(crate) fn compute_progress(&self) -> Result<Qwen2bMlpComputeProgress> {
        Ok(self
            .compute_probe
            .as_ref()
            .ok_or_else(|| Error::State("not an instrumented compute plan".into()))?
            .progress())
    }
    pub(crate) fn enter_compute_validator(&self) -> Result<()> {
        self.compute_probe
            .as_ref()
            .ok_or_else(|| Error::State("compute validator requires actual compute plan".into()))?
            .enter_validator()
    }
    fn require_flight(&self, flight: &ResidentSubmission<'_>) -> Result<()> {
        self.weights.require_initialized()?;
        if !flight.armed
            || flight.cohort.is_some()
            || !Arc::ptr_eq(flight.service, &self.weights.service)
        {
            return Err(Error::State(
                "same original armed2B MLP flight/service".into(),
            ));
        }
        for (lease, _) in self.weights.accesses() {
            if !lease.is_initialized()
                || lease.role() != BufferRole::ImmutableWeights
                || !flight.accesses.iter().any(|(actual, access)| {
                    actual.id() == lease.id() && *access == DeviceAccess::Read
                })
            {
                return Err(Error::State(
                    "actual initialized2B W Read in original submission".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn retain_before_enqueue(
        self: &Arc<Self>,
        flight: &mut ResidentSubmission<'_>,
    ) -> Result<()> {
        self.require_flight(flight)?;
        flight.retain_completion_owner(self.clone())
    }
    pub(crate) async fn complete_compute<F>(
        self: &Arc<Self>,
        flight: ResidentSubmission<'_>,
        outputs: &[ResidentLease],
        cohort: &[ResidentReadback],
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<(ResidentReady, Vec<Bytes>)>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        flight
            .complete_with_readback_and_source_verification(
                SourceCompletionMode::Compute(outputs),
                cohort,
                self,
                cancelled,
                validate,
            )
            .await
    }
    pub(crate) async fn complete_retirement<F>(
        self: &Arc<Self>,
        flight: ResidentSubmission<'_>,
        cohort: &[ResidentReadback],
        cancelled: Option<&AtomicBool>,
        validate: F,
    ) -> Result<Vec<Bytes>>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        let (ready, bytes) = flight
            .complete_with_readback_and_source_verification(
                SourceCompletionMode::Retirement,
                cohort,
                self,
                cancelled,
                validate,
            )
            .await?;
        drop(ready);
        Ok(bytes)
    }
    pub(in crate::resident_leases) fn require_installed(
        self: &Arc<Self>,
        flight: &ResidentSubmission<'_>,
    ) -> Result<()> {
        self.require_flight(flight)?;
        let owner: CompletionOwner = self.clone();
        if !flight
            .completion_owners
            .iter()
            .any(|actual| Arc::ptr_eq(actual, &owner))
        {
            return Err(Error::State(
                "exact source verification plan absent from armed flight".into(),
            ));
        }
        Ok(())
    }
    pub(in crate::resident_leases) async fn verify(
        self: &Arc<Self>,
        flight: &ResidentSubmission<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.require_installed(flight)?;
        let mut count = 0;
        for weight in [&self.weights.gate_up, &self.weights.down] {
            let mut destination = 0;
            for matrix in &weight.owner.matrices {
                let rows = 1024 * 1024 / (matrix.columns() * 2);
                for first in (0..matrix.rows()).step_by(rows) {
                    not_cancelled(cancelled)?;
                    let end = (first + rows).min(matrix.rows());
                    let (window, read, bytes) = weight.checked_verification_window(
                        matrix,
                        first..end,
                        destination,
                        cancelled,
                    )?;
                    // Source window is host comparison data only, never a GPU operand.
                    // The awaited D2H uses the existing flight's retained device graph.
                    let actual = flight
                        .readback_checked(&[read], |raw| {
                            if raw.len() != 1 || &raw[0][..] != window.as_slice() {
                                return Err(Error::State(
                                    "actual2B W differs fresh bounded source".into(),
                                ));
                            }
                            not_cancelled(cancelled)
                        })
                        .await?;
                    drop(actual);
                    drop(window);
                    not_cancelled(cancelled)?;
                    destination += bytes;
                    count += 1;
                    if let Some(probe) = &self.cold_probe {
                        probe.after_completed_read(count, bytes)?;
                    }
                    if let Some(probe) = &self.compute_probe {
                        probe.after_completed_read(count, bytes)?;
                    }
                }
            }
            if destination != weight.lease().bytes() {
                return Err(Error::State(
                    "complete actual2B W serial read coverage".into(),
                ));
            }
        }
        if count != 73 {
            return Err(Error::State(
                "all73 actual2B MLP source windows required".into(),
            ));
        }
        not_cancelled(cancelled)?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests;
