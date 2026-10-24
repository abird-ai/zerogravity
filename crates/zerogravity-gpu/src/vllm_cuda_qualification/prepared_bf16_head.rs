//! Prepared selected-row BF16 head on the original provider actor and flight.
//! Source/profile/kernel eligibility stays exact; no family recipe or publisher.
use crate::{
    resident_leases::resident_matrix::Bf16MatrixSource,
    resident_leases::{ResidentInitialization, ResidentLease, ResidentMatrix},
    transfer_retirement::{RetirementCounts, RetirementMark, TransferRetirementRing},
    vllm_cuda_qualification::projection_cubin::resident::PreparedImages,
};
use cubecl::cuda::{
    Bf16MatrixLayout, Bf16VocabularyHead, CubinKernelManifest, OwnedCudaKernel,
    OwnedCudaTransferRetirement, Vllm08WeightLayout,
};
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::source_weights::TensorWindow;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{ProgramNumericalProfile, SourceTensor},
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::HostWorkspace,
};
#[path = "short_hf_head/performance.rs"]
mod performance;
const WINDOW: usize = 1024 * 1024;
const RETIREMENT_SLOTS: usize = 2;
// Actual source is scanned in1MiB integrity chunks. A possibly unaligned1MiB
// tile retains at most two complete chunks plus their exact offset metadata.
pub(crate) const STREAMED_WINDOW_STAGING_UPPER: usize =
    2 * (WINDOW + std::mem::size_of::<(usize, std::ops::Range<usize>)>());
type HeadRetirementRing = TransferRetirementRing<OwnedCudaTransferRetirement, Arc<WindowBytes>>;
impl RetirementMark for OwnedCudaTransferRetirement {
    fn ready(&mut self) -> Result<bool> {
        OwnedCudaTransferRetirement::ready(self).map_err(native)
    }
    fn wait(&mut self) -> Result<()> {
        OwnedCudaTransferRetirement::wait(self).map_err(native)
    }
}
fn cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("streamed head cancelled".into()));
    }
    Ok(())
}
fn native(e: cubecl::server::ServerError) -> Error {
    Error::State(format!("streamed head original actor: {e}"))
}
struct WindowBytes(TensorWindow);
impl AsRef<[u8]> for WindowBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// Actual source operand and existing numerical profile from the original plan.
/// This is prepared consumer metadata, not another semantic operation/IR.
pub(crate) struct HeadPreparation {
    pub(crate) matrix: Bf16MatrixSource,
    pub(crate) operand: SourceTensor,
    pub(crate) numerical: ProgramNumericalProfile,
    pub(crate) manifest: CubinKernelManifest,
    pub(crate) images: Arc<PreparedImages>,
    pub(crate) resident: Option<Arc<ResidentMatrix>>,
    pub(crate) audit: bool,
}
pub(crate) struct PreparedBf16Head {
    lease: ResidentLease,
    resident: Option<Arc<ResidentMatrix>>,
    verification: Option<Arc<Mutex<Vec<u8>>>>,
    active: Mutex<Weak<HeadRun>>,
    retired_marks: Mutex<Vec<OwnedCudaTransferRetirement>>,
    kernel: Arc<OwnedCudaKernel>,
    _images: Arc<PreparedImages>,
    matrix: Bf16MatrixSource,
    operand: SourceTensor,
    numerical: ProgramNumericalProfile,
    layout: Vllm08WeightLayout,
    _metadata: Arc<HostWorkspace>,
    _verification_admission: Option<Arc<HostWorkspace>>,
}
#[derive(Default)]
struct Progress {
    pending: Option<Arc<WindowBytes>>,
    retirement: Option<HeadRetirementRing>,
    retirement_counts: RetirementCounts,
    next: usize,
    uploaded: usize,
    native: usize,
    verified: usize,
    checkpoints: usize,
    started: bool,
    complete: bool,
}
/// Attached to the original flight before active execution becomes visible.
pub(crate) struct HeadRun {
    head: Arc<Bf16HeadCall>,
}
pub(crate) struct Bf16HeadCall {
    progress: Mutex<Progress>,
    cold: Arc<PreparedBf16Head>,
    input_rows: usize,
    selected_row: usize,
    input: ResidentLease,
    output: ResidentLease,
    registered: AtomicBool,
    fault: AtomicUsize,
    _binding_metadata: HostWorkspace,
}
impl PreparedBf16Head {
    pub(crate) fn prepare(
        prepared: HeadPreparation,
        service: &Arc<crate::resident_leases::ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        cancelled(cancel)?;
        let HeadPreparation {
            matrix,
            operand,
            numerical,
            manifest,
            images,
            resident,
            audit,
        } = prepared;
        require_operand(&matrix, &operand, numerical)?;
        if matrix.source().source_identity() != service.source_identity() {
            return Err(Error::State(
                "head original matrix/service source differs".into(),
            ));
        }
        let source = matrix.source();
        let metadata = Arc::new(
            source
                .reserve_workspace(128 * 1024)?
                .into_host_workspace()?,
        );
        if resident.is_some() && audit {
            return Err(Error::State(
                "short-HF resident source mode/weight/audit contract".into(),
            ));
        }
        // These are finite native packet shapes, never universal family geometry.
        let layout = match (matrix.rows(), matrix.columns()) {
            (248320, 1024) => Vllm08WeightLayout::Vocabulary08,
            (248320, 2048) => Vllm08WeightLayout::Vocabulary2b,
            _ => {
                return Err(Error::Unsupported(
                    "streamed head native source geometry".into(),
                ));
            }
        };
        let expected = if resident.is_some() {
            Bf16VocabularyHead::resident_manifest(
                Bf16MatrixLayout::new(matrix.rows(), matrix.columns()).map_err(native)?,
            )
            .map_err(native)?
        } else {
            Bf16VocabularyHead::streamed_manifest(layout).map_err(native)?
        };
        if manifest != expected {
            return Err(Error::State(
                "head exact prepared numerical/source packet".into(),
            ));
        }
        let verification_admission = audit
            .then(|| {
                source
                    .reserve_workspace(WINDOW)?
                    .into_host_workspace()
                    .map(Arc::new)
            })
            .transpose()?;
        // Both image and original init must be the same admitted service.
        images.require_service(service)?;
        init.require_service(service)?;
        init.retain_completion_owner(metadata.clone())?;
        if let Some(admission) = &verification_admission {
            init.retain_completion_owner(admission.clone())?;
        }
        init.retain_completion_owner(images.clone())?;
        let lease = if let Some(weight) = &resident {
            weight.require_matrix(source, service, matrix.matrix())?;
            init.retain_completion_owner(weight.clone())?;
            weight.lease().clone()
        } else {
            init.allocate_empty(WINDOW, BufferRole::Scratch)?
        };
        let image = images.image(
            crate::vllm_cuda_qualification::cubin_manifests::ProjectionImage::Bf16Gemv2TSm86,
        )?;
        // SAFETY: original initialization retains exact image/native admission.
        let kernel = unsafe { images.set().prepare(image, manifest) }.map_err(native)?;
        init.retain_completion_owner(kernel.clone())?;
        let result = Arc::new(Self {
            lease,
            resident,
            verification: audit.then(|| Arc::new(Mutex::new(vec![0; WINDOW]))),
            active: Mutex::new(Weak::new()),
            retired_marks: Mutex::new(Vec::with_capacity(RETIREMENT_SLOTS)),
            kernel,
            _images: images,
            matrix,
            operand,
            numerical,
            layout,
            _metadata: metadata,
            _verification_admission: verification_admission,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn rows(&self) -> usize {
        self.matrix.rows()
    }
    pub(crate) fn columns(&self) -> usize {
        self.matrix.columns()
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn is_active(&self) -> Result<bool> {
        Ok(self
            .active
            .lock()
            .map_err(|_| Error::State("streamed head busy lock".into()))?
            .upgrade()
            .is_some())
    }
    pub(crate) fn service(&self) -> &Arc<crate::resident_leases::ResidentService> {
        self.lease.service()
    }
    pub(crate) fn source(&self) -> &zerogravity_artifacts::source_weights::VerifiedTensorSource {
        self.matrix.source()
    }
    pub(crate) fn bind(
        self: &Arc<Self>,
        input_rows: usize,
        selected_row: usize,
        operand: &SourceTensor,
        numerical: ProgramNumericalProfile,
        input: ResidentLease,
        output: ResidentLease,
    ) -> Result<Arc<Bf16HeadCall>> {
        if operand != &self.operand
            || numerical != self.numerical
            || selected_row.checked_add(1) != Some(input_rows)
        {
            return Err(Error::State(
                "head original source/contract/selected final row".into(),
            ));
        }
        let binding_metadata = self
            .matrix
            .source()
            .reserve_workspace(64 * 1024)?
            .into_host_workspace()?;
        let result = Arc::new(Bf16HeadCall {
            progress: Mutex::new(Progress::default()),
            cold: self.clone(),
            input_rows,
            selected_row,
            input,
            output,
            registered: AtomicBool::new(false),
            fault: AtomicUsize::new(0),
            _binding_metadata: binding_metadata,
        });
        result.require_binding()?;
        Ok(result)
    }
}
fn require_operand(
    matrix: &Bf16MatrixSource,
    operand: &SourceTensor,
    numerical: ProgramNumericalProfile,
) -> Result<()> {
    let actual = matrix.matrix();
    if numerical != ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1
        || operand.source != matrix.source().source_identity()
        || operand.name != actual.tensor().name
        || operand.kind != actual.tensor().kind
        || operand.axes != actual.axes()
        || operand.columns != matrix.columns()
        || operand.rows != matrix.rows()
    {
        return Err(Error::State(
            "head exact source operand/numerical contract".into(),
        ));
    }
    Ok(())
}
fn require_selected_extent(
    columns: usize,
    vocabulary: usize,
    input_rows: usize,
    selected_row: usize,
    input_bytes: usize,
    output_bytes: usize,
) -> Result<()> {
    if ![1, 47, 57, 62, 67, 72].contains(&input_rows)
        || selected_row.checked_add(1) != Some(input_rows)
        || input_rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(2))
            != Some(input_bytes)
        || vocabulary.checked_mul(2) != Some(output_bytes)
    {
        return Err(Error::State(
            "head exact selected final-row input/output extents".into(),
        ));
    }
    Ok(())
}
impl Bf16HeadCall {
    fn require_binding(&self) -> Result<()> {
        require_operand(&self.cold.matrix, &self.cold.operand, self.cold.numerical)?;
        let service = self.cold.lease.service();
        require_selected_extent(
            self.cold.matrix.columns(),
            self.cold.matrix.rows(),
            self.input_rows,
            self.selected_row,
            self.input.bytes(),
            self.output.bytes(),
        )?;
        if self.cold.matrix.source().source_identity() != service.source_identity()
            || self
                .cold
                .resident
                .as_ref()
                .is_some_and(|weight| !weight.matches_matrix(self.cold.matrix.matrix()))
            || !matches!(
                self.input.role(),
                BufferRole::Activation | BufferRole::Output | BufferRole::Scratch
            )
            || !matches!(
                self.output.role(),
                BufferRole::Activation | BufferRole::Output | BufferRole::Scratch
            )
            || self.input.id() == self.output.id()
            || self.input.id() == self.cold.lease.id()
            || self.output.id() == self.cold.lease.id()
            || !Arc::ptr_eq(self.input.service(), service)
            || !Arc::ptr_eq(self.output.service(), service)
            || self.input.service_generation() != service.generation()
            || self.output.service_generation() != service.generation()
            || self.input.source_identity() != service.source_identity()
            || self.output.source_identity() != service.source_identity()
        {
            return Err(Error::State(
                "head SAME source/service/generation/full WXY owners".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn input(&self) -> &ResidentLease {
        &self.input
    }
    pub(crate) fn output(&self) -> &ResidentLease {
        &self.output
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        vec![
            (
                self.cold.lease.clone(),
                if self.cold.resident.is_some() {
                    DeviceAccess::Read
                } else {
                    DeviceAccess::Write
                },
            ),
            (self.input.clone(), DeviceAccess::Read),
            (self.output.clone(), DeviceAccess::Write),
        ]
    }
    /// # Safety
    /// Caller validated original producer/operation and exact accesses on its
    /// armed flight; retain must attach this run to that SAME original flight.
    pub(crate) unsafe fn register(
        self: &Arc<Self>,
        retain: impl FnOnce(Arc<HeadRun>) -> Result<()>,
    ) -> Result<()> {
        self.require_binding()?;
        self.cold.lease.service().ensure_usable()?;
        let mut active = self
            .cold
            .active
            .lock()
            .map_err(|_| Error::State("streamed head run lock".into()))?;
        if active.upgrade().is_some() || self.registered.swap(true, Ordering::AcqRel) {
            return Err(Error::State(
                "streamed head one original registered run".into(),
            ));
        }
        let run = Arc::new(HeadRun { head: self.clone() });
        // Retain on actual flight before exposing the busy Weak to execution.
        retain(run.clone())?;
        *active = Arc::downgrade(&run);
        drop(active);
        if self.cold.resident.is_none() && self.cold.verification.is_none() {
            let marks = std::mem::take(
                &mut *self
                    .cold
                    .retired_marks
                    .lock()
                    .map_err(|_| Error::State("streamed head retired-mark pool lock".into()))?,
            );
            let cold = self.cold.clone();
            let input = self.input.clone();
            let output = self.output.clone();
            let client = self.cold.lease.service().client().clone();
            let handles = vec![cold.lease.handle(), input.handle(), output.handle()];
            let ring = HeadRetirementRing::new(RETIREMENT_SLOTS, marks, move |pending| {
                // Acyclic: Cold only has a Weak<Run>. Retain actual lease
                // charges/native/source owners, not merely raw handle clones.
                let keepalive: crate::physical_quarantine::CompletionOwner =
                    Arc::new((cold, input, output, pending));
                crate::physical_quarantine::quarantine_uncertain_with_owners(
                    client,
                    handles,
                    vec![keepalive],
                );
            })?;
            self.progress
                .lock()
                .map_err(|_| Error::State("streamed head progress lock".into()))?
                .retirement = Some(ring);
        }
        Ok(())
    }
    /// Family adapter validates actual original operation/producer first; original
    /// outer flight queues full-output certification/greedy after completion.
    /// # Safety
    /// Original family adapter rechecked current operation, producer ordering
    /// and same call/leases on the registered original flight before this call.
    pub(crate) unsafe fn enqueue(self: &Arc<Self>, cancel: Option<&AtomicBool>) -> Result<usize> {
        let mut timing = performance::HeadTiming::new(
            self.cold.resident.is_some(),
            self.input_rows,
            self.cold.verification.is_some(),
        );
        self.require_binding()?;
        let run = self
            .cold
            .active
            .lock()
            .map_err(|_| Error::State("streamed head run lock".into()))?
            .upgrade()
            .ok_or_else(|| Error::State("streamed head requires original armed owner".into()))?;
        if !Arc::ptr_eq(&run.head, self) {
            return Err(Error::State("streamed head foreign registered run".into()));
        }
        let mut progress = self
            .progress
            .lock()
            .map_err(|_| Error::State("streamed head progress lock".into()))?;
        if progress.started {
            return Err(Error::State("streamed head record replay".into()));
        }
        progress.started = true;
        drop(progress);
        if self.cold.resident.is_some() {
            return self.enqueue_resident(cancel, &mut timing);
        }
        let windows = self.cold.matrix.windows();
        let queue = self.cold.lease.service().pinned_cuda_queue()?;
        for ordinal in 0..windows {
            cancelled(cancel)?;
            self.cold.lease.service().ensure_usable()?;
            // Bound host source owners BEFORE the next authenticated read.
            if self.cold.verification.is_none() {
                let phase = timing.start();
                let mut p = self
                    .progress
                    .lock()
                    .map_err(|_| Error::State("streamed head progress lock".into()))?;
                let ring = p.retirement.as_mut().ok_or_else(|| {
                    Error::State("streamed head original transfer run absent".into())
                })?;
                ring.make_room()?;
                p.retirement_counts = ring.counts();
                timing.retirement = p.retirement_counts;
                timing.record(performance::CHECKPOINT, phase);
            }
            let phase = timing.start();
            let window = self.cold.matrix.read_window(ordinal, cancel)?;
            timing.record(performance::READ, phase);
            let phase = timing.start();
            self.cold.matrix.require_finite_window(
                ordinal,
                &window,
                STREAMED_WINDOW_STAGING_UPPER,
            )?;
            // This native packet consumes complete1MiB tiles; generic source
            // geometry also supports short tails, which this candidate refuses.
            if window.as_slice().len() != WINDOW {
                return Err(Error::Unsupported("streamed head native full tile".into()));
            }
            timing.record(performance::VALIDATE, phase);
            let window = Arc::new(WindowBytes(window));
            {
                let mut p = self
                    .progress
                    .lock()
                    .map_err(|_| Error::State("streamed head progress lock".into()))?;
                if p.next != ordinal || p.pending.is_some() {
                    return Err(Error::State("streamed head source coverage order".into()));
                }
                p.pending = Some(window.clone());
            }
            // SAFETY: registered original flight owns body, WXY, same native
            // set, source/bounce and this exact pending immutable host window.
            let phase = timing.start();
            unsafe {
                queue.enqueue_streamed_head_window(
                    self.cold.lease.handle(),
                    self.cold.layout,
                    window.clone(),
                )
            }
            .map_err(native)?;
            timing.record(performance::UPLOAD, phase);
            timing.uploaded += WINDOW;
            self.progress
                .lock()
                .map_err(|_| Error::State("streamed head progress lock".into()))?
                .uploaded += 1;
            cancelled(cancel)?;
            let phase = timing.start();
            unsafe {
                self.cold.kernel.enqueue_streamed_vocabulary_head_bindings(
                    [
                        self.cold.lease.cubin_binding()?,
                        self.input.cubin_binding()?,
                        self.output.cubin_binding()?,
                    ],
                    self.cold.layout,
                    self.input_rows,
                    ordinal,
                )
            }
            .map_err(native)?;
            timing.record(performance::KERNEL, phase);
            timing.native += 1;
            self.progress
                .lock()
                .map_err(|_| Error::State("streamed head progress lock".into()))?
                .native += 1;
            if ordinal == windows / 2 && self.fault.load(Ordering::Acquire) == 2 {
                return Err(Error::State(
                    "streamed head actual native cut before checkpoint".into(),
                ));
            }
            let phase = timing.start();
            if self.cold.verification.is_some() {
                // Diagnostic audit keeps its actual W D2H/compare + barrier.
                unsafe {
                    queue.checkpoint_streamed_head(
                        self.cold.lease.handle(),
                        self.output.handle(),
                        self.cold.layout,
                        ordinal,
                        window.clone(),
                        self.cold.verification.clone(),
                    )
                }
                .map_err(native)?;
                timing.checkpoints += 1;
                let mut p = self
                    .progress
                    .lock()
                    .map_err(|_| Error::State("streamed head progress lock".into()))?;
                p.checkpoints += 1;
                p.verified += 1;
            } else {
                let reusable = self
                    .progress
                    .lock()
                    .map_err(|_| Error::State("streamed head progress lock".into()))?
                    .retirement
                    .as_mut()
                    .ok_or_else(|| Error::State("streamed head transfer run absent".into()))?
                    .take_reusable();
                // SAFETY: exact window was retained before H2D; same original
                // stream ordered H2D/native before this mark and next overwrite.
                let mark = unsafe {
                    queue.mark_streamed_head_retirement(
                        self.cold.lease.handle(),
                        self.output.handle(),
                        self.cold.layout,
                        ordinal,
                        reusable,
                    )
                }
                .map_err(native)?;
                let mut p = self
                    .progress
                    .lock()
                    .map_err(|_| Error::State("streamed head progress lock".into()))?;
                let ring = p
                    .retirement
                    .as_mut()
                    .ok_or_else(|| Error::State("streamed head transfer run absent".into()))?;
                ring.push(mark, window.clone())?;
                p.retirement_counts = ring.counts();
                timing.retirement = p.retirement_counts;
            }
            timing.record(performance::CHECKPOINT, phase);
            {
                let mut p = self
                    .progress
                    .lock()
                    .map_err(|_| Error::State("streamed head progress lock".into()))?;
                if ordinal == windows / 2 && self.fault.load(Ordering::Acquire) == 3 {
                    // This fault promised known current-window completion before
                    // refusing the next overwrite. Keep that exact boundary.
                    if let Some(ring) = p.retirement.as_mut() {
                        ring.drain()?;
                        p.retirement_counts = ring.counts();
                        timing.retirement = p.retirement_counts;
                    }
                    let flag = cancel.ok_or_else(|| {
                        Error::State("streamed head late-cancel requires actual flag".into())
                    })?;
                    flag.store(true, Ordering::Release);
                    return Err(Error::State(
                        "streamed head actual cancellation before next overwrite".into(),
                    ));
                }
                p.next += 1;
                drop(p.pending.take());
            }
            drop(window);
            cancelled(cancel)?;
        }
        let mut p = self
            .progress
            .lock()
            .map_err(|_| Error::State("streamed head progress lock".into()))?;
        let expected_verified = if self.cold.verification.is_some() {
            windows
        } else {
            0
        };
        if p.pending.is_some()
            || (p.next, p.uploaded, p.native, p.checkpoints, p.verified)
                != (
                    windows,
                    windows,
                    windows,
                    expected_verified,
                    expected_verified,
                )
        {
            return Err(Error::State(
                "streamed head incomplete actual source/native/verification coverage".into(),
            ));
        }
        if self.cold.verification.is_none() && p.retirement_counts.issued != windows {
            return Err(Error::State(
                "streamed head incomplete actual retirement marks".into(),
            ));
        }
        // Last marks stay on the original flight. Terminal readback completion
        // must query every remaining mark before its validator can return Ready.
        p.complete = true;
        timing.complete = true;
        Ok(p.native)
    }
    fn enqueue_resident(
        &self,
        cancel: Option<&AtomicBool>,
        timing: &mut performance::HeadTiming,
    ) -> Result<usize> {
        cancelled(cancel)?;
        let tied = self
            .cold
            .resident
            .as_ref()
            .ok_or_else(|| Error::State("resident head actual weight owner absent".into()))?;
        tied.require_matrix(
            self.cold.matrix.source(),
            self.cold.lease.service(),
            self.cold.matrix.matrix(),
        )?;
        let shape = Bf16MatrixLayout::new(self.cold.matrix.rows(), self.cold.matrix.columns())
            .map_err(native)?;
        let phase = timing.start();
        crate::resident_leases::profile_head_enqueue(|| {
            // SAFETY: original registered flight holds the SAME initialized tied W,
            // actual FinalRMS producer, complete X/Y owners and exact native set.
            unsafe {
                self.cold.kernel.enqueue_resident_vocabulary_head(
                    [
                        tied.lease().cubin_binding()?,
                        self.input.cubin_binding()?,
                        self.output.cubin_binding()?,
                    ],
                    shape,
                    self.input_rows,
                )
            }
            .map_err(native)
        })?;
        timing.record(performance::KERNEL, phase);
        timing.native += 1;
        let mut p = self
            .progress
            .lock()
            .map_err(|_| Error::State("resident head progress lock".into()))?;
        p.native += 1;
        match self.fault.load(Ordering::Acquire) {
            2 => {
                return Err(Error::State(
                    "resident head actual native cut before original completion".into(),
                ));
            }
            3 => {
                cancel
                    .ok_or_else(|| {
                        Error::State("resident head late cancel requires actual flag".into())
                    })?
                    .store(true, Ordering::Release);
                return Err(Error::State(
                    "resident head actual cancellation after native enqueue".into(),
                ));
            }
            _ => {}
        }
        cancelled(cancel)?;
        p.complete = true;
        timing.complete = true;
        Ok(1)
    }
    pub(crate) fn enqueue_placement(&self) -> crate::resident_leases::EnqueuePlacement {
        if self.cold.resident.is_some() {
            crate::resident_leases::EnqueuePlacement::DeviceActor
        } else {
            // Source-window IO/checkpoints retain their original caller scope.
            crate::resident_leases::EnqueuePlacement::Caller
        }
    }
    /// Qualification-only real partial cuts; normal native callers never arm.
    pub(crate) fn arm_fault(&self, cut: HeadFault) -> Result<()> {
        if self.registered.load(Ordering::Acquire)
            || self
                .progress
                .lock()
                .map_err(|_| Error::State("streamed head progress lock".into()))?
                .started
            || self
                .fault
                .compare_exchange(0, cut as usize, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(Error::State(
                "streamed head fault must precede original run".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn retention_witness(self: &Arc<Self>) -> Result<HeadWitness> {
        let p = self
            .progress
            .lock()
            .map_err(|_| Error::State("streamed head progress lock".into()))?;
        Ok(HeadWitness {
            head: Arc::downgrade(self),
            run: self
                .cold
                .active
                .lock()
                .map_err(|_| Error::State("streamed head run lock".into()))?
                .clone(),
            window: p.pending.as_ref().map(Arc::downgrade),
        })
    }
    pub(crate) fn complete(&self) -> Result<bool> {
        Ok(self
            .progress
            .lock()
            .map_err(|_| Error::State("streamed head progress lock".into()))?
            .complete)
    }
    /// Called only inside the original successful physical-completion validator.
    /// A late pending/error refuses Ready; Drop is not a completion proof.
    pub(crate) fn retire_transfers_after_completion(&self) -> Result<()> {
        let mut p = self
            .progress
            .lock()
            .map_err(|_| Error::State("streamed head progress lock".into()))?;
        if !p.complete || p.pending.is_some() {
            return Err(Error::State(
                "streamed head terminal completion before coverage".into(),
            ));
        }
        if let Some(ring) = p.retirement.as_mut() {
            ring.confirm_after_completion()?;
            let counts = ring.counts();
            let marks = ring.take_retired_marks()?;
            let mut pool = self
                .cold
                .retired_marks
                .lock()
                .map_err(|_| Error::State("streamed head retired-mark pool lock".into()))?;
            if !pool.is_empty() || marks.len() > RETIREMENT_SLOTS {
                return Err(Error::State(
                    "streamed head original event pool owner".into(),
                ));
            }
            *pool = marks;
            p.retirement_counts = counts;
            p.retirement = None;
        }
        Ok(())
    }
    pub(crate) fn retirement_counts(&self) -> Result<[usize; 5]> {
        let p = self
            .progress
            .lock()
            .map_err(|_| Error::State("streamed head progress lock".into()))?;
        let c = p
            .retirement
            .as_ref()
            .map(|r| r.counts())
            .unwrap_or(p.retirement_counts);
        Ok([c.issued, c.retired, c.queries, c.waits, c.pending])
    }
    fn retained_source_staging_bytes(&self) -> Result<usize> {
        let p = self
            .progress
            .lock()
            .map_err(|_| Error::State("streamed head progress lock".into()))?;
        let mut bytes = p
            .retirement
            .as_ref()
            .map(|r| {
                r.owners()
                    .map(|w| w.0.charged_staging_bytes())
                    .sum::<usize>()
            })
            .unwrap_or(0);
        if let Some(current) = &p.pending {
            let already_retained = p
                .retirement
                .as_ref()
                .is_some_and(|r| r.owners().any(|w| Arc::ptr_eq(w, current)));
            if !already_retained {
                bytes += current.0.charged_staging_bytes();
            }
        }
        Ok(bytes)
    }
    pub(crate) fn counts(&self) -> Result<[usize; 4]> {
        let p = self
            .progress
            .lock()
            .map_err(|_| Error::State("streamed head progress lock".into()))?;
        Ok([p.uploaded, p.native, p.checkpoints, p.verified])
    }
    pub(crate) fn retained_graph(&self) -> Result<bool> {
        // Metadata only; callable after original service is fenced.
        self.require_binding()?;
        let weight = match &self.cold.resident {
            Some(weight) => {
                weight.lease().id() == self.cold.lease.id()
                    && weight.lease().allocation_generation()
                        == self.cold.lease.allocation_generation()
                    && self.cold.lease.is_initialized()
                    && self.cold.lease.bytes()
                        == self.cold.matrix.rows() * self.cold.matrix.columns() * 2
                    && self.cold.lease.role() == BufferRole::ImmutableWeights
            }
            None => {
                self.cold.lease.bytes() == WINDOW && self.cold.lease.role() == BufferRole::Scratch
            }
        };
        Ok(weight
            && self.cold._metadata.source_identity() == self.cold.matrix.source().source_identity())
    }
    pub(crate) fn kernel(&self) -> &Arc<OwnedCudaKernel> {
        &self.cold.kernel
    }
    pub(crate) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: caller's original source retirement flight proves completion;
        // SAME model pool owns all entries and retires its set separately.
        unsafe { self.cold.kernel.retire_after_physical_completion() }
    }
}

impl Drop for Bf16HeadCall {
    fn drop(&mut self) {
        let p = self
            .progress
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        if p.pending.is_none()
            && p.retirement
                .as_ref()
                .is_none_or(|ring| ring.counts().pending == 0)
        {
            return;
        }
        // Defense in depth for an exceptional owner Drop. In particular the
        // current H2D/native may have failed BEFORE obtaining a mark. Retain
        // that exact unmarked window too, plus native and actual lease charges.
        // Normal completion explicitly queried/returned marks before Ready.
        let owner: crate::physical_quarantine::CompletionOwner = Arc::new((
            self.cold.clone(),
            self.input.clone(),
            self.output.clone(),
            p.pending.take(),
            p.retirement.take(),
        ));
        crate::physical_quarantine::quarantine_uncertain_with_owners(
            self.cold.lease.service().client().clone(),
            vec![
                self.cold.lease.handle(),
                self.input.handle(),
                self.output.handle(),
            ],
            vec![owner],
        );
    }
}
#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum HeadFault {
    UnknownAfterNative = 2,
    CancelBeforeOverwrite = 3,
}
pub(crate) struct HeadWitness {
    head: Weak<Bf16HeadCall>,
    run: Weak<HeadRun>,
    window: Option<Weak<WindowBytes>>,
}
pub(crate) struct HeadRetained {
    pub(crate) graph: bool,
    pub(crate) run: bool,
    pub(crate) pending_window: bool,
    pub(crate) actual_counts: [usize; 4],
    pub(crate) retirement_counts: [usize; 5],
    pub(crate) source_staging_bytes: usize,
    pub(crate) resident_immutable_weight: bool,
}
impl HeadWitness {
    pub(crate) fn inspect(&self) -> Result<HeadRetained> {
        let head = self
            .head
            .upgrade()
            .ok_or_else(|| Error::State("streamed head original graph not retained".into()))?;
        let window = self.window.as_ref().and_then(Weak::upgrade);
        Ok(HeadRetained {
            graph: head.retained_graph()?,
            run: self.run.upgrade().is_some(),
            pending_window: window.is_some(),
            actual_counts: head.counts()?,
            retirement_counts: head.retirement_counts()?,
            source_staging_bytes: head.retained_source_staging_bytes()?,
            resident_immutable_weight: head.cold.resident.is_some() && head.retained_graph()?,
        })
    }
}
#[cfg(test)]
mod retirement_tests {
    use super::*;
    #[test]
    fn selected_head_is_not_independent_request_batching() {
        for columns in [1024, 2048] {
            for input_rows in [1, 47, 57, 62, 67, 72] {
                let input = input_rows * columns * 2;
                let output = 248320 * 2;
                require_selected_extent(columns, 248320, input_rows, input_rows - 1, input, output)
                    .unwrap();
                assert!(
                    require_selected_extent(columns, 248320, input_rows, input_rows, input, output)
                        .is_err()
                );
                assert!(
                    require_selected_extent(
                        columns,
                        248320,
                        input_rows,
                        input_rows - 1,
                        input - 2,
                        output
                    )
                    .is_err()
                );
                assert!(
                    require_selected_extent(
                        columns,
                        248320,
                        input_rows,
                        input_rows - 1,
                        input,
                        output - 2
                    )
                    .is_err()
                );
                if input_rows > 1 {
                    assert!(
                        require_selected_extent(columns, 248320, input_rows, 0, input, output)
                            .is_err()
                    );
                    assert!(
                        require_selected_extent(
                            columns,
                            248320,
                            input_rows,
                            input_rows - 1,
                            input,
                            output * input_rows
                        )
                        .is_err()
                    );
                }
            }
            for rows in [2, 4, 8, 16] {
                assert!(
                    require_selected_extent(
                        columns,
                        248320,
                        rows,
                        rows - 1,
                        rows * columns * 2,
                        248320 * 2
                    )
                    .is_err()
                );
            }
        }
        assert!(require_selected_extent(usize::MAX, 248320, 47, 46, 0, 248320 * 2).is_err());
        assert!(require_selected_extent(1024, usize::MAX, 47, 46, 47 * 1024 * 2, 0).is_err());
    }
    #[test]
    fn two_window_control_and_authenticated_source_bounds_are_explicit() {
        let controls = std::mem::size_of::<HeadRetirementRing>()
            + RETIREMENT_SLOTS
                * (2 * std::mem::size_of::<OwnedCudaTransferRetirement>()
                    + std::mem::size_of::<Arc<WindowBytes>>())
            + std::mem::size_of::<(Arc<PreparedBf16Head>, ResidentLease, ResidentLease)>();
        assert!(controls < 16 * 1024);
        assert_eq!(RETIREMENT_SLOTS, 2);
        assert_eq!(STREAMED_WINDOW_STAGING_UPPER, 2_097_200);
        assert_eq!(RETIREMENT_SLOTS * STREAMED_WINDOW_STAGING_UPPER, 4_194_400);
    }
}
