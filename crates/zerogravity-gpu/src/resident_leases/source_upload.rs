//! Finite original tied-table upload; existing initialization owns completion.
use super::*;
use std::sync::Weak;
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedMatrix, WindowWorkspace};
use zerogravity_core::accelerator::program::SourceTensor;
use zerogravity_core::quant::QuantKind;
use zerogravity_model_qwen35::original_binding::{OriginalWindowModelSource, QwenMatrixRole};

use super::resident_matrix::layout::{Coverage, Geometry};

// Retained .8B qualification fixtures only; production derives source geometry.
pub(crate) const TIED_BYTES: usize = 508_559_360;
pub(crate) const TIED_ROW_BYTES: usize = 2048;
pub(crate) const TIED_WINDOW_BYTES: usize = 1024 * 1024;
pub(crate) const TIED_WINDOWS: usize = 485;
const METADATA_BYTES: usize = 64 * 1024;

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("original tied-table upload cancelled".into()));
    }
    Ok(())
}

struct WindowBytes(TensorWindow);
impl AsRef<[u8]> for WindowBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}
struct Progress {
    pending: Option<Arc<WindowBytes>>,
    coverage: Coverage,
    enqueued: usize,
    checkpoints: usize,
}

pub(super) struct UploadOwner {
    // Pending actual host storage drops before its source/metadata charges.
    progress: Mutex<Progress>,
    lease: ResidentLease,
    source: Arc<OriginalWindowModelSource>,
    descriptor: SourceTensor,
    matrix: VerifiedMatrix,
    geometry: Geometry,
    _metadata: WindowWorkspace,
}
impl UploadOwner {
    fn progress(&self) -> Result<std::sync::MutexGuard<'_, Progress>> {
        self.progress
            .lock()
            .map_err(|_| Error::State("tied upload progress poisoned".into()))
    }
    pub(super) fn require_complete(&self) -> Result<()> {
        let progress = self.progress()?;
        progress.coverage.require_complete()?;
        if progress.pending.is_some()
            || progress.enqueued != self.geometry.windows()
            || progress.checkpoints != self.geometry.windows()
            || self.lease.bytes() != self.geometry.bytes()
            || self.lease.role() != BufferRole::ImmutableWeights
            || self.lease.source_identity() != self.source.source_identity()
        {
            return Err(Error::State(
                "original full tied-table upload incomplete".into(),
            ));
        }
        self.source
            .validate_matrix(QwenMatrixRole::Output, &self.descriptor)?;
        self.source
            .validate_matrix(QwenMatrixRole::Embedding, &self.descriptor)
    }
}

pub(crate) struct PendingTiedWeight {
    owner: Arc<UploadOwner>,
}
pub(crate) struct TiedWeight {
    owner: Arc<UploadOwner>,
    matrix: Arc<super::ResidentMatrix>,
}
impl PendingTiedWeight {
    /// Original lease metadata only; no initialization/completion authority.
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.owner.lease
    }
    pub(crate) fn bind_initialized(self) -> Result<Arc<TiedWeight>> {
        self.owner.require_complete()?;
        if !self.owner.lease.is_initialized() {
            return Err(Error::State(
                "tied-table original initialization is not complete".into(),
            ));
        }
        self.owner.lease.service().ensure_usable()?;
        let matrix = super::ResidentMatrix::from_completed_upload(
            self.owner.lease.clone(),
            self.owner.source.source().clone(),
            self.owner.matrix.clone(),
            self.owner.clone(),
        )?;
        Ok(Arc::new(TiedWeight {
            owner: self.owner,
            matrix,
        }))
    }
}
impl TiedWeight {
    pub(crate) fn source_layout(
        source: &OriginalWindowModelSource,
    ) -> Result<cubecl::cuda::Bf16MatrixLayout> {
        let member = source.tied_output_descriptor()?;
        let matrix = source.source().matrix(&member.name, member.axes)?;
        super::ResidentMatrix::authenticate(source.source(), &matrix)?.cuda()
    }
    pub(crate) fn matrix(&self) -> &Arc<super::ResidentMatrix> {
        &self.matrix
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.owner.lease
    }
    pub(crate) fn source(&self) -> &Arc<OriginalWindowModelSource> {
        &self.owner.source
    }
    pub(crate) async fn verify_source_windows(
        self: &Arc<Self>,
        parent: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.owner.require_complete()?;
        let service = self.lease().service();
        if !self.lease().is_initialized() {
            return Err(Error::State(
                "verify original initialized tied table".into(),
            ));
        }
        let admission = Arc::new(self.source().source().reserve_workspace(32 * 1024 * 1024)?);
        for index in 0..self.owner.geometry.windows() {
            not_cancelled(cancelled)?;
            let first = index * (TIED_WINDOW_BYTES / self.owner.geometry.row_bytes());
            let region = self
                .owner
                .geometry
                .region(first)?
                .ok_or_else(|| Error::State("tied verification region absent".into()))?;
            let end = region.end;
            let window = Arc::new(match cancelled {
                Some(flag) => self.source().source().read_rows_cancellable(
                    &self.owner.matrix,
                    first..end,
                    flag,
                )?,
                None => self
                    .source()
                    .source()
                    .read_rows(&self.owner.matrix, first..end)?,
            });
            if window.row_range() != (first..end)
                || window.as_slice().len() != region.bytes
                || window.source_offset()
                    != self.owner.matrix.tensor().byte_offset + region.offset as u64
                || window.matrix().source_identity() != self.source().source_identity()
            {
                return Err(Error::State(
                    "verify tied exact authenticated source window".into(),
                ));
            }
            let ticket = service.queue().submit(
                service.budget(),
                service.owner().clone(),
                &[zerogravity_core::resource_arbiter::AllocationRequest {
                    kind: zerogravity_core::resource_arbiter::AllocationClass::TransferStaging,
                    bytes: 32 * 1024 * 1024,
                }],
                None,
            )?;
            let mut slot = ticket.wait().await?;
            not_cancelled(cancelled)?;
            slot.begin_submission()?;
            let mut flight = service.submit(0, &[(self.lease().clone(), DeviceAccess::Read)])?;
            flight.retain_completion_owner(parent.clone())?;
            flight.retain_completion_owner(self.clone())?;
            flight.retain_completion_owner(admission.clone())?;
            flight.retain_completion_owner(window.clone())?;
            let reads = [self.lease().readback(region.offset, region.bytes)?];
            drop(
                flight
                    .complete_read_only_readback(&reads, |raw| {
                        if raw.len() != 1 || &raw[0][..] != window.as_slice() {
                            return Err(Error::State(
                                "complete resident tied source bytes changed".into(),
                            ));
                        }
                        Ok(())
                    })
                    .await?,
            );
            slot.finish()?;
        }
        Ok(self.owner.geometry.windows())
    }
    pub(crate) fn bytes(&self) -> usize {
        self.owner.geometry.bytes()
    }
    pub(crate) fn windows(&self) -> usize {
        self.owner.geometry.windows()
    }
    pub(crate) fn copy_counts(&self) -> Result<(usize, usize)> {
        self.owner.require_complete()?;
        let progress = self.owner.progress()?;
        Ok((progress.enqueued, progress.checkpoints))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TiedUploadCut {
    Complete,
    CancelBeforeFirstEnqueue,
    UnknownAfterMiddleEnqueue,
    CancelAfterMiddleEnqueue,
}
pub(crate) enum TiedUploadAttempt {
    Complete(PendingTiedWeight),
    Cut(TiedUploadWitness),
}
pub(crate) struct TiedUploadWitness {
    owner: Weak<UploadOwner>,
    source: Weak<OriginalWindowModelSource>,
    window: Weak<WindowBytes>,
    service: Weak<ResidentService>,
    lease: BufferLeaseId,
    cut: TiedUploadCut,
}
pub(crate) struct TiedUploadRetained {
    pub(crate) source: bool,
    pub(crate) owner: bool,
    pub(crate) window: bool,
    pub(crate) service: bool,
    pub(crate) enqueued: usize,
    pub(crate) checkpoints: usize,
    pub(crate) initialized: bool,
    pub(crate) staging_bytes: usize,
    pub(crate) resident_bytes: usize,
    pub(crate) pending_events: usize,
    pub(crate) uploaded_bytes: u64,
    pub(crate) handles: usize,
    pub(crate) completion_owners: usize,
    pub(crate) reserved: u64,
    pub(crate) fenced: bool,
}
impl TiedUploadWitness {
    fn new(owner: &Arc<UploadOwner>, window: &Arc<WindowBytes>, cut: TiedUploadCut) -> Self {
        Self {
            owner: Arc::downgrade(owner),
            source: Arc::downgrade(&owner.source),
            window: Arc::downgrade(window),
            service: Arc::downgrade(owner.lease.service()),
            lease: owner.lease.id(),
            cut,
        }
    }
    // Call only after ALL caller-owned strong references and the armed
    // initialization/fair slot have dropped. No GPU operation is permitted.
    pub(crate) fn inspect(self) -> Result<TiedUploadRetained> {
        let owner = self.owner.upgrade();
        let source = self.source.upgrade();
        let window = self.window.upgrade();
        let service = self.service.upgrade();
        let retained = [
            owner.is_some(),
            source.is_some(),
            window.is_some(),
            service.is_some(),
        ];
        let owner = owner.ok_or_else(|| Error::State("unknown tied upload lost owner".into()))?;
        let source =
            source.ok_or_else(|| Error::State("unknown tied upload lost source".into()))?;
        let window =
            window.ok_or_else(|| Error::State("unknown tied upload lost window".into()))?;
        let service =
            service.ok_or_else(|| Error::State("unknown tied upload lost service".into()))?;
        let progress = owner.progress()?;
        let expected = match self.cut {
            TiedUploadCut::CancelBeforeFirstEnqueue => (0, 0),
            TiedUploadCut::UnknownAfterMiddleEnqueue | TiedUploadCut::CancelAfterMiddleEnqueue => (
                owner.geometry.windows() / 2 + 1,
                owner.geometry.windows() / 2,
            ),
            TiedUploadCut::Complete => {
                return Err(Error::State(
                    "complete upload has no unknown witness".into(),
                ));
            }
        };
        let first = expected.1 * (TIED_WINDOW_BYTES / owner.geometry.row_bytes());
        let region = owner
            .geometry
            .region(first)?
            .ok_or_else(|| Error::State("retained tied region absent".into()))?;
        let source_offset = owner
            .matrix
            .tensor()
            .byte_offset
            .checked_add(region.offset as u64)
            .ok_or_else(|| Error::Limit("retained tied source offset overflow".into()))?;
        if !Arc::ptr_eq(&source, &owner.source)
            || !Arc::ptr_eq(&service, owner.lease.service())
            || progress
                .pending
                .as_ref()
                .is_none_or(|pending| !Arc::ptr_eq(pending, &window))
            || owner.lease.id() != self.lease
            || owner.lease.bytes() != owner.geometry.bytes()
            || owner.lease.role() != BufferRole::ImmutableWeights
            || owner.lease.source_identity() != source.source_identity()
            || owner.lease.is_initialized()
            || window.0.row_range() != (first..region.end)
            || window.0.source_offset() != source_offset
            || window.0.as_slice().len() != region.bytes
            || window.0.matrix().source_identity() != source.source_identity()
            || window.0.matrix().tensor().name != owner.descriptor.name
            || owner._metadata.bytes() != METADATA_BYTES
            || !owner._metadata.source().same_issuer(source.source())
            || (progress.enqueued, progress.checkpoints) != expected
            || service.ensure_usable().is_ok()
            || service.resident_bytes()? != owner.geometry.bytes()
            || service.pending_events()? != 1
            || service.uploaded_value_bytes() != (expected.0 * TIED_WINDOW_BYTES) as u64
        {
            return Err(Error::State(
                "unknown tied upload original graph/coverage differs".into(),
            ));
        }
        let snapshot = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?;
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let completion_owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        if handles != 1
            || completion_owners == 0
            || !snapshot.fenced
            || snapshot.reservations != 1
            || snapshot.per_class.len() != 1
            || snapshot
                .per_class
                .get(&zerogravity_core::resource_arbiter::AllocationClass::ProviderEnvelope)
                != Some(&snapshot.reserved)
        {
            return Err(Error::State(
                "unknown tied upload original quarantine absent".into(),
            ));
        }
        Ok(TiedUploadRetained {
            source: retained[1],
            owner: retained[0],
            window: retained[2],
            service: retained[3],
            enqueued: progress.enqueued,
            checkpoints: progress.checkpoints,
            initialized: owner.lease.is_initialized(),
            staging_bytes: window.0.charged_staging_bytes(),
            resident_bytes: service.resident_bytes()?,
            pending_events: service.pending_events()?,
            uploaded_bytes: service.uploaded_value_bytes(),
            handles,
            completion_owners,
            reserved: snapshot.reserved,
            fenced: snapshot.fenced,
        })
    }
}

fn descriptor(
    source: &OriginalWindowModelSource,
    service: &Arc<ResidentService>,
) -> Result<(SourceTensor, VerifiedMatrix)> {
    let embed = source.tied_output_descriptor()?;
    if embed.source != service.source_identity() || embed.kind != QuantKind::Bf16 {
        return Err(Error::Unsupported("source-bound tied BF16 matrix".into()));
    }
    let matrix = source.source().matrix(&embed.name, embed.axes)?;
    super::ResidentMatrix::authenticate(source.source(), &matrix)?;
    Ok((embed, matrix))
}

impl ResidentInitialization<'_> {
    pub(crate) async fn upload_original_tied_bf16(
        &mut self,
        source: Arc<OriginalWindowModelSource>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PendingTiedWeight> {
        match self
            .upload_tied_bf16_inner(source, TiedUploadCut::Complete, cancelled)
            .await?
        {
            TiedUploadAttempt::Complete(pending) => Ok(pending),
            TiedUploadAttempt::Cut(_) => Err(Error::State(
                "positive tied upload reached fault cut".into(),
            )),
        }
    }

    pub(crate) async fn upload_tied_bf16_inner(
        &mut self,
        source: Arc<OriginalWindowModelSource>,
        cut: TiedUploadCut,
        cancelled: Option<&AtomicBool>,
    ) -> Result<TiedUploadAttempt> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        if !self.source_uploads.is_empty() {
            return Err(Error::State(
                "one original tied upload per initialization".into(),
            ));
        }
        let metadata = source.source().reserve_workspace(METADATA_BYTES)?;
        let (descriptor, matrix) = descriptor(&source, self.service)?;
        let queue = self.service.pinned_cuda_queue()?.clone();
        let geometry = super::ResidentMatrix::authenticate(source.source(), &matrix)?;
        let lease = self.allocate_empty(geometry.bytes(), BufferRole::ImmutableWeights)?;
        let owner = Arc::new(UploadOwner {
            progress: Mutex::new(Progress {
                coverage: Coverage::new(geometry),
                pending: None,
                enqueued: 0,
                checkpoints: 0,
            }),
            lease,
            source,
            descriptor,
            matrix,
            geometry,
            _metadata: metadata,
        });
        self.retain_completion_owner(owner.clone())?;
        self.source_uploads.push(owner.clone());
        for index in 0..geometry.windows() {
            not_cancelled(cancelled)?;
            let region = owner
                .progress()?
                .coverage
                .next()?
                .ok_or_else(|| Error::State("tied upload coverage ended early".into()))?;
            let window = match cancelled {
                Some(flag) => owner.source.source().read_rows_cancellable(
                    &owner.matrix,
                    region.first..region.end,
                    flag,
                )?,
                None => owner
                    .source
                    .source()
                    .read_rows(&owner.matrix, region.first..region.end)?,
            };
            let expected_source_offset = owner
                .matrix
                .tensor()
                .byte_offset
                .checked_add(region.offset as u64)
                .ok_or_else(|| Error::Limit("tied source byte offset".into()))?;
            if window.row_range() != (region.first..region.end)
                || window.source_offset() != expected_source_offset
                || window.as_slice().len() != region.bytes
                || window.matrix().source_identity() != owner.source.source_identity()
                || window.matrix().tensor().name != owner.descriptor.name
                || window.matrix().tensor().kind != QuantKind::Bf16
                || window
                    .as_slice()
                    .chunks_exact(2)
                    .any(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7f80 == 0x7f80)
            {
                return Err(Error::State(
                    "tied upload original immutable row window".into(),
                ));
            }
            not_cancelled(cancelled)?;
            let window = Arc::new(WindowBytes(window));
            {
                let mut progress = owner.progress()?;
                progress.coverage.begin(region)?;
                if progress.pending.replace(window.clone()).is_some() {
                    return Err(Error::State("overlapping original host window".into()));
                }
            }
            if index == 0 && cut == TiedUploadCut::CancelBeforeFirstEnqueue {
                let injected = AtomicBool::new(true);
                if not_cancelled(Some(&injected)).is_ok() {
                    return Err(Error::State("injected cancellation did not refuse".into()));
                }
                return Ok(TiedUploadAttempt::Cut(TiedUploadWitness::new(
                    &owner, &window, cut,
                )));
            }
            not_cancelled(cancelled)?;
            // SAFETY: exact immutable TensorWindow/admission Arc is already
            // retained in this original armed initialization's pending owner.
            unsafe {
                queue.enqueue_bf16_matrix_window(
                    owner.lease.handle(),
                    cubecl::cuda::Bf16MatrixLayout::new(
                        owner.matrix.rows(),
                        owner.matrix.columns(),
                    )
                    .map_err(|e| Error::State(format!("tied source matrix layout: {e}")))?,
                    region.offset,
                    window.clone(),
                )
            }
            .map_err(|e| Error::State(format!("original tied H2D enqueue: {e}")))?;
            owner.progress()?.enqueued += 1;
            self.service
                .uploaded_value_bytes
                .fetch_add(region.bytes as u64, Ordering::AcqRel);
            if index == geometry.windows() / 2
                && matches!(
                    cut,
                    TiedUploadCut::UnknownAfterMiddleEnqueue
                        | TiedUploadCut::CancelAfterMiddleEnqueue
                )
            {
                if cut == TiedUploadCut::CancelAfterMiddleEnqueue {
                    let injected = AtomicBool::new(true);
                    if not_cancelled(Some(&injected)).is_ok() {
                        return Err(Error::State(
                            "injected late cancellation did not refuse".into(),
                        ));
                    }
                }
                return Ok(TiedUploadAttempt::Cut(TiedUploadWitness::new(
                    &owner, &window, cut,
                )));
            }
            not_cancelled(cancelled)?;
            let view = owner
                .lease
                .handle()
                .offset_start(region.offset as u64)
                .offset_end((geometry.bytes() - region.offset - region.bytes) as u64);
            self.service
                .client
                .sync_buffers([&view])
                .await
                .map_err(|e| Error::State(format!("original tied window completion: {e}")))?;
            self.service.ensure_usable()?;
            not_cancelled(cancelled)?;
            // This is the ONLY production coverage-credit call. A matching
            // number/host callback cannot enter it without that actual barrier.
            let retired = {
                let mut progress = owner.progress()?;
                progress.coverage.checkpoint(region)?;
                progress.checkpoints += 1;
                progress.pending.take()
            };
            drop(retired);
            drop(window);
        }
        owner.require_complete()?;
        self.uploaded.insert(owner.lease.id());
        Ok(TiedUploadAttempt::Complete(PendingTiedWeight { owner }))
    }
}
