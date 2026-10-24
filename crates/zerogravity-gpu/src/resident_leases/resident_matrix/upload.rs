//! Exact original row coverage, credited only after the original buffer barrier.
use super::*;
use layout::{CONTROL, Coverage};
use std::sync::Weak;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Cut {
    Never,
    First,
    Last,
}
pub(super) enum Attempt {
    Complete(PendingResidentMatrix),
    Cut(Weak<UploadOwner>),
}
pub(super) struct Snapshot {
    pub(super) commands: usize,
    pub(super) checkpoints: usize,
    pub(super) pending: bool,
    pub(super) initialized: bool,
    pub(super) source: SourceIdentity,
}
use zerogravity_artifacts::source_weights::{TensorWindow, WindowWorkspace};

struct PendingWindow {
    _window: TensorWindow,
    bytes: Bytes,
    _staging: WindowWorkspace,
}
struct Progress {
    coverage: Coverage,
    pending: Option<PendingWindow>,
    commands: usize,
    checkpoints: usize,
}
pub(in crate::resident_leases) struct UploadOwner {
    progress: Mutex<Progress>,
    lease: ResidentLease,
    matrix: Bf16MatrixSource,
    _control: WindowWorkspace,
}
impl UploadOwner {
    pub(super) fn snapshot(&self) -> Result<Snapshot> {
        let p = self
            .progress
            .lock()
            .map_err(|_| Error::State("BF16 upload progress poisoned".into()))?;
        Ok(Snapshot {
            commands: p.commands,
            checkpoints: p.checkpoints,
            pending: p.pending.is_some(),
            initialized: self.lease.is_initialized(),
            source: self.matrix.source().source_identity(),
        })
    }
    pub(in crate::resident_leases) fn require_complete(&self) -> Result<()> {
        let progress = self
            .progress
            .lock()
            .map_err(|_| Error::State("BF16 upload progress poisoned".into()))?;
        progress.coverage.require_complete()?;
        if progress.pending.is_some()
            || progress.commands != self.matrix.windows()
            || progress.commands != progress.checkpoints
        {
            return Err(Error::State(
                "BF16 actual upload/checkpoint incomplete".into(),
            ));
        }
        Ok(())
    }
}
pub(crate) struct PendingResidentMatrix {
    owner: Arc<UploadOwner>,
}
impl PendingResidentMatrix {
    #[cfg(any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ))]
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.owner.lease
    }
    pub(super) fn counts(&self) -> Result<[usize; 2]> {
        let p = self.owner.snapshot()?;
        Ok([p.commands, p.checkpoints])
    }
    pub(crate) fn bind_initialized(self) -> Result<Arc<ResidentMatrix>> {
        self.owner.require_complete()?;
        ResidentMatrix::from_completed_upload(
            self.owner.lease.clone(),
            self.owner.matrix.source().clone(),
            self.owner.matrix.matrix().clone(),
            self.owner,
        )
    }
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err(Error::State("verified matrix upload cancelled".into()));
    }
    Ok(())
}
impl ResidentInitialization<'_> {
    pub(crate) async fn upload_bf16_matrix(
        &mut self,
        source: VerifiedTensorSource,
        matrix: VerifiedMatrix,
        cancelled: Option<&AtomicBool>,
    ) -> Result<PendingResidentMatrix> {
        match self
            .upload_bf16_matrix_cut(source, matrix, cancelled, Cut::Never)
            .await?
        {
            Attempt::Complete(pending) => Ok(pending),
            Attempt::Cut(_) => Err(Error::State("unexpected BF16 upload cut".into())),
        }
    }
    pub(super) async fn upload_bf16_matrix_cut(
        &mut self,
        source: VerifiedTensorSource,
        matrix: VerifiedMatrix,
        cancelled: Option<&AtomicBool>,
        cut: Cut,
    ) -> Result<Attempt> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        let matrix = Bf16MatrixSource::new(source, matrix)?;
        let geometry = matrix.geometry();
        require_portable(self.service, geometry)?;
        if matrix.source().source_identity() != self.service.source_identity()
            || self.service.client().is_capturing()
        {
            return Err(Error::State("BF16 upload source/service/capture".into()));
        }
        let control = matrix.source().reserve_workspace(CONTROL)?;
        let lease = self.allocate_empty(geometry.bytes(), BufferRole::ImmutableWeights)?;
        let owner = Arc::new(UploadOwner {
            progress: Mutex::new(Progress {
                coverage: Coverage::new(geometry),
                pending: None,
                commands: 0,
                checkpoints: 0,
            }),
            lease,
            matrix,
            _control: control,
        });
        self.retain_completion_owner(owner.clone())?;
        self.matrix_source_uploads.push(owner.clone());
        loop {
            not_cancelled(cancelled)?;
            let region = owner
                .progress
                .lock()
                .map_err(|_| Error::State("BF16 upload progress poisoned".into()))?
                .coverage
                .next()?;
            let Some(region) = region else {
                break;
            };
            let window = owner.matrix.read_region(region, cancelled)?;
            owner.matrix.require_window(&window, region)?;
            // TensorWindow already owns its exact source staging. This separate
            // ticket prices Bytes plus the runtime pinned/fallback bounce.
            let staging = owner.matrix.source().reserve_workspace(2 * region.bytes)?;
            let copy = window.as_slice().to_vec();
            if copy.capacity() != region.bytes {
                return Err(Error::Limit("BF16 upload copy capacity".into()));
            }
            let bytes = Bytes::from_bytes_vec(copy);
            if bytes.capacity() > region.bytes {
                return Err(Error::Limit("BF16 aligned upload capacity".into()));
            }
            let bytes = bytes.shared();
            {
                let mut progress = owner
                    .progress
                    .lock()
                    .map_err(|_| Error::State("BF16 upload progress poisoned".into()))?;
                progress.coverage.begin(region)?;
                progress.pending = Some(PendingWindow {
                    _window: window,
                    bytes,
                    _staging: staging,
                });
            }
            not_cancelled(cancelled)?;
            let bytes = owner
                .progress
                .lock()
                .map_err(|_| Error::State("BF16 upload progress poisoned".into()))?
                .pending
                .as_ref()
                .ok_or_else(|| Error::State("BF16 pending source absent".into()))?
                .bytes
                .clone();
            let view = owner
                .lease
                .handle()
                .offset_start(region.offset as u64)
                .offset_end((geometry.bytes() - region.offset - region.bytes) as u64);
            self.service.client().write(&view, bytes);
            owner
                .progress
                .lock()
                .map_err(|_| Error::State("BF16 upload progress poisoned".into()))?
                .commands += 1;
            self.service.record_value_upload(region.bytes);
            if (cut == Cut::First && region.first == 0)
                || (cut == Cut::Last && region.end == geometry.rows())
            {
                return Ok(Attempt::Cut(Arc::downgrade(&owner)));
            }
            self.service
                .client()
                .sync_buffers([&view])
                .await
                .map_err(|e| Error::State(format!("BF16 exact upload barrier: {e}")))?;
            self.service.ensure_usable()?;
            not_cancelled(cancelled)?;
            let retired = {
                let mut progress = owner
                    .progress
                    .lock()
                    .map_err(|_| Error::State("BF16 upload progress poisoned".into()))?;
                progress.coverage.checkpoint(region)?;
                progress.checkpoints += 1;
                progress.pending.take()
            };
            drop(retired);
        }
        owner.require_complete()?;
        self.uploaded.insert(owner.lease.id());
        Ok(Attempt::Complete(PendingResidentMatrix { owner }))
    }
}
