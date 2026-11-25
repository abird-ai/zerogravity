//! Bounded authenticated Safetensors -> one packed resident BF16 projection.
//! Source member order is the mathematical concatenation order. This transport
//! never chooses model operations or launch eligibility.

use super::{ResidentInitialization, ResidentLease, ResidentService};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedMatrix, VerifiedTensorSource};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{MatrixAxes, SourceTensor},
    device_events::BufferRole,
    encoding::StoredTensorEncoding,
};

use super::source_page::{PAGE_BYTES, PageGeometry};

#[derive(Clone)]
struct PackedSourceOwner {
    source: VerifiedTensorSource,
    matrices: Arc<[VerifiedMatrix]>,
    members: Arc<[SourceTensor]>,
}

/// Ordered authenticated matrix members, with optional packed resident storage.
/// A streamed binding retains source identity, not a whole-matrix device lease.
pub(crate) struct BoundSourceMatrices {
    owner: Arc<PackedSourceOwner>,
    lease: Option<ResidentLease>,
    rows: usize,
    columns: usize,
}
impl BoundSourceMatrices {
    pub(crate) fn same_source(&self, source: &VerifiedTensorSource) -> bool {
        self.owner.source.same_issuer(source)
    }
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
    pub(crate) fn columns(&self) -> usize {
        self.columns
    }
    pub(crate) fn lease(&self) -> Option<&ResidentLease> {
        self.lease.as_ref()
    }
    pub(crate) fn matrices(&self) -> &[VerifiedMatrix] {
        &self.owner.matrices
    }

    /// Every member was authenticated once against the exact immutable source
    /// owner before binding. Request descriptors must remain exactly equal.
    pub(crate) fn require_members(&self, members: &[SourceTensor]) -> Result<()> {
        if members != self.owner.members.as_ref() || members.len() != self.owner.matrices.len() {
            return Err(Error::State(
                "CUDA forward source matrix members changed".into(),
            ));
        }
        Ok(())
    }
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "CUDA forward projection upload cancelled".into(),
        ));
    }
    Ok(())
}

impl ResidentInitialization<'_> {
    /// Pack one to three source-declared full matrices in order into one resident
    /// allocation. Each bounded source window is copied then physically synced
    /// before the next window; the full source owner remains a completion owner.
    pub(crate) async fn prepare_forward_matrices(
        &mut self,
        service: &Arc<ResidentService>,
        source: VerifiedTensorSource,
        members: &[SourceTensor],
        resident: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<BoundSourceMatrices>> {
        self.require_service(service)?;
        not_cancelled(cancelled)?;
        if members.is_empty()
            || members.len() > 3
            || source.source_identity() != service.source_identity()
            || service.client().is_capturing()
        {
            return Err(Error::State(
                "CUDA forward source matrix/member ownership differs".into(),
            ));
        }

        let mut matrices = Vec::with_capacity(members.len());
        let mut rows = 0usize;
        let mut columns = None;
        let mut total_bytes = 0usize;
        for member in members {
            if member.source != source.source_identity()
                || member.kind
                    != StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16)
                || member.axes != MatrixAxes::OutputInput
                || member.rows == 0
                || member.columns == 0
            {
                return Err(Error::State(
                    "CUDA forward source matrix member contract differs".into(),
                ));
            }
            let matrix = source.bind_declared_matrix(member)?;
            PageGeometry::new(matrix.rows(), matrix.columns())?;
            if columns
                .replace(member.columns)
                .is_some_and(|value| value != member.columns)
            {
                return Err(Error::State(
                    "CUDA forward source matrix input widths differ".into(),
                ));
            }
            rows = rows
                .checked_add(member.rows)
                .ok_or_else(|| Error::Limit("CUDA forward source matrix rows overflow".into()))?;
            total_bytes = total_bytes
                .checked_add(
                    member
                        .rows
                        .checked_mul(member.columns)
                        .and_then(|elements| elements.checked_mul(2))
                        .ok_or_else(|| {
                            Error::Limit("CUDA forward source matrix bytes overflow".into())
                        })?,
                )
                .ok_or_else(|| {
                    Error::Limit("CUDA forward packed source matrix bytes overflow".into())
                })?;
            matrices.push(matrix);
        }
        let columns = columns
            .ok_or_else(|| Error::State("CUDA forward source matrix columns absent".into()))?;
        let owner = Arc::new(PackedSourceOwner {
            source: source.clone(),
            matrices: matrices.into(),
            members: members.to_vec().into(),
        });
        self.retain_completion_owner(owner.clone())?;
        if !resident {
            return Ok(Arc::new(BoundSourceMatrices {
                owner,
                lease: None,
                rows,
                columns,
            }));
        }
        if total_bytes as u64 > service.client().properties().memory.max_page_size {
            return Err(Error::Unsupported(
                "packed source matrix exceeds device allocation page".into(),
            ));
        }
        let queue = service.pinned_cuda_queue()?;
        let lease = self.allocate_empty(total_bytes, BufferRole::ImmutableWeights)?;
        let pending = Arc::new(Mutex::new(None::<Arc<TensorWindow>>));
        self.retain_completion_owner(pending.clone())?;

        let mut destination_offset = 0usize;
        for matrix in owner.matrices.iter() {
            let geometry = PageGeometry::new(matrix.rows(), matrix.columns())?;
            for first in (0..geometry.rows).step_by(geometry.rows_per_page) {
                not_cancelled(cancelled)?;
                let end = first + (geometry.rows - first).min(geometry.rows_per_page);
                let window = match cancelled {
                    Some(flag) => source.read_rows_cancellable(matrix, first..end, flag)?,
                    None => source.read_rows(matrix, first..end)?,
                };
                window.require_rows(matrix, first..end)?;
                let window = Arc::new(window);
                let bytes = window.as_slice().len();
                if bytes > PAGE_BYTES {
                    return Err(Error::State(
                        "packed source exceeds bounded transfer".into(),
                    ));
                }
                let start = destination_offset
                    .checked_add(first * geometry.row_bytes)
                    .ok_or_else(|| Error::Limit("packed source destination offset".into()))?;
                let view = lease.view(start, bytes)?;
                {
                    let mut held = pending
                        .lock()
                        .map_err(|_| Error::State("packed source pending owner poisoned".into()))?;
                    if held.is_some() {
                        return Err(Error::State("packed source window not retired".into()));
                    }
                    *held = Some(window.clone());
                }
                not_cancelled(cancelled)?;
                // SAFETY: the original initialization owns the complete write
                // lease and the exact pending source through its real barrier
                // or unknown-completion quarantine. No uncharged host copy.
                unsafe {
                    queue.enqueue_owned_source_window(lease.handle(), total_bytes, start, window)
                }
                .map_err(|error| Error::State(format!("packed original source H2D: {error}")))?;
                service.record_value_upload(bytes);
                service
                    .client()
                    .sync_buffers([&view])
                    .await
                    .map_err(|error| {
                        Error::State(format!("packed source upload barrier: {error}"))
                    })?;
                service.ensure_usable()?;
                let retired = pending
                    .lock()
                    .map_err(|_| Error::State("packed source pending owner poisoned".into()))?
                    .take();
                drop(retired);
            }
            destination_offset = destination_offset
                .checked_add(geometry.rows * geometry.row_bytes)
                .ok_or_else(|| Error::Limit("packed source destination overflow".into()))?;
        }
        not_cancelled(cancelled)?;
        if destination_offset != total_bytes {
            return Err(Error::State(
                "CUDA forward source matrix packed byte coverage incomplete".into(),
            ));
        }
        self.uploaded.insert(lease.id());

        Ok(Arc::new(BoundSourceMatrices {
            owner,
            lease: Some(lease),
            rows,
            columns,
        }))
    }

    pub(crate) async fn upload_forward_projection(
        &mut self,
        service: &Arc<ResidentService>,
        source: VerifiedTensorSource,
        members: &[SourceTensor],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<BoundSourceMatrices>> {
        self.prepare_forward_matrices(service, source, members, true, cancelled)
            .await
    }
}
