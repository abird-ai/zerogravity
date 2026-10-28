//! Bounded authenticated Safetensors -> one packed resident BF16 projection.
//! Source member order is the mathematical concatenation order. This transport
//! never chooses model operations or launch eligibility.

use super::{ResidentInitialization, ResidentLease, ResidentService};
use cubecl::bytes::Bytes;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::{
    quant::QuantKind,
    source_weights::{VerifiedMatrix, VerifiedTensorSource},
};
use zerogravity_core::{
    Error, Result,
    accelerator::{
        SourceIdentity,
        program::{MatrixAxes, SourceTensor},
    },
    device_events::BufferRole,
};

const WINDOW_BYTES: usize = 1024 * 1024;

#[derive(Clone)]
struct PackedSourceOwner {
    source: VerifiedTensorSource,
    matrices: Arc<[VerifiedMatrix]>,
    members: Arc<[SourceTensor]>,
}

/// One exact packed [sum(output), input] BF16 matrix, retaining the authenticated
/// source owner and every original member binding.
pub(crate) struct PackedProjectionWeights {
    owner: Arc<PackedSourceOwner>,
    lease: ResidentLease,
    rows: usize,
    columns: usize,
}
impl PackedProjectionWeights {
    pub(crate) fn source_identity(&self) -> SourceIdentity {
        self.owner.source.source_identity()
    }
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
    pub(crate) fn columns(&self) -> usize {
        self.columns
    }
    pub(crate) fn lease(&self) -> &ResidentLease {
        &self.lease
    }
    pub(crate) fn require_members(&self, members: &[SourceTensor]) -> Result<()> {
        if members != self.owner.members.as_ref() || members.len() != self.owner.matrices.len() {
            return Err(Error::State(
                "ordinary CUDA projection source members changed".into(),
            ));
        }
        for (member, matrix) in members.iter().zip(self.owner.matrices.iter()) {
            if member.source != self.source_identity()
                || member.kind != QuantKind::Bf16
                || member.axes != MatrixAxes::OutputInput
                || member.rows != matrix.rows()
                || member.columns != matrix.columns()
                || !self
                    .owner
                    .source
                    .matrix(&member.name, member.axes)?
                    .same_binding(matrix)
            {
                return Err(Error::State(
                    "ordinary CUDA projection authenticated member differs".into(),
                ));
            }
        }
        Ok(())
    }
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "ordinary CUDA projection upload cancelled".into(),
        ));
    }
    Ok(())
}

impl ResidentInitialization<'_> {
    /// Pack one to three source-declared full matrices in order into one resident
    /// allocation. Each bounded source window is copied then physically synced
    /// before the next window; the full source owner remains a completion owner.
    pub(crate) async fn upload_ordinary_projection(
        &mut self,
        service: &Arc<ResidentService>,
        source: VerifiedTensorSource,
        members: &[SourceTensor],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<PackedProjectionWeights>> {
        self.require_service(service)?;
        not_cancelled(cancelled)?;
        if members.is_empty()
            || members.len() > 3
            || source.source_identity() != service.source_identity()
        {
            return Err(Error::State(
                "ordinary CUDA projection source/member ownership differs".into(),
            ));
        }

        let mut matrices = Vec::with_capacity(members.len());
        let mut rows = 0usize;
        let mut columns = None;
        let mut total_bytes = 0usize;
        for member in members {
            if member.source != source.source_identity()
                || member.kind != QuantKind::Bf16
                || member.axes != MatrixAxes::OutputInput
                || member.rows == 0
                || member.columns == 0
            {
                return Err(Error::State(
                    "ordinary CUDA projection member contract differs".into(),
                ));
            }
            let matrix = source.matrix(&member.name, member.axes)?;
            if matrix.source_identity() != member.source
                || matrix.tensor().kind != QuantKind::Bf16
                || matrix.rows() != member.rows
                || matrix.columns() != member.columns
            {
                return Err(Error::State(
                    "ordinary CUDA projection source matrix differs".into(),
                ));
            }
            if columns
                .replace(member.columns)
                .is_some_and(|value| value != member.columns)
            {
                return Err(Error::State(
                    "ordinary CUDA projection member input widths differ".into(),
                ));
            }
            rows = rows
                .checked_add(member.rows)
                .ok_or_else(|| Error::Limit("ordinary CUDA projection rows overflow".into()))?;
            total_bytes = total_bytes
                .checked_add(
                    member
                        .rows
                        .checked_mul(member.columns)
                        .and_then(|elements| elements.checked_mul(2))
                        .ok_or_else(|| {
                            Error::Limit("ordinary CUDA projection bytes overflow".into())
                        })?,
                )
                .ok_or_else(|| {
                    Error::Limit("ordinary CUDA packed projection bytes overflow".into())
                })?;
            matrices.push(matrix);
        }
        let columns = columns
            .ok_or_else(|| Error::State("ordinary CUDA projection columns absent".into()))?;
        if total_bytes as u64 > service.client().properties().memory.max_page_size {
            return Err(Error::Unsupported(
                "ordinary CUDA packed projection exceeds device allocation page".into(),
            ));
        }

        let lease = self.allocate_empty(total_bytes, BufferRole::ImmutableWeights)?;
        let owner = Arc::new(PackedSourceOwner {
            source: source.clone(),
            matrices: matrices.clone().into(),
            members: members.to_vec().into(),
        });
        // Source ownership precedes the first H2D. Unknown initialization keeps
        // both the destination lease and source owner behind the service fence.
        self.retain_completion_owner(owner.clone())?;

        let mut destination_offset = 0usize;
        for matrix in &matrices {
            let row_bytes = matrix
                .columns()
                .checked_mul(2)
                .ok_or_else(|| Error::Limit("ordinary CUDA source row bytes overflow".into()))?;
            let rows_per_window = (WINDOW_BYTES / row_bytes).max(1);
            let mut first = 0usize;
            while first < matrix.rows() {
                not_cancelled(cancelled)?;
                let end = matrix.rows().min(first.saturating_add(rows_per_window));
                let window = if let Some(cancelled) = cancelled {
                    source.read_rows_cancellable(matrix, first..end, cancelled)?
                } else {
                    source.read_rows(matrix, first..end)?
                };
                if !window.matrix().same_binding(matrix) || window.row_range() != (first..end) {
                    return Err(Error::State(
                        "ordinary CUDA projection source window differs".into(),
                    ));
                }
                let payload = window.as_slice();
                let expected = (end - first).checked_mul(row_bytes).ok_or_else(|| {
                    Error::Limit("ordinary CUDA source window bytes overflow".into())
                })?;
                if payload.len() != expected {
                    return Err(Error::State(
                        "ordinary CUDA projection source window extent differs".into(),
                    ));
                }
                let start = destination_offset
                    .checked_add(first.checked_mul(row_bytes).ok_or_else(|| {
                        Error::Limit("ordinary CUDA projection destination offset overflow".into())
                    })?)
                    .ok_or_else(|| {
                        Error::Limit("ordinary CUDA projection destination offset overflow".into())
                    })?;
                let finish = start
                    .checked_add(payload.len())
                    .filter(|finish| *finish <= total_bytes)
                    .ok_or_else(|| {
                        Error::Limit("ordinary CUDA projection destination exceeds lease".into())
                    })?;
                let view = lease
                    .handle()
                    .offset_start(start as u64)
                    .offset_end((total_bytes - finish) as u64);
                service
                    .client()
                    .write(&view, Bytes::from_bytes_vec(payload.to_vec()));
                service.record_value_upload(payload.len());
                // Bound the copied payload lifetime before dropping its source
                // window. This is the same physical client, not a new queue.
                service
                    .client()
                    .sync_buffers([&view])
                    .await
                    .map_err(|error| {
                        Error::State(format!(
                            "ordinary CUDA projection upload completion: {error}"
                        ))
                    })?;
                first = end;
            }
            destination_offset = destination_offset
                .checked_add(matrix.rows().checked_mul(row_bytes).ok_or_else(|| {
                    Error::Limit("ordinary CUDA source matrix bytes overflow".into())
                })?)
                .ok_or_else(|| {
                    Error::Limit("ordinary CUDA projection destination overflow".into())
                })?;
        }
        if destination_offset != total_bytes {
            return Err(Error::State(
                "ordinary CUDA projection packed byte coverage incomplete".into(),
            ));
        }
        self.uploaded.insert(lease.id());

        Ok(Arc::new(PackedProjectionWeights {
            owner,
            lease,
            rows,
            columns,
        }))
    }
}
