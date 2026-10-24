//! One cold source-shaped selection; invocation/root authority remains elsewhere.
use super::*;
use crate::resident_program::bf16_projection::{Bf16ProjectionStorage, PreparedBf16Projection};
use zerogravity_artifacts::source_weights::VerifiedMatrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmaTextStorage {
    /// Original16GiB qualification schedule and its independently retained evidence.
    Streamed,
    /// All repeated projections, including the exact tied embedding/head operand.
    ResidentProjections,
}
pub(super) struct Selection {
    pub matrices: Vec<VerifiedMatrix>,
    pub operands: Vec<Arc<PreparedBf16Projection>>,
    pub nodes: Vec<Option<usize>>,
    pub aligned_bytes: usize,
    pub largest_binding: usize,
    pub projection_rows: usize,
    pub upload_windows: usize,
    pub cached_row_bytes: usize,
    pub cached_row_capacity: usize,
}
impl Selection {
    pub(super) fn derive(recorded: &Gemma4Bf16TextPlan, storage: GemmaTextStorage) -> Result<Self> {
        let source = recorded.source().source();
        let mut matrices: Vec<VerifiedMatrix> = Vec::new();
        let mut nodes = vec![None; recorded.plan().nodes().len()];
        {
            for (index, node) in recorded.plan().nodes().iter().enumerate() {
                if let PlanOperation::Project { tensor } = &node.operation {
                    let matrix = source.matrix(&tensor.name, tensor.axes)?;
                    if matrix.rows() != tensor.rows
                        || matrix.columns() != tensor.columns
                        || matrix.source_identity() != recorded.plan().source()
                        || matrix.tensor().kind != zerogravity_core::quant::QuantKind::Bf16
                    {
                        return Err(Error::State(
                            "Gemma resident actual projection descriptor".into(),
                        ));
                    }
                    let operand = match matrices.iter().position(|m| m.same_binding(&matrix)) {
                        Some(i) => i,
                        None => {
                            matrices.push(matrix);
                            matrices.len() - 1
                        }
                    };
                    nodes[index] = Some(operand);
                }
            }
        }
        if storage == GemmaTextStorage::ResidentProjections {
            // Only an EXACT authenticated binding may reuse projection residency.
            // The family established the tie; equal shapes/names are insufficient.
            for (index, node) in recorded.plan().nodes().iter().enumerate() {
                if let PlanOperation::Embed { tensor, .. } = &node.operation {
                    let matrix = source.matrix(&tensor.name, tensor.axes)?;
                    nodes[index] = matrices.iter().position(|m| m.same_binding(&matrix));
                }
            }
        }
        let cached_row_bytes = if storage == GemmaTextStorage::ResidentProjections {
            recorded
                .plan()
                .nodes()
                .iter()
                .enumerate()
                .filter_map(|(i, node)| {
                    if nodes[i].is_none()
                        && let PlanOperation::Embed { tensor, .. } = &node.operation
                    {
                        Some(
                            tensor
                                .columns
                                .checked_mul(2)
                                .ok_or_else(|| Error::Limit("source selected row extent".into())),
                        )
                    } else {
                        None
                    }
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let cached_row_capacity = if cached_row_bytes == 0 {
            0
        } else {
            source_page::cached_row_capacity(cached_row_bytes)?
        };
        let operands = matrices
            .iter()
            .map(|m| {
                PreparedBf16Projection::new(
                    source.clone(),
                    m.clone(),
                    match storage {
                        GemmaTextStorage::Streamed => Bf16ProjectionStorage::Streamed,
                        GemmaTextStorage::ResidentProjections => Bf16ProjectionStorage::Resident,
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let resident = operands
            .iter()
            .any(|o| o.storage() == Bf16ProjectionStorage::Resident);
        let aligned_bytes = layout::sum(
            matrices
                .iter()
                .map(|m| layout::aligned(m.tensor().byte_len as usize))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let largest_binding = matrices
            .iter()
            .map(|m| m.tensor().byte_len as usize)
            .max()
            .unwrap_or(0);
        let projection_rows = match storage {
            GemmaTextStorage::Streamed => 2048,
            GemmaTextStorage::ResidentProjections => matrices
                .iter()
                .map(VerifiedMatrix::rows)
                .max()
                .ok_or_else(|| {
                    Error::State("Gemma resident selection has no projections".into())
                })?,
        };
        if projection_rows == 0
            || projection_rows > 262144
            || projection_rows > i32::MAX as usize
            || matrices.iter().any(|m| {
                m.columns() > 12288
                    || !m.columns().is_multiple_of(64)
                    || m.rows()
                        .checked_mul(m.columns())
                        .is_none_or(|n| n / 2 > i32::MAX as usize)
            })
        {
            return Err(Error::Limit(
                "Gemma exact64lane full-K resident grid/u32 binding bounds".into(),
            ));
        }
        let upload_windows = layout::sum(
            matrices
                .iter()
                .map(|m| m.rows().div_ceil((1024 * 1024) / (m.columns() * 2))),
        )?;
        Ok(Self {
            cached_row_bytes,
            cached_row_capacity,
            operands,
            matrices,
            nodes,
            aligned_bytes: if resident { aligned_bytes } else { 0 },
            largest_binding: if resident { largest_binding } else { 0 },
            projection_rows,
            upload_windows: if resident { upload_windows } else { 0 },
        })
    }
}
