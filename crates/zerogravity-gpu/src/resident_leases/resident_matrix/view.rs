//! Authenticated borrowed full-K ranges on the caller's original armed flight.
use super::*;
use std::ops::Range;

pub(crate) struct ResidentMatrixView {
    handle: cubecl::server::Handle,
    matrix: Arc<ResidentMatrix>,
    rows: Range<usize>,
}
impl ResidentMatrixView {
    pub(crate) fn handle(&self) -> &cubecl::server::Handle {
        &self.handle
    }
    pub(crate) fn matrix(&self) -> &VerifiedMatrix {
        &self.matrix.matrix
    }
    pub(crate) fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }
    pub(crate) fn bytes(&self) -> usize {
        self.rows.len() * self.matrix.layout.row_bytes()
    }
}
impl ResidentMatrix {
    pub(crate) fn with_rows(
        self: &Arc<Self>,
        flight: &mut ResidentSubmission<'_>,
        source: &VerifiedTensorSource,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        enqueue: &mut dyn FnMut(&ResidentMatrixView) -> Result<usize>,
    ) -> Result<usize> {
        self.require_matrix(source, self.lease.service(), matrix)?;
        if rows.is_empty() || rows.end > self.layout.rows() {
            return Err(Error::State(
                "resident matrix exact full-K row range".into(),
            ));
        }
        flight.require_accesses([(&self.lease, DeviceAccess::Read)])?;
        flight.retain_completion_owner(self.clone())?;
        let offset = rows
            .start
            .checked_mul(self.layout.row_bytes())
            .ok_or_else(|| Error::Limit("resident matrix row offset".into()))?;
        let bytes = rows
            .len()
            .checked_mul(self.layout.row_bytes())
            .ok_or_else(|| Error::Limit("resident matrix row extent".into()))?;
        let handle = self
            .lease
            .handle()
            .offset_start(offset as u64)
            .offset_end((self.layout.bytes() - offset - bytes) as u64);
        enqueue(&ResidentMatrixView {
            matrix: self.clone(),
            rows,
            handle,
        })
    }
}
