//! Authenticated HF BF16 windows. No family role, kernel or completion authority.
use super::{Geometry, ResidentMatrix, layout::Region};
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedMatrix, VerifiedTensorSource};
use zerogravity_core::{Error, Result};

pub(crate) struct Bf16MatrixSource {
    source: VerifiedTensorSource,
    matrix: VerifiedMatrix,
    geometry: Geometry,
}
impl Bf16MatrixSource {
    pub(crate) fn new(source: VerifiedTensorSource, matrix: VerifiedMatrix) -> Result<Self> {
        let geometry = ResidentMatrix::authenticate(&source, &matrix)?;
        // Retain the exact authentic member/range, including expert planes.
        source.require_matrix_binding(&matrix)?;
        Ok(Self {
            source,
            matrix,
            geometry,
        })
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(crate) fn matrix(&self) -> &VerifiedMatrix {
        &self.matrix
    }
    pub(crate) fn rows(&self) -> usize {
        self.geometry.rows()
    }
    pub(crate) fn columns(&self) -> usize {
        self.geometry.row_bytes() / 2
    }
    pub(crate) fn windows(&self) -> usize {
        self.geometry.windows()
    }
    #[cfg(feature = "cuda-native-qualification")]
    fn region(&self, ordinal: usize) -> Result<super::layout::Region> {
        let first = ordinal
            .checked_mul(super::layout::WINDOW / self.geometry.row_bytes())
            .ok_or_else(|| Error::Limit("BF16 source window ordinal overflow".into()))?;
        self.geometry
            .region(first)?
            .ok_or_else(|| Error::State("BF16 source window absent".into()))
    }
    #[cfg(feature = "cuda-native-qualification")]
    pub(crate) fn read_window(
        &self,
        ordinal: usize,
        cancel: Option<&AtomicBool>,
    ) -> Result<TensorWindow> {
        self.read_region(self.region(ordinal)?, cancel)
    }
    pub(in crate::resident_leases) fn geometry(&self) -> Geometry {
        self.geometry
    }
    pub(in crate::resident_leases) fn read_region(
        &self,
        region: Region,
        cancel: Option<&AtomicBool>,
    ) -> Result<TensorWindow> {
        if self.geometry.region(region.first)? != Some(region) {
            return Err(Error::State(
                "BF16 source region differs from authenticated geometry".into(),
            ));
        }
        match cancel {
            Some(flag) => {
                self.source
                    .read_rows_cancellable(&self.matrix, region.first..region.end, flag)
            }
            None => self
                .source
                .read_rows(&self.matrix, region.first..region.end),
        }
    }
    pub(in crate::resident_leases) fn require_window(
        &self,
        window: &TensorWindow,
        region: Region,
    ) -> Result<()> {
        let actual = window.matrix().tensor();
        let expected = self.matrix.tensor();
        let offset = self
            .matrix
            .byte_range()?
            .start
            .checked_add(region.offset as u64)
            .ok_or_else(|| Error::Limit("BF16 source window byte offset overflow".into()))?;
        if self.geometry.region(region.first)? != Some(region)
            || window.row_range() != (region.first..region.end)
            || window.as_slice().len() != region.bytes
            || window.source_offset() != offset
            || window.matrix().source_identity() != self.source.source_identity()
            || window.matrix().rows() != self.rows()
            || window.matrix().columns() != self.columns()
            || window.matrix().axes() != self.matrix.axes()
            || actual.name != expected.name
            || actual.kind != expected.kind
            || actual.shape != expected.shape
            || actual.byte_offset != expected.byte_offset
            || actual.byte_len != expected.byte_len
        {
            return Err(Error::State(
                "BF16 exact authenticated source window".into(),
            ));
        }
        Ok(())
    }
    /// Arithmetic consumers opt into finite BF16; byte transport alone does not.
    #[cfg(feature = "cuda-native-qualification")]
    pub(crate) fn require_finite_window(
        &self,
        ordinal: usize,
        window: &TensorWindow,
        staging_upper: usize,
    ) -> Result<()> {
        self.require_window(window, self.region(ordinal)?)?;
        if window.charged_staging_bytes() > staging_upper
            || window
                .as_slice()
                .chunks_exact(2)
                .any(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::State("BF16 finite window/staging bound".into()));
        }
        Ok(())
    }
}
