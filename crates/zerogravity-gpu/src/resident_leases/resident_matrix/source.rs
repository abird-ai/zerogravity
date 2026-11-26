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
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    pub(crate) fn rows(&self) -> usize {
        self.geometry.rows()
    }
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
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
        if self.geometry.region(region.first)? != Some(region) {
            return Err(Error::State("BF16 upload region differs".into()));
        }
        window.require_rows(&self.matrix, region.first..region.end)
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
