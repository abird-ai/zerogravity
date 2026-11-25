//! Checked byte geometry only; source owners, physical leases and Ready stay separate.
use crate::{Error, Result, encoding::StoredTensorEncoding};
use std::num::NonZeroUsize;

/// Lossless row layout. No allocation, cast, source binding or readiness grant.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedTensorLayout {
    alignment: NonZeroUsize,
    row_bytes: NonZeroUsize,
    row_stride_bytes: NonZeroUsize,
    rows: usize,
}
impl PreparedTensorLayout {
    pub fn strided(
        alignment: usize,
        row_bytes: usize,
        row_stride_bytes: usize,
        rows: usize,
    ) -> Result<Self> {
        if !alignment.is_power_of_two()
            || row_bytes == 0
            || row_stride_bytes < row_bytes
            || !row_stride_bytes.is_multiple_of(alignment)
        {
            return Err(Error::Malformed(
                "tensor row layout/alignment invalid".into(),
            ));
        }
        let layout = Self {
            alignment: NonZeroUsize::new(alignment).expect("validated alignment"),
            row_bytes: NonZeroUsize::new(row_bytes).expect("validated row bytes"),
            row_stride_bytes: NonZeroUsize::new(row_stride_bytes).expect("validated stride"),
            rows,
        };
        layout.byte_len()?;
        Ok(layout)
    }
    pub fn dense(alignment: usize, row_bytes: usize, rows: usize) -> Result<Self> {
        Self::strided(alignment, row_bytes, row_bytes, rows)
    }
    /// Original encoded row bytes, never a dequantized or reinterpreted extent.
    pub fn packed_rows(
        encoding: StoredTensorEncoding,
        columns: usize,
        rows: usize,
    ) -> Result<Self> {
        let columns = u64::try_from(columns)
            .map_err(|_| Error::Limit("tensor columns exceed encoding domain".into()))?;
        let row_bytes = usize::try_from(encoding.byte_len(columns)?)
            .map_err(|_| Error::Limit("tensor row bytes exceed address space".into()))?;
        Self::dense(1, row_bytes, rows)
    }
    pub fn row_bytes(self) -> usize {
        self.row_bytes.get()
    }
    pub fn row_stride_bytes(self) -> usize {
        self.row_stride_bytes.get()
    }
    pub fn alignment(self) -> usize {
        self.alignment.get()
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    /// Restrict an already checked physical row capacity without changing
    /// stride, alignment or encoding.
    pub fn prefix_rows(self, rows: usize) -> Result<Self> {
        if rows > self.rows {
            return Err(Error::State(
                "tensor row prefix exceeds prepared capacity".into(),
            ));
        }
        Ok(Self { rows, ..self })
    }
    /// Includes inter-row padding, not padding after the final row. Empty spans
    /// remain describable; callers still enforce their nonempty operation law.
    pub fn byte_len(self) -> Result<usize> {
        if self.rows == 0 {
            return Ok(0);
        }
        (self.rows - 1)
            .checked_mul(self.row_stride_bytes.get())
            .and_then(|n| n.checked_add(self.row_bytes.get()))
            .ok_or_else(|| Error::Limit("tensor row extent overflow".into()))
    }
    pub fn validate_region(self, offset: u64, bytes: u64) -> Result<()> {
        let expected = u64::try_from(self.byte_len()?)
            .map_err(|_| Error::Limit("tensor byte extent exceeds physical domain".into()))?;
        if !offset.is_multiple_of(self.alignment.get() as u64)
            || bytes != expected
            || offset.checked_add(bytes).is_none()
        {
            return Err(Error::State(
                "tensor physical extent/alignment differs".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{QuantScheme, ScalarDType};

    #[test]
    fn packed_source_layout_preserves_exact_codec_rows() {
        for (kind, columns, row_bytes) in [
            (StoredTensorEncoding::Scalar(ScalarDType::F32), 7, 28),
            (StoredTensorEncoding::Scalar(ScalarDType::F16), 7, 14),
            (StoredTensorEncoding::Scalar(ScalarDType::Bf16), 7, 14),
            (StoredTensorEncoding::Quantized(QuantScheme::Q4_0), 32, 18),
            (StoredTensorEncoding::Quantized(QuantScheme::Q4_1), 32, 20),
            (StoredTensorEncoding::Quantized(QuantScheme::Q8_0), 32, 34),
            (StoredTensorEncoding::Quantized(QuantScheme::Q6K), 256, 210),
            (StoredTensorEncoding::Quantized(QuantScheme::Q4K), 256, 144),
        ] {
            let layout = PreparedTensorLayout::packed_rows(kind, columns, 3).unwrap();
            assert_eq!(layout.row_bytes(), row_bytes);
            assert_eq!(layout.row_stride_bytes(), row_bytes);
            assert_eq!(layout.byte_len().unwrap(), row_bytes * 3);
            assert_eq!(layout.rows(), 3);
            layout.validate_region(1, (row_bytes * 3) as u64).unwrap();
            assert!(
                layout
                    .validate_region(1, (row_bytes * 3 - 1) as u64)
                    .is_err()
            );
        }
        assert!(
            PreparedTensorLayout::packed_rows(
                StoredTensorEncoding::Quantized(QuantScheme::Q4_0),
                31,
                1
            )
            .is_err()
        );
    }
    #[test]
    fn padded_layout_never_accepts_overlapping_or_overflowing_views() {
        let layout = PreparedTensorLayout::strided(4, 8, 16, 3).unwrap();
        assert_eq!(layout.byte_len().unwrap(), 40);
        let prefix = layout.prefix_rows(2).unwrap();
        assert_eq!(
            (
                prefix.row_bytes(),
                prefix.row_stride_bytes(),
                prefix.byte_len().unwrap()
            ),
            (8, 16, 24)
        );
        assert!(layout.prefix_rows(4).is_err());
        layout.validate_region(4, 40).unwrap();
        for (offset, bytes) in [(2, 40), (4, 48), (u64::MAX - 3, 40)] {
            assert!(layout.validate_region(offset, bytes).is_err());
        }
        for (align, row, stride, rows) in [
            (0, 8, 8, 1),
            (3, 8, 8, 1),
            (4, 0, 4, 1),
            (4, 8, 4, 1),
            (4, 8, 9, 1),
            (1, 2, usize::MAX, 2),
        ] {
            assert!(PreparedTensorLayout::strided(align, row, stride, rows).is_err());
        }
        assert_eq!(
            PreparedTensorLayout::dense(4, 4, 0)
                .unwrap()
                .byte_len()
                .unwrap(),
            0
        );
    }
}
