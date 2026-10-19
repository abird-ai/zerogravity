//! Private output rows from an authentic owned original matrix window.
//! Transport partitions output rows only; original full-width CPU math stays
//! unchanged. No partial provider region, BF16 cast or physical output admission.

use crate::accelerator::{MatVecKernel, program::MatrixAxes, program::SourceTensor};
use crate::quant::QuantKind;
use crate::weights::{MAX_MATRIX_ROWS, SourceProjectionProfile, dot_bf16_cpu_avx2};
use crate::{Error, Result};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::TensorWindow;

/// Full original identity and the exact transported row range. This is
/// immutable result metadata, not a source owner or resource admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowProjectionIdentity {
    descriptor: SourceTensor,
    original_shape: [u64; 2],
    original_byte_offset: u64,
    original_byte_len: u64,
    rows: Range<usize>,
    source_offset: u64,
    window_bytes: usize,
}
impl WindowProjectionIdentity {
    pub fn descriptor(&self) -> &SourceTensor {
        &self.descriptor
    }
    pub fn original_shape(&self) -> [u64; 2] {
        self.original_shape
    }
    pub fn original_byte_offset(&self) -> u64 {
        self.original_byte_offset
    }
    pub fn original_byte_len(&self) -> u64 {
        self.original_byte_len
    }
    pub fn row_range(&self) -> Range<usize> {
        self.rows.clone()
    }
    pub fn source_offset(&self) -> u64 {
        self.source_offset
    }
    pub fn window_bytes(&self) -> usize {
        self.window_bytes
    }
    pub fn profile(&self) -> SourceProjectionProfile {
        SourceProjectionProfile::TorchCpuBf16Avx2
    }
}

/// Private FP32 accumulations in increasing original output-row order.
/// Caller owns final BF16 rounding and any later complete-output assembly.
#[derive(Debug)]
pub struct WindowProjectionRows {
    identity: WindowProjectionIdentity,
    accumulations: Vec<f32>,
}
impl WindowProjectionRows {
    pub fn identity(&self) -> &WindowProjectionIdentity {
        &self.identity
    }
    pub fn accumulations(&self) -> &[f32] {
        &self.accumulations
    }
    pub fn into_parts(self) -> (WindowProjectionIdentity, Vec<f32>) {
        (self.identity, self.accumulations)
    }
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("original window projection cancelled".into()));
    }
    Ok(())
}

fn validate(
    expected: &SourceTensor,
    rows: &Range<usize>,
    window: &TensorWindow,
    input: &[f32],
    cancelled: Option<&AtomicBool>,
) -> Result<usize> {
    not_cancelled(cancelled)?;
    let matrix = window.matrix();
    let original = matrix.tensor();
    if expected.kind != QuantKind::Bf16 || original.kind != QuantKind::Bf16 {
        return Err(Error::Unsupported(
            "original CPU window projection requires BF16 source rows".into(),
        ));
    }
    let full_rows = u64::try_from(expected.rows)
        .map_err(|_| Error::Limit("original window full row count overflow".into()))?;
    let columns = u64::try_from(expected.columns)
        .map_err(|_| Error::Limit("original window full width overflow".into()))?;
    if !expected.source.is_bound()
        || expected.source != matrix.source_identity()
        || expected.name.is_empty()
        || expected.name != original.name
        || expected.axes != MatrixAxes::OutputInput
        || matrix.axes() != MatrixAxes::OutputInput
        || expected.rows == 0
        || full_rows > MAX_MATRIX_ROWS
        || expected.columns == 0
        || matrix.rows() != expected.rows
        || matrix.columns() != expected.columns
        || original.shape.as_slice() != [full_rows, columns]
        || input.len() != expected.columns
        || rows.start >= rows.end
        || rows.end > expected.rows
        || *rows != window.row_range()
    {
        return Err(Error::State(
            "original window differs from full source descriptor/range".into(),
        ));
    }
    let row_bytes = expected
        .columns
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("original window BF16 row bytes overflow".into()))?;
    let full_bytes = columns
        .checked_mul(2)
        .and_then(|n| n.checked_mul(full_rows))
        .ok_or_else(|| Error::Limit("original window full byte length overflow".into()))?;
    let source_offset = u64::try_from(rows.start)
        .ok()
        .and_then(|n| n.checked_mul(columns))
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| original.byte_offset.checked_add(n))
        .ok_or_else(|| Error::Limit("original window source byte offset overflow".into()))?;
    let window_bytes = (rows.end - rows.start)
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::Limit("original window row-range byte length overflow".into()))?;
    let tensor_end = original
        .byte_offset
        .checked_add(full_bytes)
        .ok_or_else(|| Error::Limit("original window tensor byte end overflow".into()))?;
    let window_end = u64::try_from(window_bytes)
        .ok()
        .and_then(|n| source_offset.checked_add(n))
        .ok_or_else(|| Error::Limit("original window byte end overflow".into()))?;
    if original.byte_len != full_bytes
        || window.source_offset() != source_offset
        || window.as_slice().len() != window_bytes
        || window_end > tensor_end
    {
        return Err(Error::State(
            "original window full tensor/offset/packed length differs".into(),
        ));
    }
    if input
        .iter()
        .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "original CPU window requires finite BF16-exact activations".into(),
        ));
    }
    // Check every transported learned word before any FMA/output allocation.
    // The sealed source authenticates bytes; it does not make NaN/Inf executable.
    for row in window.as_slice().chunks_exact(row_bytes) {
        not_cancelled(cancelled)?;
        if row
            .chunks_exact(2)
            .any(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::Malformed(
                "original CPU window has nonfinite BF16 source weights".into(),
            ));
        }
    }
    not_cancelled(cancelled)?;
    Ok(row_bytes)
}

/// Reuse the existing pinned Torch CPU BF16 B1 dot on each authentic full-width
/// row. Expected descriptor always retains ALL original matrix rows. Existing
/// providers cannot represent this contract and refuse before invocation.
///
/// Borrowed window retains its original owner/staging through synchronous math.
/// This allocates private output only: caller must arrange output/scratch
/// admission beforehand. No hidden host/provider fallback, final BF16 cast,
/// model/profile permit, SessionAuthority or StateRoot publication.
pub fn project_bf16_window(
    expected: &SourceTensor,
    expected_rows: Range<usize>,
    window: &TensorWindow,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
    cancelled: Option<&AtomicBool>,
) -> Result<WindowProjectionRows> {
    let row_bytes = validate(expected, &expected_rows, window, input, cancelled)?;
    if provider.is_some() {
        return Err(Error::Unsupported(
            "physical provider does not declare original matrix-window CPU profile".into(),
        ));
    }
    let rows = expected_rows.end - expected_rows.start;
    let mut accumulations = Vec::new();
    accumulations
        .try_reserve_exact(rows)
        .map_err(|_| Error::Limit("original window private output allocation".into()))?;
    for row in window.as_slice().chunks_exact(row_bytes) {
        not_cancelled(cancelled)?;
        let value = dot_bf16_cpu_avx2(row, input);
        if !value.is_finite() {
            return Err(Error::Malformed(
                "nonfinite original window projection output".into(),
            ));
        }
        accumulations.push(value);
    }
    not_cancelled(cancelled)?;
    let original = window.matrix().tensor();
    Ok(WindowProjectionRows {
        identity: WindowProjectionIdentity {
            descriptor: expected.clone(),
            original_shape: [expected.rows as u64, expected.columns as u64],
            original_byte_offset: original.byte_offset,
            original_byte_len: original.byte_len,
            rows: expected_rows,
            source_offset: window.source_offset(),
            window_bytes: window.as_slice().len(),
        },
        accumulations,
    })
}

#[cfg(test)]
mod tests;
