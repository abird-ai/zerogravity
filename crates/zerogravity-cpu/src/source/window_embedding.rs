//! Authentic original BF16 embedding row through the shared source decoder.
//! Caller admits private output before entry; no model/session/provider permit.
use crate::accelerator::{
    MatVecKernel,
    program::{MatrixAxes, SourceTensor},
};
use crate::quant::{QuantKind, dequantize};
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::source_weights::TensorWindow;

pub fn embedding_bf16_window(
    expected: &SourceTensor,
    token: u32,
    window: &TensorWindow,
    provider: Option<&dyn MatVecKernel>,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<f32>> {
    if provider.is_some() {
        return Err(Error::Unsupported(
            "provider lacks original embedding-window contract".into(),
        ));
    }
    let check_cancel = || {
        if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            Err(Error::State("original embedding window cancelled".into()))
        } else {
            Ok(())
        }
    };
    check_cancel()?;
    let matrix = window.matrix();
    let raw = matrix.tensor();
    let token =
        usize::try_from(token).map_err(|_| Error::Limit("embedding token domain".into()))?;
    let token_end = token
        .checked_add(1)
        .ok_or_else(|| Error::Limit("embedding token end".into()))?;
    let row_bytes = expected
        .columns
        .checked_mul(2)
        .ok_or_else(|| Error::Limit("embedding window row bytes".into()))?;
    let full_bytes = expected
        .rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::Limit("embedding window full bytes".into()))?;
    let offset = token
        .checked_mul(row_bytes)
        .and_then(|n| raw.byte_offset.checked_add(n as u64))
        .ok_or_else(|| Error::Limit("embedding window offset".into()))?;
    if !expected.source.is_bound()
        || expected.source != matrix.source_identity()
        || expected.name.is_empty()
        || expected.name != raw.name
        || expected.kind != QuantKind::Bf16
        || raw.kind != QuantKind::Bf16
        || expected.axes != MatrixAxes::OutputInput
        || matrix.axes() != MatrixAxes::OutputInput
        || expected.rows != matrix.rows()
        || expected.columns != matrix.columns()
        || expected.columns == 0
        || expected.columns > 5120
        || raw.shape.as_slice() != [expected.rows as u64, expected.columns as u64]
        || raw.byte_len != full_bytes as u64
        || token >= expected.rows
        || window.row_range() != (token..token_end)
        || window.source_offset() != offset
        || window.as_slice().len() != row_bytes
    {
        return Err(Error::State(
            "original embedding full descriptor/row window disagreement".into(),
        ));
    }
    let values = dequantize(QuantKind::Bf16, window.as_slice(), expected.columns)?;
    if values
        .iter()
        .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "nonfinite original BF16 embedding row".into(),
        ));
    }
    check_cancel()?;
    Ok(values)
}

#[cfg(test)]
mod tests;
