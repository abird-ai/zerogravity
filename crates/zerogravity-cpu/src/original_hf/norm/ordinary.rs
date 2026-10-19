//! Closed pinned CPU ordinary RMS; no model/profile eligibility.
//!
//! Modeling762feb6c…412c3b: widen BF16 input, FP32 normalization and
//! runtime(1+raw weight), then final BF16. Torch cf30153… CPU SumKernel
//! uses four Vec8 groups and16-iteration carries at these four widths.

use crate::accelerator::SourceIdentity;
use crate::quant::QuantKind;
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

const MAX_ELEMENTS: usize = 1_048_576;
const EPSILON: f32 = 1e-6;

/// Private source-bound stages, before any session publication.
#[derive(Debug)]
pub struct OrdinaryRms {
    source: SourceIdentity,
    width: usize,
    input_f32: Vec<f32>,
    squares_f32: Vec<f32>,
    sums_f32: Vec<f32>,
    mean_f32: Vec<f32>,
    denominator_f32: Vec<f32>,
    inverse_f32: Vec<f32>,
    normalized_f32: Vec<f32>,
    weight_f32: Vec<f32>,
    runtime_gamma_f32: Vec<f32>,
    weighted_f32: Vec<f32>,
    output_bf16: Vec<f32>,
}

impl OrdinaryRms {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn rows(&self) -> usize {
        self.mean_f32.len()
    }
    pub fn input_f32(&self) -> &[f32] {
        &self.input_f32
    }
    pub fn squares_f32(&self) -> &[f32] {
        &self.squares_f32
    }
    pub fn sums_f32(&self) -> &[f32] {
        &self.sums_f32
    }
    pub fn mean_f32(&self) -> &[f32] {
        &self.mean_f32
    }
    pub fn denominator_f32(&self) -> &[f32] {
        &self.denominator_f32
    }
    pub fn inverse_f32(&self) -> &[f32] {
        &self.inverse_f32
    }
    pub fn normalized_f32(&self) -> &[f32] {
        &self.normalized_f32
    }
    /// Raw learned words widened only; source storage remains unchanged.
    pub fn weight_f32(&self) -> &[f32] {
        &self.weight_f32
    }
    pub fn runtime_gamma_f32(&self) -> &[f32] {
        &self.runtime_gamma_f32
    }
    pub fn weighted_f32(&self) -> &[f32] {
        &self.weighted_f32
    }
    /// Exact BF16 words widened to F32, in logical row order.
    pub fn output_bf16(&self) -> &[f32] {
        &self.output_bf16
    }
}

fn validate(
    source: SourceIdentity,
    weight: SourceTensorRef<'_>,
    input: &[f32],
    width: usize,
) -> Result<Vec<f32>> {
    if !source.is_bound() {
        return Err(Error::Malformed(
            "pinned ordinary RMS source binding".into(),
        ));
    }
    if !matches!(width, 256 | 1024 | 2048 | 5120) {
        return Err(Error::Unsupported("pinned ordinary RMS width".into()));
    }
    if input.len() > MAX_ELEMENTS {
        return Err(Error::Limit("pinned ordinary RMS element bound".into()));
    }
    if input.is_empty() || !input.len().is_multiple_of(width) || weight.shape != [width as u64] {
        return Err(Error::Malformed("pinned ordinary RMS geometry".into()));
    }
    if !matches!(weight.kind, QuantKind::Bf16 | QuantKind::F32) {
        return Err(Error::Unsupported(
            "pinned ordinary RMS requires raw BF16/F32 weight".into(),
        ));
    }
    weight.validate()?;
    if input
        .iter()
        .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "pinned ordinary RMS requires finite exact BF16 input".into(),
        ));
    }
    let values = match weight.kind {
        QuantKind::Bf16 => weight
            .packed_bytes
            .chunks_exact(2)
            .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
            .collect::<Vec<_>>(),
        QuantKind::F32 => weight
            .packed_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("validated F32 weight")))
            .collect::<Vec<_>>(),
        _ => unreachable!("validated original ordinary RMS weight kind"),
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Malformed("nonfinite ordinary RMS raw weight".into()));
    }
    Ok(values)
}

// Source multi_row_sum size=width/32, four Vec8 groups, level_step16.
// Size<=160 never reaches the next256-iteration carry. Preserve remaining
// zero levels, ordered group/lane finish, and final StoreAccumulate(+0).
fn sum_row(row: &[f32]) -> f32 {
    let mut subtotal = [[0.0f32; 8]; 4];
    let mut carry = [[0.0f32; 8]; 4];
    for (iteration, values) in row.chunks_exact(32).enumerate() {
        for (group, lanes) in subtotal.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                *sum += values[group * 8 + lane];
            }
        }
        if (iteration + 1).is_multiple_of(16) {
            for (subtotal, carry) in subtotal.iter_mut().zip(&mut carry) {
                for (value, sum) in subtotal.iter_mut().zip(carry) {
                    *sum += *value;
                    *value = 0.0;
                }
            }
        }
    }
    for (subtotal, carry) in subtotal.iter_mut().zip(carry) {
        for (value, sum) in subtotal.iter_mut().zip(carry) {
            *value += sum;
            *value += 0.0;
            *value += 0.0;
        }
    }
    let mut lanes = subtotal[0];
    for group in &subtotal[1..] {
        for (sum, value) in lanes.iter_mut().zip(group) {
            *sum += value;
        }
    }
    let mut sum = 0.0f32;
    for value in lanes {
        sum += value;
    }
    0.0f32 + sum
}

fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite pinned ordinary RMS stage".into(),
        ));
    }
    Ok(())
}

/// Pinned original ordinary RMS, fixed epsilon1e-6. Input is logical row-major
/// exact BF16-widened values after original `.float()`; a strided raw tensor
/// needs explicit bit transport. Raw rank1 BF16/F32 learned bytes are borrowed,
/// widened privately and never replaced by adjusted learned weights. F32 weight
/// support is a closed public-operator path, not a trained model/profile claim.
pub fn ordinary_rms(
    source: SourceIdentity,
    weight: SourceTensorRef<'_>,
    input: &[f32],
    width: usize,
) -> Result<OrdinaryRms> {
    let weight_f32 = validate(source, weight, input, width)?;
    let squares_f32 = input.iter().map(|value| value * value).collect::<Vec<_>>();
    finite(&squares_f32)?;
    let sums_f32 = squares_f32
        .chunks_exact(width)
        .map(sum_row)
        .collect::<Vec<_>>();
    finite(&sums_f32)?;
    let mean_f32 = sums_f32
        .iter()
        .map(|sum| sum / width as f32)
        .collect::<Vec<_>>();
    let denominator_f32 = mean_f32
        .iter()
        .map(|mean| mean + EPSILON)
        .collect::<Vec<_>>();
    let inverse_f32 = denominator_f32
        .iter()
        .map(|value| 1.0f32 / value.sqrt())
        .collect::<Vec<_>>();
    finite(&mean_f32)?;
    finite(&denominator_f32)?;
    finite(&inverse_f32)?;
    let normalized_f32 = input
        .chunks_exact(width)
        .zip(&inverse_f32)
        .flat_map(|(row, inverse)| row.iter().map(move |value| value * inverse))
        .collect::<Vec<_>>();
    finite(&normalized_f32)?;
    let runtime_gamma_f32 = weight_f32
        .iter()
        .map(|weight| 1.0f32 + weight)
        .collect::<Vec<_>>();
    finite(&runtime_gamma_f32)?;
    let weighted_f32 = normalized_f32
        .chunks_exact(width)
        .flat_map(|row| {
            row.iter()
                .zip(&runtime_gamma_f32)
                .map(|(value, weight)| value * weight)
        })
        .collect::<Vec<_>>();
    finite(&weighted_f32)?;
    let output_bf16 = weighted_f32
        .iter()
        .map(|&value| super::round_bf16(value))
        .collect::<Result<Vec<_>>>()?;
    Ok(OrdinaryRms {
        source,
        width,
        input_f32: input.to_vec(),
        squares_f32,
        sums_f32,
        mean_f32,
        denominator_f32,
        inverse_f32,
        normalized_f32,
        weight_f32,
        runtime_gamma_f32,
        weighted_f32,
        output_bf16,
    })
}

#[cfg(test)]
mod tests;
