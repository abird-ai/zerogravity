//! Closed pinned Torch CPU F32 lastdim softmax, before any BF16 cast.
//!
//! Torch cf30153c4c131c8164ee7798e5022d810682e2cb SoftMaxKernel.cpp,
//! functional_base.h and vec256_float.h: Vec8 reduction, SLEEF exp,
//! FP32 reciprocal/multiply. Round-nearest and gradual underflow required.
//! Adapted arithmetic schedule; PyTorch BSD notice in softmax/PYTORCH-LICENSE.txt.
//! SLEEF implementation/Boost notice remain in the unchanged parent module.

use super::sleef_exp_finite_lane;
use crate::accelerator::SourceIdentity;
use crate::{Error, Result};

const LANES: usize = 8;
const MAX_WIDTH: usize = 512;
const MAX_ELEMENTS: usize = 1_048_576;

fn validate(source: SourceIdentity, input: &[f32], width: usize) -> Result<()> {
    if !source.is_bound() || input.is_empty() || width == 0 {
        return Err(Error::Malformed(
            "pinned CPU softmax source/row shape".into(),
        ));
    }
    if width > MAX_WIDTH || input.len() > MAX_ELEMENTS {
        return Err(Error::Limit(
            "pinned CPU softmax width/element bound".into(),
        ));
    }
    if !input.len().is_multiple_of(width) {
        return Err(Error::Malformed("pinned CPU softmax incomplete row".into()));
    }
    for row in input.chunks_exact(width) {
        if row
            .iter()
            .any(|value| !value.is_finite() && value.to_bits() != f32::NEG_INFINITY.to_bits())
            || !row.iter().any(|value| value.is_finite())
        {
            return Err(Error::Malformed(
                "pinned CPU softmax finite-row/mask contract".into(),
            ));
        }
    }
    Ok(())
}

// Validated finite/-Inf inputs only. x86 MAX chooses the RIGHT operand on
// equality, including opposite zero signs; Rust f32::max is not this contract.
fn maximum(left: f32, right: f32) -> f32 {
    if left > right { left } else { right }
}

fn reduce(row: &[f32], combine: impl Fn(f32, f32) -> f32) -> f32 {
    if row.len() < LANES {
        let mut value = row[0];
        for &next in &row[1..] {
            value = combine(value, next);
        }
        return value;
    }
    let mut lanes: [f32; LANES] = row[..LANES].try_into().expect("validated Vec8 row");
    let mut chunks = row[LANES..].chunks_exact(LANES);
    for chunk in &mut chunks {
        for (value, &next) in lanes.iter_mut().zip(chunk) {
            *value = combine(*value, next);
        }
    }
    // A partial Vec8 load combines only its valid low lanes. Other accumulator
    // lanes survive; no padded zero or scalar tail is reduced into them.
    for (value, &next) in lanes.iter_mut().zip(chunks.remainder()) {
        *value = combine(*value, next);
    }
    combine(
        combine(combine(lanes[0], lanes[4]), combine(lanes[2], lanes[6])),
        combine(combine(lanes[1], lanes[5]), combine(lanes[3], lanes[7])),
    )
}

/// Independent contiguous logical F32 rows, width1..512 and total<=1,048,576.
/// Negative-infinity masks require a finite value in every row. NaN, +Inf and
/// all-masked rows refuse before allocation/math. No source/cache mutation,
/// BF16 conversion, provider dispatch, model/profile or session eligibility.
pub fn softmax_f32_lastdim_avx2(
    source: SourceIdentity,
    input: &[f32],
    width: usize,
) -> Result<Vec<f32>> {
    validate(source, input, width)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| Error::Limit("pinned CPU softmax output allocation".into()))?;
    for row in input.chunks_exact(width) {
        let max = reduce(row, maximum);
        let begin = output.len();
        for &value in row {
            let shifted = value - max;
            // Source exp(-Inf) is +0, also when finite subtraction overflows.
            // Keep the parent exp helper's finite-input scope unchanged.
            let exponential = if shifted.to_bits() == f32::NEG_INFINITY.to_bits() {
                0.0
            } else {
                sleef_exp_finite_lane(shifted)
            };
            if !exponential.is_finite() || exponential.is_sign_negative() || exponential > 1.0 {
                return Err(Error::Malformed(
                    "invalid pinned CPU softmax exponential".into(),
                ));
            }
            output.push(exponential);
        }
        let exponentials = &mut output[begin..];
        let sum = reduce(exponentials, |left, right| left + right);
        if !sum.is_finite() || sum < 1.0 {
            return Err(Error::Malformed("invalid pinned CPU softmax sum".into()));
        }
        let inverse = 1.0f32 / sum;
        if !inverse.is_finite() || inverse <= 0.0 {
            return Err(Error::Malformed(
                "invalid pinned CPU softmax inverse".into(),
            ));
        }
        for value in exponentials {
            *value *= inverse;
            if !value.is_finite() || value.is_sign_negative() || *value > 1.0 {
                return Err(Error::Malformed(
                    "invalid pinned CPU softmax probability".into(),
                ));
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
