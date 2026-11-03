//! Explicit vector/scalar-tail BF16 sigmoid; caller owns tables and storage.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{add, bf16, div, finite, neg, source_bf16};
use cubecl::prelude::*;

/// Scalar-tail exp-shift is stored independently before the original F64 lane.
#[cube(launch_unchecked)]
pub(crate) fn sigmoid_shift(
    input: &[f32],
    tables: &[f64],
    shifted: &mut [f64],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        shifted[i] = crate::source_transcendentals::exp_shift(neg(input[i]), tables);
    }
}

#[allow(clippy::manual_range_contains)] // Scalar CubeCL IR has no std range contains.
#[cube(launch_unchecked)]
pub(crate) fn sigmoid(
    input: &[f32],
    tables: &[f64],
    shifted: &[f64],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let x = input[i];
        let exponent_input = neg(x);
        let mut e = crate::source_sleef::exp(exponent_input);
        if i >= elements / 32usize * 32usize {
            e = crate::source_transcendentals::source_exp(exponent_input, shifted[i], tables);
        }
        let y = bf16(div(1.0f32, add(1.0f32, e)));
        output[i] = y;
        if !source_bf16(x) || !finite(y) || y < 0.0f32 || y > 1.0f32 {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

// Whole-call boundary: heads16 has scalar lanes for one row, but vector lanes
// across pairs of rows. Repeating the one-row kernel changes source arithmetic.
#[cube(launch_unchecked)]
pub(crate) fn call_shift(input: &[f32], tables: &[f64], shifted: &mut [f64], elements: u32) {
    let i = ABSOLUTE_POS;
    if i < elements as usize {
        shifted[i] = crate::source_transcendentals::exp_shift(neg(input[i]), tables);
    }
}

#[allow(clippy::manual_range_contains)]
#[cube(launch_unchecked)]
pub(crate) fn call(
    input: &[f32],
    tables: &[f64],
    shifted: &[f64],
    output: &mut [f32],
    status: &[Atomic<u32>],
    elements: u32,
) {
    let i = ABSOLUTE_POS;
    if i < elements as usize {
        let x = input[i];
        let e_input = neg(x);
        let mut e = crate::source_sleef::exp(e_input);
        if i >= elements as usize / 32usize * 32usize {
            e = crate::source_transcendentals::source_exp(e_input, shifted[i], tables);
        }
        let y = bf16(div(1.0f32, add(1.0f32, e)));
        output[i] = y;
        if !source_bf16(x) || !finite(y) || y < 0.0f32 || y > 1.0f32 {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
