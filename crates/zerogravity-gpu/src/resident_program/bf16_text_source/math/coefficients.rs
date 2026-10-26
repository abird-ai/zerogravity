//! Original distinct softplus/vector-scalar sigmoid/standalone-exp contracts.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{add, bf16, div, finite, mul, neg, source_bf16};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(super) fn biased_softplus(
    alpha: &[f32],
    bias: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        let x = add(alpha[h], bias[h]);
        let y = crate::source_sleef::softplus(x);
        output[h] = y;
        if !source_bf16(alpha[h]) || !finite(bias[h]) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn log_decay(
    a_exp: &[f32],
    softplus: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        let y = mul(neg(a_exp[h]), softplus[h]);
        output[h] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// Scalar-tail exp-shift is stored independently before the original F64 lane.
#[cube(launch_unchecked)]
pub(super) fn sigmoid_shift(
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
pub(super) fn sigmoid(
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

#[cube(launch_unchecked)]
pub(super) fn gated(
    normalized: &[f32],
    gate: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
    #[comptime] bf16_gamma: bool,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let mut n = normalized[i];
        if comptime!(bf16_gamma) {
            n = bf16(n);
        }
        let g = crate::source_sleef::silu(gate[i]);
        let y = bf16(mul(n, g));
        output[i] = y;
        if !source_bf16(gate[i]) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn ffn_silu_mul(
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let s = bf16(crate::source_sleef::silu(gate[i]));
        let y = bf16(mul(s, up[i]));
        output[i] = y;
        if !source_bf16(gate[i]) || !source_bf16(up[i]) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
