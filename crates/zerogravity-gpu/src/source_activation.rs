//! Original distinct softplus/vector-scalar sigmoid/standalone-exp contracts.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{add, bf16, finite, mul, neg, source_bf16};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn biased_softplus(
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
pub(crate) fn log_decay(
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

#[cfg(feature = "qwen-source-qualification")]
pub(crate) use crate::source_sigmoid::{sigmoid, sigmoid_shift};

#[cube(launch_unchecked)]
pub(crate) fn gated(
    normalized: &[f32],
    gate: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
    #[comptime] bf16_gamma: bool,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        output[i] = gated_value(normalized[i], gate[i], status, bf16_gamma);
    }
}

/// Same source epilogue, with the actual borrowed output extent rather than
/// a fixed token row. One dispatch covers independent token/head groups.
#[cfg(all(feature = "hip-source", feature = "hip"))]
#[cube(launch_unchecked)]
pub(crate) fn gated_rows(
    normalized: &[f32],
    gate: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] bf16_gamma: bool,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = gated_value(normalized[i], gate[i], status, bf16_gamma);
    }
}

#[cube]
fn gated_value(
    normalized: f32,
    gate: f32,
    status: &[Atomic<u32>],
    #[comptime] bf16_gamma: bool,
) -> f32 {
    let mut n = normalized;
    if comptime!(bf16_gamma) {
        n = bf16(n);
    }
    let g = crate::source_sleef::silu(gate);
    let y = bf16(mul(n, g));
    if !source_bf16(gate) || !finite(y) {
        status[0].fetch_or(NONFINITE_VALUE);
    }
    y
}

#[cube(launch_unchecked)]
pub(crate) fn ffn_silu_mul(
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
