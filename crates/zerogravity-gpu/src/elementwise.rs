//! Source FP32 residual addition and scalar-libm attention gating.
//!
//! Attention sigmoid uses the H8 scalar `expf` policy. The FFN/GDN SiLU
//! polynomial has different source bits and continues to use `silu.rs`.
//! These kernels require physical FP32/FP64 scratch stores between stages;
//! their local casts are not a replacement for those launch boundaries.
use crate::{
    source_float::{correct_quotient, round_source, widen_source},
    source_transcendentals::{exp_shift, source_exp},
};
use cubecl::prelude::*;

/// FP32 scratch elements per attention gate: exp, denominator, sigmoid.
pub(crate) const SIGMOID_SCRATCH: usize = 3;
/// Execute every stage in order using the same scratch/output handles.
pub(crate) const SIGMOID_STAGES: usize = 6;

/// One source FP32 addition, including both negative-zero inputs and gradual
/// underflow. Source inputs and the completed output are physically stored.
#[cube(launch_unchecked)]
pub(crate) fn residual_add(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    #[comptime] count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let sum = widen_source(left[i]) + widen_source(right[i]);
        // Round-to-nearest cancellation produces +0 except for -0 + -0.
        let mut sign = (u32::reinterpret(left[i]) & u32::reinterpret(right[i])) & 0x80000000u32;
        if sum < 0.0f64 {
            sign = 0x80000000u32;
        }
        output[i] = round_source(sum, sign);
    }
}

/// Compute `value * (1 / (1 + expf(-gate)))` with the original scalar-libm
/// source schedule. `tables` is the existing `source_transcendentals::TABLES`.
///
/// The caller admits both inputs, `3 * count` FP32 scratch elements,
/// `count` FP64 rounding shifts, the immutable table and `count` FP32 outputs.
/// It launches stages 0..SIGMOID_STAGES and retains every original handle
/// until actual completion/readback, or quarantines all of them if unknown.
/// Finite inputs are required; overflow of expf is an allowed intermediate
/// and produces a positive-zero sigmoid. No device exp intrinsic is used.
#[cube(launch_unchecked)]
pub(crate) fn sigmoid_gate_step(
    gate: &[f32],
    value: &[f32],
    tables: &[f64],
    scratch: &mut [f32],
    shifted: &mut [f64],
    output: &mut [f32],
    #[comptime] count: usize,
    #[comptime] stage: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let s = i * 3;
        let negative_gate = f32::reinterpret(u32::reinterpret(gate[i]) ^ 0x80000000u32);
        if stage == 0usize {
            shifted[i] = exp_shift(negative_gate, tables);
        } else if stage == 1usize {
            scratch[s] = source_exp(negative_gate, shifted[i], tables);
        } else if stage == 2usize {
            scratch[s + 1] = f32::cast_from(1.0f64 + widen_source(scratch[s]));
        } else if stage == 3usize {
            scratch[s + 2] = round_source(1.0f64 / widen_source(scratch[s + 1]), 0u32);
        } else if stage == 4usize {
            scratch[s + 2] = correct_quotient(1.0f32, scratch[s + 1], scratch[s + 2]);
        } else {
            let sign =
                (u32::reinterpret(value[i]) ^ u32::reinterpret(scratch[s + 2])) & 0x80000000u32;
            output[i] = round_source(widen_source(value[i]) * widen_source(scratch[s + 2]), sign);
        }
    }
}
