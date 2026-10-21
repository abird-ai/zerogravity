//! Original per-head GDN coefficients with physical FP32 and exp-shift stores.
use crate::{
    source_float::{correct_quotient, round_source, widen_source},
    source_transcendentals::{exp_shift, source_exp, source_log},
};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn coefficient_step(
    input: &[f32],
    tables: &[f64],
    scratch: &mut [f32],
    shifted: &mut [f64],
    output: &mut [f32],
    #[comptime] heads: usize,
    #[comptime] stage: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        let i = h * 4;
        let s = h * 6;
        let f = h * 3;
        if stage == 0usize {
            let sum = widen_source(input[i]) + widen_source(input[i + 2]);
            let mut sign =
                (u32::reinterpret(input[i]) & u32::reinterpret(input[i + 2])) & 0x80000000u32;
            if sum < 0.0f64 {
                sign = 0x80000000u32;
            }
            scratch[s] = round_source(sum, sign);
        } else if stage == 1usize {
            shifted[f] = exp_shift(scratch[s], tables);
            shifted[f + 1] = exp_shift(
                f32::reinterpret(u32::reinterpret(input[i + 1]) ^ 0x80000000u32),
                tables,
            );
        } else if stage == 2usize {
            let mut e = 0.0f32;
            if scratch[s] <= 20.0f32 {
                e = source_exp(scratch[s], shifted[f], tables);
            }
            scratch[s + 1] = e;
            scratch[s + 2] = source_exp(
                f32::reinterpret(u32::reinterpret(input[i + 1]) ^ 0x80000000u32),
                shifted[f + 1],
                tables,
            );
        } else if stage == 3usize {
            scratch[s + 3] = f32::cast_from(1.0f64 + widen_source(scratch[s + 1]));
            scratch[s + 4] = f32::cast_from(1.0f64 + widen_source(scratch[s + 2]));
        } else if stage == 4usize {
            let x = scratch[s];
            let mut sp = x;
            if x < -20.0f32 {
                sp = scratch[s + 1];
            } else if x <= 20.0f32 {
                sp = source_log(scratch[s + 3], tables);
            }
            output[i] = sp;
            scratch[s + 5] = round_source(1.0f64 / widen_source(scratch[s + 4]), 0u32);
        } else if stage == 5usize {
            let sign =
                (u32::reinterpret(output[i]) ^ u32::reinterpret(input[i + 3])) & 0x80000000u32;
            output[i + 1] =
                round_source(widen_source(output[i]) * widen_source(input[i + 3]), sign);
            output[i + 3] = correct_quotient(1.0f32, scratch[s + 4], scratch[s + 5]);
        } else if stage == 6usize {
            shifted[f + 2] = exp_shift(output[i + 1], tables);
        } else {
            output[i + 2] = source_exp(output[i + 1], shifted[f + 2], tables);
        }
    }
}
