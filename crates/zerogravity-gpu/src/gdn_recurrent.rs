//! Source single-token GDN recurrence. Every independently rounded stage
//! crosses a real FP32 buffer boundary. Only readback returns a state proposal.
use crate::source_accumulation;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn decay(
    previous: &[f32],
    coefficients: &[f32],
    decayed: &mut [f32],
    #[comptime] width: usize,
    #[comptime] elements: usize,
) {
    let at = ABSOLUTE_POS;
    if at < elements {
        let head = at / (width * width);
        decayed[at] =
            f32::cast_from(f64::cast_from(previous[at]) * f64::cast_from(coefficients[head * 3]));
    }
}

/// Store all four-by-eight source FMA accumulators as actual FP32 before
/// any reduction. Each source FMA iteration is a separate physical launch;
/// a cast in a shader-local loop did not preserve source rounding on AMD.
#[cube(launch_unchecked)]
pub(crate) fn dot_lanes(
    state: &[f32],
    qkv: &[f32],
    partial: &mut [f32],
    #[comptime] width: usize,
    #[comptime] values: usize,
    #[comptime] vector_offset: usize,
    #[comptime] block: usize,
) {
    let at = ABSOLUTE_POS;
    if at < values * 32 {
        let column = at / 32;
        let lane = at % 32;
        let head_base = (column / width) * width + vector_offset;
        let mut sum = 0.0f32;
        if block > 0usize {
            sum = partial[at];
        }
        let i = block * 32 + lane;
        let sum = source_accumulation::source_mul_add(
            state[column * width + i],
            qkv[head_base + i],
            sum,
            true,
        );
        partial[at] = sum;
    }
}

/// Five physical FP32 tree levels: 32->16 (s0+s2,s1+s3), 16->8,
/// 8->4 (low+high), 4->2 adjacent pairs, 2->1. No compiler can
/// reassociate across the actual device buffers between these launches.
#[cube(launch_unchecked)]
pub(crate) fn dot_reduce(
    input: &[f32],
    output: &mut [f32],
    #[comptime] values: usize,
    #[comptime] level: usize,
) {
    let at = ABSOLUTE_POS;
    let lanes = 16usize >> level;
    if at < values * lanes {
        let column = at / lanes;
        let lane = at % lanes;
        let (a, b) = if level == 0usize {
            (column * 32 + lane, column * 32 + lane + 16)
        } else if level == 1usize || level == 2usize {
            (column * lanes * 2 + lane, column * lanes * 2 + lane + lanes)
        } else {
            (
                column * lanes * 2 + lane * 2,
                column * lanes * 2 + lane * 2 + 1,
            )
        };
        output[at] = f32::cast_from(f64::cast_from(input[a]) + f64::cast_from(input[b]));
    }
}

#[cube(launch_unchecked)]
pub(crate) fn residual(qkv: &[f32], delta: &mut [f32], #[comptime] values: usize) {
    let at = ABSOLUTE_POS;
    if at < values {
        delta[at] =
            f32::cast_from(f64::cast_from(qkv[values * 2 + at]) - f64::cast_from(delta[at]));
    }
}

#[cube(launch_unchecked)]
pub(crate) fn beta(
    coefficients: &[f32],
    delta: &mut [f32],
    #[comptime] width: usize,
    #[comptime] values: usize,
) {
    let at = ABSOLUTE_POS;
    if at < values {
        delta[at] = f32::cast_from(
            f64::cast_from(delta[at]) * f64::cast_from(coefficients[(at / width) * 3 + 1]),
        );
    }
}

#[cube(launch_unchecked)]
pub(crate) fn update(
    decayed: &[f32],
    qkv: &[f32],
    delta: &[f32],
    proposed: &mut [f32],
    #[comptime] width: usize,
    #[comptime] values: usize,
    #[comptime] elements: usize,
) {
    let at = ABSOLUTE_POS;
    if at < elements {
        let column = at / width;
        let k_at = values + (column / width) * width + at % width;
        proposed[at] =
            source_accumulation::source_mul_add(qkv[k_at], delta[column], decayed[at], true);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn scale(
    coefficients: &[f32],
    output: &mut [f32],
    #[comptime] width: usize,
    #[comptime] values: usize,
) {
    let at = ABSOLUTE_POS;
    if at < values {
        output[at] = f32::cast_from(
            f64::cast_from(output[at]) * f64::cast_from(coefficients[(at / width) * 3 + 2]),
        );
    }
}
