//! Source four-by-eight FMA QK tree over immutable token-major key slabs.
//! Each FMA iteration crosses physical sum/subtraction/FP32 buffers.
use crate::source_float::{correct_fma, round_source, widen_source};
use cubecl::prelude::*;

#[cube]
fn operands(
    input: &[f32],
    at: usize,
    #[comptime] width: usize,
    #[comptime] tokens: usize,
    #[comptime] qheads: usize,
    #[comptime] kvheads: usize,
    #[comptime] block: usize,
) -> (f32, f32) {
    let column = at / 32;
    let head = column / tokens;
    let token = column % tokens;
    let i = block * 32 + at % 32;
    let a = input[head * width + i];
    let b = input[qheads * width + (token * kvheads + head / (qheads / kvheads)) * width + i];
    (a, b)
}

#[cube(launch_unchecked)]
pub(crate) fn accumulate(
    input: &[f32],
    previous: &[f32],
    lanes: &mut [f32],
    sums: &mut [f64],
    subtracted: &mut [f64],
    #[comptime] width: usize,
    #[comptime] tokens: usize,
    #[comptime] qheads: usize,
    #[comptime] kvheads: usize,
    #[comptime] block: usize,
    #[comptime] stage: usize,
) {
    let at = ABSOLUTE_POS;
    if at < qheads * tokens * 32 {
        let (a, b) = operands(input, at, width, tokens, qheads, kvheads, block);
        let mut c = 0.0f32;
        if block > 0usize {
            c = previous[at];
        }
        let product = widen_source(a) * widen_source(b);
        let mut large = product;
        let mut small = widen_source(c);
        if small.abs() > product.abs() {
            large = small;
            small = product;
        }
        if stage == 0usize {
            let sum = product + widen_source(c);
            sums[at] = sum;
            let mut sign =
                ((u32::reinterpret(a) ^ u32::reinterpret(b)) & u32::reinterpret(c)) & 0x80000000u32;
            if sum < 0.0f64 {
                sign = 0x80000000u32;
            }
            lanes[at] = round_source(sum, sign);
        } else if stage == 1usize {
            subtracted[at] = sums[at] - large;
        } else {
            lanes[at] = correct_fma(sums[at], small - subtracted[at], lanes[at]);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce(
    input: &[f32],
    output: &mut [f32],
    #[comptime] columns: usize,
    #[comptime] level: usize,
) {
    let at = ABSOLUTE_POS;
    let lanes = 16usize >> level;
    if at < columns * lanes {
        let col = at / lanes;
        let lane = at % lanes;
        let (a, b) = if level == 0usize {
            (col * 32 + lane, col * 32 + lane + 16)
        } else if level == 1usize || level == 2usize {
            (col * lanes * 2 + lane, col * lanes * 2 + lane + lanes)
        } else {
            (col * lanes * 2 + lane * 2, col * lanes * 2 + lane * 2 + 1)
        };
        let sum = widen_source(input[a]) + widen_source(input[b]);
        let mut sign = (u32::reinterpret(input[a]) & u32::reinterpret(input[b])) & 0x80000000u32;
        if sum < 0.0f64 {
            sign = 0x80000000u32;
        }
        // The original complete-width SIMD path stores the tree directly;
        // Rust's empty scalar tail is -0 and preserves the same zero sign.
        output[at] = round_source(sum, sign);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn scale(output: &mut [f32], scale: f32, #[comptime] columns: usize) {
    let at = ABSOLUTE_POS;
    if at < columns {
        output[columns + at] = round_source(
            widen_source(output[at]) * widen_source(scale),
            (u32::reinterpret(output[at]) ^ u32::reinterpret(scale)) & 0x80000000u32,
        );
    }
}
