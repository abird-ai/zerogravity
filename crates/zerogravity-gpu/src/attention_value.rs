//! Source padded value-cache matmul. Runtime block index reuses one compiled
//! shader per FMA stage across the prefix; masked lanes execute canonical +0.
use crate::source_float::{correct_fma, round_source, widen_source};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn accumulate(
    input: &[f32],
    previous: &[f32],
    lanes: &mut [f32],
    sums: &mut [f64],
    subtracted: &mut [f64],
    block: u32,
    #[comptime] width: usize,
    #[comptime] tokens: usize,
    #[comptime] qheads: usize,
    #[comptime] kvheads: usize,
    #[comptime] stage: usize,
) {
    let at = ABSOLUTE_POS;
    if at < qheads * width * 32 {
        let column = at / 32;
        let head = column / width;
        let channel = column % width;
        let token = block as usize * 32 + at % 32;
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        if token < tokens {
            a = input
                [qheads * tokens + (token * kvheads + head / (qheads / kvheads)) * width + channel];
            b = input[head * tokens + token];
        }
        let mut c = 0.0f32;
        if block > 0u32 {
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
