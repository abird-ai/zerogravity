//! Split-half source products, including proportional zero-frequency pairs.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::*;
use cubecl::prelude::*;
#[cube(launch_unchecked)]
pub(crate) fn angles(
    frequencies: &[f32],
    output: &mut [f32],
    position: u32,
    #[comptime] half: usize,
) {
    let i = ABSOLUTE_POS;
    if i < half {
        output[i] = mul(frequencies[i], f32::cast_from(position));
    }
}
#[cube(launch_unchecked)]
pub(crate) fn apply(
    input: &[f32],
    trig: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
    #[comptime] count: usize,
    #[comptime] rotation: usize,
) {
    let i = ABSOLUTE_POS;
    if i < (count / width) * rotation / 2usize {
        let head = i / (rotation / 2usize);
        let pair = i % (rotation / 2usize);
        let first = head * width + pair;
        let second = first + rotation / 2usize;
        let a = input[first];
        let b = input[second];
        let sin = bf16(trig[pair * 2usize]);
        let cos = bf16(trig[pair * 2usize + 1usize]);
        let x = bf16(add(bf16(mul(a, cos)), bf16(mul(neg(b), sin))));
        let y = bf16(add(bf16(mul(b, cos)), bf16(mul(a, sin))));
        output[first] = x;
        output[second] = y;
        if !source_bf16(a) || !source_bf16(b) || !finite(x) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
