//! Direct gamma and source-defined unit RMS; CPU cascade reduction order.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::*;
use crate::source_float::{round_source, widen_source};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn squares(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let x = input[i];
        let y = mul(x, x);
        output[i] = y;
        if !source_bf16(x) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn sum(input: &[f32], zero: &[f32], output: &mut [f32], #[comptime] width: usize) {
    let lane = usize::cast_from(UNIT_POS);
    let row = usize::cast_from(CUBE_POS_X);
    let z = zero[0];
    let mut subtotal = z;
    let mut carry = z;
    let mut values = Shared::<[f32]>::new_slice(32usize);
    for iteration in 0..width / 32usize {
        subtotal = add(subtotal, input[row * width + iteration * 32usize + lane]);
        if (iteration + 1usize).is_multiple_of(16usize) {
            carry = add(carry, subtotal);
            subtotal = z;
        }
    }
    values[lane] = add(add(add(subtotal, carry), z), z);
    sync_cube();
    if lane == 0usize {
        let mut total = z;
        for i in 0..8usize {
            total = add(
                total,
                add(
                    add(add(values[i], values[8usize + i]), values[16usize + i]),
                    values[24usize + i],
                ),
            );
        }
        output[row] = add(z, total);
    }
}
#[cube(launch_unchecked)]
pub(crate) fn scalar(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] stage: u32,
    #[comptime] width: usize,
    #[comptime] rows: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows {
        let x = input[i];
        let mut y = x;
        if comptime!(stage == 0) {
            y = round_source(widen_source(x) / f64::cast_from(width), 0u32);
        }
        if comptime!(stage == 1) {
            y = add(x, f32::reinterpret(0x358637bdu32));
        }
        if comptime!(stage == 2) {
            y = f32::cast_from(widen_source(x).sqrt());
        }
        if comptime!(stage == 3) {
            y = div(1.0f32, x);
        }
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn normalize(
    input: &[f32],
    inverse: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
    #[comptime] count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let y = mul(input[i], inverse[i / width]);
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn weight(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
    #[comptime] count: usize,
    #[comptime] mode: u32,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let mut y = input[i];
        if comptime!(mode != 0u32) {
            let raw = gamma[i % width];
            let mut g = raw;
            if comptime!(mode == 2u32) {
                g = add(1.0f32, raw);
            }
            if comptime!(mode == 3u32) {
                y = bf16(y);
            }
            y = mul(y, g);
            if !finite(raw) || (mode == 1u32 && !source_bf16(raw)) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn store(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let y = bf16(input[i]);
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
