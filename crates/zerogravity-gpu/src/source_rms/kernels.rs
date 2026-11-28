//! Direct gamma and source-defined unit RMS; CPU cascade reduction order.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::*;
use crate::source_float::{round_source, widen_source};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn squares(input: &[f32], output: &mut [f32], status: &[Atomic<u32>]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let x = input[i];
        let y = mul(x, x);
        output[i] = y;
        if !source_bf16(x) || !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// Residual-RMS first pass: materialize the source-defined BF16 residual and
/// square that exact rounded value for the unchanged RMS reduction cascade.
#[cube(launch_unchecked)]
pub(crate) fn residual_squares(
    input: &[f32],
    prior: &[f32],
    residual: &mut [f32],
    squares: &mut [f32],
    status: &[Atomic<u32>],
) {
    let i = ABSOLUTE_POS;
    if i < squares.len() {
        let x = input[i];
        let y = prior[i];
        let z = bf16(add(x, y));
        let square = mul(z, z);
        residual[i] = z;
        squares[i] = square;
        if !source_bf16(x) || !source_bf16(y) || !source_bf16(z) || !finite(square) {
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
    for iteration in 0..width / 32usize {
        subtotal = add(subtotal, input[row * width + iteration * 32usize + lane]);
        if (iteration + 1usize).is_multiple_of(16usize) {
            carry = add(carry, subtotal);
            subtotal = z;
        }
    }
    let total = fold_lanes(subtotal, carry, z, false);
    if lane == 0usize {
        output[row] = total;
    }
}

/// Preserve the original CPU cascade's lane and group order. The guarded
/// arithmetic uses native FP32 only for normal operands/results.
#[cube]
fn sum_add(a: f32, b: f32, #[comptime] guarded: bool) -> f32 {
    if comptime!(guarded) {
        add_normal(a, b)
    } else {
        add(a, b)
    }
}

#[cube]
fn fold_lanes(subtotal: f32, carry: f32, zero: f32, #[comptime] guarded: bool) -> f32 {
    let lane = usize::cast_from(UNIT_POS);
    let mut values = Shared::<[f32]>::new_slice(32usize);
    values[lane] = sum_add(
        sum_add(sum_add(subtotal, carry, guarded), zero, guarded),
        zero,
        guarded,
    );
    sync_cube();
    let mut total = zero;
    if lane == 0usize {
        for i in 0..8usize {
            total = sum_add(
                total,
                sum_add(
                    sum_add(
                        sum_add(values[i], values[8usize + i], guarded),
                        values[16usize + i],
                        guarded,
                    ),
                    values[24usize + i],
                    guarded,
                ),
                guarded,
            );
        }
    }
    sum_add(zero, total, guarded)
}
#[cube]
fn scalar_value(x: f32, #[comptime] stage: u32, #[comptime] width: usize) -> f32 {
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
    y
}

#[cube(launch_unchecked)]
pub(crate) fn scalar(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] stage: u32,
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let y = scalar_value(input[i], stage, width);
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// Retained alternate RMS scalar chain: mean -> epsilon -> sqrt -> reciprocal.
/// Each stage retains the standalone kernel's f32 operation and status check.
/// Current gfx1150 Qwen keeps separate stages because the cascade loses wall time.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn scalar_cascade(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = inverse_value(input[i], status, width);
    }
}

#[cube]
fn inverse_value(total: f32, status: &[Atomic<u32>], #[comptime] width: usize) -> f32 {
    let mut y = scalar_value(total, 0u32, width);
    if !finite(y) {
        status[0].fetch_or(NONFINITE_VALUE);
    }
    y = scalar_value(y, 1u32, width);
    if !finite(y) {
        status[0].fetch_or(NONFINITE_VALUE);
    }
    y = scalar_value(y, 2u32, width);
    if !finite(y) {
        status[0].fetch_or(NONFINITE_VALUE);
    }
    y = scalar_value(y, 3u32, width);
    if !finite(y) {
        status[0].fetch_or(NONFINITE_VALUE);
    }
    y
}

/// One independent wave per norm row. Keep the exact cascade while
/// eliminating the vector squares and four scalar intermediate stores.
#[cfg(all(feature = "hip-source", feature = "hip"))]
#[cube(launch_unchecked)]
pub(crate) fn region_inverse(
    input: &[f32],
    zero: &[f32],
    inverse: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    let row = usize::cast_from(CUBE_POS_X);
    let z = zero[0];
    let mut subtotal = z;
    let mut carry = z;
    let mut invalid = false;
    for iteration in 0..width / 32usize {
        let x = input[row * width + iteration * 32usize + lane];
        let square = mul_bf16(x, x);
        invalid |= !source_bf16(x) || !finite(square);
        subtotal = add_normal(subtotal, square);
        if (iteration + 1usize).is_multiple_of(16usize) {
            carry = add_normal(carry, subtotal);
            subtotal = z;
        }
    }
    if invalid {
        status[0].fetch_or(NONFINITE_VALUE);
    }
    let total = fold_lanes(subtotal, carry, z, true);
    if lane == 0usize {
        inverse[row] = inverse_value(total, status, width);
    }
}

/// Same region with the source's separately rounded BF16 residual published
/// as the original logical output. No private residual/square transport.
#[cfg(all(feature = "hip-source", feature = "hip"))]
#[cube(launch_unchecked)]
pub(crate) fn region_residual_inverse(
    input: &[f32],
    prior: &[f32],
    residual: &mut [f32],
    zero: &[f32],
    inverse: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    let row = usize::cast_from(CUBE_POS_X);
    let z = zero[0];
    let mut subtotal = z;
    let mut carry = z;
    let mut invalid = false;
    for iteration in 0..width / 32usize {
        let i = row * width + iteration * 32usize + lane;
        let x = input[i];
        let y = prior[i];
        let r = bf16(add_normal(x, y));
        residual[i] = r;
        let square = mul_bf16(r, r);
        invalid |= !source_bf16(x) || !source_bf16(y) || !source_bf16(r) || !finite(square);
        subtotal = add_normal(subtotal, square);
        if (iteration + 1usize).is_multiple_of(16usize) {
            carry = add_normal(carry, subtotal);
            subtotal = z;
        }
    }
    if invalid {
        status[0].fetch_or(NONFINITE_VALUE);
    }
    let total = fold_lanes(subtotal, carry, z, true);
    if lane == 0usize {
        inverse[row] = inverse_value(total, status, width);
    }
}
#[cube]
fn normalize_value(input: f32, inverse: f32) -> f32 {
    mul(input, inverse)
}

#[cube]
fn weight_value(input: f32, raw: f32, #[comptime] mode: u32) -> f32 {
    let mut y = input;
    if comptime!(mode != 0u32) {
        let mut g = raw;
        if comptime!(mode == 2u32) {
            g = add(1.0f32, raw);
        }
        if comptime!(mode == 3u32) {
            y = bf16(y);
        }
        y = mul(y, g);
    }
    y
}

#[cube(launch_unchecked)]
pub(crate) fn normalize(
    input: &[f32],
    inverse: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let y = normalize_value(input[i], inverse[i / width]);
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
    #[comptime] mode: u32,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let mut raw = 0.0f32;
        if comptime!(mode != 0u32) {
            raw = gamma[i % width];
            if !finite(raw) || (mode == 1u32 && !source_bf16(raw)) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        let y = weight_value(input[i], raw, mode);
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn store(input: &[f32], output: &mut [f32], status: &[Atomic<u32>]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let y = bf16(input[i]);
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// Retained alternate RMS vector tail with the same normalize, gamma and store
/// sequence. Current gfx1150 Qwen keeps separate passes because this loses wall time.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn normalize_weight_store(
    input: &[f32],
    inverse: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] width: usize,
    #[comptime] mode: u32,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let normalized = normalize_value(input[i], inverse[i / width]);
        if !finite(normalized) {
            status[0].fetch_or(NONFINITE_VALUE);
        }

        let mut raw = 0.0f32;
        if comptime!(mode != 0u32) {
            raw = gamma[i % width];
            if !finite(raw) || (mode == 1u32 && !source_bf16(raw)) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        let weighted = weight_value(normalized, raw, mode);
        if !finite(weighted) {
            status[0].fetch_or(NONFINITE_VALUE);
        }

        let mut stored = weighted;
        if comptime!(mode != 3u32) {
            stored = bf16(weighted);
        }
        output[i] = stored;
        if !finite(stored) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
