//! P07-H2 original Qwen RMS normalization GPU stage.
//!
//! Exact source math is: round each input square to f32, accumulate in f64
//! in the original order, round mean to f32, f32 sqrt + reciprocal, and
//! multiply activation by inv and source norm coefficient separately in f32.
//! Four ordered GPU kernels share one physical completion event; intermediate
//! squares, inverse estimate and correction never read back to CPU. No alternative norm equation,
//! tolerance relaxation, or fake Ready is permitted.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn rms_squares(input: &[f32], squares: &mut [f32], #[comptime] width: usize) {
    let position = ABSOLUTE_POS;
    if position < width {
        let value = input[position];
        // Materialize the original f32 square in physical f32 device RAM.
        // A separate kernel barrier prevents the compiler folding this
        // rounding operation into the subsequent FP64 sum.
        squares[position] = f32::cast_from(f64::cast_from(value) * f64::cast_from(value));
    }
}

#[cube(launch_unchecked)]
pub(crate) fn rms_inverse(
    squares: &[f32],
    inverse: &mut [f32],
    eps: f32,
    #[comptime] width: usize,
    #[comptime] groups: usize,
) {
    let head = ABSOLUTE_POS;
    if head < groups {
        let mut sum = 0.0f64;
        let mut i = 0usize;
        while i < width {
            // Each head starts its OWN f64 sum at zero, using the original
            // stored f32 squares, in source element order.
            sum += f64::cast_from(squares[head * width + i]);
            i += 1usize;
        }
        let mean = f32::cast_from(sum / f64::cast_from(width));
        let rounded_sqrt = f32::cast_from(f64::cast_from(mean + eps).sqrt());
        inverse[head * 2] = f32::cast_from(1.0f64 / f64::cast_from(rounded_sqrt));
        inverse[head * 2 + 1] = rounded_sqrt;
    }
}

/// Source-f32 reciprocal correction in its own ordered device kernel.
/// The prior kernel stored inverse[0] and inverse[1] to actual f32 GPU
/// memory. This prevents the compiler substituting the unrounded f64
/// reciprocal during the subsequent source-candidate comparisons.
///
/// The product of two normal finite f32 values is exactly representable
/// as f64; |1 - candidate * divisor| ranks the closest f32 reciprocal.
/// A halfway tie chooses the even f32 mantissa.
#[cube(launch_unchecked)]
pub(crate) fn rms_round_inverse(inverse: &mut [f32], #[comptime] groups: usize) {
    let head = ABSOLUTE_POS;
    if head < groups {
        let divisor = f64::cast_from(inverse[head * 2 + 1]);
        let mut candidate = inverse[head * 2];
        let bits = u32::reinterpret(candidate);
        if bits > 1u32 && bits < 0x7f7ffffeu32 {
            let mut best = (1.0f64 - divisor * f64::cast_from(candidate)).abs();
            let low_bits = bits - 1u32;
            let low = f32::reinterpret(low_bits);
            let low_error = (1.0f64 - divisor * f64::cast_from(low)).abs();
            if low_error < best || (low_error == best && (low_bits & 1u32) == 0u32) {
                candidate = low;
                best = low_error;
            }
            let high_bits = bits + 1u32;
            let high = f32::reinterpret(high_bits);
            let high_error = (1.0f64 - divisor * f64::cast_from(high)).abs();
            if high_error < best || (high_error == best && (high_bits & 1u32) == 0u32) {
                candidate = high;
            }
        }
        inverse[head * 2] = candidate;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn rms_apply(
    input: &[f32],
    weights: &[f32],
    inverse: &[f32],
    output: &mut [f32],
    #[comptime] width: usize,
    #[comptime] groups: usize,
) {
    let position = ABSOLUTE_POS;
    if position < width * groups {
        let head = position / width;
        // Gamma is reused for all heads, while the source scalar f32
        // inverse has an independent value and rounding history per head.
        output[position] = (input[position] * inverse[head * 2]) * weights[position % width];
    }
}
