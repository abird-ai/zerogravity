//! Source-exact F32 projection; no quantization or model policy.
use crate::source_accumulation;
use cubecl::prelude::*;

/// One canonical Qwen F32 32-lane source reduction shared by B1 GEMV
/// and the native B>1 projection. No second numeric F32 definition.
#[cube]
pub(crate) fn source_f32_ggml_dot(
    weights: &[f32],
    x: &[f32],
    row: usize,
    input_offset: usize,
    #[comptime] cols: usize,
    #[comptime] wide_accumulate: bool,
) -> f32 {
    let mut lanes = Array::<f32>::new(32usize);
    for i in 0..32usize {
        lanes[i] = 0.0;
    }
    for base in 0..cols / 32 {
        for lane in 0..32usize {
            let i = base * 32 + lane;
            lanes[lane] = source_accumulation::source_mul_add(
                weights[row * cols + i],
                x[input_offset + i],
                lanes[lane],
                wide_accumulate,
            );
        }
    }
    let mut reduced = Array::<f32>::new(8usize);
    for i in 0..8usize {
        reduced[i] = (lanes[i] + lanes[i + 16]) + (lanes[8 + i] + lanes[24 + i]);
    }
    let p0 = reduced[0] + reduced[4];
    let p1 = reduced[1] + reduced[5];
    let p2 = reduced[2] + reduced[6];
    let p3 = reduced[3] + reduced[7];
    (p0 + p1) + (p2 + p3)
}

#[cube(launch_unchecked)]
pub(crate) fn f32_ggml_dot_gemv(
    weights: &[f32],
    x: &[f32],
    output: &mut [f32],
    #[comptime] cols: usize,
    #[comptime] rows: usize,
    #[comptime] wide_accumulate: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        output[row] = source_f32_ggml_dot(weights, x, row, 0usize, cols, wide_accumulate);
    }
}
