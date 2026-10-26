//! Source-exact Q4_1 weights and dynamic activation projection.
use super::common::eight_lane_reduce;
use crate::source_accumulation;
use cubecl::prelude::*;

/// Source-identical Q4_1 integer lanes/scale accumulation reused by
/// B1 and all native B>1 source matrix GPU projections.
#[cube]
pub(crate) fn source_q4_1_dot(
    packed: &[u8],
    scales: &[f32],
    offsets: &[f32],
    q8: &[i32],
    row: usize,
    scale_offset: usize,
    code_offset: usize,
    #[comptime] blocks: usize,
    #[comptime] wide_accumulate: bool,
) -> f32 {
    let mut lanes = Array::<f32>::new(8usize);
    for i in 0..8usize {
        lanes[i] = 0.0;
    }
    let mut summs = 0.0f32;
    for block in 0..blocks {
        let bid = row * blocks + block;
        summs += offsets[scale_offset + bid];
        for lane in 0..8usize {
            let mut dot = 0i32;
            for sub in 0..4usize {
                let at = lane * 4 + sub;
                let byte = packed[bid * 20 + 4 + (at % 16)];
                let nib = u32::cast_from(byte);
                let w = if at < 16usize {
                    i32::cast_from(nib & 15u32)
                } else {
                    i32::cast_from(nib >> 4u32)
                };
                dot += w * q8[code_offset + block * 32 + at];
            }
            lanes[lane] = source_accumulation::source_mul_add(
                f32::cast_from(dot),
                scales[scale_offset + bid],
                lanes[lane],
                wide_accumulate,
            );
        }
    }
    eight_lane_reduce(&lanes) + summs
}

#[cube(launch_unchecked)]
pub(crate) fn q4_1_q8_1_gemv(
    packed: &[u8],
    scales: &[f32],
    offsets: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] rows: usize,
    #[comptime] wide_accumulate: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        output[row] = source_q4_1_dot(
            packed,
            scales,
            offsets,
            q8,
            row,
            0usize,
            0usize,
            blocks,
            wide_accumulate,
        );
    }
}
