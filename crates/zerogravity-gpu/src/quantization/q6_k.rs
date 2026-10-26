//! Source-exact Q6_K weights and dynamic activation projection.
use super::common::{eight_lane_reduce, signed_byte};
use crate::source_accumulation;
use cubecl::prelude::*;

/// Source-identical Q6K integer lanes/scale accumulation reused by
/// B1 and all native B>1 source matrix GPU projections.
#[cube]
pub(crate) fn source_q6_k_dot(
    packed: &[u8],
    scales: &[f32],
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
    for block in 0..blocks {
        let bid = row * blocks + block;
        let base = bid * 210;
        let mut int_acc = Array::<i32>::new(8usize);
        for i in 0..8usize {
            int_acc[i] = 0;
        }
        for half in 0..2usize {
            for at32 in 0..32usize {
                let high = u32::cast_from(packed[base + 128 + half * 32 + at32]);
                for group in 0..4usize {
                    let address = at32 + (group % 2) * 32;
                    let nib = u32::cast_from(packed[base + half * 64 + address]);
                    let low = if group < 2usize {
                        nib & 15u32
                    } else {
                        nib >> 4u32
                    };
                    let w = i32::cast_from(
                        low | (((high >> (2u32 * u32::cast_from(group))) & 3u32) << 4u32),
                    );
                    let subscale = signed_byte(u32::cast_from(
                        packed[base + 192 + half * 8 + group * 2 + at32 / 16],
                    ));
                    let at = half * 128 + group * 32 + at32;
                    let target = at32 / 4;
                    int_acc[target] += w * subscale * q8[code_offset + block * 256 + at];
                }
            }
        }
        for lane in 0..8usize {
            let mut offset = 0i32;
            for sub in 0..2usize {
                let group = lane * 2 + sub;
                let subscale = signed_byte(u32::cast_from(packed[base + 192 + group]));
                let mut sum = 0i32;
                for qidx in 0..16usize {
                    sum += q8[code_offset + block * 256 + group * 16 + qidx];
                }
                offset += sum * subscale;
            }
            int_acc[lane] -= 32i32 * offset;
            lanes[lane] = source_accumulation::source_mul_add(
                f32::cast_from(int_acc[lane]),
                scales[scale_offset + bid],
                lanes[lane],
                wide_accumulate,
            );
        }
    }
    eight_lane_reduce(&lanes)
}

#[cube(launch_unchecked)]
pub(crate) fn q6_k_q8_k_gemv(
    packed: &[u8],
    scales: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] rows: usize,
    #[comptime] wide_accumulate: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        output[row] = source_q6_k_dot(
            packed,
            scales,
            q8,
            row,
            0usize,
            0usize,
            blocks,
            wide_accumulate,
        );
    }
}
