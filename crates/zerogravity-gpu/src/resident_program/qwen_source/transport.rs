//! Word transport and explicit BF16 residual stores; no source/publication proof.
use crate::program_kernels::NONFINITE_VALUE;
pub(super) use crate::resident_program::bf16_projection::widen;
use crate::source_arithmetic::{add as rounded_add, bf16, finite, source_bf16};
use cubecl::prelude::*;
#[cube(launch_unchecked)]
pub(super) fn state_copy(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] active: usize,
    #[comptime] capacity: usize,
) {
    let i = ABSOLUTE_POS;
    if i < capacity {
        let mut x = 0.0f32;
        if i < active {
            x = input[i];
        }
        output[i] = x;
        if !finite(x) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube(launch_unchecked)]
pub(super) fn add(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let x = left[i];
        let y = right[i];
        let z = bf16(rounded_add(x, y));
        output[i] = z;
        if !source_bf16(x) || !source_bf16(y) || !finite(z) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
