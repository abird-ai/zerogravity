//! Original eager BF16 materializations, including all three softcap stores.
use super::{float::*, transcendentals};
use crate::program_kernels::NONFINITE_VALUE;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(super) fn operation(
    input: &[f32],
    other: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    scalar: f32,
    #[comptime] kind: u32,
    #[comptime] count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let x = input[i];
        let mut y = x;
        if !source_bf16(x) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
        if comptime!(kind == 0) {
            y = mul(x, scalar);
        }
        if comptime!(kind == 1) {
            y = mul(x, bf16(scalar));
        }
        if comptime!(kind == 2) {
            let v = other[i];
            y = add(x, v);
            if !source_bf16(v) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        if comptime!(kind == 3) {
            let v = other[i];
            y = mul(x, v);
            if !source_bf16(v) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        if comptime!(kind == 4) {
            y = transcendentals::gelu(x);
        }
        if comptime!(kind == 5) {
            let divided = bf16(div(x, scalar));
            let t = bf16(transcendentals::tanh(divided));
            y = mul(t, scalar);
        }
        y = bf16(y);
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
