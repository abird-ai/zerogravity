//! Original partial rotary and compact canonical prior/current KV transport.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{bf16, finite, mul};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(super) fn append(
    prior_key: &[f32],
    prior_value: &[f32],
    key: &[f32],
    value: &[f32],
    keys: &mut [f32],
    values: &mut [f32],
    #[comptime] slab: usize,
    #[comptime] position: usize,
) {
    let i = ABSOLUTE_POS;
    if i < (position + 1usize) * slab {
        if i < position * slab {
            keys[i] = prior_key[i];
            values[i] = prior_value[i];
        } else {
            keys[i] = key[i - position * slab];
            values[i] = value[i - position * slab];
        }
    }
}
#[cube(launch_unchecked)]
pub(super) fn gate(
    context: &[f32],
    sigmoid: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let y = bf16(mul(context[i], sigmoid[i]));
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
