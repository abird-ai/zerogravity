//! Fixed-capacity resident graph transport only; source math stays in qualified kernels.
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn write_token(output: &mut [u32], token: u32) {
    if ABSOLUTE_POS == 0usize {
        output[0] = token;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn coefficient_input(
    alpha: &[u32],
    beta: &[u32],
    dt: &[u32],
    a: &[u32],
    output: &mut [u32],
    #[comptime] heads: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        output[h * 4] = alpha[h];
        output[h * 4 + 1] = beta[h];
        output[h * 4 + 2] = dt[h];
        output[h * 4 + 3] = a[h];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn coefficient_output(
    input: &[u32],
    packed: &mut [u32],
    softplus: &mut [u32],
    raw: &mut [u32],
    scale: f32,
    #[comptime] heads: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        packed[h * 3] = input[h * 4 + 2];
        packed[h * 3 + 1] = input[h * 4 + 3];
        packed[h * 3 + 2] = u32::reinterpret(scale);
        softplus[h] = input[h * 4];
        raw[h] = input[h * 4 + 1];
    }
}

/// Every private candidate cell is written; no prior state or future padding
/// is writable/undefined. Device/session identity is checked by the binder.
#[cube(launch_unchecked)]
pub(crate) fn append_kv(
    previous_k: &[u32],
    previous_v: &[u32],
    k: &[u32],
    v: &[u32],
    candidate_k: &mut [u32],
    candidate_v: &mut [u32],
    position: u32,
    #[comptime] slab: usize,
    #[comptime] capacity: usize,
) {
    let i = ABSOLUTE_POS;
    if i < slab * capacity {
        let token = i / slab;
        let mut key = 0u32;
        let mut value = 0u32;
        if token < position as usize {
            key = previous_k[i];
            value = previous_v[i];
        } else if token == position as usize {
            key = k[i % slab];
            value = v[i % slab];
        }
        candidate_k[i] = key;
        candidate_v[i] = value;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn masked_scores(
    input: &[u32],
    output: &mut [u32],
    valid: u32,
    #[comptime] heads: usize,
    #[comptime] capacity: usize,
) {
    let i = ABSOLUTE_POS;
    if i < heads * capacity {
        let mut bits = 0xff800000u32;
        if i % capacity < valid as usize {
            bits = input[heads * capacity + i];
        }
        output[i] = bits;
    }
}

/// Clamp inactive exponentials to canonical +0 before the original padded
/// reduction. Active exponents and all source arithmetic remain untouched.
#[cube(launch_unchecked)]
pub(crate) fn zero_future(
    values: &mut [u32],
    valid: u32,
    #[comptime] heads: usize,
    #[comptime] capacity: usize,
) {
    let i = ABSOLUTE_POS;
    if i < heads * capacity && i % capacity >= valid as usize {
        values[i] = 0u32;
    }
}

/// Preserve each diagnostic status array and carry any refusal into the
/// attempt certificate consumed by downstream compact output selection.
#[cube(launch_unchecked)]
pub(crate) fn merge_status(input: &[u32], status: &[Atomic<u32>], #[comptime] count: usize) {
    let i = ABSOLUTE_POS;
    if i < count && input[i] != 0u32 {
        status[0].fetch_or(8u32);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn certify_interval(
    input: &[u32],
    status: &[Atomic<u32>],
    low: f32,
    high: f32,
    #[comptime] count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let bits = input[i];
        let x = f32::reinterpret(bits);
        if (bits & 0x7f800000u32) == 0x7f800000u32 || x < low || x > high {
            status[0].fetch_or(16u32);
        }
    }
}

#[allow(clippy::collapsible_if)] // Keep the physical bounds guard before device memory access.
#[cube(launch_unchecked)]
pub(crate) fn certify_causal(
    input: &[u32],
    status: &[Atomic<u32>],
    valid: u32,
    #[comptime] heads: usize,
    #[comptime] capacity: usize,
) {
    let i = ABSOLUTE_POS;
    if i < heads * capacity && i % capacity < valid as usize {
        if (input[i] & 0x7f800000u32) == 0x7f800000u32 {
            status[0].fetch_or(2u32);
        }
    }
}
