//! Stored stages of the existing explicit CPU contract; no subgroup arithmetic.
use crate::source_float::{correct_quotient, round_source, widen_source};
use cubecl::prelude::*;

#[cube]
fn add(a: f32, b: f32) -> f32 {
    let sign = u32::reinterpret(a) & u32::reinterpret(b) & 0x80000000u32;
    round_source(widen_source(a) + widen_source(b), sign)
}
#[cube]
fn mul(a: f32, b: f32) -> f32 {
    let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
    round_source(widen_source(a) * widen_source(b), sign)
}

#[cube(launch_unchecked)]
pub(super) fn widen(input: &[u32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let bits = (input[i / 2usize] >> (u32::cast_from(i % 2usize) * 16u32)) & 0xffffu32;
        output[i] = f32::reinterpret(bits << 16u32);
    }
}

#[cube(launch_unchecked)]
pub(super) fn scalar_stage(
    input: &[f32],
    output: &mut [f32],
    #[comptime] stage: u32,
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let x = input[i];
        let mut y = x;
        if comptime!(stage == 0) {
            y = mul(x, x);
        }
        if comptime!(stage == 1) {
            y = round_source(widen_source(x) / f64::cast_from(width), 0u32);
        }
        if comptime!(stage == 2) {
            y = add(x, f32::reinterpret(0x358637bdu32));
        }
        if comptime!(stage == 3) {
            y = f32::cast_from(widen_source(x).sqrt());
        }
        if comptime!(stage == 4) {
            let candidate = f32::cast_from(1.0f64 / widen_source(x));
            y = correct_quotient(1.0f32, x, candidate);
        }
        if comptime!(stage == 5) {
            y = add(1.0f32, x);
        }
        output[i] = y;
    }
}

#[cube(launch_unchecked)]
pub(super) fn sum(input: &[f32], zero: &[f32], output: &mut [f32], #[comptime] width: usize) {
    let lane = usize::cast_from(UNIT_POS);
    let z = zero[0];
    let mut subtotal = z;
    let mut carry = z;
    let mut values = Shared::<[f32]>::new_slice(32usize);
    for iteration in 0..width / 32usize {
        subtotal = add(subtotal, input[iteration * 32usize + lane]);
        if (iteration + 1usize).is_multiple_of(16usize) {
            carry = add(carry, subtotal);
            subtotal = z;
        }
    }
    subtotal = add(add(add(subtotal, carry), z), z);
    values[lane] = subtotal;
    sync_cube();
    if lane == 0usize {
        let mut total = z;
        for i in 0..8usize {
            let a = add(values[i], values[8usize + i]);
            let b = add(a, values[16usize + i]);
            total = add(total, add(b, values[24usize + i]));
        }
        output[0] = add(z, total);
    }
}

#[cube(launch_unchecked)]
pub(super) fn apply(input: &[f32], scale: &[f32], output: &mut [f32], #[comptime] scalar: bool) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let mut at = i;
        if comptime!(scalar) {
            at = 0usize;
        }
        let s = scale[at];
        output[i] = mul(input[i], s);
    }
}

#[cube(launch_unchecked)]
pub(super) fn round_pack(input: &[f32], output: &mut [f32], packed: &mut [u32]) {
    let i = ABSOLUTE_POS;
    if i < packed.len() {
        let a = u32::reinterpret(input[2usize * i]);
        let b = u32::reinterpret(input[2usize * i + 1usize]);
        let ra = (a + (0x7fffu32 + ((a >> 16u32) & 1u32))) & 0xffff0000u32;
        let rb = (b + (0x7fffu32 + ((b >> 16u32) & 1u32))) & 0xffff0000u32;
        output[2usize * i] = f32::reinterpret(ra);
        output[2usize * i + 1usize] = f32::reinterpret(rb);
        packed[i] = (ra >> 16u32) | rb;
    }
}
