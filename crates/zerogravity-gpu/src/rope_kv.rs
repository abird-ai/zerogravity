//! Source IMRoPE angles/products/FMA, plus a new private current-token KV slab.
use crate::source_float::{round_source, widen_source};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn angle_step(
    angles: &mut [f32],
    position: f32,
    theta_scale: f32,
    #[comptime] stage: usize,
) {
    if ABSOLUTE_POS == 0usize {
        if stage == 0usize {
            angles[0] = position;
        } else {
            angles[stage] = round_source(
                widen_source(angles[stage - 1]) * widen_source(theta_scale),
                0u32,
            );
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn rope_products(
    input: &[f32],
    trig: &[f32],
    products: &mut [f32],
    #[comptime] width: usize,
    #[comptime] half: usize,
    #[comptime] heads: usize,
) {
    let p = ABSOLUTE_POS;
    if p < heads * half {
        let lane = p % half;
        let b = input[(p / half) * width + lane + half];
        let sin = trig[lane * 2];
        let cos = trig[lane * 2 + 1];
        products[p * 2] = round_source(
            widen_source(b) * widen_source(sin),
            (u32::reinterpret(b) ^ u32::reinterpret(sin)) & 0x80000000u32,
        );
        products[p * 2 + 1] = round_source(
            widen_source(b) * widen_source(cos),
            (u32::reinterpret(b) ^ u32::reinterpret(cos)) & 0x80000000u32,
        );
    }
}

#[cube(launch_unchecked)]
pub(crate) fn rope_sums(
    input: &[f32],
    trig: &[f32],
    products: &[f32],
    sums: &mut [f64],
    output: &mut [f32],
    #[comptime] width: usize,
    #[comptime] half: usize,
    #[comptime] heads: usize,
    #[comptime] count: usize,
) {
    let p = ABSOLUTE_POS;
    if p < count {
        let head = p / width;
        let lane = p % width;
        let mut value = input[p];
        if head < heads && lane < half * 2 {
            let pair = lane % half;
            let a = input[head * width + pair];
            let r = (head * half + pair) * 2;
            let mut b = trig[pair * 2 + 1];
            let mut c = f32::reinterpret(u32::reinterpret(products[r]) ^ 0x80000000u32);
            let mut index = r;
            if lane >= half {
                b = trig[pair * 2];
                c = products[r + 1];
                index = r + 1;
            }
            let sum = widen_source(a) * widen_source(b) + widen_source(c);
            sums[index] = sum;
            let mut sign =
                ((u32::reinterpret(a) ^ u32::reinterpret(b)) & u32::reinterpret(c)) & 0x80000000u32;
            if sum < 0.0f64 {
                sign = 0x80000000u32;
            }
            value = round_source(sum, sign);
        }
        output[p] = value;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn rope_subtract(
    input: &[f32],
    trig: &[f32],
    products: &[f32],
    sums: &[f64],
    subtracted: &mut [f64],
    #[comptime] width: usize,
    #[comptime] half: usize,
    #[comptime] heads: usize,
) {
    let p = ABSOLUTE_POS;
    if p < heads * half * 2 {
        let pair = p / 2;
        let lane = pair % half;
        let a = input[(pair / half) * width + lane];
        let mut b = trig[lane * 2 + 1];
        let mut c = -widen_source(products[p]);
        if p % 2 == 1usize {
            b = trig[lane * 2];
            c = widen_source(products[p]);
        }
        let product = widen_source(a) * widen_source(b);
        let mut large = product;
        if c.abs() > product.abs() {
            large = c;
        }
        subtracted[p] = sums[p] - large;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn rope_apply(
    input: &[f32],
    trig: &[f32],
    products: &[f32],
    sums: &[f64],
    subtracted: &[f64],
    output: &mut [f32],
    #[comptime] width: usize,
    #[comptime] half: usize,
    #[comptime] heads: usize,
) {
    let p = ABSOLUTE_POS;
    if p < heads * half * 2 {
        let pair = p / 2;
        let lane = pair % half;
        let head = pair / half;
        let a = input[head * width + lane];
        let mut b = trig[lane * 2 + 1];
        let mut c = -widen_source(products[p]);
        let mut index = head * width + lane;
        if p % 2 == 1usize {
            b = trig[lane * 2];
            c = widen_source(products[p]);
            index += half;
        }
        let product = widen_source(a) * widen_source(b);
        let mut small = c;
        if c.abs() > product.abs() {
            small = product;
        }
        let error = small - subtracted[p];
        output[index] = crate::source_float::correct_fma(sums[p], error, output[index]);
    }
}
