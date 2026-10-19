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
pub(crate) fn angle_step_rows(
    angles: &mut [f32],
    positions: &[u32],
    theta_scale: f32,
    #[comptime] half: usize,
    #[comptime] batch: usize,
    #[comptime] stage: usize,
) {
    let row = ABSOLUTE_POS;
    if row < batch {
        let at = row * half + stage;
        if stage == 0usize {
            angles[at] = f32::cast_from(positions[row]);
        } else {
            angles[at] = round_source(
                widen_source(angles[at - 1]) * widen_source(theta_scale),
                0u32,
            );
        }
    }
}

#[cube]
fn trig_index(
    head: usize,
    lane: usize,
    #[comptime] half: usize,
    #[comptime] heads: usize,
    #[comptime] query_heads: usize,
    #[comptime] batch: usize,
) -> usize {
    let mut row = 0usize;
    if batch > 1usize {
        if head < batch * query_heads {
            row = head / query_heads;
        } else {
            row = (head - batch * query_heads) / (heads / batch - query_heads);
        }
    }
    (row * half + lane) * 2
}

#[cube(launch_unchecked)]
pub(crate) fn rope_products(
    input: &[f32],
    trig: &[f32],
    products: &mut [f32],
    #[comptime] width: usize,
    #[comptime] half: usize,
    #[comptime] heads: usize,
    #[comptime] query_heads: usize,
    #[comptime] batch: usize,
) {
    let p = ABSOLUTE_POS;
    if p < heads * half {
        let lane = p % half;
        let b = input[(p / half) * width + lane + half];
        let t = trig_index(p / half, lane, half, heads, query_heads, batch);
        let sin = trig[t];
        let cos = trig[t + 1];
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
    #[comptime] query_heads: usize,
    #[comptime] batch: usize,
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
            let t = trig_index(head, pair, half, heads, query_heads, batch);
            let mut b = trig[t + 1];
            let mut c = f32::reinterpret(u32::reinterpret(products[r]) ^ 0x80000000u32);
            let mut index = r;
            if lane >= half {
                b = trig[t];
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
    #[comptime] query_heads: usize,
    #[comptime] batch: usize,
) {
    let p = ABSOLUTE_POS;
    if p < heads * half * 2 {
        let pair = p / 2;
        let lane = pair % half;
        let a = input[(pair / half) * width + lane];
        let t = trig_index(pair / half, lane, half, heads, query_heads, batch);
        let mut b = trig[t + 1];
        let mut c = -widen_source(products[p]);
        if p % 2 == 1usize {
            b = trig[t];
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
    #[comptime] query_heads: usize,
    #[comptime] batch: usize,
) {
    let p = ABSOLUTE_POS;
    if p < heads * half * 2 {
        let pair = p / 2;
        let lane = pair % half;
        let head = pair / half;
        let a = input[head * width + lane];
        let t = trig_index(head, lane, half, heads, query_heads, batch);
        let mut b = trig[t + 1];
        let mut c = -widen_source(products[p]);
        let mut index = head * width + lane;
        if p % 2 == 1usize {
            b = trig[t];
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
