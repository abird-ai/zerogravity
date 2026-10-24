// CubeCL predicates use scalar comparisons.
#![allow(clippy::manual_range_contains)]
//! Explicit grouped QK, source Vec8 softmax and full-call BF16 PV.
use crate::program_kernels::NONFINITE_VALUE;
use crate::{
    source_arithmetic::{add, bf16, div, finite, fused, mul, source_bf16, sub},
    source_sleef as transcendentals,
};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn qk(
    query: &[f32],
    keys: &[f32],
    dots: &mut [f32],
    scores: &mut [f32],
    status: &[Atomic<u32>],
    zero: &[f32],
    #[comptime] width: usize,
    #[comptime] heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] capacity: usize,
    score_scale: f32,
    active: u32,
    first: u32,
) {
    let active = usize::cast_from(active);
    let first = usize::cast_from(first);
    let column = usize::cast_from(CUBE_POS_X);
    let head = column / capacity;
    let token = column % capacity;
    let lane = usize::cast_from(UNIT_POS);
    let mut sums = Shared::<[f32]>::new_slice(64usize);
    let z = zero[0];
    let mut sum = z;
    if token < active && token >= first {
        for block in 0..width / 64usize {
            let i = block * 64usize + lane;
            let a = query[head * width + i];
            let b = keys[(token * kv_heads + head / (heads / kv_heads)) * width + i];
            if !source_bf16(a) || !source_bf16(b) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
            sum = fused(a, b, sum);
        }
    }
    sums[lane] = sum;
    sync_cube();
    if lane < 32usize {
        sums[lane] = add(sums[lane], sums[lane + 32usize]);
    }
    sync_cube();
    if lane < 16usize {
        sums[lane] = add(sums[lane], sums[lane + 16usize]);
    }
    sync_cube();
    if lane < 8usize {
        sums[lane] = add(sums[lane], sums[lane + 8usize]);
    }
    sync_cube();
    if lane == 0usize {
        let mut dot = 0.0f32;
        let mut score = f32::reinterpret(0xff800000u32);
        if token < active && token >= first {
            let r0 = add(sums[0], sums[4]);
            let r1 = add(sums[1], sums[5]);
            let r2 = add(sums[2], sums[6]);
            let r3 = add(sums[3], sums[7]);
            dot = bf16(add(add(add(r0, r2), add(r1, r3)), z));
            score = bf16(mul(dot, score_scale));
            if !finite(dot) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        // Source's active masked dots are the same masked scores. Capacity
        // tail dots have the original generic finite-certificate +0 convention.
        if token < active && token < first {
            dot = score;
        }
        dots[column] = dot;
        scores[column] = score;
    }
}
#[cube]
fn maximum(a: f32, b: f32) -> f32 {
    let mut result = b;
    if a > b {
        result = a;
    }
    result
}
#[cube]
fn combine(a: f32, b: f32, #[comptime] is_max: bool) -> f32 {
    let mut result = add(a, b);
    if comptime!(is_max) {
        result = maximum(a, b);
    }
    result
}
#[cube(launch_unchecked)]
pub(crate) fn reduce(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    active: u32,
    #[comptime] is_max: bool,
    #[comptime] capacity: usize,
) {
    let active = usize::cast_from(active);
    let lane = usize::cast_from(UNIT_POS);
    let head = usize::cast_from(CUBE_POS_X);
    let mut lanes = Shared::<[f32]>::new_slice(8usize);
    if lane < 8usize && lane < active {
        let mut v = input[head * capacity + lane];
        if !finite(v) && (!is_max || u32::reinterpret(v) != 0xff800000u32) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
        for block in 1..active.div_ceil(8usize) {
            let i = block * 8usize + lane;
            if i < active {
                let x = input[head * capacity + i];
                if !finite(x) && (!is_max || u32::reinterpret(x) != 0xff800000u32) {
                    status[0].fetch_or(NONFINITE_VALUE);
                }
                v = combine(v, x, is_max);
            }
        }
        lanes[lane] = v;
    }
    sync_cube();
    if lane == 0usize {
        let mut v = lanes[0];
        if active < 8usize {
            for i in 1..active {
                v = combine(v, lanes[i], is_max);
            }
        } else {
            v = combine(
                combine(
                    combine(lanes[0], lanes[4], is_max),
                    combine(lanes[2], lanes[6], is_max),
                    is_max,
                ),
                combine(
                    combine(lanes[1], lanes[5], is_max),
                    combine(lanes[3], lanes[7], is_max),
                    is_max,
                ),
                is_max,
            );
        }
        output[head] = v;
        if !finite(v) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn exponentials(
    input: &[f32],
    maximum: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    active: u32,
    #[comptime] heads: usize,
    #[comptime] capacity: usize,
) {
    let active = usize::cast_from(active);
    let i = ABSOLUTE_POS;
    if i < heads * capacity {
        let token = i % capacity;
        let mut y = 0.0f32;
        if token < active {
            let shifted = sub(input[i], maximum[i / capacity]);
            if u32::reinterpret(shifted) != 0xff800000u32 {
                y = transcendentals::exp(shifted);
            }
            if !finite(y) || y < 0.0f32 || y > 1.0f32 {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        output[i] = y;
    }
}
#[cube(launch_unchecked)]
pub(crate) fn inverse(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let i = ABSOLUTE_POS;
    if i < heads {
        let x = input[i];
        let y = div(1.0f32, x);
        output[i] = y;
        if !finite(y) || x < 1.0f32 || y <= 0.0f32 {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn probabilities(
    input: &[f32],
    inverse: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    active: u32,
    #[comptime] heads: usize,
    #[comptime] capacity: usize,
) {
    let active = usize::cast_from(active);
    let i = ABSOLUTE_POS;
    if i < heads * capacity {
        let mut y = 0.0f32;
        if i % capacity < active {
            y = bf16(mul(input[i], inverse[i / capacity]));
        }
        output[i] = y;
        if !finite(y) || y < 0.0f32 || y > 1.0f32 {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
#[cube]
fn product(
    probabilities: &[f32],
    values: &[f32],
    status: &[Atomic<u32>],
    head: usize,
    lane: usize,
    token: usize,
    #[comptime] width: usize,
    #[comptime] heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] capacity: usize,
    active: usize,
) -> f32 {
    let mut y = 0.0f32;
    if token < active {
        let p = probabilities[head * capacity + token];
        let v = values[(token * kv_heads + head / (heads / kv_heads)) * width + lane];
        if !source_bf16(p) || !source_bf16(v) || p < 0.0f32 || p > 1.0f32 {
            status[0].fetch_or(NONFINITE_VALUE);
        }
        y = mul(v, p);
    }
    y
}
#[cube(launch_unchecked)]
pub(crate) fn pv(
    probabilities: &[f32],
    values: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    zero: &[f32],
    #[comptime] width: usize,
    #[comptime] heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] capacity: usize,
    active: u32,
    query_rows: u32,
    contraction: u32,
) {
    let active = usize::cast_from(active);
    let query_rows = usize::cast_from(query_rows);
    let contraction = usize::cast_from(contraction);
    let i = ABSOLUTE_POS;
    if i < heads * width {
        let head = i / width;
        let lane = i % width;
        let z = zero[0];
        let mut sum = z;
        if query_rows * contraction * width < 400usize {
            for t in 0..contraction {
                sum = add(
                    sum,
                    product(
                        probabilities,
                        values,
                        status,
                        head,
                        lane,
                        t,
                        width,
                        heads,
                        kv_heads,
                        capacity,
                        active,
                    ),
                );
            }
        } else {
            let mut p0 = z;
            let mut p1 = z;
            let mut p2 = z;
            let mut p3 = z;
            let end = contraction / 4usize * 4usize;
            for block in 0..contraction / 4usize {
                let t = block * 4usize;
                p0 = add(
                    p0,
                    product(
                        probabilities,
                        values,
                        status,
                        head,
                        lane,
                        t,
                        width,
                        heads,
                        kv_heads,
                        capacity,
                        active,
                    ),
                );
                p1 = add(
                    p1,
                    product(
                        probabilities,
                        values,
                        status,
                        head,
                        lane,
                        t + 1usize,
                        width,
                        heads,
                        kv_heads,
                        capacity,
                        active,
                    ),
                );
                p2 = add(
                    p2,
                    product(
                        probabilities,
                        values,
                        status,
                        head,
                        lane,
                        t + 2usize,
                        width,
                        heads,
                        kv_heads,
                        capacity,
                        active,
                    ),
                );
                p3 = add(
                    p3,
                    product(
                        probabilities,
                        values,
                        status,
                        head,
                        lane,
                        t + 3usize,
                        width,
                        heads,
                        kv_heads,
                        capacity,
                        active,
                    ),
                );
            }
            for t in end..contraction {
                p0 = add(
                    p0,
                    product(
                        probabilities,
                        values,
                        status,
                        head,
                        lane,
                        t,
                        width,
                        heads,
                        kv_heads,
                        capacity,
                        active,
                    ),
                );
            }
            sum = add(add(add(p0, p1), p2), p3);
        }
        let y = bf16(sum);
        output[i] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
