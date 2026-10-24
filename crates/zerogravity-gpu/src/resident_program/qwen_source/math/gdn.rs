//! Closed original GDN operation; no family traversal or state publication.
//! Canonical recurrence is FP32[head,key,value], full Conv4 stays independent.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{add, bf16, div, finite, fused, mul, source_bf16, sub};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(super) fn convolution(
    input: &[f32],
    history: &[f32],
    weights: &[f32],
    raw: &mut [f32],
    activated: &mut [f32],
    proposed: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] channels: usize,
    #[comptime] initial: bool,
) {
    let i = ABSOLUTE_POS;
    if i < channels {
        let h = i * 3usize;
        let w = i * 4usize;
        let a = history[h];
        let b = history[h + 1usize];
        let c = history[h + 2usize];
        let x = input[i];
        let p0 = mul(a, weights[w]);
        let p1 = mul(b, weights[w + 1usize]);
        let p2 = mul(c, weights[w + 2usize]);
        let p3 = mul(x, weights[w + 3usize]);
        let y = bf16(add(add(add(p0, p1), p2), p3));
        raw[i] = y;
        activated[i] = bf16(crate::source_sleef::silu(y));
        proposed[h] = b;
        proposed[h + 1usize] = c;
        proposed[h + 2usize] = x;
        if !source_bf16(a)
            || !source_bf16(b)
            || !source_bf16(c)
            || !source_bf16(x)
            || !finite(activated[i])
            || (initial
                && (u32::reinterpret(a) != 0u32
                    || u32::reinterpret(b) != 0u32
                    || u32::reinterpret(c) != 0u32))
        {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn full_convolution(
    prior: &[f32],
    minimal: &[f32],
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] channels: usize,
) {
    let i = ABSOLUTE_POS;
    if i < channels {
        let a = i * 4usize;
        let b = i * 3usize;
        for j in 0..3usize {
            if u32::reinterpret(prior[a + j + 1usize]) != u32::reinterpret(minimal[b + j]) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
            output[a + j] = minimal[b + j];
        }
        output[a + 3usize] = input[i];
    }
}

/// Original four Vec8 groups, then sequential group/lane finish.
#[cube(launch_unchecked)]
pub(super) fn normalize_qk(
    activated: &[f32],
    query: &mut [f32],
    key: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] kh: usize,
    #[comptime] vh: usize,
    #[comptime] cached: bool,
) {
    let h = ABSOLUTE_POS;
    if h < vh {
        let group = h / (vh / kh);
        let mut qs = Array::<f32>::new(32usize);
        let mut ks = Array::<f32>::new(32usize);
        for lane in 0..32usize {
            qs[lane] = 0.0f32;
            ks[lane] = 0.0f32;
        }
        for block in 0..4usize {
            for lane in 0..32usize {
                let q = activated[group * 128usize + block * 32usize + lane];
                let k = activated[(kh + group) * 128usize + block * 32usize + lane];
                qs[lane] = add(qs[lane], mul(q, q));
                ks[lane] = add(ks[lane], mul(k, k));
            }
        }
        let mut qsum = 0.0f32;
        let mut ksum = 0.0f32;
        for lane in 0..8usize {
            qsum = add(
                qsum,
                add(
                    add(add(qs[lane], qs[lane + 8usize]), qs[lane + 16usize]),
                    qs[lane + 24usize],
                ),
            );
            ksum = add(
                ksum,
                add(
                    add(add(ks[lane], ks[lane + 8usize]), ks[lane + 16usize]),
                    ks[lane + 24usize],
                ),
            );
        }
        let qden = add(qsum, f32::reinterpret(0x358637bdu32));
        let kden = add(ksum, f32::reinterpret(0x358637bdu32));
        let qi = div(
            1.0f32,
            f32::cast_from(crate::source_float::widen_source(qden).sqrt()),
        );
        let ki = div(
            1.0f32,
            f32::cast_from(crate::source_float::widen_source(kden).sqrt()),
        );
        for i in 0..128usize {
            let q = mul(activated[group * 128usize + i], qi);
            let mut scaled = mul(q, f32::reinterpret(0x3db504f3u32));
            if comptime!(cached) {
                scaled = div(q, f32::reinterpret(0x413504f3u32));
            }
            query[h * 128usize + i] = scaled;
            key[h * 128usize + i] = mul(activated[(kh + group) * 128usize + i], ki);
        }
        if !finite(qi) || !finite(ki) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn initial_dot(
    query: &[f32],
    key: &[f32],
    dot: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        let mut x = 0.0f32;
        for i in 0..128usize {
            x = fused(query[h * 128usize + i], key[h * 128usize + i], x);
        }
        dot[h] = x;
        if !finite(x) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[allow(clippy::manual_range_contains)] // Scalar CubeCL IR has no std range contains.
#[cube(launch_unchecked)]
pub(super) fn decayed(
    previous: &[f32],
    decay: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
    #[comptime] initial: bool,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let x = previous[i];
        let d = decay[i / 16384usize];
        let y = mul(x, d);
        output[i] = y;
        if !finite(x)
            || !finite(y)
            || d < 0.0f32
            || d > 1.0f32
            || (initial && u32::reinterpret(x) != 0u32)
        {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// Materialized product rounding +eight16-key subtotal/carry reductions.
#[cube(launch_unchecked)]
pub(super) fn key_reduce(
    state: &[f32],
    vector: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] values: usize,
    #[comptime] round_bf16: bool,
) {
    let at = ABSOLUTE_POS;
    if at < values {
        let h = at / 128usize;
        let j = at % 128usize;
        let mut carry = 0.0f32;
        for block in 0..8usize {
            let mut subtotal = 0.0f32;
            for key in block * 16usize..(block + 1usize) * 16usize {
                let product = mul(
                    state[(h * 128usize + key) * 128usize + j],
                    vector[h * 128usize + key],
                );
                subtotal = add(subtotal, product);
            }
            carry = add(carry, subtotal);
        }
        let mut y = carry;
        if comptime!(round_bf16) {
            y = bf16(y);
        }
        output[at] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn delta(
    activated: &[f32],
    memory: &[f32],
    beta: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] kh: usize,
    #[comptime] values: usize,
) {
    let at = ABSOLUTE_POS;
    if at < values {
        let y = mul(
            sub(activated[kh * 256usize + at], memory[at]),
            beta[at / 128usize],
        );
        output[at] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn state_update(
    previous: &[f32],
    key: &[f32],
    delta: &[f32],
    activated: &[f32],
    beta: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] kh: usize,
    #[comptime] elements: usize,
    #[comptime] initial: bool,
) {
    let at = ABSOLUTE_POS;
    if at < elements {
        let h = at / 16384usize;
        let k = (at / 128usize) % 128usize;
        let j = at % 128usize;
        let d = if comptime!(initial) {
            mul(activated[kh * 256usize + h * 128usize + j], beta[h])
        } else {
            delta[h * 128usize + j]
        };
        let update = mul(key[h * 128usize + k], d);
        let mut y = add(previous[at], update);
        if comptime!(initial) {
            y = update;
        }
        output[at] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn initial_core(
    activated: &[f32],
    beta: &[f32],
    dot: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] kh: usize,
    #[comptime] values: usize,
) {
    let at = ABSOLUTE_POS;
    if at < values {
        let h = at / 128usize;
        let vb = mul(activated[kh * 256usize + at], beta[h]);
        let y = bf16(add(0.0f32, mul(dot[h], vb)));
        output[at] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
