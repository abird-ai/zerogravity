//! Padded64 source chunk arithmetic, distinct from cached single-row recurrence.
//! Grouped inputs [row,head,128], internal vectors [head,64,128], key-major state.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{add, bf16, div, finite, fused, mul, source_bf16, sub};
use crate::source_standalone_exp::negative::{add64, lane as exp};
use cubecl::prelude::*;
pub(crate) mod execution;
pub(crate) mod prepared;

#[cfg(all(
    test,
    feature = "qwen-source-qualification",
    feature = "hip",
    feature = "hardware-abort-probe"
))]
mod physical;

#[cube(launch_unchecked)]
pub(crate) fn normalize(
    query: &[f32],
    key: &[f32],
    scaled_query: &mut [f32],
    normalized_key: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] heads: usize,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 64usize {
        let h = at / 64usize;
        let t = at % 64usize;
        let mut qs = Array::<f32>::new(32usize);
        let mut ks = Array::<f32>::new(32usize);
        for lane in 0..32usize {
            qs[lane] = 0.0f32;
            ks[lane] = 0.0f32;
        }
        if t < usize::cast_from(rows) {
            let base = (t * heads + h) * 128usize;
            for block in 0..4usize {
                for lane in 0..32usize {
                    let q = query[base + block * 32usize + lane];
                    let k = key[base + block * 32usize + lane];
                    qs[lane] = add(qs[lane], mul(q, q));
                    ks[lane] = add(ks[lane], mul(k, k));
                    if !source_bf16(q) || !source_bf16(k) {
                        status[0].fetch_or(NONFINITE_VALUE);
                    }
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
            if !finite(qden) || !finite(kden) || !finite(qi) || !finite(ki) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
            for d in 0..128usize {
                let q = mul(mul(query[base + d], qi), f32::reinterpret(0x3db504f3u32));
                let k = mul(key[base + d], ki);
                scaled_query[at * 128usize + d] = q;
                normalized_key[at * 128usize + d] = k;
                if !finite(q) || !finite(k) {
                    status[0].fetch_or(NONFINITE_VALUE);
                }
            }
        } else {
            for d in 0..128usize {
                scaled_query[at * 128usize + d] = 0.0f32;
                normalized_key[at * 128usize + d] = 0.0f32;
            }
        }
    }
}

#[allow(clippy::manual_range_contains)]
#[cube(launch_unchecked)]
pub(crate) fn beta_vectors(
    key: &[f32],
    value: &[f32],
    beta: &[f32],
    k_beta: &mut [f32],
    v_beta: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] heads: usize,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 64usize * 128usize {
        let h = at / 8192usize;
        let t = (at / 128usize) % 64usize;
        let d = at % 128usize;
        let mut kb = 0.0f32;
        let mut vb = 0.0f32;
        if t < usize::cast_from(rows) {
            let b = beta[t * heads + h];
            let v = value[(t * heads + h) * 128usize + d];
            kb = mul(key[at], b);
            vb = mul(v, b);
            if !source_bf16(b) || b < 0.0f32 || b > 1.0f32 || !source_bf16(v) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        k_beta[at] = kb;
        v_beta[at] = vb;
        if !finite(kb) || !finite(vb) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn cumulative(
    decay: &[f32],
    table: &[u64],
    output: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] heads: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        let mut sum = f64::reinterpret(table[1usize]);
        for t in 0..64usize {
            let mut x = 0.0f32;
            if t < usize::cast_from(rows) {
                x = decay[t * heads + h];
            }
            if !finite(x) || x > 0.0f32 {
                status[0].fetch_or(NONFINITE_VALUE);
                x = 0.0f32;
            }
            sum = add64(sum, crate::source_float::widen_source(x), table);
            let mut y = crate::source_float::round_source(sum, 0u32);
            if !finite(y) {
                status[0].fetch_or(NONFINITE_VALUE);
                y = 0.0f32;
            }
            output[h * 64usize + t] = y;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn decays(
    cumulative: &[f32],
    normal: &[u32],
    rare: &[u64],
    pairwise: &mut [f32],
    exponents: &mut [f32],
    final_exponents: &mut [f32],
    chunk_decay: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 4096usize {
        let h = at / 4096usize;
        let t = (at / 64usize) % 64usize;
        let s = at % 64usize;
        let mut x = f32::reinterpret(0xff800000u32);
        if s <= t {
            x = sub(cumulative[h * 64usize + t], cumulative[h * 64usize + s]);
        }
        pairwise[at] = exp(x, normal, rare, status);
    }
    if at < heads * 64usize {
        let h = at / 64usize;
        let x = cumulative[at];
        exponents[at] = exp(x, normal, rare, status);
        final_exponents[at] = exp(
            sub(cumulative[h * 64usize + 63usize], x),
            normal,
            rare,
            status,
        );
    }
    if at < heads {
        chunk_decay[at] = exp(cumulative[at * 64usize + 63usize], normal, rare, status);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn gram(
    query: &[f32],
    key: &[f32],
    beta_key: &[f32],
    decay: &[f32],
    system: &mut [f32],
    attention: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 4096usize {
        let h = at / 4096usize;
        let t = (at / 64usize) % 64usize;
        let s = at % 64usize;
        let left = (h * 64usize + t) * 128usize;
        let right = (h * 64usize + s) * 128usize;
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for d in 0..128usize {
            let k = key[right + d];
            a = fused(beta_key[left + d], k, a);
            b = fused(query[left + d], k, b);
        }
        // Original SGEMM adds its separately zero-initialized C plane.
        a = mul(add(0.0f32, a), decay[at]);
        b = mul(add(0.0f32, b), decay[at]);
        system[at] = a;
        attention[at] = b;
        if !finite(a) || !finite(b) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn decay_vectors(
    query: &[f32],
    key: &[f32],
    beta_key: &[f32],
    exponents: &[f32],
    final_exponents: &[f32],
    decayed_beta: &mut [f32],
    decayed_query: &mut [f32],
    decayed_key: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 8192usize {
        let d = exponents[at / 128usize];
        let b = mul(beta_key[at], d);
        let q = mul(query[at], d);
        let k = mul(key[at], final_exponents[at / 128usize]);
        decayed_beta[at] = b;
        decayed_query[at] = q;
        decayed_key[at] = k;
        if !finite(b) || !finite(q) || !finite(k) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube]
fn triangular_accumulate(a: f32, b: f32, c: f32, #[comptime] guarded: bool) -> f32 {
    if comptime!(guarded) {
        crate::source_arithmetic::mul_then_add_normal(a, b, c)
    } else {
        add(c, mul(a, b))
    }
}

#[cube(launch_unchecked)]
pub(crate) fn triangular(
    system: &[f32],
    values: &[f32],
    decayed_beta: &[f32],
    new_values: &mut [f32],
    cumulative_key: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    triangular_column(
        system,
        values,
        decayed_beta,
        new_values,
        cumulative_key,
        status,
        heads,
        false,
    );
}

/// Exact guarded FP32 inner products; the original sequential solve, final
/// reductions, subtraction and status publication remain unchanged.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn triangular_guarded(
    system: &[f32],
    values: &[f32],
    decayed_beta: &[f32],
    new_values: &mut [f32],
    cumulative_key: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    triangular_column(
        system,
        values,
        decayed_beta,
        new_values,
        cumulative_key,
        status,
        heads,
        true,
    );
}

#[cube]
fn triangular_column(
    system: &[f32],
    values: &[f32],
    decayed_beta: &[f32],
    new_values: &mut [f32],
    cumulative_key: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
    #[comptime] guarded: bool,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 128usize {
        let h = at / 128usize;
        let col = at % 128usize;
        // Each work item owns one complete column; no cross-thread dependency.
        for row in 0..64usize {
            let mut v = Array::<f32>::new(4usize);
            let mut k = Array::<f32>::new(4usize);
            for lane in 0..4usize {
                v[lane] = 0.0f32;
                k[lane] = 0.0f32;
            }
            let groups = row / 4usize;
            for group in 0..groups {
                for lane in 0..4usize {
                    let j = group * 4usize + lane;
                    let a = system[(h * 64usize + row) * 64usize + j];
                    let prior = (h * 64usize + j) * 128usize + col;
                    v[lane] = triangular_accumulate(a, new_values[prior], v[lane], guarded);
                    k[lane] = triangular_accumulate(a, cumulative_key[prior], k[lane], guarded);
                }
            }
            let mut sv = add(add(v[0usize], v[2usize]), add(v[1usize], v[3usize]));
            let mut sk = add(add(k[0usize], k[2usize]), add(k[1usize], k[3usize]));
            for j in groups * 4usize..row {
                let a = system[(h * 64usize + row) * 64usize + j];
                let prior = (h * 64usize + j) * 128usize + col;
                sv = triangular_accumulate(a, new_values[prior], sv, guarded);
                sk = triangular_accumulate(a, cumulative_key[prior], sk, guarded);
            }
            let out = (h * 64usize + row) * 128usize + col;
            let vv = sub(values[out], sv);
            let kk = sub(decayed_beta[out], sk);
            new_values[out] = vv;
            cumulative_key[out] = kk;
            if !finite(vv) || !finite(kk) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn prior_products(
    cumulative_key: &[f32],
    decayed_query: &[f32],
    prior: &[f32],
    new_values: &[f32],
    correction: &mut [f32],
    inter: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 8192usize {
        let h = at / 8192usize;
        let row = (at / 128usize) % 64usize;
        let col = at % 128usize;
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for j in 0..128usize {
            let state = prior[(h * 128usize + j) * 128usize + col];
            let pos = (h * 64usize + row) * 128usize + j;
            a = fused(cumulative_key[pos], state, a);
            b = fused(decayed_query[pos], state, b);
            if !finite(state) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        a = sub(new_values[at], add(0.0f32, a));
        b = add(0.0f32, b);
        correction[at] = a;
        inter[at] = b;
        if !finite(a) || !finite(b) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn core_product(
    attention: &[f32],
    correction: &[f32],
    inter: &[f32],
    core: &mut [f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] heads: usize,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 8192usize {
        let h = at / 8192usize;
        let row = (at / 128usize) % 64usize;
        let col = at % 128usize;
        let mut dot = 0.0f32;
        for j in 0..64usize {
            dot = fused(
                attention[(h * 64usize + row) * 64usize + j],
                correction[(h * 64usize + j) * 128usize + col],
                dot,
            );
        }
        let y = add(inter[at], add(0.0f32, dot));
        core[at] = y;
        if row < usize::cast_from(rows) {
            let out = bf16(y);
            output[(row * heads + h) * 128usize + col] = out;
            if !finite(out) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn state_product(
    decayed_key: &[f32],
    correction: &[f32],
    prior: &[f32],
    chunk_decay: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] heads: usize,
) {
    let at = ABSOLUTE_POS;
    if at < heads * 16384usize {
        let h = at / 16384usize;
        let row = (at / 128usize) % 128usize;
        let col = at % 128usize;
        let mut dot = 0.0f32;
        for j in 0..64usize {
            dot = fused(
                decayed_key[(h * 64usize + j) * 128usize + row],
                correction[(h * 64usize + j) * 128usize + col],
                dot,
            );
        }
        let y = add(mul(prior[at], chunk_decay[h]), add(0.0f32, dot));
        output[at] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cfg(all(test, feature = "hip-source"))]
mod triangular_ab;
