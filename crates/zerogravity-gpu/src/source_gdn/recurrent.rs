//! Closed original GDN operation; no family traversal or state publication.
//! Canonical recurrence is FP32[head,key,value], full Conv4 stays independent.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{add, bf16, div, finite, fused, mul, source_bf16, sub};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn convolution(
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
pub(crate) fn full_convolution(
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
pub(crate) fn normalize_qk(
    activated: &[f32],
    query: &mut [f32],
    key: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] kh: usize,
    #[comptime] vh: usize,
    #[comptime] cached: bool,
) {
    let at = ABSOLUTE_POS;
    if at < rows as usize * vh {
        let row = at / vh;
        let h = at % vh;
        let group = h / (vh / kh);
        let channels = (2usize * kh + vh) * 128usize;
        let input_base = row * channels;
        let output_base = row * vh * 128usize;
        let mut qs = Array::<f32>::new(32usize);
        let mut ks = Array::<f32>::new(32usize);
        for lane in 0..32usize {
            qs[lane] = 0.0f32;
            ks[lane] = 0.0f32;
        }
        for block in 0..4usize {
            for lane in 0..32usize {
                let q = activated[input_base + group * 128usize + block * 32usize + lane];
                let k = activated[input_base + (kh + group) * 128usize + block * 32usize + lane];
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
            let q = mul(activated[input_base + group * 128usize + i], qi);
            let mut scaled = mul(q, f32::reinterpret(0x3db504f3u32));
            if comptime!(cached) {
                scaled = div(q, f32::reinterpret(0x413504f3u32));
            }
            query[output_base + h * 128usize + i] = scaled;
            key[output_base + h * 128usize + i] =
                mul(activated[input_base + (kh + group) * 128usize + i], ki);
        }
        if !finite(qi) || !finite(ki) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// Cached-decode normalization with one 32-lane workgroup per value head.
/// Each lane retains one of the original 32 partials and its four-element
/// accumulation order; lane 0 performs the unchanged final eight-part reduction.
#[cube(launch_unchecked)]
pub(crate) fn normalize_qk_plane32(
    activated: &[f32],
    query: &mut [f32],
    key: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] kh: usize,
    #[comptime] vh: usize,
    #[comptime] cached: bool,
) {
    let at = usize::cast_from(CUBE_POS_X);
    if at < rows as usize * vh {
        let row = at / vh;
        let h = at % vh;
        let group = h / (vh / kh);
        let lane = usize::cast_from(UNIT_POS);
        let channels = (2usize * kh + vh) * 128usize;
        let input_base = row * channels;
        let output_base = row * vh * 128usize;

        let mut qpart = 0.0f32;
        let mut kpart = 0.0f32;
        for block in 0..4usize {
            let i = block * 32usize + lane;
            let q = activated[input_base + group * 128usize + i];
            let k = activated[input_base + (kh + group) * 128usize + i];
            qpart = add(qpart, mul(q, q));
            kpart = add(kpart, mul(k, k));
        }

        let mut parts = Shared::<[f32]>::new_slice(64usize);
        parts[lane] = qpart;
        parts[32usize + lane] = kpart;
        sync_cube();

        let mut inverse = Shared::<[f32]>::new_slice(2usize);
        if lane == 0usize {
            let mut qsum = 0.0f32;
            let mut ksum = 0.0f32;
            for i in 0..8usize {
                qsum = add(
                    qsum,
                    add(
                        add(add(parts[i], parts[i + 8usize]), parts[i + 16usize]),
                        parts[i + 24usize],
                    ),
                );
                ksum = add(
                    ksum,
                    add(
                        add(
                            add(parts[32usize + i], parts[40usize + i]),
                            parts[48usize + i],
                        ),
                        parts[56usize + i],
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
            inverse[0] = qi;
            inverse[1] = ki;
            if !finite(qi) || !finite(ki) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        sync_cube();

        let qi = inverse[0];
        let ki = inverse[1];
        for block in 0..4usize {
            let i = block * 32usize + lane;
            let q = mul(activated[input_base + group * 128usize + i], qi);
            let mut scaled = mul(q, f32::reinterpret(0x3db504f3u32));
            if comptime!(cached) {
                scaled = div(q, f32::reinterpret(0x413504f3u32));
            }
            query[output_base + h * 128usize + i] = scaled;
            key[output_base + h * 128usize + i] =
                mul(activated[input_base + (kh + group) * 128usize + i], ki);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn initial_dot(
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
pub(crate) fn decayed(
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

/// Cached-decode reduction over one or more independent private states. The
/// decay multiplication stays in the original FP32 order but is not materialized
/// into a dense state buffer.
#[cube(launch_unchecked)]
pub(crate) fn key_reduce_decayed_private(
    p0: &[f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    p4: &[f32],
    p5: &[f32],
    p6: &[f32],
    p7: &[f32],
    decay: &[f32],
    vector: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] rows: usize,
    #[comptime] values: usize,
) {
    let at = ABSOLUTE_POS;
    if at < rows * values {
        let row = at / values;
        let local = at % values;
        let heads = values / 128usize;
        let h = local / 128usize;
        let j = local % 128usize;
        let d = decay[row * heads + h];
        let mut carry = 0.0f32;
        for block in 0..8usize {
            let mut subtotal = 0.0f32;
            for key in block * 16usize..(block + 1usize) * 16usize {
                let state_at = (h * 128usize + key) * 128usize + j;
                let mut x = p0[state_at];
                if row == 1usize {
                    x = p1[state_at];
                } else if row == 2usize {
                    x = p2[state_at];
                } else if row == 3usize {
                    x = p3[state_at];
                } else if row == 4usize {
                    x = p4[state_at];
                } else if row == 5usize {
                    x = p5[state_at];
                } else if row == 6usize {
                    x = p6[state_at];
                } else if row == 7usize {
                    x = p7[state_at];
                }
                let decayed = mul(x, d);
                subtotal = add(
                    subtotal,
                    mul(decayed, vector[row * values + h * 128usize + key]),
                );
                if !finite(x) || !finite(decayed) || d < 0.0f32 || d > 1.0f32 {
                    status[0].fetch_or(NONFINITE_VALUE);
                }
            }
            carry = add(carry, subtotal);
        }
        output[at] = carry;
        if !finite(carry) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// Cached-decode update into each row's original private candidate state. The
/// decayed prior is recomputed in the same FP32 order, avoiding a dense temporary
/// and its extra state-sized traffic.
#[cube(launch_unchecked)]
pub(crate) fn state_update_decayed_private(
    p0: &[f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    p4: &[f32],
    p5: &[f32],
    p6: &[f32],
    p7: &[f32],
    decay: &[f32],
    key: &[f32],
    delta: &[f32],
    n0: &mut [f32],
    n1: &mut [f32],
    n2: &mut [f32],
    n3: &mut [f32],
    n4: &mut [f32],
    n5: &mut [f32],
    n6: &mut [f32],
    n7: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] rows: usize,
    #[comptime] elements: usize,
) {
    let at = ABSOLUTE_POS;
    if at < rows * elements {
        let row = at / elements;
        let local = at % elements;
        let values = elements / 128usize;
        let heads = elements / 16384usize;
        let h = local / 16384usize;
        let k = (local / 128usize) % 128usize;
        let j = local % 128usize;
        let mut x = p0[local];
        if row == 1usize {
            x = p1[local];
        } else if row == 2usize {
            x = p2[local];
        } else if row == 3usize {
            x = p3[local];
        } else if row == 4usize {
            x = p4[local];
        } else if row == 5usize {
            x = p5[local];
        } else if row == 6usize {
            x = p6[local];
        } else if row == 7usize {
            x = p7[local];
        }
        let d = decay[row * heads + h];
        let decayed = mul(x, d);
        let update = mul(
            key[row * values + h * 128usize + k],
            delta[row * values + h * 128usize + j],
        );
        let y = add(decayed, update);
        if row == 0usize {
            n0[local] = y;
        } else if row == 1usize {
            n1[local] = y;
        } else if row == 2usize {
            n2[local] = y;
        } else if row == 3usize {
            n3[local] = y;
        } else if row == 4usize {
            n4[local] = y;
        } else if row == 5usize {
            n5[local] = y;
        } else if row == 6usize {
            n6[local] = y;
        } else if row == 7usize {
            n7[local] = y;
        }
        if !finite(x) || !finite(decayed) || !finite(y) || d < 0.0f32 || d > 1.0f32 {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

/// Reduction from one or more independent private states into row-major vectors.
#[cube(launch_unchecked)]
pub(crate) fn key_reduce_private(
    p0: &[f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    p4: &[f32],
    p5: &[f32],
    p6: &[f32],
    p7: &[f32],
    vector: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] rows: usize,
    #[comptime] values: usize,
    #[comptime] round_bf16: bool,
) {
    let at = ABSOLUTE_POS;
    if at < rows * values {
        let row = at / values;
        let local = at % values;
        let h = local / 128usize;
        let j = local % 128usize;
        let mut carry = 0.0f32;
        for block in 0..8usize {
            let mut subtotal = 0.0f32;
            for key in block * 16usize..(block + 1usize) * 16usize {
                let state_at = (h * 128usize + key) * 128usize + j;
                let mut x = p0[state_at];
                if row == 1usize {
                    x = p1[state_at];
                } else if row == 2usize {
                    x = p2[state_at];
                } else if row == 3usize {
                    x = p3[state_at];
                } else if row == 4usize {
                    x = p4[state_at];
                } else if row == 5usize {
                    x = p5[state_at];
                } else if row == 6usize {
                    x = p6[state_at];
                } else if row == 7usize {
                    x = p7[state_at];
                }
                subtotal = add(subtotal, mul(x, vector[row * values + h * 128usize + key]));
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
pub(crate) fn key_reduce(
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
pub(crate) fn delta(
    activated: &[f32],
    memory: &[f32],
    beta: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] kh: usize,
    #[comptime] values: usize,
) {
    let at = ABSOLUTE_POS;
    if at < rows as usize * values {
        let row = at / values;
        let local = at % values;
        let heads = values / 128usize;
        let channels = kh * 256usize + values;
        let y = mul(
            sub(
                activated[row * channels + kh * 256usize + local],
                memory[at],
            ),
            beta[row * heads + local / 128usize],
        );
        output[at] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn state_update(
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
        // Initial chunk also adds its decayed empty state; keep signed-zero semantics.
        let y = add(previous[at], update);
        output[at] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn initial_core(
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
