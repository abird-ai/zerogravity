//! Runtime-row transports and source BF16 arithmetic. No state publication.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{add, bf16, finite, mul, neg, source_bf16};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(super) fn columns(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    input_offset: u32,
    output_offset: u32,
    #[comptime] width: usize,
    #[comptime] input_stride: usize,
    #[comptime] output_stride: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows as usize * width {
        let row = i / width;
        let column = i % width;
        let value = input[row * input_stride + input_offset as usize + column];
        output[row * output_stride + output_offset as usize + column] = value;
        if !source_bf16(value) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn residual(
    input: &[f32],
    prior: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    elements: u32,
) {
    let i = ABSOLUTE_POS;
    if i < elements as usize {
        let x = input[i];
        let y = prior[i];
        let z = bf16(add(x, y));
        output[i] = z;
        if !source_bf16(x) || !source_bf16(y) || !finite(z) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn multiply(
    input: &[f32],
    gate: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    elements: u32,
) {
    let i = ABSOLUTE_POS;
    if i < elements as usize {
        let x = input[i];
        let y = gate[i];
        let z = bf16(mul(x, y));
        output[i] = z;
        if !source_bf16(x) || !source_bf16(y) || !finite(z) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn silu_mul(
    input: &[f32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows as usize * width {
        let at = i / width * (width * 2usize) + i % width;
        let x = input[at];
        let y = input[at + width];
        let z = bf16(mul(bf16(crate::source_sleef::silu(x)), y));
        output[i] = z;
        if !source_bf16(x) || !source_bf16(y) || !finite(z) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn convolution(
    packed: &[f32],
    previous: &[f32],
    weights: &[f32],
    activated: &mut [f32],
    gate: &mut [f32],
    proposed: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    initialized: u32,
    #[comptime] channels: usize,
    #[comptime] gates: usize,
) {
    let c = ABSOLUTE_POS;
    if c < channels {
        let stride = channels + gates;
        for j in 0..4usize {
            let x = previous[c * 4usize + j];
            if !source_bf16(x) || (initialized == 0u32 && u32::reinterpret(x) != 0u32) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        for t in 0..rows as usize {
            // Keep the four independent products and left-associated source sum.
            let mut values = Array::<f32>::new(4usize);
            for j in 0..4usize {
                let combined = t + j + 1usize;
                let x = if combined < 4usize {
                    previous[c * 4usize + combined]
                } else {
                    packed[(combined - 4usize) * stride + c]
                };
                let w = weights[c * 4usize + j];
                values[j] = mul(x, w);
                if !source_bf16(x) || !source_bf16(w) {
                    status[0].fetch_or(NONFINITE_VALUE);
                }
            }
            let raw = bf16(add(
                add(add(values[0usize], values[1usize]), values[2usize]),
                values[3usize],
            ));
            let y = bf16(crate::source_sleef::silu(raw));
            activated[t * channels + c] = y;
            if !finite(y) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
            if c < gates {
                let z = packed[t * stride + channels + c];
                gate[t * gates + c] = z;
                if !source_bf16(z) {
                    status[0].fetch_or(NONFINITE_VALUE);
                }
            }
        }
        for j in 0..4usize {
            let combined = rows as usize + j;
            let x = if combined < 4usize {
                previous[c * 4usize + combined]
            } else {
                packed[(combined - 4usize) * stride + c]
            };
            proposed[c * 4usize + j] = x;
        }
    }
}

/// Independent cached-decode rows sharing one Conv4 launch. The bounded ABI
/// keeps every session's state in its original buffer; inactive bindings are
/// unreachable and may alias row zero.
#[cube(launch_unchecked)]
pub(super) fn convolution_cohort(
    packed: &[f32],
    p0: &[f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    p4: &[f32],
    p5: &[f32],
    p6: &[f32],
    p7: &[f32],
    weights: &[f32],
    activated: &mut [f32],
    gate: &mut [f32],
    n0: &mut [f32],
    n1: &mut [f32],
    n2: &mut [f32],
    n3: &mut [f32],
    n4: &mut [f32],
    n5: &mut [f32],
    n6: &mut [f32],
    n7: &mut [f32],
    status: &[Atomic<u32>],
    initialized: u32,
    #[comptime] rows: usize,
    #[comptime] channels: usize,
    #[comptime] gates: usize,
) {
    let at = ABSOLUTE_POS;
    if at < rows * channels {
        let row = at / channels;
        let c = at % channels;
        let stride = channels + gates;
        let base = c * 4usize;
        for j in 0..4usize {
            let mut x = p0[base + j];
            if row == 1usize {
                x = p1[base + j];
            } else if row == 2usize {
                x = p2[base + j];
            } else if row == 3usize {
                x = p3[base + j];
            } else if row == 4usize {
                x = p4[base + j];
            } else if row == 5usize {
                x = p5[base + j];
            } else if row == 6usize {
                x = p6[base + j];
            } else if row == 7usize {
                x = p7[base + j];
            }
            if !source_bf16(x) || (initialized == 0u32 && u32::reinterpret(x) != 0u32) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        let mut values = Array::<f32>::new(4usize);
        for j in 0..4usize {
            let combined = j + 1usize;
            let mut x = if combined < 4usize {
                p0[base + combined]
            } else {
                packed[row * stride + c]
            };
            if combined < 4usize {
                if row == 1usize {
                    x = p1[base + combined];
                } else if row == 2usize {
                    x = p2[base + combined];
                } else if row == 3usize {
                    x = p3[base + combined];
                } else if row == 4usize {
                    x = p4[base + combined];
                } else if row == 5usize {
                    x = p5[base + combined];
                } else if row == 6usize {
                    x = p6[base + combined];
                } else if row == 7usize {
                    x = p7[base + combined];
                }
            }
            let w = weights[c * 4usize + j];
            values[j] = mul(x, w);
            if !source_bf16(x) || !source_bf16(w) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        let raw = bf16(add(
            add(add(values[0usize], values[1usize]), values[2usize]),
            values[3usize],
        ));
        let y = bf16(crate::source_sleef::silu(raw));
        activated[row * channels + c] = y;
        if !finite(y) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
        if c < gates {
            let z = packed[row * stride + channels + c];
            gate[row * gates + c] = z;
            if !source_bf16(z) {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        }
        for j in 0..4usize {
            let combined = j + 1usize;
            let mut x = if combined < 4usize {
                p0[base + combined]
            } else {
                packed[row * stride + c]
            };
            if combined < 4usize {
                if row == 1usize {
                    x = p1[base + combined];
                } else if row == 2usize {
                    x = p2[base + combined];
                } else if row == 3usize {
                    x = p3[base + combined];
                } else if row == 4usize {
                    x = p4[base + combined];
                } else if row == 5usize {
                    x = p5[base + combined];
                } else if row == 6usize {
                    x = p6[base + combined];
                } else if row == 7usize {
                    x = p7[base + combined];
                }
            }
            if row == 0usize {
                n0[base + j] = x;
            } else if row == 1usize {
                n1[base + j] = x;
            } else if row == 2usize {
                n2[base + j] = x;
            } else if row == 3usize {
                n3[base + j] = x;
            } else if row == 4usize {
                n4[base + j] = x;
            } else if row == 5usize {
                n5[base + j] = x;
            } else if row == 6usize {
                n6[base + j] = x;
            } else if row == 7usize {
                n7[base + j] = x;
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn repeat_qkv(
    activated: &[f32],
    query: &mut [f32],
    key: &mut [f32],
    value: &mut [f32],
    rows: u32,
    #[comptime] kh: usize,
    #[comptime] vh: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows as usize * vh * 128usize {
        let t = i / (vh * 128usize);
        let h = (i / 128usize) % vh;
        let c = i % 128usize;
        let base = t * (kh * 2usize + vh) * 128usize;
        let group = h / (vh / kh);
        query[i] = activated[base + group * 128usize + c];
        key[i] = activated[base + (kh + group) * 128usize + c];
        value[i] = activated[base + kh * 256usize + h * 128usize + c];
    }
}

/// Cached-decode coefficients with the immutable per-head exp evaluated in
/// the same source arithmetic, avoiding a separate tiny-vector launch/scratch.
#[cube(launch_unchecked)]
pub(super) fn coefficients_decode(
    packed: &[f32],
    bias: &[f32],
    a_log: &[f32],
    normal: &[u32],
    beta_input: &mut [f32],
    log_decay: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] heads: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows as usize * heads {
        let h = i % heads;
        let at = i / heads * heads * 2usize + h;
        let b = packed[at];
        let a = packed[at + heads];
        let x = a_log[h];
        let bits = u32::reinterpret(x) & 0x7fffffffu32;
        let mut a_exp = 0.0f32;
        if bits <= 0x42ae0000u32 {
            a_exp = crate::source_standalone_exp::mkl_ha_normal(x, normal);
        } else {
            status[0].fetch_or(NONFINITE_VALUE);
        }
        let d = mul(neg(a_exp), crate::source_sleef::softplus(add(a, bias[h])));
        beta_input[i] = b;
        log_decay[i] = d;
        if !source_bf16(a) || !source_bf16(b) || !finite(bias[h]) || !finite(d) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn coefficients(
    packed: &[f32],
    bias: &[f32],
    a_exp: &[f32],
    beta_input: &mut [f32],
    log_decay: &mut [f32],
    status: &[Atomic<u32>],
    rows: u32,
    #[comptime] heads: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows as usize * heads {
        let h = i % heads;
        let at = i / heads * heads * 2usize + h;
        let b = packed[at];
        let a = packed[at + heads];
        let d = mul(
            neg(a_exp[h]),
            crate::source_sleef::softplus(add(a, bias[h])),
        );
        beta_input[i] = b;
        log_decay[i] = d;
        if !source_bf16(a) || !source_bf16(b) || !finite(bias[h]) || !finite(d) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn split_attention(
    packed: &[f32],
    query: &mut [f32],
    key: &mut [f32],
    gate: &mut [f32],
    rows: u32,
    #[comptime] qh: usize,
    #[comptime] kh: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows as usize * qh * 256usize {
        let t = i / (qh * 256usize);
        let h = (i / 256usize) % qh;
        let c = i % 256usize;
        let base = t * (qh * 2usize + kh * 2usize) * 256usize;
        query[i] = packed[base + h * 512usize + c];
        gate[i] = packed[base + h * 512usize + 256usize + c];
        if h < kh {
            key[t * kh * 256usize + h * 256usize + c] =
                packed[base + qh * 512usize + h * 256usize + c];
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn append_kv(
    prior_key: &[f32],
    prior_value: &[f32],
    key: &[f32],
    packed: &[f32],
    next_key: &mut [f32],
    next_value: &mut [f32],
    status: &[Atomic<u32>],
    base: u32,
    rows: u32,
    value_offset: u32,
    #[comptime] width: usize,
    #[comptime] packed_stride: usize,
    #[comptime] capacity: usize,
) {
    let i = ABSOLUTE_POS;
    if i < capacity * width {
        let t = i / width;
        let c = i % width;
        let mut k = 0.0f32;
        let mut v = 0.0f32;
        if t < base as usize {
            k = prior_key[i];
            v = prior_value[i];
        } else if t < (base + rows) as usize {
            let row = t - base as usize;
            k = key[row * width + c];
            v = packed[row * packed_stride + value_offset as usize + c];
        }
        next_key[i] = k;
        next_value[i] = v;
        if !source_bf16(k) || !source_bf16(v) {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}
