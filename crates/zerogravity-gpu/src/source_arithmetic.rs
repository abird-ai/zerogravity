//! Explicit source FP32 operations; no provider exp/rsqrt/fast-math policy.
use crate::source_float::{correct_quotient, round_source, widen_source};
use cubecl::prelude::*;

#[cube]
pub(crate) fn neg(x: f32) -> f32 {
    f32::reinterpret(u32::reinterpret(x) ^ 0x80000000u32)
}
#[cube]
pub(crate) fn add(a: f32, b: f32) -> f32 {
    round_source(
        widen_source(a) + widen_source(b),
        u32::reinterpret(a) & u32::reinterpret(b) & 0x80000000u32,
    )
}
#[cube]
pub(crate) fn sub(a: f32, b: f32) -> f32 {
    add(a, neg(b))
}
#[cube]
pub(crate) fn mul(a: f32, b: f32) -> f32 {
    round_source(
        widen_source(a) * widen_source(b),
        (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32,
    )
}
#[cube]
pub(crate) fn div(a: f32, b: f32) -> f32 {
    let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
    let candidate = round_source(widen_source(a) / widen_source(b), sign);
    correct_quotient(a, b, candidate)
}
#[cube]
pub(crate) fn fused(a: f32, b: f32, c: f32) -> f32 {
    // Integer significands retain a fused product's exact48 bits and the
    // addend, with a jammed low bit only when cancellation cannot expose it.
    // Unlike an inline FP64 FastTwoSum, this needs no optimizer-sensitive
    // residual store and cannot double-round a source FP32 midpoint.
    let ab = u32::reinterpret(a);
    let bb = u32::reinterpret(b);
    let cb = u32::reinterpret(c);
    let ae = (ab >> 23u32) & 255u32;
    let be = (bb >> 23u32) & 255u32;
    let ce = (cb >> 23u32) & 255u32;
    let mut output = c;
    if ae != 255u32 && be != 255u32 && ce != 255u32 {
        let mut am = u64::cast_from(ab & 0x7fffffu32);
        let mut bm = u64::cast_from(bb & 0x7fffffu32);
        let mut cm = u64::cast_from(cb & 0x7fffffu32);
        let mut ax = i32::cast_from(-149i32);
        let mut bx = i32::cast_from(-149i32);
        let mut cx = i32::cast_from(-149i32);
        if ae != 0u32 {
            am |= 0x800000u64;
            ax = i32::cast_from(ae) - 150i32;
        }
        if be != 0u32 {
            bm |= 0x800000u64;
            bx = i32::cast_from(be) - 150i32;
        }
        if ce != 0u32 {
            cm |= 0x800000u64;
            cx = i32::cast_from(ce) - 150i32;
        }
        let product = am * bm;
        let px = ax + bx;
        let ps = (ab ^ bb) & 0x80000000u32;
        let cs = cb & 0x80000000u32;
        if product == 0u64 && cm == 0u64 {
            output = f32::reinterpret(ps & cs);
        } else if product != 0u64 {
            let mut highest = px + bit_index(product);
            if cm != 0u64 {
                let ch = cx + bit_index(cm);
                if ch > highest {
                    highest = ch;
                }
            }
            let base = highest - 60i32;
            let p = align(product, px - base);
            let q = align(cm, cx - base);
            let mut sign = ps;
            let mut magnitude = p + q;
            if ps != cs {
                if p >= q {
                    magnitude = p - q;
                } else {
                    magnitude = q - p;
                    sign = cs;
                }
            }
            if magnitude == 0u64 {
                sign = 0u32;
            }
            output = f32::reinterpret(pack(magnitude, base, sign));
        }
    } else {
        output = fma(a, b, c);
    }
    output
}
#[cube]
fn bit_index(value: u64) -> i32 {
    let mut x = value;
    let mut index = 0i32;
    if x >= 0x100000000u64 {
        x >>= 32u64;
        index += 32i32;
    }
    if x >= 0x10000u64 {
        x >>= 16u64;
        index += 16i32;
    }
    if x >= 0x100u64 {
        x >>= 8u64;
        index += 8i32;
    }
    if x >= 0x10u64 {
        x >>= 4u64;
        index += 4i32;
    }
    if x >= 4u64 {
        x >>= 2u64;
        index += 2i32;
    }
    if x >= 2u64 {
        index += 1i32;
    }
    index
}
#[cube]
fn align(value: u64, shift: i32) -> u64 {
    let mut result = value;
    if value != 0u64 {
        if shift >= 0i32 {
            result = value << u64::cast_from(shift);
        } else {
            let right = u64::cast_from(-shift);
            if right >= 64u64 {
                result = 1u64;
            } else {
                result = value >> right;
                if (value & ((1u64 << right) - 1u64)) != 0u64 {
                    result |= 1u64;
                }
            }
        }
    }
    result
}
#[cube]
fn nearest(value: u64, shift: i32) -> u64 {
    let mut result = 0u64;
    if shift <= 0i32 {
        result = value << u64::cast_from(-shift);
    } else if shift < 64i32 {
        let n = u64::cast_from(shift);
        result = value >> n;
        let remainder = value & ((1u64 << n) - 1u64);
        let half = 1u64 << (n - 1u64);
        if remainder > half || (remainder == half && (result & 1u64) != 0u64) {
            result += 1u64;
        }
    }
    result
}
#[cube]
fn pack(value: u64, base: i32, sign: u32) -> u32 {
    let mut result = sign;
    if value != 0u64 {
        let high = bit_index(value);
        let mut exponent = base + high;
        if exponent < -126i32 {
            result |= u32::cast_from(nearest(value, -149i32 - base));
        } else {
            let mut mantissa = nearest(value, high - 23i32);
            if mantissa >= 0x1000000u64 {
                mantissa >>= 1u64;
                exponent += 1i32;
            }
            if exponent > 127i32 {
                result |= 0x7f800000u32;
            } else {
                result |= (u32::cast_from(exponent + 127i32) << 23u32)
                    | (u32::cast_from(mantissa) & 0x7fffffu32);
            }
        }
    }
    result
}
#[cube]
pub(crate) fn bf16(x: f32) -> f32 {
    let b = u32::reinterpret(x);
    f32::reinterpret((b + (0x7fffu32 + ((b >> 16u32) & 1u32))) & 0xffff0000u32)
}
#[cube]
pub(crate) fn finite(x: f32) -> bool {
    (u32::reinterpret(x) & 0x7f800000u32) != 0x7f800000u32
}
#[cube]
pub(crate) fn source_bf16(x: f32) -> bool {
    finite(x) && (u32::reinterpret(x) & 0xffffu32) == 0u32
}
