//! Shared physical source-FP32 conversion and quotient rounding policy.
use cubecl::prelude::*;

/// Preserve stored f32 subnormals across Vulkan conversion/arithmetic modes.
#[cube]
pub(crate) fn widen_source(x: f32) -> f64 {
    let bits = u32::reinterpret(x);
    let mut result = f64::cast_from(x);
    if (bits & 0x7f800000u32) == 0u32 {
        result = f64::cast_from(bits & 0x007fffffu32) * 1.401298464324817e-45f64;
        if (bits & 0x80000000u32) != 0u32 {
            result = -result;
        }
    }
    result
}

/// Nearest-even subnormal packing; explicit sign settles exact zero only.
#[cube]
pub(crate) fn round_source(x: f64, sign: u32) -> f32 {
    let mut result = f32::cast_from(x);
    if x.abs() < 1.1754943508222875e-38f64 {
        let scaled = x.abs() * 7.1362384635298e44f64;
        let mut bits = u32::cast_from(scaled);
        let remainder = scaled - f64::cast_from(bits);
        if remainder > 0.5f64 || (remainder == 0.5f64 && (bits & 1u32) != 0u32) {
            bits += 1u32;
        }
        let mut packed_sign = sign;
        if x != 0.0f64 {
            packed_sign = 0u32;
            if x < 0.0f64 {
                packed_sign = 0x80000000u32;
            }
        }
        result = f32::reinterpret(bits | packed_sign);
    }
    result
}

/// Correct a physically stored quotient estimate with source nearest-even bits.
#[cube]
pub(crate) fn correct_quotient(numerator: f32, divisor: f32, candidate: f32) -> f32 {
    let divisor = widen_source(divisor);
    let numerator = widen_source(numerator);
    let mut candidate = candidate;
    let bits = u32::reinterpret(candidate);
    let magnitude = bits & 0x7fffffffu32;
    if magnitude > 1u32 && magnitude < 0x7f7ffffeu32 {
        let mut best = (numerator - divisor * widen_source(candidate)).abs();
        let low_bits = bits - 1u32;
        let low = f32::reinterpret(low_bits);
        let low_error = (numerator - divisor * widen_source(low)).abs();
        if low_error < best || (low_error == best && (low_bits & 1u32) == 0u32) {
            candidate = low;
            best = low_error;
        }
        let high_bits = bits + 1u32;
        let high = f32::reinterpret(high_bits);
        let high_error = (numerator - divisor * widen_source(high)).abs();
        if high_error < best || (high_error == best && (high_bits & 1u32) == 0u32) {
            candidate = high;
        }
    }
    candidate
}

/// Correct FP64-to-FP32 double rounding at a midpoint. `candidate` and `sum`
/// are physically stored, and `error` is the FastTwoSum residual constructed
/// using a separately stored rounded subtraction. Finite f32 operands have an
/// exact f64 product, so sum + error represents the exact source FMA result.
#[cube]
pub(crate) fn correct_fma(sum: f64, error: f64, candidate: f32) -> f32 {
    let bits = u32::reinterpret(candidate);
    let magnitude = bits & 0x7fffffffu32;
    let mut cv = widen_source(candidate);
    if magnitude == 0x7f800000u32 {
        cv = 3.402823669209385e38f64;
        if (bits & 0x80000000u32) != 0u32 {
            cv = -cv;
        }
    }
    let mut output = candidate;
    if sum != cv && magnitude <= 0x7f800000u32 && error != 0.0f64 {
        let mut next = bits;
        if (sum > cv && (bits & 0x80000000u32) == 0u32)
            || (sum < cv && (bits & 0x80000000u32) != 0u32)
        {
            next += 1u32;
        } else {
            next -= 1u32;
        }
        let mut nv = widen_source(f32::reinterpret(next));
        if (next & 0x7fffffffu32) == 0x7f800000u32 {
            nv = 3.402823669209385e38f64;
            if (next & 0x80000000u32) != 0u32 {
                nv = -nv;
            }
        }
        if sum == (cv + nv) * 0.5f64
            && ((sum > cv && error > 0.0f64) || (sum < cv && error < 0.0f64))
        {
            output = f32::reinterpret(next);
        }
    }
    output
}

#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    any(feature = "hip", feature = "vulkan"),
    target_os = "linux"
))]
pub(crate) mod qualification;
