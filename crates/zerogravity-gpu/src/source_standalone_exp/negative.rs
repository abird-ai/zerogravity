//! Separate nonpositive/causal-mask source exp. Normal-only profile unchanged.
use crate::program_kernels::NONFINITE_VALUE;
use cubecl::prelude::*;

// The authenticated table starts with (1, +0). Runtime operands prevent the
// IR optimizer from eliding separate FP64 rounding/compensated residuals.
// Each FMA is exactly one multiply OR add, never a contracted source pair.
#[cube]
pub(crate) fn add64(a: f64, b: f64, table: &[u64]) -> f64 {
    fma(a, f64::reinterpret(table[0usize]), b)
}
#[cube]
fn mul64(a: f64, b: f64, table: &[u64]) -> f64 {
    fma(a, b, f64::reinterpret(table[1usize]))
}

#[cube]
pub(crate) fn lane(x: f32, normal: &[u32], table: &[u64], status: &[Atomic<u32>]) -> f32 {
    let mut result = 0.0f32;
    let magnitude = u32::reinterpret(x) & 0x7fffffffu32;
    if magnitude > 0x7f800000u32 || x > 0.0f32 {
        status[0].fetch_or(NONFINITE_VALUE);
    } else if x >= -f32::reinterpret(0x42aeac4fu32) {
        result = crate::source_standalone_exp::mkl_ha_normal(x, normal);
    } else if x >= f32::reinterpret(0xc2cff1b4u32) {
        let wide = crate::source_float::widen_source(x);
        let scaled = mul64(wide, f64::reinterpret(0x40571547652b82feu64), table);
        let rounded = add64(scaled, 6755399441055744.0f64, table);
        let n = add64(rounded, -6755399441055744.0f64, table);
        let hi = mul64(n, f64::reinterpret(0xbf862e42fefa0000u64), table);
        let lo = mul64(n, f64::reinterpret(0xbd1cf79abc9e3b3au64), table);
        let r = add64(add64(hi, wide, table), lo, table);
        let p = mul64(f64::reinterpret(0x3f56c16a1c2a3ffdu64), r, table);
        let p = add64(p, f64::reinterpret(0x3f8111123aaf20d3u64), table);
        let p = mul64(p, r, table);
        let p = add64(p, f64::reinterpret(0x3fa5555555558fccu64), table);
        let p = mul64(p, r, table);
        let p = add64(p, f64::reinterpret(0x3fc55555555548f8u64), table);
        let p = mul64(p, r, table);
        let p = add64(p, 0.5f64, table);
        let p = mul64(p, r, table);
        let p = mul64(p, r, table);
        let p = add64(p, r, table);
        let bits = u32::cast_from(u64::reinterpret(rounded));
        let index = usize::cast_from((bits & 63u32) * 2u32);
        let correction = add64(p, f64::reinterpret(table[index + 1usize]), table);
        let hi = f64::reinterpret(table[index]);
        let correction = mul64(correction, hi, table);
        let exponent = ((bits >> 6u32) + 0x43bu32) & 0x7ffu32;
        let scale = f64::reinterpret(u64::cast_from(exponent) << 52u64);
        let correction = mul64(correction, scale, table);
        let hi = mul64(hi, scale, table);
        let sum = add64(hi, correction, table);
        let residual = add64(add64(hi, -sum, table), correction, table);
        let shifted = mul64(sum, 6442450944.0f64, table);
        let split = add64(add64(sum, shifted, table), -shifted, table);
        let tail = add64(add64(sum, -split, table), residual, table);
        let down = f64::reinterpret(0x3c30000000000000u64);
        let hi = mul64(split, down, table);
        let lo = mul64(tail, down, table);
        result = crate::source_float::round_source(add64(hi, lo, table), 0u32);
    }
    result
}
