//! Original CPU HA sine/cosine schedule; independent of scalar sincosf.
//! SSE2/EXHA reduction, polynomial and compensated 256-sector reconstruction.
#[cfg(test)]
mod tests;
use crate::source_arithmetic::{add, mul, sub};
use crate::source_float::{round_source, widen_source};
use cubecl::prelude::*;
mod tables;
pub(crate) use tables::TABLES;

const RECIPROCAL: usize = 16;
const SINE: usize = RECIPROCAL + 115 * 3;
const COSINE: usize = SINE + 256 * 4;

#[cube]
fn coefficient(table: &[u32], #[comptime] index: usize) -> f64 {
    f64::reinterpret(
        u64::cast_from(table[index * 2]) | (u64::cast_from(table[index * 2 + 1]) << 32u64),
    )
}

// Loaded zero/one anchor separate FP64 rounds despite compiler contraction.
#[cube]
fn product(a: f64, b: f64, zero: f64) -> f64 {
    let sign = (u64::reinterpret(a) ^ u64::reinterpret(b)) & 0x8000000000000000u64;
    fma(a, b, f64::reinterpret(u64::reinterpret(zero) | sign))
}
#[cube]
fn sum(a: f64, b: f64, one: f64) -> f64 {
    fma(a, one, b)
}

#[cube]
fn normal(x: f32, table: &[u32], #[comptime] cosine: bool) -> f32 {
    let mut shifted = x;
    if cosine {
        shifted = add(x, f32::reinterpret(0x3fc90fdbu32));
    }
    let q = add(mul(shifted, f32::reinterpret(0x3ea2f983u32)), 12582912.0f32);
    let mut n = sub(q, 12582912.0f32);
    if cosine {
        n = sub(n, 0.5f32);
    }
    let n = widen_source(n);
    let zero = coefficient(table, 0usize);
    let one = coefficient(table, 1usize);
    let hi = product(n, coefficient(table, 2usize), zero);
    let r = sum(widen_source(x), -hi, one);
    let lo = product(n, coefficient(table, 3usize), zero);
    let r = sum(r, -lo, one);
    let square = product(r, r, zero);
    let p = product(square, coefficient(table, 4usize), zero);
    let p = sum(p, coefficient(table, 5usize), one);
    let p = product(p, square, zero);
    let p = sum(p, coefficient(table, 6usize), one);
    let p = product(p, square, zero);
    let p = sum(p, coefficient(table, 7usize), one);
    let p = product(p, square, zero);
    let p = product(p, r, zero);
    let value = sum(p, r, one);
    let sign = u32::cast_from(u64::reinterpret(value) >> 32u64) & 0x80000000u32;
    let value = round_source(value, sign);
    f32::reinterpret(u32::reinterpret(value) ^ ((u32::reinterpret(q) & 1u32) << 31u32))
}

#[cube]
fn large(x: f32, table: &[u32], #[comptime] cosine: bool) -> f32 {
    let bits = u32::reinterpret(x);
    let exponent = usize::cast_from(bits >> 23u32);
    let mantissa = u64::cast_from((bits & 0x7fffffu32) | 0x800000u32);
    let index = RECIPROCAL + (exponent - 140usize) * 3usize;
    // Exact 24x96-bit product; retain the first64 fractional phase bits.
    let p0 = mantissa * u64::cast_from(table[index + 2]);
    let p1 = mantissa * u64::cast_from(table[index + 1]) + (p0 >> 32u64);
    let p2 = mantissa * u64::cast_from(table[index]) + (p1 >> 32u64);
    let hi = u32::cast_from(p2);
    let mid = u32::cast_from(p1);
    let top = f32::reinterpret(0x3f800000u32 | (hi >> 9u32));
    let grid = add(top, 49152.0f32);
    let sector = usize::cast_from(u32::reinterpret(grid) & 255u32);
    let top = sub(top, sub(grid, 49152.0f32));
    let middle = sub(
        f32::reinterpret(0x34000000u32 | ((hi << 14u32) & 0x7fc000u32) | (mid >> 18u32)),
        f32::reinterpret(0x34000000u32),
    );
    let low = sub(
        f32::reinterpret(0x28800000u32 | ((mid << 5u32) & 0x7fffe0u32)),
        f32::reinterpret(0x28800000u32),
    );
    let y = add(middle, top);
    let tail = add(add(sub(top, y), middle), low);
    let high = f32::reinterpret(u32::reinterpret(y) & 0xfffff000u32);
    let lower = sub(y, high);
    let ph = mul(high, f32::reinterpret(0x40c91000u32));
    let pl = sub(
        mul(lower, f32::reinterpret(0x40c91000u32)),
        mul(high, f32::reinterpret(0x3795777au32)),
    );
    let pt = sub(
        mul(tail, f32::reinterpret(0x40c90fdbu32)),
        mul(lower, f32::reinterpret(0x3795777au32)),
    );
    let pt = add(pt, pl);
    let r = add(ph, pt);
    let rl = add(sub(ph, r), pt);
    let mut index = SINE + sector * 4usize;
    if cosine {
        index = COSINE + sector * 4usize;
    }
    let s0 = f32::reinterpret(table[index]);
    let c0 = f32::reinterpret(table[index + 1]);
    let c1 = f32::reinterpret(table[index + 2]);
    let s1 = f32::reinterpret(table[index + 3]);
    let s = add(s1, s0);
    let product0 = mul(s0, r);
    let product1 = mul(s1, r);
    let square = mul(r, r);
    let sp = mul(
        mul(
            add(
                mul(square, f32::reinterpret(0x3c08885cu32)),
                f32::reinterpret(0xbe2aaaabu32),
            ),
            square,
        ),
        r,
    );
    let cp = mul(
        sub(mul(square, f32::reinterpret(0x3d2aaa7cu32)), 0.5f32),
        square,
    );
    let p0 = add(product1, c0);
    let e0 = add(sub(c0, p0), product1);
    let p1 = add(product0, p0);
    let e1 = add(sub(p0, p1), product0);
    let error = add(e1, e0);
    let s = sub(s, mul(r, c0));
    let correction = add(mul(rl, s), c1);
    let ct = add(mul(cp, c0), correction);
    add(add(add(mul(s, sp), error), ct), p1)
}

#[cube]
fn value(x: f32, table: &[u32], #[comptime] cosine: bool) -> f32 {
    let bits = u32::reinterpret(x);
    let magnitude = f32::reinterpret(bits & 0x7fffffffu32);
    let mut result = if magnitude <= 10000.0f32 {
        normal(magnitude, table, cosine)
    } else {
        large(magnitude, table, cosine)
    };
    if !cosine {
        result = f32::reinterpret(u32::reinterpret(result) ^ (bits & 0x80000000u32));
    }
    result
}

#[cube(launch_unchecked)]
pub(crate) fn sincos_angles(
    angles: &[f32],
    table: &[u32],
    trig: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] half: usize,
) {
    let i = ABSOLUTE_POS;
    if i < half {
        let x = angles[i];
        if (u32::reinterpret(x) & 0x7f800000u32) != 0x7f800000u32 {
            trig[i * 2] = value(x, table, false);
            trig[i * 2 + 1] = value(x, table, true);
        } else {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
            trig[i * 2] = 0.0f32;
            trig[i * 2 + 1] = 0.0f32;
        }
    }
}
