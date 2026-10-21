//! Source-profile scalar sincosf adapted from Arm optimized-routines
//! 97362b2eab91d35d30c4ac2c8c79dcd4dae4331b, math/sincosf{.c,.h,_data.c}.
//! Copyright (c) 2018-2024, Arm Limited. SPDX-License-Identifier: MIT
//! Full permission notice: ../LICENSE-ARM-MATH. No glibc code incorporated.

pub(crate) const TRIG_TABLES: [f64; 35] = [
    f64::from_bits(0x41645f306dc9c883),
    f64::from_bits(0x3ff921fb54442d18),
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0xbfdffffffd0c621c),
    f64::from_bits(0x3fa55553e1068f19),
    f64::from_bits(0xbf56c087e89a359d),
    f64::from_bits(0x3ef99343027bf8c3),
    f64::from_bits(0xbfc555545995a603),
    f64::from_bits(0x3f81107605230bc4),
    f64::from_bits(0xbf2994eb3774cf24),
    f64::from_bits(0x3c1921fb54442d18),
    f64::from_bits(0x4064400000000000),
    f64::from_bits(0x40e45f2000000000),
    f64::from_bits(0x41645f3060000000),
    f64::from_bits(0x41e45f306dc00000),
    f64::from_bits(0x41ef306dc9c00000),
    f64::from_bits(0x41e06dc9c8800000),
    f64::from_bits(0x41db939105400000),
    f64::from_bits(0x41d391054a400000),
    f64::from_bits(0x41d1054a7f000000),
    f64::from_bits(0x41b529fc27000000),
    f64::from_bits(0x41c4fe13ab800000),
    f64::from_bits(0x41ef84eafa200000),
    f64::from_bits(0x41c3abe8fa800000),
    f64::from_bits(0x41d5f47d4d000000),
    f64::from_bits(0x41ea3ea69ba00000),
    f64::from_bits(0x41eea69bb8000000),
    f64::from_bits(0x41ca6ee06d800000),
    f64::from_bits(0x41ebb81b6c400000),
    f64::from_bits(0x41e81b6c52a00000),
    f64::from_bits(0x41eb6c52b3200000),
    f64::from_bits(0x41d8a5664f000000),
    f64::from_bits(0x41e2b32788600000),
    f64::from_bits(0x41e3278872000000),
    f64::from_bits(0x41ce21c820800000),
];

use crate::source_float::{round_source, widen_source};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn sincos_angles(
    angles: &[f32],
    tables: &[f64],
    trig: &mut [f32],
    #[comptime] half: usize,
) {
    let i = ABSOLUTE_POS;
    if i < half {
        let y = angles[i];
        let top = (u32::reinterpret(y) >> 20u32) & 0x7ffu32;
        let mut sin = y;
        let mut cos = 1.0f32;
        if top >= 0x398u32 {
            let mut x = widen_source(y);
            let mut n = 0i32;
            let mut original_sign = 0i32;
            if top >= 0x3f4u32 {
                if top < 0x42fu32 {
                    n = (i32::cast_from(x * tables[0]) + 0x800000i32) >> 24i32;
                    x = fma(-f64::cast_from(n), tables[1], x);
                } else {
                    let bits = u32::reinterpret(y);
                    let j = usize::cast_from((bits >> 26u32) & 15u32);
                    let xi = ((bits & 0xffffffu32) | 0x800000u32) << ((bits >> 23u32) & 7u32);
                    let r0 = u64::cast_from(xi) * u64::cast_from(tables[11 + j]);
                    let r1 = u64::cast_from(xi) * u64::cast_from(tables[15 + j]);
                    let r2 = u64::cast_from(xi) * u64::cast_from(tables[19 + j]);
                    let r = ((r2 >> 32u64) | (r0 << 32u64)) + r1;
                    let q = (r + (1u64 << 61u64)) >> 62u64;
                    let remainder = r - (q << 62u64);
                    n = i32::cast_from(q);
                    x = f64::cast_from(i64::reinterpret(remainder)) * tables[10];
                    original_sign = i32::cast_from(bits >> 31u32);
                }
            }
            let quadrant = n + original_sign;
            let mut xs = x;
            if (quadrant & 3i32) == 1i32 || (quadrant & 3i32) == 2i32 {
                xs = -x;
            }
            let mut csign = 1.0f64;
            if (quadrant & 2i32) != 0i32 {
                csign = -1.0f64;
            }
            let x2 = x * x;
            let x3 = x2 * xs;
            let x4 = x2 * x2;
            let x5 = x3 * x2;
            let x6 = x4 * x2;
            let s = fma(x5, fma(tables[9], x2, tables[8]), fma(x3, tables[7], xs));
            let c = fma(
                x6,
                fma(tables[6] * csign, x2, tables[5] * csign),
                fma(
                    x4,
                    tables[4] * csign,
                    fma(x2, tables[3] * csign, tables[2] * csign),
                ),
            );
            let mut ssign = 0u32;
            if s < 0.0f64 {
                ssign = 0x80000000u32;
            }
            let mut csign_bits = 0u32;
            if c < 0.0f64 {
                csign_bits = 0x80000000u32;
            }
            if (n & 1i32) != 0i32 {
                sin = round_source(c, csign_bits);
                cos = round_source(s, ssign);
            } else {
                sin = round_source(s, ssign);
                cos = round_source(c, csign_bits);
            }
        }
        trig[i * 2] = sin;
        trig[i * 2 + 1] = cos;
    }
}

#[cfg(test)]
mod tests {
    use super::TRIG_TABLES;
    fn host_sincos(y: f32) -> (f32, f32) {
        let t = &TRIG_TABLES;
        let top = (y.to_bits() >> 20) & 0x7ff;
        let (mut x, mut n, mut sign) = (y as f64, 0i32, 0usize);
        if top < 0x398 {
            return (y, 1.);
        }
        if top >= 0x3f4 {
            if top < 0x42f {
                n = ((x * t[0]) as i32 + 0x800000) >> 24;
                x = (-(n as f64)).mul_add(t[1], x);
            } else {
                let bits = y.to_bits();
                let i = ((bits >> 26) & 15) as usize;
                let xi = ((bits & 0xffffff) | 0x800000) << ((bits >> 23) & 7);
                let r0 = (xi as u64).wrapping_mul(t[11 + i] as u64);
                let r1 = (xi as u64).wrapping_mul(t[15 + i] as u64);
                let r2 = (xi as u64).wrapping_mul(t[19 + i] as u64);
                let r = (r2 >> 32) | (r0 << 32);
                let r = r.wrapping_add(r1);
                let q = r.wrapping_add(1 << 61) >> 62;
                let r = r.wrapping_sub(q << 62);
                n = q as i32;
                x = (r as i64 as f64) * t[10];
                sign = (bits >> 31) as usize;
            }
        }
        let quadrant = n as usize + sign;
        let xs = if quadrant & 3 == 1 || quadrant & 3 == 2 {
            -x
        } else {
            x
        };
        let negative = quadrant & 2 != 0;
        let x2 = x * x;
        let x3 = x2 * xs;
        let x4 = x2 * x2;
        let x5 = x3 * x2;
        let x6 = x4 * x2;
        let sin = x5.mul_add(t[9].mul_add(x2, t[8]), x3.mul_add(t[7], xs));
        let signc = if negative { -1. } else { 1. };
        let cos = x6.mul_add(
            (t[6] * signc).mul_add(x2, t[5] * signc),
            x4.mul_add(t[4] * signc, x2.mul_add(t[3] * signc, t[2] * signc)),
        );
        if n & 1 != 0 {
            (cos as f32, sin as f32)
        } else {
            (sin as f32, cos as f32)
        }
    }

    #[test]
    #[ignore = "requires qualified glibc 2.42 FMA sincosf profile; no GPU proof"]
    fn scalar_trig_matches_host_million_inputs_and_all_32k_rope_phases() {
        let mut seed = 42u32;
        for _ in 0..1_000_000 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = (seed as i32 as f64 / i32::MAX as f64 * 32768.) as f32;
            let a = host_sincos(x);
            let e = x.sin_cos();
            assert_eq!(
                (a.0.to_bits(), a.1.to_bits()),
                (e.0.to_bits(), e.1.to_bits()),
                "x={x}"
            );
        }
        let scale = 10_000_000f32.powf(-2. / 64.);
        for pos in 0..=32768 {
            let mut x = pos as f32;
            for _ in 0..32 {
                let a = host_sincos(x);
                let e = x.sin_cos();
                assert_eq!(
                    (a.0.to_bits(), a.1.to_bits()),
                    (e.0.to_bits(), e.1.to_bits()),
                    "position={pos} angle={x}"
                );
                x *= scale;
            }
        }
    }
}
