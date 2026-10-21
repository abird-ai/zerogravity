//! Scalar table/polynomial source policy matching this host's glibc 2.42.
//! Adapted from Arm optimized-routines, revision
//! 97362b2eab91d35d30c4ac2c8c79dcd4dae4331b, math/{expf,logf,exp2f_data,logf_data}.c.
//! Copyright (c) 2017-2025 Arm Limited. SPDX-License-Identifier: MIT
//! Full permission notice: ../LICENSE-ARM-MATH. No glibc code is incorporated.

pub(crate) const TABLES: [f64; 72] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3ff059b0d3158574),
    f64::from_bits(0x3ff0b5586cf9890f),
    f64::from_bits(0x3ff11301d0125b51),
    f64::from_bits(0x3ff172b83c7d517b),
    f64::from_bits(0x3ff1d4873168b9aa),
    f64::from_bits(0x3ff2387a6e756238),
    f64::from_bits(0x3ff29e9df51fdee1),
    f64::from_bits(0x3ff306fe0a31b715),
    f64::from_bits(0x3ff371a7373aa9cb),
    f64::from_bits(0x3ff3dea64c123422),
    f64::from_bits(0x3ff44e086061892d),
    f64::from_bits(0x3ff4bfdad5362a27),
    f64::from_bits(0x3ff5342b569d4f82),
    f64::from_bits(0x3ff5ab07dd485429),
    f64::from_bits(0x3ff6247eb03a5585),
    f64::from_bits(0x3ff6a09e667f3bcd),
    f64::from_bits(0x3ff71f75e8ec5f74),
    f64::from_bits(0x3ff7a11473eb0187),
    f64::from_bits(0x3ff82589994cce13),
    f64::from_bits(0x3ff8ace5422aa0db),
    f64::from_bits(0x3ff93737b0cdc5e5),
    f64::from_bits(0x3ff9c49182a3f090),
    f64::from_bits(0x3ffa5503b23e255d),
    f64::from_bits(0x3ffae89f995ad3ad),
    f64::from_bits(0x3ffb7f76f2fb5e47),
    f64::from_bits(0x3ffc199bdd85529c),
    f64::from_bits(0x3ffcb720dcef9069),
    f64::from_bits(0x3ffd5818dcfba487),
    f64::from_bits(0x3ffdfc97337b9b5f),
    f64::from_bits(0x3ffea4afa2a490da),
    f64::from_bits(0x3fff50765b6e4540),
    f64::from_bits(0x3ff661ec79f8f3be),
    f64::from_bits(0xbfd57bf7808caade),
    f64::from_bits(0x3ff571ed4aaf883d),
    f64::from_bits(0xbfd2bef0a7c06ddb),
    f64::from_bits(0x3ff49539f0f010b0),
    f64::from_bits(0xbfd01eae7f513a67),
    f64::from_bits(0x3ff3c995b0b80385),
    f64::from_bits(0xbfcb31d8a68224e9),
    f64::from_bits(0x3ff30d190c8864a5),
    f64::from_bits(0xbfc6574f0ac07758),
    f64::from_bits(0x3ff25e227b0b8ea0),
    f64::from_bits(0xbfc1aa2bc79c8100),
    f64::from_bits(0x3ff1bb4a4a1a343f),
    f64::from_bits(0xbfba4e76ce8c0e5e),
    f64::from_bits(0x3ff12358f08ae5ba),
    f64::from_bits(0xbfb1973c5a611ccc),
    f64::from_bits(0x3ff0953f419900a7),
    f64::from_bits(0xbfa252f438e10c1e),
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x0000000000000000),
    f64::from_bits(0x3fee608cfd9a47ac),
    f64::from_bits(0x3faaa5aa5df25984),
    f64::from_bits(0x3feca4b31f026aa0),
    f64::from_bits(0x3fbc5e53aa362eb4),
    f64::from_bits(0x3feb2036576afce6),
    f64::from_bits(0x3fc526e57720db08),
    f64::from_bits(0x3fe9c2d163a1aa2d),
    f64::from_bits(0x3fcbc2860d224770),
    f64::from_bits(0x3fe886e6037841ed),
    f64::from_bits(0x3fd1058bc8a07ee1),
    f64::from_bits(0x3fe767dcf5534862),
    f64::from_bits(0x3fd4043057b6ee09),
    f64::from_bits(0x40471547652b82fe),
    f64::from_bits(0x3ebc6af84b912394),
    f64::from_bits(0x3f2ebfce50fac4f3),
    f64::from_bits(0x3f962e42ff0c52d6),
    f64::from_bits(0x3fe62e42fefa39ef),
    f64::from_bits(0xbfd00ea348b88334),
    f64::from_bits(0x3fd5575b0be00b6a),
    f64::from_bits(0xbfdffffef20a4123),
];

use crate::source_float::{round_source, widen_source};
use cubecl::prelude::*;

/// This source FMA uses a large shift whose rounding MUST be physically stored.
#[cube]
pub(crate) fn exp_shift(x: f32, tables: &[f64]) -> f64 {
    fma(widen_source(x), tables[64], 6755399441055744.0f64)
}

/// Arm scalar expf with this source host's contracted FMA reduction/polynomial.
/// `shifted` was stored to a real f64 GPU buffer by the prior kernel.
#[cube]
pub(crate) fn source_exp(x: f32, shifted: f64, tables: &[f64]) -> f32 {
    let mut output = 0.0f32;
    if x > f32::reinterpret(0x42b17217u32) {
        output = f32::reinterpret(0x7f800000u32);
    } else if x >= f32::reinterpret(0xc2cff1b4u32) {
        let kd = shifted - 6755399441055744.0f64;
        let k = i32::cast_from(kd);
        let r = fma(widen_source(x), tables[64], -kd);
        let exponent = u64::cast_from((k >> 5i32) + 1023i32) << 52u64;
        let scale = tables[usize::cast_from(k & 31i32)] * f64::reinterpret(exponent);
        let p = fma(tables[65], r, tables[66]);
        let r2 = r * r;
        let y = fma(tables[67], r, 1.0f64);
        output = round_source(fma(p, r2, y) * scale, 0u32);
    }
    output
}

/// GDN softplus calls logf only on a physically rounded finite x >= 1.
#[cube]
pub(crate) fn source_log(x: f32, tables: &[f64]) -> f32 {
    let mut output = 0.0f32;
    if x != 1.0f32 {
        let ix = u32::reinterpret(x);
        let tmp = ix - 0x3f330000u32;
        let i = usize::cast_from((tmp >> 19u32) & 15u32);
        let k = i32::reinterpret(tmp) >> 23i32;
        let iz = ix - (tmp & 0xff800000u32);
        let z = f64::cast_from(f32::reinterpret(iz));
        let r = fma(z, tables[32 + i * 2], -1.0f64);
        let y0 = fma(f64::cast_from(k), tables[68], tables[33 + i * 2]);
        let r2 = r * r;
        let y = fma(tables[70], r, tables[71]);
        let y = fma(tables[69], r2, y);
        output = f32::cast_from(fma(y, r2, y0 + r));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::TABLES;
    fn host_exp(x: f32) -> f32 {
        if x > f32::from_bits(0x42b17217) {
            return f32::INFINITY;
        }
        if x < f32::from_bits(0xc2cff1b4) {
            return 0.;
        }
        let kd = (x as f64).mul_add(TABLES[64], 6755399441055744.) - 6755399441055744.;
        let k = kd as i32;
        let r = (x as f64).mul_add(TABLES[64], -kd);
        let s = TABLES[(k & 31) as usize] * f64::from_bits((((k >> 5) + 1023) as u64) << 52);
        let p = TABLES[65].mul_add(r, TABLES[66]);
        let r2 = r * r;
        let y = TABLES[67].mul_add(r, 1.);
        (p.mul_add(r2, y) * s) as f32
    }
    fn host_log(x: f32) -> f32 {
        if x == 1. {
            return 0.;
        }
        let ix = x.to_bits();
        let tmp = ix.wrapping_sub(0x3f330000);
        let i = ((tmp >> 19) & 15) as usize;
        let k = (tmp as i32) >> 23;
        let iz = ix.wrapping_sub(tmp & 0xff800000);
        let z = f32::from_bits(iz) as f64;
        let r = z.mul_add(TABLES[32 + i * 2], -1.);
        let y0 = (k as f64).mul_add(TABLES[68], TABLES[33 + i * 2]);
        let r2 = r * r;
        let y = TABLES[70].mul_add(r, TABLES[71]);
        let y = TABLES[69].mul_add(r2, y);
        y.mul_add(r2, y0 + r) as f32
    }

    #[test]
    #[ignore = "requires the qualified glibc 2.42 FMA host profile; no GPU proof"]
    fn scalar_policy_matches_independent_host_libm_million_inputs() {
        let mut seed = 42u32;
        for _ in 0..1_000_000 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = (seed as f64 / u32::MAX as f64 * 192.0 - 104.0) as f32;
            let y = (1.0 + x.exp()).max(1.0);
            assert_eq!(host_exp(x).to_bits(), x.exp().to_bits(), "expf x={x}");
            assert_eq!(host_log(y).to_bits(), y.ln().to_bits(), "logf x={y}");
        }
    }
}
