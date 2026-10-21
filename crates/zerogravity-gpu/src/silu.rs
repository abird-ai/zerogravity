//! GGML Haswell polynomial SiLU with actual FP32 intermediate stores.
//! Each source FMA is a separate launch: compiler-local casts did not preserve
//! source rounding in earlier GDN qualification. No device exp intrinsic.
use crate::source_accumulation::source_mul_add;
use crate::source_float::{correct_quotient, round_source, widen_source};
use cubecl::prelude::*;

pub(crate) const SCRATCH: usize = 10;

#[cube(launch_unchecked)]
pub(crate) fn silu_step(
    input: &[f32],
    scratch: &mut [f32],
    output: &mut [f32],
    #[comptime] count: usize,
    #[comptime] stage: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let s = i * 10;
        let x = 0.0f32 - input[i];
        if stage == 0usize {
            scratch[s] = source_mul_add(
                x,
                f32::reinterpret(0x3fb8aa3bu32),
                f32::reinterpret(0x4b400000u32),
                true,
            );
        } else if stage == 1usize {
            scratch[s + 1] = f32::cast_from(
                f64::cast_from(scratch[s]) - f64::cast_from(f32::reinterpret(0x4b400000u32)),
            );
        } else if stage == 2usize {
            scratch[s + 2] =
                source_mul_add(-scratch[s + 1], f32::reinterpret(0x3f317200u32), x, true);
        } else if stage == 3usize {
            scratch[s + 3] = source_mul_add(
                -scratch[s + 1],
                f32::reinterpret(0x35bfbe8eu32),
                scratch[s + 2],
                true,
            );
        } else if stage == 4usize {
            let b = scratch[s + 3];
            scratch[s + 4] = f32::cast_from(f64::cast_from(b) * f64::cast_from(b));
            scratch[s + 5] = source_mul_add(
                f32::reinterpret(0x3c072010u32),
                b,
                f32::reinterpret(0x3d2b9f17u32),
                true,
            );
            scratch[s + 6] = source_mul_add(
                f32::reinterpret(0x3e2aaf33u32),
                b,
                f32::reinterpret(0x3efffedbu32),
                true,
            );
            scratch[s + 7] =
                f32::cast_from(f64::cast_from(f32::reinterpret(0x3f7ffff6u32)) * f64::cast_from(b));
        } else if stage == 5usize {
            scratch[s + 5] = source_mul_add(scratch[s + 5], scratch[s + 4], scratch[s + 6], true);
        } else if stage == 6usize {
            scratch[s + 5] = source_mul_add(scratch[s + 5], scratch[s + 4], scratch[s + 7], true);
        } else if stage == 7usize {
            let n = scratch[s + 1];
            let e = u32::reinterpret(scratch[s]) << 23u32;
            let j = scratch[s + 5];
            let k = f32::reinterpret(e + 0x3f800000u32);
            let mut base = source_mul_add(j, k, k, true);
            let mut factor = 1.0f32;
            if n.abs() > 126.0f32 {
                let mut g = 0u32;
                if n <= 0.0f32 {
                    g = 0x82000000u32;
                }
                let s1 = f32::reinterpret(g + 0x7f000000u32);
                let s2 = f32::reinterpret(e - g);
                if n.abs() > 192.0f32 {
                    base = s1;
                } else {
                    base = source_mul_add(s2, j, s2, true);
                }
                factor = s1;
            }
            scratch[s + 2] = base;
            scratch[s + 8] = factor;
        } else if stage == 8usize {
            scratch[s + 2] =
                f32::cast_from(f64::cast_from(scratch[s + 2]) * f64::cast_from(scratch[s + 8]));
        } else if stage == 9usize {
            scratch[s + 9] = f32::cast_from(1.0f64 + f64::cast_from(scratch[s + 2]));
        } else if stage == 10usize {
            output[i] = round_source(
                widen_source(input[i]) / widen_source(scratch[s + 9]),
                u32::reinterpret(input[i]) & 0x80000000u32,
            );
        } else {
            // Rank neighboring source-FP32 quotient candidates using exact
            // double products. A separate launch forces the prior estimate
            // into actual FP32 memory before this rounding correction.
            output[i] = correct_quotient(input[i], scratch[s + 9], output[i]);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn silu_gate(multiplier: &[f32], output: &mut [f32], #[comptime] count: usize) {
    let i = ABSOLUTE_POS;
    if i < count {
        let sign = (u32::reinterpret(multiplier[i]) ^ u32::reinterpret(output[i])) & 0x80000000u32;
        output[i] = round_source(widen_source(multiplier[i]) * widen_source(output[i]), sign);
    }
}
