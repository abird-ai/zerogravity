//! Explicit standalone-exp lane; no family/model dispatch or fallback.
//! Original HA/FTZ-off normal domain only; physical qualification is separate.
use crate::program_kernels::NONFINITE_VALUE;
use crate::source_arithmetic::{add, mul, sub};
use cubecl::prelude::*;

/// Every add/multiply crosses the shared integer-rounding FP32 boundary.
/// Tables are the mathematically derived core contract, uploaded unchanged.
#[cube]
pub(crate) fn mkl_ha_normal(x: f32, table: &[u32]) -> f32 {
    let scaled = mul(x, f32::reinterpret(0x4338aa3bu32));
    let rounded = add(scaled, 12582912.0f32);
    let n = sub(rounded, 12582912.0f32);
    let index = usize::cast_from((u32::reinterpret(rounded) & 127u32) * 2u32);
    let high = mul(n, f32::reinterpret(0x3bb18000u32));
    let r = sub(x, high);
    let low = mul(n, f32::reinterpret(0x35de8083u32));
    let r = add(r, low);
    let square = mul(r, r);
    let quadratic = mul(square, 0.5f32);
    let correction = add(quadratic, r);
    let hi = f32::reinterpret(table[index]);
    let lo = f32::reinterpret(table[index + 1usize]);
    let corrected = mul(correction, hi);
    let corrected = add(corrected, lo);
    let unscaled = add(corrected, hi);
    // Modulo2^32 exponent arithmetic. Splitting low/high bits avoids overflow
    // in the sum even in host-debug extraction of this actual shader body.
    let exponent = ((u32::reinterpret(rounded) & 0x7fffu32) + 0x3f80u32) & 0x7f80u32;
    let scale = f32::reinterpret(exponent << 16u32);
    mul(unscaled, scale)
}

#[cube(launch_unchecked)]
pub(crate) fn normal_vector(
    input: &[f32],
    table: &[u32],
    output: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] elements: usize,
) {
    let i = ABSOLUTE_POS;
    if i < elements {
        let x = input[i];
        let bits = u32::reinterpret(x) & 0x7fffffffu32;
        if bits <= 0x42ae0000u32 {
            output[i] = mkl_ha_normal(x, table);
        } else {
            // Refuse through the original flight status before any publication.
            status[0].fetch_or(NONFINITE_VALUE);
            output[i] = 0.0f32;
        }
    }
}
