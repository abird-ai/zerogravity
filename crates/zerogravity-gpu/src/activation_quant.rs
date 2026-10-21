//! Source GGML dynamic Q8 activation preparation on a physical device.
//!
//! H14 is source-bit qualified on AMD/NVIDIA Vulkan. This module does not submit work,
//! admit memory, perform readbacks, publish state or provide a CPU fallback.
//! Its seven source-rounded stages consume resident activations and immutable
//! source scales/biases, producing the SAME layout as `host_descriptor::dynamic`.
//! Q8_K may omit the two sum-only stages. Every dependent stage must be a
//! separate physical dispatch; no host-calculated dynamic scale/code is input.
//!
//! Source equations are the existing Rust `weights::QuantizedActivation`,
//! `host_descriptor::StaticSource::dynamic`, `q4_q8::prepare_source` and
//! `mixed_ops::prepare`. There is no external implementation dependency.

use crate::source_float::{round_source, widen_source};
use cubecl::prelude::*;
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result;

/// Parameter words for each source activation block: inverse, unrounded d,
/// execution scale (half-roundtripped d, or Q8_K d), and source Q8_1 sum scale.
pub(crate) const PARAMETER_WORDS: usize = 4;
pub(crate) const INPUT_NONFINITE: u32 = 1;
pub(crate) const SCALE_NONFINITE: u32 = 2;
pub(crate) const SUM_NONFINITE: u32 = 4;
pub(crate) const COMPOSITE_NONFINITE: u32 = 8;

/// Bounded host metadata only. All actual allocations must be included in the
/// caller's existing physical reservation/retention and completion receipts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ActivationLayout {
    pub columns: usize,
    pub rows: usize,
    pub block_width: usize,
    pub blocks: usize,
    pub composite_count: usize,
    pub signed_scale: bool,
    pub with_bias: bool,
    /// Includes parameters, scaled inputs, codes, both status arrays and full
    /// composite output buffers. Excludes resident input/static source buffers.
    /// The no-bias case includes one valid dummy offset element for the ABI.
    pub workspace_bytes: usize,
}

impl ActivationLayout {
    pub fn new(kind: QuantKind, columns: usize, rows: usize) -> Result<Self> {
        let (block_width, signed_scale, with_bias) = match kind {
            QuantKind::Q4_0 | QuantKind::Q8_0 => (32, false, false),
            QuantKind::Q4_1 => (32, false, true),
            QuantKind::Q6K => (256, true, false),
            _ => {
                return Err(Error::Unsupported(
                    "device activation encoding is Q8_0/Q8_1/Q8_K only".into(),
                ));
            }
        };
        if rows == 0
            || columns == 0
            || !columns.is_multiple_of(block_width)
            || u32::try_from(rows).is_err()
            || u32::try_from(columns).is_err()
        {
            return Err(Error::Malformed(
                "device activation preparation geometry invalid".into(),
            ));
        }
        let blocks = columns / block_width;
        let composite_count = rows
            .checked_mul(blocks)
            .ok_or_else(|| Error::Limit("device composite scale count overflow".into()))?;
        if u32::try_from(composite_count).is_err() {
            return Err(Error::Limit(
                "device composite buffer exceeds shader index domain".into(),
            ));
        }
        let offset_count = if with_bias { composite_count } else { 1 };
        let words = blocks
            .checked_mul(PARAMETER_WORDS + 1)
            .and_then(|n| columns.checked_mul(2).and_then(|m| n.checked_add(m)))
            .and_then(|n| n.checked_add(composite_count))
            .and_then(|n| n.checked_add(offset_count))
            .and_then(|n| n.checked_add(rows))
            .ok_or_else(|| Error::Limit("device activation workspace overflow".into()))?;
        let workspace_bytes = words
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("device activation workspace byte overflow".into()))?;
        Ok(Self {
            columns,
            rows,
            block_width,
            blocks,
            composite_count,
            signed_scale,
            with_bias,
            workspace_bytes,
        })
    }
}

/// Normalize a finite, nonzero binary32 magnitude using integer operations.
/// A subnormal's numerical bits must survive hardware FTZ/DAZ conversion.
#[cube]
fn normalized(magnitude: u32) -> (u64, i32) {
    let exponent = (magnitude >> 23u32) as i32;
    let mut m = magnitude & 0x007fffffu32;
    let mut e = exponent - 127i32;
    if exponent != 0i32 {
        m |= 0x00800000u32;
    } else {
        e = -126i32;
        for _iteration in 0..23usize {
            if m < 0x00800000u32 {
                m <<= 1u32;
                e -= 1i32;
            }
        }
    }
    (m as u64, e)
}

/// Exact integer quotient/remainder rounding to nearest-even. The caller's
/// operands are at most 48 and 25 bits, so no product/remainder can overflow.
#[cube]
fn rounded_ratio(n: u64, d: u64) -> u64 {
    let mut q = n / d;
    let remainder = n - q * d;
    let twice = remainder << 1u64;
    if twice > d || (twice == d && (q & 1u64) != 0u64) {
        q += 1u64;
    }
    q
}

/// Source binary32 division with one RN-even rounding. No native GPU divide,
/// reciprocal estimate or FP64->FP32 double rounding is involved. The actual
/// quantizer uses finite inputs and +/-127 or 1, but infinity/zero/NaN are
/// handled because a tiny finite block can legitimately produce infinite inv.
#[cube]
pub(crate) fn divide_bits(a_bits: u32, b_bits: u32) -> u32 {
    let sign = (a_bits ^ b_bits) & 0x80000000u32;
    let a = a_bits & 0x7fffffffu32;
    let b = b_bits & 0x7fffffffu32;
    let mut encoded = sign;
    if a > 0x7f800000u32
        || b > 0x7f800000u32
        || (a == 0u32 && b == 0u32)
        || (a == 0x7f800000u32 && b == 0x7f800000u32)
    {
        encoded = 0x7fc00000u32;
    } else if a == 0x7f800000u32 || b == 0u32 {
        encoded |= 0x7f800000u32;
    } else if a != 0u32 && b != 0x7f800000u32 {
        let (mut ma, ea) = normalized(a);
        let (mb, eb) = normalized(b);
        let mut e = ea - eb;
        if ma < mb {
            ma <<= 1u64;
            e -= 1i32;
        }
        if e > 127i32 {
            encoded |= 0x7f800000u32;
        } else if e >= -126i32 {
            let mut q = rounded_ratio(ma << 23u64, mb);
            if q == 0x01000000u64 {
                q >>= 1u64;
                e += 1i32;
            }
            if e > 127i32 {
                encoded |= 0x7f800000u32;
            } else {
                encoded |= (((e + 127i32) as u32) << 23u32) | ((q as u32) & 0x007fffffu32);
            }
        } else if e >= -150i32 {
            // A rounded subnormal can carry directly into the minimum normal.
            let shift = e + 149i32;
            let mut n = ma;
            let mut d = mb;
            if shift >= 0i32 {
                n <<= shift as u64;
            } else {
                d <<= 1u64;
            }
            encoded |= rounded_ratio(n, d) as u32;
        }
    }
    encoded
}

#[cube]
pub(crate) fn divide_rne(numerator: f32, denominator: f32) -> f32 {
    f32::reinterpret(divide_bits(
        u32::reinterpret(numerator),
        u32::reinterpret(denominator),
    ))
}

/// Exact source F32->F16 RN-even->F32 bit roundtrip, including signed zero,
/// half subnormals, normal carry and overflow. No shader Float16 is required.
#[cube]
pub(crate) fn half_roundtrip_bits(bits: u32) -> u32 {
    let sign = bits & 0x80000000u32;
    let exponent = ((bits >> 23u32) & 255u32) as i32;
    let fraction = bits & 0x007fffffu32;
    let mut half = 0u32;
    if exponent == 255i32 {
        half = 0x7c00u32;
        if fraction != 0u32 {
            half |= (fraction >> 13u32) | 0x0200u32;
        }
    } else {
        let half_exp = exponent - 112i32;
        if half_exp >= 31i32 {
            half = 0x7c00u32;
        } else if half_exp <= 0i32 {
            if half_exp >= -10i32 {
                let m = fraction | 0x00800000u32;
                let shift = (14i32 - half_exp) as u32;
                half = (m + ((1u32 << (shift - 1u32)) - 1u32) + ((m >> shift) & 1u32)) >> shift;
            }
        } else {
            let rounded = fraction + 0x0fffu32 + ((fraction >> 13u32) & 1u32);
            if rounded >= 0x00800000u32 {
                half = ((half_exp + 1i32) as u32) << 10u32;
            } else {
                half = ((half_exp as u32) << 10u32) | (rounded >> 13u32);
            }
        }
    }
    let he = (half >> 10u32) & 31u32;
    let mut hm = half & 0x03ffu32;
    let mut encoded = sign;
    if he == 0u32 {
        if hm != 0u32 {
            let mut e = 113u32;
            for _iteration in 0..10usize {
                if (hm & 0x0400u32) == 0u32 {
                    hm <<= 1u32;
                    e -= 1u32;
                }
            }
            encoded |= (e << 23u32) | ((hm & 0x03ffu32) << 13u32);
        }
    } else if he == 31u32 {
        encoded |= 0x7f800000u32 | (hm << 13u32);
    } else {
        encoded |= ((he + 112u32) << 23u32) | (hm << 13u32);
    }
    encoded
}

#[cube]
pub(crate) fn half_roundtrip(value: f32) -> f32 {
    f32::reinterpret(half_roundtrip_bits(u32::reinterpret(value)))
}

/// Source round_ties_even().clamp(-128,127) as i32, from stored source F32.
/// Rust's saturating NaN-to-int cast produces zero; infinities clamp to an end.
#[cube]
pub(crate) fn quantized_code_bits(bits: u32) -> i32 {
    let magnitude = bits & 0x7fffffffu32;
    let negative = (bits & 0x80000000u32) != 0u32;
    let mut code = 0i32;
    if magnitude <= 0x7f800000u32 && magnitude > 0x3f000000u32 {
        if magnitude >= 0x43000000u32 {
            code = 127i32;
            if negative {
                code = -128i32;
            }
        } else {
            let exponent = (magnitude >> 23u32) as i32 - 127i32;
            let shift = (23i32 - exponent) as u32;
            let m = (magnitude & 0x007fffffu32) | 0x00800000u32;
            let mut q = m >> shift;
            let remainder = m & ((1u32 << shift) - 1u32);
            let halfway = 1u32 << (shift - 1u32);
            if remainder > halfway || (remainder == halfway && (q & 1u32) != 0u32) {
                q += 1u32;
            }
            if negative {
                code = -(q as i32);
            } else {
                code = q as i32;
                if code > 127i32 {
                    code = 127i32;
                }
            }
        }
    }
    code
}

#[cube]
pub(crate) fn quantized_code(value: f32) -> i32 {
    quantized_code_bits(u32::reinterpret(value))
}

/// Launch 1: strict ordered source maximum and exact source divisions.
/// Integer magnitude comparisons preserve subnormals under FTZ and choose the
/// FIRST signed max on equal magnitude, as the Q8_K source loop requires.
#[cube(launch_unchecked)]
pub(crate) fn block_parameters(
    input: &[f32],
    parameters: &mut [f32],
    status: &mut [u32],
    #[comptime] blocks: usize,
    #[comptime] width: usize,
    #[comptime] signed_scale: bool,
) {
    let block = ABSOLUTE_POS;
    if block < blocks {
        let mut largest = 0u32;
        let mut selected = 0u32;
        let mut flags = 0u32;
        for lane in 0..width {
            let bits = u32::reinterpret(input[block * width + lane]);
            let magnitude = bits & 0x7fffffffu32;
            if magnitude >= 0x7f800000u32 {
                flags |= INPUT_NONFINITE;
            } else if magnitude > largest {
                largest = magnitude;
                selected = bits;
            }
        }
        let mut inverse = 0.0f32;
        let mut d = 0.0f32;
        if largest != 0u32 {
            if signed_scale {
                inverse = divide_rne(-127.0f32, f32::reinterpret(selected));
                d = divide_rne(1.0f32, inverse);
            } else {
                inverse = divide_rne(127.0f32, f32::reinterpret(largest));
                d = divide_rne(f32::reinterpret(largest), 127.0f32);
            }
        }
        parameters[block * 4] = inverse;
        parameters[block * 4 + 1] = d;
        status[block] = flags;
    }
}

/// Launch 2: read source-rounded d physically, then source half roundtrip.
#[cube(launch_unchecked)]
pub(crate) fn block_scales(
    parameters: &mut [f32],
    status: &mut [u32],
    #[comptime] blocks: usize,
    #[comptime] signed_scale: bool,
) {
    let block = ABSOLUTE_POS;
    if block < blocks {
        let mut d = parameters[block * 4 + 1];
        if !signed_scale {
            d = half_roundtrip(d);
        }
        parameters[block * 4 + 2] = d;
        parameters[block * 4 + 3] = 0.0f32;
        if (u32::reinterpret(d) & 0x7fffffffu32) >= 0x7f800000u32 {
            status[block] |= SCALE_NONFINITE;
        }
    }
}

/// Launch 3: store the source F32 x*inv BEFORE nearest-even integer coding.
#[cube(launch_unchecked)]
pub(crate) fn scaled_inputs(
    input: &[f32],
    parameters: &[f32],
    scaled: &mut [f32],
    #[comptime] columns: usize,
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    if i < columns {
        let inverse = parameters[(i / width) * 4];
        let sign = (u32::reinterpret(input[i]) ^ u32::reinterpret(inverse)) & 0x80000000u32;
        scaled[i] = round_source(widen_source(input[i]) * widen_source(inverse), sign);
    }
}

/// Launch 4: complete source byte codes stored as existing signed i32 words.
#[cube(launch_unchecked)]
pub(crate) fn codes(scaled: &[f32], output: &mut [i32], #[comptime] columns: usize) {
    let i = ABSOLUTE_POS;
    if i < columns {
        output[i] = quantized_code(scaled[i]);
    }
}

/// Launch 5 (small blocks only): exact integer sum, separately rounded F32
/// sum(q)*UNROUNDED d. Q8_1 does not multiply by the half-rounded scale.
#[cube(launch_unchecked)]
pub(crate) fn sum_products(codes: &[i32], parameters: &mut [f32], #[comptime] blocks: usize) {
    let block = ABSOLUTE_POS;
    if block < blocks {
        let mut sum = 0i32;
        for lane in 0..32usize {
            sum += codes[block * 32 + lane];
        }
        let a = f32::cast_from(sum);
        let b = parameters[block * 4 + 1];
        let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
        parameters[block * 4 + 3] = round_source(widen_source(a) * widen_source(b), sign);
    }
}

/// Launch 6 (small blocks only): read the rounded sum product before F16.
/// The sum may be infinite for codecs that never consume it; only Q4_1 marks
/// that as an error. The authoritative host descriptor likewise ignores the
/// unused small-block sum when no source bias exists.
#[cube(launch_unchecked)]
pub(crate) fn sum_halves(
    parameters: &mut [f32],
    status: &mut [u32],
    #[comptime] blocks: usize,
    #[comptime] with_bias: bool,
) {
    let block = ABSOLUTE_POS;
    if block < blocks {
        let sum = half_roundtrip(parameters[block * 4 + 3]);
        parameters[block * 4 + 3] = sum;
        if with_bias && (u32::reinterpret(sum) & 0x7fffffffu32) >= 0x7f800000u32 {
            status[block] |= SUM_NONFINITE;
        }
    }
}

/// Launch 7: materialize existing per-source-block F32 composite scales and
/// optional Q4_1 offsets from IMMUTABLE static arrays. One invocation per row
/// emits one status, avoiding unchecked atomics or an N*blocks status buffer.
/// No bias/offset element is read/written when with_bias=false; bind a valid
/// single dummy element for each unused buffer argument.
#[cube(launch_unchecked)]
pub(crate) fn composites(
    weight_scales: &[f32],
    biases: &[f32],
    parameters: &[f32],
    scales: &mut [f32],
    offsets: &mut [f32],
    status: &mut [u32],
    #[comptime] blocks: usize,
    #[comptime] rows: usize,
    #[comptime] with_bias: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut flags = 0u32;
        for block in 0..blocks {
            let index = row * blocks + block;
            let a = weight_scales[index];
            let b = parameters[block * 4 + 2];
            let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
            let scale = round_source(widen_source(a) * widen_source(b), sign);
            scales[index] = scale;
            if (u32::reinterpret(scale) & 0x7fffffffu32) >= 0x7f800000u32 {
                flags |= COMPOSITE_NONFINITE;
            }
            if with_bias {
                let a = biases[index];
                let b = parameters[block * 4 + 3];
                let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
                let offset = round_source(widen_source(a) * widen_source(b), sign);
                offsets[index] = offset;
                if (u32::reinterpret(offset) & 0x7fffffffu32) >= 0x7f800000u32 {
                    flags |= COMPOSITE_NONFINITE;
                }
            }
        }
        status[row] = flags;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_artifacts::quant::{f16_to_f32, f32_to_f16_rne};

    #[test]
    fn integer_division_matches_source_binary32_rounding() {
        let boundary = [
            0, 1, 2, 3, 0x003fffff, 0x00400000, 0x007fffff, 0x00800000, 0x00800001, 0x3effffff,
            0x3f000000, 0x3f000001, 0x3f7fffff, 0x3f800000, 0x3f800001, 0x42fdffff, 0x42fe0000,
            0x42fe0001, 0x7f7ffffe, 0x7f7fffff, 0x7f800000,
        ];
        for a in boundary {
            for b in boundary {
                for sign in [0u32, 0x80000000] {
                    let a = f32::from_bits(a | sign);
                    let b = f32::from_bits(b);
                    let expected = a / b;
                    let actual = f32::from_bits(divide_bits(a.to_bits(), b.to_bits()));
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(actual.to_bits(), expected.to_bits(), "{a:?}/{b:?}");
                    }
                }
            }
        }
        let mut rng = 0x6135_d6f4_9eb2_741du64;
        for _ in 0..1_000_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let a = f32::from_bits(rng as u32);
            let b = f32::from_bits((rng >> 32) as u32);
            let expected = a / b;
            let actual = f32::from_bits(divide_bits(a.to_bits(), b.to_bits()));
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{a:?}/{b:?}");
            }
            if a.is_finite() {
                for numerator in [127.0f32, -127.0, 1.0] {
                    let expected = numerator / a;
                    let actual = f32::from_bits(divide_bits(numerator.to_bits(), a.to_bits()));
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(actual.to_bits(), expected.to_bits(), "{numerator}/{a:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn integer_half_roundtrip_matches_source_at_all_half_boundaries() {
        for word in 0..=u16::MAX {
            let center = f16_to_f32(word);
            let bits = center.to_bits();
            for offset in [0u32, 1, 0x0fff, 0x1000, 0x1001, 0x1fff] {
                for bits in [bits.wrapping_add(offset), bits.wrapping_sub(offset)] {
                    let value = f32::from_bits(bits);
                    let expected = f16_to_f32(f32_to_f16_rne(value));
                    let actual = f32::from_bits(half_roundtrip_bits(value.to_bits()));
                    assert_eq!(actual.to_bits(), expected.to_bits(), "input={bits:08x}");
                }
            }
        }
    }

    #[test]
    fn integer_codes_match_source_ties_and_saturating_cast() {
        for value in -512..512 {
            let center = value as f32 * 0.5;
            let bits = center.to_bits();
            for bits in [bits.wrapping_sub(1), bits, bits.wrapping_add(1)] {
                let input = f32::from_bits(bits);
                let expected = input.round_ties_even().clamp(-128.0, 127.0) as i32;
                assert_eq!(
                    quantized_code_bits(input.to_bits()),
                    expected,
                    "input={bits:08x}"
                );
            }
        }
        for bits in [
            0, 1, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000, 0x7fc00000,
        ] {
            for sign in [0, 0x80000000] {
                let input = f32::from_bits(bits | sign);
                assert_eq!(
                    quantized_code_bits(input.to_bits()),
                    input.round_ties_even().clamp(-128.0, 127.0) as i32
                );
            }
        }
    }

    #[test]
    fn layout_refuses_bad_geometry_and_accounts_every_workspace_buffer() {
        for (kind, columns, rows) in [
            (QuantKind::Q4_0, 1024usize, 2048usize),
            (QuantKind::Q4_1, 1024, 2048),
            (QuantKind::Q8_0, 2048, 128),
            (QuantKind::Q6K, 2048, 4096),
        ] {
            let layout = ActivationLayout::new(kind, columns, rows).unwrap();
            assert_eq!(layout.columns, columns);
            assert_eq!(layout.rows, rows);
            assert_eq!(layout.blocks * layout.block_width, columns);
            assert_eq!(layout.composite_count, rows * layout.blocks);
            assert_eq!(layout.signed_scale, kind == QuantKind::Q6K);
            assert_eq!(layout.with_bias, kind == QuantKind::Q4_1);
            let offsets = if layout.with_bias {
                rows * layout.blocks
            } else {
                1
            };
            assert_eq!(
                layout.workspace_bytes,
                4 * (layout.blocks * 5 + columns * 2 + rows * layout.blocks + offsets + rows)
            );
        }
        assert!(ActivationLayout::new(QuantKind::Q4_0, 0, 1).is_err());
        assert!(ActivationLayout::new(QuantKind::Q4_0, 32, 0).is_err());
        assert!(ActivationLayout::new(QuantKind::Q6K, 32, 1).is_err());
        assert!(ActivationLayout::new(QuantKind::F32, 32, 1).is_err());
        assert!(ActivationLayout::new(QuantKind::Q4_1, usize::MAX, 1).is_err());
        assert!(ActivationLayout::new(QuantKind::Q4_1, 64, u32::MAX as usize).is_err());
    }
}
