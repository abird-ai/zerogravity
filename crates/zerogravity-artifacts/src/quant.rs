//! Exact scalar source encoding codecs.
//! Integer family homes: `quant/README.md`; BF16: `precision/bf16/README.md`.

mod q4;
mod q6;
mod q8;

use crate::{Error, Result};
pub use zerogravity_core::quant::QuantKind;

/// Decodes IEEE binary16 exactly into f32, including subnormals and NaN/Inf.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits >> 10) & 0x1f;
    let mant = u32::from(bits & 0x03ff);
    let word = match exp {
        0 if mant == 0 => sign,
        0 => {
            let mut m = mant;
            let mut e = 127i32 - 14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (mant << 13),
    };
    f32::from_bits(word)
}

/// Reference decoding, intended for correctness fixtures rather than speed.
/// Q4_0 has 16 low-nibble values followed by 16 high-nibble values per block.
pub fn dequantize(kind: QuantKind, data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let expected = kind.byte_len(elements as u64)?;
    if data.len() as u64 != expected {
        return Err(Error::Malformed(format!(
            "{} expected {expected} bytes, got {}",
            kind.name(),
            data.len()
        )));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(elements)
        .map_err(|_| Error::Limit("reference tensor allocation failed".into()))?;
    match kind {
        QuantKind::F32 => {
            for b in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
        }
        QuantKind::F16 | QuantKind::Bf16 => {
            for b in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                out.push(if kind == QuantKind::F16 {
                    f16_to_f32(bits)
                } else {
                    f32::from_bits(u32::from(bits) << 16)
                });
            }
        }
        QuantKind::Q8_0 => q8::decode_q8_0(data, &mut out)?,
        QuantKind::Q4_0 => q4::decode_q4_0(data, &mut out)?,
        QuantKind::Q4_1 => q4::decode_q4_1(data, &mut out)?,
        QuantKind::Q6K => q6::decode_q6_k(data, &mut out)?,
        QuantKind::Q4K => q4::decode_q4_k(data, &mut out)?,
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_specials_subnormal_and_normal() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn dequantize_floats_and_blocks() {
        let raw = [0x80, 0x3f, 0x80, 0xbf];
        assert_eq!(dequantize(QuantKind::Bf16, &raw, 2).unwrap(), [1.0, -1.0]);
        let mut q8 = vec![0x00, 0x3c];
        q8.extend([0x80, 0x7f]);
        q8.extend([0u8; 30]);
        assert_eq!(
            &dequantize(QuantKind::Q8_0, &q8, 32).unwrap()[..2],
            &[-128.0, 127.0]
        );
        let mut q4 = vec![0x00, 0x3c];
        q4.extend([0x80; 16]);
        let values = dequantize(QuantKind::Q4_0, &q4, 32).unwrap();
        assert_eq!(&values[..16], &[-8.0; 16]);
        assert_eq!(&values[16..], &[0.0; 16]);
        let mut q4k = [0u8; 144];
        // d = 1, dmin = 0.5; first 32 elements: 2*3 - 4*0.5 = 4.
        q4k[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        q4k[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        q4k[4] = 2;
        q4k[8] = 4;
        q4k[16..48].fill(0x53);
        let data = dequantize(QuantKind::Q4K, &q4k, 256).unwrap();
        assert_eq!(&data[..32], &[4.0; 32]);
        assert_eq!(&data[32..64], &[0.0; 32]);
        // Packed 6-bit group high bits (group index 4).
        q4k[4] = 0b0100_0000;
        q4k[8] = 0b1000_0000;
        q4k[12] = 0b0010_0110;
        q4k[80..112].fill(0x33);
        let data = dequantize(QuantKind::Q4K, &q4k, 256).unwrap();
        assert_eq!(&data[128..160], &[49.0; 32]);
    }

    #[test]
    fn rejects_truncation_and_invalid_dimensions() {
        assert!(dequantize(QuantKind::Q8_0, &[0; 3], 32).is_err());
        assert!(QuantKind::Q4_0.byte_len(31).is_err());
        assert!(QuantKind::Q4K.byte_len(u64::MAX).is_err());
        assert!(QuantKind::from_ggml(40).is_err());
        let mut q8 = [0u8; 34];
        q8[0..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(dequantize(QuantKind::Q8_0, &q8, 32).is_err());
    }
}

/// IEEE binary32 -> binary16, nearest-even, without platform ISA dependence.
/// Used by GGUF's ephemeral quantized-activation codec to reproduce source
/// quant/block semantics on CPUs without an external inference runtime.
pub fn f32_to_f16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) as u16 & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7fffff;
    if exponent == 255 {
        return sign
            | 0x7c00
            | if fraction == 0 {
                0
            } else {
                ((fraction >> 13) as u16) | 0x0200
            };
    }
    let half_exp = exponent - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = fraction | 0x800000;
        let shift = (14 - half_exp) as u32;
        let half = (mantissa + ((1u32 << (shift - 1)) - 1) + ((mantissa >> shift) & 1)) >> shift;
        return sign | half as u16;
    }
    let rounded = fraction + 0x0fff + ((fraction >> 13) & 1);
    if rounded >= 0x800000 {
        return sign | (((half_exp + 1) as u16) << 10);
    }
    sign | ((half_exp as u16) << 10) | ((rounded >> 13) as u16)
}

#[cfg(test)]
mod half_roundtrip_tests {
    use super::*;
    #[test]
    fn round_to_even_half_roundtrip_source_activation_scales() {
        let cases = [
            (0.0f32, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-1.0, 0xbc00),
            (65504.0, 0x7bff),
            (2f32.powi(-24), 0x0001),
            (2f32.powi(-25), 0x0000),
            (1.0 + 2f32.powi(-11), 0x3c00),
            (1.0 + 3.0 * 2f32.powi(-11), 0x3c02),
            (f32::INFINITY, 0x7c00),
        ];
        for (x, raw) in cases {
            assert_eq!(f32_to_f16_rne(x), raw, "{x:?}");
        }
        assert_eq!(f32_to_f16_rne(-2f32.powi(-24)), 0x8001);
        assert!(f16_to_f32(f32_to_f16_rne(f32::NAN)).is_nan());
    }
}
