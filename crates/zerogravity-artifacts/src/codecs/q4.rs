//! Exact scalar Q4_0, Q4_1 and Q4_K source decoding.

use super::f16_to_f32;
use crate::{Error, Result};

pub(super) fn decode_q4_0(data: &[u8], out: &mut Vec<f32>) -> Result<()> {
    for b in data.chunks_exact(18) {
        let scale = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        if !scale.is_finite() {
            return Err(Error::Malformed("nonfinite Q4_0 scale".into()));
        }
        for v in &b[2..] {
            out.push(f32::from(v & 0x0f) * scale - 8.0 * scale);
        }
        for v in &b[2..] {
            out.push(f32::from(v >> 4) * scale - 8.0 * scale);
        }
    }
    Ok(())
}

pub(super) fn decode_q4_1(data: &[u8], out: &mut Vec<f32>) -> Result<()> {
    for b in data.chunks_exact(20) {
        let scale = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let min = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        if !scale.is_finite() || !min.is_finite() {
            return Err(Error::Malformed("nonfinite Q4_1 scale/offset".into()));
        }
        for v in &b[4..] {
            out.push(f32::from(v & 15).mul_add(scale, min));
        }
        for v in &b[4..] {
            out.push(f32::from(v >> 4).mul_add(scale, min));
        }
    }
    Ok(())
}

pub(super) fn decode_q4_k(data: &[u8], out: &mut Vec<f32>) -> Result<()> {
    // 256 elements in 144 bytes: 2 F16 base scales, 12 scale/min
    // bytes with 6-bit packed values, and 128 nibble-packed weights.
    for block in data.chunks_exact(144) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        if !d.is_finite() || !dmin.is_finite() {
            return Err(Error::Malformed("nonfinite Q4_K scale/min".into()));
        }
        let packed = &block[4..16];
        let values = &block[16..144];
        for group in 0..8 {
            let (scale, min) = if group < 4 {
                (packed[group] & 63, packed[group + 4] & 63)
            } else {
                (
                    (packed[group + 4] & 15) | ((packed[group - 4] >> 6) << 4),
                    (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4),
                )
            };
            let scale = d * f32::from(scale);
            let min = dmin * f32::from(min);
            let start = (group / 2) * 32;
            let high_nibble = group % 2 == 1;
            for &byte in &values[start..start + 32] {
                let value = if high_nibble { byte >> 4 } else { byte & 15 };
                out.push(scale * f32::from(value) - min);
            }
        }
    }
    Ok(())
}
