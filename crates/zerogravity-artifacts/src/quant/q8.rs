//! Exact scalar Q8_0 source decoding.

use super::f16_to_f32;
use crate::{Error, Result};

pub(super) fn decode_q8_0(data: &[u8], out: &mut Vec<f32>) -> Result<()> {
    for b in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        if !scale.is_finite() {
            return Err(Error::Malformed("nonfinite Q8_0 scale".into()));
        }
        out.extend(b[2..].iter().map(|v| (*v as i8 as f32) * scale));
    }
    Ok(())
}
