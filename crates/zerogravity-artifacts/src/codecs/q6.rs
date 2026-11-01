//! Exact scalar Q6_K source decoding.

use super::f16_to_f32;
use crate::{Error, Result};

pub(super) fn decode_q6_k(data: &[u8], out: &mut Vec<f32>) -> Result<()> {
    for b in data.chunks_exact(210) {
        let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
        if !d.is_finite() {
            return Err(Error::Malformed("nonfinite Q6_K superblock scale".into()));
        }
        for half in 0..2 {
            let low = &b[half * 64..half * 64 + 64];
            let high = &b[128 + half * 32..128 + half * 32 + 32];
            let scales = &b[192 + half * 8..192 + half * 8 + 8];
            let mut decoded = [0f32; 128];
            for lane in 0..32 {
                let bits = high[lane];
                for group in 0..4 {
                    let nibble = if group < 2 {
                        low[lane + group * 32] & 15
                    } else {
                        low[lane + (group - 2) * 32] >> 4
                    };
                    let val = (nibble | (((bits >> (2 * group)) & 3) << 4)) as i32 - 32;
                    let scale = scales[group * 2 + lane / 16] as i8 as f32;
                    decoded[group * 32 + lane] = d * scale * val as f32;
                }
            }
            out.extend(decoded);
        }
    }
    Ok(())
}
