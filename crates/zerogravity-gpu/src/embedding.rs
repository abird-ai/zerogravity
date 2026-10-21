//! Draft source-byte embedding gather for Q8_0 and Q6_K GGUF tensors.
//!
//! No row is dequantized or copied on the host during token execution. The
//! runtime token is one resident u32, permitting direct GPU sampler binding.
//! Caller-owned admission, source identity, completion and quarantine remain
//! mandatory; this module neither submits work nor publishes model state.
use crate::source_float::{round_source, widen_source};
use cubecl::prelude::*;
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result;

pub(crate) const TOKEN_OUT_OF_RANGE: u32 = 1;
pub(crate) const NONFINITE_SCALE: u32 = 2;
pub(crate) const NONFINITE_OUTPUT: u32 = 4;

/// Shape/admission metadata only; no model-name-specific behavior or source
/// allocation is introduced. Each output channel has one complete status word.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbeddingLayout {
    pub source_bytes: usize,
    /// Runtime token, FP32 output, and per-output u32 status. Excludes source.
    pub workspace_bytes: usize,
    pub q6_k: bool,
}

impl EmbeddingLayout {
    pub fn new(kind: QuantKind, vocabulary: usize, width: usize) -> Result<Self> {
        let (block_width, block_bytes, q6_k) = match kind {
            QuantKind::Q8_0 => (32, 34, false),
            QuantKind::Q6K => (256, 210, true),
            _ => {
                return Err(Error::Unsupported(
                    "source embedding GPU codec is Q8_0/Q6_K only".into(),
                ));
            }
        };
        if vocabulary == 0
            || width == 0
            || !width.is_multiple_of(block_width)
            || u32::try_from(vocabulary).is_err()
            || u32::try_from(width).is_err()
        {
            return Err(Error::Malformed(
                "source embedding GPU geometry invalid".into(),
            ));
        }
        let blocks_per_row = width / block_width;
        let row_bytes = blocks_per_row
            .checked_mul(block_bytes)
            .ok_or_else(|| Error::Limit("source embedding row byte overflow".into()))?;
        let source_bytes = row_bytes
            .checked_mul(vocabulary)
            .ok_or_else(|| Error::Limit("source embedding byte overflow".into()))?;
        if u32::try_from(source_bytes).is_err() {
            return Err(Error::Limit(
                "source embedding exceeds shader index domain".into(),
            ));
        }
        let workspace_bytes = width
            .checked_mul(8)
            .and_then(|n| n.checked_add(4))
            .ok_or_else(|| Error::Limit("source embedding workspace overflow".into()))?;
        Ok(Self {
            source_bytes,
            workspace_bytes,
            q6_k,
        })
    }
}

/// Integer-only IEEE binary16 expansion, identical to quant::f16_to_f32.
/// Half subnormals become normal binary32 values; both zero signs survive.
#[cube]
fn half_to_float(bits: u32) -> f32 {
    let sign = (bits & 0x8000u32) << 16u32;
    let exponent = (bits >> 10u32) & 31u32;
    let mantissa = bits & 0x03ffu32;
    let mut word = sign;
    if exponent == 0u32 {
        if mantissa != 0u32 {
            let mut m = mantissa;
            let mut e = 113i32;
            while (m & 0x0400u32) == 0u32 {
                m <<= 1u32;
                e -= 1i32;
            }
            word |= (u32::cast_from(e) << 23u32) | ((m & 0x03ffu32) << 13u32);
        }
    } else if exponent == 31u32 {
        word |= 0x7f800000u32 | (mantissa << 13u32);
    } else {
        word |= ((exponent + 112u32) << 23u32) | (mantissa << 13u32);
    }
    f32::reinterpret(word)
}

#[cube]
fn signed_byte(byte: u8) -> i32 {
    let value = u32::cast_from(byte);
    let mut signed = i32::cast_from(value);
    if value >= 128u32 {
        signed -= 256i32;
    }
    signed
}

/// One launch reads the runtime token and original compressed row, writes each
/// decoded FP32 value and its status. Any nonzero status rejects the WHOLE
/// region; zeroed invalid values are never accepted as model answers.
///
/// Bind exactly `layout.source_bytes` u8 source elements, one u32 token,
/// `layout.width` FP32 outputs and `layout.width` u32 statuses. Host metadata
/// must first pass EmbeddingLayout::new; every handle participates in existing
/// physical admission/retention. The token itself is checked on the device.
#[cube(launch_unchecked)]
pub(crate) fn gather_quantized(
    packed: &[u8],
    token: &[u32],
    output: &mut [f32],
    status: &mut [u32],
    #[comptime] vocabulary: usize,
    #[comptime] width: usize,
    #[comptime] q6_k: bool,
) {
    let i = ABSOLUTE_POS;
    if i < width {
        let mut flags = 0u32;
        let mut decoded = 0.0f32;
        if token[0] >= u32::cast_from(vocabulary) {
            flags = TOKEN_OUT_OF_RANGE;
        } else {
            let row = usize::cast_from(token[0]);
            let (scale_bits, code, local_scale) = if q6_k {
                let base = (row * (width / 256) + i / 256) * 210;
                let scale_bits = u32::cast_from(packed[base + 208])
                    | (u32::cast_from(packed[base + 209]) << 8u32);
                let half = (i % 256) / 128;
                let group = (i % 128) / 32;
                let lane = i % 32;
                let low = u32::cast_from(packed[base + half * 64 + (group % 2) * 32 + lane]);
                let high = u32::cast_from(packed[base + 128 + half * 32 + lane]);
                let mut nibble = low & 15u32;
                if group >= 2usize {
                    nibble = low >> 4u32;
                }
                let code =
                    i32::cast_from(nibble | (((high >> u32::cast_from(2 * group)) & 3u32) << 4u32))
                        - 32i32;
                let local_scale =
                    signed_byte(packed[base + 192 + half * 8 + group * 2 + lane / 16]);
                (scale_bits, code, local_scale)
            } else {
                let base = (row * (width / 32) + i / 32) * 34;
                let scale_bits =
                    u32::cast_from(packed[base]) | (u32::cast_from(packed[base + 1]) << 8u32);
                (scale_bits, signed_byte(packed[base + 2 + i % 32]), 1i32)
            };
            if (scale_bits & 0x7c00u32) == 0x7c00u32 {
                flags = NONFINITE_SCALE;
            } else {
                let d = half_to_float(scale_bits);
                let scale = f32::cast_from(local_scale);
                let value = f32::cast_from(code);
                let sign =
                    (u32::reinterpret(d) ^ u32::reinterpret(scale) ^ u32::reinterpret(value))
                        & 0x80000000u32;
                // Q6_K first evaluates d * signed block scale in source F32.
                // This product is exact: binary16 has at most 11 significant
                // bits, and i8 adds at most 7. No physical intermediate store
                // is needed before the final source-rounded code product.
                decoded = round_source(
                    widen_source(d) * widen_source(scale) * widen_source(value),
                    sign,
                );
                if (u32::reinterpret(decoded) & 0x7f800000u32) == 0x7f800000u32 {
                    flags = NONFINITE_OUTPUT;
                    decoded = 0.0f32;
                }
            }
        }
        output[i] = decoded;
        status[i] = flags;
    }
}
