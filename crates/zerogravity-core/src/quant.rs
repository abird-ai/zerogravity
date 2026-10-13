//! Source quantization codecs, independent of model family and device layout.
//! A recognized source encoding is not necessarily an executable model.

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum QuantKind {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q8_0,
    Q6K,
    Q4K,
}

impl QuantKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
        }
    }

    /// (logical elements per block, physical bytes per block).
    pub fn block(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q6K => (256, 210),
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
        }
    }

    pub fn byte_len(self, elements: u64) -> Result<u64> {
        let (block, bytes) = self.block();
        if !elements.is_multiple_of(block) {
            return Err(Error::Malformed(format!(
                "{} requires an element count divisible by {block}",
                self.name()
            )));
        }
        (elements / block)
            .checked_mul(bytes)
            .ok_or_else(|| Error::Limit("tensor byte size overflow".into()))
    }

    pub fn from_ggml(code: u32) -> Result<Self> {
        match code {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            14 => Ok(Self::Q6K),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4K),
            30 => Ok(Self::Bf16),
            other => Err(Error::Unsupported(format!(
                "GGUF ggml tensor type {other}: no validated byte codec"
            ))),
        }
    }

    pub fn from_safetensors(code: &str) -> Result<Self> {
        match code {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            other => Err(Error::Unsupported(format!(
                "safetensors dtype {other}: no validated tensor codec"
            ))),
        }
    }
}
