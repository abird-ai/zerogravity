//! Stored bytes, not activation arithmetic or execution eligibility.
use crate::{Result, quant::QuantKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarDType {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    Q4_0,
    Q4_1,
    Q4K,
    Q6K,
    Q8_0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredTensorEncoding {
    Scalar(ScalarDType),
    Quantized(QuantScheme),
}

impl From<QuantKind> for StoredTensorEncoding {
    fn from(value: QuantKind) -> Self {
        match value {
            QuantKind::F32 => Self::Scalar(ScalarDType::F32),
            QuantKind::F16 => Self::Scalar(ScalarDType::F16),
            QuantKind::Bf16 => Self::Scalar(ScalarDType::Bf16),
            QuantKind::Q4_0 => Self::Quantized(QuantScheme::Q4_0),
            QuantKind::Q4_1 => Self::Quantized(QuantScheme::Q4_1),
            QuantKind::Q4K => Self::Quantized(QuantScheme::Q4K),
            QuantKind::Q6K => Self::Quantized(QuantScheme::Q6K),
            QuantKind::Q8_0 => Self::Quantized(QuantScheme::Q8_0),
        }
    }
}
impl From<StoredTensorEncoding> for QuantKind {
    fn from(value: StoredTensorEncoding) -> Self {
        match value {
            StoredTensorEncoding::Scalar(ScalarDType::F32) => Self::F32,
            StoredTensorEncoding::Scalar(ScalarDType::F16) => Self::F16,
            StoredTensorEncoding::Scalar(ScalarDType::Bf16) => Self::Bf16,
            StoredTensorEncoding::Quantized(QuantScheme::Q4_0) => Self::Q4_0,
            StoredTensorEncoding::Quantized(QuantScheme::Q4_1) => Self::Q4_1,
            StoredTensorEncoding::Quantized(QuantScheme::Q4K) => Self::Q4K,
            StoredTensorEncoding::Quantized(QuantScheme::Q6K) => Self::Q6K,
            StoredTensorEncoding::Quantized(QuantScheme::Q8_0) => Self::Q8_0,
        }
    }
}
impl StoredTensorEncoding {
    // S1 moves the codec identity; until then retain the one existing mapping.
    pub fn byte_len(self, elements: u64) -> Result<u64> {
        QuantKind::from(self).byte_len(elements)
    }
    pub fn from_ggml(code: u32) -> Result<Self> {
        QuantKind::from_ggml(code).map(Self::from)
    }
    pub fn from_safetensors(code: &str) -> Result<Self> {
        QuantKind::from_safetensors(code).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fabric_encoding_preserves_original_mapping_lengths_and_refusals() {
        for code in [0, 1, 2, 3, 8, 12, 14, 30] {
            let original = QuantKind::from_ggml(code).unwrap();
            let encoding = StoredTensorEncoding::from_ggml(code).unwrap();
            assert_eq!(QuantKind::from(encoding), original);
            assert_eq!(
                encoding.byte_len(256).unwrap(),
                original.byte_len(256).unwrap()
            );
        }
        assert!(StoredTensorEncoding::from_ggml(10).is_err());
        assert!(StoredTensorEncoding::from_safetensors("NVFP4").is_err());
        assert!(
            StoredTensorEncoding::Quantized(QuantScheme::Q4K)
                .byte_len(1)
                .is_err()
        );
        assert!(
            StoredTensorEncoding::Scalar(ScalarDType::F32)
                .byte_len(u64::MAX)
                .is_err()
        );
    }
}
