//! Stored bytes, not activation arithmetic or execution eligibility.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScalarDType {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum QuantScheme {
    Q4_0,
    Q4_1,
    Q8_0,
    Q6K,
    Q4K,
}

// Declaration order preserves source encoding ordering in inventories.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StoredTensorEncoding {
    Scalar(ScalarDType),
    Quantized(QuantScheme),
    /// Original integer buffers (e.g. BERT position IDs), not activation
    /// precision and never a floating-point or quantized projection codec.
    I64,
}

impl std::fmt::Debug for StoredTensorEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Preserve source diagnostics independently of the scalar/quant split.
        match self {
            Self::Scalar(dtype) => std::fmt::Debug::fmt(dtype, f),
            Self::Quantized(scheme) => std::fmt::Debug::fmt(scheme, f),
            Self::I64 => f.write_str("I64"),
        }
    }
}

impl StoredTensorEncoding {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar(ScalarDType::F32) => "F32",
            Self::Scalar(ScalarDType::F16) => "F16",
            Self::Scalar(ScalarDType::Bf16) => "BF16",
            Self::Quantized(QuantScheme::Q4_0) => "Q4_0",
            Self::Quantized(QuantScheme::Q4_1) => "Q4_1",
            Self::Quantized(QuantScheme::Q6K) => "Q6_K",
            Self::Quantized(QuantScheme::Q8_0) => "Q8_0",
            Self::Quantized(QuantScheme::Q4K) => "Q4_K",
            Self::I64 => "I64",
        }
    }

    /// (logical elements per block, physical bytes per block).
    pub fn block(self) -> (u64, u64) {
        match self {
            Self::Scalar(ScalarDType::F32) => (1, 4),
            Self::Scalar(ScalarDType::F16) | Self::Scalar(ScalarDType::Bf16) => (1, 2),
            Self::Quantized(QuantScheme::Q4_0) => (32, 18),
            Self::Quantized(QuantScheme::Q4_1) => (32, 20),
            Self::Quantized(QuantScheme::Q6K) => (256, 210),
            Self::Quantized(QuantScheme::Q8_0) => (32, 34),
            Self::Quantized(QuantScheme::Q4K) => (256, 144),
            Self::I64 => (1, 8),
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
            0 => Ok(Self::Scalar(ScalarDType::F32)),
            1 => Ok(Self::Scalar(ScalarDType::F16)),
            2 => Ok(Self::Quantized(QuantScheme::Q4_0)),
            3 => Ok(Self::Quantized(QuantScheme::Q4_1)),
            14 => Ok(Self::Quantized(QuantScheme::Q6K)),
            8 => Ok(Self::Quantized(QuantScheme::Q8_0)),
            12 => Ok(Self::Quantized(QuantScheme::Q4K)),
            30 => Ok(Self::Scalar(ScalarDType::Bf16)),
            other => Err(Error::Unsupported(format!(
                "GGUF ggml tensor type {other}: no validated byte codec"
            ))),
        }
    }

    pub fn from_safetensors(code: &str) -> Result<Self> {
        match code {
            "F32" => Ok(Self::Scalar(ScalarDType::F32)),
            "F16" => Ok(Self::Scalar(ScalarDType::F16)),
            "BF16" => Ok(Self::Scalar(ScalarDType::Bf16)),
            "I64" => Ok(Self::I64),
            other => Err(Error::Unsupported(format!(
                "safetensors dtype {other}: no validated tensor codec"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_source_mapping_names_geometry_and_inventory_order() {
        let cases = [
            (
                0,
                StoredTensorEncoding::Scalar(ScalarDType::F32),
                "F32",
                "F32",
                (1, 4),
            ),
            (
                1,
                StoredTensorEncoding::Scalar(ScalarDType::F16),
                "F16",
                "F16",
                (1, 2),
            ),
            (
                30,
                StoredTensorEncoding::Scalar(ScalarDType::Bf16),
                "BF16",
                "Bf16",
                (1, 2),
            ),
            (
                2,
                StoredTensorEncoding::Quantized(QuantScheme::Q4_0),
                "Q4_0",
                "Q4_0",
                (32, 18),
            ),
            (
                3,
                StoredTensorEncoding::Quantized(QuantScheme::Q4_1),
                "Q4_1",
                "Q4_1",
                (32, 20),
            ),
            (
                8,
                StoredTensorEncoding::Quantized(QuantScheme::Q8_0),
                "Q8_0",
                "Q8_0",
                (32, 34),
            ),
            (
                14,
                StoredTensorEncoding::Quantized(QuantScheme::Q6K),
                "Q6_K",
                "Q6K",
                (256, 210),
            ),
            (
                12,
                StoredTensorEncoding::Quantized(QuantScheme::Q4K),
                "Q4_K",
                "Q4K",
                (256, 144),
            ),
        ];
        for &(code, encoding, name, debug, (block, bytes)) in &cases {
            assert_eq!(StoredTensorEncoding::from_ggml(code).unwrap(), encoding);
            assert_eq!(encoding.name(), name);
            assert_eq!(format!("{encoding:?}"), debug);
            assert_eq!(encoding.block(), (block, bytes));
            assert_eq!(encoding.byte_len(0).unwrap(), 0);
            assert_eq!(encoding.byte_len(block).unwrap(), bytes);
            assert_eq!(encoding.byte_len(256).unwrap(), 256 / block * bytes);
            if block > 1 {
                assert!(matches!(
                    encoding.byte_len(block - 1),
                    Err(Error::Malformed(_))
                ));
                assert!(matches!(
                    encoding.byte_len(block + 1),
                    Err(Error::Malformed(_))
                ));
            } else {
                assert_eq!(
                    StoredTensorEncoding::from_safetensors(name).unwrap(),
                    encoding
                );
                assert!(matches!(encoding.byte_len(u64::MAX), Err(Error::Limit(_))));
            }
        }
        assert!(cases.windows(2).all(|pair| pair[0].1 < pair[1].1));
    }

    #[test]
    fn unknown_and_future_codecs_remain_unavailable() {
        for code in (0..=40).chain([u32::MAX]) {
            if ![0, 1, 2, 3, 8, 12, 14, 30].contains(&code) {
                assert!(matches!(
                    StoredTensorEncoding::from_ggml(code),
                    Err(Error::Unsupported(_))
                ));
            }
        }
        for code in [
            "", "f32", "F64", "I8", "Q1", "Q2_K", "Q3_K", "Q4_K", "NVFP4", "EXL3",
        ] {
            assert!(matches!(
                StoredTensorEncoding::from_safetensors(code),
                Err(Error::Unsupported(_))
            ));
        }
    }
}
