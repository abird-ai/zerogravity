//! Source arithmetic and activation coding, independent of weight storage or readiness.
pub use crate::accelerator::program::StandaloneExpProfile;
use crate::encoding::{QuantScheme, ScalarDType, StoredTensorEncoding};
use crate::{Error, Result};

/// Mathematical precision carried in F32 storage, independent of weight codec.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ActivationPrecision {
    #[default]
    F32,
    Bf16WidenedF32,
}
impl ActivationPrecision {
    pub fn validate_values(self, values: &[f32]) -> Result<()> {
        if values.is_empty()
            || values.iter().any(|value| {
                !value.is_finite() || self == Self::Bf16WidenedF32 && value.to_bits() & 0xffff != 0
            })
        {
            return Err(Error::Malformed(
                "activation differs from explicit precision".into(),
            ));
        }
        Ok(())
    }
}

/// Reserved program contract, not inference support or a backend-global dtype.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProgramNumericalProfile {
    #[default]
    GgufSourceV1,
    TorchCpuBf16Avx2MixedSourceV1,
    /// Prior name: Vllm031CudaSm86Bf16SourceV1. Arithmetic unchanged; no eligibility.
    SourceCudaSm86Bf16V1,
    /// Bounded mathematical text reference; no trained/framework parity grant.
    AnalyticF32TextV1,
    /// Prior name: Gemma4E2bBf16TextCandidateV1. Direct gamma; no qualification.
    DirectGammaBf16TextCandidateV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingContract {
    Source,
    Bf16OutputInput,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionContract {
    Source,
    TorchCpuBf16Avx2,
}

/// Exact historical arithmetic identity, independent of a model's source roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceNumericalContract {
    // Prior arithmetic name: Vllm031CudaSm86Bf16V1.
    ReferenceCudaSm86Bf16V1,
    /// Original Torch CPU AVX2 BF16/mixed parameters and normal-domain MKL exp.
    /// A mathematical identity, not CPU/HIP execution eligibility.
    TorchCpuBf16Avx2MixedSourceV1,
    GgmlSourceF32V1,
    GgmlQ4Repack8x8F32V1,
    OpenBlas033Zen16Q8F32V1,
}
impl SourceNumericalContract {
    pub fn is_f32(self) -> bool {
        matches!(
            self,
            Self::GgmlSourceF32V1 | Self::GgmlQ4Repack8x8F32V1 | Self::OpenBlas033Zen16Q8F32V1
        )
    }
    pub fn program_profile(self) -> ProgramNumericalProfile {
        match self {
            Self::ReferenceCudaSm86Bf16V1 => ProgramNumericalProfile::SourceCudaSm86Bf16V1,
            Self::TorchCpuBf16Avx2MixedSourceV1 => {
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            }
            Self::GgmlSourceF32V1 | Self::GgmlQ4Repack8x8F32V1 | Self::OpenBlas033Zen16Q8F32V1 => {
                ProgramNumericalProfile::GgufSourceV1
            }
        }
    }
}
/// Input coding for an explicit numerical operation; not a kernel permit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationEncoding {
    Scalar(ActivationPrecision),
    GgmlQ8_0,
    GgmlQ8_1,
    GgmlQ8K,
}

impl ProjectionContract {
    /// Source pairing only. Shape, source ownership, backend and implementation
    /// checks remain mandatory; Q4_K's known pairing does not implement its dot.
    pub fn activation(self, weights: StoredTensorEncoding) -> Result<ActivationEncoding> {
        match self {
            Self::Source => Ok(match weights {
                StoredTensorEncoding::Scalar(_) => {
                    ActivationEncoding::Scalar(ActivationPrecision::F32)
                }
                StoredTensorEncoding::Quantized(QuantScheme::Q4_0 | QuantScheme::Q8_0) => {
                    ActivationEncoding::GgmlQ8_0
                }
                StoredTensorEncoding::Quantized(QuantScheme::Q4_1) => ActivationEncoding::GgmlQ8_1,
                StoredTensorEncoding::Quantized(QuantScheme::Q6K | QuantScheme::Q4K) => {
                    ActivationEncoding::GgmlQ8K
                }
            }),
            Self::TorchCpuBf16Avx2 => {
                if weights != StoredTensorEncoding::Scalar(ScalarDType::Bf16) {
                    return Err(Error::Unsupported(
                        "CPU BF16 AVX2 profile requires BF16 source rows".into(),
                    ));
                }
                Ok(ActivationEncoding::Scalar(
                    ActivationPrecision::Bf16WidenedF32,
                ))
            }
        }
    }
}

/// Finite greedy readout: first numeric maximum, including ±0 ties.
/// Returns the original selected value, without normalizing its bits.
pub fn greedy_finite(logits: &[f32]) -> Result<(u32, f32)> {
    if logits.is_empty() || u32::try_from(logits.len()).is_err() {
        return Err(Error::Limit("greedy vocabulary extent".into()));
    }
    let mut best = 0;
    for (i, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::State("greedy requires finite logits".into()));
        }
        if value > logits[best] {
            best = i;
        }
    }
    Ok((best as u32, logits[best]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_greedy_preserves_first_numeric_tie_and_source_bits() {
        for values in [[-0.0, 0.0], [0.0, -0.0], [3.0, 3.0]] {
            let (id, maximum) = greedy_finite(&values).unwrap();
            assert_eq!(id, 0);
            assert_eq!(maximum.to_bits(), values[0].to_bits());
        }
        assert_eq!(greedy_finite(&[-2.0, 4.0, 4.0]).unwrap(), (1, 4.0));
        for values in [
            &[][..],
            &[f32::NAN, 1.0],
            &[1.0, f32::INFINITY],
            &[f32::NEG_INFINITY],
        ] {
            assert!(greedy_finite(values).is_err());
        }
    }

    #[test]
    fn source_storage_does_not_select_the_numerical_profile() {
        let weights = StoredTensorEncoding::Scalar(ScalarDType::Bf16);
        assert_eq!(
            ProjectionContract::Source.activation(weights).unwrap(),
            ActivationEncoding::Scalar(ActivationPrecision::F32)
        );
        assert_eq!(
            ProjectionContract::TorchCpuBf16Avx2
                .activation(weights)
                .unwrap(),
            ActivationEncoding::Scalar(ActivationPrecision::Bf16WidenedF32)
        );
        for dtype in [ScalarDType::F32, ScalarDType::F16] {
            assert!(matches!(
                ProjectionContract::TorchCpuBf16Avx2
                    .activation(StoredTensorEncoding::Scalar(dtype)),
                Err(Error::Unsupported(_))
            ));
        }
        for (scheme, expected) in [
            (QuantScheme::Q4_0, ActivationEncoding::GgmlQ8_0),
            (QuantScheme::Q4_1, ActivationEncoding::GgmlQ8_1),
            (QuantScheme::Q8_0, ActivationEncoding::GgmlQ8_0),
            (QuantScheme::Q6K, ActivationEncoding::GgmlQ8K),
            (QuantScheme::Q4K, ActivationEncoding::GgmlQ8K),
        ] {
            let storage = StoredTensorEncoding::Quantized(scheme);
            assert_eq!(
                ProjectionContract::Source.activation(storage).unwrap(),
                expected
            );
            assert!(matches!(
                ProjectionContract::TorchCpuBf16Avx2.activation(storage),
                Err(Error::Unsupported(_))
            ));
        }
    }
}
