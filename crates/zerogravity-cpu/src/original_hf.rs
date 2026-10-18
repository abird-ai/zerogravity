//! Original Hugging Face output-first source rows using the existing CPU math.
//! Torch linear matrices are [output,input] in row-major memory. No whole
//! learned matrix transpose or duplicate mathematical runtime is required.
//! This does not enable complete trained Qwen3.8 inference.

use crate::accelerator::{MatVecKernel, MatVecRegion, RegionExecution, validate_completed};
use crate::quant::QuantKind;
use crate::weights::{MAX_MATRIX_ROWS, source_reference_projection};
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

/// Project one already source-qualified original [output,input] tensor
/// through the same real CPU kernel and optional physical provider.
pub fn output_first_projection(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
) -> Result<Vec<f32>> {
    tensor.validate()?;
    if !source_id.is_bound()
        || tensor_name.is_empty()
        || tensor.shape.len() != 2
        || tensor.shape[1] != input.len() as u64
        || tensor.shape[0] > MAX_MATRIX_ROWS
        || input.is_empty()
        || !input.iter().all(|x| x.is_finite())
        || !matches!(
            tensor.kind,
            QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16
        )
    {
        return Err(Error::Malformed(
            "original output-first source GEMV geometry/encoding invalid".into(),
        ));
    }
    let rows = usize::try_from(tensor.shape[0])
        .map_err(|_| Error::Limit("output-first matrix rows exceed address space".into()))?;
    let columns = usize::try_from(tensor.shape[1])
        .map_err(|_| Error::Limit("output-first matrix columns exceed address space".into()))?;
    let region = MatVecRegion {
        source_identity: source_id,
        tensor_name,
        kind: tensor.kind,
        columns,
        rows,
        packed_weights: tensor.packed_bytes,
        activation: input,
    };
    if let Some(provider) = provider {
        match provider.try_execute(region)? {
            RegionExecution::Unsupported => {}
            RegionExecution::Completed(values) => {
                validate_completed(region, &values)?;
                return Ok(values);
            }
        }
    }
    source_reference_projection(region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, RegionExecution, SourceIdentity};

    fn source_id() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn projected(kind: QuantKind, shape: &[u64], bytes: &[u8], input: &[f32]) -> Result<Vec<f32>> {
        output_first_projection(
            source_id(),
            "original.linear.weight",
            SourceTensorRef {
                kind,
                shape,
                packed_bytes: bytes,
            },
            input,
            None,
        )
    }
    #[test]
    fn original_f32_hf_output_first_uses_same_reference_projection_without_transposition() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4., 5., 6.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        assert_eq!(
            projected(QuantKind::F32, &[2, 3], &bytes, &[7., 8., 9.]).unwrap(),
            [50., 122.]
        );
        assert!(projected(QuantKind::F32, &[3, 2], &bytes, &[7., 8., 9.]).is_err());
    }
    #[test]
    fn original_bf16_and_f16_rows_reuse_existing_raw_source_math() {
        let mut bf16 = Vec::new();
        for bits in [0x3f80u16, 0x4000, 0x4040, 0x4080] {
            bf16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::Bf16, &[2, 2], &bf16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        let mut f16 = Vec::new();
        for bits in [0x3c00u16, 0x4000, 0x4200, 0x4400] {
            f16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::F16, &[2, 2], &f16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
    }
    struct UnsupportedProvider;
    impl MatVecKernel for UnsupportedProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
    }
    struct BadProvider;
    impl MatVecKernel for BadProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Completed(vec![f32::NAN]))
        }
    }
    #[test]
    fn optional_physical_unsupported_allows_cpu_but_invalid_completion_refuses() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        let tensor = SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[2, 2],
            packed_bytes: &bytes,
        };
        let normal =
            output_first_projection(source_id(), "q.weight", tensor, &[5., 6.], None).unwrap();
        let fallback = output_first_projection(
            source_id(),
            "q.weight",
            tensor,
            &[5., 6.],
            Some(&UnsupportedProvider),
        )
        .unwrap();
        assert_eq!(normal, fallback);
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                tensor,
                &[5., 6.],
                Some(&BadProvider)
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                SourceIdentity::unbound(),
                "q.weight",
                tensor,
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(source_id(), "q.weight", tensor, &[f32::NAN, 6.], None)
                .is_err()
        );
        assert!(output_first_projection(source_id(), "q.weight", tensor, &[1.], None).is_err());
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    kind: QuantKind::Q4_0,
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    shape: &[2, 3],
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        // Original source bound MUST apply before a physical provider
        // is ever permitted to override the CPU's matrix safety cap.
        let large = vec![0u8; 2 * 1_000_001];
        let huge = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &large,
        };
        assert!(
            output_first_projection(
                source_id(),
                "oversized.hf.weight",
                huge,
                &[1.0],
                Some(&UnsupportedProvider)
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ObservedProvider<'a>(&'a AtomicBool);
    impl MatVecKernel for ObservedProvider<'_> {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Unsupported)
        }
    }

    #[test]
    fn oversize_hf_output_rows_refuse_before_accelerator_invocation() {
        let called = AtomicBool::new(false);
        let source = SourceIdentity::for_new_allocation().unwrap();
        let bytes = vec![0u8; 2 * 1_000_001];
        let tensor = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &bytes,
        };
        let result = output_first_projection(
            source,
            "original.large.weight",
            tensor,
            &[1.0],
            Some(&ObservedProvider(&called)),
        );
        assert!(matches!(result, Err(Error::Malformed(_))));
        assert!(!called.load(Ordering::Relaxed));
    }
}
