use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Original {
    source: SourceIdentity,
    rows: [u64; 2],
    packed: Vec<u8>,
    reads: AtomicUsize,
}
impl Original {
    fn new() -> Self {
        Self {
            source: SourceIdentity::for_new_allocation().unwrap(),
            rows: [8, 4],
            packed: vec![0; 64],
            reads: AtomicUsize::new(0),
        }
    }
    fn matrix(&self, name: &str) -> SourceTensor {
        SourceTensor::from_source(self, name, MatrixAxes::OutputInput).unwrap()
    }
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if name == "narrow" {
            return Ok(SourceTensorRef {
                kind: QuantKind::Bf16,
                shape: &[8, 2],
                packed_bytes: &self.packed[..32],
            });
        }
        Ok(SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &self.rows,
            packed_bytes: &self.packed,
        })
    }
}
fn builder(source: SourceIdentity) -> PlanBuilder {
    PlanBuilder::new_operator_slice(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap()
}
fn input(b: &mut PlanBuilder, matrix: SourceTensor) -> PlanValue {
    b.embed_profiled(
        matrix,
        PlanBuilder::inputs().token,
        EmbeddingContract::Bf16OutputInput,
    )
    .unwrap()
}

#[test]
fn source_parameter_revalidates_real_kind_shape_bytes_and_foreign_owner_before_read() {
    let original = Original::new();
    let descriptor = SourceParameter::from_source(&original, "raw.parameter").unwrap();
    assert_eq!(descriptor.kind, QuantKind::Bf16);
    assert_eq!(descriptor.shape, [8, 4]);
    assert_eq!(descriptor.bytes, 64);
    descriptor.validate_source(&original).unwrap();
    let foreign = Original::new();
    assert!(descriptor.validate_source(&foreign).is_err());
    assert_eq!(foreign.reads.load(Ordering::Relaxed), 0);
    for changed in [
        SourceParameter {
            kind: QuantKind::F32,
            ..descriptor.clone()
        },
        SourceParameter {
            shape: vec![4, 8],
            ..descriptor.clone()
        },
        SourceParameter {
            bytes: 63,
            ..descriptor.clone()
        },
        SourceParameter {
            source: SourceIdentity::unbound(),
            ..descriptor.clone()
        },
    ] {
        assert!(changed.validate_source(&original).is_err());
    }
    assert!(SourceParameter::from_source(&original, "").is_err());
}

#[test]
fn explicit_precision_refuses_nonfinite_and_non_bf16_words_without_inference() {
    ActivationPrecision::F32.validate_values(&[1.0001]).unwrap();
    assert!(
        ActivationPrecision::Bf16WidenedF32
            .validate_values(&[1.0001])
            .is_err()
    );
    ActivationPrecision::Bf16WidenedF32
        .validate_values(&[1.0, -0.0, f32::from_bits(0x0001_0000)])
        .unwrap();
    for values in [&[][..], &[f32::NAN][..], &[f32::INFINITY][..]] {
        assert!(ActivationPrecision::F32.validate_values(values).is_err());
        assert!(
            ActivationPrecision::Bf16WidenedF32
                .validate_values(values)
                .is_err()
        );
    }
}

#[test]
fn pure_typed_slice_records_contracts_views_and_no_fake_state_or_frontier() {
    let source = Original::new();
    let mut b = builder(source.source);
    let embedded = input(&mut b, source.matrix("embedding"));
    let viewed = b
        .view(
            &embedded,
            TensorView {
                offset: 0,
                rows: 2,
                width: 1,
                row_stride: 2,
            },
        )
        .unwrap();
    assert_eq!(viewed.precision(), ActivationPrecision::Bf16WidenedF32);
    let packed = b.contiguous(&viewed).unwrap();
    assert_eq!(packed.precision(), embedded.precision());
    let matrix = source.matrix("narrow");
    let projected = b
        .project_profiled(matrix, &packed, ProjectionContract::TorchCpuBf16Avx2)
        .unwrap();
    b.assert_finite(0, &projected).unwrap();
    assert!(b.advance_frontier().is_err());
    let slice = b.finish_operator_slice(projected).unwrap();
    let plan = slice.plan();
    assert!(plan.is_operator_slice());
    assert_eq!(
        plan.numerical_profile(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    );
    assert_eq!(plan.values().len(), plan.precisions().len());
    assert!(
        plan.precisions()
            .iter()
            .all(|precision| *precision == ActivationPrecision::Bf16WidenedF32)
    );
    assert!(plan.required_domains().is_empty() && plan.proposed_domains().is_empty());
    assert_eq!(plan.frontier_delta(), 0);
    assert!(plan.logits().is_none());
    assert!(matches!(
        plan.nodes()[0].operation,
        PlanOperation::TypedEmbed { .. }
    ));
    assert!(matches!(
        plan.nodes()[3].operation,
        PlanOperation::TypedProject {
            contract: ProjectionContract::TorchCpuBf16Avx2,
            ..
        }
    ));
}

#[test]
fn typed_recorder_refuses_foreign_values_wrong_precision_axes_codec_and_legacy_math() {
    let source = Original::new();
    let mut b = builder(source.source);
    let embedded = input(&mut b, source.matrix("embedding"));
    let mut foreign = builder(source.source);
    let other = input(&mut foreign, source.matrix("embedding"));
    let projection = source.matrix("projection");
    assert!(
        b.project_profiled(
            projection.clone(),
            &other,
            ProjectionContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    assert!(b.project(projection.clone(), &embedded).is_err());
    assert!(b.silu("legacy", &embedded, None).is_err());
    assert!(b.add("legacy", &embedded, &embedded).is_err());
    for matrix in [
        SourceTensor {
            kind: QuantKind::F32,
            ..projection.clone()
        },
        SourceTensor {
            axes: MatrixAxes::InputOutput,
            ..projection.clone()
        },
        SourceTensor {
            source: SourceIdentity::for_new_allocation().unwrap(),
            ..projection.clone()
        },
    ] {
        assert!(
            b.project_profiled(matrix, &embedded, ProjectionContract::TorchCpuBf16Avx2)
                .is_err()
        );
    }
    let mut forged = embedded;
    forged.precision = ActivationPrecision::F32;
    assert!(b.contiguous(&forged).is_err());
    assert!(b.finish_operator_slice(other).is_err());
    let mut full = builder(source.source);
    let value = input(&mut full, source.matrix("embedding"));
    assert!(full.finish(Some(value)).is_err());
}

#[test]
fn packed_bf16_does_not_change_legacy_f32_operations_or_select_cpu64() {
    let source = Original::new();
    let mut old =
        PlanBuilder::new_operator_slice(source.source, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let embedded = old
        .embed(source.matrix("embedding"), PlanBuilder::inputs().token)
        .unwrap();
    assert_eq!(embedded.precision(), ActivationPrecision::F32);
    let projected = old.project(source.matrix("projection"), &embedded).unwrap();
    assert!(
        old.project_profiled(
            source.matrix("projection"),
            &embedded,
            ProjectionContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    let slice = old.finish_operator_slice(projected).unwrap();
    assert!(matches!(
        slice.plan().nodes()[0].operation,
        PlanOperation::Embed { .. }
    ));
    assert!(matches!(
        slice.plan().nodes()[1].operation,
        PlanOperation::Project { .. }
    ));
    assert!(
        slice
            .plan()
            .precisions()
            .iter()
            .all(|precision| *precision == ActivationPrecision::F32)
    );
}
