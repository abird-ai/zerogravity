use super::*;
fn span() -> TeacherSpan {
    TeacherSpan::new(
        SourceIdentity::for_new_allocation().unwrap(),
        0,
        8448,
        248320,
        &(0..64).collect::<Vec<_>>(),
    )
    .unwrap()
}
fn matrix(span: &TeacherSpan) -> BulkF32Matrix {
    let left = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Left,
        "actual external V",
        BulkF32Layout::new([256, 256, 2, 1], [1, 8448, 2162688, 4325376], 0, 4317184).unwrap(),
    )
    .unwrap();
    let right = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "actual external P",
        BulkF32Layout::dense([256, 64, 8, 1]).unwrap(),
    )
    .unwrap();
    BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap()
}
#[test]
fn actual_temporal_n_is_not_cohort_width_and_full_k_is_preserved() {
    let span = span();
    let operation = matrix(&span);
    assert_eq!(operation.left().layout().shape(), [256, 256, 2, 1]);
    assert_eq!(operation.right().layout().shape(), [256, 64, 8, 1]);
    assert_eq!(operation.output().shape(), [256, 64, 8, 1]);
    assert_eq!(operation.output().backing_words(), 131072);
    // Real N2 is a different declared teacher span, never a B2 request label.
    let wrong = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "P",
        BulkF32Layout::dense([256, 2, 8, 1]).unwrap(),
    );
    assert!(wrong.is_err());
    assert!(TeacherSpan::new(span.source(), 0, 8448, 248320, &[11]).is_err());
    assert!(TeacherSpan::new(span.source(), 8447, 8448, 248320, &[11, 353]).is_err());
}
#[test]
fn same_ids_source_names_do_not_merge_issuers_and_wrong_full_k_refuses() {
    let span = span();
    let operation = matrix(&span);
    let foreign = TeacherSpan::new(
        span.source(),
        span.base(),
        span.capacity(),
        248320,
        span.ids(),
    )
    .unwrap();
    let right = BulkF32Operand::external(
        foreign,
        BulkF32Side::Right,
        "actual external P",
        operation.right().layout(),
    )
    .unwrap();
    assert!(
        BulkF32Matrix::new(
            operation.left().clone(),
            right,
            BulkF32Contract::HaswellTinyBlasF32V1
        )
        .is_err()
    );
    let right = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "P",
        BulkF32Layout::dense([128, 64, 8, 1]).unwrap(),
    )
    .unwrap();
    assert!(
        BulkF32Matrix::new(
            operation.left().clone(),
            right,
            BulkF32Contract::HaswellTinyBlasF32V1
        )
        .is_err()
    );
}
#[test]
fn real_nonzero_offsets_and_gaps_need_full_checked_storage() {
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 47).is_ok());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 46).is_err());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 7, 28, 28], 0, 32).is_err());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], usize::MAX, 64).is_err());
    assert!(BulkF32Layout::dense([8192, 4096, 8, 1]).is_err());
    let span = span();
    assert!(
        BulkF32Operand::external(
            span,
            BulkF32Side::Right,
            "strided P",
            BulkF32Layout::new([8, 64, 1, 1], [1, 12, 768, 768], 0, 764).unwrap()
        )
        .is_err()
    );
}
#[test]
fn same_builder_three_nodes_stay_pure_and_legacy_use_or_foreign_values_refuse() {
    let span = span();
    let operation = matrix(&span);
    let mut b =
        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let left = b.bulk_f32_input(operation.left()).unwrap();
    let right = b.bulk_f32_input(operation.right()).unwrap();
    let mut other =
        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let foreign = other.bulk_f32_input(operation.right()).unwrap();
    assert!(b.bulk_f32_matmul(&operation, &left, &foreign).is_err());
    assert!(b.silu("legacy", &right, None).is_err());
    let output = b.bulk_f32_matmul(&operation, &left, &right).unwrap();
    let slice = b.finish_operator_slice(output).unwrap();
    assert_eq!(slice.plan().nodes().len(), 3);
    assert!(slice.plan().required_domains().is_empty());
    assert!(slice.plan().proposed_domains().is_empty());
    assert_eq!(slice.plan().frontier_delta(), 0);
    assert!(slice.plan().logits().is_none());
    assert!(slice.plan().is_operator_slice());
    let slot = StateSlot::Conv(ConvDomain(0));
    let mut full = PlanBuilder::new(span.source(), 8448, vec![slot], false).unwrap();
    assert!(full.bulk_f32_input(operation.left()).is_err());
}

struct Original {
    identity: SourceIdentity,
    kind: QuantKind,
    shape: [u64; 2],
    bytes: Vec<u8>,
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        if name != "original.weight" {
            return Err(Error::Malformed("unknown original name".into()));
        }
        Ok(SourceTensorRef {
            kind: self.kind,
            shape: &self.shape,
            packed_bytes: &self.bytes,
        })
    }
}
#[test]
fn whole_raw_source_matrix_keeps_axes_dtype_extent_and_exact_issuer() {
    let original = Original {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        kind: QuantKind::F32,
        shape: [8, 4],
        bytes: vec![0; 128],
    };
    let span = TeacherSpan::new(original.identity, 0, 8, 248320, &[11, 353]).unwrap();
    let left = BulkF32Operand::source_matrix(span.clone(), &original, "original.weight").unwrap();
    assert_eq!(left.layout().shape(), [8, 4, 1, 1]);
    let BulkF32Origin::SourceMatrix(raw) = left.origin() else {
        panic!("original raw source expected")
    };
    assert_eq!(raw.axes, MatrixAxes::InputOutput);
    raw.validate_source(&original).unwrap();
    let foreign = Original {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        ..original
    };
    assert!(BulkF32Operand::source_matrix(span.clone(), &foreign, "original.weight").is_err());
    let bf16 = Original {
        identity: span.source(),
        kind: QuantKind::Bf16,
        shape: [8, 4],
        bytes: vec![0; 64],
    };
    assert!(BulkF32Operand::source_matrix(span.clone(), &bf16, "original.weight").is_err());
    let short = Original {
        identity: span.source(),
        kind: QuantKind::F32,
        shape: [8, 4],
        bytes: vec![0; 124],
    };
    assert!(BulkF32Operand::source_matrix(span, &short, "original.weight").is_err());
}
