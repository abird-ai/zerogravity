use super::*;

struct Original {
    identity: SourceIdentity,
    kind: QuantKind,
    shape: [u64; 2],
    bytes: Vec<u8>,
}
impl Original {
    fn new() -> Self {
        Self {
            identity: SourceIdentity::for_new_allocation().unwrap(),
            kind: QuantKind::Q8_0,
            shape: [2048, 2048],
            bytes: vec![0; 2048 * 2048 / 32 * 34],
        }
    }
    fn span(&self) -> TeacherSpan {
        TeacherSpan::new(self.identity, 0, 8448, 248320, &(0..64).collect::<Vec<_>>()).unwrap()
    }
    fn descriptor(&self, span: TeacherSpan) -> Result<ZenSsmOutDescriptor> {
        ZenSsmOutDescriptor::new(
            span,
            self,
            "blk.0.ssm_out.weight",
            ZenSsmOutContract::OpenBlas033Zen16SsmOutV1,
        )
    }
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        if name != "blk.0.ssm_out.weight" {
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
fn original_q8_matrix_and_exact64_teachers_are_distinct_from_tinyblas() {
    let mut source = Original::new();
    let span = source.span();
    let descriptor = source.descriptor(span.clone()).unwrap();
    assert_eq!(descriptor.tensor().kind, QuantKind::Q8_0);
    assert_eq!(descriptor.tensor().axes, MatrixAxes::InputOutput);
    assert_eq!(descriptor.layout().shape(), [2048, 64, 1, 1]);
    assert_eq!(
        descriptor.contract(),
        ZenSsmOutContract::OpenBlas033Zen16SsmOutV1
    );
    assert!(BulkF32Operand::source_matrix(span.clone(), &source, "blk.0.ssm_out.weight").is_err());
    let short = TeacherSpan::new(source.identity, 0, 8448, 248320, &[11, 353]).unwrap();
    assert!(source.descriptor(short).is_err());
    source.bytes.pop();
    assert!(source.descriptor(span.clone()).is_err());
    source.bytes.push(0);
    source.shape = [1024, 4096];
    assert!(source.descriptor(span.clone()).is_err());
    source.shape = [2048, 2048];
    source.kind = QuantKind::F32;
    assert!(source.descriptor(span.clone()).is_err());
    source.kind = QuantKind::Q8_0;
    source.identity = SourceIdentity::for_new_allocation().unwrap();
    assert!(source.descriptor(span).is_err());
}

#[test]
fn recorded_cut_has_no_state_and_rejects_other_issuers_builders_or_full_routes() {
    let source = Original::new();
    let span = source.span();
    let descriptor = source.descriptor(span.clone()).unwrap();
    let same_ids_foreign_issuer = source.descriptor(source.span()).unwrap();
    assert_ne!(descriptor, same_ids_foreign_issuer);
    let mut b =
        PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let input = b.zen_ssm_out_input(&descriptor).unwrap();
    assert!(
        b.zen_ssm_out_project(&same_ids_foreign_issuer, &input)
            .is_err()
    );
    let mut other =
        PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let foreign = other.zen_ssm_out_input(&descriptor).unwrap();
    assert!(b.zen_ssm_out_project(&descriptor, &foreign).is_err());
    assert!(b.silu("legacy", &input, None).is_err());
    assert!(b.view(&input, TensorView::vector(0, 131072)).is_err());
    let output = b.zen_ssm_out_project(&descriptor, &input).unwrap();
    assert!(b.zen_ssm_out_project(&descriptor, &output).is_err());
    assert!(b.silu("legacy", &output, None).is_err());
    let slice = b.finish_operator_slice(output).unwrap();
    assert_eq!(slice.plan().nodes().len(), 2);
    assert!(slice.plan().required_domains().is_empty());
    assert!(slice.plan().proposed_domains().is_empty());
    assert_eq!(slice.plan().frontier_delta(), 0);
    assert!(slice.plan().logits().is_none());
    let mut full = PlanBuilder::new(
        source.identity,
        8448,
        vec![StateSlot::Conv(ConvDomain(0))],
        false,
    )
    .unwrap();
    assert!(full.zen_ssm_out_input(&descriptor).is_err());
    let mut bf16 = PlanBuilder::new_operator_slice(
        source.identity,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    assert!(bf16.zen_ssm_out_input(&descriptor).is_err());
    let mut other_source = PlanBuilder::new_operator_slice(
        SourceIdentity::for_new_allocation().unwrap(),
        ProgramNumericalProfile::GgufSourceV1,
    )
    .unwrap();
    assert!(other_source.zen_ssm_out_input(&descriptor).is_err());
}

#[test]
fn equal_f32_storage_does_not_bridge_tinyblas_or_arbitrary_leaves() {
    let source = Original::new();
    let span = source.span();
    let descriptor = source.descriptor(span.clone()).unwrap();
    let left = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Left,
        "A",
        BulkF32Layout::dense([2048, 4, 1, 1]).unwrap(),
    )
    .unwrap();
    let right =
        BulkF32Operand::external(span, BulkF32Side::Right, "B", descriptor.layout()).unwrap();
    let tiny = BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap();
    let mut b =
        PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let a = b.bulk_f32_input(tiny.left()).unwrap();
    let observed = b.bulk_f32_input(tiny.right()).unwrap();
    let retained = b.zen_ssm_out_input(&descriptor).unwrap();
    assert!(b.zen_ssm_out_project(&descriptor, &observed).is_err());
    assert!(b.bulk_f32_matmul(&tiny, &a, &retained).is_err());
    assert!(b.zen_ssm_out_project(&descriptor, &a).is_err());
    b.bulk_f32_matmul(&tiny, &a, &observed).unwrap();
    b.zen_ssm_out_project(&descriptor, &retained).unwrap();
}
