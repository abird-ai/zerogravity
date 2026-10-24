use super::*;

struct Original {
    identity: SourceIdentity,
    embedding: Vec<u8>,
    norm: Vec<u8>,
}
impl Original {
    fn new() -> Self {
        Self {
            identity: SourceIdentity::for_new_allocation().unwrap(),
            embedding: vec![0; 1680 * 16],
            norm: vec![0; 8192],
        }
    }
    fn span(&self) -> TeacherSpan {
        TeacherSpan::new(self.identity, 0, 8448, 16, &[11; 64]).unwrap()
    }
    fn descriptor(&self, span: TeacherSpan) -> GgufPrefixDescriptor {
        GgufPrefixDescriptor::new(span, self, GgufPrefixContract::GgufSource2BFirst64V1, 1e-6)
            .unwrap()
    }
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        match name {
            "token_embd.weight" => Ok(SourceTensorRef {
                kind: QuantKind::Q6K,
                shape: &[2048, 16],
                packed_bytes: &self.embedding,
            }),
            "blk.0.attn_norm.weight" => Ok(SourceTensorRef {
                kind: QuantKind::F32,
                shape: &[2048],
                packed_bytes: &self.norm,
            }),
            _ => Err(Error::Malformed("unknown test source".into())),
        }
    }
}

#[test]
fn exact_prefix_source_contract_rejects_other_geometry_epsilon_and_ids() {
    let mut source = Original::new();
    let span = source.span();
    let descriptor = source.descriptor(span.clone());
    assert_eq!(descriptor.embedding().kind, QuantKind::Q6K);
    assert_eq!(descriptor.norm().kind, QuantKind::F32);
    assert_eq!(descriptor.layout().shape(), [2048, 64, 1, 1]);
    assert_eq!(descriptor.epsilon().to_bits(), 1e-6_f32.to_bits());
    assert!(GgufPrefixDescriptor::new(span.clone(), &source, descriptor.contract(), 1e-5).is_err());
    let shifted = TeacherSpan::new(source.identity, 1, 8448, 16, &[11; 64]).unwrap();
    assert!(GgufPrefixDescriptor::new(shifted, &source, descriptor.contract(), 1e-6).is_err());
    let short = TeacherSpan::new(source.identity, 0, 8448, 16, &[11; 2]).unwrap();
    assert!(GgufPrefixDescriptor::new(short, &source, descriptor.contract(), 1e-6).is_err());
    let out_of_source = TeacherSpan::new(source.identity, 0, 8448, 17, &[16; 64]).unwrap();
    assert!(
        GgufPrefixDescriptor::new(out_of_source, &source, descriptor.contract(), 1e-6).is_err()
    );
    source.norm.pop();
    assert!(GgufPrefixDescriptor::new(span.clone(), &source, descriptor.contract(), 1e-6).is_err());
    source.norm.push(0);
    source.embedding.pop();
    assert!(GgufPrefixDescriptor::new(span.clone(), &source, descriptor.contract(), 1e-6).is_err());
    source.embedding.push(0);
    source.identity = SourceIdentity::for_new_allocation().unwrap();
    assert!(GgufPrefixDescriptor::new(span, &source, descriptor.contract(), 1e-6).is_err());
}

#[test]
fn prefix_records_two_nodes_and_refuses_foreign_issuers_builders_and_legacy_consumers() {
    let source = Original::new();
    let descriptor = source.descriptor(source.span());
    let foreign = source.descriptor(source.span());
    let mut builder =
        PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let embedding = builder.gguf_prefix_embedding(&descriptor).unwrap();
    assert!(builder.gguf_prefix_rms(&foreign, &embedding).is_err());
    let mut other =
        PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let foreign_value = other.gguf_prefix_embedding(&descriptor).unwrap();
    assert!(
        builder
            .gguf_prefix_rms(&descriptor, &foreign_value)
            .is_err()
    );
    assert!(
        builder
            .view(&embedding, TensorView::vector(0, 131072))
            .is_err()
    );
    assert!(builder.silu("legacy", &embedding, None).is_err());
    let normalized = builder.gguf_prefix_rms(&descriptor, &embedding).unwrap();
    assert!(builder.gguf_prefix_rms(&descriptor, &normalized).is_err());
    assert!(builder.silu("legacy", &normalized, None).is_err());
    let recorded = builder.finish_operator_slice(normalized).unwrap();
    assert_eq!(recorded.plan().nodes().len(), 2);
    assert!(recorded.plan().required_domains().is_empty());
    assert!(recorded.plan().proposed_domains().is_empty());
    assert_eq!(recorded.plan().frontier_delta(), 0);
    assert!(recorded.plan().logits().is_none());
    let mut full = PlanBuilder::new(
        source.identity,
        8448,
        vec![StateSlot::Conv(ConvDomain(0))],
        false,
    )
    .unwrap();
    assert!(full.gguf_prefix_embedding(&descriptor).is_err());
    let mut bf16 = PlanBuilder::new_operator_slice(
        source.identity,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    assert!(bf16.gguf_prefix_embedding(&descriptor).is_err());
    let mut wrong_source = PlanBuilder::new_operator_slice(
        SourceIdentity::for_new_allocation().unwrap(),
        ProgramNumericalProfile::GgufSourceV1,
    )
    .unwrap();
    assert!(wrong_source.gguf_prefix_embedding(&descriptor).is_err());
}

#[test]
fn observed_equal_f32_words_cannot_replace_retained_prefix_origin() {
    let source = Original::new();
    let descriptor = source.descriptor(source.span());
    let right = BulkF32Operand::retained_normalized(descriptor.clone()).unwrap();
    assert!(
        matches!(right.origin(), BulkF32Origin::RetainedNormalized { prefix } if prefix.as_ref() == &descriptor)
    );
    let left = BulkF32Operand::external(
        descriptor.span().clone(),
        BulkF32Side::Left,
        "left",
        BulkF32Layout::dense([2048, 16, 1, 1]).unwrap(),
    )
    .unwrap();
    let observed = BulkF32Operand::external(
        descriptor.span().clone(),
        BulkF32Side::Right,
        "observed",
        descriptor.layout(),
    )
    .unwrap();
    let matrix = BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap();
    let mut builder =
        PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let a = builder.bulk_f32_input(matrix.left()).unwrap();
    let b = builder.bulk_f32_input(&observed).unwrap();
    assert!(builder.bulk_f32_matmul(&matrix, &a, &b).is_err());
    let embedding = builder.gguf_prefix_embedding(&descriptor).unwrap();
    assert!(builder.bulk_f32_matmul(&matrix, &a, &embedding).is_err());
    let actual = builder.bulk_f32_input(matrix.right()).unwrap();
    builder.bulk_f32_matmul(&matrix, &a, &actual).unwrap();
    let foreign = source.descriptor(source.span());
    assert_ne!(
        BulkF32Operand::retained_normalized(foreign).unwrap(),
        *matrix.right()
    );
}
