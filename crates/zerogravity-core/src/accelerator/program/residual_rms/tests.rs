use super::*;

// Synthetic descriptor metadata only; no trained input or numerical proof.
fn descriptor(source: SourceIdentity, rows: usize) -> VllmBf16Embedding {
    let teachers = VllmBf16Teachers::new(source, 0, 64, &vec![11; rows]).unwrap();
    VllmBf16Embedding::new(
        SourceTensor {
            source,
            name: EMBEDDING.into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            columns: WIDTH,
            rows: VOCABULARY,
        },
        teachers,
    )
    .unwrap()
}
fn parameter(source: SourceIdentity) -> SourceParameter {
    SourceParameter {
        source,
        name: POST_NORM.into(),
        kind: QuantKind::Bf16,
        shape: vec![1024],
        bytes: 2048,
    }
}
fn builder(source: SourceIdentity) -> PlanBuilder {
    PlanBuilder::new_operator_slice(source, PROFILE).unwrap()
}
fn inputs(b: &mut PlanBuilder, rows: usize) -> (PlanValue, PlanValue) {
    let d = descriptor(b.source, rows);
    (
        b.vllm_bf16_source_embedding(&d).unwrap(),
        b.vllm_bf16_source_embedding(&d).unwrap(),
    )
}

#[test]
fn dual_output_slice_preserves_bf16_source_metadata_and_no_state() {
    for rows in [1, 2, 64] {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut b = builder(source);
        let (branch, prior) = inputs(&mut b, rows);
        let value = b
            .residual_rms_profiled(
                parameter(source),
                &branch,
                &prior,
                ResidualRmsContract::Vllm031CudaAtenBf16H1024Rows1_2_64,
            )
            .unwrap();
        for output in [value.normalized, value.residual] {
            assert_eq!(output.shape, ValueShape::Vector(rows * 1024));
            assert_eq!(output.precision(), ActivationPrecision::Bf16WidenedF32);
        }
        assert_ne!(value.normalized.id, value.residual.id);
        let recorded = b.finish_operator_slice(value.normalized).unwrap();
        let plan = recorded.plan();
        assert_eq!(plan.nodes().len(), 3);
        assert_eq!(
            plan.values().len(),
            4,
            "FP32 sum stays private physical workspace"
        );
        assert!(
            plan.precisions()
                .iter()
                .all(|p| *p == ActivationPrecision::Bf16WidenedF32)
        );
        assert!(matches!(
            plan.nodes()[2].operation,
            PlanOperation::TypedResidualRms { .. }
        ));
        assert_eq!(plan.nodes()[2].inputs.len(), 2);
        assert_eq!(plan.nodes()[2].outputs.len(), 2);
        assert!(plan.nodes().iter().all(|node| node.state.is_none()));
        assert!(plan.required_domains().is_empty() && plan.proposed_domains().is_empty());
        assert_eq!(plan.frontier_delta(), 0);
        assert!(plan.logits().is_none());
    }
}

#[test]
fn raw_parameter_binding_dtype_shape_name_and_rows_are_closed() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let contract = ResidualRmsContract::Vllm031CudaAtenBf16H1024Rows1_2_64;
    let valid = parameter(source);
    assert!(
        contract
            .validate(
                &SourceParameter {
                    kind: QuantKind::F32,
                    bytes: 4096,
                    ..valid.clone()
                },
                1024
            )
            .is_ok()
    );
    for wrong in [
        SourceParameter {
            source: SourceIdentity::unbound(),
            ..valid.clone()
        },
        SourceParameter {
            name: String::new(),
            ..valid.clone()
        },
        SourceParameter {
            name: "model.language_model.layers.0.input_layernorm.weight".into(),
            ..valid.clone()
        },
        SourceParameter {
            kind: QuantKind::Q8_0,
            ..valid.clone()
        },
        SourceParameter {
            shape: vec![2048],
            ..valid.clone()
        },
        SourceParameter {
            bytes: 2047,
            ..valid.clone()
        },
    ] {
        assert!(contract.validate(&wrong, 1024).is_err());
    }
    for elements in [0, 1023, 3 * 1024, 63 * 1024, 65 * 1024, usize::MAX] {
        assert!(contract.validate(&valid, elements).is_err());
    }
    let mut b = builder(source);
    let (x, r) = inputs(&mut b, 2);
    let before = b.nodes.len();
    assert!(
        b.residual_rms_profiled(
            parameter(SourceIdentity::for_new_allocation().unwrap()),
            &x,
            &r,
            contract
        )
        .is_err()
    );
    assert!(b.residual_rms_profiled(valid, &x, &x, contract).is_err());
    assert_eq!(b.nodes.len(), before);
}

#[test]
fn source_rows_cannot_relax_cpu_legacy_or_full_plan_profile_guards() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let descriptor = descriptor(source, 2);
    for profile in [
        ProgramNumericalProfile::GgufSourceV1,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    ] {
        let mut b = PlanBuilder::new_operator_slice(source, profile).unwrap();
        assert!(b.vllm_bf16_source_embedding(&descriptor).is_err());
        assert!(b.nodes.is_empty() && b.values.is_empty());
    }
    let mut full = PlanBuilder::new_with_profile(
        source,
        64,
        vec![StateSlot::Conv(ConvDomain(0))],
        false,
        PROFILE,
    )
    .unwrap();
    assert!(full.vllm_bf16_source_embedding(&descriptor).is_err());
    let mut b = builder(source);
    assert!(
        b.embed_profiled(
            descriptor.tensor().clone(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput
        )
        .is_err(),
        "CPU embedding contract must stay CPU-only"
    );
    assert!(b.nodes.is_empty());
    let (x, _) = inputs(&mut b, 1);
    assert!(
        b.project_profiled(
            descriptor.tensor().clone(),
            &x,
            ProjectionContract::TorchCpuBf16Avx2
        )
        .is_err()
    );
    assert!(
        b.ordinary_rms_profiled(parameter(source), &x, OrdinaryRmsContract::TorchCpuBf16Avx2)
            .is_err()
    );
    assert!(b.add("unqualified eager residual", &x, &x).is_err());
    assert!(b.advance_frontier().is_err());
}

#[test]
fn dense_inputs_refuse_foreign_builder_f32_wrong_length_and_unpacked_views() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let contract = ResidualRmsContract::Vllm031CudaAtenBf16H1024Rows1_2_64;
    let mut b = builder(source);
    let (x, prior) = inputs(&mut b, 2);
    let mut foreign = builder(source);
    let (other, _) = inputs(&mut foreign, 2);
    assert!(
        b.residual_rms_profiled(parameter(source), &other, &prior, contract)
            .is_err()
    );
    let (short, _) = inputs(&mut b, 1);
    assert!(
        b.residual_rms_profiled(parameter(source), &short, &prior, contract)
            .is_err()
    );
    // Obtain a real recorded F32 value through the unchanged GGUF recorder.
    let mut legacy =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1).unwrap();
    legacy
        .embed(
            descriptor(source, 1).tensor().clone(),
            PlanBuilder::inputs().token,
        )
        .unwrap();
    let f32_value = legacy
        .embed(
            descriptor(source, 1).tensor().clone(),
            PlanBuilder::inputs().token,
        )
        .unwrap();
    assert!(result_precision(&b, &parameter(source), contract, &[f32_value, x]).is_err());
    let (large, _) = inputs(&mut b, 64);
    let strided = b
        .view(
            &large,
            TensorView {
                offset: 0,
                rows: 2,
                width: 1024,
                row_stride: 2048,
            },
        )
        .unwrap();
    let before = b.nodes.len();
    assert!(
        matches!(b.residual_rms_profiled(parameter(source), &strided, &prior, contract), Err(Error::State(message)) if message.contains("contiguous"))
    );
    assert_eq!(b.nodes.len(), before);
    let packed = b.contiguous(&strided).unwrap();
    // Packing storage cannot mint a matching 2-row teacher issuer from64rows.
    assert!(
        b.residual_rms_profiled(parameter(source), &packed, &prior, contract)
            .is_err()
    );
}

#[test]
fn source_row_descriptor_refuses_wrong_geometry_issuer_and_teacher_vocabulary() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let valid = descriptor(source, 2);
    for wrong in [
        SourceTensor {
            source: SourceIdentity::for_new_allocation().unwrap(),
            ..valid.tensor().clone()
        },
        SourceTensor {
            name: "other.embedding".into(),
            ..valid.tensor().clone()
        },
        SourceTensor {
            kind: QuantKind::F32,
            ..valid.tensor().clone()
        },
        SourceTensor {
            columns: 2048,
            ..valid.tensor().clone()
        },
        SourceTensor {
            rows: 248_319,
            ..valid.tensor().clone()
        },
        SourceTensor {
            axes: MatrixAxes::InputOutput,
            ..valid.tensor().clone()
        },
    ] {
        assert!(VllmBf16Embedding::new(wrong, valid.teachers().clone()).is_err());
    }
    assert!(VllmBf16Teachers::new(source, 0, 64, &[11, 11, 11]).is_err());
    assert!(VllmBf16Teachers::new(source, 0, 64, &[248_320, 11]).is_err());
    let mut foreign = builder(SourceIdentity::for_new_allocation().unwrap());
    assert!(foreign.vllm_bf16_source_embedding(&valid).is_err());
}

#[test]
fn new_profile_single_teacher_preserves_existing_teacher_span_contract() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    assert!(TeacherSpan::new(source, 0, 64, 248_320, &[11]).is_err());
    let one = VllmBf16Teachers::new(source, 0, 64, &[11]).unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one.ids(), [11]);
    assert!(VllmBf16Teachers::new(source, 64, 64, &[11]).is_err());
    assert!(VllmBf16Teachers::new(source, 0, 64, &[248_320]).is_err());
    for rows in [2, 64] {
        let span = VllmBf16Teachers::new(source, 0, 64, &vec![11; rows]).unwrap();
        assert!(matches!(span.0, TeacherRows::Many(_)));
    }
}

#[test]
fn raw_gamma_descriptor_requires_original_backing_kind_and_issuer() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct GammaOwner {
        source: SourceIdentity,
        bytes: Vec<u8>,
        reads: AtomicUsize,
    }
    impl TensorSource for GammaOwner {
        fn source_identity(&self) -> SourceIdentity {
            self.source
        }
        fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            if name != POST_NORM {
                return Err(Error::Malformed(
                    "no embedding backing in gamma-only fixture".into(),
                ));
            }
            Ok(SourceTensorRef {
                kind: QuantKind::Bf16,
                shape: &[1024],
                packed_bytes: &self.bytes,
            })
        }
    }
    let owner = GammaOwner {
        source: SourceIdentity::for_new_allocation().unwrap(),
        bytes: vec![0; 2048],
        reads: AtomicUsize::new(0),
    };
    let original = SourceParameter::from_source(&owner, POST_NORM).unwrap();
    original.validate_source(&owner).unwrap();
    let changed = SourceParameter {
        kind: QuantKind::F32,
        bytes: 4096,
        ..original.clone()
    };
    assert!(
        ResidualRmsContract::Vllm031CudaAtenBf16H1024Rows1_2_64
            .validate(&changed, 1024)
            .is_ok()
    );
    assert!(
        changed.validate_source(&owner).is_err(),
        "valid metadata cannot replace original source dtype"
    );
    let foreign = GammaOwner {
        source: SourceIdentity::for_new_allocation().unwrap(),
        bytes: vec![0; 2048],
        reads: AtomicUsize::new(0),
    };
    assert!(original.validate_source(&foreign).is_err());
    assert_eq!(foreign.reads.load(Ordering::Relaxed), 0);
    assert!(
        descriptor(owner.source, 1).validate_source(&owner).is_err(),
        "metadata does not invent an embedding backing"
    );
}
