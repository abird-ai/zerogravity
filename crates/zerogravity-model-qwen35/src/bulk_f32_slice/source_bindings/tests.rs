use super::*;
use zerogravity_core::source::SourceTensorRef;

mod prefix {
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
        fn descriptor(&self, span: TeacherSpan) -> PrefixBinding {
            PrefixBinding::new(span, self, PrefixContract::First64GgmlF32V1, 1e-6).unwrap()
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
        assert!(PrefixBinding::new(span.clone(), &source, descriptor.contract(), 1e-5).is_err());
        let shifted = TeacherSpan::new(source.identity, 1, 8448, 16, &[11; 64]).unwrap();
        assert!(PrefixBinding::new(shifted, &source, descriptor.contract(), 1e-6).is_err());
        let short = TeacherSpan::new(source.identity, 0, 8448, 16, &[11; 2]).unwrap();
        assert!(PrefixBinding::new(short, &source, descriptor.contract(), 1e-6).is_err());
        let out_of_source = TeacherSpan::new(source.identity, 0, 8448, 17, &[16; 64]).unwrap();
        assert!(PrefixBinding::new(out_of_source, &source, descriptor.contract(), 1e-6).is_err());
        source.norm.pop();
        assert!(PrefixBinding::new(span.clone(), &source, descriptor.contract(), 1e-6).is_err());
        source.norm.push(0);
        source.embedding.pop();
        assert!(PrefixBinding::new(span.clone(), &source, descriptor.contract(), 1e-6).is_err());
        source.embedding.push(0);
        source.identity = SourceIdentity::for_new_allocation().unwrap();
        assert!(PrefixBinding::new(span, &source, descriptor.contract(), 1e-6).is_err());
    }

    #[test]
    fn prefix_records_two_nodes_and_refuses_foreign_issuers_builders_and_legacy_consumers() {
        let source = Original::new();
        let descriptor = source.descriptor(source.span());
        let foreign = source.descriptor(source.span());
        let mut builder =
            PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
                .unwrap();
        let embedding = one(&mut builder, descriptor.embedding_operator(), &[]).unwrap();
        assert!(one(&mut builder, foreign.rms_operator(), &[embedding]).is_err());
        let mut other =
            PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
                .unwrap();
        let foreign_value = one(&mut other, descriptor.embedding_operator(), &[]).unwrap();
        assert!(one(&mut builder, descriptor.rms_operator(), &[foreign_value]).is_err());
        assert!(
            builder
                .view(&embedding, TensorView::vector(0, 131072))
                .is_err()
        );
        assert!(builder.silu("legacy", &embedding, None).is_err());
        let normalized = one(&mut builder, descriptor.rms_operator(), &[embedding]).unwrap();
        assert!(one(&mut builder, descriptor.rms_operator(), &[normalized]).is_err());
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
        assert!(one(&mut full, descriptor.embedding_operator(), &[]).is_err());
        let mut bf16 = PlanBuilder::new_operator_slice(
            source.identity,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        assert!(one(&mut bf16, descriptor.embedding_operator(), &[]).is_err());
        let mut wrong_source = PlanBuilder::new_operator_slice(
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::GgufSourceV1,
        )
        .unwrap();
        assert!(one(&mut wrong_source, descriptor.embedding_operator(), &[]).is_err());
    }

    #[test]
    fn observed_equal_f32_words_cannot_replace_retained_prefix_origin() {
        let source = Original::new();
        let descriptor = source.descriptor(source.span());
        let right =
            BulkF32Operand::retained_normalized(descriptor.retained_rows().unwrap()).unwrap();
        assert!(
            matches!(right.origin(), BulkF32Origin::RetainedNormalized { binding } if binding == &descriptor.retained_rows().unwrap())
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
        let matrix =
            BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap();
        let mut builder =
            PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
                .unwrap();
        let a = builder.bulk_f32_input(matrix.left()).unwrap();
        let b = builder.bulk_f32_input(&observed).unwrap();
        assert!(builder.bulk_f32_matmul(&matrix, &a, &b).is_err());
        let embedding = one(&mut builder, descriptor.embedding_operator(), &[]).unwrap();
        assert!(builder.bulk_f32_matmul(&matrix, &a, &embedding).is_err());
        let actual = builder.bulk_f32_input(matrix.right()).unwrap();
        builder.bulk_f32_matmul(&matrix, &a, &actual).unwrap();
        let foreign = source.descriptor(source.span());
        assert_ne!(
            BulkF32Operand::retained_normalized(foreign.retained_rows().unwrap()).unwrap(),
            *matrix.right()
        );
    }
}

mod upstream {
    use super::*;
    struct Original {
        source: SourceIdentity,
        embedding: Vec<u8>,
        norm: Vec<u8>,
        qkv: Vec<u8>,
        z: Vec<u8>,
        conv: Vec<u8>,
    }
    impl Original {
        fn new() -> Self {
            Self {
                source: SourceIdentity::for_new_allocation().unwrap(),
                embedding: vec![0; 1680 * 16],
                norm: vec![0; 8192],
                qkv: vec![0; 1152 * 6144],
                z: vec![0; 1152 * 2048],
                conv: vec![0; 4 * 6144 * 4],
            }
        }
        fn descriptor(&self) -> GdnUpstreamBinding {
            let span = TeacherSpan::new(self.source, 0, 8448, 16, &[11; 64]).unwrap();
            let prefix =
                PrefixBinding::new(span, self, PrefixContract::First64GgmlF32V1, 1e-6).unwrap();
            GdnUpstreamBinding::new(prefix, self, GdnUpstreamContract::First64Q4RepackF32V1)
                .unwrap()
        }
    }
    impl TensorSource for Original {
        fn source_identity(&self) -> SourceIdentity {
            self.source
        }
        fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
            let (kind, shape, packed_bytes): (_, &[u64], &[u8]) = match name {
                "token_embd.weight" => (QuantKind::Q6K, &[2048, 16], &self.embedding),
                "blk.0.attn_norm.weight" => (QuantKind::F32, &[2048], &self.norm),
                "blk.0.attn_qkv.weight" => (QuantKind::Q4_0, &[2048, 6144], &self.qkv),
                "blk.0.attn_gate.weight" => (QuantKind::Q4_0, &[2048, 2048], &self.z),
                "blk.0.ssm_conv1d.weight" => (QuantKind::F32, &[4, 6144], &self.conv),
                _ => return Err(Error::Malformed("unknown original test tensor".into())),
            };
            Ok(SourceTensorRef {
                kind,
                shape,
                packed_bytes,
            })
        }
    }
    #[test]
    fn original_source_descriptor_refuses_changed_source_or_missing_raw_bytes() {
        let mut source = Original::new();
        let descriptor = source.descriptor();
        assert_eq!(descriptor.tensor(GdnProjection::Qkv).rows, 6144);
        assert_eq!(descriptor.tensor(GdnProjection::Z).rows, 2048);
        source.qkv.pop();
        assert!(
            GdnUpstreamBinding::new(descriptor.prefix().clone(), &source, descriptor.contract())
                .is_err()
        );
        source.qkv.push(0);
        source.source = SourceIdentity::for_new_allocation().unwrap();
        assert!(
            GdnUpstreamBinding::new(descriptor.prefix().clone(), &source, descriptor.contract())
                .is_err()
        );
    }
    #[test]
    fn five_node_slice_retains_exact_producers_and_refuses_legacy_views_or_role_swaps() {
        let source = Original::new();
        let descriptor = source.descriptor();
        let mut b =
            PlanBuilder::new_operator_slice(source.source, ProgramNumericalProfile::GgufSourceV1)
                .unwrap();
        let input = one(&mut b, descriptor.input_operator(), &[]).unwrap();
        assert!(b.silu("legacy", &input, None).is_err());
        assert!(b.view(&input, TensorView::vector(0, 131072)).is_err());
        let qkv = one(
            &mut b,
            descriptor.projection_operator(GdnProjection::Qkv),
            &[input],
        )
        .unwrap();
        let z = one(
            &mut b,
            descriptor.projection_operator(GdnProjection::Z),
            &[input],
        )
        .unwrap();
        assert!(one(&mut b, descriptor.convolution_operator(), &[z]).is_err());
        assert!(one(&mut b, descriptor.qk_operator(), &[qkv]).is_err());
        let conv = one(&mut b, descriptor.convolution_operator(), &[qkv]).unwrap();
        assert!(
            one(
                &mut b,
                descriptor.projection_operator(GdnProjection::Qkv),
                &[conv]
            )
            .is_err()
        );
        let operands = one(&mut b, descriptor.qk_operator(), &[conv]).unwrap();
        assert!(b.silu("legacy", &operands, None).is_err());
        let plan = b.finish_operator_slice(operands).unwrap();
        assert_eq!(plan.plan().nodes().len(), 5);
        assert!(plan.plan().required_domains().is_empty());
        assert!(plan.plan().proposed_domains().is_empty());
        assert!(plan.plan().logits().is_none());
        assert_eq!(plan.plan().frontier_delta(), 0);
    }
    #[test]
    fn equal_ids_foreign_issuers_or_builders_and_other_profiles_cannot_substitute() {
        let source = Original::new();
        let descriptor = source.descriptor();
        let foreign = source.descriptor();
        let mut b =
            PlanBuilder::new_operator_slice(source.source, ProgramNumericalProfile::GgufSourceV1)
                .unwrap();
        let input = one(&mut b, descriptor.input_operator(), &[]).unwrap();
        assert!(
            one(
                &mut b,
                foreign.projection_operator(GdnProjection::Qkv),
                &[input]
            )
            .is_err()
        );
        let mut other =
            PlanBuilder::new_operator_slice(source.source, ProgramNumericalProfile::GgufSourceV1)
                .unwrap();
        let value = one(&mut other, descriptor.input_operator(), &[]).unwrap();
        assert!(
            one(
                &mut b,
                descriptor.projection_operator(GdnProjection::Qkv),
                &[value]
            )
            .is_err()
        );
        let mut bf16 = PlanBuilder::new_operator_slice(
            source.source,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        assert!(one(&mut bf16, descriptor.input_operator(), &[]).is_err());
        let mut full = PlanBuilder::new(
            source.source,
            8448,
            vec![StateSlot::Conv(ConvDomain(0))],
            false,
        )
        .unwrap();
        assert!(one(&mut full, descriptor.input_operator(), &[]).is_err());
    }
}

mod output {
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
        fn descriptor(&self, span: TeacherSpan) -> Result<GdnOutputBinding> {
            GdnOutputBinding::new(
                span,
                self,
                "blk.0.ssm_out.weight",
                GdnOutputContract::OpenBlas033Zen16F32V1,
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
            GdnOutputContract::OpenBlas033Zen16F32V1
        );
        assert!(
            BulkF32Operand::source_matrix(span.clone(), &source, "blk.0.ssm_out.weight").is_err()
        );
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
        let input = one(&mut b, descriptor.input_operator(), &[]).unwrap();
        assert!(
            one(
                &mut b,
                same_ids_foreign_issuer.projection_operator(),
                &[input]
            )
            .is_err()
        );
        let mut other =
            PlanBuilder::new_operator_slice(source.identity, ProgramNumericalProfile::GgufSourceV1)
                .unwrap();
        let foreign = one(&mut other, descriptor.input_operator(), &[]).unwrap();
        assert!(one(&mut b, descriptor.projection_operator(), &[foreign]).is_err());
        assert!(b.silu("legacy", &input, None).is_err());
        assert!(b.view(&input, TensorView::vector(0, 131072)).is_err());
        let output = one(&mut b, descriptor.projection_operator(), &[input]).unwrap();
        assert!(one(&mut b, descriptor.projection_operator(), &[output]).is_err());
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
        assert!(one(&mut full, descriptor.input_operator(), &[]).is_err());
        let mut bf16 = PlanBuilder::new_operator_slice(
            source.identity,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        assert!(one(&mut bf16, descriptor.input_operator(), &[]).is_err());
        let mut other_source = PlanBuilder::new_operator_slice(
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::GgufSourceV1,
        )
        .unwrap();
        assert!(one(&mut other_source, descriptor.input_operator(), &[]).is_err());
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
        let retained = one(&mut b, descriptor.input_operator(), &[]).unwrap();
        assert!(one(&mut b, descriptor.projection_operator(), &[observed]).is_err());
        assert!(b.bulk_f32_matmul(&tiny, &a, &retained).is_err());
        assert!(one(&mut b, descriptor.projection_operator(), &[a]).is_err());
        b.bulk_f32_matmul(&tiny, &a, &observed).unwrap();
        one(&mut b, descriptor.projection_operator(), &[retained]).unwrap();
    }
}
