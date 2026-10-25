use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn invocation(source: SourceIdentity, layer: usize, phase: VllmBf16Phase) -> VllmBf16Invocation {
    let base = match phase {
        VllmBf16Phase::InitialOne | VllmBf16Phase::Initial64 => 0,
        VllmBf16Phase::CachedOne => 66,
        VllmBf16Phase::ContinuedTwo => 64,
    };
    VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, layer).unwrap(),
        phase,
        VllmBf16Teachers::new(source, base, 512, &vec![11; phase.rows()]).unwrap(),
    )
    .unwrap()
}
fn members(call: &VllmBf16Invocation, kind: VllmGdnProjectionKind) -> Vec<SourceTensor> {
    kind.members()
        .iter()
        .map(|&(suffix, rows)| SourceTensor {
            source: call.layer.source(),
            name: format!(
                "model.language_model.layers.{}.{suffix}",
                call.layer.index()
            ),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            columns: kind.input_width(),
            rows,
        })
        .collect()
}
fn projection(call: &VllmBf16Invocation, kind: VllmGdnProjectionKind) -> VllmGdnProjection {
    VllmGdnProjection::new(call.clone(), kind, members(call, kind)).unwrap()
}
fn embedding(call: &VllmBf16Invocation) -> VllmBf16Embedding {
    VllmBf16Embedding::new(
        SourceTensor {
            source: call.layer.source(),
            name: "model.language_model.embed_tokens.weight".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            columns: 1024,
            rows: 248320,
        },
        call.teachers.clone(),
    )
    .unwrap()
}
fn builder(source: SourceIdentity) -> PlanBuilder {
    PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1)
        .unwrap()
}

#[test]
fn exact_projection_roles_keep_original_order_axes_widths_and_layers() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for layer in 0..3 {
        for phase in [
            VllmBf16Phase::InitialOne,
            VllmBf16Phase::CachedOne,
            VllmBf16Phase::Initial64,
            VllmBf16Phase::ContinuedTwo,
        ] {
            let call = invocation(source, layer, phase);
            for kind in [
                VllmGdnProjectionKind::Qkvz,
                VllmGdnProjectionKind::Ba,
                VllmGdnProjectionKind::GdnOut,
                VllmGdnProjectionKind::GateUp,
                VllmGdnProjectionKind::Down,
            ] {
                let original = members(&call, kind);
                let descriptor = projection(&call, kind);
                assert_eq!(descriptor.members(), original);
                assert_eq!(
                    descriptor.input_elements(),
                    phase.rows() * kind.input_width()
                );
                assert_eq!(
                    descriptor.output_elements(),
                    phase.rows() * kind.output_width()
                );
                assert_eq!(
                    descriptor.members().iter().map(|t| t.rows).sum::<usize>(),
                    kind.output_width()
                );
                if original.len() == 2 {
                    let mut wrong = original.clone();
                    wrong.swap(0, 1);
                    assert!(VllmGdnProjection::new(call.clone(), kind, wrong).is_err());
                }
                for fault in 0..6 {
                    let mut wrong = original.clone();
                    match fault {
                        0 => wrong[0].source = SourceIdentity::for_new_allocation().unwrap(),
                        1 => wrong[0].axes = MatrixAxes::InputOutput,
                        2 => wrong[0].kind = QuantKind::F32,
                        3 => wrong[0].rows += 1,
                        4 => wrong[0].columns += 1,
                        _ => {
                            wrong[0].name = format!(
                                "model.language_model.layers.{layer}.fabricated.qkvz.weight"
                            )
                        }
                    }
                    assert!(
                        VllmGdnProjection::new(call.clone(), kind, wrong).is_err(),
                        "{kind:?}/{fault}"
                    );
                }
                assert!(VllmGdnProjection::new(call.clone(), kind, Vec::new()).is_err());
                let mut excess = Vec::with_capacity(8);
                excess.extend(original);
                assert!(VllmGdnProjection::new(call.clone(), kind, excess).is_err());
            }
        }
    }
}

#[test]
fn genuine_phase_cannot_turn_continued_two_into_decode_or_cross_source() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let key = GdnPrefixLayer::new(source, 0).unwrap();
    for (phase, base, rows) in [
        (VllmBf16Phase::InitialOne, 1, 1),
        (VllmBf16Phase::CachedOne, 0, 1),
        (VllmBf16Phase::CachedOne, 64, 2),
        (VllmBf16Phase::Initial64, 1, 64),
        (VllmBf16Phase::ContinuedTwo, 0, 2),
    ] {
        assert!(
            VllmBf16Invocation::new(
                key,
                phase,
                VllmBf16Teachers::new(source, base, 512, &vec![11; rows]).unwrap()
            )
            .is_err()
        );
    }
    let foreign = SourceIdentity::for_new_allocation().unwrap();
    assert!(
        VllmBf16Invocation::new(
            key,
            VllmBf16Phase::InitialOne,
            VllmBf16Teachers::new(foreign, 0, 512, &[11]).unwrap()
        )
        .is_err()
    );
}

#[test]
fn single_and_multi_teacher_clones_keep_issuer_equal_ids_do_not_mint_it() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for rows in [1, 2, 64] {
        let a = VllmBf16Teachers::new(source, 0, 512, &vec![11; rows]).unwrap();
        assert_eq!(a, a.clone());
        let b = VllmBf16Teachers::new(source, 0, 512, &vec![11; rows]).unwrap();
        assert_eq!(a.ids(), b.ids());
        assert_ne!(a, b);
    }
}

struct OriginalBa {
    source: SourceIdentity,
    first: Vec<u8>,
    second: Vec<u8>,
    reads: AtomicUsize,
}
impl TensorSource for OriginalBa {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let raw = match name {
            "model.language_model.layers.0.linear_attn.in_proj_b.weight" => &self.first,
            "model.language_model.layers.0.linear_attn.in_proj_a.weight" => &self.second,
            _ => return Err(Error::State("unknown synthetic original Ba role".into())),
        };
        Ok(SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[16, 1024],
            packed_bytes: raw,
        })
    }
}
#[test]
fn original_member_source_revalidation_refuses_foreign_before_any_read_and_bad_backing() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = invocation(source, 0, VllmBf16Phase::InitialOne);
    let descriptor = projection(&call, VllmGdnProjectionKind::Ba);
    let mut original = OriginalBa {
        source,
        first: vec![0; 32768],
        second: vec![0; 32768],
        reads: AtomicUsize::new(0),
    };
    descriptor.validate_source(&original).unwrap();
    assert_eq!(original.reads.load(Ordering::Relaxed), 2);
    let foreign = OriginalBa {
        source: SourceIdentity::for_new_allocation().unwrap(),
        first: vec![],
        second: vec![],
        reads: AtomicUsize::new(0),
    };
    assert!(descriptor.validate_source(&foreign).is_err());
    assert_eq!(foreign.reads.load(Ordering::Relaxed), 0);
    original.second.pop();
    assert!(descriptor.validate_source(&original).is_err());
}

#[test]
fn same_plan_embedding_dual_rms_projection_activation_preserve_real_teacher_tag() {
    for layer in 0..3 {
        for phase in [
            VllmBf16Phase::InitialOne,
            VllmBf16Phase::CachedOne,
            VllmBf16Phase::Initial64,
            VllmBf16Phase::ContinuedTwo,
        ] {
            let source = SourceIdentity::for_new_allocation().unwrap();
            let call = invocation(source, layer, phase);
            let mut b = builder(source);
            let e = embedding(&call);
            let branch = b.vllm_bf16_source_embedding(&e).unwrap();
            let prior = b.vllm_bf16_source_embedding(&e).unwrap();
            let binding = call.layer.rms(GdnPrefixRmsSite::PostAttention).unwrap();
            let gamma = SourceParameter {
                source,
                name: binding.gamma_name(),
                kind: QuantKind::Bf16,
                shape: vec![1024],
                bytes: 2048,
            };
            let rms = b
                .residual_rms_profiled(
                    gamma,
                    &branch,
                    &prior,
                    ResidualRmsContract::Vllm031CudaGdnPrefix(binding),
                )
                .unwrap();
            let gate = b
                .vllm_gdn_project(
                    &projection(&call, VllmGdnProjectionKind::GateUp),
                    &rms.normalized,
                )
                .unwrap();
            let activated = b
                .vllm_silu_and_mul(&VllmSiluAndMul::new(call.clone()).unwrap(), &gate)
                .unwrap();
            let down = b
                .vllm_gdn_project(&projection(&call, VllmGdnProjectionKind::Down), &activated)
                .unwrap();
            for value in [
                branch,
                prior,
                rms.normalized,
                rms.residual,
                gate,
                activated,
                down,
            ] {
                assert_eq!(value.precision(), ActivationPrecision::Bf16WidenedF32);
                let p = b.vllm_value_provenance(&value).unwrap().unwrap();
                assert_eq!(p.teachers(), call.teachers());
                assert_eq!(p.phase(), Some(phase));
            }
            let slice = b.finish_operator_slice(down).unwrap();
            let plan = slice.plan();
            assert_eq!(plan.nodes().len(), 6);
            assert_eq!(plan.values().len(), 7);
            assert!(plan.nodes().iter().all(|node| node.state.is_none()));
            assert!(plan.required_domains().is_empty() && plan.proposed_domains().is_empty());
            assert_eq!(plan.frontier_delta(), 0);
            for id in 0..7 {
                assert!(plan.vllm_value_provenance(ValueId(id)).unwrap().is_some());
            }
            assert!(plan.vllm_value_provenance(ValueId(7)).is_err());
        }
    }
}

#[test]
fn same_builder_wrong_teacher_or_phase_foreign_plan_and_generic_pack_refuse_without_mutation() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for phase in [
        VllmBf16Phase::InitialOne,
        VllmBf16Phase::Initial64,
        VllmBf16Phase::ContinuedTwo,
    ] {
        let a = invocation(source, 0, phase);
        let equal_ids_new_issuer = invocation(source, 0, phase);
        let mut b = builder(source);
        let input = b.vllm_bf16_source_embedding(&embedding(&a)).unwrap();
        let other = b
            .vllm_bf16_source_embedding(&embedding(&equal_ids_new_issuer))
            .unwrap();
        let mut foreign = builder(source);
        let value = foreign.vllm_bf16_source_embedding(&embedding(&a)).unwrap();
        let snapshot = (
            b.nodes.len(),
            b.values.len(),
            b.precisions.len(),
            b.vllm_provenance.len(),
            b.constants.len(),
            b.domains.len(),
        );
        for (descriptor, value) in [
            (projection(&a, VllmGdnProjectionKind::Qkvz), other),
            (projection(&a, VllmGdnProjectionKind::Qkvz), value),
            (projection(&a, VllmGdnProjectionKind::GdnOut), input),
        ] {
            assert!(b.vllm_gdn_project(&descriptor, &value).is_err());
            assert_eq!(
                snapshot,
                (
                    b.nodes.len(),
                    b.values.len(),
                    b.precisions.len(),
                    b.vllm_provenance.len(),
                    b.constants.len(),
                    b.domains.len()
                )
            );
        }
        if phase == VllmBf16Phase::InitialOne {
            let cached = invocation(source, 0, VllmBf16Phase::CachedOne);
            assert!(
                b.vllm_gdn_project(&projection(&cached, VllmGdnProjectionKind::Qkvz), &input)
                    .is_err()
            );
        }
        let foreign_source = invocation(SourceIdentity::for_new_allocation().unwrap(), 0, phase);
        assert!(
            b.vllm_gdn_project(
                &projection(&foreign_source, VllmGdnProjectionKind::Qkvz),
                &input
            )
            .is_err()
        );
        assert!(
            b.vllm_silu_and_mul(&VllmSiluAndMul::new(a.clone()).unwrap(), &input)
                .is_err()
        );
        assert_eq!(
            snapshot,
            (
                b.nodes.len(),
                b.values.len(),
                b.precisions.len(),
                b.vllm_provenance.len(),
                b.constants.len(),
                b.domains.len()
            )
        );
        let packed = b.contiguous(&input).unwrap();
        assert!(b.vllm_value_provenance(&packed).unwrap().is_none());
        assert!(
            b.vllm_gdn_project(&projection(&a, VllmGdnProjectionKind::Qkvz), &packed)
                .is_err()
        );
        let gamma = SourceParameter {
            source,
            name: "model.language_model.layers.0.post_attention_layernorm.weight".into(),
            kind: QuantKind::Bf16,
            shape: vec![1024],
            bytes: 2048,
        };
        assert!(
            b.residual_rms_profiled(
                gamma,
                &input,
                &other,
                ResidualRmsContract::Vllm031CudaAtenBf16H1024Rows1_2_64
            )
            .is_err()
        );
    }
}

#[test]
fn source_and_old_profile_never_fall_into_cuda_contract_or_mint_provenance() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = invocation(source, 0, VllmBf16Phase::InitialOne);
    for profile in [
        ProgramNumericalProfile::GgufSourceV1,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    ] {
        let mut b = PlanBuilder::new_operator_slice(source, profile).unwrap();
        let tensor = embedding(&call).tensor().clone();
        let input = if profile == ProgramNumericalProfile::GgufSourceV1 {
            b.embed(tensor, PlanBuilder::inputs().token).unwrap()
        } else {
            b.embed_profiled(
                tensor,
                PlanBuilder::inputs().token,
                EmbeddingContract::Bf16OutputInput,
            )
            .unwrap()
        };
        assert!(b.vllm_value_provenance(&input).unwrap().is_none());
        assert!(
            b.vllm_gdn_project(&projection(&call, VllmGdnProjectionKind::GateUp), &input)
                .is_err()
        );
        assert_eq!(b.nodes.len(), 1);
    }
}
