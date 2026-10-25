//! Structural fixtures only; no trained source, native eligibility or inference.
use super::*;
use crate::accelerator::program::typed_tests::UnawareBackend;

const WORKLOADS: [ShortHfWorkload; 5] = [
    ShortHfWorkload::Prompt47,
    ShortHfWorkload::Prompt57,
    ShortHfWorkload::Prompt62,
    ShortHfWorkload::Prompt67,
    ShortHfWorkload::Prompt72,
];
fn embedding(call: &ShortHfCall) -> VllmBf16Embedding {
    VllmBf16Embedding::for_call(
        SourceTensor {
            source: call.span().source(),
            name: "model.language_model.embed_tokens.weight".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows: 248_320,
            columns: call.model_profile().hidden_size(),
        },
        call.as_vllm_call(),
    )
    .unwrap()
}
fn rms(call: &ShortHfCall) -> Vllm08Operator {
    let h = call.model_profile().hidden_size();
    Vllm08Operator::new(
        call.as_vllm_call().clone(),
        Some(call.as_vllm_call().layer(0).unwrap()),
        Vllm08Kind::InputRms,
        Vec::new(),
        vec![SourceParameter {
            source: call.span().source(),
            name: "model.language_model.layers.0.input_layernorm.weight".into(),
            kind: QuantKind::Bf16,
            shape: vec![h as u64],
            bytes: h * 2,
        }],
    )
    .unwrap()
}
fn projection(call: &ShortHfCall, kind: Vllm08Projection) -> Vllm08Operator {
    let profile = call.model_profile();
    let members = match kind {
        Vllm08Projection::Qkvz => vec![
            ("linear_attn.in_proj_qkv.weight", 6144),
            ("linear_attn.in_proj_z.weight", 2048),
        ],
        Vllm08Projection::Ba => vec![
            ("linear_attn.in_proj_b.weight", 16),
            ("linear_attn.in_proj_a.weight", 16),
        ],
        Vllm08Projection::GdnOut => vec![("linear_attn.out_proj.weight", profile.hidden_size())],
        Vllm08Projection::AttentionQkv => vec![
            ("self_attn.q_proj.weight", 4096),
            ("self_attn.k_proj.weight", 512),
            ("self_attn.v_proj.weight", 512),
        ],
        Vllm08Projection::AttentionOut => vec![("self_attn.o_proj.weight", profile.hidden_size())],
        Vllm08Projection::GateUp => vec![
            ("mlp.gate_proj.weight", profile.intermediate_size()),
            ("mlp.up_proj.weight", profile.intermediate_size()),
        ],
        Vllm08Projection::Down => vec![("mlp.down_proj.weight", profile.hidden_size())],
        Vllm08Projection::Head => vec![("model.language_model.embed_tokens.weight", 248_320)],
    };
    let layer = if kind == Vllm08Projection::Head {
        None
    } else {
        Some(
            call.as_vllm_call()
                .layer(
                    if matches!(
                        kind,
                        Vllm08Projection::AttentionQkv | Vllm08Projection::AttentionOut
                    ) {
                        3
                    } else {
                        0
                    },
                )
                .unwrap(),
        )
    };
    let matrices = members
        .into_iter()
        .map(|(name, rows)| SourceTensor {
            source: call.span().source(),
            name: match layer {
                Some(layer) => format!("model.language_model.layers.{}.{name}", layer.index()),
                None => name.into(),
            },
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows,
            columns: profile.projection_input(kind),
        })
        .collect();
    Vllm08Operator::new(
        call.as_vllm_call().clone(),
        layer,
        Vllm08Kind::Projection(kind),
        matrices,
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn finite_actual_prefills_and_all_seven_cached_spans_preserve_issuer() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for workload in WORKLOADS {
        let rows = workload.prompt_rows();
        assert_eq!(ShortHfWorkload::from_prompt_rows(rows).unwrap(), workload);
        let ids = vec![11; rows];
        let span = ShortHfSpan::prefill(source, workload, &ids).unwrap();
        let clone = span.clone();
        assert_eq!(span, clone);
        assert_ne!(span, ShortHfSpan::prefill(source, workload, &ids).unwrap());
        assert_eq!((span.base(), span.len(), span.capacity()), (0, rows, 512));
        for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
            let call = ShortHfCall::new(profile, span.clone()).unwrap();
            assert_eq!(call.request().teachers().short_hf_span(), Some(&span));
            assert_eq!(call.request().teachers().ids(), &ids);
            assert_eq!(call.as_vllm_call().rows(), rows);
            assert!(call.request().legacy_phase().is_err());
            for base in workload.cached_bases() {
                let cached = ShortHfSpan::cached(source, workload, base, 353).unwrap();
                let call = ShortHfCall::new(profile, cached.clone()).unwrap();
                assert_eq!(call.span(), &cached);
                assert_eq!((call.span().base(), call.span().len()), (base, 1));
                assert!(call.request().legacy_phase().is_err());
            }
        }
        assert_eq!(
            *workload.cached_bases().end() + 1,
            workload.final_frontier()
        );
        assert!(workload.final_frontier() <= 79);
    }
}

#[test]
fn invalid_lengths_ids_and_cached_workload_boundaries_refuse_before_adoption() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for rows in [0, 1, 46, 48, 56, 58, 61, 63, 64, 66, 68, 71, 73, usize::MAX] {
        assert!(ShortHfWorkload::from_prompt_rows(rows).is_err());
    }
    for workload in WORKLOADS {
        let rows = workload.prompt_rows();
        for ids in [vec![11; rows - 1], vec![11; rows + 1], vec![248_320; rows]] {
            assert!(ShortHfSpan::prefill(source, workload, &ids).is_err());
        }
        assert!(
            ShortHfSpan::prefill(SourceIdentity::unbound(), workload, &vec![11; rows]).is_err()
        );
        for base in [0, rows - 1, rows + 7, usize::MAX] {
            assert!(ShortHfSpan::cached(source, workload, base, 11).is_err());
        }
        assert!(ShortHfSpan::cached(source, workload, rows, 248_320).is_err());
        assert!(TeacherSpan::new(source, 0, 512, 248_320, &vec![11; 67]).is_err());
    }
}

#[test]
fn short_cached66_cannot_be_retagged_as_legacy_phase_or_standalone_embedding() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let span = ShortHfSpan::cached(source, ShortHfWorkload::Prompt62, 66, 11).unwrap();
    let call = ShortHfCall::new(FullModelProfile::Qwen08, span).unwrap();
    let request = call.request();
    let layer = GdnPrefixLayer::new(source, 0).unwrap();
    assert!(request.legacy_phase().is_err());
    assert!(
        VllmBf16Invocation::new(layer, VllmBf16Phase::CachedOne, request.teachers().clone())
            .is_err()
    );
    assert!(
        VllmBf16Invocation::from_span(layer, VllmBf16Mode::Decode, request.teachers().clone())
            .is_err()
    );
    assert!(Vllm08Call::new(request.clone()).is_err());
    assert!(Vllm08Call::from_span(request.clone()).is_err());
    assert!(Vllm08Call::for_profile(FullModelProfile::Qwen2B, request.clone()).is_err());
    let raw = embedding(&call);
    assert!(VllmBf16Embedding::new(raw.tensor().clone(), request.teachers().clone()).is_err());
    assert!(PlanBuilder::new_vllm_gdn_attempt_phase(request.clone()).is_err());
    let mut slice = PlanBuilder::new_operator_slice(
        source,
        ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
    )
    .unwrap();
    assert!(slice.vllm_bf16_source_embedding(&raw).is_err());
    let old = VllmBf16Invocation::new(
        layer,
        VllmBf16Phase::CachedOne,
        VllmBf16Teachers::new(source, 66, 512, &[11]).unwrap(),
    )
    .unwrap();
    assert_eq!(old.legacy_phase().unwrap(), VllmBf16Phase::CachedOne);
}

#[test]
fn full_operator_shapes_use_true_rows_source_roles_and_family_widths() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        for workload in WORKLOADS {
            for span in [
                ShortHfSpan::prefill(source, workload, &vec![11; workload.prompt_rows()]).unwrap(),
                ShortHfSpan::cached(source, workload, *workload.cached_bases().end(), 353).unwrap(),
            ] {
                let call = ShortHfCall::new(profile, span).unwrap();
                let rows = call.span().len();
                assert_eq!(embedding(&call).rows(), rows);
                for kind in [
                    Vllm08Projection::Qkvz,
                    Vllm08Projection::Ba,
                    Vllm08Projection::GdnOut,
                    Vllm08Projection::AttentionQkv,
                    Vllm08Projection::AttentionOut,
                    Vllm08Projection::GateUp,
                    Vllm08Projection::Down,
                    Vllm08Projection::Head,
                ] {
                    let op = projection(&call, kind);
                    assert_eq!(
                        op.input_elements(),
                        vec![rows * profile.projection_input(kind)]
                    );
                    assert_eq!(
                        op.output_elements(),
                        vec![if kind == Vllm08Projection::Head {
                            248_320
                        } else {
                            rows * profile.projection_output(kind)
                        }]
                    );
                    assert_eq!(op.call(), call.as_vllm_call());
                    if kind == Vllm08Projection::Head {
                        assert_eq!(op.selected_row(), Some(rows - 1));
                    }
                }
                let gate = projection(&call, Vllm08Projection::GateUp);
                let mut reversed = gate.matrices().to_vec();
                reversed.swap(0, 1);
                assert!(
                    Vllm08Operator::new(
                        call.as_vllm_call().clone(),
                        gate.layer(),
                        gate.kind(),
                        reversed,
                        Vec::new()
                    )
                    .is_err()
                );
            }
        }
    }
}

#[test]
fn full_builder_keeps_actual_short_teacher_provenance_and_no_legacy_phase() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        for span in [
            ShortHfSpan::prefill(source, ShortHfWorkload::Prompt72, &[11; 72]).unwrap(),
            ShortHfSpan::cached(source, ShortHfWorkload::Prompt62, 66, 353).unwrap(),
        ] {
            let call = ShortHfCall::new(profile, span.clone()).unwrap();
            let mut builder = PlanBuilder::new_full_model(call.as_vllm_call().clone()).unwrap();
            let raw = builder
                .vllm_bf16_source_embedding(&embedding(&call))
                .unwrap();
            assert_eq!(raw.elements, span.len() * profile.hidden_size());
            let normal = builder.vllm08_operator(&rms(&call), &[raw]).unwrap()[0];
            let tag = builder.vllm_value_provenance(&normal).unwrap().unwrap();
            assert_eq!(tag.teachers().short_hf_span(), Some(&span));
            assert_eq!(tag.mode(), Some(span.mode()));
            assert_eq!(tag.phase(), None);
            tag.require_invocation(call.request()).unwrap();
            let other_span = ShortHfSpan::new(
                source,
                span.workload(),
                span.mode(),
                span.base(),
                span.ids(),
            )
            .unwrap();
            let foreign = ShortHfCall::new(profile, other_span).unwrap();
            assert!(tag.require_invocation(foreign.request()).is_err());
            let nodes = builder.nodes.len();
            assert!(builder.vllm08_operator(&rms(&foreign), &[raw]).is_err());
            assert_eq!(builder.nodes.len(), nodes);
        }
    }
}

#[test]
fn short_metadata_does_not_enable_an_unaware_backend() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = ShortHfCall::new(
        FullModelProfile::Qwen2B,
        ShortHfSpan::prefill(source, ShortHfWorkload::Prompt67, &[11; 67]).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        UnawareBackend.vllm_bf16_source_embedding(&embedding(&call)),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        UnawareBackend.vllm08_operator(&rms(&call), &[vec![0.0; 67 * 2048]]),
        Err(Error::Unsupported(_))
    ));
}
