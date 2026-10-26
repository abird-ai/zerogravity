//! Structural fixtures only; no trained source, native eligibility or inference.
use super::*;
use crate::plan::tests_support::UnawareBackend;

const PROMPT_TOKEN_COUNTS: [usize; 5] = [47, 57, 62, 67, 72];
fn embedding(call: &PromptCall) -> QwenEmbedding {
    QwenEmbedding::for_call(
        SourceTensor {
            source: call.span().source(),
            name: "model.language_model.embed_tokens.weight".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows: 248_320,
            columns: call.model_profile().hidden_size(),
        },
        call.as_source_call(),
    )
    .unwrap()
}
fn rms(call: &PromptCall) -> QwenOperator {
    let h = call.model_profile().hidden_size();
    QwenOperator::new(
        call.as_source_call().clone(),
        Some(call.as_source_call().layer(0).unwrap()),
        QwenOperation::InputRms,
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
fn projection(call: &PromptCall, kind: QwenProjection) -> QwenOperator {
    let profile = call.model_profile();
    let members = match kind {
        QwenProjection::Qkvz => vec![
            ("linear_attn.in_proj_qkv.weight", 6144),
            ("linear_attn.in_proj_z.weight", 2048),
        ],
        QwenProjection::Ba => vec![
            ("linear_attn.in_proj_b.weight", 16),
            ("linear_attn.in_proj_a.weight", 16),
        ],
        QwenProjection::GdnOut => vec![("linear_attn.out_proj.weight", profile.hidden_size())],
        QwenProjection::AttentionQkv => vec![
            ("self_attn.q_proj.weight", 4096),
            ("self_attn.k_proj.weight", 512),
            ("self_attn.v_proj.weight", 512),
        ],
        QwenProjection::AttentionOut => vec![("self_attn.o_proj.weight", profile.hidden_size())],
        QwenProjection::GateUp => vec![
            ("mlp.gate_proj.weight", profile.intermediate_size()),
            ("mlp.up_proj.weight", profile.intermediate_size()),
        ],
        QwenProjection::Down => vec![("mlp.down_proj.weight", profile.hidden_size())],
        QwenProjection::Head => vec![("model.language_model.embed_tokens.weight", 248_320)],
    };
    let layer = if kind == QwenProjection::Head {
        None
    } else {
        Some(
            call.as_source_call()
                .layer(
                    if matches!(
                        kind,
                        QwenProjection::AttentionQkv | QwenProjection::AttentionOut
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
    QwenOperator::new(
        call.as_source_call().clone(),
        layer,
        QwenOperation::Projection(kind),
        matrices,
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn dynamic_prefills_and_cached_spans_preserve_issuer() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for prompt_tokens in [1, 47, 48, 49, 50, 73] {
        let workload = PromptWorkload::new(prompt_tokens, 512).unwrap();
        let rows = workload.prompt_tokens();
        assert_eq!(PromptWorkload::from_prompt_tokens(rows).unwrap(), workload);
        let ids = vec![11; rows];
        let span = PromptSpan::prefill(source, workload, &ids).unwrap();
        let clone = span.clone();
        assert_eq!(span, clone);
        assert_ne!(span, PromptSpan::prefill(source, workload, &ids).unwrap());
        assert_eq!((span.base(), span.len(), span.capacity()), (0, rows, 512));
        for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
            let call = PromptCall::new(profile, span.clone()).unwrap();
            assert_eq!(call.request().teachers().prompt_span(), Some(&span));
            assert_eq!(call.request().teachers().ids(), &ids);
            assert_eq!(call.as_source_call().rows(), rows);
            assert!(call.request().reference_phase().is_err());
            for base in workload.cached_bases() {
                let cached = PromptSpan::cached(source, workload, base, 353).unwrap();
                let call = PromptCall::new(profile, cached.clone()).unwrap();
                assert_eq!(call.span(), &cached);
                assert_eq!((call.span().base(), call.span().len()), (base, 1));
                assert!(call.request().reference_phase().is_err());
            }
        }
        assert_eq!(
            *workload.cached_bases().end() + 1,
            workload.final_frontier()
        );
        assert!(workload.final_frontier() <= workload.context_capacity_tokens());
    }
}

#[test]
fn invalid_ids_empty_spans_and_context_frontiers_refuse_before_adoption() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (prompt, capacity) in [(0, 512), (1, 0), (513, 512)] {
        assert!(PromptWorkload::new(prompt, capacity).is_err());
    }
    for prompt_tokens in [1, 48, 49, 50, 73, 511] {
        let workload = PromptWorkload::new(prompt_tokens, 512).unwrap();
        let rows = workload.prompt_tokens();
        for ids in [
            vec![11; rows.saturating_sub(1)],
            vec![11; rows + 1],
            vec![248_320; rows],
        ] {
            assert!(PromptSpan::prefill(source, workload, &ids).is_err());
        }
        assert!(PromptSpan::prefill(SourceIdentity::unbound(), workload, &vec![11; rows]).is_err());
        for base in [0, rows.saturating_sub(1), 512, usize::MAX] {
            assert!(PromptSpan::cached(source, workload, base, 11).is_err());
        }
        assert!(PromptSpan::cached(source, workload, rows, 248_320).is_err());
    }
    assert!(TeacherSpan::new(source, 0, 512, 248_320, &vec![11; 67]).is_err());
}

#[test]
fn short_cached66_cannot_be_retagged_as_reference_phase_or_standalone_embedding() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let span = PromptSpan::cached(
        source,
        PromptWorkload::from_prompt_tokens(62).unwrap(),
        66,
        11,
    )
    .unwrap();
    let call = PromptCall::new(QwenSourceProfile::Qwen08, span).unwrap();
    let request = call.request();
    let layer = QwenPrefixLayer::new(source, 0).unwrap();
    assert!(request.reference_phase().is_err());
    assert!(
        QwenGdnCall::new(layer, ReferencePhase::CachedOne, request.teachers().clone()).is_err()
    );
    assert!(
        QwenGdnCall::from_span(layer, SourceCallMode::Decode, request.teachers().clone()).is_err()
    );
    assert!(QwenSourceCall::new(request.clone()).is_err());
    assert!(QwenSourceCall::from_span(request.clone()).is_err());
    assert!(QwenSourceCall::for_profile(QwenSourceProfile::Qwen2B, request.clone()).is_err());
    let raw = embedding(&call);
    assert!(QwenEmbedding::new(raw.tensor().clone(), request.teachers().clone()).is_err());
    assert!(QwenPlanBuilder::new_gdn_attempt_phase(request.clone()).is_err());
    let mut slice =
        QwenPlanBuilder::new_operator_slice(source, ProgramNumericalProfile::SourceCudaSm86Bf16V1)
            .unwrap();
    assert!(slice.source_embedding(&raw).is_err());
    let old = QwenGdnCall::new(
        layer,
        ReferencePhase::CachedOne,
        QwenInputRows::new(source, 66, 512, &[11]).unwrap(),
    )
    .unwrap();
    assert_eq!(old.reference_phase().unwrap(), ReferencePhase::CachedOne);
}

#[test]
fn full_operator_shapes_use_true_rows_source_roles_and_family_widths() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        for prompt_tokens in PROMPT_TOKEN_COUNTS {
            let workload = PromptWorkload::from_prompt_tokens(prompt_tokens).unwrap();
            for span in [
                PromptSpan::prefill(source, workload, &vec![11; workload.prompt_tokens()]).unwrap(),
                PromptSpan::cached(source, workload, *workload.cached_bases().end(), 353).unwrap(),
            ] {
                let call = PromptCall::new(profile, span).unwrap();
                let rows = call.span().len();
                assert_eq!(embedding(&call).rows(), rows);
                for kind in [
                    QwenProjection::Qkvz,
                    QwenProjection::Ba,
                    QwenProjection::GdnOut,
                    QwenProjection::AttentionQkv,
                    QwenProjection::AttentionOut,
                    QwenProjection::GateUp,
                    QwenProjection::Down,
                    QwenProjection::Head,
                ] {
                    let op = projection(&call, kind);
                    assert_eq!(
                        op.input_elements(),
                        vec![rows * profile.projection_input(kind)]
                    );
                    assert_eq!(
                        op.output_elements(),
                        vec![if kind == QwenProjection::Head {
                            248_320
                        } else {
                            rows * profile.projection_output(kind)
                        }]
                    );
                    assert_eq!(op.call(), call.as_source_call());
                    if kind == QwenProjection::Head {
                        assert_eq!(op.selected_row(), Some(rows - 1));
                    }
                }
                let gate = projection(&call, QwenProjection::GateUp);
                let mut reversed = gate.matrices().to_vec();
                reversed.swap(0, 1);
                assert!(
                    QwenOperator::new(
                        call.as_source_call().clone(),
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
fn full_builder_keeps_actual_short_teacher_provenance_and_no_reference_phase() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        for span in [
            PromptSpan::prefill(
                source,
                PromptWorkload::from_prompt_tokens(72).unwrap(),
                &[11; 72],
            )
            .unwrap(),
            PromptSpan::cached(
                source,
                PromptWorkload::from_prompt_tokens(62).unwrap(),
                66,
                353,
            )
            .unwrap(),
        ] {
            let call = PromptCall::new(profile, span.clone()).unwrap();
            let mut builder = QwenPlanBuilder::new_model(call.as_source_call().clone()).unwrap();
            let raw = builder.source_embedding(&embedding(&call)).unwrap();
            assert_eq!(raw.elements(), span.len() * profile.hidden_size());
            let normal = builder.source_operator(&rms(&call), &[raw]).unwrap()[0];
            let tag = builder.value_provenance(&normal).unwrap().unwrap();
            assert_eq!(tag.teachers().prompt_span(), Some(&span));
            assert_eq!(tag.mode(), Some(span.mode()));
            assert_eq!(tag.phase(), None);
            tag.require_invocation(call.request()).unwrap();
            let other_span = PromptSpan::new(
                source,
                span.workload(),
                span.mode(),
                span.base(),
                span.ids(),
            )
            .unwrap();
            let foreign = PromptCall::new(profile, other_span).unwrap();
            assert!(tag.require_invocation(foreign.request()).is_err());
            let nodes = builder.core.nodes().len();
            assert!(builder.source_operator(&rms(&foreign), &[raw]).is_err());
            assert_eq!(builder.core.nodes().len(), nodes);
        }
    }
}

#[test]
fn short_metadata_does_not_enable_an_unaware_backend() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = PromptCall::new(
        QwenSourceProfile::Qwen2B,
        PromptSpan::prefill(
            source,
            PromptWorkload::from_prompt_tokens(67).unwrap(),
            &[11; 67],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        UnawareBackend.source_embedding(&embedding(&call)),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        UnawareBackend.source_operator(&rms(&call), &[vec![0.0; 67 * 2048]]),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn positive_outputs_and_checked_context_frontier_precede_backend_admission() {
    for n in [1, 47, 48, 49, 50, 73, 511] {
        let workload = PromptWorkload::new(n, 512).unwrap();
        let mut ids = vec![248_319; n];
        workload.validate_request(&ids, 512 - n, 248_320).unwrap();
        assert!(workload.validate_request(&ids, 0, 248_320).is_err());
        assert!(
            workload
                .validate_request(&ids, 513usize.saturating_sub(n), 248_320)
                .is_err()
        );
        ids[n - 1] = 248_320;
        assert!(workload.validate_request(&ids, 1, 248_320).is_err());
    }
}
