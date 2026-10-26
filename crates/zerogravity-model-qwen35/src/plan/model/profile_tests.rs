//! Synthetic structural metadata only; no trained arithmetic/runtime proof.
use super::*;
use crate::plan::tests_support::UnawareBackend;

fn request(source: SourceIdentity, phase: ReferencePhase) -> QwenGdnCall {
    let base = match phase {
        ReferencePhase::CachedOne => 1,
        ReferencePhase::ContinuedTwo => 64,
        _ => 0,
    };
    QwenGdnCall::new(
        QwenPrefixLayer::new(source, 0).unwrap(),
        phase,
        QwenInputRows::new(source, base, 512, &vec![11; phase.rows()]).unwrap(),
    )
    .unwrap()
}
fn operator(call: &QwenSourceCall, projection: QwenProjection) -> QwenOperator {
    let layer = (projection != QwenProjection::Head).then(|| call.layer(0).unwrap());
    let matrices = projection
        .members_for_profile(call.model_profile())
        .iter()
        .copied()
        .map(|(suffix, rows)| SourceTensor {
            source: call.source(),
            name: layer.map_or_else(|| suffix.into(), |l| l.name(suffix)),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows,
            columns: call.model_profile().projection_input(projection),
        })
        .collect();
    QwenOperator::new(
        call.clone(),
        layer,
        QwenOperation::Projection(projection),
        matrices,
        Vec::new(),
    )
    .unwrap()
}
fn embedding(call: &QwenSourceCall) -> QwenEmbedding {
    QwenEmbedding::for_model(
        call.model_profile(),
        SourceTensor {
            source: call.source(),
            name: "model.language_model.embed_tokens.weight".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows: 248320,
            columns: call.model_profile().hidden_size(),
        },
        call.request().teachers().clone(),
    )
    .unwrap()
}
fn rms(call: &QwenSourceCall) -> QwenOperator {
    let h = call.model_profile().hidden_size();
    QwenOperator::new(
        call.clone(),
        Some(call.layer(0).unwrap()),
        QwenOperation::InputRms,
        Vec::new(),
        vec![SourceParameter {
            source: call.source(),
            name: "model.language_model.layers.0.input_layernorm.weight".into(),
            kind: QuantKind::Bf16,
            shape: vec![h as u64],
            bytes: h * 2,
        }],
    )
    .unwrap()
}

#[test]
fn old_constructor_and_projection_geometry_remain_qwen08() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = QwenSourceCall::new(request(source, ReferencePhase::InitialOne)).unwrap();
    assert_eq!(call.model_profile(), QwenSourceProfile::Qwen08);
    for p in [
        QwenProjection::Qkvz,
        QwenProjection::Ba,
        QwenProjection::GdnOut,
        QwenProjection::AttentionQkv,
        QwenProjection::AttentionOut,
        QwenProjection::GateUp,
        QwenProjection::Down,
        QwenProjection::Head,
    ] {
        assert_eq!(call.model_profile().projection_input(p), p.input_width());
        assert_eq!(call.model_profile().projection_output(p), p.output_width());
        assert_eq!(p.members_for_profile(call.model_profile()), p.members());
    }
}

#[test]
fn qwen2b_exact_source_members_and_truephase_head_dimensions() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for phase in [
        ReferencePhase::InitialOne,
        ReferencePhase::Initial64,
        ReferencePhase::ContinuedTwo,
        ReferencePhase::CachedOne,
    ] {
        let call =
            QwenSourceCall::for_profile(QwenSourceProfile::Qwen2B, request(source, phase)).unwrap();
        for (p, k, m) in [
            (QwenProjection::Qkvz, 2048, 8192),
            (QwenProjection::Ba, 2048, 32),
            (QwenProjection::GdnOut, 2048, 2048),
            (QwenProjection::GateUp, 2048, 12288),
            (QwenProjection::Down, 6144, 2048),
            (QwenProjection::Head, 2048, 248320),
        ] {
            let d = operator(&call, p);
            assert_eq!(d.input_elements(), vec![phase.rows() * k]);
            assert_eq!(
                d.output_elements(),
                vec![if p == QwenProjection::Head {
                    m
                } else {
                    phase.rows() * m
                }]
            );
            assert_eq!(d.call().request(), call.request());
            if p == QwenProjection::Head {
                assert_eq!(d.selected_row(), Some(phase.rows() - 1));
            }
        }
        let gate = operator(&call, QwenProjection::GateUp);
        assert_eq!(
            gate.matrices().iter().map(|m| m.rows).collect::<Vec<_>>(),
            vec![6144, 6144]
        );
        let mut reversed = gate.matrices().to_vec();
        reversed.swap(0, 1);
        assert!(
            QwenOperator::new(
                call.clone(),
                gate.layer(),
                gate.kind(),
                reversed,
                Vec::new()
            )
            .is_err()
        );
        for (p, layer) in [
            (QwenProjection::AttentionQkv, 3),
            (QwenProjection::AttentionOut, 3),
        ] {
            let layer = call.layer(layer).unwrap();
            let matrices = p
                .members_for_profile(call.model_profile())
                .iter()
                .copied()
                .map(|(s, rows)| SourceTensor {
                    source,
                    name: layer.name(s),
                    kind: QuantKind::Bf16,
                    axes: MatrixAxes::OutputInput,
                    rows,
                    columns: 2048,
                })
                .collect();
            assert!(
                QwenOperator::new(
                    call.clone(),
                    Some(layer),
                    QwenOperation::Projection(p),
                    matrices,
                    Vec::new()
                )
                .is_ok()
            );
        }
    }
}

#[test]
fn qwen2b_embedding_requires_matching_full_model_builder() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let request = request(source, ReferencePhase::InitialOne);
    let old = QwenSourceCall::new(request.clone()).unwrap();
    let two = QwenSourceCall::for_profile(QwenSourceProfile::Qwen2B, request).unwrap();
    let mut old_builder = QwenPlanBuilder::new_small_model(old.clone()).unwrap();
    assert!(old_builder.source_embedding(&embedding(&two)).is_err());
    assert!(QwenPlanBuilder::new_small_model(two.clone()).is_err());
    let mut two_builder = QwenPlanBuilder::new_model(two.clone()).unwrap();
    assert!(two_builder.source_embedding(&embedding(&old)).is_err());
    let mut slice =
        QwenPlanBuilder::new_operator_slice(source, ProgramNumericalProfile::SourceCudaSm86Bf16V1)
            .unwrap();
    assert!(slice.source_embedding(&embedding(&two)).is_err());
    let raw = two_builder.source_embedding(&embedding(&two)).unwrap();
    assert_eq!(raw.elements(), 2048);
    assert_eq!(raw.precision(), ActivationPrecision::Bf16WidenedF32);
    assert!(
        QwenEmbedding::new(
            embedding(&two).tensor().clone(),
            two.request().teachers().clone()
        )
        .is_err()
    );
}

#[test]
fn full_model_value_origin_preserves_profile_and_same_teacher_issuer() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = QwenSourceCall::for_profile(
        QwenSourceProfile::Qwen2B,
        request(source, ReferencePhase::InitialOne),
    )
    .unwrap();
    let mut builder = QwenPlanBuilder::new_model(call.clone()).unwrap();
    let raw = builder.source_embedding(&embedding(&call)).unwrap();
    let normalized = builder.source_operator(&rms(&call), &[raw]).unwrap()[0];
    let tag = builder.value_provenance(&normalized).unwrap().unwrap();
    let QwenValueOrigin::FullModel(origin) = tag.origin() else {
        panic!("missing actual full-model origin")
    };
    assert_eq!(origin.call().model_profile(), QwenSourceProfile::Qwen2B);
    assert_eq!(
        origin.call().request().teachers(),
        call.request().teachers()
    );
    let foreign = QwenSourceCall::for_profile(
        QwenSourceProfile::Qwen2B,
        request(source, ReferencePhase::InitialOne),
    )
    .unwrap();
    let before = (
        builder.core.nodes().len(),
        builder.core.values().len(),
        builder.provenance.clone(),
    );
    assert!(
        builder
            .source_operator(&operator(&foreign, QwenProjection::GateUp), &[normalized])
            .is_err()
    );
    assert_eq!(
        (
            builder.core.nodes().len(),
            builder.core.values().len(),
            builder.provenance.clone()
        ),
        before
    );
    let mut other = QwenPlanBuilder::new_model(call.clone()).unwrap();
    let foreign_value = other.source_embedding(&embedding(&call)).unwrap();
    assert!(
        builder
            .source_operator(&rms(&call), &[foreign_value])
            .is_err()
    );
}

#[test]
fn qwen2b_parameter_shape_and_layer_kind_do_not_accept_legacy_width() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = QwenSourceCall::for_profile(
        QwenSourceProfile::Qwen2B,
        request(source, ReferencePhase::InitialOne),
    )
    .unwrap();
    let d = rms(&call);
    let mut parameters = d.parameters().to_vec();
    parameters[0].shape = vec![1024];
    parameters[0].bytes = 2048;
    assert!(QwenOperator::new(call.clone(), d.layer(), d.kind(), Vec::new(), parameters).is_err());
    let qkvz = operator(&call, QwenProjection::Qkvz);
    assert!(
        QwenOperator::new(
            call.clone(),
            Some(call.layer(3).unwrap()),
            qkvz.kind(),
            qkvz.matrices().to_vec(),
            Vec::new()
        )
        .is_err()
    );
    assert!(
        QwenOperator::new(
            call.clone(),
            d.layer(),
            d.kind(),
            Vec::new(),
            vec![SourceParameter {
                kind: QuantKind::F32,
                bytes: 8192,
                ..d.parameters()[0].clone()
            }]
        )
        .is_err()
    );
}

#[test]
fn qwen2b_metadata_never_uses_legacy_backend_fallback() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = QwenSourceCall::for_profile(
        QwenSourceProfile::Qwen2B,
        request(source, ReferencePhase::InitialOne),
    )
    .unwrap();
    let d = operator(&call, QwenProjection::GateUp);
    assert!(matches!(
        UnawareBackend.source_operator(&d, &[vec![0.0; 2048]]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        UnawareBackend.source_embedding(&embedding(&call)),
        Err(Error::Unsupported(_))
    ));
}
