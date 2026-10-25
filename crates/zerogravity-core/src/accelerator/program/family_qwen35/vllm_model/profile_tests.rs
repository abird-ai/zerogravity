//! Synthetic structural metadata only; no trained arithmetic/runtime proof.
use super::*;
use crate::accelerator::program::typed_tests::UnawareBackend;

fn request(source: SourceIdentity, phase: VllmBf16Phase) -> VllmBf16Invocation {
    let base = match phase {
        VllmBf16Phase::CachedOne => 1,
        VllmBf16Phase::ContinuedTwo => 64,
        _ => 0,
    };
    VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        phase,
        VllmBf16Teachers::new(source, base, 512, &vec![11; phase.rows()]).unwrap(),
    )
    .unwrap()
}
fn operator(call: &Vllm08Call, projection: Vllm08Projection) -> Vllm08Operator {
    let layer = (projection != Vllm08Projection::Head).then(|| call.layer(0).unwrap());
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
    Vllm08Operator::new(
        call.clone(),
        layer,
        Vllm08Kind::Projection(projection),
        matrices,
        Vec::new(),
    )
    .unwrap()
}
fn embedding(call: &Vllm08Call) -> VllmBf16Embedding {
    VllmBf16Embedding::for_model(
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
fn rms(call: &Vllm08Call) -> Vllm08Operator {
    let h = call.model_profile().hidden_size();
    Vllm08Operator::new(
        call.clone(),
        Some(call.layer(0).unwrap()),
        Vllm08Kind::InputRms,
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
    let call = Vllm08Call::new(request(source, VllmBf16Phase::InitialOne)).unwrap();
    assert_eq!(call.model_profile(), FullModelProfile::Qwen08);
    for p in [
        Vllm08Projection::Qkvz,
        Vllm08Projection::Ba,
        Vllm08Projection::GdnOut,
        Vllm08Projection::AttentionQkv,
        Vllm08Projection::AttentionOut,
        Vllm08Projection::GateUp,
        Vllm08Projection::Down,
        Vllm08Projection::Head,
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
        VllmBf16Phase::InitialOne,
        VllmBf16Phase::Initial64,
        VllmBf16Phase::ContinuedTwo,
        VllmBf16Phase::CachedOne,
    ] {
        let call =
            Vllm08Call::for_profile(FullModelProfile::Qwen2B, request(source, phase)).unwrap();
        for (p, k, m) in [
            (Vllm08Projection::Qkvz, 2048, 8192),
            (Vllm08Projection::Ba, 2048, 32),
            (Vllm08Projection::GdnOut, 2048, 2048),
            (Vllm08Projection::GateUp, 2048, 12288),
            (Vllm08Projection::Down, 6144, 2048),
            (Vllm08Projection::Head, 2048, 248320),
        ] {
            let d = operator(&call, p);
            assert_eq!(d.input_elements(), vec![phase.rows() * k]);
            assert_eq!(
                d.output_elements(),
                vec![if p == Vllm08Projection::Head {
                    m
                } else {
                    phase.rows() * m
                }]
            );
            assert_eq!(d.call().request(), call.request());
            if p == Vllm08Projection::Head {
                assert_eq!(d.selected_row(), Some(phase.rows() - 1));
            }
        }
        let gate = operator(&call, Vllm08Projection::GateUp);
        assert_eq!(
            gate.matrices().iter().map(|m| m.rows).collect::<Vec<_>>(),
            vec![6144, 6144]
        );
        let mut reversed = gate.matrices().to_vec();
        reversed.swap(0, 1);
        assert!(
            Vllm08Operator::new(
                call.clone(),
                gate.layer(),
                gate.kind(),
                reversed,
                Vec::new()
            )
            .is_err()
        );
        for (p, layer) in [
            (Vllm08Projection::AttentionQkv, 3),
            (Vllm08Projection::AttentionOut, 3),
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
                Vllm08Operator::new(
                    call.clone(),
                    Some(layer),
                    Vllm08Kind::Projection(p),
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
    let request = request(source, VllmBf16Phase::InitialOne);
    let old = Vllm08Call::new(request.clone()).unwrap();
    let two = Vllm08Call::for_profile(FullModelProfile::Qwen2B, request).unwrap();
    let mut old_builder = PlanBuilder::new_vllm08_model(old.clone()).unwrap();
    assert!(
        old_builder
            .vllm_bf16_source_embedding(&embedding(&two))
            .is_err()
    );
    assert!(PlanBuilder::new_vllm08_model(two.clone()).is_err());
    let mut two_builder = PlanBuilder::new_full_model(two.clone()).unwrap();
    assert!(
        two_builder
            .vllm_bf16_source_embedding(&embedding(&old))
            .is_err()
    );
    let mut slice = PlanBuilder::new_operator_slice(
        source,
        ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
    )
    .unwrap();
    assert!(slice.vllm_bf16_source_embedding(&embedding(&two)).is_err());
    let raw = two_builder
        .vllm_bf16_source_embedding(&embedding(&two))
        .unwrap();
    assert_eq!(raw.elements, 2048);
    assert_eq!(raw.precision(), ActivationPrecision::Bf16WidenedF32);
    assert!(
        VllmBf16Embedding::new(
            embedding(&two).tensor().clone(),
            two.request().teachers().clone()
        )
        .is_err()
    );
}

#[test]
fn full_model_value_origin_preserves_profile_and_same_teacher_issuer() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = Vllm08Call::for_profile(
        FullModelProfile::Qwen2B,
        request(source, VllmBf16Phase::InitialOne),
    )
    .unwrap();
    let mut builder = PlanBuilder::new_full_model(call.clone()).unwrap();
    let raw = builder
        .vllm_bf16_source_embedding(&embedding(&call))
        .unwrap();
    let normalized = builder.vllm08_operator(&rms(&call), &[raw]).unwrap()[0];
    let tag = builder.vllm_value_provenance(&normalized).unwrap().unwrap();
    let VllmValueOrigin::FullModel(origin) = tag.origin() else {
        panic!("missing actual full-model origin")
    };
    assert_eq!(origin.call().model_profile(), FullModelProfile::Qwen2B);
    assert_eq!(
        origin.call().request().teachers(),
        call.request().teachers()
    );
    let foreign = Vllm08Call::for_profile(
        FullModelProfile::Qwen2B,
        request(source, VllmBf16Phase::InitialOne),
    )
    .unwrap();
    let before = (
        builder.nodes.len(),
        builder.values.len(),
        builder.vllm_provenance.clone(),
    );
    assert!(
        builder
            .vllm08_operator(&operator(&foreign, Vllm08Projection::GateUp), &[normalized])
            .is_err()
    );
    assert_eq!(
        (
            builder.nodes.len(),
            builder.values.len(),
            builder.vllm_provenance.clone()
        ),
        before
    );
    let mut other = PlanBuilder::new_full_model(call.clone()).unwrap();
    let foreign_value = other.vllm_bf16_source_embedding(&embedding(&call)).unwrap();
    assert!(
        builder
            .vllm08_operator(&rms(&call), &[foreign_value])
            .is_err()
    );
}

#[test]
fn qwen2b_parameter_shape_and_layer_kind_do_not_accept_legacy_width() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = Vllm08Call::for_profile(
        FullModelProfile::Qwen2B,
        request(source, VllmBf16Phase::InitialOne),
    )
    .unwrap();
    let d = rms(&call);
    let mut parameters = d.parameters().to_vec();
    parameters[0].shape = vec![1024];
    parameters[0].bytes = 2048;
    assert!(
        Vllm08Operator::new(call.clone(), d.layer(), d.kind(), Vec::new(), parameters).is_err()
    );
    let qkvz = operator(&call, Vllm08Projection::Qkvz);
    assert!(
        Vllm08Operator::new(
            call.clone(),
            Some(call.layer(3).unwrap()),
            qkvz.kind(),
            qkvz.matrices().to_vec(),
            Vec::new()
        )
        .is_err()
    );
    assert!(
        Vllm08Operator::new(
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
    let call = Vllm08Call::for_profile(
        FullModelProfile::Qwen2B,
        request(source, VllmBf16Phase::InitialOne),
    )
    .unwrap();
    let d = operator(&call, Vllm08Projection::GateUp);
    assert!(matches!(
        UnawareBackend.vllm08_operator(&d, &[vec![0.0; 2048]]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        UnawareBackend.vllm_bf16_source_embedding(&embedding(&call)),
        Err(Error::Unsupported(_))
    ));
}
