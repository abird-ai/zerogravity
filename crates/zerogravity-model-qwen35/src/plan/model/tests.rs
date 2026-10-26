//! Structural refusals only; no source math/native/profile qualification.
use super::*;

fn call(source: SourceIdentity, phase: ReferencePhase) -> QwenSourceCall {
    let base = match phase {
        ReferencePhase::CachedOne => 1,
        ReferencePhase::ContinuedTwo => 64,
        _ => 0,
    };
    QwenSourceCall::new(
        QwenGdnCall::new(
            QwenPrefixLayer::new(source, 0).unwrap(),
            phase,
            QwenInputRows::new(source, base, 512, &vec![11; phase.rows()]).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}
fn descriptor(call: &QwenSourceCall, index: Option<usize>, kind: QwenOperation) -> QwenOperator {
    let layer = index.map(|i| call.layer(i).unwrap());
    let mut matrices = Vec::new();
    if let QwenOperation::Projection(projection) = kind {
        matrices = Vec::with_capacity(projection.members().len());
        for &(suffix, rows) in projection.members() {
            matrices.push(SourceTensor {
                source: call.source(),
                name: if projection == QwenProjection::Head {
                    suffix.into()
                } else {
                    layer.unwrap().name(suffix)
                },
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                rows,
                columns: projection.input_width(),
            });
        }
    }
    let parameters = match kind {
        QwenOperation::InputRms => vec![SourceParameter {
            source: call.source(),
            name: layer.unwrap().name("input_layernorm.weight"),
            kind: QuantKind::Bf16,
            shape: vec![1024],
            bytes: 2048,
        }],
        QwenOperation::AttentionQk => ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
            .into_iter()
            .map(|suffix| SourceParameter {
                source: call.source(),
                name: layer.unwrap().name(suffix),
                kind: QuantKind::Bf16,
                shape: vec![256],
                bytes: 512,
            })
            .collect(),
        QwenOperation::GdnConv => vec![SourceParameter {
            source: call.source(),
            name: layer.unwrap().name("linear_attn.conv1d.weight"),
            kind: QuantKind::Bf16,
            shape: vec![6144, 1, 4],
            bytes: 49152,
        }],
        QwenOperation::GdnRecurrent => [
            ("linear_attn.A_log", QuantKind::F32, 16, 64),
            ("linear_attn.dt_bias", QuantKind::Bf16, 16, 32),
            ("linear_attn.norm.weight", QuantKind::F32, 128, 512),
        ]
        .into_iter()
        .map(|(suffix, kind, n, bytes)| SourceParameter {
            source: call.source(),
            name: layer.unwrap().name(suffix),
            kind,
            shape: vec![n],
            bytes,
        })
        .collect(),
        _ => Vec::new(),
    };
    QwenOperator::new(call.clone(), layer, kind, matrices, parameters).unwrap()
}
fn embedding(builder: &mut QwenPlanBuilder, call: &QwenSourceCall) -> PlanValue {
    builder
        .source_embedding(
            &QwenEmbedding::new(
                SourceTensor {
                    source: call.source(),
                    name: "model.language_model.embed_tokens.weight".into(),
                    kind: QuantKind::Bf16,
                    axes: MatrixAxes::OutputInput,
                    rows: 248320,
                    columns: 1024,
                },
                call.request().teachers().clone(),
            )
            .unwrap(),
        )
        .unwrap()
}
fn one(builder: &mut QwenPlanBuilder, d: &QwenOperator, input: PlanValue) -> PlanValue {
    builder.source_operator(d, &[input]).unwrap()[0]
}

#[test]
fn original_indices_after_attention_are_never_gdn_ordinals_or_prefix_capabilities() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = call(source, ReferencePhase::InitialOne);
    let domains = all_domains().unwrap();
    assert_eq!(domains.len(), 42);
    for i in 0..24 {
        assert_eq!(call.layer(i).unwrap().is_attention(), i % 4 == 3);
        assert_eq!(domains.contains(&StateSlot::Kv(KvDomain(i))), i % 4 == 3);
        assert_eq!(
            domains.contains(&StateSlot::Conv(ConvDomain(i))),
            i % 4 != 3
        );
        assert_eq!(
            domains.contains(&StateSlot::Recurrent(RecurrentDomain(i))),
            i % 4 != 3
        );
    }
    assert!(call.layer(24).is_err());
    assert!(call.layer(usize::MAX).is_err());
    assert!(QwenPrefixLayer::new(source, 3).is_err());
    assert!(QwenPrefixLayer::new(source, 4).is_err());
    let mut builder = QwenPlanBuilder::new_small_model(call.clone()).unwrap();
    let raw = embedding(&mut builder, &call);
    let qkvz = one(
        &mut builder,
        &descriptor(
            &call,
            Some(4),
            QwenOperation::Projection(QwenProjection::Qkvz),
        ),
        raw,
    );
    let ba = one(
        &mut builder,
        &descriptor(
            &call,
            Some(4),
            QwenOperation::Projection(QwenProjection::Ba),
        ),
        raw,
    );
    let conv = builder
        .source_operator(
            &descriptor(&call, Some(4), QwenOperation::GdnConv),
            &[qkvz, ba],
        )
        .unwrap();
    assert_eq!(
        builder
            .core
            .nodes()
            .last()
            .unwrap()
            .state
            .unwrap()
            .write
            .slot,
        StateSlot::Conv(ConvDomain(4))
    );
    builder
        .source_operator(
            &descriptor(&call, Some(4), QwenOperation::GdnRecurrent),
            &[conv[0], conv[1], ba],
        )
        .unwrap();
    assert_eq!(
        builder
            .core
            .nodes()
            .last()
            .unwrap()
            .state
            .unwrap()
            .write
            .slot,
        StateSlot::Recurrent(RecurrentDomain(4))
    );
    assert!(builder.advance_frontier().is_err());
}

#[test]
fn full_request_issuer_and_foreign_plan_values_refuse_without_recording() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = call(source, ReferencePhase::InitialOne);
    let b = call(source, ReferencePhase::InitialOne);
    assert_ne!(
        a, b,
        "same token bits do not replace teacher allocation issuer"
    );
    let mut builder = QwenPlanBuilder::new_small_model(a.clone()).unwrap();
    let raw = embedding(&mut builder, &a);
    let mut foreign = QwenPlanBuilder::new_small_model(a.clone()).unwrap();
    let foreign_raw = embedding(&mut foreign, &a);
    let d = descriptor(&a, Some(0), QwenOperation::Projection(QwenProjection::Qkvz));
    let before = (
        builder.core.nodes().len(),
        builder.core.values().len(),
        builder.core.proposed_domains().len(),
    );
    assert!(builder.source_operator(&d, &[foreign_raw]).is_err());
    assert!(
        builder
            .source_operator(&descriptor(&b, Some(0), d.kind()), &[raw])
            .is_err()
    );
    assert_eq!(
        (
            builder.core.nodes().len(),
            builder.core.values().len(),
            builder.core.proposed_domains().len()
        ),
        before
    );
    let mut old = QwenPlanBuilder::new_gdn_fragment(a.request().clone()).unwrap();
    let old_raw = embedding(&mut old, &a);
    assert!(old.source_operator(&d, &[old_raw]).is_err());
}

#[test]
fn same_call_different_qk_cohort_cannot_supply_private_appended_kv() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = call(source, ReferencePhase::Initial64);
    let mut builder = QwenPlanBuilder::new_small_model(call.clone()).unwrap();
    let raw = embedding(&mut builder, &call);
    let projection = descriptor(
        &call,
        Some(3),
        QwenOperation::Projection(QwenProjection::AttentionQkv),
    );
    let packed_a = one(&mut builder, &projection, raw);
    let packed_b = one(&mut builder, &projection, raw);
    let qk = descriptor(&call, Some(3), QwenOperation::AttentionQk);
    let a = builder.source_operator(&qk, &[packed_a]).unwrap();
    let b = builder.source_operator(&qk, &[packed_b]).unwrap();
    let append = descriptor(&call, Some(3), QwenOperation::KvAppend);
    let before = (builder.core.nodes().len(), builder.core.values().len());
    assert!(builder.source_operator(&append, &[a[1], packed_b]).is_err());
    assert_eq!(
        (builder.core.nodes().len(), builder.core.values().len()),
        before
    );
    builder.source_operator(&append, &[a[1], packed_a]).unwrap();
    let fa = descriptor(&call, Some(3), QwenOperation::FlashAttention);
    let before = (builder.core.nodes().len(), builder.core.values().len());
    assert!(builder.source_operator(&fa, &[b[0]]).is_err());
    assert_eq!(
        (builder.core.nodes().len(), builder.core.values().len()),
        before
    );
    builder.source_operator(&fa, &[a[0]]).unwrap();
    assert!(builder.source_operator(&append, &[a[1], packed_a]).is_err());
}

#[test]
fn packed_original_member_order_kind_axes_and_layer_cannot_be_substituted() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = call(source, ReferencePhase::InitialOne);
    let d = descriptor(
        &call,
        Some(7),
        QwenOperation::Projection(QwenProjection::AttentionQkv),
    );
    for mode in 0..6 {
        let mut bad = d.clone();
        match mode {
            0 => bad.matrices.swap(1, 2),
            1 => bad.matrices[0].axes = MatrixAxes::InputOutput,
            2 => bad.matrices[0].kind = QuantKind::F32,
            3 => bad.matrices[0].source = SourceIdentity::for_new_allocation().unwrap(),
            4 => {
                bad.matrices[0].name =
                    "model.language_model.layers.3.self_attn.q_proj.weight".into()
            }
            _ => bad.layer = Some(call.layer(6).unwrap()),
        }
        assert!(bad.validate().is_err(), "fault{mode}");
    }
    assert!(
        QwenOperator::new(
            call.clone(),
            Some(call.layer(3).unwrap()),
            QwenOperation::GdnConv,
            Vec::new(),
            Vec::new()
        )
        .is_err()
    );
}

#[test]
fn selected_head_is_a_view_of_full_phase_not_a_new_one_row_invocation() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for phase in [
        ReferencePhase::InitialOne,
        ReferencePhase::Initial64,
        ReferencePhase::ContinuedTwo,
        ReferencePhase::CachedOne,
    ] {
        let call = call(source, phase);
        let head = descriptor(&call, None, QwenOperation::Projection(QwenProjection::Head));
        assert_eq!(head.selected_row(), Some(phase.rows() - 1));
        assert_eq!(head.input_elements(), vec![phase.rows() * 1024]);
        assert_eq!(head.output_elements(), vec![248320]);
        assert_eq!(head.call().request().reference_phase().unwrap(), phase);
        assert_eq!(head.call().request().teachers(), call.request().teachers());
    }
}
