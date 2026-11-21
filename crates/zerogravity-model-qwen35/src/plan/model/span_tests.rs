//! Semantic metadata only; no trained operands, root or backend eligibility.
use super::*;
use crate::plan::tests_support::UnawareBackend;

fn teachers(source: SourceIdentity, base: usize, rows: usize) -> QwenInputRows {
    if rows == 1 {
        QwenInputRows::new(source, base, 512, &[11]).unwrap()
    } else {
        QwenInputRows::from_span(
            TeacherSpan::new(source, base, 512, 248_320, &vec![11; rows]).unwrap(),
        )
        .unwrap()
    }
}
fn request(
    source: SourceIdentity,
    mode: InvocationPhase,
    base: usize,
    rows: usize,
) -> QwenGdnInvocation {
    QwenGdnInvocation::from_span(
        QwenPrefixLayer::new(source, 0).unwrap(),
        mode,
        teachers(source, base, rows),
    )
    .unwrap()
}
fn embedding(call: &QwenModelInvocation) -> QwenEmbedding {
    QwenEmbedding::for_model_invocation(
        SourceTensor {
            source: call.source(),
            name: "model.language_model.embed_tokens.weight".into(),
            kind: StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            axes: MatrixAxes::OutputInput,
            rows: 248_320,
            columns: 1024,
        },
        call,
    )
    .unwrap()
}
fn rms(call: &QwenModelInvocation) -> QwenOperator {
    QwenOperator::new(
        call.clone(),
        Some(call.layer(0).unwrap()),
        QwenOperation::InputRms,
        Vec::new(),
        vec![SourceParameter {
            source: call.source(),
            name: "model.language_model.layers.0.input_layernorm.weight".into(),
            kind: StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            shape: vec![1024],
            bytes: 2048,
        }],
    )
    .unwrap()
}

#[test]
fn adoption_preserves_existing_teacher_issuer_and_rechecks_original_vocabulary() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let span = TeacherSpan::new(source, 0, 512, 248_320, &[11; 47]).unwrap();
    let a = QwenInputRows::from_span(span.clone()).unwrap();
    let b = QwenInputRows::from_span(span).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, teachers(source, 0, 47));
    assert!(QwenInputRows::new(source, 0, 512, &[11; 47]).is_err());
    let foreign_vocabulary = TeacherSpan::new(source, 0, 512, 248_321, &[248_320; 47]).unwrap();
    assert!(QwenInputRows::from_span(foreign_vocabulary).is_err());
}

#[test]
fn legacy_invocation_constructors_keep_exact_finite_guards() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let layer = QwenPrefixLayer::new(source, 0).unwrap();
    for (phase, base, rows) in [
        (ReferencePhase::InitialOne, 0, 1),
        (ReferencePhase::CachedOne, 1, 1),
        (ReferencePhase::Initial64, 0, 64),
        (ReferencePhase::ContinuedTwo, 64, 2),
        (ReferencePhase::CachedOne, 66, 1),
    ] {
        let t = teachers(source, base, rows);
        let old = QwenGdnInvocation::new(layer, phase, t.clone()).unwrap();
        let semantic = QwenGdnInvocation::from_span(layer, phase.invocation_phase(), t).unwrap();
        assert_eq!(old, semantic);
        assert_eq!(old.reference_phase().unwrap(), phase);
        let call = QwenModelInvocation::new(old).unwrap();
        assert_eq!(
            QwenEmbedding::for_model(
                QwenModelVariant::Qwen800M,
                embedding(&call).tensor().clone(),
                call.request().teachers().clone()
            )
            .unwrap(),
            embedding(&call),
        );
    }
    for base in 47..=53 {
        let t = teachers(source, base, 1);
        assert!(QwenGdnInvocation::new(layer, ReferencePhase::CachedOne, t).is_err());
    }
    let t = teachers(source, 0, 47);
    for phase in [ReferencePhase::InitialOne, ReferencePhase::Initial64] {
        assert!(QwenGdnInvocation::new(layer, phase, t.clone()).is_err());
    }
    let semantic = request(source, InvocationPhase::Prefill, 0, 47);
    assert!(QwenModelInvocation::new(semantic.clone()).is_err());
    assert!(QwenModelInvocation::for_profile(QwenModelVariant::Qwen2B, semantic.clone()).is_err());
    let call = QwenModelInvocation::from_span(semantic.clone()).unwrap();
    assert!(
        QwenEmbedding::new(
            embedding(&call).tensor().clone(),
            call.request().teachers().clone()
        )
        .is_err()
    );
    assert!(
        QwenEmbedding::for_model(
            QwenModelVariant::Qwen800M,
            embedding(&call).tensor().clone(),
            call.request().teachers().clone()
        )
        .is_err()
    );
    assert!(QwenPlanBuilder::new_gdn_fragment(semantic.clone()).is_err());
    assert!(QwenPlanBuilder::new_gdn_attempt_phase(semantic).is_err());
}

#[test]
fn prefill47_and_decode_bases47_through53_have_no_legacy_reference_phase() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let initial = request(source, InvocationPhase::Prefill, 0, 47);
    assert_eq!((initial.rows(), initial.teachers().base()), (47, 0));
    assert!(initial.reference_phase().is_err());
    let call = QwenModelInvocation::from_span(initial).unwrap();
    assert_eq!(call.model_profile(), QwenModelVariant::Qwen800M);
    for base in 47..=53 {
        let next = request(source, InvocationPhase::Decode, base, 1);
        assert_eq!(next.teachers().base() + next.rows(), base + 1);
        assert!(next.reference_phase().is_err());
        assert!(QwenModelInvocation::from_span(next).is_ok());
    }
    let head = QwenOperator::new(
        call.clone(),
        None,
        QwenOperation::Projection(QwenProjection::Head),
        vec![embedding(&call).tensor().clone()],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(head.selected_row(), Some(46));
    assert_eq!(head.input_elements(), vec![47 * 1024]);
    assert_eq!(head.output_elements(), vec![248_320]);
    assert_eq!(head.model_invocation(), &call);
}

#[test]
fn invalid_source_phase_capacity_and_span_never_become_an_invocation() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let other = SourceIdentity::for_new_allocation().unwrap();
    let layer = QwenPrefixLayer::new(source, 0).unwrap();
    assert!(
        QwenGdnInvocation::from_span(
            QwenPrefixLayer::new(other, 0).unwrap(),
            InvocationPhase::Prefill,
            teachers(source, 0, 47)
        )
        .is_err()
    );
    assert!(
        QwenGdnInvocation::from_span(layer, InvocationPhase::Decode, teachers(source, 0, 1))
            .is_err()
    );
    assert!(
        QwenGdnInvocation::from_span(layer, InvocationPhase::Decode, teachers(source, 0, 47))
            .is_err()
    );
    for (base, capacity, count) in [
        (500, 512, 47),
        (usize::MAX, 512, 2),
        (0, 0, 47),
        (0, 512, 65),
    ] {
        assert!(TeacherSpan::new(source, base, capacity, 248_320, &vec![11; count]).is_err());
    }
    let small = TeacherSpan::new(source, 0, 64, 248_320, &[11; 47]).unwrap();
    let semantic = QwenGdnInvocation::from_span(
        layer,
        InvocationPhase::Prefill,
        QwenInputRows::from_span(small).unwrap(),
    )
    .unwrap();
    assert!(QwenModelInvocation::from_span(semantic).is_err());
}

#[test]
fn full_model_embedding_records_same_builder_span_and_actual_producer_origin() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (mode, base, rows) in [
        (InvocationPhase::Prefill, 0, 47),
        (InvocationPhase::Decode, 47, 1),
    ] {
        let call = QwenModelInvocation::from_span(request(source, mode, base, rows)).unwrap();
        let mut builder = QwenPlanBuilder::new_model(call.clone()).unwrap();
        let raw = builder.source_embedding(&embedding(&call)).unwrap();
        assert_eq!(raw.elements(), rows * 1024);
        let tag = builder.value_provenance(&raw).unwrap().unwrap();
        assert_eq!(tag.invocation_phase(), Some(mode));
        assert_eq!(tag.reference_phase(), None);
        assert_eq!(tag.teachers(), call.request().teachers());
        let normal = builder.source_operator(&rms(&call), &[raw]).unwrap()[0];
        let tag = builder.value_provenance(&normal).unwrap().unwrap();
        let QwenValueOrigin::FullModel(origin) = tag.origin() else {
            panic!("full-model origin absent")
        };
        assert_eq!(origin.model_invocation(), &call);
        assert_eq!(tag.invocation_phase(), Some(mode));
        let head = QwenOperator::new(
            call.clone(),
            None,
            QwenOperation::Projection(QwenProjection::Head),
            vec![embedding(&call).tensor().clone()],
            Vec::new(),
        )
        .unwrap();
        let before = (
            builder.core.nodes().len(),
            builder.core.values().len(),
            builder.provenance.clone(),
        );
        assert!(
            builder.source_operator(&head, &[normal]).is_err(),
            "input RMS cannot impersonate final RMS"
        );
        assert_eq!(
            (
                builder.core.nodes().len(),
                builder.core.values().len(),
                builder.provenance.clone()
            ),
            before
        );
    }
}

#[test]
fn foreign_teacher_issuer_or_phase_fails_before_graph_mutation() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (mode, base, rows) in [
        (InvocationPhase::Prefill, 0, 47),
        (InvocationPhase::Decode, 47, 1),
    ] {
        let call = QwenModelInvocation::from_span(request(source, mode, base, rows)).unwrap();
        let foreign = QwenModelInvocation::from_span(request(source, mode, base, rows)).unwrap();
        let mut builder = QwenPlanBuilder::new_model(call.clone()).unwrap();
        let before = (
            builder.core.nodes().len(),
            builder.core.values().len(),
            builder.provenance.clone(),
        );
        assert!(builder.source_embedding(&embedding(&foreign)).is_err());
        assert_eq!(
            (
                builder.core.nodes().len(),
                builder.core.values().len(),
                builder.provenance.clone()
            ),
            before
        );
        let raw = builder.source_embedding(&embedding(&call)).unwrap();
        let before = (
            builder.core.nodes().len(),
            builder.core.values().len(),
            builder.provenance.clone(),
        );
        assert!(builder.source_operator(&rms(&foreign), &[raw]).is_err());
        assert_eq!(
            (
                builder.core.nodes().len(),
                builder.core.values().len(),
                builder.provenance.clone()
            ),
            before
        );
        if rows == 1 {
            let changed = QwenModelInvocation::from_span(
                QwenGdnInvocation::from_span(
                    call.request().layer(),
                    InvocationPhase::Prefill,
                    call.request().teachers().clone(),
                )
                .unwrap(),
            )
            .unwrap();
            assert!(builder.source_operator(&rms(&changed), &[raw]).is_err());
            assert_eq!(
                (
                    builder.core.nodes().len(),
                    builder.core.values().len(),
                    builder.provenance.clone()
                ),
                before
            );
        }
    }
}

#[test]
fn new_semantic_span_never_enables_unaware_backend_or_unbound_slice_provenance() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call =
        QwenModelInvocation::from_span(request(source, InvocationPhase::Prefill, 0, 47)).unwrap();
    assert!(matches!(
        UnawareBackend.source_operator(&rms(&call), &[vec![0.0; 47 * 1024]]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        UnawareBackend.source_embedding(&embedding(&call)),
        Err(Error::Unsupported(_))
    ));
    let mut slice =
        QwenPlanBuilder::new_operator_slice(source, NumericalProfile::CudaSm86Bf16ReferenceV1)
            .unwrap();
    let raw = slice.source_embedding(&embedding(&call)).unwrap();
    let tag = slice.value_provenance(&raw).unwrap().unwrap();
    assert_eq!(tag.invocation_phase(), None);
    assert!(tag.require_invocation(call.request()).is_err());
}
