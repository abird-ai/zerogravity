//! Semantic metadata only; no trained operands, root or backend eligibility.
use super::*;
use crate::accelerator::program::typed_tests::UnawareBackend;

fn teachers(source: SourceIdentity, base: usize, rows: usize) -> VllmBf16Teachers {
    if rows == 1 {
        VllmBf16Teachers::new(source, base, 512, &[11]).unwrap()
    } else {
        VllmBf16Teachers::from_span(
            TeacherSpan::new(source, base, 512, 248_320, &vec![11; rows]).unwrap(),
        )
        .unwrap()
    }
}
fn request(
    source: SourceIdentity,
    mode: VllmBf16Mode,
    base: usize,
    rows: usize,
) -> VllmBf16Invocation {
    VllmBf16Invocation::from_span(
        GdnPrefixLayer::new(source, 0).unwrap(),
        mode,
        teachers(source, base, rows),
    )
    .unwrap()
}
fn embedding(call: &Vllm08Call) -> VllmBf16Embedding {
    VllmBf16Embedding::for_call(
        SourceTensor {
            source: call.source(),
            name: "model.language_model.embed_tokens.weight".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            rows: 248_320,
            columns: 1024,
        },
        call,
    )
    .unwrap()
}
fn rms(call: &Vllm08Call) -> Vllm08Operator {
    Vllm08Operator::new(
        call.clone(),
        Some(call.layer(0).unwrap()),
        Vllm08Kind::InputRms,
        Vec::new(),
        vec![SourceParameter {
            source: call.source(),
            name: "model.language_model.layers.0.input_layernorm.weight".into(),
            kind: QuantKind::Bf16,
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
    let a = VllmBf16Teachers::from_span(span.clone()).unwrap();
    let b = VllmBf16Teachers::from_span(span).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, teachers(source, 0, 47));
    assert!(VllmBf16Teachers::new(source, 0, 512, &[11; 47]).is_err());
    let foreign_vocabulary = TeacherSpan::new(source, 0, 512, 248_321, &[248_320; 47]).unwrap();
    assert!(VllmBf16Teachers::from_span(foreign_vocabulary).is_err());
}

#[test]
fn old_invocation_and_call_constructors_keep_exact_finite_guards() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let layer = GdnPrefixLayer::new(source, 0).unwrap();
    for (phase, base, rows) in [
        (VllmBf16Phase::InitialOne, 0, 1),
        (VllmBf16Phase::CachedOne, 1, 1),
        (VllmBf16Phase::Initial64, 0, 64),
        (VllmBf16Phase::ContinuedTwo, 64, 2),
        (VllmBf16Phase::CachedOne, 66, 1),
    ] {
        let t = teachers(source, base, rows);
        let old = VllmBf16Invocation::new(layer, phase, t.clone()).unwrap();
        let semantic = VllmBf16Invocation::from_span(layer, phase.mode(), t).unwrap();
        assert_eq!(old, semantic);
        assert_eq!(old.legacy_phase().unwrap(), phase);
        let call = Vllm08Call::new(old).unwrap();
        assert_eq!(
            VllmBf16Embedding::for_model(
                FullModelProfile::Qwen08,
                embedding(&call).tensor().clone(),
                call.request().teachers().clone()
            )
            .unwrap(),
            embedding(&call),
        );
    }
    for base in 47..=53 {
        let t = teachers(source, base, 1);
        assert!(VllmBf16Invocation::new(layer, VllmBf16Phase::CachedOne, t).is_err());
    }
    let t = teachers(source, 0, 47);
    for phase in [VllmBf16Phase::InitialOne, VllmBf16Phase::Initial64] {
        assert!(VllmBf16Invocation::new(layer, phase, t.clone()).is_err());
    }
    let semantic = request(source, VllmBf16Mode::Prefill, 0, 47);
    assert!(Vllm08Call::new(semantic.clone()).is_err());
    assert!(Vllm08Call::for_profile(FullModelProfile::Qwen2B, semantic.clone()).is_err());
    let call = Vllm08Call::from_span(semantic.clone()).unwrap();
    assert!(
        VllmBf16Embedding::new(
            embedding(&call).tensor().clone(),
            call.request().teachers().clone()
        )
        .is_err()
    );
    assert!(
        VllmBf16Embedding::for_model(
            FullModelProfile::Qwen08,
            embedding(&call).tensor().clone(),
            call.request().teachers().clone()
        )
        .is_err()
    );
    assert!(PlanBuilder::new_vllm_gdn_fragment(semantic.clone()).is_err());
    assert!(PlanBuilder::new_vllm_gdn_attempt_phase(semantic).is_err());
}

#[test]
fn true47_and_cached47_through53_spans_have_no_legacy_native_phase() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let initial = request(source, VllmBf16Mode::Prefill, 0, 47);
    assert_eq!((initial.rows(), initial.teachers().base()), (47, 0));
    assert!(initial.legacy_phase().is_err());
    let call = Vllm08Call::from_span(initial).unwrap();
    assert_eq!(call.model_profile(), FullModelProfile::Qwen08);
    for base in 47..=53 {
        let next = request(source, VllmBf16Mode::Decode, base, 1);
        assert_eq!(next.teachers().base() + next.rows(), base + 1);
        assert!(next.legacy_phase().is_err());
        assert!(Vllm08Call::from_span(next).is_ok());
    }
    let head = Vllm08Operator::new(
        call.clone(),
        None,
        Vllm08Kind::Projection(Vllm08Projection::Head),
        vec![embedding(&call).tensor().clone()],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(head.selected_row(), Some(46));
    assert_eq!(head.input_elements(), vec![47 * 1024]);
    assert_eq!(head.output_elements(), vec![248_320]);
    assert_eq!(head.call(), &call);
}

#[test]
fn invalid_source_mode_capacity_and_span_never_become_an_invocation() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let other = SourceIdentity::for_new_allocation().unwrap();
    let layer = GdnPrefixLayer::new(source, 0).unwrap();
    assert!(
        VllmBf16Invocation::from_span(
            GdnPrefixLayer::new(other, 0).unwrap(),
            VllmBf16Mode::Prefill,
            teachers(source, 0, 47)
        )
        .is_err()
    );
    assert!(
        VllmBf16Invocation::from_span(layer, VllmBf16Mode::Decode, teachers(source, 0, 1)).is_err()
    );
    assert!(
        VllmBf16Invocation::from_span(layer, VllmBf16Mode::Decode, teachers(source, 0, 47))
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
    let semantic = VllmBf16Invocation::from_span(
        layer,
        VllmBf16Mode::Prefill,
        VllmBf16Teachers::from_span(small).unwrap(),
    )
    .unwrap();
    assert!(Vllm08Call::from_span(semantic).is_err());
}

#[test]
fn full_model_embedding_records_same_builder_span_and_actual_producer_origin() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (mode, base, rows) in [
        (VllmBf16Mode::Prefill, 0, 47),
        (VllmBf16Mode::Decode, 47, 1),
    ] {
        let call = Vllm08Call::from_span(request(source, mode, base, rows)).unwrap();
        let mut builder = PlanBuilder::new_full_model(call.clone()).unwrap();
        let raw = builder
            .vllm_bf16_source_embedding(&embedding(&call))
            .unwrap();
        assert_eq!(raw.elements, rows * 1024);
        let tag = builder.vllm_value_provenance(&raw).unwrap().unwrap();
        assert_eq!(tag.mode(), Some(mode));
        assert_eq!(tag.phase(), None);
        assert_eq!(tag.teachers(), call.request().teachers());
        let normal = builder.vllm08_operator(&rms(&call), &[raw]).unwrap()[0];
        let tag = builder.vllm_value_provenance(&normal).unwrap().unwrap();
        let VllmValueOrigin::FullModel(origin) = tag.origin() else {
            panic!("full-model origin absent")
        };
        assert_eq!(origin.call(), &call);
        assert_eq!(tag.mode(), Some(mode));
        let head = Vllm08Operator::new(
            call.clone(),
            None,
            Vllm08Kind::Projection(Vllm08Projection::Head),
            vec![embedding(&call).tensor().clone()],
            Vec::new(),
        )
        .unwrap();
        let before = (
            builder.nodes.len(),
            builder.values.len(),
            builder.vllm_provenance.clone(),
        );
        assert!(
            builder.vllm08_operator(&head, &[normal]).is_err(),
            "input RMS cannot impersonate final RMS"
        );
        assert_eq!(
            (
                builder.nodes.len(),
                builder.values.len(),
                builder.vllm_provenance.clone()
            ),
            before
        );
    }
}

#[test]
fn foreign_teacher_issuer_or_mode_fails_before_graph_mutation() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (mode, base, rows) in [
        (VllmBf16Mode::Prefill, 0, 47),
        (VllmBf16Mode::Decode, 47, 1),
    ] {
        let call = Vllm08Call::from_span(request(source, mode, base, rows)).unwrap();
        let foreign = Vllm08Call::from_span(request(source, mode, base, rows)).unwrap();
        let mut builder = PlanBuilder::new_full_model(call.clone()).unwrap();
        let before = (
            builder.nodes.len(),
            builder.values.len(),
            builder.vllm_provenance.clone(),
        );
        assert!(
            builder
                .vllm_bf16_source_embedding(&embedding(&foreign))
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
        let raw = builder
            .vllm_bf16_source_embedding(&embedding(&call))
            .unwrap();
        let before = (
            builder.nodes.len(),
            builder.values.len(),
            builder.vllm_provenance.clone(),
        );
        assert!(builder.vllm08_operator(&rms(&foreign), &[raw]).is_err());
        assert_eq!(
            (
                builder.nodes.len(),
                builder.values.len(),
                builder.vllm_provenance.clone()
            ),
            before
        );
        if rows == 1 {
            let changed = Vllm08Call::from_span(
                VllmBf16Invocation::from_span(
                    call.request().layer(),
                    VllmBf16Mode::Prefill,
                    call.request().teachers().clone(),
                )
                .unwrap(),
            )
            .unwrap();
            assert!(builder.vllm08_operator(&rms(&changed), &[raw]).is_err());
            assert_eq!(
                (
                    builder.nodes.len(),
                    builder.values.len(),
                    builder.vllm_provenance.clone()
                ),
                before
            );
        }
    }
}

#[test]
fn new_semantic_span_never_enables_unaware_backend_or_unbound_slice_provenance() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = Vllm08Call::from_span(request(source, VllmBf16Mode::Prefill, 0, 47)).unwrap();
    assert!(matches!(
        UnawareBackend.vllm08_operator(&rms(&call), &[vec![0.0; 47 * 1024]]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        UnawareBackend.vllm_bf16_source_embedding(&embedding(&call)),
        Err(Error::Unsupported(_))
    ));
    let mut slice = PlanBuilder::new_operator_slice(
        source,
        ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
    )
    .unwrap();
    let raw = slice.vllm_bf16_source_embedding(&embedding(&call)).unwrap();
    let tag = slice.vllm_value_provenance(&raw).unwrap().unwrap();
    assert_eq!(tag.mode(), None);
    assert!(tag.require_invocation(call.request()).is_err());
}
