//! Pure geometry/issuer rejection; no source/native qualification.
use super::*;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{GdnPrefixLayer, TeacherSpan, VllmBf16Teachers},
};
fn request(
    rows: usize,
    base: usize,
    capacity: usize,
    layer: usize,
    mode: VllmBf16Mode,
) -> VllmBf16Invocation {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let ids = vec![11; rows];
    let teachers = if rows == 1 {
        VllmBf16Teachers::new(source, base, capacity, &ids).unwrap()
    } else {
        VllmBf16Teachers::from_span(TeacherSpan::new(source, base, capacity, 248320, &ids).unwrap())
            .unwrap()
    };
    VllmBf16Invocation::from_span(GdnPrefixLayer::new(source, layer).unwrap(), mode, teachers)
        .unwrap()
}
#[test]
fn finite_source_family_initial_span_rejects_other_phase_geometry() {
    for (rows, phase) in [(1, VllmGdnPhase::InitialOne), (64, VllmGdnPhase::Initial64)] {
        let req = request(rows, 0, 512, 0, VllmBf16Mode::Prefill);
        assert_eq!(
            require_request(FullModelProfile::Qwen2B, &req).unwrap(),
            phase
        );
        assert!(require_request(FullModelProfile::Qwen08, &req).is_err());
    }
    for req in [
        request(47, 0, 512, 0, VllmBf16Mode::Prefill),
        request(64, 0, 512, 1, VllmBf16Mode::Prefill),
        request(1, 1, 512, 0, VllmBf16Mode::Prefill),
        request(64, 0, 1024, 0, VllmBf16Mode::Prefill),
        request(1, 64, 512, 0, VllmBf16Mode::Decode),
    ] {
        assert!(require_request(FullModelProfile::Qwen2B, &req).is_err());
    }
}
#[test]
fn matching_teacher_words_do_not_replace_actual_invocation_issuer() {
    let req = request(64, 0, 512, 0, VllmBf16Mode::Prefill);
    let replacement = VllmBf16Invocation::from_span(
        req.layer(),
        req.mode(),
        VllmBf16Teachers::from_span(
            TeacherSpan::new(req.layer().source(), 0, 512, 248320, req.teachers().ids()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(req.teachers().ids(), replacement.teachers().ids());
    assert_ne!(req, replacement);
}
