//! Metadata rejection tests; actual source/lease/physical gates remain separate.
use super::*;
use crate::vllm_cuda_gdn::Canonical47GdnArithmetic as A;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{GdnPrefixLayer, TeacherSpan, VllmBf16Invocation, VllmBf16Teachers},
};
fn request(ids: &[u32], base: usize, capacity: usize, layer: usize) -> VllmBf16Invocation {
    let source = SourceIdentity::for_new_allocation().unwrap();
    VllmBf16Invocation::from_span(
        GdnPrefixLayer::new(source, layer).unwrap(),
        VllmBf16Mode::Prefill,
        VllmBf16Teachers::from_span(TeacherSpan::new(source, base, capacity, 248320, ids).unwrap())
            .unwrap(),
    )
    .unwrap()
}
#[test]
fn finite47_request_is_not_legacy64_or_plain_tuple_or_another_prompt() {
    let kind = ComponentKind::Canonical47(A::OwnedCanonical47Solve4);
    let exact = request(&CANONICAL47_PROMPT, 0, 512, 0);
    kind.require_request(&exact).unwrap();
    assert!(exact.legacy_phase().is_err());
    assert!(ComponentKind::Qwen2b.require_request(&exact).is_err());
    assert!(
        ComponentKind::Canonical47(A::OriginalPlainSolve2)
            .require_request(&exact)
            .is_err()
    );
    let mut changed = CANONICAL47_PROMPT;
    changed[12] = 353;
    for invalid in [
        request(&changed, 0, 512, 0),
        request(&[11; 64], 0, 512, 0),
        request(&CANONICAL47_PROMPT, 1, 512, 0),
        request(&CANONICAL47_PROMPT, 0, 1024, 0),
        request(&CANONICAL47_PROMPT, 0, 512, 1),
    ] {
        assert!(kind.require_request(&invalid).is_err());
    }
}
#[test]
fn old_source_family_and_equal_id_new_teacher_issuer_remain_distinct() {
    let old = request(&[11; 64], 0, 512, 0);
    ComponentKind::Qwen2b.require_request(&old).unwrap();
    assert_eq!(ComponentKind::Qwen2b.profile(), FullModelProfile::Qwen2B);
    assert_eq!(ComponentKind::Qwen2b.columns(), 2048);
    let new = ComponentKind::Canonical47(A::OwnedCanonical47Solve4);
    assert_eq!(new.profile(), FullModelProfile::Qwen08);
    assert_eq!(new.columns(), 1024);
    assert!(new.require_request(&old).is_err());
    let a = request(&CANONICAL47_PROMPT, 0, 512, 0);
    let b = VllmBf16Invocation::from_span(
        a.layer(),
        a.mode(),
        VllmBf16Teachers::from_span(
            TeacherSpan::new(a.layer().source(), 0, 512, 248320, &CANONICAL47_PROMPT).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(a.teachers().ids(), b.teachers().ids());
    assert_ne!(a.teachers(), b.teachers());
    assert_ne!(a, b);
}

#[test]
fn short_hf_issuer_cannot_reuse_initial47_component_kind() {
    use zerogravity_core::accelerator::program::{ShortHfCall, ShortHfSpan, ShortHfWorkload};
    let source = SourceIdentity::for_new_allocation().unwrap();
    let kind = ComponentKind::Canonical47(A::OwnedCanonical47Solve4);
    kind.require_request(&request(&CANONICAL47_PROMPT, 0, 512, 0))
        .unwrap();
    let short = ShortHfCall::new(
        FullModelProfile::Qwen08,
        ShortHfSpan::prefill(source, ShortHfWorkload::Prompt47, &CANONICAL47_PROMPT).unwrap(),
    )
    .unwrap();
    assert_eq!(short.request().teachers().ids(), CANONICAL47_PROMPT);
    assert!(short.request().legacy_phase().is_err());
    assert!(matches!(
        kind.require_request(short.request()),
        Err(Error::Unsupported(_))
    ));
    for base in 47..=53 {
        let short = ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::cached(source, ShortHfWorkload::Prompt47, base, 11).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            kind.require_request(short.request()),
            Err(Error::Unsupported(_))
        ));
    }
}
