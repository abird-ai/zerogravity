//! Pure span/source-role refusals; no loaded model, device, tensor or module.
use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{GdnPrefixLayer, TeacherSpan, VllmBf16Teachers};

fn request(
    mode: VllmBf16Mode,
    base: usize,
    ids: &[u32],
    capacity: usize,
    layer: usize,
) -> VllmBf16Invocation {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let teachers = if ids.len() == 1 {
        VllmBf16Teachers::new(source, base, capacity, ids).unwrap()
    } else {
        VllmBf16Teachers::from_span(TeacherSpan::new(source, base, capacity, 248320, ids).unwrap())
            .unwrap()
    };
    VllmBf16Invocation::from_span(GdnPrefixLayer::new(source, layer).unwrap(), mode, teachers)
        .unwrap()
}
#[test]
fn exact47_prefill_is_separate_from_legacy_and_other_prompts() {
    let exact = request(VllmBf16Mode::Prefill, 0, &CANONICAL47_PROMPT, 512, 0);
    assert_eq!(
        Canonical47CallKind::of(&exact).unwrap(),
        Canonical47CallKind::Prefill
    );
    assert!(exact.legacy_phase().is_err());
    let mut changed = CANONICAL47_PROMPT;
    changed[12] = 353;
    for wrong in [
        request(VllmBf16Mode::Prefill, 0, &changed, 512, 0),
        request(VllmBf16Mode::Prefill, 0, &CANONICAL47_PROMPT, 512, 1),
        request(VllmBf16Mode::Prefill, 0, &CANONICAL47_PROMPT, 1024, 0),
        request(VllmBf16Mode::Prefill, 1, &CANONICAL47_PROMPT, 512, 0),
        request(VllmBf16Mode::Prefill, 0, &[11; 64], 512, 0),
    ] {
        assert!(Canonical47CallKind::of(&wrong).is_err());
    }
}
#[test]
fn seven_cached_coordinates_remain_semantic_decode_and_never_legacy_cached_one() {
    for base in 47..=53 {
        let call = request(VllmBf16Mode::Decode, base, &[11], 512, 0);
        assert_eq!(
            Canonical47CallKind::of(&call).unwrap(),
            Canonical47CallKind::Decode { base }
        );
        assert!(call.legacy_phase().is_err());
    }
    for wrong in [
        request(VllmBf16Mode::Decode, 46, &[11], 512, 0),
        request(VllmBf16Mode::Decode, 54, &[11], 512, 0),
        request(VllmBf16Mode::Decode, 1, &[11], 512, 0),
        request(VllmBf16Mode::Prefill, 47, &[11], 512, 0),
    ] {
        assert!(Canonical47CallKind::of(&wrong).is_err());
    }
}
#[test]
fn equal_teacher_ids_do_not_replace_the_actual_record_issuer() {
    let a = request(VllmBf16Mode::Prefill, 0, &CANONICAL47_PROMPT, 512, 0);
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
    assert_ne!(a, b);
    assert_ne!(a.teachers(), b.teachers());
    assert_eq!(
        Canonical47CallKind::of(&a).unwrap(),
        Canonical47CallKind::of(&b).unwrap()
    );
    // Coordinate identity never replaces record/source/physical issuer checks.
}
#[test]
fn only_declared_prefill_owned_tuple_may_reach_the_exact_prepared_join() {
    assert!(
        Canonical47CallKind::Prefill
            .require_prefill_gdn(Canonical47GdnArithmetic::OwnedCanonical47Solve4)
            .is_ok()
    );
    assert!(
        Canonical47CallKind::Prefill
            .require_prefill_gdn(Canonical47GdnArithmetic::OriginalPlainSolve2)
            .is_err()
    );
    for base in 47..=53 {
        assert!(
            Canonical47CallKind::Decode { base }
                .require_prefill_gdn(Canonical47GdnArithmetic::OwnedCanonical47Solve4)
                .is_err()
        );
    }
}

#[test]
fn short_hf_prompt47_and_cached47_issuers_cannot_select_old_canonical_native_kind() {
    use zerogravity_core::accelerator::program::{ShortHfCall, ShortHfSpan, ShortHfWorkload};
    let source = SourceIdentity::for_new_allocation().unwrap();
    let short = ShortHfCall::new(
        FullModelProfile::Qwen08,
        ShortHfSpan::prefill(source, ShortHfWorkload::Prompt47, &CANONICAL47_PROMPT).unwrap(),
    )
    .unwrap();
    assert_eq!(short.request().teachers().ids(), CANONICAL47_PROMPT);
    assert_eq!(
        (short.request().rows(), short.request().teachers().base()),
        (47, 0)
    );
    assert!(short.request().legacy_phase().is_err());
    assert!(matches!(
        Canonical47CallKind::of(short.request()),
        Err(Error::Unsupported(_))
    ));
    for base in 47..=53 {
        let short = ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::cached(source, ShortHfWorkload::Prompt47, base, 11).unwrap(),
        )
        .unwrap();
        let old = request(VllmBf16Mode::Decode, base, &[11], 512, 0);
        assert_eq!(
            Canonical47CallKind::of(&old).unwrap(),
            Canonical47CallKind::Decode { base }
        );
        assert_eq!(short.request().teachers().ids(), old.teachers().ids());
        assert!(matches!(
            Canonical47CallKind::of(short.request()),
            Err(Error::Unsupported(_))
        ));
    }
}
