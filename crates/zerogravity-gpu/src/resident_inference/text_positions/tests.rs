//! Pure original issuer/RootAttempt metadata and exact integer control checks.
use super::*;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{GdnPrefixLayer, VllmBf16Phase, VllmBf16Teachers},
};

fn call(source: SourceIdentity, phase: VllmBf16Phase, base: usize) -> VllmBf16Invocation {
    VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        phase,
        VllmBf16Teachers::new(source, base, 512, &vec![11; phase.rows()]).unwrap(),
    )
    .unwrap()
}
#[test]
fn exact_five_source_position_spans_are_i64_three_rows_with_defined_private_padding() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (phase, base) in [
        (VllmBf16Phase::InitialOne, 0),
        (VllmBf16Phase::CachedOne, 1),
        (VllmBf16Phase::Initial64, 0),
        (VllmBf16Phase::ContinuedTwo, 64),
        (VllmBf16Phase::CachedOne, 66),
    ] {
        let invocation = call(source, phase, base);
        let span = PositionSpan::new(&invocation).unwrap();
        let raw = span.bytes().unwrap();
        assert_eq!(raw.len(), 1560);
        for axis in 0..3 {
            for row in 0..65 {
                let offset = (axis * 65 + row) * 8;
                let actual = i64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
                assert_eq!(
                    actual,
                    if row < phase.rows() {
                        (base + row) as i64
                    } else {
                        0
                    }
                );
            }
        }
    }
}
#[test]
fn invalid_spans_capacity_and_layer_retag_refuse_without_storage_or_gpu() {
    for span in [
        PositionSpan {
            base: usize::MAX,
            rows: 1,
        },
        PositionSpan { base: 0, rows: 0 },
        PositionSpan { base: 0, rows: 65 },
        PositionSpan { base: 64, rows: 1 },
        PositionSpan { base: 1, rows: 64 },
    ] {
        assert!(span.bytes().is_err());
    }
    let source = SourceIdentity::for_new_allocation().unwrap();
    let old = call(source, VllmBf16Phase::InitialOne, 0);
    let retag = VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 1).unwrap(),
        old.legacy_phase().unwrap(),
        old.teachers().clone(),
    )
    .unwrap();
    assert!(PositionSpan::new(&retag).is_err());
    let wrong_capacity = VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        old.legacy_phase().unwrap(),
        VllmBf16Teachers::new(source, 0, 256, &[11]).unwrap(),
    )
    .unwrap();
    assert!(PositionSpan::new(&wrong_capacity).is_err());
}
#[test]
fn current_original_root_attempt_span_is_required_and_restore_revokes_it() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut authority = SessionAuthority::new(source).unwrap();
    let original = authority.binding();
    let invocation = call(source, VllmBf16Phase::Initial64, 0);
    let attempt = authority.begin_advance(64).unwrap();
    assert!(require_published_span(original, &attempt, &invocation).is_ok());
    assert!(
        require_published_span(original, &authority.begin_advance(65).unwrap(), &invocation)
            .is_err()
    );
    let foreign = SessionAuthority::new(source).unwrap();
    assert!(require_published_span(foreign.binding(), &attempt, &invocation).is_err());
    let other_source = SourceIdentity::for_new_allocation().unwrap();
    assert!(
        require_published_span(
            original,
            &attempt,
            &call(other_source, invocation.legacy_phase().unwrap(), 0)
        )
        .is_err()
    );
    // Pure logical-authority transition test; no physical model readiness claim.
    let restore = authority.begin_restore(original).unwrap();
    authority.prepare_publish(restore).unwrap().commit();
    assert_eq!(authority.binding().frontier(), 0);
    assert!(require_published_span(authority.binding(), &attempt, &invocation).is_err());
}
#[test]
fn request_teacher_identity_is_preserved_and_not_recreated_by_shape() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let first = call(source, VllmBf16Phase::InitialOne, 0);
    let same = first.clone();
    let reminted = call(source, VllmBf16Phase::InitialOne, 0);
    assert_eq!(first.teachers(), same.teachers());
    assert_ne!(first.teachers(), reminted.teachers());
    assert_eq!(
        PositionSpan::new(&first).unwrap(),
        PositionSpan::new(&reminted).unwrap()
    );
    // Equal integer controls grant no issuer/root/producer equivalence.
}
