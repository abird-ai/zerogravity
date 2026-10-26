//! Pure original issuer/RootAttempt metadata and exact integer control checks.
use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{
    SourceCall, SourceCallMode, SourceRows, TextCallGeometry,
};
fn call(source: SourceIdentity, mode: SourceCallMode, base: usize, rows: usize) -> SourceCall {
    SourceCall::new(
        SourceRows::new(
            TextCallGeometry::new(source, 248320, 512).unwrap(),
            base,
            Arc::from(vec![11; rows]),
        )
        .unwrap(),
        mode,
    )
    .unwrap()
}
#[test]
fn exact_five_source_position_spans_are_i64_three_rows_with_defined_private_padding() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (mode, base, rows) in [
        (SourceCallMode::Prefill, 0, 1),
        (SourceCallMode::Decode, 1, 1),
        (SourceCallMode::Prefill, 0, 64),
        (SourceCallMode::Prefill, 64, 2),
        (SourceCallMode::Decode, 66, 1),
    ] {
        let invocation = call(source, mode, base, rows);
        let span = PositionSpan::new(&invocation).unwrap();
        let raw = span.bytes().unwrap();
        assert_eq!(raw.len(), 1560);
        for axis in 0..3 {
            for row in 0..65 {
                let offset = (axis * 65 + row) * 8;
                let actual = i64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
                assert_eq!(actual, if row < rows { (base + row) as i64 } else { 0 });
            }
        }
    }
}
#[test]
fn invalid_spans_capacity_vocabulary_mode_refuse_without_storage_or_gpu() {
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
    for (mode, capacity, vocabulary) in [
        (SourceCallMode::Prefill, 256, 248320),
        (SourceCallMode::Prefill, 512, 248319),
        (SourceCallMode::Decode, 512, 248320),
    ] {
        let wrong = SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, vocabulary, capacity).unwrap(),
                0,
                Arc::from([11]),
            )
            .unwrap(),
            mode,
        );
        if mode == SourceCallMode::Decode {
            assert!(wrong.is_err());
        } else {
            assert!(PositionSpan::new(&wrong.unwrap()).is_err());
        }
    }
}
#[test]
fn current_original_root_attempt_span_is_required_and_restore_revokes_it() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut authority = SessionAuthority::new(source).unwrap();
    let original = authority.binding();
    let invocation = call(source, SourceCallMode::Prefill, 0, 64);
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
            &call(other_source, invocation.mode(), 0, invocation.rows().len())
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
    let first = call(source, SourceCallMode::Prefill, 0, 1);
    let same = first.clone();
    let reminted = call(source, SourceCallMode::Prefill, 0, 1);
    assert_eq!(first.rows(), same.rows());
    assert_ne!(first.rows(), reminted.rows());
    assert_eq!(
        PositionSpan::new(&first).unwrap(),
        PositionSpan::new(&reminted).unwrap()
    );
    // Equal integer controls grant no issuer/root/producer equivalence.
}
