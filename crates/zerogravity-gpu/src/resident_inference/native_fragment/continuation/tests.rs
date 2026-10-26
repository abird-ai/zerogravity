use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{SourceRows, TextCallGeometry};
fn invocation(
    source: SourceIdentity,
    mode: SourceCallMode,
    base: usize,
    ids: &[u32],
) -> SourceCall {
    SourceCall::new(
        SourceRows::new(
            TextCallGeometry::new(source, 248320, 512).unwrap(),
            base,
            Arc::from(ids),
        )
        .unwrap(),
        mode,
    )
    .unwrap()
}
fn short(source: SourceIdentity) -> Vec<SourceCall> {
    vec![
        invocation(source, SourceCallMode::Prefill, 0, &[11]),
        invocation(source, SourceCallMode::Decode, 1, &[353]),
    ]
}
fn long(source: SourceIdentity) -> Vec<SourceCall> {
    vec![
        // IDs here exercise only original span metadata, not numerical fixtures.
        invocation(source, SourceCallMode::Prefill, 0, &[11; 64]),
        invocation(source, SourceCallMode::Prefill, 64, &[353, 353]),
        invocation(source, SourceCallMode::Decode, 66, &[353]),
    ]
}
#[test]
fn one_original_attempt_covers_closed_spans_without_publishing() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for (tokens, calls) in [(2, short(source)), (67, long(source))] {
        let authority = SessionAuthority::new(source).unwrap();
        let base = authority.binding();
        let schedule =
            GdnAttemptSchedule::new(base, authority.begin_advance(tokens).unwrap(), calls, 3)
                .unwrap();
        assert_eq!(schedule.base, base);
        assert_eq!(authority.binding(), base);
        assert_eq!(authority.binding().frontier(), 0);
        assert_eq!(schedule.attempt.frontier(), tokens);
        assert!(schedule.invocation(schedule.invocations.len()).is_err());
    }
}
#[test]
fn wrong_attempt_source_order_extent_or_completed_layers_is_refused() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let foreign_source = SourceIdentity::for_new_allocation().unwrap();
    let authority = SessionAuthority::new(source).unwrap();
    let foreign_session = SessionAuthority::new(source).unwrap();
    let base = authority.binding();
    assert!(
        GdnAttemptSchedule::new(
            base,
            foreign_session.begin_advance(2).unwrap(),
            short(source),
            3
        )
        .is_err()
    );
    assert!(
        GdnAttemptSchedule::new(base, authority.begin_advance(1).unwrap(), short(source), 3)
            .is_err()
    );
    assert!(
        GdnAttemptSchedule::new(
            base,
            authority.begin_advance(2).unwrap(),
            short(foreign_source),
            3
        )
        .is_err()
    );
    assert!(
        GdnAttemptSchedule::new(base, authority.begin_advance(2).unwrap(), short(source), 2)
            .is_err()
    );
    let mut reversed = short(source);
    reversed.swap(0, 1);
    assert!(
        GdnAttemptSchedule::new(base, authority.begin_advance(2).unwrap(), reversed, 3).is_err()
    );
    let mut wrong_cached_base = short(source);
    wrong_cached_base[1] = invocation(source, SourceCallMode::Decode, 66, &[353]);
    assert!(
        GdnAttemptSchedule::new(
            base,
            authority.begin_advance(67).unwrap(),
            wrong_cached_base,
            3
        )
        .is_err()
    );
    let mut oversized_metadata = Vec::with_capacity(4);
    oversized_metadata.extend(short(source));
    assert!(
        GdnAttemptSchedule::new(
            base,
            authority.begin_advance(2).unwrap(),
            oversized_metadata,
            3
        )
        .is_err()
    );
    assert_eq!(authority.binding(), base);
}
#[test]
fn even_actual_logical_publication_cannot_retag_an_initial_prefix_schedule() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut authority = SessionAuthority::new(source).unwrap();
    let base = authority.binding();
    let stale = authority.begin_advance(2).unwrap();
    let schedule = GdnAttemptSchedule::new(base, stale.clone(), short(source), 3).unwrap();
    assert!(schedule.require_current(base).is_ok());
    let real_transition = authority.begin_advance(1).unwrap();
    let next = authority.prepare_publish(real_transition).unwrap().commit();
    assert!(schedule.require_current(next).is_err());
    assert!(GdnAttemptSchedule::new(next, stale, short(source), 3).is_err());
    assert!(
        GdnAttemptSchedule::new(next, authority.begin_advance(2).unwrap(), short(source), 3)
            .is_err()
    );
    assert_eq!(base.frontier(), 0);
    assert_eq!(authority.binding().frontier(), 1);
}
#[test]
fn same_frontier_restore_revokes_the_actual_check_used_before_enqueue() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut authority = SessionAuthority::new(source).unwrap();
    let base = authority.binding();
    let schedule =
        GdnAttemptSchedule::new(base, authority.begin_advance(2).unwrap(), short(source), 3)
            .unwrap();
    let replacement = authority.begin_restore(base).unwrap();
    let restored = authority.prepare_publish(replacement).unwrap().commit();
    assert_eq!(restored.frontier(), base.frontier());
    assert_ne!(restored.session_epoch(), base.session_epoch());
    assert!(schedule.require_current(restored).is_err());
    // Metadata authority test only; no fake ResidentReady/GPU/model is built.
}

#[test]
fn wrong_mode_capacity_vocabulary_and_row_population_refuse() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let authority = SessionAuthority::new(source).unwrap();
    for (mode, capacity, vocabulary, ids) in [
        (SourceCallMode::Decode, 512, 248320, vec![11]),
        (SourceCallMode::Prefill, 511, 248320, vec![11]),
        (SourceCallMode::Prefill, 512, 248319, vec![11]),
        (SourceCallMode::Prefill, 512, 248320, vec![11, 12]),
    ] {
        let mut calls = short(source);
        let replacement = SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, vocabulary, capacity).unwrap(),
                0,
                Arc::from(ids),
            )
            .unwrap(),
            mode,
        );
        if mode == SourceCallMode::Decode {
            assert!(replacement.is_err());
            continue;
        }
        calls[0] = replacement.unwrap();
        assert!(
            GdnAttemptSchedule::new(
                authority.binding(),
                authority.begin_advance(2).unwrap(),
                calls,
                3
            )
            .is_err()
        );
    }
    assert_eq!(authority.binding().frontier(), 0);
}
