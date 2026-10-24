use super::*;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{GdnPrefixLayer, VllmBf16Teachers},
};

fn invocation(
    source: SourceIdentity,
    phase: VllmBf16Phase,
    base: usize,
    ids: &[u32],
) -> VllmBf16Invocation {
    VllmBf16Invocation::new(
        GdnPrefixLayer::new(source, 0).unwrap(),
        phase,
        VllmBf16Teachers::new(source, base, 512, ids).unwrap(),
    )
    .unwrap()
}
fn short(source: SourceIdentity) -> Vec<VllmBf16Invocation> {
    vec![
        invocation(source, VllmBf16Phase::InitialOne, 0, &[11]),
        invocation(source, VllmBf16Phase::CachedOne, 1, &[353]),
    ]
}
fn long(source: SourceIdentity) -> Vec<VllmBf16Invocation> {
    vec![
        // IDs here exercise only original span metadata, not numerical fixtures.
        invocation(source, VllmBf16Phase::Initial64, 0, &[11; 64]),
        invocation(source, VllmBf16Phase::ContinuedTwo, 64, &[353, 353]),
        invocation(source, VllmBf16Phase::CachedOne, 66, &[353]),
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
    wrong_cached_base[1] = invocation(source, VllmBf16Phase::CachedOne, 66, &[353]);
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
