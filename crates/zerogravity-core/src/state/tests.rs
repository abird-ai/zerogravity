use super::*;

fn required() -> [(String, u64); 3] {
    [
        ("kv".into(), 1),
        ("recurrence".into(), 2),
        ("rng".into(), 3),
    ]
}

fn prepared(frontier: u64) -> Vec<PreparedDomain> {
    required()
        .into_iter()
        .zip([vec![1, 2], vec![3], vec![4]])
        .map(|((name, schema), bytes)| PreparedDomain {
            name,
            schema,
            bytes,
            frontier,
        })
        .collect()
}

fn coordinator() -> TaskCoordinator {
    TaskCoordinator::new(7, required()).unwrap()
}

fn source() -> SourceIdentity {
    SourceIdentity::for_new_allocation().unwrap()
}

fn snapshot(c: &TaskCoordinator) -> (RootBinding, StateRoot, u64) {
    (c.binding(), c.root().clone(), c.output_sequence)
}

fn assert_unchanged(c: &TaskCoordinator, before: &(RootBinding, StateRoot, u64)) {
    assert_eq!(c.binding(), before.0);
    assert_eq!(c.root(), &before.1);
    assert_eq!(c.output_sequence, before.2);
    for (name, domain) in &before.1.domains {
        assert!(Arc::ptr_eq(&c.root().domains[name].bytes, &domain.bytes));
    }
}

fn malformed(frontier: u64, case: usize) -> Vec<PreparedDomain> {
    let mut domains = prepared(frontier);
    match case {
        0 => {
            let _ = domains.pop();
        }
        1 => domains[2].name = "kv".into(),
        2 => domains[2].name = "unregistered".into(),
        3 => domains[1].schema = 888,
        4 => domains[2].frontier += 1,
        _ => unreachable!(),
    }
    domains
}

#[test]
fn reference_and_ordinary_sessions_share_one_identity_issuer() {
    let source = source();
    let ordinary = SessionAuthority::new(source).unwrap();
    let plain = coordinator();
    let bound = TaskCoordinator::new_source_bound(7, source, required()).unwrap();
    let sibling = TaskCoordinator::new_source_bound(7, source, required()).unwrap();
    let another_ordinary = SessionAuthority::new(source).unwrap();
    let bindings = [
        ordinary.binding(),
        plain.binding(),
        bound.binding(),
        sibling.binding(),
        another_ordinary.binding(),
    ];
    let sessions: BTreeSet<_> = bindings.iter().map(|b| b.session()).collect();
    assert_eq!(sessions.len(), bindings.len());
    assert!(sessions.iter().all(|id| id.get() != 0));
    assert_eq!(plain.coordinator_id(), plain.binding().session().get());
    assert_eq!(bound.coordinator_id(), bound.binding().session().get());
    assert_eq!(plain.source_identity(), None);
    assert_eq!(plain.binding().source(), SourceIdentity::unbound());
    assert_eq!(bound.source_identity(), Some(source));
    assert_eq!(bound.binding().source(), source);
    assert_eq!(ordinary.binding().session_epoch(), 0);
    assert!(SessionAuthority::new(SourceIdentity::unbound()).is_err());
    assert!(TaskCoordinator::new_source_bound(7, SourceIdentity::unbound(), required()).is_err());
}

#[test]
fn given_epoch_is_retained_and_multi_unit_progress_uses_one_root() {
    for epoch in [0, 7, u64::MAX] {
        let source = source();
        let mut plain = TaskCoordinator::new(epoch, required()).unwrap();
        let mut bound = TaskCoordinator::new_source_bound(epoch, source, required()).unwrap();
        for c in [&mut plain, &mut bound] {
            let initial = c.binding();
            assert_eq!(
                (initial.root(), initial.frontier(), c.epoch()),
                (0, 0, epoch)
            );
            assert!(c.begin_advance(0).is_err());
            let attempt = c.begin_advance(11).unwrap();
            assert_eq!(attempt.base(), initial);
            assert_eq!(attempt.frontier(), 11);
            assert_eq!(c.binding(), initial);
            let out = if initial.source().is_bound() {
                c.publish_source_bound(source, attempt, prepared(11), vec![42])
            } else {
                c.publish(0, 11, prepared(11), vec![42])
            }
            .unwrap();
            assert_eq!((out.root_id, out.epoch, out.sequence), (1, epoch, 1));
            assert_eq!(out.payload, [42]);
            let committed = c.binding();
            assert_eq!(committed.session(), initial.session());
            assert_eq!(committed.source(), initial.source());
            assert_eq!(
                (
                    committed.root(),
                    committed.frontier(),
                    committed.session_epoch()
                ),
                (1, 11, epoch)
            );
            assert_eq!((c.root().id, c.root().frontier, c.epoch()), (1, 11, epoch));
        }
        assert_eq!(plain.source_identity(), None);
        assert!(!plain.binding().source().is_bound());
    }
}

#[test]
fn atomic_three_domain_publication_retains_arc_snapshots() {
    let mut c = coordinator();
    let before = snapshot(&c);
    assert!(
        c.publish(0, 4, malformed(4, 3), b"uncommitted".to_vec())
            .is_err()
    );
    assert_unchanged(&c, &before);
    let output = c.publish(0, 4, prepared(4), b"committed".to_vec()).unwrap();
    assert_eq!((output.epoch, output.sequence, output.root_id), (7, 1, 1));
    assert_eq!(output.payload, b"committed");
    assert!(before.1.domains["kv"].bytes.is_empty());
    assert_eq!(c.root().domains["kv"].bytes.as_ref(), &[1, 2]);
    let saved = c.root().clone();
    let retained = saved.domains["kv"].bytes.clone();
    assert!(Arc::ptr_eq(&retained, &c.root().domains["kv"].bytes));
    let mut next = prepared(9);
    next[0].bytes = vec![8, 9, 10];
    c.publish(1, 9, next, vec![]).unwrap();
    assert_eq!((saved.id, saved.frontier), (1, 4));
    assert!(Arc::ptr_eq(&saved.domains["kv"].bytes, &retained));
    assert_eq!(retained.as_ref(), &[1, 2]);
    assert_eq!(c.root().domains["kv"].bytes.as_ref(), &[8, 9, 10]);
    assert_eq!((c.binding().root(), c.binding().frontier()), (2, 9));
}

#[test]
fn generic_publication_rejects_invalid_domains_frontiers_and_stale_roots_atomically() {
    let mut c = coordinator();
    let before = snapshot(&c);
    for case in 0..5 {
        assert!(c.publish(0, 3, malformed(3, case), vec![]).is_err());
        assert_unchanged(&c, &before);
    }
    assert!(c.publish(1, 3, prepared(3), vec![]).is_err());
    assert!(c.publish(0, 0, prepared(0), vec![]).is_err());
    let attempt = c.begin_advance(3).unwrap();
    assert!(
        c.publish_source_bound(source(), attempt, prepared(3), vec![])
            .is_err()
    );
    assert_unchanged(&c, &before);
    c.publish(0, 3, prepared(3), vec![]).unwrap();
    let committed = snapshot(&c);
    for (root, frontier) in [(0, 4), (1, 3), (1, 2)] {
        assert!(
            c.publish(root, frontier, prepared(frontier), vec![])
                .is_err()
        );
        assert_unchanged(&c, &committed);
    }
}

#[test]
fn source_bound_publication_checks_actual_source_full_binding_and_all_domains() {
    let source = source();
    let mut c = TaskCoordinator::new_source_bound(7, source, required()).unwrap();
    let sibling = TaskCoordinator::new_source_bound(7, source, required()).unwrap();
    let ordinary = SessionAuthority::new(source).unwrap();
    let foreign = TaskCoordinator::new_source_bound(7, self::source(), required()).unwrap();
    let plain = coordinator();
    let before = snapshot(&c);
    let attempt = c.begin_advance(4).unwrap();
    assert!(c.publish(0, 4, prepared(4), vec![]).is_err());
    for actual_source in [self::source(), SourceIdentity::unbound()] {
        assert!(
            c.publish_source_bound(actual_source, attempt.clone(), prepared(4), vec![])
                .is_err()
        );
        assert_unchanged(&c, &before);
    }
    for foreign_attempt in [
        sibling.begin_advance(4).unwrap(),
        ordinary.begin_advance(4).unwrap(),
        foreign.begin_advance(4).unwrap(),
        plain.begin_advance(4).unwrap(),
    ] {
        assert!(
            c.publish_source_bound(source, foreign_attempt, prepared(4), vec![])
                .is_err()
        );
        assert_unchanged(&c, &before);
    }
    for case in 0..5 {
        assert!(
            c.publish_source_bound(source, attempt.clone(), malformed(4, case), vec![])
                .is_err()
        );
        assert_unchanged(&c, &before);
    }
    // A complete but uniformly wrong frontier is no substitute for the attempt.
    assert!(
        c.publish_source_bound(source, attempt.clone(), prepared(5), vec![])
            .is_err()
    );
    assert_unchanged(&c, &before);
    let out = c
        .publish_source_bound(source, attempt, prepared(4), vec![42])
        .unwrap();
    assert_eq!((out.root_id, out.sequence, out.epoch), (1, 1, 7));
    assert_eq!((c.binding().root(), c.binding().frontier()), (1, 4));
}

#[test]
fn source_bound_stale_competing_and_cloned_attempts_cannot_republish() {
    let source = source();
    let mut c = TaskCoordinator::new_source_bound(7, source, required()).unwrap();
    let initial = snapshot(&c);
    {
        let _abandoned = c.begin_advance(100).unwrap();
    }
    assert_unchanged(&c, &initial);
    let first = c.begin_advance(4).unwrap();
    let cloned = first.clone();
    let competing = c.begin_advance(7).unwrap();
    c.publish_source_bound(source, first, prepared(4), vec![1])
        .unwrap();
    let committed = snapshot(&c);
    for stale in [cloned, competing] {
        let domains = prepared(stale.frontier());
        assert!(
            c.publish_source_bound(source, stale, domains, vec![2])
                .is_err()
        );
        assert_unchanged(&c, &committed);
    }
    let next = c.begin_advance(2).unwrap();
    let out = c
        .publish_source_bound(source, next, prepared(6), vec![3])
        .unwrap();
    assert_eq!((out.root_id, out.sequence, out.epoch), (2, 2, 7));
    assert_eq!(c.binding().frontier(), 6);
    assert_eq!(committed.1.domains["kv"].bytes.as_ref(), &[1, 2]);
}

#[test]
fn output_sequence_overflow_leaves_authority_and_payload_untouched() {
    let source = source();
    for mut c in [
        coordinator(),
        TaskCoordinator::new_source_bound(7, source, required()).unwrap(),
    ] {
        let publish = |c: &mut TaskCoordinator, attempt: RootAttempt| {
            let domains = prepared(attempt.frontier());
            if c.source_identity().is_some() {
                c.publish_source_bound(source, attempt, domains, vec![])
            } else {
                c.publish(attempt.base().root(), attempt.frontier(), domains, vec![])
            }
        };
        let initial = c.begin_advance(2).unwrap();
        publish(&mut c, initial).unwrap();
        c.output_sequence = u64::MAX;
        let before = snapshot(&c);
        let attempt = c.begin_advance(3).unwrap();
        assert!(publish(&mut c, attempt.clone()).is_err());
        assert_unchanged(&c, &before);
        // Test-only counter repair proves the failed publication did not stale
        // the reservation or burn a root.
        c.output_sequence = 1;
        let out = publish(&mut c, attempt).unwrap();
        assert_eq!((out.root_id, out.sequence), (2, 2));
        assert_eq!(c.binding().frontier(), 5);
    }
}

#[test]
fn invalid_required_domain_maps_are_rejected() {
    for required in [
        vec![],
        vec![("".into(), 1)],
        vec![("kv".into(), 1), ("kv".into(), 2)],
    ] {
        assert!(TaskCoordinator::new(7, required.clone()).is_err());
        assert!(TaskCoordinator::new_source_bound(7, source(), required).is_err());
    }
}
