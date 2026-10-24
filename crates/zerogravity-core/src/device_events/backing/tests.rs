use super::*;

#[test]
fn idle_live_and_fresh_logical_uses_have_one_continuous_backing_charge() {
    let mut ledger = DeviceEventLedger::new(7, 128).unwrap();
    let backing = ledger
        .admit_backing(96, BufferRole::RecurrentState)
        .unwrap();
    assert_eq!(ledger.resident_bytes(), 96);
    let first = ledger
        .admit_backing_view(backing, 0, 96, BufferRole::RecurrentState)
        .unwrap();
    assert_eq!(ledger.resident_bytes(), 96);
    assert!(ledger.admit(33, BufferRole::Scratch).is_err());
    assert!(ledger.release_backing(backing).is_err());
    ledger.release_lease(first).unwrap();
    assert_eq!(ledger.resident_bytes(), 96); // idle is still physically occupied
    let second = ledger
        .admit_backing_view(backing, 0, 96, BufferRole::RecurrentState)
        .unwrap();
    assert_ne!(first, second);
    assert!(ledger.submit(0, &[(first, DeviceAccess::Write)]).is_err());
    ledger.release_lease(second).unwrap();
    ledger.release_backing(backing).unwrap();
    assert_eq!(ledger.resident_bytes(), 0);
    assert!(ledger.release_backing(backing).is_err());
}

#[test]
fn cancellation_and_ready_do_not_return_backing_before_actual_retirement_and_last_view() {
    let mut ledger = DeviceEventLedger::new(1, 128).unwrap();
    let backing = ledger
        .admit_backing(128, BufferRole::RecurrentState)
        .unwrap();
    let view = ledger
        .admit_backing_view(backing, 0, 128, BufferRole::RecurrentState)
        .unwrap();
    let event = ledger.submit(0, &[(view, DeviceAccess::Write)]).unwrap();
    ledger.request_cancel(event).unwrap();
    assert!(ledger.release_lease(view).is_err());
    assert!(ledger.require_idle_backing(backing).is_err());
    ledger.report_physical_ready(event).unwrap(); // cancelled work retires, no publication
    assert!(ledger.require_idle_backing(backing).is_err());
    ledger.release_lease(view).unwrap();
    ledger.require_idle_backing(backing).unwrap();
    assert_eq!(ledger.resident_bytes(), 128);
    ledger.release_backing(backing).unwrap();
    assert_eq!(ledger.resident_bytes(), 0);
}

#[test]
fn overlapping_geometry_views_conflict_across_distinct_ids_and_disjoint_ranges_do_not() {
    let mut ledger = DeviceEventLedger::new(1, 256).unwrap();
    let b = ledger.admit_backing(256, BufferRole::Scratch).unwrap();
    let a = ledger
        .admit_backing_view(b, 0, 128, BufferRole::Scratch)
        .unwrap();
    let overlap = ledger
        .admit_backing_view(b, 64, 128, BufferRole::Scratch)
        .unwrap();
    let tail = ledger
        .admit_backing_view(b, 128, 128, BufferRole::Scratch)
        .unwrap();
    assert!(
        ledger
            .submit(
                0,
                &[(a, DeviceAccess::Read), (overlap, DeviceAccess::Write)]
            )
            .is_err()
    );
    assert_eq!(ledger.pending_events(), 0);
    let event = ledger
        .submit(0, &[(a, DeviceAccess::Write), (tail, DeviceAccess::Write)])
        .unwrap();
    assert!(ledger.submit(0, &[(overlap, DeviceAccess::Read)]).is_err());
    ledger.report_physical_ready(event).unwrap();
    assert!(ledger.submit(0, &[(overlap, DeviceAccess::Read)]).is_err()); // Ready still pins
    ledger.retire_physical_ready(event).unwrap();
    let read1 = ledger.submit(0, &[(a, DeviceAccess::Read)]).unwrap();
    let read2 = ledger.submit(0, &[(overlap, DeviceAccess::Read)]).unwrap();
    assert!(ledger.submit(0, &[(tail, DeviceAccess::Write)]).is_err());
    for e in [read1, read2] {
        ledger.report_physical_ready(e).unwrap();
        ledger.retire_physical_ready(e).unwrap();
    }
    for v in [a, overlap, tail] {
        ledger.release_lease(v).unwrap();
    }
    ledger.release_backing(b).unwrap();
    assert_eq!(ledger.resident_bytes(), 0);
}

#[test]
fn unknown_fences_entire_backing_including_idle_aliases_and_preserves_charge() {
    let mut ledger = DeviceEventLedger::new(1, 128).unwrap();
    let b = ledger.admit_backing(128, BufferRole::Output).unwrap();
    let a = ledger
        .admit_backing_view(b, 0, 64, BufferRole::Output)
        .unwrap();
    let tail = ledger
        .admit_backing_view(b, 64, 64, BufferRole::Output)
        .unwrap();
    let e = ledger.submit(0, &[(a, DeviceAccess::Write)]).unwrap();
    ledger.report_completion_unknown(e).unwrap();
    assert!(ledger.submit(0, &[(tail, DeviceAccess::Write)]).is_err());
    assert!(
        ledger
            .admit_backing_view(b, 0, 4, BufferRole::Output)
            .is_err()
    );
    assert!(ledger.release_lease(a).is_err());
    assert!(ledger.release_backing(b).is_err());
    assert!(ledger.report_physical_ready(e).is_err());
    assert_eq!(ledger.resident_bytes(), 128);
    assert_eq!(ledger.pending_events(), 1);
}

#[test]
fn role_extent_overflow_and_sequence_failures_never_mint_a_view_or_drop_charge() {
    let mut ledger = DeviceEventLedger::new(1, 128).unwrap();
    assert!(ledger.admit_backing(0, BufferRole::Scratch).is_err());
    assert!(ledger.admit_backing(129, BufferRole::Scratch).is_err());
    let b = ledger.admit_backing(128, BufferRole::Scratch).unwrap();
    for (offset, bytes, role) in [
        (0, 0, BufferRole::Scratch),
        (127, 2, BufferRole::Scratch),
        (usize::MAX, 4, BufferRole::Scratch),
        (0, 128, BufferRole::ImmutableWeights),
    ] {
        assert!(ledger.admit_backing_view(b, offset, bytes, role).is_err());
    }
    ledger.next_lease = u64::MAX;
    assert!(
        ledger
            .admit_backing_view(b, 0, 128, BufferRole::Scratch)
            .is_err()
    );
    ledger.require_idle_backing(b).unwrap();
    ledger.release_backing(b).unwrap();
    assert_eq!(ledger.resident_bytes(), 0);
}
