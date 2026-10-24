use super::*;

fn plane(bytes: usize) -> ResidentPlaneLayout {
    ResidentPlaneLayout {
        bytes,
        role: BufferRole::Scratch,
        alignment: 512,
    }
}
fn view(offset: usize, bytes: usize) -> ResidentViewLayout {
    ResidentViewLayout {
        plane: 0,
        offset,
        bytes,
        role: BufferRole::Scratch,
        alignment: 512,
    }
}

#[test]
fn exact_geometry_accepts_full_and_checked_aligned_tail_without_retagging_capacity() {
    require_view(plane(4096), view(0, 4096)).unwrap();
    require_view(plane(4096), view(512, 3584)).unwrap();
    for bad in [
        view(1, 2048),
        view(0, 0),
        view(3584, 1024),
        view(usize::MAX, 512),
    ] {
        assert!(require_view(plane(4096), bad).is_err());
    }
    let mut wrong_role = view(0, 1024);
    wrong_role.role = BufferRole::ImmutableWeights;
    assert!(require_view(plane(4096), wrong_role).is_err());
    let mut too_aligned = view(0, 1024);
    too_aligned.alignment = 1024;
    assert!(require_view(plane(4096), too_aligned).is_err());
}

#[test]
fn whole_set_and_metadata_caps_check_overflow_before_any_table_allocation() {
    assert!(bytes_total(&[]).is_err());
    assert!(bytes_total(&[plane(usize::MAX), plane(1)]).is_err());
    assert!(metadata_bytes(usize::MAX, 14, 0).is_err());
    assert!(metadata_bytes(48, usize::MAX, 0).is_err());
    assert!(metadata_bytes(1, 1, usize::MAX).is_err());
    assert_eq!(ResidentPool::metadata_bytes(48, 14).unwrap(), 768000);
    assert!(metadata_bytes(1, 1, 33 * 1024).is_err());
}

#[test]
fn total_geometry_admission_is_shared_and_released_by_its_last_owner() {
    let gate = Arc::new(WorkspaceGate {
        active: AtomicU64::new(0),
        next_geometry: AtomicU64::new(1),
        next_run: AtomicU64::new(1),
        run: Mutex::new(Weak::new()),
        remaining_views: std::sync::atomic::AtomicUsize::new(8),
    });
    let a = WorkspaceViewAdmission::reserve(&gate, 5).unwrap();
    let snapshot = a.clone();
    let b = WorkspaceViewAdmission::reserve(&gate, 3).unwrap();
    assert!(WorkspaceViewAdmission::reserve(&gate, 1).is_err());
    drop(a);
    assert_eq!(gate.remaining_views.load(Ordering::Acquire), 0);
    drop(snapshot);
    assert_eq!(gate.remaining_views.load(Ordering::Acquire), 5);
    drop(b);
    assert_eq!(gate.remaining_views.load(Ordering::Acquire), 8);
}

#[test]
fn all_geometries_share_one_nonblocking_activation_and_stale_release_refuses() {
    let gate = Arc::new(WorkspaceGate {
        active: AtomicU64::new(0),
        next_geometry: AtomicU64::new(1),
        next_run: AtomicU64::new(1),
        run: Mutex::new(Weak::new()),
        remaining_views: std::sync::atomic::AtomicUsize::new(2),
    });
    let a = WorkspaceMembership {
        gate: gate.clone(),
        geometry: 1,
        _admission: WorkspaceViewAdmission::reserve(&gate, 1).unwrap(),
    };
    let b = WorkspaceMembership {
        gate: gate.clone(),
        geometry: 2,
        _admission: WorkspaceViewAdmission::reserve(&gate, 1).unwrap(),
    };
    assert!(a.require_active().is_err());
    let first = gate.acquire(1).unwrap();
    a.require_active().unwrap();
    assert!(b.require_active().is_err());
    assert!(gate.acquire(2).is_err());
    assert!(gate.release(2).is_err());
    gate.release(1).unwrap();
    let second = gate.acquire(2).unwrap();
    assert!(second > first);
    assert!(a.require_active().is_err());
    b.require_active().unwrap();
    assert!(gate.release(1).is_err());
    gate.release(2).unwrap();
}
