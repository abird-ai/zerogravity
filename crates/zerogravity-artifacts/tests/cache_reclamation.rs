//! Exact-target reclamation of uniquely owned synthetic fixture caches only.
#![cfg(all(feature = "cache", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;

use cache_transaction_support::*;
use std::fs;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::Error;
use zerogravity_artifacts::cache::{
    CacheStore, DiscoveredEntry, FileRole, Manifest, ReclaimMode, ReclaimOutcome, ReclaimReport,
    TransferAdmission,
};

fn apply(store: &CacheStore, snapshot: &str) -> ReclaimReport {
    match store
        .reclaim_snapshot(snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
        .unwrap()
    {
        ReclaimOutcome::Durable(report) => report,
        other => panic!("expected durable reclamation: {other:?}"),
    }
}

fn publish_manifest(
    store: &CacheStore,
    admission: &TransferAdmission,
    manifest: Manifest,
    bytes: &[u8],
) -> String {
    let paths: Vec<_> = manifest
        .entries()
        .iter()
        .map(|entry| entry.path().to_string())
        .collect();
    let mut tx = store
        .begin(manifest, admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    for path in paths {
        let mut input = bytes;
        tx.import_object(&path, &mut input).unwrap();
    }
    durable(tx.commit(None).unwrap())
}

fn private_record(path: &std::path::Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

#[test]
fn exact_dry_run_preserves_names_and_apply_rebuilds_lower_unique_inode_charge() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let bytes = vec![7; 128 * 1024];
    let snapshot = durable(publish(&store, &admission, &bytes, None).unwrap());
    let before = store.usage().unwrap();
    let ReclaimOutcome::DryRun(preview) = store
        .reclaim_snapshot(&snapshot, ReclaimMode::DryRun, &AtomicBool::new(false))
        .unwrap()
    else {
        panic!("dry-run outcome required");
    };
    assert!(!preview.retired());
    assert_eq!(preview.used_before_bytes(), before.used_bytes);
    assert_eq!(preview.used_after_bytes(), Some(before.used_bytes));
    assert_eq!(
        (preview.removed_files(), preview.removed_directories()),
        (0, 0)
    );
    assert_eq!(preview.candidate_files(), 4);
    assert_eq!(preview.candidate_directories(), 2);
    assert_payload(&store.open_snapshot(&snapshot).unwrap(), &bytes);
    assert_eq!(store.usage().unwrap(), before);
    let report = apply(&store, &snapshot);
    assert!(report.retired());
    assert_eq!(report.removed_files(), preview.candidate_files());
    assert_eq!(
        report.removed_directories(),
        preview.candidate_directories()
    );
    assert!(report.used_after_bytes().unwrap() < before.used_bytes);
    assert_eq!(
        report.used_after_bytes(),
        Some(store.usage().unwrap().used_bytes)
    );
    assert_not_found(store.open_snapshot(&snapshot));
    assert!(
        !fixture
            .root
            .join(format!("blobs/sha256/{}", digest(&bytes)))
            .exists()
    );
    let absent = apply(&store, &snapshot);
    assert!(!absent.retired());
    assert_eq!((absent.candidate_files(), absent.removed_files()), (0, 0));
    assert_eq!(absent.used_after_bytes(), Some(absent.used_before_bytes()));
    assert_no_host_reservation(&admission);
}

#[test]
fn every_current_reference_protects_target_and_retargeting_does_not_delete_it() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("first")).unwrap());
    publish(&store, &admission, OLD, Some("second")).unwrap();
    for mode in [ReclaimMode::DryRun, ReclaimMode::Apply] {
        assert!(matches!(
            store
                .reclaim_snapshot(&previous, mode, &AtomicBool::new(false))
                .unwrap(),
            ReclaimOutcome::Referenced { references: 2 }
        ));
    }
    let current = durable(publish(&store, &admission, NEW, Some("first")).unwrap());
    assert!(matches!(
        store
            .reclaim_snapshot(&previous, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Referenced { references: 1 }
    ));
    publish(&store, &admission, NEW, Some("second")).unwrap();
    apply(&store, &previous);
    assert_payload(&store.resolve_ref("first").unwrap(), NEW);
    assert_payload(&store.resolve_ref("second").unwrap(), NEW);
    assert!(store.list_snapshots().unwrap().contains(&current));
    assert!(
        fs::read_dir(fixture.root.join("reference-records"))
            .unwrap()
            .count()
            >= 4
    );
}

#[test]
fn duplicate_alias_probes_deduplicate_inodes_and_remaining_snapshot_protects_blob() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let target = publish_manifest(
        &store,
        &admission,
        manifest(&[
            ("a.bin", OLD, FileRole::Auxiliary),
            ("nested/b.bin", OLD, FileRole::Auxiliary),
        ]),
        OLD,
    );
    let retained = durable(publish(&store, &admission, OLD, None).unwrap());
    let report = apply(&store, &target);
    assert_eq!(report.retained_blobs(), 1);
    assert_payload(&store.open_snapshot(&retained).unwrap(), OLD);
    assert!(
        fixture
            .root
            .join(format!("blobs/sha256/{}", digest(OLD)))
            .exists()
    );
    let final_report = apply(&store, &retained);
    assert_eq!(final_report.retained_blobs(), 0);
    assert!(
        !fixture
            .root
            .join(format!("blobs/sha256/{}", digest(OLD)))
            .exists()
    );
}

#[test]
fn escaped_managed_file_clone_blocks_removal_after_wrapper_drop() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
    let lease = store.open_snapshot(&snapshot).unwrap();
    let escaped = lease.file(PAYLOAD_PATH).unwrap().try_clone().unwrap();
    drop(lease);
    let before = store.usage().unwrap();
    assert!(matches!(
        store
            .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Busy
    ));
    assert_eq!(store.usage().unwrap(), before);
    assert_eq!(file_bytes(&escaped), OLD);
    drop(escaped);
    apply(&store, &snapshot);
}

#[test]
fn payload_named_manifest_json_still_requires_actual_descriptor_pin() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let snapshot = publish_manifest(
        &store,
        &admission,
        manifest(&[("manifest.json", OLD, FileRole::Auxiliary)]),
        OLD,
    );
    let lease = store.open_snapshot(&snapshot).unwrap();
    assert!(matches!(
        store
            .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Busy
    ));
    drop(lease);
    apply(&store, &snapshot);
}

#[test]
fn live_first_seen_dedup_transaction_excludes_collection_until_real_cleanup() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
    let mut transaction = store
        .begin_discovered(
            payload_manifest(OLD).source().clone(),
            vec![
                DiscoveredEntry::new(PAYLOAD_PATH, OLD.len() as u64, FileRole::Auxiliary, None)
                    .unwrap(),
            ],
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    let mut input = OLD;
    transaction.import_object(PAYLOAD_PATH, &mut input).unwrap();
    assert!(matches!(
        store
            .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Busy
    ));
    let mut prefix = [0; 128];
    let count = transaction.read_prefix(PAYLOAD_PATH, &mut prefix).unwrap();
    assert_eq!(&prefix[..count], OLD);
    drop(transaction);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    apply(&store, &snapshot);
    assert_no_host_reservation(&admission);
}

#[test]
fn abandoned_staged_manifest_and_payload_names_remain_conservative_graph_roots() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
    let stage = fixture.root.join("staging/abandoned-fixture");
    fs::DirBuilder::new().mode(0o700).create(&stage).unwrap();
    private_record(
        &stage.join("manifest.json"),
        &payload_manifest(OLD).canonical_bytes().unwrap(),
    );
    fs::hard_link(
        fixture.root.join(format!("blobs/sha256/{}", digest(OLD))),
        stage.join("object-0"),
    )
    .unwrap();
    let report = apply(&store, &snapshot);
    assert_eq!(report.retained_blobs(), 1);
    assert_eq!(fs::read(stage.join("object-0")).unwrap(), OLD);
    assert!(stage.join("manifest.json").exists());
    assert!(
        fixture
            .root
            .join(format!("blobs/sha256/{}", digest(OLD)))
            .exists()
    );
}

#[test]
fn malformed_graph_or_unexpected_target_entry_refuses_before_retirement() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
    let foreign = fixture
        .root
        .join(format!("manifests/{}.json", "a".repeat(64)));
    private_record(&foreign, b"{}");
    assert!(matches!(
        store.reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false)),
        Err(Error::Malformed(_))
    ));
    assert!(fixture.root.join(format!("snapshots/{snapshot}")).exists());
    fs::remove_file(foreign).unwrap();
    let extra = fixture
        .root
        .join(format!("snapshots/{snapshot}/unexpected"));
    private_record(&extra, b"retain unrecognized bytes");
    assert!(matches!(
        store.reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false)),
        Err(Error::Malformed(_))
    ));
    assert!(extra.exists());
    assert!(!fixture.root.join(format!("retirement/{snapshot}")).exists());
}

#[test]
fn empty_retirement_marker_finishes_and_absence_preserves_orphan_manifest() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let snapshot = payload_manifest(OLD).digest().unwrap();
    let marker = fixture.root.join(format!("retirement/{snapshot}"));
    fs::DirBuilder::new().mode(0o700).create(&marker).unwrap();
    let report = apply(&store, &snapshot);
    assert!(report.retired());
    assert_eq!(
        (report.removed_files(), report.removed_directories()),
        (0, 1)
    );
    let orphan = fixture.root.join(format!("manifests/{snapshot}.json"));
    private_record(&orphan, &payload_manifest(OLD).canonical_bytes().unwrap());
    let absent = apply(&store, &snapshot);
    assert!(!absent.retired());
    assert_eq!(
        (absent.removed_files(), absent.removed_directories()),
        (0, 0)
    );
    assert!(orphan.exists());
}

#[test]
fn cancelled_entry_and_control_peak_refuse_without_mutation() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
    assert!(matches!(
        store.reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(true)),
        Err(Error::State(_))
    ));
    let before = store.usage().unwrap().used_bytes;
    // Fill only our synthetic root; leave less than the conservative retirement
    // control peak while staying within its original persistent quota.
    let filler = fixture.root.join("fixture-padding");
    private_record(
        &filler,
        &vec![0; (DISK_LIMIT - before - 64 * 1024) as usize],
    );
    assert!(store.usage().unwrap().used_bytes < DISK_LIMIT);
    assert!(matches!(
        store.reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false)),
        Err(Error::Limit(_))
    ));
    assert!(fixture.root.join(format!("snapshots/{snapshot}")).exists());
    assert!(!fixture.root.join(format!("retirement/{snapshot}")).exists());
}

#[cfg(feature = "diagnostic-cache")]
mod interleavings {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, mpsc};
    use std::thread;
    use std::time::Duration;
    use zerogravity_artifacts::{
        Result,
        cache::{CacheHook, CacheMoment, CacheOperation, CachePoint},
    };

    struct ReplaceAfterBoundary {
        path: PathBuf,
        saved: PathBuf,
        operation: CacheOperation,
        symlink: bool,
        fired: AtomicBool,
    }
    impl CacheHook for ReplaceAfterBoundary {
        fn point(&self, point: CachePoint) -> Result<()> {
            if point.operation == self.operation
                && point.moment == CacheMoment::Before
                && !self.fired.swap(true, Ordering::AcqRel)
            {
                fs::rename(&self.path, &self.saved).unwrap();
                if self.symlink {
                    std::os::unix::fs::symlink(&self.saved, &self.path).unwrap();
                } else {
                    private_record(&self.path, OLD);
                }
            }
            Ok(())
        }
    }

    #[test]
    fn replaced_activity_or_publication_lock_refuses_mutation_and_sync_after_before_hook() {
        for lock in ["activity", "publication"] {
            for operation in [
                CacheOperation::ReclaimUnlink,
                CacheOperation::ReclaimDirectorySync,
            ] {
                let fixture = Fixture::new();
                let store = fixture.open();
                let admission = admission(256 * 1024 * 1024);
                let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
                let path = fixture.root.join(format!("locks/{lock}"));
                let saved = fixture.base.join("original-lock");
                let hooked = store.clone().with_hook(Arc::new(ReplaceAfterBoundary {
                    path: path.clone(),
                    saved: saved.clone(),
                    operation,
                    symlink: false,
                    fired: AtomicBool::new(false),
                }));
                let ReclaimOutcome::Incomplete { report, cause } = hooked
                    .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
                    .unwrap()
                else {
                    panic!("lost authority after retirement must be incomplete");
                };
                assert!(matches!(cause, Error::State(_)));
                assert_eq!(report.used_after_bytes(), None);
                assert_eq!(report.removed_files(), 0);
                assert!(
                    fixture
                        .root
                        .join(format!("retirement/{snapshot}/files/{PAYLOAD_PATH}"))
                        .exists()
                );
                drop(hooked);
                fs::remove_file(&path).unwrap();
                fs::rename(&saved, &path).unwrap();
                apply(&store, &snapshot);
            }
        }
    }

    #[test]
    fn replaced_regular_or_symlink_leaf_is_never_unlinked_after_before_hook() {
        for symlink in [false, true] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(256 * 1024 * 1024);
            let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
            let path = fixture
                .root
                .join(format!("retirement/{snapshot}/files/{PAYLOAD_PATH}"));
            let saved = fixture.base.join("original-payload-link");
            let hooked = store.clone().with_hook(Arc::new(ReplaceAfterBoundary {
                path: path.clone(),
                saved: saved.clone(),
                operation: CacheOperation::ReclaimUnlink,
                symlink,
                fired: AtomicBool::new(false),
            }));
            let ReclaimOutcome::Incomplete { report, cause } = hooked
                .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
                .unwrap()
            else {
                panic!("changed leaf after retirement must be incomplete");
            };
            assert!(matches!(cause, Error::State(_)));
            assert_eq!(report.removed_files(), 0);
            assert!(fs::symlink_metadata(&path).is_ok());
            if symlink {
                assert_eq!(report.used_after_bytes(), None);
            }
            drop(hooked);
            fs::remove_file(&path).unwrap();
            fs::rename(&saved, &path).unwrap();
            apply(&store, &snapshot);
        }
    }

    struct PauseAfterRetirement {
        ready: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl CacheHook for PauseAfterRetirement {
        fn point(&self, point: CachePoint) -> Result<()> {
            if point.operation == CacheOperation::RetirementRename
                && point.moment == CacheMoment::After
            {
                self.ready.send(()).unwrap();
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
            Ok(())
        }
    }

    #[test]
    fn new_openers_wait_behind_collector_and_cancellation_does_not_release_its_barrier() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
        let (ready, reached) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let collector = store.clone().with_hook(Arc::new(PauseAfterRetirement {
            ready,
            release: Mutex::new(released),
        }));
        let cancel = AtomicBool::new(false);
        thread::scope(|scope| {
            let collection = scope.spawn(|| {
                collector
                    .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
                    .unwrap()
            });
            reached.recv_timeout(Duration::from_secs(10)).unwrap();
            let (started, entered) = mpsc::channel();
            let (opening_store, opening_snapshot, opening_cancel) = (&store, &snapshot, &cancel);
            let cancellation = scope.spawn(move || {
                started.send(()).unwrap();
                opening_store.open_snapshot_cancellable(opening_snapshot, opening_cancel)
            });
            entered.recv_timeout(Duration::from_secs(10)).unwrap();
            thread::sleep(Duration::from_millis(20));
            assert!(!cancellation.is_finished());
            cancel.store(true, Ordering::Release);
            assert!(matches!(cancellation.join().unwrap(), Err(Error::State(_))));
            let (started, entered) = mpsc::channel();
            let (opening_store, opening_snapshot) = (&store, &snapshot);
            let opening = scope.spawn(move || {
                started.send(()).unwrap();
                opening_store.open_snapshot(opening_snapshot)
            });
            entered.recv_timeout(Duration::from_secs(10)).unwrap();
            thread::sleep(Duration::from_millis(20));
            assert!(!opening.is_finished());
            release.send(()).unwrap();
            assert!(matches!(
                collection.join().unwrap(),
                ReclaimOutcome::Durable(_)
            ));
            assert_not_found(opening.join().unwrap());
        });
    }
}

#[cfg(feature = "diagnostic-cache")]
mod interrupted {
    use super::*;
    use std::sync::atomic::Ordering;
    use zerogravity_artifacts::{
        Result,
        cache::{CacheHook, CacheMoment, CacheOperation, CachePoint},
    };

    struct CancelAfterUnlink(Arc<AtomicBool>);
    impl CacheHook for CancelAfterUnlink {
        fn point(&self, point: CachePoint) -> Result<()> {
            if point.operation == CacheOperation::ReclaimUnlink
                && point.moment == CacheMoment::After
            {
                self.0.store(true, Ordering::Release);
            }
            Ok(())
        }
    }

    #[test]
    fn cancellation_after_mutation_retains_manifest_and_exact_retry_finishes() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let snapshot = durable(publish(&store, &admission, OLD, None).unwrap());
        let cancel = Arc::new(AtomicBool::new(false));
        let hooked = store
            .clone()
            .with_hook(Arc::new(CancelAfterUnlink(cancel.clone())));
        let ReclaimOutcome::Incomplete { report, .. } = hooked
            .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &cancel)
            .unwrap()
        else {
            panic!("post-mutation cancellation must be incomplete");
        };
        assert!(report.retired());
        assert_eq!(report.removed_files(), 1);
        assert!(
            fixture
                .root
                .join(format!("retirement/{snapshot}/manifest.json"))
                .exists()
        );
        assert_not_found(store.open_snapshot(&snapshot));
        drop(hooked);
        drop(store);
        let reopened = CacheStore::open_existing(options(&fixture.root)).unwrap();
        let final_report = apply(&reopened, &snapshot);
        assert!(final_report.retired());
        assert!(final_report.used_after_bytes().unwrap() < final_report.used_before_bytes());
        assert!(!fixture.root.join(format!("retirement/{snapshot}")).exists());
    }
}
