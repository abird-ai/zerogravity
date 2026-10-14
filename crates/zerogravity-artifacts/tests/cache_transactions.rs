//! A1 local cache transactions: tiny synthetic files, no model or provider.
//! The diagnostic feature adds explicit hooks and bounded owned subprocesses;
//! no production environment-variable fault injection is used.
#![cfg(all(feature = "cache", target_os = "linux"))]

mod cache_transaction_support;

use cache_transaction_support::*;
use std::fs;
use std::io::{self, Read};
use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use zerogravity_artifacts::Error;
#[cfg(feature = "diagnostic-cache")]
use zerogravity_artifacts::Result;
#[cfg(feature = "diagnostic-cache")]
use zerogravity_artifacts::cache::CommitOutcome;
use zerogravity_artifacts::cache::{CacheStore, FileRole};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};

#[test]
fn local_multifile_snapshot_ref_replacement_and_retained_lease() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let files: &[(&str, &[u8], FileRole)] = &[
        ("config.json", b"{\"model\":\"fixture\"}", FileRole::Config),
        (
            "tokenizer.json",
            b"{\"tokens\":[\"a\",\"b\"]}",
            FileRole::Tokenizer,
        ),
        ("LICENSE", b"synthetic fixture only", FileRole::License),
    ];
    let expected = manifest(files);
    let expected_digest = expected.digest().unwrap();
    let mut transaction = store
        .begin(expected, &admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    for (path, bytes, _) in files {
        let mut reader = *bytes;
        transaction.import_object(path, &mut reader).unwrap();
    }
    assert_eq!(
        durable(transaction.commit(Some("fixture")).unwrap()),
        expected_digest
    );
    let lease = store.resolve_ref("fixture").unwrap();
    for (path, bytes, _) in files {
        assert_eq!(file_bytes(lease.file(path).unwrap()), *bytes);
    }
    let replacement = durable(publish(&store, &admission, NEW, Some("fixture")).unwrap());
    assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
    assert_payload(&store.open_snapshot(&replacement).unwrap(), NEW);
    assert_payload(&store.verify_snapshot(&replacement).unwrap(), NEW);
    for (path, bytes, _) in files {
        assert_eq!(file_bytes(lease.file(path).unwrap()), *bytes);
    }
    assert!(matches!(
        store.remove_snapshot(&expected_digest),
        Err(Error::Unsupported(_))
    ));
    assert_not_found(store.resolve_ref("not-present"));
    assert_no_host_reservation(&admission);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_eq!(
        fs::metadata(&fixture.root).unwrap().permissions().mode() & 0o777,
        0o700
    );
}

#[test]
fn wrong_truncated_oversized_and_incomplete_objects_preserve_previous_ref() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let original = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let wrong = vec![b'x'; NEW.len()];
    for mut bytes in [
        b"short".as_slice(),
        wrong.as_slice(),
        b"oversized object longer than expected replacement bytes",
    ] {
        let mut transaction = store
            .begin(
                payload_manifest(NEW),
                &admission,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        assert!(transaction.import_object(PAYLOAD_PATH, &mut bytes).is_err());
        drop(transaction);
        assert_payload(&store.resolve_ref("fixture").unwrap(), OLD);
        assert_no_host_reservation(&admission);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    }
    let mut incomplete = store
        .begin(
            payload_manifest(NEW),
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    assert!(
        incomplete
            .import_object("not-declared", &mut b"x".as_slice())
            .is_err()
    );
    assert!(incomplete.commit(Some("fixture")).is_err());
    assert_payload(&store.resolve_ref("fixture").unwrap(), OLD);
    assert_payload(&store.open_snapshot(&original).unwrap(), OLD);
    assert_no_host_reservation(&admission);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
}

#[test]
fn opened_root_capability_survives_path_replacement_without_escape() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let pinned = fixture.base.join("pinned-original-root");
    let outside = fixture.base.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("untouched"), b"outside sentinel").unwrap();
    let inode = fs::metadata(&fixture.root).unwrap().ino();
    fs::rename(&fixture.root, &pinned).unwrap();
    symlink(&outside, &fixture.root).unwrap();
    durable(publish(&store, &admission, NEW, Some("fixture")).unwrap());
    assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
    assert_eq!(fs::metadata(&pinned).unwrap().ino(), inode);
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 1);
    assert_eq!(
        fs::read(outside.join("untouched")).unwrap(),
        b"outside sentinel"
    );
    assert!(CacheStore::open(options(&fixture.root)).is_err());
    assert_no_host_reservation(&admission);
}

#[test]
fn symlinked_storage_ancestor_cannot_write_outside_cache() {
    let fixture = Fixture::new();
    fs::create_dir(&fixture.root).unwrap();
    fs::set_permissions(&fixture.root, fs::Permissions::from_mode(0o700)).unwrap();
    let outside = fixture.base.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("untouched"), b"outside sentinel").unwrap();
    symlink(&outside, fixture.root.join("blobs")).unwrap();
    let admission = admission(128 * 1024 * 1024);
    let result = CacheStore::open(options(&fixture.root))
        .and_then(|store| publish(&store, &admission, NEW, Some("fixture")));
    assert!(result.is_err());
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 1);
    assert_eq!(
        fs::read(outside.join("untouched")).unwrap(),
        b"outside sentinel"
    );
    assert_no_host_reservation(&admission);
}

#[test]
fn verification_hashes_actual_cached_bytes_and_refuses_same_size_mutation() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let snapshot = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let blob = fixture.root.join("blobs/sha256").join(digest(OLD));
    // A same-UID writer can chmod and modify even a nominally read-only blob.
    // A1 verification must detect it; descriptor possession is not immutability.
    fs::set_permissions(&blob, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&blob, vec![b'x'; OLD.len()]).unwrap();
    assert!(matches!(
        store.verify_snapshot(&snapshot),
        Err(Error::Malformed(_))
    ));
    assert!(matches!(
        store.resolve_ref("fixture"),
        Err(Error::Malformed(_))
    ));
    assert_no_host_reservation(&admission);
}

#[test]
fn encoded_reference_bound_is_checked_before_publication() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let quoted = "\"".repeat(100);
    durable(publish(&store, &admission, OLD, Some(&quoted)).unwrap());
    assert_payload(&store.resolve_ref(&quoted).unwrap(), OLD);
    let excessive = "\"".repeat(4096);
    assert!(matches!(
        publish(&store, &admission, NEW, Some(&excessive)),
        Err(Error::Limit(_))
    ));
    assert_payload(&store.resolve_ref("fixture").unwrap(), OLD);
    assert_payload(&store.resolve_ref(&quoted).unwrap(), OLD);
    assert_not_found(store.resolve_ref(&excessive));
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn retained_reference_descriptor_has_named_backing_after_dedup_and_replacement() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    durable(publish(&store, &admission, NEW, Some("fixture")).unwrap());
    // Exercise existing identical record deduplication before retaining the
    // actual alias descriptor; equal bytes alone must not leave a new inode.
    durable(publish(&store, &admission, NEW, Some("fixture")).unwrap());
    let aliases = fs::read_dir(fixture.root.join("refs"))
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(aliases.len(), 1);
    let old = fs::File::open(aliases[0].path()).unwrap();
    let old_bytes = file_bytes(&old);
    let backing = fixture
        .root
        .join("reference-records")
        .join(format!("{}.json", digest(&old_bytes)));
    let before = old.metadata().unwrap();
    let named = fs::metadata(&backing).unwrap();
    assert_eq!((before.dev(), before.ino()), (named.dev(), named.ino()));
    assert!(
        before.nlink() >= 2,
        "live alias must share the immutable backing inode"
    );
    let usage_before = store.usage().unwrap();
    durable(publish(&store, &admission, THIRD, Some("fixture")).unwrap());
    assert_payload(&store.resolve_ref("fixture").unwrap(), THIRD);
    assert_eq!(file_bytes(&old), old_bytes);
    let retained = old.metadata().unwrap();
    let named = fs::metadata(&backing).unwrap();
    assert!(
        retained.nlink() > 0,
        "retained reference payload became unnamed"
    );
    assert_eq!((retained.dev(), retained.ino()), (named.dev(), named.ino()));
    assert!(store.usage().unwrap().used_bytes >= usage_before.used_bytes);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

struct ObservedReader<'a> {
    bytes: &'a [u8],
    reads: Arc<AtomicUsize>,
    observe: Box<dyn FnMut() + 'a>,
}

impl Read for ObservedReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        (self.observe)();
        self.bytes.read(output)
    }
}

#[test]
fn admission_precedes_body_read_and_cancel_releases_synchronous_resources() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let too_small = admission(1);
    assert!(matches!(
        store.begin(
            payload_manifest(NEW),
            &too_small,
            Arc::new(AtomicBool::new(false))
        ),
        Err(Error::Limit(_))
    ));
    assert_no_host_reservation(&too_small);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);

    let admission = admission(128 * 1024 * 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicUsize::new(0));
    let mut transaction = store
        .begin(payload_manifest(NEW), &admission, cancel.clone())
        .unwrap();
    let mut reader = ObservedReader {
        bytes: NEW,
        reads: reads.clone(),
        observe: Box::new(|| {
            let host = admission.arbiter.snapshot(&admission.resource).unwrap();
            assert!(
                host.per_class
                    .get(&AllocationClass::TransferStaging)
                    .copied()
                    .unwrap_or(0)
                    > 0
            );
            assert!(store.usage().unwrap().reserved_bytes > 0);
            cancel.store(true, Ordering::Release);
        }),
    };
    assert!(matches!(
        transaction.import_object(PAYLOAD_PATH, &mut reader),
        Err(Error::State(_))
    ));
    drop(reader);
    assert!(reads.load(Ordering::Relaxed) > 0);
    drop(transaction);
    assert_no_host_reservation(&admission);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_not_found(store.resolve_ref("fixture"));
}

#[test]
fn host_and_cooperative_disk_reservations_are_shared_before_source_reads() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let transaction = store
        .begin(
            payload_manifest(NEW),
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    let host = admission.arbiter.snapshot(&admission.resource).unwrap();
    let filler = admission
        .arbiter
        .admit(
            &admission.resource,
            ResourceOwner::new("other-host-user", "fixture").unwrap(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: host.remaining,
            }],
        )
        .unwrap();
    assert!(matches!(
        store.begin(
            payload_manifest(THIRD),
            &admission,
            Arc::new(AtomicBool::new(false))
        ),
        Err(Error::Limit(_))
    ));
    drop(filler);
    drop(transaction);
    assert_no_host_reservation(&admission);

    // A declared object reserves space before any body is supplied. No large
    // allocation/file is needed to exercise the actual cooperative quota.
    let first = store
        .begin(
            quota_manifest(),
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    let reserved = store.usage().unwrap().reserved_bytes;
    assert!(reserved >= 6 * 1024 * 1024);
    assert!(matches!(
        store.begin(
            quota_manifest(),
            &admission,
            Arc::new(AtomicBool::new(false))
        ),
        Err(Error::Limit(_))
    ));
    drop(first);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    let second = store
        .begin(
            quota_manifest(),
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    drop(second);
    assert_no_host_reservation(&admission);
}

#[cfg(feature = "diagnostic-cache")]
mod diagnostics {
    use super::*;
    use cache_transaction_support::process::*;
    use std::collections::BTreeMap;
    use std::io::{BufRead, Write};
    use std::path::Path;
    use std::sync::Mutex;
    use std::time::Duration;
    use zerogravity_artifacts::cache::{CacheHook, CachePoint};
    use zerogravity_artifacts::cache::{Manifest, ManifestEntry, ManifestSource};

    struct FailAt {
        point: CachePoint,
        ordinal: usize,
        calls: AtomicUsize,
        reached: Arc<AtomicUsize>,
    }

    impl CacheHook for FailAt {
        fn point(&self, point: CachePoint) -> Result<()> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == self.ordinal {
                assert_eq!(point, self.point, "publication callback order changed");
                self.reached.fetch_add(1, Ordering::Relaxed);
                // Inject a real I/O error kind at an explicit boundary. This
                // does not fill the host filesystem or pretend power loss.
                return Err(Error::Io(io::Error::from_raw_os_error(28)));
            }
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordPoints(Mutex<Vec<CachePoint>>);

    impl CacheHook for RecordPoints {
        fn point(&self, point: CachePoint) -> Result<()> {
            self.0.lock().unwrap().push(point);
            Ok(())
        }
    }

    fn observed_points() -> Vec<CachePoint> {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let recorder = Arc::new(RecordPoints::default());
        durable(
            publish(
                &store.with_hook(recorder.clone()),
                &admission,
                NEW,
                Some("fixture"),
            )
            .unwrap(),
        );
        let observed = recorder.0.lock().unwrap().clone();
        for required in points() {
            assert!(
                observed.contains(&required),
                "required publication hook absent: {required:?}"
            );
        }
        eprintln!(
            "A04 actual publication callbacks including repeated directory syncs: {}",
            observed.len()
        );
        observed
    }

    fn lock_inodes(root: &Path) -> BTreeMap<std::ffi::OsString, (u64, u64)> {
        fs::read_dir(root.join("locks"))
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                let metadata = entry.metadata().unwrap();
                assert!(metadata.is_file());
                (entry.file_name(), (metadata.dev(), metadata.ino()))
            })
            .collect()
    }

    fn assert_lock_inodes_preserved(
        root: &Path,
        before: &BTreeMap<std::ffi::OsString, (u64, u64)>,
    ) {
        let after = lock_inodes(root);
        for (name, identity) in before {
            assert_eq!(
                after.get(name),
                Some(identity),
                "stable lock inode replaced or removed: {name:?}"
            );
        }
    }

    #[test]
    fn every_publication_boundary_preserves_old_or_reports_complete_uncertain_new() {
        for (ordinal, point) in observed_points().into_iter().enumerate() {
            for has_previous in [false, true] {
                let fixture = Fixture::new();
                let store = fixture.open();
                let admission = admission(128 * 1024 * 1024);
                if has_previous {
                    durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
                }
                let reached = Arc::new(AtomicUsize::new(0));
                let faulted = store.clone().with_hook(Arc::new(FailAt {
                    point,
                    ordinal,
                    calls: AtomicUsize::new(0),
                    reached: reached.clone(),
                }));
                let reads = Arc::new(AtomicUsize::new(0));
                let mut reader = ObservedReader {
                    bytes: NEW,
                    reads: reads.clone(),
                    observe: Box::new(|| {}),
                };
                let result = faulted
                    .begin(
                        payload_manifest(NEW),
                        &admission,
                        Arc::new(AtomicBool::new(false)),
                    )
                    .and_then(|mut transaction| {
                        transaction.import_object(PAYLOAD_PATH, &mut reader)?;
                        transaction.commit(Some("fixture"))
                    });
                assert!(reached.load(Ordering::Relaxed) > 0, "unreached {point:?}");
                if before_body(point) {
                    assert_eq!(
                        reads.load(Ordering::Relaxed),
                        0,
                        "body read before begin admission/sync completed"
                    );
                }
                if reference_changed(point) {
                    match result {
                        Ok(CommitOutcome::DurabilityUnknown {
                            snapshot,
                            reference,
                            ..
                        }) => {
                            assert_eq!(reference.as_deref(), Some("fixture"));
                            assert_payload(&store.open_snapshot(snapshot.digest()).unwrap(), NEW);
                        }
                        _ => panic!("post-publication failure must report uncertainty: {point:?}"),
                    }
                    assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
                } else {
                    assert!(
                        result.is_err(),
                        "pre-publication fault returned success: {point:?}"
                    );
                    if has_previous {
                        assert_payload(&store.resolve_ref("fixture").unwrap(), OLD);
                    } else {
                        assert_not_found(store.resolve_ref("fixture"));
                    }
                }
                assert_no_host_reservation(&admission);
                assert_eq!(store.usage().unwrap().reserved_bytes, 0);
                assert!(store.usage().unwrap().used_bytes <= DISK_LIMIT);
            }
        }
    }

    #[test]
    fn real_child_exit_at_each_publication_boundary_recovers_complete_snapshots() {
        for (index, point) in observed_points().into_iter().enumerate() {
            for has_previous in [false, true] {
                let fixture = Fixture::new();
                let store = fixture.open();
                let admission = admission(128 * 1024 * 1024);
                if has_previous {
                    durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
                }
                let output = RunningChild::spawn(&fixture.root, "crash", index, "fixture").finish();
                assert_eq!(
                    output.status.code(),
                    Some(86),
                    "child missed {point:?}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let lock_ids = lock_inodes(&fixture.root);
                if before_body(point) {
                    assert!(
                        String::from_utf8_lossy(&output.stdout)
                            .contains("CACHE_HELPER_EXIT body_reads=0")
                    );
                }
                let reopened = fixture.open();
                if reference_changed(point) {
                    assert_payload(&reopened.resolve_ref("fixture").unwrap(), NEW);
                } else if has_previous {
                    assert_payload(&reopened.resolve_ref("fixture").unwrap(), OLD);
                } else {
                    assert_not_found(reopened.resolve_ref("fixture"));
                }
                // Reusing the same object/ref after OS release proves that the
                // dead publisher cannot leave a permanent live-lock reservation.
                durable(publish(&reopened, &admission, NEW, Some("fixture")).unwrap());
                assert_payload(&reopened.resolve_ref("fixture").unwrap(), NEW);
                let usage = reopened.usage().unwrap();
                assert_eq!(usage.reserved_bytes, 0);
                assert!(usage.used_bytes <= usage.limit_bytes);
                assert_lock_inodes_preserved(&fixture.root, &lock_ids);
                assert_no_host_reservation(&admission);
            }
        }
    }

    #[test]
    fn contended_publication_lock_has_deadline_and_preserves_its_owner() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let mut owner = RunningChild::spawn(&fixture.root, "hold", 0, "fixture");
        owner.event("CACHE_HELPER_LOCK_HELD");
        let lock_ids = lock_inodes(&fixture.root);
        let waiter = RunningChild::spawn(&fixture.root, "timeout", 0, "fixture");
        assert_success(waiter.finish());
        assert_payload(&store.resolve_ref("fixture").unwrap(), OLD);
        owner.send("release");
        assert_success(owner.finish());
        assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
        assert_lock_inodes_preserved(&fixture.root, &lock_ids);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }

    #[test]
    fn distinct_objects_and_references_progress_while_another_object_is_held() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let mut first = RunningChild::spawn(&fixture.root, "hold-object", 0, "first");
        first.event("CACHE_HELPER_LOCK_HELD");
        // Cancellation at transaction entry must not cancel the separately
        // held owner. This makes no claim about a measured lock-wait interval.
        assert_success(RunningChild::spawn(&fixture.root, "canceled", 0, "first").finish());
        assert_success(RunningChild::spawn(&fixture.root, "publish-third", 0, "second").finish());
        assert_payload(&store.resolve_ref("second").unwrap(), THIRD);
        assert_not_found(store.resolve_ref("first"));
        first.send("release");
        assert_success(first.finish());
        assert_payload(&store.resolve_ref("first").unwrap(), NEW);
        assert_payload(&store.resolve_ref("second").unwrap(), THIRD);
        let first_snapshot =
            durable(publish(&store, &admission, NEW, Some("deduplicated")).unwrap());
        let a = store.open_snapshot(&first_snapshot).unwrap();
        let b = store.resolve_ref("first").unwrap();
        assert_eq!(
            a.file(PAYLOAD_PATH).unwrap().metadata().unwrap().ino(),
            b.file(PAYLOAD_PATH).unwrap().metadata().unwrap().ino()
        );
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }

    #[test]
    fn cooperating_processes_observe_live_disk_reservations_and_release() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let mut owner = RunningChild::spawn(&fixture.root, "hold-reservation", 0, "fixture");
        owner.event("CACHE_HELPER_RESERVATION_HELD");
        assert!(store.usage().unwrap().reserved_bytes >= 6 * 1024 * 1024);
        assert!(matches!(
            store.begin(
                quota_manifest(),
                &admission,
                Arc::new(AtomicBool::new(false))
            ),
            Err(Error::Limit(_))
        ));
        assert_no_host_reservation(&admission);
        owner.send("release");
        assert_success(owner.finish());
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        let next = store
            .begin(
                quota_manifest(),
                &admission,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        drop(next);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }

    #[test]
    fn cached_deep_snapshot_rebuild_retains_future_space_until_private_work_exists() {
        let fixture = Fixture::new();
        let mut selected_options = options(&fixture.root);
        selected_options.disk_limit_bytes = 64 * 1024 * 1024;
        let store = CacheStore::open(selected_options).unwrap();
        let admission = admission(512 * 1024 * 1024);
        let names: Vec<String> = (0..64)
            .map(|index| {
                let mut path = format!("d{index:03}");
                for depth in 0..16 {
                    path.push_str(&format!("/level{depth:02}"));
                }
                path.push_str("/data.bin");
                path
            })
            .collect();
        let files: Vec<_> = names
            .iter()
            .enumerate()
            .map(|(index, path)| {
                (
                    path.as_str(),
                    if index % 2 == 0 {
                        b"".as_slice()
                    } else {
                        b"x".as_slice()
                    },
                    FileRole::Auxiliary,
                )
            })
            .collect();
        let mut first = store
            .begin(
                manifest(&files),
                &admission,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        for (path, bytes, _) in &files {
            let mut reader = *bytes;
            first.import_object(path, &mut reader).unwrap();
        }
        let original = durable(first.commit(Some("deep")).unwrap());

        struct CheckPrivatePeak {
            store: CacheStore,
            admission: zerogravity_artifacts::cache::TransferAdmission,
            observed: Arc<AtomicBool>,
        }
        impl CacheHook for CheckPrivatePeak {
            fn point(&self, point: CachePoint) -> Result<()> {
                use zerogravity_artifacts::cache::{CacheMoment, CacheOperation};
                if point.operation == CacheOperation::SnapshotEntryInstall
                    && point.moment == CacheMoment::After
                    && point.object == Some(32)
                {
                    let usage = self.store.usage()?;
                    assert!(
                        usage.used_bytes + usage.reserved_bytes <= usage.limit_bytes,
                        "a cached final tree was credited before its separate private replacement existed: {usage:?}"
                    );
                    assert!(
                        self.admission
                            .arbiter
                            .snapshot(&self.admission.resource)?
                            .remaining
                            >= 128 * 1024 * 1024
                    );
                    let refusal = std::thread::scope(|scope| {
                        scope
                            .spawn(|| {
                                self.store
                                    .begin(
                                        quota_manifest(),
                                        &self.admission,
                                        Arc::new(AtomicBool::new(false)),
                                    )
                                    .map(drop)
                            })
                            .join()
                            .unwrap()
                    });
                    assert!(
                        matches!(refusal, Err(Error::Limit(_))),
                        "overlapping owner was admitted beyond the reserved private-work peak"
                    );
                    self.observed.store(true, Ordering::Release);
                }
                Ok(())
            }
        }
        let observed = Arc::new(AtomicBool::new(false));
        let checked = store.clone().with_hook(Arc::new(CheckPrivatePeak {
            store: store.clone(),
            admission: admission.clone(),
            observed: observed.clone(),
        }));
        let mut repeat = checked
            .begin(
                manifest(&files),
                &admission,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        let usage = store.usage().unwrap();
        // Fill the remaining cooperative allowance with another live plan;
        // this transfers no body and allocates no declared payload bytes.
        let filler_bytes = usage
            .limit_bytes
            .checked_sub(usage.used_bytes + usage.reserved_bytes + 512 * 1024)
            .unwrap();
        assert!(filler_bytes > 0);
        let filler_plan = Manifest::new(
            ManifestSource::new("local", "unread-space-plan", "immutable-revision").unwrap(),
            vec![
                ManifestEntry::new(
                    "unread.bin",
                    digest(b"never-imported"),
                    filler_bytes,
                    FileRole::Auxiliary,
                )
                .unwrap(),
            ],
            &Default::default(),
        )
        .unwrap();
        let filler = store
            .begin(filler_plan, &admission, Arc::new(AtomicBool::new(false)))
            .unwrap();
        for (path, bytes, _) in &files {
            let mut reader = *bytes;
            repeat.import_object(path, &mut reader).unwrap();
        }
        assert_eq!(durable(repeat.commit(Some("deep")).unwrap()), original);
        assert!(
            observed.load(Ordering::Acquire),
            "private entry construction was not observed"
        );
        drop(filler);
        let lease = store.resolve_ref("deep").unwrap();
        for (path, bytes, _) in &files {
            assert_eq!(file_bytes(lease.file(path).unwrap()), *bytes);
        }
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }

    pub(super) fn child() {
        let mode = std::env::var(CHILD_MODE).expect("subprocess fixture only");
        let root = std::path::PathBuf::from(std::env::var_os(CHILD_ROOT).unwrap());
        let reference = std::env::var(CHILD_REFERENCE).unwrap();
        let admission = admission(128 * 1024 * 1024);
        let mut selected_options = options(&root);
        if mode == "timeout" {
            selected_options.lock_timeout = Duration::from_millis(150);
        }
        let store = match CacheStore::open(selected_options) {
            Ok(store) => store,
            Err(Error::Limit(_)) if mode == "timeout" => {
                assert_no_host_reservation(&admission);
                return;
            }
            Err(error) => panic!("child could not open its cache: {error}"),
        };
        struct ChildHook {
            point: CachePoint,
            crash: bool,
            crash_ordinal: usize,
            calls: AtomicUsize,
            reads: Arc<AtomicUsize>,
            held: AtomicBool,
        }
        impl CacheHook for ChildHook {
            fn point(&self, point: CachePoint) -> Result<()> {
                let ordinal = self.calls.fetch_add(1, Ordering::Relaxed);
                if self.crash && ordinal == self.crash_ordinal {
                    // A real process exit without Rust Drop, on this same
                    // running kernel. This is not a power-failure model.
                    println!(
                        "CACHE_HELPER_EXIT body_reads={}",
                        self.reads.load(Ordering::Relaxed)
                    );
                    io::stdout().flush().unwrap();
                    std::process::exit(86);
                }
                if point == self.point && !self.held.swap(true, Ordering::AcqRel) {
                    if self.crash {
                        return Ok(());
                    }
                    println!("CACHE_HELPER_LOCK_HELD");
                    io::stdout().flush().unwrap();
                    let mut command = String::new();
                    io::stdin().lock().read_line(&mut command).unwrap();
                    assert_eq!(command.trim(), "release");
                }
                Ok(())
            }
        }
        match mode.as_str() {
            "crash" => {
                let index: usize = std::env::var(CHILD_POINT).unwrap().parse().unwrap();
                let reads = Arc::new(AtomicUsize::new(0));
                let hook = Arc::new(ChildHook {
                    point: points()[0],
                    crash: true,
                    crash_ordinal: index,
                    calls: AtomicUsize::new(0),
                    reads: reads.clone(),
                    held: AtomicBool::new(false),
                });
                let mut reader = ObservedReader {
                    bytes: NEW,
                    reads,
                    observe: Box::new(|| {}),
                };
                let result = store
                    .with_hook(hook)
                    .begin(
                        payload_manifest(NEW),
                        &admission,
                        Arc::new(AtomicBool::new(false)),
                    )
                    .and_then(|mut transaction| {
                        transaction.import_object(PAYLOAD_PATH, &mut reader)?;
                        transaction.commit(Some(&reference))
                    });
                assert!(result.is_err(), "crash point was not reached");
                panic!("crash point returned instead of exiting");
            }
            "hold" | "hold-object" => {
                use zerogravity_artifacts::cache::{CacheMoment, CacheOperation};
                let hook = Arc::new(ChildHook {
                    point: if mode == "hold" {
                        CachePoint {
                            operation: CacheOperation::RefRename,
                            moment: CacheMoment::Before,
                            object: None,
                        }
                    } else {
                        CachePoint {
                            operation: CacheOperation::ObjectInstall,
                            moment: CacheMoment::Before,
                            object: Some(0),
                        }
                    },
                    crash: false,
                    crash_ordinal: usize::MAX,
                    calls: AtomicUsize::new(0),
                    reads: Arc::new(AtomicUsize::new(0)),
                    held: AtomicBool::new(false),
                });
                durable(
                    publish(&store.with_hook(hook), &admission, NEW, Some(&reference)).unwrap(),
                );
            }
            "timeout" => {
                assert!(matches!(
                    publish(&store, &admission, THIRD, Some(&reference)),
                    Err(Error::Limit(_))
                ));
            }
            "publish-third" => {
                durable(publish(&store, &admission, THIRD, Some(&reference)).unwrap());
            }
            "canceled" => {
                assert!(matches!(
                    store.begin(
                        payload_manifest(NEW),
                        &admission,
                        Arc::new(AtomicBool::new(true))
                    ),
                    Err(Error::State(_))
                ));
            }
            "hold-reservation" => {
                let transaction = store
                    .begin(
                        quota_manifest(),
                        &admission,
                        Arc::new(AtomicBool::new(false)),
                    )
                    .unwrap();
                println!("CACHE_HELPER_RESERVATION_HELD");
                io::stdout().flush().unwrap();
                let mut command = String::new();
                io::stdin().lock().read_line(&mut command).unwrap();
                assert_eq!(command.trim(), "release");
                drop(transaction);
            }
            _ => panic!("unknown test subprocess mode"),
        }
        assert_no_host_reservation(&admission);
    }
}

#[cfg(feature = "diagnostic-cache")]
#[test]
#[ignore = "owned subprocess entry point; invoked explicitly by transaction fixtures"]
fn cache_child_worker() {
    diagnostics::child();
}
