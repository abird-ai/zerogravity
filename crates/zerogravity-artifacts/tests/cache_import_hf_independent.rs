//! Tiny offline source fixtures, actual opened descriptors and original publisher.
#![cfg(all(feature = "cache", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use std::fs;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, symlink};
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::cache::import_hf::{HfImportEntry, HfImportRequest};
use zerogravity_artifacts::cache::{
    CacheLimits, CacheOptions, CacheStore, DigestProvenance, FileRole,
};
const COMMIT: &str = "1111111111111111111111111111111111111111";
const CONTENT: &[u8] = b"owned offline import fixture";
struct Local {
    fixture: Fixture,
    source: PathBuf,
    destination: PathBuf,
    blob: PathBuf,
    logical: PathBuf,
}
impl Local {
    fn new() -> Self {
        let fixture = Fixture::new();
        let source = fixture.base.join("models--owner--repo");
        let destination = fixture.base.join("import-destination");
        fs::create_dir_all(source.join(format!("snapshots/{COMMIT}"))).unwrap();
        fs::create_dir(source.join("blobs")).unwrap();
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&destination)
            .unwrap();
        // A Git-style40hex name is deliberately NOT this file's SHA256.
        let blob = source.join(format!("blobs/{}", "a".repeat(40)));
        fs::write(&blob, CONTENT).unwrap();
        let logical = source.join(format!("snapshots/{COMMIT}/note.txt"));
        symlink(format!("../../blobs/{}", "a".repeat(40)), &logical).unwrap();
        Self {
            fixture,
            source,
            destination,
            blob,
            logical,
        }
    }
    fn request(&self, expected: Option<String>) -> HfImportRequest {
        HfImportRequest::new(
            self.source.clone(),
            "owner/repo",
            COMMIT,
            vec![HfImportEntry::new("note.txt", FileRole::Auxiliary, expected).unwrap()],
        )
        .unwrap()
    }
    fn options(&self) -> CacheOptions {
        options(&self.destination)
    }
}
type Stamp = (u64, u64, u32, u64, i64, i64, i64, i64);
fn stamp(path: &Path) -> Stamp {
    let m = fs::symlink_metadata(path).unwrap();
    (
        m.dev(),
        m.ino(),
        m.mode(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    )
}
#[test]
fn selected_original_bytes_are_copied_not_linked_and_source_survives_removal() {
    let f = Local::new();
    let admission = admission(128 * 1024 * 1024);
    let before = (stamp(&f.source), stamp(&f.blob), stamp(&f.logical));
    let plan = f
        .request(None)
        .prepare(
            &CacheLimits::default(),
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    assert_eq!(plan.source().provider(), "local-hf-cache");
    assert_eq!(plan.total_bytes(), CONTENT.len() as u64);
    assert_eq!(plan.entries().len(), 1);
    plan.check_destination(&f.options()).unwrap();
    assert_eq!(fs::read_dir(&f.destination).unwrap().count(), 0);
    let receipt = plan.apply(f.options()).unwrap();
    let id = match receipt.outcome() {
        zerogravity_artifacts::cache::CommitOutcome::Durable(s) => s.digest().to_string(),
        other => panic!("{other:?}"),
    };
    assert_eq!(
        receipt.entries()[0].digest_provenance(),
        DigestProvenance::FirstSeen
    );
    assert_eq!(receipt.entries()[0].digest(), digest(CONTENT));
    assert_eq!(receipt.entries()[0].copied_bytes(), CONTENT.len() as u64);
    let store = CacheStore::open_existing(f.options()).unwrap();
    let lease = store.open_snapshot(&id).unwrap();
    let object = lease.file("note.txt").unwrap();
    assert_eq!(file_bytes(object), CONTENT);
    assert_ne!(
        (
            object.metadata().unwrap().dev(),
            object.metadata().unwrap().ino()
        ),
        (stamp(&f.blob).0, stamp(&f.blob).1)
    );
    assert_eq!(
        before,
        (stamp(&f.source), stamp(&f.blob), stamp(&f.logical))
    );
    assert_eq!(fs::read(&f.blob).unwrap(), CONTENT);
    drop(lease);
    drop(receipt);
    drop(store);
    assert_no_host_reservation(&admission);
    fs::remove_dir_all(&f.destination).unwrap();
    assert_eq!(fs::read(&f.blob).unwrap(), CONTENT);
    assert_eq!(
        fs::read_link(&f.logical).unwrap(),
        PathBuf::from(format!("../../blobs/{}", "a".repeat(40)))
    );
    let _ = &f.fixture;
}
#[test]
fn overlap_absence_and_aliases_refuse_before_destination_initialization() {
    for case in 0..5 {
        let f = Local::new();
        let admission = admission(128 * 1024 * 1024);
        let plan = f
            .request(None)
            .prepare(
                &CacheLimits::default(),
                &admission,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        let destination = match case {
            0 => f.source.clone(),
            1 => {
                let p = f.source.join("destination");
                fs::DirBuilder::new().mode(0o700).create(&p).unwrap();
                p
            }
            2 => f.fixture.base.clone(),
            3 => f.source.join("absent-child"),
            _ => {
                let p = f.fixture.base.join("alias");
                symlink(&f.source, &p).unwrap();
                p
            }
        };
        let options = options(&destination);
        assert!(plan.check_destination(&options).is_err(), "case{case}");
        assert!(plan.apply(options).is_err(), "case{case}");
        assert!(!f.source.join("policy.json").exists());
        assert!(!f.fixture.base.join("policy.json").exists());
        assert!(!f.source.join("destination/policy.json").exists());
        assert!(!f.source.join("absent-child").exists());
        assert_eq!(fs::read(&f.blob).unwrap(), CONTENT);
        assert_no_host_reservation(&admission);
    }
}
#[test]
fn path_escapes_chained_symlinks_and_special_files_refuse_without_copy() {
    for case in 0..5 {
        let f = Local::new();
        fs::remove_file(&f.logical).unwrap();
        match case {
            0 => symlink("../../../outside", &f.logical).unwrap(),
            1 => {
                let chain = f.source.join(format!("blobs/{}", "b".repeat(40)));
                symlink(&f.blob, &chain).unwrap();
                symlink(format!("../../blobs/{}", "b".repeat(40)), &f.logical).unwrap();
            }
            2 => symlink("/dev/null", &f.logical).unwrap(),
            3 => {
                rustix::fs::mknodat(
                    rustix::fs::CWD,
                    &f.logical,
                    rustix::fs::FileType::Fifo,
                    rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
                    0,
                )
                .unwrap();
            }
            _ => {
                // sockaddr_un is shorter than a canonical HF snapshot path.
                // Bind privately at a short name, then move the socket inode.
                let short = f.fixture.base.join("socket");
                let socket = std::os::unix::net::UnixListener::bind(&short).unwrap();
                fs::rename(short, &f.logical).unwrap();
                drop(socket);
            }
        }
        let admission = admission(128 * 1024 * 1024);
        assert!(
            f.request(None)
                .prepare(
                    &CacheLimits::default(),
                    &admission,
                    Arc::new(AtomicBool::new(false))
                )
                .is_err(),
            "case{case}"
        );
        assert_eq!(fs::read_dir(&f.destination).unwrap().count(), 0);
        assert_no_host_reservation(&admission);
    }
}
#[test]
fn retained_fd_never_imports_a_replaced_path_or_changed_source() {
    for case in 0..3 {
        let f = Local::new();
        let admission = admission(128 * 1024 * 1024);
        let plan = f
            .request(Some(digest(CONTENT)))
            .prepare(
                &CacheLimits::default(),
                &admission,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        match case {
            0 => {
                fs::rename(&f.blob, f.source.join("old-original")).unwrap();
                fs::write(&f.blob, b"attacker replacement").unwrap();
            }
            1 => fs::write(&f.blob, vec![b'x'; CONTENT.len()]).unwrap(),
            _ => fs::write(&f.blob, b"truncated").unwrap(),
        }
        let result = plan.apply(f.options());
        // The original descriptor can either survive a pathname-only rename or
        // fail its ctime stability check. It must never publish substituted data.
        if case == 0 {
            if let Ok(receipt) = result {
                assert_eq!(receipt.entries()[0].digest(), digest(CONTENT));
                let store = CacheStore::open_existing(f.options()).unwrap();
                let snapshot = store.list_snapshots().unwrap();
                assert_eq!(
                    file_bytes(
                        store
                            .open_snapshot(&snapshot[0])
                            .unwrap()
                            .file("note.txt")
                            .unwrap()
                    ),
                    CONTENT
                );
            }
        } else {
            assert!(result.is_err(), "changed source accepted case{case}");
        }
        assert_no_host_reservation(&admission);
    }
}
#[test]
fn wrong_digest_cancel_and_denied_control_admission_preserve_existing_data() {
    let f = Local::new();
    let admission = admission(128 * 1024 * 1024);
    let store = CacheStore::open(f.options()).unwrap();
    let prior = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let plan = f
        .request(Some(digest(b"wrong")))
        .prepare(
            &CacheLimits::default(),
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    assert!(plan.apply(f.options()).is_err());
    assert_eq!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        prior
    );
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
    let cancel = Arc::new(AtomicBool::new(false));
    let plan = f
        .request(None)
        .prepare(&CacheLimits::default(), &admission, cancel.clone())
        .unwrap();
    cancel.store(true, std::sync::atomic::Ordering::Release);
    assert!(plan.apply(f.options()).is_err());
    assert_eq!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        prior
    );
    assert_no_host_reservation(&admission);
    let denied = admission_fixture(1);
    assert!(
        f.request(None)
            .prepare(
                &CacheLimits::default(),
                &denied,
                Arc::new(AtomicBool::new(false))
            )
            .is_err()
    );
    assert_no_host_reservation(&denied);
}
fn admission_fixture(limit: u64) -> zerogravity_artifacts::cache::TransferAdmission {
    cache_transaction_support::admission(limit)
}

#[cfg(feature = "diagnostic-cache")]
#[test]
fn mutation_after_copy_before_install_refuses_new_snapshot_and_keeps_old_ref() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use zerogravity_artifacts::cache::{CacheHook, CacheMoment, CacheOperation, CachePoint};
    struct Mutate {
        path: PathBuf,
        reached: AtomicBool,
    }
    impl CacheHook for Mutate {
        fn point(&self, point: CachePoint) -> zerogravity_artifacts::Result<()> {
            if point.operation == CacheOperation::ObjectInstall
                && point.moment == CacheMoment::Before
                && !self.reached.swap(true, Ordering::SeqCst)
            {
                fs::write(&self.path, vec![b'x'; CONTENT.len()])?;
            }
            Ok(())
        }
    }
    let f = Local::new();
    let admission = admission(128 * 1024 * 1024);
    let store = CacheStore::open(f.options()).unwrap();
    let prior = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let plan = f
        .request(Some(digest(CONTENT)))
        .prepare(
            &CacheLimits::default(),
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    let hook = Arc::new(Mutate {
        path: f.blob.clone(),
        reached: AtomicBool::new(false),
    });
    assert!(plan.apply_with_hook(f.options(), hook.clone()).is_err());
    assert!(hook.reached.load(Ordering::SeqCst));
    assert_eq!(store.list_snapshots().unwrap(), vec![prior.clone()]);
    assert_eq!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        prior
    );
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}
