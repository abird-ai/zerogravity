//! Explicit physical qualification with tiny private directories; no mounts.
#![cfg(all(feature = "cache", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use std::fs::{self, File};
use std::os::fd::AsFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, symlink};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::cache::import_hf::{HfImportEntry, HfImportPlan, HfImportRequest};
use zerogravity_artifacts::cache::{
    CacheLimits, CacheOptions, CacheStore, CommitOutcome, DigestProvenance, FileRole,
};
const COMMIT: &str = "1111111111111111111111111111111111111111";
const CONTENT: &[u8] = b"original cross-filesystem fixture";
const BTRFS: u64 = 0x9123_683e;
const TMPFS: u64 = 0x0102_1994;
#[derive(Debug, Clone, Copy)]
struct Profile {
    device: u64,
    mount: u64,
    kind: u64,
}
fn profile(path: &Path) -> Profile {
    let file = File::open(path).unwrap();
    let stat = rustix::fs::statx(
        file.as_fd(),
        "",
        rustix::fs::AtFlags::EMPTY_PATH,
        rustix::fs::StatxFlags::MNT_ID,
    )
    .unwrap();
    assert_ne!(
        stat.stx_mask & rustix::fs::StatxFlags::MNT_ID.bits(),
        0,
        "mount identity required"
    );
    Profile {
        device: file.metadata().unwrap().dev(),
        mount: stat.stx_mnt_id,
        kind: u64::from(rustix::fs::fstatfs(&file).unwrap().f_type as u32),
    }
}
struct PrivateDir(PathBuf);
impl PrivateDir {
    fn new(parent: &Path, label: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = parent.join(format!(
            "zg-crossfs-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        Self(root)
    }
}
impl Drop for PrivateDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct CrossFixture {
    left: PrivateDir,
    right: PrivateDir,
    source: PathBuf,
    target: PathBuf,
}
impl CrossFixture {
    fn new(source_parent: &Path, target_parent: &Path) -> Self {
        let left = PrivateDir::new(source_parent, "source");
        let right = PrivateDir::new(target_parent, "target");
        let source = left.0.join("models--owner--repo");
        let target = right.0.join("cache");
        fs::create_dir_all(source.join(format!("snapshots/{COMMIT}/nested"))).unwrap();
        fs::create_dir(source.join("blobs")).unwrap();
        fs::DirBuilder::new().mode(0o700).create(&target).unwrap();
        fs::write(source.join("blobs").join("a".repeat(40)), CONTENT).unwrap();
        fs::write(
            source.join(format!("snapshots/{COMMIT}/direct.bin")),
            CONTENT,
        )
        .unwrap();
        fs::write(source.join(format!("snapshots/{COMMIT}/empty.bin")), b"").unwrap();
        fs::write(
            source.join(format!("snapshots/{COMMIT}/nested/direct.bin")),
            b"nested",
        )
        .unwrap();
        symlink(
            format!("../../../blobs/{}", "a".repeat(40)),
            source.join(format!("snapshots/{COMMIT}/nested/linked.bin")),
        )
        .unwrap();
        Self {
            left,
            right,
            source,
            target,
        }
    }
    fn request(&self, trusted: bool) -> HfImportRequest {
        let entries = [
            ("direct.bin", CONTENT),
            ("empty.bin", b"".as_slice()),
            ("nested/direct.bin", b"nested".as_slice()),
            ("nested/linked.bin", CONTENT),
        ]
        .into_iter()
        .map(|(path, bytes)| {
            HfImportEntry::new(path, FileRole::Auxiliary, trusted.then(|| digest(bytes))).unwrap()
        })
        .collect();
        HfImportRequest::new(self.source.clone(), "owner/repo", COMMIT, entries).unwrap()
    }
    fn prepare(
        &self,
        trusted: bool,
        admission: &zerogravity_artifacts::cache::TransferAdmission,
        cancel: Arc<AtomicBool>,
    ) -> HfImportPlan {
        self.request(trusted)
            .prepare(&CacheLimits::default(), admission, cancel)
            .unwrap()
    }
    fn options(&self) -> CacheOptions {
        options(&self.target)
    }
}
fn repository_scratch() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .canonicalize()
        .unwrap()
}
fn physical_pairs(tmpfs_only: bool) -> [(PathBuf, PathBuf); 2] {
    let left = if tmpfs_only {
        PathBuf::from("/tmp")
    } else {
        repository_scratch()
    };
    let right = PathBuf::from("/dev/shm");
    let a = profile(&left);
    let b = profile(&right);
    assert_ne!(
        a.device, b.device,
        "physical test needs distinct devices: {a:?}/{b:?}"
    );
    assert_ne!(
        a.mount, b.mount,
        "physical test needs distinct mounts: {a:?}/{b:?}"
    );
    assert_eq!(
        a.kind,
        if tmpfs_only { TMPFS } else { BTRFS },
        "unqualified source profile {a:?}"
    );
    assert_eq!(b.kind, TMPFS, "unqualified destination profile {b:?}");
    eprintln!(
        "physical import roots: {} {a:?} <-> {} {b:?}",
        left.display(),
        right.display()
    );
    [(left.clone(), right.clone()), (right, left)]
}
type Stamp = (u64, u64, u32, u64, u64, i64, i64, i64, i64);
fn stamp(path: &Path) -> Stamp {
    let m = fs::symlink_metadata(path).unwrap();
    (
        m.dev(),
        m.ino(),
        m.mode(),
        m.nlink(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    )
}
fn source_stamps(source: &Path) -> Vec<(PathBuf, Stamp)> {
    let mut pending = vec![source.to_path_buf()];
    let mut result = Vec::new();
    while let Some(path) = pending.pop() {
        let m = fs::symlink_metadata(&path).unwrap();
        result.push((path.clone(), stamp(&path)));
        if m.is_dir() {
            pending.extend(
                fs::read_dir(path)
                    .unwrap()
                    .map(|entry| entry.unwrap().path()),
            );
        }
    }
    result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    result
}
fn id(outcome: &CommitOutcome) -> String {
    match outcome {
        CommitOutcome::Durable(snapshot) => snapshot.digest().to_string(),
        other => panic!("{other:?}"),
    }
}
fn qualify_copy(source: &Path, target: &Path) {
    let f = CrossFixture::new(source, target);
    let admission = admission(128 * 1024 * 1024);
    let before = source_stamps(&f.source);
    let plan = f.prepare(false, &admission, Arc::new(AtomicBool::new(false)));
    plan.check_destination(&f.options()).unwrap();
    assert_eq!(fs::read_dir(&f.target).unwrap().count(), 0);
    let receipt = plan.apply(f.options()).unwrap();
    let digest = id(receipt.outcome());
    assert_eq!(receipt.source().provider(), "local-hf-cache");
    assert_eq!(receipt.entries().len(), 4);
    for entry in receipt.entries() {
        assert_eq!(entry.digest_provenance(), DigestProvenance::FirstSeen);
        assert_eq!(entry.bytes(), entry.copied_bytes());
    }
    assert_eq!(before, source_stamps(&f.source));
    let store = CacheStore::open_existing(f.options()).unwrap();
    let lease = store.open_snapshot(&digest).unwrap();
    for (path, bytes) in [
        ("direct.bin", CONTENT),
        ("empty.bin", b"".as_slice()),
        ("nested/direct.bin", b"nested".as_slice()),
        ("nested/linked.bin", CONTENT),
    ] {
        let file = lease.file(path).unwrap();
        assert_eq!(file_bytes(file), bytes);
        assert_eq!(file.metadata().unwrap().dev(), profile(target).device);
        let entry = lease
            .manifest()
            .entries()
            .iter()
            .find(|e| e.path() == path)
            .unwrap();
        assert_eq!(entry.bytes(), bytes.len() as u64);
        assert_eq!(entry.digest(), cache_transaction_support::digest(bytes));
    }
    assert_no_host_reservation(&admission);
    fs::remove_dir_all(&f.source).unwrap();
    for (path, bytes) in [("direct.bin", CONTENT), ("nested/linked.bin", CONTENT)] {
        assert_eq!(file_bytes(lease.file(path).unwrap()), bytes);
    }
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    let _ = (&f.left, &f.right);
}
#[test]
fn same_mount_public_api_compatibility_uses_the_original_publisher() {
    let parent = std::env::temp_dir();
    qualify_copy(&parent, &parent);
}
#[test]
#[ignore = "requires actual writable distinct Btrfs/tmpfs roots; run explicitly, no runtime skips"]
fn btrfs_tmpfs_both_directions_preserve_zero_nested_regular_and_hf_links() {
    for (source, target) in physical_pairs(false) {
        qualify_copy(&source, &target);
    }
}
#[test]
#[ignore = "requires actual writable distinct tmpfs mounts at /tmp and /dev/shm; run explicitly"]
fn distinct_tmpfs_both_directions_preserve_original_bytes() {
    for (source, target) in physical_pairs(true) {
        qualify_copy(&source, &target);
    }
}
#[test]
#[ignore = "requires actual Btrfs/tmpfs roots and original public cache-hit verification"]
fn crossfs_trusted_hits_still_hash_external_bytes_and_refuse_changed_source() {
    for (source, target) in physical_pairs(false) {
        let f = CrossFixture::new(&source, &target);
        let admission = admission(128 * 1024 * 1024);
        let first = f
            .prepare(true, &admission, Arc::new(AtomicBool::new(false)))
            .apply(f.options())
            .unwrap();
        let first_id = id(first.outcome());
        let second = f
            .prepare(true, &admission, Arc::new(AtomicBool::new(false)))
            .apply(f.options())
            .unwrap();
        assert_eq!(id(second.outcome()), first_id);
        assert!(
            second
                .entries()
                .iter()
                .all(|entry| entry.copied_bytes() == 0)
        );
        fs::write(
            f.source.join(format!("snapshots/{COMMIT}/direct.bin")),
            vec![b'x'; CONTENT.len()],
        )
        .unwrap();
        // Same length and stable metadata at preparation; only full actual
        // external-body hashing detects that a valid target hit is insufficient.
        assert!(
            f.prepare(true, &admission, Arc::new(AtomicBool::new(false)))
                .apply(f.options())
                .is_err()
        );
        let store = CacheStore::open_existing(f.options()).unwrap();
        assert_eq!(store.list_snapshots().unwrap(), vec![first_id.clone()]);
        assert_eq!(
            file_bytes(
                store
                    .open_snapshot(&first_id)
                    .unwrap()
                    .file("direct.bin")
                    .unwrap()
            ),
            CONTENT
        );
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}
#[test]
#[ignore = "requires actual Btrfs/tmpfs roots; exercises rollback/cancel/alias boundaries"]
fn crossfs_hash_cancel_admission_and_alias_refusals_preserve_existing_refs() {
    for (source, target) in physical_pairs(false) {
        let f = CrossFixture::new(&source, &target);
        let admission = admission(128 * 1024 * 1024);
        let store = CacheStore::open(f.options()).unwrap();
        let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let request = HfImportRequest::new(
            f.source.clone(),
            "owner/repo",
            COMMIT,
            vec![
                HfImportEntry::new("direct.bin", FileRole::Auxiliary, Some(digest(b"wrong")))
                    .unwrap(),
            ],
        )
        .unwrap();
        let plan = request
            .prepare(
                &CacheLimits::default(),
                &admission,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        assert!(plan.apply(f.options()).is_err());
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = f.prepare(false, &admission, cancel.clone());
        cancel.store(true, Ordering::Release);
        assert!(plan.apply(f.options()).is_err());
        let denied = cache_transaction_support::admission(1);
        assert!(
            f.request(false)
                .prepare(
                    &CacheLimits::default(),
                    &denied,
                    Arc::new(AtomicBool::new(false))
                )
                .is_err()
        );
        assert_no_host_reservation(&denied);
        let alias = f.right.0.join("alias");
        symlink(&f.source, &alias).unwrap();
        let plan = f.prepare(false, &admission, Arc::new(AtomicBool::new(false)));
        assert!(plan.apply(options(&alias)).is_err());
        let source_child = f.source.join("target");
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&source_child)
            .unwrap();
        let plan = f.prepare(false, &admission, Arc::new(AtomicBool::new(false)));
        assert!(plan.apply(options(&source_child)).is_err());
        assert!(!source_child.join("policy.json").exists());
        assert_eq!(
            store
                .resolve_ref("keep")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            previous
        );
        assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
        assert_eq!(
            fs::read(f.source.join("blobs").join("a".repeat(40))).unwrap(),
            CONTENT
        );
    }
}

#[cfg(feature = "diagnostic-cache")]
#[test]
#[ignore = "requires actual Btrfs/tmpfs roots and real post-copy publisher hook"]
fn crossfs_postcopy_cancellation_preserves_old_reference_and_source() {
    use zerogravity_artifacts::cache::{CacheHook, CacheMoment, CacheOperation, CachePoint};
    struct CancelAtInstall {
        cancel: Arc<AtomicBool>,
        reached: AtomicBool,
    }
    impl CacheHook for CancelAtInstall {
        fn point(&self, point: CachePoint) -> zerogravity_artifacts::Result<()> {
            if point.operation == CacheOperation::ObjectInstall
                && point.moment == CacheMoment::Before
            {
                self.reached.store(true, Ordering::Release);
                self.cancel.store(true, Ordering::Release);
            }
            Ok(())
        }
    }
    for (source, target) in physical_pairs(false) {
        let f = CrossFixture::new(&source, &target);
        let admission = admission(128 * 1024 * 1024);
        let store = CacheStore::open(f.options()).unwrap();
        let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let before = source_stamps(&f.source);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = f.prepare(false, &admission, cancel.clone());
        let hook = Arc::new(CancelAtInstall {
            cancel,
            reached: AtomicBool::new(false),
        });
        assert!(plan.apply_with_hook(f.options(), hook.clone()).is_err());
        assert!(
            hook.reached.load(Ordering::Acquire),
            "post-copy hook was not reached"
        );
        assert_eq!(source_stamps(&f.source), before);
        assert_eq!(store.list_snapshots().unwrap(), vec![previous.clone()]);
        assert_eq!(
            store
                .resolve_ref("keep")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            previous
        );
        assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}
