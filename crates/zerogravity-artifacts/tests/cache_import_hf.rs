//! Explicit local imports retain external descriptors and the original cache owner.
#![cfg(all(feature = "cache", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;

use cache_transaction_support::*;
use std::fs::{self, File};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::Error;
use zerogravity_artifacts::cache::import_hf::{HfImportEntry, HfImportPlan, HfImportRequest};
use zerogravity_artifacts::cache::{
    CacheLimits, CacheOptions, CacheStore, CommitOutcome, DigestProvenance, FileRole,
    TransferAdmission,
};

const REPOSITORY: &str = "owner/repo";
const COMMIT: &str = "1111111111111111111111111111111111111111";

struct HfFixture {
    cache: Fixture,
    source: PathBuf,
}

impl HfFixture {
    fn new() -> Self {
        let cache = Fixture::new();
        let source = cache.base.join("models--owner--repo");
        fs::DirBuilder::new().mode(0o700).create(&source).unwrap();
        fs::create_dir_all(source.join("snapshots").join(COMMIT)).unwrap();
        fs::create_dir(source.join("blobs")).unwrap();
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&cache.root)
            .unwrap();
        Self { cache, source }
    }

    fn file(&self, logical: &str, bytes: &[u8]) -> PathBuf {
        let path = self.source.join("snapshots").join(COMMIT).join(logical);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        path
    }

    fn request(&self, entries: Vec<HfImportEntry>) -> HfImportRequest {
        HfImportRequest::new(self.source.clone(), REPOSITORY, COMMIT, entries).unwrap()
    }

    fn plan(&self, entries: Vec<HfImportEntry>, memory: &TransferAdmission) -> HfImportPlan {
        self.request(entries)
            .prepare(
                &CacheLimits::default(),
                memory,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap()
    }

    fn options(&self) -> CacheOptions {
        options(&self.cache.root)
    }
}

fn selected(path: &str, role: FileRole, expected: Option<&[u8]>) -> HfImportEntry {
    HfImportEntry::new(path, role, expected.map(digest)).unwrap()
}

fn snapshot(outcome: &CommitOutcome) -> &str {
    match outcome {
        CommitOutcome::Durable(snapshot) => snapshot.digest(),
        CommitOutcome::DurabilityUnknown { cause, .. } => panic!("unexpected uncertainty: {cause}"),
    }
}

fn stamp(path: &Path) -> (u64, u64, u64, u32, i64, i64) {
    let m = fs::metadata(path).unwrap();
    (
        m.dev(),
        m.ino(),
        m.len(),
        m.mode(),
        m.mtime(),
        m.mtime_nsec(),
    )
}

#[test]
fn dry_run_is_readonly_and_holds_admitted_source_controls() {
    let fixture = HfFixture::new();
    let source = fixture.file("tiny.bin", NEW);
    let before = stamp(&source);
    let memory = admission(128 * 1024 * 1024);
    let plan = fixture.plan(
        vec![selected("tiny.bin", FileRole::Auxiliary, None)],
        &memory,
    );
    assert_eq!(plan.source().provider(), "local-hf-cache");
    assert_eq!(plan.source().repository(), REPOSITORY);
    assert_eq!(plan.source().revision(), COMMIT);
    assert_eq!(plan.total_bytes(), NEW.len() as u64);
    assert_eq!(plan.entries()[0].bytes(), NEW.len() as u64);
    assert_eq!(plan.entries()[0].expected_sha256(), None);
    assert_eq!(plan.entries()[0].role(), FileRole::Auxiliary);
    assert!(memory.arbiter.snapshot(&memory.resource).unwrap().reserved > 0);
    plan.check_destination(&fixture.options()).unwrap();
    assert_eq!(fs::read_dir(&fixture.cache.root).unwrap().count(), 0);
    assert_eq!(stamp(&source), before);
    assert_eq!(fs::read(&source).unwrap(), NEW);
    drop(plan);
    assert_no_host_reservation(&memory);
}

#[test]
fn exact_regular_and_nested_hf_links_copy_private_bytes_with_honest_provenance() {
    let fixture = HfFixture::new();
    let regular = fixture.file("config.json", b"{}");
    let blob = fixture.source.join("blobs").join(digest(NEW));
    fs::write(&blob, NEW).unwrap();
    let link = fixture
        .source
        .join("snapshots")
        .join(COMMIT)
        .join("nested/data.bin");
    fs::create_dir(link.parent().unwrap()).unwrap();
    symlink(format!("../../../blobs/{}", digest(NEW)), &link).unwrap();
    let before_regular = stamp(&regular);
    let before_blob = stamp(&blob);
    let memory = admission(128 * 1024 * 1024);
    let receipt = fixture
        .plan(
            vec![
                selected("nested/data.bin", FileRole::Auxiliary, None),
                selected("config.json", FileRole::Config, Some(b"{}")),
            ],
            &memory,
        )
        .apply(fixture.options())
        .unwrap();
    let store = CacheStore::open(fixture.options()).unwrap();
    let lease = store.open_snapshot(snapshot(receipt.outcome())).unwrap();
    assert_eq!(lease.manifest().source(), receipt.source());
    for entry in receipt.entries() {
        let (bytes, original, provenance) = if entry.path() == "config.json" {
            (&b"{}"[..], &regular, DigestProvenance::TrustedSha256)
        } else {
            (NEW, &blob, DigestProvenance::FirstSeen)
        };
        assert_eq!(entry.bytes(), bytes.len() as u64);
        assert_eq!(entry.copied_bytes(), bytes.len() as u64);
        assert_eq!(entry.digest(), digest(bytes));
        assert_eq!(entry.digest_provenance(), provenance);
        let owned = lease.file(entry.path()).unwrap();
        assert_eq!(file_bytes(owned), bytes);
        let metadata = owned.metadata().unwrap();
        assert_ne!(
            (metadata.dev(), metadata.ino()),
            (stamp(original).0, stamp(original).1)
        );
    }
    assert_eq!(stamp(&regular), before_regular);
    assert_eq!(stamp(&blob), before_blob);
    assert_eq!(
        fs::read_link(&link).unwrap(),
        PathBuf::from(format!("../../../blobs/{}", digest(NEW)))
    );
    // Only this private synthetic external source is removed, after publication.
    fs::remove_dir_all(&fixture.source).unwrap();
    assert_eq!(file_bytes(lease.file("nested/data.bin").unwrap()), NEW);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&memory);
}

#[test]
fn known_digest_reuse_still_hashes_external_body_and_never_calls_it_copied() {
    let fixture = HfFixture::new();
    let source = fixture.file("data.bin", NEW);
    let memory = admission(128 * 1024 * 1024);
    let entries = || vec![selected("data.bin", FileRole::Auxiliary, Some(NEW))];
    let first = fixture
        .plan(entries(), &memory)
        .apply(fixture.options())
        .unwrap();
    let first_snapshot = snapshot(first.outcome()).to_owned();
    let second = fixture
        .plan(entries(), &memory)
        .apply(fixture.options())
        .unwrap();
    assert_eq!(snapshot(second.outcome()), first_snapshot);
    assert_eq!(second.entries()[0].copied_bytes(), 0);
    assert_eq!(second.entries()[0].bytes(), NEW.len() as u64);
    assert_eq!(
        second.entries()[0].digest_provenance(),
        DigestProvenance::TrustedSha256
    );
    fs::write(&source, vec![b'x'; NEW.len()]).unwrap();
    // Metadata is freshly prepared and stable; only hashing the skipped source
    // body can catch this mismatch against the already verified cache object.
    assert!(
        fixture
            .plan(entries(), &memory)
            .apply(fixture.options())
            .is_err()
    );
    let store = CacheStore::open(fixture.options()).unwrap();
    assert_eq!(store.list_snapshots().unwrap(), vec![first_snapshot]);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&memory);
}

#[test]
fn overlap_missing_targets_and_selected_inode_alias_refuse_before_initialization() {
    let fixture = HfFixture::new();
    let source = fixture.file("data.bin", NEW);
    let memory = admission(128 * 1024 * 1024);
    for target in [fixture.source.clone(), fixture.cache.base.clone()] {
        let plan = fixture.plan(
            vec![selected("data.bin", FileRole::Auxiliary, None)],
            &memory,
        );
        assert!(plan.apply(options(&target)).is_err());
        assert!(!target.join("policy.json").exists());
        assert!(!target.join("locks").exists());
    }
    let nested = fixture.source.join("target");
    fs::DirBuilder::new().mode(0o700).create(&nested).unwrap();
    for target in [nested, fixture.cache.base.join("missing/child")] {
        let plan = fixture.plan(
            vec![selected("data.bin", FileRole::Auxiliary, None)],
            &memory,
        );
        assert!(plan.apply(options(&target)).is_err());
        assert!(!target.join("policy.json").exists());
        assert!(!target.join("locks").exists());
    }
    assert!(!fixture.cache.base.join("missing").exists());
    let alias = fixture.cache.base.join("destination-alias");
    symlink(&fixture.source, &alias).unwrap();
    assert!(
        fixture
            .plan(
                vec![selected("data.bin", FileRole::Auxiliary, None)],
                &memory
            )
            .apply(options(&alias))
            .is_err()
    );
    fs::hard_link(&source, fixture.cache.root.join("source-alias")).unwrap();
    assert!(
        fixture
            .plan(
                vec![selected("data.bin", FileRole::Auxiliary, None)],
                &memory
            )
            .apply(fixture.options())
            .is_err()
    );
    assert_eq!(fs::read_dir(&fixture.cache.root).unwrap().count(), 1);
    assert_no_host_reservation(&memory);
}

#[test]
fn source_mutation_or_replacement_after_prepare_cannot_publish() {
    let fixture = HfFixture::new();
    let source = fixture.file("data.bin", NEW);
    let memory = admission(128 * 1024 * 1024);
    for mutation in 0..3 {
        fs::write(&source, NEW).unwrap();
        let plan = fixture.plan(
            vec![selected("data.bin", FileRole::Auxiliary, None)],
            &memory,
        );
        match mutation {
            0 => fs::write(&source, vec![b'x'; NEW.len()]).unwrap(),
            1 => File::options()
                .write(true)
                .open(&source)
                .unwrap()
                .set_len(1)
                .unwrap(),
            _ => {
                fs::rename(&source, source.with_extension("old")).unwrap();
                fs::write(&source, NEW).unwrap();
            }
        }
        assert!(plan.apply(fixture.options()).is_err());
        if fixture.cache.root.join("policy.json").exists() {
            let store = CacheStore::open(fixture.options()).unwrap();
            assert!(store.list_snapshots().unwrap().is_empty());
            assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        }
        assert_no_host_reservation(&memory);
    }
}

#[test]
fn denied_control_payload_quota_limits_and_cancellation_publish_nothing() {
    let fixture = HfFixture::new();
    fixture.file("data.bin", NEW);
    let request = || fixture.request(vec![selected("data.bin", FileRole::Auxiliary, None)]);
    let denied = admission(1);
    let missing = fixture.cache.base.join("models--missing--repo");
    let invalid_source = HfImportRequest::new(
        missing,
        "missing/repo",
        COMMIT,
        vec![selected("data.bin", FileRole::Auxiliary, None)],
    )
    .unwrap();
    assert!(matches!(
        invalid_source.prepare(
            &CacheLimits::default(),
            &denied,
            Arc::new(AtomicBool::new(false))
        ),
        Err(Error::Limit(_))
    ));
    assert_no_host_reservation(&denied);
    let memory = admission(128 * 1024 * 1024);
    let limits = CacheLimits {
        max_file_bytes: 1,
        ..CacheLimits::default()
    };
    assert!(
        request()
            .prepare(&limits, &memory, Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    assert_no_host_reservation(&memory);
    let payload_denied = admission(32 * 1024 * 1024);
    let plan = fixture.plan(
        vec![selected("data.bin", FileRole::Auxiliary, None)],
        &payload_denied,
    );
    assert!(matches!(
        plan.apply(fixture.options()),
        Err(Error::Limit(_))
    ));
    assert_no_host_reservation(&payload_denied);
    let store = CacheStore::open(fixture.options()).unwrap();
    assert!(store.list_snapshots().unwrap().is_empty());
    let cancel = Arc::new(AtomicBool::new(false));
    let plan = request()
        .prepare(&CacheLimits::default(), &memory, cancel.clone())
        .unwrap();
    cancel.store(true, Ordering::Release);
    assert!(matches!(
        plan.apply(fixture.options()),
        Err(Error::State(_))
    ));
    assert_no_host_reservation(&memory);
    let plan = fixture.plan(
        vec![selected("data.bin", FileRole::Auxiliary, None)],
        &memory,
    );
    let mut tiny_disk = fixture.options();
    tiny_disk.disk_limit_bytes = 1;
    assert!(plan.apply(tiny_disk).is_err());
    assert!(store.list_snapshots().unwrap().is_empty());
    assert_no_host_reservation(&memory);
}

#[test]
fn malformed_explicit_selections_refuse_without_source_io() {
    let root = PathBuf::from("/absent/models--owner--repo");
    for commit in [
        "main",
        "",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "111111111111111111111111111111111111111",
    ] {
        assert!(
            HfImportRequest::new(
                root.clone(),
                REPOSITORY,
                commit,
                vec![selected("a", FileRole::Auxiliary, None)]
            )
            .is_err()
        );
    }
    for repository in [
        "owner",
        "owner/repo/extra",
        "../repo",
        "owner/..",
        "owner/r?x",
    ] {
        assert!(
            HfImportRequest::new(
                root.clone(),
                repository,
                COMMIT,
                vec![selected("a", FileRole::Auxiliary, None)]
            )
            .is_err()
        );
    }
    assert!(
        HfImportRequest::new(
            root.clone(),
            REPOSITORY,
            COMMIT,
            vec![
                selected("a", FileRole::Auxiliary, None),
                selected("a", FileRole::Config, None)
            ]
        )
        .is_err()
    );
    assert!(
        HfImportRequest::new(
            root,
            REPOSITORY,
            COMMIT,
            vec![
                selected("a", FileRole::Auxiliary, None),
                selected("a/b", FileRole::Auxiliary, None)
            ]
        )
        .is_err()
    );
    assert!(HfImportEntry::new("../outside", FileRole::Auxiliary, None).is_err());
    assert!(HfImportEntry::new("a", FileRole::Auxiliary, Some("A".repeat(64))).is_err());
}

#[cfg(feature = "diagnostic-cache")]
#[test]
fn postcopy_source_stability_and_original_faults_gate_snapshot_publication() {
    use zerogravity_artifacts::Result;
    use zerogravity_artifacts::cache::{CacheHook, CacheMoment, CacheOperation, CachePoint};

    struct ChangeSource(PathBuf);
    impl CacheHook for ChangeSource {
        fn point(&self, point: CachePoint) -> Result<()> {
            if point.operation == CacheOperation::ObjectFileSync
                && point.moment == CacheMoment::After
            {
                fs::write(&self.0, b"changed after original source EOF")?;
            }
            Ok(())
        }
    }
    struct FailPublication;
    impl CacheHook for FailPublication {
        fn point(&self, point: CachePoint) -> Result<()> {
            if point.operation == CacheOperation::SnapshotInstall
                && point.moment == CacheMoment::Before
            {
                return Err(std::io::Error::other("injected original publication failure").into());
            }
            Ok(())
        }
    }
    let fixture = HfFixture::new();
    let source = fixture.file("data.bin", NEW);
    let memory = admission(128 * 1024 * 1024);
    let store = CacheStore::open(fixture.options()).unwrap();
    let old_ref = durable(publish(&store, &memory, OLD, Some("unrelated-ref")).unwrap());
    let plan = fixture.plan(
        vec![selected("data.bin", FileRole::Auxiliary, None)],
        &memory,
    );
    assert!(
        plan.apply_with_hook(fixture.options(), Arc::new(ChangeSource(source.clone())))
            .is_err()
    );
    assert_eq!(store.list_snapshots().unwrap(), vec![old_ref.clone()]);
    assert_eq!(
        store
            .resolve_ref("unrelated-ref")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old_ref
    );
    fs::write(&source, NEW).unwrap();
    let plan = fixture.plan(
        vec![selected("data.bin", FileRole::Auxiliary, None)],
        &memory,
    );
    assert!(
        plan.apply_with_hook(fixture.options(), Arc::new(FailPublication))
            .is_err()
    );
    assert_eq!(store.list_snapshots().unwrap(), vec![old_ref.clone()]);
    assert_eq!(
        store
            .resolve_ref("unrelated-ref")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old_ref
    );
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&memory);
}

#[test]
fn local_pointer_payload_refuses_before_snapshot_and_caller_metadata_limits_remain_effective() {
    let fixture = HfFixture::new();
    let pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:not-an-object\nsize 9\n";
    fixture.file("model.gguf", pointer);
    let memory = admission(128 * 1024 * 1024);
    let plan = fixture.plan(
        vec![selected("model.gguf", FileRole::Weights, None)],
        &memory,
    );
    assert!(matches!(
        plan.apply(fixture.options()),
        Err(Error::Unsupported(_))
    ));
    let store = CacheStore::open(fixture.options()).unwrap();
    assert!(store.list_snapshots().unwrap().is_empty());
    let entries = (0..2500)
        .map(|index| selected(&format!("unopened-{index}"), FileRole::Auxiliary, None))
        .collect();
    let request = fixture.request(entries);
    let limits = CacheLimits {
        max_manifest_bytes: 8 * 1024 * 1024,
        ..CacheLimits::default()
    };
    // Construction uses the hard8MiB bound. Raised caller controls reach source
    // metadata instead of being silently replaced by the default1MiB envelope.
    assert!(
        matches!(request.prepare(&limits, &admission(128 * 1024 * 1024),
        Arc::new(AtomicBool::new(false))), Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound)
    );
    assert_no_host_reservation(&memory);
}
