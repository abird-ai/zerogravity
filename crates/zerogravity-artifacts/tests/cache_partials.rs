//! Synthetic local partials only: no provider, model, network or device work.
#![cfg(all(feature = "cache", target_os = "linux"))]

use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use zerogravity_artifacts::cache::{
    CacheOptions, CacheStore, CommitOutcome, DiscoveredEntry, DiscoveredTransaction, FileRole,
    Manifest, ManifestEntry, ManifestSource, TransferAdmission,
};
use zerogravity_artifacts::{Error, Result};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

const LIMIT: u64 = 8 * 1024 * 1024;
const BODY: &[u8] = b"the immutable synthetic replacement artifact";
const IDENTITY: &str = "test:revision:object:immutable";

struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let root = std::env::temp_dir().join(format!(
            "zerogravity-partial-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        Self { root }
    }
    fn open(&self) -> CacheStore {
        open(&self.root)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn open(root: &Path) -> CacheStore {
    let mut options = CacheOptions::new(Some(root.to_path_buf()), LIMIT);
    options.lock_timeout = Duration::from_millis(40);
    CacheStore::open(options).unwrap()
}
fn admission(capacity: u64) -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("test-host", "partial-cache").unwrap();
    arbiter.register(resource.clone(), capacity).unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "partial-cache").unwrap(),
    }
}
fn source(revision: &str) -> ManifestSource {
    ManifestSource::new("local", "synthetic-partials", revision).unwrap()
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn begin(
    store: &CacheStore,
    admission: &TransferAdmission,
    cancel: Arc<AtomicBool>,
) -> DiscoveredTransaction {
    begin_variant(
        store,
        admission,
        cancel,
        "immutable-v1",
        BODY.len() as u64,
        Some(hash(BODY)),
    )
    .unwrap()
}
fn begin_variant(
    store: &CacheStore,
    admission: &TransferAdmission,
    cancel: Arc<AtomicBool>,
    revision: &str,
    bytes: u64,
    digest: Option<String>,
) -> Result<DiscoveredTransaction> {
    store.begin_discovered(
        source(revision),
        vec![DiscoveredEntry::new(
            "payload.bin",
            bytes,
            FileRole::Auxiliary,
            digest,
        )?],
        admission,
        cancel,
    )
}
fn retained_prefix(store: &CacheStore, admission: &TransferAdmission) {
    let mut transaction = begin(store, admission, Arc::new(AtomicBool::new(false)));
    let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
    partial.set_validator("\"strong\"").unwrap();
    partial.write_chunk(&BODY[..8]).unwrap();
    partial.checkpoint().unwrap();
}
fn checkpoints(root: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    for attempt in fs::read_dir(root.join("staging")).unwrap() {
        for entry in fs::read_dir(attempt.unwrap().path()).unwrap() {
            let entry = entry.unwrap();
            if entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("partial-") && name.ends_with(".json"))
            {
                result.push(entry.path());
            }
        }
    }
    result
}
fn no_host_lease(admission: &TransferAdmission) {
    let snapshot = admission.arbiter.snapshot(&admission.resource).unwrap();
    assert_eq!(snapshot.reserved, 0);
    assert_eq!(snapshot.reservations, 0);
    assert!(!snapshot.fenced);
}
fn old_ref(fixture: &Fixture, store: &CacheStore, admission: &TransferAdmission) -> Vec<u8> {
    let body = b"previous complete reference";
    let manifest = Manifest::new(
        source("previous"),
        vec![
            ManifestEntry::new(
                "payload.bin",
                hash(body),
                body.len() as u64,
                FileRole::Auxiliary,
            )
            .unwrap(),
        ],
        &Default::default(),
    )
    .unwrap();
    let mut transaction = store
        .begin(manifest, admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    transaction
        .import_object("payload.bin", &mut body.as_slice())
        .unwrap();
    assert!(matches!(
        transaction.commit(Some("stable")).unwrap(),
        CommitOutcome::Durable(_)
    ));
    fs::read(fixture.root.join(format!("refs/{}.json", hash(b"stable")))).unwrap()
}

#[test]
fn durable_prefix_is_rehashed_and_crash_tail_is_not_copied() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let old = old_ref(&fixture, &store, &admission);
    {
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
        partial.set_validator("\"strong\"").unwrap();
        partial.write_chunk(&BODY[..8]).unwrap();
        partial.checkpoint().unwrap();
        partial.write_chunk(b"uncommitted wrong tail").unwrap();
    }
    let retained = store.usage().unwrap();
    assert_eq!(retained.reserved_bytes, 0);
    no_host_lease(&admission);
    drop(store);
    let store = fixture.open();
    let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    let live = store.usage().unwrap();
    assert!(live.used_bytes >= retained.used_bytes);
    assert!(live.reserved_bytes > 0);
    let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
    assert_eq!(partial.resume_bytes(), 8);
    assert_eq!(partial.bytes(), 8);
    assert_eq!(partial.validator(), Some("\"strong\""));
    let mut prefix = [0; 32];
    assert_eq!(partial.read_prefix(&mut prefix).unwrap(), 8);
    assert_eq!(&prefix[..8], &BODY[..8]);
    partial.write_chunk(&BODY[8..]).unwrap();
    assert_eq!(partial.finish().unwrap().digest(), hash(BODY));
    assert_eq!(
        fs::read(fixture.root.join(format!("refs/{}.json", hash(b"stable")))).unwrap(),
        old
    );
    let outcome = transaction.commit(Some("stable")).unwrap();
    assert!(matches!(outcome, CommitOutcome::Durable(_)));
    let snapshot = store.resolve_ref("stable").unwrap();
    let mut bytes = Vec::new();
    snapshot
        .file("payload.bin")
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes, BODY);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    // The old attempt remains named and charged; successful recovery does not
    // claim physical reclamation or delete a retained checkpoint's inodes.
    assert_eq!(checkpoints(&fixture.root).len(), 1);
    assert!(store.usage().unwrap().used_bytes >= retained.used_bytes);
    no_host_lease(&admission);
}

#[test]
fn changed_binding_never_restores_and_tampered_prefix_refuses() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    retained_prefix(&store, &admission);
    for (revision, bytes, digest, identity) in [
        (
            "changed-revision",
            BODY.len() as u64,
            Some(hash(BODY)),
            IDENTITY,
        ),
        (
            "immutable-v1",
            BODY.len() as u64 + 1,
            Some(hash(BODY)),
            IDENTITY,
        ),
        ("immutable-v1", BODY.len() as u64, None, IDENTITY),
        (
            "immutable-v1",
            BODY.len() as u64,
            Some(hash(BODY)),
            "changed-object",
        ),
    ] {
        let mut transaction = begin_variant(
            &store,
            &admission,
            Arc::new(AtomicBool::new(false)),
            revision,
            bytes,
            digest,
        )
        .unwrap();
        assert_eq!(
            transaction
                .partial_object("payload.bin", identity)
                .unwrap()
                .resume_bytes(),
            0
        );
    }
    let record = checkpoints(&fixture.root).pop().unwrap();
    fs::write(record.parent().unwrap().join("object-0"), b"tampered").unwrap();
    let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    assert!(matches!(
        transaction.partial_object("payload.bin", IDENTITY),
        Err(Error::Malformed(_))
    ));
}

#[test]
fn no_validator_is_ephemeral_and_changed_validator_never_appends() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    {
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
        partial.write_chunk(&BODY[..8]).unwrap();
        partial.checkpoint().unwrap();
        assert_eq!(partial.bytes(), 8);
    }
    assert!(checkpoints(&fixture.root).is_empty());
    retained_prefix(&store, &admission);
    let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
    assert_eq!(partial.resume_bytes(), 8);
    assert!(partial.set_validator("\"changed\"").is_err());
    for bad in ["W/\"weak\"", "unquoted", "\"a\"b\"", "\"line\nbreak\""] {
        assert!(partial.set_validator(bad).is_err());
    }
    assert!(
        partial
            .set_validator(&format!("\"{}\"", "x".repeat(1024)))
            .is_err()
    );
    assert!(partial.write_chunk(&vec![0; BODY.len()]).is_err());
    assert_eq!(partial.bytes(), 8);
}

#[test]
fn cancellation_can_checkpoint_written_bytes_and_waiter_cannot_steal_ownership() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut transaction = begin(&store, &admission, cancel.clone());
    let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
    partial.set_validator("\"strong\"").unwrap();
    partial.write_chunk(&BODY[..8]).unwrap();
    let mut waiter = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    assert!(matches!(
        waiter.partial_object("payload.bin", IDENTITY),
        Err(Error::Limit(_))
    ));
    drop(waiter);
    cancel.store(true, Ordering::Release);
    assert!(partial.write_chunk(&BODY[8..]).is_err());
    partial.checkpoint().unwrap();
    drop(partial);
    drop(transaction);
    no_host_lease(&admission);
    let mut recovered = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    assert_eq!(
        recovered
            .partial_object("payload.bin", IDENTITY)
            .unwrap()
            .resume_bytes(),
        8
    );
}

#[test]
fn full_digest_length_and_admission_remain_authoritative() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let old = old_ref(&fixture, &store, &admission);
    for bytes in [&BODY[..8], &vec![b'x'; BODY.len()][..]] {
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
        partial.write_chunk(bytes).unwrap();
        assert!(partial.finish().is_err());
    }
    assert_eq!(
        fs::read(fixture.root.join(format!("refs/{}.json", hash(b"stable")))).unwrap(),
        old
    );
    let denied = self::admission(1);
    assert!(
        begin_variant(
            &store,
            &denied,
            Arc::new(AtomicBool::new(false)),
            "v1",
            BODY.len() as u64,
            None
        )
        .is_err()
    );
    no_host_lease(&denied);
    retained_prefix(&store, &admission);
    // A retained prefix does not credit a newly copied payload reservation.
    let before = store.usage().unwrap();
    assert!(
        begin_variant(
            &store,
            &admission,
            Arc::new(AtomicBool::new(false)),
            "v1",
            LIMIT,
            None
        )
        .is_err()
    );
    assert_eq!(store.usage().unwrap(), before);
    no_host_lease(&admission);
}

#[test]
fn stable_partial_lock_rebinding_refuses_further_writes() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
    let lock = fs::read_dir(fixture.root.join("locks"))
        .unwrap()
        .find_map(|entry| {
            let entry = entry.unwrap();
            entry
                .file_name()
                .to_str()
                .unwrap()
                .starts_with("partial-")
                .then(|| entry.path())
        })
        .unwrap();
    let original = fs::metadata(&lock).unwrap().ino();
    fs::rename(&lock, fixture.root.join("locks/retired-partial")).unwrap();
    fs::write(&lock, b"").unwrap();
    fs::set_permissions(&lock, fs::Permissions::from_mode(0o600)).unwrap();
    assert_ne!(original, fs::metadata(&lock).unwrap().ino());
    assert!(partial.write_chunk(b"a").is_err());
}

#[test]
fn exact_total_checkpoint_does_not_create_a_terminal_proof() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    {
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
        partial.set_validator("\"strong\"").unwrap();
        partial.write_chunk(BODY).unwrap();
        partial.checkpoint().unwrap();
    }
    assert!(checkpoints(&fixture.root).is_empty());
    retained_prefix(&store, &admission);
    {
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
        partial.write_chunk(&BODY[8..]).unwrap();
        partial.checkpoint().unwrap();
    }
    let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    assert_eq!(
        transaction
            .partial_object("payload.bin", IDENTITY)
            .unwrap()
            .resume_bytes(),
        8
    );
}

#[test]
fn retained_old_prefix_plus_new_full_copy_must_fit_disk_peak() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let total = 6 * 1024 * 1024;
    {
        let mut transaction = begin_variant(
            &store,
            &admission,
            Arc::new(AtomicBool::new(false)),
            "large-v1",
            total,
            None,
        )
        .unwrap();
        let mut partial = transaction
            .partial_object("payload.bin", "large-object")
            .unwrap();
        partial.set_validator("\"strong\"").unwrap();
        let bytes = [b'a'; 64 * 1024];
        for _ in 0..48 {
            partial.write_chunk(&bytes).unwrap();
        }
        partial.checkpoint().unwrap();
    }
    no_host_lease(&admission);
    let retained = store.usage().unwrap();
    assert!(retained.used_bytes > 3 * 1024 * 1024);
    assert_eq!(retained.reserved_bytes, 0);
    assert!(
        begin_variant(
            &store,
            &admission,
            Arc::new(AtomicBool::new(false)),
            "large-v1",
            total,
            None,
        )
        .is_err()
    );
    assert_eq!(store.usage().unwrap(), retained);
    no_host_lease(&admission);
}

#[test]
fn unchanged_resumed_prefix_is_not_retained_again_on_refused_request() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    retained_prefix(&store, &admission);
    let original = checkpoints(&fixture.root);
    assert_eq!(original.len(), 1);
    for _ in 0..3 {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut transaction = begin(&store, &admission, cancel.clone());
        let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
        assert_eq!(partial.resume_bytes(), 8);
        cancel.store(true, Ordering::Release);
        partial.checkpoint().unwrap();
        drop(partial);
        drop(transaction);
        assert_eq!(checkpoints(&fixture.root), original);
        assert_eq!(
            fs::read_dir(fixture.root.join("staging")).unwrap().count(),
            1
        );
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        no_host_lease(&admission);
    }
}

#[test]
fn repeated_trusted_digest_paths_require_dedup_before_second_private_copy() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let body = vec![b'a'; 4 * 1024 * 1024];
    let digest = hash(&body);
    let mut transaction = store
        .begin_discovered(
            source("same-digest"),
            ["a.bin", "b.bin"]
                .into_iter()
                .map(|path| {
                    DiscoveredEntry::new(
                        path,
                        body.len() as u64,
                        FileRole::Auxiliary,
                        Some(digest.clone()),
                    )
                    .unwrap()
                })
                .collect(),
            &admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    let mut first = transaction.partial_object("a.bin", "a").unwrap();
    first.write_chunk(&body).unwrap();
    first.finish().unwrap();
    assert!(matches!(
        transaction.partial_object("b.bin", "b"),
        Err(Error::State(_))
    ));
    assert!(
        !fs::read_dir(fixture.root.join("staging"))
            .unwrap()
            .any(|attempt| attempt.unwrap().path().join("object-1").exists())
    );
    assert!(transaction.try_cached("b.bin").unwrap().is_some());
    let usage = store.usage().unwrap();
    assert!(usage.used_bytes + usage.reserved_bytes <= usage.limit_bytes);
    assert!(matches!(
        transaction.commit(None).unwrap(),
        CommitOutcome::Durable(_)
    ));
    no_host_lease(&admission);
}

#[test]
fn trusted_partial_locks_competing_publication_and_refuses_existing_blob_credit() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
    let manifest = Manifest::new(
        source("competing"),
        vec![
            ManifestEntry::new(
                "payload.bin",
                hash(BODY),
                BODY.len() as u64,
                FileRole::Auxiliary,
            )
            .unwrap(),
        ],
        &Default::default(),
    )
    .unwrap();
    let mut competitor = store
        .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    struct NoBody;
    impl Read for NoBody {
        fn read(&mut self, _output: &mut [u8]) -> std::io::Result<usize> {
            panic!("competing trusted publication must not read a second body");
        }
    }
    assert!(matches!(
        competitor.import_object("payload.bin", &mut NoBody),
        Err(Error::Limit(_))
    ));
    partial.write_chunk(BODY).unwrap();
    partial.finish().unwrap();
    competitor
        .import_object("payload.bin", &mut NoBody)
        .unwrap();
    competitor.commit(None).unwrap();
    transaction.commit(None).unwrap();
    let mut cached = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    assert!(matches!(
        cached.partial_object("payload.bin", IDENTITY),
        Err(Error::State(_))
    ));
    assert!(cached.try_cached("payload.bin").unwrap().is_some());
    cached.commit(None).unwrap();
    let usage = store.usage().unwrap();
    assert!(usage.used_bytes + usage.reserved_bytes <= usage.limit_bytes);
    no_host_lease(&admission);
}

#[cfg(feature = "diagnostic-cache")]
mod fault {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    use std::process::{Command, Output, Stdio};
    use std::thread;
    use std::time::Instant;
    use zerogravity_artifacts::cache::{CacheHook, CacheMoment, CacheOperation, CachePoint};

    const ROOT: &str = "ZEROGRAVITY_PARTIAL_CHILD_ROOT";
    const POINT: &str = "ZEROGRAVITY_PARTIAL_CHILD_POINT";

    fn points() -> Vec<CachePoint> {
        [
            CacheOperation::PartialFileSync,
            CacheOperation::PartialRecordSync,
            CacheOperation::PartialRecordInstall,
            CacheOperation::PartialDirectorySync,
        ]
        .into_iter()
        .flat_map(|operation| {
            [CacheMoment::Before, CacheMoment::After].map(move |moment| CachePoint {
                operation,
                moment,
                object: Some(0),
            })
        })
        .collect()
    }

    struct Hook {
        point: CachePoint,
        armed: Arc<AtomicBool>,
        exit: bool,
    }
    impl CacheHook for Hook {
        fn point(&self, point: CachePoint) -> Result<()> {
            if self.armed.load(Ordering::Acquire) && point == self.point {
                if self.exit {
                    std::process::exit(74);
                }
                return Err(Error::Io(std::io::Error::other(
                    "synthetic partial checkpoint fault",
                )));
            }
            Ok(())
        }
    }
    fn interrupted(
        store: CacheStore,
        admission: &TransferAdmission,
        point: CachePoint,
        exit: bool,
    ) {
        let armed = Arc::new(AtomicBool::new(false));
        let store = store.with_hook(Arc::new(Hook {
            point,
            armed: armed.clone(),
            exit,
        }));
        let mut transaction = begin(&store, admission, Arc::new(AtomicBool::new(false)));
        let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
        let start = partial.resume_bytes() as usize;
        partial.set_validator("\"strong\"").unwrap();
        partial.write_chunk(&BODY[start..12]).unwrap();
        armed.store(true, Ordering::Release);
        assert!(partial.checkpoint().is_err());
    }
    fn child(root: &Path, point: usize) -> Output {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "fault::partial_child_worker",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(ROOT, root)
            .env(POINT, point.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                panic!("owned partial child exceeded deadline: {output:?}");
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
    #[test]
    #[ignore = "owned subprocess fixture; invoked by durable checkpoint matrix"]
    fn partial_child_worker() {
        let root = PathBuf::from(std::env::var_os(ROOT).unwrap());
        let selected = std::env::var(POINT).unwrap();
        if selected == "hold" {
            let store = open(&root);
            let admission = admission(128 * 1024 * 1024);
            let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
            let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
            partial.set_validator("\"strong\"").unwrap();
            partial.write_chunk(&BODY[..8]).unwrap();
            partial.checkpoint().unwrap();
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(root.join("holder-ready"))
                .unwrap()
                .write_all(b"ready")
                .unwrap();
            std::io::stdin().read_exact(&mut [0; 1]).unwrap();
            return;
        }
        let point = selected.parse::<usize>().unwrap();
        interrupted(
            open(&root),
            &admission(128 * 1024 * 1024),
            points()[point],
            true,
        );
        panic!("selected exit boundary was not reached");
    }

    struct HeldChild(Option<std::process::Child>);
    impl Drop for HeldChild {
        fn drop(&mut self) {
            if let Some(child) = &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    #[test]
    fn live_child_partial_owner_blocks_other_process_and_recovers_after_exit() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let mut child = HeldChild(Some(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "fault::partial_child_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(ROOT, &fixture.root)
                .env(POINT, "hold")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        ));
        let deadline = Instant::now() + Duration::from_secs(20);
        while !fixture.root.join("holder-ready").exists() {
            assert!(
                Instant::now() < deadline,
                "partial child did not establish ownership"
            );
            assert!(child.0.as_mut().unwrap().try_wait().unwrap().is_none());
            thread::sleep(Duration::from_millis(5));
        }
        let lock = fs::read_dir(fixture.root.join("locks"))
            .unwrap()
            .find_map(|entry| {
                let entry = entry.unwrap();
                entry
                    .file_name()
                    .to_str()
                    .unwrap()
                    .starts_with("partial-")
                    .then(|| entry.path())
            })
            .unwrap();
        let inode = fs::metadata(&lock).unwrap().ino();
        let mut waiter = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        assert!(matches!(
            waiter.partial_object("payload.bin", IDENTITY),
            Err(Error::Limit(_))
        ));
        drop(waiter);
        child
            .0
            .as_mut()
            .unwrap()
            .stdin
            .as_mut()
            .unwrap()
            .write_all(b"x")
            .unwrap();
        loop {
            if child.0.as_mut().unwrap().try_wait().unwrap().is_some() {
                break;
            }
            assert!(Instant::now() < deadline, "partial child failed to exit");
            thread::sleep(Duration::from_millis(5));
        }
        let output = child.0.take().unwrap().wait_with_output().unwrap();
        assert!(output.status.success(), "{output:?}");
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        assert_eq!(
            transaction
                .partial_object("payload.bin", IDENTITY)
                .unwrap()
                .resume_bytes(),
            8
        );
        assert_eq!(fs::metadata(lock).unwrap().ino(), inode);
    }

    #[test]
    fn checkpoint_faults_and_real_process_exits_recover_only_committed_prefixes() {
        for had_prefix in [false, true] {
            for (ordinal, point) in points().into_iter().enumerate() {
                for process_exit in [false, true] {
                    let fixture = Fixture::new();
                    let store = fixture.open();
                    let admission = admission(128 * 1024 * 1024);
                    let original = old_ref(&fixture, &store, &admission);
                    if had_prefix {
                        retained_prefix(&store, &admission);
                    }
                    if process_exit {
                        let output = child(&fixture.root, ordinal);
                        assert_eq!(output.status.code(), Some(74), "{point:?}: {output:?}");
                    } else {
                        interrupted(store.clone(), &admission, point, false);
                    }
                    let recovered = fixture.open();
                    assert_eq!(recovered.usage().unwrap().reserved_bytes, 0);
                    assert_eq!(
                        fs::read(fixture.root.join(format!("refs/{}.json", hash(b"stable"))))
                            .unwrap(),
                        original
                    );
                    let mut transaction =
                        begin(&recovered, &admission, Arc::new(AtomicBool::new(false)));
                    let mut partial = transaction.partial_object("payload.bin", IDENTITY).unwrap();
                    let installed = point.operation == CacheOperation::PartialDirectorySync
                        || (point.operation == CacheOperation::PartialRecordInstall
                            && point.moment == CacheMoment::After);
                    let expected = if installed {
                        12
                    } else if had_prefix {
                        8
                    } else {
                        0
                    };
                    assert_eq!(
                        partial.resume_bytes(),
                        expected,
                        "{point:?}, exit={process_exit}, previous={had_prefix}"
                    );
                    partial.write_chunk(&BODY[expected as usize..]).unwrap();
                    partial.finish().unwrap();
                    assert!(matches!(
                        transaction.commit(Some("stable")).unwrap(),
                        CommitOutcome::Durable(_)
                    ));
                    assert_eq!(recovered.usage().unwrap().reserved_bytes, 0);
                    no_host_lease(&admission);
                }
            }
        }
    }
}
