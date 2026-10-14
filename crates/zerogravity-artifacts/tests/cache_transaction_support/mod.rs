use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::{DirBuilderExt, FileExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use zerogravity_artifacts::cache::{
    CacheOptions, CacheStore, CommitOutcome, FileRole, Manifest, ManifestEntry, ManifestSource,
    SnapshotLease, TransferAdmission,
};
use zerogravity_artifacts::{Error, Result};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

pub const DISK_LIMIT: u64 = 8 * 1024 * 1024;
pub const PAYLOAD_PATH: &str = "payload.bin";
pub const OLD: &[u8] = b"the previous complete artifact";
pub const NEW: &[u8] = b"the proposed replacement artifact";
pub const THIRD: &[u8] = b"another publisher's complete artifact";
static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

pub struct Fixture {
    pub base: PathBuf,
    pub root: PathBuf,
}

impl Fixture {
    pub fn new() -> Self {
        let serial = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!(
            "zerogravity-cache-test-{}-{serial}",
            std::process::id()
        ));
        fs::DirBuilder::new().mode(0o700).create(&base).unwrap();
        Self {
            root: base.join("cache"),
            base,
        }
    }

    pub fn open(&self) -> CacheStore {
        CacheStore::open(options(&self.root)).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // All paths are inside this uniquely created private fixture. Unix
        // remove_dir_all does not follow a substituted final symlink.
        let _ = fs::remove_dir_all(&self.base);
    }
}

pub fn options(root: &Path) -> CacheOptions {
    CacheOptions::new(Some(root.to_owned()), DISK_LIMIT)
}

pub fn admission(limit: u64) -> TransferAdmission {
    // Isolated fixture admission, not an attestation of physical host RAM.
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("test-host-pool", "cache-transactions").unwrap();
    arbiter.register(resource.clone(), limit).unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("tiny-local-fixture", "cache-import").unwrap(),
    }
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn manifest(entries: &[(&str, &[u8], FileRole)]) -> Manifest {
    let entries = entries
        .iter()
        .map(|(path, bytes, role)| {
            ManifestEntry::new(*path, digest(bytes), bytes.len() as u64, *role).unwrap()
        })
        .collect();
    Manifest::new(
        ManifestSource::new("local", "synthetic-fixture", "immutable-test-revision").unwrap(),
        entries,
        &Default::default(),
    )
    .unwrap()
}

pub fn payload_manifest(bytes: &[u8]) -> Manifest {
    manifest(&[(PAYLOAD_PATH, bytes, FileRole::Auxiliary)])
}

pub fn quota_manifest() -> Manifest {
    // This plan is only reserved, never imported or presented as verified data.
    Manifest::new(
        ManifestSource::new("local", "quota-fixture", "immutable-revision").unwrap(),
        vec![
            ManifestEntry::new(
                "large.bin",
                digest(b"not-read"),
                6 * 1024 * 1024,
                FileRole::Auxiliary,
            )
            .unwrap(),
        ],
        &Default::default(),
    )
    .unwrap()
}

pub fn publish(
    store: &CacheStore,
    admission: &TransferAdmission,
    bytes: &[u8],
    reference: Option<&str>,
) -> Result<CommitOutcome> {
    let mut transaction = store.begin(
        payload_manifest(bytes),
        admission,
        Arc::new(AtomicBool::new(false)),
    )?;
    let mut reader = bytes;
    transaction.import_object(PAYLOAD_PATH, &mut reader)?;
    transaction.commit(reference)
}

pub fn durable(outcome: CommitOutcome) -> String {
    match outcome {
        CommitOutcome::Durable(snapshot) => snapshot.digest().to_owned(),
        CommitOutcome::DurabilityUnknown { cause, .. } => {
            panic!("unexpected uncertain commit: {cause}")
        }
    }
}

pub fn file_bytes(file: &File) -> Vec<u8> {
    let length = usize::try_from(file.metadata().unwrap().len()).unwrap();
    let mut bytes = vec![0; length];
    let mut done = 0;
    while done < bytes.len() {
        let count = file.read_at(&mut bytes[done..], done as u64).unwrap();
        assert!(count != 0, "file became truncated while checking the lease");
        done += count;
    }
    bytes
}

pub fn assert_payload(lease: &SnapshotLease, expected: &[u8]) {
    assert_eq!(file_bytes(lease.file(PAYLOAD_PATH).unwrap()), expected);
}

pub fn assert_no_host_reservation(admission: &TransferAdmission) {
    let usage = admission.arbiter.snapshot(&admission.resource).unwrap();
    assert_eq!(usage.reserved, 0);
    assert_eq!(usage.reservations, 0);
    assert!(!usage.fenced, "synchronous file work must not fence a GPU");
}

pub fn assert_not_found<T>(result: Result<T>) {
    assert!(matches!(result, Err(Error::Io(error)) if error.kind() == io::ErrorKind::NotFound));
}

#[cfg(feature = "diagnostic-cache")]
pub mod process {
    use super::*;
    use std::io::{BufRead, Read, Write};
    use std::process::{Child, Command, Output, Stdio};
    use std::sync::mpsc::{self, Receiver};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::cache::{CacheMoment, CacheOperation, CachePoint};

    pub const CHILD_MODE: &str = "ZEROGRAVITY_CACHE_TEST_CHILD_MODE";
    pub const CHILD_ROOT: &str = "ZEROGRAVITY_CACHE_TEST_CHILD_ROOT";
    pub const CHILD_POINT: &str = "ZEROGRAVITY_CACHE_TEST_CHILD_POINT";
    pub const CHILD_REFERENCE: &str = "ZEROGRAVITY_CACHE_TEST_CHILD_REFERENCE";
    const TIMEOUT: Duration = Duration::from_secs(20);
    const MAX_OUTPUT: u64 = 512 * 1024;

    pub fn points() -> Vec<CachePoint> {
        use CacheMoment::{After, Before};
        use CacheOperation::*;
        [
            ReservationFileSync,
            ReservationDirectorySync,
            StagingDirectorySync,
            ObjectFileSync,
            ObjectInstall,
            ObjectDirectorySync,
            ManifestFileSync,
            ManifestInstall,
            ManifestDirectorySync,
            SnapshotEntryInstall,
            SnapshotDirectorySync,
            SnapshotInstall,
            SnapshotParentSync,
            RefFileSync,
            ReferenceRecordInstall,
            ReferenceRecordDirectorySync,
            RefRename,
            RefDirectorySync,
        ]
        .into_iter()
        .flat_map(|operation| {
            [Before, After].map(|moment| CachePoint {
                operation,
                moment,
                object: matches!(
                    operation,
                    ObjectFileSync | ObjectInstall | ObjectDirectorySync | SnapshotEntryInstall
                )
                .then_some(0),
            })
        })
        .collect()
    }

    pub fn reference_changed(point: CachePoint) -> bool {
        point.operation == CacheOperation::RefDirectorySync
            || (point.operation == CacheOperation::RefRename && point.moment == CacheMoment::After)
    }

    pub fn before_body(point: CachePoint) -> bool {
        matches!(
            point.operation,
            CacheOperation::ReservationFileSync
                | CacheOperation::ReservationDirectorySync
                | CacheOperation::StagingDirectorySync
        )
    }

    pub struct RunningChild {
        child: Child,
        lines: Receiver<String>,
        stdout: Option<JoinHandle<Vec<u8>>>,
        stderr: Option<JoinHandle<Vec<u8>>>,
        deadline: Instant,
        finished: bool,
    }

    impl RunningChild {
        pub fn spawn(root: &Path, mode: &str, point: usize, reference: &str) -> Self {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "cache_child_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_MODE, mode)
                .env(CHILD_ROOT, root)
                .env(CHILD_POINT, point.to_string())
                .env(CHILD_REFERENCE, reference)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let stdout = child.stdout.take().unwrap();
            let stderr = child.stderr.take().unwrap();
            let (sender, lines) = mpsc::channel();
            let stdout = thread::spawn(move || {
                let mut input = io::BufReader::new(stdout.take(MAX_OUTPUT));
                let mut all = Vec::new();
                loop {
                    let mut line = Vec::new();
                    if input.read_until(b'\n', &mut line).unwrap() == 0 {
                        break;
                    }
                    all.extend_from_slice(&line);
                    let _ = sender.send(String::from_utf8_lossy(&line).into_owned());
                }
                all
            });
            let stderr = thread::spawn(move || {
                let mut all = Vec::new();
                stderr.take(MAX_OUTPUT).read_to_end(&mut all).unwrap();
                all
            });
            Self {
                child,
                lines,
                stdout: Some(stdout),
                stderr: Some(stderr),
                deadline: Instant::now() + TIMEOUT,
                finished: false,
            }
        }

        pub fn event(&self, expected: &str) {
            loop {
                let timeout = self.deadline.saturating_duration_since(Instant::now());
                let line = self
                    .lines
                    .recv_timeout(timeout)
                    .unwrap_or_else(|error| panic!("child did not reach {expected}: {error}"));
                // libtest can prefix its test name without a newline before
                // a nocapture marker emitted by this owned helper.
                if line.trim_end().ends_with(expected) {
                    return;
                }
            }
        }

        pub fn send(&mut self, command: &str) {
            let input = self.child.stdin.as_mut().unwrap();
            writeln!(input, "{command}").unwrap();
            input.flush().unwrap();
        }

        pub fn finish(mut self) -> Output {
            let status = loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    break status;
                }
                assert!(
                    Instant::now() < self.deadline,
                    "owned child exceeded its deadline"
                );
                thread::sleep(Duration::from_millis(5));
            };
            self.finished = true;
            Output {
                status,
                stdout: self.stdout.take().unwrap().join().unwrap(),
                stderr: self.stderr.take().unwrap().join().unwrap(),
            }
        }
    }

    impl Drop for RunningChild {
        fn drop(&mut self) {
            if !self.finished {
                // Only this test's owned subprocess. It has no GPU handles or
                // descendants; forced cleanup establishes no device result.
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
            for reader in [&mut self.stdout, &mut self.stderr] {
                if let Some(reader) = reader.take() {
                    let _ = reader.join();
                }
            }
        }
    }

    pub fn assert_success(output: Output) {
        assert!(
            output.status.success(),
            "child status {:?}\nstdout={}\nstderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
