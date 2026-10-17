//! Independent actual-descriptor and crash/restart removal gates. Tiny local data.
#![cfg(all(feature = "cache", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;

use cache_transaction_support::*;
use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::{FileRole, ReclaimMode, ReclaimOutcome};

// Command::spawn may briefly inherit every thread's managed descriptors until
// CLOEXEC runs in the child. Keep unrelated cases quiescent while another case
// forks, so an exact fault/retry assertion cannot observe a legitimate inherited
// activity pin. Deliberate descriptor/child overlap inside its own case remains.
#[cfg(feature = "diagnostic-cache")]
static PROCESS_FIXTURE_SCHEDULE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gguf() -> Vec<u8> {
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    string(&mut bytes, "value");
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(2u64.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    while !bytes.len().is_multiple_of(32) {
        bytes.push(0);
    }
    bytes.extend(1f32.to_le_bytes());
    bytes.extend(2f32.to_le_bytes());
    bytes
}

#[test]
fn managed_lazy_artifact_clone_blocks_reclamation_until_actual_descriptor_drop() {
    #[cfg(feature = "diagnostic-cache")]
    let _spawn_schedule = PROCESS_FIXTURE_SCHEDULE.lock().unwrap();
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let bytes = gguf();
    let manifest = manifest(&[("model.gguf", &bytes, FileRole::Weights)]);
    let target = manifest.digest().unwrap();
    let mut transaction = store
        .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    transaction
        .import_object("model.gguf", &mut Cursor::new(&bytes))
        .unwrap();
    durable(transaction.commit(None).unwrap());
    let lease = store.open_snapshot(&target).unwrap();
    let mut lazy = Artifact::from_file(
        lease.file("model.gguf").unwrap().try_clone().unwrap(),
        Path::new("model.gguf"),
    )
    .unwrap();
    drop(lease);
    assert!(matches!(
        store
            .reclaim_snapshot(&target, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Busy
    ));
    assert_eq!(lazy.read_reference_f32("value").unwrap(), [1., 2.]);
    drop(lazy);
    assert!(matches!(
        store
            .reclaim_snapshot(&target, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Durable(_)
    ));
    assert!(store.open_snapshot(&target).is_err());
    assert_no_host_reservation(&admission);
}

#[cfg(feature = "diagnostic-cache")]
mod diagnostics {
    use super::*;
    use cache_transaction_support::process::*;
    use std::collections::BTreeMap;
    use std::fs::{self, File};
    use std::io::{self, BufRead, Read};
    use std::os::unix::fs::MetadataExt;
    use std::process::{Child, Command, Stdio};
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::cache::{
        CacheHook, CacheMoment, CacheOperation, CachePoint, CacheStore,
    };
    use zerogravity_artifacts::{Error, Result};

    const RELEASE: &str = "ZEROGRAVITY_RECLAIM_TEST_RELEASE";
    fn removal(point: CachePoint) -> bool {
        matches!(
            point.operation,
            CacheOperation::RetirementRename
                | CacheOperation::ReclaimUnlink
                | CacheOperation::ReclaimDirectorySync
        )
    }
    #[derive(Default)]
    struct Trace(Mutex<Vec<CachePoint>>);
    impl CacheHook for Trace {
        fn point(&self, point: CachePoint) -> Result<()> {
            if removal(point) {
                self.0.lock().unwrap().push(point);
            }
            Ok(())
        }
    }
    struct FailAt {
        index: usize,
        count: AtomicUsize,
    }
    impl CacheHook for FailAt {
        fn point(&self, point: CachePoint) -> Result<()> {
            if removal(point) && self.count.fetch_add(1, Ordering::SeqCst) == self.index {
                return Err(Error::Io(io::Error::other(
                    "injected removal operation failure",
                )));
            }
            Ok(())
        }
    }

    // Independent std::fs walk of the published cooperative policy. It does
    // not call the production capability walker, usage helper or charge code.
    fn observed_charge(root: &Path) -> u64 {
        let policy: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("policy.json")).unwrap()).unwrap();
        let block = policy["block_allowance"].as_u64().unwrap();
        let inode = policy["inode_allowance"].as_u64().unwrap();
        assert_eq!(block, 4096);
        let mut pending = vec![root.to_path_buf()];
        let mut physical = BTreeMap::<(u64, u64), u64>::new();
        while let Some(path) = pending.pop() {
            let metadata = fs::symlink_metadata(&path).unwrap();
            assert!(!metadata.file_type().is_symlink());
            let logical = metadata.len().div_ceil(block) * block;
            let observed = metadata.blocks() * 512;
            let charge = logical.max(observed) + inode;
            physical
                .entry((metadata.dev(), metadata.ino()))
                .and_modify(|old| *old = (*old).max(charge))
                .or_insert(charge);
            if metadata.is_dir() {
                pending.extend(
                    fs::read_dir(path)
                        .unwrap()
                        .map(|entry| entry.unwrap().path()),
                );
            }
        }
        physical.into_values().sum()
    }

    fn prepare(fixture: &Fixture) -> (CacheStore, String, String) {
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let target = durable(publish(&store, &admission, OLD, None).unwrap());
        let survivor = durable(publish(&store, &admission, NEW, Some("survivor")).unwrap());
        (store, target, survivor)
    }
    fn trace() -> Vec<CachePoint> {
        let fixture = Fixture::new();
        let (store, target, _) = prepare(&fixture);
        let trace = Arc::new(Trace::default());
        let outcome = store
            .with_hook(trace.clone())
            .reclaim_snapshot(&target, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap();
        assert!(
            matches!(&outcome, ReclaimOutcome::Durable(_)),
            "trace outcome: {outcome:?}"
        );
        let points = trace.0.lock().unwrap().clone();
        assert!(!points.is_empty());
        assert_eq!(points.len() % 2, 0);
        for pair in points.chunks_exact(2) {
            assert_eq!(pair[0].moment, CacheMoment::Before);
            assert_eq!(pair[1].moment, CacheMoment::After);
            assert_eq!(pair[0].operation, pair[1].operation);
            assert_eq!(pair[0].object, pair[1].object);
            assert!(pair[0].object.is_some());
        }
        for operation in [
            CacheOperation::RetirementRename,
            CacheOperation::ReclaimUnlink,
            CacheOperation::ReclaimDirectorySync,
        ] {
            assert!(points.iter().any(|point| point.operation == operation));
        }
        points
    }
    fn verify_restart(fixture: &Fixture, target: &str, survivor: &str, context: &str) {
        let reopened = CacheStore::open_existing(options(&fixture.root)).unwrap();
        assert_payload(&reopened.resolve_ref("survivor").unwrap(), NEW);
        assert_payload(&reopened.open_snapshot(survivor).unwrap(), NEW);
        let before = reopened.usage().unwrap();
        assert_eq!(before.reserved_bytes, 0);
        assert_eq!(before.used_bytes, observed_charge(&fixture.root));
        let outcome = reopened
            .reclaim_snapshot(target, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap();
        let report = match outcome {
            ReclaimOutcome::Durable(report) => report,
            other => panic!("{context}: restart could not finish exact target: {other:?}"),
        };
        assert_eq!(report.used_before_bytes(), before.used_bytes);
        assert_eq!(
            report.used_after_bytes(),
            Some(observed_charge(&fixture.root))
        );
        assert_eq!(
            reopened.usage().unwrap().used_bytes,
            observed_charge(&fixture.root)
        );
        assert!(reopened.open_snapshot(target).is_err());
        assert_payload(&reopened.resolve_ref("survivor").unwrap(), NEW);
        let again = reopened
            .reclaim_snapshot(target, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap();
        assert!(
            matches!(&again, ReclaimOutcome::Durable(report) if !report.retired() && report.removed_files() == 0 && report.removed_directories() == 0),
            "{context}: absent-target retry returned {again:?}"
        );
    }

    #[test]
    fn every_observed_removal_io_boundary_preserves_survivor_and_restarts() {
        let _spawn_schedule = PROCESS_FIXTURE_SCHEDULE.lock().unwrap();
        let points = trace();
        for (index, point) in points.iter().enumerate() {
            let fixture = Fixture::new();
            let (store, target, survivor) = prepare(&fixture);
            let hook = Arc::new(FailAt {
                index,
                count: AtomicUsize::new(0),
            });
            let result = store.with_hook(hook.clone()).reclaim_snapshot(
                &target,
                ReclaimMode::Apply,
                &AtomicBool::new(false),
            );
            assert!(
                hook.count.load(Ordering::SeqCst) > index,
                "unreached {point:?}"
            );
            assert!(
                matches!(result, Err(_) | Ok(ReclaimOutcome::Incomplete { .. })),
                "fault incorrectly succeeded at {point:?}: {result:?}"
            );
            verify_restart(
                &fixture,
                &target,
                &survivor,
                &format!("I/O callback {index} {point:?}"),
            );
        }
        eprintln!(
            "RECLAIM_IO_BOUNDARIES_PASS callbacks={} survivor=exact restarted=1 independent_charge=exact",
            points.len()
        );
    }

    #[test]
    fn every_observed_removal_process_exit_boundary_keeps_charge_and_restarts() {
        let _spawn_schedule = PROCESS_FIXTURE_SCHEDULE.lock().unwrap();
        let points = trace();
        for (index, point) in points.iter().enumerate() {
            let fixture = Fixture::new();
            let (_, target, survivor) = prepare(&fixture);
            let output =
                RunningChild::spawn(&fixture.root, "crash-reclaim", index, &target).finish();
            assert_eq!(
                output.status.code(),
                Some(86),
                "child did not exit at {point:?}: {}",
                String::from_utf8_lossy(&output.stdout)
            );
            verify_restart(
                &fixture,
                &target,
                &survivor,
                &format!("exit callback {index} {point:?}"),
            );
        }
        eprintln!(
            "RECLAIM_PROCESS_EXIT_BOUNDARIES_PASS callbacks={} actual_exit=86 survivor=exact restarted=1 independent_charge=exact",
            points.len()
        );
    }

    struct InheritedChild {
        child: Child,
        ready: mpsc::Receiver<()>,
        output: Option<JoinHandle<Vec<u8>>>,
        deadline: Instant,
        finished: bool,
    }
    impl InheritedChild {
        fn spawn(root: &Path, release: &Path, inherited: File) -> Self {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "cache_child_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_MODE, "inherited-reader")
                .env(CHILD_ROOT, root)
                .env(RELEASE, release)
                .stdin(Stdio::from(inherited))
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap();
            let output = child.stdout.take().unwrap();
            let (sender, ready) = mpsc::channel();
            let output = thread::spawn(move || {
                let mut reader = io::BufReader::new(output.take(128 * 1024));
                let mut all = Vec::new();
                loop {
                    let mut line = Vec::new();
                    if reader.read_until(b'\n', &mut line).unwrap() == 0 {
                        break;
                    }
                    if line.ends_with(b"INHERITED_READY\n") {
                        let _ = sender.send(());
                    }
                    all.extend(line);
                }
                all
            });
            Self {
                child,
                ready,
                output: Some(output),
                deadline: Instant::now() + Duration::from_secs(20),
                finished: false,
            }
        }
        fn ready(&self) {
            self.ready
                .recv_timeout(self.deadline.saturating_duration_since(Instant::now()))
                .expect("inherited child did not become ready");
        }
        fn finish(mut self) {
            let status = loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    break status;
                }
                assert!(Instant::now() < self.deadline, "inherited child deadline");
                thread::sleep(Duration::from_millis(5));
            };
            self.finished = true;
            let output = self.output.take().unwrap().join().unwrap();
            assert!(
                status.success(),
                "inherited child failed: {}",
                String::from_utf8_lossy(&output)
            );
        }
    }
    impl Drop for InheritedChild {
        fn drop(&mut self) {
            if !self.finished {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
            if let Some(output) = self.output.take() {
                let _ = output.join();
            }
        }
    }

    #[test]
    fn inherited_dup_outlives_lease_and_parent_descriptor_until_child_exit() {
        let _spawn_schedule = PROCESS_FIXTURE_SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let (store, target, survivor) = prepare(&fixture);
        let lease = store.open_snapshot(&target).unwrap();
        let inherited = lease.file(PAYLOAD_PATH).unwrap().try_clone().unwrap();
        let release = fixture.base.join("release-child");
        let child = InheritedChild::spawn(&fixture.root, &release, inherited);
        drop(lease);
        child.ready();
        assert!(matches!(
            store
                .reclaim_snapshot(&target, ReclaimMode::Apply, &AtomicBool::new(false))
                .unwrap(),
            ReclaimOutcome::Busy
        ));
        assert_payload(&store.open_snapshot(&target).unwrap(), OLD);
        fs::write(&release, b"release").unwrap();
        child.finish();
        verify_restart(
            &fixture,
            &target,
            &survivor,
            "after inherited descriptor close",
        );
    }

    pub fn worker() {
        match std::env::var(CHILD_MODE).unwrap().as_str() {
            "inherited-reader" => {
                let release = std::env::var_os(RELEASE).unwrap();
                let mut bytes = Vec::new();
                std::io::stdin().read_to_end(&mut bytes).unwrap();
                assert_eq!(bytes, OLD);
                println!("INHERITED_READY");
                let deadline = Instant::now() + Duration::from_secs(10);
                while !Path::new(&release).exists() {
                    assert!(Instant::now() < deadline, "release not received");
                    thread::sleep(Duration::from_millis(5));
                }
                // The inherited stdin descriptor remains open until process exit.
            }
            "crash-reclaim" => {
                struct ExitAt {
                    index: usize,
                    count: AtomicUsize,
                }
                impl CacheHook for ExitAt {
                    fn point(&self, point: CachePoint) -> Result<()> {
                        if removal(point) && self.count.fetch_add(1, Ordering::SeqCst) == self.index
                        {
                            std::process::exit(86);
                        }
                        Ok(())
                    }
                }
                let root = std::env::var_os(CHILD_ROOT).unwrap();
                let target = std::env::var(CHILD_REFERENCE).unwrap();
                let index = std::env::var(CHILD_POINT).unwrap().parse().unwrap();
                let store = CacheStore::open_existing(options(Path::new(&root)))
                    .unwrap()
                    .with_hook(Arc::new(ExitAt {
                        index,
                        count: AtomicUsize::new(0),
                    }));
                let _ =
                    store.reclaim_snapshot(&target, ReclaimMode::Apply, &AtomicBool::new(false));
                panic!("requested removal exit boundary was not reached");
            }
            _ => panic!("unknown isolated test helper mode"),
        }
    }
}

#[cfg(feature = "diagnostic-cache")]
#[test]
#[ignore = "owned subprocess fixture, explicitly selected by parent tests"]
fn cache_child_worker() {
    diagnostics::worker();
}
