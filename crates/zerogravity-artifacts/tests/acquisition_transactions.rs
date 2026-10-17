//! Tiny admitted scripted transfers, with the real cache publisher and actual
//! child exits. No model, external endpoint, GPU or alternate cache runtime.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)] // Reuse the already-qualified larger A1 fixture owner.
mod cache_transaction_support;
use cache_transaction_support::*;
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Cursor, Read};
use std::sync::{
    Arc, Barrier, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use zerogravity_artifacts::acquisition::{
    DigestProvenance, PullLimits, ResolvedFile, ResolvedPullPlan, execute,
};
use zerogravity_artifacts::cache::{
    CacheStore, DiscoveredEntry, FileRole, Manifest, ManifestSource, TransferAdmission,
};
use zerogravity_artifacts::{Error, Result};

type ResponseCase = (Option<u64>, u16, BTreeMap<String, String>, Vec<u8>);

struct Script {
    responses: Mutex<VecDeque<(Option<u64>, HttpResponse)>>,
    calls: AtomicUsize,
    body_bytes: Arc<AtomicUsize>,
    admission: TransferAdmission,
    store: CacheStore,
}
struct Counted {
    reader: Cursor<Vec<u8>>,
    bytes: Arc<AtomicUsize>,
}
impl Read for Counted {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let n = self.reader.read(output)?;
        self.bytes.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }
}
impl Script {
    fn new(
        store: &CacheStore,
        admission: &TransferAdmission,
        responses: Vec<ResponseCase>,
    ) -> Self {
        let body_bytes = Arc::new(AtomicUsize::new(0));
        let responses = responses
            .into_iter()
            .map(|(range, status, headers, body)| {
                (
                    range,
                    HttpResponse {
                        status,
                        headers,
                        body: Box::new(Counted {
                            reader: Cursor::new(body),
                            bytes: body_bytes.clone(),
                        }),
                    },
                )
            })
            .collect();
        Self {
            responses: Mutex::new(responses),
            calls: AtomicUsize::new(0),
            body_bytes,
            admission: admission.clone(),
            store: store.clone(),
        }
    }
}
impl HttpTransport for Script {
    fn get(&self, request: &HttpRequest, _cancel: &AtomicBool) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0,
            "host admission must precede request"
        );
        assert!(
            self.store.usage()?.reserved_bytes > 0,
            "disk peak admission must precede request"
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (range, response) = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP request");
        assert_eq!(request.range.as_ref().map(|range| range.start), range);
        Ok(response)
    }
}
fn source() -> ManifestSource {
    ManifestSource::new("github", "fixture/repo", "1".repeat(40)).unwrap()
}
fn file(path: &str, bytes: &[u8], trusted: bool) -> ResolvedFile {
    ResolvedFile::new(
        path,
        format!("https://example.invalid/{path}?secret=DO_NOT_PRINT"),
        FileRole::Auxiliary,
        bytes.len() as u64,
        trusted.then(|| digest(bytes)),
        format!("immutable/{path}"),
    )
    .unwrap()
}
fn plan(files: Vec<ResolvedFile>) -> ResolvedPullPlan {
    ResolvedPullPlan::new(source(), files, &PullLimits::default()).unwrap()
}
fn ok(body: &[u8]) -> (Option<u64>, u16, BTreeMap<String, String>, Vec<u8>) {
    (None, 200, BTreeMap::new(), body.to_vec())
}
fn discovered(bytes: &[u8]) -> DiscoveredEntry {
    DiscoveredEntry::new(PAYLOAD_PATH, bytes.len() as u64, FileRole::Auxiliary, None).unwrap()
}
fn publish_discovered(
    store: &CacheStore,
    admission: &TransferAdmission,
    mut bytes: &[u8],
    reference: &str,
) -> String {
    let mut tx = store
        .begin_discovered(
            source(),
            vec![discovered(bytes)],
            admission,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    tx.import_object(PAYLOAD_PATH, &mut bytes).unwrap();
    durable(tx.commit(Some(reference)).unwrap())
}

#[test]
fn admitted_single_pass_multifile_provenance_survives_offline_reopen() {
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let plan = plan(vec![
        file("a", b"first seen", false),
        file("b", b"provider checked", true),
    ]);
    assert!(!format!("{plan:?}").contains("DO_NOT_PRINT"));
    let script = Script::new(
        &store,
        &admission,
        vec![ok(b"first seen"), ok(b"provider checked")],
    );
    let receipt = execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
    )
    .unwrap();
    assert_eq!(
        script.body_bytes.load(Ordering::Relaxed),
        b"first seenprovider checked".len()
    );
    assert_eq!(receipt.files()[0].provenance(), DigestProvenance::FirstSeen);
    assert_eq!(
        receipt.files()[1].provenance(),
        DigestProvenance::TrustedSha256
    );
    let canonical = receipt.manifest().canonical_bytes().unwrap();
    assert!(!String::from_utf8_lossy(&canonical).contains("https"));
    let lease = f.open().resolve_ref("fixture").unwrap();
    assert_eq!(lease.manifest().canonical_bytes().unwrap(), canonical);
    assert_eq!(
        lease.manifest().entries()[0].digest_provenance(),
        DigestProvenance::FirstSeen
    );
    assert_eq!(file_bytes(lease.file("a").unwrap()), b"first seen");
    assert_eq!(
        store.list_snapshots().unwrap(),
        vec![receipt.manifest().digest().unwrap()]
    );
    assert_no_host_reservation(&admission);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    // A1 canonical format remains unchanged for absent provenance.
    let legacy = manifest(&[("x", b"old", FileRole::Auxiliary)]);
    assert!(
        !String::from_utf8(legacy.canonical_bytes().unwrap())
            .unwrap()
            .contains("digest_provenance")
    );
    assert_eq!(
        Manifest::from_json(&legacy.canonical_bytes().unwrap(), &Default::default()).unwrap(),
        legacy
    );
}

#[test]
fn known_hash_cache_hit_avoids_http_and_keeps_first_seen_lease_exact() {
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    publish_discovered(&store, &admission, NEW, "fixture");
    let old = store.resolve_ref("fixture").unwrap();
    let script = Script::new(&store, &admission, vec![]);
    let receipt = execute(
        &store,
        &plan(vec![file(PAYLOAD_PATH, NEW, true)]),
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
    )
    .unwrap();
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        old.manifest().entries()[0].digest_provenance(),
        DigestProvenance::FirstSeen
    );
    assert_eq!(
        receipt.manifest().entries()[0].digest_provenance(),
        DigestProvenance::TrustedSha256
    );
    assert_payload(&old, NEW);
    assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
    assert_no_host_reservation(&admission);
}

#[test]
fn cancellation_after_body_progress_drops_only_owned_attempt_and_preserves_ref() {
    struct CancelingBody {
        inner: Box<dyn Read + Send>,
        read: usize,
        cancel: Arc<AtomicBool>,
    }
    impl Read for CancelingBody {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let count = output.len().min(1024);
            let read = self.inner.read(&mut output[..count])?;
            self.read += read;
            if self.read >= 4096 {
                self.cancel.store(true, Ordering::Release);
            }
            Ok(read)
        }
    }
    struct CancelingTransport {
        inner: Script,
        cancel: Arc<AtomicBool>,
    }
    impl HttpTransport for CancelingTransport {
        fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
            let mut response = self.inner.get(request, cancel)?;
            response.body = Box::new(CancelingBody {
                inner: response.body,
                read: 0,
                cancel: self.cancel.clone(),
            });
            Ok(response)
        }
    }
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let cancel = Arc::new(AtomicBool::new(false));
    let body = vec![b'a'; 16 * 1024];
    let transport = CancelingTransport {
        inner: Script::new(&store, &admission, vec![ok(&body)]),
        cancel: cancel.clone(),
    };
    assert!(
        execute(
            &store,
            &plan(vec![file("partial", &body, false)]),
            &transport,
            &admission,
            &cancel,
            Some("fixture")
        )
        .is_err()
    );
    assert!(transport.inner.body_bytes.load(Ordering::Relaxed) >= 4096);
    assert_eq!(
        store
            .resolve_ref("fixture")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        previous
    );
    assert_no_host_reservation(&admission);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
}

#[test]
fn public_resolve_admission_and_cancellation_precede_provider_metadata() {
    use zerogravity_artifacts::acquisition::{Provider, PullRequest, Selection, resolve};
    struct NoMetadata;
    impl HttpTransport for NoMetadata {
        fn get(&self, _request: &HttpRequest, _cancel: &AtomicBool) -> Result<HttpResponse> {
            panic!("metadata requested before mandatory admission/cancellation check");
        }
    }
    let request = PullRequest {
        provider: Provider::GitHub,
        repository: "fixture/repository".into(),
        revision: "main".into(),
        selection: Selection::Files {
            paths: vec!["config.json".into()],
            subfolder: None,
        },
    };
    for (capacity, canceled) in [(1, false), (128 * 1024 * 1024, true)] {
        let admission = admission(capacity);
        let result = resolve(
            &NoMetadata,
            &request,
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(canceled),
        );
        if canceled {
            assert!(matches!(result, Err(Error::State(_))));
        } else {
            assert!(matches!(result, Err(Error::Limit(_))));
        }
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn unfit_host_or_distinct_unknown_disk_peak_never_requests_a_body() {
    let f = Fixture::new();
    let store = f.open();
    for cap in [1, 128 * 1024 * 1024] {
        let admission = admission(cap);
        let bytes = if cap == 1 { 1 } else { 6 * 1024 * 1024 };
        let files = ["a", "b"]
            .into_iter()
            .map(|path| {
                ResolvedFile::new(
                    path,
                    "https://example.invalid/x",
                    FileRole::Weights,
                    bytes,
                    None,
                    path,
                )
                .unwrap()
            })
            .collect();
        let plan = plan(files);
        let script = Script::new(&store, &admission, vec![]);
        assert!(matches!(
            execute(
                &store,
                &plan,
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                None
            ),
            Err(Error::Limit(_))
        ));
        assert_eq!(script.calls.load(Ordering::Relaxed), 0);
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn wrong_digest_length_pointer_and_cancel_preserve_exact_prior_ref() {
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 20\n";
    for (candidate, body, canceled) in [
        (file("bad", NEW, true), vec![b'x'; NEW.len()], false),
        (file("short", NEW, false), b"short".to_vec(), false),
        (file("long", b"x", false), b"longer".to_vec(), false),
        (file("pointer", pointer, false), pointer.to_vec(), false),
        (file("cancel", NEW, false), NEW.to_vec(), true),
    ] {
        let script = Script::new(&store, &admission, vec![ok(&body)]);
        let result = execute(
            &store,
            &plan(vec![candidate]),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(canceled)),
            Some("fixture"),
        );
        assert!(result.is_err());
        if body == pointer {
            assert!(matches!(result, Err(Error::Unsupported(_))));
        }
        assert_eq!(
            store
                .resolve_ref("fixture")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            previous
        );
        assert_no_host_reservation(&admission);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    }
}

#[test]
fn complete_split_and_index_closure_required_before_publication() {
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let split = plan(vec![file("m-00001-of-00002.gguf", b"a", false)]);
    let script = Script::new(&store, &admission, vec![]);
    assert!(
        execute(
            &store,
            &split,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture")
        )
        .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    let index = br#"{"metadata":{"total_size":1},"weight_map":{"x":"missing.safetensors"}}"#;
    let plan = plan(vec![file("model.safetensors.index.json", index, false)]);
    let script = Script::new(&store, &admission, vec![ok(index)]);
    assert!(
        execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture")
        )
        .is_err()
    );
    assert_not_found(store.resolve_ref("fixture"));
    assert_no_host_reservation(&admission);
}

#[test]
fn acquired_index_parser_redacts_remote_fields_and_preserves_previous_ref() {
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let secret = "SIGNED_QUERY_TOKEN_DO_NOT_ECHO";
    let index = br#"{"SIGNED_QUERY_TOKEN_DO_NOT_ECHO":true,"metadata":{"total_size":1},"weight_map":{"x":"weights.safetensors"}}"#;
    // A public caller-built plan is valid before its first-seen body is parsed.
    // A provider's earlier metadata inspection cannot authorize this later body.
    let plan = plan(vec![file("model.safetensors.index.json", index, false)]);
    let transport = Script::new(&store, &admission, vec![ok(index)]);
    let error = execute(
        &store,
        &plan,
        &transport,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Malformed(_)));
    assert!(
        error
            .to_string()
            .contains("invalid acquired safetensors index")
    );
    assert!(!format!("{error} {error:?}").contains(secret));
    let lease = store.resolve_ref("fixture").unwrap();
    assert_eq!(lease.manifest().digest().unwrap(), previous);
    assert_payload(&lease, OLD);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn validator_bound_range_retry_is_single_stream_and_rejects_restart_or_change() {
    for mode in ["append", "restart", "changed"] {
        let f = Fixture::new();
        let store = f.open();
        let admission = admission(128 * 1024 * 1024);
        let bytes = b"hello world";
        let initial = BTreeMap::from([
            ("etag".into(), "\"stable\"".into()),
            ("content-length".into(), "11".into()),
        ]);
        let second = BTreeMap::from([
            (
                "etag".into(),
                if mode == "changed" {
                    "\"changed\""
                } else {
                    "\"stable\""
                }
                .into(),
            ),
            ("content-range".into(), "bytes 5-10/11".into()),
        ]);
        let script = Script::new(
            &store,
            &admission,
            vec![
                (None, 200, initial, b"hello".to_vec()),
                (
                    Some(5),
                    if mode == "restart" { 200 } else { 206 },
                    second,
                    b" world".to_vec(),
                ),
            ],
        );
        let result = execute(
            &store,
            &plan(vec![file("x", bytes, true)]),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        );
        if mode == "append" {
            assert_eq!(
                file_bytes(
                    store
                        .open_snapshot(result.unwrap().manifest().digest().unwrap().as_str())
                        .unwrap()
                        .file("x")
                        .unwrap()
                ),
                bytes
            );
        } else {
            assert!(result.is_err());
            assert_not_found(store.resolve_ref("fixture"));
        }
        assert_eq!(script.calls.load(Ordering::Relaxed), 2);
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn concurrent_unknown_identical_and_different_payloads_dedup_without_quota_loss() {
    for same in [true, false] {
        let f = Fixture::new();
        let store = f.open();
        let admission = admission(512 * 1024 * 1024);
        let barrier = Arc::new(Barrier::new(2));
        let mut joins = Vec::new();
        for index in 0..2 {
            let store = store.clone();
            let admission = admission.clone();
            let barrier = barrier.clone();
            joins.push(std::thread::spawn(move || {
                let bytes = vec![if same || index == 0 { b'a' } else { b'b' }; 512 * 1024];
                let mut tx = store
                    .begin_discovered(
                        source(),
                        vec![discovered(&bytes)],
                        &admission,
                        Arc::new(AtomicBool::new(false)),
                    )
                    .unwrap();
                barrier.wait();
                tx.import_object(PAYLOAD_PATH, &mut bytes.as_slice())
                    .unwrap();
                let manifest = tx.manifest().unwrap();
                let usage = store.usage().unwrap();
                assert!(usage.used_bytes + usage.reserved_bytes <= usage.limit_bytes);
                let expected = digest(&bytes);
                durable(tx.commit(Some(&format!("concurrent-{index}"))).unwrap());
                (manifest.digest().unwrap(), expected)
            }));
        }
        let a = joins.remove(0).join().unwrap();
        let b = joins.remove(0).join().unwrap();
        assert_eq!(a.0 == b.0, same);
        for index in 0..2 {
            let lease = store.resolve_ref(&format!("concurrent-{index}")).unwrap();
            assert_eq!(
                lease.manifest().entries()[0].digest(),
                if index == 0 { &a.1 } else { &b.1 }
            );
        }
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}

#[cfg(feature = "diagnostic-cache")]
mod crash {
    use super::*;
    use cache_transaction_support::process::*;
    use zerogravity_artifacts::cache::{CacheHook, CachePoint, CommitOutcome};
    struct At {
        point: CachePoint,
        ordinal: usize,
        calls: AtomicUsize,
        crash: bool,
    }
    impl CacheHook for At {
        fn point(&self, point: CachePoint) -> Result<()> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == self.ordinal {
                assert_eq!(point, self.point, "discovered callback schedule changed");
                if self.crash {
                    std::process::exit(73);
                }
                return Err(Error::Io(io::Error::from_raw_os_error(28)));
            }
            Ok(())
        }
    }
    #[derive(Default)]
    struct Recorder(Mutex<Vec<CachePoint>>);
    impl CacheHook for Recorder {
        fn point(&self, point: CachePoint) -> Result<()> {
            self.0.lock().unwrap().push(point);
            Ok(())
        }
    }
    fn observed_points() -> Vec<CachePoint> {
        let f = Fixture::new();
        let store = f.open();
        let admission = admission(128 * 1024 * 1024);
        durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let recorder = Arc::new(Recorder::default());
        publish_discovered(
            &store.with_hook(recorder.clone()),
            &admission,
            NEW,
            "fixture",
        );
        let points = recorder.0.lock().unwrap().clone();
        assert!(
            points.len() > 38,
            "new path must include all A1 operations plus actual-manifest transition"
        );
        points
    }
    fn attempt(store: &CacheStore, admission: &TransferAdmission) -> Result<CommitOutcome> {
        let mut tx = store.begin_discovered(
            source(),
            vec![discovered(NEW)],
            admission,
            Arc::new(AtomicBool::new(false)),
        )?;
        let mut reader = NEW;
        tx.import_object(PAYLOAD_PATH, &mut reader)?;
        tx.commit(Some("fixture"))
    }
    #[test]
    fn every_discovery_publication_fault_and_real_child_exit_recovers_complete_state() {
        let points = observed_points();
        eprintln!(
            "discovery observed callbacks: {}; IO failures: {}; real child exits: {}",
            points.len(),
            points.len() * 2,
            points.len() * 2
        );
        for has_prior in [false, true] {
            for crash in [false, true] {
                for (ordinal, point) in points.iter().copied().enumerate() {
                    let f = Fixture::new();
                    let store = f.open();
                    let admission = admission(128 * 1024 * 1024);
                    let previous = has_prior.then(|| {
                        durable(publish(&store, &admission, OLD, Some("fixture")).unwrap())
                    });
                    if crash {
                        let output =
                            RunningChild::spawn(&f.root, "discovery", ordinal, "fixture").finish();
                        assert_eq!(
                            output.status.code(),
                            Some(73),
                            "point {ordinal}/{point:?}: {}",
                            String::from_utf8_lossy(&output.stderr)
                        );
                    } else {
                        let hooked = store.clone().with_hook(Arc::new(At {
                            point,
                            ordinal,
                            calls: AtomicUsize::new(0),
                            crash: false,
                        }));
                        let result = attempt(&hooked, &admission);
                        if reference_changed(point) {
                            assert!(matches!(
                                result,
                                Ok(CommitOutcome::DurabilityUnknown { .. })
                            ));
                        } else {
                            assert!(
                                result.is_err(),
                                "point {ordinal}/{point:?} falsely reported success"
                            );
                        }
                    }
                    let reopened = f.open();
                    if reference_changed(point) {
                        let lease = reopened.resolve_ref("fixture").unwrap();
                        assert_payload(&lease, NEW);
                        assert_eq!(
                            lease.manifest().entries()[0].digest_provenance(),
                            DigestProvenance::FirstSeen
                        );
                    } else if let Some(previous) = previous {
                        let lease = reopened.resolve_ref("fixture").unwrap();
                        assert_eq!(lease.manifest().digest().unwrap(), previous);
                        assert_payload(&lease, OLD);
                    } else {
                        assert_not_found(reopened.resolve_ref("fixture"));
                    }
                    assert_eq!(reopened.usage().unwrap().reserved_bytes, 0);
                    let next = publish_discovered(&reopened, &admission, NEW, "next");
                    assert_payload(&reopened.open_snapshot(&next).unwrap(), NEW);
                    let usage = reopened.usage().unwrap();
                    assert!(usage.used_bytes <= usage.limit_bytes);
                    assert_no_host_reservation(&admission);
                }
            }
        }
    }
    pub(super) fn child() {
        let root = std::path::PathBuf::from(std::env::var_os(CHILD_ROOT).unwrap());
        let ordinal = std::env::var(CHILD_POINT)
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let points = observed_points();
        let store = CacheStore::open(options(&root))
            .unwrap()
            .with_hook(Arc::new(At {
                point: points[ordinal],
                ordinal,
                calls: AtomicUsize::new(0),
                crash: true,
            }));
        attempt(&store, &admission(128 * 1024 * 1024)).unwrap();
        panic!("required discovered boundary missed");
    }
}
#[cfg(feature = "diagnostic-cache")]
#[test]
#[ignore = "owned child helper, invoked only by bounded parent"]
fn cache_child_worker() {
    crash::child();
}
