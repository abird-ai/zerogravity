//! Independent capped-body framing, cancellation and actual publication failures.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
#[cfg(any(feature = "diagnostic-acquisition", feature = "diagnostic-cache"))]
use std::io;
use std::io::Cursor;
#[cfg(feature = "diagnostic-acquisition")]
use std::io::Read;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use zerogravity_artifacts::acquisition::{
    PullLimits, ResolvedFile, ResolvedPullPlan, execute, execute_with_progress,
};
use zerogravity_artifacts::cache::{
    ByteLength, CacheStore, FileRole, ManifestSource, TransferAdmission,
};
use zerogravity_artifacts::{Error, Result};

// A sibling fork can inherit another test's temporary OFDs until exec. Keep
// unrelated subprocess fixtures quiescent, without changing production locks.
static SCHEDULE: Mutex<()> = Mutex::new(());
fn plan(url: &str, cap: u64, sha: Option<String>) -> ResolvedPullPlan {
    let limits = PullLimits {
        unknown_length_cap: Some(cap),
        ..Default::default()
    };
    ResolvedPullPlan::new(
        ManifestSource::new("github", "fixture/capped", "1".repeat(40)).unwrap(),
        vec![
            ResolvedFile::new_with_length(
                PAYLOAD_PATH,
                url,
                FileRole::Auxiliary,
                ByteLength::AtMost(cap),
                sha,
                "immutable-auxiliary",
            )
            .unwrap(),
        ],
        &limits,
    )
    .unwrap()
}
struct Script {
    response: Mutex<Option<HttpResponse>>,
    calls: AtomicUsize,
    store: CacheStore,
    admission: TransferAdmission,
}
impl Script {
    fn new(
        store: &CacheStore,
        admission: &TransferAdmission,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Self {
        Self {
            response: Mutex::new(Some(HttpResponse {
                status: 200,
                headers: headers
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
                body: Box::new(Cursor::new(body.to_vec())),
            })),
            calls: AtomicUsize::new(0),
            store: store.clone(),
            admission: admission.clone(),
        }
    }
}
impl HttpTransport for Script {
    fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        assert!(request.range.is_none(), "unknown bodies must never resume");
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0
        );
        assert!(self.store.usage()?.reserved_bytes > 0);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .response
            .lock()
            .unwrap()
            .take()
            .expect("unknown body retried"))
    }
}
#[cfg(feature = "diagnostic-cache")]
fn attempt(
    store: &CacheStore,
    admission: &TransferAdmission,
    bytes: &[u8],
) -> Result<zerogravity_artifacts::acquisition::PullReceipt> {
    let script = Script::new(store, admission, &[("transfer-encoding", "chunked")], bytes);
    execute(
        store,
        &plan("https://example.invalid/aux", 1024, None),
        &script,
        admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
    )
}
#[test]
fn scripted_lengths_enforce_cap_declared_length_and_old_reference() {
    let _schedule = SCHEDULE.lock().unwrap();
    type Case<'a> = (&'a [(&'a str, &'a str)], &'a [u8], bool);
    let cases: &[Case<'_>] = &[
        (&[("transfer-encoding", "chunked")], b"", true),
        (&[("transfer-encoding", "chunked")], b"abc", true),
        (&[("transfer-encoding", "chunked")], b"abcd", true),
        (&[("transfer-encoding", "chunked")], b"abcde", false),
        (&[("content-length", "4")], b"abc", false),
        (&[("content-length", "3")], b"abcd", false),
        (&[("content-length", "5")], b"abcde", false),
        (&[], b"abc", false),
        (
            &[
                ("transfer-encoding", "chunked"),
                ("content-encoding", "gzip"),
            ],
            b"abc",
            false,
        ),
    ];
    for (headers, body, success) in cases {
        let f = Fixture::new();
        let store = f.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let script = Script::new(&store, &admission, headers, body);
        let result = execute(
            &store,
            &plan("https://example.invalid/aux", 4, None),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        );
        assert_eq!(
            result.is_ok(),
            *success,
            "headers={headers:?}, len={}, result={result:?}",
            body.len()
        );
        let lease = store.resolve_ref("fixture").unwrap();
        if *success {
            assert_payload(&lease, body);
            assert_eq!(lease.manifest().entries()[0].bytes(), body.len() as u64);
            assert_eq!(lease.manifest().entries()[0].digest(), digest(body));
        } else {
            assert_eq!(lease.manifest().digest().unwrap(), old);
            assert_payload(&lease, OLD);
        }
        assert_eq!(script.calls.load(Ordering::Relaxed), 1);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}
#[test]
fn reaching_cap_is_not_eof_and_cancellation_cannot_publish() {
    let _schedule = SCHEDULE.lock().unwrap();
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let cancel = Arc::new(AtomicBool::new(false));
    let script = Script::new(
        &store,
        &admission,
        &[("transfer-encoding", "chunked")],
        b"abcdEXTRA",
    );
    let result = execute_with_progress(
        &store,
        &plan("https://example.invalid/aux", 4, None),
        &script,
        &admission,
        &cancel,
        Some("fixture"),
        &mut |event| {
            if event.file_bytes() == 4 {
                assert!(!event.file_complete());
                assert_eq!(event.file_total_bytes(), None);
                cancel.store(true, Ordering::Release);
            }
        },
    );
    assert!(matches!(result, Err(Error::State(_))));
    assert!(cancel.load(Ordering::Acquire));
    assert_eq!(
        store
            .resolve_ref("fixture")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 1);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}
#[test]
fn known_digest_cache_hits_use_actual_length_and_refuse_a_smaller_cap() {
    let _schedule = SCHEDULE.lock().unwrap();
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    durable(publish(&store, &admission, b"abc", None).unwrap());
    for (cap, success) in [(4, true), (2, false)] {
        let script = Script::new(&store, &admission, &[], b"must not read");
        let result = execute(
            &store,
            &plan("https://example.invalid/aux", cap, Some(digest(b"abc"))),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
        );
        assert_eq!(result.is_ok(), success, "{result:?}");
        if let Ok(receipt) = result {
            assert_eq!(receipt.files()[0].bytes(), 3);
        }
        assert_eq!(script.calls.load(Ordering::Relaxed), 0);
        assert_no_host_reservation(&admission);
    }
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_http_body_framing_and_zero_chunk_completion() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    let _schedule = SCHEDULE.lock().unwrap();
    let cases: &[(&str, &str, bool)] = &[
        ("Transfer-Encoding: chunked\r\n", "0\r\n\r\n", true),
        (
            "Transfer-Encoding: chunked\r\n",
            "3\r\nabc\r\n0\r\n\r\n",
            true,
        ),
        (
            "Transfer-Encoding: chunked\r\n",
            "4\r\nabcd\r\n0\r\n\r\n",
            true,
        ),
        (
            "Transfer-Encoding: chunked\r\n",
            "5\r\nabcde\r\n0\r\n\r\n",
            false,
        ),
        ("Transfer-Encoding: chunked\r\n", "3\r\nabc\r\n", false),
        // ureq accepts a received zero chunk line at FIN. This establishes its
        // body-completion semantics, not validity of a complete trailer section.
        ("Transfer-Encoding: chunked\r\n", "3\r\nabc\r\n0\r\n", true),
        ("Transfer-Encoding: chunked\r\n", "4\r\nabc", false),
        ("", "abc", false),
        ("Content-Length: 3\r\n", "abc", true),
        ("Content-Length: 0\r\n", "", true),
        ("Content-Length: 4\r\n", "abc", false),
        ("Content-Length: 5\r\n", "abcde", false),
        (
            "Transfer-Encoding: chunked\r\nContent-Encoding: gzip\r\n",
            "3\r\nabc\r\n0\r\n\r\n",
            false,
        ),
    ];
    for (headers, body, success) in cases {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let wire = format!("HTTP/1.1 200 OK\r\nConnection: close\r\n{headers}\r\n{body}");
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut stream = loop {
                match listener.accept() {
                    Ok((s, _)) => break s,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "client never connected");
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(e) => panic!("{e}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = Vec::new();
            let mut byte = [0];
            while !request.ends_with(b"\r\n\r\n") {
                assert!(request.len() < 16384);
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            assert!(request.starts_with(b"GET "));
            assert!(
                !String::from_utf8_lossy(&request)
                    .to_ascii_lowercase()
                    .contains("range:")
            );
            let _ = stream.write_all(wire.as_bytes());
        });
        let f = Fixture::new();
        let store = f.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let client = HttpsClient::diagnostic_loopback(
            &origin,
            None,
            HttpOptions {
                max_retries: 0,
                total_timeout: Duration::from_secs(2),
                ..Default::default()
            },
        )
        .unwrap();
        let result = execute(
            &store,
            &plan(&format!("{origin}/aux"), 4, None),
            &client,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        );
        server.join().unwrap();
        assert_eq!(
            result.is_ok(),
            *success,
            "headers={headers:?}, body={body:?}, result={result:?}"
        );
        if !success {
            assert_eq!(
                store
                    .resolve_ref("fixture")
                    .unwrap()
                    .manifest()
                    .digest()
                    .unwrap(),
                old
            );
        }
        assert_no_host_reservation(&admission);
    }
}

#[cfg(feature = "diagnostic-cache")]
mod faults {
    use super::*;
    use cache_transaction_support::process::*;
    use zerogravity_artifacts::cache::{CacheHook, CachePoint, CommitOutcome};
    #[derive(Default)]
    struct Recorder(Mutex<Vec<CachePoint>>);
    impl CacheHook for Recorder {
        fn point(&self, p: CachePoint) -> Result<()> {
            self.0.lock().unwrap().push(p);
            Ok(())
        }
    }
    struct Fail {
        ordinal: usize,
        calls: AtomicUsize,
        crash: bool,
    }
    impl CacheHook for Fail {
        fn point(&self, _: CachePoint) -> Result<()> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == self.ordinal {
                if self.crash {
                    std::process::exit(86);
                }
                return Err(Error::Io(io::Error::other("capped publication fault")));
            }
            Ok(())
        }
    }
    #[test]
    fn every_observed_capped_publication_failure_and_child_exit_preserves_complete_refs() {
        let _schedule = SCHEDULE.lock().unwrap();
        let f = Fixture::new();
        let store = f.open();
        let a = admission(128 * 1024 * 1024);
        let recorder = Arc::new(Recorder::default());
        attempt(&store.with_hook(recorder.clone()), &a, NEW).unwrap();
        let points = recorder.0.lock().unwrap().clone();
        assert!(!points.is_empty());
        eprintln!(
            "capped callbacks {}; I/O fixtures {}; child-exit fixtures {}",
            points.len(),
            points.len() * 2,
            points.len() * 2
        );
        for prior in [false, true] {
            for crash in [false, true] {
                for (ordinal, point) in points.iter().copied().enumerate() {
                    let f = Fixture::new();
                    let store = f.open();
                    let a = admission(128 * 1024 * 1024);
                    let old =
                        prior.then(|| durable(publish(&store, &a, OLD, Some("fixture")).unwrap()));
                    if crash {
                        let output =
                            RunningChild::spawn(&f.root, "capped", ordinal, "fixture").finish();
                        assert_eq!(
                            output.status.code(),
                            Some(86),
                            "{ordinal}/{point:?}: {}",
                            String::from_utf8_lossy(&output.stderr)
                        );
                    } else {
                        let result = attempt(
                            &store.clone().with_hook(Arc::new(Fail {
                                ordinal,
                                calls: AtomicUsize::new(0),
                                crash: false,
                            })),
                            &a,
                            NEW,
                        );
                        if reference_changed(point) {
                            assert!(
                                matches!(
                                    result.as_ref().map(|r| r.outcome()),
                                    Ok(CommitOutcome::DurabilityUnknown { .. })
                                ),
                                "{ordinal}/{point:?}"
                            );
                        } else {
                            assert!(result.is_err(), "unreached fault {ordinal}/{point:?}");
                        }
                    }
                    let reopened = f.open();
                    if reference_changed(point) {
                        let lease = reopened.resolve_ref("fixture").unwrap();
                        assert_payload(&lease, NEW);
                        assert_eq!(lease.manifest().entries()[0].bytes(), NEW.len() as u64);
                        assert_eq!(lease.manifest().entries()[0].digest(), digest(NEW));
                    } else if let Some(old) = old {
                        assert_eq!(
                            reopened
                                .resolve_ref("fixture")
                                .unwrap()
                                .manifest()
                                .digest()
                                .unwrap(),
                            old
                        );
                    } else {
                        assert_not_found(reopened.resolve_ref("fixture"));
                    }
                    assert_eq!(reopened.usage().unwrap().reserved_bytes, 0);
                    attempt(&reopened, &a, NEW).unwrap();
                    assert_payload(&reopened.resolve_ref("fixture").unwrap(), NEW);
                    assert_no_host_reservation(&a);
                }
            }
        }
    }
    pub(super) fn child() {
        let root = std::path::PathBuf::from(std::env::var_os(CHILD_ROOT).unwrap());
        let ordinal = std::env::var(CHILD_POINT).unwrap().parse().unwrap();
        let store = CacheStore::open(options(&root))
            .unwrap()
            .with_hook(Arc::new(Fail {
                ordinal,
                calls: AtomicUsize::new(0),
                crash: true,
            }));
        attempt(&store, &admission(128 * 1024 * 1024), NEW).unwrap();
        panic!("fault callback not reached");
    }
}
#[cfg(feature = "diagnostic-cache")]
#[test]
#[ignore = "bounded owned child helper"]
fn cache_child_worker() {
    faults::child();
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_head_405_oversized_error_length_does_not_read_a_payload() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    let _schedule = SCHEDULE.lock().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("{error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            assert!(request.len() < 16384);
            stream.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
        }
        assert!(request.starts_with(b"HEAD "));
        // The deliberately truncated nonconforming error body would fail if
        // read. Its advertised representation length is not the object length.
        let _ = stream.write_all(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 9999\r\nConnection: close\r\n\r\nDO_NOT_READ");
    });
    let client = HttpsClient::diagnostic_loopback(
        &origin,
        None,
        HttpOptions {
            max_retries: 0,
            total_timeout: Duration::from_secs(2),
            ..Default::default()
        },
    )
    .unwrap();
    let result = client.head(
        &HttpRequest::new(format!("{origin}/aux"), 16),
        &AtomicBool::new(false),
    );
    server.join().unwrap();
    let mut response = result.unwrap();
    assert_eq!(response.status, 405);
    assert_eq!(response.header("content-length"), Some("9999"));
    let mut sentinel = [0u8; 1];
    assert_eq!(response.body.read(&mut sentinel).unwrap(), 0);
}
