//! Real cache ownership with tiny scripted/loopback transfers and fresh processes.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Cursor, Read};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use zerogravity_artifacts::acquisition::{
    PullLimits, ResolvedFile, ResolvedPullPlan, execute, execute_with_progress,
};
use zerogravity_artifacts::cache::{FileRole, ManifestSource};
use zerogravity_artifacts::{Error, Result};

const PREFIX: u64 = 128;
const VALIDATOR: &str = "\"fixture-stable\"";
const SECRET: &str = "SIGNED_QUERY_DO_NOT_DISCLOSE";
type Case = (Option<u64>, u16, BTreeMap<String, String>, Vec<u8>);

struct Script {
    responses: Mutex<VecDeque<Case>>,
    calls: AtomicUsize,
    reads: Arc<AtomicUsize>,
}
struct Counted {
    body: Cursor<Vec<u8>>,
    reads: Arc<AtomicUsize>,
}
impl Read for Counted {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = self.body.read(output)?;
        self.reads.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }
}
impl Script {
    fn new(cases: Vec<Case>) -> Self {
        Self {
            responses: Mutex::new(cases.into()),
            calls: AtomicUsize::new(0),
            reads: Arc::new(AtomicUsize::new(0)),
        }
    }
}
impl HttpTransport for Script {
    fn get(&self, request: &HttpRequest, _cancel: &AtomicBool) -> Result<HttpResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (start, status, headers, body) = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        assert_eq!(request.range.as_ref().map(|range| range.start), start);
        if let Some(range) = &request.range {
            assert_eq!(range.validator, VALIDATOR);
            assert_eq!(request.max_body_bytes, range.total - range.start);
        }
        Ok(HttpResponse {
            status,
            headers,
            body: Box::new(Counted {
                body: Cursor::new(body),
                reads: self.reads.clone(),
            }),
        })
    }
}
fn bytes() -> Vec<u8> {
    (0..1024).map(|n| (n % 251) as u8).collect()
}
fn plan(body: &[u8]) -> ResolvedPullPlan {
    plan_at(body, "1", "asset-42", "https://example.invalid/file")
}
fn plan_at(body: &[u8], revision: &str, identity: &str, url: &str) -> ResolvedPullPlan {
    ResolvedPullPlan::new(
        ManifestSource::new("github", "fixture/repo", revision.repeat(40)).unwrap(),
        vec![
            ResolvedFile::new(
                PAYLOAD_PATH,
                format!("{url}?signature={SECRET}"),
                FileRole::Auxiliary,
                body.len() as u64,
                Some(digest(body)),
                identity,
            )
            .unwrap(),
        ],
        &PullLimits::default(),
    )
    .unwrap()
}
fn initial(body: &[u8]) -> Case {
    (
        None,
        200,
        BTreeMap::from([
            ("etag".into(), VALIDATOR.into()),
            ("content-length".into(), body.len().to_string()),
        ]),
        body.to_vec(),
    )
}
fn continuation(body: &[u8]) -> Case {
    continuation_at(body, PREFIX)
}
fn continuation_at(body: &[u8], start: u64) -> Case {
    (
        Some(start),
        206,
        BTreeMap::from([
            ("etag".into(), VALIDATOR.into()),
            (
                "content-range".into(),
                format!("bytes {start}-{}/{}", body.len() - 1, body.len()),
            ),
            (
                "content-length".into(),
                (body.len() as u64 - start).to_string(),
            ),
        ]),
        body[start as usize..].to_vec(),
    )
}
fn save_prefix(store: &zerogravity_artifacts::cache::CacheStore, body: &[u8]) {
    let admission = admission(128 * 1024 * 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![initial(body)]);
    let result = execute_with_progress(
        store,
        &plan(body),
        &script,
        &admission,
        &cancel,
        Some("fixture"),
        &mut |event| {
            if event.file_bytes() >= PREFIX {
                assert_eq!(event.file_bytes(), PREFIX);
                cancel.store(true, Ordering::Release);
            }
        },
    );
    assert!(result.is_err());
    assert_eq!(script.reads.load(Ordering::Relaxed), PREFIX as usize);
    assert_no_host_reservation(&admission);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
}

#[test]
fn fresh_invocation_rehashes_prefix_and_reports_distinct_monotonic_counters() {
    let fixture = Fixture::new();
    let body = bytes();
    save_prefix(&fixture.open(), &body);
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let script = Script::new(vec![continuation(&body)]);
    let mut previous = 0;
    let mut events = Vec::new();
    let receipt = execute_with_progress(
        &store,
        &plan(&body),
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
        &mut |event| {
            assert_eq!(event.source().repository(), "fixture/repo");
            assert_eq!(event.path(), PAYLOAD_PATH);
            assert_eq!(event.identity(), "asset-42");
            assert!(!format!("{event:?}").contains(SECRET));
            assert!(event.total_bytes() >= previous);
            assert!(event.file_bytes() <= event.file_total_bytes());
            assert!(event.total_bytes() <= event.total_expected_bytes());
            assert_eq!(
                event.downloaded_bytes() + event.cached_bytes() + event.resumed_bytes(),
                event.total_bytes()
            );
            previous = event.total_bytes();
            events.push((
                event.total_bytes(),
                event.resumed_bytes(),
                event.cached_bytes(),
            ));
        },
    )
    .unwrap();
    assert_eq!(events.last(), Some(&(body.len() as u64, PREFIX, 0)));
    assert_eq!(
        script.reads.load(Ordering::Relaxed),
        body.len() - PREFIX as usize
    );
    assert_payload(&store.resolve_ref("fixture").unwrap(), &body);
    assert_eq!(receipt.manifest().entries()[0].digest(), digest(&body));
    let cached = Script::new(vec![]);
    let mut final_counts = None;
    execute_with_progress(
        &store,
        &plan(&body),
        &cached,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
        &mut |event| {
            final_counts = Some((
                event.downloaded_bytes(),
                event.resumed_bytes(),
                event.cached_bytes(),
            ));
        },
    )
    .unwrap();
    assert_eq!(final_counts, Some((0, 0, body.len() as u64)));
    assert_eq!(cached.calls.load(Ordering::Relaxed), 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn resumed_response_refusals_read_no_new_bytes_and_keep_exact_previous_ref() {
    for mode in ["200", "validator", "start", "total", "encoding", "length"] {
        let fixture = Fixture::new();
        let body = bytes();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        save_prefix(&store, &body);
        let mut response = continuation(&body);
        match mode {
            "200" => response.1 = 200,
            "validator" => {
                response.2.insert("etag".into(), "\"changed\"".into());
            }
            "start" => {
                response
                    .2
                    .insert("content-range".into(), "bytes 127-1023/1024".into());
            }
            "total" => {
                response
                    .2
                    .insert("content-range".into(), "bytes 128-1024/1025".into());
            }
            "encoding" => {
                response.2.insert("content-encoding".into(), "gzip".into());
            }
            "length" => {
                response.2.insert("content-length".into(), "897".into());
            }
            _ => unreachable!(),
        }
        let script = Script::new(vec![response]);
        let error = execute(
            &fixture.open(),
            &plan(&body),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        )
        .unwrap_err();
        assert!(!format!("{error} {error:?}").contains(SECRET));
        assert_eq!(script.reads.load(Ordering::Relaxed), 0, "{mode}");
        let lease = store.resolve_ref("fixture").unwrap();
        assert_eq!(lease.manifest().digest().unwrap(), previous);
        assert_payload(&lease, OLD);
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn changed_source_or_object_identity_never_reuses_saved_bytes() {
    for (revision, identity) in [("2", "asset-42"), ("1", "asset-43")] {
        let fixture = Fixture::new();
        let body = bytes();
        save_prefix(&fixture.open(), &body);
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let script = Script::new(vec![initial(&body)]);
        execute_with_progress(
            &store,
            &plan_at(&body, revision, identity, "https://example.invalid/file"),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |event| assert_eq!(event.resumed_bytes(), 0),
        )
        .unwrap();
        assert_eq!(script.reads.load(Ordering::Relaxed), body.len());
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn resumed_suffix_still_requires_the_complete_trusted_digest() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let body = bytes();
    save_prefix(&store, &body);
    let mut response = continuation(&body);
    response.3[0] ^= 1;
    let script = Script::new(vec![response]);
    let error = execute(
        &fixture.open(),
        &plan(&body),
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Malformed(_)));
    let lease = store.resolve_ref("fixture").unwrap();
    assert_eq!(lease.manifest().digest().unwrap(), previous);
    assert_payload(&lease, OLD);
    assert_no_host_reservation(&admission);
}

#[test]
fn absent_or_weak_validator_never_creates_a_resumable_prefix() {
    for validator in [None, Some("W/\"weak\"")] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let body = bytes();
        let mut response = initial(&body);
        response.2.remove("etag");
        if let Some(validator) = validator {
            response.2.insert("etag".into(), validator.into());
        }
        let script = Script::new(vec![response]);
        let cancel = Arc::new(AtomicBool::new(false));
        assert!(
            execute_with_progress(
                &store,
                &plan(&body),
                &script,
                &admission,
                &cancel,
                None,
                &mut |event| {
                    if event.file_bytes() >= PREFIX {
                        cancel.store(true, Ordering::Release);
                    }
                }
            )
            .is_err()
        );
        let next = Script::new(vec![initial(&body)]);
        execute_with_progress(
            &fixture.open(),
            &plan(&body),
            &next,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |event| assert_eq!(event.resumed_bytes(), 0),
        )
        .unwrap();
        assert_eq!(next.reads.load(Ordering::Relaxed), body.len());
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn live_range_retry_never_credits_repeated_prefix_bytes() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let body = bytes();
    let mut first = initial(&body);
    first.3.truncate(500);
    let second = (
        Some(500),
        206,
        BTreeMap::from([
            ("etag".into(), VALIDATOR.into()),
            ("content-range".into(), "bytes 500-1023/1024".into()),
            ("content-length".into(), "524".into()),
        ]),
        body[500..].to_vec(),
    );
    let script = Script::new(vec![first, second]);
    let mut counts = Vec::new();
    execute_with_progress(
        &store,
        &plan(&body),
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
        &mut |event| {
            assert_eq!(event.resumed_bytes(), 0);
            assert_eq!(event.cached_bytes(), 0);
            counts.push(event.downloaded_bytes());
        },
    )
    .unwrap();
    assert_eq!(counts, [0, 128, 500, 1024]);
    assert_eq!(script.reads.load(Ordering::Relaxed), body.len());
    assert_eq!(script.calls.load(Ordering::Relaxed), 2);
    assert_no_host_reservation(&admission);
}

#[test]
fn progress_cancellation_precedes_requests_and_external_errors_are_redacted() {
    struct SecretError;
    impl HttpTransport for SecretError {
        fn get(&self, _request: &HttpRequest, _cancel: &AtomicBool) -> Result<HttpResponse> {
            Err(Error::State(format!("upstream {SECRET}")))
        }
    }
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let never = Script::new(vec![]);
    assert!(
        execute_with_progress(
            &store,
            &plan(&bytes()),
            &never,
            &admission,
            &cancel,
            None,
            &mut |_| cancel.store(true, Ordering::Release)
        )
        .is_err()
    );
    assert_eq!(never.calls.load(Ordering::Relaxed), 0);
    let error = execute(
        &store,
        &plan(&bytes()),
        &SecretError,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap_err();
    assert!(!format!("{error} {error:?}").contains(SECRET));
    assert_no_host_reservation(&admission);
}

#[test]
fn cancellation_when_read_returns_preserves_the_earlier_accepted_prefix() {
    struct CancelOnSecondRead {
        inner: Box<dyn Read + Send>,
        reads: usize,
        cancel: Arc<AtomicBool>,
    }
    impl Read for CancelOnSecondRead {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let count = output.len().min(PREFIX as usize);
            let read = self.inner.read(&mut output[..count])?;
            self.reads += 1;
            if self.reads == 2 {
                self.cancel.store(true, Ordering::Release);
            }
            Ok(read)
        }
    }
    struct CancelingTransport {
        script: Script,
        cancel: Arc<AtomicBool>,
    }
    impl HttpTransport for CancelingTransport {
        fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
            let mut response = self.script.get(request, cancel)?;
            response.body = Box::new(CancelOnSecondRead {
                inner: response.body,
                reads: 0,
                cancel: self.cancel.clone(),
            });
            Ok(response)
        }
    }
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let body = bytes();
    let cancel = Arc::new(AtomicBool::new(false));
    let transport = CancelingTransport {
        script: Script::new(vec![initial(&body)]),
        cancel: cancel.clone(),
    };
    let mut accepted = Vec::new();
    let error = execute_with_progress(
        &store,
        &plan(&body),
        &transport,
        &admission,
        &cancel,
        Some("fixture"),
        &mut |event| accepted.push(event.file_bytes()),
    )
    .unwrap_err();
    assert!(matches!(error, Error::State(_)));
    assert_eq!(
        accepted,
        [0, PREFIX],
        "canceled second chunk must not be accepted"
    );
    assert_eq!(
        transport.script.reads.load(Ordering::Relaxed),
        2 * PREFIX as usize
    );
    assert_eq!(
        store
            .resolve_ref("fixture")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        previous
    );
    let next = Script::new(vec![continuation(&body)]);
    execute(
        &fixture.open(),
        &plan(&body),
        &next,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
    )
    .unwrap();
    assert_eq!(
        next.reads.load(Ordering::Relaxed),
        body.len() - PREFIX as usize
    );
    assert_payload(&store.resolve_ref("fixture").unwrap(), &body);
    assert_no_host_reservation(&admission);
}

#[test]
fn cancel_at_full_length_cannot_save_an_unproved_terminal_prefix() {
    struct ExactThenExtra {
        body: Cursor<Vec<u8>>,
        total: usize,
    }
    impl Read for ExactThenExtra {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let before_end = self.total.saturating_sub(self.body.position() as usize);
            let count = if before_end == 0 {
                output.len()
            } else {
                output.len().min(before_end)
            };
            self.body.read(&mut output[..count])
        }
    }
    struct Overlong(Vec<u8>);
    impl HttpTransport for Overlong {
        fn get(&self, request: &HttpRequest, _cancel: &AtomicBool) -> Result<HttpResponse> {
            assert!(request.range.is_none());
            let mut body = self.0.clone();
            body.push(b'x');
            Ok(HttpResponse {
                status: 200,
                headers: BTreeMap::from([("etag".into(), VALIDATOR.into())]),
                body: Box::new(ExactThenExtra {
                    body: Cursor::new(body),
                    total: self.0.len(),
                }),
            })
        }
    }
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let body = bytes();
    let plan = ResolvedPullPlan::new(
        ManifestSource::new("github", "fixture/repo", "1".repeat(40)).unwrap(),
        vec![
            ResolvedFile::new(
                PAYLOAD_PATH,
                "https://example.invalid/object",
                FileRole::Auxiliary,
                body.len() as u64,
                None,
                "asset-42",
            )
            .unwrap(),
        ],
        &PullLimits::default(),
    )
    .unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    assert!(
        execute_with_progress(
            &store,
            &plan,
            &Overlong(body.clone()),
            &admission,
            &cancel,
            Some("fixture"),
            &mut |event| {
                if event.file_bytes() == event.file_total_bytes() {
                    cancel.store(true, Ordering::Release);
                }
            }
        )
        .is_err()
    );
    assert_eq!(
        store
            .resolve_ref("fixture")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        previous
    );
    let next = Script::new(vec![initial(&body)]);
    execute(
        &fixture.open(),
        &plan,
        &next,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
    )
    .unwrap();
    assert_eq!(
        next.calls.load(Ordering::Relaxed),
        1,
        "terminal EOF must be proved by a new request"
    );
    assert_eq!(next.reads.load(Ordering::Relaxed), body.len());
    assert_payload(&store.resolve_ref("fixture").unwrap(), &body);
    assert_no_host_reservation(&admission);
}

#[cfg(feature = "diagnostic-acquisition")]
mod loopback {
    use super::*;
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};

    fn accept(listener: &TcpListener) -> TcpStream {
        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match listener.accept() {
                Ok((stream, _)) => return stream,
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("bounded loopback accept failed: {error}"),
            }
        }
    }
    fn server(
        body: &[u8],
        mode: &'static str,
    ) -> (String, Arc<Mutex<Vec<String>>>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let captured = Arc::new(Mutex::new(Vec::new()));
        let copy = captured.clone();
        let body = body.to_vec();
        let worker = std::thread::spawn(move || {
            for initial in [true, false] {
                let mut stream = accept(&listener);
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                let mut byte = [0];
                while !request.ends_with(b"\r\n\r\n") {
                    assert!(request.len() < 16 * 1024);
                    assert_eq!(stream.read(&mut byte).unwrap(), 1);
                    request.push(byte[0]);
                }
                copy.lock()
                    .unwrap()
                    .push(String::from_utf8(request).unwrap());
                let (status, validator, data, range) = if initial {
                    ("200 OK", VALIDATOR, body.as_slice(), String::new())
                } else {
                    (
                        if mode == "200" {
                            "200 OK"
                        } else {
                            "206 Partial Content"
                        },
                        if mode == "changed" {
                            "\"changed\""
                        } else {
                            VALIDATOR
                        },
                        &body[PREFIX as usize..],
                        format!(
                            "Content-Range: bytes {PREFIX}-{}/{}\r\n",
                            body.len() - 1,
                            body.len()
                        ),
                    )
                };
                write!(stream, "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\nETag: {validator}\r\n{range}\r\n", data.len()).unwrap();
                stream.write_all(data).unwrap();
            }
        });
        (origin, captured, worker)
    }
    #[test]
    fn persisted_prefix_restarts_with_exact_if_range_on_real_loopback() {
        for mode in ["append", "200", "changed"] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
            let body = bytes();
            let (origin, requests, worker) = server(&body, mode);
            let plan = plan_at(&body, "1", "asset-42", &format!("{origin}/object"));
            let client =
                || HttpsClient::diagnostic_loopback(&origin, None, HttpOptions::default()).unwrap();
            let cancel = Arc::new(AtomicBool::new(false));
            assert!(
                execute_with_progress(
                    &store,
                    &plan,
                    &client(),
                    &admission,
                    &cancel,
                    Some("fixture"),
                    &mut |event| {
                        if event.file_bytes() >= PREFIX {
                            cancel.store(true, Ordering::Release);
                        }
                    }
                )
                .is_err()
            );
            let result = execute(
                &fixture.open(),
                &plan,
                &client(),
                &admission,
                &Arc::new(AtomicBool::new(false)),
                Some("fixture"),
            );
            worker.join().unwrap();
            let requests = requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            let first = requests[0].to_ascii_lowercase();
            let second = requests[1].to_ascii_lowercase();
            assert!(!first.contains("\r\nrange:"));
            assert!(second.contains("\r\nrange: bytes=128-\r\n"));
            assert!(second.contains("\r\nif-range: \"fixture-stable\"\r\n"));
            assert!(second.contains("\r\naccept-encoding: identity\r\n"));
            let lease = store.resolve_ref("fixture").unwrap();
            if mode == "append" {
                assert!(result.is_ok());
                assert_payload(&lease, &body);
            } else {
                assert!(result.is_err());
                assert_eq!(lease.manifest().digest().unwrap(), previous);
                assert_payload(&lease, OLD);
            }
            assert_no_host_reservation(&admission);
        }
    }
}

#[cfg(feature = "diagnostic-cache")]
mod crash {
    use super::*;
    use cache_transaction_support::process::{CHILD_ROOT, RunningChild};
    use zerogravity_artifacts::cache::{CacheHook, CacheMoment, CacheOperation, CachePoint};

    struct ExitAfterCheckpoint;
    impl CacheHook for ExitAfterCheckpoint {
        fn point(&self, point: CachePoint) -> Result<()> {
            if point.operation == CacheOperation::PartialDirectorySync
                && point.moment == CacheMoment::After
            {
                std::process::exit(73);
            }
            Ok(())
        }
    }
    #[test]
    fn actual_child_exit_after_periodic_checkpoint_recovers_prefix_and_exact_old_reference() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let output = RunningChild::spawn(&fixture.root, "partial-transfer", 0, "fixture").finish();
        assert_eq!(
            output.status.code(),
            Some(73),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let reopened = fixture.open();
        let lease = reopened.resolve_ref("fixture").unwrap();
        assert_eq!(lease.manifest().digest().unwrap(), previous);
        assert_payload(&lease, OLD);
        let body = bytes();
        let script = Script::new(vec![continuation_at(&body, 2 * PREFIX)]);
        execute(
            &reopened,
            &plan(&body),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        )
        .unwrap();
        assert_payload(&reopened.resolve_ref("fixture").unwrap(), &body);
        assert_eq!(
            script.reads.load(Ordering::Relaxed),
            body.len() - 2 * PREFIX as usize
        );
        assert_no_host_reservation(&admission);
    }
    pub(super) fn child() {
        struct DelayedChunks {
            inner: Box<dyn Read + Send>,
            bytes: usize,
        }
        impl Read for DelayedChunks {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                if self.bytes == PREFIX as usize {
                    std::thread::sleep(std::time::Duration::from_millis(1100));
                }
                let count = output.len().min(PREFIX as usize);
                let read = self.inner.read(&mut output[..count])?;
                self.bytes += read;
                Ok(read)
            }
        }
        struct DelayedTransport(Script);
        impl HttpTransport for DelayedTransport {
            fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
                let mut response = self.0.get(request, cancel)?;
                response.body = Box::new(DelayedChunks {
                    inner: response.body,
                    bytes: 0,
                });
                Ok(response)
            }
        }
        let root = std::path::PathBuf::from(std::env::var_os(CHILD_ROOT).unwrap());
        let store = zerogravity_artifacts::cache::CacheStore::open(options(&root))
            .unwrap()
            .with_hook(Arc::new(ExitAfterCheckpoint));
        let body = bytes();
        execute(
            &store,
            &plan(&body),
            &DelayedTransport(Script::new(vec![initial(&body)])),
            &admission(128 * 1024 * 1024),
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        )
        .unwrap();
        panic!("required checkpoint exit was missed");
    }
}
#[cfg(feature = "diagnostic-cache")]
#[test]
#[ignore = "owned child helper, invoked by bounded parent"]
fn cache_child_worker() {
    crash::child();
}
