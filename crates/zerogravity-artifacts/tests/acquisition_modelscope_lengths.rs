//! Exact selected HEAD sizing under the original native commit/transfer owners.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Cursor, Read};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::Result;
use zerogravity_artifacts::acquisition::http::{
    HttpRequest, HttpResponse, HttpTarget, HttpTransport,
};
use zerogravity_artifacts::acquisition::modelscope::ModelScopeRequest;
use zerogravity_artifacts::acquisition::{PullLimits, execute};
use zerogravity_artifacts::cache::{CacheLimits, DigestProvenance, TransferAdmission};
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const ORIGIN: &str = "https://modelscope.cn";
const TAG: &str = "\"exact-observation\"";
fn request(paths: &[&str]) -> ModelScopeRequest {
    ModelScopeRequest::new(
        ORIGIN,
        "owner/repo",
        COMMIT,
        paths.iter().map(|s| (*s).to_owned()).collect(),
        None,
    )
    .unwrap()
}
fn row(path: &str, body: &[u8], size: bool, sha: bool) -> Value {
    let mut value = json!({"Path":path,"Type":"blob"});
    if size {
        value["Size"] = json!(body.len())
    }
    if sha {
        value["Sha256"] = json!(digest(body))
    }
    value
}
struct Reply {
    head: bool,
    file: Option<String>,
    status: u16,
    headers: BTreeMap<String, String>,
    data: Vec<u8>,
    never: bool,
    cancel: bool,
}
fn tree(rows: Vec<Value>) -> Reply {
    let data =
        serde_json::to_vec(&json!({"Code":200,"Success":true,"Data":{"Files":rows}})).unwrap();
    Reply {
        head: false,
        file: None,
        status: 200,
        headers: BTreeMap::from([("content-length".into(), data.len().to_string())]),
        data,
        never: false,
        cancel: false,
    }
}
fn raw(path: &str, data: &[u8]) -> Reply {
    Reply {
        head: false,
        file: Some(path.into()),
        status: 200,
        headers: BTreeMap::from([
            ("content-length".into(), data.len().to_string()),
            ("etag".into(), TAG.into()),
        ]),
        data: data.to_vec(),
        never: false,
        cancel: false,
    }
}
fn head(path: &str, size: u64) -> Reply {
    let mut r = raw(path, b"");
    r.head = true;
    r.never = true;
    r.headers.insert("content-length".into(), size.to_string());
    r
}
struct Never;
impl Read for Never {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("HEAD/refused metadata body consumed")
    }
}
struct Counted {
    reader: Cursor<Vec<u8>>,
    bytes: Arc<AtomicUsize>,
}
impl Read for Counted {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let n = self.reader.read(out)?;
        self.bytes.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }
}
struct Script {
    rows: Mutex<VecDeque<Reply>>,
    admission: TransferAdmission,
    heads: AtomicUsize,
    payload: Arc<AtomicUsize>,
}
impl Script {
    fn new(rows: Vec<Reply>, admission: &TransferAdmission) -> Self {
        Self {
            rows: Mutex::new(rows.into()),
            admission: admission.clone(),
            heads: AtomicUsize::new(0),
            payload: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn response(
        &self,
        req: &HttpRequest,
        target: &HttpTarget,
        is_head: bool,
        cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0
        );
        assert!(!format!("{target:?}").contains("api/v1"));
        assert!(req.range.is_none());
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request/fallback");
        assert_eq!(row.head, is_head);
        let url = url::Url::parse(&req.url).unwrap();
        assert_eq!(url.origin().ascii_serialization(), ORIGIN);
        let query = url.query_pairs().collect::<BTreeMap<_, _>>();
        assert_eq!(query.get("Revision").unwrap(), COMMIT);
        let counted = if let Some(path) = row.file {
            assert_eq!(url.path(), "/api/v1/models/owner/repo/repo");
            assert_eq!(query.get("FilePath").unwrap().as_ref(), path.as_str());
            if is_head {
                self.heads.fetch_add(1, Ordering::Relaxed);
            }
            self.payload.clone()
        } else {
            assert!(!is_head);
            assert_eq!(url.path(), "/api/v1/models/owner/repo/repo/files");
            assert_eq!(query.get("Recursive").unwrap(), "False");
            Arc::new(AtomicUsize::new(0))
        };
        if row.cancel {
            cancel.store(true, Ordering::Release)
        }
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body: if row.never {
                Box::new(Never)
            } else {
                Box::new(Counted {
                    reader: Cursor::new(row.data),
                    bytes: counted,
                })
            },
        })
    }
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty())
    }
}
impl HttpTransport for Script {
    fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("untyped metadata/payload request")
    }
    fn head(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("untyped HEAD or redirect downgrade")
    }
    fn get_target(&self, r: &HttpRequest, t: &HttpTarget, c: &AtomicBool) -> Result<HttpResponse> {
        self.response(r, t, false, c)
    }
    fn head_target(&self, r: &HttpRequest, t: &HttpTarget, c: &AtomicBool) -> Result<HttpResponse> {
        self.response(r, t, true, c)
    }
}
#[test]
fn selected_missing_sizes_probe_exact_zero_and_raw_trust_but_unselected_never_probes() {
    let a = admission(128 * 1024 * 1024);
    let script = Script::new(
        vec![
            tree(vec![
                row("a.txt", NEW, false, true),
                row("empty.txt", b"", false, false),
                row("unused.txt", b"", false, false),
            ]),
            head("a.txt", NEW.len() as u64),
            head("empty.txt", 0),
            raw("a.txt", NEW),
            raw("empty.txt", b""),
        ],
        &a,
    );
    let req = request(&["a.txt", "empty.txt"]);
    let plan = req
        .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(script.heads.load(Ordering::Relaxed), 2);
    assert_eq!(script.payload.load(Ordering::Relaxed), 0);
    assert_eq!(plan.total_bytes(), Some(NEW.len() as u64));
    let fixture = Fixture::new();
    let store = fixture.open();
    let cancel = Arc::new(AtomicBool::new(false));
    let receipt = execute(&store, &plan, &script, &a, &cancel, None).unwrap();
    script.finished();
    assert_eq!(
        receipt.files()[0].provenance(),
        DigestProvenance::TrustedSha256
    );
    assert_eq!(receipt.files()[1].provenance(), DigestProvenance::FirstSeen);
    let lease = req
        .open_cached(
            &store,
            &receipt.manifest().digest().unwrap(),
            &PullLimits::default(),
            &cancel,
        )
        .unwrap();
    assert_eq!(file_bytes(lease.file("a.txt").unwrap()), NEW);
    assert!(file_bytes(lease.file("empty.txt").unwrap()).is_empty());
    assert_no_host_reservation(&a);
}
fn index() -> Vec<u8> {
    serde_json::to_vec(&json!({"metadata":{"total_size":4},"weight_map":{"a":"shard-1.safetensors","b":"shard-2.safetensors"}})).unwrap()
}
fn inventory(index: &[u8]) -> Reply {
    tree(vec![
        row("config.json", b"{}", true, true),
        row("tokenizer.json", b"{}", true, true),
        row("model.safetensors.index.json", index, false, true),
        row("shard-1.safetensors", b"aa", false, true),
        row("shard-2.safetensors", b"bb", true, true),
    ])
}
#[test]
fn missing_index_size_is_observed_once_before_metadata_get_and_complete_shard_closure() {
    let data = index();
    let a = admission(128 * 1024 * 1024);
    let script = Script::new(
        vec![
            inventory(&data),
            head("model.safetensors.index.json", data.len() as u64),
            raw("model.safetensors.index.json", &data),
            head("shard-1.safetensors", 2),
        ],
        &a,
    );
    let plan = request(&["shard-1.safetensors"])
        .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
        .unwrap();
    script.finished();
    assert_eq!(script.heads.load(Ordering::Relaxed), 2);
    assert_eq!(script.payload.load(Ordering::Relaxed), data.len());
    assert_eq!(plan.files().len(), 5);
    assert_eq!(
        plan.files()
            .iter()
            .find(|f| f.path() == "model.safetensors.index.json")
            .unwrap()
            .bytes(),
        Some(data.len() as u64)
    );
    assert_no_host_reservation(&a);
}
#[test]
fn tree_head_index_and_later_probe_share_one_request_and_header_byte_budget() {
    let data = index();
    let a = admission(128 * 1024 * 1024);
    let script = Script::new(
        vec![
            inventory(&data),
            head("model.safetensors.index.json", data.len() as u64),
            raw("model.safetensors.index.json", &data),
        ],
        &a,
    );
    let limits = PullLimits {
        max_pages: 3,
        ..Default::default()
    };
    assert!(
        request(&["shard-1.safetensors"])
            .resolve(&script, &limits, &a, &AtomicBool::new(false))
            .is_err()
    );
    script.finished();
    assert_eq!(script.heads.load(Ordering::Relaxed), 1);
    let mut meta = head("a.txt", 3);
    meta.headers
        .insert("x-large-untrusted".into(), "x".repeat(4096));
    let script = Script::new(
        vec![tree(vec![row("a.txt", b"abc", false, false)]), meta],
        &a,
    );
    let limits = PullLimits {
        max_metadata_bytes: 4096,
        ..Default::default()
    };
    assert!(
        request(&["a.txt"])
            .resolve(&script, &limits, &a, &AtomicBool::new(false))
            .is_err()
    );
    script.finished();
    assert_eq!(script.payload.load(Ordering::Relaxed), 0);
    assert_no_host_reservation(&a);
}
#[test]
fn malformed_present_sizes_refuse_even_unselected_while_only_absence_may_probe() {
    let a = admission(128 * 1024 * 1024);
    for size in [Value::Null, json!(-1), json!(1.0), json!("3"), json!(true)] {
        let mut bad = row("unselected.txt", b"", false, false);
        bad["Size"] = size;
        let script = Script::new(vec![tree(vec![row("a.txt", NEW, true, false), bad])], &a);
        assert!(
            request(&["a.txt"])
                .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
                .is_err()
        );
        script.finished();
        assert_eq!(script.heads.load(Ordering::Relaxed), 0);
        assert_no_host_reservation(&a);
    }
}
#[test]
fn exact_head_refuses_errors_redirects_unknown_lengths_encoding_ranges_and_overflow_without_get() {
    let a = admission(128 * 1024 * 1024);
    for mode in 0..12 {
        let mut metadata = head("a.txt", 3);
        match mode {
            0 => metadata.status = 302,
            1 => metadata.status = 405,
            2 => metadata.status = 501,
            3 => {
                metadata.headers.remove("content-length");
            }
            4 => {
                metadata
                    .headers
                    .insert("content-encoding".into(), "gzip".into());
            }
            5 => {
                metadata
                    .headers
                    .insert("transfer-encoding".into(), "chunked".into());
            }
            6 => {
                metadata
                    .headers
                    .insert("content-range".into(), "bytes 0-2/3".into());
            }
            7 => {
                metadata
                    .headers
                    .insert("content-length".into(), "-1".into());
            }
            8 => {
                metadata
                    .headers
                    .insert("content-length".into(), "1.0".into());
            }
            9 => {
                metadata
                    .headers
                    .insert("content-length".into(), "18446744073709551616".into());
            }
            10 => {
                metadata.headers.insert("content-length".into(), "4".into());
            }
            _ => {
                metadata
                    .headers
                    .insert("content-length".into(), "3x".into());
            }
        }
        let script = Script::new(
            vec![tree(vec![row("a.txt", b"abc", false, false)]), metadata],
            &a,
        );
        let limits = PullLimits {
            cache: CacheLimits {
                max_file_bytes: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            request(&["a.txt"])
                .resolve(&script, &limits, &a, &AtomicBool::new(false))
                .is_err()
        );
        script.finished();
        assert_eq!(script.payload.load(Ordering::Relaxed), 0);
        assert_no_host_reservation(&a);
    }
}
#[test]
fn denied_or_precanceled_resolution_never_requests_and_cancel_after_head_publishes_no_plan() {
    let denied = admission(1);
    let script = Script::new(vec![], &denied);
    assert!(
        request(&["a.txt"])
            .resolve(
                &script,
                &PullLimits::default(),
                &denied,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    script.finished();
    assert_no_host_reservation(&denied);
    let a = admission(128 * 1024 * 1024);
    let script = Script::new(vec![], &a);
    assert!(
        request(&["a.txt"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(true))
            .is_err()
    );
    script.finished();
    let mut metadata = head("a.txt", 3);
    metadata.cancel = true;
    let script = Script::new(
        vec![tree(vec![row("a.txt", b"abc", false, false)]), metadata],
        &a,
    );
    assert!(
        request(&["a.txt"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
            .is_err()
    );
    script.finished();
    assert_eq!(script.payload.load(Ordering::Relaxed), 0);
    assert_no_host_reservation(&a);
}
#[test]
fn strong_head_validator_is_checked_before_index_or_payload_body_while_weak_disables_resume() {
    let a = admission(128 * 1024 * 1024);
    let data = index();
    let mut body = raw("model.safetensors.index.json", &data);
    body.headers.insert("etag".into(), "\"changed\"".into());
    body.never = true;
    let script = Script::new(
        vec![
            inventory(&data),
            head("model.safetensors.index.json", data.len() as u64),
            body,
        ],
        &a,
    );
    assert!(
        request(&["shard-1.safetensors"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
            .is_err()
    );
    script.finished();
    assert_eq!(script.payload.load(Ordering::Relaxed), 0);
    for tag in [None, Some("W/\"weak\"")] {
        let mut metadata = head("a.txt", 3);
        if let Some(tag) = tag {
            metadata.headers.insert("etag".into(), tag.into());
        } else {
            metadata.headers.remove("etag");
        }
        let mut body = raw("a.txt", b"abc");
        body.headers.remove("etag");
        let script = Script::new(
            vec![
                tree(vec![row("a.txt", b"abc", false, false)]),
                metadata,
                body,
            ],
            &a,
        );
        let plan = request(&["a.txt"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
            .unwrap();
        let fixture = Fixture::new();
        execute(
            &fixture.open(),
            &plan,
            &script,
            &a,
            &Arc::new(AtomicBool::new(false)),
            None,
        )
        .unwrap();
        script.finished();
    }
    assert_no_host_reservation(&a);
}
#[test]
fn listed_and_probed_exact_bytes_keep_same_source_file_identity_and_aggregate_bound() {
    let a = admission(128 * 1024 * 1024);
    let req = request(&["a.txt"]);
    let listed = Script::new(vec![tree(vec![row("a.txt", NEW, true, true)])], &a);
    let p = req
        .resolve(&listed, &PullLimits::default(), &a, &AtomicBool::new(false))
        .unwrap();
    listed.finished();
    let probed = Script::new(
        vec![
            tree(vec![row("a.txt", NEW, false, true)]),
            head("a.txt", NEW.len() as u64),
        ],
        &a,
    );
    let q = req
        .resolve(&probed, &PullLimits::default(), &a, &AtomicBool::new(false))
        .unwrap();
    probed.finished();
    assert_eq!(p.source(), q.source());
    assert_eq!(p.files()[0].identity(), q.files()[0].identity());
    let script = Script::new(
        vec![
            tree(vec![
                row("a.txt", b"123456", false, false),
                row("b.txt", b"123456", false, false),
            ]),
            head("a.txt", 6),
            head("b.txt", 6),
        ],
        &a,
    );
    let limits = PullLimits {
        cache: CacheLimits {
            max_snapshot_bytes: 10,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(
        request(&["a.txt", "b.txt"])
            .resolve(&script, &limits, &a, &AtomicBool::new(false))
            .is_err()
    );
    script.finished();
    assert_eq!(script.payload.load(Ordering::Relaxed), 0);
    assert_no_host_reservation(&a);
}

#[test]
fn probed_strong_validator_refuses_missing_changed_or_weak_payload_before_read_and_preserves_ref() {
    let a = admission(128 * 1024 * 1024);
    for tag in [None, Some("\"changed\""), Some("W/\"weak\"")] {
        let mut body = raw("a.txt", NEW);
        if let Some(tag) = tag {
            body.headers.insert("etag".into(), tag.into());
        } else {
            body.headers.remove("etag");
        }
        body.never = true;
        let script = Script::new(
            vec![
                tree(vec![row("a.txt", NEW, false, false)]),
                head("a.txt", NEW.len() as u64),
                body,
            ],
            &a,
        );
        let plan = request(&["a.txt"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
            .unwrap();
        let fixture = Fixture::new();
        let store = fixture.open();
        let old = durable(publish(&store, &a, OLD, Some("keep")).unwrap());
        assert!(
            execute(
                &store,
                &plan,
                &script,
                &a,
                &Arc::new(AtomicBool::new(false)),
                Some("keep")
            )
            .is_err()
        );
        script.finished();
        assert_eq!(script.payload.load(Ordering::Relaxed), 0);
        assert_eq!(
            store
                .resolve_ref("keep")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            old
        );
        assert_no_host_reservation(&a);
    }
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_modelscope_token_tree_head_and_body_keep_exact_commit_origin_and_never_follow_redirect() {
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    fn accept(listener: &TcpListener) -> TcpStream {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    return stream;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < deadline,
                        "owned ModelScope HEAD fixture timeout"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("{e}"),
            }
        }
    }
    for redirect in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let unused = TcpListener::bind("127.0.0.1:0").unwrap();
        unused.set_nonblocking(true).unwrap();
        let location = format!("http://{}/NEVER_ECHO", unused.local_addr().unwrap());
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            for index in 0..if redirect { 2 } else { 3 } {
                let mut stream = accept(&listener);
                let mut request = Vec::new();
                let mut buf = [0; 2048];
                while !request.windows(4).any(|s| s == b"\r\n\r\n") {
                    let n = stream.read(&mut buf).unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&buf[..n]);
                    assert!(request.len() < 16384)
                }
                let text = std::str::from_utf8(&request).unwrap();
                let first = text.lines().next().unwrap();
                assert!(first.starts_with(if index == 1 { "HEAD " } else { "GET " }));
                assert!(first.contains(&format!("Revision={COMMIT}")));
                if index == 0 {
                    assert!(first.contains("/repo/files?"));
                } else {
                    assert!(first.contains("FilePath=a.txt"));
                }
                let lower = text.to_ascii_lowercase();
                assert_eq!(lower.matches("\r\nauthorization:").count(), 1);
                assert!(lower.contains("authorization: bearer synthetic-head-token\r\n"));
                assert_eq!(lower.matches("\r\ncookie:").count(), 1);
                assert!(lower.contains("cookie: m_session_id=synthetic-head-token\r\n"));
                assert!(!lower.contains("range:"));
                if redirect && index == 1 {
                    let response = format!(
                        "HTTP/1.1 302 Found\r\nConnection: close\r\nLocation: {location}\r\nContent-Length: 0\r\n\r\n"
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                    continue;
                }
                let data = if index == 0 {
                    tree(vec![row("a.txt", NEW, false, true)]).data
                } else {
                    NEW.to_vec()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nConnection: close\r\nETag: {TAG}\r\nContent-Length: {}\r\n\r\n",
                    data.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                if index != 1 {
                    stream.write_all(&data).unwrap()
                }
            }
        });
        let req = ModelScopeRequest::new(&origin, "owner/repo", COMMIT, vec!["a.txt".into()], None)
            .unwrap();
        let client = HttpsClient::diagnostic_modelscope_loopback(
            &origin,
            "synthetic-head-token",
            HttpOptions {
                max_retries: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let a = admission(128 * 1024 * 1024);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = req.resolve(&client, &PullLimits::default(), &a, &cancel);
        if redirect {
            let error = plan.unwrap_err().to_string();
            assert!(!error.contains("NEVER_ECHO"));
        } else {
            let fixture = Fixture::new();
            let store = fixture.open();
            let receipt = execute(&store, &plan.unwrap(), &client, &a, &cancel, None).unwrap();
            assert_eq!(
                file_bytes(
                    req.open_cached(
                        &store,
                        &receipt.manifest().digest().unwrap(),
                        &PullLimits::default(),
                        &cancel
                    )
                    .unwrap()
                    .file("a.txt")
                    .unwrap()
                ),
                NEW
            );
        }
        worker.join().unwrap();
        assert!(matches!(unused.accept(),Err(e)if e.kind()==io::ErrorKind::WouldBlock));
        assert_no_host_reservation(&a);
    }
}
