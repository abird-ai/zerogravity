//! Exact pinned length recovery; HEAD bodies are never consumed as payloads.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use serde_json::json;
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Cursor, Read};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use zerogravity_artifacts::acquisition::{
    ComponentKind, ComponentPullRequest, ComponentRequest, ComponentTransport, DigestProvenance,
    Provider, PullLimits, PullRequest, Selection, execute, execute_with_progress, resolve,
    resolve_components,
};
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::{Error, Result};

const DATA: &[u8] = b"exact tiny source";
const TAG: &str = "\"pinned-object\"";
const SECRET: &str = "SIGNED_QUERY_DO_NOT_EXPOSE";
fn pin() -> String {
    "1".repeat(40)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    Get,
    Head,
}
struct Row {
    method: Method,
    path: String,
    status: u16,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
    range: Option<u64>,
    forbidden_body: bool,
}
struct Script {
    rows: Mutex<VecDeque<Row>>,
    heads: AtomicUsize,
    calls: AtomicUsize,
    admission: TransferAdmission,
}
struct NoHeadBody;
impl Read for NoHeadBody {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("metadata probe or refused object body must never be read")
    }
}
impl Script {
    fn new(rows: Vec<Row>, admission: &TransferAdmission) -> Self {
        Self {
            rows: Mutex::new(rows.into()),
            heads: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            admission: admission.clone(),
        }
    }
    fn response(&self, method: Method, request: &HttpRequest) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0,
            "metadata/transfer admission must precede HTTP"
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        if method == Method::Head {
            self.heads.fetch_add(1, Ordering::Relaxed);
        }
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        assert_eq!(method, row.method);
        if method == Method::Head
            && row
                .headers
                .get("content-length")
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|bytes| bytes > 8 * 1024 * 1024)
        {
            assert!(
                request.max_body_bytes >= 64 * 1024 * 1024,
                "a HEAD object bound cannot be clamped to the small metadata body budget"
            );
        }
        assert!(
            request.url.contains(&row.path),
            "unexpected pinned fixture URL"
        );
        assert_eq!(request.range.as_ref().map(|range| range.start), row.range);
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body: if method == Method::Head || row.forbidden_body {
                Box::new(NoHeadBody) as Box<dyn Read + Send>
            } else {
                Box::new(Cursor::new(row.body))
            },
        })
    }
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        self.response(Method::Get, request)
    }
    fn head(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        self.response(Method::Head, request)
    }
}
impl ComponentTransport for Script {
    fn get(&self, _: &str, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        HttpTransport::get(self, request, cancel)
    }
    fn head(&self, _: &str, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        HttpTransport::head(self, request, cancel)
    }
}
fn get(path: impl Into<String>, body: Vec<u8>) -> Row {
    Row {
        method: Method::Get,
        path: path.into(),
        status: 200,
        headers: BTreeMap::from([("content-length".into(), body.len().to_string())]),
        body,
        range: None,
        forbidden_body: false,
    }
}
fn metadata(path: impl Into<String>, value: serde_json::Value) -> Row {
    get(path, serde_json::to_vec(&value).unwrap())
}
fn head(path: impl Into<String>, size: Option<u64>) -> Row {
    Row {
        method: Method::Head,
        path: path.into(),
        status: 200,
        headers: size
            .map(|size| {
                BTreeMap::from([
                    ("content-length".into(), size.to_string()),
                    ("etag".into(), TAG.into()),
                ])
            })
            .unwrap_or_default(),
        body: Vec::new(),
        range: None,
        forbidden_body: true,
    }
}
fn hf_tree(entries: serde_json::Value) -> Vec<Row> {
    vec![
        metadata("/revision/main", json!({"sha":pin()})),
        metadata(format!("/tree/{}?", pin()), entries),
    ]
}
fn entry(path: &str, size: Option<u64>) -> serde_json::Value {
    json!({"type":"file","path":path,"size":size})
}
fn file_request(provider: Provider, path: &str) -> PullRequest {
    PullRequest {
        provider,
        repository: "fixture/repo".into(),
        revision: "main".into(),
        selection: Selection::Files {
            paths: vec![path.into()],
            subfolder: None,
        },
    }
}

#[test]
fn hf_lfs_size_fallback_and_exact_zero_need_no_probe() {
    for body in [DATA, &b""[..]] {
        let admission = admission(128 * 1024 * 1024);
        let mut listed = entry("data.txt", None);
        listed["lfs"] = json!({"oid":digest(body),"size":body.len()});
        let script = Script::new(
            hf_tree(json!([listed, entry("unselected.bin", None)])),
            &admission,
        );
        let plan = resolve(
            &script,
            &file_request(Provider::HuggingFace, "data.txt"),
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(plan.files()[0].bytes().unwrap(), body.len() as u64);
        assert_eq!(plan.files()[0].sha256(), Some(digest(body).as_str()));
        assert_eq!(
            plan.files()[0].digest_provenance(),
            DigestProvenance::TrustedSha256
        );
        assert_eq!(script.heads.load(Ordering::Relaxed), 0);
        script.finished();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn missing_hf_length_uses_body_free_pinned_head_and_preserves_first_seen_trust() {
    for body in [DATA, &b""[..]] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let mut rows = hf_tree(json!([entry("data.txt", None)]));
        rows.push(head(
            format!("/resolve/{}/data.txt", pin()),
            Some(body.len() as u64),
        ));
        let script = Script::new(rows, &admission);
        let plan = resolve(
            &script,
            &file_request(Provider::HuggingFace, "data.txt"),
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
        script.finished();
        assert_eq!(
            script.calls.load(Ordering::Relaxed),
            3,
            "metadata-only planning cannot issue payload GET"
        );
        assert_eq!(plan.files()[0].bytes().unwrap(), body.len() as u64);
        assert_eq!(
            plan.files()[0].digest_provenance(),
            DigestProvenance::FirstSeen
        );
        let mut object = get(format!("/resolve/{}/data.txt", pin()), body.to_vec());
        object.headers.insert("etag".into(), TAG.into());
        let transfer = Script::new(vec![object], &admission);
        let receipt = execute(
            &store,
            &plan,
            &transfer,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        )
        .unwrap();
        assert_eq!(receipt.files()[0].provenance(), DigestProvenance::FirstSeen);
        assert_eq!(
            file_bytes(
                store
                    .resolve_ref("fixture")
                    .unwrap()
                    .file("data.txt")
                    .unwrap()
            ),
            body
        );
        transfer.finished();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn head_rejects_unknown_conflicting_encoded_ranged_oversize_or_failed_metadata() {
    for mode in [
        "missing", "number", "encoding", "range", "transfer", "conflict", "oversize", "status",
        "commit",
    ] {
        let admission = admission(128 * 1024 * 1024);
        let mut rows = hf_tree(json!([entry("data.txt", None)]));
        let mut response = head(
            format!("/resolve/{}/data.txt", pin()),
            Some(DATA.len() as u64),
        );
        match mode {
            "missing" => {
                response.headers.remove("content-length");
            }
            "number" => {
                response
                    .headers
                    .insert("content-length".into(), format!("{SECRET}bad"));
            }
            "encoding" => {
                response
                    .headers
                    .insert("content-encoding".into(), "gzip".into());
            }
            "range" => {
                response
                    .headers
                    .insert("content-range".into(), "bytes0-16/17".into());
            }
            "transfer" => {
                response
                    .headers
                    .insert("transfer-encoding".into(), "chunked".into());
            }
            "conflict" => {
                response
                    .headers
                    .insert("x-linked-size".into(), "999".into());
            }
            "oversize" => {
                response
                    .headers
                    .insert("content-length".into(), "4097".into());
            }
            "status" => response.status = 503,
            "commit" => {
                response
                    .headers
                    .insert("x-repo-commit".into(), "2".repeat(40));
            }
            _ => unreachable!(),
        }
        rows.push(response);
        let script = Script::new(rows, &admission);
        let mut limits = PullLimits::default();
        limits.cache.max_file_bytes = 4096;
        let error = resolve(
            &script,
            &file_request(Provider::HuggingFace, "data.txt"),
            &limits,
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(!format!("{error} {error:?}").contains(SECRET));
        script.finished();
        assert_no_host_reservation(&admission);
    }
    let admission = admission(128 * 1024 * 1024);
    let mut listed = entry("data.txt", Some(DATA.len() as u64 + 1));
    listed["lfs"] = json!({"oid":digest(DATA),"size":DATA.len()});
    let script = Script::new(hf_tree(json!([listed])), &admission);
    assert!(matches!(
        resolve(
            &script,
            &file_request(Provider::HuggingFace, "data.txt"),
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Malformed(_))
    ));
    assert_eq!(script.heads.load(Ordering::Relaxed), 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn strong_probe_validator_must_match_get_without_changing_existing_ref() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let mut rows = hf_tree(json!([entry("data.txt", None)]));
    rows.push(head("/data.txt", Some(DATA.len() as u64)));
    let plan = resolve(
        &Script::new(rows, &admission),
        &file_request(Provider::HuggingFace, "data.txt"),
        &PullLimits::default(),
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    for validator in [None, Some("\"changed\"")] {
        let mut row = get("/data.txt", DATA.to_vec());
        if let Some(value) = validator {
            row.headers.insert("etag".into(), value.into());
        }
        row.forbidden_body = true;
        let error = execute(
            &store,
            &plan,
            &Script::new(vec![row], &admission),
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::State(_)));
        assert_eq!(
            store
                .resolve_ref("fixture")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            previous
        );
    }
    assert_no_host_reservation(&admission);
}

#[test]
fn changed_probe_validator_cannot_append_an_older_saved_prefix() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let body = vec![b'a'; 512];
    let make_plan = |tag: &str| {
        let mut rows = hf_tree(json!([entry("data.txt", None)]));
        let mut probe = head("/data.txt", Some(body.len() as u64));
        probe.headers.insert("etag".into(), tag.into());
        rows.push(probe);
        resolve(
            &Script::new(rows, &admission),
            &file_request(Provider::HuggingFace, "data.txt"),
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap()
    };
    let plan = make_plan(TAG);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut first = get("/data.txt", body.clone());
    first.headers.insert("etag".into(), TAG.into());
    assert!(
        execute_with_progress(
            &store,
            &plan,
            &Script::new(vec![first], &admission),
            &admission,
            &cancel,
            Some("fixture"),
            &mut |event| {
                if event.file_bytes() >= 128 {
                    cancel.store(true, Ordering::Release);
                }
            }
        )
        .is_err()
    );
    let changed = make_plan("\"different-probe\"");
    let mut range = get("/data.txt", body[128..].to_vec());
    range.status = 206;
    range.range = Some(128);
    range.forbidden_body = true;
    range.headers.insert("etag".into(), TAG.into());
    range
        .headers
        .insert("content-range".into(), "bytes 128-511/512".into());
    assert!(matches!(
        execute(
            &fixture.open(),
            &changed,
            &Script::new(vec![range], &admission),
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture")
        ),
        Err(Error::State(_))
    ));
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
}

#[test]
fn missing_index_length_preserves_full_closure_and_checks_probe_validator_before_index_read() {
    let index = br#"{"metadata":{"total_size":1},"weight_map":{"x":"weights.safetensors"}}"#;
    for changed in [false, true] {
        let admission = admission(128 * 1024 * 1024);
        let mut rows = hf_tree(json!([
            entry("model.safetensors.index.json", None),
            entry("weights.safetensors", None),
            entry("config.json", Some(2)),
            entry("tokenizer.json", Some(2)),
        ]));
        rows.push(head(
            "/model.safetensors.index.json",
            Some(index.len() as u64),
        ));
        let mut metadata = get("/model.safetensors.index.json", index.to_vec());
        metadata.headers.insert(
            "etag".into(),
            if changed { "\"changed\"" } else { TAG }.into(),
        );
        metadata.forbidden_body = changed;
        rows.push(metadata);
        if !changed {
            rows.push(head("/weights.safetensors", Some(DATA.len() as u64)));
        }
        let script = Script::new(rows, &admission);
        let result = resolve(
            &script,
            &file_request(Provider::HuggingFace, "model.safetensors.index.json"),
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        );
        if changed {
            assert!(matches!(result, Err(Error::State(_))));
        } else {
            let plan = result.unwrap();
            assert_eq!(
                plan.files()
                    .iter()
                    .map(|file| file.path())
                    .collect::<Vec<_>>(),
                [
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "weights.safetensors"
                ]
            );
            assert_eq!(script.heads.load(Ordering::Relaxed), 2);
        }
        script.finished();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn failed_head_observes_cancellation_before_error_redaction_and_preserves_limit_kind() {
    struct FailingHead {
        script: Script,
        cancel: bool,
    }
    impl HttpTransport for FailingHead {
        fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
            HttpTransport::get(&self.script, request, cancel)
        }
        fn head(&self, _: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
            if self.cancel {
                cancel.store(true, Ordering::Release);
                Err(Error::Unsupported(format!("upstream {SECRET}")))
            } else {
                Err(Error::Limit(format!("upstream {SECRET}")))
            }
        }
    }
    for canceled in [false, true] {
        let admission = admission(128 * 1024 * 1024);
        let script = FailingHead {
            script: Script::new(hf_tree(json!([entry("data.txt", None)])), &admission),
            cancel: canceled,
        };
        let error = resolve(
            &script,
            &file_request(Provider::HuggingFace, "data.txt"),
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        if canceled {
            assert!(matches!(error, Error::State(_)));
            assert!(error.to_string().contains("canceled"));
        } else {
            assert!(matches!(error, Error::Limit(_)));
        }
        assert!(!format!("{error} {error:?}").contains(SECRET));
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn github_missing_tree_and_release_asset_sizes_probe_the_exact_object() {
    for release in [false, true] {
        let admission = admission(128 * 1024 * 1024);
        let request = if release {
            PullRequest {
                provider: Provider::GitHub,
                repository: "fixture/repo".into(),
                revision: "v1".into(),
                selection: Selection::ReleaseAsset {
                    name: "data.txt".into(),
                },
            }
        } else {
            file_request(Provider::GitHub, "data.txt")
        };
        let rows = if release {
            vec![
                metadata("/releases/tags/v1", json!({"id":9,"tag_name":"v1"})),
                metadata(
                    "/releases/9/assets?",
                    json!([{"id":7,"name":"data.txt","state":"uploaded","size":null,"digest":null}]),
                ),
                head("/releases/assets/7", Some(DATA.len() as u64)),
            ]
        } else {
            vec![
                metadata(
                    "/commits/main",
                    json!({"sha":pin(),"commit":{"tree":{"sha":"2".repeat(40)}}}),
                ),
                metadata(
                    "/git/trees/",
                    json!({"sha":"2".repeat(40),"truncated":false,"tree":[{"path":"data.txt","mode":"100644","type":"blob","sha":"3".repeat(40),"size":null}]}),
                ),
                head(
                    format!("/fixture/repo/{}/data.txt", pin()),
                    Some(DATA.len() as u64),
                ),
            ]
        };
        let script = Script::new(rows, &admission);
        let plan = resolve(
            &script,
            &request,
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(plan.files()[0].bytes().unwrap(), DATA.len() as u64);
        assert_eq!(
            plan.files()[0].sha256(),
            None,
            "Git OID/ETag cannot become raw SHA256"
        );
        if release {
            assert_eq!(plan.source().revision(), "release-9-asset-7");
        }
        script.finished();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn ordinary_and_component_head_probe_counts_are_bounded() {
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits {
        max_pages: 1,
        ..PullLimits::default()
    };
    let mut request = file_request(Provider::HuggingFace, "a.txt");
    request.selection = Selection::Files {
        paths: vec!["a.txt".into(), "b.txt".into()],
        subfolder: None,
    };
    let mut rows = hf_tree(json!([entry("a.txt", None), entry("b.txt", None)]));
    rows.push(head("/a.txt", Some(1)));
    let script = Script::new(rows, &admission);
    assert!(matches!(
        resolve(
            &script,
            &request,
            &limits,
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(script.heads.load(Ordering::Relaxed), 1);
    script.finished();

    let limits = PullLimits {
        max_pages: 6,
        ..PullLimits::default()
    };
    let components = ComponentPullRequest::new(
        vec![
            ComponentRequest::new(
                "weights",
                ComponentKind::Weights,
                file_request(Provider::HuggingFace, "model.gguf"),
            )
            .unwrap(),
            ComponentRequest::new(
                "config",
                ComponentKind::Config,
                file_request(Provider::HuggingFace, "config.json"),
            )
            .unwrap(),
            ComponentRequest::new(
                "tokenizer",
                ComponentKind::Tokenizer,
                file_request(Provider::HuggingFace, "tokenizer.json"),
            )
            .unwrap(),
        ],
        &limits,
    )
    .unwrap();
    let entries = json!([
        entry("model.gguf", None),
        entry("config.json", Some(2)),
        entry("tokenizer.json", Some(2))
    ]);
    let mut rows = hf_tree(entries.clone());
    rows.push(head("/model.gguf", Some(DATA.len() as u64)));
    rows.push(metadata(
        format!("/revision/{}", pin()),
        json!({"sha":pin()}),
    ));
    rows.push(metadata("/tree/", entries));
    rows.push(metadata(
        format!("/revision/{}", pin()),
        json!({"sha":pin()}),
    ));
    let script = Script::new(rows, &admission);
    assert!(
        resolve_components(
            &components,
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 6);
    assert_eq!(script.heads.load(Ordering::Relaxed), 1);
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn large_exact_head_length_does_not_consume_the_component_json_body_budget() {
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits {
        max_metadata_bytes: 4096,
        max_pages: 3,
        ..PullLimits::default()
    };
    let request = ComponentPullRequest::new(
        vec![
            ComponentRequest::new(
                "weights",
                ComponentKind::Weights,
                file_request(Provider::HuggingFace, "model.gguf"),
            )
            .unwrap(),
        ],
        &limits,
    )
    .unwrap();
    let mut rows = hf_tree(json!([
        entry("model.gguf", None),
        entry("config.json", Some(2)),
        entry("tokenizer.json", Some(2))
    ]));
    rows.push(head("/model.gguf", Some(64 * 1024 * 1024)));
    let script = Script::new(rows, &admission);
    let plan = resolve_components(
        &request,
        &script,
        &limits,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(plan.total_bytes().unwrap(), 64 * 1024 * 1024 + 4);
    script.finished();
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
                Err(error) => panic!("bounded HEAD fixture accept failed: {error}"),
            }
        }
    }
    fn request(stream: &mut TcpStream) -> String {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut byte = [0];
        while !bytes.ends_with(b"\r\n\r\n") {
            assert!(bytes.len() < 16 * 1024);
            assert_eq!(stream.read(&mut byte).unwrap(), 1);
            bytes.push(byte[0]);
        }
        String::from_utf8(bytes).unwrap()
    }
    fn serve(response: String) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            let mut stream = accept(&listener);
            let captured = request(&mut stream);
            stream.write_all(response.as_bytes()).unwrap();
            captured
        });
        (origin, worker)
    }
    #[test]
    fn actual_head_redirect_strips_cross_origin_auth_and_never_reads_a_declared_body() {
        let (second, second_worker) = serve(format!(
            "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\nETag: {TAG}\r\n\r\n",
            64 * 1024 * 1024
        ));
        let (first, first_worker) = serve(format!(
            "HTTP/1.1 302 Found\r\nConnection: close\r\nContent-Length: 0\r\nLocation: {second}/object?signature={SECRET}\r\n\r\n"
        ));
        let client = HttpsClient::diagnostic_loopback(
            &first,
            Some("fixture_bearer".into()),
            HttpOptions::default(),
        )
        .unwrap();
        let mut response = client
            .head(
                &HttpRequest::new(format!("{first}/object"), 64 * 1024 * 1024),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(response.header("content-length"), Some("67108864"));
        assert_eq!(response.body.read(&mut [0]).unwrap(), 0);
        let first = first_worker.join().unwrap().to_ascii_lowercase();
        let second = second_worker.join().unwrap().to_ascii_lowercase();
        assert!(first.starts_with("head /object "));
        assert!(first.contains("authorization: bearer fixture_bearer"));
        assert!(second.starts_with("head /object?signature="));
        assert!(!second.contains("authorization:"));
        assert!(second.contains("accept-encoding: identity"));
    }
    #[test]
    fn actual_head_header_deadline_and_cancellation_are_bounded_and_redacted() {
        for canceled in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let cancel = Arc::new(AtomicBool::new(false));
            let worker_cancel = cancel.clone();
            let worker = std::thread::spawn(move || {
                let mut stream = accept(&listener);
                let captured = request(&mut stream);
                assert!(captured.starts_with("HEAD "));
                if canceled {
                    worker_cancel.store(true, Ordering::Release);
                }
                std::thread::sleep(Duration::from_millis(160));
            });
            let client = HttpsClient::diagnostic_loopback(
                &origin,
                None,
                HttpOptions {
                    max_retries: 0,
                    total_timeout: Duration::from_millis(60),
                    response_header_timeout: Duration::from_millis(60),
                    ..HttpOptions::default()
                },
            )
            .unwrap();
            let start = Instant::now();
            let error = client
                .head(
                    &HttpRequest::new(format!("{origin}/object?signature={SECRET}"), 4096),
                    &cancel,
                )
                .err()
                .expect("delayed HEAD must exceed its deadline");
            assert!(start.elapsed() < Duration::from_secs(1));
            assert!(!format!("{error} {error:?}").contains(SECRET));
            worker.join().unwrap();
        }
    }
}
