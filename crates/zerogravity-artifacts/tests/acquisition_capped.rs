//! Explicit capped auxiliary acquisition, with tiny admitted synthetic data.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;

use cache_transaction_support::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Cursor, Read};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use zerogravity_artifacts::acquisition::{
    ByteLength, ComponentKind, ComponentPullRequest, ComponentRequest, ComponentTransport,
    DigestProvenance, Provider, PullLimits, PullRequest, ResolvedFile, ResolvedPullPlan, Selection,
    execute, execute_with_progress, resolve, resolve_components,
};
use zerogravity_artifacts::cache::{FileRole, ManifestSource};
use zerogravity_artifacts::{Error, Result};

const PIN: &str = "1111111111111111111111111111111111111111";

struct Row {
    head: bool,
    path: String,
    response: HttpResponse,
}
struct Script(Mutex<VecDeque<Row>>);
impl Script {
    fn new(rows: Vec<Row>) -> Self {
        Self(Mutex::new(rows.into()))
    }
    fn take(&self, request: &HttpRequest, head: bool) -> Result<HttpResponse> {
        let row = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request or range retry");
        assert_eq!(head, row.head);
        assert!(
            request.url.contains(&row.path),
            "unexpected pinned object path"
        );
        assert!(request.range.is_none(), "capped requests never resume");
        Ok(row.response)
    }
    fn finished(&self) {
        assert!(self.0.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        self.take(request, false)
    }
    fn head(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        self.take(request, true)
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
fn response(
    status: u16,
    headers: &[(&str, &str)],
    body: impl Read + Send + 'static,
) -> HttpResponse {
    HttpResponse {
        status,
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<BTreeMap<_, _>>(),
        body: Box::new(body),
    }
}
fn get(path: &str, headers: &[(&str, &str)], bytes: &[u8]) -> Row {
    Row {
        head: false,
        path: path.into(),
        response: response(200, headers, Cursor::new(bytes.to_vec())),
    }
}
fn metadata(path: &str, value: Value) -> Row {
    get(path, &[], &serde_json::to_vec(&value).unwrap())
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("refused or HEAD response body must not be read")
    }
}
fn head(status: u16, headers: &[(&str, &str)]) -> Row {
    Row {
        head: true,
        path: "/note.txt".into(),
        response: response(status, headers, NoRead),
    }
}
fn request(provider: Provider, path: &str) -> PullRequest {
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
fn hf_rows(entries: Value) -> Vec<Row> {
    vec![
        metadata("/revision/", json!({"sha":PIN})),
        metadata("/tree/", entries),
    ]
}
fn hf_note() -> Vec<Row> {
    hf_rows(json!([{"type":"file","path":"note.txt"}]))
}
fn limits(cap: u64) -> PullLimits {
    PullLimits {
        unknown_length_cap: Some(cap),
        ..Default::default()
    }
}
fn plan(cap: u64, hash: Option<String>) -> ResolvedPullPlan {
    ResolvedPullPlan::new(
        ManifestSource::new("github", "fixture/repo", PIN).unwrap(),
        vec![
            ResolvedFile::new_with_length(
                "note.txt",
                "https://example.invalid/note.txt",
                FileRole::Auxiliary,
                ByteLength::AtMost(cap),
                hash,
                "pinned-note",
            )
            .unwrap(),
        ],
        &limits(cap),
    )
    .unwrap()
}

#[test]
fn empty_short_exact_cap_and_declared_get_lengths_publish_actual_counts_with_unknown_progress() {
    for (bytes, cap, declared) in [
        (b"".as_slice(), 8, false),
        (b"abc".as_slice(), 32, false),
        (b"eight123".as_slice(), 8, false),
        (b"abc".as_slice(), 32, true),
    ] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let plan = plan(cap, None);
        assert_eq!(plan.total_bytes(), None);
        assert_eq!(plan.maximum_bytes(), cap);
        let length = bytes.len().to_string();
        let headers = if declared {
            vec![("content-length", length.as_str())]
        } else {
            vec![("transfer-encoding", "chunked")]
        };
        let script = Script::new(vec![get("/note.txt", &headers, bytes)]);
        let mut events = Vec::new();
        let receipt = execute_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
            &mut |event| {
                assert_eq!(event.file_total_bytes(), None);
                assert_eq!(event.total_expected_bytes(), None);
                assert_eq!(event.file_maximum_bytes(), cap);
                assert_eq!(event.total_maximum_bytes(), cap);
                assert_eq!((event.cached_bytes(), event.resumed_bytes()), (0, 0));
                assert!(event.total_bytes() <= cap);
                events.push((event.file_bytes(), event.file_complete()));
            },
        )
        .unwrap();
        assert_eq!(receipt.files()[0].bytes(), bytes.len() as u64);
        assert_eq!(receipt.files()[0].provenance(), DigestProvenance::FirstSeen);
        assert_eq!(receipt.manifest().entries()[0].bytes(), bytes.len() as u64);
        assert_eq!(events.last(), Some(&(bytes.len() as u64, true)));
        assert!(
            events[..events.len() - 1]
                .iter()
                .all(|(_, complete)| !complete)
        );
        assert_eq!(
            store
                .resolve_ref("fixture")
                .unwrap()
                .read_verified("note.txt", cap)
                .unwrap(),
            bytes
        );
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
        script.finished();
    }
}

#[test]
fn capped_framing_length_digest_pointer_and_oversize_failures_preserve_previous_reference() {
    for case in 0..9 {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let hash = (case == 7).then(|| digest(b"different"));
        let cap = if case == 8 { 128 } else { 8 };
        let row = match case {
            0 => get("/note.txt", &[("content-length", "4")], b"abc"),
            1 => get(
                "/note.txt",
                &[("transfer-encoding", "chunked")],
                b"123456789",
            ),
            2 => Row {
                head: false,
                path: "/note.txt".into(),
                response: response(200, &[], NoRead),
            },
            3 => Row {
                head: false,
                path: "/note.txt".into(),
                response: response(200, &[("content-length", "9")], NoRead),
            },
            4 => Row {
                head: false,
                path: "/note.txt".into(),
                response: response(
                    200,
                    &[("content-length", "3"), ("transfer-encoding", "chunked")],
                    NoRead,
                ),
            },
            5 => Row {
                head: false,
                path: "/note.txt".into(),
                response: response(200, &[("transfer-encoding", "gzip, chunked")], NoRead),
            },
            6 => Row {
                head: false,
                path: "/note.txt".into(),
                response: response(
                    200,
                    &[
                        ("transfer-encoding", "chunked"),
                        ("content-range", "bytes 0-2/3"),
                    ],
                    NoRead,
                ),
            },
            7 => get("/note.txt", &[("content-length", "3")], b"abc"),
            _ => get(
                "/note.txt",
                &[("transfer-encoding", "chunked")],
                b"version https://git-lfs.github.com/spec/v1\noid sha256:unqualified\n",
            ),
        };
        let script = Script::new(vec![row]);
        assert!(
            execute(
                &store,
                &plan(cap, hash),
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                Some("fixture")
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
            old
        );
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert!(
            std::fs::read_dir(fixture.root.join("staging"))
                .unwrap()
                .next()
                .is_none()
        );
        assert_no_host_reservation(&admission);
        script.finished();
    }
}

struct Interrupted(Cursor<Vec<u8>>);
impl Read for Interrupted {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.0.position() == self.0.get_ref().len() as u64 {
            return Err(io::Error::other("secret upstream failure"));
        }
        self.0.read(output)
    }
}
#[test]
fn capped_read_error_has_no_range_retry_checkpoint_or_error_echo() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let script = Script::new(vec![Row {
        head: false,
        path: "/note.txt".into(),
        response: response(
            200,
            &[("transfer-encoding", "chunked"), ("etag", "\"strong\"")],
            Interrupted(Cursor::new(vec![7; 200])),
        ),
    }]);
    let error = execute(
        &store,
        &plan(512, None),
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap_err();
    assert!(matches!(error, Error::State(_)));
    assert!(!error.to_string().contains("secret"));
    script.finished();
    assert!(
        std::fs::read_dir(fixture.root.join("staging"))
            .unwrap()
            .next()
            .is_none()
    );
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
}

struct CountReads {
    body: Cursor<Vec<u8>>,
    calls: Arc<AtomicUsize>,
}
impl Read for CountReads {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.body.read(output)
    }
}
#[test]
fn cancellation_at_cap_prevents_terminal_read_and_next_invocation_starts_fresh() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![Row {
        head: false,
        path: "/note.txt".into(),
        response: response(
            200,
            &[("transfer-encoding", "chunked")],
            CountReads {
                body: Cursor::new(b"abcd".to_vec()),
                calls: calls.clone(),
            },
        ),
    }]);
    let mut completed = false;
    assert!(
        execute_with_progress(
            &store,
            &plan(4, None),
            &script,
            &admission,
            &cancel,
            Some("fixture"),
            &mut |event| {
                completed |= event.file_complete();
                if event.file_bytes() == 4 {
                    cancel.store(true, Ordering::Release);
                }
            }
        )
        .is_err()
    );
    assert!(!completed);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    script.finished();
    assert_eq!(
        store
            .resolve_ref("fixture")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old
    );
    assert!(
        std::fs::read_dir(fixture.root.join("staging"))
            .unwrap()
            .next()
            .is_none()
    );
    let fresh = Script::new(vec![get(
        "/note.txt",
        &[("transfer-encoding", "chunked")],
        b"abcd",
    )]);
    execute(
        &store,
        &plan(4, None),
        &fresh,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
    )
    .unwrap();
    fresh.finished();
}

#[test]
fn trusted_cached_unknown_length_reports_actual_cached_bytes_without_get() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    publish(&store, &admission, OLD, None).unwrap();
    let script = Script::new(Vec::new());
    let mut last = None;
    let receipt = execute_with_progress(
        &store,
        &plan(256, Some(digest(OLD))),
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
        &mut |event| {
            last = Some((
                event.file_bytes(),
                event.cached_bytes(),
                event.downloaded_bytes(),
                event.file_complete(),
            ));
        },
    )
    .unwrap();
    assert_eq!(last, Some((OLD.len() as u64, OLD.len() as u64, 0, true)));
    assert_eq!(
        receipt.files()[0].provenance(),
        DigestProvenance::TrustedSha256
    );
    script.finished();
}

#[test]
fn explicit_auxiliary_head_unavailability_is_narrow_and_body_free() {
    let admission = admission(256 * 1024 * 1024);
    for (status, headers) in [
        (200, vec![]),
        (200, vec![("transfer-encoding", "chunked")]),
        (
            405,
            vec![("content-length", "9999"), ("etag", "\"error-page\"")],
        ),
        (501, vec![]),
    ] {
        let mut rows = hf_note();
        rows.push(head(status, &headers));
        let script = Script::new(rows);
        let mut bounds = limits(16);
        bounds.cache.max_file_bytes = 32;
        let plan = resolve(
            &script,
            &request(Provider::HuggingFace, "note.txt"),
            &bounds,
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(plan.files()[0].length(), ByteLength::AtMost(16));
        assert_eq!(plan.total_bytes(), None);
        script.finished();
    }
    for (status, headers) in [
        (401, vec![]),
        (503, vec![]),
        (200, vec![("content-length", "bad")]),
        (
            200,
            vec![("content-length", "3"), ("transfer-encoding", "chunked")],
        ),
        (200, vec![("content-encoding", "gzip")]),
        (200, vec![("x-repo-commit", "changed")]),
        (200, vec![("x-linked-size", "3")]),
    ] {
        let mut rows = hf_note();
        rows.push(head(status, &headers));
        let script = Script::new(rows);
        assert!(
            resolve(
                &script,
                &request(Provider::HuggingFace, "note.txt"),
                &limits(16),
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        script.finished();
    }
    let mut rows = hf_note();
    rows.push(head(200, &[]));
    let script = Script::new(rows);
    assert!(
        resolve(
            &script,
            &request(Provider::HuggingFace, "note.txt"),
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    script.finished();
}

#[test]
fn github_raw_and_release_auxiliary_files_keep_exact_pins_and_no_oid_hash_upgrade() {
    let admission = admission(256 * 1024 * 1024);
    for release in [false, true] {
        let mut req = request(Provider::GitHub, "note.txt");
        let mut rows = if release {
            req.selection = Selection::ReleaseAsset {
                name: "note.txt".into(),
            };
            vec![
                metadata("/releases/tags/", json!({"id":9,"tag_name":"main"})),
                metadata(
                    "/releases/9/assets",
                    json!([{"id":7,"name":"note.txt","state":"uploaded","digest":null}]),
                ),
            ]
        } else {
            vec![
                metadata(
                    "/commits/",
                    json!({"sha":PIN,"commit":{"tree":{"sha":PIN}}}),
                ),
                metadata(
                    "/git/trees/",
                    json!({"sha":PIN,"truncated":false,"tree":[{"path":"note.txt","mode":"100644","type":"blob","sha":PIN}]}),
                ),
            ]
        };
        rows.push(Row {
            head: true,
            path: if release {
                "/releases/assets/7".into()
            } else {
                format!("/{PIN}/note.txt")
            },
            response: response(200, &[], NoRead),
        });
        let script = Script::new(rows);
        let plan = resolve(
            &script,
            &req,
            &limits(32),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(plan.files()[0].bytes(), None);
        assert_eq!(plan.files()[0].sha256(), None);
        assert_eq!(
            plan.source().revision(),
            if release { "release-9-asset-7" } else { PIN }
        );
        script.finished();
    }
}

#[test]
fn nonauxiliary_zero_cap_and_aggregate_cap_plans_refuse() {
    for role in [
        FileRole::Weights,
        FileRole::Config,
        FileRole::Tokenizer,
        FileRole::License,
        FileRole::Index,
    ] {
        assert!(matches!(
            ResolvedFile::new_with_length(
                "entry",
                "https://example.invalid/entry",
                role,
                ByteLength::AtMost(8),
                None,
                "pinned"
            ),
            Err(Error::Unsupported(_))
        ));
    }
    assert!(
        ResolvedFile::new_with_length(
            "entry",
            "https://example.invalid/entry",
            FileRole::Auxiliary,
            ByteLength::AtMost(0),
            None,
            "pinned"
        )
        .is_err()
    );
    assert!(limits(0).validate().is_err());
    let mut bounds = limits(8);
    bounds.cache.max_snapshot_bytes = 15;
    bounds.cache.max_file_bytes = 8;
    let files = ["a", "b"]
        .into_iter()
        .map(|p| {
            ResolvedFile::new_with_length(
                p,
                format!("https://example.invalid/{p}"),
                FileRole::Auxiliary,
                ByteLength::AtMost(8),
                None,
                p,
            )
            .unwrap()
        })
        .collect();
    assert!(
        ResolvedPullPlan::new(
            ManifestSource::new("github", "fixture/repo", PIN).unwrap(),
            files,
            &bounds
        )
        .is_err()
    );
}

struct CountBytes {
    body: Cursor<Vec<u8>>,
    read: Arc<AtomicUsize>,
}
impl Read for CountBytes {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let read = self.body.read(output)?;
        self.read.fetch_add(read, Ordering::Relaxed);
        Ok(read)
    }
}

#[test]
fn custom_transport_reads_at_most_one_sentinel_byte_beyond_cap_or_declared_length() {
    for declared in [false, true] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let read = Arc::new(AtomicUsize::new(0));
        let headers = if declared {
            vec![("content-length", "5")]
        } else {
            vec![("transfer-encoding", "chunked")]
        };
        let script = Script::new(vec![Row {
            head: false,
            path: "/note.txt".into(),
            response: response(
                200,
                &headers,
                CountBytes {
                    body: Cursor::new(vec![1; 1024]),
                    read: read.clone(),
                },
            ),
        }]);
        assert!(
            execute(
                &store,
                &plan(8, None),
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                None
            )
            .is_err()
        );
        assert_eq!(read.load(Ordering::Relaxed), if declared { 6 } else { 9 });
        script.finished();
    }
}

#[test]
fn unknown_plan_preserves_strong_head_validator_and_ignores_error_page_etag() {
    for tag in [None, Some("\"changed\"")] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let mut rows = hf_note();
        rows.push(head(200, &[("etag", "\"strong\"")]));
        let mut headers = vec![("transfer-encoding", "chunked")];
        if let Some(tag) = tag {
            headers.push(("etag", tag));
        }
        rows.push(Row {
            head: false,
            path: "/note.txt".into(),
            response: response(200, &headers, NoRead),
        });
        let script = Script::new(rows);
        let plan = resolve(
            &script,
            &request(Provider::HuggingFace, "note.txt"),
            &limits(32),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(matches!(
            execute(
                &store,
                &plan,
                &script,
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
            old
        );
        script.finished();
    }
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let mut rows = hf_note();
    rows.push(head(
        405,
        &[("etag", "\"error\""), ("content-length", "9999")],
    ));
    rows.push(get(
        "/note.txt",
        &[("transfer-encoding", "chunked")],
        b"abc",
    ));
    let script = Script::new(rows);
    let plan = resolve(
        &script,
        &request(Provider::HuggingFace, "note.txt"),
        &limits(32),
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    let receipt = execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap();
    assert_eq!(receipt.files()[0].bytes(), 3);
    script.finished();
}

#[test]
fn unknown_model_sidecar_license_and_index_lengths_stay_strict_despite_opt_in() {
    let admission = admission(256 * 1024 * 1024);
    for path in [
        "model.gguf",
        "config.json",
        "tokenizer.json",
        "LICENSE",
        "model.safetensors.index.json",
    ] {
        let mut rows = hf_rows(json!([
            {"type":"file","path":path},{"type":"file","path":"unselected.txt"},
            {"type":"file","path":"other/config.json","size":2},
            {"type":"file","path":"other/tokenizer.json","size":2}
        ]));
        rows.push(Row {
            head: true,
            path: format!("/{path}"),
            response: response(200, &[], NoRead),
        });
        let script = Script::new(rows);
        assert!(
            resolve(
                &script,
                &request(Provider::HuggingFace, path),
                &limits(32),
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(script.0.lock().unwrap().len() <= 1);
    }
}

#[test]
fn components_sum_capped_auxiliary_worst_case_under_existing_aggregate_budget() {
    let admission = admission(256 * 1024 * 1024);
    let components = vec![
        ComponentRequest::new(
            "weights",
            ComponentKind::Weights,
            request(Provider::HuggingFace, "model.gguf"),
        )
        .unwrap(),
        ComponentRequest::new(
            "config",
            ComponentKind::Config,
            request(Provider::HuggingFace, "config.json"),
        )
        .unwrap(),
        ComponentRequest::new(
            "tokenizer",
            ComponentKind::Tokenizer,
            request(Provider::HuggingFace, "tokenizer.json"),
        )
        .unwrap(),
        ComponentRequest::new(
            "note",
            ComponentKind::Auxiliary,
            request(Provider::HuggingFace, "note.txt"),
        )
        .unwrap(),
    ];
    for maximum in [100, 128] {
        let mut bounds = limits(100);
        bounds.cache.max_snapshot_bytes = maximum;
        bounds.cache.max_file_bytes = 100;
        let req = ComponentPullRequest::new(components.clone(), &bounds).unwrap();
        let mut rows = Vec::new();
        for _ in 0..4 {
            rows.extend(hf_rows(json!([
            {"type":"file","path":"model.gguf","size":3},{"type":"file","path":"config.json","size":2},
            {"type":"file","path":"tokenizer.json","size":2},{"type":"file","path":"note.txt"}
        ])));
        }
        rows.push(head(200, &[]));
        let script = Script::new(rows);
        let result =
            resolve_components(&req, &script, &bounds, &admission, &AtomicBool::new(false));
        if maximum == 100 {
            assert!(matches!(result, Err(Error::Limit(_))));
        } else {
            let plan = result.unwrap();
            assert_eq!(plan.total_bytes(), None);
            assert_eq!(plan.maximum_bytes(), 111);
        }
        script.finished();
        assert_no_host_reservation(&admission);
    }
}
