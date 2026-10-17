//! Discovery uses original provider metadata owners, never payload/HEAD/cache publication.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::{Fixture, admission, assert_no_host_reservation, digest, durable};
use std::collections::{BTreeMap, VecDeque};
use std::io::Cursor;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{
    HttpAccept, HttpRequest, HttpResponse, HttpTransport,
};
use zerogravity_artifacts::acquisition::{
    CandidateCoverage, CandidateRequest, Provider, PullLimits, discover_candidates,
    inspect_cached_candidates,
};
use zerogravity_artifacts::cache::{
    CacheLimits, DigestProvenance, FileRole, Manifest, ManifestEntry, ManifestSource,
};
use zerogravity_artifacts::{Error, Result};
const PIN: &str = "1111111111111111111111111111111111111111";
const TREE: &str = "2222222222222222222222222222222222222222";
const OID: &str = "3333333333333333333333333333333333333333";
const POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1\n";
struct Row {
    url: String,
    bytes: Vec<u8>,
    link: Option<String>,
}
impl Row {
    fn json(url: String, value: serde_json::Value) -> Self {
        Self {
            url,
            bytes: serde_json::to_vec(&value).unwrap(),
            link: None,
        }
    }
}
struct Script {
    rows: Mutex<VecDeque<Row>>,
    calls: AtomicUsize,
}
impl Script {
    fn new(rows: Vec<Row>) -> Self {
        Self {
            rows: Mutex::new(rows.into()),
            calls: AtomicUsize::new(0),
        }
    }
    fn empty(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert!(matches!(
            request.accept,
            HttpAccept::Json | HttpAccept::GitHubJson
        ));
        assert!(request.range.is_none());
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("undeclared metadata/payload request");
        assert_eq!(request.url, row.url);
        let mut headers = BTreeMap::new();
        if let Some(link) = row.link {
            headers.insert("link".into(), link);
        }
        Ok(HttpResponse {
            status: 200,
            headers,
            body: Box::new(Cursor::new(row.bytes)),
        })
    }
    fn head(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("discovery cannot HEAD or transfer a payload")
    }
}
fn request(provider: Provider, subfolder: Option<&str>) -> CandidateRequest {
    CandidateRequest::new(provider, "owner/repo", "main", subfolder).unwrap()
}
fn hf_rows(entries: serde_json::Value) -> Vec<Row> {
    vec![
        Row::json(
            "https://huggingface.co/api/models/owner/repo/revision/main".into(),
            serde_json::json!({"sha":PIN}),
        ),
        Row::json(
            format!(
                "https://huggingface.co/api/models/owner/repo/tree/{PIN}?recursive=true&expand=false&limit=1000"
            ),
            entries,
        ),
    ]
}
fn hf_file(path: &str, size: Option<u64>) -> serde_json::Value {
    let mut row = serde_json::json!({"type":"file","path":path});
    if let Some(size) = size {
        row["size"] = serde_json::json!(size);
    }
    row
}
fn gh_rows(entries: serde_json::Value) -> Vec<Row> {
    vec![
        Row::json(
            "https://api.github.com/repos/owner/repo/commits/main".into(),
            serde_json::json!({"sha":PIN,"commit":{"tree":{"sha":TREE}}}),
        ),
        Row::json(
            format!("https://api.github.com/repos/owner/repo/git/trees/{TREE}?recursive=1"),
            serde_json::json!({"sha":TREE,"truncated":false,"tree":entries}),
        ),
    ]
}
fn gh_file(path: &str, size: Option<u64>, mode: &str) -> serde_json::Value {
    let mut row = serde_json::json!({"type":"blob","mode":mode,"path":path,"sha":OID});
    if let Some(size) = size {
        row["size"] = serde_json::json!(size);
    }
    row
}
#[test]
fn multiple_choices_unknown_zero_and_huge_sizes_need_only_one_pin_and_tree() {
    let memory = admission(128 * 1024 * 1024);
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let entries = match provider {
            Provider::HuggingFace => serde_json::json!([
                hf_file("Q4.gguf", Some(u64::MAX)),
                hf_file("Q8.gguf", Some(u64::MAX)),
                hf_file("unknown.bin", None),
                hf_file("zero", Some(0)),
                hf_file("model.safetensors.index.json", Some(64))
            ]),
            Provider::GitHub => serde_json::json!([
                gh_file("Q4.gguf", Some(u64::MAX), "100644"),
                gh_file("Q8.gguf", Some(u64::MAX), "100755"),
                gh_file("unknown.bin", None, "100644"),
                gh_file("zero", Some(0), "100644"),
                gh_file("model.safetensors.index.json", Some(64), "100644")
            ]),
        };
        let script = Script::new(match provider {
            Provider::HuggingFace => hf_rows(entries),
            Provider::GitHub => gh_rows(entries),
        });
        let limits = PullLimits {
            cache: CacheLimits {
                max_file_bytes: 1,
                max_snapshot_bytes: 1,
                ..CacheLimits::default()
            },
            ..PullLimits::default()
        };
        let result = discover_candidates(
            &script,
            &request(provider, None),
            &limits,
            &memory,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(result.coverage(), CandidateCoverage::PinnedRepositoryTree);
        assert_eq!(result.source().revision(), PIN);
        assert_eq!(result.files().len(), 5);
        assert_eq!(result.files()[0].path(), "Q4.gguf");
        assert_eq!(result.files()[0].bytes(), Some(u64::MAX));
        assert_eq!(
            result
                .files()
                .iter()
                .find(|f| f.path() == "zero")
                .unwrap()
                .bytes(),
            Some(0)
        );
        assert_eq!(
            result
                .files()
                .iter()
                .find(|f| f.path() == "unknown.bin")
                .unwrap()
                .bytes(),
            None
        );
        assert!(result.files().iter().all(|f| f.sha256().is_none())); // Git OID is not SHA256.
        assert_eq!(script.calls.load(Ordering::Relaxed), 2);
        script.empty();
        assert_no_host_reservation(&memory);
    }
}
#[test]
fn hf_page_links_stay_pinned_subfolder_filter_has_component_boundary_and_lfs_fallback() {
    let mut lfs = hf_file("q/a.gguf", None);
    lfs["lfs"] = serde_json::json!({"oid":"A".repeat(64),"size":7});
    let mut rows = hf_rows(serde_json::json!([lfs, hf_file("qq/other.gguf", Some(9))]));
    let next = format!("https://huggingface.co/api/models/owner/repo/tree/{PIN}?cursor=two");
    rows[1].link = Some(format!("<{next}>; rel=\"next\""));
    rows.push(Row::json(
        next,
        serde_json::json!([hf_file("q/nested/b.gguf", Some(8))]),
    ));
    let script = Script::new(rows);
    let memory = admission(128 * 1024 * 1024);
    let result = discover_candidates(
        &script,
        &request(Provider::HuggingFace, Some("q")),
        &PullLimits::default(),
        &memory,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(
        result.files().iter().map(|f| f.path()).collect::<Vec<_>>(),
        ["q/a.gguf", "q/nested/b.gguf"]
    );
    assert_eq!(result.files()[0].bytes(), Some(7));
    assert_eq!(result.files()[0].sha256(), Some("a".repeat(64).as_str()));
    assert_eq!(script.calls.load(Ordering::Relaxed), 3);
    script.empty();
}
#[test]
fn github_nonregular_entries_are_not_download_choices_and_truncation_refuses() {
    let entries = serde_json::json!([gh_file("model.gguf",Some(3),"100644"),gh_file("link.gguf",Some(3),"120000"),
 {"type":"commit","mode":"160000","path":"submodule","sha":OID},
 {"type":"tree","mode":"040000","path":"dir","sha":OID}]);
    let memory = admission(128 * 1024 * 1024);
    let script = Script::new(gh_rows(entries.clone()));
    let result = discover_candidates(
        &script,
        &request(Provider::GitHub, None),
        &PullLimits::default(),
        &memory,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(result.files().len(), 1);
    script.empty();
    let mut rows = gh_rows(entries);
    let mut tree: serde_json::Value = serde_json::from_slice(&rows[1].bytes).unwrap();
    tree["truncated"] = serde_json::json!(true);
    rows[1].bytes = serde_json::to_vec(&tree).unwrap();
    assert!(matches!(
        discover_candidates(
            &Script::new(rows),
            &request(Provider::GitHub, None),
            &PullLimits::default(),
            &memory,
            &AtomicBool::new(false)
        ),
        Err(Error::Unsupported(_))
    ));
}
#[test]
fn logical_request_body_count_and_returned_control_limits_are_cumulative() {
    let memory = admission(128 * 1024 * 1024);
    let rows = || {
        hf_rows(serde_json::json!([
            hf_file("a.gguf", Some(1)),
            hf_file("b.gguf", Some(1))
        ]))
    };
    let page = PullLimits {
        max_pages: 1,
        ..PullLimits::default()
    };
    let script = Script::new(rows());
    assert!(matches!(
        discover_candidates(
            &script,
            &request(Provider::HuggingFace, None),
            &page,
            &memory,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(script.calls.load(Ordering::Relaxed), 1);
    let data = rows();
    let body = PullLimits {
        max_metadata_bytes: data.iter().map(|r| r.bytes.len()).sum::<usize>() - 1,
        ..PullLimits::default()
    };
    assert!(matches!(
        discover_candidates(
            &Script::new(data),
            &request(Provider::HuggingFace, None),
            &body,
            &memory,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    let count = PullLimits {
        cache: CacheLimits {
            max_files: 1,
            ..CacheLimits::default()
        },
        ..PullLimits::default()
    };
    assert!(matches!(
        discover_candidates(
            &Script::new(rows()),
            &request(Provider::HuggingFace, None),
            &count,
            &memory,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    let control = PullLimits {
        cache: CacheLimits {
            max_manifest_bytes: 1,
            ..CacheLimits::default()
        },
        ..PullLimits::default()
    };
    assert!(matches!(
        discover_candidates(
            &Script::new(hf_rows(serde_json::json!([]))),
            &request(Provider::HuggingFace, None),
            &control,
            &memory,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    assert_no_host_reservation(&memory);
}
#[test]
fn unsafe_duplicate_escaping_pages_and_exact_pin_contradiction_refuse() {
    let memory = admission(128 * 1024 * 1024);
    for entries in [
        serde_json::json!([hf_file("../escape.gguf", Some(1))]),
        serde_json::json!([hf_file("a", Some(1)), hf_file("a", Some(1))]),
        serde_json::json!([hf_file("a", Some(1)), hf_file("a/b", Some(1))]),
    ] {
        assert!(
            discover_candidates(
                &Script::new(hf_rows(entries)),
                &request(Provider::HuggingFace, None),
                &PullLimits::default(),
                &memory,
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    for next in [
        "https://untrusted.invalid/tree",
        &format!(
            "https://huggingface.co/api/models/owner/repo/tree/{PIN}?recursive=true&expand=false&limit=1000"
        ),
    ] {
        let mut rows = hf_rows(serde_json::json!([]));
        rows[1].link = Some(format!("<{next}>; rel=\"next\""));
        assert!(
            discover_candidates(
                &Script::new(rows),
                &request(Provider::HuggingFace, None),
                &PullLimits::default(),
                &memory,
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    let exact =
        CandidateRequest::new(Provider::HuggingFace, "owner/repo", &"A".repeat(40), None).unwrap();
    let rows = vec![Row::json(
        format!(
            "https://huggingface.co/api/models/owner/repo/revision/{}",
            "A".repeat(40)
        ),
        serde_json::json!({"sha":PIN}),
    )];
    assert!(matches!(
        discover_candidates(
            &Script::new(rows),
            &exact,
            &PullLimits::default(),
            &memory,
            &AtomicBool::new(false)
        ),
        Err(Error::State(_))
    ));
}
#[test]
fn subfolder_cannot_hide_a_regular_ancestor_collision_in_either_provider() {
    let memory = admission(128 * 1024 * 1024);
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let rows = match provider {
            Provider::HuggingFace => hf_rows(serde_json::json!([
                hf_file("q", Some(3)),
                hf_file("q/a.gguf", Some(4))
            ])),
            Provider::GitHub => gh_rows(serde_json::json!([
                gh_file("q", Some(3), "100644"),
                gh_file("q/a.gguf", Some(4), "100644")
            ])),
        };
        let script = Script::new(rows);
        let error = discover_candidates(
            &script,
            &request(provider, Some("q")),
            &PullLimits::default(),
            &memory,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Malformed(_)), "{error:?}");
        assert!(error.to_string().contains("file/directory collision"));
        script.empty();
    }
    // A real directory ancestor is not a regular file and remains valid.
    let script = Script::new(gh_rows(serde_json::json!([
        {"type":"tree","mode":"040000","path":"q","sha":OID},
        gh_file("q/a.gguf", Some(4), "100644")
    ])));
    let result = discover_candidates(
        &script,
        &request(Provider::GitHub, Some("q")),
        &PullLimits::default(),
        &memory,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(result.files().len(), 1);
    assert_eq!(result.files()[0].path(), "q/a.gguf");
    script.empty();
    assert_no_host_reservation(&memory);
}
#[test]
fn canceled_or_denied_admission_precedes_metadata_and_postcall_cancel_wins() {
    let script = Script::new(vec![]);
    let request = request(Provider::HuggingFace, None);
    assert!(matches!(
        discover_candidates(
            &script,
            &request,
            &PullLimits::default(),
            &admission(1),
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        discover_candidates(
            &script,
            &request,
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(true)
        ),
        Err(Error::State(_))
    ));
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    struct Cancel;
    impl HttpTransport for Cancel {
        fn get(&self, _: &HttpRequest, c: &AtomicBool) -> Result<HttpResponse> {
            c.store(true, Ordering::Release);
            Err(Error::Unsupported("NEVER_ECHO_URL_TOKEN".into()))
        }
    }
    let error = discover_candidates(
        &Cancel,
        &request,
        &PullLimits::default(),
        &admission(128 * 1024 * 1024),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error, Error::State(_)));
    assert!(!format!("{error:?}").contains("NEVER_ECHO"));
}
fn cached(
    provider: &str,
    revision: &str,
    pointer: bool,
) -> (Fixture, zerogravity_artifacts::cache::SnapshotLease) {
    let fixture = Fixture::new();
    let store = fixture.open();
    let data = if pointer { POINTER } else { b"abc" };
    let manifest = Manifest::new(
        ManifestSource::new(provider, "owner/repo", revision).unwrap(),
        vec![
            ManifestEntry::new(
                "q/q4.gguf",
                digest(data),
                data.len() as u64,
                FileRole::Auxiliary,
            )
            .unwrap()
            .with_digest_provenance(DigestProvenance::FirstSeen),
            ManifestEntry::new("q/q8.gguf", digest(b"def"), 3, FileRole::Auxiliary)
                .unwrap()
                .with_digest_provenance(DigestProvenance::TrustedSha256),
        ],
        &Default::default(),
    )
    .unwrap();
    let mut tx = store
        .begin(
            manifest,
            &admission(128 * 1024 * 1024),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    tx.import_object("q/q4.gguf", &mut Cursor::new(data))
        .unwrap();
    tx.import_object("q/q8.gguf", &mut Cursor::new(b"def"))
        .unwrap();
    let snapshot = durable(tx.commit(None).unwrap());
    let lease = store.open_snapshot(&snapshot).unwrap();
    (fixture, lease)
}
#[test]
fn cached_subset_keeps_multiple_choices_without_promoting_firstseen_or_remote_coverage() {
    let (_fixture, lease) = cached("huggingface", PIN, false);
    let result = inspect_cached_candidates(
        &lease,
        &request(Provider::HuggingFace, Some("q")),
        &PullLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(result.coverage(), CandidateCoverage::VerifiedCacheSubset);
    assert_eq!(result.files().len(), 2);
    assert_eq!(result.files()[0].sha256(), None);
    assert_eq!(result.files()[1].sha256(), Some(digest(b"def").as_str()));
    assert_eq!(result.files()[0].role(), FileRole::Weights); // Filename hint, not the generic manifest role.
    let limits = PullLimits {
        cache: CacheLimits {
            max_file_bytes: 2,
            ..CacheLimits::default()
        },
        ..PullLimits::default()
    };
    assert!(matches!(
        inspect_cached_candidates(
            &lease,
            &request(Provider::HuggingFace, None),
            &limits,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
}
#[test]
fn cached_foreign_source_release_locator_pointers_and_cancellation_refuse() {
    for (provider, revision, pointer) in [
        ("github", PIN, false),
        (
            "huggingface",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            false,
        ),
        ("huggingface", "release-1-asset-2", false),
        ("huggingface", PIN, true),
    ] {
        let (_fixture, lease) = cached(provider, revision, pointer);
        let request =
            CandidateRequest::new(Provider::HuggingFace, "owner/repo", PIN, None).unwrap();
        assert!(
            inspect_cached_candidates(
                &lease,
                &request,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    let (_fixture, lease) = cached("huggingface", PIN, false);
    assert!(matches!(
        inspect_cached_candidates(
            &lease,
            &request(Provider::HuggingFace, None),
            &PullLimits::default(),
            &AtomicBool::new(true)
        ),
        Err(Error::State(_))
    ));
}
#[test]
fn request_validation_is_bounded_and_does_not_need_a_fake_file_selection() {
    for (repo, revision, folder) in [
        ("https://user:secret@host/repo", "main", None),
        ("owner/repo", "https://secret.invalid", None),
        ("owner/repo", "main", Some("../escape")),
    ] {
        assert!(CandidateRequest::new(Provider::HuggingFace, repo, revision, folder).is_err());
    }
    assert!(CandidateRequest::new(Provider::GitHub, "owner/repo", "tag", Some("q")).is_ok());
    assert!(
        CandidateRequest::new(Provider::HuggingFace, "owner/repo", &"r".repeat(257), None).is_err()
    );
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn injected_mapping_uses_original_http_auth_owner_for_only_two_metadata_requests() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let stopping = stop.clone();
    let expected = hf_rows(serde_json::json!([
        hf_file("a.gguf", Some(3)),
        hf_file("b.gguf", Some(4))
    ]));
    let worker = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(5);
        let mut count = 0;
        while count < 2 && !stopping.load(Ordering::Acquire) {
            assert!(Instant::now() < until, "owned metadata fixture deadline");
            let (mut stream, _) = match listener.accept() {
                Ok(value) => value,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                Err(e) => panic!("{e}"),
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut head = Vec::new();
            while !head.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                head.push(byte[0]);
                assert!(head.len() <= 8192);
            }
            let wire = String::from_utf8(head).unwrap().to_ascii_lowercase();
            assert!(wire.starts_with("get /api/models/"));
            assert!(wire.contains("authorization: bearer fixture-secret"));
            let body = &expected[count].bytes;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
            count += 1;
        }
        count
    });
    struct Owned {
        stop: Arc<AtomicBool>,
        worker: Option<std::thread::JoinHandle<usize>>,
    }
    impl Drop for Owned {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(worker) = self.worker.take() {
                let joined = worker.join();
                if !std::thread::panicking() {
                    joined.unwrap();
                }
            }
        }
    }
    let mut owned = Owned {
        stop,
        worker: Some(worker),
    };
    struct Mapping {
        origin: String,
        client: HttpsClient,
    }
    impl HttpTransport for Mapping {
        fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
            let suffix = request
                .url
                .strip_prefix("https://huggingface.co")
                .expect("no other provider/payload authority");
            let mut mapped =
                HttpRequest::new(format!("{}{suffix}", self.origin), request.max_body_bytes);
            mapped.accept = request.accept;
            self.client.get(&mapped, cancel)
        }
    }
    let transport = Mapping {
        client: HttpsClient::diagnostic_loopback(
            &origin,
            Some("fixture-secret".into()),
            HttpOptions::default(),
        )
        .unwrap(),
        origin,
    };
    let result = discover_candidates(
        &transport,
        &request(Provider::HuggingFace, None),
        &PullLimits::default(),
        &admission(128 * 1024 * 1024),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(result.files().len(), 2);
    assert!(!format!("{result:?}").contains("fixture-secret"));
    assert_eq!(owned.worker.take().unwrap().join().unwrap(), 2);
}
