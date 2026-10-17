//! Filename choice under the original pin/metadata/closure/publisher owners.
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
use zerogravity_artifacts::acquisition::http::{
    HttpRequest, HttpResponse, HttpTarget, HttpTransport,
};
use zerogravity_artifacts::acquisition::{
    self, CandidateRequest, GgufQuantRequest, Provider, PullLimits, PullRequest, ResolvedPullPlan,
    Selection, quant, xet,
};
use zerogravity_artifacts::cache::{
    CacheLimits, DigestProvenance, FileRole, Manifest, ManifestEntry, ManifestSource, SnapshotLease,
};
use zerogravity_artifacts::{Error, Result};
const PIN: &str = "1111111111111111111111111111111111111111";
const TREE: &str = "2222222222222222222222222222222222222222";
const OID: &str = "3333333333333333333333333333333333333333";
const XET: &str = "55b9df318a7c45f470d57b04d8ca79a15b0637ffe94b25041a58d00c3528afff";
fn request(provider: Provider, tag: &str, folder: Option<&str>) -> GgufQuantRequest {
    GgufQuantRequest::new(
        CandidateRequest::new(provider, "owner/repo", "main", folder).unwrap(),
        tag,
    )
    .unwrap()
}
#[derive(Clone)]
struct Row {
    method: &'static str,
    url: String,
    status: u16,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}
impl Row {
    fn json(url: String, value: serde_json::Value) -> Self {
        let body = serde_json::to_vec(&value).unwrap();
        Self {
            method: "GET",
            url,
            status: 200,
            headers: BTreeMap::new(),
            body,
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
    fn response(&self, method: &str, req: &HttpRequest) -> Result<HttpResponse> {
        let r = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("undeclared request/relist/payload");
        assert_eq!(method, r.method);
        assert_eq!(req.url, r.url);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(HttpResponse {
            status: r.status,
            headers: r.headers,
            body: Box::new(Cursor::new(r.body)),
        })
    }
}
impl HttpTransport for Script {
    fn get(&self, r: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        self.response("GET", r)
    }
    fn head(&self, r: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        self.response("HEAD", r)
    }
    fn get_target(
        &self,
        r: &HttpRequest,
        target: &HttpTarget,
        c: &AtomicBool,
    ) -> Result<HttpResponse> {
        target.begin_attempt(c)?;
        let response = self.response("GET", r)?;
        target.charge_metadata(64)?;
        struct Charged {
            body: Box<dyn Read + Send>,
            target: HttpTarget,
        }
        impl Read for Charged {
            fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
                let n = self
                    .target
                    .body_read_limit(out.len())
                    .map_err(io::Error::other)?;
                let got = self.body.read(&mut out[..n])?;
                self.target
                    .charge_body(got as u64)
                    .map_err(io::Error::other)?;
                Ok(got)
            }
        }
        Ok(HttpResponse {
            status: response.status,
            headers: response.headers,
            body: Box::new(Charged {
                body: response.body,
                target: target.clone(),
            }),
        })
    }
}
fn rows(provider: Provider, paths: &[&str]) -> Vec<Row> {
    match provider {
        Provider::HuggingFace => vec![
            Row::json(
                "https://huggingface.co/api/models/owner/repo/revision/main".into(),
                serde_json::json!({"sha":PIN}),
            ),
            Row::json(
                format!(
                    "https://huggingface.co/api/models/owner/repo/tree/{PIN}?recursive=true&expand=false&limit=1000"
                ),
                serde_json::json!(
                    paths
                        .iter()
                        .map(|p| serde_json::json!({"type":"file","path":p,"size":3}))
                        .collect::<Vec<_>>()
                ),
            ),
        ],
        Provider::GitHub => vec![
            Row::json(
                "https://api.github.com/repos/owner/repo/commits/main".into(),
                serde_json::json!({"sha":PIN,"commit":{"tree":{"sha":TREE}}}),
            ),
            Row::json(
                format!("https://api.github.com/repos/owner/repo/git/trees/{TREE}?recursive=1"),
                serde_json::json!({"sha":TREE,"truncated":false,"tree":paths.iter().map(|p|serde_json::json!({"type":"blob","mode":"100644","path":p,"sha":OID,"size":3})).collect::<Vec<_>>()}),
            ),
        ],
    }
}
fn resolve(provider: Provider, tag: &str, paths: &[&str]) -> Result<ResolvedPullPlan> {
    let script = Script::new(rows(provider, paths));
    quant::resolve(
        &script,
        &request(provider, tag, None),
        &PullLimits::default(),
        &admission(128 * 1024 * 1024),
        &AtomicBool::new(false),
    )
}
#[test]
fn tag_grammar_is_bounded_ascii_label_not_a_numeric_whitelist() {
    for tag in ["Q4_K_M", "q8_0", "IQ4_XS", "F16", "BF16", "Future_123"] {
        assert!(
            GgufQuantRequest::new(
                CandidateRequest::new(Provider::HuggingFace, "owner/repo", "main", None).unwrap(),
                tag
            )
            .is_ok()
        );
    }
    for tag in [
        "", "4Q", "Q-4", "Q.4", "Q*", " Q4", "Q4 ", "Q4/xx", "Q٤", "Ｑ4", "M\n",
    ] {
        assert!(
            GgufQuantRequest::new(
                CandidateRequest::new(Provider::HuggingFace, "owner/repo", "main", None).unwrap(),
                tag
            )
            .is_err()
        );
    }
    assert!(
        GgufQuantRequest::new(
            CandidateRequest::new(Provider::HuggingFace, "owner/repo", "main", None).unwrap(),
            &"Q".repeat(33)
        )
        .is_err()
    );
    assert_eq!(request(Provider::GitHub, "q4_k_m", None).quant(), "Q4_K_M");
}
#[test]
fn terminal_tag_start_dash_dot_and_unicode_prefix_match_only_tag_case() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        for path in ["Q4_K_M.gguf", "model-q4_k_m.gguf", "模型.Q4_K_M.gguf"] {
            let plan =
                resolve(provider, "q4_k_m", &[path, "config.json", "tokenizer.json"]).unwrap();
            assert!(plan.files().iter().any(|f| f.path() == path));
        }
        for path in [
            "model-Q4_K_MX.gguf",
            "model-Q4_K_M-extra.gguf",
            "model_Q4_K_M.gguf",
            "Q4_K_M.GGUF",
            "Q4_K_M/model.gguf",
        ] {
            assert!(resolve(provider, "Q4_K_M", &[path, "config.json", "tokenizer.json"]).is_err());
        }
        assert!(
            resolve(
                provider,
                "Q4",
                &["model-Q4_K_M.gguf", "config.json", "tokenizer.json"]
            )
            .is_err()
        );
        assert!(
            resolve(
                provider,
                "M",
                &["model-Q4_K_M.gguf", "config.json", "tokenizer.json"]
            )
            .is_err()
        );
    }
}
#[test]
fn unrelated_directory_tag_or_split_text_never_changes_basename_quant_choice() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let plan = resolve(
            provider,
            "Q4_K_M",
            &[
                "model-Q4_K_M.gguf",
                "archive-Q4_K_M-of-folder/other.gguf",
                "archive-Q4_K_M-0001-of-00002/other.gguf",
                "config.json",
                "tokenizer.json",
            ],
        )
        .unwrap();
        assert_eq!(
            plan.files()
                .iter()
                .filter(|f| f.role() == FileRole::Weights)
                .map(|f| f.path())
                .collect::<Vec<_>>(),
            ["model-Q4_K_M.gguf"]
        );
        // The same malformed pattern in an actual matching basename refuses,
        // even when another complete family could otherwise be selected.
        assert!(
            resolve(
                provider,
                "Q4_K_M",
                &[
                    "model-Q4_K_M.gguf",
                    "archive/noise-Q4_K_M-0001-of-00002.gguf",
                    "config.json",
                    "tokenizer.json",
                ]
            )
            .is_err()
        );
    }
}
#[test]
fn complete_split_uses_one_pin_and_has_identical_original_explicit_plan() {
    let paths = [
        "q/model-Q4_K_M-00001-of-00002.gguf",
        "q/model-Q4_K_M-00002-of-00002.gguf",
        "q/model-Q8_0.gguf",
        "q/config.json",
        "q/tokenizer.json",
    ];
    let memory = admission(128 * 1024 * 1024);
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let script = Script::new(rows(provider, &paths));
        let result = quant::resolve(
            &script,
            &request(provider, "Q4_K_M", Some("q")),
            &PullLimits::default(),
            &memory,
            &AtomicBool::new(false),
        )
        .unwrap();
        script.empty();
        assert_eq!(script.calls.load(Ordering::Relaxed), 2);
        assert_eq!(result.source().revision(), PIN);
        assert_eq!(result.files().len(), 4);
        let explicit = PullRequest {
            provider,
            repository: "owner/repo".into(),
            revision: "main".into(),
            selection: Selection::Files {
                paths: vec![paths[0].to_owned()],
                subfolder: None,
            },
        };
        let script = Script::new(rows(provider, &paths));
        let old = acquisition::resolve(
            &script,
            &explicit,
            &PullLimits::default(),
            &memory,
            &AtomicBool::new(false),
        )
        .unwrap();
        script.empty();
        assert_eq!(result.source(), old.source());
        for (a, b) in result.files().iter().zip(old.files()) {
            assert_eq!(a.path(), b.path());
            assert_eq!(a.url(), b.url());
            assert_eq!(a.bytes(), b.bytes());
            assert_eq!(a.sha256(), b.sha256());
            assert_eq!(a.identity(), b.identity());
            assert_eq!(a.role(), b.role());
        }
    }
    assert_no_host_reservation(&memory);
}
#[test]
fn missing_malformed_conflicting_or_ambiguous_matching_cohorts_never_fallback() {
    let cohorts = vec![
        vec!["a-Q4_K_M-00001-of-00002.gguf"],
        vec![
            "a-Q4_K_M-00002-of-00003.gguf",
            "a-Q4_K_M-00003-of-00003.gguf",
        ],
        vec![
            "a-Q4_K_M-00001-of-00002.gguf",
            "a-Q4_K_M-00002-of-00003.gguf",
        ],
        vec!["a-Q4_K_M-00000-of-00002.gguf"],
        vec!["a-Q4_K_M-00001-of-00000.gguf"],
        vec!["a-Q4_K_M-of-00002.gguf", "b-Q4_K_M.gguf"],
        vec!["a-Q4_K_M-00003-of-00002.gguf"],
        vec!["a-Q4_K_M-0001-of-00002.gguf"],
        vec!["a-Q4_K_M-00001-of-abcde.gguf"],
        vec!["a-Q4_K_M.gguf", "b-Q4_K_M.gguf"],
        vec!["a-Q4_K_M.gguf", "a-q4_k_m.gguf"],
        vec!["a-Q4_K_M.gguf", "a-Q4_K_M-00001-of-00001.gguf"],
        vec!["a-Q4_K_M.gguf", "b-Q4_K_M-00001-of-00002.gguf"],
        vec!["a-Q8_0.gguf"],
    ];
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        for mut c in cohorts.clone() {
            c.extend(["config.json", "tokenizer.json"]);
            assert!(resolve(provider, "Q4_K_M", &c).is_err(), "{c:?}");
        }
    }
}
#[test]
fn source_inventory_collision_subfolder_boundary_and_required_sidecars_are_strict() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        for paths in [
            vec!["q", "q/a-Q4_K_M.gguf", "q/config.json", "q/tokenizer.json"],
            vec!["q/a-Q4_K_M.gguf", "q/config.json"],
        ] {
            let script = Script::new(rows(provider, &paths));
            assert!(
                quant::resolve(
                    &script,
                    &request(provider, "Q4_K_M", Some("q")),
                    &PullLimits::default(),
                    &admission(128 * 1024 * 1024),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
        }
        let script = Script::new(rows(
            provider,
            &["qq/a-Q4_K_M.gguf", "qq/config.json", "qq/tokenizer.json"],
        ));
        assert!(
            quant::resolve(
                &script,
                &request(provider, "Q4_K_M", Some("q")),
                &PullLimits::default(),
                &admission(128 * 1024 * 1024),
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    let mut r = rows(
        Provider::GitHub,
        &["a-Q4_K_M.gguf", "config.json", "tokenizer.json"],
    );
    let mut tree: serde_json::Value = serde_json::from_slice(&r[1].body).unwrap();
    tree["tree"][0]["mode"] = serde_json::json!("120000");
    r[1].body = serde_json::to_vec(&tree).unwrap();
    assert!(
        quant::resolve(
            &Script::new(r),
            &request(Provider::GitHub, "Q4_K_M", None),
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false)
        )
        .is_err()
    );
}
#[test]
fn shared_logical_budget_includes_exact_length_heads_without_repin() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let mut r = rows(
            provider,
            &["a-Q4_K_M.gguf", "config.json", "tokenizer.json"],
        );
        let mut tree: serde_json::Value = serde_json::from_slice(&r[1].body).unwrap();
        match provider {
            Provider::HuggingFace => {
                tree[0].as_object_mut().unwrap().remove("size");
            }
            Provider::GitHub => {
                tree["tree"][0].as_object_mut().unwrap().remove("size");
            }
        }
        r[1].body = serde_json::to_vec(&tree).unwrap();
        r.push(Row {
            method: "HEAD",
            url: match provider {
                Provider::HuggingFace => {
                    format!("https://huggingface.co/owner/repo/resolve/{PIN}/a-Q4_K_M.gguf")
                }
                Provider::GitHub => {
                    format!("https://raw.githubusercontent.com/owner/repo/{PIN}/a-Q4_K_M.gguf")
                }
            },
            status: 200,
            headers: BTreeMap::from([("content-length".into(), "3".into())]),
            body: vec![],
        });
        let script = Script::new(r);
        let lim = PullLimits {
            max_pages: 2,
            ..PullLimits::default()
        };
        assert!(matches!(
            quant::resolve(
                &script,
                &request(provider, "Q4_K_M", None),
                &lim,
                &admission(128 * 1024 * 1024),
                &AtomicBool::new(false)
            ),
            Err(Error::Limit(_))
        ));
        assert_eq!(script.calls.load(Ordering::Relaxed), 2);
        assert_eq!(script.rows.lock().unwrap().len(), 1);
    }
}
#[test]
fn cancelled_denied_or_capped_request_never_touches_metadata() {
    let script = Script::new(vec![]);
    let req = request(Provider::HuggingFace, "Q4_K_M", None);
    assert!(matches!(
        quant::resolve(
            &script,
            &req,
            &PullLimits::default(),
            &admission(1),
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        quant::resolve(
            &script,
            &req,
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(true)
        ),
        Err(Error::State(_))
    ));
    let limits = PullLimits {
        unknown_length_cap: Some(1),
        ..PullLimits::default()
    };
    assert!(matches!(
        quant::resolve(
            &script,
            &req,
            &limits,
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false)
        ),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
}
fn seed(
    provider: &str,
    revision: &str,
    paths: &[(&str, FileRole)],
    pointer: bool,
) -> (Fixture, SnapshotLease) {
    let fixture = Fixture::new();
    let store = fixture.open();
    let body = if pointer {
        b"version https://git-lfs.github.com/spec/v1\n".as_slice()
    } else {
        b"abc".as_slice()
    };
    let manifest = Manifest::new(
        ManifestSource::new(provider, "owner/repo", revision).unwrap(),
        paths
            .iter()
            .map(|(p, r)| ManifestEntry::new(*p, digest(body), body.len() as u64, *r).unwrap())
            .collect(),
        &CacheLimits::default(),
    )
    .unwrap();
    let mut tx = store
        .begin(
            manifest,
            &admission(128 * 1024 * 1024),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    for (p, _) in paths {
        tx.import_object(p, &mut Cursor::new(body)).unwrap();
    }
    let d = durable(tx.commit(None).unwrap());
    let lease = store.open_snapshot(&d).unwrap();
    (fixture, lease)
}
#[test]
fn offline_quant_is_complete_role_bound_subset_with_current_source_pointer_limits() {
    let complete = [
        ("a-Q4_K_M-00001-of-00002.gguf", FileRole::Weights),
        ("a-Q4_K_M-00002-of-00002.gguf", FileRole::Weights),
        ("config.json", FileRole::Config),
        ("tokenizer.json", FileRole::Tokenizer),
    ];
    let req = request(Provider::HuggingFace, "Q4_K_M", None);
    let (_f, lease) = seed("huggingface", PIN, &complete, false);
    quant::validate_cached_selection(
        &lease,
        &req,
        &PullLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    let mut bad = complete.to_vec();
    bad[0].1 = FileRole::Auxiliary;
    for paths in [bad, complete[1..].to_vec(), complete[..3].to_vec()] {
        let (_f, l) = seed("huggingface", PIN, &paths, false);
        assert!(
            quant::validate_cached_selection(
                &l,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    for (provider, rev, ptr) in [
        ("github", PIN, false),
        ("huggingface", "release-1-asset-2", false),
        ("huggingface", PIN, true),
    ] {
        let (_f, l) = seed(provider, rev, &complete, ptr);
        assert!(
            quant::validate_cached_selection(
                &l,
                &req,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    let small = PullLimits {
        cache: CacheLimits {
            max_file_bytes: 2,
            ..CacheLimits::default()
        },
        ..PullLimits::default()
    };
    assert!(
        quant::validate_cached_selection(&lease, &req, &small, &AtomicBool::new(false)).is_err()
    );
}
#[test]
fn ordinary_publication_retains_original_content_provenance_and_offline_binding() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(128 * 1024 * 1024);
    let paths = ["a-Q4_K_M.gguf", "config.json", "tokenizer.json"];
    let mut r = rows(Provider::HuggingFace, &paths);
    let mut tree: serde_json::Value = serde_json::from_slice(&r[1].body).unwrap();
    for entry in tree.as_array_mut().unwrap() {
        entry["lfs"] = serde_json::json!({"oid":digest(b"abc"),"size":3});
    }
    r[1].body = serde_json::to_vec(&tree).unwrap();
    let script = Script::new(r);
    let req = request(Provider::HuggingFace, "Q4_K_M", None);
    let plan = quant::resolve(
        &script,
        &req,
        &PullLimits::default(),
        &memory,
        &AtomicBool::new(false),
    )
    .unwrap();
    script.empty();
    let old = durable(publish(&store, &memory, OLD, Some("selected")).unwrap());
    let corrupt = Script::new(vec![Row {
        method: "GET",
        url: plan.files()[0].url().into(),
        status: 200,
        headers: BTreeMap::from([("content-length".into(), "3".into())]),
        body: b"abd".to_vec(),
    }]);
    assert!(
        acquisition::execute(
            &store,
            &plan,
            &corrupt,
            &memory,
            &Arc::new(AtomicBool::new(false)),
            Some("selected")
        )
        .is_err()
    );
    assert_eq!(
        store
            .resolve_ref("selected")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old
    );
    assert_no_host_reservation(&memory);
    // All three roles intentionally share one trusted content hash; only the
    // first object needs a body, subsequent members reuse verified cache data.
    let payload = Script::new(vec![Row {
        method: "GET",
        url: plan.files()[0].url().into(),
        status: 200,
        headers: BTreeMap::from([("content-length".into(), "3".into())]),
        body: b"abc".to_vec(),
    }]);
    let receipt = acquisition::execute(
        &store,
        &plan,
        &payload,
        &memory,
        &Arc::new(AtomicBool::new(false)),
        Some("selected"),
    )
    .unwrap();
    payload.empty();
    assert!(
        receipt
            .files()
            .iter()
            .all(|f| f.provenance() == DigestProvenance::TrustedSha256)
    );
    let lease = store.resolve_ref("selected").unwrap();
    quant::validate_cached_selection(
        &lease,
        &req,
        &PullLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(file_bytes(lease.file(paths[0]).unwrap()), b"abc");
    assert_no_host_reservation(&memory);
}
#[test]
fn native_quant_uses_one_hub_inventory_then_existing_xet_announcement_and_budget() {
    let paths = [
        "a-Q4_K_M.gguf",
        "a-Q8_0.gguf",
        "config.json",
        "tokenizer.json",
    ];
    let mut r = rows(Provider::HuggingFace, &paths);
    let mut tree: serde_json::Value = serde_json::from_slice(&r[1].body).unwrap();
    tree[0]["lfs"] = serde_json::json!({"oid":digest(b"abc"),"size":3});
    r[1].body = serde_json::to_vec(&tree).unwrap();
    r.push(Row {
        method: "GET",
        url: format!("https://huggingface.co/owner/repo/resolve/{PIN}/a-Q4_K_M.gguf"),
        status: 302,
        headers: BTreeMap::from([
            ("x-xet-hash".into(), XET.into()),
            ("x-repo-commit".into(), PIN.into()),
            ("x-linked-size".into(), "3".into()),
            ("x-linked-etag".into(), digest(b"abc")),
        ]),
        body: vec![],
    });
    let blocked = Script::new(r.clone());
    let limit = PullLimits {
        max_pages: 2,
        ..PullLimits::default()
    };
    assert!(matches!(
        xet::resolve_quant(
            &blocked,
            &request(Provider::HuggingFace, "Q4_K_M", None),
            &limit,
            &xet::XetOptions::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(blocked.calls.load(Ordering::Relaxed), 2);
    let mut changed = r.clone();
    changed[2]
        .headers
        .insert("x-repo-commit".into(), "a".repeat(40));
    assert!(matches!(
        xet::resolve_quant(
            &Script::new(changed),
            &request(Provider::HuggingFace, "Q4_K_M", None),
            &PullLimits::default(),
            &xet::XetOptions::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false)
        ),
        Err(Error::State(_))
    ));
    let script = Script::new(r);
    let memory = admission(128 * 1024 * 1024);
    let plan = xet::resolve_quant(
        &script,
        &request(Provider::HuggingFace, "Q4_K_M", None),
        &PullLimits::default(),
        &xet::XetOptions::default(),
        &memory,
        &AtomicBool::new(false),
    )
    .unwrap();
    script.empty();
    assert_eq!(script.calls.load(Ordering::Relaxed), 3);
    assert_eq!(plan.files().len(), 3);
    assert_eq!(plan.source().revision(), PIN);
    assert_no_host_reservation(&memory);
}
