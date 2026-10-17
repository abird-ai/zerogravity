//! Additive Unicode full-path filters, closure and native discovery ownership.
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
    self, CandidateRequest, GgufQuantRequest, Provider, PullLimits, ResolvedPullPlan, filters,
    quant, xet,
};
use zerogravity_artifacts::cache::{
    CacheLimits, FileRole, Manifest, ManifestEntry, ManifestSource, SnapshotLease,
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
fn resolve_quant(provider: Provider, tag: &str, paths: &[&str]) -> Result<ResolvedPullPlan> {
    let script = Script::new(rows(provider, paths));
    quant::resolve(
        &script,
        &request(provider, tag, None),
        &PullLimits::default(),
        &admission(128 * 1024 * 1024),
        &AtomicBool::new(false),
    )
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

fn filtered(
    provider: Provider,
    patterns: &[&str],
    excludes: &[&str],
    paths: &[&str],
) -> Result<ResolvedPullPlan> {
    let source = CandidateRequest::new(provider, "owner/repo", "main", None)?;
    let r = filters::FilterRequest::new(
        source,
        patterns.iter().map(|x| (*x).to_owned()).collect(),
        excludes.iter().map(|x| (*x).to_owned()).collect(),
    )?;
    let script = Script::new(rows(provider, paths));
    let result = filters::resolve(
        &script,
        &r,
        &PullLimits::default(),
        &admission(128 * 1024 * 1024),
        &AtomicBool::new(false),
    );
    script.empty();
    result
}
#[test]
fn include_union_exclude_before_family_disambiguation_and_no_implicit_fallback() {
    let all = [
        "a-Q4_K_M.gguf",
        "b-Q8_0.gguf",
        "config.json",
        "tokenizer.json",
        "README.md",
        "LICENSE",
        "generation_config.json",
    ];
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        assert!(filtered(provider, &["*.gguf"], &[], &all).is_err());
        let p = filtered(
            provider,
            &["*.gguf", "not-there*"],
            &["b-*.gguf", "README.md", "LICENSE"],
            &all,
        )
        .unwrap();
        assert_eq!(p.source().revision(), PIN);
        assert_eq!(
            p.files().iter().map(|x| x.path()).collect::<Vec<_>>(),
            [
                "a-Q4_K_M.gguf",
                "config.json",
                "generation_config.json",
                "tokenizer.json"
            ]
        );
        let old = resolve_quant(provider, "Q4_K_M", &all).unwrap();
        assert_eq!(p.source(), old.source());
        for f in p.files() {
            let counterpart = old.files().iter().find(|o| o.path() == f.path()).unwrap();
            assert_eq!(f.url(), counterpart.url());
            assert_eq!(f.identity(), counterpart.identity());
            assert_eq!(f.role(), counterpart.role());
        }
        assert!(filtered(provider, &["*.gguf"], &["*.gguf"], &all).is_err());
    }
}
#[test]
fn split_and_required_index_sidecar_exclusions_refuse() {
    let paths = [
        "a-Q4_K_M-00001-of-00002.gguf",
        "a-Q4_K_M-00002-of-00002.gguf",
        "b-Q4_K_M.gguf",
        "config.json",
        "tokenizer.json",
        "README.md",
    ];
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        assert!(
            filtered(
                provider,
                &["*.gguf"],
                &["a-Q4_K_M-00001-of-00002.gguf", "b-Q4_K_M.gguf"],
                &paths
            )
            .is_err()
        );
        assert!(filtered(provider, &["*.gguf"], &["config.json"], &paths).is_err());
        assert!(filtered(provider, &["*.gguf"], &["tokenizer.json"], &paths).is_err());
        let r = filters::FilterRequest::for_quant(
            request(provider, "Q4_K_M", None),
            vec!["b-Q4_K_M.gguf".into()],
        )
        .unwrap();
        let script = Script::new(rows(provider, &paths));
        let p = filters::resolve(
            &script,
            &r,
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false),
        )
        .unwrap();
        script.empty();
        assert_eq!(p.files().len(), 5); // split two, sidecars two and README
        assert!(!p.files().iter().any(|f| f.path() == "b-Q4_K_M.gguf"));
    }
}
#[test]
fn index_exclusions_veto_before_index_body_and_gguf_does_not_touch_unrelated_index() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let paths = [
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
        ];
        let script = Script::new(rows(provider, &paths));
        let r = filters::FilterRequest::new(
            CandidateRequest::new(provider, "owner/repo", "main", None).unwrap(),
            vec!["*.safetensors".into()],
            vec!["*.index.json".into()],
        )
        .unwrap();
        assert!(matches!(
            filters::resolve(
                &script,
                &r,
                &PullLimits::default(),
                &admission(128 * 1024 * 1024),
                &AtomicBool::new(false)
            ),
            Err(Error::Unsupported(_))
        ));
        script.empty(); // only the pin and tree; index GET was vetoed
        let p = filtered(
            provider,
            &["*.gguf"],
            &["*.index.json"],
            &[
                "only-Q8_0.gguf",
                "model.safetensors.index.json",
                "config.json",
                "tokenizer.json",
            ],
        )
        .unwrap();
        assert!(p.files().iter().any(|f| f.path() == "only-Q8_0.gguf"));
    }
}
#[test]
fn literal_bracketed_folder_unicode_scalar_filter_and_exact_case() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let paths = [
            "folder[one]/模型.gguf",
            "folder[one]/config.json",
            "folder[one]/tokenizer.json",
            "folder2/模型.gguf",
        ];
        let req = filters::FilterRequest::new(
            CandidateRequest::new(provider, "owner/repo", "main", Some("folder[one]")).unwrap(),
            vec!["模?.gguf".into()],
            vec![],
        )
        .unwrap();
        let script = Script::new(rows(provider, &paths));
        let p = filters::resolve(
            &script,
            &req,
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(p.files().len(), 3);
        script.empty();
        assert!(filtered(provider, &["*.GGUF"], &[], &paths).is_err());
    }
}
#[test]
fn malformed_denied_and_cancelled_before_provider_calls() {
    let p = Provider::HuggingFace;
    let source = CandidateRequest::new(p, "owner/repo", "main", None).unwrap();
    for bad in ["", &"a".repeat(4097), "bad\nname", "bad\u{0000}name"] {
        assert!(filters::FilterRequest::new(source.clone(), vec![bad.into()], vec![]).is_err());
    }
    assert!(filters::FilterRequest::new(source.clone(), vec!["a".into(); 4097], vec![]).is_err());
    let req = filters::FilterRequest::new(source, vec!["*".into()], vec![]).unwrap();
    let script = Script::new(vec![]);
    for (limit, cancel) in [(1, false), (128 * 1024 * 1024, true)] {
        assert!(
            filters::resolve(
                &script,
                &req,
                &PullLimits::default(),
                &admission(limit),
                &AtomicBool::new(cancel)
            )
            .is_err()
        );
    }
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
}
#[test]
fn exact_cached_filter_closure_refuses_extras_wrong_roles_and_pointers() {
    let complete = [
        ("a-Q4_K_M.gguf", FileRole::Weights),
        ("config.json", FileRole::Config),
        ("tokenizer.json", FileRole::Tokenizer),
    ];
    let r = filters::FilterRequest::new(
        CandidateRequest::new(Provider::HuggingFace, "owner/repo", PIN, None).unwrap(),
        vec!["*.gguf".into()],
        vec![],
    )
    .unwrap();
    let (_f, l) = seed("huggingface", PIN, &complete, false);
    filters::validate_cached_selection(&l, &r, &PullLimits::default(), &AtomicBool::new(false))
        .unwrap();
    let mut extra = complete.to_vec();
    extra.push(("README.md", FileRole::Auxiliary));
    let (_f, l) = seed("huggingface", PIN, &extra, false);
    // optional sidecar is original closure: another unselected arbitrary extra is not
    let mut other = complete.to_vec();
    other.push(("unrelated.txt", FileRole::Auxiliary));
    let (_f, l2) = seed("huggingface", PIN, &other, false);
    assert!(
        filters::validate_cached_selection(
            &l2,
            &r,
            &PullLimits::default(),
            &AtomicBool::new(false)
        )
        .is_err()
    );
    assert!(
        filters::validate_cached_selection(&l, &r, &PullLimits::default(), &AtomicBool::new(false))
            .is_ok()
    );
    extra[0].1 = FileRole::Auxiliary;
    let (_f, bad) = seed("huggingface", PIN, &extra, false);
    assert!(
        filters::validate_cached_selection(
            &bad,
            &r,
            &PullLimits::default(),
            &AtomicBool::new(false)
        )
        .is_err()
    );
    let (_f, pointer) = seed("huggingface", PIN, &complete, true);
    assert!(
        filters::validate_cached_selection(
            &pointer,
            &r,
            &PullLimits::default(),
            &AtomicBool::new(false)
        )
        .is_err()
    );
}
#[test]
fn native_filtered_selection_uses_one_hub_inventory_and_announces_exact_source() {
    let mut scripted = rows(
        Provider::HuggingFace,
        &[
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "config.json",
            "tokenizer.json",
        ],
    );
    // Native discovery is meaningful only for a pinned LFS object; ordinary
    // Git entries remain sidecars and never acquire CAS grants.
    let mut tree: serde_json::Value = serde_json::from_slice(&scripted[1].body).unwrap();
    tree[0]["lfs"] = serde_json::json!({
        "oid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "size": 3
    });
    scripted[1].body = serde_json::to_vec(&tree).unwrap();
    let mut headers = BTreeMap::new();
    headers.insert("x-repo-commit".into(), PIN.into());
    headers.insert("x-xet-hash".into(), XET.into());
    scripted.push(Row {
        method: "GET",
        url: format!("https://huggingface.co/owner/repo/resolve/{PIN}/model-Q4_K_M.gguf"),
        status: 302,
        headers,
        body: vec![],
    });
    let script = Script::new(scripted);
    let r = filters::FilterRequest::new(
        CandidateRequest::new(Provider::HuggingFace, "owner/repo", "main", None).unwrap(),
        vec!["*.gguf".into()],
        vec!["model-Q8_0.gguf".into()],
    )
    .unwrap();
    let plan = xet::resolve_filtered(
        &script,
        &r,
        &PullLimits::default(),
        &xet::XetOptions::default(),
        &admission(256 * 1024 * 1024),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(plan.files().len(), 3);
    script.empty(); // Native discovery only; never requested CAS/signed bytes
}
#[test]
fn bounded_control_memory_and_metadata_call_budget_refuse_before_unplanned_io() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let request = filters::FilterRequest::new(
            CandidateRequest::new(provider, "owner/repo", "main", None).unwrap(),
            vec!["*.gguf".into()],
            vec![],
        )
        .unwrap();
        let paths = ["a-Q4_K_M.gguf", "config.json", "tokenizer.json"];
        let script = Script::new(rows(provider, &paths));
        assert!(matches!(
            filters::resolve(
                &script,
                &request,
                &PullLimits::default(),
                &admission(96 * 1024 * 1024 - 1),
                &AtomicBool::new(false)
            ),
            Err(Error::Limit(_))
        ));
        assert_eq!(script.calls.load(Ordering::Relaxed), 0);
        let mut planned = rows(provider, &paths);
        let mut tree: serde_json::Value = serde_json::from_slice(&planned[1].body).unwrap();
        match provider {
            Provider::HuggingFace => {
                tree[0].as_object_mut().unwrap().remove("size");
            }
            Provider::GitHub => {
                tree["tree"][0].as_object_mut().unwrap().remove("size");
            }
        }
        planned[1].body = serde_json::to_vec(&tree).unwrap();
        planned.push(Row {
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
        let script = Script::new(planned);
        assert!(matches!(
            filters::resolve(
                &script,
                &request,
                &PullLimits {
                    max_pages: 2,
                    ..PullLimits::default()
                },
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
fn hidden_regular_ancestor_is_rejected_before_subfolder_projection() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let r = filters::FilterRequest::new(
            CandidateRequest::new(provider, "owner/repo", "main", Some("q")).unwrap(),
            vec!["*.gguf".into()],
            vec![],
        )
        .unwrap();
        let script = Script::new(rows(
            provider,
            &[
                "q",
                "q/model-Q4_K_M.gguf",
                "q/config.json",
                "q/tokenizer.json",
            ],
        ));
        assert!(
            filters::resolve(
                &script,
                &r,
                &PullLimits::default(),
                &admission(128 * 1024 * 1024),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        script.empty();
    }
}
#[test]
fn indexed_safetensors_closure_reuses_same_pin_and_index_body_once() {
    let index =
        br#"{"metadata":{"total_size":1},"weight_map":{"x":"model-00001-of-00001.safetensors"}}"#;
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let paths = [
            "model-00001-of-00001.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "README.md",
        ];
        let mut planned = rows(provider, &paths);
        let mut tree: serde_json::Value = serde_json::from_slice(&planned[1].body).unwrap();
        let nodes = if provider == Provider::HuggingFace {
            tree.as_array_mut().unwrap()
        } else {
            tree["tree"].as_array_mut().unwrap()
        };
        for node in nodes {
            if node["path"] == "model.safetensors.index.json" {
                node["size"] = serde_json::json!(index.len());
            }
        }
        planned[1].body = serde_json::to_vec(&tree).unwrap();
        planned.push(Row{
            method:"GET",
            url:if provider==Provider::HuggingFace {
                format!("https://huggingface.co/owner/repo/resolve/{PIN}/model.safetensors.index.json")
            } else {
                format!("https://raw.githubusercontent.com/owner/repo/{PIN}/model.safetensors.index.json")
            },
            status:200,headers:BTreeMap::new(),body:index.to_vec()
        });
        let script = Script::new(planned);
        let request = filters::FilterRequest::new(
            CandidateRequest::new(provider, "owner/repo", "main", None).unwrap(),
            vec!["*.safetensors".into()],
            vec!["README.md".into()],
        )
        .unwrap();
        let plan = filters::resolve(
            &script,
            &request,
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false),
        )
        .unwrap();
        script.empty();
        assert_eq!(script.calls.load(Ordering::Relaxed), 3);
        assert_eq!(plan.files().len(), 4);
        assert!(
            plan.files()
                .iter()
                .any(|f| f.path() == "model.safetensors.index.json")
        );
        assert!(!plan.files().iter().any(|f| f.path() == "README.md"));
    }
}
#[test]
fn filtered_plan_publishes_only_selected_verified_bytes_via_original_cache_executor() {
    use zerogravity_artifacts::cache::CommitOutcome;
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let files = [
            "a-Q4_K_M.gguf",
            "b-Q8_0.gguf",
            "config.json",
            "tokenizer.json",
            "README.md",
        ];
        let mut scripted = rows(provider, &files);
        for path in ["a-Q4_K_M.gguf", "config.json", "tokenizer.json"] {
            let url = match provider {
                Provider::HuggingFace => {
                    format!("https://huggingface.co/owner/repo/resolve/{PIN}/{path}")
                }
                Provider::GitHub => {
                    format!("https://raw.githubusercontent.com/owner/repo/{PIN}/{path}")
                }
            };
            scripted.push(Row {
                method: "GET",
                url,
                status: 200,
                headers: BTreeMap::new(),
                body: b"abc".to_vec(),
            });
        }
        let script = Script::new(scripted);
        let limits = PullLimits::default();
        let admission = admission(128 * 1024 * 1024);
        let req = filters::FilterRequest::new(
            CandidateRequest::new(provider, "owner/repo", "main", None).unwrap(),
            vec!["*.gguf".into()],
            vec!["b-*.gguf".into(), "README.md".into()],
        )
        .unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = filters::resolve(&script, &req, &limits, &admission, &cancel).unwrap();
        assert_eq!(plan.files().len(), 3);
        let fixture = Fixture::new();
        let store = fixture.open();
        let receipt = acquisition::execute(
            &store,
            &plan,
            &script,
            &admission,
            &cancel,
            Some("filtered"),
        )
        .unwrap();
        assert!(matches!(receipt.outcome(), CommitOutcome::Durable(_)));
        let lease = store.resolve_ref("filtered").unwrap();
        filters::validate_cached_selection(&lease, &req, &limits, &cancel).unwrap();
        assert_eq!(lease.manifest().entries().len(), 3);
        assert!(receipt.files().iter().all(|f| f.bytes() == 3));
        script.empty();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn qwen_and_gemma_companions_are_bounded_same_folder_and_not_executable() {
    let newly_supported = [
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.txt",
        "tokenizer.tiktoken",
        "chat_template.jinja",
        "processor_config.json",
        "preprocessor_config.json",
        "image_processor_config.json",
        "video_processor_config.json",
        "video_preprocessor_config.json",
        "feature_extractor_config.json",
        "audio_processor_config.json",
        "crc32.txt",
    ];
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let mut paths = vec![
            "q/weights-Q8_0.gguf",
            "q/config.json",
            "q/tokenizer.json",
            "q/README.md",
            "q/deliberate-other-quant-Q4_0.gguf",
            "q/subdir/processor_config.json",
            "other/chat_template.jinja",
            "q/unrelated.txt",
        ];
        let companions = newly_supported
            .iter()
            .map(|leaf| format!("q/{leaf}"))
            .collect::<Vec<_>>();
        paths.extend(companions.iter().map(String::as_str));
        let quant = request(provider, "Q8_0", Some("q"));
        let script = Script::new(rows(provider, &paths));
        let plan = quant::resolve(
            &script,
            &quant,
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false),
        )
        .unwrap();
        script.empty();
        for companion in &companions {
            assert!(
                plan.files().iter().any(|file| file.path() == companion),
                "missing bounded source companion {companion}"
            );
        }
        for unrelated in [
            "q/unrelated.txt",
            "other/chat_template.jinja",
            "q/subdir/processor_config.json",
            "q/deliberate-other-quant-Q4_0.gguf",
        ] {
            assert!(
                !plan.files().iter().any(|file| file.path() == unrelated),
                "model pull may not silently traverse or select other family"
            );
        }
        assert_eq!(
            plan.files().len(),
            newly_supported.len() + 4,
            "one weight, config, tokenizer, README plus known companions"
        );
        let skip_template = filters::FilterRequest::for_quant(
            quant,
            vec!["chat_template.jinja".into(), "README.md".into()],
        )
        .unwrap();
        let script = Script::new(rows(provider, &paths));
        let filtered = filters::resolve(
            &script,
            &skip_template,
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false),
        )
        .unwrap();
        script.empty();
        assert!(
            !filtered
                .files()
                .iter()
                .any(|file| file.path() == "q/chat_template.jinja")
        );
        assert!(
            !filtered
                .files()
                .iter()
                .any(|file| file.path() == "q/README.md")
        );
        assert!(
            filtered
                .files()
                .iter()
                .any(|file| file.path() == "q/processor_config.json")
        );
    }
}

#[test]
fn qwen_gemma_large_indexed_model_pull_plans_refuse_unadmitted_weight_bytes() {
    // Representative pinned original HF-style safetensors index layouts.
    // No learned tensor/header/model qualification is inferred; this validates
    // transfer source closure, full sizes and the original disk admission.
    use zerogravity_artifacts::acquisition::{PullRequest, Selection};
    let index = serde_json::to_vec(&serde_json::json!({
        "metadata": {"total_size": 51_600_000_000u64},
        "weight_map": {
            "language_model.layers.0.weight": "model-00001-of-00002.safetensors",
            "language_model.layers.1.weight": "model-00002-of-00002.safetensors"
        }
    }))
    .unwrap();
    let paths = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "unrelated-fp8.gguf",
    ];
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let mut planned = rows(provider, &paths);
        let mut tree: serde_json::Value = serde_json::from_slice(&planned[1].body).unwrap();
        let items = if provider == Provider::HuggingFace {
            tree.as_array_mut().unwrap()
        } else {
            tree["tree"].as_array_mut().unwrap()
        };
        for node in items {
            let size = match node["path"].as_str().unwrap() {
                "model-00001-of-00002.safetensors" => 49_900_000_000,
                "model-00002-of-00002.safetensors" => 1_700_000_000,
                "model.safetensors.index.json" => index.len() as u64,
                _ => 3,
            };
            node["size"] = serde_json::json!(size);
        }
        planned[1].body = serde_json::to_vec(&tree).unwrap();
        planned.push(Row {
            method: "GET",
            url: match provider {
                Provider::HuggingFace => {
                    format!("https://huggingface.co/owner/repo/resolve/{PIN}/model.safetensors.index.json")
                }
                Provider::GitHub => {
                    format!("https://raw.githubusercontent.com/owner/repo/{PIN}/model.safetensors.index.json")
                }
            },
            status: 200,
            headers: BTreeMap::new(),
            body: index.clone(),
        });
        let script = Script::new(planned);
        let request = PullRequest {
            provider,
            repository: "owner/repo".into(),
            revision: "main".into(),
            selection: Selection::Files {
                paths: vec!["model.safetensors.index.json".into()],
                subfolder: None,
            },
        };
        let admission = admission(128 * 1024 * 1024);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = acquisition::resolve(
            &script,
            &request,
            &PullLimits::default(),
            &admission,
            &cancel,
        )
        .unwrap();
        assert_eq!(plan.source().revision(), PIN);
        assert!(plan.total_bytes().unwrap() > 51_600_000_000);
        for expected in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "processor_config.json",
            "chat_template.jinja",
        ] {
            assert!(plan.files().iter().any(|file| file.path() == expected));
        }
        assert!(
            !plan
                .files()
                .iter()
                .any(|file| file.path() == "unrelated-fp8.gguf")
        );
        // Resolve only pins one commit/tree and reads one bounded index. No
        // credential, weight GET, cache publication or 50GiB fixture copy.
        script.empty();
        let fixture = Fixture::new();
        let store = fixture.open(); // 8MiB cooperative local quota
        let result = acquisition::execute(
            &store,
            &plan,
            &script,
            &admission,
            &cancel,
            Some("never-publish"),
        );
        assert!(
            result.is_err(),
            "unadmitted >50GB payload must refuse before GET"
        );
        assert!(store.resolve_ref("never-publish").is_err());
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn known_companions_roundtrip_through_exact_offline_cache_closure() {
    let paths = [
        ("weights-Q8_0.gguf", FileRole::Weights),
        ("config.json", FileRole::Config),
        ("tokenizer.json", FileRole::Tokenizer),
        ("chat_template.jinja", FileRole::Auxiliary),
        ("processor_config.json", FileRole::Config),
        ("preprocessor_config.json", FileRole::Config),
        ("tokenizer.model", FileRole::Tokenizer),
    ];
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let (name, _display) = match provider {
            Provider::HuggingFace => ("huggingface", "HF"),
            Provider::GitHub => ("github", "GH"),
        };
        let source = CandidateRequest::new(provider, "owner/repo", PIN, None).unwrap();
        let request =
            filters::FilterRequest::new(source.clone(), vec!["*.gguf".into()], vec![]).unwrap();
        let (_fixture, lease) = seed(name, PIN, &paths, false);
        filters::validate_cached_selection(
            &lease,
            &request,
            &PullLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap();
        let exclude = filters::FilterRequest::new(
            source.clone(),
            vec!["*.gguf".into()],
            vec!["processor_config.json".into()],
        )
        .unwrap();
        assert!(
            filters::validate_cached_selection(
                &lease,
                &exclude,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err(),
            "cached excluded companion must not be silently removed"
        );
        let mut extra = paths.to_vec();
        extra.push(("irrelevant-other-family.bin", FileRole::Auxiliary));
        let (_fixture, lease) = seed(name, PIN, &extra, false);
        assert!(
            filters::validate_cached_selection(
                &lease,
                &request,
                &PullLimits::default(),
                &AtomicBool::new(false)
            )
            .is_err(),
            "cached unrelated file is never declared source closure"
        );
    }
}

#[test]
fn explicit_opaque_multicomponent_bundle_uses_one_pinned_source_and_original_cache_publisher() {
    use zerogravity_artifacts::cache::CommitOutcome;
    // This is source data, NOT a verified Diffusers, ASR or TTS assembly.
    // In particular, no arbitrary nested safetensors index is interpreted
    // as a model's only index or a right to execute model-owned weights.
    let files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/model.safetensors",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "other-fp8/model.safetensors",
        "LICENSE",
    ];
    let selected = &files[..6];
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let mut scripted = rows(provider, &files);
        for path in selected {
            let url = match provider {
                Provider::HuggingFace => {
                    format!("https://huggingface.co/owner/repo/resolve/{PIN}/{path}")
                }
                Provider::GitHub => {
                    format!("https://raw.githubusercontent.com/owner/repo/{PIN}/{path}")
                }
            };
            scripted.push(Row {
                method: "GET",
                url,
                status: 200,
                headers: BTreeMap::new(),
                body: b"abc".to_vec(),
            });
        }
        let script = Script::new(scripted);
        let source = CandidateRequest::new(provider, "owner/repo", "main", None).unwrap();
        let req = filters::FilterRequest::bundle(
            source.clone(),
            [
                "model_index.json",
                "scheduler/*",
                "text_encoder/*",
                "transformer/*",
                "vae/*",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            vec!["other-fp8/*".into()],
        )
        .unwrap();
        let limits = PullLimits::default();
        let budget = admission(128 * 1024 * 1024);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = filters::resolve(&script, &req, &limits, &budget, &cancel).unwrap();
        assert_eq!(plan.files().len(), selected.len());
        assert_eq!(plan.source().repository(), "owner/repo");
        assert_eq!(plan.source().revision(), PIN);
        assert_eq!(
            plan.source().provider(),
            if provider == Provider::HuggingFace {
                "huggingface-bundle"
            } else {
                "github-bundle"
            }
        );
        assert!(
            plan.files()
                .iter()
                .all(|f| f.role() == FileRole::Auxiliary && f.bytes() == Some(3))
        );
        let fixture = Fixture::new();
        let store = fixture.open();
        let receipt = acquisition::execute(
            &store,
            &plan,
            &script,
            &budget,
            &cancel,
            Some("opaque-bundle"),
        )
        .unwrap();
        assert!(matches!(receipt.outcome(), CommitOutcome::Durable(_)));
        let lease = store.resolve_ref("opaque-bundle").unwrap();
        assert_eq!(lease.manifest().entries().len(), selected.len());
        filters::validate_cached_selection(&lease, &req, &limits, &cancel).unwrap();
        let exact = filters::FilterRequest::bundle(
            CandidateRequest::new(provider, "owner/repo", PIN, None).unwrap(),
            req.includes().to_vec(),
            req.excludes().to_vec(),
        )
        .unwrap();
        filters::validate_cached_selection(&lease, &exact, &limits, &cancel).unwrap();
        let wrong = filters::FilterRequest::bundle(
            CandidateRequest::new(provider, "owner/repo", PIN, None).unwrap(),
            vec!["model_index.json".into()],
            vec![],
        )
        .unwrap();
        assert!(filters::validate_cached_selection(&lease, &wrong, &limits, &cancel).is_err());
        let changed_source = filters::FilterRequest::bundle(
            CandidateRequest::new(provider, "owner/repo", &"f".repeat(40), None).unwrap(),
            req.includes().to_vec(),
            req.excludes().to_vec(),
        )
        .unwrap();
        assert!(
            filters::validate_cached_selection(&lease, &changed_source, &limits, &cancel).is_err()
        );
        let ordinary =
            filters::FilterRequest::new(source, vec!["*.safetensors".into()], vec![]).unwrap();
        assert!(filters::validate_cached_selection(&lease, &ordinary, &limits, &cancel).is_err());
        script.empty();
        assert_no_host_reservation(&budget);
    }
}

#[test]
fn bundle_is_explicit_and_cannot_infer_validated_model_or_codec_from_one_weight_file() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let source = CandidateRequest::new(provider, "owner/repo", "main", None).unwrap();
        assert!(filters::FilterRequest::bundle(source.clone(), vec![], vec![]).is_err());
        assert!(
            filters::FilterRequest::bundle(source.clone(), vec!["*".into()], vec!["*".into()])
                .is_ok()
        );
        let script = Script::new(rows(
            provider,
            &["component/model.safetensors", "README.md"],
        ));
        let req = filters::FilterRequest::bundle(
            source,
            vec!["component/model.safetensors".into()],
            vec![],
        )
        .unwrap();
        let plan = filters::resolve(
            &script,
            &req,
            &PullLimits::default(),
            &admission(128 * 1024 * 1024),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(plan.files().len(), 1);
        assert_eq!(plan.files()[0].role(), FileRole::Auxiliary);
        assert!(!plan.source().provider().eq(provider.label()));
        script.empty();
    }
}

#[test]
fn native_xet_bundle_qualifier_refuses_before_hub_metadata_or_cas_grants() {
    let script = Script::new(vec![]);
    let request = filters::FilterRequest::bundle(
        CandidateRequest::new(Provider::HuggingFace, "owner/repo", PIN, None).unwrap(),
        vec!["model_index.json".into(), "transformer/*".into()],
        vec![],
    )
    .unwrap();
    let result = xet::resolve_filtered(
        &script,
        &request,
        &PullLimits::default(),
        &xet::XetOptions::default(),
        &admission(256 * 1024 * 1024),
        &AtomicBool::new(false),
    );
    assert!(matches!(result, Err(Error::Unsupported(_))));
    script.empty();
}
