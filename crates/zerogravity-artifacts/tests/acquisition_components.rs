//! Independently pinned tiny sources, original cache snapshots and real leases.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use serde_json::json;
use std::collections::{BTreeMap, VecDeque};
use std::io::Cursor;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use zerogravity_artifacts::acquisition::{
    ComponentKind, ComponentPullRequest, ComponentRequest, ComponentTransport, Provider,
    PullLimits, PullRequest, Selection, execute_components, execute_components_with_progress,
    open_components, resolve, resolve_components,
};
use zerogravity_artifacts::cache::{FileRole, Manifest, ManifestEntry, ManifestSource};
use zerogravity_artifacts::{Error, Result};

const WEIGHTS: &[u8] = b"tiny synthetic original weight data";
const CONFIG: &[u8] = br#"{"model_type":"fixture","url":"https://unrequested.invalid/config"}"#;
const TOKENIZER: &[u8] = br#"{"version":"fixture"}"#;
fn pin(digit: char) -> String {
    digit.to_string().repeat(40)
}
struct Row {
    label: &'static str,
    url: String,
    body: Vec<u8>,
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
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl ComponentTransport for Script {
    fn get(
        &self,
        label: &str,
        request: &HttpRequest,
        _cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("undeclared remote request");
        assert_eq!(label, row.label);
        assert!(
            request.url.contains(&row.url),
            "expected fixture path {}",
            row.url
        );
        Ok(HttpResponse {
            status: 200,
            headers: BTreeMap::from([("content-length".into(), row.body.len().to_string())]),
            body: Box::new(Cursor::new(row.body)),
        })
    }
}
fn request(provider: Provider, repository: &str, path: &str) -> PullRequest {
    PullRequest {
        provider,
        repository: repository.into(),
        revision: "main".into(),
        selection: Selection::Files {
            paths: vec![path.into()],
            subfolder: None,
        },
    }
}
fn components(provider: Provider, limits: &PullLimits) -> ComponentPullRequest {
    ComponentPullRequest::new(
        vec![
            ComponentRequest::new(
                "weights",
                ComponentKind::Weights,
                request(provider, "converted/weights", "model.gguf"),
            )
            .unwrap(),
            ComponentRequest::new(
                "config",
                ComponentKind::Config,
                request(Provider::HuggingFace, "author/original", "config.json"),
            )
            .unwrap(),
            ComponentRequest::new(
                "tokenizer",
                ComponentKind::Tokenizer,
                request(Provider::HuggingFace, "author/original", "tokenizer.json"),
            )
            .unwrap(),
        ],
        limits,
    )
    .unwrap()
}
fn hf_entry(path: &str, bytes: &[u8]) -> serde_json::Value {
    json!({"type":"file","path":path,"size":bytes.len(),"lfs":{"oid":digest(bytes),"size":bytes.len()}})
}
fn hf_rows(
    label: &'static str,
    repository: &str,
    revision: &str,
    exact: char,
    entries: serde_json::Value,
) -> Vec<Row> {
    vec![
        Row {
            label,
            url: format!("/api/models/{repository}/revision/{revision}"),
            body: serde_json::to_vec(&json!({"sha":pin(exact)})).unwrap(),
        },
        Row {
            label,
            url: format!("/api/models/{repository}/tree/{}?", pin(exact)),
            body: serde_json::to_vec(&entries).unwrap(),
        },
    ]
}
fn metadata(provider: Provider) -> Vec<Row> {
    let mut rows = match provider {
        Provider::HuggingFace => hf_rows("weights", "converted/weights", "main", '1', json!([hf_entry("model.gguf", WEIGHTS)])),
        Provider::GitHub => vec![
            Row { label: "weights", url: "/repos/converted/weights/commits/main".into(), body: serde_json::to_vec(&json!({"sha":pin('1'),"commit":{"tree":{"sha":pin('3')}}})).unwrap() },
            Row { label: "weights", url: format!("/repos/converted/weights/git/trees/{}?", pin('3')), body: serde_json::to_vec(&json!({"sha":pin('3'),"truncated":false,"tree":[{"path":"model.gguf","mode":"100644","type":"blob","size":WEIGHTS.len(),"sha":pin('4')}]})).unwrap() },
        ],
    };
    let entries = json!([
        hf_entry("config.json", CONFIG),
        hf_entry("tokenizer.json", TOKENIZER)
    ]);
    rows.extend(hf_rows(
        "config",
        "author/original",
        "main",
        '2',
        entries.clone(),
    ));
    rows.extend(hf_rows(
        "tokenizer",
        "author/original",
        &pin('2'),
        '2',
        entries,
    ));
    rows
}
fn bodies(provider: Provider) -> Vec<Row> {
    vec![
        Row {
            label: "weights",
            url: match provider {
                Provider::HuggingFace => {
                    format!("/converted/weights/resolve/{}/model.gguf", pin('1'))
                }
                Provider::GitHub => format!("/converted/weights/{}/model.gguf", pin('1')),
            },
            body: WEIGHTS.to_vec(),
        },
        Row {
            label: "config",
            url: format!("/author/original/resolve/{}/config.json", pin('2')),
            body: CONFIG.to_vec(),
        },
        Row {
            label: "tokenizer",
            url: format!("/author/original/resolve/{}/tokenizer.json", pin('2')),
            body: TOKENIZER.to_vec(),
        },
    ]
}

#[test]
fn explicit_sources_pin_once_keep_separate_snapshots_and_open_offline() {
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let previous = durable(publish(&store, &admission, OLD, Some("previous")).unwrap());
        let limits = PullLimits::default();
        let request = components(provider, &limits);
        let script = Script::new(metadata(provider));
        let plan = resolve_components(
            &request,
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
        script.finished();
        assert_eq!(script.calls.load(Ordering::Relaxed), 6);
        assert_eq!(
            plan.total_bytes().unwrap(),
            (WEIGHTS.len() + CONFIG.len() + TOKENIZER.len()) as u64
        );
        assert_eq!(plan.components()[0].plan().source().revision(), pin('1'));
        assert_eq!(plan.components()[1].plan().source().revision(), pin('2'));
        assert_eq!(plan.components()[2].plan().source().revision(), pin('2'));
        assert_no_host_reservation(&admission);
        let script = Script::new(bodies(provider));
        let mut seen = Vec::new();
        let receipt = execute_components_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            &mut |label, event| {
                assert!(event.total_bytes() <= event.total_expected_bytes().unwrap());
                seen.push(label.to_string());
            },
        )
        .unwrap();
        script.finished();
        assert!(seen.iter().any(|label| label == "weights"));
        assert_eq!(receipt.receipts().len(), 3);
        let set = receipt.verified_set();
        let weights = set.component("weights").unwrap().lease();
        assert_eq!(
            weights.manifest().source().repository(),
            "converted/weights"
        );
        assert_eq!(weights.manifest().entries().len(), 1);
        assert_eq!(file_bytes(weights.file("model.gguf").unwrap()), WEIGHTS);
        assert_eq!(
            file_bytes(
                set.component("config")
                    .unwrap()
                    .lease()
                    .file("config.json")
                    .unwrap()
            ),
            CONFIG
        );
        assert_eq!(
            file_bytes(
                set.component("tokenizer")
                    .unwrap()
                    .lease()
                    .file("tokenizer.json")
                    .unwrap()
            ),
            TOKENIZER
        );
        let bindings: Vec<_> = set
            .components()
            .iter()
            .map(|component| {
                (
                    component.label().to_string(),
                    component.lease().manifest().digest().unwrap(),
                )
            })
            .collect();
        let offline = open_components(&fixture.open(), &request, &bindings, &limits).unwrap();
        assert_eq!(offline.components().len(), 3);
        let previous_lease = store.resolve_ref("previous").unwrap();
        assert_eq!(previous_lease.manifest().digest().unwrap(), previous);
        assert_payload(&previous_lease, OLD);
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn later_component_failure_preserves_refs_and_reports_no_complete_set() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("previous")).unwrap());
    let limits = PullLimits::default();
    let request = components(Provider::HuggingFace, &limits);
    let plan = resolve_components(
        &request,
        &Script::new(metadata(Provider::HuggingFace)),
        &limits,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    let mut rows = bodies(Provider::HuggingFace);
    rows[1].body[0] ^= 1;
    let script = Script::new(rows);
    assert!(
        execute_components(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false))
        )
        .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 2);
    assert_eq!(script.rows.lock().unwrap().len(), 1);
    assert_eq!(
        store.list_snapshots().unwrap().len(),
        2,
        "old snapshot plus individually complete weights may remain"
    );
    let lease = store.resolve_ref("previous").unwrap();
    assert_eq!(lease.manifest().digest().unwrap(), previous);
    assert_payload(&lease, OLD);
    assert_no_host_reservation(&admission);
}

#[test]
fn ordinary_resolve_keeps_missing_sidecars_strict_for_both_providers() {
    struct Direct(Script);
    impl HttpTransport for Direct {
        fn get(&self, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
            self.0.get("weights", request, cancel)
        }
    }
    for provider in [Provider::HuggingFace, Provider::GitHub] {
        let rows = metadata(provider).into_iter().take(2).collect();
        let script = Direct(Script::new(rows));
        let admission = admission(128 * 1024 * 1024);
        assert!(matches!(
            resolve(
                &script,
                &request(provider, "converted/weights", "model.gguf"),
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(false)
            ),
            Err(Error::Unsupported(_))
        ));
        script.0.finished();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn explicit_sidecars_do_not_relax_missing_weight_shards() {
    let limits = PullLimits::default();
    let mut components = components(Provider::HuggingFace, &limits)
        .components()
        .to_vec();
    components[0] = ComponentRequest::new(
        "weights",
        ComponentKind::Weights,
        request(
            Provider::HuggingFace,
            "converted/weights",
            "model.safetensors.index.json",
        ),
    )
    .unwrap();
    let request = ComponentPullRequest::new(components, &limits).unwrap();
    let index = br#"{"metadata":{"total_size":1},"weight_map":{"x":"missing.safetensors"}}"#;
    let mut rows = hf_rows(
        "weights",
        "converted/weights",
        "main",
        '1',
        json!([hf_entry("model.safetensors.index.json", index)]),
    );
    rows.push(Row {
        label: "weights",
        url: format!("/resolve/{}/model.safetensors.index.json", pin('1')),
        body: index.to_vec(),
    });
    let script = Script::new(rows);
    let admission = admission(128 * 1024 * 1024);
    assert!(matches!(
        resolve_components(
            &request,
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Unsupported(_))
    ));
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn declared_sidecar_roles_and_reused_exact_pins_are_checked_before_payloads() {
    let limits = PullLimits::default();
    let mut members = components(Provider::HuggingFace, &limits)
        .components()
        .to_vec();
    members[1] = ComponentRequest::new(
        "config",
        ComponentKind::Config,
        request(Provider::HuggingFace, "author/original", "LICENSE"),
    )
    .unwrap();
    let request = ComponentPullRequest::new(members, &limits).unwrap();
    let mut rows: Vec<_> = metadata(Provider::HuggingFace)
        .into_iter()
        .take(2)
        .collect();
    rows.extend(hf_rows(
        "config",
        "author/original",
        "main",
        '2',
        json!([hf_entry("LICENSE", b"a license")]),
    ));
    let script = Script::new(rows);
    let admission = admission(128 * 1024 * 1024);
    assert!(matches!(
        resolve_components(
            &request,
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Unsupported(_))
    ));
    script.finished();
    let mut rows = metadata(Provider::HuggingFace);
    rows[4].body = serde_json::to_vec(&json!({"sha":pin('3')})).unwrap();
    rows.truncate(5);
    let script = Script::new(rows);
    assert!(matches!(
        resolve_components(
            &components(Provider::HuggingFace, &limits),
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::State(_))
    ));
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn aggregate_metadata_request_and_byte_limits_precede_payloads() {
    let base = metadata(Provider::HuggingFace);
    let first_bytes = base[..2].iter().map(|row| row.body.len()).sum::<usize>();
    for (metadata_limit, requests, expected_calls) in
        [(first_bytes + 10, 64, 3), (8 * 1024 * 1024, 2, 2)]
    {
        let limits = PullLimits {
            max_metadata_bytes: metadata_limit,
            max_pages: requests,
            ..PullLimits::default()
        };
        let request = components(Provider::HuggingFace, &limits);
        let script = Script::new(metadata(Provider::HuggingFace));
        let admission = admission(128 * 1024 * 1024);
        assert!(
            resolve_components(
                &request,
                &script,
                &limits,
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(script.calls.load(Ordering::Relaxed), expected_calls);
        assert_no_host_reservation(&admission);
    }
    let mut limits = PullLimits::default();
    limits.cache.max_snapshot_bytes = (WEIGHTS.len() + CONFIG.len() + TOKENIZER.len() - 1) as u64;
    let script = Script::new(metadata(Provider::HuggingFace));
    let admission = admission(128 * 1024 * 1024);
    assert!(matches!(
        resolve_components(
            &components(Provider::HuggingFace, &limits),
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    script.finished();
    assert_no_host_reservation(&admission);

    let mut limits = PullLimits::default();
    limits.cache.max_files = 3;
    let mut rows = metadata(Provider::HuggingFace);
    rows[1].body = serde_json::to_vec(&json!([
        hf_entry("model.gguf", WEIGHTS),
        hf_entry("config.json", CONFIG),
        hf_entry("tokenizer.json", TOKENIZER),
    ]))
    .unwrap();
    rows.truncate(4);
    let script = Script::new(rows);
    assert!(matches!(
        resolve_components(
            &components(Provider::HuggingFace, &limits),
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn admission_and_cancellation_precede_every_component_metadata_request() {
    struct Never;
    impl ComponentTransport for Never {
        fn get(&self, _: &str, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("request before admission/cancellation");
        }
    }
    let limits = PullLimits::default();
    let request = components(Provider::HuggingFace, &limits);
    for (bytes, canceled) in [(1, false), (128 * 1024 * 1024, true)] {
        let admission = admission(bytes);
        assert!(
            resolve_components(
                &request,
                &Never,
                &limits,
                &admission,
                &AtomicBool::new(canceled)
            )
            .is_err()
        );
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn cached_shared_source_must_bind_the_same_exact_commit() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let request = components(Provider::HuggingFace, &limits);
    let plan = resolve_components(
        &request,
        &Script::new(metadata(Provider::HuggingFace)),
        &limits,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    let receipt = execute_components(
        &store,
        &plan,
        &Script::new(bodies(Provider::HuggingFace)),
        &admission,
        &Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let mut bindings: Vec<_> = receipt
        .verified_set()
        .components()
        .iter()
        .map(|component| {
            (
                component.label().to_string(),
                component.lease().manifest().digest().unwrap(),
            )
        })
        .collect();
    let manifest = Manifest::new(
        ManifestSource::new("huggingface", "author/original", pin('3')).unwrap(),
        vec![
            ManifestEntry::new(
                "tokenizer.json",
                digest(TOKENIZER),
                TOKENIZER.len() as u64,
                FileRole::Tokenizer,
            )
            .unwrap(),
        ],
        &limits.cache,
    )
    .unwrap();
    let mut tx = store
        .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    let mut tokenizer = TOKENIZER;
    tx.import_object("tokenizer.json", &mut tokenizer).unwrap();
    bindings[2].1 = durable(tx.commit(None).unwrap());
    assert!(matches!(
        open_components(&store, &request, &bindings, &limits),
        Err(Error::Malformed(_))
    ));
    bindings[2].0 = "config".into();
    assert!(open_components(&store, &request, &bindings, &limits).is_err());
    assert_no_host_reservation(&admission);
}

#[test]
fn offline_components_apply_caller_file_limit_even_with_a_broader_open_store() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let request = components(Provider::HuggingFace, &limits);
    let plan = resolve_components(
        &request,
        &Script::new(metadata(Provider::HuggingFace)),
        &limits,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    let receipt = execute_components(
        &store,
        &plan,
        &Script::new(bodies(Provider::HuggingFace)),
        &admission,
        &Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let bindings: Vec<_> = receipt
        .verified_set()
        .components()
        .iter()
        .map(|component| {
            (
                component.label().to_string(),
                component.lease().manifest().digest().unwrap(),
            )
        })
        .collect();
    let mut narrow = limits;
    narrow.cache.max_file_bytes = WEIGHTS.len() as u64 - 1;
    assert!(matches!(
        open_components(&store, &request, &bindings, &narrow),
        Err(Error::Limit(_))
    ));
    assert_no_host_reservation(&admission);
}

#[test]
fn component_authority_labels_role_suppliers_and_release_counts_are_bounded() {
    let limits = PullLimits::default();
    assert!(
        ComponentRequest::new(
            "https://secret.invalid",
            ComponentKind::Weights,
            request(Provider::HuggingFace, "converted/weights", "model.gguf")
        )
        .is_err()
    );
    let mut repeated = components(Provider::HuggingFace, &limits)
        .components()
        .to_vec();
    repeated.push(repeated[0].clone());
    assert!(ComponentPullRequest::new(repeated, &limits).is_err());
    let mut too_many = components(Provider::HuggingFace, &limits)
        .components()
        .to_vec();
    for number in 0..14 {
        too_many.push(
            ComponentRequest::new(
                format!("aux{number}"),
                ComponentKind::Auxiliary,
                request(Provider::HuggingFace, "author/original", "LICENSE"),
            )
            .unwrap(),
        );
    }
    assert!(matches!(
        ComponentPullRequest::new(too_many, &limits),
        Err(Error::Limit(_))
    ));
    let mut releases = components(Provider::HuggingFace, &limits)
        .components()
        .to_vec();
    for (label, name) in [("release1", "one.txt"), ("release2", "two.txt")] {
        releases.push(
            ComponentRequest::new(
                label,
                ComponentKind::Auxiliary,
                PullRequest {
                    provider: Provider::GitHub,
                    repository: "author/original".into(),
                    revision: "release".into(),
                    selection: Selection::ReleaseAsset { name: name.into() },
                },
            )
            .unwrap(),
        );
    }
    assert_eq!(
        ComponentPullRequest::new(releases, &limits)
            .unwrap()
            .components()
            .len(),
        5
    );
}

fn release_request(repository: &str, tag: &str, name: &str) -> PullRequest {
    PullRequest {
        provider: Provider::GitHub,
        repository: repository.into(),
        revision: tag.into(),
        selection: Selection::ReleaseAsset { name: name.into() },
    }
}

fn shared_file_and_release_request(limits: &PullLimits) -> ComponentPullRequest {
    ComponentPullRequest::new(
        vec![
            ComponentRequest::new(
                "weights",
                ComponentKind::Weights,
                request(Provider::GitHub, "shared/repo", "model.gguf"),
            )
            .unwrap(),
            ComponentRequest::new(
                "config",
                ComponentKind::Config,
                release_request("shared/repo", "main", "config.json"),
            )
            .unwrap(),
            ComponentRequest::new(
                "tokenizer",
                ComponentKind::Tokenizer,
                release_request("shared/repo", "main", "tokenizer.json"),
            )
            .unwrap(),
        ],
        limits,
    )
    .unwrap()
}

fn release_assets() -> Vec<u8> {
    serde_json::to_vec(&json!([
        {"id":113,"name":"config.json","size":CONFIG.len(),"state":"uploaded","digest":format!("sha256:{}",digest(CONFIG))},
        {"id":114,"name":"tokenizer.json","size":TOKENIZER.len(),"state":"uploaded","digest":format!("sha256:{}",digest(TOKENIZER))},
    ])).unwrap()
}

fn shared_file_metadata() -> Vec<Row> {
    vec![
        Row {label:"weights",url:"/repos/shared/repo/commits/main".into(),body:serde_json::to_vec(&json!({"sha":pin('1'),"commit":{"tree":{"sha":pin('3')}}})).unwrap()},
        Row {label:"weights",url:format!("/repos/shared/repo/git/trees/{}?",pin('3')),body:serde_json::to_vec(&json!({"sha":pin('3'),"truncated":false,"tree":[{"path":"model.gguf","mode":"100644","type":"blob","size":WEIGHTS.len(),"sha":pin('4')}]})).unwrap()},
    ]
}

fn shared_release_metadata() -> Vec<Row> {
    vec![
        Row {
            label: "config",
            url: "/repos/shared/repo/releases/tags/main".into(),
            body: serde_json::to_vec(&json!({"id":91,"tag_name":"main"})).unwrap(),
        },
        Row {
            label: "config",
            url: "/repos/shared/repo/releases/91/assets?".into(),
            body: release_assets(),
        },
    ]
}

fn shared_tokenizer_metadata() -> Row {
    Row {
        label: "tokenizer",
        url: "/repos/shared/repo/releases/91/assets?".into(),
        body: release_assets(),
    }
}

fn shared_bodies() -> Vec<Row> {
    vec![
        Row {
            label: "weights",
            url: format!("/shared/repo/{}/model.gguf", pin('1')),
            body: WEIGHTS.to_vec(),
        },
        Row {
            label: "config",
            url: "/repos/shared/repo/releases/assets/113".into(),
            body: CONFIG.to_vec(),
        },
        Row {
            label: "tokenizer",
            url: "/repos/shared/repo/releases/assets/114".into(),
            body: TOKENIZER.to_vec(),
        },
    ]
}

#[test]
fn file_and_release_aliases_with_the_same_tuple_keep_separate_pin_namespaces() {
    for release_first in [false, true] {
        let limits = PullLimits::default();
        let fixture = Fixture::new();
        let store = fixture.open();
        let memory = admission(128 * 1024 * 1024);
        let mut selections = shared_file_and_release_request(&limits)
            .components()
            .to_vec();
        let mut rows = Vec::new();
        if release_first {
            selections.swap(0, 1);
            rows.extend(shared_release_metadata());
            rows.extend(shared_file_metadata());
        } else {
            rows.extend(shared_file_metadata());
            rows.extend(shared_release_metadata());
        }
        rows.push(shared_tokenizer_metadata());
        let request = ComponentPullRequest::new(selections, &limits).unwrap();
        let metadata = Script::new(rows);
        let plan = resolve_components(
            &request,
            &metadata,
            &limits,
            &memory,
            &AtomicBool::new(false),
        )
        .unwrap();
        metadata.finished();
        assert_eq!(metadata.calls.load(Ordering::Relaxed), 5);
        let mut rows = shared_bodies();
        if release_first {
            rows.swap(0, 1);
        }
        let payloads = Script::new(rows);
        let receipt = execute_components(
            &store,
            &plan,
            &payloads,
            &memory,
            &Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        payloads.finished();
        let verified = receipt.verified_set();
        assert_eq!(
            verified
                .component("weights")
                .unwrap()
                .lease()
                .manifest()
                .source()
                .revision(),
            pin('1')
        );
        assert_eq!(
            verified
                .component("config")
                .unwrap()
                .lease()
                .manifest()
                .source()
                .revision(),
            "release-91-asset-113"
        );
        assert_eq!(
            verified
                .component("tokenizer")
                .unwrap()
                .lease()
                .manifest()
                .source()
                .revision(),
            "release-91-asset-114"
        );
        let bindings: Vec<_> = verified
            .components()
            .iter()
            .map(|component| {
                (
                    component.label().to_string(),
                    component.lease().manifest().digest().unwrap(),
                )
            })
            .collect();
        let offline = open_components(&store, &request, &bindings, &limits).unwrap();
        assert_eq!(offline.components().len(), 3);
        assert_no_host_reservation(&memory);
    }
}

fn cached_component(
    store: &zerogravity_artifacts::cache::CacheStore,
    memory: &zerogravity_artifacts::cache::TransferAdmission,
    repository: &str,
    revision: &str,
    path: &str,
    role: FileRole,
    bytes: &[u8],
) -> String {
    let manifest = Manifest::new(
        ManifestSource::new("github", repository, revision).unwrap(),
        vec![ManifestEntry::new(path, digest(bytes), bytes.len() as u64, role).unwrap()],
        &Default::default(),
    )
    .unwrap();
    let mut transaction = store
        .begin(manifest, memory, Arc::new(AtomicBool::new(false)))
        .unwrap();
    transaction
        .import_object(path, &mut Cursor::new(bytes))
        .unwrap();
    durable(transaction.commit(None).unwrap())
}

#[test]
fn offline_release_coherence_requires_canonical_positive_locators_but_not_equal_assets() {
    let limits = PullLimits::default();
    let request = shared_file_and_release_request(&limits);
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(128 * 1024 * 1024);
    let weights = cached_component(
        &store,
        &memory,
        "shared/repo",
        &pin('1'),
        "model.gguf",
        FileRole::Weights,
        WEIGHTS,
    );
    let config = cached_component(
        &store,
        &memory,
        "shared/repo",
        "release-91-asset-113",
        "config.json",
        FileRole::Config,
        CONFIG,
    );
    let tokenizer = cached_component(
        &store,
        &memory,
        "shared/repo",
        "release-91-asset-114",
        "tokenizer.json",
        FileRole::Tokenizer,
        TOKENIZER,
    );
    let mut bindings = vec![
        ("weights".into(), weights),
        ("config".into(), config),
        ("tokenizer".into(), tokenizer),
    ];
    assert_eq!(
        open_components(&store, &request, &bindings, &limits)
            .unwrap()
            .components()
            .len(),
        3
    );
    let file_pin = pin('1');
    for revision in [
        "release-92-asset-114",
        "release-0-asset-114",
        "release-091-asset-114",
        "release-91-asset-0",
        "release-91-asset-0114",
        "release-91-asset-+114",
        "release-18446744073709551616-asset-114",
        "release-91-asset-18446744073709551616",
        "release-91-asset-114-extra",
        "release-91-asset--114",
        file_pin.as_str(),
    ] {
        bindings[2].1 = cached_component(
            &store,
            &memory,
            "shared/repo",
            revision,
            "tokenizer.json",
            FileRole::Tokenizer,
            TOKENIZER,
        );
        assert!(
            matches!(
                open_components(&store, &request, &bindings, &limits),
                Err(Error::Malformed(_))
            ),
            "accepted {revision}"
        );
    }
    assert_no_host_reservation(&memory);
}

#[test]
fn repeated_release_metadata_still_consumes_aggregate_requests_and_payload_bounds() {
    for request_budget in [true, false] {
        let mut limits = PullLimits::default();
        if request_budget {
            limits.max_pages = 4;
        } else {
            limits.cache.max_snapshot_bytes =
                (WEIGHTS.len() + CONFIG.len() + TOKENIZER.len() - 1) as u64;
        }
        let request = shared_file_and_release_request(&limits);
        let memory = admission(128 * 1024 * 1024);
        let mut rows = shared_file_metadata();
        rows.extend(shared_release_metadata());
        rows.push(shared_tokenizer_metadata());
        let metadata = Script::new(rows);
        assert!(matches!(
            resolve_components(
                &request,
                &metadata,
                &limits,
                &memory,
                &AtomicBool::new(false)
            ),
            Err(Error::Limit(_))
        ));
        assert_eq!(
            metadata.calls.load(Ordering::Relaxed),
            if request_budget { 4 } else { 5 }
        );
        if !request_budget {
            metadata.finished();
        }
        assert_no_host_reservation(&memory);
    }
    struct NoRequests;
    impl ComponentTransport for NoRequests {
        fn get(&self, _: &str, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("metadata before admission/cancel");
        }
    }
    let limits = PullLimits::default();
    let request = shared_file_and_release_request(&limits);
    for (bytes, canceled) in [(1, false), (128 * 1024 * 1024, true)] {
        let memory = admission(bytes);
        assert!(
            resolve_components(
                &request,
                &NoRequests,
                &limits,
                &memory,
                &AtomicBool::new(canceled)
            )
            .is_err()
        );
        assert_no_host_reservation(&memory);
    }
}
