//! Independent release-alias fixtures, retaining original source snapshots.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::io::Cursor;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse};
use zerogravity_artifacts::acquisition::{
    ComponentKind, ComponentPullRequest, ComponentRequest, ComponentTransport, Provider,
    PullLimits, PullRequest, Selection, execute_components, open_components, resolve_components,
};
use zerogravity_artifacts::cache::{
    CacheStore, FileRole, Manifest, ManifestEntry, ManifestSource, TransferAdmission,
};
use zerogravity_artifacts::{Error, Result};
const REPO: &str = "owner/repo";
const TAG: &str = "stable";
const A: &[u8] = b"synthetic weight data";
const B: &[u8] = b"config bytes";
const C: &[u8] = b"tokenizer bytes";
fn role(name: &str) -> FileRole {
    match name {
        "model.gguf" => FileRole::Weights,
        "config.json" => FileRole::Config,
        "tokenizer.json" => FileRole::Tokenizer,
        _ => FileRole::Auxiliary,
    }
}
fn component(label: &str, repo: &str, tag: &str, name: &str) -> ComponentRequest {
    let kind = match role(name) {
        FileRole::Weights => ComponentKind::Weights,
        FileRole::Config => ComponentKind::Config,
        FileRole::Tokenizer => ComponentKind::Tokenizer,
        _ => ComponentKind::Auxiliary,
    };
    ComponentRequest::new(
        label,
        kind,
        PullRequest {
            provider: Provider::GitHub,
            repository: repo.into(),
            revision: tag.into(),
            selection: Selection::ReleaseAsset { name: name.into() },
        },
    )
    .unwrap()
}
fn members() -> Vec<ComponentRequest> {
    vec![
        component("first", REPO, TAG, "model.gguf"),
        component("second", REPO, TAG, "config.json"),
        component("third", REPO, TAG, "tokenizer.json"),
    ]
}
fn request(limits: &PullLimits) -> ComponentPullRequest {
    ComponentPullRequest::new(members(), limits).unwrap()
}
fn asset(id: u64, name: &str, bytes: &[u8]) -> Value {
    json!({"id":id,"name":name,"size":bytes.len(),"state":"uploaded","digest":format!("sha256:{}",digest(bytes))})
}
fn assets() -> Value {
    json!([
        asset(11, "model.gguf", A),
        asset(12, "config.json", B),
        asset(13, "tokenizer.json", C)
    ])
}
fn release(id: u64, tag: &str) -> Value {
    json!({"id":id,"tag_name":tag})
}
struct Row {
    label: String,
    url: String,
    body: Vec<u8>,
}
impl Row {
    fn json(label: &str, path: &str, value: Value) -> Self {
        Self {
            label: label.into(),
            url: format!("https://api.github.com{path}"),
            body: serde_json::to_vec(&value).unwrap(),
        }
    }
    fn body(label: &str, id: u64, bytes: &[u8]) -> Self {
        Self {
            label: label.into(),
            url: format!("https://api.github.com/repos/{REPO}/releases/assets/{id}"),
            body: bytes.to_vec(),
        }
    }
}
struct Script {
    rows: Mutex<VecDeque<Row>>,
    calls: AtomicUsize,
    admission: TransferAdmission,
    cancel_after: Option<usize>,
}
impl Script {
    fn new(rows: Vec<Row>, admission: &TransferAdmission) -> Self {
        Self {
            rows: Mutex::new(rows.into()),
            calls: AtomicUsize::new(0),
            admission: admission.clone(),
            cancel_after: None,
        }
    }
    fn finished(&self) {
        assert!(
            self.rows.lock().unwrap().is_empty(),
            "unconsumed declared requests"
        );
    }
}
impl ComponentTransport for Script {
    fn get(&self, label: &str, request: &HttpRequest, cancel: &AtomicBool) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0,
            "admission precedes metadata/payload"
        );
        let count = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("undeclared alias/payload lookup");
        assert_eq!(
            label, row.label,
            "each original label keeps its credential adapter"
        );
        assert_eq!(
            request.url, row.url,
            "mutable alias reread or changed asset-ID route"
        );
        assert!(request.range.is_none());
        if self.cancel_after == Some(count) {
            cancel.store(true, Ordering::Release);
        }
        Ok(HttpResponse {
            status: 200,
            headers: BTreeMap::from([("content-length".into(), row.body.len().to_string())]),
            body: Box::new(Cursor::new(row.body)),
        })
    }
}
fn metadata_rows() -> Vec<Row> {
    let mut rows = vec![Row::json(
        "first",
        &format!("/repos/{REPO}/releases/tags/{TAG}"),
        release(91, TAG),
    )];
    for label in ["first", "second", "third"] {
        // The tag could now name release92. Later labels must not read it.
        let listed = if label == "second" {
            json!([
                asset(99, "model.gguf", b"replacement"),
                asset(12, "config.json", B),
                asset(13, "tokenizer.json", C)
            ])
        } else {
            assets()
        };
        rows.push(Row::json(
            label,
            &format!("/repos/{REPO}/releases/91/assets?per_page=100&page=1"),
            listed,
        ));
    }
    rows
}
fn publish_source(
    store: &CacheStore,
    admission: &TransferAdmission,
    repo: &str,
    locator: &str,
    name: &str,
    bytes: &[u8],
) -> String {
    let manifest = Manifest::new(
        ManifestSource::new("github", repo, locator).unwrap(),
        vec![ManifestEntry::new(name, digest(bytes), bytes.len() as u64, role(name)).unwrap()],
        &Default::default(),
    )
    .unwrap();
    let mut tx = store
        .begin(manifest, admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    tx.import_object(name, &mut Cursor::new(bytes)).unwrap();
    durable(tx.commit(None).unwrap())
}
fn bindings(store: &CacheStore, admission: &TransferAdmission) -> Vec<(String, String)> {
    vec![
        (
            "first".into(),
            publish_source(
                store,
                admission,
                REPO,
                "release-91-asset-11",
                "model.gguf",
                A,
            ),
        ),
        (
            "second".into(),
            publish_source(
                store,
                admission,
                REPO,
                "release-91-asset-12",
                "config.json",
                B,
            ),
        ),
        (
            "third".into(),
            publish_source(
                store,
                admission,
                REPO,
                "release-91-asset-13",
                "tokenizer.json",
                C,
            ),
        ),
    ]
}
#[test]
fn offline_release_members_share_only_canonical_release_id_not_asset_id() {
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let bindings = bindings(&store, &admission);
    let request = request(&limits);
    let set = open_components(&store, &request, &bindings, &limits).unwrap();
    assert_eq!(
        file_bytes(
            set.component("first")
                .unwrap()
                .lease()
                .file("model.gguf")
                .unwrap()
        ),
        A
    );
    assert_eq!(
        file_bytes(
            set.component("second")
                .unwrap()
                .lease()
                .file("config.json")
                .unwrap()
        ),
        B
    );
    for locator in [
        "release-92-asset-12",
        "release-091-asset-12",
        "release-91-asset-0",
        "release-91-asset-12-extra",
        "1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ] {
        let bad = publish_source(&store, &admission, REPO, locator, "config.json", B);
        let mut selected = bindings.clone();
        selected[1].1 = bad;
        assert!(
            open_components(&store, &request, &selected, &limits).is_err(),
            "accepted {locator}"
        );
    }
    let mut file_members = members();
    file_members[0] = ComponentRequest::new(
        "first",
        ComponentKind::Weights,
        PullRequest {
            provider: Provider::GitHub,
            repository: REPO.into(),
            revision: TAG.into(),
            selection: Selection::Files {
                paths: vec!["model.gguf".into()],
                subfolder: None,
            },
        },
    )
    .unwrap();
    let mixed = ComponentPullRequest::new(file_members, &limits).unwrap();
    let mut mixed_bindings = bindings;
    mixed_bindings[0].1 =
        publish_source(&store, &admission, REPO, &"1".repeat(40), "model.gguf", A);
    assert!(
        open_components(&store, &mixed, &mixed_bindings, &limits).is_ok(),
        "file/release namespaces must not collide"
    );
    assert_no_host_reservation(&admission);
}
#[test]
fn moving_tag_is_looked_up_once_then_each_label_uses_release_and_asset_ids() {
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let script = Script::new(metadata_rows(), &admission);
    let request = request(&limits);
    let plan = resolve_components(
        &request,
        &script,
        &limits,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    script.finished();
    assert_eq!(script.calls.load(Ordering::Relaxed), 4);
    assert_eq!(
        plan.total_bytes(),
        Some((A.len() + B.len() + C.len()) as u64)
    );
    for (member, id) in plan.components().iter().zip([11, 12, 13]) {
        assert_eq!(
            member.plan().source().revision(),
            format!("release-91-asset-{id}")
        );
        assert_eq!(
            member.plan().files()[0].identity(),
            format!("release:91:asset:{id}")
        );
        assert_eq!(
            member.plan().files()[0].url(),
            format!("https://api.github.com/repos/{REPO}/releases/assets/{id}")
        );
    }
    assert_no_host_reservation(&admission);
    let bodies = Script::new(
        vec![
            Row::body("first", 11, A),
            Row::body("second", 12, B),
            Row::body("third", 13, C),
        ],
        &admission,
    );
    let receipt = execute_components(
        &store,
        &plan,
        &bodies,
        &admission,
        &Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    bodies.finished();
    let mut bound = Vec::new();
    let mut identities = Vec::new();
    for (label, name, bytes, id) in [
        ("first", "model.gguf", A, 11),
        ("second", "config.json", B, 12),
        ("third", "tokenizer.json", C, 13),
    ] {
        let lease = receipt.verified_set().component(label).unwrap().lease();
        assert_eq!(
            lease.manifest().source().revision(),
            format!("release-91-asset-{id}")
        );
        assert_eq!(lease.manifest().entries().len(), 1);
        assert_eq!(file_bytes(lease.file(name).unwrap()), bytes);
        let digest = lease.manifest().digest().unwrap();
        identities.push(digest.clone());
        bound.push((label.into(), digest));
    }
    identities.sort_unstable();
    identities.dedup();
    assert_eq!(identities.len(), 3);
    assert!(open_components(&store, &request, &bound, &limits).is_ok());
    assert_eq!(store.list_snapshots().unwrap().len(), 3);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}
#[test]
fn later_replaced_payload_retains_prior_ref_and_earlier_original_snapshot() {
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let metadata = Script::new(metadata_rows(), &admission);
    let plan = resolve_components(
        &request(&limits),
        &metadata,
        &limits,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    metadata.finished();
    let replaced = vec![b'x'; B.len()];
    let bodies = Script::new(
        vec![
            Row::body("first", 11, A),
            Row::body("second", 12, &replaced),
        ],
        &admission,
    );
    assert!(
        execute_components(
            &store,
            &plan,
            &bodies,
            &admission,
            &Arc::new(AtomicBool::new(false))
        )
        .is_err()
    );
    bodies.finished();
    assert_eq!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old
    );
    let snapshots = store.list_snapshots().unwrap();
    let earlier = snapshots
        .iter()
        .filter_map(|digest| {
            let lease = store.open_snapshot(digest).unwrap();
            (lease.manifest().source().revision() == "release-91-asset-11").then_some(lease)
        })
        .collect::<Vec<_>>();
    assert_eq!(earlier.len(), 1);
    assert_eq!(file_bytes(earlier[0].file("model.gguf").unwrap()), A);
    assert!(!snapshots.iter().any(|digest| {
        store
            .open_snapshot(digest)
            .unwrap()
            .manifest()
            .source()
            .revision()
            == "release-91-asset-12"
    }));
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}
#[test]
fn malformed_or_missing_later_asset_refuses_during_metadata_only() {
    let limits = PullLimits::default();
    let mut state = asset(12, "config.json", B);
    state["state"] = json!("new");
    let mut digest_bad = asset(12, "config.json", B);
    digest_bad["digest"] = json!("sha1:untrusted-secret");
    for assets in [
        json!([]),
        json!([asset(12, "other.txt", B)]),
        json!([asset(12, "config.json", B), asset(13, "config.json", B)]),
        json!([asset(12, "config.json", B), asset(12, "other.txt", B)]),
        json!([asset(0, "config.json", B)]),
        json!([state]),
        json!([digest_bad]),
    ] {
        let admission = admission(128 * 1024 * 1024);
        let mut rows = metadata_rows();
        rows[2].body = serde_json::to_vec(&assets).unwrap();
        let script = Script::new(rows, &admission);
        assert!(
            resolve_components(
                &request(&limits),
                &script,
                &limits,
                &admission,
                &AtomicBool::new(false)
            )
            .is_err(),
            "{assets}"
        );
        assert_eq!(script.calls.load(Ordering::Relaxed), 3);
        assert_no_host_reservation(&admission);
    }
}
#[test]
fn shared_request_byte_and_file_limits_and_cancel_precede_payload() {
    for change in 0..4 {
        let mut limits = PullLimits::default();
        match change {
            0 => limits.max_pages = 3,
            1 => limits.max_metadata_bytes = 64,
            2 => limits.cache.max_file_bytes = (A.len() - 1) as u64,
            _ => limits.cache.max_snapshot_bytes = (A.len() + B.len() + C.len() - 1) as u64,
        }
        let admission = admission(128 * 1024 * 1024);
        let script = Script::new(metadata_rows(), &admission);
        assert!(
            resolve_components(
                &request(&limits),
                &script,
                &limits,
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(script.calls.load(Ordering::Relaxed) <= 4);
        assert_no_host_reservation(&admission);
    }
    let limits = PullLimits::default();
    let denied = admission(1);
    let script = Script::new(Vec::new(), &denied);
    assert!(matches!(
        resolve_components(
            &request(&limits),
            &script,
            &limits,
            &denied,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    assert_no_host_reservation(&denied);
    let admission = admission(128 * 1024 * 1024);
    let script = Script::new(Vec::new(), &admission);
    assert!(matches!(
        resolve_components(
            &request(&limits),
            &script,
            &limits,
            &admission,
            &AtomicBool::new(true)
        ),
        Err(Error::State(_))
    ));
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    let mut script = Script::new(metadata_rows(), &admission);
    script.cancel_after = Some(1);
    assert!(
        resolve_components(
            &request(&limits),
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 1);
    assert_no_host_reservation(&admission);
}
#[test]
fn different_requested_tags_and_repositories_keep_independent_release_pins() {
    let limits = PullLimits::default();
    let admission = admission(128 * 1024 * 1024);
    let mut members = members();
    members.extend([
        component("other_tag", REPO, "v2", "c.txt"),
        component("other_repo", "other/repo", TAG, "d.txt"),
    ]);
    let request = ComponentPullRequest::new(members, &limits).unwrap();
    let mut rows = metadata_rows();
    rows.extend([
        Row::json(
            "other_tag",
            &format!("/repos/{REPO}/releases/tags/v2"),
            release(92, "v2"),
        ),
        Row::json(
            "other_tag",
            &format!("/repos/{REPO}/releases/92/assets?per_page=100&page=1"),
            json!([asset(21, "c.txt", b"c")]),
        ),
        Row::json(
            "other_repo",
            "/repos/other/repo/releases/tags/stable",
            release(93, TAG),
        ),
        Row::json(
            "other_repo",
            "/repos/other/repo/releases/93/assets?per_page=100&page=1",
            json!([asset(31, "d.txt", b"d")]),
        ),
    ]);
    let script = Script::new(rows, &admission);
    let plan = resolve_components(
        &request,
        &script,
        &limits,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    script.finished();
    assert_eq!(
        plan.components()
            .iter()
            .map(|c| c.plan().source().revision())
            .collect::<Vec<_>>(),
        vec![
            "release-91-asset-11",
            "release-91-asset-12",
            "release-91-asset-13",
            "release-92-asset-21",
            "release-93-asset-31"
        ]
    );
    assert_eq!(
        plan.components()[4].plan().source().repository(),
        "other/repo"
    );
    assert_no_host_reservation(&admission);
}
#[test]
fn repeated_asset_pages_consume_one_shared_request_budget() {
    let page: Vec<_> = (0..100)
        .map(|n| asset(1000 + n, &format!("unused{n}.txt"), b"x"))
        .collect();
    let rows = || {
        let mut rows = vec![Row::json(
            "first",
            &format!("/repos/{REPO}/releases/tags/{TAG}"),
            release(91, TAG),
        )];
        for label in ["first", "second", "third"] {
            rows.extend([
                Row::json(
                    label,
                    &format!("/repos/{REPO}/releases/91/assets?per_page=100&page=1"),
                    json!(page),
                ),
                Row::json(
                    label,
                    &format!("/repos/{REPO}/releases/91/assets?per_page=100&page=2"),
                    assets(),
                ),
            ]);
        }
        rows
    };
    for max_pages in [7, 6] {
        let limits = PullLimits {
            max_pages,
            ..Default::default()
        };
        let admission = admission(128 * 1024 * 1024);
        let script = Script::new(rows(), &admission);
        let result = resolve_components(
            &request(&limits),
            &script,
            &limits,
            &admission,
            &AtomicBool::new(false),
        );
        assert_eq!(result.is_ok(), max_pages == 7, "{result:?}");
        assert_eq!(script.calls.load(Ordering::Relaxed), max_pages);
        if max_pages == 7 {
            script.finished();
        }
        assert_no_host_reservation(&admission);
    }
}
