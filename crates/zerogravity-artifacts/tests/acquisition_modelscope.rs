//! Independent ModelScope pin/completeness/selection and original-publisher tests.
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
use zerogravity_artifacts::acquisition::http::{
    HttpRequest, HttpResponse, HttpTarget, HttpTransport,
};
use zerogravity_artifacts::acquisition::modelscope::ModelScopeRequest;
use zerogravity_artifacts::acquisition::{PullLimits, execute, execute_with_progress};
use zerogravity_artifacts::cache::{
    DigestProvenance, FileRole, Manifest, ManifestEntry, TransferAdmission,
};
use zerogravity_artifacts::{Error, Result};
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const ORIGIN: &str = "https://modelscope.cn";
static SCHEDULE: Mutex<()> = Mutex::new(());
fn request(paths: &[&str]) -> ModelScopeRequest {
    ModelScopeRequest::new(
        ORIGIN,
        "owner/repo",
        COMMIT,
        paths.iter().map(|s| (*s).into()).collect(),
        None,
    )
    .unwrap()
}
fn row(path: &str, bytes: &[u8], trusted: bool) -> Value {
    let mut row = json!({"Path":path,"Name":path.rsplit('/').next().unwrap(),"Type":"blob","Size":bytes.len()});
    if trusted {
        row["Sha256"] = json!(digest(bytes));
    }
    row
}
fn directory(path: &str) -> Value {
    json!({"Path":path,"Name":path.rsplit('/').next().unwrap(),"Type":"tree"})
}
struct Reply {
    parent: Option<String>,
    file: Option<String>,
    status: u16,
    bytes: Vec<u8>,
}
fn tree(parent: &str, rows: Vec<Value>) -> Reply {
    Reply {
        parent: Some(parent.into()),
        file: None,
        status: 200,
        bytes: serde_json::to_vec(&json!({"Code":200,"Success":true,"Data":{"Files":rows}}))
            .unwrap(),
    }
}
fn raw(path: &str, bytes: &[u8]) -> Reply {
    Reply {
        parent: None,
        file: Some(path.into()),
        status: 200,
        bytes: bytes.to_vec(),
    }
}
struct Script {
    replies: Mutex<VecDeque<Reply>>,
    admission: TransferAdmission,
    payload: Arc<AtomicUsize>,
}
impl Script {
    fn new(replies: Vec<Reply>, admission: &TransferAdmission) -> Self {
        Self {
            replies: Mutex::new(replies.into()),
            admission: admission.clone(),
            payload: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn finished(&self) {
        assert!(self.replies.lock().unwrap().is_empty());
    }
}
struct Counted {
    data: Cursor<Vec<u8>>,
    count: Arc<AtomicUsize>,
}
impl Read for Counted {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let n = self.data.read(out)?;
        self.count.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }
}
impl HttpTransport for Script {
    fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("ModelScope must use original no-redirect target capability")
    }
    fn get_target(
        &self,
        req: &HttpRequest,
        target: &HttpTarget,
        _: &AtomicBool,
    ) -> Result<HttpResponse> {
        assert!(!format!("{target:?}").contains("api/v1"));
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0,
            "admission before metadata and body"
        );
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        let url = url::Url::parse(&req.url).unwrap();
        assert_eq!(url.origin().ascii_serialization(), ORIGIN);
        let query = url.query_pairs().collect::<BTreeMap<_, _>>();
        assert_eq!(query.get("Revision").unwrap(), COMMIT);
        assert!(req.range.is_none());
        let count = if let Some(parent) = reply.parent {
            assert_eq!(url.path(), "/api/v1/models/owner/repo/repo/files");
            assert_eq!(query.get("Recursive").unwrap(), "False");
            assert_eq!(
                query.get("Root").map(|s| s.as_ref()),
                if parent.is_empty() {
                    None
                } else {
                    Some(parent.as_str())
                }
            );
            Arc::new(AtomicUsize::new(0))
        } else {
            assert_eq!(url.path(), "/api/v1/models/owner/repo/repo");
            assert_eq!(query.get("FilePath").unwrap().as_ref(), reply.file.unwrap());
            self.payload.clone()
        };
        Ok(HttpResponse {
            status: reply.status,
            headers: BTreeMap::from([("content-length".into(), reply.bytes.len().to_string())]),
            body: Box::new(Counted {
                data: Cursor::new(reply.bytes),
                count,
            }),
        })
    }
}
fn snapshot(receipt: &zerogravity_artifacts::acquisition::PullReceipt) -> String {
    match receipt.outcome() {
        zerogravity_artifacts::cache::CommitOutcome::Durable(s) => s.digest().into(),
        other => panic!("{other:?}"),
    }
}
#[test]
fn complete_nested_tree_exact_glob_subfolder_zero_and_endpoint_pin_survive_offline() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let f = Fixture::new();
    let store = f.open();
    let a = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let req = ModelScopeRequest::new(
        ORIGIN,
        "owner/repo",
        COMMIT,
        vec!["*.txt".into()],
        Some("nested".into()),
    )
    .unwrap();
    let script = Script::new(
        vec![
            tree("", vec![directory("nested"), directory("empty")]),
            tree(
                "nested",
                vec![
                    row("nested/a.txt", NEW, true),
                    row("nested/zero.txt", b"", false),
                ],
            ),
            tree("empty", vec![]),
            raw("nested/a.txt", NEW),
            raw("nested/zero.txt", b""),
        ],
        &a,
    );
    let plan = req
        .resolve(&script, &limits, &a, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(plan.source().revision(), COMMIT);
    assert!(plan.source().repository().ends_with("/owner/repo"));
    assert_eq!(plan.files().len(), 2);
    assert_eq!(plan.files()[1].bytes(), Some(0));
    let receipt = execute(
        &store,
        &plan,
        &script,
        &a,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap();
    script.finished();
    let id = snapshot(&receipt);
    let lease = req
        .open_cached(&store, &id, &limits, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(file_bytes(lease.file("nested/a.txt").unwrap()), NEW);
    assert_eq!(
        lease.manifest().entries()[0].digest_provenance(),
        DigestProvenance::TrustedSha256
    );
    assert_eq!(
        lease.manifest().entries()[1].digest_provenance(),
        DigestProvenance::FirstSeen
    );
    assert!(
        req.open_cached(&store, "alias", &limits, &AtomicBool::new(false))
            .is_err()
    );
    assert!(
        request(&["other.txt"])
            .open_cached(&store, &id, &limits, &AtomicBool::new(false))
            .is_err()
    );
    let other = ModelScopeRequest::new(
        ORIGIN,
        "owner/other",
        COMMIT,
        vec!["nested/*.txt".into()],
        None,
    )
    .unwrap();
    assert!(
        other
            .open_cached(&store, &id, &limits, &AtomicBool::new(false))
            .is_err()
    );
    assert_no_host_reservation(&a);
}
#[test]
fn bounded_walk_refuses_truncation_even_when_selected_file_is_visible() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let a = admission(128 * 1024 * 1024);
    for n in [3000, 3001] {
        let rows = (0..n)
            .map(|i| row(&format!("a{i}.txt"), b"", false))
            .collect();
        let script = Script::new(vec![tree("", rows)], &a);
        assert!(
            request(&["a0.txt"])
                .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
                .is_err()
        );
        script.finished();
        assert_eq!(script.payload.load(Ordering::Relaxed), 0);
    }
    let script = Script::new(
        vec![tree(
            "",
            vec![row("a.txt", NEW, false), directory("unseen")],
        )],
        &a,
    );
    let limits = PullLimits {
        max_pages: 1,
        ..Default::default()
    };
    assert!(matches!(
        request(&["a.txt"]).resolve(&script, &limits, &a, &AtomicBool::new(false)),
        Err(Error::Limit(_))
    ));
    script.finished();
}
#[test]
fn unsafe_duplicate_wrong_parent_types_sizes_and_secret_json_refuse() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let a = admission(128 * 1024 * 1024);
    let cases = vec![
        vec![row("a.txt", NEW, false), row("a.txt", NEW, false)],
        vec![row("nested/a.txt", NEW, false)],
        vec![row("../escape", NEW, false)],
        vec![json!({"Path":"a.txt","Type":"link","Size":1})],
        vec![json!({"Path":"a.txt","Type":"blob","Size":null})],
        vec![json!({"Path":"a.txt","Type":"blob","Size":-1})],
        vec![json!({"Path":"a.txt","Name":"other","Type":"blob","Size":0})],
        vec![json!({"Path":"a.txt","Type":"blob","Size":0,"Sha256":"notSHA"})],
        vec![directory("a"), row("a", NEW, false)],
    ];
    for rows in cases {
        let script = Script::new(vec![tree("", rows)], &a);
        assert!(
            request(&["a.txt"])
                .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
                .is_err()
        );
        script.finished();
    }
    for bytes in [
        br#"{"Code":200,"Success":true,"Data":{"Files":"SECRET"}}"#.as_slice(),
        br#"{"Code":200,"Success":true,"Data":null}"#,
        br#"{"Code":200,"Success":true,"Success":false,"Data":[]}"#,
    ] {
        let mut reply = tree("", vec![]);
        reply.bytes = bytes.to_vec();
        let script = Script::new(vec![reply], &a);
        let error = request(&["a.txt"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
            .unwrap_err();
        assert!(!error.to_string().contains("SECRET"));
    }
}
#[test]
fn existing_shared_model_closure_adds_all_index_shards_sidecars_and_split_parts() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let a = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let index=br#"{"metadata":{"total_size":63},"weight_map":{"a":"part-00001-of-00002.safetensors","b":"part-00002-of-00002.safetensors"}}"#;
    let script = Script::new(
        vec![
            tree(
                "",
                vec![
                    row("model.safetensors.index.json", index, true),
                    row("part-00001-of-00002.safetensors", NEW, false),
                    row("part-00002-of-00002.safetensors", OLD, false),
                    row("config.json", b"{}", false),
                    row("tokenizer.json", b"{}", false),
                ],
            ),
            raw("model.safetensors.index.json", index),
        ],
        &a,
    );
    let plan = request(&["part-00001-of-00002.safetensors"])
        .resolve(&script, &limits, &a, &AtomicBool::new(false))
        .unwrap();
    script.finished();
    assert_eq!(plan.files().len(), 5);
    let files = vec![
        row("m-00001-of-00002.gguf", NEW, false),
        row("m-00002-of-00002.gguf", OLD, false),
        row("config.json", b"{}", false),
        row("tokenizer.json", b"{}", false),
    ];
    let script = Script::new(vec![tree("", files.clone())], &a);
    assert_eq!(
        request(&["m-00001-of-00002.gguf"])
            .resolve(&script, &limits, &a, &AtomicBool::new(false))
            .unwrap()
            .files()
            .len(),
        4
    );
    for missing in [0, 2, 3] {
        let mut rows = files.clone();
        rows.remove(missing);
        let script = Script::new(vec![tree("", rows)], &a);
        assert!(
            request(&["m-00002-of-00002.gguf"])
                .resolve(&script, &limits, &a, &AtomicBool::new(false))
                .is_err()
        );
    }
    let script = Script::new(
        vec![tree(
            "",
            vec![
                row("a.gguf", NEW, false),
                row("b.gguf", NEW, false),
                row("config.json", b"{}", false),
                row("tokenizer.json", b"{}", false),
            ],
        )],
        &a,
    );
    assert!(
        request(&["a.gguf", "b.gguf"])
            .resolve(&script, &limits, &a, &AtomicBool::new(false))
            .is_err()
    );
}
#[test]
fn shared_index_metadata_budget_digest_and_lfs_checks_precede_any_weight_body() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let a = admission(128 * 1024 * 1024);
    for bad in [
        b"version https://git-lfs.github.com/spec/v1\n".as_slice(),
        b"{\"weight_map\":\"SECRET\"}",
    ] {
        let script = Script::new(
            vec![
                tree("", vec![row("m.safetensors.index.json", bad, false)]),
                raw("m.safetensors.index.json", bad),
            ],
            &a,
        );
        let err = request(&["m.safetensors.index.json"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
            .unwrap_err();
        assert!(!err.to_string().contains("SECRET"));
        script.finished();
    }
    let rows = vec![row("m.safetensors.index.json", NEW, true)];
    let mut corrupt = raw("m.safetensors.index.json", NEW);
    corrupt.bytes[0] ^= 1;
    let script = Script::new(vec![tree("", rows.clone()), corrupt], &a);
    assert!(
        request(&["m.safetensors.index.json"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
            .is_err()
    );
    script.finished();
    let script = Script::new(vec![tree("", rows)], &a);
    let limits = PullLimits {
        max_pages: 1,
        ..Default::default()
    };
    assert!(matches!(
        request(&["m.safetensors.index.json"]).resolve(
            &script,
            &limits,
            &a,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    script.finished();
}
#[test]
fn denied_admission_precancel_and_invalid_pin_are_before_any_request() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let a = admission(1);
    let script = Script::new(vec![], &a);
    assert!(
        request(&["a.txt"])
            .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
            .is_err()
    );
    let a = admission(128 * 1024 * 1024);
    let script = Script::new(vec![], &a);
    assert!(matches!(
        request(&["a.txt"]).resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(true)),
        Err(Error::State(_))
    ));
    for commit in [
        "master",
        "v1",
        "abc123",
        "0123456789ABCDEF0123456789ABCDEF01234567",
    ] {
        assert!(
            ModelScopeRequest::new(ORIGIN, "owner/repo", commit, vec!["a.txt".into()], None)
                .is_err()
        );
    }
    for origin in [
        "https://modelscope.ai",
        "https://modelscope.cn/",
        "https://modelscope.cn/../",
        "https://modelscope.cn?secret=x",
    ] {
        assert!(
            ModelScopeRequest::new(origin, "owner/repo", COMMIT, vec!["a.txt".into()], None)
                .is_err()
        );
    }
    script.finished();
    assert_no_host_reservation(&a);
}
#[test]
fn actual_body_hash_length_lfs_and_post_read_cancel_preserve_old_ref() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let f = Fixture::new();
    let store = f.open();
    let a = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let old = durable(publish(&store, &a, OLD, Some("keep")).unwrap());
    for bad in [
        b"wrong".as_slice(),
        &NEW[..NEW.len() - 1],
        b"version https://git-lfs.github.com/spec/v1\n",
    ] {
        let script = Script::new(
            vec![tree("", vec![row("a.txt", NEW, true)]), raw("a.txt", bad)],
            &a,
        );
        let plan = request(&["a.txt"])
            .resolve(&script, &limits, &a, &AtomicBool::new(false))
            .unwrap();
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
        assert_eq!(
            store
                .resolve_ref("keep")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            old
        );
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&a);
    }
    let script = Script::new(
        vec![tree("", vec![row("a.txt", NEW, false)]), raw("a.txt", NEW)],
        &a,
    );
    let plan = request(&["a.txt"])
        .resolve(&script, &limits, &a, &AtomicBool::new(false))
        .unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    assert!(
        execute_with_progress(
            &store,
            &plan,
            &script,
            &a,
            &cancel,
            Some("keep"),
            &mut |p| {
                if p.file_bytes() > 0 {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        )
        .is_err()
    );
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
#[test]
fn offline_rejects_wrong_role_unqualified_provenance_and_tighter_limits() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let f = Fixture::new();
    let store = f.open();
    let a = admission(128 * 1024 * 1024);
    let req = request(&["a.txt"]);
    for (role, provenance) in [
        (FileRole::Weights, DigestProvenance::FirstSeen),
        (FileRole::Auxiliary, DigestProvenance::Unspecified),
    ] {
        let manifest = Manifest::new(
            req.source().clone(),
            vec![
                ManifestEntry::new("a.txt", digest(NEW), NEW.len() as u64, role)
                    .unwrap()
                    .with_digest_provenance(provenance),
            ],
            &Default::default(),
        )
        .unwrap();
        let mut tx = store
            .begin(manifest, &a, Arc::new(AtomicBool::new(false)))
            .unwrap();
        let mut reader = NEW;
        tx.import_object("a.txt", &mut reader).unwrap();
        let id = durable(tx.commit(None).unwrap());
        assert!(
            req.open_cached(&store, &id, &PullLimits::default(), &AtomicBool::new(false))
                .is_err()
        );
    }
    let script = Script::new(
        vec![tree("", vec![row("a.txt", NEW, true)]), raw("a.txt", NEW)],
        &a,
    );
    let plan = req
        .resolve(&script, &PullLimits::default(), &a, &AtomicBool::new(false))
        .unwrap();
    let receipt = execute(
        &store,
        &plan,
        &script,
        &a,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap();
    let mut limits = PullLimits::default();
    limits.cache.max_file_bytes = 1;
    assert!(
        req.open_cached(
            &store,
            &snapshot(&receipt),
            &limits,
            &AtomicBool::new(false)
        )
        .is_err()
    );
}

#[test]
fn offline_manual_pointer_seed_cannot_masquerade_as_verified_provider_data() {
    let _schedule = SCHEDULE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let f = Fixture::new();
    let store = f.open();
    let a = admission(128 * 1024 * 1024);
    let request = request(&["a.txt"]);
    let pointer=b"version https://git-lfs.github.com/spec/v1\noid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nsize 123\n";
    let manifest = Manifest::new(
        request.source().clone(),
        vec![
            ManifestEntry::new(
                "a.txt",
                digest(pointer),
                pointer.len() as u64,
                FileRole::Auxiliary,
            )
            .unwrap()
            .with_digest_provenance(DigestProvenance::FirstSeen),
        ],
        &Default::default(),
    )
    .unwrap();
    let mut tx = store
        .begin(manifest, &a, Arc::new(AtomicBool::new(false)))
        .unwrap();
    let mut reader = pointer.as_slice();
    tx.import_object("a.txt", &mut reader).unwrap();
    let id = durable(tx.commit(None).unwrap());
    // Generic A1 data remains valid, but it did not pass the native provider's
    // body eligibility boundary. Exact source/role/hash alone is insufficient.
    assert_eq!(
        file_bytes(store.open_snapshot(&id).unwrap().file("a.txt").unwrap()),
        pointer
    );
    let error = request
        .open_cached(&store, &id, &PullLimits::default(), &AtomicBool::new(false))
        .expect_err("native offline provider must reject a generic pointer seed");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    assert_no_host_reservation(&a);
}
