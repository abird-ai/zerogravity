//! Native LFS-backed required safetensors indexes: full root/raw SHA/EOF before parser.
//! Custom transport explicitly charges opaque work capabilities; it does not
//! receive or implement the original owner's private CAS authentication profile.
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
use zerogravity_artifacts::acquisition::xet::{self, XetOptions};
use zerogravity_artifacts::acquisition::{Provider, PullLimits, PullRequest, Selection, execute};
use zerogravity_artifacts::cache::{CommitOutcome, TransferAdmission};
use zerogravity_artifacts::{Error, Result};

const PIN: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const WEIGHTS: &str = "weights.safetensors";
const INDEX: &str = "model.safetensors.index.json";
const WEIGHT_XET: &str = "55b9df318a7c45f470d57b04d8ca79a15b0637ffe94b25041a58d00c3528afff";
fn request() -> PullRequest {
    PullRequest {
        provider: Provider::HuggingFace,
        repository: "owner/repo".into(),
        revision: "main".into(),
        selection: Selection::Files {
            paths: vec![WEIGHTS.into()],
            subfolder: None,
        },
    }
}
#[derive(Clone)]
struct Row {
    suffix: String,
    status: u16,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
    never: bool,
    cut: Option<usize>,
}
impl Row {
    fn json(suffix: String, value: serde_json::Value) -> Self {
        let body = serde_json::to_vec(&value).unwrap();
        Self {
            suffix,
            status: 200,
            headers: BTreeMap::from([("content-length".into(), body.len().to_string())]),
            body,
            never: false,
            cut: None,
        }
    }
}
struct Body {
    bytes: Cursor<Vec<u8>>,
    target: HttpTarget,
    cut: Option<usize>,
    never: bool,
    read: Arc<AtomicUsize>,
}
impl Read for Body {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        assert!(
            !self.never,
            "refused/discovery response body must not be read"
        );
        if self.cut == Some(0) {
            return Err(io::Error::other("fixture transport interruption"));
        }
        let count = self
            .target
            .body_read_limit(output.len())
            .map_err(|_| io::Error::other("fixture native bound"))?
            .min(self.cut.unwrap_or(usize::MAX));
        let read = self.bytes.read(&mut output[..count])?;
        if let Some(left) = &mut self.cut {
            *left -= read;
        }
        self.read.fetch_add(read, Ordering::Relaxed);
        self.target
            .charge_body(read as u64)
            .map_err(|_| io::Error::other("fixture native bound"))?;
        Ok(read)
    }
}
struct Script {
    rows: Mutex<VecDeque<Row>>,
    calls: AtomicUsize,
    read: Arc<AtomicUsize>,
    admission: TransferAdmission,
}
impl Script {
    fn new(rows: Vec<Row>, admission: &TransferAdmission) -> Self {
        Self {
            rows: Mutex::new(rows.into()),
            calls: AtomicUsize::new(0),
            read: Arc::new(AtomicUsize::new(0)),
            admission: admission.clone(),
        }
    }
    fn empty(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("native operations must use the typed original owner capability")
    }
    fn get_target(
        &self,
        request: &HttpRequest,
        target: &HttpTarget,
        cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        target.begin_attempt(cancel)?;
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0
        );
        assert!(
            request.range.is_none(),
            "native streams never use suffix resume"
        );
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected native request");
        assert!(
            request.url.ends_with(&row.suffix),
            "native request differs from pinned script"
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        let bytes = row
            .headers
            .iter()
            .map(|(name, value)| name.len() + value.len() + 4)
            .sum::<usize>()
            + 32;
        target.charge_metadata(bytes as u64)?;
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body: Box::new(Body {
                bytes: Cursor::new(row.body),
                target: target.clone(),
                cut: row.cut,
                never: row.never,
                read: self.read.clone(),
            }),
        })
    }
}
fn chunk(raw: &[u8]) -> Vec<u8> {
    let length = (raw.len() as u32).to_le_bytes();
    let mut result = vec![
        0, length[0], length[1], length[2], 0, length[0], length[1], length[2],
    ];
    result.extend_from_slice(raw);
    result
}
/// Independent direct BLAKE3 composition of protocol 1.1's single-chunk
/// Xet file root, checked against the published abc golden.
fn file_root(data: &[u8]) -> String {
    const DATA_KEY: [u8; 32] = [
        102, 151, 245, 119, 91, 149, 80, 222, 49, 53, 203, 172, 165, 151, 24, 28, 157, 228, 33, 16,
        155, 235, 43, 88, 180, 208, 176, 75, 147, 173, 242, 41,
    ];
    let chunk = blake3::keyed_hash(&DATA_KEY, data);
    let root = blake3::keyed_hash(&[0u8; 32], chunk.as_bytes());
    root.as_bytes()
        .chunks_exact(8)
        .flat_map(|limb| limb.iter().rev())
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
const INDEX_BYTES: &[u8] =
    br#"{"metadata":{"total_size":3},"weight_map":{"tensor":"weights.safetensors"}}"#;

fn discovery(path: &str, data: &[u8], override_root: Option<&str>) -> Row {
    Row {
        suffix: format!("/owner/repo/resolve/{PIN}/{path}"),
        status: 302,
        headers: BTreeMap::from([
            (
                "x-xet-hash".into(),
                override_root.map_or_else(|| file_root(data), str::to_owned),
            ),
            ("x-repo-commit".into(), PIN.into()),
            ("x-linked-size".into(), data.len().to_string()),
        ]),
        body: Vec::new(),
        never: true,
        cut: None,
    }
}
fn native_body(data: &[u8]) -> Vec<Row> {
    let root = file_root(data);
    let encoded = chunk(data);
    let end = encoded.len() - 1;
    let xorb = "1".repeat(64);
    let signed = format!(
        "https://transfer.example/xorb?X-Xet-Signed-Range=bytes%3D0-{end}&Signature=NEVER_PERSIST"
    );
    let mut groups = serde_json::Map::new();
    groups.insert(
        xorb.clone(),
        serde_json::json!([{"url":signed,"ranges":[{"chunks":{"start":0,"end":1},"bytes":{"start":0,"end":end}}]}]),
    );
    vec![
        Row::json(
            format!("/api/models/owner/repo/xet-read-token/{PIN}"),
            serde_json::json!({"accessToken":"CAS_NEVER_PERSIST","exp":3_000_000_000u64,"casUrl":"https://cas-server.xethub.hf.co"}),
        ),
        Row::json(
            format!("/v2/reconstructions/{root}"),
            serde_json::json!({"offset_into_first_range":0,"terms":[{"hash":xorb,"unpacked_length":data.len(),"range":{"start":0,"end":1}}],"xorbs":groups}),
        ),
        Row {
            suffix: format!("/xorb?X-Xet-Signed-Range=bytes%3D0-{end}&Signature=NEVER_PERSIST"),
            status: 206,
            headers: BTreeMap::from([
                (
                    "content-range".into(),
                    format!("bytes 0-{end}/{}", encoded.len()),
                ),
                ("content-length".into(), encoded.len().to_string()),
            ]),
            body: encoded,
            never: false,
            cut: None,
        },
    ]
}
fn planned_index(
    raw: &[u8],
    index_sha: &str,
    include_shard: bool,
    index_xet_root: Option<&str>,
) -> Vec<Row> {
    let mut tree = vec![
        serde_json::json!({
            "type":"file","path":INDEX,"size":raw.len(),
            "lfs":{"oid":index_sha,"size":raw.len()}
        }),
        serde_json::json!({"type":"file","path":"config.json","size":2}),
        serde_json::json!({"type":"file","path":"tokenizer.json","size":2}),
    ];
    if include_shard {
        tree.push(serde_json::json!({
            "type":"file","path":WEIGHTS,"size":3,
            "lfs":{"oid":digest(b"abc"),"size":3}
        }));
    }
    vec![
        Row::json(
            "/api/models/owner/repo/revision/main".into(),
            serde_json::json!({"sha":PIN}),
        ),
        Row::json(
            format!("/api/models/owner/repo/tree/{PIN}?recursive=true&expand=false&limit=1000"),
            serde_json::json!(tree),
        ),
        discovery(INDEX, raw, index_xet_root),
    ]
}
fn resolved_index_rows() -> Vec<Row> {
    let mut out = planned_index(INDEX_BYTES, &digest(INDEX_BYTES), true, None);
    out.extend(native_body(INDEX_BYTES)); // C4 planning reads complete native index
    out.push(discovery(WEIGHTS, b"abc", None));
    out
}
fn payload_rows() -> Vec<Row> {
    let mut out = vec![Row::json(
        format!("/owner/repo/resolve/{PIN}/config.json"),
        serde_json::json!({}),
    )];
    out.extend(native_body(INDEX_BYTES)); // C4 publisher reconstitutes same index again
    out.push(Row::json(
        format!("/owner/repo/resolve/{PIN}/tokenizer.json"),
        serde_json::json!({}),
    ));
    out.extend(native_body(b"abc")); // same original content, source-specific root
    out
}
#[test]
fn independent_protocol_file_root_checks_the_original_abc_golden() {
    assert_eq!(file_root(b"abc"), WEIGHT_XET);
    assert_ne!(file_root(INDEX_BYTES), WEIGHT_XET);
}
#[test]
fn required_native_index_default_and_dryrun_refuse_before_any_cas_grant() {
    let admission = admission(256 * 1024 * 1024);
    let source = planned_index(INDEX_BYTES, &digest(INDEX_BYTES), true, None);
    let script = Script::new(source, &admission);
    assert!(matches!(
        xet::resolve(
            &script,
            &request(),
            &PullLimits::default(),
            &XetOptions::default(),
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(script.calls.load(Ordering::Relaxed), 2);
    assert_eq!(script.rows.lock().unwrap().len(), 1); // even discovery remains unread
    assert_no_host_reservation(&admission);
}
#[test]
fn exact_native_index_is_verified_before_shards_and_published_again_in_original_cache() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let mut rows = resolved_index_rows();
    rows.extend(payload_rows());
    let script = Script::new(rows, &admission);
    let options = XetOptions::default().with_index_reconstruction();
    let plan = xet::resolve(
        &script,
        &request(),
        &PullLimits::default(),
        &options,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(plan.files().len(), 4);
    assert_eq!(plan.source().revision(), PIN);
    assert_eq!(script.calls.load(Ordering::Relaxed), 7); // pin/tree/index3/weight discovery
    let display = format!("{plan:?}");
    for secret in ["CAS_NEVER_PERSIST", "Signature=", "transfer.example"] {
        assert!(!display.contains(secret));
    }
    let result = execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
    )
    .unwrap();
    assert!(matches!(result.outcome(), CommitOutcome::Durable(_)));
    assert_ne!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old
    );
    let lease = store.resolve_ref("keep").unwrap();
    assert_eq!(file_bytes(lease.file(INDEX).unwrap()), INDEX_BYTES);
    assert_eq!(file_bytes(lease.file(WEIGHTS).unwrap()), b"abc");
    assert_eq!(file_bytes(lease.file("config.json").unwrap()), b"{}");
    assert_eq!(file_bytes(lease.file("tokenizer.json").unwrap()), b"{}");
    script.empty();
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}
#[test]
fn original_admission_denies_index_scratch_before_read_grant_or_signed_body() {
    let small = admission(128 * 1024 * 1024);
    let script = Script::new(resolved_index_rows(), &small);
    assert!(matches!(
        xet::resolve(
            &script,
            &request(),
            &PullLimits::default(),
            &XetOptions::default().with_index_reconstruction(),
            &small,
            &AtomicBool::new(false)
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(script.calls.load(Ordering::Relaxed), 3);
    assert_eq!(script.rows.lock().unwrap().len(), 4); // no CAS-grant/signed body
    assert_no_host_reservation(&small);
}
#[test]
fn missing_pinned_shard_is_rejected_after_verified_index_before_any_shard_body() {
    let admission = admission(256 * 1024 * 1024);
    let mut rows = planned_index(INDEX_BYTES, &digest(INDEX_BYTES), false, None);
    rows.extend(native_body(INDEX_BYTES));
    let script = Script::new(rows, &admission);
    let mut requested = request();
    requested.selection = Selection::Files {
        paths: vec![INDEX.into()],
        subfolder: None,
    };
    assert!(
        xet::resolve(
            &script,
            &requested,
            &PullLimits::default(),
            &XetOptions::default().with_index_reconstruction(),
            &admission,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    script.empty();
    assert_no_host_reservation(&admission);
}

#[test]
fn wrong_raw_sha_wrong_file_root_or_invalid_json_never_parses_or_publishes() {
    for kind in [
        "wrong-raw-sha",
        "wrong-root",
        "invalid-json",
        "truncated-signed-data",
    ] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let raw = if kind == "invalid-json" {
            b"{bad-json".as_slice()
        } else {
            INDEX_BYTES
        };
        let incorrect_hash = "f".repeat(64);
        let sha = if kind == "wrong-raw-sha" {
            incorrect_hash.as_str()
        } else {
            &digest(raw)
        };
        let bad_root = (kind == "wrong-root").then_some(WEIGHT_XET);
        let mut planned = planned_index(raw, sha, true, bad_root);
        let mut streamed = native_body(raw);
        if kind == "wrong-root" {
            streamed[1].suffix = format!("/v2/reconstructions/{WEIGHT_XET}");
        }
        if kind == "truncated-signed-data" {
            streamed[2].cut = Some(1);
        }
        planned.extend(streamed);
        let script = Script::new(planned, &admission);
        let mut options = XetOptions::default().with_index_reconstruction();
        if kind == "truncated-signed-data" {
            // Six real metadata/grant/range attempts; a subsequent retry must
            // refuse at the ORIGINAL counter before another signed GET.
            options.max_requests = 6;
        }
        let result = xet::resolve(
            &script,
            &request(),
            &PullLimits::default(),
            &options,
            &admission,
            &AtomicBool::new(false),
        );
        assert!(
            result.is_err(),
            "{kind}: incorrectly accepted a broken native index"
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
        assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
        script.empty(); // no shard discovery or body before validated closure
        assert_no_host_reservation(&admission);
    }
}
#[test]
fn cumulative_native_index_then_publication_then_shard_cannot_reset_term_budget() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let mut rows = resolved_index_rows();
    rows.extend(payload_rows());
    let script = Script::new(rows, &admission);
    let mut options = XetOptions::default().with_index_reconstruction();
    options.max_terms = 2; // each object fits; planning+publication+weight do not
    let plan = xet::resolve(
        &script,
        &request(),
        &PullLimits::default(),
        &options,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    let result = execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
    );
    assert!(
        result.is_err(),
        "invocation budget must include pre-plan index work"
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 14);
    assert_eq!(script.rows.lock().unwrap().len(), 1); // signed shard body never read
    assert_eq!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old
    );
    assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}
#[test]
fn filters_exclude_required_adjacent_native_index_before_discovery_or_cas() {
    use zerogravity_artifacts::acquisition::{CandidateRequest, filters};
    let admission = admission(256 * 1024 * 1024);
    let planned = planned_index(INDEX_BYTES, &digest(INDEX_BYTES), true, None);
    let script = Script::new(planned, &admission);
    let request = filters::FilterRequest::new(
        CandidateRequest::new(Provider::HuggingFace, "owner/repo", "main", None).unwrap(),
        vec!["*.safetensors".into()],
        vec!["*.safetensors.index.json".into()],
    )
    .unwrap();
    assert!(matches!(
        xet::resolve_filtered(
            &script,
            &request,
            &PullLimits::default(),
            &XetOptions::default().with_index_reconstruction(),
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(script.calls.load(Ordering::Relaxed), 2);
    assert_eq!(script.rows.lock().unwrap().len(), 1);
    assert_no_host_reservation(&admission);
}
#[test]
fn cancellation_and_untrusted_index_sha_refuse_before_any_signed_body() {
    let admission = admission(256 * 1024 * 1024);
    let original = planned_index(INDEX_BYTES, &digest(INDEX_BYTES), true, None);
    let script = Script::new(original, &admission);
    assert!(
        xet::resolve(
            &script,
            &request(),
            &PullLimits::default(),
            &XetOptions::default().with_index_reconstruction(),
            &admission,
            &AtomicBool::new(true)
        )
        .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    let invalid = planned_index(INDEX_BYTES, "not-a-raw-sha", true, None);
    let script = Script::new(invalid, &admission);
    assert!(
        xet::resolve(
            &script,
            &request(),
            &PullLimits::default(),
            &XetOptions::default().with_index_reconstruction(),
            &admission,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    assert!(script.calls.load(Ordering::Relaxed) <= 3);
    assert_no_host_reservation(&admission);
}

#[test]
fn c3_filtered_native_selection_can_reconstruct_c4_index_without_repin_or_fallback() {
    use zerogravity_artifacts::acquisition::{CandidateRequest, filters};
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let mut rows = resolved_index_rows();
    rows.extend(payload_rows());
    let script = Script::new(rows, &admission);
    let request = filters::FilterRequest::new(
        CandidateRequest::new(Provider::HuggingFace, "owner/repo", "main", None).unwrap(),
        vec!["*.safetensors".into()],
        vec!["not-a-source*".into()],
    )
    .unwrap();
    let options = XetOptions::default().with_index_reconstruction();
    let plan = xet::resolve_filtered(
        &script,
        &request,
        &PullLimits::default(),
        &options,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(script.calls.load(Ordering::Relaxed), 7);
    assert_eq!(plan.files().len(), 4);
    execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("c3-c4"),
    )
    .unwrap();
    script.empty();
    let lease = store.resolve_ref("c3-c4").unwrap();
    assert_eq!(file_bytes(lease.file(INDEX).unwrap()), INDEX_BYTES);
    assert_eq!(file_bytes(lease.file(WEIGHTS).unwrap()), b"abc");
    assert_no_host_reservation(&admission);
}
