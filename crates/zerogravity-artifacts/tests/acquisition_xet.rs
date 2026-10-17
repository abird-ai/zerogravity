//! Tiny native wire scripts through the original cache admission/publisher.
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
use zerogravity_artifacts::acquisition::{
    Provider, PullLimits, PullRequest, Selection, execute, execute_with_progress,
};
use zerogravity_artifacts::cache::{CommitOutcome, DigestProvenance, TransferAdmission};
use zerogravity_artifacts::{Error, Result};

const PIN: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const NATIVE_PATH: &str = "payload.dat";
fn assert_native_payload(lease: &zerogravity_artifacts::cache::SnapshotLease, expected: &[u8]) {
    assert_eq!(file_bytes(lease.file(NATIVE_PATH).unwrap()), expected);
}
// Frozen independent scalar BLAKE3 oracle, after official keyed/unkeyed vectors.
const ABC_FILE: &str = "55b9df318a7c45f470d57b04d8ca79a15b0637ffe94b25041a58d00c3528afff";
const ABC_TWICE: &str = "9d1b355c4f3b15db0956f1d0f9338ddf303492290f550689666535057b522f85";
const LONG_FILE: &str = "c591135b587d04a52721338102602e8731a4e5b967af004d0b845c81e02c5a2b";
fn request() -> PullRequest {
    PullRequest {
        provider: Provider::HuggingFace,
        repository: "owner/repo".into(),
        revision: "main".into(),
        selection: Selection::Files {
            paths: vec![NATIVE_PATH.into()],
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
    fn refused(suffix: &str, status: u16) -> Self {
        Self {
            suffix: suffix.into(),
            status,
            headers: BTreeMap::new(),
            body: vec![],
            never: true,
            cut: None,
        }
    }
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
fn rows(
    data: &[u8],
    file: &str,
    chunks: &[&[u8]],
    selected: (u32, u32),
    repeats: usize,
) -> Vec<Row> {
    let encoded: Vec<u8> = chunks.iter().flat_map(|raw| chunk(raw)).collect();
    let end = encoded.len() - 1;
    let xorb = "1".repeat(64);
    let signed = format!(
        "https://transfer.example/xorb?X-Xet-Signed-Range=bytes%3D0-{end}&Signature=NEVER_PERSIST"
    );
    let mut groups = serde_json::Map::new();
    groups.insert(xorb.clone(),serde_json::json!([{"url":signed,"ranges":[{"chunks":{"start":0,"end":chunks.len()},"bytes":{"start":0,"end":end}}]}]));
    let term = serde_json::json!({"hash":xorb,"unpacked_length":data.len()/repeats,"range":{"start":selected.0,"end":selected.1}});
    let terms: Vec<_> = (0..repeats).map(|_| term.clone()).collect();
    let mut result = vec![
        Row::json(
            "/api/models/owner/repo/revision/main".into(),
            serde_json::json!({"sha":PIN}),
        ),
        Row::json(
            format!("/api/models/owner/repo/tree/{PIN}?recursive=true&expand=false&limit=1000"),
            serde_json::json!([{"type":"file","path":NATIVE_PATH,"size":data.len(),"lfs":{"oid":digest(data),"size":data.len()}}]),
        ),
        Row {
            suffix: format!("/owner/repo/resolve/{PIN}/{NATIVE_PATH}"),
            status: 302,
            headers: BTreeMap::from([
                ("x-xet-hash".into(), file.into()),
                ("x-repo-commit".into(), PIN.into()),
                ("x-linked-size".into(), data.len().to_string()),
            ]),
            body: vec![],
            never: true,
            cut: None,
        },
        Row::json(
            format!("/api/models/owner/repo/xet-read-token/{PIN}"),
            serde_json::json!({"accessToken":"CAS_NEVER_PERSIST","exp":3_000_000_000u64,"casUrl":"https://cas-server.xethub.hf.co"}),
        ),
        Row::json(
            format!("/v2/reconstructions/{file}"),
            serde_json::json!({"offset_into_first_range":0,"terms":terms,"xorbs":groups}),
        ),
    ];
    for _ in 0..repeats {
        result.push(Row {
            suffix: format!("/xorb?X-Xet-Signed-Range=bytes%3D0-{end}&Signature=NEVER_PERSIST"),
            status: 206,
            headers: BTreeMap::from([
                (
                    "content-range".into(),
                    format!("bytes 0-{end}/{}", encoded.len()),
                ),
                ("content-length".into(), encoded.len().to_string()),
            ]),
            body: encoded.clone(),
            never: false,
            cut: None,
        });
    }
    result
}
fn resolve(
    script: &Script,
    options: &XetOptions,
    admission: &TransferAdmission,
) -> zerogravity_artifacts::acquisition::ResolvedPullPlan {
    xet::resolve(
        script,
        &request(),
        &PullLimits::default(),
        options,
        admission,
        &AtomicBool::new(false),
    )
    .unwrap()
}

#[test]
fn default_native_envelope_refuses_128mib_and_succeeds_with_256mib_without_reducing_limits() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let small = admission(128 * 1024 * 1024);
    let old = durable(publish(&store, &small, OLD, Some("keep")).unwrap());
    let script = Script::new(rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1), &small);
    let plan = resolve(&script, &XetOptions::default(), &small);
    let error = execute(
        &store,
        &plan,
        &script,
        &small,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Limit(_)));
    assert_eq!(
        script.calls.load(Ordering::Relaxed),
        3,
        "admission must refuse before token/CAS/signed payload attempts"
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
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&small);
    let sufficient = admission(256 * 1024 * 1024);
    let same = Script::new(
        rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1)
            .into_iter()
            .skip(3)
            .collect(),
        &sufficient,
    );
    execute(
        &store,
        &plan,
        &same,
        &sufficient,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
    )
    .unwrap();
    same.empty();
    assert_native_payload(&store.resolve_ref("keep").unwrap(), b"abc");
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&sufficient);
}

#[test]
fn discovery_identity_contradictions_refuse_without_compatibility_body_or_client_fallback() {
    for mode in [
        "absent",
        "noncanonical",
        "commit",
        "size",
        "raw-sha",
        "not-announced",
    ] {
        let admission = admission(256 * 1024 * 1024);
        let mut source: Vec<_> = rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1)
            .into_iter()
            .take(3)
            .collect();
        match mode {
            "absent" => {
                source[2].headers.remove("x-xet-hash");
            }
            "noncanonical" => {
                source[2]
                    .headers
                    .insert("x-xet-hash".into(), "A".repeat(64));
            }
            "commit" => {
                source[2]
                    .headers
                    .insert("x-repo-commit".into(), "b".repeat(40));
            }
            "size" => {
                source[2].headers.insert("x-linked-size".into(), "4".into());
            }
            "raw-sha" => {
                source[2]
                    .headers
                    .insert("x-linked-etag".into(), format!("\"{}\"", "f".repeat(64)));
            }
            "not-announced" => source[2].status = 200,
            _ => unreachable!(),
        }
        let script = Script::new(source, &admission);
        assert!(
            xet::resolve(
                &script,
                &request(),
                &PullLimits::default(),
                &XetOptions::default(),
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        script.empty();
        assert_no_host_reservation(&admission);
    }
    struct LegacyOnly;
    impl HttpTransport for LegacyOnly {
        fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("missing typed target capability cannot fall back to ordinary GET")
        }
    }
    let admission = admission(256 * 1024 * 1024);
    assert!(matches!(
        xet::resolve(
            &LegacyOnly,
            &request(),
            &PullLimits::default(),
            &XetOptions::default(),
            &admission,
            &AtomicBool::new(false)
        ),
        Err(Error::Unsupported(_))
    ));
    assert_no_host_reservation(&admission);
}

#[test]
fn explicit_native_route_requires_at_least_one_announced_lfs_object() {
    let admission = admission(256 * 1024 * 1024);
    let mut source: Vec<_> = rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1)
        .into_iter()
        .take(2)
        .collect();
    source[1] = Row::json(
        source[1].suffix.clone(),
        serde_json::json!([{"type":"file","path":NATIVE_PATH,"size":3}]),
    );
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
    script.empty();
    assert_no_host_reservation(&admission);
}

#[test]
fn declared_git_sidecar_keeps_durable_suffix_resume_and_shared_native_attempt_budget() {
    use zerogravity_artifacts::cache::{DiscoveredEntry, FileRole, ManifestSource};
    struct WithOrdinary {
        native: Script,
        bytes: Vec<u8>,
        calls: AtomicUsize,
    }
    impl HttpTransport for WithOrdinary {
        fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("ordinary native sidecar still uses budgeted original target")
        }
        fn get_target(
            &self,
            request: &HttpRequest,
            target: &HttpTarget,
            cancel: &AtomicBool,
        ) -> Result<HttpResponse> {
            if !request.url.ends_with("/sidecar.txt") {
                return self.native.get_target(request, target, cancel);
            }
            target.begin_attempt(cancel)?;
            self.calls.fetch_add(1, Ordering::Relaxed);
            let range = request
                .range
                .as_ref()
                .expect("verified proper prefix must request suffix");
            assert_eq!(range.start, 128);
            assert_eq!(range.total, self.bytes.len() as u64);
            assert_eq!(range.validator, "\"sidecar-stable\"");
            assert_eq!(request.max_body_bytes, (self.bytes.len() - 128) as u64);
            let headers = BTreeMap::from([
                ("etag".into(), "\"sidecar-stable\"".into()),
                (
                    "content-length".into(),
                    (self.bytes.len() - 128).to_string(),
                ),
                (
                    "content-range".into(),
                    format!("bytes 128-{}/{}", self.bytes.len() - 1, self.bytes.len()),
                ),
            ]);
            target.charge_metadata(128)?;
            Ok(HttpResponse {
                status: 206,
                headers,
                body: Box::new(Body {
                    bytes: Cursor::new(self.bytes[128..].to_vec()),
                    target: target.clone(),
                    cut: None,
                    never: false,
                    read: self.native.read.clone(),
                }),
            })
        }
    }
    for maximum in [6, 7] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let bytes: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let mut saved = store
            .begin_discovered(
                ManifestSource::new("huggingface", "owner/repo", PIN).unwrap(),
                vec![
                    DiscoveredEntry::new(
                        "sidecar.txt",
                        bytes.len() as u64,
                        FileRole::Auxiliary,
                        None,
                    )
                    .unwrap(),
                ],
                &admission,
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        let mut prefix = saved
            .partial_object("sidecar.txt", &format!("hf:owner/repo:{PIN}:sidecar.txt"))
            .unwrap();
        prefix.set_validator("\"sidecar-stable\"").unwrap();
        prefix.write_chunk(&bytes[..128]).unwrap();
        prefix.checkpoint().unwrap();
        drop(prefix);
        drop(saved);
        let mut source = rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1);
        let mut tree: serde_json::Value = serde_json::from_slice(&source[1].body).unwrap();
        tree.as_array_mut()
            .unwrap()
            .push(serde_json::json!({"type":"file","path":"sidecar.txt","size":bytes.len()}));
        source[1] = Row::json(source[1].suffix.clone(), tree);
        let transport = WithOrdinary {
            native: Script::new(source, &admission),
            bytes: bytes.clone(),
            calls: AtomicUsize::new(0),
        };
        let mut requested = request();
        let Selection::Files { paths, .. } = &mut requested.selection else {
            unreachable!()
        };
        paths.push("sidecar.txt".into());
        let mut options = XetOptions::default();
        options.max_requests = maximum;
        let plan = xet::resolve(
            &transport,
            &requested,
            &PullLimits::default(),
            &options,
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
        let mut resumed = 0;
        let result = execute_with_progress(
            &store,
            &plan,
            &transport,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("keep"),
            &mut |event| resumed = event.resumed_bytes(),
        );
        if maximum == 6 {
            assert!(
                matches!(result, Err(Error::State(ref text)) if text == "HTTP object request failed")
            );
            assert_eq!(transport.calls.load(Ordering::Relaxed), 0);
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
        } else {
            result.unwrap();
            assert_eq!(transport.calls.load(Ordering::Relaxed), 1);
            assert_eq!(resumed, 128);
            let lease = store.resolve_ref("keep").unwrap();
            assert_native_payload(&lease, b"abc");
            assert_eq!(file_bytes(lease.file("sidecar.txt").unwrap()), bytes);
        }
        transport.native.empty();
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn native_weights_keep_the_original_complete_config_and_tokenizer_closure() {
    use zerogravity_artifacts::cache::FileRole;
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let mut source = rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1);
    source[1] = Row::json(
        source[1].suffix.clone(),
        serde_json::json!([
            {"type":"file","path":"model.gguf","size":3,"lfs":{"oid":digest(b"abc"),"size":3}},
            {"type":"file","path":"config.json","size":2},
            {"type":"file","path":"tokenizer.json","size":2}
        ]),
    );
    source[2].suffix = format!("/owner/repo/resolve/{PIN}/model.gguf");
    source.insert(
        3,
        Row::json(
            format!("/owner/repo/resolve/{PIN}/config.json"),
            serde_json::json!({}),
        ),
    );
    source.push(Row::json(
        format!("/owner/repo/resolve/{PIN}/tokenizer.json"),
        serde_json::json!({}),
    ));
    let script = Script::new(source, &admission);
    let mut requested = request();
    requested.selection = Selection::Files {
        paths: vec!["model.gguf".into()],
        subfolder: None,
    };
    let plan = xet::resolve(
        &script,
        &requested,
        &PullLimits::default(),
        &XetOptions::default(),
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(plan.files().len(), 3);
    execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
    )
    .unwrap();
    let lease = store.resolve_ref("keep").unwrap();
    assert_eq!(file_bytes(lease.file("model.gguf").unwrap()), b"abc");
    assert_eq!(file_bytes(lease.file("config.json").unwrap()), b"{}");
    assert_eq!(file_bytes(lease.file("tokenizer.json").unwrap()), b"{}");
    for (path, role) in [
        ("model.gguf", FileRole::Weights),
        ("config.json", FileRole::Config),
        ("tokenizer.json", FileRole::Tokenizer),
    ] {
        assert_eq!(
            lease
                .manifest()
                .entries()
                .iter()
                .find(|entry| entry.path() == path)
                .unwrap()
                .role(),
            role
        );
    }
    assert_eq!(script.calls.load(Ordering::Relaxed), 8);
    script.empty();
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn native_resolution_is_metadata_only_and_never_persists_signed_grants() {
    let admission = admission(256 * 1024 * 1024);
    let script = Script::new(
        rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1)
            .into_iter()
            .take(3)
            .collect(),
        &admission,
    );
    let plan = resolve(&script, &XetOptions::default(), &admission);
    script.empty();
    assert_eq!(plan.total_bytes(), Some(3));
    assert_eq!(
        plan.files()[0].digest_provenance(),
        DigestProvenance::TrustedSha256
    );
    for secret in ["CAS_NEVER_PERSIST", "Signature=", "transfer.example"] {
        assert!(!format!("{plan:?}").contains(secret));
    }
    assert_no_host_reservation(&admission);
}

#[test]
fn native_exact_bytes_publish_once_and_verified_cache_hit_needs_no_grant() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let script = Script::new(rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1), &admission);
    let plan = resolve(&script, &XetOptions::default(), &admission);
    let mut progress = Vec::new();
    let receipt = execute_with_progress(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
        &mut |event| {
            progress.push((
                event.file_bytes(),
                event.resumed_bytes(),
                event.cached_bytes(),
                event.file_complete(),
            ))
        },
    )
    .unwrap();
    script.empty();
    assert_eq!(
        progress,
        [(0, 0, 0, false), (3, 0, 0, false), (3, 0, 0, true)]
    );
    let snapshot = match receipt.outcome() {
        CommitOutcome::Durable(snapshot) => snapshot.digest(),
        _ => panic!("fixture commit must be durable"),
    };
    let lease = store.open_snapshot(snapshot).unwrap();
    assert_native_payload(&lease, b"abc");
    xet::validate_cached_selection(
        &lease,
        &request(),
        &PullLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    let mut limits = PullLimits::default();
    limits.cache.max_file_bytes = 2;
    assert!(
        xet::validate_cached_selection(&lease, &request(), &limits, &AtomicBool::new(false))
            .is_err()
    );
    let before = script.calls.load(Ordering::Relaxed);
    execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap();
    assert_eq!(script.calls.load(Ordering::Relaxed), before);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn transport_edges_do_not_enter_logical_hash_and_repeated_terms_do() {
    for (data, file, chunks, selected, repeats) in [
        (
            b"abc".as_slice(),
            ABC_FILE,
            vec![b"edge".as_slice(), b"abc".as_slice(), b"tail".as_slice()],
            (1, 2),
            1,
        ),
        (
            b"abcabc".as_slice(),
            ABC_TWICE,
            vec![b"abc".as_slice()],
            (0, 1),
            2,
        ),
    ] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let script = Script::new(rows(data, file, &chunks, selected, repeats), &admission);
        let plan = resolve(&script, &XetOptions::default(), &admission);
        let receipt = execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
        )
        .unwrap();
        assert_eq!(receipt.files()[0].bytes(), data.len() as u64);
        script.empty();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn actual_decode_bound_counts_discarded_edges_before_successful_eof() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let script = Script::new(
        rows(b"abc", ABC_FILE, &[b"edge", b"abc", b"tail"], (1, 2), 1),
        &admission,
    );
    let mut options = XetOptions::default();
    options.max_chunks = 2;
    let plan = resolve(&script, &options, &admission);
    assert!(
        execute(
            &store,
            &plan,
            &script,
            &admission,
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
    assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn decoded_chunk_refetch_compares_accepted_leaves_without_duplicate_progress() {
    for changed_replay in [false, true] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let mut source = rows(b"abcabc", ABC_TWICE, &[b"abc", b"abc"], (0, 2), 1);
        let interrupted = Row {
            suffix: source[5].suffix.clone(),
            status: 206,
            headers: source[5].headers.clone(),
            body: source[5].body.clone(),
            never: false,
            cut: Some(20),
        };
        // First full11B chunk has already contributed its leaf. Interrupt after
        // the second8B header and1payload byte, then replay the complete group.
        if changed_replay {
            source[5].body[8..11].copy_from_slice(b"xyz");
        }
        source.insert(5, interrupted);
        let script = Script::new(source, &admission);
        let plan = resolve(&script, &XetOptions::default(), &admission);
        let mut accepted = Vec::new();
        let result = execute_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("keep"),
            &mut |event| accepted.push(event.file_bytes()),
        );
        if changed_replay {
            assert!(result.is_err());
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
        } else {
            result.unwrap();
            assert_eq!(accepted, [0, 6, 6]);
            assert_native_payload(&store.resolve_ref("keep").unwrap(), b"abcabc");
        }
        assert_eq!(script.calls.load(Ordering::Relaxed), 7);
        script.empty();
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn full_reader_reorders_multiple_parts_by_reissuing_exact_signed_sets() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let mut source = rows(b"abcabc", ABC_TWICE, &[b"abc", b"abc"], (0, 2), 1);
    let suffix = "/xorb?X-Xet-Signed-Range=bytes%3D0-10%2C11-21&Signature=NEVER_PERSIST";
    let mut metadata: serde_json::Value = serde_json::from_slice(&source[4].body).unwrap();
    let groups = metadata["xorbs"].as_object_mut().unwrap();
    let group = &mut groups.values_mut().next().unwrap()[0];
    group["url"] = serde_json::json!(format!("https://transfer.example{suffix}"));
    group["ranges"] = serde_json::json!([
        {"chunks":{"start":0,"end":1},"bytes":{"start":0,"end":10}},
        {"chunks":{"start":1,"end":2},"bytes":{"start":11,"end":21}}
    ]);
    source[4] = Row::json(source[4].suffix.clone(), metadata);
    source.truncate(5);
    for _ in 0..2 {
        let mut body = Vec::new();
        // The later logical descriptor arrives first on both full-set requests.
        for (start, end) in [(11, 21), (0, 10)] {
            body.extend_from_slice(
                format!("--fixture\r\nContent-Range: bytes {start}-{end}/22\r\n\r\n").as_bytes(),
            );
            body.extend(chunk(b"abc"));
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--fixture--\r\n");
        source.push(Row {
            suffix: suffix.into(),
            status: 206,
            headers: BTreeMap::from([
                (
                    "content-type".into(),
                    "multipart/byteranges; boundary=fixture".into(),
                ),
                ("content-length".into(), body.len().to_string()),
            ]),
            body,
            never: false,
            cut: None,
        });
    }
    let script = Script::new(source, &admission);
    let plan = resolve(&script, &XetOptions::default(), &admission);
    execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
    )
    .unwrap();
    assert_native_payload(&store.resolve_ref("keep").unwrap(), b"abcabc");
    assert_eq!(script.calls.load(Ordering::Relaxed), 7);
    script.empty();
    assert_no_host_reservation(&admission);
}

#[test]
fn framed_lz4_and_bg4_publish_only_decoded_bytes_under_independent_file_identity() {
    let raw = [b"abc".repeat(6), b"END!!".to_vec()].concat();
    // Frozen scalar oracle after official vectors; no decoder-under-test used.
    let identity = "0758a14ef418f95226029f9eb3e249213b2530b6e54500935805b9a6825c12fa";
    for (codec, hex) in [
        (
            1u8,
            "04224d186040820c0000003b616263030050454e44212100000000",
        ),
        (
            2u8,
            "04224d1860408217000080616263616244626361626321636162634521616263614e00000000",
        ),
    ] {
        let frame: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let compressed = (frame.len() as u32).to_le_bytes();
        let bytes = (raw.len() as u32).to_le_bytes();
        let mut wire = vec![
            0,
            compressed[0],
            compressed[1],
            compressed[2],
            codec,
            bytes[0],
            bytes[1],
            bytes[2],
        ];
        wire.extend(frame);
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let mut source = rows(&raw, identity, &[&raw], (0, 1), 1);
        let end = wire.len() - 1;
        let suffix = format!("/xorb?X-Xet-Signed-Range=bytes%3D0-{end}&Signature=NEVER_PERSIST");
        let mut metadata: serde_json::Value = serde_json::from_slice(&source[4].body).unwrap();
        let group = &mut metadata["xorbs"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap()[0];
        group["url"] = serde_json::json!(format!("https://transfer.example{suffix}"));
        group["ranges"][0]["bytes"]["end"] = serde_json::json!(end);
        source[4] = Row::json(source[4].suffix.clone(), metadata);
        source[5] = Row {
            suffix,
            status: 206,
            headers: BTreeMap::from([
                (
                    "content-range".into(),
                    format!("bytes 0-{end}/{}", wire.len()),
                ),
                ("content-length".into(), wire.len().to_string()),
            ]),
            body: wire,
            never: false,
            cut: None,
        };
        let script = Script::new(source, &admission);
        let plan = resolve(&script, &XetOptions::default(), &admission);
        execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("keep"),
        )
        .unwrap();
        assert_native_payload(&store.resolve_ref("keep").unwrap(), &raw);
        script.empty();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn long_independent_keyed_hash_vector_crosses_native_update_and_tree_boundaries() {
    let data: Vec<u8> = (0..128 * 1024).map(|i| (i % 251) as u8).collect();
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let script = Script::new(rows(&data, LONG_FILE, &[&data], (0, 1), 1), &admission);
    let plan = resolve(&script, &XetOptions::default(), &admission);
    let receipt = execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap();
    assert_eq!(receipt.files()[0].bytes(), data.len() as u64);
    script.empty();
    assert_no_host_reservation(&admission);
}

#[test]
fn pre_canceled_or_initial_callback_canceled_native_execute_never_requests_a_grant() {
    for initially in [true, false] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let script = Script::new(rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1), &admission);
        let plan = resolve(&script, &XetOptions::default(), &admission);
        let cancel = Arc::new(AtomicBool::new(initially));
        let mut events = 0;
        assert!(
            execute_with_progress(
                &store,
                &plan,
                &script,
                &admission,
                &cancel,
                Some("keep"),
                &mut |event| {
                    assert_eq!(event.file_bytes(), 0);
                    events += 1;
                    cancel.store(true, Ordering::Release);
                }
            )
            .is_err()
        );
        assert_eq!(events, usize::from(!initially));
        assert_eq!(script.calls.load(Ordering::Relaxed), 3);
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
}

#[test]
fn final_progress_cancel_preserves_old_ref_and_does_not_save_a_native_prefix() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let script = Script::new(rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1), &admission);
    let plan = resolve(&script, &XetOptions::default(), &admission);
    let cancel = Arc::new(AtomicBool::new(false));
    assert!(
        execute_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &cancel,
            Some("keep"),
            &mut |event| {
                if event.file_bytes() == 3 {
                    cancel.store(true, Ordering::Release);
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
    let retry = Script::new(
        rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1)
            .into_iter()
            .skip(3)
            .collect(),
        &admission,
    );
    execute(
        &store,
        &plan,
        &retry,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap();
    retry.empty();
    assert_no_host_reservation(&admission);
}

#[test]
fn authorization_and_signed_reissues_keep_the_same_original_logical_source() {
    for mode in ["cas401", "signed403", "after-leaf", "v1-fallback"] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let (data, id, chunks, span) = if mode == "after-leaf" {
            (
                b"abcabc".as_slice(),
                ABC_TWICE,
                vec![b"abc".as_slice(), b"abc".as_slice()],
                (0, 2),
            )
        } else {
            (b"abc".as_slice(), ABC_FILE, vec![b"abc".as_slice()], (0, 1))
        };
        let mut source = rows(data, id, &chunks, span, 1);
        match mode {
            "cas401" => {
                let token = source[3].clone();
                let denied = Row::refused(&source[4].suffix, 401);
                source.insert(4, denied);
                source.insert(5, token);
            }
            "signed403" | "after-leaf" => {
                let mut fresh = source[4].clone();
                let mut metadata: serde_json::Value = serde_json::from_slice(&fresh.body).unwrap();
                let url = &mut metadata["xorbs"]
                    .as_object_mut()
                    .unwrap()
                    .values_mut()
                    .next()
                    .unwrap()[0]["url"];
                *url = serde_json::json!(
                    url.as_str()
                        .unwrap()
                        .replace("NEVER_PERSIST", "ROTATED_NEVER_PERSIST")
                );
                fresh = Row::json(fresh.suffix, metadata);
                let denied = Row::refused(&source[5].suffix, 403);
                source[5].suffix = source[5]
                    .suffix
                    .replace("NEVER_PERSIST", "ROTATED_NEVER_PERSIST");
                if mode == "after-leaf" {
                    let mut interrupted = source[5].clone();
                    interrupted.suffix = denied.suffix.clone();
                    interrupted.cut = Some(20);
                    source.insert(5, interrupted);
                    source.insert(6, denied);
                    source.insert(7, fresh);
                } else {
                    source.insert(5, denied);
                    source.insert(6, fresh);
                }
            }
            "v1-fallback" => {
                let original: serde_json::Value = serde_json::from_slice(&source[4].body).unwrap();
                let (key, group) = original["xorbs"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .next()
                    .unwrap();
                let mut fetch = serde_json::Map::new();
                fetch.insert(key.clone(),serde_json::json!([{"range":group[0]["ranges"][0]["chunks"],"url":group[0]["url"],"url_range":group[0]["ranges"][0]["bytes"]}]));
                let v1 = Row::json(
                    format!("/v1/reconstructions/{id}"),
                    serde_json::json!({"offset_into_first_range":0,"terms":original["terms"],"fetch_info":fetch}),
                );
                source[4] = Row::refused(&source[4].suffix, 404);
                source.insert(5, v1);
            }
            _ => unreachable!(),
        }
        let expected = source.len();
        let script = Script::new(source, &admission);
        let plan = resolve(&script, &XetOptions::default(), &admission);
        execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("keep"),
        )
        .unwrap();
        assert_native_payload(&store.resolve_ref("keep").unwrap(), data);
        assert_eq!(script.calls.load(Ordering::Relaxed), expected);
        script.empty();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn changed_reissue_or_refused_read_scope_never_falls_back_or_overwrites_old_ref() {
    for mode in ["changed", "no-refresh", "cas403", "v2-500"] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let mut source = rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1);
        let mut options = XetOptions::default();
        match mode {
            "changed" => {
                let mut metadata: serde_json::Value =
                    serde_json::from_slice(&source[4].body).unwrap();
                let groups = metadata["xorbs"].as_object_mut().unwrap();
                let key = groups.keys().next().unwrap().clone();
                let group = groups.remove(&key).unwrap();
                let changed = "2".repeat(64);
                groups.insert(changed.clone(), group);
                metadata["terms"][0]["hash"] = serde_json::json!(changed);
                let fresh = Row::json(source[4].suffix.clone(), metadata);
                let denied = Row::refused(&source[5].suffix, 403);
                source.insert(5, denied);
                source.insert(6, fresh);
            }
            "no-refresh" => {
                options.max_refreshes = 0;
                source.insert(5, Row::refused(&source[5].suffix, 403));
            }
            "cas403" => source[4] = Row::refused(&source[4].suffix, 403),
            "v2-500" => source[4] = Row::refused(&source[4].suffix, 500),
            _ => unreachable!(),
        }
        let script = Script::new(source, &admission);
        let plan = resolve(&script, &options, &admission);
        assert!(
            execute(
                &store,
                &plan,
                &script,
                &admission,
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
        assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn accepted_reconstruction_cardinality_does_not_reset_on_signed_reissue() {
    for dimension in ["terms", "groups", "ranges"] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let mut source = rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1);
        let fresh = source[4].clone();
        let denied = Row::refused(&source[5].suffix, 403);
        source.insert(5, denied);
        source.insert(6, fresh);
        let mut options = XetOptions::default();
        match dimension {
            "terms" => options.max_terms = 1,
            "groups" => options.max_fetch_groups = 1,
            "ranges" => options.max_ranges = 1,
            _ => unreachable!(),
        }
        let script = Script::new(source, &admission);
        let plan = resolve(&script, &options, &admission);
        assert!(
            execute(
                &store,
                &plan,
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                Some("keep")
            )
            .is_err()
        );
        assert_eq!(script.calls.load(Ordering::Relaxed), 7);
        assert_eq!(
            script.rows.lock().unwrap().len(),
            1,
            "second payload must remain unrequested after individually fitting documents jointly exceed native control cardinality"
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
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn source_hash_codec_expiry_and_actual_attempt_failures_preserve_old_reference() {
    for failure in ["filehash", "rawhash", "codec", "expiry", "attempts"] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(256 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let wrong = "e".repeat(64);
        let mut source = rows(
            b"abc",
            if failure == "filehash" {
                &wrong
            } else {
                ABC_FILE
            },
            &[b"abc"],
            (0, 1),
            1,
        );
        let mut options = XetOptions::default();
        match failure {
            "rawhash" => {
                let mut tree: serde_json::Value = serde_json::from_slice(&source[1].body).unwrap();
                tree[0]["lfs"]["oid"] = serde_json::json!("f".repeat(64));
                source[1] = Row::json(source[1].suffix.clone(), tree);
            }
            "codec" => source[5].body[4] = 9,
            "expiry" => {
                source[3] = Row::json(
                    source[3].suffix.clone(),
                    serde_json::json!({"accessToken":"CAS_NEVER_PERSIST","exp":1,"casUrl":"https://cas-server.xethub.hf.co"}),
                )
            }
            "attempts" => options.max_requests = 3,
            _ => {}
        }
        let script = Script::new(source, &admission);
        let plan = resolve(&script, &options, &admission);
        let error = execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("keep"),
        )
        .unwrap_err();
        assert!(!format!("{error:?}").contains("NEVER_PERSIST"));
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
}

const CHILD_ROOT: &str = "ZEROGRAVITY_XET_EPHEMERAL_CHILD_ROOT";
#[test]
#[ignore = "invoked only by the bounded native process-exit fixture"]
fn native_ephemeral_child_worker() {
    let Some(root) = std::env::var_os(CHILD_ROOT) else {
        return;
    };
    let store = zerogravity_artifacts::cache::CacheStore::open(
        zerogravity_artifacts::cache::CacheOptions::new(Some(root.into()), DISK_LIMIT),
    )
    .unwrap();
    let admission = admission(256 * 1024 * 1024);
    let data: Vec<u8> = (0..128 * 1024).map(|i| (i % 251) as u8).collect();
    let script = Script::new(rows(&data, LONG_FILE, &[&data], (0, 1), 1), &admission);
    let plan = resolve(&script, &XetOptions::default(), &admission);
    let _ = execute_with_progress(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
        &mut |event| {
            if event.file_bytes() >= 128 {
                std::process::exit(73);
            }
        },
    );
    panic!("native worker did not reach its written prefix exit");
}

#[test]
fn actual_process_exit_leaves_charge_but_fresh_native_invocation_starts_at_zero() {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let before = store.usage().unwrap().used_bytes;
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "native_ephemeral_child_worker",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ROOT, &fixture.root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("bounded native child timed out");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        output.status.code(),
        Some(73),
        "{}",
        String::from_utf8_lossy(&output.stderr)
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
    let after = store.usage().unwrap();
    assert!(after.used_bytes > before);
    assert_eq!(after.reserved_bytes, 0);
    let data: Vec<u8> = (0..128 * 1024).map(|i| (i % 251) as u8).collect();
    let script = Script::new(rows(&data, LONG_FILE, &[&data], (0, 1), 1), &admission);
    let plan = resolve(&script, &XetOptions::default(), &admission);
    let mut resumed = Vec::new();
    execute_with_progress(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
        &mut |event| resumed.push(event.resumed_bytes()),
    )
    .unwrap();
    assert!(resumed.iter().all(|bytes| *bytes == 0));
    assert_eq!(script.calls.load(Ordering::Relaxed), 6);
    script.empty();
    assert_native_payload(&store.resolve_ref("keep").unwrap(), &data);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_loopback_native_flow_uses_scoped_grants_and_credential_free_signed_body() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let mut source = rows(b"abc", ABC_FILE, &[b"abc"], (0, 1), 1);
    source[3] = Row::json(
        source[3].suffix.clone(),
        serde_json::json!({
            "accessToken":"CAS_NEVER_PERSIST","exp":3_000_000_000u64,"casUrl":origin
        }),
    );
    let mut reconstruction: serde_json::Value = serde_json::from_slice(&source[4].body).unwrap();
    reconstruction["xorbs"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .next()
        .unwrap()[0]["url"] = serde_json::json!(format!("{origin}{}", source[5].suffix));
    source[4] = Row::json(source[4].suffix.clone(), reconstruction);
    let captured = Arc::new(Mutex::new(Vec::new()));
    let copies = captured.clone();
    let worker = std::thread::spawn(move || {
        for row in source {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(_) => panic!("bounded native fixture accept failed"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut wire = Vec::new();
            let mut byte = [0u8; 1];
            while !wire.ends_with(b"\r\n\r\n") {
                assert!(wire.len() < 96 * 1024);
                assert_eq!(stream.read(&mut byte).unwrap(), 1);
                wire.push(byte[0]);
            }
            let wire = String::from_utf8(wire).unwrap();
            assert!(
                wire.starts_with(&format!("GET {} HTTP/1.1\r\n", row.suffix)),
                "native fixture received a changed immutable target"
            );
            copies.lock().unwrap().push(wire);
            let mut head = format!("HTTP/1.1 {} Fixture\r\nConnection: close\r\n", row.status);
            if !row.headers.contains_key("content-length") {
                head.push_str(&format!("Content-Length: {}\r\n", row.body.len()));
            }
            for (name, value) in row.headers {
                head.push_str(&format!("{name}: {value}\r\n"));
            }
            head.push_str("\r\n");
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(&row.body).unwrap();
        }
    });
    let options = XetOptions::default()
        .diagnostic_endpoints(&origin, &origin)
        .unwrap();
    let client = HttpsClient::diagnostic_loopback(
        &origin,
        Some("HUB_SECRET".into()),
        HttpOptions {
            backoff: Duration::ZERO,
            ..HttpOptions::default()
        },
    )
    .unwrap();
    let plan = xet::resolve(
        &client,
        &request(),
        &PullLimits::default(),
        &options,
        &admission,
        &AtomicBool::new(false),
    )
    .unwrap();
    execute(
        &store,
        &plan,
        &client,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
    )
    .unwrap();
    worker.join().unwrap();
    let wire = captured.lock().unwrap();
    assert_eq!(wire.len(), 6);
    for request in &wire[..4] {
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer hub_secret")
        );
    }
    assert!(
        wire[4]
            .to_ascii_lowercase()
            .contains("authorization: bearer cas_never_persist")
    );
    assert!(!wire[4].contains("HUB_SECRET"));
    let signed = wire[5].to_ascii_lowercase();
    assert!(signed.contains("range: bytes=0-10"));
    for name in [
        "authorization:",
        "cookie:",
        "x-amz-security-token:",
        "x-ms-date:",
    ] {
        assert!(!signed.contains(name));
    }
    assert_native_payload(&store.resolve_ref("keep").unwrap(), b"abc");
    assert_no_host_reservation(&admission);
}
