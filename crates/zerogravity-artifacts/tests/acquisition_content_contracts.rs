//! Explicit trusted raw content at mutable cloud locators, without metadata I/O.
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
use zerogravity_artifacts::acquisition::azure::{
    AzureContentSpec, AzureObjectPin, AzureObjectRequest, AzureRevision,
};
use zerogravity_artifacts::acquisition::http::{
    HttpRequest, HttpResponse, HttpTarget, HttpTransport,
};
use zerogravity_artifacts::acquisition::object::HttpsObjectSpec;
use zerogravity_artifacts::acquisition::s3::{
    S3Addressing, S3ContentSpec, S3Endpoint, S3ObjectPin, S3ObjectRequest,
};
use zerogravity_artifacts::acquisition::{
    PullLimits, ResolvedPullPlan, execute, execute_with_progress,
};
use zerogravity_artifacts::cache::{
    CacheLimits, CommitOutcome, DigestProvenance, FileRole, Manifest, ManifestEntry,
    ManifestSource, SnapshotLease, TransferAdmission,
};
use zerogravity_artifacts::{Error, Result};
const AZURE: &str = "https://fixtureaccount.blob.core.windows.net";
const S3: &str = "https://object.example";
const TAG: &str = "\"content-validator\"";
#[derive(Clone, Copy, Debug)]
enum Provider {
    S3,
    Azure,
}
impl Provider {
    fn spec(self, data: &[u8]) -> Spec {
        self.contract(data.len() as u64, &digest(data))
    }
    fn contract(self, bytes: u64, sha: &str) -> Spec {
        match self {
            Self::S3 => Spec::S3(
                S3ContentSpec::new(
                    &S3Endpoint::compatible(S3, "us-east-1", S3Addressing::PathStyle).unwrap(),
                    "bucket",
                    "folder/data.bin",
                    PAYLOAD_PATH,
                    FileRole::Auxiliary,
                    bytes,
                    sha,
                )
                .unwrap(),
            ),
            Self::Azure => Spec::Azure(
                AzureContentSpec::new(
                    AZURE,
                    "container",
                    "folder/data.bin",
                    PAYLOAD_PATH,
                    FileRole::Auxiliary,
                    bytes,
                    sha,
                )
                .unwrap(),
            ),
        }
    }
    fn row(self, data: &[u8]) -> Row {
        let mut headers = BTreeMap::from([
            ("content-length".into(), data.len().to_string()),
            ("etag".into(), TAG.into()),
        ]);
        if matches!(self, Self::Azure) {
            headers.insert("x-ms-blob-type".into(), "BlockBlob".into());
        }
        Row {
            headers,
            status: 200,
            body: data.to_vec(),
            range: None,
            cut: None,
            never: false,
        }
    }
    fn range(self, data: &[u8], start: usize) -> Row {
        let mut row = self.row(&data[start..]);
        row.range = Some(start as u64);
        row.status = 206;
        row.headers.insert(
            "content-range".into(),
            format!("bytes {start}-{}/{}", data.len() - 1, data.len()),
        );
        row
    }
    fn version_header(self) -> &'static str {
        match self {
            Self::S3 => "x-amz-version-id",
            Self::Azure => "x-ms-version-id",
        }
    }
}
enum Spec {
    S3(S3ContentSpec),
    Azure(AzureContentSpec),
}
impl Spec {
    fn prepare(
        &self,
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<ResolvedPullPlan> {
        match self {
            Self::S3(s) => s.prepare(limits, admission, cancel),
            Self::Azure(s) => s.prepare(limits, admission, cancel),
        }
    }
    fn source(&self) -> &ManifestSource {
        match self {
            Self::S3(s) => s.source(),
            Self::Azure(s) => s.source(),
        }
    }
    fn open(
        &self,
        store: &zerogravity_artifacts::cache::CacheStore,
        snapshot: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        match self {
            Self::S3(s) => s.open_cached(store, snapshot, limits, cancel),
            Self::Azure(s) => s.open_cached(store, snapshot, limits, cancel),
        }
    }
}
struct Row {
    headers: BTreeMap<String, String>,
    status: u16,
    body: Vec<u8>,
    range: Option<u64>,
    cut: Option<usize>,
    never: bool,
}
struct Never;
impl Read for Never {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("rejected body consumed")
    }
}
struct Body {
    data: Cursor<Vec<u8>>,
    bytes: Arc<AtomicUsize>,
    remaining: Option<usize>,
}
impl Read for Body {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.remaining == Some(0) {
            return Err(io::Error::other("synthetic interruption"));
        }
        let cap = self.remaining.unwrap_or(out.len()).min(out.len());
        let n = self.data.read(&mut out[..cap])?;
        if let Some(remaining) = &mut self.remaining {
            *remaining -= n;
        }
        self.bytes.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }
}
struct Script {
    provider: Provider,
    rows: Mutex<VecDeque<Row>>,
    admission: TransferAdmission,
    bytes: Arc<AtomicUsize>,
    calls: AtomicUsize,
}
impl Script {
    fn new(provider: Provider, rows: Vec<Row>, admission: &TransferAdmission) -> Self {
        Self {
            provider,
            rows: Mutex::new(rows.into()),
            admission: admission.clone(),
            bytes: Arc::new(AtomicUsize::new(0)),
            calls: AtomicUsize::new(0),
        }
    }
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("native contract lost typed target")
    }
    fn head(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("content contract is metadata-free")
    }
    fn head_target(&self, _: &HttpRequest, _: &HttpTarget, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("content contract is metadata-free")
    }
    fn get_target(
        &self,
        request: &HttpRequest,
        target: &HttpTarget,
        _: &AtomicBool,
    ) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0
        );
        assert_eq!(
            request.url,
            match self.provider {
                Provider::S3 => format!("{S3}/bucket/folder/data.bin"),
                Provider::Azure => format!("{AZURE}/container/folder/data.bin"),
            }
        );
        assert!(!format!("{target:?}").contains("folder/data.bin"));
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        assert_eq!(request.range.as_ref().map(|r| r.start), row.range);
        self.calls.fetch_add(1, Ordering::Relaxed);
        let body: Box<dyn Read + Send> = if row.never {
            Box::new(Never)
        } else {
            Box::new(Body {
                data: Cursor::new(row.body),
                bytes: self.bytes.clone(),
                remaining: row.cut,
            })
        };
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body,
        })
    }
}
fn snapshot(outcome: &CommitOutcome) -> String {
    match outcome {
        CommitOutcome::Durable(s) => s.digest().to_owned(),
        CommitOutcome::DurabilityUnknown { cause, .. } => panic!("{cause}"),
    }
}
fn seed(
    store: &zerogravity_artifacts::cache::CacheStore,
    admission: &TransferAdmission,
    source: &ManifestSource,
    body: &[u8],
    selected: (&str, FileRole, DigestProvenance),
    extra: bool,
) -> String {
    let (path, role, trust) = selected;
    let mut entries = vec![
        ManifestEntry::new(path, digest(body), body.len() as u64, role)
            .unwrap()
            .with_digest_provenance(trust),
    ];
    if extra {
        entries.push(
            ManifestEntry::new(
                "extra.bin",
                digest(body),
                body.len() as u64,
                FileRole::Auxiliary,
            )
            .unwrap()
            .with_digest_provenance(trust),
        );
    }
    let manifest = Manifest::new(source.clone(), entries, &CacheLimits::default()).unwrap();
    let id = manifest.digest().unwrap();
    let mut tx = store
        .begin(manifest, admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    tx.import_object(path, &mut Cursor::new(body)).unwrap();
    if extra {
        tx.import_object("extra.bin", &mut Cursor::new(body))
            .unwrap();
    }
    tx.commit(Some("existing-ref")).unwrap();
    id
}

#[test]
fn pure_content_identity_is_distinct_from_remote_version_and_binds_complete_contract() {
    let bytes = b"abc";
    let sha = digest(bytes);
    let endpoint = S3Endpoint::compatible(S3, "us-east-1", S3Addressing::PathStyle).unwrap();
    let s3 = S3ContentSpec::new(
        &endpoint,
        "bucket",
        "folder/data.bin",
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        3,
        &sha,
    )
    .unwrap();
    let old = S3ObjectRequest::new(
        &endpoint,
        "bucket",
        "folder/data.bin",
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        None,
        Some(&sha),
    )
    .unwrap();
    assert_ne!(
        s3.source(),
        S3ObjectPin::new(&old, "real-version", 3).unwrap().source()
    );
    assert_eq!(s3.request().requested_version(), None);
    let azure = AzureContentSpec::new(
        AZURE,
        "container",
        "folder/data.bin",
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        3,
        &sha,
    )
    .unwrap();
    let old = AzureObjectRequest::new(
        AZURE,
        "container",
        "folder/data.bin",
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        AzureRevision::version("real-version").unwrap(),
        Some(&sha),
    )
    .unwrap();
    assert_ne!(
        azure.source(),
        AzureObjectPin::new(&old, 3).unwrap().source()
    );
    for changed in [
        AzureContentSpec::new(
            AZURE,
            "container",
            "other.bin",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            3,
            &sha,
        )
        .unwrap(),
        AzureContentSpec::new(
            AZURE,
            "container",
            "folder/data.bin",
            "other.bin",
            FileRole::Auxiliary,
            3,
            &sha,
        )
        .unwrap(),
        AzureContentSpec::new(
            AZURE,
            "container",
            "folder/data.bin",
            PAYLOAD_PATH,
            FileRole::Config,
            3,
            &sha,
        )
        .unwrap(),
        AzureContentSpec::new(
            AZURE,
            "container",
            "folder/data.bin",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            4,
            &sha,
        )
        .unwrap(),
        AzureContentSpec::new(
            AZURE,
            "container",
            "folder/data.bin",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            3,
            &digest(b"def"),
        )
        .unwrap(),
    ] {
        assert_ne!(azure.source(), changed.source());
    }
    for bad in ["", "short", "UPPERCASE"] {
        assert!(
            S3ContentSpec::new(
                &endpoint,
                "bucket",
                "key",
                PAYLOAD_PATH,
                FileRole::Auxiliary,
                3,
                bad
            )
            .is_err()
        );
        assert!(
            AzureContentSpec::new(
                AZURE,
                "container",
                "key",
                PAYLOAD_PATH,
                FileRole::Auxiliary,
                3,
                bad
            )
            .is_err()
        );
    }
    assert!(!format!("{s3:?} {azure:?}").contains("folder/data.bin"));
}

#[test]
fn preparation_is_local_admitted_exact_and_cancellation_releases_control() {
    for provider in [Provider::S3, Provider::Azure] {
        let spec = provider.spec(b"abc");
        let denied = admission(1);
        assert!(matches!(
            spec.prepare(&PullLimits::default(), &denied, &AtomicBool::new(false)),
            Err(Error::Limit(_))
        ));
        assert_no_host_reservation(&denied);
        let allowed = admission(128 * 1024 * 1024);
        assert!(
            spec.prepare(&PullLimits::default(), &allowed, &AtomicBool::new(true))
                .is_err()
        );
        assert_no_host_reservation(&allowed);
        let plan = spec
            .prepare(&PullLimits::default(), &allowed, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(plan.total_bytes(), Some(3));
        assert_eq!(plan.files()[0].sha256(), Some(digest(b"abc").as_str()));
        assert_eq!(
            plan.files()[0].digest_provenance(),
            DigestProvenance::TrustedSha256
        );
        assert!(!plan.files()[0].url().contains('?'));
        let small = PullLimits {
            cache: CacheLimits {
                max_file_bytes: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            spec.prepare(&small, &allowed, &AtomicBool::new(false))
                .is_err()
        );
        assert_no_host_reservation(&allowed);
    }
}

#[test]
fn current_missing_null_or_present_version_headers_never_replace_trusted_raw_content() {
    for provider in [Provider::S3, Provider::Azure] {
        for version in [None, Some("null"), Some("current-opaque-version")] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let spec = provider.spec(NEW);
            let plan = spec
                .prepare(&PullLimits::default(), &admission, &AtomicBool::new(false))
                .unwrap();
            let mut row = provider.row(NEW);
            if let Some(version) = version {
                row.headers
                    .insert(provider.version_header().into(), version.into());
            }
            let script = Script::new(provider, vec![row], &admission);
            let cancel = Arc::new(AtomicBool::new(false));
            let receipt = execute(&store, &plan, &script, &admission, &cancel, None).unwrap();
            script.finished();
            assert_eq!(script.bytes.load(Ordering::Relaxed), NEW.len());
            let id = snapshot(receipt.outcome());
            assert_payload(
                &spec
                    .open(&store, &id, &PullLimits::default(), &cancel)
                    .unwrap(),
                NEW,
            );
            let cached = Script::new(provider, vec![], &admission);
            let mut cached_bytes = 0;
            execute_with_progress(
                &store,
                &plan,
                &cached,
                &admission,
                &cancel,
                None,
                &mut |p| cached_bytes = p.cached_bytes(),
            )
            .unwrap();
            cached.finished();
            assert_eq!(cached_bytes, NEW.len() as u64);
            assert_no_host_reservation(&admission);
        }
    }
}

#[test]
fn native_offline_content_rejects_refs_and_wrong_source_trust_path_role_size_digest_or_closure() {
    for provider in [Provider::S3, Provider::Azure] {
        for mode in 0..7 {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let spec = provider.spec(NEW);
            let source = if mode == 0 {
                ManifestSource::new("local", "wrong", "source").unwrap()
            } else {
                spec.source().clone()
            };
            let wrong_digest = vec![b'x'; NEW.len()];
            let body: &[u8] = if mode == 4 {
                b"wrong size"
            } else if mode == 5 {
                &wrong_digest
            } else {
                NEW
            };
            let id = seed(
                &store,
                &admission,
                &source,
                body,
                (
                    if mode == 2 { "wrong.bin" } else { PAYLOAD_PATH },
                    if mode == 3 {
                        FileRole::Config
                    } else {
                        FileRole::Auxiliary
                    },
                    if mode == 1 {
                        DigestProvenance::FirstSeen
                    } else {
                        DigestProvenance::TrustedSha256
                    },
                ),
                mode == 6,
            );
            assert!(
                spec.open(&store, &id, &PullLimits::default(), &AtomicBool::new(false))
                    .is_err(),
                "{provider:?} mode{mode}"
            );
            assert!(
                spec.open(
                    &store,
                    "existing-ref",
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            assert_no_host_reservation(&admission);
        }
    }
}

#[test]
fn shared_trusted_verifier_keeps_https_digest_and_ref_behavior_and_pointer_policy() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let spec = HttpsObjectSpec::new(
        "https://example.invalid",
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        NEW.len() as u64,
        digest(NEW),
    )
    .unwrap();
    let id = seed(
        &store,
        &admission,
        spec.source(),
        NEW,
        (
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            DigestProvenance::TrustedSha256,
        ),
        false,
    );
    for selector in [&id, "existing-ref"] {
        assert_payload(
            &spec
                .open_cached(
                    &store,
                    selector,
                    &PullLimits::default(),
                    &AtomicBool::new(false),
                )
                .unwrap(),
            NEW,
        );
    }
    for provider in [Provider::S3, Provider::Azure] {
        let body = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n";
        let spec = provider.spec(body);
        let id = seed(
            &store,
            &admission,
            spec.source(),
            body,
            (
                PAYLOAD_PATH,
                FileRole::Auxiliary,
                DigestProvenance::TrustedSha256,
            ),
            false,
        );
        assert!(
            spec.open(&store, &id, &PullLimits::default(), &AtomicBool::new(false))
                .is_err()
        );
    }
}

#[test]
fn digest_size_eof_and_pointer_failures_never_move_previous_reference() {
    let data = vec![b'a'; 256];
    let mut pointer =
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 256\n".to_vec();
    pointer.resize(256, b' ');
    for provider in [Provider::S3, Provider::Azure] {
        for body in [
            data[..255].to_vec(),
            [data.as_slice(), b"extra"].concat(),
            vec![b'b'; 256],
            pointer.clone(),
        ] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
            let spec = provider.spec(&data);
            let plan = spec
                .prepare(&PullLimits::default(), &admission, &AtomicBool::new(false))
                .unwrap();
            let mut first = provider.row(&body);
            first
                .headers
                .insert("content-length".into(), data.len().to_string());
            let mut rows = vec![first];
            if body.len() < data.len() {
                for _ in 0..2 {
                    let mut retry = provider.range(&data, body.len());
                    retry.body.clear();
                    rows.push(retry);
                }
            }
            let script = Script::new(provider, rows, &admission);
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
            script.finished();
            assert_eq!(
                store
                    .resolve_ref("keep")
                    .unwrap()
                    .manifest()
                    .digest()
                    .unwrap(),
                old
            );
            assert_no_host_reservation(&admission);
        }
    }
}

#[test]
fn content_representation_delete_and_status_refusals_precede_payload_reads() {
    for provider in [Provider::S3, Provider::Azure] {
        let guards = match provider {
            Provider::S3 => vec![("x-amz-delete-marker", "true")],
            Provider::Azure => vec![
                ("x-ms-blob-type", "PageBlob"),
                ("x-ms-copy-status", "pending"),
                ("x-ms-structured-body", "true"),
                ("x-ms-encryption-key-sha256", "key"),
                ("content-encoding", "gzip"),
            ],
        };
        for (header, value) in guards {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
            let plan = provider
                .spec(NEW)
                .prepare(&PullLimits::default(), &admission, &AtomicBool::new(false))
                .unwrap();
            let mut row = provider.row(NEW);
            row.headers.insert(header.into(), value.into());
            row.never = true;
            let script = Script::new(provider, vec![row], &admission);
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
            script.finished();
            assert_eq!(script.bytes.load(Ordering::Relaxed), 0);
            assert_eq!(
                store
                    .resolve_ref("keep")
                    .unwrap()
                    .manifest()
                    .digest()
                    .unwrap(),
                old
            );
        }
        for status in [302, 403, 404, 412] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let plan = provider
                .spec(NEW)
                .prepare(&PullLimits::default(), &admission, &AtomicBool::new(false))
                .unwrap();
            let mut row = provider.row(NEW);
            row.status = status;
            row.never = true;
            let script = Script::new(provider, vec![row], &admission);
            assert!(
                execute(
                    &store,
                    &plan,
                    &script,
                    &admission,
                    &Arc::new(AtomicBool::new(false)),
                    None
                )
                .is_err()
            );
            script.finished();
        }
    }
}

#[test]
fn durable_content_prefix_is_rehashed_and_bound_to_strong_validator_and_complete_sha() {
    for provider in [Provider::S3, Provider::Azure] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let data = vec![b'r'; 1024];
        let spec = provider.spec(&data);
        let plan = spec
            .prepare(&PullLimits::default(), &admission, &AtomicBool::new(false))
            .unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let first = Script::new(provider, vec![provider.row(&data)], &admission);
        assert!(
            execute_with_progress(
                &store,
                &plan,
                &first,
                &admission,
                &cancel,
                Some("keep"),
                &mut |p| {
                    if p.file_bytes() == 128 {
                        cancel.store(true, Ordering::Release);
                    }
                }
            )
            .is_err()
        );
        first.finished();
        assert_eq!(first.bytes.load(Ordering::Relaxed), 128);
        let mut bad = provider.range(&data, 128);
        bad.headers.insert("etag".into(), "\"changed\"".into());
        bad.never = true;
        let script = Script::new(provider, vec![bad], &admission);
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
        script.finished();
        assert_eq!(
            store
                .resolve_ref("keep")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            old
        );
        let mut row = provider.range(&data, 128);
        row.headers.insert(
            provider.version_header().into(),
            "new-current-metadata-only".into(),
        );
        let script = Script::new(provider, vec![row], &admission);
        let mut resumed = 0;
        execute_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("keep"),
            &mut |p| resumed = p.resumed_bytes(),
        )
        .unwrap();
        script.finished();
        assert_eq!(resumed, 128);
        assert_payload(&store.resolve_ref("keep").unwrap(), &data);
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn missing_validator_cancellation_restarts_full_body_without_persisted_prefix() {
    for provider in [Provider::S3, Provider::Azure] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let data = vec![b'n'; 512];
        let plan = provider
            .spec(&data)
            .prepare(&PullLimits::default(), &admission, &AtomicBool::new(false))
            .unwrap();
        let mut row = provider.row(&data);
        row.headers.remove("etag");
        let script = Script::new(provider, vec![row], &admission);
        let cancel = Arc::new(AtomicBool::new(false));
        assert!(
            execute_with_progress(
                &store,
                &plan,
                &script,
                &admission,
                &cancel,
                None,
                &mut |p| {
                    if p.file_bytes() == 128 {
                        cancel.store(true, Ordering::Release);
                    }
                }
            )
            .is_err()
        );
        script.finished();
        let mut row = provider.row(&data);
        row.headers.remove("etag");
        let script = Script::new(provider, vec![row], &admission);
        let mut resumed = 1;
        execute_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |p| resumed = p.resumed_bytes(),
        )
        .unwrap();
        script.finished();
        assert_eq!(resumed, 0);
        assert_eq!(script.bytes.load(Ordering::Relaxed), data.len());
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn live_retry_uses_current_target_with_full_sha_even_when_version_metadata_changes() {
    for provider in [Provider::S3, Provider::Azure] {
        for corrupt in [false, true] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
            let data = vec![b'l'; 512];
            let plan = provider
                .spec(&data)
                .prepare(&PullLimits::default(), &admission, &AtomicBool::new(false))
                .unwrap();
            let mut first = provider.row(&data);
            first.cut = Some(128);
            let mut next = provider.range(&data, 128);
            next.headers.insert(
                provider.version_header().into(),
                "different-current-version".into(),
            );
            if corrupt {
                next.body[0] = b'X';
            }
            let script = Script::new(provider, vec![first, next], &admission);
            let result = execute(
                &store,
                &plan,
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                Some("keep"),
            );
            script.finished();
            assert_eq!(script.calls.load(Ordering::Relaxed), 2);
            if corrupt {
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
            } else {
                result.unwrap();
                assert_payload(&store.resolve_ref("keep").unwrap(), &data);
            }
            assert_no_host_reservation(&admission);
        }
    }
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_azure_content_get_only_public_bearer_sas_and_independent_shared_key_wire() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use zerogravity_artifacts::acquisition::http::{AzureSharedKey, HttpOptions, HttpsClient};
    const KEY: &str =
        "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+Pw==";
    const SAS: &str = "sv=2023-11-03&sr=b&sp=r&se=2030-01-01T00%3a00%3A00Z&spr=https&sig=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA%3d";
    let data = b"actual isolated Azure payload";
    for mode in ["public", "bearer", "key", "sas"] {
        let mode = format!("content-{mode}");
        let mut child = Command::new("python3")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/azure_wire.py"
            ))
            .arg(&mode)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        let port = line.trim().parse::<u16>().unwrap();
        let origin = format!("http://127.0.0.1:{port}");
        let http = HttpOptions {
            max_retries: 0,
            ..Default::default()
        };
        let client = match mode.as_str() {
            "content-key" => HttpsClient::diagnostic_azure_shared_key(
                &origin,
                AzureSharedKey::new("fixtureaccount", KEY).unwrap(),
                http,
            ),
            "content-sas" => HttpsClient::diagnostic_azure_sas(&origin, SAS, http),
            "content-bearer" => HttpsClient::diagnostic_loopback(
                &origin,
                Some("synthetic-azure-bearer".into()),
                http,
            ),
            _ => HttpsClient::diagnostic_loopback(&origin, None, http),
        }
        .unwrap();
        let spec = AzureContentSpec::new(
            &origin,
            "container",
            "dir/a b.bin",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            data.len() as u64,
            &digest(data),
        )
        .unwrap();
        let admission = admission(128 * 1024 * 1024);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = spec
            .prepare(&PullLimits::default(), &admission, &cancel)
            .unwrap();
        assert!(!plan.files()[0].url().contains('?'));
        assert!(!format!("{plan:?}").contains(SAS));
        let fixture = Fixture::new();
        let store = fixture.open();
        let receipt = execute(&store, &plan, &client, &admission, &cancel, None).unwrap();
        assert_payload(
            &spec
                .open_cached(
                    &store,
                    &snapshot(receipt.outcome()),
                    &PullLimits::default(),
                    &cancel,
                )
                .unwrap(),
            data,
        );
        let mut verified = String::new();
        stdout.read_line(&mut verified).unwrap();
        assert_eq!(verified.trim(), format!("VERIFIED {mode}"));
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "independent Azure content verifier failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_no_host_reservation(&admission);
    }
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_s3_current_target_keeps_raw_path_no_version_query_and_independent_sigv4_wire() {
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient, S3Credentials};
    fn accept(listener: &TcpListener) -> TcpStream {
        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    return stream;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < deadline,
                        "owned content fixture accept timeout"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("{e}"),
            }
        }
    }
    // Independent stdlib verifier, not the production HMAC/signing helper. Raw
    // query is empty; dot segments/repeated slashes/escaped bytes stay unchanged.
    const VERIFY: &str = r#"import hashlib,hmac,sys
raw=sys.stdin.buffer.read().decode('ascii').split('\r\n')
method,target,_=raw[0].split(' ')
headers={}
for line in raw[1:]:
    if not line: break
    name,value=line.split(':',1)
    headers[name.lower()]=value.strip()
assert target=='/bucket/dir/../raw//a%20b%2B%25'
assert headers['x-amz-content-sha256']==hashlib.sha256(b'').hexdigest()
assert headers['x-amz-security-token']=='synthetic-session'
auth=headers['authorization']
assert auth.startswith('AWS4-HMAC-SHA256 ')
parts=dict(part.strip().split('=',1)for part in auth.split(' ',1)[1].split(','))
assert parts['SignedHeaders']=='host;x-amz-content-sha256;x-amz-date;x-amz-security-token'
access,scope=parts['Credential'].split('/',1)
assert access=='SYNTHETICACCESS'
date,region,service,terminal=scope.split('/')
assert region=='us-east-1'and service=='s3'and terminal=='aws4_request'
stamp=headers['x-amz-date']
assert stamp.startswith(date)
canonical_headers=''.join(name+':'+ ' '.join(headers[name].split())+'\n'for name in parts['SignedHeaders'].split(';'))
canonical='\n'.join([method,target,'',canonical_headers,parts['SignedHeaders'],headers['x-amz-content-sha256']])
to_sign='\n'.join(['AWS4-HMAC-SHA256',stamp,scope,hashlib.sha256(canonical.encode()).hexdigest()])
key=('AWS4'+'synthetic-secret').encode()
for value in [date,region,service,terminal]:key=hmac.new(key,value.encode(),hashlib.sha256).digest()
assert hmac.compare_digest(hmac.new(key,to_sign.encode(),hashlib.sha256).hexdigest(),parts['Signature'])
print('VERIFIED')
"#;
    for signed in [false, true] {
        let data = b"actual isolated S3 content";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            let mut stream = accept(&listener);
            let mut raw = Vec::new();
            let mut buffer = [0u8; 2048];
            while !raw.windows(4).any(|x| x == b"\r\n\r\n") {
                let n = stream.read(&mut buffer).unwrap();
                assert_ne!(n, 0);
                raw.extend_from_slice(&buffer[..n]);
                assert!(raw.len() < 16384);
            }
            let text = std::str::from_utf8(&raw).unwrap();
            assert!(text.starts_with("GET /bucket/dir/../raw//a%20b%2B%25 HTTP/1.1\r\n"));
            assert!(!text.contains("versionId="));
            if signed {
                let mut child = Command::new("python3")
                    .args(["-c", VERIFY])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap();
                child.stdin.take().unwrap().write_all(&raw).unwrap();
                let output = child.wait_with_output().unwrap();
                assert!(
                    output.status.success(),
                    "independent SigV4 verifier failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(output.stdout, b"VERIFIED\n");
            } else {
                assert!(!text.to_ascii_lowercase().contains("authorization:"));
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\nETag: \"wire\"\r\n\r\n",
                data.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(data).unwrap();
        });
        let endpoint =
            S3Endpoint::compatible(&origin, "us-east-1", S3Addressing::PathStyle).unwrap();
        let spec = S3ContentSpec::new(
            &endpoint,
            "bucket",
            "dir/../raw//a b+%",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            data.len() as u64,
            &digest(data),
        )
        .unwrap();
        let options = HttpOptions {
            max_retries: 0,
            ..Default::default()
        };
        let client = if signed {
            HttpsClient::diagnostic_s3_loopback(
                &origin,
                "us-east-1",
                S3Credentials::new(
                    "SYNTHETICACCESS",
                    "synthetic-secret",
                    Some("synthetic-session"),
                )
                .unwrap(),
                options,
            )
        } else {
            HttpsClient::diagnostic_loopback(&origin, None, options)
        }
        .unwrap();
        let admission = admission(128 * 1024 * 1024);
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = spec
            .prepare(&PullLimits::default(), &admission, &cancel)
            .unwrap();
        let fixture = Fixture::new();
        let store = fixture.open();
        let receipt = execute(&store, &plan, &client, &admission, &cancel, None).unwrap();
        worker.join().unwrap();
        assert_payload(
            &spec
                .open_cached(
                    &store,
                    &snapshot(receipt.outcome()),
                    &PullLimits::default(),
                    &cancel,
                )
                .unwrap(),
            data,
        );
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn exact_empty_content_stream_still_proves_eof_before_installation() {
    for provider in [Provider::S3, Provider::Azure] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let spec = provider.spec(b"");
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = spec
            .prepare(&PullLimits::default(), &admission, &cancel)
            .unwrap();
        let script = Script::new(provider, vec![provider.row(b"")], &admission);
        let receipt = execute(&store, &plan, &script, &admission, &cancel, None).unwrap();
        script.finished();
        assert_eq!(script.calls.load(Ordering::Relaxed), 1);
        assert_eq!(receipt.files()[0].bytes(), 0);
        assert_payload(
            &spec
                .open(
                    &store,
                    &snapshot(receipt.outcome()),
                    &PullLimits::default(),
                    &cancel,
                )
                .unwrap(),
            b"",
        );
        assert_no_host_reservation(&admission);
    }
}
