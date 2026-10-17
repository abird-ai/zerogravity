//! Synthetic native Azure protocol through the original executor/publisher.
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
use zerogravity_artifacts::Result;
use zerogravity_artifacts::acquisition::azure::{
    AzureObjectPin, AzureObjectRequest, AzurePrefixRequest, AzureRevision,
};
use zerogravity_artifacts::acquisition::http::{
    HttpRequest, HttpResponse, HttpTarget, HttpTransport,
};
use zerogravity_artifacts::acquisition::{
    PullLimits, ResolvedPullPlan, execute, execute_with_progress,
};
use zerogravity_artifacts::cache::{CommitOutcome, DigestProvenance, FileRole, TransferAdmission};
const ORIGIN: &str = "https://fixtureaccount.blob.core.windows.net";
const VERSION: &str = "2025-10-16T12:00:00.0000000Z";
const TAG: &str = "\"strong-azure-fixture\"";
const DATA: &[u8] = b"a complete synthetic block blob";
fn request(sha: Option<&str>) -> AzureObjectRequest {
    AzureObjectRequest::new(
        ORIGIN,
        "container",
        "folder/data.bin",
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        AzureRevision::version(VERSION).unwrap(),
        sha,
    )
    .unwrap()
}
struct NeverRead;
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("HEAD or rejected body was read")
    }
}
struct CutReader {
    body: Cursor<Vec<u8>>,
    remaining: usize,
}
impl Read for CutReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("synthetic cut"));
        }
        let n = out.len().min(self.remaining);
        let n = self.body.read(&mut out[..n])?;
        self.remaining -= n;
        Ok(n)
    }
}
struct Row {
    head: bool,
    suffix: String,
    status: u16,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
    never: bool,
    range: Option<u64>,
    cut: Option<usize>,
}
impl Row {
    fn get(data: &[u8]) -> Self {
        Self {
            head: false,
            suffix: format!(
                "/container/folder/data.bin?versionid={}",
                VERSION.replace(':', "%3A")
            ),
            status: 200,
            headers: BTreeMap::from([
                ("content-length".into(), data.len().to_string()),
                ("etag".into(), TAG.into()),
                ("x-ms-blob-type".into(), "BlockBlob".into()),
            ]),
            body: data.into(),
            never: false,
            range: None,
            cut: None,
        }
    }
    fn head(bytes: u64) -> Self {
        let mut row = Self::get(&[]);
        row.head = true;
        row.never = true;
        row.headers
            .insert("content-length".into(), bytes.to_string());
        row
    }
    fn range(data: &[u8], start: usize) -> Self {
        let mut row = Self::get(&data[start..]);
        row.status = 206;
        row.range = Some(start as u64);
        row.headers.insert(
            "content-range".into(),
            format!("bytes {start}-{}/{}", data.len() - 1, data.len()),
        );
        row
    }
    fn page(xml: String, marker: Option<&str>) -> Self {
        let mut suffix =
            "/container?restype=container&comp=list&include=copy&prefix=folder%2F&maxresults=4096"
                .to_owned();
        if let Some(marker) = marker {
            suffix.push_str("&marker=");
            suffix.push_str(marker);
        }
        Self {
            head: false,
            suffix,
            status: 200,
            headers: BTreeMap::from([("content-length".into(), xml.len().to_string())]),
            body: xml.into_bytes(),
            never: false,
            range: None,
            cut: None,
        }
    }
}
struct Script {
    rows: Mutex<VecDeque<Row>>,
    admission: TransferAdmission,
    calls: AtomicUsize,
}
impl Script {
    fn new(rows: Vec<Row>, admission: &TransferAdmission) -> Self {
        Self {
            rows: Mutex::new(rows.into()),
            admission: admission.clone(),
            calls: AtomicUsize::new(0),
        }
    }
    fn response(&self, request: &HttpRequest, head: bool) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0,
            "admission precedes metadata/payload"
        );
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected provider call");
        assert_eq!(row.head, head);
        assert!(
            request.url.ends_with(&row.suffix),
            "unexpected request target"
        );
        assert_eq!(request.range.as_ref().map(|r| r.start), row.range);
        self.calls.fetch_add(1, Ordering::Relaxed);
        let body: Box<dyn Read + Send> = if row.never {
            Box::new(NeverRead)
        } else if let Some(remaining) = row.cut {
            Box::new(CutReader {
                body: Cursor::new(row.body),
                remaining,
            })
        } else {
            Box::new(Cursor::new(row.body))
        };
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body,
        })
    }
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("Azure must use canonical target capability")
    }
    fn get_target(
        &self,
        request: &HttpRequest,
        target: &HttpTarget,
        _: &AtomicBool,
    ) -> Result<HttpResponse> {
        assert!(!format!("{target:?}").contains(VERSION));
        self.response(request, false)
    }
    fn head_target(
        &self,
        request: &HttpRequest,
        _: &HttpTarget,
        _: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.response(request, true)
    }
}
fn resolve(
    req: &AzureObjectRequest,
    bytes: u64,
    admission: &TransferAdmission,
) -> (AzureObjectPin, ResolvedPullPlan) {
    let script = Script::new(vec![Row::head(bytes)], admission);
    let result = req
        .resolve_with_pin(
            &script,
            &PullLimits::default(),
            admission,
            &AtomicBool::new(false),
        )
        .unwrap();
    script.finished();
    result
}
fn snapshot(outcome: &CommitOutcome) -> String {
    match outcome {
        CommitOutcome::Durable(s) => s.digest().to_owned(),
        CommitOutcome::DurabilityUnknown { cause, .. } => panic!("{cause}"),
    }
}
fn listing(rows: &str, marker: &str, next: &str) -> String {
    format!(
        "<EnumerationResults ServiceEndpoint=\"{ORIGIN}/\" ContainerName=\"container\"><Prefix>folder/</Prefix><Marker>{marker}</Marker><Blobs>{rows}</Blobs><NextMarker>{next}</NextMarker></EnumerationResults>"
    )
}
fn listed(name: &str, bytes: u64) -> String {
    format!(
        "<Blob><Name>folder/{name}</Name><VersionId>{VERSION}</VersionId><Properties><Etag>&quot;strong-azure-fixture&quot;</Etag><Content-Length>{bytes}</Content-Length><BlobType>BlockBlob</BlobType></Properties></Blob>"
    )
}

#[test]
fn exact_properties_then_original_publish_and_offline_contract_without_etag() {
    for sha in [None, Some(digest(DATA))] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let cancel = Arc::new(AtomicBool::new(false));
        let req = request(sha.as_deref());
        let (pin, plan) = resolve(&req, DATA.len() as u64, &admission);
        assert_eq!(pin.bytes(), DATA.len() as u64);
        assert_eq!(pin.revision().value(), VERSION);
        assert_eq!(pin.source(), plan.source());
        let script = Script::new(vec![Row::get(DATA)], &admission);
        let receipt = execute(&store, &plan, &script, &admission, &cancel, None).unwrap();
        script.finished();
        let id = snapshot(receipt.outcome());
        let lease = pin
            .open_cached(&store, &id, &PullLimits::default(), &cancel)
            .unwrap();
        assert_payload(&lease, DATA);
        assert_eq!(
            lease.manifest().entries()[0].digest_provenance(),
            if sha.is_some() {
                DigestProvenance::TrustedSha256
            } else {
                DigestProvenance::FirstSeen
            }
        );
        let rebuilt = AzureObjectPin::new(&req, pin.bytes()).unwrap();
        assert!(
            rebuilt
                .open_cached(&store, &id, &PullLimits::default(), &cancel)
                .is_ok()
        );
        if sha.is_some() {
            let cached = Script::new(vec![], &admission);
            let mut cached_bytes = 0;
            execute_with_progress(
                &store,
                &plan,
                &cached,
                &admission,
                &cancel,
                None,
                &mut |event| {
                    cached_bytes = event.cached_bytes();
                },
            )
            .unwrap();
            cached.finished();
            assert_eq!(cached_bytes, DATA.len() as u64);
        }
        let wrong_size = AzureObjectPin::new(&req, pin.bytes() + 1).unwrap();
        assert!(
            wrong_size
                .open_cached(&store, &id, &PullLimits::default(), &cancel)
                .is_err()
        );
        let other = AzureObjectRequest::new(
            ORIGIN,
            "container",
            "folder/data.bin",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            AzureRevision::snapshot(VERSION).unwrap(),
            sha.as_deref(),
        )
        .unwrap();
        assert!(
            AzureObjectPin::new(&other, pin.bytes())
                .unwrap()
                .open_cached(&store, &id, &PullLimits::default(), &cancel)
                .is_err()
        );
        let narrow = PullLimits {
            cache: zerogravity_artifacts::cache::CacheLimits {
                max_file_bytes: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(pin.open_cached(&store, &id, &narrow, &cancel).is_err());
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn metadata_refuses_unsupported_representations_without_reading_head_body() {
    let admission = admission(128 * 1024 * 1024);
    for (header, value) in [
        ("x-ms-blob-type", "PageBlob"),
        ("x-ms-blob-type", "AppendBlob"),
        ("x-ms-version-id", "different"),
        ("content-encoding", "gzip"),
        ("x-ms-structured-body", "true"),
        ("x-ms-encryption-key-sha256", "keydigest"),
        ("x-ms-copy-status", "pending"),
        ("etag", "W/\"weak\""),
        ("content-range", "bytes 0-1/2"),
        ("transfer-encoding", "chunked"),
        ("content-length", "-1"),
    ] {
        let mut row = Row::head(2);
        row.headers.insert(header.into(), value.into());
        let script = Script::new(vec![row], &admission);
        assert!(
            request(None)
                .resolve(
                    &script,
                    &PullLimits::default(),
                    &admission,
                    &AtomicBool::new(false)
                )
                .is_err(),
            "{header}"
        );
        script.finished();
    }
    for missing in ["x-ms-blob-type", "etag", "content-length"] {
        let mut row = Row::head(2);
        row.headers.remove(missing);
        let script = Script::new(vec![row], &admission);
        assert!(
            request(None)
                .resolve(
                    &script,
                    &PullLimits::default(),
                    &admission,
                    &AtomicBool::new(false)
                )
                .is_err()
        );
    }
    assert_no_host_reservation(&admission);
}

#[test]
fn optional_version_echo_and_snapshot_distinct_namespace_are_accepted() {
    let admission = admission(128 * 1024 * 1024);
    for echo in [None, Some(VERSION)] {
        let mut row = Row::head(0);
        if let Some(echo) = echo {
            row.headers.insert("x-ms-version-id".into(), echo.into());
        }
        let script = Script::new(vec![row], &admission);
        assert!(
            request(None)
                .resolve(
                    &script,
                    &PullLimits::default(),
                    &admission,
                    &AtomicBool::new(false)
                )
                .is_ok()
        );
        script.finished();
    }
    let req = AzureObjectRequest::new(
        ORIGIN,
        "container",
        "folder/data.bin",
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        AzureRevision::snapshot("snapshot-opaque").unwrap(),
        None,
    )
    .unwrap();
    let mut head = Row::head(0);
    head.suffix = "/container/folder/data.bin?snapshot=snapshot-opaque".into();
    head.headers.insert(
        "x-ms-version-id".into(),
        "different-version-namespace".into(),
    );
    let script = Script::new(vec![head], &admission);
    assert!(
        req.resolve(
            &script,
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false)
        )
        .is_ok()
    );
    script.finished();
}

#[test]
fn body_binding_refuses_before_bytes_and_preserves_old_reference() {
    for (header, value) in [
        ("x-ms-version-id", "different"),
        ("x-ms-blob-type", "PageBlob"),
        ("x-ms-copy-status", "failed"),
        ("x-ms-structured-body", "true"),
        ("x-ms-encryption-key-sha256", "key"),
        ("content-encoding", "gzip"),
    ] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let (_, plan) = resolve(&request(None), DATA.len() as u64, &admission);
        let mut row = Row::get(DATA);
        row.never = true;
        row.headers.insert(header.into(), value.into());
        let script = Script::new(vec![row], &admission);
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

#[test]
fn full_hash_exact_eof_and_pointer_refusals_keep_old_reference() {
    let mut pointer =
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abcdef\nsize 128\n".to_vec();
    pointer.resize(128, b' ');
    for (expected, body) in [
        (DATA.len() as u64, DATA[..DATA.len() - 1].to_vec()),
        (DATA.len() as u64, [DATA, b"extra"].concat()),
        (DATA.len() as u64, [&DATA[..DATA.len() - 1], b"X"].concat()),
        (128, pointer),
    ] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let (_, plan) = resolve(&request(Some(&digest(DATA))), expected, &admission);
        let mut row = Row::get(&body);
        row.headers
            .insert("content-length".into(), expected.to_string());
        let short = (body.len() as u64) < expected;
        let mut rows = vec![row];
        if short {
            // The original reader legitimately retries a strong exact prefix.
            // Both valid206 replies end prematurely at the same offset, so its
            // existing two-retry budget ends in failure rather than recovery.
            for _ in 0..2 {
                let mut retry = Row::range(DATA, body.len());
                retry.body.clear();
                rows.push(retry);
            }
        }
        let script = Script::new(rows, &admission);
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
            script.calls.load(Ordering::Relaxed),
            if short { 3 } else { 1 }
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
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn durable_resume_rechecks_version_and_progress_uses_original_counts() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let data = vec![b'r'; 1024];
    let (_, plan) = resolve(&request(Some(&digest(&data))), 1024, &admission);
    let cancel = Arc::new(AtomicBool::new(false));
    let first = Script::new(vec![Row::get(&data)], &admission);
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
    let mut bad = Row::range(&data, 128);
    bad.never = true;
    bad.headers
        .insert("x-ms-version-id".into(), "changed".into());
    let script = Script::new(vec![bad], &admission);
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
    let script = Script::new(vec![Row::range(&data, 128)], &admission);
    let mut resumed = 0;
    let mut complete = 0;
    execute_with_progress(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
        &mut |p| {
            resumed = p.resumed_bytes();
            if p.file_complete() {
                complete += 1;
            }
        },
    )
    .unwrap();
    assert_eq!(resumed, 128);
    assert_eq!(complete, 1);
    assert_payload(&store.resolve_ref("keep").unwrap(), &data);
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn live_retry_dispatch_preserves_exact_target_and_rechecks_present_version() {
    for bad in [false, true] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let data = vec![b'a'; 512];
        let (_, plan) = resolve(&request(Some(&digest(&data))), 512, &admission);
        let mut first = Row::get(&data);
        first.cut = Some(128);
        let mut next = Row::range(&data, 128);
        if bad {
            next.never = true;
            next.headers
                .insert("x-ms-version-id".into(), "wrong".into());
        }
        let script = Script::new(vec![first, next], &admission);
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
        if bad {
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

#[test]
fn prefix_short_continuation_pins_independent_current_auxiliary_versions() {
    let admission = admission(128 * 1024 * 1024);
    let req = AzurePrefixRequest::new(ORIGIN, "container", "folder/").unwrap();
    let script = Script::new(
        vec![
            Row::page(listing(&listed("a.bin", 2), "", "next"), None),
            Row::page(listing(&listed("b.bin", 3), "next", ""), Some("next")),
        ],
        &admission,
    );
    let plans = req
        .resolve_with_pins(
            &script,
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
    script.finished();
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].1.files()[0].path(), "a.bin");
    assert_eq!(plans[1].1.files()[0].path(), "b.bin");
    assert_ne!(plans[0].0.source(), plans[1].0.source());
    for (pin, plan) in plans {
        assert_eq!(pin.revision().value(), VERSION);
        assert_eq!(plan.files()[0].role(), FileRole::Auxiliary);
        assert_eq!(
            plan.files()[0].digest_provenance(),
            DigestProvenance::FirstSeen
        );
    }
    assert_no_host_reservation(&admission);
}

#[test]
fn prefix_rejects_source_membership_duplicates_missing_versions_and_budget_exhaustion() {
    let admission = admission(128 * 1024 * 1024);
    let req = AzurePrefixRequest::new(ORIGIN, "container", "folder/").unwrap();
    let valid = listed("a.bin", 2);
    for xml in [
        listing(&valid, "", "").replace(ORIGIN, "https://other.blob.core.windows.net"),
        listing(&valid, "", "").replace("ContainerName=\"container\"", "ContainerName=\"other\""),
        listing(&valid, "", "").replace("<Prefix>folder/</Prefix>", "<Prefix>other/</Prefix>"),
        listing(&valid, "", "").replace("folder/a.bin", "other/a.bin"),
        listing(&valid, "", "").replace("folder/a.bin", "folder/../unsafe"),
        listing(&format!("{valid}{valid}"), "", ""),
        listing(&format!("{}{}", listed("a", 1), listed("a/b", 1)), "", ""),
        listing(&valid, "", "").replace(&format!("<VersionId>{VERSION}</VersionId>"), ""),
        listing("", "", ""),
    ] {
        let script = Script::new(vec![Row::page(xml, None)], &admission);
        assert!(
            req.resolve(
                &script,
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        script.finished();
    }
    let limits = PullLimits {
        max_pages: 1,
        ..Default::default()
    };
    let script = Script::new(
        vec![Row::page(listing(&valid, "", "next"), None)],
        &admission,
    );
    assert!(
        req.resolve(&script, &limits, &admission, &AtomicBool::new(false))
            .is_err()
    );
    script.finished();
    let limits = PullLimits {
        cache: zerogravity_artifacts::cache::CacheLimits {
            max_snapshot_bytes: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let script = Script::new(vec![Row::page(listing(&valid, "", ""), None)], &admission);
    assert!(
        req.resolve(&script, &limits, &admission, &AtomicBool::new(false))
            .is_err()
    );
    let script = Script::new(
        vec![
            Row::page(listing(&valid, "", "next"), None),
            Row::page(listing(&listed("b.bin", 1), "next", "next"), Some("next")),
        ],
        &admission,
    );
    assert!(
        req.resolve(
            &script,
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn metadata_admission_cancel_and_legacy_transport_fail_before_any_body() {
    struct Legacy;
    impl HttpTransport for Legacy {
        fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("no request expected")
        }
    }
    let denied = admission(1);
    assert!(
        request(None)
            .resolve(
                &Legacy,
                &PullLimits::default(),
                &denied,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_no_host_reservation(&denied);
    let allowed = admission(128 * 1024 * 1024);
    assert!(
        request(None)
            .resolve(
                &Legacy,
                &PullLimits::default(),
                &allowed,
                &AtomicBool::new(true)
            )
            .is_err()
    );
    assert!(
        request(None)
            .resolve(
                &Legacy,
                &PullLimits::default(),
                &allowed,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_no_host_reservation(&allowed);
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_original_http_owner_public_bearer_sas_and_independent_shared_key_wire() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use zerogravity_artifacts::acquisition::http::{AzureSharedKey, HttpOptions, HttpsClient};
    const KEY: &str =
        "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+Pw==";
    const SAS: &str = "sv=2023-11-03&sr=bv&sp=r&se=2030-01-01T00%3a00%3A00Z&spr=https&sig=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA%3d";
    for mode in ["public", "bearer", "key", "sas"] {
        let mut child = Command::new("python3")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/azure_wire.py"
            ))
            .arg(mode)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        let port = line.trim().parse::<u16>().unwrap();
        let origin = format!("http://127.0.0.1:{port}");
        let options = HttpOptions {
            max_retries: 0,
            ..Default::default()
        };
        let client = match mode {
            "key" => HttpsClient::diagnostic_azure_shared_key(
                &origin,
                AzureSharedKey::new("fixtureaccount", KEY).unwrap(),
                options,
            ),
            "sas" => HttpsClient::diagnostic_azure_sas(&origin, SAS, options),
            "bearer" => HttpsClient::diagnostic_loopback(
                &origin,
                Some("synthetic-azure-bearer".to_owned()),
                options,
            ),
            _ => HttpsClient::diagnostic_loopback(&origin, None, options),
        }
        .unwrap();
        let req = AzureObjectRequest::new(
            &origin,
            "container",
            "dir/a b.bin",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            AzureRevision::version(VERSION).unwrap(),
            Some(&digest(b"actual isolated Azure payload")),
        )
        .unwrap();
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let cancel = Arc::new(AtomicBool::new(false));
        let (pin, plan) = req
            .resolve_with_pin(&client, &PullLimits::default(), &admission, &cancel)
            .unwrap();
        assert!(!format!("{plan:?}").contains(SAS));
        assert!(!plan.files()[0].url().contains("sig="));
        let receipt = execute(&store, &plan, &client, &admission, &cancel, None).unwrap();
        assert_payload(
            &pin.open_cached(
                &store,
                &snapshot(receipt.outcome()),
                &PullLimits::default(),
                &cancel,
            )
            .unwrap(),
            b"actual isolated Azure payload",
        );
        let mut verified = String::new();
        stdout.read_line(&mut verified).unwrap();
        assert_eq!(verified.trim(), format!("VERIFIED {mode}"));
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "independent wire verifier failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_no_host_reservation(&admission);
    }
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_provider_redirects_and_duplicate_azure_headers_refuse() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    for extra in [
        "Location: /container/dir/a.bin\r\n",
        "x-ms-blob-type: BlockBlob\r\nx-ms-blob-type: BlockBlob\r\n",
        "x-ms-version-id: one\r\nx-ms-version-id: two\r\n",
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let redirect = extra.starts_with("Location");
        let response = format!(
            "HTTP/1.1 {}\r\nConnection: close\r\nContent-Length: 1\r\nETag: \"e\"\r\n{extra}\r\n",
            if redirect { "302 Found" } else { "200 OK" }
        );
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            stream.write_all(response.as_bytes()).unwrap();
        });
        let client = HttpsClient::diagnostic_loopback(
            &origin,
            None,
            HttpOptions {
                max_retries: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let req = AzureObjectRequest::new(
            &origin,
            "container",
            "dir/a.bin",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            AzureRevision::version(VERSION).unwrap(),
            None,
        )
        .unwrap();
        assert!(
            req.resolve(
                &client,
                &PullLimits::default(),
                &admission(128 * 1024 * 1024),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        worker.join().unwrap();
    }
}

#[test]
fn exact_empty_version_object_proves_eof_and_publishes_trusted_empty_data() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let (pin, plan) = resolve(&request(Some(&digest(b""))), 0, &admission);
    let script = Script::new(vec![Row::get(b"")], &admission);
    let receipt = execute(&store, &plan, &script, &admission, &cancel, None).unwrap();
    script.finished();
    assert_eq!(receipt.files()[0].bytes(), 0);
    assert_payload(
        &pin.open_cached(
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

#[test]
fn prefix_combined_long_validator_controls_refuse_even_when_members_fit_alone() {
    let admission = admission(128 * 1024 * 1024);
    let request = AzurePrefixRequest::new(ORIGIN, "container", "folder/").unwrap();
    let limits = PullLimits {
        cache: zerogravity_artifacts::cache::CacheLimits {
            max_manifest_bytes: 2500,
            ..Default::default()
        },
        ..Default::default()
    };
    let long_tag = format!("&quot;{}&quot;", "v".repeat(900));
    let first = listed("a.bin", 2).replace("&quot;strong-azure-fixture&quot;", &long_tag);
    let second = listed("b.bin", 3).replace("&quot;strong-azure-fixture&quot;", &long_tag);
    // Both original individual plans fit the caller's control budget. The
    // selected pair must refuse before any payload is requested under that same
    // budget; this does not depend on a private byte-count formula.
    for row in [&first, &second] {
        let script = Script::new(vec![Row::page(listing(row, "", ""), None)], &admission);
        let selected = request
            .resolve(&script, &limits, &admission, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(selected.len(), 1);
        script.finished();
    }
    let script = Script::new(
        vec![Row::page(
            listing(&format!("{first}{second}"), "", ""),
            None,
        )],
        &admission,
    );
    assert!(matches!(
        request.resolve(&script, &limits, &admission, &AtomicBool::new(false)),
        Err(zerogravity_artifacts::Error::Limit(_))
    ));
    script.finished();
    assert_eq!(script.calls.load(Ordering::Relaxed), 1);
    assert_no_host_reservation(&admission);
}
