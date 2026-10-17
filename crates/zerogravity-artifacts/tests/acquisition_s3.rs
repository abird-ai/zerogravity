//! Synthetic protocol fixtures through the sole cache executor/publisher.
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
use zerogravity_artifacts::acquisition::s3::{
    S3Addressing, S3Endpoint, S3Listing, S3ObjectPin, S3ObjectRequest, S3PrefixRequest,
};
use zerogravity_artifacts::acquisition::{PullLimits, execute, execute_with_progress};
use zerogravity_artifacts::cache::{CommitOutcome, DigestProvenance, FileRole, TransferAdmission};
use zerogravity_artifacts::{Error, Result};

const DATA: &[u8] = b"synthetic exact S3 object";
fn endpoint() -> S3Endpoint {
    S3Endpoint::compatible(
        "https://object.example",
        "us-east-1",
        S3Addressing::PathStyle,
    )
    .unwrap()
}
fn object(version: Option<&str>, sha: Option<&str>) -> S3ObjectRequest {
    S3ObjectRequest::new(
        &endpoint(),
        "bucket",
        "source/key",
        "model.gguf",
        FileRole::Weights,
        version,
        sha,
    )
    .unwrap()
}
struct NeverRead;
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("refused or HEAD response body must not be read")
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
    fn head(key: &str, requested: Option<&str>, actual: &str, bytes: u64) -> Self {
        Self {
            head: true,
            suffix: format!(
                "/bucket/{key}{}",
                requested
                    .map(|v| format!("?versionId={v}"))
                    .unwrap_or_default()
            ),
            status: 200,
            headers: BTreeMap::from([
                ("content-length".into(), bytes.to_string()),
                ("x-amz-version-id".into(), actual.into()),
                ("etag".into(), "\"e\"".into()),
            ]),
            body: vec![],
            never: true,
            range: None,
            cut: None,
        }
    }
    fn get(key: &str, version: &str, data: &[u8]) -> Self {
        Self {
            head: false,
            suffix: format!("/bucket/{key}?versionId={version}"),
            status: 200,
            headers: BTreeMap::from([
                ("content-length".into(), data.len().to_string()),
                ("x-amz-version-id".into(), version.into()),
                ("etag".into(), "\"e\"".into()),
            ]),
            body: data.to_vec(),
            never: false,
            range: None,
            cut: None,
        }
    }
    fn listing(suffix: &str, raw: String) -> Self {
        Self {
            head: false,
            suffix: suffix.into(),
            status: 200,
            headers: BTreeMap::from([("content-length".into(), raw.len().to_string())]),
            body: raw.into_bytes(),
            never: false,
            range: None,
            cut: None,
        }
    }
    fn range(key: &str, version: &str, data: &[u8], start: usize) -> Self {
        let mut row = Self::get(key, version, &data[start..]);
        row.status = 206;
        row.range = Some(start as u64);
        row.headers.insert(
            "content-range".into(),
            format!("bytes {start}-{}/{}", data.len() - 1, data.len()),
        );
        row
    }
}
struct CutReader {
    body: Cursor<Vec<u8>>,
    remaining: usize,
}
impl Read for CutReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("owned fixture interrupted stream"));
        }
        let count = output.len().min(self.remaining);
        let count = self.body.read(&mut output[..count])?;
        self.remaining -= count;
        Ok(count)
    }
}
struct Script {
    rows: Mutex<VecDeque<Row>>,
    calls: AtomicUsize,
    admission: TransferAdmission,
}
impl Script {
    fn new(rows: Vec<Row>, admission: &TransferAdmission) -> Self {
        Self {
            rows: Mutex::new(rows.into()),
            calls: AtomicUsize::new(0),
            admission: admission.clone(),
        }
    }
    fn response(&self, request: &HttpRequest, head: bool) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected S3 request");
        assert_eq!(head, row.head);
        assert_eq!(request.range.as_ref().map(|range| range.start), row.range);
        assert!(
            request.url.ends_with(&row.suffix),
            "expected suffix {}",
            row.suffix
        );
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body: if row.never {
                Box::new(NeverRead)
            } else if let Some(remaining) = row.cut {
                Box::new(CutReader {
                    body: Cursor::new(row.body),
                    remaining,
                })
            } else {
                Box::new(Cursor::new(row.body))
            },
        })
    }
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("native S3 must use the exact target capability")
    }
    fn get_target(
        &self,
        request: &HttpRequest,
        target: &HttpTarget,
        _: &AtomicBool,
    ) -> Result<HttpResponse> {
        assert!(!format!("{target:?}").contains("source/key"));
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
fn snapshot(outcome: &CommitOutcome) -> String {
    match outcome {
        CommitOutcome::Durable(s) => s.digest().into(),
        CommitOutcome::DurabilityUnknown { cause, .. } => panic!("{cause}"),
    }
}

#[test]
fn latest_head_pins_one_version_then_original_publication_and_exact_offline_contract() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let script = Script::new(
        vec![
            Row::head("source/key", None, "v1", DATA.len() as u64),
            Row::get("source/key", "v1", DATA),
        ],
        &admission,
    );
    let request = object(None, Some(&digest(DATA)));
    let cancel = Arc::new(AtomicBool::new(false));
    let limits = PullLimits::default();
    let (pin, plan) = request
        .resolve_with_pin(&script, &limits, &admission, &cancel)
        .unwrap();
    assert_eq!(pin.version(), "v1");
    assert_eq!(pin.bytes(), DATA.len() as u64);
    let receipt = execute(&store, &plan, &script, &admission, &cancel, None).unwrap();
    script.finished();
    let id = snapshot(receipt.outcome());
    let lease = pin.open_cached(&store, &id, &limits, &cancel).unwrap();
    assert_eq!(file_bytes(lease.file("model.gguf").unwrap()), DATA);
    assert_eq!(
        lease.manifest().entries()[0].digest_provenance(),
        DigestProvenance::TrustedSha256
    );
    for wrong in [
        S3ObjectPin::new(&request, "v2", pin.bytes()).unwrap(),
        S3ObjectPin::new(&request, "v1", pin.bytes() + 1).unwrap(),
        S3ObjectPin::new(&object(None, None), "v1", pin.bytes()).unwrap(),
    ] {
        assert!(wrong.open_cached(&store, &id, &limits, &cancel).is_err());
    }
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

#[test]
fn head_is_body_free_and_missing_null_changed_encoded_ranged_and_oversize_metadata_refuse() {
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let cancel = AtomicBool::new(false);
    let mut rows = Vec::new();
    let mut absent = Row::head("source/key", Some("v1"), "v1", 1);
    absent.headers.remove("x-amz-version-id");
    rows.push(absent);
    rows.push(Row::head("source/key", Some("v1"), "null", 1));
    rows.push(Row::head("source/key", Some("v1"), "v2", 1));
    for (name, value) in [
        ("content-encoding", "gzip"),
        ("content-range", "bytes 0-0/1"),
        ("x-amz-delete-marker", "true"),
        ("content-length", "1.5"),
        ("etag", "W/\"weak\""),
    ] {
        let mut row = Row::head("source/key", Some("v1"), "v1", 1);
        row.headers.insert(name.into(), value.into());
        rows.push(row);
    }
    for status in [403, 404, 405] {
        let mut row = Row::head("source/key", Some("v1"), "v1", 1);
        row.status = status;
        rows.push(row);
    }
    for row in rows {
        let script = Script::new(vec![row], &admission);
        assert!(
            object(Some("v1"), None)
                .resolve(&script, &limits, &admission, &cancel)
                .is_err()
        );
        script.finished();
    }
    let script = Script::new(vec![Row::head("source/key", None, "v1", 2)], &admission);
    let limits = PullLimits {
        cache: zerogravity_artifacts::cache::CacheLimits {
            max_file_bytes: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(
        object(None, None)
            .resolve(&script, &limits, &admission, &cancel)
            .is_err()
    );
    script.finished();
    let script = Script::new(vec![Row::head("source/key", None, "v1", 0)], &admission);
    assert_eq!(
        object(None, None)
            .resolve(&script, &PullLimits::default(), &admission, &cancel)
            .unwrap()
            .files()[0]
            .bytes(),
        Some(0)
    );
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn body_version_binding_refuses_before_reads_and_failures_preserve_previous_reference() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let limits = PullLimits::default();
    let pin = S3ObjectPin::new(
        &object(Some("v1"), Some(&digest(DATA))),
        "v1",
        DATA.len() as u64,
    )
    .unwrap();
    for change in 0..3 {
        let mut row = Row::get("source/key", "v1", DATA);
        row.never = true;
        match change {
            0 => {
                row.headers.remove("x-amz-version-id");
            }
            1 => {
                row.headers.insert("x-amz-version-id".into(), "v2".into());
            }
            _ => {
                row.headers
                    .insert("x-amz-delete-marker".into(), "true".into());
            }
        }
        let script = Script::new(vec![row], &admission);
        assert!(
            execute(
                &store,
                &pin.plan(&limits).unwrap(),
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                Some("keep")
            )
            .is_err()
        );
        script.finished();
    }
    let script = Script::new(
        vec![Row::get("source/key", "v1", &vec![b'x'; DATA.len()])],
        &admission,
    );
    assert!(
        execute(
            &store,
            &pin.plan(&limits).unwrap(),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("keep")
        )
        .is_err()
    );
    script.finished();
    let data = vec![b'x'; 512];
    let pin = S3ObjectPin::new(&object(Some("v1"), Some(&digest(&data))), "v1", 512).unwrap();
    let script = Script::new(vec![Row::get("source/key", "v1", &data)], &admission);
    let cancel = Arc::new(AtomicBool::new(false));
    assert!(
        execute_with_progress(
            &store,
            &pin.plan(&limits).unwrap(),
            &script,
            &admission,
            &cancel,
            Some("keep"),
            &mut |progress| if progress.downloaded_bytes() > 0 {
                cancel.store(true, Ordering::Release);
            }
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
        previous
    );
    assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

fn reject_version(row: &mut Row, change: usize) {
    row.never = true;
    match change {
        0 => {
            row.headers
                .insert("x-amz-version-id".into(), "changed".into());
        }
        1 => {
            row.headers.remove("x-amz-version-id");
        }
        _ => {
            row.headers
                .insert("x-amz-delete-marker".into(), "true".into());
        }
    }
}

#[test]
fn durable_s3_prefix_recovery_keeps_the_exact_target_and_checks_every_resumed_version_before_body()
{
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let data = vec![b'r'; 512];
    let limits = PullLimits::default();
    let pin = S3ObjectPin::new(&object(Some("v1"), Some(&digest(&data))), "v1", 512).unwrap();
    let first = Script::new(vec![Row::get("source/key", "v1", &data)], &admission);
    let cancel = Arc::new(AtomicBool::new(false));
    assert!(
        execute_with_progress(
            &store,
            &pin.plan(&limits).unwrap(),
            &first,
            &admission,
            &cancel,
            Some("keep"),
            &mut |progress| if progress.downloaded_bytes() >= 128 {
                cancel.store(true, Ordering::Release);
            }
        )
        .is_err()
    );
    first.finished();
    assert_eq!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        previous
    );
    for change in 0..3 {
        let mut row = Row::range("source/key", "v1", &data, 128);
        reject_version(&mut row, change);
        let script = Script::new(vec![row], &admission);
        assert!(
            execute(
                &store,
                &pin.plan(&limits).unwrap(),
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                Some("keep")
            )
            .is_err()
        );
        script.finished();
        assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
    }
    let script = Script::new(vec![Row::range("source/key", "v1", &data, 128)], &admission);
    let receipt = execute(
        &store,
        &pin.plan(&limits).unwrap(),
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("keep"),
    )
    .unwrap();
    script.finished();
    let id = snapshot(receipt.outcome());
    assert_eq!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        id
    );
    assert_eq!(
        file_bytes(
            store
                .resolve_ref("keep")
                .unwrap()
                .file("model.gguf")
                .unwrap()
        ),
        data
    );
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn live_retry_s3_target_dispatch_rechecks_present_missing_and_deleted_versions_before_new_bytes() {
    for change in 0..4 {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let data = vec![b'l'; 512];
        let limits = PullLimits::default();
        let pin = S3ObjectPin::new(&object(Some("v1"), Some(&digest(&data))), "v1", 512).unwrap();
        let mut first = Row::get("source/key", "v1", &data);
        first.cut = Some(128);
        let mut second = Row::range("source/key", "v1", &data, 128);
        if change > 0 {
            reject_version(&mut second, change - 1);
        }
        let script = Script::new(vec![first, second], &admission);
        let result = execute(
            &store,
            &pin.plan(&limits).unwrap(),
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("keep"),
        );
        script.finished();
        assert_eq!(script.calls.load(Ordering::Relaxed), 2);
        if change == 0 {
            let receipt = result.unwrap();
            let id = snapshot(receipt.outcome());
            assert_eq!(
                store
                    .resolve_ref("keep")
                    .unwrap()
                    .manifest()
                    .digest()
                    .unwrap(),
                id
            );
            assert_eq!(
                file_bytes(
                    store
                        .resolve_ref("keep")
                        .unwrap()
                        .file("model.gguf")
                        .unwrap()
                ),
                data
            );
        } else {
            assert!(result.is_err());
            assert_eq!(
                store
                    .resolve_ref("keep")
                    .unwrap()
                    .manifest()
                    .digest()
                    .unwrap(),
                previous
            );
            assert_payload(&store.resolve_ref("keep").unwrap(), OLD);
        }
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}

fn page(versions: bool, rows: &str, truncated: bool, next: &str) -> String {
    let root = if versions {
        "ListVersionsResult"
    } else {
        "ListBucketResult"
    };
    format!(
        "<{root} xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Name>bucket</Name><Prefix>aux%2F</Prefix><EncodingType>url</EncodingType><IsTruncated>{truncated}</IsTruncated>{rows}{next}</{root}>"
    )
}
fn content(key: &str) -> String {
    format!("<Contents><Key>aux%2F{key}</Key><Size>3</Size><ETag>&quot;e&quot;</ETag></Contents>")
}
fn version(key: &str, id: &str, latest: bool, deleted: bool) -> String {
    let kind = if deleted { "DeleteMarker" } else { "Version" };
    format!(
        "<{kind}><Key>aux%2F{key}</Key><VersionId>{id}</VersionId><IsLatest>{latest}</IsLatest>{}</{kind}>",
        if deleted {
            "".into()
        } else {
            "<Size>3</Size><ETag>&quot;e&quot;</ETag>".to_owned()
        }
    )
}

#[test]
fn bounded_v2_prefix_tokens_pin_each_independent_auxiliary_object() {
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let cancel = AtomicBool::new(false);
    let first = "/bucket/?encoding-type=url&list-type=2&max-keys=1000&prefix=aux%2F";
    let second = "/bucket/?continuation-token=next%2B%2F%3D&encoding-type=url&list-type=2&max-keys=1000&prefix=aux%2F";
    let script = Script::new(
        vec![
            Row::listing(
                first,
                page(
                    false,
                    &content("a"),
                    true,
                    "<NextContinuationToken>next+/=</NextContinuationToken>",
                ),
            ),
            Row::head("aux/a", None, "v1", 3),
            Row::listing(second, page(false, &content("b"), false, "")),
            Row::head("aux/b", None, "v2", 3),
        ],
        &admission,
    );
    let request =
        S3PrefixRequest::new(&endpoint(), "bucket", "aux/", S3Listing::CurrentObjects).unwrap();
    let plans = request
        .resolve(&script, &limits, &admission, &cancel)
        .unwrap();
    script.finished();
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].0.version(), "v1");
    assert_eq!(plans[1].0.version(), "v2");
    assert!(
        plans
            .iter()
            .all(|(_, p)| p.files()[0].role() == FileRole::Auxiliary
                && p.files()[0].digest_provenance() == DigestProvenance::FirstSeen)
    );
    assert_no_host_reservation(&admission);
}

#[test]
fn version_marker_pair_skips_historical_and_deleted_latest_without_fetching_them() {
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let cancel = AtomicBool::new(false);
    let first = "/bucket/?encoding-type=url&max-keys=1000&prefix=aux%2F&versions=";
    let second = "/bucket/?encoding-type=url&key-marker=aux%2Fa&max-keys=1000&prefix=aux%2F&version-id-marker=old&versions=";
    let script = Script::new(
        vec![
            Row::listing(
                first,
                page(
                    true,
                    &format!(
                        "{}{}{}",
                        version("a", "v1", true, false),
                        version("a", "old", false, false),
                        version("deleted", "gone", true, true)
                    ),
                    true,
                    "<NextKeyMarker>aux%2Fa</NextKeyMarker><NextVersionIdMarker>old</NextVersionIdMarker>",
                ),
            ),
            Row::head("aux/a", Some("v1"), "v1", 3),
            Row::listing(
                second,
                page(true, &version("b", "v2", true, false), false, ""),
            ),
            Row::head("aux/b", Some("v2"), "v2", 3),
        ],
        &admission,
    );
    let plans = S3PrefixRequest::new(&endpoint(), "bucket", "aux/", S3Listing::LatestVersions)
        .unwrap()
        .resolve(&script, &limits, &admission, &cancel)
        .unwrap();
    script.finished();
    assert_eq!(plans.len(), 2);
    assert_no_host_reservation(&admission);
}

#[test]
fn aggregate_calls_bytes_and_xml_refusals_never_return_a_partial_ready_selection() {
    let admission = admission(128 * 1024 * 1024);
    let cancel = AtomicBool::new(false);
    let request =
        S3PrefixRequest::new(&endpoint(), "bucket", "aux/", S3Listing::CurrentObjects).unwrap();
    let first = "/bucket/?encoding-type=url&list-type=2&max-keys=1000&prefix=aux%2F";
    let script = Script::new(
        vec![Row::listing(first, page(false, &content("a"), false, ""))],
        &admission,
    );
    let limits = PullLimits {
        max_pages: 1,
        ..Default::default()
    };
    assert!(
        request
            .resolve(&script, &limits, &admission, &cancel)
            .is_err()
    );
    script.finished();
    for raw in [
        page(false, &content("../escape"), false, ""),
        page(false, "", true, ""),
        page(false, "", false, "<IsTruncated>false</IsTruncated>"),
        "<!DOCTYPE x [<!ENTITY y SYSTEM 'https://NEVER_FOLLOW'>]><x/>".into(),
    ] {
        let script = Script::new(vec![Row::listing(first, raw)], &admission);
        assert!(
            request
                .resolve(&script, &PullLimits::default(), &admission, &cancel)
                .is_err()
        );
        script.finished();
    }
    let script = Script::new(
        vec![
            Row::listing(first, page(false, &content("a"), false, "")),
            Row::head("aux/a", None, "v1", 3),
        ],
        &admission,
    );
    let limits = PullLimits {
        cache: zerogravity_artifacts::cache::CacheLimits {
            max_snapshot_bytes: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(
        request
            .resolve(&script, &limits, &admission, &cancel)
            .is_err()
    );
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn denied_admission_precancel_and_legacy_transports_refuse_before_metadata() {
    let denied = admission(1);
    let script = Script::new(vec![], &denied);
    assert!(
        object(None, None)
            .resolve(
                &script,
                &PullLimits::default(),
                &denied,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    let admission = admission(128 * 1024 * 1024);
    let script = Script::new(vec![], &admission);
    assert!(
        object(None, None)
            .resolve(
                &script,
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(true)
            )
            .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    struct Legacy;
    impl HttpTransport for Legacy {
        fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("no fallback to legacy GET")
        }
    }
    assert!(
        object(None, None)
            .resolve(
                &Legacy,
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    struct FailedHead(bool);
    impl HttpTransport for FailedHead {
        fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("no fallback")
        }
        fn head_target(
            &self,
            _: &HttpRequest,
            _: &HttpTarget,
            cancel: &AtomicBool,
        ) -> Result<HttpResponse> {
            if self.0 {
                cancel.store(true, Ordering::Release);
            }
            Err(Error::State(
                "SECRET https://unrequested.invalid/private".into(),
            ))
        }
    }
    for canceled in [false, true] {
        let cancel = AtomicBool::new(false);
        let error = object(None, None)
            .resolve(
                &FailedHead(canceled),
                &PullLimits::default(),
                &admission,
                &cancel,
            )
            .unwrap_err();
        assert!(!format!("{error} {error:?}").contains("SECRET"));
        if canceled {
            assert!(error.to_string().contains("canceled"));
        }
    }
    assert_no_host_reservation(&admission);
    assert_no_host_reservation(&denied);
}
