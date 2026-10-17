//! Explicit AWS-standard HeadObject FULL_OBJECT SHA256 trust gate.
//! Synthetic typed transport, original metadata admission and cache publisher.
//! The transport does not assert AWS authenticity outside the selected HTTPS
//! canonical AWS profile (and is never used for compatibility promotion).
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use cache_transaction_support::*;
use sha2::Digest;
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
    S3Addressing, S3Endpoint, S3ObjectPin, S3ObjectRequest,
};
use zerogravity_artifacts::acquisition::{PullLimits, execute};
use zerogravity_artifacts::cache::{CommitOutcome, DigestProvenance, FileRole, TransferAdmission};
use zerogravity_artifacts::{Error, Result};

const DATA: &[u8] = b"fixed trusted AWS S3 full object";
const VERSION: &str = "immutable-00000001";
fn aws() -> S3Endpoint {
    S3Endpoint::aws("us-east-1", S3Addressing::PathStyle).unwrap()
}
fn object(sha: Option<&str>, version: Option<&str>) -> S3ObjectRequest {
    S3ObjectRequest::new(
        &aws(),
        "bucket",
        "prefix/model.gguf",
        "model.gguf",
        FileRole::Weights,
        version,
        sha,
    )
    .unwrap()
}
fn header_sha(data: &[u8]) -> String {
    STANDARD.encode(sha2::Sha256::digest(data))
}
struct NeverRead;
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("HEAD must not consume a response body")
    }
}
#[derive(Clone)]
struct Row {
    head: bool,
    suffix: String,
    headers: BTreeMap<String, String>,
    data: Vec<u8>,
    status: u16,
    checksum_mode: bool,
}
impl Row {
    fn head(data: &[u8], mode: bool) -> Self {
        Self {
            head: true,
            suffix: "/bucket/prefix/model.gguf".into(),
            data: vec![],
            status: 200,
            headers: BTreeMap::from([
                ("content-length".into(), data.len().to_string()),
                ("x-amz-version-id".into(), VERSION.into()),
                ("etag".into(), "\"stable-etag\"".into()),
                ("x-amz-checksum-type".into(), "FULL_OBJECT".into()),
                ("x-amz-checksum-sha256".into(), header_sha(data)),
            ]),
            checksum_mode: mode,
        }
    }
    fn get(data: &[u8]) -> Self {
        Self {
            head: false,
            suffix: format!("/bucket/prefix/model.gguf?versionId={VERSION}"),
            headers: BTreeMap::from([
                ("content-length".into(), data.len().to_string()),
                ("x-amz-version-id".into(), VERSION.into()),
                ("etag".into(), "\"stable-etag\"".into()),
            ]),
            data: data.to_vec(),
            status: 200,
            checksum_mode: false,
        }
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
    fn response(&self, req: &HttpRequest, target: &HttpTarget, head: bool) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0
        );
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unscripted S3 request");
        assert_eq!(head, row.head);
        assert_eq!(target.s3_checksum_mode_enabled(), row.checksum_mode);
        assert!(req.url.ends_with(&row.suffix), "{:?}", row.suffix);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body: if head {
                Box::new(NeverRead)
            } else {
                Box::new(Cursor::new(row.data))
            },
        })
    }
    fn empty(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("S3 must use typed target")
    }
    fn get_target(
        &self,
        req: &HttpRequest,
        target: &HttpTarget,
        _: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.response(req, target, false)
    }
    fn head_target(
        &self,
        req: &HttpRequest,
        target: &HttpTarget,
        _: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.response(req, target, true)
    }
}
#[test]
fn opted_whole_object_sha256_is_trusted_then_verified_by_original_importer_and_cache() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let script = Script::new(vec![Row::head(DATA, true), Row::get(DATA)], &admission);
    let request = object(None, None).with_aws_full_sha256().unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let limits = PullLimits::default();
    let (pin, plan) = request
        .resolve_with_pin(&script, &limits, &admission, &cancel)
        .unwrap();
    assert_eq!(script.calls.load(Ordering::Relaxed), 1);
    assert_eq!(pin.version(), VERSION);
    assert_eq!(pin.bytes(), DATA.len() as u64);
    assert_eq!(plan.files()[0].sha256(), Some(digest(DATA).as_str()));
    assert_eq!(
        plan.files()[0].digest_provenance(),
        DigestProvenance::TrustedSha256
    );
    let receipt = execute(&store, &plan, &script, &admission, &cancel, Some("trusted")).unwrap();
    assert!(matches!(receipt.outcome(), CommitOutcome::Durable(_)));
    let lease = store.resolve_ref("trusted").unwrap();
    assert_eq!(file_bytes(lease.file("model.gguf").unwrap()), DATA);
    assert_eq!(
        lease.manifest().entries()[0].digest_provenance(),
        DigestProvenance::TrustedSha256
    );
    script.empty();
    let offline = object(Some(&digest(DATA)), Some(VERSION))
        .with_aws_full_sha256()
        .unwrap();
    let expected = S3ObjectPin::new(&offline, VERSION, DATA.len() as u64).unwrap();
    let id = lease.manifest().digest().unwrap();
    expected.open_cached(&store, &id, &limits, &cancel).unwrap();
    let wrong = object(Some(&digest(b"other")), Some(VERSION))
        .with_aws_full_sha256()
        .unwrap();
    assert!(
        S3ObjectPin::new(&wrong, VERSION, DATA.len() as u64)
            .unwrap()
            .open_cached(&store, &id, &limits, &cancel)
            .is_err()
    );
    assert!(
        S3ObjectPin::new(&offline, VERSION, DATA.len() as u64 + 1)
            .unwrap()
            .open_cached(&store, &id, &limits, &cancel)
            .is_err()
    );
    assert_no_host_reservation(&admission);
}
#[test]
fn without_opt_in_checksum_headers_cannot_upgrade_first_seen_digest() {
    let admission = admission(128 * 1024 * 1024);
    let script = Script::new(vec![Row::head(DATA, false)], &admission);
    let (_, plan) = object(None, None)
        .resolve_with_pin(
            &script,
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(plan.files()[0].sha256(), None);
    assert_eq!(
        plan.files()[0].digest_provenance(),
        DigestProvenance::FirstSeen
    );
    script.empty();
    assert_no_host_reservation(&admission);
}
#[test]
fn checksum_authority_must_be_standard_aws_not_compatible_even_if_endpoint_mimics_aws() {
    for origin in [
        "https://s3.us-east-1.amazonaws.com",
        "https://object.example",
    ] {
        let endpoint =
            S3Endpoint::compatible(origin, "us-east-1", S3Addressing::PathStyle).unwrap();
        let request = S3ObjectRequest::new(
            &endpoint,
            "bucket",
            "prefix/model.gguf",
            "model.gguf",
            FileRole::Weights,
            None,
            None,
        )
        .unwrap();
        assert!(matches!(
            request.with_aws_full_sha256(),
            Err(Error::Unsupported(_))
        ));
    }
}
#[test]
fn missing_malformed_composite_or_contradictory_head_data_refuses_before_get() {
    let admission = admission(128 * 1024 * 1024);
    for violation in [
        "missing-sha",
        "missing-type",
        "composite",
        "wrong-type",
        "short",
        "bad-padding",
        "lowercase",
        "wrong-expected",
        "wrong-version",
        "missing-version",
        "wrong-length",
    ] {
        let mut row = Row::head(DATA, true);
        let mut sha = None;
        match violation {
            "missing-sha" => {
                row.headers.remove("x-amz-checksum-sha256");
            }
            "missing-type" => {
                row.headers.remove("x-amz-checksum-type");
            }
            "composite" => {
                row.headers
                    .insert("x-amz-checksum-type".into(), "COMPOSITE".into());
            }
            "wrong-type" => {
                row.headers
                    .insert("x-amz-checksum-type".into(), "FULL_OBJECT,COMPOSITE".into());
            }
            "short" => {
                row.headers
                    .insert("x-amz-checksum-sha256".into(), "YWJj".into());
            }
            "bad-padding" => {
                row.headers
                    .insert("x-amz-checksum-sha256".into(), "!".repeat(44));
            }
            "lowercase" => {
                row.headers
                    .insert("x-amz-checksum-type".into(), "full_object".into());
            }
            "wrong-expected" => {
                sha = Some(digest(b"wrong"));
            }
            "wrong-version" => {
                row.suffix.push_str(&format!("?versionId={VERSION}"));
                row.headers
                    .insert("x-amz-version-id".into(), "different".into());
            }
            "missing-version" => {
                row.headers.remove("x-amz-version-id");
            }
            "wrong-length" => {
                row.headers.insert("content-length".into(), "-1".into());
            }
            _ => unreachable!(),
        }
        let script = Script::new(vec![row], &admission);
        let request = object(
            sha.as_deref(),
            (violation == "wrong-version").then_some(VERSION),
        )
        .with_aws_full_sha256()
        .unwrap();
        assert!(
            request
                .resolve_with_pin(
                    &script,
                    &PullLimits::default(),
                    &admission,
                    &AtomicBool::new(false)
                )
                .is_err(),
            "{violation}"
        );
        script.empty();
        assert_no_host_reservation(&admission);
    }
}
#[test]
fn offline_checksum_mode_requires_explicit_snapshot_digest_contract_not_inferred_sha() {
    let no_sha = object(None, Some(VERSION)).with_aws_full_sha256().unwrap();
    assert!(matches!(
        S3ObjectPin::new(&no_sha, VERSION, DATA.len() as u64),
        Err(Error::Unsupported(_))
    ));
}
#[test]
fn incorrect_get_bytes_cannot_publish_or_replace_existing_reference() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let script = Script::new(
        vec![Row::head(DATA, true), Row::get(&vec![b'x'; DATA.len()])],
        &admission,
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let (_pin, plan) = object(None, None)
        .with_aws_full_sha256()
        .unwrap()
        .resolve_with_pin(&script, &PullLimits::default(), &admission, &cancel)
        .unwrap();
    assert!(execute(&store, &plan, &script, &admission, &cancel, Some("keep")).is_err());
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
    script.empty();
    assert_no_host_reservation(&admission);
}
#[test]
fn mode_admission_and_cancellation_refuse_before_http_request() {
    let insufficient = admission(1);
    let request = object(None, None).with_aws_full_sha256().unwrap();
    let script = Script::new(vec![], &insufficient);
    assert!(
        request
            .resolve(
                &script,
                &PullLimits::default(),
                &insufficient,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    let sufficient = admission(128 * 1024 * 1024);
    let script = Script::new(vec![], &sufficient);
    assert!(
        request
            .resolve(
                &script,
                &PullLimits::default(),
                &sufficient,
                &AtomicBool::new(true)
            )
            .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    assert_no_host_reservation(&sufficient);
}
