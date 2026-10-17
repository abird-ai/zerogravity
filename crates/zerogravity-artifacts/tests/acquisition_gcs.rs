//! GCS protocol selection is independent of the original cache publisher.
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
use zerogravity_artifacts::Result;
use zerogravity_artifacts::acquisition::gcs::{GcsObjectPin, GcsObjectRequest, GcsPrefixRequest};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use zerogravity_artifacts::acquisition::{
    DigestProvenance, PullLimits, ResolvedPullPlan, execute, execute_with_progress,
};
use zerogravity_artifacts::cache::{FileRole, TransferAdmission};

const ORIGIN: &str = "https://storage.googleapis.com";
const OBJECT: &str = "folder/data.bin";
const GENERATION: &str = "9001";
const METAGENERATION: &str = "3";
const TAG: &str = "\"strong-opaque-validator\"";
const SECRET: &str = "UNTRUSTED_MEDIA_LINK_SIGNED_QUERY_SECRET";

enum Body {
    Data(Vec<u8>),
    Never,
    Interrupted(Vec<u8>),
}
struct Row {
    path: String,
    query: Vec<(String, String)>,
    range: Option<u64>,
    status: u16,
    headers: BTreeMap<String, String>,
    body: Body,
    payload: bool,
}
struct Script {
    rows: Mutex<VecDeque<Row>>,
    admission: TransferAdmission,
    payload_bytes: Arc<AtomicUsize>,
}
struct Counted {
    data: Cursor<Vec<u8>>,
    bytes: Arc<AtomicUsize>,
    interrupted: bool,
}
impl Read for Counted {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let read = self.data.read(output)?;
        if read == 0 && self.interrupted {
            return Err(io::Error::other("fixture interruption"));
        }
        self.bytes.fetch_add(read, Ordering::Relaxed);
        Ok(read)
    }
}
struct Forbidden;
impl Read for Forbidden {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("refused response body was read")
    }
}
impl Script {
    fn new(rows: Vec<Row>, admission: &TransferAdmission) -> Self {
        Self {
            rows: Mutex::new(rows.into()),
            admission: admission.clone(),
            payload_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get_without_redirects(
        &self,
        request: &HttpRequest,
        cancel: &AtomicBool,
    ) -> Result<HttpResponse> {
        self.get(request, cancel)
    }
    fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0,
            "admission precedes all metadata/body requests"
        );
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        let url = url::Url::parse(&request.url).unwrap();
        assert_eq!(url.origin().ascii_serialization(), ORIGIN);
        assert_eq!(url.path(), row.path);
        let query: BTreeMap<_, _> = url
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        for (key, value) in row.query {
            assert_eq!(query.get(&key), Some(&value));
        }
        assert_eq!(request.range.as_ref().map(|range| range.start), row.range);
        let bytes = if row.payload {
            self.payload_bytes.clone()
        } else {
            Arc::new(AtomicUsize::new(0))
        };
        let body: Box<dyn Read + Send> = match row.body {
            Body::Never => Box::new(Forbidden),
            Body::Data(data) => Box::new(Counted {
                data: Cursor::new(data),
                bytes,
                interrupted: false,
            }),
            Body::Interrupted(data) => Box::new(Counted {
                data: Cursor::new(data),
                bytes,
                interrupted: true,
            }),
        };
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body,
        })
    }
}
fn object(name: &str, size: u64) -> Value {
    json!({"kind":"storage#object", "bucket":"fixture-bucket", "name":name, "generation":GENERATION,
        "metageneration":METAGENERATION,"size":size.to_string(),"md5Hash":"not-a-raw-sha256",
        "crc32c":"not-a-raw-sha256","etag":"not-a-raw-sha256",
        "mediaLink":format!("https://example.invalid/other?signature={SECRET}")})
}
fn metadata(name: &str, value: Value) -> Row {
    let data = serde_json::to_vec(&value).unwrap();
    Row { path: format!("/storage/v1/b/fixture-bucket/o/{}", name.replace('/', "%2F")), query: vec![("fields".into(), "kind,bucket,name,generation,metageneration,size,contentEncoding,customerEncryption,softDeleteTime,restoreToken".into())], range: None, status:200,
        headers:BTreeMap::from([("content-length".into(),data.len().to_string())]),body:Body::Data(data),payload:false }
}
fn page(items: Vec<Value>, token: Option<&str>, expected_token: Option<&str>) -> Row {
    let mut value = json!({"items":items});
    if let Some(token) = token {
        value["nextPageToken"] = json!(token);
    }
    let data = serde_json::to_vec(&value).unwrap();
    let mut query = vec![("prefix".into(), "folder/".into())];
    if let Some(token) = expected_token {
        query.push(("pageToken".into(), token.into()));
    }
    Row {
        path: "/storage/v1/b/fixture-bucket/o".into(),
        query,
        range: None,
        status: 200,
        headers: BTreeMap::new(),
        body: Body::Data(data),
        payload: false,
    }
}
fn media(data: &[u8], start: Option<u64>) -> Row {
    let mut headers = BTreeMap::from([
        ("etag".into(), TAG.into()),
        (
            "content-length".into(),
            (data.len() as u64 - start.unwrap_or(0)).to_string(),
        ),
    ]);
    if let Some(start) = start {
        headers.insert(
            "content-range".into(),
            format!("bytes {start}-{}/{}", data.len() - 1, data.len()),
        );
    }
    Row {
        path: "/storage/v1/b/fixture-bucket/o/folder%2Fdata.bin".into(),
        query: vec![
            ("alt".into(), "media".into()),
            ("generation".into(), GENERATION.into()),
            ("ifMetagenerationMatch".into(), METAGENERATION.into()),
        ],
        range: start,
        status: if start.is_some() { 206 } else { 200 },
        headers,
        body: Body::Data(data[start.unwrap_or(0) as usize..].to_vec()),
        payload: true,
    }
}
fn request(hash: Option<&str>) -> GcsObjectRequest {
    GcsObjectRequest::new(
        ORIGIN,
        "fixture-bucket",
        OBJECT,
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        None,
        hash,
    )
    .unwrap()
}
fn resolve(
    request: &GcsObjectRequest,
    size: u64,
    admission: &TransferAdmission,
) -> ResolvedPullPlan {
    let script = Script::new(vec![metadata(OBJECT, object(OBJECT, size))], admission);
    let (pin, plan) = request
        .resolve_with_pin(
            &script,
            &PullLimits::default(),
            admission,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(pin.generation(), 9001);
    assert_eq!(pin.metageneration(), 3);
    assert_eq!(pin.bytes(), size);
    assert_eq!(pin.source(), plan.source());
    script.finished();
    plan
}

#[test]
fn pure_inputs_origin_normalization_and_version_contracts_are_strict() {
    for origin in [
        "https://storage.googleapis.com/?secret=1",
        "https://@storage.googleapis.com",
        "https://storage.googleapis.com/a/..",
        " https://storage.googleapis.com",
        "https://storage.googleapis.com/#fragment",
        "https://other.invalid",
    ] {
        assert!(
            GcsObjectRequest::new(
                origin,
                "fixture-bucket",
                OBJECT,
                PAYLOAD_PATH,
                FileRole::Auxiliary,
                None,
                None
            )
            .is_err()
        );
    }
    let a = GcsObjectRequest::new(
        "HTTPS://STORAGE.GOOGLEAPIS.COM:443/",
        "fixture-bucket",
        OBJECT,
        PAYLOAD_PATH,
        FileRole::Auxiliary,
        Some(GENERATION),
        None,
    )
    .unwrap();
    assert_eq!(a.origin(), ORIGIN);
    for generation in ["0", "01", "+1", "-1", "1.0", "1e2", "9223372036854775808"] {
        assert!(
            GcsObjectRequest::new(
                ORIGIN,
                "fixture-bucket",
                OBJECT,
                PAYLOAD_PATH,
                FileRole::Auxiliary,
                Some(generation),
                None
            )
            .is_err()
        );
        assert!(GcsObjectPin::new(&a, GENERATION, generation, 1).is_err());
    }
    assert!(GcsObjectPin::new(&a, "9002", METAGENERATION, 1).is_err());
    let pin = GcsObjectPin::new(&a, GENERATION, METAGENERATION, 1).unwrap();
    assert_eq!(pin.generation(), 9001);
    assert_eq!(pin.metageneration(), 3);
    assert!(
        pin.source()
            .revision()
            .starts_with("generation-9001-metageneration-3-contract-")
    );
    assert_ne!(
        pin.source(),
        GcsObjectPin::new(&a, GENERATION, "4", 1).unwrap().source()
    );
    assert!(GcsPrefixRequest::new(ORIGIN, "fixture-bucket", "folder").is_err());
    assert!(
        GcsObjectRequest::new(
            ORIGIN,
            "fixture-bucket",
            "..",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            None,
            None
        )
        .is_err()
    );
}

#[test]
fn metadata_admission_cancel_and_limits_precede_requests() {
    struct Never;
    impl HttpTransport for Never {
        fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("request before admission/cancel")
        }
    }
    let request = request(None);
    let denied = admission(1);
    assert!(
        request
            .resolve(
                &Never,
                &PullLimits::default(),
                &denied,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_no_host_reservation(&denied);
    let allowed = admission(128 * 1024 * 1024);
    assert!(
        request
            .resolve(
                &Never,
                &PullLimits::default(),
                &allowed,
                &AtomicBool::new(true)
            )
            .is_err()
    );
    assert_no_host_reservation(&allowed);
    let invalid = PullLimits {
        max_pages: 0,
        ..PullLimits::default()
    };
    assert!(
        request
            .resolve(&Never, &invalid, &allowed, &AtomicBool::new(false))
            .is_err()
    );
}

#[test]
fn metadata_pin_ignores_media_links_and_crc_md5_never_upgrade_raw_trust() {
    let admission = admission(128 * 1024 * 1024);
    let request = request(None);
    let plan = resolve(&request, NEW.len() as u64, &admission);
    assert_eq!(plan.files()[0].sha256(), None);
    assert_eq!(
        plan.files()[0].digest_provenance(),
        DigestProvenance::FirstSeen
    );
    assert!(!format!("{plan:?}").contains(SECRET));
    assert!(!plan.files()[0].url().contains(SECRET));
    assert_eq!(
        plan.source(),
        GcsObjectPin::new(&request, GENERATION, METAGENERATION, NEW.len() as u64)
            .unwrap()
            .source()
    );
    assert_no_host_reservation(&admission);
}

#[test]
fn contradictory_or_unsupported_metadata_refuses_before_media() {
    let admission = admission(128 * 1024 * 1024);
    for (field, replacement) in [
        ("kind", json!("other")),
        ("bucket", json!("other-bucket")),
        ("name", json!("other.bin")),
        ("generation", json!("0")),
        ("generation", json!("01")),
        ("metageneration", json!("-1")),
        ("size", json!("1.0")),
        ("size", json!("01")),
        ("size", json!("18446744073709551616")),
        ("contentEncoding", json!("gzip")),
        (
            "customerEncryption",
            json!({"encryptionAlgorithm":"AES256"}),
        ),
        ("softDeleteTime", json!("now")),
        ("restoreToken", json!("unimplemented")),
    ] {
        let mut value = object(OBJECT, NEW.len() as u64);
        value[field] = replacement;
        let script = Script::new(vec![metadata(OBJECT, value)], &admission);
        assert!(
            request(None)
                .resolve(
                    &script,
                    &PullLimits::default(),
                    &admission,
                    &AtomicBool::new(false)
                )
                .is_err(),
            "field {field}"
        );
        script.finished();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn absent_or_matching_media_version_headers_succeed_and_offline_is_exact() {
    for trusted in [false, true] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let hash = digest(NEW);
        let request = request(trusted.then_some(hash.as_str()));
        let plan = resolve(&request, NEW.len() as u64, &admission);
        let mut row = media(NEW, None);
        if trusted {
            row.headers
                .insert("x-goog-generation".into(), GENERATION.into());
            row.headers
                .insert("x-goog-metageneration".into(), METAGENERATION.into());
        }
        let script = Script::new(vec![row], &admission);
        let receipt = execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        )
        .unwrap();
        script.finished();
        assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
        assert_eq!(script.payload_bytes.load(Ordering::Relaxed), NEW.len());
        assert_eq!(
            receipt.files()[0].provenance(),
            if trusted {
                DigestProvenance::TrustedSha256
            } else {
                DigestProvenance::FirstSeen
            }
        );
        let pin =
            GcsObjectPin::new(&request, GENERATION, METAGENERATION, NEW.len() as u64).unwrap();
        let digest = receipt.manifest().digest().unwrap();
        assert_payload(
            &pin.open_cached(
                &store,
                &digest,
                &PullLimits::default(),
                &AtomicBool::new(false),
            )
            .unwrap(),
            NEW,
        );
        assert!(
            GcsObjectPin::new(&request, GENERATION, "4", NEW.len() as u64)
                .unwrap()
                .open_cached(
                    &store,
                    &digest,
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        let low = PullLimits {
            cache: zerogravity_artifacts::cache::CacheLimits {
                max_file_bytes: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            pin.open_cached(&store, &digest, &low, &AtomicBool::new(false))
                .is_err()
        );
        assert!(
            pin.open_cached(
                &store,
                &digest,
                &PullLimits::default(),
                &AtomicBool::new(true)
            )
            .is_err()
        );
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn changed_present_versions_or_precondition_failure_read_no_body_and_preserve_ref() {
    for (header, value, status) in [
        ("x-goog-generation", "9002", 200),
        ("x-goog-metageneration", "4", 200),
        ("x-goog-generation", "09001", 200),
        ("x-goog-metageneration", "", 200),
        ("ignored", "", 412),
    ] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let plan = resolve(&request(None), NEW.len() as u64, &admission);
        let mut row = media(NEW, None);
        row.status = status;
        row.headers.insert(header.into(), value.into());
        row.body = Body::Never;
        let script = Script::new(vec![row], &admission);
        assert!(
            execute(
                &store,
                &plan,
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                Some("fixture")
            )
            .is_err()
        );
        assert_eq!(
            store
                .resolve_ref("fixture")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            old
        );
        assert_eq!(script.payload_bytes.load(Ordering::Relaxed), 0);
        script.finished();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn prefix_pages_are_bounded_independent_first_seen_objects() {
    let admission = admission(128 * 1024 * 1024);
    let token = "opaque+/=token";
    let script = Script::new(
        vec![
            page(vec![object("folder/a.bin", 2)], Some(token), None),
            page(vec![object("folder/b.bin", 3)], None, Some(token)),
        ],
        &admission,
    );
    let objects = GcsPrefixRequest::new(ORIGIN, "fixture-bucket", "folder/")
        .unwrap()
        .resolve_with_pins(
            &script,
            &PullLimits::default(),
            &admission,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(objects[0].0.generation(), 9001);
    assert_eq!(objects[0].0.metageneration(), 3);
    assert_eq!(objects[0].0.bytes(), 2);
    assert_eq!(objects[1].0.bytes(), 3);
    assert_eq!(objects[0].0.request().object(), "folder/a.bin");
    let plans: Vec<_> = objects.into_iter().map(|(_, plan)| plan).collect();
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].files()[0].path(), "a.bin");
    assert_eq!(plans[1].files()[0].path(), "b.bin");
    assert_ne!(plans[0].source(), plans[1].source());
    assert!(plans.iter().all(|plan| plan.files().len() == 1
        && plan.files()[0].role() == FileRole::Auxiliary
        && plan.files()[0].digest_provenance() == DigestProvenance::FirstSeen));
    assert!(!format!("{plans:?}").contains(token));
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn prefix_unsafe_duplicate_cycle_and_aggregate_bounds_refuse_without_payload() {
    let admission = admission(128 * 1024 * 1024);
    for items in [
        vec![],
        vec![object("outside/a", 1)],
        vec![object("folder/../a", 1)],
        vec![object("folder/", 1)],
        vec![object("folder/a", 1), object("folder/a", 1)],
        vec![object("folder/a", 1), object("folder/a/b", 1)],
    ] {
        let script = Script::new(vec![page(items, None, None)], &admission);
        assert!(
            GcsPrefixRequest::new(ORIGIN, "fixture-bucket", "folder/")
                .unwrap()
                .resolve(
                    &script,
                    &PullLimits::default(),
                    &admission,
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        script.finished();
    }
    for limits in [
        PullLimits {
            max_pages: 1,
            ..Default::default()
        },
        PullLimits {
            max_metadata_bytes: 1,
            ..Default::default()
        },
        PullLimits {
            cache: zerogravity_artifacts::cache::CacheLimits {
                max_files: 1,
                ..Default::default()
            },
            ..Default::default()
        },
        PullLimits {
            cache: zerogravity_artifacts::cache::CacheLimits {
                max_snapshot_bytes: 2,
                ..Default::default()
            },
            ..Default::default()
        },
    ] {
        let rows = if limits.max_pages == 1 {
            vec![page(vec![object("folder/a", 1)], Some("next"), None)]
        } else {
            vec![page(
                vec![object("folder/a", 2), object("folder/b", 2)],
                None,
                None,
            )]
        };
        let script = Script::new(rows, &admission);
        assert!(
            GcsPrefixRequest::new(ORIGIN, "fixture-bucket", "folder/")
                .unwrap()
                .resolve(&script, &limits, &admission, &AtomicBool::new(false))
                .is_err()
        );
        script.finished();
        assert_no_host_reservation(&admission);
    }
    let script = Script::new(
        vec![
            page(vec![object("folder/a", 1)], Some("repeat"), None),
            page(vec![object("folder/b", 1)], Some("repeat"), Some("repeat")),
        ],
        &admission,
    );
    assert!(
        GcsPrefixRequest::new(ORIGIN, "fixture-bucket", "folder/")
            .unwrap()
            .resolve(
                &script,
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    script.finished();
}

#[test]
fn durable_resume_revalidates_present_versions_and_optional_headers_remain_legal() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let data: Vec<u8> = (0..1024).map(|n| (n % 251) as u8).collect();
    let plan = resolve(
        &request(Some(&digest(&data))),
        data.len() as u64,
        &admission,
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![media(&data, None)], &admission);
    assert!(
        execute_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &cancel,
            Some("fixture"),
            &mut |progress| {
                if progress.file_bytes() == 128 {
                    cancel.store(true, Ordering::Release);
                }
            }
        )
        .is_err()
    );
    assert_eq!(script.payload_bytes.load(Ordering::Relaxed), 128);
    let mut bad = media(&data, Some(128));
    bad.headers
        .insert("x-goog-metageneration".into(), "4".into());
    bad.body = Body::Never;
    let script = Script::new(vec![bad], &admission);
    assert!(
        execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture")
        )
        .is_err()
    );
    assert_eq!(script.payload_bytes.load(Ordering::Relaxed), 0);
    script.finished();
    assert_eq!(
        store
            .resolve_ref("fixture")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        old
    );
    let script = Script::new(vec![media(&data, Some(128))], &admission);
    let mut resumed = 0;
    execute_with_progress(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        Some("fixture"),
        &mut |progress| {
            resumed = progress.resumed_bytes();
        },
    )
    .unwrap();
    assert_eq!(resumed, 128);
    assert_eq!(
        script.payload_bytes.load(Ordering::Relaxed),
        data.len() - 128
    );
    assert_payload(&store.resolve_ref("fixture").unwrap(), &data);
    assert_no_host_reservation(&admission);
}

#[test]
fn live_retry_revalidates_object_version_before_accepting_new_body() {
    for valid in [false, true] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let data: Vec<u8> = (0..1024).map(|n| (n % 251) as u8).collect();
        let plan = resolve(
            &request(Some(&digest(&data))),
            data.len() as u64,
            &admission,
        );
        let mut initial = media(&data, None);
        initial.body = Body::Interrupted(data[..128].to_vec());
        let mut continuation = media(&data, Some(128));
        if !valid {
            continuation
                .headers
                .insert("x-goog-generation".into(), "9002".into());
            continuation.body = Body::Never;
        }
        let script = Script::new(vec![initial, continuation], &admission);
        let result = execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
        );
        assert_eq!(result.is_ok(), valid);
        script.finished();
        assert_eq!(
            script.payload_bytes.load(Ordering::Relaxed),
            if valid { data.len() } else { 128 }
        );
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn empty_exact_object_succeeds_and_wrong_authorized_raw_digest_preserves_ref() {
    for data in [b"".as_slice(), NEW] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let hash = if data.is_empty() {
            digest(data)
        } else {
            "0".repeat(64)
        };
        let plan = resolve(&request(Some(&hash)), data.len() as u64, &admission);
        let script = Script::new(vec![media(data, None)], &admission);
        let result = execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        );
        assert_eq!(result.is_ok(), data.is_empty());
        if !data.is_empty() {
            assert_eq!(
                store
                    .resolve_ref("fixture")
                    .unwrap()
                    .manifest()
                    .digest()
                    .unwrap(),
                old
            );
        }
        script.finished();
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn legacy_custom_transport_refuses_version_bound_media_until_explicit_capability() {
    struct Legacy;
    impl HttpTransport for Legacy {
        fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
            panic!("version-bound media must not silently use legacy redirect behavior")
        }
    }
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let plan = resolve(&request(None), NEW.len() as u64, &admission);
    assert!(
        execute(
            &store,
            &plan,
            &Legacy,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None
        )
        .is_err()
    );
    assert_no_host_reservation(&admission);
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_direct_http_preserves_query_and_optional_headers_but_refuses_all_redirects() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::time::Duration;
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    for mode in ["direct", "missing-query", "same-query"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let target = "/storage/v1/b/fixture-bucket/o/folder%2Fdata.bin?alt=media&generation=9001&ifMetagenerationMatch=3";
        let response = match mode {
            "direct" => {
                "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".to_owned()
            }
            "missing-query" => format!(
                "HTTP/1.1 302 Found\r\nLocation: /other?signature={SECRET}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            ),
            _ => format!(
                "HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            ),
        };
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
                assert!(request.len() <= 16 * 1024);
            }
            let request = String::from_utf8(request).unwrap();
            assert_eq!(
                request.lines().next(),
                Some(format!("GET {target} HTTP/1.1").as_str())
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let client = HttpsClient::diagnostic_loopback(
            &origin,
            None,
            HttpOptions {
                max_retries: 0,
                total_timeout: Duration::from_secs(3),
                ..Default::default()
            },
        )
        .unwrap();
        let result = client.get_without_redirects(
            &HttpRequest::new(format!("{origin}{target}"), 2),
            &AtomicBool::new(false),
        );
        match mode {
            "direct" => assert_eq!(
                result
                    .unwrap()
                    .read_bounded(2, &AtomicBool::new(false))
                    .unwrap(),
                b"ok"
            ),
            _ => {
                let error = result.err().expect("redirect refused");
                assert!(matches!(
                    error,
                    zerogravity_artifacts::Error::Unsupported(_)
                ));
                assert!(!error.to_string().contains(SECRET));
            }
        }
        server.join().unwrap();
    }
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_json_to_media_checks_captured_versions_and_duplicate_headers() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::time::Duration;
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    for mode in ["missing", "matching", "changed", "duplicate"] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let metadata = serde_json::to_vec(&object(OBJECT, NEW.len() as u64)).unwrap();
        let headers = match mode {
            "missing" => "",
            "matching" => "X-Goog-Generation: 9001\r\nX-Goog-Metageneration: 3\r\n",
            "changed" => "X-Goog-Generation: 9001\r\nX-Goog-Metageneration: 4\r\n",
            _ => {
                "X-Goog-Generation: 9001\r\nX-Goog-Generation: 9001\r\nX-Goog-Metageneration: 3\r\n"
            }
        };
        let server = std::thread::spawn(move || {
            for payload in [false, true] {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    let mut byte = [0];
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                    assert!(request.len() <= 16 * 1024);
                }
                let request = String::from_utf8(request).unwrap();
                assert!(
                    request.starts_with("GET /storage/v1/b/fixture-bucket/o/folder%2Fdata.bin?")
                );
                let body = if payload { NEW } else { metadata.as_slice() };
                if payload {
                    assert!(request.contains("alt=media&generation=9001&ifMetagenerationMatch=3"));
                } else {
                    assert!(request.contains("fields="));
                }
                let mut response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
                    body.len(),
                    if payload { headers } else { "" }
                )
                .into_bytes();
                response.extend_from_slice(body);
                stream.write_all(&response).unwrap();
            }
        });
        let client = HttpsClient::diagnostic_loopback(
            &origin,
            None,
            HttpOptions {
                max_retries: 0,
                total_timeout: Duration::from_secs(3),
                ..Default::default()
            },
        )
        .unwrap();
        let request = GcsObjectRequest::new(
            &origin,
            "fixture-bucket",
            OBJECT,
            PAYLOAD_PATH,
            FileRole::Auxiliary,
            None,
            None,
        )
        .unwrap();
        let plan = request
            .resolve(
                &client,
                &PullLimits::default(),
                &admission,
                &AtomicBool::new(false),
            )
            .unwrap();
        let result = execute(
            &store,
            &plan,
            &client,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            Some("fixture"),
        );
        assert_eq!(result.is_ok(), matches!(mode, "missing" | "matching"));
        if result.is_ok() {
            assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
        } else {
            assert_eq!(
                store
                    .resolve_ref("fixture")
                    .unwrap()
                    .manifest()
                    .digest()
                    .unwrap(),
                old
            );
        }
        server.join().unwrap();
        assert_no_host_reservation(&admission);
    }
}
