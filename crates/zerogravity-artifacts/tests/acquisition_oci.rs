//! Native OCI metadata and original cache publisher; no provider/model payload.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use serde_json::json;
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Cursor, Read};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{
    HttpAccept, HttpOptions, HttpRequest, HttpResponse, HttpTransport,
};
use zerogravity_artifacts::acquisition::oci::{
    GgufLayer, OciProfile, OciReference, OciRequest, docker_hub_client,
};
use zerogravity_artifacts::acquisition::{PullLimits, execute, execute_with_progress};
use zerogravity_artifacts::cache::{CommitOutcome, DigestProvenance, TransferAdmission};
use zerogravity_artifacts::{Error, Result};

const ORIGIN: &str = "https://registry.example";
const TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
const GGUF: &str = "application/vnd.docker.ai.gguf.v3";
const DATA: &[u8] = b"tiny synthetic raw OCI GGUF data";

fn snapshot(outcome: &CommitOutcome) -> String {
    match outcome {
        CommitOutcome::Durable(published) => published.digest().to_owned(),
        CommitOutcome::DurabilityUnknown { cause, .. } => panic!("{cause}"),
    }
}

fn manifest(data: &[u8]) -> Vec<u8> {
    serde_json::to_vec(&json!({"schemaVersion":2,"mediaType":TYPE,
        "config":{"mediaType":"application/vnd.oci.empty.v1+json","digest":format!("sha256:{}",digest(b"{}")),"size":2},
        "layers":[{"mediaType":GGUF,"digest":format!("sha256:{}",digest(data)),"size":data.len()}]})).unwrap()
}
fn request(
    origin: &str,
    profile: OciProfile,
    reference: OciReference,
    layer: GgufLayer,
) -> OciRequest {
    OciRequest::new(origin, profile, "ai/toy", reference, "model.gguf", layer).unwrap()
}
fn tag(origin: &str) -> OciRequest {
    request(
        origin,
        OciProfile::Anonymous,
        OciReference::Tag("stable".into()),
        GgufLayer::Unique,
    )
}
struct Row {
    path: String,
    status: u16,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
    forbid_body: bool,
}
impl Row {
    fn new(path: impl Into<String>, body: Vec<u8>, kind: &str) -> Self {
        Self {
            path: path.into(),
            status: 200,
            headers: BTreeMap::from([
                ("content-type".into(), kind.into()),
                ("content-length".into(), body.len().to_string()),
            ]),
            body,
            forbid_body: false,
        }
    }
    fn manifest(body: Vec<u8>) -> Self {
        let mut row = Self::new("/v2/ai/toy/manifests/stable", body, TYPE);
        row.headers.insert(
            "docker-content-digest".into(),
            format!("sha256:{}", digest(&row.body)),
        );
        row
    }
    fn blob(data: &[u8]) -> Self {
        Self::new(
            format!("/v2/ai/toy/blobs/sha256:{}", digest(data)),
            data.to_vec(),
            "application/octet-stream",
        )
    }
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("refused metadata body must not be read")
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
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty());
    }
}
impl HttpTransport for Script {
    fn get(&self, request: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
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
            .expect("unexpected OCI request");
        assert!(request.url.ends_with(&row.path), "{}", row.path);
        if row.path.contains("/manifests/") {
            assert!(matches!(request.accept, HttpAccept::OciManifest));
        }
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body: if row.forbid_body {
                Box::new(NoRead)
            } else {
                Box::new(Cursor::new(row.body))
            },
        })
    }
}

#[test]
fn raw_manifest_layer_and_offline_request_binding_use_the_original_publisher() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let raw = manifest(DATA);
    let source = tag(ORIGIN);
    let script = Script::new(
        vec![Row::manifest(raw.clone()), Row::blob(DATA)],
        &admission,
    );
    let limits = PullLimits::default();
    let cancel = Arc::new(AtomicBool::new(false));
    let plan = source
        .resolve(&script, &limits, &admission, &cancel)
        .unwrap();
    assert_eq!(plan.source().revision(), digest(&raw));
    assert_eq!(plan.files()[0].sha256(), Some(digest(DATA).as_str()));
    let receipt = execute(&store, &plan, &script, &admission, &cancel, None).unwrap();
    script.finished();
    let id = snapshot(receipt.outcome());
    let lease = source.open_cached(&store, &id, &limits, &cancel).unwrap();
    assert_eq!(file_bytes(lease.file("model.gguf").unwrap()), DATA);
    assert_eq!(
        lease.manifest().entries()[0].digest_provenance(),
        DigestProvenance::TrustedSha256
    );
    for different in [
        request(
            ORIGIN,
            OciProfile::Anonymous,
            OciReference::Tag("different".into()),
            GgufLayer::Unique,
        ),
        request(
            ORIGIN,
            OciProfile::Bearer,
            OciReference::Tag("stable".into()),
            GgufLayer::Unique,
        ),
        request(
            ORIGIN,
            OciProfile::Anonymous,
            OciReference::Tag("stable".into()),
            GgufLayer::Sha256(digest(DATA)),
        ),
        tag("https://other.example"),
    ] {
        assert!(
            different
                .open_cached(&store, &id, &limits, &cancel)
                .is_err()
        );
    }
    let narrow = PullLimits {
        cache: zerogravity_artifacts::cache::CacheLimits {
            max_file_bytes: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(source.open_cached(&store, &id, &narrow, &cancel).is_err());
    assert_no_host_reservation(&admission);
}

#[test]
fn explicit_manifest_and_layer_sha_reject_mutations_before_payload() {
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let cancel = AtomicBool::new(false);
    let raw = manifest(DATA);
    let source = request(
        ORIGIN,
        OciProfile::Anonymous,
        OciReference::ManifestSha256(digest(&raw)),
        GgufLayer::Sha256(digest(DATA)),
    );
    let mut row = Row::manifest(raw.clone());
    row.path = format!("/v2/ai/toy/manifests/sha256:{}", digest(&raw));
    let script = Script::new(vec![row], &admission);
    let plan = source
        .resolve(&script, &limits, &admission, &cancel)
        .unwrap();
    assert_eq!(plan.files()[0].bytes(), Some(DATA.len() as u64));
    script.finished();
    let mut row = Row::manifest(b"not-json".to_vec());
    row.path = format!("/v2/ai/toy/manifests/sha256:{}", digest(&raw));
    let script = Script::new(vec![row], &admission);
    let error = source
        .resolve(&script, &limits, &admission, &cancel)
        .unwrap_err();
    assert!(error.to_string().contains("raw SHA256"), "{error}");
    script.finished();
    assert_no_host_reservation(&admission);
}

#[test]
fn received_type_digest_schema_descriptor_and_ambiguity_are_strict() {
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let cancel = AtomicBool::new(false);
    let mut variants = Vec::new();
    let mut bad = Row::manifest(manifest(DATA));
    bad.headers.insert(
        "docker-content-digest".into(),
        format!("sha256:{}", "a".repeat(64)),
    );
    variants.push(bad);
    for header in [
        "application/vnd.oci.image.index.v1+json",
        "application/octet-stream",
    ] {
        let mut row = Row::manifest(manifest(DATA));
        row.headers.insert("content-type".into(), header.into());
        row.forbid_body = true;
        variants.push(row);
    }
    for header in ["content-encoding", "content-range"] {
        let mut row = Row::manifest(manifest(DATA));
        row.headers.insert(
            header.into(),
            if header == "content-encoding" {
                "gzip".into()
            } else {
                "bytes 0-1/2".into()
            },
        );
        row.forbid_body = true;
        variants.push(row);
    }
    for modify in 0..5 {
        let mut value: serde_json::Value = serde_json::from_slice(&manifest(DATA)).unwrap();
        match modify {
            0 => value["layers"] = json!([value["layers"][0].clone(), value["layers"][0].clone()]),
            1 => {
                value["layers"][0]["mediaType"] =
                    json!("application/vnd.oci.image.layer.v1.tar+gzip")
            }
            2 => value["layers"][0]["urls"] = json!(["https://unrequested.invalid/SECRET"]),
            3 => value["schemaVersion"] = json!(1),
            _ => value["layers"][0]["size"] = json!(1.5),
        }
        variants.push(Row::manifest(serde_json::to_vec(&value).unwrap()));
    }
    for row in variants {
        let script = Script::new(vec![row], &admission);
        assert!(
            tag(ORIGIN)
                .resolve(&script, &limits, &admission, &cancel)
                .is_err()
        );
        script.finished();
    }
    assert_no_host_reservation(&admission);
}

#[test]
fn admission_cancel_size_and_broker_profiles_refuse_without_unowned_requests() {
    let denied = admission(1);
    let script = Script::new(vec![], &denied);
    let limits = PullLimits::default();
    assert!(
        tag(ORIGIN)
            .resolve(&script, &limits, &denied, &AtomicBool::new(false))
            .is_err()
    );
    assert!(
        tag(ORIGIN)
            .resolve(&script, &limits, &denied, &AtomicBool::new(true))
            .is_err()
    );
    assert_eq!(script.calls.load(Ordering::Relaxed), 0);
    let admission = admission(128 * 1024 * 1024);
    let script = Script::new(vec![Row::manifest(manifest(DATA))], &admission);
    let narrow = PullLimits {
        cache: zerogravity_artifacts::cache::CacheLimits {
            max_file_bytes: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(
        tag(ORIGIN)
            .resolve(&script, &narrow, &admission, &AtomicBool::new(false))
            .is_err()
    );
    script.finished();
    let source = request(
        "https://registry-1.docker.io",
        OciProfile::DockerHubAnonymous,
        OciReference::Tag("stable".into()),
        GgufLayer::Unique,
    );
    let mut row = Row::manifest(vec![]);
    row.status = 401;
    row.forbid_body = true;
    row.headers.insert(
        "www-authenticate".into(),
        "Bearer realm=https://unrequested.invalid/SECRET".into(),
    );
    let script = Script::new(vec![row], &admission);
    let error = source
        .resolve(&script, &limits, &admission, &AtomicBool::new(false))
        .unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)));
    assert!(!error.to_string().contains("SECRET"));
    script.finished();
    assert_no_host_reservation(&admission);
    assert_no_host_reservation(&denied);
}

#[test]
fn fixed_docker_broker_token_shapes_are_bounded_and_redacted() {
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let cancel = AtomicBool::new(false);
    let source = request(
        "https://registry-1.docker.io",
        OciProfile::DockerHubAnonymous,
        OciReference::Tag("stable".into()),
        GgufLayer::Unique,
    );
    let path = "/token?service=registry.docker.io&scope=repository%3Aai%2Ftoy%3Apull";
    for (body, valid) in [
        (json!({"token":"fixture-bearer","expires_in":60}), true),
        (json!({"access_token":"fixture-bearer"}), true),
        (json!({"token":"SECRET","access_token":"DIFFERENT"}), false),
        (json!({"token":"SECRET","expires_in":0}), false),
        (json!({"token":"SECRET\n"}), false),
        (json!({"token":"x".repeat(8193)}), false),
        (json!({}), false),
    ] {
        let script = Script::new(
            vec![Row::new(
                path,
                serde_json::to_vec(&body).unwrap(),
                "application/json",
            )],
            &admission,
        );
        let result = docker_hub_client(
            &source,
            &script,
            HttpOptions::default(),
            &limits,
            &admission,
            &cancel,
        );
        assert_eq!(result.is_ok(), valid);
        if let Err(error) = result {
            assert!(!error.to_string().contains("SECRET"));
        }
        script.finished();
    }
    assert_no_host_reservation(&admission);
}

#[test]
fn full_raw_digest_and_progress_cancel_preserve_the_existing_reference() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(128 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let data = vec![b'x'; 512];
    let raw = manifest(&data);
    let limits = PullLimits::default();
    let cancel = Arc::new(AtomicBool::new(false));
    let script = Script::new(
        vec![
            Row::manifest(raw.clone()),
            Row::new(
                format!("/v2/ai/toy/blobs/sha256:{}", digest(&data)),
                vec![b'y'; 512],
                "application/octet-stream",
            ),
        ],
        &admission,
    );
    let plan = tag(ORIGIN)
        .resolve(&script, &limits, &admission, &cancel)
        .unwrap();
    assert!(execute(&store, &plan, &script, &admission, &cancel, None).is_err());
    script.finished();
    let script = Script::new(vec![Row::manifest(raw), Row::blob(&data)], &admission);
    let plan = tag(ORIGIN)
        .resolve(&script, &limits, &admission, &cancel)
        .unwrap();
    assert!(
        execute_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &cancel,
            None,
            &mut |event| {
                if event.downloaded_bytes() > 0 {
                    cancel.store(true, Ordering::Release);
                }
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

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_http_owner_preserves_oci_type_digest_and_duplicate_refusal() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::HttpsClient;
    fn serve(replies: Vec<Vec<u8>>) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            for reply in replies {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline);
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(error) => panic!("{error}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                let mut byte = [0];
                while !request.ends_with(b"\r\n\r\n") {
                    assert!(request.len() < 16384);
                    assert_eq!(stream.read(&mut byte).unwrap(), 1);
                    request.push(byte[0]);
                }
                stream.write_all(&reply).unwrap();
            }
        });
        (origin, worker)
    }
    fn reply(headers: &str, body: &[u8]) -> Vec<u8> {
        let mut bytes = format!(
            "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\n{headers}\r\n",
            body.len()
        )
        .into_bytes();
        bytes.extend(body);
        bytes
    }
    let raw = manifest(DATA);
    let headers = format!(
        "Content-Type: {TYPE}\r\nDocker-Content-Digest: sha256:{}\r\n",
        digest(&raw)
    );
    let (origin, worker) = serve(vec![
        reply(&headers, &raw),
        reply("Content-Type: application/octet-stream\r\n", DATA),
    ]);
    let client = HttpsClient::diagnostic_loopback(
        &origin,
        None,
        HttpOptions {
            max_retries: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let admission = admission(128 * 1024 * 1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let limits = PullLimits::default();
    let source = tag(&origin);
    let plan = source
        .resolve(&client, &limits, &admission, &cancel)
        .unwrap();
    let fixture = Fixture::new();
    let store = fixture.open();
    let receipt = execute(&store, &plan, &client, &admission, &cancel, None).unwrap();
    worker.join().unwrap();
    let id = snapshot(receipt.outcome());
    assert_eq!(
        file_bytes(
            source
                .open_cached(&store, &id, &limits, &cancel)
                .unwrap()
                .file("model.gguf")
                .unwrap()
        ),
        DATA
    );
    for duplicate in [
        format!("{headers}Content-Type: {TYPE}\r\n"),
        format!(
            "{headers}Docker-Content-Digest: sha256:{}\r\n",
            digest(&raw)
        ),
    ] {
        let (origin, worker) = serve(vec![reply(&duplicate, &raw)]);
        let client = HttpsClient::diagnostic_loopback(
            &origin,
            None,
            HttpOptions {
                max_retries: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            tag(&origin)
                .resolve(&client, &limits, &admission, &cancel)
                .is_err()
        );
        worker.join().unwrap();
    }
    assert_no_host_reservation(&admission);
}
