//! Explicit strong conditional observations, without caller raw SHA authority.
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
use zerogravity_artifacts::acquisition::azure::{AzureConditionalPin, AzureConditionalRequest};
use zerogravity_artifacts::acquisition::http::{
    HttpRequest, HttpResponse, HttpTarget, HttpTransport,
};
use zerogravity_artifacts::acquisition::s3::{
    S3Addressing, S3ConditionalPin, S3ConditionalRequest, S3Endpoint,
};
use zerogravity_artifacts::acquisition::{
    PullLimits, ResolvedPullPlan, execute, execute_with_progress,
};
use zerogravity_artifacts::cache::{
    CacheLimits, CacheStore, DigestProvenance, FileRole, Manifest, ManifestEntry, ManifestSource,
    SnapshotLease, TransferAdmission,
};
const TAG: &str = "\"observed-multipart-2\"";
const S3: &str = "https://object.example";
const AZURE: &str = "https://fixtureaccount.blob.core.windows.net";
#[derive(Clone, Copy, Debug)]
enum Provider {
    S3,
    Azure,
}
enum Request {
    S3(S3ConditionalRequest),
    Azure(AzureConditionalRequest),
}
enum Pin {
    S3(S3ConditionalPin),
    Azure(AzureConditionalPin),
}
impl Provider {
    fn request(self) -> Request {
        match self {
            Self::S3 => Request::S3(
                S3ConditionalRequest::new(
                    &S3Endpoint::compatible(S3, "us-east-1", S3Addressing::PathStyle).unwrap(),
                    "bucket",
                    "folder/data.bin",
                    PAYLOAD_PATH,
                    FileRole::Auxiliary,
                )
                .unwrap(),
            ),
            Self::Azure => Request::Azure(
                AzureConditionalRequest::new(
                    AZURE,
                    "container",
                    "folder/data.bin",
                    PAYLOAD_PATH,
                    FileRole::Auxiliary,
                )
                .unwrap(),
            ),
        }
    }
    fn row(self, bytes: &[u8]) -> Row {
        let mut headers = BTreeMap::from([
            ("content-length".into(), bytes.len().to_string()),
            ("etag".into(), TAG.into()),
        ]);
        if matches!(self, Self::Azure) {
            headers.insert("x-ms-blob-type".into(), "BlockBlob".into());
        }
        Row {
            head: false,
            status: 200,
            headers,
            data: bytes.to_vec(),
            range: None,
            cut: None,
            never: false,
        }
    }
    fn head(self, bytes: u64) -> Row {
        let mut row = self.row(b"");
        row.headers
            .insert("content-length".into(), bytes.to_string());
        row.head = true;
        row.never = true;
        row
    }
    fn range(self, data: &[u8], start: usize) -> Row {
        let mut row = self.row(&data[start..]);
        row.status = 206;
        row.range = Some(start as u64);
        row.headers.insert(
            "content-range".into(),
            format!("bytes {start}-{}/{}", data.len() - 1, data.len()),
        );
        row
    }
}
impl Request {
    fn pin(&self, bytes: u64, tag: &str) -> Result<Pin> {
        match self {
            Self::S3(r) => S3ConditionalPin::new(r, bytes, tag).map(Pin::S3),
            Self::Azure(r) => AzureConditionalPin::new(r, bytes, tag).map(Pin::Azure),
        }
    }
    fn resolve(
        &self,
        script: &impl HttpTransport,
        limits: &PullLimits,
        admission: &TransferAdmission,
        cancel: &AtomicBool,
    ) -> Result<(Pin, ResolvedPullPlan)> {
        match self {
            Self::S3(r) => r
                .resolve_with_pin(script, limits, admission, cancel)
                .map(|(pin, plan)| (Pin::S3(pin), plan)),
            Self::Azure(r) => r
                .resolve_with_pin(script, limits, admission, cancel)
                .map(|(pin, plan)| (Pin::Azure(pin), plan)),
        }
    }
}
impl Pin {
    fn source(&self) -> &ManifestSource {
        match self {
            Self::S3(p) => p.source(),
            Self::Azure(p) => p.source(),
        }
    }
    fn open(
        &self,
        store: &CacheStore,
        id: &str,
        limits: &PullLimits,
        cancel: &AtomicBool,
    ) -> Result<SnapshotLease> {
        match self {
            Self::S3(p) => p.open_cached(store, id, limits, cancel),
            Self::Azure(p) => p.open_cached(store, id, limits, cancel),
        }
    }
}
struct Row {
    head: bool,
    status: u16,
    headers: BTreeMap<String, String>,
    data: Vec<u8>,
    range: Option<u64>,
    cut: Option<usize>,
    never: bool,
}
struct Never;
impl Read for Never {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("refused or HEAD body consumed")
    }
}
struct Body {
    data: Cursor<Vec<u8>>,
    cut: Option<usize>,
    bytes: Arc<AtomicUsize>,
}
impl Read for Body {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.cut == Some(0) {
            return Err(io::Error::other("synthetic interruption"));
        }
        let n = out.len().min(self.cut.unwrap_or(out.len()));
        let n = self.data.read(&mut out[..n])?;
        if let Some(left) = &mut self.cut {
            *left -= n
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
    fn response(
        &self,
        request: &HttpRequest,
        target: &HttpTarget,
        head: bool,
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
        assert_eq!(target.if_match(), if head { None } else { Some(TAG) });
        let row = self
            .rows
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected call/fallback");
        assert_eq!(row.head, head);
        assert_eq!(request.range.as_ref().map(|r| r.start), row.range);
        if let Some(range) = &request.range {
            assert_eq!(range.validator, TAG)
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(HttpResponse {
            status: row.status,
            headers: row.headers,
            body: if row.never {
                Box::new(Never)
            } else {
                Box::new(Body {
                    data: Cursor::new(row.data),
                    cut: row.cut,
                    bytes: self.bytes.clone(),
                })
            },
        })
    }
    fn finished(&self) {
        assert!(self.rows.lock().unwrap().is_empty())
    }
}
impl HttpTransport for Script {
    fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("conditional target downgraded")
    }
    fn head(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
        panic!("conditional target downgraded")
    }
    fn head_target(&self, r: &HttpRequest, t: &HttpTarget, _: &AtomicBool) -> Result<HttpResponse> {
        self.response(r, t, true)
    }
    fn get_target(&self, r: &HttpRequest, t: &HttpTarget, _: &AtomicBool) -> Result<HttpResponse> {
        self.response(r, t, false)
    }
}
fn resolve(
    provider: Provider,
    data: &[u8],
    admission: &TransferAdmission,
) -> (Pin, ResolvedPullPlan) {
    let script = Script::new(provider, vec![provider.head(data.len() as u64)], admission);
    let result = provider
        .request()
        .resolve(
            &script,
            &PullLimits::default(),
            admission,
            &AtomicBool::new(false),
        )
        .unwrap();
    script.finished();
    assert_eq!(script.bytes.load(Ordering::Relaxed), 0);
    result
}
fn seed(
    store: &CacheStore,
    admission: &TransferAdmission,
    source: &ManifestSource,
    body: &[u8],
    selected: (&str, FileRole, DigestProvenance),
) -> String {
    let manifest = Manifest::new(
        source.clone(),
        vec![
            ManifestEntry::new(selected.0, digest(body), body.len() as u64, selected.1)
                .unwrap()
                .with_digest_provenance(selected.2),
        ],
        &CacheLimits::default(),
    )
    .unwrap();
    let id = manifest.digest().unwrap();
    let mut tx = store
        .begin(manifest, admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    tx.import_object(selected.0, &mut Cursor::new(body))
        .unwrap();
    tx.commit(Some("old")).unwrap();
    id
}
#[test]
fn pure_pins_bind_observed_etag_size_locator_role_without_invented_sha_or_version() {
    for provider in [Provider::S3, Provider::Azure] {
        let request = provider.request();
        let pin = request.pin(3, TAG).unwrap();
        assert_ne!(pin.source(), request.pin(4, TAG).unwrap().source());
        assert_ne!(pin.source(), request.pin(3, "\"other\"").unwrap().source());
        for tag in [
            "",
            "*",
            "W/\"weak\"",
            "unquoted",
            "\"a\",\"b\"",
            "\"bad\n\"",
        ] {
            assert!(request.pin(3, tag).is_err())
        }
        assert!(
            request
                .pin(3, &format!("\"{}\"", "x".repeat(1023)))
                .is_err()
        );
        assert!(request.pin(0, TAG).is_ok());
        assert!(pin.source().provider().ends_with("-conditional"));
        assert!(!pin.source().revision().contains(TAG));
    }
}
#[test]
fn admission_cancel_and_metadata_bounds_precede_head_or_plan_and_head_never_reads_body() {
    for provider in [Provider::S3, Provider::Azure] {
        let denied = admission(1);
        let script = Script::new(provider, vec![], &denied);
        assert!(
            provider
                .request()
                .resolve(
                    &script,
                    &PullLimits::default(),
                    &denied,
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        script.finished();
        assert_no_host_reservation(&denied);
        let allowed = admission(128 * 1024 * 1024);
        assert!(
            provider
                .request()
                .resolve(
                    &script,
                    &PullLimits::default(),
                    &allowed,
                    &AtomicBool::new(true)
                )
                .is_err()
        );
        let (_, plan) = resolve(provider, b"abc", &allowed);
        assert_eq!(plan.total_bytes(), Some(3));
        assert_eq!(plan.files()[0].sha256(), None);
        assert_eq!(
            plan.files()[0].digest_provenance(),
            DigestProvenance::FirstSeen
        );
        for limits in [
            PullLimits {
                max_metadata_bytes: 16,
                ..Default::default()
            },
            PullLimits {
                cache: CacheLimits {
                    max_manifest_bytes: 128,
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            let script = Script::new(provider, vec![provider.head(3)], &allowed);
            assert!(
                provider
                    .request()
                    .resolve(&script, &limits, &allowed, &AtomicBool::new(false))
                    .is_err()
            );
            script.finished();
        }
        assert_no_host_reservation(&allowed);
    }
}
#[test]
fn incomplete_weak_missing_encoded_or_deleted_head_refuses_without_payload_fallback() {
    for provider in [Provider::S3, Provider::Azure] {
        let admission = admission(128 * 1024 * 1024);
        for mode in 0..9 {
            let mut row = provider.head(3);
            match mode {
                0 => row.status = 405,
                1 => row.status = 501,
                2 => {
                    row.headers.remove("content-length");
                }
                3 => {
                    row.headers.remove("etag");
                }
                4 => {
                    row.headers.insert("etag".into(), "W/\"weak\"".into());
                }
                5 => {
                    row.headers.insert("content-encoding".into(), "gzip".into());
                }
                6 => {
                    row.headers
                        .insert("content-range".into(), "bytes 0-2/3".into());
                }
                7 => {
                    row.headers
                        .insert("transfer-encoding".into(), "chunked".into());
                }
                _ => {
                    row.headers.insert(
                        if matches!(provider, Provider::S3) {
                            "x-amz-delete-marker"
                        } else {
                            "x-ms-blob-type"
                        }
                        .into(),
                        if matches!(provider, Provider::S3) {
                            "true"
                        } else {
                            "AppendBlob"
                        }
                        .into(),
                    );
                }
            }
            let script = Script::new(provider, vec![row], &admission);
            assert!(
                provider
                    .request()
                    .resolve(
                        &script,
                        &PullLimits::default(),
                        &admission,
                        &AtomicBool::new(false)
                    )
                    .is_err()
            );
            script.finished();
            assert_eq!(script.bytes.load(Ordering::Relaxed), 0);
            assert_no_host_reservation(&admission);
        }
    }
}
#[test]
fn matching_reads_are_first_seen_while_changed_missing_weak_etag_or_412_never_reads_or_moves_ref() {
    for provider in [Provider::S3, Provider::Azure] {
        for mode in 0..5 {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
            let (pin, plan) = resolve(provider, NEW, &admission);
            let mut row = provider.row(NEW);
            match mode {
                0 => {}
                1 => row.status = 412,
                2 => {
                    row.headers.remove("etag");
                }
                3 => {
                    row.headers.insert("etag".into(), "\"changed\"".into());
                }
                _ => {
                    row.headers.insert("etag".into(), "W/\"weak\"".into());
                }
            }
            row.never = mode != 0;
            let script = Script::new(provider, vec![row], &admission);
            let result = execute(
                &store,
                &plan,
                &script,
                &admission,
                &Arc::new(AtomicBool::new(false)),
                Some("keep"),
            );
            script.finished();
            if mode == 0 {
                let receipt = result.unwrap();
                assert_eq!(receipt.files()[0].provenance(), DigestProvenance::FirstSeen);
                assert_eq!(receipt.files()[0].digest(), digest(NEW));
                let id = receipt.manifest().digest().unwrap();
                assert_payload(
                    &pin.open(&store, &id, &PullLimits::default(), &AtomicBool::new(false))
                        .unwrap(),
                    NEW,
                );
            } else {
                assert!(result.is_err());
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
            assert_no_host_reservation(&admission);
        }
    }
}
#[test]
fn first_seen_offline_is_exact_and_digest_only_before_any_transport_or_admission() {
    for provider in [Provider::S3, Provider::Azure] {
        for mode in 0..6 {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let pin = provider.request().pin(NEW.len() as u64, TAG).unwrap();
            let source = if mode == 1 {
                ManifestSource::new("local", "wrong", "source").unwrap()
            } else {
                pin.source().clone()
            };
            let body = if mode == 4 {
                b"wrong size".as_slice()
            } else {
                NEW
            };
            let id = seed(
                &store,
                &admission,
                &source,
                body,
                (
                    if mode == 2 { "other.bin" } else { PAYLOAD_PATH },
                    if mode == 3 {
                        FileRole::Config
                    } else {
                        FileRole::Auxiliary
                    },
                    if mode == 5 {
                        DigestProvenance::TrustedSha256
                    } else {
                        DigestProvenance::FirstSeen
                    },
                ),
            );
            let opened = pin.open(&store, &id, &PullLimits::default(), &AtomicBool::new(false));
            if mode == 0 {
                assert_payload(&opened.unwrap(), NEW)
            } else {
                assert!(opened.is_err())
            }
            assert!(
                pin.open(
                    &store,
                    "old",
                    &PullLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            assert!(
                provider
                    .request()
                    .pin(NEW.len() as u64, "\"different\"")
                    .unwrap()
                    .open(&store, &id, &PullLimits::default(), &AtomicBool::new(false))
                    .is_err()
            );
            assert_no_host_reservation(&admission);
        }
    }
}
#[test]
fn durable_prefix_and_live_retry_keep_same_if_match_and_cannot_refresh_on_failed_condition() {
    for provider in [Provider::S3, Provider::Azure] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let data = vec![b'r'; 1024];
        let (_, plan) = resolve(provider, &data, &admission);
        let cancel = Arc::new(AtomicBool::new(false));
        let script = Script::new(provider, vec![provider.row(&data)], &admission);
        assert!(
            execute_with_progress(
                &store,
                &plan,
                &script,
                &admission,
                &cancel,
                Some("keep"),
                &mut |p| {
                    if p.file_bytes() == 128 {
                        cancel.store(true, Ordering::Release)
                    }
                }
            )
            .is_err()
        );
        script.finished();
        assert_eq!(script.bytes.load(Ordering::Relaxed), 128);
        let mut row = provider.range(&data, 128);
        row.status = 412;
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
        assert_eq!(
            store
                .resolve_ref("keep")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            old
        );
        let script = Script::new(provider, vec![provider.range(&data, 128)], &admission);
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
        let fixture = Fixture::new();
        let store = fixture.open();
        let mut row = provider.row(&data);
        row.cut = Some(128);
        let script = Script::new(provider, vec![row, provider.range(&data, 128)], &admission);
        execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
        )
        .unwrap();
        script.finished();
        assert_eq!(script.calls.load(Ordering::Relaxed), 2);
        assert_no_host_reservation(&admission);
    }
}
#[test]
fn exact_zero_observation_requires_one_get_and_terminal_eof() {
    for provider in [Provider::S3, Provider::Azure] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(128 * 1024 * 1024);
        let (pin, plan) = resolve(provider, b"", &admission);
        let script = Script::new(provider, vec![provider.row(b"")], &admission);
        let receipt = execute(
            &store,
            &plan,
            &script,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
        )
        .unwrap();
        script.finished();
        assert_eq!(script.calls.load(Ordering::Relaxed), 1);
        assert_eq!(receipt.files()[0].bytes(), 0);
        assert_eq!(receipt.files()[0].provenance(), DigestProvenance::FirstSeen);
        assert_payload(
            &pin.open(
                &store,
                &receipt.manifest().digest().unwrap(),
                &PullLimits::default(),
                &AtomicBool::new(false),
            )
            .unwrap(),
            b"",
        );
    }
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_azure_conditional_retry_signs_if_match_and_range_and_412_never_falls_back() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use zerogravity_artifacts::acquisition::http::{AzureSharedKey, HttpOptions, HttpsClient};
    const KEY: &str =
        "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+Pw==";
    const SAS: &str = "sv=2023-11-03&sr=b&sp=r&se=2030-01-01T00%3a00%3A00Z&spr=https&sig=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA%3d";
    for mode in ["public", "bearer", "key", "sas", "key-412"] {
        let mode = format!("conditional-{mode}");
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
        let mut out = BufReader::new(child.stdout.take().unwrap());
        let mut port = String::new();
        out.read_line(&mut port).unwrap();
        let origin = format!("http://127.0.0.1:{}", port.trim());
        let options = HttpOptions {
            max_retries: 0,
            ..Default::default()
        };
        let client = match mode.as_str() {
            "conditional-key" | "conditional-key-412" => HttpsClient::diagnostic_azure_shared_key(
                &origin,
                AzureSharedKey::new("fixtureaccount", KEY).unwrap(),
                options,
            ),
            "conditional-sas" => HttpsClient::diagnostic_azure_sas(&origin, SAS, options),
            "conditional-bearer" => HttpsClient::diagnostic_loopback(
                &origin,
                Some("synthetic-azure-bearer".into()),
                options,
            ),
            _ => HttpsClient::diagnostic_loopback(&origin, None, options),
        }
        .unwrap();
        let request = AzureConditionalRequest::new(
            &origin,
            "container",
            "dir/a b.bin",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
        )
        .unwrap();
        let admission = admission(128 * 1024 * 1024);
        let cancel = Arc::new(AtomicBool::new(false));
        let (pin, plan) = request
            .resolve_with_pin(&client, &PullLimits::default(), &admission, &cancel)
            .unwrap();
        assert_eq!(pin.etag(), "\"wire-etag\"");
        let fixture = Fixture::new();
        let store = fixture.open();
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let result = execute(&store, &plan, &client, &admission, &cancel, Some("keep"));
        if mode.ends_with("-412") {
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
            let receipt = result.unwrap();
            assert_eq!(receipt.files()[0].provenance(), DigestProvenance::FirstSeen);
            assert_payload(
                &pin.open_cached(
                    &store,
                    &receipt.manifest().digest().unwrap(),
                    &PullLimits::default(),
                    &cancel,
                )
                .unwrap(),
                b"actual isolated Azure payload",
            );
        }
        let mut verified = String::new();
        out.read_line(&mut verified).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(verified.trim(), format!("VERIFIED {mode}"));
        assert_no_host_reservation(&admission);
    }
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn actual_s3_conditional_head_initial_and_range_retry_have_independently_valid_signatures() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient, S3Credentials};
    const VERIFY: &str = r#"import hashlib,hmac,sys
raw=sys.stdin.buffer.read().decode('ascii').split('\r\n')
method,target,_=raw[0].split(' ')
headers={}
for line in raw[1:]:
    if not line:break
    name,value=line.split(':',1)
    assert name.lower()not in headers
    headers[name.lower()]=value.strip()
assert target=='/bucket/dir/../raw//a%20b%2B%25'
base=['host','x-amz-content-sha256','x-amz-date','x-amz-security-token']
if method=='GET':
    base+=['if-match']
    assert headers['if-match']=='"wire-etag"'
if 'range'in headers:
    base+=['range','if-range']
    assert headers['range']=='bytes=8-'and headers['if-range']=='"wire-etag"'
parts=dict(part.strip().split('=',1)for part in headers['authorization'].split(' ',1)[1].split(','))
assert parts['SignedHeaders']==';'.join(sorted(base))
access,scope=parts['Credential'].split('/',1)
assert access=='SYNTHETICACCESS'
date,region,service,terminal=scope.split('/')
assert region=='us-east-1'and service=='s3'and terminal=='aws4_request'
assert headers['x-amz-security-token']=='synthetic-session'
assert headers['x-amz-content-sha256']==hashlib.sha256(b'').hexdigest()
canonical_headers=''.join(name+':'+ ' '.join(headers[name].split())+'\n'for name in sorted(base))
canonical='\n'.join([method,target,'',canonical_headers,parts['SignedHeaders'],headers['x-amz-content-sha256']])
to_sign='\n'.join(['AWS4-HMAC-SHA256',headers['x-amz-date'],scope,hashlib.sha256(canonical.encode()).hexdigest()])
key=b'AWS4synthetic-secret'
for value in [date,region,service,terminal]:key=hmac.new(key,value.encode(),hashlib.sha256).digest()
assert hmac.compare_digest(hmac.new(key,to_sign.encode(),hashlib.sha256).hexdigest(),parts['Signature'])
print('VERIFIED')
"#;
    for (signed, refuse) in [(false, false), (true, false), (true, true)] {
        let data = b"actual isolated S3 conditional bytes";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            for index in 0..if refuse { 2 } else { 3 } {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline);
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(e) => panic!("{e}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut raw = Vec::new();
                let mut buf = [0; 2048];
                while !raw.windows(4).any(|s| s == b"\r\n\r\n") {
                    let n = stream.read(&mut buf).unwrap();
                    assert_ne!(n, 0);
                    raw.extend_from_slice(&buf[..n]);
                    assert!(raw.len() < 16384)
                }
                let text = std::str::from_utf8(&raw).unwrap();
                assert!(text.starts_with(if index == 0 {
                    "HEAD /bucket/dir/../raw//a%20b%2B%25 HTTP/1.1\r\n"
                } else {
                    "GET /bucket/dir/../raw//a%20b%2B%25 HTTP/1.1\r\n"
                }));
                let lower = text.to_ascii_lowercase();
                assert_eq!(
                    lower.matches("\r\nif-match:").count(),
                    if index == 0 { 0 } else { 1 }
                );
                if index == 0 {
                    assert!(!lower.contains("if-match:"))
                } else {
                    assert!(lower.contains("if-match: \"wire-etag\"\r\n"))
                }
                if index == 2 {
                    assert!(lower.contains("range: bytes=8-\r\n"));
                    assert!(lower.contains("if-range: \"wire-etag\"\r\n"))
                }
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
                        "{}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                    assert_eq!(output.stdout, b"VERIFIED\n")
                } else {
                    assert!(!lower.contains("authorization:"))
                }
                if refuse && index == 1 {
                    stream.write_all(b"HTTP/1.1 412 Precondition Failed\r\nConnection: close\r\nContent-Length: 0\r\n\r\n").unwrap();
                    continue;
                }
                let payload = if index == 2 {
                    &data[8..]
                } else {
                    data.as_slice()
                };
                let extra = if index == 2 {
                    format!(
                        "Content-Range: bytes 8-{}/{}\r\n",
                        data.len() - 1,
                        data.len()
                    )
                } else {
                    String::new()
                };
                let response = format!(
                    "HTTP/1.1 {}\r\nConnection: close\r\nContent-Length: {}\r\nETag: \"wire-etag\"\r\n{}\r\n",
                    if index == 2 {
                        "206 Partial Content"
                    } else {
                        "200 OK"
                    },
                    payload.len(),
                    extra
                );
                stream.write_all(response.as_bytes()).unwrap();
                if index > 0 {
                    stream
                        .write_all(if index == 1 { &payload[..8] } else { payload })
                        .unwrap()
                }
            }
        });
        let endpoint =
            S3Endpoint::compatible(&origin, "us-east-1", S3Addressing::PathStyle).unwrap();
        let request = S3ConditionalRequest::new(
            &endpoint,
            "bucket",
            "dir/../raw//a b+%",
            PAYLOAD_PATH,
            FileRole::Auxiliary,
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
        let (pin, plan) = request
            .resolve_with_pin(&client, &PullLimits::default(), &admission, &cancel)
            .unwrap();
        let fixture = Fixture::new();
        let store = fixture.open();
        let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let result = execute(&store, &plan, &client, &admission, &cancel, Some("keep"));
        worker.join().unwrap();
        if refuse {
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
            let receipt = result.unwrap();
            assert_payload(
                &pin.open_cached(
                    &store,
                    &receipt.manifest().digest().unwrap(),
                    &PullLimits::default(),
                    &cancel,
                )
                .unwrap(),
                data,
            );
            assert_eq!(receipt.files()[0].provenance(), DigestProvenance::FirstSeen);
        }
        assert_no_host_reservation(&admission);
    }
}

#[test]
fn short_oversized_or_pointer_observed_body_preserves_old_ref_and_does_not_become_complete() {
    let expected = vec![b'a'; 256];
    let mut pointer =
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 256\n".to_vec();
    pointer.resize(256, b' ');
    for provider in [Provider::S3, Provider::Azure] {
        for body in [
            expected[..255].to_vec(),
            [expected.as_slice(), b"extra"].concat(),
            pointer.clone(),
        ] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let admission = admission(128 * 1024 * 1024);
            let old = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
            let (_, plan) = resolve(provider, &expected, &admission);
            let mut first = provider.row(&body);
            first
                .headers
                .insert("content-length".into(), expected.len().to_string());
            let mut rows = vec![first];
            if body.len() < expected.len() {
                for _ in 0..2 {
                    let mut next = provider.range(&expected, body.len());
                    next.data.clear();
                    rows.push(next);
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
