//! Independent content-bound object fixtures using the original transfer owner.
#![cfg(all(feature = "acquisition", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;
use cache_transaction_support::*;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
use zerogravity_artifacts::acquisition::object::HttpsObjectSpec;
use zerogravity_artifacts::acquisition::{PullLimits, execute, execute_with_progress};
use zerogravity_artifacts::cache::{
    CacheStore, DigestProvenance, FileRole, Manifest, ManifestEntry, TransferAdmission,
};
// Keep unrelated fork fixtures quiescent: CLOEXEC copies can briefly retain
// another test's managed descriptors until the child execs.
static SCHEDULE: Mutex<()> = Mutex::new(());
const URL: &str = "https://objects.example.invalid/private/opaque?signature=NEVER_PERSIST";
fn spec(bytes: &[u8]) -> HttpsObjectSpec {
    HttpsObjectSpec::new(
        "https://objects.example.invalid",
        "payload.bin",
        FileRole::Auxiliary,
        bytes.len() as u64,
        digest(bytes),
    )
    .unwrap()
}
struct Script {
    response: Mutex<Option<HttpResponse>>,
    calls: AtomicUsize,
    store: CacheStore,
    admission: TransferAdmission,
}
impl Script {
    fn new(store: &CacheStore, admission: &TransferAdmission, bytes: &[u8]) -> Self {
        Self {
            response: Mutex::new(Some(HttpResponse {
                status: 200,
                headers: BTreeMap::from([("content-length".into(), bytes.len().to_string())]),
                body: Box::new(Cursor::new(bytes.to_vec())),
            })),
            calls: AtomicUsize::new(0),
            store: store.clone(),
            admission: admission.clone(),
        }
    }
}
impl HttpTransport for Script {
    fn get(
        &self,
        request: &HttpRequest,
        _: &AtomicBool,
    ) -> zerogravity_artifacts::Result<HttpResponse> {
        assert_eq!(request.url, URL);
        assert!(request.range.is_none());
        assert!(
            self.admission
                .arbiter
                .snapshot(&self.admission.resource)?
                .reserved
                > 0
        );
        assert!(self.store.usage()?.reserved_bytes > 0);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .response
            .lock()
            .unwrap()
            .take()
            .expect("unplanned retry"))
    }
}
#[test]
fn preparation_is_admitted_origin_bound_and_contains_no_signed_locator_metadata() {
    let _schedule = SCHEDULE.lock().unwrap();
    let admission = admission(128 * 1024 * 1024);
    let spec = spec(NEW);
    let limits = PullLimits::default();
    let plan = spec
        .prepare(URL, &limits, &admission, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(plan.files().len(), 1);
    assert_eq!(plan.files()[0].bytes(), Some(NEW.len() as u64));
    assert_eq!(plan.files()[0].sha256(), Some(digest(NEW).as_str()));
    for text in [
        format!("{spec:?}"),
        format!("{plan:?}"),
        format!("{:?}", spec.source()),
    ] {
        assert!(!text.contains("NEVER_PERSIST"));
        assert!(!text.contains("private/opaque"));
    }
    for url in [
        "https://other.invalid/path?secret=HIDE",
        "http://objects.example.invalid/object",
        "https://user:secret@objects.example.invalid/object",
        "https://objects.example.invalid/path#fragment",
    ] {
        assert!(
            spec.prepare(url, &limits, &admission, &AtomicBool::new(false))
                .is_err()
        );
    }
    let denied = cache_transaction_support::admission(1);
    assert!(
        spec.prepare(URL, &limits, &denied, &AtomicBool::new(false))
            .is_err()
    );
    assert_no_host_reservation(&denied);
    assert!(
        spec.prepare(URL, &limits, &admission, &AtomicBool::new(true))
            .is_err()
    );
    assert_no_host_reservation(&admission);
}
#[test]
fn exact_caller_sha_object_publishes_no_ref_and_offline_checks_the_whole_contract() {
    let _schedule = SCHEDULE.lock().unwrap();
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let spec = spec(NEW);
    let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
    let plan = spec
        .prepare(URL, &limits, &admission, &AtomicBool::new(false))
        .unwrap();
    let script = Script::new(&store, &admission, NEW);
    let receipt = execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap();
    let snapshot = receipt.manifest().digest().unwrap();
    assert_eq!(script.calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        receipt.files()[0].provenance(),
        DigestProvenance::TrustedSha256
    );
    let metadata = String::from_utf8(receipt.manifest().canonical_bytes().unwrap()).unwrap();
    assert!(!metadata.contains("NEVER_PERSIST"));
    assert!(!metadata.contains("objects.example.invalid"));
    assert!(!metadata.contains("private/opaque"));
    let lease = spec
        .open_cached(&store, &snapshot, &limits, &AtomicBool::new(false))
        .unwrap();
    assert_payload(&lease, NEW);
    assert_eq!(
        store
            .resolve_ref("keep")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        previous
    );
    for (origin, path, role, bytes, sha) in [
        (
            "https://other.invalid",
            "payload.bin",
            FileRole::Auxiliary,
            NEW.len() as u64,
            digest(NEW),
        ),
        (
            "https://objects.example.invalid",
            "other.bin",
            FileRole::Auxiliary,
            NEW.len() as u64,
            digest(NEW),
        ),
        (
            "https://objects.example.invalid",
            "payload.bin",
            FileRole::Weights,
            NEW.len() as u64,
            digest(NEW),
        ),
        (
            "https://objects.example.invalid",
            "payload.bin",
            FileRole::Auxiliary,
            NEW.len() as u64 + 1,
            digest(NEW),
        ),
        (
            "https://objects.example.invalid",
            "payload.bin",
            FileRole::Auxiliary,
            NEW.len() as u64,
            digest(b"wrong"),
        ),
    ] {
        assert!(
            HttpsObjectSpec::new(origin, path, role, bytes, sha)
                .unwrap()
                .open_cached(&store, &snapshot, &limits, &AtomicBool::new(false))
                .is_err()
        );
    }
    let forged = Manifest::new(
        spec.source().clone(),
        vec![
            ManifestEntry::new(
                "payload.bin",
                digest(NEW),
                NEW.len() as u64,
                FileRole::Auxiliary,
            )
            .unwrap()
            .with_digest_provenance(DigestProvenance::FirstSeen),
        ],
        &limits.cache,
    )
    .unwrap();
    let forged_id = forged.digest().unwrap();
    let mut tx = store
        .begin(forged, &admission, Arc::new(AtomicBool::new(false)))
        .unwrap();
    tx.import_object("payload.bin", &mut std::io::empty())
        .unwrap();
    tx.commit(None).unwrap();
    assert!(
        spec.open_cached(&store, &forged_id, &limits, &AtomicBool::new(false))
            .is_err()
    );
    drop(lease);
    std::fs::OpenOptions::new()
        .write(true)
        .open(f.root.join("blobs/sha256").join(digest(NEW)))
        .unwrap()
        .set_len(1)
        .unwrap();
    assert!(
        spec.open_cached(&store, &snapshot, &limits, &AtomicBool::new(false))
            .is_err()
    );
    assert_no_host_reservation(&admission);
}
#[test]
fn digest_length_cancellation_and_lfs_refusals_preserve_prior_ref() {
    let _schedule = SCHEDULE.lock().unwrap();
    for case in 0..4 {
        let f = Fixture::new();
        let store = f.open();
        let admission = admission(128 * 1024 * 1024);
        let limits = PullLimits::default();
        let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let lfs = b"version https://git-lfs.github.com/spec/v1\noid sha256:000\nsize 3\n";
        let expected = if case == 3 { lfs.as_slice() } else { NEW };
        let spec = spec(expected);
        let plan = spec
            .prepare(URL, &limits, &admission, &AtomicBool::new(false))
            .unwrap();
        let data = match case {
            0 => vec![b'x'; NEW.len()],
            1 => NEW[..NEW.len() - 1].to_vec(),
            3 => lfs.to_vec(),
            _ => NEW.to_vec(),
        };
        let script = Script::new(&store, &admission, &data);
        let cancel = Arc::new(AtomicBool::new(false));
        let result = execute_with_progress(
            &store,
            &plan,
            &script,
            &admission,
            &cancel,
            None,
            &mut |event| {
                if case == 2 && event.file_bytes() > 0 {
                    cancel.store(true, Ordering::Release);
                }
            },
        );
        assert!(result.is_err(), "case{case}");
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
}
#[test]
fn exact_empty_object_is_a_verified_cached_contract() {
    let _schedule = SCHEDULE.lock().unwrap();
    let f = Fixture::new();
    let store = f.open();
    let admission = admission(128 * 1024 * 1024);
    let limits = PullLimits::default();
    let spec = spec(b"");
    let plan = spec
        .prepare(URL, &limits, &admission, &AtomicBool::new(false))
        .unwrap();
    let script = Script::new(&store, &admission, b"");
    let receipt = execute(
        &store,
        &plan,
        &script,
        &admission,
        &Arc::new(AtomicBool::new(false)),
        None,
    )
    .unwrap();
    let lease = spec
        .open_cached(
            &store,
            &receipt.manifest().digest().unwrap(),
            &limits,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_payload(&lease, b"");
    assert_no_host_reservation(&admission);
}

#[cfg(feature = "diagnostic-acquisition")]
#[test]
fn real_verified_tls_object_uses_the_existing_owner_without_persisting_signed_url() {
    use std::io::BufRead;
    use std::process::{Command, Stdio};
    use std::time::Duration;
    use zerogravity_artifacts::acquisition::http::{HttpOptions, HttpsClient};
    let _schedule = SCHEDULE.lock().unwrap();
    struct Server(std::process::Child);
    impl Drop for Server {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let certificates =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/acquisition");
    for trusted in [true, false] {
        // Reuse the exact existing HTTP-owner certificate fixture and Python
        // stdlib TLS pattern. No new TLS implementation or production option.
        let script = r#"
import signal,socket,ssl,sys
signal.alarm(8)
context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(sys.argv[1],sys.argv[2])
with socket.socket() as listener:
    listener.settimeout(5);listener.bind(('127.0.0.1',0));listener.listen(1)
    print(listener.getsockname()[1],flush=True)
    raw,_=listener.accept();raw.settimeout(5)
    try:
        with context.wrap_socket(raw,server_side=True) as stream:
            request=b''
            while not request.endswith(b'\r\n\r\n'):
                part=stream.recv(1)
                if not part:break
                request+=part
                if len(request)>16384:raise ValueError('header bound')
            assert request.startswith(b'GET /private?signature=NEVER_PERSIST ')
            stream.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok')
    except ssl.SSLError:
        raw.close()
"#;
        let mut server = Server(
            Command::new("python3")
                .arg("-u")
                .arg("-c")
                .arg(script)
                .arg(certificates.join("server.pem"))
                .arg(certificates.join("server.key"))
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let stdout = server.0.stdout.take().unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let ready = std::thread::spawn(move || {
            let mut port = String::new();
            let result = std::io::BufReader::new(stdout)
                .read_line(&mut port)
                .map(|_| port);
            let _ = sender.send(result);
        });
        let port = match receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(port)) => port,
            _ => {
                drop(server);
                ready.join().unwrap();
                panic!("owned TLS fixture readiness deadline");
            }
        };
        ready.join().unwrap();
        let port = port.trim().parse::<u16>().unwrap();
        let origin = format!("https://127.0.0.1:{port}");
        let url = format!("{origin}/private?signature=NEVER_PERSIST");
        let options = HttpOptions {
            max_retries: 0,
            total_timeout: Duration::from_secs(3),
            ..Default::default()
        };
        let client = if trusted {
            HttpsClient::diagnostic_tls(
                &origin,
                include_bytes!("fixtures/acquisition/ca.pem"),
                options,
            )
            .unwrap()
        } else {
            HttpsClient::new(&origin, options).unwrap()
        };
        let f = Fixture::new();
        let store = f.open();
        let admission = admission(128 * 1024 * 1024);
        let limits = PullLimits::default();
        let previous = durable(publish(&store, &admission, OLD, Some("keep")).unwrap());
        let spec = HttpsObjectSpec::new(
            &origin,
            "payload.bin",
            FileRole::Auxiliary,
            2,
            digest(b"ok"),
        )
        .unwrap();
        let plan = spec
            .prepare(&url, &limits, &admission, &AtomicBool::new(false))
            .unwrap();
        let result = execute(
            &store,
            &plan,
            &client,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
        );
        assert_eq!(result.is_ok(), trusted, "{result:?}");
        if let Ok(receipt) = result {
            let metadata =
                String::from_utf8(receipt.manifest().canonical_bytes().unwrap()).unwrap();
            assert!(!metadata.contains("NEVER_PERSIST"));
            assert!(!metadata.contains("127.0.0.1"));
            assert_payload(
                &spec
                    .open_cached(
                        &store,
                        &receipt.manifest().digest().unwrap(),
                        &limits,
                        &AtomicBool::new(false),
                    )
                    .unwrap(),
                b"ok",
            );
        }
        assert_eq!(
            store
                .resolve_ref("keep")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            previous
        );
        assert_no_host_reservation(&admission);
        drop(server);
    }
}
