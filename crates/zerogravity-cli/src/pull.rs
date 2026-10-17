//! Explicit acquisition frontend. Ordinary inspect/run never call this module.
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

#[path = "pull_components.rs"]
mod components;
#[path = "pull_signals.rs"]
mod signals;
use zerogravity::acquisition::http::{HttpOptions, HttpsClient};
use zerogravity::acquisition::{self, Provider, PullLimits, PullRequest, Selection};
use zerogravity::cache::{
    CacheOptions, CacheStore, CommitOutcome, SnapshotLease, TransferAdmission,
};
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
use zerogravity::{Error, Result};

const DISK_DEFAULT: u64 = 16 * 1024 * 1024 * 1024;
const MEMORY_DEFAULT: u64 = 128 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Help,
    List,
    Inspect(String),
    Verify(String),
    Fetch(PullRequest),
    Components(String),
}
#[derive(Debug)]
struct Options {
    action: Action,
    root: Option<PathBuf>,
    disk: u64,
    memory: u64,
    max_file: Option<u64>,
    max_snapshot: Option<u64>,
    max_files: Option<usize>,
    metadata: Option<usize>,
    pages: Option<usize>,
    timeout: Option<u64>,
    reference: Option<String>,
    dry_run: bool,
    offline: bool,
}
fn invalid() -> Error {
    // Do not echo unknown argv: a mistaken token option must stay out of logs.
    Error::Malformed("invalid pull arguments; use pull --help (tokens are environment-only)".into())
}
fn parse(args: &[String]) -> Result<Options> {
    if args.len() > 8192 || args.iter().any(|s| s.len() > 4096) {
        return Err(Error::Limit("pull arguments exceed bounds".into()));
    }
    let first = args.first().ok_or_else(invalid)?;
    if matches!(first.as_str(), "help" | "--help" | "-h") {
        return if args.len() == 1 {
            Ok(Options {
                action: Action::Help,
                root: None,
                disk: DISK_DEFAULT,
                memory: MEMORY_DEFAULT,
                max_file: None,
                max_snapshot: None,
                max_files: None,
                metadata: None,
                pages: None,
                timeout: None,
                reference: None,
                dry_run: false,
                offline: false,
            })
        } else {
            Err(invalid())
        };
    }
    let mut rest = args[1..].iter();
    let provider = match first.as_str() {
        "hf" => Some(Provider::HuggingFace),
        "github" => Some(Provider::GitHub),
        "list" | "inspect" | "verify" | "components" => None,
        _ => return Err(invalid()),
    };
    let positional = if first == "list" {
        None
    } else {
        Some(rest.next().ok_or_else(invalid)?.clone())
    };
    let mut result = Options {
        action: Action::List,
        root: None,
        disk: DISK_DEFAULT,
        memory: MEMORY_DEFAULT,
        max_file: None,
        max_snapshot: None,
        max_files: None,
        metadata: None,
        pages: None,
        timeout: None,
        reference: None,
        dry_run: false,
        offline: false,
    };
    let mut revision = None;
    let mut release = None;
    let mut asset = None;
    let mut subfolder = None;
    let mut files = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    while let Some(flag) = rest.next() {
        if flag != "--file" && !seen.insert(flag.as_str()) {
            return Err(invalid());
        }
        match flag.as_str() {
            "--offline" => result.offline = true,
            "--dry-run" => result.dry_run = true,
            "--cache"
            | "--ref"
            | "--revision"
            | "--release"
            | "--asset"
            | "--file"
            | "--subfolder"
            | "--disk-limit-bytes"
            | "--memory-limit-bytes"
            | "--max-file-bytes"
            | "--max-snapshot-bytes"
            | "--max-files"
            | "--max-metadata-bytes"
            | "--max-pages"
            | "--timeout-seconds" => {
                let value = rest.next().ok_or_else(invalid)?;
                if value.is_empty() {
                    return Err(invalid());
                }
                let number = || {
                    value
                        .parse::<u64>()
                        .ok()
                        .filter(|n| *n > 0)
                        .ok_or_else(invalid)
                };
                let count = || usize::try_from(number()?).map_err(|_| invalid());
                match flag.as_str() {
                    "--cache" => {
                        let path = PathBuf::from(value);
                        if !path.is_absolute() {
                            return Err(invalid());
                        }
                        result.root = Some(path);
                    }
                    "--ref" => {
                        if value.chars().any(char::is_control) {
                            return Err(invalid());
                        }
                        result.reference = Some(value.clone());
                    }
                    "--revision" => revision = Some(value.clone()),
                    "--release" => release = Some(value.clone()),
                    "--asset" => asset = Some(value.clone()),
                    "--file" => files.push(value.clone()),
                    "--subfolder" => subfolder = Some(value.clone()),
                    "--disk-limit-bytes" => result.disk = number()?,
                    "--memory-limit-bytes" => result.memory = number()?,
                    "--max-file-bytes" => result.max_file = Some(number()?),
                    "--max-snapshot-bytes" => result.max_snapshot = Some(number()?),
                    "--max-files" => result.max_files = Some(count()?),
                    "--max-metadata-bytes" => result.metadata = Some(count()?),
                    "--max-pages" => result.pages = Some(count()?),
                    "--timeout-seconds" => {
                        let seconds = number()?;
                        if seconds > 86400 {
                            return Err(invalid());
                        }
                        result.timeout = Some(seconds);
                    }
                    _ => unreachable!(),
                }
            }
            _ => return Err(invalid()),
        }
    }
    if let Some(provider) = provider {
        let (revision, selection) = match (revision, release, asset) {
            (Some(revision), None, None) if !files.is_empty() => (
                revision,
                Selection::Files {
                    paths: files,
                    subfolder,
                },
            ),
            (None, Some(tag), Some(name))
                if provider == Provider::GitHub && files.is_empty() && subfolder.is_none() =>
            {
                (tag, Selection::ReleaseAsset { name })
            }
            _ => return Err(invalid()),
        };
        let request = PullRequest {
            provider,
            repository: positional.ok_or_else(invalid)?,
            revision,
            selection,
        };
        request.validate(&limits(&result)?)?;
        result.action = Action::Fetch(request);
    } else {
        if revision.is_some()
            || release.is_some()
            || asset.is_some()
            || subfolder.is_some()
            || !files.is_empty()
            || result.reference.is_some()
            || result.dry_run && first != "components"
        {
            return Err(invalid());
        }
        result.action = match first.as_str() {
            "list" => Action::List,
            "components" => Action::Components(positional.ok_or_else(invalid)?),
            "inspect" => Action::Inspect(positional.ok_or_else(invalid)?),
            "verify" => Action::Verify(positional.ok_or_else(invalid)?),
            _ => unreachable!(),
        };
    }
    Ok(result)
}
fn limits(options: &Options) -> Result<PullLimits> {
    let mut limits = PullLimits::default();
    if let Some(n) = options.max_file {
        limits.cache.max_file_bytes = n;
    }
    if let Some(n) = options.max_snapshot {
        limits.cache.max_snapshot_bytes = n;
    }
    if let Some(n) = options.max_files {
        limits.cache.max_files = n;
    }
    if let Some(n) = options.metadata {
        limits.max_metadata_bytes = n;
    }
    if let Some(n) = options.pages {
        limits.max_pages = n;
    }
    limits.validate()?;
    Ok(limits)
}
fn open(options: &Options) -> Result<CacheStore> {
    let mut cache = CacheOptions::new(options.root.clone(), options.disk);
    cache.limits = limits(options)?.cache;
    CacheStore::open(cache)
}
fn cached(store: &CacheStore, name: &str) -> Result<SnapshotLease> {
    if name.len() == 64
        && name
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        store.open_snapshot(name)
    } else {
        store.resolve_ref(name)
    }
}
fn describe(lease: &SnapshotLease) -> Result<()> {
    use std::io::Write;
    std::io::stdout().write_all(&lease.manifest().canonical_bytes()?)?;
    eprintln!("Verified cached data; no model execution support or remote freshness inferred.");
    Ok(())
}
fn admission(memory: u64) -> Result<TransferAdmission> {
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "artifact-transfer-staging")?;
    // This synchronous CLI owns the fixed domain for its process lifetime.
    // It is not a global host-RAM, OS or cross-process reservation.
    arbiter.register(resource.clone(), memory)?;
    Ok(TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("artifact-acquisition", "cli")?,
    })
}

#[derive(Default)]
struct ProgressDisplay {
    last_report: Option<std::time::Instant>,
    last_path: String,
}
impl ProgressDisplay {
    fn report(
        &mut self,
        event: &acquisition::PullProgress<'_>,
        output: &mut impl std::io::Write,
    ) -> std::io::Result<()> {
        let now = std::time::Instant::now();
        if self.last_path != event.path()
            || event.file_bytes() == event.file_total_bytes()
            || self.last_report.is_none_or(|last| {
                now.duration_since(last) >= std::time::Duration::from_millis(200)
            })
        {
            writeln!(
                output,
                "Pull bytes: {}/{} (file {}/{}, cached {}, resumed {})",
                event.total_bytes(),
                event.total_expected_bytes(),
                event.file_bytes(),
                event.file_total_bytes(),
                event.cached_bytes(),
                event.resumed_bytes()
            )?;
            self.last_report = Some(now);
            self.last_path.clear();
            self.last_path.push_str(event.path());
        }
        Ok(())
    }
}

pub(super) fn run(args: &[String]) -> Result<()> {
    let options = parse(args)?;
    match &options.action {
        Action::Help => {
            help();
            Ok(())
        }
        Action::List => {
            for digest in open(&options)?.list_snapshots()? {
                println!("{digest}");
            }
            Ok(())
        }
        Action::Inspect(name) | Action::Verify(name) => describe(&cached(&open(&options)?, name)?),
        Action::Components(path) => components::run(path, &options),
        Action::Fetch(request) => {
            let reference = options.reference.clone().unwrap_or_else(|| {
                format!(
                    "{}/{}/{}",
                    request.provider.label(),
                    request.repository,
                    request.revision
                )
            });
            // Offline is terminal before client, credentials, metadata, DNS or
            // transfer-admission construction. Local validation is still real.
            if options.offline {
                let lease = cached(&open(&options)?, &reference)?;
                acquisition::validate_cached_selection(&lease, request)?;
                return describe(&lease);
            }
            let (origin, token) = match request.provider {
                Provider::HuggingFace => ("https://huggingface.co", "HF_TOKEN"),
                Provider::GitHub => ("https://api.github.com", "GITHUB_TOKEN"),
            };
            let mut http_options = HttpOptions::default();
            if let Some(seconds) = options.timeout {
                http_options.total_timeout = std::time::Duration::from_secs(seconds);
            }
            let mut client =
                HttpsClient::from_env(origin, request.provider.label(), Some(token), http_options)?;
            if request.provider == Provider::GitHub {
                client = client.with_credential_origin("https://raw.githubusercontent.com")?;
            }
            let admission = admission(options.memory)?;
            let cancel = Arc::new(AtomicBool::new(false));
            let _signals = signals::CancellationSignals::register(cancel.clone())?;
            let plan =
                acquisition::resolve(&client, request, &limits(&options)?, &admission, &cancel)?;
            if options.dry_run {
                println!(
                    "Provider: {}\nRepository: {}\nPinned revision: {}\nSelected bytes: {}",
                    plan.source().provider(),
                    plan.source().repository(),
                    plan.source().revision(),
                    plan.total_bytes()
                );
                for file in plan.files() {
                    println!(
                        "{} {} {:?}",
                        file.path(),
                        file.bytes(),
                        file.digest_provenance()
                    );
                }
                println!(
                    "DRY RUN: metadata only; no model-weight transfer/cache publication or inference qualification."
                );
                return Ok(());
            }
            let mut display = ProgressDisplay::default();
            let mut progress = |event: &acquisition::PullProgress<'_>| {
                // Progress output is best-effort; it cannot certify publication.
                let _ = display.report(event, &mut std::io::stderr());
            };
            let receipt = acquisition::execute_with_progress(
                &open(&options)?,
                &plan,
                &client,
                &admission,
                &cancel,
                Some(&reference),
                &mut progress,
            )?;
            match receipt.outcome() {
                CommitOutcome::Durable(snapshot) => {
                    println!("Snapshot: {}\nReference: {reference}\nVerified data only; no inference eligibility implied.", snapshot.digest());
                    use std::io::Write;
                    std::io::stdout().write_all(&receipt.manifest().canonical_bytes()?)?;
                    eprintln!("Verified during acquisition; data only, not immutable against arbitrary same-UID writers or executable model readiness.");
                    Ok(())
                }
                CommitOutcome::DurabilityUnknown { .. } => Err(Error::State("complete cache binding is visible but durability is unknown; inspect before retrying".into())),
            }
        }
    }
}
fn help() {
    println!(
        "USAGE:
    zerogravity pull hf REPO --revision REV --file PATH [--file PATH ...]
    zerogravity pull github OWNER/REPO --revision REV --file PATH
    zerogravity pull github OWNER/REPO --release TAG --asset NAME
    zerogravity pull components PLAN.json [--dry-run | --offline]
    zerogravity pull list
    zerogravity pull inspect DIGEST_OR_REF
    zerogravity pull verify DIGEST_OR_REF

OPTIONS:
    --cache ABS_PATH           Override XDG cache root
                              Default: $XDG_CACHE_HOME/abird/zerogravity/models
                              (fallback $HOME/.cache/abird/zerogravity/models)
    --ref NAME                 Local reference (default provider/repo/requested-revision)
    --subfolder PATH           Paths are relative to this repository subfolder
    --dry-run                  Resolve metadata; no model-weight transfer/publication
    --offline                  Verified local cache only; no token/client/provider construction
    --disk-limit-bytes N       Cooperative cache-owned disk policy (default16GiB)
    --memory-limit-bytes N     This process transfer-staging cap (default128MiB, not all RAM)
    --max-file-bytes N --max-snapshot-bytes N --max-files N
    --max-metadata-bytes N --max-pages N
    --timeout-seconds N        Per HTTP operation total deadline (1..86400, default120)

Component PLAN.json: schema=zerogravity.pull.components.v1, components array
(up to16) of label, role(weights/config/tokenizer/auxiliary), provider
(huggingface/github), repository and revision/files (optional subfolder), or
release/asset. Offline additionally requires snapshot per component. No tokens
or URLs in plans. Original component snapshots remain independent; no atomic
cross-source ref/model-readiness claim.

Credentials: HF_TOKEN or GITHUB_TOKEN environment only; no token command-line option.
HF --file supports bounded component-local * and ? filters, including offline.
SIGINT/SIGTERM request cooperative cancellation of online pulls; a blocked read
can take its configured HTTP deadline. Progress bytes are logical completed bytes
including verified cached/resumed prefixes, not a network-throughput measurement.
Select an exact weight/index or auxiliary path; same-repository required model
closure is added, and missing sidecars or ambiguous variants refuse. First-seen
SHA256 is weaker than a trusted provider digest. Missing listing sizes may be recovered
from pinned provider metadata or bounded body-free HEAD; totals still unknown refuse.
HEAD is metadata only and does not establish content digest trust. Protected cache
roots support explicit cache remove; legacy roots are not migrated. Explicit
run-cache uses the existing supported CPU loader; acquisition
adds no inference formats. Resume requires a rehashed saved prefix and matching strong ETag;
changed or ignored range responses refuse."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }
    #[test]
    fn exact_selection_and_release_are_explicit() {
        assert!(matches!(
            parse(&args(&[
                "github",
                "a/b",
                "--revision",
                "main",
                "--file",
                "x",
                "--file",
                "y"
            ]))
            .unwrap()
            .action,
            Action::Fetch(_)
        ));
        assert!(matches!(
            parse(&args(&[
                "github",
                "a/b",
                "--release",
                "v1",
                "--asset",
                "tiny.gguf"
            ]))
            .unwrap()
            .action,
            Action::Fetch(PullRequest {
                selection: Selection::ReleaseAsset { .. },
                ..
            })
        ));
        for parts in [
            vec!["hf", "a/b"],
            vec!["hf", "a/b", "--release", "v1", "--asset", "x"],
            vec!["github", "a/b", "--revision", "main", "--file", "../x"],
            vec!["list", "--revision", "main"],
            vec!["github", "a/b", "--token", "secret-value"],
        ] {
            let error = parse(&args(&parts)).unwrap_err().to_string();
            assert!(!error.contains("secret-value"));
        }
    }
    #[test]
    fn limits_duplicate_flags_and_relative_cache_refuse() {
        for parts in [
            vec!["list", "--cache", "relative"],
            vec!["list", "--max-pages", "0"],
            vec!["list", "--timeout-seconds", "0"],
            vec!["list", "--timeout-seconds", "86401"],
            vec!["list", "--offline", "--offline"],
            vec!["list", "--max-files", "999999"],
        ] {
            assert!(parse(&args(&parts)).and_then(|o| limits(&o)).is_err());
        }
    }
    #[test]
    fn byte_progress_reports_actual_cache_and_transfer_without_urls() {
        use std::io::Cursor;
        use std::sync::atomic::Ordering;
        use zerogravity::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
        use zerogravity::acquisition::{ResolvedFile, ResolvedPullPlan};
        use zerogravity::cache::{CacheLimits, FileRole, Manifest, ManifestEntry, ManifestSource};
        let root = std::env::temp_dir().join(format!("zg-pull-progress-{}", std::process::id()));
        assert!(!root.exists());
        let store = CacheStore::open(CacheOptions::new(Some(root.clone()), DISK_DEFAULT)).unwrap();
        // An isolated test arbiter avoids sharing the production-domain registration
        // with other parallel tests. Production still uses process_shared().
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("test-host", "progress").unwrap();
        arbiter.register(resource.clone(), MEMORY_DEFAULT).unwrap();
        let admission = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("test", "progress").unwrap(),
        };
        let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let source = ManifestSource::new("github", "fixture/progress", "1".repeat(40)).unwrap();
        let old = Manifest::new(
            source.clone(),
            vec![ManifestEntry::new("old", hash, 3, FileRole::Auxiliary).unwrap()],
            &CacheLimits::default(),
        )
        .unwrap();
        let mut tx = store
            .begin(old, &admission, Arc::new(AtomicBool::new(false)))
            .unwrap();
        tx.import_object("old", &mut Cursor::new(b"abc")).unwrap();
        tx.commit(None).unwrap();
        struct FixtureTransport(std::sync::atomic::AtomicUsize);
        impl HttpTransport for FixtureTransport {
            fn get(&self, _: &HttpRequest, _: &AtomicBool) -> Result<HttpResponse> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(HttpResponse {
                    status: 200,
                    headers: Default::default(),
                    body: Box::new(Cursor::new(b"body")),
                })
            }
        }
        let plan = ResolvedPullPlan::new(
            source,
            vec![
                ResolvedFile::new(
                    "cached",
                    "https://fixture.invalid/?token=secret-value",
                    FileRole::Auxiliary,
                    3,
                    Some(hash.into()),
                    "cached",
                )
                .unwrap(),
                ResolvedFile::new(
                    "download",
                    "https://fixture.invalid/?token=secret-value",
                    FileRole::Auxiliary,
                    4,
                    None,
                    "download",
                )
                .unwrap(),
            ],
            &PullLimits::default(),
        )
        .unwrap();
        let transport = FixtureTransport(Default::default());
        let mut display = ProgressDisplay::default();
        let mut output = Vec::new();
        acquisition::execute_with_progress(
            &store,
            &plan,
            &transport,
            &admission,
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |event| display.report(event, &mut output).unwrap(),
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(
            output.contains("Pull bytes: 7/7 (file 4/4, cached 3, resumed 0)"),
            "{output}"
        );
        assert!(!output.contains("secret-value"));
        assert!(!output.contains("https://"));
        assert_eq!(transport.0.load(Ordering::Relaxed), 1);
        assert_eq!(
            admission
                .arbiter
                .snapshot(&admission.resource)
                .unwrap()
                .reserved,
            0
        );
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn real_process_signals_cancel_pull_and_preserve_previous_reference() {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};
        struct OwnedChild(std::process::Child);
        impl Drop for OwnedChild {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        for signal in ["-INT", "-TERM"] {
            let root = std::env::temp_dir()
                .join(format!("zg-pull-signal-{}-{signal}", std::process::id()));
            assert!(!root.exists());
            let mut child = OwnedChild(
                Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--ignored",
                        "--exact",
                        "pull::tests::signal_child",
                        "--nocapture",
                    ])
                    .env("ZG_PULL_SIGNAL_FIXTURE", &root)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .unwrap(),
            );
            let stdout = child.0.stdout.take().unwrap();
            let (send, recv) = std::sync::mpsc::channel();
            let reader = std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    if line.unwrap().ends_with("PULL_SIGNAL_READY") {
                        let _ = send.send(());
                    }
                }
            });
            recv.recv_timeout(Duration::from_secs(15))
                .expect("child reached real body read");
            assert!(
                Command::new("kill")
                    .args([signal, &child.0.id().to_string()])
                    .status()
                    .unwrap()
                    .success()
            );
            let deadline = Instant::now() + Duration::from_secs(15);
            let status = loop {
                if let Some(status) = child.0.try_wait().unwrap() {
                    break status;
                }
                assert!(Instant::now() < deadline, "signal child did not retire");
                std::thread::sleep(Duration::from_millis(10));
            };
            reader.join().unwrap();
            assert!(status.success(), "signal child failed: {status}");
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    #[ignore = "owned child helper; invoked only by real-process signal test"]
    fn signal_child() {
        use std::io::{Cursor, Read, Write};
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};
        use zerogravity::cache::{CacheLimits, FileRole, Manifest, ManifestEntry, ManifestSource};
        let root = PathBuf::from(std::env::var_os("ZG_PULL_SIGNAL_FIXTURE").unwrap());
        let store = CacheStore::open(CacheOptions::new(Some(root), DISK_DEFAULT)).unwrap();
        let admission = admission(MEMORY_DEFAULT).unwrap();
        let make = |revision: &str, hash: String| {
            Manifest::new(
                ManifestSource::new("github", "fixture/source", revision).unwrap(),
                vec![ManifestEntry::new("tiny", hash, 3, FileRole::Auxiliary).unwrap()],
                &CacheLimits::default(),
            )
            .unwrap()
        };
        let old = make(
            &"1".repeat(40),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
        );
        let mut initial = store
            .begin(old, &admission, Arc::new(AtomicBool::new(false)))
            .unwrap();
        initial
            .import_object("tiny", &mut Cursor::new(b"abc"))
            .unwrap();
        initial.commit(Some("previous")).unwrap();
        let previous = store
            .resolve_ref("previous")
            .unwrap()
            .manifest()
            .digest()
            .unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let guard = signals::CancellationSignals::register(cancel.clone()).unwrap();
        let mut tx = store
            .begin(
                make(&"2".repeat(40), "0".repeat(64)),
                &admission,
                cancel.clone(),
            )
            .unwrap();
        assert!(
            admission
                .arbiter
                .snapshot(&admission.resource)
                .unwrap()
                .reserved
                > 0
        );
        struct WaitingReader {
            cancel: Arc<AtomicBool>,
            read: bool,
        }
        impl Read for WaitingReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if !self.read {
                    self.read = true;
                    buffer[0] = b'a';
                    return Ok(1);
                }
                println!("PULL_SIGNAL_READY");
                std::io::stdout().flush()?;
                let deadline = Instant::now() + Duration::from_secs(12);
                while !self.cancel.load(Ordering::Acquire) {
                    if Instant::now() >= deadline {
                        return Err(std::io::Error::other("fixture signal deadline"));
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(std::io::Error::other("fixture observed cancellation"))
            }
        }
        assert!(
            tx.import_object(
                "tiny",
                &mut WaitingReader {
                    cancel: cancel.clone(),
                    read: false
                }
            )
            .is_err()
        );
        assert!(cancel.load(Ordering::Acquire));
        drop(tx);
        assert_eq!(
            store
                .resolve_ref("previous")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            previous
        );
        assert_eq!(
            admission
                .arbiter
                .snapshot(&admission.resource)
                .unwrap()
                .reserved,
            0
        );
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        drop(guard);
        cancel.store(false, Ordering::Release);
        signal_hook::low_level::raise(signal_hook::consts::SIGINT).unwrap();
        signal_hook::low_level::raise(signal_hook::consts::SIGTERM).unwrap();
        assert!(
            !cancel.load(Ordering::Acquire),
            "scoped callbacks were not removed"
        );
    }
}
