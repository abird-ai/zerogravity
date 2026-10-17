use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn content_modes_require_optional_acquisition_without_io_or_secret_echo() {
    for provider in ["s3", "azure"] {
        let output = binary()
            .args([
                "pull",
                provider,
                "object",
                "--content-contract",
                "--secret",
                "NEVER_ECHO",
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        let text = String::from_utf8_lossy(&output.stderr);
        assert!(text.contains("optional acquisition feature"));
        assert!(!text.contains("NEVER_ECHO"));
    }
}
#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod enabled {
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::process::Stdio;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };
    use std::time::{Duration, Instant};
    use zerogravity::acquisition::azure::AzureContentSpec;
    use zerogravity::acquisition::s3::{S3Addressing, S3ContentSpec, S3Endpoint};
    use zerogravity::cache::{
        CacheLimits, CacheOptions, CacheStore, DigestProvenance, FileRole, Manifest, ManifestEntry,
        ManifestSource, TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const AZURE: &str = "https://fixtureaccount.blob.core.windows.net";
    const S3: &str = "https://object.example";
    static SCHEDULE: Mutex<()> = Mutex::new(());
    #[derive(Clone, Copy, Debug)]
    enum Provider {
        S3,
        Azure,
    }
    impl Provider {
        fn name(self) -> &'static str {
            match self {
                Self::S3 => "s3",
                Self::Azure => "azure",
            }
        }
        fn origin_args(self) -> Vec<String> {
            match self {
                Self::S3 => [
                    "--endpoint-profile",
                    "compatible",
                    "--endpoint",
                    S3,
                    "--region",
                    "us-east-1",
                    "--addressing",
                    "path",
                    "--bucket",
                    "bucket",
                    "--key",
                    "private/a.bin",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                Self::Azure => [
                    "--origin",
                    AZURE,
                    "--container",
                    "container",
                    "--blob",
                    "private/a.bin",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            }
        }
        fn source(self, empty: bool) -> ManifestSource {
            let size = if empty { 0 } else { 3 };
            let hash = if empty { EMPTY } else { SHA };
            match self {
                Self::S3 => S3ContentSpec::new(
                    &S3Endpoint::compatible(S3, "us-east-1", S3Addressing::PathStyle).unwrap(),
                    "bucket",
                    "private/a.bin",
                    "nested/a.bin",
                    FileRole::Auxiliary,
                    size,
                    hash,
                )
                .unwrap()
                .source()
                .clone(),
                Self::Azure => AzureContentSpec::new(
                    AZURE,
                    "container",
                    "private/a.bin",
                    "nested/a.bin",
                    FileRole::Auxiliary,
                    size,
                    hash,
                )
                .unwrap()
                .source()
                .clone(),
            }
        }
        fn auth(self) -> Vec<String> {
            match self {
                Self::S3 => [
                    "--profile",
                    "sigv4",
                    "--access-key-env",
                    "BAD_ACCESS",
                    "--secret-key-env",
                    "BAD_SECRET",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                Self::Azure => ["--token-env", "BAD_TOKEN"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            }
        }
        fn cache_selector(self) -> &'static str {
            match self {
                Self::S3 => "--snapshot",
                Self::Azure => "--cached-snapshot",
            }
        }
    }
    struct Fixture {
        base: PathBuf,
        cache: PathBuf,
        input: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let base = std::env::temp_dir().join(format!(
                "zg-content-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&base).unwrap();
            Self {
                cache: base.join("cache"),
                input: base.join("sas"),
                base,
            }
        }
        fn args(&self, provider: Provider, empty: bool) -> Vec<String> {
            let mut args = vec!["pull".into(), provider.name().into(), "object".into()];
            args.extend(provider.origin_args());
            args.extend(
                [
                    "--file",
                    "nested/a.bin",
                    "--role",
                    "auxiliary",
                    "--content-contract",
                    "--bytes",
                    if empty { "0" } else { "3" },
                    "--sha256",
                    if empty { EMPTY } else { SHA },
                    "--cache",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            args.push(self.cache.to_str().unwrap().into());
            args
        }
        fn seed(&self, provider: Provider, empty: bool, trusted: bool) -> String {
            let source = provider.source(empty);
            let body: &[u8] = if empty { b"" } else { b"abc" };
            let manifest = Manifest::new(
                source,
                vec![
                    ManifestEntry::new(
                        "nested/a.bin",
                        if empty { EMPTY } else { SHA },
                        body.len() as u64,
                        FileRole::Auxiliary,
                    )
                    .unwrap()
                    .with_digest_provenance(if trusted {
                        DigestProvenance::TrustedSha256
                    } else {
                        DigestProvenance::FirstSeen
                    }),
                ],
                &CacheLimits::default(),
            )
            .unwrap();
            let id = manifest.digest().unwrap();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("test-host", "content-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("test", "content-cli").unwrap(),
            };
            let store = CacheStore::open(CacheOptions::new(
                Some(self.cache.clone()),
                16 * 1024 * 1024 * 1024,
            ))
            .unwrap();
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            tx.import_object("nested/a.bin", &mut Cursor::new(body))
                .unwrap();
            tx.commit(None).unwrap();
            id
        }
        fn offline(&self, provider: Provider, id: &str, empty: bool) -> Vec<String> {
            let mut args = self.args(provider, empty);
            args.extend([
                "--offline".into(),
                provider.cache_selector().into(),
                id.into(),
            ]);
            args
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.base).unwrap();
        }
    }
    fn run(args: &[String]) -> std::process::Output {
        let mut child = binary()
            .args(args)
            .env("BAD_ACCESS", "invalid\nNEVER_ECHO")
            .env("BAD_SECRET", "invalid\nNEVER_ECHO")
            .env("BAD_TOKEN", "invalid\nNEVER_ECHO")
            .env("AWS_PROFILE", "NEVER_ECHO")
            .env("AZURE_STORAGE_CONNECTION_STRING", "NEVER_ECHO")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if Instant::now() > deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("owned content CLI timed out (stdin stays intentionally open)");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn refused(output: &std::process::Output) {
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        for text in [&output.stdout, &output.stderr] {
            assert!(!String::from_utf8_lossy(text).contains("NEVER_ECHO"));
        }
    }
    #[test]
    fn explicit_content_dry_run_is_local_before_credentials_stdin_or_cache_creation() {
        let _schedule = SCHEDULE.lock().unwrap();
        for provider in [Provider::S3, Provider::Azure] {
            let f = Fixture::new();
            let mut args = f.args(provider, false);
            args.push("--dry-run".into());
            args.extend(provider.auth());
            let output = run(&args);
            assert!(output.status.success(), "{output:?}");
            let text = String::from_utf8_lossy(&output.stdout);
            assert!(text.contains("local caller content contract only"));
            assert!(text.contains("Exact bytes: 3"));
            assert!(text.contains(SHA));
            assert!(!text.contains("Version ID"));
            assert!(!f.cache.exists());
        }
        let f = Fixture::new();
        assert!(
            Command::new("mkfifo")
                .arg(&f.input)
                .status()
                .unwrap()
                .success()
        );
        for input in [f.input.to_str().unwrap(), "-"] {
            let mut args = f.args(Provider::Azure, false);
            args.extend(["--dry-run".into(), "--sas-file".into(), input.into()]);
            assert!(run(&args).status.success());
            assert!(!f.cache.exists());
        }
    }
    #[test]
    fn offline_exact_and_empty_content_precede_all_credentials_and_sas_inputs() {
        let _schedule = SCHEDULE.lock().unwrap();
        for provider in [Provider::S3, Provider::Azure] {
            for empty in [false, true] {
                let f = Fixture::new();
                let id = f.seed(provider, empty, true);
                let mut args = f.offline(provider, &id, empty);
                args.extend(provider.auth());
                let output = run(&args);
                assert!(output.status.success(), "{output:?}");
                let text = String::from_utf8_lossy(&output.stdout);
                assert!(text.contains("Content contract:"));
                assert!(text.contains("trusted_sha256"));
                assert!(text.contains(if empty { EMPTY } else { SHA }));
                if matches!(provider, Provider::Azure) {
                    let mut args = f.offline(provider, &id, empty);
                    args.extend(["--sas-file".into(), "-".into()]);
                    assert!(run(&args).status.success());
                }
            }
        }
        let f = Fixture::new();
        let mut args = f.offline(Provider::Azure, SHA, false);
        args.extend(["--sas-file".into(), "-".into()]);
        refused(&run(&args));
        assert!(!f.cache.exists());
    }
    #[test]
    fn missing_sha_size_selectors_caps_prefix_and_implicit_content_fallback_refuse() {
        let _schedule = SCHEDULE.lock().unwrap();
        for provider in [Provider::S3, Provider::Azure] {
            let f = Fixture::new();
            for flag in ["--bytes", "--sha256"] {
                let mut args = f.args(provider, false);
                let at = args.iter().position(|v| v == flag).unwrap();
                args.drain(at..at + 2);
                args.push("--dry-run".into());
                refused(&run(&args));
                assert!(!f.cache.exists());
            }
            let mut args = f.args(provider, false);
            let at = args.iter().position(|v| v == "--content-contract").unwrap();
            args.remove(at);
            args.push("--dry-run".into());
            refused(&run(&args));
            for extra in [
                vec!["--version-id", "opaque"],
                vec!["--unknown-size-cap-bytes", "4"],
                vec!["--ref", "NEVER_ECHO"],
                vec!["--content-contract"],
                vec!["--max-file-bytes", "1"],
            ] {
                let mut args = f.args(provider, false);
                args.extend(extra.into_iter().map(str::to_owned));
                args.push("--dry-run".into());
                refused(&run(&args));
                assert!(!f.cache.exists());
            }
            let mut args = vec!["pull".into(), provider.name().into(), "prefix".into()];
            let mut source = provider.origin_args();
            let at = source
                .iter()
                .position(|v| {
                    v == if matches!(provider, Provider::S3) {
                        "--key"
                    } else {
                        "--blob"
                    }
                })
                .unwrap();
            source.drain(at..at + 2);
            args.extend(source);
            args.extend(
                [
                    "--prefix",
                    "folder/",
                    "--content-contract",
                    "--bytes",
                    "3",
                    "--sha256",
                    SHA,
                    "--dry-run",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            refused(&run(&args));
        }
        let f = Fixture::new();
        let mut args = f.args(Provider::Azure, false);
        args.extend(["--snapshot".into(), "opaque".into(), "--dry-run".into()]);
        refused(&run(&args));
    }
    #[test]
    fn offline_source_role_bytes_digest_trust_and_digest_selector_remain_exact() {
        let _schedule = SCHEDULE.lock().unwrap();
        for provider in [Provider::S3, Provider::Azure] {
            let f = Fixture::new();
            let id = f.seed(provider, false, true);
            let base = f.offline(provider, &id, false);
            for (flag, value) in [
                ("--bytes", "4"),
                ("--sha256", EMPTY),
                ("--role", "config"),
                ("--file", "other.bin"),
                (provider.cache_selector(), "some-ref"),
            ] {
                let mut args = base.clone();
                let at = args.iter().position(|v| v == flag).unwrap() + 1;
                args[at] = value.into();
                args.extend(provider.auth());
                refused(&run(&args));
            }
            let f = Fixture::new();
            let id = f.seed(provider, false, false);
            let mut args = f.offline(provider, &id, false);
            args.extend(provider.auth());
            refused(&run(&args));
        }
    }
}
