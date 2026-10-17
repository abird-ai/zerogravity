use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn conditional_modes_require_optional_acquisition_without_io_or_secret_echo() {
    for provider in ["s3", "azure"] {
        let output = binary()
            .args([
                "pull",
                provider,
                "object",
                "--conditional-read",
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
    use zerogravity::acquisition::azure::{AzureConditionalPin, AzureConditionalRequest};
    use zerogravity::acquisition::s3::{
        S3Addressing, S3ConditionalPin, S3ConditionalRequest, S3Endpoint,
    };
    use zerogravity::cache::{
        CacheLimits, CacheOptions, CacheStore, DigestProvenance, FileRole, Manifest, ManifestEntry,
        ManifestSource, TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const TAG: &str = "\"observed-2\"";
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
            let bytes = if empty { 0 } else { 3 };
            match self {
                Self::S3 => S3ConditionalPin::new(
                    &S3ConditionalRequest::new(
                        &S3Endpoint::compatible(S3, "us-east-1", S3Addressing::PathStyle).unwrap(),
                        "bucket",
                        "private/a.bin",
                        "nested/a.bin",
                        FileRole::Auxiliary,
                    )
                    .unwrap(),
                    bytes,
                    TAG,
                )
                .unwrap()
                .source()
                .clone(),
                Self::Azure => AzureConditionalPin::new(
                    &AzureConditionalRequest::new(
                        AZURE,
                        "container",
                        "private/a.bin",
                        "nested/a.bin",
                        FileRole::Auxiliary,
                    )
                    .unwrap(),
                    bytes,
                    TAG,
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
                "zg-conditional-cli-{}-{}",
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
        fn args(&self, provider: Provider, _empty: bool) -> Vec<String> {
            let mut args = vec!["pull".into(), provider.name().into(), "object".into()];
            args.extend(provider.origin_args());
            args.extend(
                [
                    "--file",
                    "nested/a.bin",
                    "--role",
                    "auxiliary",
                    "--conditional-read",
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
                        DigestProvenance::FirstSeen
                    } else {
                        DigestProvenance::TrustedSha256
                    }),
                ],
                &CacheLimits::default(),
            )
            .unwrap();
            let id = manifest.digest().unwrap();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("test-host", "conditional-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("test", "conditional-cli").unwrap(),
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
                "--bytes".into(),
                if empty { "0" } else { "3" }.into(),
                "--etag".into(),
                TAG.into(),
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
                panic!("owned conditional CLI timed out (stdin stays intentionally open)");
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
    fn conditional_help_and_conflicts_refuse_before_env_stdin_or_cache_creation() {
        let _schedule = SCHEDULE.lock().unwrap();
        for provider in [Provider::S3, Provider::Azure] {
            let output = run(&["pull".into(), provider.name().into(), "--help".into()]);
            assert!(output.status.success());
            let text = String::from_utf8_lossy(&output.stdout);
            assert!(text.contains("--conditional-read"));
            assert!(text.contains("If-Match"));
            assert!(text.contains("FirstSeen"));
            let f = Fixture::new();
            for extra in [
                vec!["--bytes", "3"],
                vec!["--etag", TAG],
                vec!["--sha256", SHA],
                vec!["--version-id", "opaque"],
                vec!["--content-contract"],
                vec!["--unknown-size-cap-bytes", "4"],
                vec!["--ref", "NEVER_ECHO"],
                vec!["--conditional-read"],
            ] {
                let mut args = f.args(provider, false);
                args.extend(extra.into_iter().map(str::to_owned));
                args.push("--dry-run".into());
                args.extend(provider.auth());
                refused(&run(&args));
                assert!(!f.cache.exists());
            }
        }
    }
    #[test]
    fn conditional_offline_exact_empty_and_fifo_stdin_precede_credentials_and_admission() {
        let _schedule = SCHEDULE.lock().unwrap();
        for provider in [Provider::S3, Provider::Azure] {
            for empty in [false, true] {
                let f = Fixture::new();
                let id = f.seed(provider, empty, true);
                let mut args = f.offline(provider, &id, empty);
                args.extend(provider.auth());
                args.extend(["--memory-limit-bytes".into(), "1".into()]);
                let out = run(&args);
                assert!(out.status.success(), "{out:?}");
                let text = String::from_utf8_lossy(&out.stdout);
                assert!(text.contains("Conditional observation: FirstSeen"));
                assert!(text.contains("ETag (JSON string):"));
                assert!(text.contains("first_seen"));
                assert!(!text.contains("Version ID"));
            }
        }
        let f = Fixture::new();
        let id = f.seed(Provider::Azure, false, true);
        assert!(
            Command::new("mkfifo")
                .arg(&f.input)
                .status()
                .unwrap()
                .success()
        );
        for input in ["-", f.input.to_str().unwrap()] {
            let mut args = f.offline(Provider::Azure, &id, false);
            args.extend(["--sas-file".into(), input.into()]);
            assert!(run(&args).status.success());
        }
        let f = Fixture::new();
        let mut args = f.offline(Provider::Azure, SHA, false);
        args.extend(["--sas-file".into(), "-".into()]);
        refused(&run(&args));
        assert!(!f.cache.exists());
    }
    #[test]
    fn conditional_offline_requires_exact_contract_first_seen_and_digest_selector() {
        let _schedule = SCHEDULE.lock().unwrap();
        for provider in [Provider::S3, Provider::Azure] {
            let f = Fixture::new();
            let id = f.seed(provider, false, true);
            let base = f.offline(provider, &id, false);
            for (flag, value) in [
                ("--bytes", "4"),
                ("--etag", "\"other\""),
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
            for tag in ["W/\"weak\"", "*", "unquoted", "\"a\",\"b\""] {
                let mut args = base.clone();
                let at = args.iter().position(|v| v == "--etag").unwrap() + 1;
                args[at] = tag.into();
                refused(&run(&args));
            }
            let f = Fixture::new();
            let id = f.seed(provider, false, false);
            refused(&run(&f.offline(provider, &id, false)));
        }
    }
    #[test]
    fn conditional_opt_in_never_falls_back_to_version_or_content_or_prefix() {
        let _schedule = SCHEDULE.lock().unwrap();
        for provider in [Provider::S3, Provider::Azure] {
            let f = Fixture::new();
            let id = f.seed(provider, false, true);
            let base = f.offline(provider, &id, false);
            for flag in ["--conditional-read", "--etag", "--bytes"] {
                let mut args = base.clone();
                let at = args.iter().position(|v| v == flag).unwrap();
                let count = if flag == "--conditional-read" { 1 } else { 2 };
                args.drain(at..at + count);
                refused(&run(&args));
            }
            let mut args = f.args(provider, false);
            args[2] = "prefix".into();
            args.extend(["--prefix".into(), "folder/".into(), "--dry-run".into()]);
            refused(&run(&args));
        }
        let f = Fixture::new();
        let mut args = f.args(Provider::Azure, false);
        args.extend(["--snapshot".into(), "opaque".into(), "--dry-run".into()]);
        refused(&run(&args));
    }
}
