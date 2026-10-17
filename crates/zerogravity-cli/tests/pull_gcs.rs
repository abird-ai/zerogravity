use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn gcs_requires_optional_acquisition_before_io_without_argv_echo() {
    let output = binary()
        .args(["pull", "gcs", "--token", "NEVER_ECHO"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("optional acquisition feature"));
    assert!(!error.contains("NEVER_ECHO"));
}
#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod enabled {
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };
    use std::time::{Duration, Instant};
    use zerogravity::acquisition::gcs::{GcsObjectPin, GcsObjectRequest};
    use zerogravity::cache::{
        CacheLimits, CacheOptions, CacheStore, DigestProvenance, FileRole, Manifest, ManifestEntry,
        TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const HASH: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const EMPTY_HASH: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    static SCHEDULE: Mutex<()> = Mutex::new(());
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "zg-gcs-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!root.exists());
            Self(root)
        }
        fn args(&self) -> Vec<String> {
            [
                "pull",
                "gcs",
                "object",
                "--bucket",
                "example-bucket",
                "--object",
                "private/a.bin",
                "--file",
                "nested/a.bin",
                "--role",
                "auxiliary",
                "--cache",
            ]
            .into_iter()
            .map(str::to_owned)
            .chain([self.0.to_str().unwrap().to_owned()])
            .collect()
        }
        fn seed(&self, trusted: bool, empty: bool) -> String {
            let hash = if empty { EMPTY_HASH } else { HASH };
            let body: &[u8] = if empty { b"" } else { b"abc" };
            let request = GcsObjectRequest::new(
                "https://storage.googleapis.com",
                "example-bucket",
                "private/a.bin",
                "nested/a.bin",
                FileRole::Auxiliary,
                Some("17"),
                trusted.then_some(hash),
            )
            .unwrap();
            let pin = GcsObjectPin::new(&request, "17", "3", body.len() as u64).unwrap();
            let provenance = if trusted {
                DigestProvenance::TrustedSha256
            } else {
                DigestProvenance::FirstSeen
            };
            let manifest = Manifest::new(
                pin.source().clone(),
                vec![
                    ManifestEntry::new(
                        "nested/a.bin",
                        hash,
                        body.len() as u64,
                        FileRole::Auxiliary,
                    )
                    .unwrap()
                    .with_digest_provenance(provenance),
                ],
                &CacheLimits::default(),
            )
            .unwrap();
            let digest = manifest.digest().unwrap();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("test-host", "gcs-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("test", "gcs-cli").unwrap(),
            };
            let store = CacheStore::open(CacheOptions::new(
                Some(self.0.clone()),
                16 * 1024 * 1024 * 1024,
            ))
            .unwrap();
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            tx.import_object("nested/a.bin", &mut Cursor::new(body))
                .unwrap();
            tx.commit(None).unwrap();
            digest
        }
        fn offline(&self, digest: &str, bytes: u64) -> Vec<String> {
            let mut args = self.args();
            args.extend(
                [
                    "--offline",
                    "--snapshot",
                    digest,
                    "--generation",
                    "17",
                    "--metageneration",
                    "3",
                    "--bytes",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            args.push(bytes.to_string());
            args
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            if self.0.exists() {
                std::fs::remove_dir_all(&self.0).unwrap();
            }
        }
    }
    fn run(args: &[String]) -> std::process::Output {
        let mut child = binary()
            .args(args)
            .env("GCS_TEST_TOKEN", "invalid\nNEVER_ECHO_TOKEN")
            .env(
                "GOOGLE_APPLICATION_CREDENTIALS",
                "/must-not-read/NEVER_ECHO",
            )
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("owned GCS CLI child exceeded fixture deadline");
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
    fn help_and_argument_errors_are_bounded_before_cache_or_credentials() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let output = run(&["pull".into(), "gcs".into(), "--help".into()]);
        assert!(output.status.success());
        let text = String::from_utf8_lossy(&output.stdout);
        for phrase in [
            "metadata only",
            "FirstSeen",
            "independent",
            "offline-only",
            "No mutable references",
            "Media redirects refuse",
        ] {
            assert!(text.contains(phrase), "{text}");
        }
        for extra in [
            vec!["--token", "NEVER_ECHO"],
            vec!["--ref", "NEVER_ECHO"],
            vec!["--generation", "01"],
            vec!["--metageneration", "3"],
            vec!["--bytes", "3"],
            vec!["--snapshot", HASH],
            vec!["--bucket", "duplicate"],
            vec!["--profile", "adc"],
            vec!["--token-env", "GCS_TEST_TOKEN"],
            vec!["--origin", "https://other.example"],
            vec!["--timeout-seconds", "86401"],
            vec!["--max-files", "0"],
            vec!["--unknown-size-cap-bytes", "5"],
        ] {
            let mut args = fixture.args();
            args.extend(extra.into_iter().map(str::to_owned));
            refused(&run(&args));
            assert!(!fixture.0.exists());
        }
        let mut missing = fixture.args();
        missing.extend(["--profile".into(), "bearer".into()]);
        refused(&run(&missing));
        missing.extend(["--token-env".into(), "GCS_TEST_TOKEN".into()]);
        refused(&run(&missing)); // malformed explicit bearer refuses before network
        assert!(!fixture.0.exists());
    }
    #[test]
    fn offline_exact_pin_firstseen_and_trusted_data_precede_credentials_and_admission() {
        let _schedule = SCHEDULE.lock().unwrap();
        for trusted in [false, true] {
            let fixture = Fixture::new();
            let digest = fixture.seed(trusted, false);
            let mut args = fixture.offline(&digest, 3);
            if trusted {
                args.extend(["--sha256".into(), HASH.into()]);
            }
            args.extend(
                [
                    "--profile",
                    "bearer",
                    "--token-env",
                    "GCS_TEST_TOKEN",
                    "--memory-limit-bytes",
                    "1",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            let output = run(&args);
            assert!(output.status.success(), "{output:?}");
            let text = String::from_utf8_lossy(&output.stdout);
            for expected in [
                "Generation: 17",
                "Metageneration: 3",
                "Exact bytes: 3",
                HASH,
                if trusted {
                    "trusted_sha256"
                } else {
                    "first_seen"
                },
            ] {
                assert!(text.contains(expected), "{text}");
            }
            assert!(!String::from_utf8_lossy(&output.stderr).contains("NEVER_ECHO_TOKEN"));
        }
        let fixture = Fixture::new();
        let digest = fixture.seed(true, true);
        let mut args = fixture.offline(&digest, 0);
        args.extend(["--sha256".into(), EMPTY_HASH.into()]);
        let output = run(&args);
        assert!(output.status.success(), "{output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("Exact bytes: 0"));
    }
    #[test]
    fn offline_contract_hash_trust_version_and_caller_limits_are_enforced() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let digest = fixture.seed(true, false);
        let mut base = fixture.offline(&digest, 3);
        base.extend(["--sha256".into(), HASH.into()]);
        assert!(run(&base).status.success());
        for (flag, value) in [
            ("--bucket", "different-bucket"),
            ("--object", "private/b.bin"),
            ("--file", "other.bin"),
            ("--role", "config"),
            ("--generation", "18"),
            ("--metageneration", "4"),
            ("--bytes", "2"),
            ("--sha256", EMPTY_HASH),
        ] {
            let mut args = base.clone();
            let index = args.iter().position(|entry| entry == flag).unwrap();
            args[index + 1] = value.into();
            refused(&run(&args));
        }
        let mut no_hash = base.clone();
        no_hash.truncate(no_hash.len() - 2);
        refused(&run(&no_hash));
        let mut tight = base.clone();
        tight.extend(["--max-file-bytes".into(), "2".into()]);
        refused(&run(&tight));
        for flag in ["--generation", "--metageneration", "--bytes", "--snapshot"] {
            let mut missing = base.clone();
            let index = missing.iter().position(|entry| entry == flag).unwrap();
            missing.drain(index..index + 2);
            refused(&run(&missing));
        }
        let mut alias = base;
        let index = alias
            .iter()
            .position(|entry| entry == "--snapshot")
            .unwrap();
        alias[index + 1] = "ordinary-ref-not-allowed".into();
        refused(&run(&alias));
    }
    #[test]
    fn offline_misses_and_prefix_refusal_never_create_cache_or_discover_adc() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        refused(&run(&fixture.offline(&"0".repeat(64), 3)));
        assert!(!fixture.0.exists());
        let prefix: Vec<String> = [
            "pull",
            "gcs",
            "prefix",
            "--bucket",
            "example-bucket",
            "--prefix",
            "private/",
            "--offline",
            "--cache",
        ]
        .into_iter()
        .map(str::to_owned)
        .chain([fixture.0.to_str().unwrap().to_owned()])
        .collect();
        let output = run(&prefix);
        refused(&output);
        assert!(String::from_utf8_lossy(&output.stderr).contains("offline GCS prefix"));
        assert!(!fixture.0.exists());
        for extra in [
            vec!["--file", "bad"],
            vec!["--sha256", HASH],
            vec!["--role", "weights"],
            vec!["--generation", "1"],
        ] {
            let mut args: Vec<String> = [
                "pull",
                "gcs",
                "prefix",
                "--bucket",
                "example-bucket",
                "--prefix",
                "private/",
                "--cache",
            ]
            .into_iter()
            .map(str::to_owned)
            .chain([fixture.0.to_str().unwrap().to_owned()])
            .collect();
            args.extend(extra.into_iter().map(str::to_owned));
            refused(&run(&args));
            assert!(!fixture.0.exists());
        }
    }
}
