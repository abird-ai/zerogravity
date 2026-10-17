use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn s3_requires_optional_acquisition_without_io_or_argument_echo() {
    let output = binary()
        .args(["pull", "s3", "--secret-key", "NEVER_ECHO"])
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
    use zerogravity::acquisition::s3::{S3Addressing, S3Endpoint, S3ObjectPin, S3ObjectRequest};
    use zerogravity::cache::{
        CacheLimits, CacheOptions, CacheStore, DigestProvenance, FileRole, Manifest, ManifestEntry,
        TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const HASH: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const EMPTY_HASH: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const VERSION: &str = "opaque/+= version\"x";
    static SCHEDULE: Mutex<()> = Mutex::new(());
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "zg-s3-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!path.exists());
            Self(path)
        }
        fn args(&self) -> Vec<String> {
            [
                "pull",
                "s3",
                "object",
                "--region",
                "us-east-1",
                "--addressing",
                "path",
                "--bucket",
                "example-bucket",
                "--key",
                "private//a/../x.bin",
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
            self.seed_endpoint(
                trusted,
                empty,
                S3Endpoint::aws("us-east-1", S3Addressing::PathStyle).unwrap(),
            )
        }
        fn seed_endpoint(&self, trusted: bool, empty: bool, endpoint: S3Endpoint) -> String {
            let hash = if empty { EMPTY_HASH } else { HASH };
            let body: &[u8] = if empty { b"" } else { b"abc" };
            let request = S3ObjectRequest::new(
                &endpoint,
                "example-bucket",
                "private//a/../x.bin",
                "nested/a.bin",
                FileRole::Auxiliary,
                Some(VERSION),
                trusted.then_some(hash),
            )
            .unwrap();
            let pin = S3ObjectPin::new(&request, VERSION, body.len() as u64).unwrap();
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
            let resource = PhysicalDeviceKey::new("test-host", "s3-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("test", "s3-cli").unwrap(),
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
                    "--version-id",
                    VERSION,
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
            .env("S3_ACCESS", "invalid\nNEVER_ECHO_ACCESS")
            .env("S3_SECRET", "invalid\nNEVER_ECHO_SECRET")
            .env("S3_SESSION", "invalid\nNEVER_ECHO_SESSION")
            .env("AWS_ACCESS_KEY_ID", "invalid\nNEVER_ECHO_AMBIENT")
            .env("AWS_SECRET_ACCESS_KEY", "invalid\nNEVER_ECHO_AMBIENT")
            .env("AWS_CONFIG_FILE", "/must-not-read/NEVER_ECHO_CONFIG")
            .env(
                "AWS_SHARED_CREDENTIALS_FILE",
                "/must-not-read/NEVER_ECHO_CREDS",
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
                panic!("owned S3 CLI child exceeded fixture deadline");
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
    fn help_and_endpoint_auth_arguments_refuse_before_cache_or_discovery() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let help = run(&["pull".into(), "s3".into(), "--help".into()]);
        assert!(help.status.success());
        let text = String::from_utf8_lossy(&help.stdout);
        for phrase in [
            "metadata only",
            "FirstSeen",
            "JSON strings",
            "No",
            "non-null",
            "no offline prefix",
        ] {
            assert!(text.contains(phrase), "{text}");
        }
        for extra in [
            vec!["--secret-key", "NEVER_ECHO"],
            vec!["--ref", "NEVER_ECHO"],
            vec!["--endpoint", "https://other.example"],
            vec!["--endpoint-profile", "compatible"],
            vec!["--profile", "sdk"],
            vec!["--profile", "sigv4"],
            vec!["--session-token-env", "S3_SESSION"],
            vec!["--version-id", "null"],
            vec!["--bytes", "3"],
            vec!["--snapshot", HASH],
            vec!["--bucket", "duplicate"],
            vec!["--listing", "current"],
            vec!["--unknown-size-cap-bytes", "3"],
            vec!["--max-files", "0"],
            vec!["--timeout-seconds", "86401"],
        ] {
            let mut args = fixture.args();
            args.extend(extra.into_iter().map(str::to_owned));
            refused(&run(&args));
            assert!(!fixture.0.exists());
        }
        for flag in ["--region", "--addressing", "--role"] {
            let mut args = fixture.args();
            let index = args.iter().position(|entry| entry == flag).unwrap();
            args.drain(index..index + 2);
            refused(&run(&args));
            assert!(!fixture.0.exists());
        }
        let mut args = fixture.args();
        args.extend(
            [
                "--profile",
                "sigv4",
                "--access-key-env",
                "S3_ACCESS",
                "--secret-key-env",
                "S3_SECRET",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        refused(&run(&args));
        assert!(!fixture.0.exists()); // malformed explicit value before network
    }
    #[test]
    fn offline_exact_pin_trust_empty_data_and_escaped_version_precede_env_admission() {
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
                    "sigv4",
                    "--access-key-env",
                    "S3_ACCESS",
                    "--secret-key-env",
                    "S3_SECRET",
                    "--session-token-env",
                    "S3_SESSION",
                    "--memory-limit-bytes",
                    "1",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            let output = run(&args);
            assert!(output.status.success(), "{output:?}");
            let text = String::from_utf8_lossy(&output.stdout);
            assert!(
                text.contains(&format!(
                    "Version ID (JSON string): {}",
                    serde_json::to_string(VERSION).unwrap()
                )),
                "{text}"
            );
            for phrase in [
                "Exact bytes: 3",
                HASH,
                if trusted {
                    "trusted_sha256"
                } else {
                    "first_seen"
                },
            ] {
                assert!(text.contains(phrase), "{text}");
            }
            assert!(!String::from_utf8_lossy(&output.stderr).contains("NEVER_ECHO"));
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
    fn offline_endpoint_addressing_region_and_data_contract_are_bound() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let digest = fixture.seed(true, false);
        let mut base = fixture.offline(&digest, 3);
        base.extend(["--sha256".into(), HASH.into()]);
        assert!(run(&base).status.success());
        for (flag, value) in [
            ("--region", "us-west-2"),
            ("--addressing", "virtual"),
            ("--bucket", "other-bucket"),
            ("--key", "private/a/x.bin"),
            ("--file", "other.bin"),
            ("--role", "config"),
            ("--version-id", "other"),
            ("--bytes", "2"),
            ("--sha256", EMPTY_HASH),
        ] {
            let mut args = base.clone();
            let index = args.iter().position(|entry| entry == flag).unwrap();
            args[index + 1] = value.into();
            refused(&run(&args));
        }
        let mut service = base.clone();
        service.extend(
            [
                "--endpoint-profile",
                "compatible",
                "--endpoint",
                "https://s3.us-east-1.amazonaws.com",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        refused(&run(&service));
        let mut no_hash = base.clone();
        no_hash.truncate(no_hash.len() - 2);
        refused(&run(&no_hash));
        for flag in ["--version-id", "--bytes", "--snapshot"] {
            let mut args = base.clone();
            let index = args.iter().position(|entry| entry == flag).unwrap();
            args.drain(index..index + 2);
            refused(&run(&args));
        }
        let mut tight = base;
        tight.extend(["--max-file-bytes".into(), "2".into()]);
        refused(&run(&tight));
        for (endpoint, extras) in [
            (
                S3Endpoint::aws("us-east-1", S3Addressing::VirtualHosted).unwrap(),
                vec!["--addressing", "virtual"],
            ),
            (
                S3Endpoint::compatible(
                    "https://objects.example",
                    "us-east-1",
                    S3Addressing::PathStyle,
                )
                .unwrap(),
                vec![
                    "--endpoint-profile",
                    "compatible",
                    "--endpoint",
                    "https://objects.example",
                ],
            ),
        ] {
            let other = Fixture::new();
            let digest = other.seed_endpoint(false, false, endpoint);
            let mut args = other.offline(&digest, 3);
            if extras[0] == "--addressing" {
                let index = args
                    .iter()
                    .position(|entry| entry == "--addressing")
                    .unwrap();
                args[index + 1] = "virtual".into();
            } else {
                args.extend(extras.into_iter().map(str::to_owned));
            }
            let output = run(&args);
            assert!(output.status.success(), "{output:?}");
        }
    }
    #[test]
    fn misses_and_prefix_offline_or_object_fields_refuse_without_root_creation() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        refused(&run(&fixture.offline(&"0".repeat(64), 3)));
        assert!(!fixture.0.exists());
        let prefix: Vec<String> = [
            "pull",
            "s3",
            "prefix",
            "--region",
            "us-east-1",
            "--addressing",
            "path",
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
        let mut offline = prefix.clone();
        offline.push("--offline".into());
        let output = run(&offline);
        refused(&output);
        assert!(String::from_utf8_lossy(&output.stderr).contains("offline S3 prefix"));
        for extra in [
            vec!["--file", "bad"],
            vec!["--sha256", HASH],
            vec!["--role", "weights"],
            vec!["--version-id", VERSION],
            vec!["--listing", "all"],
            vec!["--profile", "sigv4"],
        ] {
            let mut args = prefix.clone();
            args.extend(extra.into_iter().map(str::to_owned));
            refused(&run(&args));
            assert!(!fixture.0.exists());
        }
    }
}
