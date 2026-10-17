use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn azure_requires_optional_acquisition_before_io_without_argv_echo() {
    let output = binary()
        .args(["pull", "azure", "--sas", "NEVER_ECHO"])
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
    use std::io::{Cursor, Write};
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;
    use std::process::Stdio;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };
    use std::time::{Duration, Instant};
    use zerogravity::acquisition::azure::{AzureObjectPin, AzureObjectRequest, AzureRevision};
    use zerogravity::cache::{
        CacheLimits, CacheOptions, CacheStore, DigestProvenance, FileRole, Manifest, ManifestEntry,
        TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const ORIGIN: &str = "https://fixtureaccount.blob.core.windows.net";
    const HASH: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const EMPTY_HASH: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    static SCHEDULE: Mutex<()> = Mutex::new(());
    struct Fixture {
        base: PathBuf,
        cache: PathBuf,
        input: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let base = std::env::temp_dir().join(format!(
                "zg-azure-cli-{}-{}",
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
        fn args(&self) -> Vec<String> {
            [
                "pull",
                "azure",
                "object",
                "--origin",
                ORIGIN,
                "--container",
                "container",
                "--blob",
                "private/a.bin",
                "--file",
                "nested/a.bin",
                "--role",
                "auxiliary",
                "--version-id",
                "opaque-version",
                "--cache",
            ]
            .into_iter()
            .map(str::to_owned)
            .chain([self.cache.to_str().unwrap().to_owned()])
            .collect()
        }
        fn seed(&self, trusted: bool, empty: bool, snapshot: bool) -> String {
            let hash = if empty { EMPTY_HASH } else { HASH };
            let body: &[u8] = if empty { b"" } else { b"abc" };
            let revision = if snapshot {
                AzureRevision::snapshot("opaque-version")
            } else {
                AzureRevision::version("opaque-version")
            }
            .unwrap();
            let request = AzureObjectRequest::new(
                ORIGIN,
                "container",
                "private/a.bin",
                "nested/a.bin",
                FileRole::Auxiliary,
                revision,
                trusted.then_some(hash),
            )
            .unwrap();
            let pin = AzureObjectPin::new(&request, body.len() as u64).unwrap();
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
                    .with_digest_provenance(if trusted {
                        DigestProvenance::TrustedSha256
                    } else {
                        DigestProvenance::FirstSeen
                    }),
                ],
                &CacheLimits::default(),
            )
            .unwrap();
            let digest = manifest.digest().unwrap();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("test-host", "azure-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("test", "azure-cli").unwrap(),
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
            digest
        }
        fn offline(&self, digest: &str, bytes: u64) -> Vec<String> {
            let mut args = self.args();
            args.extend(
                ["--offline", "--cached-snapshot", digest, "--bytes"]
                    .into_iter()
                    .map(str::to_owned),
            );
            args.push(bytes.to_string());
            args
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.base).unwrap();
        }
    }
    fn run(args: &[String], input: Option<&[u8]>, hold_stdin: bool) -> std::process::Output {
        let mut child = binary()
            .args(args)
            .env("AZURE_TEST_TOKEN", "invalid\nNEVER_ECHO")
            .env("AZURE_TEST_KEY", "invalid\nNEVER_ECHO")
            .env("AZURE_STORAGE_CONNECTION_STRING", "NEVER_ECHO")
            .stdin(if input.is_some() || hold_stdin {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        if let Some(input) = input {
            child.stdin.take().unwrap().write_all(input).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if Instant::now() > deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("owned Azure CLI exceeded fixture deadline");
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
    fn help_and_invalid_selections_refuse_before_cache_credentials_or_stdin() {
        let _schedule = SCHEDULE.lock().unwrap();
        let f = Fixture::new();
        let output = run(
            &["pull".into(), "azure".into(), "--help".into()],
            None,
            true,
        );
        assert!(output.status.success());
        let help = String::from_utf8_lossy(&output.stdout)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        for phrase in [
            "metadata only",
            "FirstSeen",
            "independent",
            "No mutable refs",
            "never redirect",
            "before ALL env/file/stdin",
        ] {
            assert!(help.contains(phrase), "{help}");
        }
        for extra in [
            vec!["--token", "NEVER_ECHO"],
            vec!["--ref", "NEVER_ECHO"],
            vec!["--bytes", "3"],
            vec!["--cached-snapshot", HASH],
            vec!["--snapshot", "other"],
            vec![
                "--token-env",
                "AZURE_TEST_TOKEN",
                "--shared-key-env",
                "AZURE_TEST_KEY",
            ],
            vec!["--sas-file", "-", "--token-env", "AZURE_TEST_TOKEN"],
            vec!["--unknown-size-cap-bytes", "5"],
            vec!["--max-files", "0"],
            vec!["--origin", "https://other.example"],
            vec!["--timeout-seconds", "86401"],
        ] {
            let mut args = f.args();
            args.extend(extra.into_iter().map(str::to_owned));
            refused(&run(&args, None, true));
            assert!(!f.cache.exists());
        }
    }
    #[test]
    fn offline_hit_and_miss_precede_all_env_fifo_and_open_stdin_reads() {
        let _schedule = SCHEDULE.lock().unwrap();
        let f = Fixture::new();
        let id = f.seed(false, false, false);
        assert!(
            Command::new("mkfifo")
                .arg(&f.input)
                .status()
                .unwrap()
                .success()
        );
        for auth in [
            vec!["--token-env".into(), "AZURE_TEST_TOKEN".into()],
            vec!["--shared-key-env".into(), "AZURE_TEST_KEY".into()],
            vec!["--sas-file".into(), f.input.to_str().unwrap().into()],
            vec!["--sas-file".into(), "-".into()],
        ] {
            let mut args = f.offline(&id, 3);
            args.extend(auth);
            let output = run(&args, None, true);
            assert!(output.status.success(), "{output:?}");
            let text = String::from_utf8_lossy(&output.stdout);
            assert!(text.contains("Version ID: \"opaque-version\""));
            assert!(text.contains("Exact bytes: 3"));
            assert!(!text.contains("NEVER_ECHO"));
        }
        let missing = Fixture::new();
        let mut args = missing.offline(&id, 3);
        args.extend(["--sas-file".into(), "-".into()]);
        refused(&run(&args, None, true));
        assert!(!missing.cache.exists());
    }
    #[test]
    fn offline_exact_zero_snapshot_namespace_raw_trust_and_limits_are_checked() {
        let _schedule = SCHEDULE.lock().unwrap();
        for (trusted, empty, snapshot) in [
            (true, false, false),
            (false, false, true),
            (true, true, false),
        ] {
            let f = Fixture::new();
            let id = f.seed(trusted, empty, snapshot);
            let mut args = f.offline(&id, if empty { 0 } else { 3 });
            if snapshot {
                let at = args.iter().position(|x| x == "--version-id").unwrap();
                args[at] = "--snapshot".into();
            }
            if trusted {
                args.extend([
                    "--sha256".into(),
                    if empty { EMPTY_HASH } else { HASH }.into(),
                ]);
            }
            let output = run(&args, None, false);
            assert!(output.status.success(), "{output:?}");
            let mut wrong = args.clone();
            let at = wrong.iter().position(|x| x == "--bytes").unwrap() + 1;
            wrong[at] = "4".into();
            refused(&run(&wrong, None, false));
            if !empty {
                let mut narrow = args.clone();
                narrow.extend(["--max-file-bytes".into(), "1".into()]);
                refused(&run(&narrow, None, false));
            }
            let mut wrong = args.clone();
            let at = wrong
                .iter()
                .position(|x| {
                    x == if snapshot {
                        "--snapshot"
                    } else {
                        "--version-id"
                    }
                })
                .unwrap();
            wrong[at] = if snapshot {
                "--version-id"
            } else {
                "--snapshot"
            }
            .into();
            refused(&run(&wrong, None, false));
            let mut wrong = args.clone();
            if trusted {
                let at = wrong.iter().position(|x| x == "--sha256").unwrap();
                wrong.drain(at..at + 2);
            } else {
                wrong.extend(["--sha256".into(), HASH.into()]);
            }
            refused(&run(&wrong, None, false));
        }
    }
    #[test]
    fn secret_regular_file_bounds_and_offline_prefix_refuse_without_network() {
        let _schedule = SCHEDULE.lock().unwrap();
        let f = Fixture::new();
        assert!(
            Command::new("mkfifo")
                .arg(&f.input)
                .status()
                .unwrap()
                .success()
        );
        let sas_args = || {
            let mut a = f.args();
            a.extend([
                "--dry-run".into(),
                "--sas-file".into(),
                f.input.to_str().unwrap().into(),
            ]);
            a
        };
        refused(&run(&sas_args(), None, false));
        std::fs::remove_file(&f.input).unwrap();
        symlink("/dev/null", &f.input).unwrap();
        refused(&run(&sas_args(), None, false));
        std::fs::remove_file(&f.input).unwrap();
        std::fs::write(&f.input, b"NEVER_ECHO").unwrap();
        refused(&run(&sas_args(), None, false));
        std::fs::write(&f.input, vec![b'x'; 8193]).unwrap();
        refused(&run(&sas_args(), None, false));
        assert!(!f.cache.exists());
        let args: [&str; 12] = [
            "pull",
            "azure",
            "prefix",
            "--origin",
            ORIGIN,
            "--container",
            "container",
            "--prefix",
            "folder/",
            "--offline",
            "--sas-file",
            "-",
        ];
        refused(&run(
            &args.into_iter().map(str::to_owned).collect::<Vec<_>>(),
            None,
            true,
        ));
        let mut stdin = f.args();
        stdin.extend(["--dry-run".into(), "--sas-file".into(), "-".into()]);
        refused(&run(&stdin, Some(b"NEVER_ECHO\n"), false));
        assert!(!f.cache.exists());
    }
}
