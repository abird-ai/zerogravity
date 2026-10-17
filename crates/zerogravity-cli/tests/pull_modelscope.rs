use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn modelscope_requires_optional_acquisition_without_echo_or_io() {
    let output = binary()
        .args(["pull", "modelscope", "--token", "NEVER_ECHO"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(text.contains("optional acquisition feature"));
    assert!(!text.contains("NEVER_ECHO"));
}
#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod enabled {
    use super::*;
    use std::{
        io::Cursor,
        path::PathBuf,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, AtomicU64, Ordering},
        },
        time::{Duration, Instant},
    };
    use zerogravity::acquisition::modelscope::ModelScopeRequest;
    use zerogravity::cache::{
        CacheOptions, CacheStore, CommitOutcome, DigestProvenance, FileRole, Manifest,
        ManifestEntry, TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    static SCHEDULE: Mutex<()> = Mutex::new(());
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "zg-ms-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!root.exists());
            Self(root)
        }
        fn args(&self) -> Vec<String> {
            let mut args: Vec<_> = [
                "pull",
                "modelscope",
                "owner/repo",
                "--commit",
                COMMIT,
                "--file",
                "a.txt",
                "--cache",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect();
            args.push(self.0.to_str().unwrap().into());
            args
        }
        fn seed(&self) -> String {
            let request = ModelScopeRequest::new(
                "https://modelscope.cn",
                "owner/repo",
                COMMIT,
                vec!["a.txt".into()],
                None,
            )
            .unwrap();
            let manifest = Manifest::new(
                request.source().clone(),
                vec![
                    ManifestEntry::new("a.txt", SHA, 3, FileRole::Auxiliary)
                        .unwrap()
                        .with_digest_provenance(DigestProvenance::TrustedSha256),
                ],
                &Default::default(),
            )
            .unwrap();
            let store = CacheStore::open(CacheOptions::new(
                Some(self.0.clone()),
                16 * 1024 * 1024 * 1024,
            ))
            .unwrap();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("fixture", "ms-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("fixture", "ms-cli").unwrap(),
            };
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            tx.import_object("a.txt", &mut Cursor::new(b"abc")).unwrap();
            match tx.commit(None).unwrap() {
                CommitOutcome::Durable(s) => s.digest().into(),
                other => panic!("{other:?}"),
            }
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
            .env("MS_TEST_TOKEN", "invalid\nNEVER_ECHO")
            .env("MODELSCOPE_API_TOKEN", "invalid\nNEVER_ECHO")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if Instant::now() >= until {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("owned ModelScope child deadline");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn refused(output: std::process::Output) {
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!text.contains("NEVER_ECHO"));
    }
    #[test]
    fn help_and_invalid_arguments_precede_client_credentials_and_cache() {
        let _guard = SCHEDULE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let f = Fixture::new();
        let out = run(&["pull".into(), "modelscope".into(), "--help".into()]);
        assert!(out.status.success());
        let text = String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        for expected in [
            "3000",
            "metadata only",
            "no ambient",
            "Branches/tags",
            "modelscope.cn",
            "Bearer + m_session_id",
        ] {
            assert!(text.contains(expected), "{text}");
        }
        for extras in [
            vec!["--token", "NEVER_ECHO"],
            vec!["--ref", "NEVER_ECHO"],
            vec!["--origin", "https://NEVER_ECHO"],
            vec!["--unknown-size-cap-bytes", "10"],
            vec!["--profile", "bearer"],
            vec!["--token-env", "MS_TEST_TOKEN"],
            vec!["--dry-run", "--offline"],
            vec!["--timeout-seconds", "86401"],
            vec!["--file", "../escape"],
        ] {
            let mut args = f.args();
            args.extend(extras.into_iter().map(str::to_owned));
            refused(run(&args));
            assert!(!f.0.exists());
        }
        let mut args = f.args();
        let i = args.iter().position(|s| s == COMMIT).unwrap();
        args[i] = "master".into();
        refused(run(&args));
        assert!(!f.0.exists());
    }
    #[test]
    fn offline_explicit_digest_precedes_invalid_token_and_preserves_source_identity() {
        let _guard = SCHEDULE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let f = Fixture::new();
        let digest = f.seed();
        let mut args = f.args();
        args.extend([
            "--offline".into(),
            "--snapshot".into(),
            digest.clone(),
            "--profile".into(),
            "official-token".into(),
            "--token-env".into(),
            "MS_TEST_TOKEN".into(),
        ]);
        let out = run(&args);
        assert!(out.status.success(), "{out:?}");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("modelscope"));
        assert!(text.contains(COMMIT));
        assert!(!text.contains("NEVER_ECHO"));
        let other = COMMIT.replace('0', "1");
        let index = args.iter().position(|s| s == COMMIT).unwrap();
        args[index] = other;
        refused(run(&args));
    }
    #[test]
    fn offline_miss_never_creates_cache_and_requires_canonical_digest_selection() {
        let _guard = SCHEDULE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let f = Fixture::new();
        for snapshot in ["0".repeat(64), "some-ref".into()] {
            let mut args = f.args();
            args.extend(["--offline".into(), "--snapshot".into(), snapshot]);
            refused(run(&args));
            assert!(!f.0.exists());
        }
    }
    #[test]
    fn offline_selection_mismatch_and_tighter_bounds_refuse_without_network() {
        let _guard = SCHEDULE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let f = Fixture::new();
        let digest = f.seed();
        let mut args = f.args();
        args.extend(["--offline".into(), "--snapshot".into(), digest]);
        let i = args.iter().position(|s| s == "a.txt").unwrap();
        args[i] = "missing.txt".into();
        refused(run(&args));
        args[i] = "a.txt".into();
        args.extend(["--max-file-bytes".into(), "1".into()]);
        refused(run(&args));
    }
}
