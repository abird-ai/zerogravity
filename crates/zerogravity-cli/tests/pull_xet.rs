use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn xet_requires_optional_acquisition_without_argument_echo_or_io() {
    let output = binary()
        .args(["pull", "xet", "owner/repo", "--token", "NEVER_ECHO"])
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
    use zerogravity::cache::{
        CacheOptions, CacheStore, CommitOutcome, DigestProvenance, FileRole, Manifest,
        ManifestEntry, ManifestSource, TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1\n";
    const POINTER_SHA: &str = "8fed3c028d7d2b57b9510d41270fed0c46ebf1c7e55e432823e4bc6d83fae2bd";
    static SCHEDULE: Mutex<()> = Mutex::new(());
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "zg-xet-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!root.exists());
            Self(root)
        }
        fn args(&self) -> Vec<String> {
            let mut args: Vec<_> = [
                "pull",
                "xet",
                "owner/repo",
                "--revision",
                "main",
                "--file",
                "nested/*.txt",
                "--cache",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect();
            args.push(self.0.to_str().unwrap().into());
            args
        }
        fn seed(&self, provider: &str, bytes: &[u8], sha: &str) -> String {
            let manifest = Manifest::new(
                ManifestSource::new(provider, "owner/repo", COMMIT).unwrap(),
                vec![
                    ManifestEntry::new(
                        "nested/a.txt",
                        sha,
                        bytes.len() as u64,
                        FileRole::Auxiliary,
                    )
                    .unwrap()
                    .with_digest_provenance(DigestProvenance::FirstSeen),
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
            let resource = PhysicalDeviceKey::new("fixture", "xet-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("fixture", "xet-cli").unwrap(),
            };
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            tx.import_object("nested/a.txt", &mut Cursor::new(bytes))
                .unwrap();
            match tx.commit(Some("original")).unwrap() {
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
            .env("HF_TOKEN", "invalid\nNEVER_ECHO")
            .env("GITHUB_TOKEN", "invalid\nNEVER_ECHO")
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
                panic!("owned Xet CLI child deadline");
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
    fn help_and_all_native_bound_refusals_precede_credentials_and_cache() {
        let _guard = SCHEDULE.lock().unwrap_or_else(|e| e.into_inner());
        let f = Fixture::new();
        let out = run(&["pull".into(), "xet".into(), "--help".into()]);
        assert!(out.status.success());
        let text = String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        for expected in [
            "no silent compatibility fallback",
            "metadata/discovery only",
            "does not attest native Xet protocol execution",
            "HF_TOKEN",
            "--xet-max-requests",
            "--xet-max-wire-bytes",
        ] {
            assert!(text.contains(expected), "{text}");
        }
        for extras in [
            vec!["--token", "NEVER_ECHO"],
            vec!["--hub-origin", "https://NEVER_ECHO"],
            vec!["--cas-origin", "https://NEVER_ECHO"],
            vec!["--unknown-size-cap-bytes", "10"],
            vec!["--release", "tag"],
            vec!["--asset", "asset"],
            vec!["--provider", "github"],
            vec!["--xet-max-terms", "0"],
            vec!["--xet-max-fetch-groups", "65537"],
            vec!["--xet-max-ranges", "65537"],
            vec!["--xet-max-chunks", "1048577"],
            vec!["--xet-max-requests", "65537"],
            vec!["--xet-max-wire-bytes", "1099511627777"],
            vec!["--xet-max-refreshes", "9"],
            vec!["--xet-max-requests", "1", "--xet-max-requests", "2"],
            vec!["--xet-max-requests", "18446744073709551616"],
            vec!["--dry-run", "--offline"],
        ] {
            let mut args = f.args();
            args.extend(extras.into_iter().map(str::to_owned));
            refused(run(&args));
            assert!(!f.0.exists());
        }
    }
    #[test]
    fn offline_hit_ref_and_filters_precede_invalid_credentials_and_denied_host_budget() {
        let _guard = SCHEDULE.lock().unwrap_or_else(|e| e.into_inner());
        let f = Fixture::new();
        let id = f.seed("huggingface", b"abc", ABC_SHA);
        for reference in [id.as_str(), "original"] {
            let mut args = f.args();
            args.extend([
                "--offline".into(),
                "--ref".into(),
                reference.into(),
                "--memory-limit-bytes".into(),
                "1".into(),
                "--xet-max-refreshes".into(),
                "0".into(),
            ]);
            let out = run(&args);
            assert!(out.status.success(), "{out:?}");
            assert!(String::from_utf8_lossy(&out.stdout).contains(COMMIT));
            assert!(
                String::from_utf8_lossy(&out.stderr)
                    .contains("do not attest native Xet protocol execution")
            );
        }
        let mut args = f.args();
        let i = args.iter().position(|s| s == "nested/*.txt").unwrap();
        args[i] = "*.txt".into();
        args.extend([
            "--subfolder".into(),
            "nested".into(),
            "--offline".into(),
            "--ref".into(),
            id,
        ]);
        assert!(run(&args).status.success());
    }
    #[test]
    fn offline_miss_never_initializes_cache_or_falls_back_to_hf_network() {
        let _guard = SCHEDULE.lock().unwrap_or_else(|e| e.into_inner());
        let f = Fixture::new();
        let mut args = f.args();
        args.extend(["--offline".into(), "--ref".into(), "0".repeat(64)]);
        refused(run(&args));
        assert!(!f.0.exists());
    }
    #[test]
    fn offline_wrong_source_pin_selection_and_tighter_current_limits_refuse() {
        let _guard = SCHEDULE.lock().unwrap_or_else(|e| e.into_inner());
        let f = Fixture::new();
        let wrong = f.seed("github", b"abc", ABC_SHA);
        let mut args = f.args();
        args.extend(["--offline".into(), "--ref".into(), wrong]);
        refused(run(&args));
        let id = f.seed("huggingface", b"abc", ABC_SHA);
        let mut args = f.args();
        args.extend(["--offline".into(), "--ref".into(), id]);
        let index = args.iter().position(|s| s == "main").unwrap();
        args[index] = "f".repeat(40);
        refused(run(&args));
        args[index] = "main".into();
        let file = args.iter().position(|s| s == "nested/*.txt").unwrap();
        args[file] = "absent.txt".into();
        refused(run(&args));
        args[file] = "nested/*.txt".into();
        args.extend(["--max-file-bytes".into(), "2".into()]);
        refused(run(&args));
    }
    #[test]
    fn ordinary_valid_cache_pointer_and_postpublication_corruption_are_not_native_eligibility() {
        let _guard = SCHEDULE.lock().unwrap_or_else(|e| e.into_inner());
        let f = Fixture::new();
        let pointer = f.seed("huggingface", POINTER, POINTER_SHA);
        let mut args = f.args();
        args.extend(["--offline".into(), "--ref".into(), pointer]);
        refused(run(&args));
        let id = f.seed("huggingface", b"abc", ABC_SHA);
        let blob = f.0.join("blobs/sha256").join(ABC_SHA);
        assert!(std::fs::metadata(&blob).unwrap().is_file());
        std::fs::write(&blob, b"bad").unwrap();
        let mut args = f.args();
        args.extend(["--offline".into(), "--ref".into(), id]);
        refused(run(&args));
    }
}
