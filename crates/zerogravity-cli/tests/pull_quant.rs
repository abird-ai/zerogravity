//! New explicit quant route; legacy file selection keeps its original policy.
use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn quant_is_optional_and_does_not_read_credentials_or_echo_unknown_args() {
    let out = binary()
        .args([
            "pull",
            "hf",
            "owner/repo",
            "--revision",
            "main",
            "--quant",
            "Q4_K_M",
            "--token",
            "DO_NOT_ECHO",
        ])
        .env("HF_TOKEN", "bad\nvalue")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let error = String::from_utf8_lossy(&out.stderr);
    assert!(error.contains("optional acquisition feature"));
    assert!(!error.contains("DO_NOT_ECHO"));
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
    use zerogravity::cache::{
        CacheLimits, CacheOptions, CacheStore, CommitOutcome, FileRole, Manifest, ManifestEntry,
        ManifestSource, TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const PIN: &str = "1111111111111111111111111111111111111111";
    const HASH: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    static SCHEDULE: Mutex<()> = Mutex::new(());
    fn schedule() -> std::sync::MutexGuard<'static, ()> {
        SCHEDULE.lock().unwrap_or_else(|p| p.into_inner())
    }
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "zg-quant-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!path.exists());
            Self(path)
        }
        fn command(&self) -> Command {
            let mut c = binary();
            c.env("HF_TOKEN", "bad\nvalue")
                .env("GITHUB_TOKEN", "bad\nvalue");
            c
        }
        fn seed(&self, provider: &str, valid_roles: bool, complete: bool) -> String {
            let store = CacheStore::open(CacheOptions::new(
                Some(self.0.clone()),
                16 * 1024 * 1024 * 1024,
            ))
            .unwrap();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("test-host", "quant-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("fixture", "quant-cli").unwrap(),
            };
            let mut paths = vec![
                ("q/a-Q4_K_M-00001-of-00002.gguf", FileRole::Weights),
                ("q/config.json", FileRole::Config),
                ("q/tokenizer.json", FileRole::Tokenizer),
            ];
            if complete {
                paths.push(("q/a-Q4_K_M-00002-of-00002.gguf", FileRole::Weights));
            }
            let entries = paths
                .iter()
                .map(|(p, r)| {
                    ManifestEntry::new(
                        *p,
                        HASH,
                        3,
                        if valid_roles { *r } else { FileRole::Auxiliary },
                    )
                    .unwrap()
                })
                .collect();
            let manifest = Manifest::new(
                ManifestSource::new(provider, "owner/repo", PIN).unwrap(),
                entries,
                &CacheLimits::default(),
            )
            .unwrap();
            let snapshot = manifest.digest().unwrap();
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            for (p, _) in paths {
                tx.import_object(p, &mut Cursor::new(b"abc")).unwrap();
            }
            assert!(matches!(
                tx.commit(Some("alias")).unwrap(),
                CommitOutcome::Durable(_)
            ));
            snapshot
        }
        fn offline(&self, provider: &str, digest: &str) -> Command {
            let mut c = self.command();
            if provider == "xet" {
                c.args(["pull", "xet", "owner/repo"]);
            } else {
                c.args(["pull", provider, "owner/repo"]);
            }
            c.args([
                "--revision",
                PIN,
                "--quant",
                "q4_k_m",
                "--subfolder",
                "q",
                "--offline",
                "--ref",
                digest,
                "--cache",
            ])
            .arg(&self.0);
            c
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            if self.0.exists() {
                std::fs::remove_dir_all(&self.0).unwrap();
            }
        }
    }
    #[test]
    fn ordinary_and_native_help_describe_explicit_tag_no_fallback_and_digest_only_offline() {
        let _schedule = schedule();
        let f = Fixture::new();
        for argv in [vec!["pull", "--help"], vec!["pull", "xet", "--help"]] {
            let out = f.command().args(argv).output().unwrap();
            assert!(out.status.success());
            let t = String::from_utf8_lossy(&out.stdout)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            assert!(t.contains("--quant TAG"));
            assert!(t.contains("instead of --file"));
            assert!(t.contains("no fallback"));
            assert!(t.contains("SNAPSHOT_DIGEST"));
        }
        assert!(!f.0.exists());
    }
    #[test]
    fn offline_hf_github_and_native_use_verified_complete_cached_subset_before_credentials() {
        let _schedule = schedule();
        for (cli, provider) in [
            ("hf", "huggingface"),
            ("github", "github"),
            ("xet", "huggingface"),
        ] {
            let f = Fixture::new();
            let digest = f.seed(provider, true, true);
            let out = f.offline(cli, &digest).output().unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let text = String::from_utf8_lossy(&out.stdout);
            assert!(text.contains(PIN));
            assert!(text.contains("00001-of-00002"));
            assert!(text.contains("00002-of-00002"));
            let note = String::from_utf8_lossy(&out.stderr);
            assert!(note.contains("verified cache subset"));
            assert!(note.contains("not remote uniqueness, native Xet execution"));
            assert!(!note.contains("bad"));
        }
    }
    #[test]
    fn new_offline_role_and_complete_split_rules_do_not_change_legacy_exact_file_policy() {
        let _schedule = schedule();
        let f = Fixture::new();
        let snapshot = f.seed("huggingface", false, false);
        assert!(
            !f.offline("hf", &snapshot)
                .output()
                .unwrap()
                .status
                .success()
        );
        let out = f
            .command()
            .args([
                "pull",
                "hf",
                "owner/repo",
                "--revision",
                PIN,
                "--file",
                "q/a-Q4_K_M-00001-of-00002.gguf",
                "--offline",
                "--ref",
                "alias",
                "--cache",
            ])
            .arg(&f.0)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let f = Fixture::new();
        let snapshot = f.seed("huggingface", true, false);
        assert!(
            !f.offline("xet", &snapshot)
                .output()
                .unwrap()
                .status
                .success()
        );
    }
    #[test]
    fn conflicts_invalid_labels_offline_aliases_and_missing_digest_refuse_before_io() {
        let _schedule = schedule();
        let f = Fixture::new();
        for provider in ["hf", "github", "xet"] {
            for suffix in [
                vec!["--file", "anything.gguf"],
                vec!["--quant", "Q8_0"],
                vec!["--release", "tag", "--asset", "a.gguf"],
                vec!["--unknown-size-cap-bytes", "1"],
                vec!["--token", "DO_NOT_ECHO"],
                vec!["--offline"],
                vec!["--offline", "--ref", "alias"],
                vec![
                    "--offline",
                    "--ref",
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "--dry-run",
                ],
            ] {
                let out = f
                    .command()
                    .args([
                        "pull",
                        provider,
                        "owner/repo",
                        "--revision",
                        "main",
                        "--quant",
                        "Q4_K_M",
                    ])
                    .args(suffix)
                    .output()
                    .unwrap();
                assert!(!out.status.success());
                let err = String::from_utf8_lossy(&out.stderr);
                assert!(!err.contains("DO_NOT_ECHO"));
                assert!(!err.contains("invalid credential"));
                assert!(!f.0.exists());
            }
            for tag in ["Q*", "Q-4", "Q4/path", "Ｑ4", "Q4 K"] {
                let out = f
                    .command()
                    .args([
                        "pull",
                        provider,
                        "owner/repo",
                        "--revision",
                        "main",
                        "--quant",
                        tag,
                    ])
                    .output()
                    .unwrap();
                assert!(!out.status.success());
                assert!(!String::from_utf8_lossy(&out.stderr).contains("invalid credential"));
            }
        }
    }
    #[test]
    fn offline_missing_root_does_not_create_and_quant_source_pin_is_still_required() {
        let _schedule = schedule();
        let f = Fixture::new();
        assert!(
            !f.offline("hf", &"a".repeat(64))
                .output()
                .unwrap()
                .status
                .success()
        );
        assert!(!f.0.exists());
        let snapshot = f.seed("github", true, true);
        assert!(
            !f.offline("hf", &snapshot)
                .output()
                .unwrap()
                .status
                .success()
        );
        let out = f
            .offline("github", &snapshot)
            .args(["--max-files", "1"])
            .output()
            .unwrap();
        assert!(!out.status.success());
    }
}
