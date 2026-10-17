use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn candidate_discovery_is_optional_and_refuses_before_io_or_secret_echo() {
    let out = binary()
        .args([
            "pull",
            "candidates",
            "hf",
            "owner/repo",
            "--token",
            "NEVER_ECHO_SECRET",
        ])
        .env("HF_TOKEN", "invalid\ncredential")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let error = String::from_utf8_lossy(&out.stderr);
    assert!(error.contains("optional acquisition feature"));
    assert!(!error.contains("NEVER_ECHO_SECRET"));
    assert!(!error.contains("I/O error"));
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
        CacheLimits, CacheOptions, CacheStore, DigestProvenance, FileRole, Manifest, ManifestEntry,
        ManifestSource, TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const PIN: &str = "1111111111111111111111111111111111111111";
    const HASH: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const DISK: u64 = 16 * 1024 * 1024 * 1024;
    static SCHEDULE: Mutex<()> = Mutex::new(());
    // Acquire first, drop last: unrelated process spawns must not transiently
    // inherit another case's managed activity/payload descriptor before exec.
    fn schedule() -> std::sync::MutexGuard<'static, ()> {
        SCHEDULE.lock().unwrap_or_else(|p| p.into_inner())
    }
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "zg-candidate-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!path.exists());
            Self(path)
        }
        fn command(&self) -> Command {
            let mut c = binary();
            c.env("HF_TOKEN", "invalid\ncredential")
                .env("GITHUB_TOKEN", "invalid\ncredential");
            c
        }
        fn seed(&self, provider: &str, revision: &str) -> String {
            let store = CacheStore::open(CacheOptions::new(Some(self.0.clone()), DISK)).unwrap();
            let arbiter = ResourceArbiter::new();
            let key = PhysicalDeviceKey::new("test-host", "candidate-cli").unwrap();
            arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource: key,
                owner: ResourceOwner::new("fixture", "candidate-cli").unwrap(),
            };
            let entries = [
                "q/q4.gguf",
                "q/q8.gguf",
                "q/nested/quoted\".bin",
                "qq/outside.gguf",
            ]
            .into_iter()
            .enumerate()
            .map(|(i, path)| {
                ManifestEntry::new(path, HASH, 3, FileRole::Auxiliary)
                    .unwrap()
                    .with_digest_provenance(if i == 1 {
                        DigestProvenance::TrustedSha256
                    } else {
                        DigestProvenance::FirstSeen
                    })
            })
            .collect();
            let manifest = Manifest::new(
                ManifestSource::new(provider, "owner/repo", revision).unwrap(),
                entries,
                &CacheLimits::default(),
            )
            .unwrap();
            let snapshot = manifest.digest().unwrap();
            let paths = manifest
                .entries()
                .iter()
                .map(|e| e.path().to_owned())
                .collect::<Vec<_>>();
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            for path in paths {
                tx.import_object(&path, &mut Cursor::new(b"abc")).unwrap();
            }
            assert!(matches!(
                tx.commit(Some("original")).unwrap(),
                zerogravity::cache::CommitOutcome::Durable(_)
            ));
            snapshot
        }
        fn offline(&self, provider: &str, snapshot: &str, revision: &str) -> Command {
            let mut c = self.command();
            c.args([
                "pull",
                "candidates",
                provider,
                "owner/repo",
                "--revision",
                revision,
                "--offline",
                "--snapshot",
                snapshot,
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
    fn help_describes_metadata_only_without_reading_credentials_or_creating_cache() {
        let _schedule = schedule();
        let f = Fixture::new();
        let out = f
            .command()
            .args(["pull", "candidates", "--help"])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        for phrase in [
            "Metadata-only file discovery",
            "multiple GGUF choices",
            "No HEAD, index or model-body transfer",
            "stored Git blobs",
            "verified-cached-subset",
            "existing pull command",
        ] {
            assert!(text.contains(phrase), "missing {phrase}");
        }
        assert!(!f.0.exists());
    }
    #[test]
    fn offline_verified_subset_preserves_multiple_choices_and_source_hash_strength() {
        let _schedule = schedule();
        for (provider, label) in [("hf", "huggingface"), ("github", "github")] {
            let f = Fixture::new();
            let snapshot = f.seed(label, PIN);
            let out = f
                .offline(provider, &snapshot, "main")
                .args(["--subfolder", "q"])
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(value["source"]["revision"], PIN);
            assert_eq!(value["coverage"], "verified-cached-subset");
            assert_eq!(value["discovery_only"], true);
            let files = value["files"].as_array().unwrap();
            assert_eq!(files.len(), 3);
            assert!(files.iter().any(|f| f["path"] == "q/nested/quoted\".bin"));
            let q4 = files.iter().find(|f| f["path"] == "q/q4.gguf").unwrap();
            let q8 = files.iter().find(|f| f["path"] == "q/q8.gguf").unwrap();
            assert_eq!(q4["expected_sha256"], serde_json::Value::Null);
            assert_eq!(q8["expected_sha256"], HASH);
            assert_eq!(q4["bytes"], 3);
            assert_eq!(q4["role_hint"], "weights");
            assert!(
                files
                    .iter()
                    .all(|f| f["size_basis"] == "verified-retained-content")
            );
            // An ordinary acquisition still refuses multiple weight variants;
            // discovery must never change that selector into a fallback.
            let ordinary = f
                .command()
                .args([
                    "pull",
                    provider,
                    "owner/repo",
                    "--revision",
                    PIN,
                    "--file",
                    "*.gguf",
                    "--subfolder",
                    "q",
                    "--offline",
                    "--ref",
                    "original",
                    "--cache",
                ])
                .arg(&f.0)
                .output()
                .unwrap();
            assert!(!ordinary.status.success());
            assert!(!String::from_utf8_lossy(&out.stderr).contains("invalid credential"));
        }
    }
    #[test]
    fn alias_source_mismatch_release_and_current_file_limits_refuse_without_remote_lookup() {
        let _schedule = schedule();
        let f = Fixture::new();
        f.seed("huggingface", PIN);
        let alias = f.offline("hf", "original", PIN).output().unwrap();
        assert!(alias.status.success());
        for (provider, revision) in [
            ("github", PIN),
            ("hf", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ] {
            let out = f.offline(provider, "original", revision).output().unwrap();
            assert!(!out.status.success());
            assert!(!String::from_utf8_lossy(&out.stderr).contains("invalid credential"));
        }
        let count = f
            .offline("hf", "original", PIN)
            .args(["--max-files", "1"])
            .output()
            .unwrap();
        assert!(!count.status.success());
        let release = Fixture::new();
        let snapshot = release.seed("github", "release-1-asset-2");
        assert!(
            !release
                .offline("github", &snapshot, "tag")
                .output()
                .unwrap()
                .status
                .success()
        );
    }
    #[test]
    fn malformed_flags_are_static_and_refuse_before_io_or_credentials() {
        let _schedule = schedule();
        let f = Fixture::new();
        for suffix in [
            vec!["--token", "NEVER_ECHO_SECRET"],
            vec!["--max-file-bytes", "1"],
            vec!["--unknown-size-cap-bytes", "1"],
            vec!["--quant", "Q4_K_M"],
            vec!["--file", "*.gguf"],
            vec!["--origin", "https://NEVER_ECHO_SECRET"],
            vec!["--ref", "NEVER_ECHO_SECRET"],
            vec!["--dry-run"],
            vec!["--max-pages", "0"],
            vec!["--timeout-seconds", "86401"],
            vec!["--revision", "again"],
            vec!["--snapshot", "absent"],
        ] {
            let mut c = f.command();
            c.args([
                "pull",
                "candidates",
                "hf",
                "owner/repo",
                "--revision",
                "main",
            ]);
            c.args(suffix);
            let out = c.output().unwrap();
            assert!(!out.status.success());
            let error = String::from_utf8_lossy(&out.stderr);
            assert!(error.contains("invalid pull candidates"));
            assert!(!error.contains("NEVER_ECHO_SECRET"));
            assert!(!error.contains("invalid credential"));
            assert!(!f.0.exists());
        }
    }
    #[test]
    fn offline_missing_root_never_creates_or_acquires_and_online_cache_argument_is_not_ignored() {
        let _schedule = schedule();
        let f = Fixture::new();
        let out = f.offline("hf", "original", "main").output().unwrap();
        assert!(!out.status.success());
        assert!(!f.0.exists());
        assert!(!String::from_utf8_lossy(&out.stderr).contains("invalid credential"));
        let out = f
            .command()
            .args([
                "pull",
                "candidates",
                "hf",
                "owner/repo",
                "--revision",
                "main",
                "--cache",
            ])
            .arg(&f.0)
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(!f.0.exists());
        assert!(String::from_utf8_lossy(&out.stderr).contains("invalid pull candidates"));
    }
}
