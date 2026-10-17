use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}

#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn acquisition_is_explicit_optional_and_refuses_before_io() {
    let output = binary()
        .args([
            "pull",
            "github",
            "missing/repo",
            "--token",
            "do-not-log-secret",
        ])
        .env("GITHUB_TOKEN", "invalid\ncredential")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("optional acquisition feature"));
    assert!(!error.contains("I/O error"));
    assert!(!error.contains("do-not-log-secret"));
}

#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod enabled {
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };
    use zerogravity::cache::{
        CacheLimits, CacheOptions, CacheStore, FileRole, Manifest, ManifestEntry, ManifestSource,
        TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const COMMIT: &str = "1111111111111111111111111111111111111111";
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "zg-pull-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!path.exists());
            Self(path)
        }
        fn command(&self) -> Command {
            let mut cmd = binary();
            // Invalid credentials prove offline hit did not construct a client
            // or validate token headers. No global environment mutation.
            cmd.env("GITHUB_TOKEN", "invalid\ncredential")
                .env("HF_TOKEN", "invalid\ncredential");
            cmd
        }
        fn store(&self) -> CacheStore {
            CacheStore::open(CacheOptions::new(
                Some(self.0.clone()),
                16 * 1024 * 1024 * 1024,
            ))
            .unwrap()
        }
        fn seed(&self) -> String {
            self.seed_revision(COMMIT)
        }
        fn seed_revision(&self, revision: &str) -> String {
            self.seed_paths("github", revision, &["tiny.gguf"])
        }
        fn seed_paths(&self, provider: &str, revision: &str, paths: &[&str]) -> String {
            let store = self.store();
            let arbiter = ResourceArbiter::new();
            let key = PhysicalDeviceKey::new("test-host", "pull-cli").unwrap();
            arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource: key,
                owner: ResourceOwner::new("fixture", "cli").unwrap(),
            };
            let entries = paths
                .iter()
                .map(|path| {
                    ManifestEntry::new(
                        *path,
                        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                        3,
                        FileRole::Weights,
                    )
                    .unwrap()
                })
                .collect();
            let manifest = Manifest::new(
                ManifestSource::new(provider, "owner/repo", revision).unwrap(),
                entries,
                &CacheLimits::default(),
            )
            .unwrap();
            let digest = manifest.digest().unwrap();
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            for path in paths {
                tx.import_object(path, &mut Cursor::new(b"abc")).unwrap();
            }
            tx.commit(Some("original")).unwrap();
            digest
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
    fn offline_hit_uses_exact_local_bytes_before_credentials_and_client() {
        let fixture = Fixture::new();
        let digest = fixture.seed();
        let output = fixture
            .command()
            .args([
                "pull",
                "github",
                "owner/repo",
                "--revision",
                COMMIT,
                "--file",
                "tiny.gguf",
                "--offline",
                "--ref",
                "original",
                "--cache",
            ])
            .arg(&fixture.0)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("ba7816bf"));
        assert!(text.contains(COMMIT));
        for command in ["inspect", "verify"] {
            let output = fixture
                .command()
                .args(["pull", command, &digest, "--cache"])
                .arg(&fixture.0)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let list = fixture
            .command()
            .args(["pull", "list", "--cache"])
            .arg(&fixture.0)
            .output()
            .unwrap();
        assert!(list.status.success());
        assert!(String::from_utf8_lossy(&list.stdout).contains(&digest));
    }
    #[test]
    fn offline_miss_or_foreign_selection_never_uses_invalid_credentials() {
        let fixture = Fixture::new();
        fixture.seed();
        for (reference, file, repository) in [
            ("missing", "tiny.gguf", "owner/repo"),
            ("original", "absent.gguf", "owner/repo"),
            ("original", "tiny.gguf", "foreign/repo"),
        ] {
            let output = fixture
                .command()
                .args([
                    "pull",
                    "github",
                    repository,
                    "--revision",
                    COMMIT,
                    "--file",
                    file,
                    "--offline",
                    "--ref",
                    reference,
                    "--cache",
                ])
                .arg(&fixture.0)
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(2));
            let text = String::from_utf8_lossy(&output.stderr);
            assert!(!text.contains("credential"), "{text}");
            assert!(!text.contains("HTTP"), "{text}");
        }
    }

    #[test]
    fn uppercase_exact_commit_does_not_accept_another_cached_revision() {
        let fixture = Fixture::new();
        fixture.seed();
        let output = fixture
            .command()
            .args([
                "pull",
                "github",
                "owner/repo",
                "--revision",
                &"F".repeat(40),
                "--file",
                "tiny.gguf",
                "--offline",
                "--ref",
                "original",
                "--cache",
            ])
            .arg(&fixture.0)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("different pinned revision"));
    }

    #[test]
    fn offline_provider_locator_kind_cannot_be_changed_by_alias() {
        for (cached_revision, release_request, ok) in [
            ("release-91-asset-113", false, false),
            (COMMIT, true, false),
            ("release-91-asset-113", true, true),
        ] {
            let fixture = Fixture::new();
            fixture.seed_revision(cached_revision);
            let mut cmd = fixture.command();
            cmd.args(["pull", "github", "owner/repo"]);
            if release_request {
                cmd.args(["--release", &"A".repeat(40), "--asset", "tiny.gguf"]);
            } else {
                cmd.args(["--revision", "main", "--file", "tiny.gguf"]);
            }
            let output = cmd
                .args(["--offline", "--ref", "original", "--cache"])
                .arg(&fixture.0)
                .output()
                .unwrap();
            assert_eq!(
                output.status.success(),
                ok,
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            if !ok {
                assert!(String::from_utf8_lossy(&output.stderr).contains("locator kind"));
            }
        }
    }
    #[test]
    fn token_argv_is_rejected_and_help_does_not_create_cache() {
        let output = binary()
            .args(["pull", "github", "a/b", "--token", "do-not-log-secret"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("do-not-log-secret"));
        let output = binary()
            .args(["pull", "--help"])
            .env("GITHUB_TOKEN", "invalid\ncredential")
            .output()
            .unwrap();
        assert!(output.status.success());
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("--offline"));
        assert!(text.contains("First-seen"));
        assert!(text.contains("strong ETag"));
    }
    #[test]
    fn offline_hf_filters_reuse_online_selection_without_credentials() {
        for (paths, pattern, subfolder, expected) in [
            (
                vec!["weights/q4.gguf", "weights/config.json"],
                "q?.gguf",
                Some("weights"),
                true,
            ),
            (vec!["weights/q4.gguf"], "*.gguf", None, false),
            (vec!["q4.gguf", "q8.gguf"], "*.gguf", None, false),
            (vec!["weights/q4.gguf"], "missing*", Some("weights"), false),
            (vec!["weights/日本.json"], "日*.json", Some("weights"), true),
        ] {
            let fixture = Fixture::new();
            fixture.seed_paths("huggingface", COMMIT, &paths);
            let mut command = fixture.command();
            command
                .args([
                    "pull",
                    "hf",
                    "owner/repo",
                    "--revision",
                    COMMIT,
                    "--file",
                    pattern,
                    "--offline",
                    "--ref",
                    "original",
                    "--cache",
                ])
                .arg(&fixture.0);
            if let Some(folder) = subfolder {
                command.args(["--subfolder", folder]);
            }
            let output = command.output().unwrap();
            assert_eq!(
                output.status.success(),
                expected,
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains("credential"), "{stderr}");
            assert!(!stderr.contains("HTTP"), "{stderr}");
        }
    }
}
