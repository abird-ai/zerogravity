//! Additive full-path filters must not change legacy file selection or defaults.
use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}

#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn filters_are_disabled_without_the_optional_acquisition_feature() {
    let output = binary()
        .args([
            "pull",
            "hf",
            "owner/repo",
            "--revision",
            "main",
            "--include",
            "*.gguf",
        ])
        .env("HF_TOKEN", "not-to-print")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(message.contains("optional acquisition feature"));
    assert!(!message.contains("not-to-print"));
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
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "zg-filters-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            assert!(!path.exists());
            Self(path)
        }
        fn command(&self) -> Command {
            let mut c = binary();
            c.env("HF_TOKEN", "bad\ncredential")
                .env("GITHUB_TOKEN", "bad\ncredential");
            c
        }
        fn seed(&self, provider: &str, extra: bool) -> String {
            let store = CacheStore::open(CacheOptions::new(
                Some(self.0.clone()),
                16 * 1024 * 1024 * 1024,
            ))
            .unwrap();
            let arbiter = ResourceArbiter::new();
            let key = PhysicalDeviceKey::new("test-host", "filters-cli").unwrap();
            arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource: key,
                owner: ResourceOwner::new("fixture", "filters-cli").unwrap(),
            };
            let mut files = vec![
                ("q/model-Q4_K_M.gguf", FileRole::Weights),
                ("q/config.json", FileRole::Config),
                ("q/tokenizer.json", FileRole::Tokenizer),
            ];
            if extra {
                files.push(("unrelated.txt", FileRole::Auxiliary));
            }
            let manifest = Manifest::new(
                ManifestSource::new(provider, "owner/repo", PIN).unwrap(),
                files
                    .iter()
                    .map(|(p, r)| ManifestEntry::new(*p, HASH, 3, *r).unwrap())
                    .collect(),
                &CacheLimits::default(),
            )
            .unwrap();
            let digest = manifest.digest().unwrap();
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            for (p, _) in files {
                tx.import_object(p, &mut Cursor::new(b"abc")).unwrap();
            }
            assert!(matches!(
                tx.commit(None).unwrap(),
                CommitOutcome::Durable(_)
            ));
            digest
        }
        fn offline(&self, cli: &str, digest: &str, args: &[&str]) -> Command {
            let mut command = self.command();
            command
                .args([
                    "pull",
                    cli,
                    "owner/repo",
                    "--revision",
                    PIN,
                    "--subfolder",
                    "q",
                ])
                .args(args)
                .args(["--offline", "--ref", digest, "--cache"])
                .arg(&self.0);
            command
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
    fn help_documents_explicit_unicode_filter_and_index_refusal_policy() {
        let _lock = SCHEDULE.lock().unwrap();
        let f = Fixture::new();
        for args in [vec!["pull", "--help"], vec!["pull", "xet", "--help"]] {
            let output = f.command().args(args).output().unwrap();
            assert!(output.status.success());
            let help = String::from_utf8_lossy(&output.stdout);
            assert!(help.contains("--include GLOB"));
            assert!(help.contains("--exclude GLOB"));
            assert!(help.contains("--quant"));
        }
        assert!(!f.0.exists());
    }
    #[test]
    fn exact_cached_closure_works_for_hf_github_and_xet_with_invalid_credentials() {
        let _lock = SCHEDULE.lock().unwrap();
        for (cli, provider) in [
            ("hf", "huggingface"),
            ("github", "github"),
            ("xet", "huggingface"),
        ] {
            let f = Fixture::new();
            let digest = f.seed(provider, false);
            for request in [
                &["--include", "*.gguf"][..],
                &["--quant", "q4_k_m", "--exclude", "no-match*"][..],
            ] {
                let out = f.offline(cli, &digest, request).output().unwrap();
                assert!(
                    out.status.success(),
                    "{cli}: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                assert!(String::from_utf8_lossy(&out.stdout).contains("model-Q4_K_M.gguf"));
                let stderr = String::from_utf8_lossy(&out.stderr);
                assert!(stderr.contains("closed cached snapshot"));
                assert!(!stderr.contains("bad\ncredential"));
            }
        }
    }
    #[test]
    fn stale_or_extra_cached_data_does_not_form_an_exact_filtered_snapshot() {
        let _lock = SCHEDULE.lock().unwrap();
        let f = Fixture::new();
        let digest = f.seed("huggingface", true);
        for cli in ["hf", "xet"] {
            assert!(
                !f.offline(cli, &digest, &["--include", "*.gguf"])
                    .output()
                    .unwrap()
                    .status
                    .success()
            );
        }
        let bad = f
            .offline("hf", "alias", &["--include", "*.gguf"])
            .output()
            .unwrap();
        assert!(!bad.status.success());
        let other = Fixture::new();
        assert!(
            !other
                .offline("hf", &"a".repeat(64), &["--include", "*.gguf"])
                .output()
                .unwrap()
                .status
                .success()
        );
        assert!(!other.0.exists());
    }
    #[test]
    fn conflicts_and_pattern_input_refuse_without_cache_or_credential_access() {
        let _lock = SCHEDULE.lock().unwrap();
        let f = Fixture::new();
        for cli in ["hf", "github", "xet"] {
            for args in [
                vec!["--exclude", "*.gguf"],
                vec!["--include", "*.gguf", "--file", "anything.gguf"],
                vec!["--include", "*.gguf", "--quant", "Q4"],
                vec!["--quant", "Q4", "--include", "*.gguf"],
                vec![
                    "--include",
                    "*.gguf",
                    "--release",
                    "main",
                    "--asset",
                    "any.gguf",
                ],
                vec!["--include", "*.gguf", "--unknown-size-cap-bytes", "1"],
                vec!["--include", ""],
                vec!["--include", "invalid\nname"],
                vec!["--include", "*.gguf", "--token", "DO_NOT_ECHO"],
                vec!["--include", "*.gguf", "--offline", "--ref", "alias"],
                vec!["--include", "*.gguf", "--offline"],
                vec![
                    "--include",
                    "*.gguf",
                    "--offline",
                    "--dry-run",
                    "--ref",
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ],
            ] {
                let out = f
                    .command()
                    .args(["pull", cli, "owner/repo", "--revision", "main"])
                    .args(args)
                    .output()
                    .unwrap();
                assert!(!out.status.success(), "{cli}");
                let error = String::from_utf8_lossy(&out.stderr);
                assert!(!error.contains("DO_NOT_ECHO"));
                assert!(!error.contains("bad\ncredential"));
                assert!(!f.0.exists());
            }
        }
    }
}
#[cfg(all(feature = "acquisition", target_os = "linux"))]
#[test]
fn bundle_cli_requires_explicit_paths_and_refuses_quant_or_legacy_selection() {
    let help = binary().args(["pull", "--help"]).output().unwrap();
    assert!(help.status.success());
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(text.contains("--bundle"));
    assert!(text.contains("Auxiliary"));
    let base = ["pull", "hf", "owner/repo", "--revision", "main", "--bundle"];
    for extra in [
        vec![],
        vec!["--include", "*.safetensors", "--quant", "Q4_K_M"],
        vec!["--include", "*.safetensors", "--file", "weights.gguf"],
        vec!["--include", "*.safetensors", "--release", "v1"],
        vec![
            "--include",
            "*.safetensors",
            "--unknown-size-cap-bytes",
            "1000",
        ],
        vec!["--include", "*.safetensors", "--bundle"],
    ] {
        let mut args = base.to_vec();
        args.extend(extra);
        let out = binary()
            .args(&args)
            .env("HF_TOKEN", "NEVER_ECHO_SECRET")
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(!String::from_utf8_lossy(&out.stderr).contains("NEVER_ECHO_SECRET"));
    }
}
