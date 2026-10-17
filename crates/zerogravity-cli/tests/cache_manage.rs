use std::process::Command;

fn binary() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_zerogravity"));
    // Maintenance must never create or validate an acquisition client.
    command.env("HF_TOKEN", "invalid\ncredential");
    command.env("GITHUB_TOKEN", "invalid\ncredential");
    command
}

#[cfg(not(all(feature = "cache", target_os = "linux")))]
#[test]
fn cache_maintenance_requires_cache_before_io() {
    let output = binary()
        .args([
            "cache",
            "remove",
            "DO_NOT_ECHO_TOKEN",
            "--apply",
            "--cache",
            "/missing",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("optional cache feature"));
    assert!(!error.contains("DO_NOT_ECHO_TOKEN"));
    assert!(!error.contains("I/O error"));
}

#[cfg(all(feature = "cache", target_os = "linux"))]
mod enabled {
    use super::*;
    use std::io::{Cursor, Read};
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

    const PAYLOAD: &[u8] = b"abc";
    const DIGEST: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const DISK: u64 = 16 * 1024 * 1024 * 1024;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "zg-remove-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!path.exists());
            Self(path)
        }
        fn store(&self) -> CacheStore {
            CacheStore::open(CacheOptions::new(Some(self.0.clone()), DISK)).unwrap()
        }
        fn seed(&self, revision: &str, reference: Option<&str>) -> String {
            let store = self.store();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("test", "maintenance-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("test", "cli").unwrap(),
            };
            let manifest = Manifest::new(
                ManifestSource::new("fixture", "owned/cli", revision).unwrap(),
                vec![
                    ManifestEntry::new("data", DIGEST, PAYLOAD.len() as u64, FileRole::Auxiliary)
                        .unwrap(),
                ],
                &CacheLimits::default(),
            )
            .unwrap();
            let digest = manifest.digest().unwrap();
            let mut attempt = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            attempt
                .import_object("data", &mut Cursor::new(PAYLOAD))
                .unwrap();
            attempt.commit(reference).unwrap();
            digest
        }
        fn remove(&self, digest: &str, mode: &str) -> std::process::Output {
            binary()
                .args(["cache", "remove", digest, mode, "--cache"])
                .arg(&self.0)
                .output()
                .unwrap()
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
    fn help_invalid_inputs_and_missing_root_do_not_create_cache() {
        let fixture = Fixture::new();
        let help = binary()
            .args(["cache", "--help"])
            .env("XDG_CACHE_HOME", &fixture.0)
            .output()
            .unwrap();
        assert!(help.status.success());
        let text = String::from_utf8_lossy(&help.stdout);
        assert!(text.contains("no network"));
        assert!(text.contains("not a reference or path"));
        assert!(!fixture.0.exists());
        for args in [
            vec!["cache", "remove", DIGEST, "--cache"],
            vec!["cache", "remove", "alias", "--apply", "--cache"],
            vec!["cache", "remove", DIGEST, "--apply", "--dry-run", "--cache"],
            vec!["cache", "remove", DIGEST, "--apply", "--apply", "--cache"],
            vec![
                "cache",
                "remove",
                DIGEST,
                "--force",
                "DO_NOT_ECHO_TOKEN",
                "--cache",
            ],
            vec![
                "cache",
                "remove",
                DIGEST,
                "--apply",
                "--disk-limit-bytes",
                "0",
                "--cache",
            ],
        ] {
            let output = binary().args(args).arg(&fixture.0).output().unwrap();
            assert_eq!(output.status.code(), Some(2));
            assert!(!String::from_utf8_lossy(&output.stderr).contains("DO_NOT_ECHO_TOKEN"));
            assert!(!fixture.0.exists());
        }
        let missing_root = fixture.remove(DIGEST, "--apply");
        assert_eq!(missing_root.status.code(), Some(2));
        assert!(
            !fixture.0.exists(),
            "maintenance must not initialize an absent root"
        );
        let missing_explicit_root = binary()
            .args(["cache", "remove", DIGEST, "--apply"])
            .output()
            .unwrap();
        assert_eq!(missing_explicit_root.status.code(), Some(2));
        assert!(!String::from_utf8_lossy(&missing_explicit_root.stderr).contains("credential"));
    }

    #[test]
    fn dry_run_preserves_target_and_apply_reports_observed_charge() {
        let fixture = Fixture::new();
        let digest = fixture.seed("one", None);
        let output = fixture.remove(&digest, "--dry-run");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("DRY RUN"));
        assert!(fixture.store().open_snapshot(&digest).is_ok());
        let output = fixture.remove(&digest, "--apply");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("Observed charged-byte"));
        assert!(!text.contains("physical bytes freed"));
        assert!(fixture.store().open_snapshot(&digest).is_err());
        let repeated = fixture.remove(&digest, "--apply");
        assert!(
            repeated.status.success(),
            "{}",
            String::from_utf8_lossy(&repeated.stderr)
        );
        assert!(
            String::from_utf8_lossy(&repeated.stdout)
                .contains("Removed named entries: 0 files, 0 directories")
        );
    }

    #[test]
    fn references_and_escaped_file_handles_refuse_without_implicit_ref_deletion() {
        let referenced = Fixture::new();
        let digest = referenced.seed("referenced", Some("keep"));
        let output = referenced.remove(&digest, "--apply");
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("references"));
        assert!(referenced.store().resolve_ref("keep").is_ok());

        let fixture = Fixture::new();
        let digest = fixture.seed("escaped", None);
        let lease = fixture.store().open_snapshot(&digest).unwrap();
        let mut escaped = lease.file("data").unwrap().try_clone().unwrap();
        drop(lease);
        let output = fixture.remove(&digest, "--apply");
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("busy"));
        let mut bytes = Vec::new();
        escaped.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, PAYLOAD);
        assert!(fixture.store().open_snapshot(&digest).is_ok());
        drop(escaped);
        let output = fixture.remove(&digest, "--apply");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn removing_one_snapshot_preserves_another_shared_blob_snapshot() {
        let fixture = Fixture::new();
        let first = fixture.seed("first", None);
        let second = fixture.seed("second", None);
        let output = fixture.remove(&first, "--apply");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let lease = fixture.store().open_snapshot(&second).unwrap();
        let mut file = lease.file("data").unwrap().try_clone().unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, PAYLOAD);
        assert!(fixture.store().open_snapshot(&first).is_err());
    }
}
