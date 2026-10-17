use std::process::Command;
fn binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_zerogravity"));
    cmd.env("HF_TOKEN", "invalid\ncredential")
        .env("GITHUB_TOKEN", "invalid\ncredential");
    cmd
}
#[cfg(not(all(feature = "cache", target_os = "linux")))]
#[test]
fn import_is_cache_only_and_refuses_before_io_when_disabled() {
    let output = binary()
        .args([
            "cache",
            "import-hf",
            "--source",
            "/missing/DO_NOT_LOG_TOKEN",
            "--apply",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("optional cache feature"));
    assert!(!error.contains("DO_NOT_LOG_TOKEN"));
    assert!(!error.contains("I/O error"));
}
#[cfg(all(feature = "cache", target_os = "linux"))]
mod enabled {
    use super::*;
    use std::fs;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, symlink};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use zerogravity::cache::{CacheOptions, CacheStore, DigestProvenance};
    const COMMIT: &str = "1111111111111111111111111111111111111111";
    const SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const DISK: u64 = 16 * 1024 * 1024 * 1024;
    struct Fixture {
        base: PathBuf,
        source: PathBuf,
        destination: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let base = std::env::temp_dir().join(format!(
                "zg-import-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::DirBuilder::new().mode(0o700).create(&base).unwrap();
            let source = base.join("models--owner--repo");
            let destination = base.join("destination");
            fs::create_dir_all(source.join(format!("snapshots/{COMMIT}"))).unwrap();
            fs::create_dir(source.join("blobs")).unwrap();
            fs::DirBuilder::new()
                .mode(0o700)
                .create(&destination)
                .unwrap();
            fs::write(source.join(format!("blobs/{SHA}")), b"abc").unwrap();
            symlink(
                format!("../../blobs/{SHA}"),
                source.join(format!("snapshots/{COMMIT}/note.txt")),
            )
            .unwrap();
            Self {
                base,
                source,
                destination,
            }
        }
        fn command(&self, mode: &str) -> Command {
            let mut cmd = binary();
            cmd.args(["cache", "import-hf", "--source"])
                .arg(&self.source)
                .args([
                    "--repository",
                    "owner/repo",
                    "--commit",
                    COMMIT,
                    "--file",
                    "auxiliary:note.txt",
                    mode,
                    "--cache",
                ])
                .arg(&self.destination);
            cmd
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }
    #[test]
    fn help_and_malformed_inputs_precede_source_and_destination_io() {
        let f = Fixture::new();
        let missing = f.base.join("absent");
        let help = binary()
            .args(["cache", "import-hf", "--help"])
            .env("XDG_CACHE_HOME", &missing)
            .output()
            .unwrap();
        assert!(help.status.success());
        let text = String::from_utf8_lossy(&help.stdout);
        assert!(text.contains("no network"));
        assert!(text.contains("first-seen"));
        assert!(!missing.exists());
        for flags in [
            vec!["--file", "auxiliary:../escape"],
            vec!["--file", "secret:DO_NOT_LOG_TOKEN"],
            vec!["--sha256", "missing=bad"],
            vec!["--max-files", "0"],
            vec!["--source", "/missing"],
            vec!["--apply"],
            vec!["--token", "DO_NOT_LOG_TOKEN"],
        ] {
            let mut cmd = binary();
            cmd.args(["cache", "import-hf", "--source"])
                .arg(&missing)
                .args([
                    "--repository",
                    "owner/repo",
                    "--commit",
                    COMMIT,
                    "--file",
                    "auxiliary:note.txt",
                    "--dry-run",
                    "--cache",
                ])
                .arg(&missing)
                .args(flags);
            let output = cmd.output().unwrap();
            assert_eq!(output.status.code(), Some(2));
            let err = String::from_utf8_lossy(&output.stderr);
            assert!(!err.contains("DO_NOT_LOG_TOKEN"));
            assert!(!err.contains("I/O error"), "{err}");
            assert!(!missing.exists());
        }
    }
    #[test]
    fn dry_run_has_no_payload_or_destination_write_then_apply_imports_local_data() {
        let f = Fixture::new();
        let path = f.source.join(format!("blobs/{SHA}"));
        let before = fs::metadata(&path).unwrap();
        let output = f.command("--dry-run").output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("DRY RUN"));
        assert!(text.contains("first-seen; not yet hashed"));
        assert_eq!(fs::read_dir(&f.destination).unwrap().count(), 0);
        let output = f.command("--apply").output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("FirstSeen"));
        assert!(text.contains("copied=3"));
        assert!(text.contains("no remote-origin or model-readiness claim"));
        let store = CacheStore::open_existing(CacheOptions::new(Some(f.destination.clone()), DISK))
            .unwrap();
        let snapshots = store.list_snapshots().unwrap();
        assert_eq!(snapshots.len(), 1);
        let lease = store.open_snapshot(&snapshots[0]).unwrap();
        assert_eq!(lease.manifest().source().provider(), "local-hf-cache");
        assert_eq!(lease.manifest().entries()[0].digest(), SHA);
        assert_eq!(
            lease.manifest().entries()[0].digest_provenance(),
            DigestProvenance::FirstSeen
        );
        let after = fs::metadata(&path).unwrap();
        assert_eq!(
            (
                before.dev(),
                before.ino(),
                before.mode(),
                before.len(),
                before.mtime(),
                before.mtime_nsec(),
                before.ctime(),
                before.ctime_nsec()
            ),
            (
                after.dev(),
                after.ino(),
                after.mode(),
                after.len(),
                after.mtime(),
                after.mtime_nsec(),
                after.ctime(),
                after.ctime_nsec()
            )
        );
        assert_eq!(fs::read(path).unwrap(), b"abc");
    }
    #[test]
    fn caller_sha256_missing_destination_overlap_and_low_admission_are_explicit() {
        let f = Fixture::new();
        let mut cmd = f.command("--apply");
        cmd.args(["--sha256", &format!("note.txt={SHA}")]);
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("TrustedSha256"));
        let f = Fixture::new();
        fs::remove_dir(&f.destination).unwrap();
        let output = f.command("--apply").output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(!f.destination.exists());
        let f = Fixture::new();
        let mut cmd = f.command("--apply");
        cmd.args(["--memory-limit-bytes", "1"]);
        let output = cmd.output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(fs::read_dir(&f.destination).unwrap().count(), 0);
        let f = Fixture::new();
        fs::remove_dir(&f.destination).unwrap();
        symlink(&f.source, &f.destination).unwrap();
        let output = f.command("--apply").output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(!f.source.join("policy.json").exists());
        assert_eq!(
            fs::read(f.source.join(format!("blobs/{SHA}"))).unwrap(),
            b"abc"
        );
    }
}
