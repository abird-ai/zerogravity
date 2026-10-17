use std::process::Command;
fn binary() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_zerogravity"));
    c.env("OBJECT_TEST_TOKEN", "invalid\ncredential")
        .env("HF_TOKEN", "invalid\ncredential");
    c
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn object_requires_optional_acquisition_before_io_or_argument_echo() {
    let output = binary()
        .args([
            "pull",
            "object",
            "--url",
            "https://example.invalid/?secret=DO_NOT_ECHO",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("optional acquisition feature"));
    assert!(!err.contains("DO_NOT_ECHO"));
    assert!(!err.contains("I/O error"));
}
#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod enabled {
    use super::*;
    use std::fs;
    use std::io::{Cursor, Write};
    use std::os::unix::fs::{DirBuilderExt, symlink};
    use std::path::PathBuf;
    use std::process::{Output, Stdio};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    use std::time::{Duration, Instant};
    use zerogravity::acquisition::object::HttpsObjectSpec;
    use zerogravity::cache::{
        CacheLimits, CacheOptions, CacheStore, FileRole, Manifest, ManifestEntry, TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const HASH: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const DISK: u64 = 16 * 1024 * 1024 * 1024;
    struct Fixture {
        root: PathBuf,
        url: PathBuf,
        cache: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "zg-object-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            Self {
                url: root.join("url-input"),
                cache: root.join("absent-cache"),
                root,
            }
        }
        fn command(&self) -> Command {
            let mut c = binary();
            c.args([
                "pull",
                "object",
                "--origin",
                "https://example.invalid",
                "--file",
                "payload.bin",
                "--role",
                "auxiliary",
                "--bytes",
                "3",
                "--sha256",
                HASH,
                "--cache",
            ])
            .arg(&self.cache);
            c
        }
        fn seed(&self) -> String {
            let spec = HttpsObjectSpec::new(
                "https://example.invalid",
                "payload.bin",
                FileRole::Auxiliary,
                3,
                HASH,
            )
            .unwrap();
            let store =
                CacheStore::open(CacheOptions::new(Some(self.cache.clone()), DISK)).unwrap();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("test", "object-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let a = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("fixture", "object-cli").unwrap(),
            };
            let manifest = Manifest::new(
                spec.source().clone(),
                vec![
                    ManifestEntry::new("payload.bin", HASH, 3, FileRole::Auxiliary)
                        .unwrap()
                        .with_digest_provenance(
                            zerogravity::cache::DigestProvenance::TrustedSha256,
                        ),
                ],
                &CacheLimits::default(),
            )
            .unwrap();
            let id = manifest.digest().unwrap();
            let mut tx = store
                .begin(manifest, &a, Arc::new(AtomicBool::new(false)))
                .unwrap();
            tx.import_object("payload.bin", &mut Cursor::new(b"abc"))
                .unwrap();
            tx.commit(Some("object")).unwrap();
            id
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
    fn run(mut c: Command, input: Option<&[u8]>) -> Output {
        c.stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
        let mut child = c.spawn().unwrap();
        if let Some(bytes) = input {
            child.stdin.take().unwrap().write_all(bytes).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("owned CLI child exceeded fixture deadline");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    #[test]
    fn help_and_invalid_arguments_never_read_url_or_initialize_cache() {
        let f = Fixture::new();
        let out = run(
            {
                let mut c = binary();
                c.args(["pull", "object", "--help"]);
                c
            },
            None,
        );
        assert!(out.status.success());
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("token values are never"));
        assert!(text.contains("Exact bytes may be zero"));
        assert!(!f.cache.exists());
        for flags in [
            vec!["--url", "https://example.invalid/?token=DO_NOT_ECHO"],
            vec!["--token", "DO_NOT_ECHO"],
            vec!["--ref", "DO_NOT_ECHO"],
            vec!["--url-file", "relative", "--dry-run"],
            vec!["--snapshot", "missing", "--url-file", "-"],
            vec!["--offline", "--dry-run", "--snapshot", "missing"],
            vec!["--offline"],
            vec![
                "--token-env",
                "invalid-name",
                "--url-file",
                "-",
                "--dry-run",
            ],
            vec!["--max-file-bytes", "2", "--url-file", "-", "--dry-run"],
        ] {
            let mut c = f.command();
            c.args(flags);
            let out = run(c, None);
            assert_eq!(out.status.code(), Some(2));
            let err = String::from_utf8_lossy(&out.stderr);
            assert!(!err.contains("DO_NOT_ECHO"));
            assert!(!err.contains("I/O error"), "{err}");
            assert!(!f.cache.exists());
        }
    }
    #[test]
    fn dry_run_validates_local_signed_url_without_network_credentials_or_cache() {
        let f = Fixture::new();
        let secret = "https://example.invalid/private-object?signature=DO_NOT_ECHO";
        fs::write(&f.url, format!("{secret}\r\n")).unwrap();
        let mut c = f.command();
        c.args(["--dry-run", "--url-file"])
            .arg(&f.url)
            .args(["--token-env", "OBJECT_TEST_TOKEN"]);
        let out = run(c, None);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("caller-supplied contract only"));
        assert!(!text.contains("DO_NOT_ECHO"));
        assert!(!text.contains("private-object"));
        assert!(!f.cache.exists());
        let mut c = f.command();
        c.args(["--dry-run", "--url-file", "-"]);
        let out = run(c, Some(format!("{secret}\n").as_bytes()));
        assert!(out.status.success());
        assert!(!String::from_utf8_lossy(&out.stdout).contains("DO_NOT_ECHO"));
        assert!(!f.cache.exists());
    }
    #[test]
    fn url_input_is_regular_bounded_single_line_and_origin_bound() {
        let f = Fixture::new();
        for bytes in [
            b"https://other.invalid/?secret=DO_NOT_ECHO".to_vec(),
            b"https://example.invalid/object\n\n".to_vec(),
            b"https://example.invalid/\tsecret".to_vec(),
            vec![b'x'; 16385],
            vec![0xff],
        ] {
            fs::write(&f.url, bytes).unwrap();
            let mut c = f.command();
            c.args(["--dry-run", "--url-file"]).arg(&f.url);
            let out = run(c, None);
            assert_eq!(out.status.code(), Some(2));
            assert!(!String::from_utf8_lossy(&out.stderr).contains("DO_NOT_ECHO"));
            assert!(!f.cache.exists());
        }
        fs::remove_file(&f.url).unwrap();
        symlink("/dev/null", &f.url).unwrap();
        let mut c = f.command();
        c.args(["--dry-run", "--url-file"]).arg(&f.url);
        assert_eq!(run(c, None).status.code(), Some(2));
        fs::remove_file(&f.url).unwrap();
        rustix::fs::mknodat(
            rustix::fs::CWD,
            &f.url,
            rustix::fs::FileType::Fifo,
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            0,
        )
        .unwrap();
        let mut c = f.command();
        c.args(["--dry-run", "--url-file"]).arg(&f.url);
        assert_eq!(run(c, None).status.code(), Some(2));
        assert!(!f.cache.exists());
    }
    #[test]
    fn offline_hit_and_miss_never_read_fifo_stdin_token_or_client() {
        let f = Fixture::new();
        let id = f.seed();
        rustix::fs::mknodat(
            rustix::fs::CWD,
            &f.url,
            rustix::fs::FileType::Fifo,
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            0,
        )
        .unwrap();
        let mut c = f.command();
        c.args(["--offline", "--snapshot", &id, "--url-file"])
            .arg(&f.url)
            .args(["--token-env", "OBJECT_TEST_TOKEN"]);
        let out = run(c, None);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8_lossy(&out.stdout).contains("https-object"));
        let mut c = f.command();
        c.args([
            "--offline",
            "--snapshot",
            "object",
            "--url-file",
            "-",
            "--token-env",
            "OBJECT_TEST_TOKEN",
        ]);
        assert!(run(c, None).status.success());
        let mut c = f.command();
        c.args(["--offline", "--snapshot", "missing", "--url-file"])
            .arg(&f.url)
            .args(["--token-env", "OBJECT_TEST_TOKEN"]);
        let out = run(c, None);
        assert_eq!(out.status.code(), Some(2));
        assert!(!String::from_utf8_lossy(&out.stderr).contains("credential"));
        let f = Fixture::new();
        let mut c = f.command();
        c.args(["--offline", "--snapshot", "missing"]);
        assert_eq!(run(c, None).status.code(), Some(2));
        assert!(!f.cache.exists());
    }
}
