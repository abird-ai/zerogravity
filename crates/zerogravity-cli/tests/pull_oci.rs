use std::process::Command;
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
}
#[cfg(not(all(feature = "acquisition", target_os = "linux")))]
#[test]
fn oci_requires_optional_acquisition_without_echo_or_io() {
    let output = binary()
        .args(["pull", "oci", "--token", "NEVER_ECHO"])
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
    use std::collections::BTreeMap;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };
    use std::time::{Duration, Instant};
    use zerogravity::acquisition::PullLimits;
    use zerogravity::acquisition::http::{HttpRequest, HttpResponse, HttpTransport};
    use zerogravity::acquisition::oci::{GgufLayer, OciProfile, OciReference, OciRequest};
    use zerogravity::cache::{
        CacheOptions, CacheStore, Manifest, ManifestEntry, TransferAdmission,
    };
    use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    // Serialize unrelated process spawns before any managed descriptor is opened.
    static SCHEDULE: Mutex<()> = Mutex::new(());
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "zg-oci-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!root.exists());
            Self(root)
        }
        fn args(&self, profile: &str) -> Vec<String> {
            let mut args: Vec<String> = [
                "pull",
                "oci",
                "--origin",
                "https://registry.example",
                "--profile",
                profile,
                "--repository",
                "owner/repo",
                "--tag",
                "v1",
                "--file",
                "nested/model.gguf",
                "--cache",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect();
            args.push(self.0.to_str().unwrap().to_owned());
            args
        }
        fn seed(&self, profile: OciProfile) -> String {
            self.seed_layer(profile, GgufLayer::Unique)
        }
        fn seed_layer(&self, profile: OciProfile, layer: GgufLayer) -> String {
            let request = OciRequest::new(
                "https://registry.example",
                profile,
                "owner/repo",
                OciReference::Tag("v1".into()),
                "nested/model.gguf",
                layer,
            )
            .unwrap();
            let limits = PullLimits::default();
            let arbiter = ResourceArbiter::new();
            let resource = PhysicalDeviceKey::new("test-host", "oci-cli").unwrap();
            arbiter
                .register(resource.clone(), 128 * 1024 * 1024)
                .unwrap();
            let admission = TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("test", "oci-cli").unwrap(),
            };
            let plan = request
                .resolve(&Metadata, &limits, &admission, &AtomicBool::new(false))
                .unwrap();
            let file = &plan.files()[0];
            let manifest = Manifest::new(
                plan.source().clone(),
                vec![
                    ManifestEntry::new(file.path(), SHA, 3, file.role())
                        .unwrap()
                        .with_digest_provenance(file.digest_provenance()),
                ],
                &limits.cache,
            )
            .unwrap();
            let digest = manifest.digest().unwrap();
            let store = CacheStore::open(CacheOptions::new(
                Some(self.0.clone()),
                16 * 1024 * 1024 * 1024,
            ))
            .unwrap();
            let mut tx = store
                .begin(manifest, &admission, Arc::new(AtomicBool::new(false)))
                .unwrap();
            tx.import_object("nested/model.gguf", &mut Cursor::new(b"abc"))
                .unwrap();
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
    struct Metadata;
    impl HttpTransport for Metadata {
        fn get(&self, request: &HttpRequest, _: &AtomicBool) -> zerogravity::Result<HttpResponse> {
            assert_eq!(
                request.url,
                "https://registry.example/v2/owner/repo/manifests/v1"
            );
            let bytes = format!(r#"{{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json","config":{{"mediaType":"application/vnd.oci.empty.v1+json","digest":"sha256:{}","size":2}},"layers":[{{"mediaType":"application/vnd.docker.ai.gguf.v3","digest":"sha256:{SHA}","size":3}}]}}"#, "0".repeat(64)).into_bytes();
            Ok(HttpResponse {
                status: 200,
                headers: BTreeMap::from([(
                    "content-type".into(),
                    "application/vnd.oci.image.manifest.v1+json".into(),
                )]),
                body: Box::new(Cursor::new(bytes)),
            })
        }
    }
    fn run(args: &[String]) -> std::process::Output {
        let mut child = binary()
            .args(args)
            .env("OCI_TEST_TOKEN", "invalid\nNEVER_ECHO_TOKEN")
            .env("GITHUB_TOKEN", "invalid\nNEVER_ECHO_TOKEN")
            .env("HF_TOKEN", "invalid\nNEVER_ECHO_TOKEN")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("owned OCI CLI child exceeded fixture deadline");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn assert_refused(output: &std::process::Output) {
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!text.contains("NEVER_ECHO"));
    }
    #[test]
    fn help_and_argument_refusals_precede_credentials_and_cache() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let help = run(&["pull".into(), "oci".into(), "--help".into()]);
        assert!(help.status.success());
        let text = String::from_utf8_lossy(&help.stdout);
        for expected in [
            "metadata only",
            "no mutable reference",
            "remote membership",
            "compressed",
        ] {
            assert!(text.to_ascii_lowercase().contains(expected), "{text}");
        }
        for extra in [
            vec!["--token", "NEVER_ECHO"],
            vec!["--ref", "NEVER_ECHO"],
            vec!["--tag", "duplicate"],
            vec!["--manifest-sha256", SHA],
            vec!["--layer-sha256", "bad"],
            vec!["--token-env", "OCI_TEST_TOKEN"],
            vec!["--memory-limit-bytes", "0"],
            vec!["--file", "bad.tar"],
            vec!["--timeout-seconds", "86401"],
            vec!["--dry-run", "--offline"],
        ] {
            let mut args = fixture.args("anonymous");
            args.extend(extra.into_iter().map(str::to_owned));
            assert_refused(&run(&args));
            assert!(!fixture.0.exists());
        }
        let mut missing = fixture.args("bearer");
        assert_refused(&run(&missing));
        missing.extend(["--token-env".into(), "OCI_TEST_TOKEN".into()]);
        assert_refused(&run(&missing)); // malformed token refuses before network/cache
        assert!(!fixture.0.exists());
    }
    #[test]
    fn offline_bearer_hit_verifies_data_before_token_and_admission() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let digest = fixture.seed(OciProfile::Bearer);
        for selection in [digest.as_str(), "original"] {
            let mut args = fixture.args("bearer");
            args.extend(
                [
                    "--token-env",
                    "OCI_TEST_TOKEN",
                    "--offline",
                    "--snapshot",
                    selection,
                    "--memory-limit-bytes",
                    "1",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            let output = run(&args);
            assert!(output.status.success(), "{output:?}");
            assert!(String::from_utf8_lossy(&output.stdout).contains(SHA));
            assert!(
                String::from_utf8_lossy(&output.stderr)
                    .contains("no independent remote descriptor membership")
            );
            assert!(!String::from_utf8_lossy(&output.stderr).contains("NEVER_ECHO_TOKEN"));
        }
    }
    #[test]
    fn offline_changed_contract_and_explicit_layer_refuse() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let digest = fixture.seed(OciProfile::Anonymous);
        let mut base = fixture.args("anonymous");
        base.extend(["--offline".into(), "--snapshot".into(), digest]);
        assert!(run(&base).status.success());
        for (flag, value) in [
            ("--repository", "owner/other"),
            ("--origin", "https://other.example"),
            ("--tag", "v2"),
            ("--file", "other.gguf"),
            ("--profile", "dockerhub"),
        ] {
            let mut args = base.clone();
            let index = args.iter().position(|entry| entry == flag).unwrap();
            args[index + 1] = value.into();
            assert_refused(&run(&args));
        }
        // Layer selector is part of the source contract, not merely filename equality.
        let mut wrong = base.clone();
        wrong.extend(["--layer-sha256".into(), "1".repeat(64)]);
        assert_refused(&run(&wrong));
        let mut tight = base;
        tight.extend(["--max-file-bytes".into(), "2".into()]);
        assert_refused(&run(&tight));
        let selected = Fixture::new();
        let selected_digest =
            selected.seed_layer(OciProfile::Anonymous, GgufLayer::Sha256(SHA.into()));
        let mut selected_args = selected.args("anonymous");
        selected_args.extend([
            "--layer-sha256".into(),
            SHA.into(),
            "--offline".into(),
            "--snapshot".into(),
            selected_digest,
        ]);
        assert!(run(&selected_args).status.success());
        let index = selected_args
            .iter()
            .position(|value| value == "--layer-sha256")
            .unwrap();
        selected_args[index + 1] = "1".repeat(64);
        assert_refused(&run(&selected_args));
    }
    #[test]
    fn offline_miss_and_fixed_dockerhub_profile_do_not_create_or_broker() {
        let _schedule = SCHEDULE.lock().unwrap();
        let fixture = Fixture::new();
        let mut args = fixture.args("anonymous");
        args.extend(["--offline".into(), "--snapshot".into(), "0".repeat(64)]);
        assert_refused(&run(&args));
        assert!(!fixture.0.exists());
        let docker: Vec<String> = [
            "pull",
            "oci",
            "--profile",
            "dockerhub",
            "--repository",
            "owner/repo",
            "--tag",
            "v1",
            "--file",
            "model.gguf",
            "--offline",
            "--snapshot",
            "missing",
            "--cache",
        ]
        .into_iter()
        .map(str::to_owned)
        .chain([fixture.0.to_str().unwrap().to_owned()])
        .collect();
        assert_refused(&run(&docker));
        assert!(!fixture.0.exists());
    }
}
