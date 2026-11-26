//! Synthetic genuine GGUF/tokenizer input to the original CPU engine. The
//! explicit ignored serving integration uses an already acquired approved model.
#![cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    process::Command,
};
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}
fn gguf() -> Vec<u8> {
    let mut tensors = Vec::<(String, Vec<u64>, f32)>::new();
    tensors.push(("token_embd.weight".into(), vec![1024, 4], 0.0));
    tensors.push(("output_norm.weight".into(), vec![1024], 1.0));
    for i in 0..24 {
        for (n, shape, v) in [
            ("attn_norm.weight", vec![1024], 1.0),
            ("post_attention_norm.weight", vec![1024], 1.0),
            ("attn_q_norm.weight", vec![64], 1.0),
            ("attn_k_norm.weight", vec![64], 1.0),
            ("attn_q.weight", vec![1024, 128], 0.0),
            ("attn_k.weight", vec![1024, 64], 0.0),
            ("attn_v.weight", vec![1024, 64], 0.0),
            ("attn_output.weight", vec![64, 1024], 0.0),
            ("ffn_gate.weight", vec![1024, 1], 0.0),
            ("ffn_up.weight", vec![1024, 1], 0.0),
            ("ffn_down.weight", vec![1, 1024], 0.0),
        ] {
            tensors.push((format!("blk.{i}.{n}"), shape, v));
        }
    }
    let mut header = b"GGUF".to_vec();
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&1u64.to_le_bytes());
    string(&mut header, "general.architecture");
    header.extend_from_slice(&8u32.to_le_bytes());
    string(&mut header, "qwen35");
    let mut payload = Vec::new();
    for (name, shape, value) in tensors {
        while !payload.len().is_multiple_of(32) {
            payload.push(0);
        }
        string(&mut header, &name);
        header.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in &shape {
            header.extend_from_slice(&d.to_le_bytes());
        }
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let n = shape.iter().product::<u64>() as usize;
        for i in 0..n {
            let v = if name == "token_embd.weight" && i < 1024 {
                1.0f32
            } else {
                value
            };
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    while !header.len().is_multiple_of(32) {
        header.push(0);
    }
    header.extend(payload);
    header
}
fn fixture_named(tag: &str) -> Fixture {
    let root = std::env::temp_dir().join(format!("zg-unified-{tag}-{}", std::process::id()));
    assert!(!root.exists());
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("model.gguf"), gguf()).unwrap();
    std::fs::write(root.join("config.json"),serde_json::to_vec(&json!({"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],"text_config":{"model_type":"qwen3_5_text","hidden_size":1024,"head_dim":64,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":4,"max_position_embeddings":32768,"num_hidden_layers":24,"layer_types":vec!["full_attention";24],"full_attention_interval":1,"linear_conv_kernel_dim":4,"linear_num_key_heads":1,"linear_num_value_heads":1}})).unwrap()).unwrap();
    std::fs::write(root.join("tokenizer.json"),serde_json::to_vec(&json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":1,"content":"<|im_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"<|im_start|>":1,"<|im_end|>":2,"[UNK]":3},"unk_token":"[UNK]"}})).unwrap()).unwrap();
    Fixture(root)
}
fn fixture() -> Fixture {
    fixture_named("main")
}
/// Deterministic direct-CPU run flags shared by every scenario in this file.
const COMMON: [&str; 11] = [
    "--prompt",
    "a",
    "--raw",
    "--max-tokens",
    "1",
    "--context",
    "8",
    "--headroom",
    "0",
    "--device",
    "cpu:0",
];
fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
        .args(args)
        .current_dir(root)
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("HF_TOKEN", "invalid\nnever-used")
        .output()
        .unwrap()
}
fn passed(output: std::process::Output) -> Vec<u8> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "a");
    output.stdout
}
#[test]
fn actual_same_cpu_inference_local_alias_config_xdg_and_cached_provider_forms() {
    let f = fixture();
    let common = [
        "--prompt",
        "a",
        "--raw",
        "--max-tokens",
        "1",
        "--context",
        "8",
        "--headroom",
        "0",
        "--device",
        "cpu:0",
    ];
    let normal = run(&f.0, &[&["run", "./model.gguf"][..], &common].concat());
    let normal_stderr = String::from_utf8_lossy(&normal.stderr);
    assert!(normal_stderr.contains("prefill"), "{normal_stderr}");
    assert!(
        normal_stderr.contains("generated tokens"),
        "{normal_stderr}"
    );
    assert!(!normal_stderr.contains('\x1b'), "{normal_stderr}");
    let baseline = passed(normal);
    // Every normal prompt spelling feeds the same original CPU owner.
    let no_prompt = &common[2..];
    assert_eq!(
        passed(run(
            &f.0,
            &[&["run", "./model.gguf", "a"][..], no_prompt].concat()
        )),
        baseline
    );
    std::fs::write(f.0.join("prompt.txt"), "a").unwrap();
    assert_eq!(
        passed(run(
            &f.0,
            &[
                &["run", "./model.gguf", "--prompt-file", "prompt.txt"][..],
                no_prompt
            ]
            .concat()
        )),
        baseline
    );
    let mut child = Command::new(env!("CARGO_BIN_EXE_zerogravity"))
        .args([&["run", "./model.gguf"][..], no_prompt].concat())
        .current_dir(&f.0)
        .env("HOME", &f.0)
        .env("XDG_CACHE_HOME", f.0.join("cache"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write as _;
    child.stdin.take().unwrap().write_all(b"a").unwrap();
    assert_eq!(passed(child.wait_with_output().unwrap()), baseline);
    assert_eq!(
        passed(run(
            &f.0,
            &[&["run", "./model.gguf", "a", "--pull"][..], no_prompt].concat()
        )),
        baseline
    );
    for invalid in [
        vec!["--offline", "run", "./model.gguf", "a", "--pull"],
        vec!["run", "./model.gguf", "--pull", "--dry-run"],
        vec!["run", "./model.gguf", "a", "--prompt-file", "missing.txt"],
        vec!["run", "qwen3.5-2b", "a", "--backend", "cpu", "--pull"],
    ] {
        let output = run(&f.0, &invalid);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{invalid:?}: {:?}",
            output.stderr
        );
        assert!(output.stdout.is_empty());
        assert!(!f.0.join("cache").exists());
    }
    let quiet = run(
        &f.0,
        &[&["-q", "run", "./model.gguf"][..], &common].concat(),
    );
    assert!(
        quiet.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&quiet.stderr)
    );
    assert_eq!(passed(quiet), baseline);
    let verbose = run(
        &f.0,
        &[
            &["-vv", "--progress", "never", "run", "./model.gguf"][..],
            &common,
        ]
        .concat(),
    );
    let stderr = String::from_utf8_lossy(&verbose.stderr);
    assert!(stderr.contains("Selected CPU source:"), "{stderr}");
    assert!(
        stderr.contains("Observed ") && stderr.contains("device(s) for routing"),
        "{stderr}"
    );
    assert!(stderr.contains("generated tokens"), "{stderr}");
    assert!(!stderr.contains("prefill"), "{stderr}");
    assert_eq!(passed(verbose), baseline);
    assert_eq!(
        passed(run(
            &f.0,
            &[&["run-cpu", "./model.gguf"][..], &common].concat()
        )),
        baseline
    );
    let yaml = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: tiny\n    target: qwen3.5-0.8b\n    source: ./model.gguf\nrun:\n  entrypoint: {model: tiny}\n  backend: cpu\n  device: [cpu:0]\n  headroom: '0'\n  task: {prompt: a, raw: true, max_tokens: 1, context: 8}\n";
    std::fs::write(f.0.join("run.yaml"), yaml).unwrap();
    assert_eq!(passed(run(&f.0, &["run", "run.yaml"])), baseline);
    std::fs::create_dir_all(f.0.join("config/abird/zerogravity")).unwrap();
    let absolute = yaml.replace("./model.gguf", f.0.join("model.gguf").to_str().unwrap())
        + "ui:\n  progress: never\n";

    std::fs::write(f.0.join("config/abird/zerogravity/config.yaml"), &absolute).unwrap();
    let implicit = run(&f.0, &["run"]);
    assert!(!String::from_utf8_lossy(&implicit.stderr).contains("prefill"));
    assert_eq!(passed(implicit), baseline);
    let profiles = f.0.join("config/abird/zerogravity/profiles");
    std::fs::create_dir(&profiles).unwrap();
    // Selected profile replaces the default and retains its own relative base.
    let profile = absolute.replace("max_tokens: 1", "max_tokens: 2");
    std::fs::write(profiles.join("laptop.yaml"), profile).unwrap();
    assert_eq!(
        passed(run(
            &f.0,
            &["--profile", "laptop", "run", "--max-tokens", "1", "--quiet"]
        )),
        baseline
    );
    let profile_dry = run(
        &f.0,
        &["run", "--profile", "laptop", "--dry-run", "--format=json"],
    );
    assert!(profile_dry.status.success(), "{:?}", profile_dry.stderr);
    let decision: serde_json::Value = serde_json::from_slice(&profile_dry.stdout).unwrap();
    assert_eq!(decision["max_tokens"], 2);
    let direct = run(&f.0, &["run", "./model.gguf", "--dry-run", "--format=json"]);
    assert!(direct.status.success(), "{:?}", direct.stderr);
    let decision: serde_json::Value = serde_json::from_slice(&direct.stdout).unwrap();
    assert_eq!(decision["max_tokens"], 1);
    let selected = run(
        &f.0,
        &[
            "--profile",
            "laptop",
            "run",
            "./model.gguf",
            "--max-tokens",
            "3",
            "--dry-run",
            "--format=json",
        ],
    );
    assert!(selected.status.success(), "{:?}", selected.stderr);
    let decision: serde_json::Value = serde_json::from_slice(&selected.stdout).unwrap();
    assert_eq!(decision["max_tokens"], 3);
    assert_eq!(decision["provenance"]["task.max_tokens"], "Cli");
    for args in [
        vec!["--profile", "../escape", "run"],
        vec!["--profile", "laptop", "run", "--config", "run.yaml"],
    ] {
        let failed = run(&f.0, &args);
        assert_eq!(failed.status.code(), Some(2));
    }

    let dry = run(
        &f.0,
        &[
            "run",
            "run.yaml",
            "--dry-run",
            "--audit-logits-jsonl",
            "not-created",
        ],
    );
    assert!(
        dry.status.success(),
        "{}",
        String::from_utf8_lossy(&dry.stderr)
    );
    assert!(!f.0.join("not-created").exists());
    assert!(!f.0.join("cache").exists());
    #[cfg(all(feature = "cache", target_os = "linux"))]
    cached(&f.0, &common, &baseline);
}
#[cfg(all(feature = "cache", target_os = "linux"))]
fn cached(root: &Path, common: &[&str], baseline: &[u8]) {
    use sha2::{Digest, Sha256};
    use std::sync::{Arc, atomic::AtomicBool};
    use zerogravity::cache::*;
    use zerogravity::resource_arbiter::*;
    let cache = root.join("cache/abird/zerogravity/models");
    let store =
        CacheStore::open(CacheOptions::new(Some(cache.clone()), 16 * 1024u64.pow(3))).unwrap();
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "test-cache").unwrap();
    arbiter
        .register(resource.clone(), 128 * 1024 * 1024)
        .unwrap();
    let memory = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "import").unwrap(),
    };
    let components = [
        ("model.gguf", FileRole::Weights),
        ("config.json", FileRole::Config),
        ("tokenizer.json", FileRole::Tokenizer),
    ];
    let identities = vec![
        (
            "hf:fixture/tiny@v1",
            ManifestSource::new("huggingface", "fixture/tiny", "v1").unwrap(),
        ),
        (
            "gh:fixture/tiny@v1",
            ManifestSource::new("github", "fixture/tiny", "v1").unwrap(),
        ),
    ];
    #[cfg(feature = "acquisition")]
    let identities = {
        let mut identities = identities;
        let bytes = std::fs::read(root.join("model.gguf")).unwrap();
        let spec = zerogravity::acquisition::object::HttpsObjectSpec::new(
            "https://fixture.invalid",
            "model.gguf",
            FileRole::Weights,
            bytes.len() as u64,
            format!("{:x}", Sha256::digest(&bytes)),
        )
        .unwrap();
        identities.push((
            "https://fixture.invalid/transient/model.yaml?secret=never-persisted",
            spec.source().clone(),
        ));
        let http = zerogravity::acquisition::object::HttpsObjectSpec::cached_locator(
            "http://fixture.invalid/transient/model.yaml?secret=never-persisted",
            "model.gguf",
            FileRole::Weights,
            bytes.len() as u64,
            &format!("{:x}", Sha256::digest(&bytes)),
        )
        .unwrap();
        assert_ne!(http.source(), spec.source());
        identities.push((
            "http://fixture.invalid/transient/model.yaml?secret=never-persisted",
            http.source().clone(),
        ));
        identities
    };
    for (index, (selector, source)) in identities.into_iter().enumerate() {
        let entries = components
            .iter()
            .map(|(name, role)| {
                let raw = std::fs::read(root.join(name)).unwrap();
                ManifestEntry::new(
                    *name,
                    format!("{:x}", Sha256::digest(&raw)),
                    raw.len() as u64,
                    *role,
                )
                .unwrap()
            })
            .collect();
        let manifest = Manifest::new(source, entries, &CacheLimits::default()).unwrap();
        let digest = manifest.digest().unwrap();
        let mut tx = store
            .begin(manifest, &memory, Arc::new(AtomicBool::new(false)))
            .unwrap();
        for (name, _) in components {
            tx.import_object(name, &mut std::fs::File::open(root.join(name)).unwrap())
                .unwrap();
        }
        let reference = match index {
            0 => "huggingface/fixture/tiny/v1".to_string(),
            1 => "github/fixture/tiny/v1".to_string(),
            _ => format!("tiny{index}"),
        };
        tx.commit(Some(&reference)).unwrap();
        let metadata = CacheMetadata::open(&cache).unwrap();
        let selected = metadata
            .select(
                &zerogravity::run_policy::SourceSelector::parse(selector).unwrap(),
                Some(&digest),
            )
            .unwrap();
        assert_eq!(selected.digest, digest);
        assert!(
            cache
                .join("snapshots")
                .join(&selected.digest)
                .join("files/config.json")
                .is_file()
        );
        assert!(
            metadata
                .select(
                    &zerogravity::run_policy::SourceSelector::parse("hf:other/source@v1").unwrap(),
                    Some(&digest),
                )
                .is_err()
        );
        #[cfg(feature = "acquisition")]
        if selector.starts_with("http") {
            for wrong in [
                "http://other.invalid/model",
                "https://user@fixture.invalid/model",
                "http://@fixture.invalid/model",
                "ftp://fixture.invalid/model",
            ] {
                assert!(
                    metadata
                        .select(
                            &zerogravity::run_policy::SourceSelector::Http(wrong.into()),
                            Some(&digest),
                        )
                        .is_err()
                );
            }
        }
        if index == 0 {
            let lease = store.open_snapshot(&digest).unwrap();
            let mut wrong = zerogravity::artifact::Artifact::open(&root.join("model.gguf"))
                .unwrap()
                .info;
            wrong.tensors[0].shape[0] += 1;
            let refused = zerogravity::prepare::PreparedQwen::from_snapshots_with_layout(
                &lease,
                "model.gguf",
                &lease,
                "config.json",
                &lease,
                "tokenizer.json",
                &memory,
                Some(&wrong),
            );
            assert!(matches!(refused, Err(zerogravity::Error::State(_))));
            assert_eq!(
                memory.arbiter.snapshot(&memory.resource).unwrap().reserved,
                0
            );
        }

        assert_eq!(
            passed(run(
                root,
                &[&["run", selector, "--snapshot", &digest][..], common].concat()
            )),
            baseline
        );
        assert_eq!(
            passed(run(root, &[&["run", &reference][..], common].concat())),
            baseline
        );
        if index == 0 {
            let explicit_digest = format!("sha256:{digest}");
            store
                .set_by_name_alias("local-fixture", &digest, false)
                .unwrap();
            for selector in [
                &digest,
                &explicit_digest,
                &"by-name:local-fixture".to_owned(),
                &"local-fixture".to_owned(),
            ] {
                assert_eq!(
                    passed(run(root, &[&["run", selector][..], common].concat())),
                    baseline
                );
                let inspected = run(root, &["model", "inspect", selector, "--format=json"]);
                assert!(inspected.status.success(), "{:?}", inspected.stderr);
                let report: serde_json::Value = serde_json::from_slice(&inspected.stdout).unwrap();
                assert_eq!(report["snapshot"], digest);
                assert_eq!(report["payload_verified"], false);
            }
            assert_eq!(
                passed(run(
                    root,
                    &[&["run", "local-fixture", "--pull"][..], common].concat()
                )),
                baseline
            );
            let verified = run(
                root,
                &["model", "verify", &explicit_digest, "--format=json"],
            );
            assert!(verified.status.success(), "{:?}", verified.stderr);
            let report: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
            assert_eq!(report["snapshot"], digest);
            assert_eq!(report["payload_verified"], true);
            let tree = cache_tree(&cache);
            let listed = run(root, &["model", "list", "--format=json"]);
            assert!(listed.status.success(), "{:?}", listed.stderr);
            let reports: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
            assert!(
                reports
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|row| row["snapshot"] == digest)
            );
            assert_eq!(cache_tree(&cache), tree);
            let diagnostic = run(root, &["doctor", "--format=json"]);
            assert!(diagnostic.status.success(), "{:?}", diagnostic.stderr);
            let report: serde_json::Value = serde_json::from_slice(&diagnostic.stdout).unwrap();
            let observed = report["checks"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["component"] == "cache")
                .unwrap();
            assert_eq!(observed["status"], "ok");
            assert!(observed["details"]["snapshots"].as_u64().unwrap() >= 1);
            assert_eq!(
                observed["details"]["configured_disk_limit_bytes"],
                16 * 1024u64.pow(3)
            );
            let quota = store.usage().unwrap();
            assert_eq!(observed["details"]["physical_charge_observed"], true);
            assert_eq!(observed["details"]["used_bytes"], quota.used_bytes);
            assert_eq!(observed["details"]["reserved_bytes"], quota.reserved_bytes);
            assert_eq!(observed["details"]["payload_verified"], false);
            assert_eq!(cache_tree(&cache), tree);
            let refused = run(root, &["model", "remove", &explicit_digest]);
            assert_eq!(refused.status.code(), Some(2));
            assert!(String::from_utf8_lossy(&refused.stderr).contains("--dry-run or --yes"));
            // The default pull shape is a catalogue reference, and the explicit
            // spelling selects the same immutable snapshot.
            let explicit = format!("catalog:{reference}");
            assert_eq!(
                passed(run(root, &[&["run", &explicit][..], common].concat())),
                baseline
            );
            assert_eq!(
                passed(run(
                    root,
                    &[&["run", &explicit, "--snapshot", &digest][..], common].concat()
                )),
                baseline
            );
        }
        let locks = std::fs::read_dir(cache.join("locks")).unwrap().count();
        let tree = cache_tree(&cache);
        let dry = run(
            root,
            &[
                &["run", selector, "--snapshot", &digest, "--dry-run"][..],
                common,
            ]
            .concat(),
        );
        assert!(
            dry.status.success(),
            "{}",
            String::from_utf8_lossy(&dry.stderr)
        );
        assert!(!String::from_utf8_lossy(&dry.stdout).contains("secret="));
        assert_eq!(
            std::fs::read_dir(cache.join("locks")).unwrap().count(),
            locks
        );
        // The complete cache tree stays byte-identical across a dry-run.
        assert_eq!(cache_tree(&cache), tree);
    }
    #[cfg(not(feature = "acquisition"))]
    {
        let refused = run(
            root,
            &[
                &["run", "http://fixture.invalid/model", "--dry-run"][..],
                common,
            ]
            .concat(),
        );
        assert!(!refused.status.success());
        assert!(
            String::from_utf8_lossy(&refused.stderr)
                .contains("requires compiled acquisition contract parser")
        );
    }
}

/// Deterministic content identity of the whole cache tree; dry-runs must
/// leave it byte-identical.
#[cfg(all(feature = "cache", target_os = "linux"))]
fn cache_tree(root: &Path) -> std::collections::BTreeMap<String, String> {
    use sha2::{Digest, Sha256};
    let mut tree = std::collections::BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                tree.insert(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    format!("{:x}", Sha256::digest(&bytes)),
                );
            }
        }
    }
    tree
}

#[cfg(all(feature = "cache", target_os = "linux"))]
fn cache_store(
    root: &Path,
) -> (
    zerogravity::cache::CacheStore,
    zerogravity::cache::TransferAdmission,
) {
    use zerogravity::cache::*;
    use zerogravity::resource_arbiter::*;
    let cache = root.join("cache/abird/zerogravity/models");
    let store = CacheStore::open(CacheOptions::new(Some(cache), 16 * 1024u64.pow(3))).unwrap();
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "test-cache").unwrap();
    arbiter
        .register(resource.clone(), 128 * 1024 * 1024)
        .unwrap();
    let memory = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "import").unwrap(),
    };
    (store, memory)
}

/// Commit one snapshot built from the fixture components plus optional extra
/// role-typed files; weights/config/tokenizer stay exactly one each so
/// select-level ambiguity, never the role error, is what surfaces.
#[cfg(all(feature = "cache", target_os = "linux"))]
fn commit_snapshot(
    root: &Path,
    store: &zerogravity::cache::CacheStore,
    memory: &zerogravity::cache::TransferAdmission,
    source: zerogravity::cache::ManifestSource,
    extra: &[(&str, zerogravity::cache::FileRole)],
    reference: Option<&str>,
) -> String {
    use sha2::{Digest, Sha256};
    use std::sync::{Arc, atomic::AtomicBool};
    use zerogravity::cache::*;
    let mut components = vec![
        ("model.gguf".to_string(), FileRole::Weights),
        ("config.json".to_string(), FileRole::Config),
        ("tokenizer.json".to_string(), FileRole::Tokenizer),
    ];
    for (name, role) in extra {
        components.push((name.to_string(), *role));
    }
    let entries = components
        .iter()
        .map(|(name, role)| {
            let raw = std::fs::read(root.join(name)).unwrap();
            ManifestEntry::new(
                name,
                format!("{:x}", Sha256::digest(&raw)),
                raw.len() as u64,
                *role,
            )
            .unwrap()
        })
        .collect();
    let manifest = Manifest::new(source, entries, &CacheLimits::default()).unwrap();
    let digest = manifest.digest().unwrap();
    let mut tx = store
        .begin(manifest, memory, Arc::new(AtomicBool::new(false)))
        .unwrap();
    for (name, _) in &components {
        tx.import_object(name, &mut std::fs::File::open(root.join(name)).unwrap())
            .unwrap();
    }
    tx.commit(reference).unwrap();
    digest
}

#[test]
fn run_dry_run_decision_trace_and_forced_refusal_stay_read_only() {
    let f = fixture_named("dry-trace");
    let yaml = "schema: zerogravity.pipeline.v2\nmodel:\n  - id: tiny\n    target: qwen3.5-0.8b\n    source: ./model.gguf\nrun:\n  entrypoint: {model: tiny}\n  backend: cpu\n  device: [cpu:0]\n  headroom: '0'\n  task: {prompt: a, raw: true, max_tokens: 1, context: 8}\n";
    std::fs::write(f.0.join("run.yaml"), yaml).unwrap();
    let dry = run(
        &f.0,
        &[
            &[
                "run",
                "run.yaml",
                "--dry-run",
                "--audit-logits-jsonl",
                "not-created",
            ][..],
            &COMMON,
        ]
        .concat(),
    );
    assert!(
        dry.status.success(),
        "{}",
        String::from_utf8_lossy(&dry.stderr)
    );
    assert!(!f.0.join("not-created").exists());
    assert!(!f.0.join("cache").exists());
    let trace: serde_json::Value =
        serde_json::from_slice(&dry.stdout).expect("decision trace JSON");
    assert_eq!(trace["schema"], "zerogravity.run.decision.v1");
    assert_eq!(trace["selected"], "cpu:0");
    assert!(trace["planning_bytes"].as_u64().is_some_and(|n| n > 0));
    assert_eq!(trace["dry_run"], true);
    assert_eq!(trace["reservation"], "none");
    assert!(
        trace["provenance"]
            .as_object()
            .is_some_and(|m| !m.is_empty())
    );

    // A forced refusal still prints the full exclusion trace before failing.
    let refused = run(
        &f.0,
        &["run", "run.yaml", "--dry-run", "--headroom", "100GiB"],
    );
    assert_eq!(refused.status.code(), Some(2));
    assert!(!f.0.join("not-created").exists());
    let refused_trace: serde_json::Value =
        serde_json::from_slice(&refused.stdout).expect("refusal trace JSON");
    assert_eq!(refused_trace["schema"], "zerogravity.run.decision.v1");
    assert_eq!(refused_trace["selected"], serde_json::Value::Null);
    assert_eq!(refused_trace["dry_run"], true);
    assert_eq!(refused_trace["reservation"], "none");
    assert!(
        refused_trace["excluded"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
    );
    assert!(
        refused_trace["excluded"].as_array().is_some_and(|a| {
            a.iter().any(|e| {
                e.as_str().is_some_and(|s| {
                    s.contains("insufficient currently observed memory after routing headroom")
                })
            })
        }),
        "{refused_trace}"
    );
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("unsupported: no eligible local device; see dry-run exclusions"),
        "{stderr}"
    );
    assert!(!f.0.join("cache").exists());

    // A device-specific check is the same original planner with an explicit
    // source and physical selector. It does not inherit a configured model pin.
    let direct_profile = yaml.replace("./model.gguf", "./not-the-selected-model.gguf")
        .replace("task: {prompt: a, raw: true, max_tokens: 1, context: 8}",
                 "task: {prompt: a, raw: true, max_tokens: 1, context: 8}\n  diagnostics: {audit_logits_jsonl: should-not-exist}");
    std::fs::write(f.0.join("other.yaml"), direct_profile).unwrap();
    let decision = run(
        &f.0,
        &[
            "device",
            "inspect",
            "cpu:0",
            "--model",
            "./model.gguf",
            "--config",
            "other.yaml",
            "--format=json",
        ],
    );
    assert!(
        decision.status.success(),
        "{}",
        String::from_utf8_lossy(&decision.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&decision.stdout).unwrap();
    assert_eq!(plan["schema"], trace["schema"]);
    assert_eq!(plan["selected"], "cpu:0");
    assert_eq!(plan["max_tokens"], 1);
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["reservation"], "none");
    let explicit = run(
        &f.0,
        &[
            "--config",
            "other.yaml",
            "run",
            "./model.gguf",
            "--dry-run",
            "--format=json",
        ],
    );
    assert!(explicit.status.success(), "{:?}", explicit.stderr);
    let other: serde_json::Value = serde_json::from_slice(&explicit.stdout).unwrap();
    assert_eq!(other["selected"], plan["selected"]);
    assert_eq!(other["planning_bytes"], plan["planning_bytes"]);
    assert!(!f.0.join("should-not-exist").exists());
    assert!(!f.0.join("cache").exists());
}

#[cfg(all(feature = "cache", target_os = "linux"))]
#[test]
fn positional_yaml_named_catalog_reference_runs_end_to_end() {
    let f = fixture_named("yaml-ref");
    let baseline = passed(run(&f.0, &[&["run", "./model.gguf"][..], &COMMON].concat()));
    let (store, memory) = cache_store(&f.0);
    let reference = "huggingface/fixture/tiny/v1.yaml";
    let digest = commit_snapshot(
        &f.0,
        &store,
        &memory,
        zerogravity::cache::ManifestSource::new("huggingface", "fixture/tiny", "v1.yaml").unwrap(),
        &[],
        Some(reference),
    );
    // A `.yaml`-final catalog revision is a model source in every spelling;
    // scheme precedence beats the local run-document suffix classification.
    assert_eq!(
        passed(run(&f.0, &[&["run", reference][..], &COMMON].concat())),
        baseline
    );
    assert_eq!(
        passed(run(
            &f.0,
            &[&["run", "hf:fixture/tiny@v1.yaml"][..], &COMMON].concat()
        )),
        baseline
    );
    assert_eq!(
        passed(run(
            &f.0,
            &[
                &[
                    "run",
                    "catalog:huggingface/fixture/tiny/v1.yaml",
                    "--snapshot",
                    &digest
                ][..],
                &COMMON
            ]
            .concat()
        )),
        baseline
    );
}

#[cfg(all(feature = "cache", target_os = "linux"))]
#[test]
fn provider_variant_ambiguity_refuses_exactly_without_silent_pick() {
    let f = fixture_named("ambiguity");
    let baseline = passed(run(&f.0, &[&["run", "./model.gguf"][..], &COMMON].concat()));
    let (store, memory) = cache_store(&f.0);
    let cache = f.0.join("cache/abird/zerogravity/models");
    std::fs::write(f.0.join("index.bin"), b"index-fixture").unwrap();
    let source =
        zerogravity::cache::ManifestSource::new("huggingface", "fixture/tiny", "v1.yaml").unwrap();
    // One acquired variant: provider selection without --snapshot runs.
    let first = commit_snapshot(
        &f.0,
        &store,
        &memory,
        source.clone(),
        &[],
        Some("huggingface/fixture/tiny/v1.yaml"),
    );
    assert_eq!(
        passed(run(
            &f.0,
            &[&["run", "hf:fixture/tiny@v1.yaml"][..], &COMMON].concat()
        )),
        baseline
    );
    // Zero acquired variants: exact acquisition guidance, cache unchanged.
    let tree = cache_tree(&cache);
    let refused = run(
        &f.0,
        &[&["run", "hf:other/source@v9"][..], &COMMON].concat(),
    );
    assert_eq!(refused.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains(
            "provider source is not acquired; use models/pull then an explicit --snapshot or catalog reference"
        ),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert_eq!(cache_tree(&cache), tree);
    // A second snapshot of the same provider identity with a distinct
    // manifest digest makes the provider selector ambiguous; no silent pick.
    let second = commit_snapshot(
        &f.0,
        &store,
        &memory,
        source,
        &[("index.bin", zerogravity::cache::FileRole::Index)],
        Some("conflicting-name"),
    );
    assert_ne!(first, second);
    store
        .set_by_name_alias("conflicting-name", &first, false)
        .unwrap();
    let metadata = zerogravity::cache::CacheMetadata::open(&cache).unwrap();
    assert!(matches!(
        metadata.named("conflicting-name"),
        Err(zerogravity::Error::Malformed(_))
    ));
    let conflict = run(
        &f.0,
        &[&["run", "conflicting-name", "--pull"][..], &COMMON].concat(),
    );
    assert_eq!(conflict.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&conflict.stderr).contains("ambiguous bare model name"));
    for selector in ["catalog:conflicting-name", "by-name:conflicting-name"] {
        let inspected = run(&f.0, &["model", "inspect", selector, "--format=json"]);
        assert!(inspected.status.success(), "{:?}", inspected.stderr);
        let report: serde_json::Value = serde_json::from_slice(&inspected.stdout).unwrap();
        assert_eq!(
            report["snapshot"],
            if selector.starts_with("catalog:") {
                second.as_str()
            } else {
                first.as_str()
            }
        );
    }
    let tree = cache_tree(&cache);
    let refused = run(
        &f.0,
        &[&["run", "hf:fixture/tiny@v1.yaml"][..], &COMMON].concat(),
    );
    assert_eq!(refused.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains(
            "provider source matches 2 acquired snapshots; an explicit --snapshot or catalog reference is required"
        ),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert_eq!(cache_tree(&cache), tree);
    // Each explicit snapshot disambiguates to the same baseline output.
    for digest in [&first, &second] {
        assert_eq!(
            passed(run(
                &f.0,
                &[
                    &["run", "hf:fixture/tiny@v1.yaml", "--snapshot", digest][..],
                    &COMMON
                ]
                .concat()
            )),
            baseline
        );
    }
    // The 1:1 catalog reference stays exact by construction.
    assert_eq!(
        passed(run(
            &f.0,
            &[
                &["run", "catalog:huggingface/fixture/tiny/v1.yaml"][..],
                &COMMON
            ]
            .concat()
        )),
        baseline
    );
    // A missing catalog reference name reports acquisition guidance, not a
    // raw filesystem error.
    let refused = run(
        &f.0,
        &[&["run", "huggingface/fixture/tiny/absent"][..], &COMMON].concat(),
    );
    assert_eq!(refused.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains(
            "catalog reference huggingface/fixture/tiny/absent is not acquired; use models/pull then run the printed reference"
        ),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    // A snapshot that does not match the requested identity keeps its error.
    let third = commit_snapshot(
        &f.0,
        &store,
        &memory,
        zerogravity::cache::ManifestSource::new("github", "fixture/tiny", "v1").unwrap(),
        &[],
        None,
    );
    let refused = run(
        &f.0,
        &[
            &["run", "hf:fixture/tiny@v1.yaml", "--snapshot", &third][..],
            &COMMON,
        ]
        .concat(),
    );
    assert_eq!(refused.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&refused.stderr)
            .contains("explicit snapshot does not match requested provider/content identity"),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
}

#[cfg(all(feature = "serve", feature = "cache", target_os = "linux"))]
#[test]
fn canonical_serve_refuses_incompatible_routes_before_cache_effects() {
    let f = fixture_named("serve-preflight");
    // Preflight refuses remote bind, duplicate aliases, and conflicting
    // acquisition policy without initializing a cache or touching the source.
    let cache = f.0.join("cache/abird/zerogravity/models");
    for args in [
        vec!["serve", "not-cached", "--pull", "--listen", "0.0.0.0:8000"],
        vec![
            "serve",
            "served-fixture",
            "--model",
            "served-fixture=by-name:served-fixture",
        ],
        vec!["serve", "qwen3.5-2b", "qwen3.8-27b", "--pull"],
        vec!["serve", "./model.gguf", "qwen3.5-2b", "--pull"],
        vec!["--offline", "serve", "served-fixture", "--pull"],
        vec![
            "serve",
            "./model.gguf",
            "-d",
            "gpu:9999",
            "--backend",
            "cpu",
        ],
    ] {
        let output = run(&f.0, &args);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{args:?}: {:?}",
            output.stderr
        );
        assert!(!cache.exists());
    }
}

#[cfg(all(feature = "serve", feature = "cache", target_os = "linux"))]
#[test]
#[ignore = "existing approved local Qwen 0.8B GGUF and tokenizer; no download"]
fn canonical_serve_profile_and_cached_aliases_use_one_listener_owner() {
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::{SocketAddr, TcpStream},
        process::{Child, Stdio},
        time::{Duration, Instant},
    };
    let root = std::env::temp_dir().join(format!("zg-unified-serve-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let f = Fixture(root);
    let original = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models/qwen35-0.8b");
    let (store, memory) = cache_store(&f.0);
    let digest = commit_snapshot(
        &original,
        &store,
        &memory,
        zerogravity::cache::ManifestSource::new("huggingface", "fixture/serve", "v1").unwrap(),
        &[],
        None,
    );
    store
        .set_by_name_alias("served-fixture", &digest, false)
        .unwrap();
    drop(store);
    let profiles = f.0.join("config/abird/zerogravity/profiles");
    std::fs::create_dir_all(&profiles).unwrap();
    std::fs::write(
        profiles.join("local.yaml"),
        r#"schema: zerogravity.pipeline.v2
model:
  - id: configured
    target: qwen3.5-0.8b
    source: by-name:served-fixture
run:
  backend: cpu
  device: [cpu:0]
  headroom: '0'
  task: {context: 64, max_tokens: 2}
serve:
  listen: 127.0.0.1:0
  workers: 1
  active_per_worker: 1
  max_pending_requests: 4
  host_mib: 8192
"#,
    )
    .unwrap();

    struct Owned(Child);
    impl Drop for Owned {
        fn drop(&mut self) {
            if self.0.try_wait().ok().flatten().is_none() {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
    }
    let log = f.0.join("serve.log");
    let mut child = Owned(
        Command::new(env!("CARGO_BIN_EXE_zerogravity"))
            .args([
                "--profile",
                "local",
                "serve",
                "served-fixture",
                "--model",
                &format!("fast=sha256:{digest}"),
                "--pull",
            ])
            .current_dir(&f.0)
            .env("HOME", &f.0)
            .env("XDG_CONFIG_HOME", f.0.join("config"))
            .env("XDG_CACHE_HOME", f.0.join("cache"))
            .env("HF_TOKEN", "invalid\nnever-used")
            .env_remove("ZEROGRAVITY_API_KEY")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&log).unwrap())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(90);
    let address: SocketAddr = loop {
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        if let Some(address) = text.lines().find_map(|line| {
            line.split_once("API http://")
                .and_then(|(_, v)| v.strip_suffix("/v1"))
        }) {
            break address.parse().unwrap();
        }
        assert!(
            child.0.try_wait().unwrap().is_none() && Instant::now() < deadline,
            "server did not start: {text}"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    let stream = TcpStream::connect_timeout(&address, Duration::from_secs(5)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(60)))
        .unwrap();
    let mut connection = BufReader::new(stream);
    let mut http = |path: &str, body: Option<&str>| {
        let request = if let Some(body) = body {
            format!(
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
        } else {
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n")
        };
        connection.get_mut().write_all(request.as_bytes()).unwrap();
        let mut status = String::new();
        connection.read_line(&mut status).unwrap();
        assert!(status.starts_with("HTTP/1.1 200"), "{status}");
        let mut length = None;
        loop {
            let mut header = String::new();
            assert!(connection.read_line(&mut header).unwrap() > 0);
            if header == "\r\n" {
                break;
            }
            if let Some((name, value)) = header.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
                assert!(
                    !name.eq_ignore_ascii_case("connection")
                        || !value.trim().eq_ignore_ascii_case("close"),
                    "listener unexpectedly closed a reusable connection"
                );
            }
        }
        let length = length.expect("fixed JSON response length");
        assert!(length <= 1024 * 1024);
        let mut body = vec![0; length];
        connection.read_exact(&mut body).unwrap();
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()
    };
    let models = http("/v1/models", None);
    let mut ids = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, ["fast", "served-fixture"]);
    let outputs = ids.into_iter().map(|alias| {
        let request = serde_json::json!({"model":alias,"prompt":"Continue: 1, 2, 3,","max_tokens":2,"stream":false}).to_string();
        let response = http("/v1/completions", Some(&request));
        assert!(response["choices"][0]["text"].is_string(), "{response}");
        assert!(response["usage"]["completion_tokens"].as_u64().unwrap() > 0);
        response["choices"][0]["text"].clone()
    }).collect::<Vec<_>>();
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(http("/v1/models", None), models);
    drop(connection);
    // Aliases sharing the original snapshot must appear in one startup row.
    let text = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        text.lines()
            .filter(|line| line.contains("Model ")
                && line.contains("fast")
                && line.contains("served-fixture"))
            .count(),
        1,
        "{text}"
    );
    let signal = Command::new("python3")
        .args([
            "-c",
            "import os,signal,sys; os.kill(int(sys.argv[1]), signal.SIGTERM)",
            &child.0.id().to_string(),
        ])
        .status()
        .unwrap();
    assert!(signal.success());
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(
                status.success(),
                "{}",
                std::fs::read_to_string(&log).unwrap()
            );
            break;
        }
        assert!(Instant::now() < deadline, "server shutdown timed out");
        std::thread::sleep(Duration::from_millis(25));
    }
    let text = std::fs::read_to_string(&log).unwrap();
    assert!(
        text.contains("shutdown complete; HTTP handlers and inference workers joined"),
        "{text}"
    );
    assert!(TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_err());

    #[cfg(feature = "tui")]
    {
        // Reuse the exact acquired fixture and original HTTP output. The TUI
        // owns one foreground runtime through load, serve, stop and reload.
        let profile = profiles.join("local.yaml");
        let raw = std::fs::read_to_string(&profile).unwrap().replace(
            "task: {context: 64, max_tokens: 2}",
            "task: {context: 64, max_tokens: 2, raw: true}",
        );
        std::fs::write(&profile, raw).unwrap();
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tui_shell.py");
        let status = Command::new("python3")
            .args(["-B"])
            .arg(fixture)
            .arg(env!("CARGO_BIN_EXE_zerogravity"))
            .arg(&f.0)
            .arg(outputs[0].as_str().unwrap())
            .env("ZG_TEST_WEB", if cfg!(feature = "web") { "1" } else { "0" })
            .status()
            .unwrap();
        assert!(status.success(), "cached native control lifecycle failed");
    }
}
