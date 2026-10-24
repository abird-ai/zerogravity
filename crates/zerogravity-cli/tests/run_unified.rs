//! Synthetic genuine GGUF/tokenizer input to the original CPU engine. No
//! learned checkpoint, device, download or foreign inference runtime.
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
fn fixture() -> Fixture {
    let root = std::env::temp_dir().join(format!("zg-unified-{}", std::process::id()));
    assert!(!root.exists());
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("model.gguf"), gguf()).unwrap();
    std::fs::write(root.join("config.json"),serde_json::to_vec(&json!({"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],"text_config":{"model_type":"qwen3_5_text","hidden_size":1024,"head_dim":64,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":4,"max_position_embeddings":32768,"num_hidden_layers":24,"layer_types":vec!["full_attention";24],"full_attention_interval":1,"linear_conv_kernel_dim":4,"linear_num_key_heads":1,"linear_num_value_heads":1}})).unwrap()).unwrap();
    std::fs::write(root.join("tokenizer.json"),serde_json::to_vec(&json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":1,"content":"<|im_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"<|im_start|>":1,"<|im_end|>":2,"[UNK]":3},"unk_token":"[UNK]"}})).unwrap()).unwrap();
    Fixture(root)
}
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
    let baseline = passed(run(&f.0, &[&["run", "./model.gguf"][..], &common].concat()));
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
    let absolute = yaml.replace("./model.gguf", f.0.join("model.gguf").to_str().unwrap());
    std::fs::write(f.0.join("config/abird/zerogravity/config.yaml"), absolute).unwrap();
    assert_eq!(passed(run(&f.0, &["run"])), baseline);
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
