//! End-to-end source-only Qwen original HF owned-shard -> original
//! Qwen text binder -> the existing CPU projection, embedding, vector
//! operator. NO trained checkpoint, tokenizer, full Qwen forward or GPU.
#![cfg(all(feature = "cpu", feature = "cache", target_os = "linux"))]

use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use zerogravity::artifact::Artifact;
use zerogravity::cache::TransferAdmission;
use zerogravity::model::qwen::QwenTextConfig;
use zerogravity::original_qwen_cpu::QwenOriginalCpu;
use zerogravity::quant::QuantKind;
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
use zerogravity_artifacts::SourceWeights;
use zerogravity_artifacts::source_weights::{ExpectedSourceShard, SourceOpenLimits};

struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let root = std::env::temp_dir().join(format!(
            "zg-hf-original-operators-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self { root }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn config() -> Vec<u8> {
    let schedule = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ];
    serde_json::to_vec(&json!({
        "model_type":"qwen3_5",
        "architectures":["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings":false,
        "text_config":{
            "model_type":"qwen3_5_text",
            "num_hidden_layers":4,"layer_types":schedule,
            "hidden_size":4,"head_dim":4,
            "num_attention_heads":2,"num_key_value_heads":1,
            "vocab_size":16,"max_position_embeddings":1024,
            "full_attention_interval":4,"linear_conv_kernel_dim":2,
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":2,"linear_value_head_dim":2,
            "intermediate_size":8,"hidden_act":"silu",
            "attention_bias":false,"attention_dropout":0.0,
            "attn_output_gate":true,"output_gate_type":"swish",
            "rms_norm_eps":0.000001,"tie_word_embeddings":false,
            "partial_rotary_factor":0.5,
            "rope_parameters":{"rope_type":"default","partial_rotary_factor":0.5,
                "rope_theta":10000.0}
        }
    }))
    .unwrap()
}
fn loaded(
    fixture: &Fixture,
) -> (
    SourceWeights,
    Vec<u8>,
    Vec<u8>,
    TransferAdmission,
    PhysicalDeviceKey,
) {
    let cfg = config();
    let spec = QwenTextConfig::from_json(&cfg)
        .unwrap()
        .tensor_specs()
        .unwrap();
    assert!(spec.len() > 50);
    let mut header = BTreeMap::new();
    let mut payload = Vec::new();
    let mut map = BTreeMap::new();
    for t in &spec {
        let kind = if t.name.ends_with("mlp.gate_proj.weight") {
            "F16"
        } else if t.name.ends_with("self_attn.q_proj.weight") {
            "BF16"
        } else {
            "F32"
        };
        let shape_count = t.shape.iter().product::<u64>() as usize;
        let start = payload.len();
        for _ in 0..shape_count {
            match kind {
                "F32" => payload.extend_from_slice(&1.0f32.to_le_bytes()),
                "F16" => payload.extend_from_slice(&0x3c00u16.to_le_bytes()),
                "BF16" => payload.extend_from_slice(&0x3f80u16.to_le_bytes()),
                _ => unreachable!(),
            }
        }
        header.insert(
            t.name.clone(),
            json!({
                "dtype":kind,"shape":t.shape,"data_offsets":[start,payload.len()]
            }),
        );
        map.insert(t.name.clone(), "single.safetensors");
    }
    let original_payload_bytes = payload.len() as u64;
    let raw_header = serde_json::to_vec(&header).unwrap();
    let mut source = Vec::new();
    source.extend_from_slice(&(raw_header.len() as u64).to_le_bytes());
    source.extend(raw_header);
    source.extend(payload);
    let index = serde_json::to_vec(&json!({
        "metadata":{"total_size":original_payload_bytes},
        "weight_map":map
    }))
    .unwrap();
    let path = fixture.root.join("single.safetensors");
    fs::write(&path, &source).unwrap();
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("single.safetensors")).unwrap();
    let shard = ExpectedSourceShard::new(
        "single.safetensors".into(),
        artifact,
        source.len() as u64,
        sha(&source),
    )
    .unwrap();
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("test-host", "qwen-original-ops").unwrap();
    arbiter.register(key.clone(), 64 * 1024 * 1024).unwrap();
    let admission = TransferAdmission {
        arbiter,
        resource: key.clone(),
        owner: ResourceOwner::new("qwen", "hf-ops").unwrap(),
    };
    let weights = SourceWeights::prepare_verified_shards(
        &index,
        &sha(&index),
        vec![shard],
        SourceOpenLimits {
            max_owned_bytes: 8 * 1024 * 1024,
            max_metadata_bytes: 32 * 1024 * 1024,
            max_shards: 2,
            max_tensors: 200,
        },
        &admission,
    )
    .unwrap()
    .materialize()
    .unwrap();
    (weights, cfg, index, admission, key)
}
#[test]
fn original_borrowed_tensor_source_uses_one_owner_and_existing_cpu_value_operators() {
    let fixture = Fixture::new();
    let (source, config, index, admission, key) = loaded(&fixture);
    let retained = admission.arbiter.snapshot(&key).unwrap().reserved;
    let bridge = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
    assert_eq!(bridge.binding().source_identity(), source.source_identity());
    assert_eq!(
        bridge.binding().text_tensor_count(),
        QwenTextConfig::from_json(&config)
            .unwrap()
            .tensor_specs()
            .unwrap()
            .len()
    );
    assert_eq!(bridge.embed(0).unwrap(), [1., 1., 1., 1.]);
    assert_eq!(bridge.embed(15).unwrap(), [1., 1., 1., 1.]);
    assert!(bridge.embed(16).is_err());
    assert_eq!(
        bridge.vector("model.language_model.norm.weight").unwrap(),
        [1., 1., 1., 1.]
    );
    let activation = [1.0, 2.0, 3.0, 4.0];
    let names = [
        (
            "model.language_model.layers.3.self_attn.k_proj.weight",
            QuantKind::F32,
            4,
        ),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            QuantKind::Bf16,
            16,
        ),
        (
            "model.language_model.layers.0.mlp.gate_proj.weight",
            QuantKind::F16,
            8,
        ),
    ];
    for (name, dtype, rows) in names {
        assert_eq!(bridge.binding().text_tensor(name).unwrap().kind, dtype);
        assert_eq!(
            bridge.project(name, &activation, None).unwrap(),
            vec![10.; rows]
        );
    }
    assert!(
        bridge
            .project("model.language_model.norm.weight", &activation, None)
            .is_err()
    );
    assert!(bridge.project("lm_head.weight", &[0.; 16], None).is_err());
    assert!(
        bridge
            .vector("model.language_model.layers.0.mlp.gate_proj.weight")
            .is_err()
    );
    assert!(bridge.binding().text_tensor("unregistered").is_err());
    drop(bridge);
    assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, retained);
    drop(source);
    assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn original_binding_cannot_override_original_source_config_or_index_pin() {
    let fixture = Fixture::new();
    let (source, config, index, admission, key) = loaded(&fixture);
    assert!(QwenOriginalCpu::bind(&source, &config, &"0".repeat(64), &sha(&index)).is_err());
    assert!(QwenOriginalCpu::bind(&source, &config, &sha(&config), &"0".repeat(64)).is_err());
    assert!(QwenOriginalCpu::bind(&source, &config, "invalid", &sha(&index)).is_err());
    let mut changed = serde_json::from_slice::<serde_json::Value>(&config).unwrap();
    changed["text_config"]["head_dim"] = json!(8);
    let changed = serde_json::to_vec(&changed).unwrap();
    assert!(QwenOriginalCpu::bind(&source, &changed, &sha(&changed), &sha(&index)).is_err());
    assert!(admission.arbiter.snapshot(&key).unwrap().reserved > 0);
}
