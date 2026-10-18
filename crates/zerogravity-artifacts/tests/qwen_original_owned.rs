//! Original Qwen3.8 dense-text geometry bound to exact owned shard bytes.
#![cfg(all(feature = "cache", target_os = "linux"))]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use zerogravity_artifacts::SourceWeights;
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::model::qwen::QwenTextConfig;
use zerogravity_artifacts::model::qwen_owned::QwenOriginalText;
use zerogravity_artifacts::source_weights::{ExpectedSourceShard, SourceOpenLimits};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
const A: &str = "model-00001-of-00002.safetensors";
const B: &str = "model-00002-of-00002.safetensors";

struct Fixture {
    directory: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let directory = std::env::temp_dir().join(format!(
            "zg-qwen-owned-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        Self { directory }
    }
    fn create(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.directory.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn config() -> Vec<u8> {
    let layers = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ];
    serde_json::to_vec(&json!({
        "architectures":["Qwen3_5ForConditionalGeneration"],
        "model_type":"qwen3_5",
        "tie_word_embeddings":false,
        "text_config":{
            "model_type":"qwen3_5_text",
            "num_hidden_layers":4,
            "layer_types":layers,
            "hidden_size":4,
            "num_attention_heads":2,
            "num_key_value_heads":1,
            "head_dim":4,
            "vocab_size":16,
            "max_position_embeddings":1024,
            "full_attention_interval":4,
            "linear_conv_kernel_dim":2,
            "linear_num_key_heads":1,
            "linear_num_value_heads":2,
            "linear_key_head_dim":2,
            "linear_value_head_dim":2,
            "intermediate_size":8,
            "hidden_act":"silu",
            "attention_bias":false,
            "attention_dropout":0.0,
            "attn_output_gate":true,
            "output_gate_type":"swish",
            "rms_norm_eps":0.000001,
            "tie_word_embeddings":false,
            "partial_rotary_factor":0.5,
            "rope_parameters":{
                "rope_type":"default",
                "rope_theta":10000.0,
                "partial_rotary_factor":0.5
            }
        }
    }))
    .unwrap()
}
fn bytes_for(dtype: &str, shape: &[u64]) -> Vec<u8> {
    let count = shape.iter().product::<u64>() as usize;
    let mut out = Vec::new();
    match dtype {
        "F32" => {
            for _ in 0..count {
                out.extend(1.0f32.to_le_bytes())
            }
        }
        "F16" => {
            for _ in 0..count {
                out.extend(0x3c00u16.to_le_bytes())
            }
        }
        "BF16" => {
            for _ in 0..count {
                out.extend(0x3f80u16.to_le_bytes())
            }
        }
        _ => panic!("unregistered fixture dtype"),
    }
    out
}
struct Entry {
    name: String,
    shape: Vec<u64>,
    dtype: &'static str,
}
#[derive(Clone, Copy)]
enum Variation {
    Normal,
    WrongMatrixShape,
    MissingHead,
    ExtraTextTensor,
}
fn build_entries(raw_config: &[u8], variation: Variation) -> Vec<Entry> {
    let cfg = QwenTextConfig::from_json(raw_config).unwrap();
    let mut specs = cfg.tensor_specs().unwrap();
    assert!(specs.len() > 50);
    if matches!(variation, Variation::MissingHead) {
        specs.retain(|s| s.name != "lm_head.weight");
    }
    if matches!(variation, Variation::WrongMatrixShape) {
        let target = specs
            .iter_mut()
            .find(|s| s.name.ends_with(".linear_attn.in_proj_qkv.weight"))
            .unwrap();
        target.shape[1] += 1;
    }
    let mut entries = specs
        .into_iter()
        .map(|spec| {
            let dtype = if spec.name.ends_with("self_attn.q_proj.weight") {
                "BF16"
            } else if spec.name.ends_with("mlp.gate_proj.weight") {
                "F16"
            } else {
                "F32"
            };
            Entry {
                name: spec.name,
                shape: spec.shape,
                dtype,
            }
        })
        .collect::<Vec<_>>();
    if matches!(variation, Variation::ExtraTextTensor) {
        entries.push(Entry {
            name: "model.language_model.layers.0.unregistered_timestep.weight".into(),
            shape: vec![2],
            dtype: "F32",
        });
    }
    // Other components from multimodal/Qwen MTP may coexist in the original
    // index but this ordinary text binder must not promote them.
    entries.push(Entry {
        name: "model.visual.patch_embedding.weight".into(),
        shape: vec![2],
        dtype: "F32",
    });
    entries.push(Entry {
        name: "mtp.unqualified.weight".into(),
        shape: vec![2],
        dtype: "F16",
    });
    entries
}
fn safetensors(entries: &[Entry]) -> (Vec<u8>, u64) {
    assert!(!entries.is_empty());
    let mut names = BTreeMap::new();
    let mut data = Vec::new();
    for entry in entries {
        let bytes = bytes_for(entry.dtype, &entry.shape);
        let start = data.len() as u64;
        data.extend_from_slice(&bytes);
        names.insert(
            entry.name.clone(),
            json!({
                "dtype":entry.dtype,
                "shape":entry.shape,
                "data_offsets":[start,data.len() as u64]
            }),
        );
    }
    let header = serde_json::to_vec(&names).unwrap();
    let payload = data.len() as u64;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend(header);
    bytes.extend(data);
    (bytes, payload)
}
struct SourceFixture {
    _fixture: Fixture,
    config: Vec<u8>,
    index: Vec<u8>,
    original: SourceWeights,
    admission: TransferAdmission,
    key: PhysicalDeviceKey,
}
impl SourceFixture {
    fn new(variation: Variation) -> Self {
        let fixture = Fixture::new();
        let config = config();
        let entries = build_entries(&config, variation);
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut weight_map = BTreeMap::new();
        for (i, entry) in entries.into_iter().enumerate() {
            if i.is_multiple_of(2) {
                weight_map.insert(entry.name.clone(), A);
                first.push(entry);
            } else {
                weight_map.insert(entry.name.clone(), B);
                second.push(entry);
            }
        }
        let (raw_a, payload_a) = safetensors(&first);
        let (raw_b, payload_b) = safetensors(&second);
        fixture.create(A, &raw_a);
        fixture.create(B, &raw_b);
        let index = serde_json::to_vec(&json!({
            "metadata":{"total_size":payload_a+payload_b},
            "weight_map":weight_map
        }))
        .unwrap();
        let arbiter = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("test-host", "qwen-owned").unwrap();
        arbiter.register(key.clone(), 64 * 1024 * 1024).unwrap();
        let admission = TransferAdmission {
            arbiter,
            resource: key.clone(),
            owner: ResourceOwner::new("qwen", "original").unwrap(),
        };
        let shards = vec![
            ExpectedSourceShard::new(
                A.into(),
                Artifact::from_file(File::open(fixture.directory.join(A)).unwrap(), Path::new(A))
                    .unwrap(),
                raw_a.len() as u64,
                hash(&raw_a),
            )
            .unwrap(),
            ExpectedSourceShard::new(
                B.into(),
                Artifact::from_file(File::open(fixture.directory.join(B)).unwrap(), Path::new(B))
                    .unwrap(),
                raw_b.len() as u64,
                hash(&raw_b),
            )
            .unwrap(),
        ];
        let original = SourceWeights::prepare_verified_shards(
            &index,
            &hash(&index),
            shards,
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
        Self {
            _fixture: fixture,
            config,
            index,
            original,
            admission,
            key,
        }
    }
    fn bind(&self) -> zerogravity_artifacts::Result<QwenOriginalText<'_>> {
        QwenOriginalText::bind(
            &self.original,
            &self.config,
            &hash(&self.config),
            &hash(&self.index),
        )
    }
}
#[test]
fn original_small_hybrid_all_text_tensor_headers_bind_to_the_same_owned_index() {
    let fixture = SourceFixture::new(Variation::Normal);
    let bound = fixture.bind().unwrap();
    assert_eq!(
        bound.text_tensor_count(),
        QwenTextConfig::from_json(&fixture.config)
            .unwrap()
            .tensor_specs()
            .unwrap()
            .len()
    );
    assert!(bound.text_tensor_count() > 50);
    assert_eq!(bound.other_tensor_count(), 2);
    assert_eq!(bound.source_identity(), fixture.original.source_identity());
    assert_eq!(bound.verified_index_sha256(), hash(&fixture.index));
    assert_eq!(bound.config_sha256(), hash(&fixture.config));
    let geometry = bound.config().geometry();
    assert_eq!(geometry.hidden_size, 4);
    assert_eq!(geometry.layers.len(), 4);
    assert_eq!(
        (geometry.recurrent_key_heads, geometry.recurrent_value_heads),
        (Some(1), Some(2))
    );
    assert!(bound.numerical_state_bytes(16).unwrap().total().unwrap() > 0);
    assert!(bound.numerical_state_bytes(1025).is_err());
    let embedding = bound.embedding_region(0).unwrap();
    assert_eq!(
        embedding.source_identity,
        fixture.original.source_identity()
    );
    assert_eq!(embedding.width, 4);
    assert_eq!(embedding.vocabulary, 16);
    assert_eq!(embedding.packed_weights.len(), 16 * 4 * 4);
    assert!(bound.embedding_region(16).is_err());
    let q = bound
        .text_tensor("model.language_model.layers.3.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q.kind, QuantKind::Bf16);
    assert_eq!(q.shape, &[16, 4]);
    let gate = bound
        .text_tensor("model.language_model.layers.0.mlp.gate_proj.weight")
        .unwrap();
    assert_eq!(gate.kind, QuantKind::F16);
    assert_eq!(gate.shape, &[8, 4]);
    assert!(
        bound
            .text_tensor("model.visual.patch_embedding.weight")
            .is_err()
    );
    assert!(bound.text_tensor("mtp.unqualified.weight").is_err());
    assert!(
        fixture
            .admission
            .arbiter
            .snapshot(&fixture.key)
            .unwrap()
            .reserved
            > 0
    );
    drop(bound);
    drop(fixture);
}
#[test]
fn original_qwen_binding_refuses_wrong_missing_extra_shapes_and_source_hashes() {
    for variation in [
        Variation::WrongMatrixShape,
        Variation::MissingHead,
        Variation::ExtraTextTensor,
    ] {
        let fixture = SourceFixture::new(variation);
        assert!(fixture.bind().is_err());
    }
    let fixture = SourceFixture::new(Variation::Normal);
    assert!(
        QwenOriginalText::bind(
            &fixture.original,
            &fixture.config,
            &"0".repeat(64),
            &hash(&fixture.index)
        )
        .is_err()
    );
    assert!(
        QwenOriginalText::bind(
            &fixture.original,
            &fixture.config,
            &hash(&fixture.config),
            &"0".repeat(64)
        )
        .is_err()
    );
    let mut bad = serde_json::from_slice::<Value>(&fixture.config).unwrap();
    bad["text_config"]["hidden_act"] = json!("relu");
    let unsupported = serde_json::to_vec(&bad).unwrap();
    assert!(
        QwenOriginalText::bind(
            &fixture.original,
            &unsupported,
            &hash(&unsupported),
            &hash(&fixture.index)
        )
        .is_err()
    );
    assert!(
        QwenOriginalText::bind(
            &fixture.original,
            &fixture.config,
            "bad digest",
            &hash(&fixture.index)
        )
        .is_err()
    );
}
