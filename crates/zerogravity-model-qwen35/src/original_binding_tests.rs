//! Synthetic verified original sources only; no trained arithmetic or GPU.

use super::*;
use crate::{LoadedQwenModel, StateSpec};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use zerogravity_artifacts::SourceWeights;
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::model::qwen::QwenTextConfig;
use zerogravity_artifacts::source_weights::{ExpectedSourceShard, SourceOpenLimits};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-hf-model-binding-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn config(tied: bool) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings":tied,
        "text_config":{
            "model_type":"qwen3_5_text","num_hidden_layers":4,
            "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
            "hidden_size":4,"head_dim":4,"num_attention_heads":2,
            "num_key_value_heads":1,"vocab_size":16,"max_position_embeddings":1024,
            "full_attention_interval":4,"linear_conv_kernel_dim":4,
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":2,"linear_value_head_dim":2,
            "intermediate_size":8,"hidden_act":"silu","attention_bias":false,
            "attention_dropout":0.0,"attn_output_gate":true,"output_gate_type":"swish",
            "rms_norm_eps":0.000001,"tie_word_embeddings":tied,
            "partial_rotary_factor":0.5,
            "rope_parameters":{"rope_type":"default","partial_rotary_factor":0.5,"rope_theta":10000.0}
        }
    })).unwrap()
}
struct Loaded {
    source: SourceWeights,
    config: Vec<u8>,
    index_sha256: String,
    admission: TransferAdmission,
    key: PhysicalDeviceKey,
}
fn source(fixture: &Fixture, tied: bool, mutate: impl FnOnce(&mut Value)) -> Loaded {
    let cfg = config(tied);
    let specs = QwenTextConfig::from_json(&cfg)
        .unwrap()
        .tensor_specs()
        .unwrap();
    let mut header = BTreeMap::new();
    let mut map = BTreeMap::new();
    let mut payload = Vec::new();
    for spec in specs {
        let kind = if spec.shape.len() == 1 && spec.name.ends_with("dt_bias") {
            "F16"
        } else if spec.shape.len() == 1 {
            "F32"
        } else {
            "BF16"
        };
        let start = payload.len();
        for _ in 0..spec.shape.iter().product::<u64>() {
            match kind {
                "F32" => payload.extend_from_slice(&0.25f32.to_le_bytes()),
                "F16" => payload.extend_from_slice(&0x3400u16.to_le_bytes()),
                "BF16" => {
                    let bits = if spec.name == "lm_head.weight" {
                        0xbf80u16
                    } else {
                        0x3f80
                    };
                    payload.extend_from_slice(&bits.to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        header.insert(
            spec.name.clone(),
            json!({
                "dtype":kind,"shape":spec.shape,"data_offsets":[start,payload.len()]
            }),
        );
        map.insert(spec.name, "model.safetensors");
    }
    let mut header = serde_json::to_value(header).unwrap();
    mutate(&mut header);
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(&payload);
    let path = fixture.0.join("model.safetensors");
    fs::write(&path, &bytes).unwrap();
    let index = serde_json::to_vec(&json!({
        "metadata":{"total_size":payload.len()},"weight_map":map
    }))
    .unwrap();
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
    let shard = ExpectedSourceShard::new(
        "model.safetensors".into(),
        artifact,
        bytes.len() as u64,
        digest(&bytes),
    )
    .unwrap();
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("synthetic-host", "hf-binding").unwrap();
    arbiter.register(key.clone(), 64 * 1024 * 1024).unwrap();
    let admission = TransferAdmission {
        arbiter,
        resource: key.clone(),
        owner: ResourceOwner::new("qwen", "source-binding").unwrap(),
    };
    let index_sha256 = digest(&index);
    let source = SourceWeights::prepare_verified_shards(
        &index,
        &index_sha256,
        vec![shard],
        SourceOpenLimits {
            max_owned_bytes: 8 * 1024 * 1024,
            max_metadata_bytes: 32 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 200,
        },
        &admission,
    )
    .unwrap()
    .materialize()
    .unwrap();
    Loaded {
        source,
        config: cfg,
        index_sha256,
        admission,
        key,
    }
}
fn bind(
    loaded: Loaded,
) -> (
    std::sync::Arc<LoadedQwenModel>,
    TransferAdmission,
    PhysicalDeviceKey,
) {
    let model = LoadedQwenModel::from_original_source(
        loaded.source,
        &loaded.config,
        &digest(&loaded.config),
        &loaded.index_sha256,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    (model, loaded.admission, loaded.key)
}

#[test]
fn original_binding_retains_actual_source_pointer_dtype_axes_geometry_and_last_lease() {
    let fixture = Fixture::new();
    let loaded = source(&fixture, false, |_| {});
    let id = loaded.source.source_identity();
    let name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
    let original_ptr = loaded.source.tensor_bytes(name).unwrap().as_ptr();
    let config_sha = digest(&loaded.config);
    let index_sha = loaded.index_sha256.clone();
    let bundle_sha = loaded.source.verified_bundle_sha256().unwrap().to_owned();
    let retained = loaded
        .admission
        .arbiter
        .snapshot(&loaded.key)
        .unwrap()
        .reserved;
    let (model, admission, key) = bind(loaded);
    assert_eq!(model.source_identity(), id);
    assert_eq!(model.tensor_bytes(name).unwrap().as_ptr(), original_ptr);
    assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, retained);
    let binding = model.original_binding().unwrap();
    assert_eq!(binding.config_sha256(), config_sha);
    assert_eq!(binding.index_sha256(), index_sha);
    assert_eq!(binding.bundle_sha256(), bundle_sha);
    assert_eq!(binding.other_tensor_count(), 0);
    assert_eq!(
        binding.text_tensor_count(),
        QwenTextConfig::from_json(&config(false))
            .unwrap()
            .tensor_specs()
            .unwrap()
            .len()
    );
    let qkv = model.matrix_descriptor(QwenMatrixRole::GdnQkv(0)).unwrap();
    assert_eq!(
        (qkv.columns, qkv.rows, qkv.axes, qkv.kind),
        (4, 8, MatrixAxes::OutputInput, QuantKind::Bf16)
    );
    assert_eq!(qkv.name, name);
    qkv.validate_source(model.as_ref()).unwrap();
    assert_eq!(model.tensor_info(name).unwrap().shape, [8, 4]);
    let gamma = model
        .parameter_descriptor(QwenParameterRole::GdnNorm(0))
        .unwrap();
    assert_eq!(
        (gamma.source, gamma.kind, gamma.shape, gamma.bytes),
        (id, QuantKind::F32, vec![2], 8)
    );
    assert_eq!(
        model
            .parameter_descriptor(QwenParameterRole::GdnDt(0))
            .unwrap()
            .kind,
        QuantKind::F16
    );
    assert_eq!(
        model
            .parameter_descriptor(QwenParameterRole::GdnConvolution(0))
            .unwrap()
            .shape,
        [8, 1, 4]
    );
    assert_eq!(
        model.original_binding().unwrap().gdn_geometry(),
        OriginalGdnGeometry {
            key_heads: 1,
            value_heads: 2,
            key_width: 2,
            value_width: 2,
            convolution_kernel: 4,
        }
    );
    assert_eq!(
        model.state_specs(),
        &[
            StateSpec::Delta { conv: 24, state: 8 },
            StateSpec::Delta { conv: 24, state: 8 },
            StateSpec::Delta { conv: 24, state: 8 },
            StateSpec::Full { kv_width: 4 }
        ]
    );
    assert!(model.matrix_descriptor(QwenMatrixRole::Query(0)).is_err());
    assert!(model.matrix_descriptor(QwenMatrixRole::GdnQkv(3)).is_err());
    assert!(model.matrix_descriptor(QwenMatrixRole::FfnGate(4)).is_err());
    assert!(
        model
            .parameter_descriptor(QwenParameterRole::QueryNorm(0))
            .is_err()
    );
    let sibling = std::sync::Arc::clone(&model);
    drop(model);
    assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, retained);
    drop(sibling);
    assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, 0);
}

#[test]
fn original_binding_uses_genuine_tied_or_independent_head_without_tensor_aliases() {
    for tied in [false, true] {
        let fixture = Fixture::new();
        let (model, admission, key) = bind(source(&fixture, tied, |_| {}));
        let head = model.matrix_descriptor(QwenMatrixRole::Output).unwrap();
        let embedding = model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap();
        assert_eq!(
            head.name,
            if tied {
                "model.language_model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
        );
        assert_eq!(head == embedding, tied);
        let raw = model.tensor_bytes(&head.name).unwrap();
        assert_eq!(
            u16::from_le_bytes(raw[..2].try_into().unwrap()),
            if tied { 0x3f80 } else { 0xbf80 }
        );
        assert!(model.tensor_info("token_embd.weight").is_err());
        assert_eq!(model.tensor_info("lm_head.weight").is_err(), tied);
        let region = model.embedding_source(15).unwrap();
        assert_eq!(
            (region.source_identity, region.width, region.vocabulary),
            (model.source_identity(), 4, 16)
        );
        assert_eq!(
            region.packed_weights.as_ptr(),
            model.tensor_bytes(&embedding.name).unwrap().as_ptr()
        );
        assert!(model.embedding_source(16).is_err());
        drop(model);
        assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, 0);
    }
}

#[test]
fn original_binding_refuses_all_execution_and_legacy_checkpoint_before_state_allocation() {
    let fixture = Fixture::new();
    let (model, admission, key) = bind(source(&fixture, false, |_| {}));
    let retained = admission.arbiter.snapshot(&key).unwrap().reserved;
    assert_eq!(
        model.numerical_profile(),
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2
    );
    assert!(matches!(
        model.forward_plan(256, true, true),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        model.forward_plan(0, false, false),
        Err(Error::Unsupported(_))
    ));
    #[cfg(feature = "cpu")]
    {
        assert!(matches!(model.new_session(), Err(Error::Unsupported(_))));
        assert!(matches!(
            crate::QwenDenseCpu::from_shared(std::sync::Arc::clone(&model)),
            Err(Error::Unsupported(_))
        ));
    }
    #[cfg(feature = "diagnostic-cpu-checkpoint")]
    {
        assert!(matches!(
            model.checkpoint_from_domains(0, &[]),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            model.load_checkpoint(&fixture.0.join("absent"), [0; 32], [0; 32]),
            Err(Error::Unsupported(_))
        ));
    }
    assert_eq!(model.initialized_cpu_state_bytes(), 0);
    assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, retained);
    drop(model);
    assert_eq!(admission.arbiter.snapshot(&key).unwrap().reserved, 0);
}

#[test]
fn original_binding_refuses_changed_pins_profile_axes_and_non_bf16_matrix() {
    for bad in 0..5 {
        let fixture = Fixture::new();
        let loaded = source(&fixture, false, |header| {
            let matrix =
                &mut header["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"];
            match bad {
                3 => matrix["shape"] = json!([4, 8]),
                4 => matrix["dtype"] = json!("F16"),
                _ => {}
            }
        });
        let config_sha = if bad == 0 {
            "0".repeat(64)
        } else {
            digest(&loaded.config)
        };
        let index_sha = if bad == 1 {
            "0".repeat(64)
        } else {
            loaded.index_sha256
        };
        let profile = if bad == 2 {
            QwenNumericalProfile::GgufSourceV1
        } else {
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2
        };
        assert!(
            LoadedQwenModel::from_original_source(
                loaded.source,
                &loaded.config,
                &config_sha,
                &index_sha,
                profile
            )
            .is_err()
        );
        assert_eq!(
            loaded
                .admission
                .arbiter
                .snapshot(&loaded.key)
                .unwrap()
                .reserved,
            0
        );
    }
}
