//! Declared source contracts; these tests never read a weight/tokenizer body.
use serde_json::{Value, json};
use zerogravity_artifacts::artifact::TensorInfo;
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_artifacts::sharded::SafetensorsIndex;
use zerogravity_model_qwen35::config::LayerKind;
use zerogravity_model_qwen35::source::safetensors::{GdnOutputGate, QwenTextConfig};

fn declared_27b() -> Value {
    // Compact independent test vector from the official config. Full original
    // metadata is exercised separately by the explicit capture-bound test.
    json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 5120,
            "head_dim": 256, "num_hidden_layers": 64,
            "num_attention_heads": 24, "num_key_value_heads": 4,
            "layer_types": (0..64).map(|i| if i % 4 == 3 { "full_attention" } else { "linear_attention" }).collect::<Vec<_>>(),
            "vocab_size": 248320, "max_position_embeddings": 262144,
            "full_attention_interval": 4, "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "intermediate_size": 17408, "hidden_act": "silu",
            "attn_output_gate": true, "output_gate_type": "swish",
            "attention_bias": false, "attention_dropout": 0.0,
            "tie_word_embeddings": false, "rms_norm_eps": 0.000001,
            "mtp_num_hidden_layers": 1, "partial_rotary_factor": 0.25,
            "rope_parameters": { "rope_type": "default", "partial_rotary_factor": 0.25, "rope_theta": 10000000.0 }
        }
    })
}
fn parse(config: &Value) -> zerogravity_artifacts::Result<QwenTextConfig> {
    QwenTextConfig::from_json(config.to_string().as_bytes())
}
fn index_for(config: &QwenTextConfig) -> Value {
    let mut map = serde_json::Map::new();
    for spec in config.tensor_specs().unwrap() {
        map.insert(spec.name, json!("model-01.safetensors"));
    }
    map.insert(
        "model.visual.norm.weight".into(),
        json!("model-02.safetensors"),
    );
    map.insert("mtp.norm.weight".into(), json!("model-02.safetensors"));
    json!({"metadata": {"total_size": 55562855904.0}, "weight_map": map})
}

#[test]
fn qwen38_declared_grouping_gate_and_untied_hf_layout() {
    let config = parse(&declared_27b()).unwrap();
    assert_eq!(config.geometry().layers.len(), 64);
    assert_eq!(
        config
            .geometry()
            .layers
            .iter()
            .filter(|k| **k == LayerKind::LinearAttention)
            .count(),
        48
    );
    assert_eq!(config.gdn_output_gate(), GdnOutputGate::Silu);
    assert!(config.attention_output_gate());
    assert!(!config.tie_word_embeddings());
    assert_eq!(config.rotary_dimensions(), 64);
    assert_eq!(config.mtp_num_hidden_layers(), 1);
    let state = config.numerical_state_bytes(512).unwrap();
    assert_eq!(state.convolution, 5_898_240);
    assert_eq!(state.recurrent, 150_994_944);
    assert_eq!(state.kv, 67_108_864);
    assert_eq!(state.total().unwrap(), 224_002_048);
    assert_eq!(config.numerical_state_bytes(0).unwrap().kv, 0);
    assert!(config.numerical_state_bytes(262_145).is_err());
    let specs = config.tensor_specs().unwrap();
    assert_eq!(specs.len(), 851);
    for (name, shape) in [
        ("lm_head.weight", vec![248320, 5120]),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![10240, 5120],
        ),
        (
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![10240, 1, 4],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![6144, 5120],
        ),
        ("model.language_model.layers.0.linear_attn.A_log", vec![48]),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            vec![12288, 5120],
        ),
        (
            "model.language_model.layers.3.self_attn.k_proj.weight",
            vec![1024, 5120],
        ),
        (
            "model.language_model.layers.3.self_attn.o_proj.weight",
            vec![5120, 6144],
        ),
    ] {
        assert_eq!(specs.iter().find(|s| s.name == name).unwrap().shape, shape);
    }
    assert!(
        specs
            .iter()
            .all(|s| !s.name.starts_with("mtp.") && !s.name.starts_with("model.visual."))
    );
}

#[test]
fn gdn_swish_alias_does_not_change_full_attention_gate_or_query_layout() {
    let mut raw = declared_27b();
    let baseline = parse(&raw).unwrap();
    for gate in [None, Some("silu"), Some("swish"), Some("sigmoid")] {
        if let Some(name) = gate {
            raw["text_config"]["output_gate_type"] = json!(name);
        } else {
            raw["text_config"]
                .as_object_mut()
                .unwrap()
                .remove("output_gate_type");
        }
        let config = parse(&raw).unwrap();
        assert_eq!(
            config.gdn_output_gate(),
            if gate == Some("sigmoid") {
                GdnOutputGate::Sigmoid
            } else {
                GdnOutputGate::Silu
            }
        );
        assert!(config.attention_output_gate());
        assert_eq!(
            config.tensor_specs().unwrap(),
            baseline.tensor_specs().unwrap()
        );
    }
    raw["text_config"]["output_gate_type"] = json!("relu");
    assert!(parse(&raw).is_err());
}

#[test]
fn ordinary_text_index_rejects_missing_extra_or_wrong_layer_roles() {
    let config = parse(&declared_27b()).unwrap();
    let raw = index_for(&config);
    let directory = config
        .validate_shard_index(&SafetensorsIndex::from_json(raw.to_string().as_bytes()).unwrap())
        .unwrap();
    assert_eq!(directory.text_tensors, 851);
    assert_eq!(directory.other_tensors, 2);
    assert_eq!(directory.shards, 2);
    assert!(directory.expected_text_elements > 25_000_000_000);
    for mut bad in [raw.clone(), raw.clone(), raw.clone()]
        .into_iter()
        .enumerate()
    {
        let map = bad.1["weight_map"].as_object_mut().unwrap();
        match bad.0 {
            0 => {
                map.remove("lm_head.weight");
            }
            1 => {
                map.insert(
                    "model.language_model.layers.0.self_attn.q_proj.weight".into(),
                    json!("model-01.safetensors"),
                );
            }
            _ => {
                map.insert(
                    "model.language_model.layers.64.input_layernorm.weight".into(),
                    json!("model-01.safetensors"),
                );
            }
        }
        assert!(
            config
                .validate_shard_index(
                    &SafetensorsIndex::from_json(bad.1.to_string().as_bytes()).unwrap()
                )
                .is_err()
        );
    }
}

#[test]
fn header_shape_contract_distinguishes_hf_matrix_order_and_grouped_qkv() {
    let config = parse(&declared_27b()).unwrap();
    let spec = config
        .tensor_specs()
        .unwrap()
        .into_iter()
        .find(|s| s.name.ends_with("layers.0.linear_attn.in_proj_qkv.weight"))
        .unwrap();
    let mut header = TensorInfo {
        name: spec.name.clone(),
        shape: vec![10240, 5120],
        kind: QuantKind::Bf16,
        byte_offset: 0,
        byte_len: 104857600,
    };
    spec.validate_header(&header).unwrap();
    header.shape = vec![5120, 10240];
    assert!(spec.validate_header(&header).is_err());
    header.shape = vec![18432, 5120]; // Incorrectly treating 16 key heads as48.
    assert!(spec.validate_header(&header).is_err());
    header.shape = spec.shape.clone();
    header.name.push_str(".unbound");
    assert!(spec.validate_header(&header).is_err());
}

#[test]
fn malformed_config_cannot_silently_select_other_math() {
    for (key, value) in [
        ("linear_num_value_heads", json!(47)),
        ("linear_conv_kernel_dim", json!(0)),
        ("linear_conv_kernel_dim", json!(4097)),
        ("linear_key_head_dim", json!(0)),
        ("linear_value_head_dim", json!(8193)),
        ("attention_bias", json!(true)),
        ("attention_dropout", json!("zero")),
        ("attention_dropout", json!(0.1)),
        ("hidden_act", json!("gelu")),
        ("partial_rotary_factor", json!(0.5)),
        ("tie_word_embeddings", json!(true)),
        ("rms_norm_eps", json!(-0.1)),
        ("mtp_num_hidden_layers", json!(-1)),
    ] {
        let mut raw = declared_27b();
        raw["text_config"][key] = value;
        assert!(parse(&raw).is_err(), "{key}");
    }
    let mut raw = declared_27b();
    raw["text_config"]["num_experts"] = json!(128);
    assert!(parse(&raw).is_err());
}

#[test]
#[ignore = "requires explicitly captured official metadata directory; no weights or network"]
fn pinned_official_qwen38_config_and_complete_text_index() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("ZEROGRAVITY_QWEN38_METADATA")
            .expect("set explicit captured metadata source directory"),
    );
    let config =
        QwenTextConfig::from_json(&std::fs::read(directory.join("config.json")).unwrap()).unwrap();
    let index = SafetensorsIndex::from_json(
        &std::fs::read(directory.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let result = config.validate_shard_index(&index).unwrap();
    assert_eq!(index.declared_total_size(), 55_562_855_904);
    assert_eq!(index.tensors().len(), 1199);
    assert_eq!(result.text_tensors, 851);
    assert_eq!(result.other_tensors, 348);
    assert_eq!(result.shards, 18);
    assert_eq!(config.gdn_output_gate(), GdnOutputGate::Silu);
    assert!(config.attention_output_gate());
    println!(
        "Qwen3.8-27B official metadata: {result:?}; header shapes and trained execution remain unqualified"
    );
}
