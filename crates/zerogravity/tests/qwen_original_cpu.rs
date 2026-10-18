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
            "full_attention_interval":4,"linear_conv_kernel_dim":4,
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

#[cfg(feature = "model-qwen35")]
mod original_numerics {
    use super::*;
    use zerogravity::original_qwen_math::{
        OriginalQwenLayerMath, original_gdn_coefficients, repeat_qk_to_value_heads,
        split_interleaved_query_gate,
    };

    fn near(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "actual {actual} differs from analytical {expected}"
        );
    }

    #[test]
    fn independent_hf_query_gate_split_is_head_interleaved_not_two_flat_halves() {
        let raw = [1., 2., -1., -2., 3., 4., -3., -4.];
        let (q, gates) = split_interleaved_query_gate(&raw, 2, 2).unwrap();
        assert_eq!(q, [1., 2., 3., 4.]);
        assert_eq!(gates, [-1., -2., -3., -4.]);
        assert!(split_interleaved_query_gate(&raw, 4, 2).is_err());
        assert!(split_interleaved_query_gate(&raw, 2, 0).is_err());
        assert!(split_interleaved_query_gate(&[f32::NAN; 8], 2, 2).is_err());
        assert!(split_interleaved_query_gate(&raw, usize::MAX, 2).is_err());
    }

    #[test]
    fn exact_repeat_interleave_16_key_heads_to_48_value_heads_preserves_mapping_and_bits() {
        let mut raw = Vec::new();
        for key in 0..16 {
            let datum = if key == 0 { -0.0f32 } else { key as f32 };
            raw.extend(std::iter::repeat_n(datum, 128));
        }
        let repeated = repeat_qk_to_value_heads(&raw, 16, 48, 128).unwrap();
        assert_eq!(repeated.len(), 48 * 128);
        for h in 0..48 {
            assert!(
                repeated[h * 128..(h + 1) * 128]
                    .iter()
                    .all(|&x| x.to_bits() == raw[(h / 3) * 128].to_bits())
            );
        }
        assert!(repeat_qk_to_value_heads(&raw, 17, 48, 128).is_err());
        assert!(repeat_qk_to_value_heads(&raw, 16, 47, 128).is_err());
        assert!(repeat_qk_to_value_heads(&raw, 16, usize::MAX, 128).is_err());
        assert!(repeat_qk_to_value_heads(&[f32::INFINITY], 1, 1, 1).is_err());
    }

    #[test]
    fn original_zero_centered_rms_is_one_plus_checkpoint_weight_not_raw_gamma() {
        let fixture = Fixture::new();
        let (source, config, index, _, _) = loaded(&fixture);
        let bound = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&bound);
        let x = [3., 4.];
        let zero = math.ordinary_norm(&x, &[0., 0.]).unwrap();
        near(zero[0], 3.0 / (12.5f32 + 1e-6).sqrt(), 1e-6);
        near(zero[1], 4.0 / (12.5f32 + 1e-6).sqrt(), 1e-6);
        let doubled = math.ordinary_norm(&x, &[1., 1.]).unwrap();
        assert_eq!(doubled, [2.0 * zero[0], 2.0 * zero[1]]);
        assert_eq!(math.ordinary_norm(&x, &[-1., -1.]).unwrap(), [0., 0.]);
        assert!(math.ordinary_norm(&x, &[0.]).is_err());
        assert!(math.ordinary_norm(&x, &[f32::INFINITY, 0.]).is_err());
    }

    #[test]
    fn original_full_attention_one_token_uses_source_bound_unpublished_kv_proposal() {
        let fixture = Fixture::new();
        let (source, config, index, admit, key) = loaded(&fixture);
        let bound = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&bound);
        let x = [1.0f32; 4];
        let empty = math.empty_full_prefix(3).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.source_identity(), source.source_identity());
        assert_eq!(empty.layer(), 3);
        let first = math.propose_full_attention(3, &x, &empty).unwrap();
        assert_eq!(first.position, 0);
        assert_eq!(first.source, source.source_identity());
        assert_eq!(first.prefix.len(), 1);
        assert_eq!(first.prefix.last_key().unwrap().len(), 4);
        assert_eq!(first.prefix.last_value().unwrap().len(), 4);
        assert_eq!(first.gates.len(), 8);
        assert!(first.gates.iter().all(|v| *v > 0.0));
        assert!(first.output.iter().all(|x| x.is_finite() && *x > 0.0));
        let expected = 64.0 / (1.0 + (-8.0f32).exp());
        for &got in &first.output {
            near(got, expected, 0.01);
        }
        let second = math.propose_full_attention(3, &x, &first.prefix).unwrap();
        assert_eq!(second.position, 1);
        assert_eq!(second.prefix.len(), 2);
        assert_eq!(first.prefix.len(), 1);
        assert_eq!(second.prefix.last_key().unwrap().len(), 4);
        assert_eq!(
            second.prefix.last_value().unwrap(),
            first.prefix.last_value().unwrap()
        );
        // Partial RoPE rotates only first 2 lanes in this tiny config.
        assert_eq!(
            &second.prefix.last_key().unwrap()[2..],
            &first.prefix.last_key().unwrap()[2..]
        );
        assert_ne!(
            &second.prefix.last_key().unwrap()[..2],
            &first.prefix.last_key().unwrap()[..2]
        );
        let replay = math.propose_full_attention(3, &x, &first.prefix).unwrap();
        assert_eq!(replay.output, second.output);
        assert_eq!(replay.prefix.last_key(), second.prefix.last_key());
        assert!(math.empty_full_prefix(0).is_err());
        assert!(math.propose_full_attention(0, &x, &empty).is_err());
        assert!(
            math.propose_full_attention(3, &[f32::NAN; 4], &empty)
                .is_err()
        );
        // Independently re-opening exact same source bytes issues another
        // identity; previously proposed KV cannot be silently reused.
        let fixture_b = Fixture::new();
        let (source_b, config_b, index_b, _, _) = loaded(&fixture_b);
        let bound_b =
            QwenOriginalCpu::bind(&source_b, &config_b, &sha(&config_b), &sha(&index_b)).unwrap();
        let other = OriginalQwenLayerMath::new(&bound_b);
        assert_ne!(other.source_identity(), math.source_identity());
        assert!(other.propose_full_attention(3, &x, &first.prefix).is_err());
        assert!(admit.arbiter.snapshot(&key).unwrap().reserved > 0);
    }

    #[test]
    fn original_gdn_private_two_steps_preserve_bound_source_and_session_state_isolation() {
        let fixture = Fixture::new();
        let (source, config, index, _, _) = loaded(&fixture);
        let bound = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&bound);
        let cfg = bound.binding().config();
        let k = cfg.geometry().recurrent_key_heads.unwrap() as usize;
        let v = cfg.geometry().recurrent_value_heads.unwrap() as usize;
        let kd = cfg.linear_key_head_dim() as usize;
        let vd = cfg.linear_value_head_dim() as usize;
        let conv_channels = (2 * k * kd + v * vd) * 3;
        let state_size = v * kd * vd;
        let initial = math.empty_gdn_prefix(0).unwrap();
        assert_eq!(initial.source_identity(), math.source_identity());
        assert_eq!(initial.layer(), 0);
        assert_eq!(initial.position(), 0);
        assert_eq!(initial.convolution().len(), conv_channels);
        assert_eq!(initial.recurrent().len(), state_size);
        assert!(initial.convolution().iter().all(|&v| v == 0.0));
        assert!(initial.recurrent().iter().all(|&v| v == 0.0));
        let x = [1.0f32; 4];
        let first = math.propose_gdn(0, &x, &initial).unwrap();
        assert_eq!((first.key_heads, first.value_heads), (1, 2));
        assert_eq!(first.next.position(), 1);
        assert_eq!(first.next.convolution().len(), conv_channels);
        assert_eq!(first.next.recurrent().len(), state_size);
        assert!(first.output.iter().all(|x| x.is_finite()));
        assert!(first.output.iter().any(|x| *x > 0.0));
        // Independent analytic zero-state first step: V is SiLU(8);
        // with two equal normalized Q/K channels beta=sigmoid(8),
        // per-value output before head norm is approximately
        // V * beta / sqrt(2), and the gated output has 4 channels
        // of approximately SiLU(8) after normalized head RMS.
        let expected = 4.0 * crate_silu(8.0);
        for &actual in &first.output {
            near(actual, expected, 0.1);
        }
        assert!(initial.convolution().iter().all(|&v| v == 0.0));
        assert!(initial.recurrent().iter().all(|&v| v == 0.0));
        let repeated = math.propose_gdn(0, &x, &initial).unwrap();
        assert_eq!(first.output, repeated.output);
        assert_eq!(first.next.convolution(), repeated.next.convolution());
        assert_eq!(first.next.recurrent(), repeated.next.recurrent());
        let next = math.propose_gdn(0, &x, &first.next).unwrap();
        assert_eq!(next.next.position(), 2);
        assert_ne!(first.next.convolution(), next.next.convolution());
        assert_eq!(first.next.position(), 1); // old immutable proposal unchanged
        assert!(next.output.iter().all(|v| v.is_finite()));
        assert!(math.empty_gdn_prefix(3).is_err());
        assert!(math.propose_gdn(3, &x, &initial).is_err());
        assert!(math.propose_gdn(0, &[f32::INFINITY; 4], &initial).is_err());
        // Same checkpoint reopened under another SourceIdentity cannot
        // reuse the first model's private GDN numerical proposal.
        let fixture_b = Fixture::new();
        let (source_b, config_b, index_b, _, _) = loaded(&fixture_b);
        let bound_b =
            QwenOriginalCpu::bind(&source_b, &config_b, &sha(&config_b), &sha(&index_b)).unwrap();
        let other = OriginalQwenLayerMath::new(&bound_b);
        assert!(other.propose_gdn(0, &x, &first.next).is_err());
    }

    #[test]
    fn original_24_query_to_4_kv_gqa_matches_exact_six_queries_per_key_value() {
        use std::sync::Arc;
        use zerogravity::accelerator::{AttentionQkRegion, AttentionValueRegion, SourceIdentity};
        use zerogravity::qwen35::{source_attention_qk, source_attention_value};
        let source = SourceIdentity::for_new_allocation().unwrap();
        let query = vec![1.0; 24];
        let keys: Vec<Arc<[f32]>> = vec![Arc::from([1.0, 2.0, 3.0, 4.0])];
        let qk = source_attention_qk(AttentionQkRegion {
            source_identity: source,
            operation: "original-hf-independent-gqa",
            query: &query,
            keys: &keys,
            head_width: 1,
            query_heads: 24,
            kv_heads: 4,
            scale: 1.0,
        })
        .unwrap();
        assert_eq!(qk.scores.len(), 24);
        for (h, &score) in qk.scores.iter().enumerate() {
            assert_eq!(score, 1.0 + (h / 6) as f32);
        }
        let values: Vec<Arc<[f32]>> = vec![Arc::from([10.0, 20.0, 30.0, 40.0])];
        let attended = source_attention_value(AttentionValueRegion {
            source_identity: source,
            operation: "original-hf-independent-gqa-value",
            probabilities: &[1.0; 24],
            values: &values,
            head_width: 1,
            query_heads: 24,
            kv_heads: 4,
            padded_tokens: 32,
        })
        .unwrap();
        assert_eq!(attended.len(), 24);
        for (h, &value) in attended.iter().enumerate() {
            assert_eq!(value, 10.0 * (h / 6 + 1) as f32);
        }
    }

    #[test]
    fn original_log_decay_a_and_write_gate_have_an_independent_analytic_oracle() {
        use zerogravity::accelerator::SourceIdentity;
        let source = SourceIdentity::for_new_allocation().unwrap();
        let ready = original_gdn_coefficients(source, &[0.0], &[0.0], &[0.0], &[0.0], 4).unwrap();
        assert_eq!(ready.coefficients.len(), 1);
        near(ready.softplus[0], std::f32::consts::LN_2, 1e-6);
        near(ready.gate_raw[0], -std::f32::consts::LN_2, 1e-6);
        near(ready.coefficients[0].gain, 0.5, 1e-6);
        near(ready.coefficients[0].beta, 0.5, 1e-6);
        near(ready.coefficients[0].scale, 0.5, 1e-6);
        assert!(original_gdn_coefficients(source, &[0.0], &[0.0], &[0.0], &[100.0], 4).is_err());
        assert!(
            original_gdn_coefficients(SourceIdentity::unbound(), &[0.0], &[0.0], &[0.0], &[0.0], 4)
                .is_err()
        );
        assert!(original_gdn_coefficients(source, &[0.0], &[0.0, 1.0], &[0.0], &[0.0], 4).is_err());
        assert!(original_gdn_coefficients(source, &[0.0], &[0.0], &[0.0], &[0.0], 0).is_err());
    }

    #[test]
    fn source_bound_original_ffn_applies_swiglu_and_residual_without_model_session() {
        let fixture = Fixture::new();
        let (source, config, index, _, _) = loaded(&fixture);
        let bound = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&bound);
        let x = [1.0f32; 4];
        let ffn = math.propose_ffn(0, &x).unwrap();
        assert_eq!(ffn.len(), 4);
        let expected = 1.0 + 8.0 * (8.0 * crate_silu(8.0));
        for &value in &ffn {
            near(value, expected, 0.1);
        }
        assert!(math.propose_ffn(4, &x).is_err());
        assert!(math.propose_ffn(0, &[1.0; 3]).is_err());
        assert!(math.propose_ffn(0, &[f32::NAN; 4]).is_err());
    }

    fn crate_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
}
