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

#[cfg(feature = "model-qwen35")]
mod original_coordinator_tests {
    use super::*;
    use zerogravity::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
    use zerogravity::original_qwen_math::OriginalQwenLayerMath;
    use zerogravity::original_qwen_state::{
        ORIGINAL_CONV_V1, ORIGINAL_KV_V1, ORIGINAL_RECURRENCE_V1, OriginalQwenStateBridge,
    };
    use zerogravity::state::{PreparedDomain, TaskCoordinator};

    #[test]
    fn source_approved_gdn_state_domains_publish_all_or_nothing_on_the_original_coordinator() {
        let fixture = Fixture::new();
        let (source, config, index, admission, key) = loaded(&fixture);
        let cpu = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&cpu);
        let bridge = OriginalQwenStateBridge::new(&math);
        assert_eq!(
            bridge.typed_domains(0).unwrap(),
            [
                StateSlot::Conv(ConvDomain(0)),
                StateSlot::Recurrent(RecurrentDomain(0))
            ]
        );
        let mut coordinator = bridge.coordinator_for_layer(0, 77).unwrap();
        assert_eq!(
            coordinator.source_identity(),
            Some(source.source_identity())
        );
        assert_eq!(coordinator.root().id, 0);
        assert_eq!(coordinator.root().frontier, 0);
        assert_eq!(coordinator.root().domains.len(), 2);
        assert_eq!(
            coordinator.root().domains["qwen.original.conv.0"].schema,
            ORIGINAL_CONV_V1
        );
        assert_eq!(
            coordinator.root().domains["qwen.original.recurrent.0"].schema,
            ORIGINAL_RECURRENCE_V1
        );
        let before = coordinator.root().clone();
        let first = bridge.prepare(0, &coordinator, &[1.0; 4]).unwrap();
        assert_eq!(first.base_root(), 0);
        assert_eq!(first.proposed_frontier(), 1);
        assert_eq!(first.required_domains(), 2);
        assert_eq!(first.output().len(), 4);
        assert_eq!(coordinator.root(), &before); // tentative only
        // Abandonment of a completed numerical draft publishes nothing.
        drop(first);
        assert_eq!(coordinator.root(), &before);
        let first = bridge.prepare(0, &coordinator, &[1.0; 4]).unwrap();
        let concurrent = bridge.prepare(0, &coordinator, &[2.0; 4]).unwrap();
        let first_output = first.output().to_vec();
        let committed = first.publish(&mut coordinator).unwrap();
        assert_eq!(committed.epoch, 77);
        assert_eq!(committed.sequence, 1);
        assert_eq!(committed.root_id, 1);
        assert_eq!(committed.payload.len(), 16);
        assert_eq!(coordinator.root().frontier, 1);
        assert_eq!(
            coordinator.root().domains["qwen.original.conv.0"]
                .bytes
                .len(),
            48 * 4 / 2
        );
        assert_eq!(
            coordinator.root().domains["qwen.original.recurrent.0"]
                .bytes
                .len(),
            8 * 4
        );
        assert!(concurrent.publish(&mut coordinator).is_err());
        let saved = coordinator.root().clone();
        assert_eq!(saved.id, 1);
        let second = bridge.prepare(0, &coordinator, &[1.0; 4]).unwrap();
        assert_eq!(second.proposed_frontier(), 2);
        assert_eq!(second.output().len(), first_output.len());
        assert_eq!(coordinator.root(), &saved);
        second.publish(&mut coordinator).unwrap();
        assert_eq!(coordinator.root().id, 2);
        assert_eq!(coordinator.root().frontier, 2);
        assert_eq!(saved.id, 1);
        assert!(!saved.domains["qwen.original.recurrent.0"].bytes.is_empty());
        assert!(admission.arbiter.snapshot(&key).unwrap().reserved > 0);
    }

    #[test]
    fn full_attention_prefix_has_only_one_source_bound_kv_domain_and_replays_after_publish() {
        let fixture = Fixture::new();
        let (source, config, index, _, _) = loaded(&fixture);
        let cpu = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&cpu);
        let bridge = OriginalQwenStateBridge::new(&math);
        assert_eq!(
            bridge.typed_domains(3).unwrap(),
            [StateSlot::Kv(KvDomain(3))]
        );
        let mut coordinator = bridge.coordinator_for_layer(3, 19).unwrap();
        assert_eq!(
            coordinator.root().domains["qwen.original.kv.3"].schema,
            ORIGINAL_KV_V1
        );
        let proposed = bridge.prepare(3, &coordinator, &[1.0; 4]).unwrap();
        let first_out = proposed.output().to_vec();
        assert!(first_out.iter().all(|x| x.is_finite()));
        assert_eq!(coordinator.root().id, 0);
        proposed.publish(&mut coordinator).unwrap();
        assert_eq!(
            coordinator.root().domains["qwen.original.kv.3"].bytes.len(),
            8 * 4
        );
        let first_root = coordinator.root().clone();
        let another = bridge.prepare(3, &coordinator, &[1.0; 4]).unwrap();
        assert_eq!(another.proposed_frontier(), 2);
        another.publish(&mut coordinator).unwrap();
        assert_eq!(
            coordinator.root().domains["qwen.original.kv.3"].bytes.len(),
            2 * 8 * 4
        );
        assert_eq!(first_root.domains["qwen.original.kv.3"].bytes.len(), 8 * 4);
        assert_eq!(coordinator.root().frontier, 2);
        assert!(bridge.prepare(0, &coordinator, &[1.0; 4]).is_err()); // wrong layer domains
    }

    #[test]
    fn foreign_source_session_schema_frontier_and_corruption_refuse_without_side_effects() {
        let fixture = Fixture::new();
        let (source, config, index, _, _) = loaded(&fixture);
        let cpu = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&cpu);
        let bridge = OriginalQwenStateBridge::new(&math);
        let mut own = bridge.coordinator_for_layer(0, 1).unwrap();
        let mut another = bridge.coordinator_for_layer(0, 1).unwrap();
        assert_ne!(own.coordinator_id(), another.coordinator_id());
        let step = bridge.prepare(0, &own, &[1.0; 4]).unwrap();
        assert!(step.publish(&mut another).is_err());
        assert_eq!(another.root().id, 0);
        let step = bridge.prepare(0, &own, &[1.0; 4]).unwrap();
        let initial = own.root().clone();
        assert!(own.publish(0, 1, vec![], vec![]).is_err());
        assert_eq!(own.root(), &initial);
        step.publish(&mut own).unwrap();
        assert_eq!(own.root().id, 1);
        let wrong = TaskCoordinator::new_source_bound(
            0,
            source.source_identity(),
            [("qwen.original.conv.0".into(), ORIGINAL_CONV_V1)],
        )
        .unwrap();
        assert!(bridge.prepare(0, &wrong, &[1.0; 4]).is_err());
        let wrong_schema = TaskCoordinator::new_source_bound(
            0,
            source.source_identity(),
            [
                ("qwen.original.conv.0".into(), ORIGINAL_CONV_V1),
                ("qwen.original.recurrent.0".into(), 8),
            ],
        )
        .unwrap();
        assert!(bridge.prepare(0, &wrong_schema, &[1.0; 4]).is_err());
        let fixture_b = Fixture::new();
        let (other_source, other_config, other_index, _, _) = loaded(&fixture_b);
        let other_cpu = QwenOriginalCpu::bind(
            &other_source,
            &other_config,
            &sha(&other_config),
            &sha(&other_index),
        )
        .unwrap();
        let other_math = OriginalQwenLayerMath::new(&other_cpu);
        let other_bridge = OriginalQwenStateBridge::new(&other_math);
        assert!(other_bridge.prepare(0, &own, &[1.0; 4]).is_err());
        // Mutated owned state bytes in a trusted source-bound coordinator
        // are still independently decoded before any new proposal.
        let mut forged = bridge.coordinator_for_layer(3, 55).unwrap();
        let wrong_stored = PreparedDomain {
            name: "qwen.original.kv.3".into(),
            schema: ORIGINAL_KV_V1,
            bytes: vec![1, 2, 3],
            frontier: 1,
        };
        forged
            .publish_source_bound(source.source_identity(), 0, 1, vec![wrong_stored], vec![])
            .unwrap();
        let unchanged = forged.root().clone();
        assert!(bridge.prepare(3, &forged, &[1.0; 4]).is_err());
        assert_eq!(forged.root(), &unchanged);
        let mut forged = bridge.coordinator_for_layer(3, 56).unwrap();
        forged
            .publish_source_bound(
                source.source_identity(),
                0,
                1,
                vec![PreparedDomain {
                    name: "qwen.original.kv.3".into(),
                    schema: ORIGINAL_KV_V1,
                    bytes: f32::NAN.to_le_bytes().repeat(8),
                    frontier: 1,
                }],
                vec![],
            )
            .unwrap();
        assert!(bridge.prepare(3, &forged, &[1.0; 4]).is_err());
    }
}

#[cfg(feature = "model-qwen35")]
mod original_full_model_tests {
    use super::*;
    use zerogravity::original_qwen_math::OriginalQwenLayerMath;
    use zerogravity::original_qwen_state::OriginalQwenStateBridge;
    use zerogravity::state::PreparedDomain;

    #[test]
    fn original_hf_four_layer_model_is_one_complete_source_root_and_full_vocab() {
        let fixture = Fixture::new();
        let (source, config, index, admission, key) = loaded(&fixture);
        let cpu = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&cpu);
        let bridge = OriginalQwenStateBridge::new(&math);
        let mut coordinator = bridge.coordinator_for_model(19).unwrap();
        assert_eq!(
            coordinator.source_identity(),
            Some(source.source_identity())
        );
        assert_eq!(coordinator.root().domains.len(), 7); // 3 GDN*2 plus 1 KV
        assert_eq!(coordinator.root().frontier, 0);
        let root0 = coordinator.root().clone();

        let discarded = bridge
            .prepare_diagnostic_token_f32(&coordinator, 0)
            .unwrap();
        assert_eq!(discarded.logits().len(), 16);
        assert_eq!(discarded.required_domains(), 7);
        assert_eq!(discarded.proposed_frontier(), 1);
        assert!(discarded.logits().iter().all(|v| v.is_finite()));
        assert!(
            discarded
                .logits()
                .iter()
                .all(|v| *v == discarded.logits()[0])
        );
        // Independent synthetic all-ones analytical oracle: the
        // positive uniform final residual RMS-normalizes to ~one per
        // lane, but original Qwen ordinary RMS uses (1 + stored_weight).
        // Stored norm weights are 1, so each lane becomes ~2 and the
        // untied all-ones output row sums four lanes to approximately 8.
        assert!((discarded.logits()[0] - 8.0).abs() < 0.01);
        drop(discarded);
        assert_eq!(coordinator.root(), &root0);
        assert!(
            bridge
                .prepare_diagnostic_token_f32(&coordinator, 16)
                .is_err()
        );
        assert_eq!(coordinator.root(), &root0);

        let first = bridge
            .prepare_diagnostic_token_f32(&coordinator, 0)
            .unwrap();
        let competing = bridge
            .prepare_diagnostic_token_f32(&coordinator, 1)
            .unwrap();
        let logits = first.logits().to_vec();
        let committed = first.publish(&mut coordinator).unwrap();
        assert_eq!(
            (committed.root_id, committed.sequence, committed.epoch),
            (1, 1, 19)
        );
        assert_eq!(committed.payload.len(), 16 * 4);
        assert_eq!(coordinator.root().frontier, 1);
        assert_eq!(
            coordinator.root().domains["qwen.original.kv.3"].bytes.len(),
            32
        );
        for layer in 0..3 {
            assert_eq!(
                coordinator.root().domains[&format!("qwen.original.conv.{layer}")]
                    .bytes
                    .len(),
                96
            );
            assert_eq!(
                coordinator.root().domains[&format!("qwen.original.recurrent.{layer}")]
                    .bytes
                    .len(),
                32
            );
        }
        assert!(competing.publish(&mut coordinator).is_err());
        assert_eq!(coordinator.root().id, 1);
        let root1 = coordinator.root().clone();
        let second = bridge
            .prepare_diagnostic_token_f32(&coordinator, 1)
            .unwrap();
        assert_eq!(second.logits().len(), logits.len());
        assert!(second.logits().iter().all(|v| v.is_finite()));
        assert_eq!(coordinator.root(), &root1);
        second.publish(&mut coordinator).unwrap();
        assert_eq!(coordinator.root().id, 2);
        assert_eq!(coordinator.root().frontier, 2);
        assert_eq!(
            coordinator.root().domains["qwen.original.kv.3"].bytes.len(),
            64
        );
        assert_eq!(root1.id, 1);
        assert_eq!(root1.domains["qwen.original.kv.3"].bytes.len(), 32);
        assert!(admission.arbiter.snapshot(&key).unwrap().reserved > 0);
    }

    #[test]
    fn full_model_first_gdn_matches_independent_one_layer_published_domain_bytes() {
        let fixture = Fixture::new();
        let (source, config, index, _, _) = loaded(&fixture);
        let cpu = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&cpu);
        let bridge = OriginalQwenStateBridge::new(&math);
        let mut full = bridge.coordinator_for_model(1).unwrap();
        let mut partial = bridge.coordinator_for_layer(0, 1).unwrap();
        let first = bridge.prepare_diagnostic_token_f32(&full, 0).unwrap();
        let initial = math.embed_token(0).unwrap();
        let only_layer0 = bridge.prepare(0, &partial, &initial).unwrap();
        only_layer0.publish(&mut partial).unwrap();
        first.publish(&mut full).unwrap();
        for role in ["conv", "recurrent"] {
            let name = format!("qwen.original.{role}.0");
            assert_eq!(
                full.root().domains[&name].bytes,
                partial.root().domains[&name].bytes
            );
        }
        assert_eq!(partial.root().domains.len(), 2);
        assert_eq!(full.root().domains.len(), 7);
    }

    #[test]
    fn full_token_rejects_wrong_source_session_and_missing_late_layer_before_publish() {
        let fixture = Fixture::new();
        let (source, config, index, _, _) = loaded(&fixture);
        let cpu = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let math = OriginalQwenLayerMath::new(&cpu);
        let bridge = OriginalQwenStateBridge::new(&math);
        let mut first = bridge.coordinator_for_model(8).unwrap();
        let mut second = bridge.coordinator_for_model(8).unwrap();
        let prepared = bridge.prepare_diagnostic_token_f32(&first, 0).unwrap();
        assert_ne!(first.coordinator_id(), second.coordinator_id());
        assert!(prepared.publish(&mut second).is_err());
        assert_eq!(second.root().id, 0);
        let prepared = bridge.prepare_diagnostic_token_f32(&first, 0).unwrap();
        prepared.publish(&mut first).unwrap();
        let retained = first.root().clone();
        assert!(first.publish(1, 2, Vec::new(), Vec::new()).is_err());
        assert_eq!(first.root(), &retained);
        let malformed = bridge.coordinator_for_layer(0, 8).unwrap();
        assert!(bridge.prepare_diagnostic_token_f32(&malformed, 0).is_err()); // only 2 of 7 domains
        let fixture2 = Fixture::new();
        let (s2, c2, i2, _, _) = loaded(&fixture2);
        let cpu2 = QwenOriginalCpu::bind(&s2, &c2, &sha(&c2), &sha(&i2)).unwrap();
        let math2 = OriginalQwenLayerMath::new(&cpu2);
        let bridge2 = OriginalQwenStateBridge::new(&math2);
        assert!(bridge2.prepare_diagnostic_token_f32(&first, 0).is_err());

        // Intentionally fabricate an entire state root using the TRUSTED
        // generic TaskCoordinator publisher with a malformed *late* KV
        // domain. The original model binder must refuse it before it
        // can prepare or publish a different root.
        let mut damaged = bridge.coordinator_for_model(3).unwrap();
        let current = damaged.root().clone();
        let bogus = current
            .domains
            .iter()
            .map(|(name, domain)| PreparedDomain {
                name: name.clone(),
                schema: domain.schema,
                frontier: 1,
                bytes: if name == "qwen.original.kv.3" {
                    vec![1, 2, 3]
                } else if name.contains(".conv.") {
                    vec![0u8; 96]
                } else {
                    vec![0u8; 32]
                },
            })
            .collect();
        damaged
            .publish_source_bound(source.source_identity(), 0, 1, bogus, Vec::new())
            .unwrap();
        let untouched = damaged.root().clone();
        assert!(bridge.prepare_diagnostic_token_f32(&damaged, 0).is_err());
        assert_eq!(damaged.root(), &untouched);
    }
}

#[cfg(feature = "model-qwen35")]
mod original_bf16_diagnostic_tests {
    use super::*;
    use zerogravity::original_qwen_math::OriginalQwenLayerMath;
    use zerogravity::original_qwen_precision::OriginalNumericalDType;
    use zerogravity::original_qwen_state::{
        ORIGINAL_CONV_V1, ORIGINAL_KV_V1, ORIGINAL_RECURRENCE_V1, OriginalQwenStateBridge,
    };

    fn bf16_representable(value: f32) -> bool {
        value.is_finite() && (value.to_bits() & 0xffff) == 0
    }
    #[test]
    fn bf16_full_model_uses_distinct_state_schema_and_no_cross_dtype_session_replay() {
        let fixture = Fixture::new();
        let (source, config, index, admission, key) = loaded(&fixture);
        let cpu = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let bf16 = OriginalQwenLayerMath::bf16_diagnostic(&cpu);
        assert_eq!(bf16.dtype(), OriginalNumericalDType::Bf16Diagnostic);
        let bridge = OriginalQwenStateBridge::new(&bf16);
        let mut session = bridge.coordinator_for_model(777).unwrap();
        assert_eq!(session.root().domains.len(), 7);
        assert_eq!(
            session.root().domains["qwen.original.conv.0"].schema,
            ORIGINAL_CONV_V1 + 0x10
        );
        assert_eq!(
            session.root().domains["qwen.original.recurrent.0"].schema,
            ORIGINAL_RECURRENCE_V1 + 0x10
        );
        assert_eq!(
            session.root().domains["qwen.original.kv.3"].schema,
            ORIGINAL_KV_V1 + 0x10
        );
        assert!(bridge.prepare_diagnostic_token_f32(&session, 0).is_err());
        let before = session.root().clone();
        let first = bridge.prepare_diagnostic_token_bf16(&session, 0).unwrap();
        assert_eq!(first.logits().len(), 16);
        assert!(first.logits().iter().copied().all(bf16_representable));
        assert_eq!(first.required_domains(), 7);
        assert_eq!(session.root(), &before); // no publication yet
        let concurrent = bridge.prepare_diagnostic_token_bf16(&session, 1).unwrap();
        first.publish(&mut session).unwrap();
        assert_eq!(session.root().id, 1);
        assert_eq!(session.root().frontier, 1);
        assert!(concurrent.publish(&mut session).is_err());
        let committed = session.root().clone();
        let second = bridge.prepare_diagnostic_token_bf16(&session, 1).unwrap();
        assert!(second.logits().iter().copied().all(bf16_representable));
        second.publish(&mut session).unwrap();
        assert_eq!(session.root().id, 2);
        assert_eq!(session.root().frontier, 2);
        assert_eq!(committed.id, 1);
        assert_eq!(committed.domains["qwen.original.kv.3"].bytes.len(), 32);
        assert_eq!(session.root().domains["qwen.original.kv.3"].bytes.len(), 64);
        assert!(admission.arbiter.snapshot(&key).unwrap().reserved > 0);
        let f32 = OriginalQwenLayerMath::new(&cpu);
        let bridge_f32 = OriginalQwenStateBridge::new(&f32);
        let pure = bridge_f32.coordinator_for_model(777).unwrap();
        assert_eq!(
            pure.root().domains["qwen.original.kv.3"].schema,
            ORIGINAL_KV_V1
        );
        assert!(
            bridge_f32
                .prepare_diagnostic_token_f32(&session, 0)
                .is_err()
        );
        assert!(bridge.prepare_diagnostic_token_bf16(&pure, 0).is_err());
        assert!(bridge_f32.prepare_diagnostic_token_bf16(&pure, 0).is_err());
    }
    #[test]
    fn source_bf16_norm_ffn_attention_gdn_out_are_stage_quantized_not_a_final_logit_cast() {
        let fixture = Fixture::new();
        let (source, config, index, _, _) = loaded(&fixture);
        let cpu = QwenOriginalCpu::bind(&source, &config, &sha(&config), &sha(&index)).unwrap();
        let source_f32 = OriginalQwenLayerMath::new(&cpu);
        let source_bf16 = OriginalQwenLayerMath::bf16_diagnostic(&cpu);
        let x = [0.10123456f32, f32::from_bits(0x3f80_8000), 1.019, 1.25];
        let gamma = [0.0f32, 0.5, -0.2, 0.3];
        let normalized = source_bf16.ordinary_norm(&x, &gamma).unwrap();
        assert!(normalized.iter().copied().all(bf16_representable));
        let raw = source_f32.ordinary_norm(&x, &gamma).unwrap();
        assert_ne!(raw, normalized);
        let stage_f32 = source_f32.propose_ffn(0, &x).unwrap();
        let stage_bf16 = source_bf16.propose_ffn(0, &x).unwrap();
        assert!(stage_bf16.iter().copied().all(bf16_representable));
        assert_ne!(stage_f32, stage_bf16);
        let a = source_bf16.empty_gdn_prefix(0).unwrap();
        let gdn = source_bf16.propose_gdn(0, &x, &a).unwrap();
        assert!(gdn.output.iter().copied().all(bf16_representable));
        assert!(gdn.next.recurrent().iter().all(|v| v.is_finite()));
        let empty = source_bf16.empty_full_prefix(3).unwrap();
        let full = source_bf16.propose_full_attention(3, &x, &empty).unwrap();
        assert!(full.output.iter().copied().all(bf16_representable));
        assert!(
            full.prefix
                .last_key()
                .unwrap()
                .iter()
                .copied()
                .all(bf16_representable)
        );
        assert!(
            full.prefix
                .last_value()
                .unwrap()
                .iter()
                .copied()
                .all(bf16_representable)
        );
    }
}
