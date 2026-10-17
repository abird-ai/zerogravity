use std::collections::BTreeMap;
use zerogravity_model_gemma4::{AttentionKind, Gemma4Config, Gemma4Variant, RotaryKind};
const OFFICIAL: &[u8] = include_bytes!("fixtures/official-e2b-it-config.json");
fn altered(f: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let mut v: serde_json::Value = serde_json::from_slice(OFFICIAL).unwrap();
    f(&mut v);
    serde_json::to_vec(&v).unwrap()
}
#[test]
fn official_e2b_has_fifteen_unique_kv_owners_and_twenty_shared_consumers() {
    let cfg = Gemma4Config::from_json(OFFICIAL).unwrap();
    assert_eq!(cfg.variant(), Gemma4Variant::E2b);
    assert_eq!(cfg.layers().len(), 35);
    assert_eq!(cfg.hidden_size(), 1536);
    assert_eq!(cfg.eos_token_ids(), [1, 106]);
    assert_eq!(
        cfg.kv_state_owners()
            .iter()
            .map(|d| d.0)
            .collect::<Vec<_>>(),
        (0..15).collect::<Vec<_>>()
    );
    for (i, l) in cfg.layers().iter().enumerate() {
        assert_eq!(
            l.kv_source_layer,
            if i < 15 {
                i
            } else if (i + 1) % 5 == 0 {
                14
            } else {
                13
            }
        );
        assert_eq!(l.feed_forward_width, if i < 15 { 6144 } else { 12288 });
        assert_eq!(l.kv_heads, 1);
        if l.attention == AttentionKind::Full {
            assert_eq!(
                (l.head_dim, l.rope_rotate, l.rotary),
                (512, 128, RotaryKind::Proportional)
            );
        } else {
            assert_eq!(
                (l.head_dim, l.rope_rotate, l.rotary),
                (256, 256, RotaryKind::Default)
            );
        }
    }
}
#[test]
fn ple_dimensions_are_packed_across_layers_without_allocating_tensor_payloads() {
    let cfg = Gemma4Config::from_json(OFFICIAL).unwrap();
    let specs = cfg.tensor_specs().unwrap();
    let ple = specs
        .iter()
        .find(|s| s.name.ends_with("embed_tokens_per_layer.weight"))
        .unwrap();
    assert_eq!(ple.shape, [262144, 8960]);
    assert_eq!(ple.shape.iter().product::<usize>(), 2_348_810_240);
    let shared = specs
        .iter()
        .find(|s| s.name.ends_with("layers.15.self_attn.k_proj.weight"))
        .unwrap();
    assert!(!shared.required);
    assert!(!specs.iter().any(|s| s.name.ends_with("v_norm.weight")));
    assert_eq!(
        specs
            .iter()
            .filter(|s| s.name.ends_with("layer_scalar"))
            .count(),
        35
    );
}
#[test]
fn declared_tensor_shape_admission_rejects_missing_and_transposed_payload_metadata() {
    let cfg = Gemma4Config::from_json(OFFICIAL).unwrap();
    let mut shapes: BTreeMap<String, Vec<usize>> = cfg
        .tensor_specs()
        .unwrap()
        .into_iter()
        .filter(|s| s.required)
        .map(|s| (s.name, s.shape))
        .collect();
    let expected = shapes.len();
    assert_eq!(
        cfg.validate_tensor_shapes(|n| shapes.get(n).map(Vec::as_slice))
            .unwrap(),
        expected
    );
    shapes
        .get_mut("model.language_model.layers.4.self_attn.q_proj.weight")
        .unwrap()
        .reverse();
    assert!(
        cfg.validate_tensor_shapes(|n| shapes.get(n).map(Vec::as_slice))
            .is_err()
    );
    shapes.remove("model.language_model.embed_tokens.weight");
    assert!(
        cfg.validate_tensor_shapes(|n| shapes.get(n).map(Vec::as_slice))
            .is_err()
    );
}
#[test]
fn untrusted_head_counts_and_sharing_cannot_change_registered_source_geometry() {
    for (name, value) in [
        ("num_attention_heads", u64::MAX),
        ("num_kv_shared_layers", 35),
        ("num_hidden_layers", 4096),
        ("hidden_size_per_layer_input", 0),
    ] {
        let b = altered(|v| v["text_config"][name] = value.into());
        assert!(Gemma4Config::from_json(&b).is_err(), "{name}");
    }
}
#[test]
fn foreign_attention_activation_and_schedule_have_no_generic_transformer_fallback() {
    let changes = [
        altered(|v| v["text_config"]["hidden_activation"] = "silu".into()),
        altered(|v| v["text_config"]["layer_types"][0] = "full_attention".into()),
        altered(|v| v["architectures"][0] = "Qwen3_5ForConditionalGeneration".into()),
        altered(|v| v["text_config"]["use_bidirectional_attention"] = "all".into()),
        altered(|v| {
            v["text_config"]["per_layer_config"] = serde_json::json!({"0":{"head_dim":128}})
        }),
    ];
    for b in changes {
        assert!(Gemma4Config::from_json(&b).is_err());
    }
}
#[test]
fn token_bounds_and_config_size_are_checked_before_model_storage() {
    let b = altered(|v| v["eos_token_id"] = serde_json::json!([262144]));
    assert!(Gemma4Config::from_json(&b).is_err());
    let b = altered(|v| v["eos_token_id"] = serde_json::json!([]));
    assert!(Gemma4Config::from_json(&b).is_err());
    assert!(Gemma4Config::from_json(&vec![b' '; 1024 * 1024 + 1]).is_err());
}
#[test]
fn rotary_frequency_and_scalar_policy_are_not_silently_substituted() {
    for b in [
        altered(|v| {
            v["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] =
                0.5.into()
        }),
        altered(|v| {
            v["text_config"]["rope_parameters"]["sliding_attention"]["rope_theta"] = 500000.0.into()
        }),
        altered(|v| v["text_config"]["rms_norm_eps"] = 0.00001.into()),
        altered(|v| v["text_config"]["final_logit_softcapping"] = 50.0.into()),
    ] {
        assert!(Gemma4Config::from_json(&b).is_err());
    }
}
