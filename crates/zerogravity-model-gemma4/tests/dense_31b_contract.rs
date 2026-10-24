//! Pinned official metadata; no learned payload or tokenizer fixture.
use zerogravity_model_gemma4::{AttentionKind, Gemma4Config, Gemma4Variant, variant_31b};
const OFFICIAL: &[u8] = include_bytes!("fixtures/official-31b-it-config.json");

#[test]
fn dense31_has_sixty_distinct_owners_and_true_local_global_geometry() {
    let config = Gemma4Config::from_json(OFFICIAL).unwrap();
    assert_eq!(config.variant(), Gemma4Variant::Dense31b);
    assert_eq!(
        (config.hidden_size(), config.intermediate_size()),
        (5376, 21504)
    );
    assert_eq!(config.layers().len(), 60);
    assert_eq!(
        config
            .kv_state_owners()
            .iter()
            .map(|d| d.0)
            .collect::<Vec<_>>(),
        (0..60).collect::<Vec<_>>()
    );
    assert!(config.moe().is_none());
    assert_eq!(
        (config.per_layer_input_dim(), config.shared_kv_layers()),
        (0, 0)
    );
    assert_eq!(
        (config.sliding_window(), config.max_position_embeddings()),
        (1024, 262144)
    );
    assert_eq!(config.eos_token_ids(), [1, 106]);
    assert_eq!(
        variant_31b::INSTRUCTION_REVISION,
        "842da3794eaa0b77d5f08bae87a17459d91ff475"
    );
    for (i, layer) in config.layers().iter().enumerate() {
        assert_eq!(
            (
                layer.kv_source_layer,
                layer.query_heads,
                layer.feed_forward_width
            ),
            (i, 32, 21504)
        );
        if (i + 1) % 6 == 0 {
            assert_eq!(layer.attention, AttentionKind::Full);
            assert_eq!(
                (
                    layer.kv_heads,
                    layer.head_dim,
                    layer.rope_rotate,
                    layer.key_equals_value
                ),
                (4, 512, 128, true)
            );
        } else {
            assert_eq!(layer.attention, AttentionKind::Sliding);
            assert_eq!(
                (
                    layer.kv_heads,
                    layer.head_dim,
                    layer.rope_rotate,
                    layer.key_equals_value
                ),
                (16, 256, 256, false)
            );
        }
    }
}

#[test]
fn dense31_source_roles_preserve_wide_down_and_tied_head_without_moe_or_ple() {
    let config = Gemma4Config::from_json(OFFICIAL).unwrap();
    let specs = config.tensor_specs().unwrap();
    let by_name = specs
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(by_name.len(), specs.len());
    assert_eq!(
        by_name["model.language_model.layers.59.mlp.down_proj.weight"].shape,
        [5376, 21504]
    );
    assert_eq!(
        by_name["model.language_model.layers.5.self_attn.q_proj.weight"].shape,
        [16384, 5376]
    );
    assert_eq!(
        by_name["model.language_model.layers.4.self_attn.k_proj.weight"].shape,
        [4096, 5376]
    );
    assert_eq!(
        by_name["model.language_model.layers.5.self_attn.k_proj.weight"].shape,
        [2048, 5376]
    );
    assert!(!by_name["model.language_model.layers.5.self_attn.v_proj.weight"].required);
    assert!(by_name["model.language_model.layers.4.self_attn.v_proj.weight"].required);
    assert_eq!(
        by_name["model.language_model.embed_tokens.weight"].shape,
        [262144, 5376]
    );
    assert!(!by_name["lm_head.weight"].required);
    assert!(!specs.iter().any(|s| s.name.contains("router")
        || s.name.contains("experts")
        || s.name.contains("per_layer")));
    let shapes = specs
        .iter()
        .filter(|s| s.required)
        .map(|s| (s.name.clone(), s.shape.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        config
            .validate_tensor_shapes(|n| shapes.get(n).map(Vec::as_slice))
            .unwrap(),
        shapes.len()
    );
}

#[test]
fn dense31_does_not_inherit_e2b_sharing_or_moe_override_admission() {
    for (name, value) in [
        ("num_kv_shared_layers", serde_json::json!(20)),
        ("num_global_key_value_heads", serde_json::json!(2)),
        ("intermediate_size", serde_json::json!(16384)),
        ("num_hidden_layers", serde_json::json!(30)),
        ("enable_moe_block", serde_json::json!(true)),
        ("num_experts", serde_json::json!(128)),
        ("hidden_size_per_layer_input", serde_json::json!(256)),
    ] {
        let mut altered: serde_json::Value = serde_json::from_slice(OFFICIAL).unwrap();
        altered["text_config"][name] = value;
        assert!(
            Gemma4Config::from_json(&serde_json::to_vec(&altered).unwrap()).is_err(),
            "{name}"
        );
    }
}
