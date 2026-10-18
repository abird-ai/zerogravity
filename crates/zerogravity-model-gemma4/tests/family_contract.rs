use std::collections::BTreeSet;
use zerogravity_model_gemma4::{AttentionKind, Gemma4Config, Gemma4Variant};
const MOE: &[u8] = include_bytes!("fixtures/official-26b-it-config.json");
#[test]
fn common_family_admits_official_26b_with_distinct_global_kv_and_no_ple() {
    let config = Gemma4Config::from_json(MOE).unwrap();
    assert_eq!(config.variant(), Gemma4Variant::Moe26bA4b);
    assert_eq!(config.kv_state_owners().len(), 30);
    assert_eq!(config.per_layer_input_dim(), 0);
    assert_eq!(config.moe().unwrap().experts(), 128);
    for (i, layer) in config.layers().iter().enumerate() {
        assert_eq!(layer.kv_source_layer, i);
        if layer.attention == AttentionKind::Full {
            assert_eq!(
                (layer.head_dim, layer.kv_heads, layer.key_equals_value),
                (512, 2, true)
            );
        } else {
            assert_eq!(
                (layer.head_dim, layer.kv_heads, layer.key_equals_value),
                (256, 8, false)
            );
        }
    }
}
#[test]
fn combined_tensor_contract_keeps_moe_branch_norms_and_shared_value_semantics() {
    let config = Gemma4Config::from_json(MOE).unwrap();
    let specs = config.tensor_specs().unwrap();
    assert_eq!(
        specs.iter().map(|s| &s.name).collect::<BTreeSet<_>>().len(),
        specs.len()
    );
    let norm = specs
        .iter()
        .find(|s| {
            s.name
                .ends_with("layers.0.post_feedforward_layernorm_2.weight")
        })
        .unwrap();
    assert_eq!(norm.shape, [2816]);
    let v = specs
        .iter()
        .find(|s| s.name.ends_with("layers.5.self_attn.v_proj.weight"))
        .unwrap();
    assert!(!v.required);
    let k = specs
        .iter()
        .find(|s| s.name.ends_with("layers.5.self_attn.k_proj.weight"))
        .unwrap();
    assert!(k.required);
    assert_eq!(k.shape, [1024, 2816]);
    assert!(!specs.iter().any(|s| s.name.contains("per_layer_input")));
}
