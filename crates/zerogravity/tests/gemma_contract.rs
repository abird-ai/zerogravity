//! Public CPU-free contract surface. No checkpoint/model payload is read.
#![cfg(feature = "model-gemma4")]
use zerogravity::accelerator::program::KvDomain;
use zerogravity::gemma4::{AttentionKind, Gemma4Config, Gemma4Variant};

const E2B: &[u8] =
    include_bytes!("../../zerogravity-model-gemma4/tests/fixtures/official-e2b-it-config.json");
const MOE: &[u8] =
    include_bytes!("../../zerogravity-model-gemma4/tests/fixtures/official-26b-it-config.json");

#[test]
fn facade_exposes_distinct_official_gemma_contracts_and_existing_core_domains() {
    let dense = Gemma4Config::from_json(E2B).unwrap();
    let routed = Gemma4Config::from_json(MOE).unwrap();
    assert_eq!(dense.variant(), Gemma4Variant::E2b);
    assert_eq!(routed.variant(), Gemma4Variant::Moe26bA4b);
    let owners: Vec<KvDomain> = dense.kv_state_owners();
    assert_eq!(
        owners.iter().map(|d| d.0).collect::<Vec<_>>(),
        (0..15).collect::<Vec<_>>()
    );
    assert_eq!(routed.kv_state_owners().len(), 30);
    assert_eq!(dense.per_layer_input_dim(), 256);
    assert_eq!(routed.per_layer_input_dim(), 0);
    assert!(dense.moe().is_none());
    assert_eq!(routed.moe().unwrap().selected_experts(), 8);
    for layer in routed.layers() {
        assert_eq!(
            layer.key_equals_value,
            layer.attention == AttentionKind::Full
        );
    }
}

#[test]
fn facade_contracts_refuse_cross_family_identity_and_transposed_source_shapes() {
    use zerogravity::model::qwen::QwenTextConfig;
    assert!(
        Gemma4Config::from_json(
            br#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}"#
        )
        .is_err()
    );
    assert!(QwenTextConfig::from_json(E2B).is_err());
    assert!(QwenTextConfig::from_json(MOE).is_err());
    let config = Gemma4Config::from_json(MOE).unwrap();
    let spec = config
        .tensor_specs()
        .unwrap()
        .into_iter()
        .find(|s| s.name.ends_with("layers.5.self_attn.k_proj.weight"))
        .unwrap();
    spec.validate_shape(&[1024, 2816]).unwrap();
    assert!(spec.validate_shape(&[2816, 1024]).is_err());
}
