//! Source-pinned 26B-A4B text variant admission, not an inference support claim.

use crate::moe::MoeGeometry;
use serde_json::Value;
use zerogravity_core::{Error, Result};

pub const BASE_MODEL_ID: &str = "google/gemma-4-26B-A4B";
pub const BASE_REVISION: &str = "24548b62aa021d562695c04aaf7758a1ea47990b";
pub const INSTRUCTION_MODEL_ID: &str = "google/gemma-4-26B-A4B-it";
pub const INSTRUCTION_REVISION: &str = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52";

/// Admit exactly the pinned 26B-A4B text geometry. Multimodal data is retained
/// as source metadata; this function does not authorize vision/video execution.
pub fn validate_config(config: &Value) -> Result<MoeGeometry> {
    let text = config
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::Malformed("Gemma4 text_config must be an object".into()))?;
    for (name, expected) in [
        ("hidden_size", 2816),
        ("num_hidden_layers", 30),
        ("intermediate_size", 2112),
        ("num_experts", 128),
        ("top_k_experts", 8),
        ("moe_intermediate_size", 704),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 8),
        ("num_global_key_value_heads", 2),
        ("head_dim", 256),
        ("global_head_dim", 512),
        ("num_kv_shared_layers", 0),
        ("hidden_size_per_layer_input", 0),
        ("sliding_window", 1024),
        ("vocab_size", 262144),
        ("max_position_embeddings", 262144),
    ] {
        let actual = text
            .get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Malformed(format!("Gemma4 missing or invalid {name}")))?;
        if actual != expected {
            return Err(Error::Unsupported(format!(
                "Gemma4 26B-A4B {name}={actual}, expected {expected}"
            )));
        }
    }
    for (name, expected) in [
        ("enable_moe_block", true),
        ("attention_k_eq_v", true),
        ("use_double_wide_mlp", false),
        ("attention_bias", false),
        ("tie_word_embeddings", true),
    ] {
        if text.get(name).and_then(Value::as_bool) != Some(expected) {
            return Err(Error::Unsupported(format!(
                "Gemma4 26B-A4B unsupported {name}"
            )));
        }
    }
    if text.get("hidden_activation").and_then(Value::as_str) != Some("gelu_pytorch_tanh") {
        return Err(Error::Unsupported(
            "Gemma4 26B-A4B requires gelu_pytorch_tanh".into(),
        ));
    }
    let layers = text
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Malformed("Gemma4 layer_types must be an array".into()))?;
    if layers.len() != 30
        || layers.iter().enumerate().any(|(i, x)| {
            x.as_str()
                != Some(if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                })
        })
    {
        return Err(Error::Unsupported(
            "Gemma4 26B-A4B requires five sliding layers followed by one full layer".into(),
        ));
    }
    MoeGeometry::new(2816, 128, 8, 704)
}
