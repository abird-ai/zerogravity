//! Exact dense31B text metadata. No learned/native/serving qualification.
use crate::{Error, Result};
use serde_json::Value;

pub const INSTRUCTION_MODEL_ID: &str = "google/gemma-4-31B-it";
pub const INSTRUCTION_REVISION: &str = "842da3794eaa0b77d5f08bae87a17459d91ff475";

pub fn validate_config(config: &Value) -> Result<()> {
    let text = config
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::Malformed("Gemma4 text_config must be an object".into()))?;
    for (name, expected) in [
        ("hidden_size", 5376),
        ("num_hidden_layers", 60),
        ("intermediate_size", 21504),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 16),
        ("num_global_key_value_heads", 4),
        ("head_dim", 256),
        ("global_head_dim", 512),
        ("num_kv_shared_layers", 0),
        ("hidden_size_per_layer_input", 0),
        ("vocab_size_per_layer_input", 262144),
        ("vocab_size", 262144),
        ("sliding_window", 1024),
        ("max_position_embeddings", 262144),
    ] {
        if text.get(name).and_then(Value::as_u64) != Some(expected) {
            return Err(Error::Unsupported(format!("Gemma4 31B unsupported {name}")));
        }
    }
    for (name, expected) in [
        ("enable_moe_block", false),
        ("attention_k_eq_v", true),
        ("use_double_wide_mlp", false),
        ("attention_bias", false),
        ("tie_word_embeddings", true),
    ] {
        if text.get(name).and_then(Value::as_bool) != Some(expected) {
            return Err(Error::Unsupported(format!("Gemma4 31B unsupported {name}")));
        }
    }
    for name in [
        "num_experts",
        "top_k_experts",
        "expert_intermediate_size",
        "moe_intermediate_size",
    ] {
        if text.get(name).is_some_and(|v| !v.is_null()) {
            return Err(Error::Unsupported(format!(
                "Gemma4 dense31B unexpected {name}"
            )));
        }
    }
    if text.get("hidden_activation").and_then(Value::as_str) != Some("gelu_pytorch_tanh")
        || text
            .get("use_bidirectional_attention")
            .and_then(Value::as_str)
            != Some("vision")
    {
        return Err(Error::Unsupported(
            "Gemma4 31B activation/attention policy".into(),
        ));
    }
    let layers = text
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Malformed("Gemma4 layer_types must be an array".into()))?;
    if layers.len() != 60
        || layers.iter().enumerate().any(|(i, kind)| {
            kind.as_str()
                != Some(if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                })
        })
    {
        return Err(Error::Unsupported(
            "Gemma4 31B requires five sliding then one full layer".into(),
        ));
    }
    Ok(())
}
