//! Narrow MoE config inspection, not a source-bound Gemma4Config or execution permit.
use super::AttentionKind;
use crate::{Error, Result};
use serde_json::Value;

const MAX_CONFIG: u64 = 1024 * 1024;

/// Declared dimensions only. Supports the historical MoE inspector, not all Gemma variants.
#[derive(Debug)]
pub struct DeclaredGeometry {
    pub hidden_size: u64,
    pub head_dim: u64,
    pub layers: Vec<AttentionKind>,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    pub sliding_window: u64,
    pub num_experts: u64,
    pub top_k_experts: u64,
}
fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| Error::Malformed(format!("missing config field '{key}'")))
}
fn number(v: &Value, key: &str) -> Result<u64> {
    let n = field(v, key)?
        .as_u64()
        .ok_or_else(|| Error::Malformed(format!("invalid positive integer '{key}'")))?;
    if n == 0 {
        return Err(Error::Malformed(format!("zero config dimension '{key}'")));
    }
    Ok(n)
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    field(v, key)?
        .as_str()
        .ok_or_else(|| Error::Malformed(format!("invalid text field '{key}'")))
}
impl DeclaredGeometry {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() as u64 > MAX_CONFIG {
            return Err(Error::Limit("config.json exceeds 1 MiB".into()));
        }
        let config: Value = serde_json::from_slice(bytes)
            .map_err(|e| Error::Malformed(format!("model config JSON: {e}")))?;
        let config_type = text(&config, "model_type")?;
        let architecture = field(&config, "architectures")?
            .as_array()
            .and_then(|x| x.first())
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Malformed("missing architectures[0]".into()))?;
        let text_type = match (config_type, architecture) {
            ("gemma4", "Gemma4ForConditionalGeneration") => "gemma4_text",
            _ => {
                return Err(Error::Unsupported(format!(
                    "no registered reference geometry for model_type={config_type} architecture={architecture}"
                )));
            }
        };
        let cfg = field(&config, "text_config")?;
        if text(cfg, "model_type")? != text_type {
            return Err(Error::Malformed(
                "outer/text architecture disagreement".into(),
            ));
        }
        let raw_layers = field(cfg, "layer_types")?
            .as_array()
            .ok_or_else(|| Error::Malformed("missing text layer_types".into()))?;
        let num_layers = number(cfg, "num_hidden_layers")?;
        if num_layers != raw_layers.len() as u64 || num_layers > 4096 {
            return Err(Error::Malformed(
                "num_hidden_layers and layer_types disagree".into(),
            ));
        }
        let mut layers = Vec::with_capacity(raw_layers.len());
        for ty in raw_layers {
            let name = ty
                .as_str()
                .ok_or_else(|| Error::Malformed("layer type must be string".into()))?;
            layers.push(match name {
                "sliding_attention" => AttentionKind::Sliding,
                "full_attention" => AttentionKind::Full,
                _ => {
                    return Err(Error::Unsupported(format!(
                        "unregistered layer type {name}"
                    )));
                }
            });
        }
        let hidden_size = number(cfg, "hidden_size")?;
        let head_dim = number(cfg, "head_dim")?;
        let num_attention_heads = number(cfg, "num_attention_heads")?;
        let num_key_value_heads = number(cfg, "num_key_value_heads")?;
        let vocab_size = number(cfg, "vocab_size")?;
        let max_position_embeddings = number(cfg, "max_position_embeddings")?;
        // Untrusted JSON may advertise arbitrarily large head counts without
        // any corresponding tensor data. Reject them at geometry inspection
        // rather than overflowing CPU KV/GDN products or allocating an
        // impossible recurrent state when a model is opened.
        if num_attention_heads > 4096
            || num_key_value_heads > num_attention_heads
            || num_key_value_heads > 4096
            || num_attention_heads
                .checked_mul(head_dim)
                .is_none_or(|n| n > 1_048_576)
            || num_key_value_heads
                .checked_mul(head_dim)
                .is_none_or(|n| n > 1_048_576)
        {
            return Err(Error::Limit(
                "attention head configuration exceeds safe geometry bounds".into(),
            ));
        }
        if hidden_size > 131072 || head_dim > 8192 || vocab_size > 10_000_000 {
            return Err(Error::Limit(
                "declared model dimensions exceed safety bound".into(),
            ));
        }
        if field(cfg, "enable_moe_block")?.as_bool() != Some(true) {
            return Err(Error::Malformed("Gemma 4 MoE is not enabled".into()));
        }
        let n = number(cfg, "num_experts")?;
        let k = number(cfg, "top_k_experts")?;
        if k > n || n > 65536 {
            return Err(Error::Malformed(
                "Gemma expert routing count invalid".into(),
            ));
        }
        let window = number(cfg, "sliding_window")?;
        let global_kv = number(cfg, "num_global_key_value_heads")?;
        if global_kv > num_attention_heads || window > max_position_embeddings {
            return Err(Error::Malformed("Gemma attention geometry invalid".into()));
        }

        Ok(Self {
            hidden_size,
            head_dim,
            layers,
            num_attention_heads,
            num_key_value_heads,
            vocab_size,
            max_position_embeddings,
            sliding_window: window,
            num_experts: n,
            top_k_experts: k,
        })
    }
}
/// Declared config only; no source, tokenizer or execution claim.
impl std::fmt::Display for DeclaredGeometry {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(output, "Family: Gemma4Moe")?;
        writeln!(
            output,
            "Text geometry: {} layers, hidden {}, head dim {}, attention heads {}, KV heads {}",
            self.layers.len(),
            self.hidden_size,
            self.head_dim,
            self.num_attention_heads,
            self.num_key_value_heads
        )?;
        writeln!(
            output,
            "Vocabulary: {}; max declared positions: {}",
            self.vocab_size, self.max_position_embeddings
        )?;
        writeln!(
            output,
            "Full attention layers: {}",
            self.layers
                .iter()
                .filter(|&&kind| kind == AttentionKind::Full)
                .count()
        )?;
        writeln!(
            output,
            "Sliding attention layers: {}; window: {:?}; experts: {:?}; selected experts: {:?}",
            self.layers
                .iter()
                .filter(|&&kind| kind == AttentionKind::Sliding)
                .count(),
            Some(self.sliding_window),
            Some(self.num_experts),
            Some(self.top_k_experts)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn gemma() -> Value {
        let layers = (0..30)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect::<Vec<_>>();
        json!({"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4",
            "text_config":{"model_type":"gemma4_text","layer_types":layers,
                "num_hidden_layers":30,"num_attention_heads":16,"num_key_value_heads":8,
                "num_global_key_value_heads":2,"head_dim":256,"hidden_size":2816,
                "vocab_size":262144,"max_position_embeddings":262144,
                "enable_moe_block":true,"num_experts":128,"top_k_experts":8,
                "sliding_window":1024}})
    }

    #[test]
    fn compact_geometry_is_inspection_only() {
        let bytes = gemma().to_string();
        let declared = DeclaredGeometry::from_json(bytes.as_bytes()).unwrap();
        assert_eq!(declared.layers.len(), 30);
        assert_eq!(
            declared
                .layers
                .iter()
                .filter(|x| **x == AttentionKind::Full)
                .count(),
            5
        );
        assert_eq!((declared.num_experts, declared.top_k_experts), (128, 8));
        assert!(super::super::Gemma4Config::from_json(bytes.as_bytes()).is_err());
    }
    #[test]
    fn inspection_keeps_narrow_variant_and_untrusted_bounds() {
        for (field, value) in [
            ("num_attention_heads", json!(u64::MAX)),
            ("top_k_experts", json!(129)),
            ("enable_moe_block", json!(false)),
        ] {
            let mut raw = gemma();
            raw["text_config"][field] = value;
            assert!(DeclaredGeometry::from_json(raw.to_string().as_bytes()).is_err());
        }
        let mut raw = gemma();
        raw["model_type"] = json!("qwen3_5");
        assert!(DeclaredGeometry::from_json(raw.to_string().as_bytes()).is_err());
        assert!(
            DeclaredGeometry::from_json(include_bytes!(
                "../../tests/fixtures/official-e2b-it-config.json"
            ))
            .is_err()
        );
    }
}
