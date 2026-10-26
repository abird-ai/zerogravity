//! Declared dense Qwen geometry; config recognition alone never enables execution.
use crate::{Error, Result};
use serde_json::Value;
use std::path::Path;

pub(crate) const MAX_CONFIG: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    QwenHybrid,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    SlidingAttention,
    FullAttention,
}
#[derive(Debug)]
pub struct ModelGeometry {
    pub family: Family,
    pub hidden_size: u64,
    pub head_dim: u64,
    pub layers: Vec<LayerKind>,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    pub recurrent_key_heads: Option<u64>,
    pub recurrent_value_heads: Option<u64>,
}

pub(crate) fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| Error::Malformed(format!("missing config field '{key}'")))
}
pub(crate) fn number(v: &Value, key: &str) -> Result<u64> {
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
impl ModelGeometry {
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
        let (family, text_type) = match (config_type, architecture) {
            ("qwen3_5", "Qwen3_5ForConditionalGeneration") => (Family::QwenHybrid, "qwen3_5_text"),
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
            layers.push(match (family, name) {
                (Family::QwenHybrid, "linear_attention") => LayerKind::LinearAttention,
                (_, "full_attention") => LayerKind::FullAttention,
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
        let (recurrent_key_heads, recurrent_value_heads) = {
            // Dense Qwen target only. New routed Qwen variants must not be
            // mislabeled as supported simply because they share model_type.
            if cfg
                .get("num_experts")
                .and_then(Value::as_u64)
                .is_some_and(|n| n > 0)
                || cfg
                    .get("num_experts_per_tok")
                    .and_then(Value::as_u64)
                    .is_some_and(|n| n > 0)
            {
                return Err(Error::Unsupported(
                    "routed Qwen variant not part of Phase 1 geometry".into(),
                ));
            }
            let interval = number(cfg, "full_attention_interval")?;
            let conv = number(cfg, "linear_conv_kernel_dim")?;
            let key = number(cfg, "linear_num_key_heads")?;
            let value = number(cfg, "linear_num_value_heads")?;
            if conv > 4096
                || key > 4096
                || value > 4096
                || key.checked_mul(head_dim).is_none_or(|n| n > 1_048_576)
                || value.checked_mul(head_dim).is_none_or(|n| n > 1_048_576)
                || interval > num_layers
                || layers.iter().enumerate().any(|(i, l)| {
                    *l != if (i as u64 + 1).is_multiple_of(interval) {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
            {
                return Err(Error::Malformed(
                    "Qwen hybrid layer schedule disagrees with interval".into(),
                ));
            }
            (Some(key), Some(value))
        };
        Ok(Self {
            family,
            hidden_size,
            head_dim,
            layers,
            num_attention_heads,
            num_key_value_heads,
            vocab_size,
            max_position_embeddings,
            recurrent_key_heads,
            recurrent_value_heads,
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = zerogravity_artifacts::config::read(path, MAX_CONFIG)?;
        Self::from_json(&bytes)
    }
}

/// Declared config only; no source, tokenizer or execution claim.
impl std::fmt::Display for ModelGeometry {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(output, "Family: QwenHybrid")?;
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
                .filter(|&&kind| kind == LayerKind::FullAttention)
                .count()
        )?;
        writeln!(
            output,
            "Gated DeltaNet layers: {}; recurrent K heads: {:?}, V heads: {:?}",
            self.layers
                .iter()
                .filter(|&&kind| kind == LayerKind::LinearAttention)
                .count(),
            self.recurrent_key_heads,
            self.recurrent_value_heads
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn qwen() -> Value {
        let layers = (0..64)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        json!({"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5",
            "text_config":{"model_type":"qwen3_5_text","layer_types":layers,
                "num_hidden_layers":64,"num_attention_heads":24,"num_key_value_heads":4,
                "head_dim":256,"hidden_size":5120,"vocab_size":248320,
                "max_position_embeddings":262144,"full_attention_interval":4,
                "linear_conv_kernel_dim":4,"linear_num_key_heads":16,
                "linear_num_value_heads":48}})
    }

    #[test]
    fn declared_qwen_geometry_and_untrusted_bounds() {
        let q = ModelGeometry::from_json(qwen().to_string().as_bytes()).unwrap();
        assert_eq!(q.family, Family::QwenHybrid);
        assert_eq!(q.layers.len(), 64);
        assert_eq!(
            q.layers
                .iter()
                .filter(|x| **x == LayerKind::FullAttention)
                .count(),
            16
        );
        assert_eq!(q.recurrent_value_heads, Some(48));
        for name in [
            "num_attention_heads",
            "num_key_value_heads",
            "linear_num_value_heads",
        ] {
            let mut raw = qwen();
            raw["text_config"][name] = json!(u64::MAX);
            assert!(ModelGeometry::from_json(raw.to_string().as_bytes()).is_err());
        }
    }
    #[test]
    fn refuses_other_families_and_inconsistent_schedule() {
        let mut raw = qwen();
        raw["text_config"]["layer_types"][0] = json!("full_attention");
        assert!(ModelGeometry::from_json(raw.to_string().as_bytes()).is_err());
        let mut raw = qwen();
        raw["model_type"] = json!("gemma4");
        assert!(ModelGeometry::from_json(raw.to_string().as_bytes()).is_err());
    }
}
