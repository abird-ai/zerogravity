//! Exact *declared config geometry* for two target architecture families.
//!
//! This is config recognition, not a ModelDefinition with verified weight names,
//! tokenizer, recurrence or model execution. A name or architecture identifier
//! alone never authorizes fallback to a generic transformer.

use crate::{Error, Result};
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub mod qwen;

const MAX_CONFIG: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    QwenHybrid,
    Gemma4Moe,
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
    pub sliding_window: Option<u64>,
    pub num_experts: Option<u64>,
    pub top_k_experts: Option<u64>,
    pub recurrent_key_heads: Option<u64>,
    pub recurrent_value_heads: Option<u64>,
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
            ("gemma4", "Gemma4ForConditionalGeneration") => (Family::Gemma4Moe, "gemma4_text"),
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
                (Family::Gemma4Moe, "sliding_attention") => LayerKind::SlidingAttention,
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
        let (
            sliding_window,
            num_experts,
            top_k_experts,
            recurrent_key_heads,
            recurrent_value_heads,
        ) = match family {
            Family::QwenHybrid => {
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
                (None, None, None, Some(key), Some(value))
            }
            Family::Gemma4Moe => {
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
                (Some(window), Some(n), Some(k), None, None)
            }
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
            sliding_window,
            num_experts,
            top_k_experts,
            recurrent_key_heads,
            recurrent_value_heads,
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.by_ref().take(MAX_CONFIG + 1).read_to_end(&mut bytes)?;
        Self::from_json(&bytes)
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
    fn source_geometry_qwen_27b_and_gemma_moe() {
        // Compact structure/geometry vectors extracted from official published
        // model configs, not the original signed/checksummed full JSON.
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
        let g = ModelGeometry::from_json(gemma().to_string().as_bytes()).unwrap();
        assert_eq!(g.family, Family::Gemma4Moe);
        assert_eq!(g.layers.len(), 30);
        assert_eq!(
            g.layers
                .iter()
                .filter(|x| **x == LayerKind::FullAttention)
                .count(),
            5
        );
        assert_eq!((g.num_experts, g.top_k_experts), (Some(128), Some(8)));
    }
    #[test]
    fn untrusted_head_geometry_refuses_overflow_and_recurrent_allocation_bombs() {
        let mut q = qwen();
        q["text_config"]["num_attention_heads"] = json!(u64::MAX);
        assert!(ModelGeometry::from_json(q.to_string().as_bytes()).is_err());
        let mut q = qwen();
        q["text_config"]["num_key_value_heads"] = json!(4097);
        assert!(ModelGeometry::from_json(q.to_string().as_bytes()).is_err());
        let mut q = qwen();
        q["text_config"]["linear_num_value_heads"] = json!(u64::MAX);
        assert!(ModelGeometry::from_json(q.to_string().as_bytes()).is_err());
        let mut g = gemma();
        g["text_config"]["num_attention_heads"] = json!(u64::MAX);
        assert!(ModelGeometry::from_json(g.to_string().as_bytes()).is_err());
        // Compact valid configs must continue accepting model families.
        assert!(ModelGeometry::from_json(qwen().to_string().as_bytes()).is_ok());
        assert!(ModelGeometry::from_json(gemma().to_string().as_bytes()).is_ok());
    }

    #[test]
    fn rejects_unsupported_and_inconsistent_architectures() {
        let mut q = qwen();
        q["text_config"]["layer_types"][0] = json!("full_attention");
        assert!(ModelGeometry::from_json(q.to_string().as_bytes()).is_err());
        let mut g = gemma();
        g["text_config"]["top_k_experts"] = json!(129);
        assert!(ModelGeometry::from_json(g.to_string().as_bytes()).is_err());
        let mut g = gemma();
        g["model_type"] = json!("llama");
        assert!(ModelGeometry::from_json(g.to_string().as_bytes()).is_err());
    }
}
