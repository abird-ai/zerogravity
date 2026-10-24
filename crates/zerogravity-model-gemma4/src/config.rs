//! Immutable admission of the explicitly registered Gemma 4 text variants.
use crate::{Error, Result, moe::MoeGeometry, variant_26b, variant_31b};
use serde_json::Value;

const MAX_CONFIG_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gemma4Variant {
    E2b,
    Moe26bA4b,
    Dense31b,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    Sliding,
    Full,
}
/// This describes source geometry. It does not authorize reusing a Qwen RoPE
/// kernel: proportional Gemma rotation has its own source ordering contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotaryKind {
    Default,
    Proportional,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4LayerSpec {
    pub attention: AttentionKind,
    pub head_dim: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub kv_source_layer: usize,
    pub feed_forward_width: usize,
    pub rotary: RotaryKind,
    pub rope_rotate: usize,
    pub rope_theta: f32,
    pub key_equals_value: bool,
}
/// Validated fields stay private so a caller cannot mutate geometry after
/// deriving a tensor/state contract. Accessors borrow immutable source metadata.
#[derive(Clone, Debug)]
pub struct Gemma4Config {
    variant: Gemma4Variant,
    hidden_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    sliding_window: usize,
    shared_kv_layers: usize,
    per_layer_input_dim: usize,
    intermediate_size: usize,
    final_logit_softcap: f32,
    rms_norm_eps: f32,
    eos_token_ids: Vec<u32>,
    layers: Vec<Gemma4LayerSpec>,
    moe: Option<MoeGeometry>,
}

fn integer(v: &Value, name: &str) -> Result<usize> {
    let n = v
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Malformed(format!("Gemma4 integer {name} missing or invalid")))?;
    usize::try_from(n).map_err(|_| Error::Limit(format!("Gemma4 {name} exceeds address space")))
}
fn boolean(v: &Value, name: &str) -> Result<bool> {
    v.get(name)
        .and_then(Value::as_bool)
        .ok_or_else(|| Error::Malformed(format!("Gemma4 boolean {name} missing or invalid")))
}
fn exact_integer(v: &Value, name: &str, expected: usize) -> Result<()> {
    let n = integer(v, name)?;
    if n != expected {
        return Err(Error::Unsupported(format!(
            "Gemma4 {name}={n}, expected {expected}"
        )));
    }
    Ok(())
}
fn exact_float(v: &Value, name: &str, expected: f32) -> Result<f32> {
    let n = v
        .get(name)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::Malformed(format!("Gemma4 scalar {name} missing or invalid")))?;
    if !n.is_finite() || (n as f32).to_bits() != expected.to_bits() {
        return Err(Error::Unsupported(format!("Gemma4 unsupported {name}")));
    }
    Ok(expected)
}

impl Gemma4Config {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_CONFIG_BYTES {
            return Err(Error::Limit("Gemma4 config exceeds 1 MiB".into()));
        }
        let root: Value = serde_json::from_slice(bytes)
            .map_err(|e| Error::Malformed(format!("Gemma4 config JSON: {e}")))?;
        if root.get("model_type").and_then(Value::as_str) != Some("gemma4")
            || root
                .get("architectures")
                .and_then(Value::as_array)
                .is_none_or(|a| {
                    a.len() != 1 || a[0].as_str() != Some("Gemma4ForConditionalGeneration")
                })
        {
            return Err(Error::Unsupported(
                "unregistered Gemma4 outer architecture".into(),
            ));
        }
        let cfg = root
            .get("text_config")
            .filter(|v| v.is_object())
            .ok_or_else(|| Error::Malformed("Gemma4 text_config object missing".into()))?;
        if cfg.get("model_type").and_then(Value::as_str) != Some("gemma4_text") {
            return Err(Error::Malformed(
                "Gemma4 outer/text architecture disagreement".into(),
            ));
        }
        let enabled = boolean(cfg, "enable_moe_block")?;
        let hidden_size = integer(cfg, "hidden_size")?;
        let variant = match (enabled, hidden_size) {
            (true, _) => Gemma4Variant::Moe26bA4b,
            (false, 5376) => Gemma4Variant::Dense31b,
            (false, _) => Gemma4Variant::E2b,
        };
        let moe = if enabled {
            Some(variant_26b::validate_config(&root)?)
        } else {
            None
        };
        if variant == Gemma4Variant::Dense31b {
            variant_31b::validate_config(&root)?;
        }
        let layer_count = integer(cfg, "num_hidden_layers")?;
        if variant == Gemma4Variant::E2b {
            for (name, expected) in [
                ("hidden_size", 1536),
                ("num_hidden_layers", 35),
                ("num_attention_heads", 8),
                ("num_key_value_heads", 1),
                ("head_dim", 256),
                ("global_head_dim", 512),
                ("intermediate_size", 6144),
                ("hidden_size_per_layer_input", 256),
                ("vocab_size_per_layer_input", 262144),
                ("vocab_size", 262144),
                ("num_kv_shared_layers", 20),
                ("sliding_window", 512),
                ("max_position_embeddings", 131072),
            ] {
                exact_integer(cfg, name, expected)?;
            }
            for (name, expected) in [
                ("attention_bias", false),
                ("attention_k_eq_v", false),
                ("use_double_wide_mlp", true),
                ("tie_word_embeddings", true),
            ] {
                if boolean(cfg, name)? != expected {
                    return Err(Error::Unsupported(format!("Gemma4 E2B unsupported {name}")));
                }
            }
            for name in [
                "num_experts",
                "top_k_experts",
                "expert_intermediate_size",
                "moe_intermediate_size",
                "num_global_key_value_heads",
            ] {
                if cfg.get(name).is_some_and(|v| !v.is_null()) {
                    return Err(Error::Unsupported(format!("Gemma4 E2B unexpected {name}")));
                }
            }
        }
        if cfg.get("hidden_activation").and_then(Value::as_str) != Some("gelu_pytorch_tanh")
            || cfg
                .get("use_bidirectional_attention")
                .is_some_and(|v| !v.is_null() && v.as_str() != Some("vision"))
            || cfg.get("per_layer_config").is_some_and(|v| !v.is_null())
        {
            return Err(Error::Unsupported(
                "Gemma4 activation, attention policy or per-layer override is unregistered".into(),
            ));
        }
        if root.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true) {
            return Err(Error::Unsupported(
                "Gemma4 requires tied source embeddings".into(),
            ));
        }
        exact_integer(cfg, "bos_token_id", 2)?;
        exact_integer(cfg, "pad_token_id", 0)?;
        let rms_norm_eps = exact_float(cfg, "rms_norm_eps", 1e-6)?;
        let final_logit_softcap = exact_float(cfg, "final_logit_softcapping", 30.0)?;
        let shared_kv_layers = integer(cfg, "num_kv_shared_layers")?;
        let intermediate_size = integer(cfg, "intermediate_size")?;
        let per_layer_input_dim = integer(cfg, "hidden_size_per_layer_input")?;
        let local_heads = integer(cfg, "num_key_value_heads")?;
        let global_heads = cfg
            .get("num_global_key_value_heads")
            .and_then(Value::as_u64)
            .map(|n| {
                usize::try_from(n).map_err(|_| Error::Limit("Gemma4 global head count".into()))
            })
            .transpose()?
            .unwrap_or(local_heads);
        let query_heads = integer(cfg, "num_attention_heads")?;
        let local_dim = integer(cfg, "head_dim")?;
        let global_dim = integer(cfg, "global_head_dim")?;
        let unshared = layer_count
            .checked_sub(shared_kv_layers)
            .ok_or_else(|| Error::Malformed("Gemma4 shared KV count exceeds layers".into()))?;
        let layer_types = cfg
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Malformed("Gemma4 layer_types missing".into()))?;
        if layer_types.len() != layer_count {
            return Err(Error::Malformed(
                "Gemma4 layer schedule length mismatch".into(),
            ));
        }
        let ropes = cfg
            .get("rope_parameters")
            .ok_or_else(|| Error::Malformed("Gemma4 rope_parameters missing".into()))?;
        let local_rope = ropes
            .get("sliding_attention")
            .ok_or_else(|| Error::Malformed("Gemma4 sliding rope missing".into()))?;
        let global_rope = ropes
            .get("full_attention")
            .ok_or_else(|| Error::Malformed("Gemma4 global rope missing".into()))?;
        if local_rope.get("rope_type").and_then(Value::as_str) != Some("default")
            || global_rope.get("rope_type").and_then(Value::as_str) != Some("proportional")
        {
            return Err(Error::Unsupported(
                "unregistered Gemma4 rotary policy".into(),
            ));
        }
        exact_float(local_rope, "rope_theta", 10000.0)?;
        exact_float(global_rope, "rope_theta", 1000000.0)?;
        exact_float(global_rope, "partial_rotary_factor", 0.25)?;
        let interval = if variant == Gemma4Variant::E2b { 5 } else { 6 };
        let mut layers = Vec::with_capacity(layer_count);
        for (i, kind) in layer_types.iter().enumerate() {
            let full = (i + 1) % interval == 0;
            if kind.as_str()
                != Some(if full {
                    "full_attention"
                } else {
                    "sliding_attention"
                })
            {
                return Err(Error::Unsupported(
                    "Gemma4 source attention schedule differs from registered variant".into(),
                ));
            }
            let (head_dim, kv_heads) = if full {
                (global_dim, global_heads)
            } else {
                (local_dim, local_heads)
            };
            if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
                return Err(Error::Malformed(
                    "Gemma4 grouped head counts invalid".into(),
                ));
            }
            let kv_source_layer = if i < unshared {
                i
            } else if full {
                unshared - 1
            } else {
                unshared - 2
            };
            layers.push(Gemma4LayerSpec {
                attention: if full {
                    AttentionKind::Full
                } else {
                    AttentionKind::Sliding
                },
                head_dim,
                query_heads,
                kv_heads,
                kv_source_layer,
                feed_forward_width: if i >= unshared && variant == Gemma4Variant::E2b {
                    intermediate_size * 2
                } else {
                    intermediate_size
                },
                rotary: if full {
                    RotaryKind::Proportional
                } else {
                    RotaryKind::Default
                },
                rope_rotate: if full { head_dim / 4 } else { head_dim },
                rope_theta: if full { 1000000.0 } else { 10000.0 },
                key_equals_value: full && boolean(cfg, "attention_k_eq_v")?,
            });
        }
        let eos = root.get("eos_token_id").unwrap_or(&cfg["eos_token_id"]);
        let values = if let Some(a) = eos.as_array() {
            a.clone()
        } else {
            vec![eos.clone()]
        };
        let mut eos_token_ids = Vec::new();
        for value in values {
            let token = value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .filter(|v| (*v as usize) < 262144)
                .ok_or_else(|| Error::Malformed("Gemma4 EOS token out of vocabulary".into()))?;
            if !eos_token_ids.contains(&token) {
                eos_token_ids.push(token);
            }
        }
        if eos_token_ids.is_empty() {
            return Err(Error::Malformed("Gemma4 EOS list empty".into()));
        }
        Ok(Self {
            variant,
            hidden_size,
            vocab_size: integer(cfg, "vocab_size")?,
            max_position_embeddings: integer(cfg, "max_position_embeddings")?,
            sliding_window: integer(cfg, "sliding_window")?,
            shared_kv_layers,
            per_layer_input_dim,
            intermediate_size,
            final_logit_softcap,
            rms_norm_eps,
            eos_token_ids,
            layers,
            moe,
        })
    }
    pub fn variant(&self) -> Gemma4Variant {
        self.variant
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    pub fn sliding_window(&self) -> usize {
        self.sliding_window
    }
    pub fn shared_kv_layers(&self) -> usize {
        self.shared_kv_layers
    }
    pub fn per_layer_input_dim(&self) -> usize {
        self.per_layer_input_dim
    }
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    pub fn final_logit_softcap(&self) -> f32 {
        self.final_logit_softcap
    }
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
    pub fn layers(&self) -> &[Gemma4LayerSpec] {
        &self.layers
    }
    pub fn moe(&self) -> Option<&MoeGeometry> {
        self.moe.as_ref()
    }
}
