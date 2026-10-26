//! Qwen3.5-family text contracts, including the published Qwen3.8-27B layout.
//!
//! Config/index recognition neither opens weights nor enables inference. Shapes
//! here follow the pinned HF module layout (PyTorch output-first order), not
//! GGUF dimension order. The native source remains subject to header, bytes,
//! dtype, identity, memory admission and numerical qualification.
use crate::config::{Family, LayerKind, MAX_CONFIG, ModelGeometry, field, number};
use crate::{Error, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use zerogravity_artifacts::sharded::SafetensorsIndex;

/// Activation of the GDN output RMSNorm gate, distinct from full attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnOutputGate {
    Silu,
    Sigmoid,
}

/// Fully declared text geometry; never contains session or source allocations.
#[derive(Debug)]
pub struct QwenTextConfig {
    geometry: ModelGeometry,
    intermediate_size: u64,
    linear_key_head_dim: u64,
    linear_value_head_dim: u64,
    linear_conv_kernel_dim: u64,
    gdn_output_gate: GdnOutputGate,
    attention_output_gate: bool,
    tie_word_embeddings: bool,
    rms_norm_eps: f64,
    rope_theta: f64,
    rotary_dimensions: u64,
    mtp_num_hidden_layers: u64,
}

/// An expected role in ordinary text decode, not an inspected tensor header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTensorSpec {
    pub name: String,
    /// Hugging Face source order. Matrix shape is `[output, input]`.
    pub shape: Vec<u64>,
}

impl QwenTensorSpec {
    /// Validate one independently read header's name and source-order shape.
    /// Dtype, bytes, offsets, shard completeness and source hashes are separate.
    pub fn validate_header(&self, tensor: &crate::artifact::TensorInfo) -> Result<()> {
        if self.name != tensor.name || self.shape != tensor.shape {
            return Err(Error::Malformed(format!(
                "Qwen source tensor name/shape disagreement: {}",
                self.name
            )));
        }
        Ok(())
    }
}

/// Directory evidence only. No tensor's shape/dtype/data was read from a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenTextDirectory {
    pub text_tensors: usize,
    pub other_tensors: usize,
    pub shards: usize,
    pub expected_text_elements: u64,
}

/// Ideal FP32 numerical state sizes. These exclude COW drafts, allocation
/// padding, temporary values and weights and therefore are NOT admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenTextStateBytes {
    pub convolution: u64,
    pub recurrent: u64,
    pub kv: u64,
}

impl QwenTextStateBytes {
    pub fn total(self) -> Result<u64> {
        self.convolution
            .checked_add(self.recurrent)
            .and_then(|n| n.checked_add(self.kv))
            .ok_or_else(|| Error::Limit("Qwen numerical state size overflow".into()))
    }
}

fn boolean(value: &Value, name: &str, default: bool) -> Result<bool> {
    value.get(name).map_or(Ok(default), |v| {
        v.as_bool()
            .ok_or_else(|| Error::Malformed(format!("invalid boolean '{name}'")))
    })
}
fn positive_float(value: &Value, name: &str) -> Result<f64> {
    let n = field(value, name)?
        .as_f64()
        .filter(|n| n.is_finite() && *n > 0.0)
        .ok_or_else(|| Error::Malformed(format!("invalid positive number '{name}'")))?;
    Ok(n)
}
fn product(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Limit("Qwen tensor dimension overflow".into()))
}

impl QwenTextConfig {
    pub fn geometry(&self) -> &ModelGeometry {
        &self.geometry
    }
    pub fn intermediate_size(&self) -> u64 {
        self.intermediate_size
    }
    pub fn linear_key_head_dim(&self) -> u64 {
        self.linear_key_head_dim
    }
    pub fn linear_value_head_dim(&self) -> u64 {
        self.linear_value_head_dim
    }
    pub fn linear_conv_kernel_dim(&self) -> u64 {
        self.linear_conv_kernel_dim
    }
    pub fn gdn_output_gate(&self) -> GdnOutputGate {
        self.gdn_output_gate
    }
    pub fn attention_output_gate(&self) -> bool {
        self.attention_output_gate
    }
    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    pub fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
    }
    pub fn rotary_dimensions(&self) -> u64 {
        self.rotary_dimensions
    }
    pub fn mtp_num_hidden_layers(&self) -> u64 {
        self.mtp_num_hidden_layers
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() as u64 > MAX_CONFIG {
            return Err(Error::Limit("config.json exceeds 1 MiB".into()));
        }
        let geometry = ModelGeometry::from_json(bytes)?;
        if geometry.family != Family::QwenHybrid {
            return Err(Error::Unsupported(
                "Qwen text config needs dense qwen3_5".into(),
            ));
        }
        let config: Value = serde_json::from_slice(bytes)
            .map_err(|e| Error::Malformed(format!("Qwen text config: {e}")))?;
        let text = field(&config, "text_config")?;
        let dropout = match text.get("attention_dropout") {
            None => 0.0,
            Some(value) => value
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| Error::Malformed("invalid Qwen attention dropout".into()))?,
        };
        if text.get("hidden_act").and_then(Value::as_str) != Some("silu")
            || boolean(text, "attention_bias", false)?
            || dropout != 0.0
        {
            return Err(Error::Unsupported(
                "unregistered Qwen text activation/bias/dropout".into(),
            ));
        }
        // Pinned vLLM QwenGatedDeltaNetAttention consumes this field for its
        // RMSNormGated. "swish" is a SiLU alias HERE; it never selects the
        // full-attention gate, whose source operation remains sigmoid.
        let gdn_output_gate = match text.get("output_gate_type") {
            None => GdnOutputGate::Silu,
            Some(Value::String(name)) if matches!(name.as_str(), "silu" | "swish") => {
                GdnOutputGate::Silu
            }
            Some(Value::String(name)) if name == "sigmoid" => GdnOutputGate::Sigmoid,
            _ => {
                return Err(Error::Unsupported(
                    "unregistered Qwen GDN output gate".into(),
                ));
            }
        };
        let intermediate_size = number(text, "intermediate_size")?;
        let linear_key_head_dim = number(text, "linear_key_head_dim")?;
        let linear_value_head_dim = number(text, "linear_value_head_dim")?;
        let linear_conv_kernel_dim = number(text, "linear_conv_kernel_dim")?;
        let key_heads = geometry.recurrent_key_heads.unwrap_or(0);
        let value_heads = geometry.recurrent_value_heads.unwrap_or(0);
        if !(1..=4096).contains(&linear_conv_kernel_dim)
            || intermediate_size > 1_048_576
            || linear_key_head_dim > 8192
            || linear_value_head_dim > 8192
            || product(key_heads, linear_key_head_dim)? > 1_048_576
            || product(value_heads, linear_value_head_dim)? > 1_048_576
            || !value_heads.is_multiple_of(key_heads)
            || !geometry
                .num_attention_heads
                .is_multiple_of(geometry.num_key_value_heads)
        {
            return Err(Error::Malformed(
                "invalid Qwen grouped head/FFN geometry".into(),
            ));
        }
        let tie_word_embeddings = boolean(text, "tie_word_embeddings", false)?;
        if boolean(&config, "tie_word_embeddings", tie_word_embeddings)? != tie_word_embeddings {
            return Err(Error::Malformed(
                "outer/text embedding tying disagreement".into(),
            ));
        }
        let rope = field(text, "rope_parameters")?;
        if rope.get("rope_type").and_then(Value::as_str) != Some("default") {
            return Err(Error::Unsupported(
                "Qwen layout requires default source RoPE".into(),
            ));
        }
        let partial = positive_float(rope, "partial_rotary_factor")?;
        if partial > 1.0 {
            return Err(Error::Malformed(
                "Qwen partial rotary factor exceeds one".into(),
            ));
        }
        if let Some(outer_partial) = text.get("partial_rotary_factor")
            && outer_partial.as_f64() != Some(partial)
        {
            return Err(Error::Malformed(
                "Qwen partial rotary factors disagree".into(),
            ));
        }
        let rotary_width = geometry.head_dim as f64 * partial;
        if rotary_width.fract() != 0.0
            || rotary_width < 2.0
            || !(rotary_width as u64).is_multiple_of(2)
        {
            return Err(Error::Malformed(
                "Qwen rotary dimension must be positive and even".into(),
            ));
        }
        let mtp_num_hidden_layers = match text.get("mtp_num_hidden_layers") {
            None => 0,
            Some(v) => v
                .as_u64()
                .filter(|v| *v <= 4096)
                .ok_or_else(|| Error::Malformed("invalid Qwen MTP layer count".into()))?,
        };
        Ok(Self {
            intermediate_size,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            gdn_output_gate,
            attention_output_gate: boolean(text, "attn_output_gate", true)?,
            tie_word_embeddings,
            rms_norm_eps: positive_float(text, "rms_norm_eps")?,
            rope_theta: positive_float(rope, "rope_theta")?,
            rotary_dimensions: rotary_width as u64,
            mtp_num_hidden_layers,
            geometry,
        })
    }

    /// Expected original HF names/shapes; no conversion and no weight read.
    pub fn tensor_specs(&self) -> Result<Vec<QwenTensorSpec>> {
        let g = &self.geometry;
        let h = g.hidden_size;
        let q = product(g.num_attention_heads, g.head_dim)?;
        let kv = product(g.num_key_value_heads, g.head_dim)?;
        let key = product(g.recurrent_key_heads.unwrap_or(0), self.linear_key_head_dim)?;
        let value = product(
            g.recurrent_value_heads.unwrap_or(0),
            self.linear_value_head_dim,
        )?;
        let qkv = product(key, 2)?
            .checked_add(value)
            .ok_or_else(|| Error::Limit("Qwen QKV width overflow".into()))?;
        let mut specs = Vec::new();
        let mut add = |name: String, shape: Vec<u64>| specs.push(QwenTensorSpec { name, shape });
        add(
            "model.language_model.embed_tokens.weight".into(),
            vec![g.vocab_size, h],
        );
        add("model.language_model.norm.weight".into(), vec![h]);
        if !self.tie_word_embeddings {
            add("lm_head.weight".into(), vec![g.vocab_size, h]);
        }
        for (i, layer) in g.layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{i}");
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                add(format!("{prefix}.{name}"), vec![h]);
            }
            for name in ["gate_proj", "up_proj"] {
                add(
                    format!("{prefix}.mlp.{name}.weight"),
                    vec![self.intermediate_size, h],
                );
            }
            add(
                format!("{prefix}.mlp.down_proj.weight"),
                vec![h, self.intermediate_size],
            );
            match layer {
                LayerKind::LinearAttention => {
                    let p = format!("{prefix}.linear_attn");
                    let heads = g.recurrent_value_heads.unwrap_or(0);
                    for name in ["A_log", "dt_bias"] {
                        add(format!("{p}.{name}"), vec![heads]);
                    }
                    add(
                        format!("{p}.conv1d.weight"),
                        vec![qkv, 1, self.linear_conv_kernel_dim],
                    );
                    for name in ["in_proj_a", "in_proj_b"] {
                        add(format!("{p}.{name}.weight"), vec![heads, h]);
                    }
                    add(format!("{p}.in_proj_qkv.weight"), vec![qkv, h]);
                    add(format!("{p}.in_proj_z.weight"), vec![value, h]);
                    add(format!("{p}.norm.weight"), vec![self.linear_value_head_dim]);
                    add(format!("{p}.out_proj.weight"), vec![h, value]);
                }
                LayerKind::FullAttention => {
                    let p = format!("{prefix}.self_attn");
                    let query_width = product(q, if self.attention_output_gate { 2 } else { 1 })?;
                    add(format!("{p}.q_proj.weight"), vec![query_width, h]);
                    for name in ["k_proj", "v_proj"] {
                        add(format!("{p}.{name}.weight"), vec![kv, h]);
                    }
                    add(format!("{p}.o_proj.weight"), vec![h, q]);
                    for name in ["q_norm", "k_norm"] {
                        add(format!("{p}.{name}.weight"), vec![g.head_dim]);
                    }
                }
                LayerKind::SlidingAttention => {
                    return Err(Error::Unsupported("Qwen sliding layer".into()));
                }
            }
        }
        Ok(specs)
    }

    /// Size only the ordinary text decoder's numerical state, not the vision
    /// tower or speculative MTP. No state allocation or root is constructed.
    pub fn numerical_state_bytes(&self, prefix: u64) -> Result<QwenTextStateBytes> {
        let g = &self.geometry;
        if prefix > g.max_position_embeddings {
            return Err(Error::Limit(
                "Qwen state prefix exceeds declared context".into(),
            ));
        }
        let linear = g
            .layers
            .iter()
            .filter(|&&k| k == LayerKind::LinearAttention)
            .count() as u64;
        let full = g
            .layers
            .iter()
            .filter(|&&k| k == LayerKind::FullAttention)
            .count() as u64;
        let key = product(g.recurrent_key_heads.unwrap_or(0), self.linear_key_head_dim)?;
        let value = product(
            g.recurrent_value_heads.unwrap_or(0),
            self.linear_value_head_dim,
        )?;
        let qkv = product(key, 2)?
            .checked_add(value)
            .ok_or_else(|| Error::Limit("Qwen state QKV width overflow".into()))?;
        let history = self
            .linear_conv_kernel_dim
            .checked_sub(1)
            .ok_or_else(|| Error::Malformed("zero Qwen convolution kernel".into()))?;
        Ok(QwenTextStateBytes {
            convolution: product(product(product(qkv, history)?, linear)?, 4)?,
            recurrent: product(
                product(product(value, self.linear_key_head_dim)?, linear)?,
                4,
            )?,
            kv: product(
                product(
                    product(
                        product(product(g.num_key_value_heads, g.head_dim)?, prefix)?,
                        full,
                    )?,
                    2,
                )?,
                4,
            )?,
        })
    }

    /// Compare a complete ordinary-text namespace, without opening any shard.
    /// Expected shapes remain unverified until actual tensor headers are bound.
    pub fn validate_shard_index(&self, index: &SafetensorsIndex) -> Result<QwenTextDirectory> {
        let specs = self.tensor_specs()?;
        let names: BTreeSet<&str> = specs.iter().map(|t| t.name.as_str()).collect();
        let mut elements = 0u64;
        for spec in &specs {
            if !index.tensors().contains_key(&spec.name) {
                return Err(Error::Malformed(format!(
                    "Qwen text tensor missing: {}",
                    spec.name
                )));
            }
            let count = spec.shape.iter().try_fold(1u64, |n, &d| product(n, d))?;
            elements = elements
                .checked_add(count)
                .ok_or_else(|| Error::Limit("Qwen text element count overflow".into()))?;
        }
        for name in index.tensors().keys() {
            if (name.starts_with("model.language_model.") || name.starts_with("lm_head."))
                && !names.contains(name.as_str())
            {
                return Err(Error::Malformed(format!(
                    "unexpected Qwen text tensor: {name}"
                )));
            }
        }
        Ok(QwenTextDirectory {
            text_tensors: specs.len(),
            other_tensors: index.tensors().len() - specs.len(),
            shards: index.shards().len(),
            expected_text_elements: elements,
        })
    }
}
