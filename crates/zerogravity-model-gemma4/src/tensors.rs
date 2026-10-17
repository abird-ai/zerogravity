//! Hugging Face conditional-model tensor metadata, in original row-major order.
//!
//! Shape validation does not read weights, establish numerical parity, map GGUF
//! names, authorize source storage or construct an executable forward plan.
use crate::{Error, Gemma4Config, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma4TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub required: bool,
}
impl Gemma4TensorSpec {
    pub fn validate_shape(&self, actual: &[usize]) -> Result<()> {
        if actual != self.shape {
            return Err(Error::Malformed(format!(
                "Gemma4 tensor {} shape {:?}, expected {:?}",
                self.name, actual, self.shape
            )));
        }
        Ok(())
    }
}

impl Gemma4Config {
    /// Enumerate source-defined metadata without allocating tensor payloads.
    /// Optional K/V tensors on shared layers are redundant checkpoint entries,
    /// never additional KV state owners. A tied lm_head may be omitted.
    pub fn tensor_specs(&self) -> Result<Vec<Gemma4TensorSpec>> {
        let mut out = Vec::new();
        let mut add = |name: String, shape: Vec<usize>, required| {
            out.push(Gemma4TensorSpec {
                name,
                shape,
                required,
            })
        };
        let h = self.hidden_size();
        let v = self.vocab_size();
        let ple = self.per_layer_input_dim();
        add(
            "model.language_model.embed_tokens.weight".into(),
            vec![v, h],
            true,
        );
        add("model.language_model.norm.weight".into(), vec![h], true);
        add("lm_head.weight".into(), vec![v, h], false);
        if ple > 0 {
            let width = ple
                .checked_mul(self.layers().len())
                .ok_or_else(|| Error::Limit("Gemma4 PLE width overflow".into()))?;
            add(
                "model.language_model.embed_tokens_per_layer.weight".into(),
                vec![v, width],
                true,
            );
            add(
                "model.language_model.per_layer_model_projection.weight".into(),
                vec![width, h],
                true,
            );
            add(
                "model.language_model.per_layer_projection_norm.weight".into(),
                vec![ple],
                true,
            );
        }
        for (i, layer) in self.layers().iter().enumerate() {
            let prefix = format!("model.language_model.layers.{i}");
            let query = layer
                .query_heads
                .checked_mul(layer.head_dim)
                .ok_or_else(|| Error::Limit("Gemma4 query width overflow".into()))?;
            let kv = layer
                .kv_heads
                .checked_mul(layer.head_dim)
                .ok_or_else(|| Error::Limit("Gemma4 KV width overflow".into()))?;
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                add(format!("{prefix}.{name}.weight"), vec![h], true);
            }
            add(format!("{prefix}.layer_scalar"), vec![1], true);
            add(
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![query, h],
                true,
            );
            add(
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![h, query],
                true,
            );
            add(
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![layer.head_dim],
                true,
            );
            let owns_kv = layer.kv_source_layer == i;
            add(
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv, h],
                owns_kv,
            );
            add(
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![layer.head_dim],
                owns_kv,
            );
            add(
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv, h],
                owns_kv && !layer.key_equals_value,
            );
            let ff = layer.feed_forward_width;
            for name in ["gate_proj", "up_proj"] {
                add(format!("{prefix}.mlp.{name}.weight"), vec![ff, h], true);
            }
            add(format!("{prefix}.mlp.down_proj.weight"), vec![h, ff], true);
            if ple > 0 {
                add(
                    format!("{prefix}.per_layer_input_gate.weight"),
                    vec![ple, h],
                    true,
                );
                add(
                    format!("{prefix}.per_layer_projection.weight"),
                    vec![h, ple],
                    true,
                );
                add(
                    format!("{prefix}.post_per_layer_input_norm.weight"),
                    vec![h],
                    true,
                );
            }
            if let Some(moe) = self.moe() {
                for name in [
                    "pre_feedforward_layernorm_2",
                    "post_feedforward_layernorm_1",
                    "post_feedforward_layernorm_2",
                ] {
                    add(format!("{prefix}.{name}.weight"), vec![h], true);
                }
                for tensor in moe.tensors(i) {
                    add(tensor.name, tensor.shape, true);
                }
            }
        }
        Ok(out)
    }
    /// Validate an artifact's declared shapes only. The caller still owns
    /// source-byte ranges, codecs, source identity and immutable storage.
    pub fn validate_tensor_shapes<'a>(
        &self,
        mut lookup: impl FnMut(&str) -> Option<&'a [usize]>,
    ) -> Result<usize> {
        let mut checked = 0;
        for spec in self.tensor_specs()? {
            if let Some(actual) = lookup(&spec.name) {
                spec.validate_shape(actual)?;
                checked += 1;
            } else if spec.required {
                return Err(Error::Malformed(format!(
                    "Gemma4 required source tensor {} missing",
                    spec.name
                )));
            }
        }
        Ok(checked)
    }
    /// Source layer indices that actually append KV. Shared consumers only read
    /// those domains; a later common-engine adapter must publish through the
    /// existing SessionAuthority, never through a family-specific publisher.
    pub fn kv_state_owners(&self) -> Vec<zerogravity_core::accelerator::program::KvDomain> {
        self.layers()
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                (l.kv_source_layer == i)
                    .then_some(zerogravity_core::accelerator::program::KvDomain(i))
            })
            .collect()
    }
}
