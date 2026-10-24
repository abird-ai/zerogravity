//! Final full-attention block and existing FFN/head tail from an external row.
//! The immutable prefix is diagnostic KV, not SessionAuthority or a StateRoot.
use super::{
    attention::{AttentionProjectionValues, attention_roles},
    output_tail::OutputTailValues,
};
use crate::accelerator::program::{TorchCpuAttentionGeometry, TorchCpuAttentionValues};
use crate::original_binding::{OriginalWindowOperators, QwenMatrixRole, QwenParameterRole};
use crate::{Error, Result};
#[cfg(feature = "cpu")]
use crate::{host_math::HostValue, operator_slice::pure::PureSliceInput};

pub(crate) struct AttentionTailSpec {
    pub layer: usize,
    pub hidden: usize,
    pub geometry: TorchCpuAttentionGeometry,
}
pub(crate) struct AttentionTailValues<V> {
    pub normalized_input: V,
    pub projections: AttentionProjectionValues<V>,
    pub attention: TorchCpuAttentionValues<V>,
    pub branch: V,
    pub residual: V,
    pub tail: OutputTailValues<V>,
}
impl OriginalWindowOperators {
    pub(crate) fn attention_tail_spec(&self) -> Result<AttentionTailSpec> {
        let layer = self.output_tail_layer()?;
        let r = attention_roles(self, layer)?;
        let config = self.binding().config();
        let geometry = TorchCpuAttentionGeometry {
            query_heads: r.query_heads,
            kv_heads: r.kv_heads,
            width: r.width,
            rotary: usize::try_from(config.rotary_dimensions())
                .map_err(|_| Error::Limit("CPU rotary width".into()))?,
            theta: config.rope_theta() as f32,
        };
        geometry.validate(0)?;
        self.ordinary_hidden_role(QwenParameterRole::InputNorm(layer))?;
        let output = self.projection_role(QwenMatrixRole::AttentionOutput(layer))?;
        if !config.attention_output_gate()
            || output.columns != r.query_heads * r.width
            || output.rows != r.hidden
        {
            return Err(Error::State(
                "final attention source output projection/gate".into(),
            ));
        }
        Ok(AttentionTailSpec {
            layer,
            hidden: r.hidden,
            geometry,
        })
    }
    /// One genuine current row. Prior comes only from this same source/layer's
    /// completed diagnostic call; external callers cannot supply cache bytes.
    #[cfg(feature = "cpu")]
    pub fn execute_attention_output_tail(
        &self,
        input: PureSliceInput<'_>,
        prior: Option<&WindowAttentionPrefix>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<WindowAttentionTail> {
        use crate::accelerator::program::MathematicalBackend;
        if cancelled.is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "CPU attention tail cancelled before input".into(),
            ));
        }
        let spec = self.attention_tail_spec()?;
        if let Some(p) = prior
            && (p.source != self.source().source_identity()
                || p.layer != spec.layer
                || p.geometry != spec.geometry)
        {
            return Err(Error::State(
                "CPU attention prior source/layer/geometry".into(),
            ));
        }
        let position = prior.map_or(0, |p| p.tokens);
        spec.geometry.validate(position)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let input = math.import_pure_input(input, spec.hidden)?;
        let values = crate::forward::attention_output_tail_slice(
            self,
            &mut math,
            &input,
            prior.map(|p| (&p.keys, &p.values)),
            position,
        )?;
        math.assert_finite(spec.layer, &values.tail.logits)?;
        if cancelled.is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "CPU attention tail cancelled before return".into(),
            ));
        }
        let prefix = WindowAttentionPrefix {
            source: self.source().source_identity(),
            layer: spec.layer,
            geometry: spec.geometry,
            tokens: position + 1,
            keys: values.attention.keys.clone(),
            values: values.attention.values.clone(),
        };
        Ok(WindowAttentionTail {
            input,
            values,
            prefix,
        })
    }
}

#[cfg(feature = "cpu")]
#[derive(Clone)]
pub struct WindowAttentionPrefix {
    source: crate::accelerator::SourceIdentity,
    layer: usize,
    geometry: TorchCpuAttentionGeometry,
    tokens: usize,
    keys: HostValue,
    values: HostValue,
}
#[cfg(feature = "cpu")]
impl WindowAttentionPrefix {
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn keys(&self) -> Result<&[f32]> {
        self.keys.as_slice()
    }
    pub fn values(&self) -> Result<&[f32]> {
        self.values.as_slice()
    }
}
#[cfg(feature = "cpu")]
pub struct WindowAttentionTail {
    input: HostValue,
    values: AttentionTailValues<HostValue>,
    prefix: WindowAttentionPrefix,
}
#[cfg(feature = "cpu")]
impl WindowAttentionTail {
    pub fn prefix(&self) -> &WindowAttentionPrefix {
        &self.prefix
    }
    pub fn layer(&self) -> usize {
        self.prefix.layer
    }
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.prefix.source
    }
    pub fn input(&self) -> Result<&[f32]> {
        self.input.as_slice()
    }
    /// Exact stage labels used by the bounded external reference/replay.
    pub fn stage(&self, label: &str) -> Result<&[f32]> {
        let v = &self.values;
        let value = match label {
            "input" => &self.input,
            "normalized_input" => &v.normalized_input,
            "query_gate_projection" => &v.projections.query_gate_projection,
            "key_projection" => &v.projections.key_projection,
            "value_projection" => &v.projections.value_projection,
            "query" => &v.projections.query,
            "gate" => &v.projections.gate,
            "normalized_query" => &v.projections.normalized_query,
            "normalized_key" => &v.projections.normalized_key,
            "rotated_query" => &v.attention.rotated_query,
            "rotated_key" => &v.attention.rotated_key,
            "keys" => &v.attention.keys,
            "values" => &v.attention.values,
            "dots" => &v.attention.dots,
            "scores" => &v.attention.scores,
            "probabilities" => &v.attention.probabilities,
            "context" => &v.attention.context,
            "gated_attention" => &v.attention.gated,
            "attention_branch" => &v.branch,
            "post_attention" => &v.residual,
            "normalized_ffn" => &v.tail.normalized_ffn,
            "ffn_gate" => &v.tail.gate,
            "ffn_up" => &v.tail.up,
            "ffn_gated" => &v.tail.gated,
            "ffn_branch" => &v.tail.branch,
            "residual" => &v.tail.residual,
            "normalized" => &v.tail.normalized,
            "logits" => &v.tail.logits,
            _ => {
                return Err(Error::Unsupported(
                    "unknown complete attention-tail stage".into(),
                ));
            }
        };
        value.as_slice()
    }
    pub fn full_vocabulary(&self) -> Result<&[f32]> {
        self.values.tail.logits.as_slice()
    }
}
