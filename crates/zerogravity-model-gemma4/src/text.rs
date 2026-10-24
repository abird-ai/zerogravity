//! One E2B/26B text schedule over the shared mathematical backend.
//! Analytic F32 and explicit E2B BF16 cast diagnostics; no trained/framework qualification.
use crate::{Error, Gemma4Config, Gemma4LayerSpec, Result};
use std::collections::BTreeMap;
use zerogravity_core::accelerator::{SourceIdentity, program::*};
mod original;
#[cfg(feature = "text-tokenizer")]
pub mod tokenizer;
pub use original::{
    Gemma4Bf16Inventory, Gemma4Bf16NormSource, Gemma4Bf16Observation, Gemma4Bf16TextPlan,
    Gemma4CpuInventory, Gemma4ReferencePlan, Gemma4ReferenceReadback, Gemma4ReferenceSession,
    Gemma4SourcePins, Gemma4SourceTile, Gemma4SourceTiles, OriginalGemma4Text,
};
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gemma4MatrixRole {
    Embedding,
    Output,
    PerLayerEmbedding,
    PerLayerModelProjection,
    Query(usize),
    Key(usize),
    Value(usize),
    AttentionOutput(usize),
    FfnGate(usize),
    FfnUp(usize),
    FfnDown(usize),
    PerLayerGate(usize),
    PerLayerProjection(usize),
    Router(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gemma4ParameterRole {
    FinalNorm,
    PerLayerProjectionNorm,
    InputNorm(usize),
    PostAttentionNorm(usize),
    PreFfnNorm(usize),
    PostFfnNorm(usize),
    QueryNorm(usize),
    KeyNorm(usize),
    PostPerLayerNorm(usize),
    LayerScalar(usize),
    MoePreFfnNorm(usize),
    DensePostFfnNorm(usize),
    MoePostFfnNorm(usize),
    RouterScale(usize),
    ExpertScale(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gemma4ExpertRole {
    GateUp(usize),
    Down(usize),
}
#[derive(Clone)]
pub(crate) struct Geometry {
    pub hidden: usize,
    pub vocab: usize,
    pub ple: usize,
    pub max_position: usize,
    pub window: usize,
    pub epsilon: f32,
    pub softcap: f32,
    pub layers: Vec<Gemma4LayerSpec>,
    pub moe: Option<crate::moe::MoeGeometry>,
}
impl Geometry {
    pub(crate) fn from_config(config: &Gemma4Config) -> Result<Self> {
        Ok(Self {
            hidden: config.hidden_size(),
            vocab: config.vocab_size(),
            ple: config.per_layer_input_dim(),
            max_position: config.max_position_embeddings(),
            window: config.sliding_window(),
            epsilon: config.rms_norm_eps(),
            softcap: config.final_logit_softcap(),
            layers: config.layers().to_vec(),
            moe: config.moe().cloned(),
        })
    }
    pub fn domains(&self) -> Vec<KvDomain> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(i, l)| (l.kv_source_layer == i).then_some(KvDomain(i)))
            .collect()
    }
}
pub(crate) trait Weights {
    fn source_identity(&self) -> SourceIdentity;
    fn geometry(&self) -> &Geometry;
    fn activation_precision(&self) -> ActivationPrecision {
        ActivationPrecision::F32
    }
    fn matrix(&self, role: Gemma4MatrixRole) -> Result<SourceTensor>;
    fn expert(&self, _role: Gemma4ExpertRole) -> Result<TextExpertTensor> {
        Err(Error::Unsupported("source has no sparse experts".into()))
    }
    fn parameter(&self, role: Gemma4ParameterRole) -> Result<SourceVector<'_>>;
}
fn shape<B: MathematicalBackend>(
    weights: &impl Weights,
    math: &B,
    value: &B::Value,
    elements: usize,
) -> Result<()> {
    if math.elements(value) != elements || math.precision(value) != weights.activation_precision() {
        return Err(Error::State(
            "analytic text actual activation geometry/precision".into(),
        ));
    }
    Ok(())
}
// Candidate-only taps preserve the old analytic recorded plan byte-for-byte.
fn producer<B: MathematicalBackend>(
    weights: &impl Weights,
    math: &mut B,
    layer: usize,
    stage: &'static str,
    value: &B::Value,
) -> Result<()> {
    if weights.activation_precision() == ActivationPrecision::Bf16WidenedF32 {
        math.observe(layer, stage, value)
    } else {
        Ok(())
    }
}
fn unary<B: MathematicalBackend>(
    math: &mut B,
    name: &str,
    operation: TextElementwise,
    input: &B::Value,
) -> Result<B::Value> {
    math.text_elementwise(name, operation, &[input])
}
fn binary<B: MathematicalBackend>(
    math: &mut B,
    name: &str,
    operation: TextElementwise,
    a: &B::Value,
    b: &B::Value,
) -> Result<B::Value> {
    math.text_elementwise(name, operation, &[a, b])
}
fn projection<B: MathematicalBackend>(
    weights: &impl Weights,
    math: &mut B,
    role: Gemma4MatrixRole,
    input: &B::Value,
) -> Result<B::Value> {
    let matrix = weights.matrix(role)?;
    if matrix.source != weights.source_identity() {
        return Err(Error::State("Gemma projection immutable issuer".into()));
    }
    let rows = matrix.rows;
    let output = math.project(matrix, input)?;
    shape(weights, math, &output, rows)?;
    Ok(output)
}
fn norm<B: MathematicalBackend>(
    weights: &impl Weights,
    math: &mut B,
    role: Gemma4ParameterRole,
    input: &B::Value,
    groups: usize,
) -> Result<B::Value> {
    let gamma = weights.parameter(role)?;
    if gamma.source != weights.source_identity() {
        return Err(Error::State("Gemma normalization immutable issuer".into()));
    }
    let name = gamma.name.to_owned();
    math.grouped_rms(&name, input, gamma, groups, weights.geometry().epsilon)
}

/// The sole family token traversal; host evaluation and plan recording call this.
pub(crate) fn forward<B: MathematicalBackend>(
    weights: &impl Weights,
    math: &mut B,
    input: ForwardInputs<B::Index>,
    need_logits: bool,
) -> Result<Option<B::Value>> {
    use Gemma4MatrixRole as M;
    use Gemma4ParameterRole as P;
    let g = weights.geometry();
    let mut x = math.embed(weights.matrix(M::Embedding)?, input.token)?;
    shape(weights, math, &x, g.hidden)?;
    x = unary(
        math,
        "text.embedding.scale",
        if weights.activation_precision() == ActivationPrecision::Bf16WidenedF32 {
            TextElementwise::ScaleBf16Scalar((g.hidden as f64).sqrt() as f32)
        } else {
            TextElementwise::Scale((g.hidden as f32).sqrt())
        },
        &x,
    )?;
    math.observe(usize::MAX, "embedding", &x)?;
    let ple = if g.ple > 0 {
        let ple = math.embed(weights.matrix(M::PerLayerEmbedding)?, input.token)?;
        let ple = unary(
            math,
            "text.ple.embedding.scale",
            if weights.activation_precision() == ActivationPrecision::Bf16WidenedF32 {
                TextElementwise::ScaleBf16Scalar((g.ple as f64).sqrt() as f32)
            } else {
                TextElementwise::Scale((g.ple as f32).sqrt())
            },
            &ple,
        )?;
        producer(weights, math, usize::MAX, "ple.embedding", &ple)?;
        let projected = projection(weights, math, M::PerLayerModelProjection, &x)?;
        producer(weights, math, usize::MAX, "ple.projection", &projected)?;
        let projected = unary(
            math,
            "text.ple.projection.scale",
            TextElementwise::Scale(
                if weights.activation_precision() == ActivationPrecision::Bf16WidenedF32 {
                    (g.hidden as f64).powf(-0.5) as f32
                } else {
                    (g.hidden as f32).sqrt().recip()
                },
            ),
            &projected,
        )?;
        let projected = norm(
            weights,
            math,
            P::PerLayerProjectionNorm,
            &projected,
            g.layers.len(),
        )?;
        producer(weights, math, usize::MAX, "ple.norm", &projected)?;
        let ple = binary(math, "text.ple.add", TextElementwise::Add, &projected, &ple)?;
        let ple = unary(
            math,
            "text.ple.combine.scale",
            TextElementwise::Scale(
                if weights.activation_precision() == ActivationPrecision::Bf16WidenedF32 {
                    2.0f64.powf(-0.5) as f32
                } else {
                    2.0f32.sqrt().recip()
                },
            ),
            &ple,
        )?;
        shape(weights, math, &ple, g.layers.len() * g.ple)?;
        Some(ple)
    } else {
        None
    };
    for (layer, spec) in g.layers.iter().enumerate() {
        producer(weights, math, layer, "block.input", &x)?;
        let normalized = norm(weights, math, P::InputNorm(layer), &x, 1)?;
        producer(weights, math, layer, "input.norm", &normalized)?;
        let q = projection(weights, math, M::Query(layer), &normalized)?;
        producer(weights, math, layer, "query.projection", &q)?;
        let q = norm(weights, math, P::QueryNorm(layer), &q, spec.query_heads)?;
        producer(weights, math, layer, "query.norm", &q)?;
        let rotary = TextRotary {
            head_width: spec.head_dim,
            rotated_pairs: spec.rope_rotate / 2,
            theta: spec.rope_theta,
        };
        let q = math.text_rotary("text.query.rotary", &q, input.position, rotary)?;
        producer(weights, math, layer, "query.rotary", &q)?;
        let domain = KvDomain(spec.kv_source_layer);
        if spec.kv_source_layer == layer {
            math.validate_kv_prefix(domain, input.position)?;
            let k = projection(weights, math, M::Key(layer), &normalized)?;
            producer(weights, math, layer, "key.projection", &k)?;
            let v = if spec.key_equals_value {
                k.clone()
            } else {
                projection(weights, math, M::Value(layer), &normalized)?
            };
            if !spec.key_equals_value {
                producer(weights, math, layer, "value.projection", &v)?;
            }
            let k = norm(weights, math, P::KeyNorm(layer), &k, spec.kv_heads)?;
            producer(weights, math, layer, "key.norm", &k)?;
            let k = math.text_rotary("text.key.rotary", &k, input.position, rotary)?;
            producer(weights, math, layer, "key.rotary", &k)?;
            let unit = vec![1.0; spec.head_dim];
            let unit_name = format!("gemma4.pure_value_rms.{}", spec.head_dim);
            let v = math.grouped_rms(
                "text.value.unit_rms",
                &v,
                SourceVector {
                    source: weights.source_identity(),
                    name: &unit_name,
                    values: &unit,
                },
                spec.kv_heads,
                g.epsilon,
            )?;
            producer(weights, math, layer, "value.norm", &v)?;
            math.append_kv(domain, &k, &v)?;
        }
        let geometry = AttentionGeometry {
            width: spec.head_dim,
            query_heads: spec.query_heads,
            kv_heads: spec.kv_heads,
        };
        let window = (spec.attention == crate::AttentionKind::Sliding).then_some(g.window);
        let qk = math.text_window_qk(
            "text.attention.unit_scale_qk",
            domain,
            &q,
            input.valid_tokens,
            geometry,
            window,
        )?;
        producer(weights, math, layer, "attention.qk", &qk.scores)?;
        let probabilities = math.attention_softmax(
            "text.attention.softmax",
            &qk.scores,
            spec.query_heads,
            input.valid_tokens,
        )?;
        producer(weights, math, layer, "attention.softmax", &probabilities)?;
        let attended = math.attention_value(
            "text.attention.value",
            domain,
            &probabilities,
            input.valid_tokens,
            geometry,
        )?;
        producer(weights, math, layer, "attention.pv", &attended)?;
        let branch = projection(weights, math, M::AttentionOutput(layer), &attended)?;
        producer(weights, math, layer, "attention.projection", &branch)?;
        let branch = norm(weights, math, P::PostAttentionNorm(layer), &branch, 1)?;
        producer(weights, math, layer, "attention.norm", &branch)?;
        x = binary(
            math,
            "text.attention.residual",
            TextElementwise::Add,
            &x,
            &branch,
        )?;
        producer(weights, math, layer, "attention.residual", &x)?;
        let normalized = norm(weights, math, P::PreFfnNorm(layer), &x, 1)?;
        producer(weights, math, layer, "ffn.norm", &normalized)?;
        let gate = projection(weights, math, M::FfnGate(layer), &normalized)?;
        let up = projection(weights, math, M::FfnUp(layer), &normalized)?;
        producer(weights, math, layer, "ffn.gate", &gate)?;
        producer(weights, math, layer, "ffn.up", &up)?;
        let gate = unary(math, "text.ffn.gelu", TextElementwise::GeluTanh, &gate)?;
        producer(weights, math, layer, "ffn.gelu", &gate)?;
        let activation = binary(
            math,
            "text.ffn.multiply",
            TextElementwise::Multiply,
            &gate,
            &up,
        )?;
        producer(weights, math, layer, "ffn.activation", &activation)?;
        let branch = projection(weights, math, M::FfnDown(layer), &activation)?;
        producer(weights, math, layer, "ffn.down", &branch)?;
        let branch = if let Some(moe) = &g.moe {
            let dense = norm(weights, math, P::DensePostFfnNorm(layer), &branch, 1)?;
            let normalized = norm(weights, math, P::MoePreFfnNorm(layer), &x, 1)?;
            let contract = TextMoeContract {
                router: weights.matrix(M::Router(layer))?,
                gate_up: weights.expert(Gemma4ExpertRole::GateUp(layer))?,
                down: weights.expert(Gemma4ExpertRole::Down(layer))?,
                selected: moe.selected_experts(),
                epsilon: g.epsilon,
            };
            let sparse = math.text_moe(
                "text.ffn.sparse",
                &contract,
                TextMoeInputs {
                    normalized: &normalized,
                    router_input: &x,
                    router_scale: weights.parameter(P::RouterScale(layer))?,
                    expert_scale: weights.parameter(P::ExpertScale(layer))?,
                },
            )?;
            let sparse = norm(weights, math, P::MoePostFfnNorm(layer), &sparse, 1)?;
            binary(
                math,
                "text.ffn.dense_plus_sparse",
                TextElementwise::Add,
                &dense,
                &sparse,
            )?
        } else {
            branch
        };
        let branch = norm(weights, math, P::PostFfnNorm(layer), &branch, 1)?;
        producer(weights, math, layer, "ffn.post_norm", &branch)?;
        x = binary(math, "text.ffn.residual", TextElementwise::Add, &x, &branch)?;
        producer(weights, math, layer, "ffn.residual", &x)?;
        if let Some(ple) = &ple {
            let gate = projection(weights, math, M::PerLayerGate(layer), &x)?;
            producer(weights, math, layer, "ple.gate", &gate)?;
            let gate = unary(math, "text.ple.gelu", TextElementwise::GeluTanh, &gate)?;
            producer(weights, math, layer, "ple.gelu", &gate)?;
            let local = math.view(ple, TensorView::vector(layer * g.ple, g.ple))?;
            producer(weights, math, layer, "ple.input", &local)?;
            let gate = binary(
                math,
                "text.ple.multiply",
                TextElementwise::Multiply,
                &gate,
                &local,
            )?;
            producer(weights, math, layer, "ple.activation", &gate)?;
            let branch = projection(weights, math, M::PerLayerProjection(layer), &gate)?;
            producer(weights, math, layer, "ple.projection", &branch)?;
            let branch = norm(weights, math, P::PostPerLayerNorm(layer), &branch, 1)?;
            producer(weights, math, layer, "ple.post_norm", &branch)?;
            x = binary(math, "text.ple.residual", TextElementwise::Add, &x, &branch)?;
        }
        let scalar = weights.parameter(P::LayerScalar(layer))?;
        if scalar.values.len() != 1 || scalar.source != weights.source_identity() {
            return Err(Error::State(
                "actual one-word Gemma layer scalar source".into(),
            ));
        }
        x = unary(
            math,
            scalar.name,
            TextElementwise::Scale(scalar.values[0]),
            &x,
        )?;
        shape(weights, math, &x, g.hidden)?;
        math.assert_finite(layer, &x)?;
        math.observe(layer, "block.output", &x)?;
    }
    math.advance_frontier()?;
    if !need_logits {
        return Ok(None);
    }
    let normalized = norm(weights, math, P::FinalNorm, &x, 1)?;
    math.observe(usize::MAX, "final_norm", &normalized)?;
    let logits = projection(weights, math, M::Output, &normalized)?;
    shape(weights, math, &logits, g.vocab)?;
    producer(weights, math, usize::MAX, "head", &logits)?;
    let logits = unary(
        math,
        "text.final.softcap",
        TextElementwise::TanhSoftcap(g.softcap),
        &logits,
    )?;
    producer(weights, math, usize::MAX, "logits", &logits)?;
    Ok(Some(logits))
}

pub(crate) fn matrix_names(g: &Geometry) -> BTreeMap<Gemma4MatrixRole, String> {
    use Gemma4MatrixRole as M;
    let root = "model.language_model";
    let mut out = BTreeMap::from([
        (M::Embedding, format!("{root}.embed_tokens.weight")),
        (M::Output, format!("{root}.embed_tokens.weight")),
    ]);
    if g.ple > 0 {
        out.insert(
            M::PerLayerEmbedding,
            format!("{root}.embed_tokens_per_layer.weight"),
        );
        out.insert(
            M::PerLayerModelProjection,
            format!("{root}.per_layer_model_projection.weight"),
        );
    }
    for (i, layer) in g.layers.iter().enumerate() {
        let p = format!("{root}.layers.{i}");
        for (role, suffix) in [
            (M::Query(i), "self_attn.q_proj.weight"),
            (M::AttentionOutput(i), "self_attn.o_proj.weight"),
            (M::FfnGate(i), "mlp.gate_proj.weight"),
            (M::FfnUp(i), "mlp.up_proj.weight"),
            (M::FfnDown(i), "mlp.down_proj.weight"),
        ] {
            out.insert(role, format!("{p}.{suffix}"));
        }
        if g.ple > 0 {
            out.insert(
                M::PerLayerGate(i),
                format!("{p}.per_layer_input_gate.weight"),
            );
            out.insert(
                M::PerLayerProjection(i),
                format!("{p}.per_layer_projection.weight"),
            );
        }
        if g.moe.is_some() {
            out.insert(M::Router(i), format!("{p}.router.proj.weight"));
        }
        if layer.kv_source_layer == i {
            out.insert(M::Key(i), format!("{p}.self_attn.k_proj.weight"));
            if !layer.key_equals_value {
                out.insert(M::Value(i), format!("{p}.self_attn.v_proj.weight"));
            }
        }
    }
    out
}
pub(crate) fn parameter_names(g: &Geometry) -> BTreeMap<Gemma4ParameterRole, String> {
    use Gemma4ParameterRole as P;
    let root = "model.language_model";
    let mut out = BTreeMap::from([(P::FinalNorm, format!("{root}.norm.weight"))]);
    if g.ple > 0 {
        out.insert(
            P::PerLayerProjectionNorm,
            format!("{root}.per_layer_projection_norm.weight"),
        );
    }
    for (i, layer) in g.layers.iter().enumerate() {
        let p = format!("{root}.layers.{i}");
        for (role, suffix) in [
            (P::InputNorm(i), "input_layernorm.weight"),
            (P::PostAttentionNorm(i), "post_attention_layernorm.weight"),
            (P::PreFfnNorm(i), "pre_feedforward_layernorm.weight"),
            (P::PostFfnNorm(i), "post_feedforward_layernorm.weight"),
            (P::QueryNorm(i), "self_attn.q_norm.weight"),
            (P::LayerScalar(i), "layer_scalar"),
        ] {
            out.insert(role, format!("{p}.{suffix}"));
        }
        if g.ple > 0 {
            out.insert(
                P::PostPerLayerNorm(i),
                format!("{p}.post_per_layer_input_norm.weight"),
            );
        }
        if g.moe.is_some() {
            for (role, name) in [
                (P::MoePreFfnNorm(i), "pre_feedforward_layernorm_2.weight"),
                (
                    P::DensePostFfnNorm(i),
                    "post_feedforward_layernorm_1.weight",
                ),
                (P::MoePostFfnNorm(i), "post_feedforward_layernorm_2.weight"),
                (P::RouterScale(i), "router.scale"),
                (P::ExpertScale(i), "router.per_expert_scale"),
            ] {
                out.insert(role, format!("{p}.{name}"));
            }
        }
        if layer.kv_source_layer == i {
            out.insert(P::KeyNorm(i), format!("{p}.self_attn.k_norm.weight"));
        }
    }
    out
}

pub(crate) fn expert_names(g: &Geometry) -> BTreeMap<Gemma4ExpertRole, String> {
    let mut out = BTreeMap::new();
    if g.moe.is_some() {
        for i in 0..g.layers.len() {
            let p = format!("model.language_model.layers.{i}.experts");
            out.insert(Gemma4ExpertRole::GateUp(i), format!("{p}.gate_up_proj"));
            out.insert(Gemma4ExpertRole::Down(i), format!("{p}.down_proj"));
        }
    }
    out
}
