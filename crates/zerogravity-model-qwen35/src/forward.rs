//! Single Qwen forward schedule over host values or recorded values.
//! No backend owns a copied Qwen token/layer loop.

use super::{CachedLayer, LoadedQwenModel, layer_tensor};
use crate::accelerator::program::{
    AttentionGeometry, ConvDomain, ForwardInputs, ForwardPlan, KvDomain, MathematicalBackend,
    PlanBuilder, RecurrentDomain, RopeGeometry, SourceVector, StateSlot, TensorView,
};
use crate::model::LayerKind;
use crate::original_binding::QwenMatrixRole;
use crate::{Error, Result};

/// One family-neutral Qwen **layer traversal** shared by the existing
/// GGUF CPU/recorded MathematicalBackend and source-verified original HF
/// analytical execution. Layer mathematics remain profile-specific; this
/// iterator never allocates a session, buffers a model or publishes a root.
pub fn run_layer_sequence<V>(
    layers: &[LayerKind],
    mut residual: V,
    mut evaluate_layer: impl FnMut(usize, LayerKind, V) -> Result<V>,
) -> Result<V> {
    if layers.is_empty() || layers.len() > 4096 {
        return Err(Error::Limit(
            "Qwen layer schedule exceeds qualified bounds".into(),
        ));
    }
    for (index, &kind) in layers.iter().enumerate() {
        residual = evaluate_layer(index, kind, residual)?;
    }
    Ok(residual)
}

fn vector<'a>(model: &LoadedQwenModel, name: &'a str, values: &'a [f32]) -> SourceVector<'a> {
    SourceVector {
        source: model.source_identity(),
        name,
        values,
    }
}
fn project<B: MathematicalBackend>(
    model: &LoadedQwenModel,
    math: &mut B,
    role: QwenMatrixRole,
    input: &B::Value,
) -> Result<B::Value> {
    math.project(model.matrix_descriptor(role)?, input)
}
fn expect_elements<B: MathematicalBackend>(
    math: &B,
    value: &B::Value,
    expected: usize,
    message: &str,
) -> Result<()> {
    if math.elements(value) != expected {
        return Err(Error::Malformed(message.into()));
    }
    Ok(())
}

pub(super) fn forward_token<B: MathematicalBackend>(
    model: &LoadedQwenModel,
    math: &mut B,
    input: ForwardInputs<B::Index>,
    need_logits: bool,
) -> Result<Option<B::Value>> {
    model.require_execution_profile()?;
    let x = math.embed(
        model.matrix_descriptor(QwenMatrixRole::Embedding)?,
        input.token,
    )?;
    math.observe(usize::MAX, "embedding", &x)?;
    let x = run_layer_sequence(&model.geometry.layers, x, |i, kind, mut x| {
        let layer = &model.layers[i];
        let name = layer_tensor(i, "attn_norm.weight");
        let normal = math.rms(&name, &x, vector(model, &name, &layer.attn_norm), model.eps)?;
        math.observe(i, "attention_input_norm", &normal)?;
        let attn = match kind {
            LayerKind::LinearAttention => gdn_layer(model, math, i, &normal, layer)?,
            LayerKind::FullAttention => attention_layer(model, math, i, &normal, layer, input)?,
            LayerKind::SlidingAttention => {
                return Err(Error::Unsupported(
                    "Qwen dense forward has no sliding layer".into(),
                ));
            }
        };
        expect_elements(
            math,
            &attn,
            math.elements(&x),
            "attention output hidden width",
        )?;
        math.observe(i, "attention_output", &attn)?;
        x = math.add(&layer_tensor(i, "attn_residual"), &x, &attn)?;
        math.observe(i, "post_attention_residual", &x)?;
        let name = layer_tensor(i, "post_attention_norm.weight");
        let norm = math.rms(&name, &x, vector(model, &name, &layer.ffn_norm), model.eps)?;
        math.observe(i, "ffn_input_norm", &norm)?;
        let gate = project(model, math, QwenMatrixRole::FfnGate(i), &norm)?;
        let up = project(model, math, QwenMatrixRole::FfnUp(i), &norm)?;
        expect_elements(math, &gate, layer.ffn_dim, "FFN projection mismatch")?;
        expect_elements(math, &up, layer.ffn_dim, "FFN projection mismatch")?;
        let fused = math.silu(&layer_tensor(i, "ffn_silu"), &gate, Some(&up))?;
        let ff = project(model, math, QwenMatrixRole::FfnDown(i), &fused)?;
        math.observe(i, "ffn_output", &ff)?;
        x = math.add(&layer_tensor(i, "ffn_residual"), &x, &ff)?;
        math.assert_finite(i, &x)?;
        math.observe(i, "post_ffn_residual", &x)?;
        Ok(x)
    })?;
    // Private proposed metadata advances at the same point as the old CPU
    // schedule. Later final norm/vocabulary failures still abort the draft.
    math.advance_frontier()?;
    if !need_logits {
        return Ok(None);
    }
    let name = "output_norm.weight";
    let norm = math.rms(name, &x, vector(model, name, &model.final_norm), model.eps)?;
    math.observe(usize::MAX, "final_norm", &norm)?;
    Ok(Some(project(model, math, QwenMatrixRole::Output, &norm)?))
}

fn gdn_layer<B: MathematicalBackend>(
    model: &LoadedQwenModel,
    math: &mut B,
    i: usize,
    x: &B::Value,
    layer: &CachedLayer,
) -> Result<B::Value> {
    let heads = model.geometry.recurrent_value_heads.unwrap() as usize;
    let d = layer.ssm_norm.as_ref().unwrap().len();
    let width = heads * d;
    let qkv = project(model, math, QwenMatrixRole::GdnQkv(i), x)?;
    let z = project(model, math, QwenMatrixRole::GdnGate(i), x)?;
    let alpha = project(model, math, QwenMatrixRole::GdnAlpha(i), x)?;
    let beta = project(model, math, QwenMatrixRole::GdnBeta(i), x)?;
    expect_elements(math, &qkv, width * 3, "GDN projected dimensions mismatch")?;
    expect_elements(math, &z, width, "GDN projected dimensions mismatch")?;
    expect_elements(math, &alpha, heads, "GDN projected dimensions mismatch")?;
    expect_elements(math, &beta, heads, "GDN projected dimensions mismatch")?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "gdn_qkv_raw", &qkv)?;
        math.observe(i, "gdn_z_raw", &z)?;
    }
    let name = layer_tensor(i, "ssm_conv1d.weight");
    let raw = math.conv_step(
        &name,
        ConvDomain(i),
        &qkv,
        vector(model, &name, layer.conv_kernel.as_ref().unwrap()),
    )?;
    let mixed = math.silu(&layer_tensor(i, "gdn_conv_silu"), &raw, None)?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "gdn_conv_raw", &raw)?;
        math.observe(i, "gdn_conv_silu", &mixed)?;
    }
    let dt_name = layer_tensor(i, "ssm_dt.bias");
    let a_name = layer_tensor(i, "ssm_a");
    let coefficients = math.coefficients(
        &layer_tensor(i, "gdn_coefficients"),
        &alpha,
        &beta,
        vector(model, &dt_name, layer.dt.as_ref().unwrap()),
        vector(model, &a_name, layer.a.as_ref().unwrap()),
        1.0 / (d as f32).sqrt(),
    )?;
    let qk = math.view(&mixed, TensorView::vector(0, width * 2))?;
    let scale = vec![1.0 / (d as f32).sqrt(); d];
    let name = layer_tensor(i, "gdn_qk_l2");
    let normalized_qk = math.grouped_rms(
        &name,
        &qk,
        vector(model, &name, &scale),
        heads * 2,
        model.eps / d as f32,
    )?;
    let q = math.view(&normalized_qk, TensorView::vector(0, width))?;
    let k = math.view(&normalized_qk, TensorView::vector(width, width))?;
    let v = math.view(&mixed, TensorView::vector(width * 2, width))?;
    let core = math.recurrent_step(
        &layer_tensor(i, "gdn_recurrent"),
        RecurrentDomain(i),
        &q,
        &k,
        &v,
        &coefficients.packed,
        d,
    )?;
    let name = layer_tensor(i, "ssm_norm.weight");
    let normalized = math.grouped_rms(
        &name,
        &core,
        vector(model, &name, layer.ssm_norm.as_ref().unwrap()),
        heads,
        model.eps,
    )?;
    let output = math.silu(&layer_tensor(i, "gdn_output_gate"), &z, Some(&normalized))?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        if i == 0 {
            math.observe_state(i, "gdn_state", StateSlot::Recurrent(RecurrentDomain(i)))?;
        }
        math.observe(i, "gdn_q_norm", &q)?;
        math.observe(i, "gdn_k_norm", &k)?;
        math.observe(i, "gdn_v", &v)?;
        math.observe(i, "gdn_alpha", &alpha)?;
        math.observe(i, "gdn_alpha_softplus", &coefficients.softplus)?;
        math.observe(i, "gdn_gate_raw", &coefficients.gate_raw)?;
        let gain = math.view(
            &coefficients.packed,
            TensorView {
                offset: 0,
                rows: heads,
                width: 1,
                row_stride: 3,
            },
        )?;
        let gain = math.contiguous(&gain)?;
        math.observe(i, "gdn_gain", &gain)?;
        let beta = math.view(
            &coefficients.packed,
            TensorView {
                offset: 1,
                rows: heads,
                width: 1,
                row_stride: 3,
            },
        )?;
        let beta = math.contiguous(&beta)?;
        math.observe(i, "gdn_beta", &beta)?;
        math.observe(i, "gdn_core_output", &core)?;
        math.observe(i, "gdn_gated_output", &output)?;
    }
    project(model, math, QwenMatrixRole::GdnOutput(i), &output)
}

fn attention_layer<B: MathematicalBackend>(
    model: &LoadedQwenModel,
    math: &mut B,
    i: usize,
    x: &B::Value,
    layer: &CachedLayer,
    input: ForwardInputs<B::Index>,
) -> Result<B::Value> {
    let n_q = model.geometry.num_attention_heads as usize;
    let n_kv = model.geometry.num_key_value_heads as usize;
    let d = model.geometry.head_dim as usize;
    let q_proj = project(model, math, QwenMatrixRole::Query(i), x)?;
    let k_proj = project(model, math, QwenMatrixRole::Key(i), x)?;
    let v_proj = project(model, math, QwenMatrixRole::Value(i), x)?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "full_qg_raw", &q_proj)?;
        math.observe(i, "full_k_raw", &k_proj)?;
        math.observe(i, "full_v_raw", &v_proj)?;
    }
    if n_kv == 0 || !n_q.is_multiple_of(n_kv) {
        return Err(Error::Malformed("GQA tensor head count mismatch".into()));
    }
    expect_elements(math, &q_proj, n_q * d * 2, "GQA tensor head count mismatch")?;
    expect_elements(math, &k_proj, n_kv * d, "GQA tensor head count mismatch")?;
    expect_elements(math, &v_proj, n_kv * d, "GQA tensor head count mismatch")?;
    let q_view = math.view(
        &q_proj,
        TensorView {
            offset: 0,
            rows: n_q,
            width: d,
            row_stride: d * 2,
        },
    )?;
    let raw_q = math.contiguous(&q_view)?;
    let g_view = math.view(
        &q_proj,
        TensorView {
            offset: d,
            rows: n_q,
            width: d,
            row_stride: d * 2,
        },
    )?;
    let gates = math.contiguous(&g_view)?;
    let name = layer_tensor(i, "attn_q_norm.weight");
    let norm_q = math.grouped_rms(
        &name,
        &raw_q,
        vector(model, &name, layer.q_norm.as_ref().unwrap()),
        n_q,
        model.eps,
    )?;
    let name = layer_tensor(i, "attn_k_norm.weight");
    let norm_k = math.grouped_rms(
        &name,
        &k_proj,
        vector(model, &name, layer.k_norm.as_ref().unwrap()),
        n_kv,
        model.eps,
    )?;
    math.validate_kv_prefix(KvDomain(i), input.position)?;
    let geometry = AttentionGeometry {
        width: d,
        query_heads: n_q,
        kv_heads: n_kv,
    };
    let rotated = math.rope(
        &layer_tensor(i, "attn_rope_kv"),
        &norm_q,
        &norm_k,
        &v_proj,
        input.position,
        RopeGeometry {
            attention: geometry,
            rotate: 64,
            theta_scale: 10_000_000.0f32.powf(-2.0 / 64.0),
        },
    )?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "full_q_norm", &norm_q)?;
        math.observe(i, "full_k_norm", &norm_k)?;
        math.observe(i, "full_q_rope", &rotated.query)?;
        math.observe(i, "full_k_rope", &rotated.key)?;
        math.observe(i, "full_gate_raw", &gates)?;
    }
    math.append_kv(KvDomain(i), &rotated.key, &rotated.value)?;
    let qk = math.attention_qk(
        &layer_tensor(i, "attn_qk"),
        KvDomain(i),
        &rotated.query,
        input.valid_tokens,
        geometry,
        1.0 / (d as f32).sqrt(),
    )?;
    let probabilities = math.attention_softmax(
        &layer_tensor(i, "attn_softmax"),
        &qk.scores,
        n_q,
        input.valid_tokens,
    )?;
    let result = math.attention_value(
        &layer_tensor(i, "attn_value"),
        KvDomain(i),
        &probabilities,
        input.valid_tokens,
        geometry,
    )?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    {
        math.observe(i, "full_qk_dots", &qk.dots)?;
        math.observe(i, "full_scores", &qk.scores)?;
        math.observe(i, "full_probs", &probabilities)?;
        math.observe(i, "full_pregate", &result)?;
    }
    let gated = math.sigmoid_mul(&layer_tensor(i, "attn_gate"), &result, &gates)?;
    #[cfg(feature = "diagnostic-gdn-trace")]
    math.observe(i, "full_gated", &gated)?;
    project(model, math, QwenMatrixRole::AttentionOutput(i), &gated)
}

impl LoadedQwenModel {
    /// Record one bounded symbolic forward. The result contains no live
    /// device/session handles and is not evidence that any backend executed.
    pub fn forward_plan(
        &self,
        prefix_capacity: usize,
        need_logits: bool,
        capture: bool,
    ) -> Result<ForwardPlan> {
        self.require_execution_profile()?;
        if prefix_capacity == 0 || prefix_capacity > self.geometry.max_position_embeddings as usize
        {
            return Err(Error::Limit(
                "forward-plan prefix capacity exceeds model".into(),
            ));
        }
        let mut domains = Vec::new();
        for (layer, kind) in self.geometry.layers.iter().enumerate() {
            match kind {
                LayerKind::LinearAttention => {
                    domains.push(StateSlot::Conv(ConvDomain(layer)));
                    domains.push(StateSlot::Recurrent(RecurrentDomain(layer)));
                }
                LayerKind::FullAttention => domains.push(StateSlot::Kv(KvDomain(layer))),
                LayerKind::SlidingAttention => {
                    return Err(Error::Unsupported("Qwen forward-plan sliding state".into()));
                }
            }
        }
        let mut builder =
            PlanBuilder::new(self.source_identity(), prefix_capacity, domains, capture)?;
        let output = forward_token(self, &mut builder, PlanBuilder::inputs(), need_logits)?;
        builder.finish(output)
    }
}

#[cfg(all(test, feature = "cpu"))]
mod matrix_axes_tests {
    use super::*;
    use crate::accelerator::program::{MatrixAxes, SourceTensor};
    use crate::accelerator::{MatVecRegion, SourceIdentity};
    use zerogravity_core::quant::QuantKind;
    use zerogravity_core::source::{SourceTensorRef, TensorSource};

    struct OriginalRows {
        source: SourceIdentity,
        shape: [u64; 2],
        bytes: Vec<u8>,
    }
    impl TensorSource for OriginalRows {
        fn source_identity(&self) -> SourceIdentity {
            self.source
        }
        fn tensor(&self, _name: &str) -> Result<SourceTensorRef<'_>> {
            Ok(SourceTensorRef {
                kind: QuantKind::F32,
                shape: &self.shape,
                packed_bytes: &self.bytes,
            })
        }
    }

    #[test]
    fn gguf_and_hf_raw_axes_use_identical_original_cpu_row_math() {
        for (axes, shape) in [
            (MatrixAxes::InputOutput, [3, 2]),
            (MatrixAxes::OutputInput, [2, 3]),
        ] {
            let source = OriginalRows {
                source: SourceIdentity::for_new_allocation().unwrap(),
                shape,
                bytes: [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            };
            let tensor = SourceTensor::from_source(&source, "matrix", axes).unwrap();
            let actual = crate::weights::source_reference_projection(MatVecRegion {
                source_identity: tensor.source,
                tensor_name: &tensor.name,
                kind: tensor.kind,
                columns: tensor.columns,
                rows: tensor.rows,
                packed_weights: source.tensor("matrix").unwrap().packed_bytes,
                activation: &[7.0, 8.0, 9.0],
            })
            .unwrap();
            assert_eq!(actual, [50.0, 122.0]);
            assert_eq!(source.shape, shape);
        }
    }
}
