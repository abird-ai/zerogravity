//! Single Qwen forward schedule over host values or recorded values.
//! No backend owns a copied Qwen token/layer loop.

use super::{CachedLayer, LoadedQwenModel, layer_tensor};
use crate::accelerator::program::{
    AttentionGeometry, ConvDomain, ForwardInputs, ForwardPlan, KvDomain, MathematicalBackend,
    OrdinaryRmsContract, PlanBuilder, RecurrentDomain, RopeGeometry, SourceVector, StateSlot,
    TensorView,
};
use crate::config::LayerKind;
use crate::operator_slice::OperatorSliceSource;
use crate::source::binding::{QwenMatrixRole, QwenParameterRole};
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
    model: &impl OperatorSliceSource,
    math: &mut B,
    role: QwenMatrixRole,
    input: &B::Value,
) -> Result<B::Value> {
    math.project_profiled(
        model.matrix_descriptor(role)?,
        input,
        model.projection_contract(),
    )
}

#[cfg(all(feature = "source-binding", target_os = "linux"))]
pub(crate) fn pure_projection_slice<B: MathematicalBackend>(
    source: &impl OperatorSliceSource,
    math: &mut B,
    role: QwenMatrixRole,
    input: &B::Value,
) -> Result<B::Value> {
    use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
    let tensor = source.matrix_descriptor(role)?;
    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || math.precision(input) != ActivationPrecision::Bf16WidenedF32
        || math.elements(input) != tensor.columns
    {
        return Err(Error::State(
            "pure projection source profile/input geometry".into(),
        ));
    }
    let rows = tensor.rows;
    let output = math.project_profiled(tensor, input, source.projection_contract())?;
    if math.elements(&output) != rows
        || math.precision(&output) != ActivationPrecision::Bf16WidenedF32
    {
        return Err(Error::State("pure projection complete typed output".into()));
    }
    Ok(output)
}

#[cfg(all(feature = "source-binding", target_os = "linux"))]
pub(crate) fn pure_ordinary_rms_slice<B: MathematicalBackend>(
    source: &impl OperatorSliceSource,
    math: &mut B,
    role: QwenParameterRole,
    input: &B::Value,
) -> Result<B::Value> {
    use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || source.rms_epsilon().to_bits() != 1e-6f32.to_bits()
        || math.precision(input) != ActivationPrecision::Bf16WidenedF32
    {
        return Err(Error::State(
            "pure ordinary RMS source profile/precision".into(),
        ));
    }
    let elements = math.elements(input);
    let hidden = usize::try_from(source.geometry().hidden_size)
        .map_err(|_| Error::Limit("pure ordinary RMS hidden width".into()))?;
    if elements != hidden {
        return Err(Error::Malformed(
            "pure ordinary RMS exact hidden input".into(),
        ));
    }
    let output = math.ordinary_rms_profiled(
        source.parameter_descriptor(role)?,
        input,
        OrdinaryRmsContract::TorchCpuBf16Avx2,
    )?;
    if math.elements(&output) != elements
        || math.precision(&output) != ActivationPrecision::Bf16WidenedF32
    {
        return Err(Error::State(
            "pure ordinary RMS complete typed output".into(),
        ));
    }
    Ok(output)
}

/// Final decoder FFN and complete head on the caller's post-attention value.
/// Same backend/schedule; no layer traversal, state issuer or hidden producer.
#[cfg(all(feature = "source-binding", target_os = "linux"))]
pub(crate) fn output_tail_slice<B: MathematicalBackend>(
    source: &crate::source::binding::OriginalWindowOperators,
    math: &mut B,
    input: &B::Value,
) -> Result<crate::operator_slice::output_tail::OutputTailValues<B::Value>> {
    use crate::operator_slice::output_tail::OutputTailValues;
    let layer = source.output_tail_layer()?;
    let SourceFfnValues {
        normalized_ffn,
        gate,
        up,
        gated,
        branch,
        residual,
    } = source_ffn_layer(source, math, layer, input)?;
    let normalized =
        pure_ordinary_rms_slice(source, math, QwenParameterRole::FinalNorm, &residual)?;
    let logits = pure_projection_slice(source, math, QwenMatrixRole::Output, &normalized)?;
    Ok(OutputTailValues {
        normalized_ffn,
        gate,
        up,
        gated,
        branch,
        residual,
        normalized,
        logits,
    })
}
#[cfg(all(feature = "source-binding", target_os = "linux"))]
struct SourceFfnValues<V> {
    normalized_ffn: V,
    gate: V,
    up: V,
    gated: V,
    branch: V,
    residual: V,
}
#[cfg(all(feature = "source-binding", target_os = "linux"))]
fn source_ffn_layer<B: MathematicalBackend>(
    source: &crate::source::binding::OriginalWindowOperators,
    math: &mut B,
    layer: usize,
    input: &B::Value,
) -> Result<SourceFfnValues<B::Value>> {
    let normalized_ffn =
        pure_ordinary_rms_slice(source, math, QwenParameterRole::FfnNorm(layer), input)?;
    let gate = pure_projection_slice(
        source,
        math,
        QwenMatrixRole::FfnGate(layer),
        &normalized_ffn,
    )?;
    let up = pure_projection_slice(source, math, QwenMatrixRole::FfnUp(layer), &normalized_ffn)?;
    let gated = math.torch_cpu_bf16_silu_mul("original.output_tail.silu_mul", &gate, &up)?;
    let branch = pure_projection_slice(source, math, QwenMatrixRole::FfnDown(layer), &gated)?;
    let residual = math.torch_cpu_bf16_add("original.output_tail.residual", input, &branch)?;
    Ok(SourceFfnValues {
        normalized_ffn,
        gate,
        up,
        gated,
        branch,
        residual,
    })
}

/// Diagnostic CPU adapter over the same recordable source traversal.
#[cfg(all(feature = "source-binding", target_os = "linux", feature = "cpu"))]
pub(crate) fn window_diagnostic_token(
    source: &crate::source::binding::OriginalWindowOperators,
    math: &mut crate::host_math::HostMath<'_, '_, '_>,
    states: &mut [crate::source::binding::DiagnosticLayerState],
    token: u32,
    position: usize,
    trace: Option<crate::source::binding::WindowDiagnosticTrace<'_>>,
) -> Result<crate::host_math::HostValue> {
    window_source_token(source, math, states, token as usize, position, trace)
}

/// One original family traversal; state producers belong to the backend.
#[cfg(all(feature = "source-binding", target_os = "linux", feature = "cpu"))]
pub(crate) fn window_source_token<B: MathematicalBackend>(
    source: &crate::source::binding::OriginalWindowOperators,
    math: &mut B,
    states: &mut [crate::source::binding::DiagnosticLayerState<B::Value>],
    token: B::Index,
    position: usize,
    trace: Option<crate::source::binding::WindowDiagnosticTrace<'_, B::Value>>,
) -> Result<B::Value> {
    if position >= 64 || states.len() != source.geometry().layers.len() {
        return Err(Error::State("diagnostic full actual state/position".into()));
    }
    let prefix = trace.as_ref().is_some_and(|t| t.prefix);
    let layers = &source.geometry().layers[..if prefix {
        10
    } else {
        source.geometry().layers.len()
    }];
    let mut adapter = SourceDecoder {
        source,
        math,
        states,
        token: Some(token),
        position,
        trace,
        prefix,
        evolved: None,
    };
    crate::hybrid_decoder::compose(layers, &mut adapter)
}

/// One final source-bound attention block followed by the existing output tail.
/// KV is an explicit immutable diagnostic prefix, never an engine StateRoot.
#[cfg(all(feature = "source-binding", target_os = "linux", feature = "cpu"))]
pub(crate) fn attention_output_tail_slice<B: MathematicalBackend>(
    source: &crate::source::binding::OriginalWindowOperators,
    math: &mut B,
    input: &B::Value,
    prior: Option<(&B::Value, &B::Value)>,
    position: usize,
) -> Result<crate::operator_slice::attention_tail::AttentionTailValues<B::Value>> {
    use crate::accelerator::program::TorchCpuAttentionInputs;
    use crate::operator_slice::attention_tail::AttentionTailValues;
    let s = source.attention_tail_spec()?;
    let normalized_input =
        pure_ordinary_rms_slice(source, math, QwenParameterRole::InputNorm(s.layer), input)?;
    let projections = attention_projection_norm_slice(source, math, s.layer, &normalized_input)?;
    let attention = math.torch_cpu_bf16_attention(
        "original.final_attention",
        TorchCpuAttentionInputs {
            query: &projections.normalized_query,
            key: &projections.normalized_key,
            value: &projections.value_projection,
            gate: &projections.gate,
            prior,
            geometry: s.geometry,
            position,
        },
    )?;
    let branch = pure_projection_slice(
        source,
        math,
        QwenMatrixRole::AttentionOutput(s.layer),
        &attention.gated,
    )?;
    let residual = math.torch_cpu_bf16_add("original.final_attention.residual", input, &branch)?;
    let tail = output_tail_slice(source, math, &residual)?;
    Ok(AttentionTailValues {
        normalized_input,
        projections,
        attention,
        branch,
        residual,
        tail,
    })
}

/// Pure fragment of this same source-role schedule. It never touches state,
/// normalizes a hidden activation or represents a complete Qwen layer/token.
pub(crate) fn embedding_projection_slice<B: MathematicalBackend>(
    model: &impl OperatorSliceSource,
    math: &mut B,
    token: B::Index,
    role: QwenMatrixRole,
) -> Result<(B::Value, B::Value)> {
    let expected = match model.program_numerical_profile() {
        crate::accelerator::program::ProgramNumericalProfile::GgufSourceV1 => {
            crate::accelerator::program::ActivationPrecision::F32
        }
        crate::accelerator::program::ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1 => {
            crate::accelerator::program::ActivationPrecision::Bf16WidenedF32
        }
        crate::accelerator::program::ProgramNumericalProfile::SourceCudaSm86Bf16V1 => {
            return Err(Error::Unsupported(
                "existing embedding/projection slice has no vLLM CUDA contract".into(),
            ));
        }
        crate::accelerator::program::ProgramNumericalProfile::AnalyticF32TextV1 => {
            return Err(Error::Unsupported(
                "Qwen slice has no analytic F32 text contract".into(),
            ));
        }
        crate::accelerator::program::ProgramNumericalProfile::DirectGammaBf16TextCandidateV1 => {
            return Err(Error::Unsupported(
                "Qwen slice has no Gemma BF16 text contract".into(),
            ));
        }
    };
    let embedding = math.embed_profiled(
        model.matrix_descriptor(QwenMatrixRole::Embedding)?,
        token,
        model.embedding_contract(),
    )?;
    let projected = project(model, math, role, &embedding)?;

    if math.precision(&embedding) != expected || math.precision(&projected) != expected {
        return Err(Error::State(
            "source operator slice activation precision".into(),
        ));
    }
    Ok((embedding, projected))
}
/// Pure original layer0 input prefix through this same backend/schedule seam.
/// No layer loop, cache proposal, token advance or full-HF execution admission.
pub(crate) fn embedding_input_norm_projection_slice<B: MathematicalBackend>(
    model: &impl OperatorSliceSource,
    math: &mut B,
    token: B::Index,
) -> Result<(B::Value, B::Value, B::Value)> {
    use crate::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
    if model.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || model.rms_epsilon().to_bits() != 1e-6f32.to_bits()
    {
        return Err(Error::Unsupported(
            "original ordinary RMS slice profile/epsilon".into(),
        ));
    }
    let embedding = math.embed_profiled(
        model.matrix_descriptor(QwenMatrixRole::Embedding)?,
        token,
        model.embedding_contract(),
    )?;
    let normalized = math.ordinary_rms_profiled(
        model.parameter_descriptor(QwenParameterRole::InputNorm(0))?,
        &embedding,
        OrdinaryRmsContract::TorchCpuBf16Avx2,
    )?;
    let projected = project(model, math, QwenMatrixRole::GdnQkv(0), &normalized)?;
    if [&embedding, &normalized, &projected]
        .iter()
        .any(|value| math.precision(value) != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State(
            "ordinary RMS slice activation precision".into(),
        ));
    }
    Ok((embedding, normalized, projected))
}

/// Pure prefix of the existing attention role schedule, before rotary/state.
pub(crate) fn attention_projection_norm_slice<B: MathematicalBackend>(
    model: &impl OperatorSliceSource,
    math: &mut B,
    layer: usize,
    input: &B::Value,
) -> Result<crate::operator_slice::attention::AttentionProjectionValues<B::Value>> {
    use crate::accelerator::program::ActivationPrecision;
    use crate::operator_slice::attention::{AttentionProjectionValues, attention_roles};
    let roles = attention_roles(model, layer)?;
    if math.elements(input) != roles.hidden
        || math.precision(input) != ActivationPrecision::Bf16WidenedF32
    {
        return Err(Error::Malformed(
            "attention slice typed full hidden input".into(),
        ));
    }
    let query_gate_projection =
        math.project_profiled(roles.query, input, model.projection_contract())?;
    let key_projection = math.project_profiled(roles.key, input, model.projection_contract())?;
    let value_projection =
        math.project_profiled(roles.value, input, model.projection_contract())?;
    for (value, elements) in [
        (&query_gate_projection, roles.query_heads * roles.width * 2),
        (&key_projection, roles.kv_heads * roles.width),
        (&value_projection, roles.kv_heads * roles.width),
    ] {
        expect_elements(math, value, elements, "attention slice projected geometry")?;
    }
    let query_view = math.view(
        &query_gate_projection,
        TensorView {
            offset: 0,
            rows: roles.query_heads,
            width: roles.width,
            row_stride: roles.width * 2,
        },
    )?;
    let query = math.contiguous(&query_view)?;
    let gate_view = math.view(
        &query_gate_projection,
        TensorView {
            offset: roles.width,
            rows: roles.query_heads,
            width: roles.width,
            row_stride: roles.width * 2,
        },
    )?;
    let gate = math.contiguous(&gate_view)?;
    let normalized_query = math.ordinary_rms_profiled(
        roles.query_norm,
        &query,
        OrdinaryRmsContract::TorchCpuBf16Avx2,
    )?;
    let normalized_key = math.ordinary_rms_profiled(
        roles.key_norm,
        &key_projection,
        OrdinaryRmsContract::TorchCpuBf16Avx2,
    )?;
    if [
        &query_gate_projection,
        &key_projection,
        &value_projection,
        &query,
        &gate,
        &normalized_query,
        &normalized_key,
    ]
    .iter()
    .any(|value| math.precision(value) != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State("attention slice output precision".into()));
    }
    Ok(AttentionProjectionValues {
        query_gate_projection,
        key_projection,
        value_projection,
        query,
        gate,
        normalized_query,
        normalized_key,
    })
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

pub(super) fn forward_token<'a, B: MathematicalBackend>(
    model: impl Into<crate::QwenModelMetadata<'a>>,
    math: &mut B,
    input: ForwardInputs<B::Index>,
    need_logits: bool,
) -> Result<Option<B::Value>> {
    let source = model.into();
    source.require_execution_profile()?;
    // Legacy learned-cache/packed-source math remains the unchanged GGUF route.
    // Verified windows never fabricate this interface or select a CPU fallback.
    let model = source.legacy_loaded()?;
    let x = math.embed_profiled(
        model.matrix_descriptor(QwenMatrixRole::Embedding)?,
        input.token,
        model.embedding_contract(),
    )?;
    math.observe(usize::MAX, "embedding", &x)?;
    let x = run_layer_sequence(&source.geometry().layers, x, |i, kind, mut x| {
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
            let actual = zerogravity_cpu::source::source_reference_projection(MatVecRegion {
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

/// Pure original window only; proposed history is not a StateEffect.
pub(crate) fn gdn_convolution_silu_slice<B: MathematicalBackend>(
    source: &impl OperatorSliceSource,
    math: &mut B,
    layer: usize,
    input: &B::Value,
    history: &B::Value,
) -> Result<crate::accelerator::program::CausalConvValues<B::Value>> {
    use crate::accelerator::program::{ActivationPrecision, CausalConvSiluContract};
    let (parameter, channels) = crate::operator_slice::gdn_conv::convolution_role(source, layer)?;
    if math.elements(input) != channels
        || math.elements(history) != channels * 3
        || [input, history]
            .iter()
            .any(|v| math.precision(v) != ActivationPrecision::Bf16WidenedF32)
    {
        return Err(Error::State(
            "original convolution slice geometry/precision".into(),
        ));
    }
    let values = math.causal_conv_silu_profiled(
        parameter,
        input,
        history,
        CausalConvSiluContract::TorchCpuBf16Avx2,
    )?;
    for (value, elements) in [
        (&values.raw, channels),
        (&values.activated, channels),
        (&values.next_history, channels * 3),
    ] {
        if math.elements(value) != elements
            || math.precision(value) != ActivationPrecision::Bf16WidenedF32
        {
            return Err(Error::State(
                "original convolution region result geometry/precision".into(),
            ));
        }
    }
    Ok(values)
}

#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
#[path = "forward/bulk_f32.rs"]
mod bulk_f32;
#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
pub(crate) use bulk_f32::bulk_f32_slice;

#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
#[path = "forward/zen_ssm_out.rs"]
mod zen_ssm_out;
#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
pub(crate) use zen_ssm_out::zen_ssm_out_slice;

#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn gguf_prefix_slice<M: MathematicalBackend>(
    math: &mut M,
    descriptor: &crate::operator_slice::bulk_f32::source_bindings::PrefixBinding,
) -> Result<(M::Value, M::Value)> {
    use crate::operator_slice::bulk_f32::source_bindings::one;
    let embedding = one(math, descriptor.embedding_operator(), &[])?;
    let normalized = one(
        math,
        descriptor.rms_operator(),
        std::slice::from_ref(&embedding),
    )?;
    Ok((embedding, normalized))
}
#[cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn gguf_gdn_upstream_slice<M: MathematicalBackend>(
    math: &mut M,
    descriptor: &crate::operator_slice::bulk_f32::source_bindings::GdnUpstreamBinding,
) -> Result<crate::operator_slice::bulk_f32::source_bindings::UpstreamValues<M::Value>> {
    use crate::operator_slice::bulk_f32::source_bindings::{GdnProjection, UpstreamValues, one};
    let input = one(math, descriptor.input_operator(), &[])?;
    let qkv = one(
        math,
        descriptor.projection_operator(GdnProjection::Qkv),
        std::slice::from_ref(&input),
    )?;
    let z = one(
        math,
        descriptor.projection_operator(GdnProjection::Z),
        std::slice::from_ref(&input),
    )?;
    let convolution = one(
        math,
        descriptor.convolution_operator(),
        std::slice::from_ref(&qkv),
    )?;
    let operands = one(
        math,
        descriptor.qk_operator(),
        std::slice::from_ref(&convolution),
    )?;
    Ok(UpstreamValues {
        qkv,
        z,
        convolution,
        operands,
    })
}

#[cfg(all(feature = "source-binding", target_os = "linux", feature = "cpu"))]
struct SourceDecoder<'a, 't, B: MathematicalBackend> {
    source: &'a crate::source::binding::OriginalWindowOperators,
    math: &'a mut B,
    states: &'a mut [crate::source::binding::DiagnosticLayerState<B::Value>],
    token: Option<B::Index>,
    position: usize,
    trace: Option<crate::source::binding::WindowDiagnosticTrace<'t, B::Value>>,
    prefix: bool,
    evolved: Option<(
        usize,
        crate::source::binding::DiagnosticLayerState<B::Value>,
    )>,
}
#[cfg(all(feature = "source-binding", target_os = "linux", feature = "cpu"))]
impl<B: MathematicalBackend> SourceDecoder<'_, '_, B> {
    fn observe(
        &mut self,
        layer: usize,
        stage: crate::source::binding::WindowDiagnosticLayer9Stage,
        value: &B::Value,
    ) {
        if layer == 9
            && let Some(trace) = self.trace.as_mut()
            && trace.layer9
        {
            trace.boundaries.push((
                crate::source::binding::WindowDiagnosticBoundary::Layer9(stage),
                value.clone(),
            ));
        }
    }
}
#[cfg(all(feature = "source-binding", target_os = "linux", feature = "cpu"))]
impl<B: MathematicalBackend> crate::hybrid_decoder::HybridDecoder for SourceDecoder<'_, '_, B> {
    type Value = B::Value;
    type GdnInputs = [B::Value; 4];
    type AttentionInputs = crate::operator_slice::attention::AttentionProjectionValues<B::Value>;
    type FfnInputs = [B::Value; 2];
    fn embedding(&mut self) -> Result<B::Value> {
        let token = self
            .token
            .take()
            .ok_or_else(|| Error::State("source embedding invoked twice".into()))?;
        let x = self.math.embed_profiled(
            self.source.matrix_descriptor(QwenMatrixRole::Embedding)?,
            token,
            self.source.embedding_contract(),
        )?;
        self.math.observe(0, "embedding", &x)?;
        if let Some(t) = self.trace.as_mut() {
            t.boundaries.push((
                crate::source::binding::WindowDiagnosticBoundary::Embedding,
                x.clone(),
            ));
        }
        Ok(x)
    }
    fn input_norm(
        &mut self,
        layer: usize,
        branch: B::Value,
        residual: Option<B::Value>,
    ) -> Result<(B::Value, B::Value)> {
        if residual.is_some() {
            return Err(Error::State(
                "source Torch residual must materialize at original BF16 boundary".into(),
            ));
        }
        let normal = pure_ordinary_rms_slice(
            self.source,
            self.math,
            QwenParameterRole::InputNorm(layer),
            &branch,
        )?;
        self.observe(
            layer,
            crate::source::binding::WindowDiagnosticLayer9Stage::InputNorm,
            &normal,
        );
        Ok((normal, branch))
    }
    fn gdn_inputs(&mut self, layer: usize, normal: &B::Value) -> Result<Self::GdnInputs> {
        use crate::source::binding::WindowDiagnosticLayer9Stage as S;
        let roles = crate::hybrid_decoder::ProjectionGroup::GdnInput.roles(layer);
        let coefficients = crate::hybrid_decoder::ProjectionGroup::GdnCoefficients.roles(layer);
        let qkv = pure_projection_slice(self.source, self.math, roles[0], normal)?;
        let gate = pure_projection_slice(self.source, self.math, roles[1], normal)?;
        let alpha = pure_projection_slice(self.source, self.math, coefficients[1], normal)?;
        let beta = pure_projection_slice(self.source, self.math, coefficients[0], normal)?;
        self.observe(layer, S::Qkv, &qkv);
        self.observe(layer, S::Z, &gate);
        self.observe(layer, S::B, &beta);
        self.observe(layer, S::A, &alpha);
        Ok([qkv, gate, alpha, beta])
    }
    fn gdn_evolve(
        &mut self,
        layer: usize,
        [qkv, gate, alpha, beta]: Self::GdnInputs,
    ) -> Result<B::Value> {
        use crate::accelerator::program::{TorchCpuGdnInputs, TorchCpuGdnPhase};
        use crate::source::binding::{DiagnosticLayerState, WindowDiagnosticLayer9Stage as S};
        let DiagnosticLayerState::Gdn {
            history,
            full_history,
            state,
        } = &self.states[layer]
        else {
            return Err(Error::State(
                "diagnostic source GDN state domain mismatch".into(),
            ));
        };
        let proposed_full = full_history
            .as_ref()
            .map(|full| {
                self.math
                    .torch_cpu_full_convolution_cache(full, history, &qkv)
            })
            .transpose()?;
        let core = self.math.torch_cpu_bf16_gdn(
            &self.source.gdn_contract(layer)?,
            TorchCpuGdnInputs {
                qkv: &qkv,
                gate: &gate,
                alpha: &alpha,
                beta: &beta,
                history,
                state,
                phase: if self.position == 0 {
                    TorchCpuGdnPhase::InitialOne
                } else {
                    TorchCpuGdnPhase::CachedOne
                },
            },
        )?;
        self.observe(layer, S::Core, &core.core);
        self.observe(layer, S::NormGate, &gate);
        self.observe(layer, S::Gated, &core.gated);
        self.evolved = Some((
            layer,
            DiagnosticLayerState::Gdn {
                history: core.history,
                full_history: proposed_full,
                state: core.state,
            },
        ));
        Ok(core.gated)
    }
    fn attention_inputs(
        &mut self,
        layer: usize,
        normal: &B::Value,
    ) -> Result<Self::AttentionInputs> {
        // Preserve prior geometry refusal before the original projected producers.
        let r = crate::operator_slice::attention::attention_roles(self.source, layer)?;
        let crate::source::binding::DiagnosticLayerState::Attention { keys, values } =
            &self.states[layer]
        else {
            return Err(Error::State(
                "diagnostic source attention state domain mismatch".into(),
            ));
        };
        match (keys.as_ref(), values.as_ref()) {
            (None, None) if self.position == 0 => {}
            (Some(k), Some(v))
                if self.math.elements(k) == self.position * r.kv_heads * r.width
                    && self.math.elements(v) == self.math.elements(k) => {}
            _ => {
                return Err(Error::State(
                    "diagnostic exact source KV producer/frontier".into(),
                ));
            }
        }
        attention_projection_norm_slice(self.source, self.math, layer, normal)
    }
    fn attention_evolve(&mut self, layer: usize, p: Self::AttentionInputs) -> Result<B::Value> {
        use crate::accelerator::program::{TorchCpuAttentionGeometry, TorchCpuAttentionInputs};
        use crate::source::binding::DiagnosticLayerState;
        let r = crate::operator_slice::attention::attention_roles(self.source, layer)?;
        let DiagnosticLayerState::Attention { keys, values } = &self.states[layer] else {
            return Err(Error::State(
                "diagnostic source attention state domain mismatch".into(),
            ));
        };
        let prior = keys.as_ref().zip(values.as_ref());
        let c = self.source.binding().config();
        let a = self.math.torch_cpu_bf16_attention(
            "original.model.attention",
            TorchCpuAttentionInputs {
                query: &p.normalized_query,
                key: &p.normalized_key,
                value: &p.value_projection,
                gate: &p.gate,
                prior,
                position: self.position,
                geometry: TorchCpuAttentionGeometry {
                    query_heads: r.query_heads,
                    kv_heads: r.kv_heads,
                    width: r.width,
                    rotary: c.rotary_dimensions() as usize,
                    theta: c.rope_theta() as f32,
                },
            },
        )?;
        self.evolved = Some((
            layer,
            DiagnosticLayerState::Attention {
                keys: Some(a.keys),
                values: Some(a.values),
            },
        ));
        Ok(a.gated)
    }
    fn attention_output(
        &mut self,
        layer: usize,
        kind: LayerKind,
        value: B::Value,
    ) -> Result<B::Value> {
        let role = if kind == LayerKind::FullAttention {
            QwenMatrixRole::AttentionOutput(layer)
        } else {
            QwenMatrixRole::GdnOutput(layer)
        };
        let branch = pure_projection_slice(self.source, self.math, role, &value)?;
        if kind == LayerKind::LinearAttention {
            self.observe(
                layer,
                crate::source::binding::WindowDiagnosticLayer9Stage::GdnOut,
                &branch,
            );
        }
        let (actual, state) = self
            .evolved
            .take()
            .ok_or_else(|| Error::State("source layer state producer absent".into()))?;
        if actual != layer {
            return Err(Error::State("source layer state producer identity".into()));
        }
        self.states[layer] = state;
        Ok(branch)
    }
    fn post_attention_norm(
        &mut self,
        layer: usize,
        branch: B::Value,
        carried: B::Value,
    ) -> Result<(B::Value, B::Value)> {
        use crate::source::binding::WindowDiagnosticLayer9Stage as S;
        let residual =
            self.math
                .torch_cpu_bf16_add("original.model.attention.residual", &carried, &branch)?;
        self.observe(layer, S::PostAttention, &residual);
        let normal = pure_ordinary_rms_slice(
            self.source,
            self.math,
            QwenParameterRole::FfnNorm(layer),
            &residual,
        )?;
        self.observe(layer, S::PostNorm, &normal);
        Ok((normal, residual))
    }
    fn ffn_inputs(&mut self, layer: usize, normal: &B::Value) -> Result<Self::FfnInputs> {
        use crate::source::binding::WindowDiagnosticLayer9Stage as S;
        let roles = crate::hybrid_decoder::ProjectionGroup::FfnInput.roles(layer);
        let gate = pure_projection_slice(self.source, self.math, roles[0], normal)?;
        let up = pure_projection_slice(self.source, self.math, roles[1], normal)?;
        self.observe(layer, S::FfnGate, &gate);
        self.observe(layer, S::FfnUp, &up);
        Ok([gate, up])
    }
    fn ffn_activate(&mut self, layer: usize, [gate, up]: Self::FfnInputs) -> Result<B::Value> {
        let gated =
            self.math
                .torch_cpu_bf16_silu_mul("original.output_tail.silu_mul", &gate, &up)?;
        self.observe(
            layer,
            crate::source::binding::WindowDiagnosticLayer9Stage::FfnGated,
            &gated,
        );
        Ok(gated)
    }
    fn ffn_output(&mut self, layer: usize, value: B::Value) -> Result<B::Value> {
        let branch = pure_projection_slice(
            self.source,
            self.math,
            QwenMatrixRole::FfnDown(layer),
            &value,
        )?;
        self.observe(
            layer,
            crate::source::binding::WindowDiagnosticLayer9Stage::FfnDown,
            &branch,
        );
        Ok(branch)
    }
    fn finish_layer(
        &mut self,
        layer: usize,
        branch: B::Value,
        carried: B::Value,
    ) -> Result<(B::Value, Option<B::Value>)> {
        let out =
            self.math
                .torch_cpu_bf16_add("original.output_tail.residual", &carried, &branch)?;
        self.math.assert_finite(layer, &out)?;
        self.math.observe(layer, "layer_output", &out)?;
        if let Some(t) = self.trace.as_mut() {
            t.boundaries.push((
                crate::source::binding::WindowDiagnosticBoundary::Layer(layer),
                out.clone(),
            ));
        }
        Ok((out, None))
    }
    fn finish_model(&mut self, branch: B::Value, residual: Option<B::Value>) -> Result<B::Value> {
        if residual.is_some() || self.evolved.is_some() {
            return Err(Error::State(
                "source decoder has unpublished adapter state".into(),
            ));
        }
        if self.prefix {
            return Ok(branch);
        }
        let normalized = pure_ordinary_rms_slice(
            self.source,
            self.math,
            QwenParameterRole::FinalNorm,
            &branch,
        )?;
        self.math.observe(
            self.source.geometry().layers.len() - 1,
            "final_norm",
            &normalized,
        )?;
        if let Some(t) = self.trace.as_mut() {
            t.boundaries.push((
                crate::source::binding::WindowDiagnosticBoundary::FinalNorm,
                normalized.clone(),
            ));
        }
        pure_projection_slice(self.source, self.math, QwenMatrixRole::Output, &normalized)
    }
}
