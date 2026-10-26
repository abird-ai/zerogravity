//! Semantic fixtures only; no trained source/device/root authority.
use super::*;
use zerogravity_core::accelerator::program::*;
use zerogravity_core::quant::QuantKind;
use zerogravity_model_qwen35::plan::QwenPlanBuilder;
use zerogravity_model_qwen35::plan::{
    PromptCall, PromptSpan, PromptWorkload, QwenEmbedding, QwenLayer, QwenOperation, QwenOperator,
    QwenPlan, QwenProjection, QwenRmsSite, QwenSourceCall, QwenSourceProfile,
};
fn members(p: QwenProjection, profile: QwenSourceProfile) -> Vec<(&'static str, usize)> {
    match p {
        QwenProjection::Qkvz => vec![
            ("linear_attn.in_proj_qkv.weight", 6144),
            ("linear_attn.in_proj_z.weight", 2048),
        ],
        QwenProjection::Ba => vec![
            ("linear_attn.in_proj_b.weight", 16),
            ("linear_attn.in_proj_a.weight", 16),
        ],
        QwenProjection::GdnOut => vec![("linear_attn.out_proj.weight", profile.hidden_size())],
        QwenProjection::AttentionQkv => vec![
            ("self_attn.q_proj.weight", 4096),
            ("self_attn.k_proj.weight", 512),
            ("self_attn.v_proj.weight", 512),
        ],
        QwenProjection::AttentionOut => vec![("self_attn.o_proj.weight", profile.hidden_size())],
        QwenProjection::GateUp => vec![
            ("mlp.gate_proj.weight", profile.intermediate_size()),
            ("mlp.up_proj.weight", profile.intermediate_size()),
        ],
        QwenProjection::Down => vec![("mlp.down_proj.weight", profile.hidden_size())],
        QwenProjection::Head => vec![("model.language_model.embed_tokens.weight", 248320)],
    }
}
pub(super) fn call(
    source: SourceIdentity,
    profile: QwenSourceProfile,
    workload: PromptWorkload,
    base: Option<usize>,
    id: u32,
) -> PromptCall {
    let span = match base {
        None => PromptSpan::prefill(source, workload, &vec![id; workload.prompt_tokens()]),
        Some(base) => PromptSpan::cached(source, workload, base, id),
    }
    .unwrap();
    PromptCall::new(profile, span).unwrap()
}
fn descriptor(
    call: &QwenSourceCall,
    layer: Option<QwenLayer>,
    kind: QwenOperation,
) -> QwenOperator {
    let name = |suffix: &str| {
        format!(
            "model.language_model.layers.{}.{suffix}",
            layer.unwrap().index()
        )
    };
    let h = call.model_profile().hidden_size();
    let matrices = if let QwenOperation::Projection(p) = kind {
        members(p, call.model_profile())
            .iter()
            .map(|&(suffix, rows)| SourceTensor {
                source: call.source(),
                name: if p == QwenProjection::Head {
                    suffix.into()
                } else {
                    name(suffix)
                },
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                rows,
                columns: call.model_profile().projection_input(p),
            })
            .collect()
    } else {
        Vec::new()
    };
    let parameters = match kind {
        QwenOperation::InputRms | QwenOperation::ResidualRms(QwenRmsSite::Input) => vec![(
            name("input_layernorm.weight"),
            QuantKind::Bf16,
            vec![h as u64],
            h * 2,
        )],
        QwenOperation::ResidualRms(QwenRmsSite::PostAttention) => vec![(
            name("post_attention_layernorm.weight"),
            QuantKind::Bf16,
            vec![h as u64],
            h * 2,
        )],
        QwenOperation::ResidualRms(QwenRmsSite::Final) => vec![(
            "model.language_model.norm.weight".into(),
            QuantKind::Bf16,
            vec![h as u64],
            h * 2,
        )],
        QwenOperation::GdnConv => vec![(
            name("linear_attn.conv1d.weight"),
            QuantKind::Bf16,
            vec![6144, 1, 4],
            49152,
        )],
        QwenOperation::GdnRecurrent => [
            ("linear_attn.A_log", QuantKind::F32, 16, 64),
            ("linear_attn.dt_bias", QuantKind::Bf16, 16, 32),
            ("linear_attn.norm.weight", QuantKind::F32, 128, 512),
        ]
        .into_iter()
        .map(|(suffix, kind, n, bytes)| (name(suffix), kind, vec![n], bytes))
        .collect(),
        QwenOperation::AttentionQk => ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
            .into_iter()
            .map(|suffix| (name(suffix), QuantKind::Bf16, vec![256], 512))
            .collect(),
        _ => Vec::new(),
    }
    .into_iter()
    .map(|(name, kind, shape, bytes)| SourceParameter {
        source: call.source(),
        name,
        kind,
        shape,
        bytes,
    })
    .collect();
    QwenOperator::new(call.clone(), layer, kind, matrices, parameters).unwrap()
}
pub(super) fn record(call: &PromptCall) -> QwenPlan {
    let call = call.as_source_call();
    let mut builder = QwenPlanBuilder::new_model(call.clone()).unwrap();
    let mut branch = builder
        .source_embedding(
            &QwenEmbedding::for_call(
                SourceTensor {
                    source: call.source(),
                    name: "model.language_model.embed_tokens.weight".into(),
                    kind: QuantKind::Bf16,
                    axes: MatrixAxes::OutputInput,
                    rows: 248320,
                    columns: call.model_profile().hidden_size(),
                },
                call,
            )
            .unwrap(),
        )
        .unwrap();
    let mut residual = None;
    let mut boundaries = Vec::with_capacity(24);
    let mut apply = |layer, kind, inputs: &[PlanValue]| {
        builder
            .source_operator(&descriptor(call, layer, kind), inputs)
            .unwrap()
    };
    for i in 0..24 {
        let actual_layer = call.layer(i).unwrap();
        let layer = Some(actual_layer);
        let (normal, carried) = if i == 0 {
            (apply(layer, QwenOperation::InputRms, &[branch])[0], branch)
        } else {
            let out = apply(
                layer,
                QwenOperation::ResidualRms(QwenRmsSite::Input),
                &[branch, residual.unwrap()],
            );
            (out[0], out[1])
        };
        let attention = if actual_layer.is_attention() {
            let packed = apply(
                layer,
                QwenOperation::Projection(QwenProjection::AttentionQkv),
                &[normal],
            )[0];
            let qk = apply(layer, QwenOperation::AttentionQk, &[packed]);
            assert!(apply(layer, QwenOperation::KvAppend, &[qk[1], packed]).is_empty());
            let fa = apply(layer, QwenOperation::FlashAttention, &[qk[0]])[0];
            let sig = apply(layer, QwenOperation::Sigmoid, &[qk[2]])[0];
            let gated = apply(layer, QwenOperation::Multiply, &[fa, sig])[0];
            apply(
                layer,
                QwenOperation::Projection(QwenProjection::AttentionOut),
                &[gated],
            )[0]
        } else {
            let packed = apply(
                layer,
                QwenOperation::Projection(QwenProjection::Qkvz),
                &[normal],
            )[0];
            let ba = apply(
                layer,
                QwenOperation::Projection(QwenProjection::Ba),
                &[normal],
            )[0];
            let conv = apply(layer, QwenOperation::GdnConv, &[packed, ba]);
            let gated = apply(layer, QwenOperation::GdnRecurrent, &[conv[0], conv[1], ba])[0];
            apply(
                layer,
                QwenOperation::Projection(QwenProjection::GdnOut),
                &[gated],
            )[0]
        };
        let post = apply(
            layer,
            QwenOperation::ResidualRms(QwenRmsSite::PostAttention),
            &[attention, carried],
        );
        let gate_up = apply(
            layer,
            QwenOperation::Projection(QwenProjection::GateUp),
            &[post[0]],
        )[0];
        let activated = apply(layer, QwenOperation::SiluAndMul, &[gate_up])[0];
        branch = apply(
            layer,
            QwenOperation::Projection(QwenProjection::Down),
            &[activated],
        )[0];
        residual = Some(post[1]);
        boundaries.push([branch, post[1]]);
    }
    let normal = apply(
        None,
        QwenOperation::ResidualRms(QwenRmsSite::Final),
        &[branch, residual.unwrap()],
    )[0];
    let logits = apply(
        None,
        QwenOperation::Projection(QwenProjection::Head),
        &[normal],
    )[0];
    builder.finish_model(boundaries, normal, logits).unwrap()
}
