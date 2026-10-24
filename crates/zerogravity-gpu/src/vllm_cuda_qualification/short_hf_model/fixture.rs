//! Semantic fixtures only; no trained source/device/root authority.
use super::*;
use zerogravity_core::accelerator::program::*;
use zerogravity_core::quant::QuantKind;
fn members(p: Vllm08Projection, profile: FullModelProfile) -> Vec<(&'static str, usize)> {
    match p {
        Vllm08Projection::Qkvz => vec![
            ("linear_attn.in_proj_qkv.weight", 6144),
            ("linear_attn.in_proj_z.weight", 2048),
        ],
        Vllm08Projection::Ba => vec![
            ("linear_attn.in_proj_b.weight", 16),
            ("linear_attn.in_proj_a.weight", 16),
        ],
        Vllm08Projection::GdnOut => vec![("linear_attn.out_proj.weight", profile.hidden_size())],
        Vllm08Projection::AttentionQkv => vec![
            ("self_attn.q_proj.weight", 4096),
            ("self_attn.k_proj.weight", 512),
            ("self_attn.v_proj.weight", 512),
        ],
        Vllm08Projection::AttentionOut => vec![("self_attn.o_proj.weight", profile.hidden_size())],
        Vllm08Projection::GateUp => vec![
            ("mlp.gate_proj.weight", profile.intermediate_size()),
            ("mlp.up_proj.weight", profile.intermediate_size()),
        ],
        Vllm08Projection::Down => vec![("mlp.down_proj.weight", profile.hidden_size())],
        Vllm08Projection::Head => vec![("model.language_model.embed_tokens.weight", 248320)],
    }
}
pub(super) fn call(
    source: SourceIdentity,
    profile: FullModelProfile,
    workload: ShortHfWorkload,
    base: Option<usize>,
    id: u32,
) -> ShortHfCall {
    let span = match base {
        None => ShortHfSpan::prefill(source, workload, &vec![id; workload.prompt_rows()]),
        Some(base) => ShortHfSpan::cached(source, workload, base, id),
    }
    .unwrap();
    ShortHfCall::new(profile, span).unwrap()
}
fn descriptor(call: &Vllm08Call, layer: Option<Vllm08Layer>, kind: Vllm08Kind) -> Vllm08Operator {
    let name = |suffix: &str| {
        format!(
            "model.language_model.layers.{}.{suffix}",
            layer.unwrap().index()
        )
    };
    let h = call.model_profile().hidden_size();
    let matrices = if let Vllm08Kind::Projection(p) = kind {
        members(p, call.model_profile())
            .iter()
            .map(|&(suffix, rows)| SourceTensor {
                source: call.source(),
                name: if p == Vllm08Projection::Head {
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
        Vllm08Kind::InputRms | Vllm08Kind::ResidualRms(Vllm08RmsSite::Input) => vec![(
            name("input_layernorm.weight"),
            QuantKind::Bf16,
            vec![h as u64],
            h * 2,
        )],
        Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention) => vec![(
            name("post_attention_layernorm.weight"),
            QuantKind::Bf16,
            vec![h as u64],
            h * 2,
        )],
        Vllm08Kind::ResidualRms(Vllm08RmsSite::Final) => vec![(
            "model.language_model.norm.weight".into(),
            QuantKind::Bf16,
            vec![h as u64],
            h * 2,
        )],
        Vllm08Kind::GdnConv => vec![(
            name("linear_attn.conv1d.weight"),
            QuantKind::Bf16,
            vec![6144, 1, 4],
            49152,
        )],
        Vllm08Kind::GdnRecurrent => [
            ("linear_attn.A_log", QuantKind::F32, 16, 64),
            ("linear_attn.dt_bias", QuantKind::Bf16, 16, 32),
            ("linear_attn.norm.weight", QuantKind::F32, 128, 512),
        ]
        .into_iter()
        .map(|(suffix, kind, n, bytes)| (name(suffix), kind, vec![n], bytes))
        .collect(),
        Vllm08Kind::AttentionQk => ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
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
    Vllm08Operator::new(call.clone(), layer, kind, matrices, parameters).unwrap()
}
pub(super) fn record(call: &ShortHfCall) -> RecordedVllm08Model {
    let call = call.as_vllm_call();
    let mut builder = PlanBuilder::new_full_model(call.clone()).unwrap();
    let mut branch = builder
        .vllm_bf16_source_embedding(
            &VllmBf16Embedding::for_call(
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
            .vllm08_operator(&descriptor(call, layer, kind), inputs)
            .unwrap()
    };
    for i in 0..24 {
        let actual_layer = call.layer(i).unwrap();
        let layer = Some(actual_layer);
        let (normal, carried) = if i == 0 {
            (apply(layer, Vllm08Kind::InputRms, &[branch])[0], branch)
        } else {
            let out = apply(
                layer,
                Vllm08Kind::ResidualRms(Vllm08RmsSite::Input),
                &[branch, residual.unwrap()],
            );
            (out[0], out[1])
        };
        let attention = if actual_layer.is_attention() {
            let packed = apply(
                layer,
                Vllm08Kind::Projection(Vllm08Projection::AttentionQkv),
                &[normal],
            )[0];
            let qk = apply(layer, Vllm08Kind::AttentionQk, &[packed]);
            assert!(apply(layer, Vllm08Kind::KvAppend, &[qk[1], packed]).is_empty());
            let fa = apply(layer, Vllm08Kind::FlashAttention, &[qk[0]])[0];
            let sig = apply(layer, Vllm08Kind::Sigmoid, &[qk[2]])[0];
            let gated = apply(layer, Vllm08Kind::Multiply, &[fa, sig])[0];
            apply(
                layer,
                Vllm08Kind::Projection(Vllm08Projection::AttentionOut),
                &[gated],
            )[0]
        } else {
            let packed = apply(
                layer,
                Vllm08Kind::Projection(Vllm08Projection::Qkvz),
                &[normal],
            )[0];
            let ba = apply(
                layer,
                Vllm08Kind::Projection(Vllm08Projection::Ba),
                &[normal],
            )[0];
            let conv = apply(layer, Vllm08Kind::GdnConv, &[packed, ba]);
            let gated = apply(layer, Vllm08Kind::GdnRecurrent, &[conv[0], conv[1], ba])[0];
            apply(
                layer,
                Vllm08Kind::Projection(Vllm08Projection::GdnOut),
                &[gated],
            )[0]
        };
        let post = apply(
            layer,
            Vllm08Kind::ResidualRms(Vllm08RmsSite::PostAttention),
            &[attention, carried],
        );
        let gate_up = apply(
            layer,
            Vllm08Kind::Projection(Vllm08Projection::GateUp),
            &[post[0]],
        )[0];
        let activated = apply(layer, Vllm08Kind::SiluAndMul, &[gate_up])[0];
        branch = apply(
            layer,
            Vllm08Kind::Projection(Vllm08Projection::Down),
            &[activated],
        )[0];
        residual = Some(post[1]);
        boundaries.push([branch, post[1]]);
    }
    let normal = apply(
        None,
        Vllm08Kind::ResidualRms(Vllm08RmsSite::Final),
        &[branch, residual.unwrap()],
    )[0];
    let logits = apply(
        None,
        Vllm08Kind::Projection(Vllm08Projection::Head),
        &[normal],
    )[0];
    builder
        .finish_full_model(boundaries, normal, logits)
        .unwrap()
}
