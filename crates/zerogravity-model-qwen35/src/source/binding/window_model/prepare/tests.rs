//! Metadata refusals only; no trained source, native execution or root permit.
use super::*;
use crate::plan::{PromptSpan, PromptWorkload, QwenSourceProfile};
use zerogravity_core::accelerator::SourceIdentity;

mod fixture {
    //! Semantic fixtures only; no trained source/device/root authority.
    use super::*;
    use crate::plan::QwenPlanBuilder;
    use crate::plan::{
        PromptCall, PromptSpan, PromptWorkload, QwenEmbedding, QwenLayer, QwenOperation,
        QwenOperator, QwenPlan, QwenProjection, QwenRmsSite, QwenSourceCall, QwenSourceProfile,
    };
    use zerogravity_core::quant::QuantKind;
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
            QwenProjection::AttentionOut => {
                vec![("self_attn.o_proj.weight", profile.hidden_size())]
            }
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
}

#[test]
fn head_selects_actual_last_row_for_each_variant_and_phase() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        let row = profile.hidden_size() * 2;
        for count in [47, 57, 62, 67, 72] {
            let workload = PromptWorkload::from_prompt_tokens(count)?;
            let span = PromptSpan::prefill(source, workload, &vec![11; count])?;
            let call = PromptCall::new(profile, span)?;
            let head = HeadView::for_call(&call)?;
            assert_eq!(
                head,
                HeadView {
                    full_bytes: count * row,
                    offset: (count - 1) * row,
                    selected_bytes: row,
                    output_bytes: 496640
                }
            );
            assert!(call.request().reference_phase().is_err());
            for base in workload.cached_bases() {
                let call =
                    PromptCall::new(profile, PromptSpan::cached(source, workload, base, 353)?)?;
                let head = HeadView::for_call(&call)?;
                assert_eq!(
                    (head.full_bytes, head.offset, head.selected_bytes),
                    (row, 0, row)
                );
                assert_eq!(head.output_bytes, 496640);
            }
        }
    }
    Ok(())
}

#[test]
fn fa_uses_the_original_appended_private_read_and_never_a_fake_write() -> Result<()> {
    use zerogravity_core::accelerator::program::{ConvDomain, StateEffect, StateVersion};
    let source = SourceIdentity::for_new_allocation()?;
    let call = PromptCall::new(
        QwenSourceProfile::Qwen2B,
        PromptSpan::prefill(
            source,
            PromptWorkload::from_prompt_tokens(67).unwrap(),
            &[11; 67],
        )?,
    )?;
    for layer in [3, 7, 11, 15, 19, 23] {
        let descriptor = QwenOperator::new(
            call.as_source_call().clone(),
            Some(call.as_source_call().layer(layer)?),
            QwenOperation::FlashAttention,
            Vec::new(),
            Vec::new(),
        )?;
        let read = StateVersion {
            slot: StateSlot::Kv(KvDomain(layer)),
            ordinal: 1,
        };
        let actual = PlanNode::new(
            "actual FA read".into(),
            PlanOperation::SourceOperator {
                descriptor: Box::new(descriptor.mathematical_operator()?),
                state: Some(read),
            },
            vec![ValueId(1)],
            vec![ValueId(2)],
            None,
        );
        require_state_read(&actual, &descriptor)?;
        for bad in [
            None,
            Some(StateVersion { ordinal: 0, ..read }),
            Some(StateVersion { ordinal: 2, ..read }),
            Some(StateVersion {
                slot: StateSlot::Kv(KvDomain((layer + 4) % 24)),
                ..read
            }),
            Some(StateVersion {
                slot: StateSlot::Conv(ConvDomain(layer)),
                ..read
            }),
        ] {
            let mut node = actual.clone();
            if let PlanOperation::SourceOperator { state, .. } = &mut node.operation {
                *state = bad;
            }
            assert!(require_state_read(&node, &descriptor).is_err());
        }
        let mut fake_write = actual.clone();
        fake_write.state = Some(StateEffect {
            read: StateVersion { ordinal: 0, ..read },
            write: read,
        });
        assert!(require_state_read(&fake_write, &descriptor).is_err());
    }
    Ok(())
}

#[test]
fn complete_metadata_for_both_variants_and_every_supported_call() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        for prompt_tokens in [47, 57, 62, 67, 72] {
            let workload = PromptWorkload::from_prompt_tokens(prompt_tokens).unwrap();
            for base in std::iter::once(None).chain(workload.cached_bases().map(Some)) {
                let call = fixture::call(source, profile, workload, base, 11);
                let record = fixture::record(&call);
                let mut members = 0;
                let analysis = analyze_text_record(&call, &record, |operator| {
                    members += 1;
                    assert_eq!(operator.call(), call.as_source_call());
                    Ok(())
                })?;
                require_analysis_values(&record, &analysis)?;
                assert_eq!(members, 254);
                assert_eq!(
                    (
                        analysis.rms.len(),
                        analysis.projections.len(),
                        analysis.gdn.len(),
                        analysis.attention.len()
                    ),
                    (49, 114, 18, 6)
                );
                assert_eq!(analysis.head.selected_bytes, profile.hidden_size() * 2);
                assert_eq!(analysis.head.output_bytes, 248320 * 2);
                for id in [record.normalized(), record.logits()] {
                    require_selected_value(record.plan(), id)?;
                }
                for pair in record.boundaries() {
                    for &id in pair {
                        require_selected_value(record.plan(), id)?;
                    }
                }
                assert!(require_selected_value(record.plan(), ValueId(327)).is_err());
            }
        }
    }
    Ok(())
}

#[test]
fn fresh_ids_profile_phase_and_foreign_source_cannot_retag_a_record() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let other = SourceIdentity::for_new_allocation()?;
    let first = fixture::call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    );
    let record = fixture::record(&first);
    for (identity, profile, workload, base, id) in [
        (
            source,
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            None,
            11,
        ),
        (
            source,
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            None,
            353,
        ),
        (
            source,
            QwenSourceProfile::Qwen2B,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            None,
            11,
        ),
        (
            source,
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(57).unwrap(),
            None,
            11,
        ),
        (
            source,
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            Some(47),
            11,
        ),
        (
            other,
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            None,
            11,
        ),
    ] {
        let wrong = fixture::call(identity, profile, workload, base, id);
        assert!(require_text_record(&wrong, &record).is_err());
    }
    require_text_record(&first, &record)?;
    Ok(())
}

#[test]
fn every_source_member_failure_stops_preparation() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let call = fixture::call(
        source,
        QwenSourceProfile::Qwen2B,
        PromptWorkload::from_prompt_tokens(67).unwrap(),
        None,
        11,
    );
    let record = fixture::record(&call);
    let mut visits = 0;
    assert!(
        analyze_text_record(&call, &record, |_| {
            visits += 1;
            if visits == 133 {
                Err(Error::State("actual member issuer differs".into()))
            } else {
                Ok(())
            }
        })
        .is_err()
    );
    assert_eq!(visits, 133);
    Ok(())
}

#[test]
fn into_plan_moves_graph_and_preserves_unpublished_math_and_issuer() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let call = fixture::call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    );
    let record = fixture::record(&call);
    let nodes = record.plan().nodes().as_ptr();
    let source_call = record.plan().source_call().cloned();
    let output = record.logits();
    let plan = record.into_plan();
    assert_eq!(plan.nodes().as_ptr(), nodes);
    assert_eq!(plan.source_call(), source_call.as_ref());
    assert_eq!(plan.frontier_delta(), 0);
    assert!(plan.logits().is_none());
    require_selected_value(&plan, output)?;
    Ok(())
}

#[test]
fn selection_outside_the_actual_graph_is_refused() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let call = fixture::call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    );
    let record = fixture::record(&call);
    for id in [ValueId(327), ValueId(u32::MAX)] {
        assert!(require_selected_value(record.plan(), id).is_err());
    }
    Ok(())
}

#[test]
fn actual_role_owner_rejects_swapped_members_and_foreign_source() -> Result<()> {
    use super::super::source_plan::validate_source_operator;
    let source = SourceIdentity::for_new_allocation()?;
    let foreign = SourceIdentity::for_new_allocation()?;
    let call = fixture::call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    );
    let record = fixture::record(&call);
    let operator = record.operator(2)?.unwrap();
    let members = operator.matrices();
    assert_eq!(members.len(), 2);
    let validate = |swapped: bool, role, descriptor: &SourceTensor| -> Result<()> {
        let member = match role {
            QwenMatrixRole::GdnQkv(0) => &members[usize::from(swapped)],
            QwenMatrixRole::GdnGate(0) => &members[usize::from(!swapped)],
            _ => return Err(Error::State("unexpected source role".into())),
        };
        if member != descriptor {
            return Err(Error::State("actual role owner differs".into()));
        }
        Ok(())
    };
    validate_source_operator(
        source,
        &operator,
        |r, d| validate(false, r, d),
        |_, _| Err(Error::State("projection has no source parameter".into())),
    )?;
    assert!(
        validate_source_operator(
            source,
            &operator,
            |r, d| validate(true, r, d),
            |_, _| { Err(Error::State("projection has no source parameter".into())) }
        )
        .is_err()
    );
    assert!(
        validate_source_operator(
            foreign,
            &operator,
            |_, _| { panic!("foreign source must fail before any actual member access") },
            |_, _| { panic!("foreign source must fail before any parameter access") }
        )
        .is_err()
    );

    let rms = record.operator(1)?.unwrap();
    let mut wrong = rms.parameters()[0].clone();
    wrong.source = foreign;
    assert!(
        validate_source_operator(
            source,
            &rms,
            |_, _| { Err(Error::State("RMS has no source matrix".into())) },
            |_, actual| {
                if actual == &wrong {
                    Ok(())
                } else {
                    Err(Error::State("actual gamma issuer differs".into()))
                }
            }
        )
        .is_err()
    );
    Ok(())
}
