//! Complete semantic fixtures; no trained source, root, native or device authority.
use super::*;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

fn call(
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
    let name = |suffix: &str| layer.unwrap().name(suffix);
    let h = call.model_profile().hidden_size();
    let matrices = if let QwenOperation::Projection(p) = kind {
        p.members_for_profile(call.model_profile())
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
fn record(call: &PromptCall) -> QwenPlan {
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

#[test]
fn every_shape_shares_semantics_but_rebinds_all_nodes_and_producers() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        for prompt_tokens in [47, 57, 62, 67, 72] {
            let workload = PromptWorkload::from_prompt_tokens(prompt_tokens).unwrap();
            for base in std::iter::once(None).chain(workload.cached_bases().map(Some)) {
                let first = call(source, profile, workload, base, 11);
                let second = call(source, profile, workload, base, 353);
                let prepared = record(&first);
                let bound = prepared.bind_prompt_call(&second).unwrap();
                let local_bytes = bound.binding_metadata_bytes().unwrap();
                assert!(local_bytes <= QwenPlan::BINDING_METADATA_BYTES);
                assert_eq!(
                    bound.require_binding_metadata(local_bytes).unwrap(),
                    local_bytes
                );
                assert!(bound.require_binding_metadata(local_bytes - 1).is_err());
                assert!(prepared.shares_prepared_semantics(&bound));
                assert_ne!(prepared.call(), bound.call());
                assert_eq!(bound.call(), second.as_source_call());
                assert_eq!(
                    (bound.plan().nodes().len(), bound.plan().values().len()),
                    (255, 327)
                );
                assert_eq!(
                    bound.plan().required_domains(),
                    prepared.plan().required_domains()
                );
                assert_eq!(
                    bound.plan().proposed_domains(),
                    prepared.plan().proposed_domains()
                );
                assert_eq!(bound.plan().frontier_delta(), 0);
                assert!(bound.plan().logits().is_none());
                assert_eq!(
                    bound.plan().source_call(),
                    Some(&second.request().source_call().unwrap())
                );
                assert!(!std::ptr::eq(prepared.plan(), bound.plan()));
                assert!(std::ptr::eq(
                    prepared.plan().values(),
                    bound.plan().values()
                ));
                assert!(std::ptr::eq(
                    prepared.plan().precisions(),
                    bound.plan().precisions()
                ));
                for (a, b) in prepared.plan().nodes().iter().zip(bound.plan().nodes()) {
                    assert!(a.shares_topology(b));
                    match (&a.operation, &b.operation) {
                        (
                            PlanOperation::SourceOperator { descriptor: a, .. },
                            PlanOperation::SourceOperator { descriptor: b, .. },
                        ) => {
                            assert!(a.shares_semantics(b));
                            assert_eq!(a.call(), &first.request().source_call().unwrap());
                            assert_eq!(b.call(), &second.request().source_call().unwrap());
                            b.validate().unwrap();
                        }
                        _ => panic!("foreign prepared operation"),
                    }
                }
                for index in 0..327 {
                    let tag = bound.value_provenance(ValueId(index)).unwrap().unwrap();
                    tag.require_invocation(second.request()).unwrap();
                    assert!(tag.require_invocation(first.request()).is_err());
                    if let QwenValueOrigin::FullModel(origin) = tag.origin() {
                        assert_eq!(origin.call(), second.as_source_call());
                    }
                }
                // Equal token bytes minted separately still have independent issuers.
                let same_ids = call(source, profile, workload, base, 353);
                assert!(
                    bound
                        .value_provenance(ValueId(326))
                        .unwrap()
                        .unwrap()
                        .require_invocation(same_ids.request())
                        .is_err()
                );
                assert!(!prepared.shares_prepared_semantics(&record(&first)));
            }
        }
    }
}

#[test]
fn prepared_binding_refuses_source_profile_phase_frontier_and_workload_changes() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let prepared = record(&call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    ));
    for actual in [
        call(
            SourceIdentity::for_new_allocation().unwrap(),
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            None,
            11,
        ),
        call(
            source,
            QwenSourceProfile::Qwen2B,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            None,
            11,
        ),
        call(
            source,
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(57).unwrap(),
            None,
            11,
        ),
        call(
            source,
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            Some(47),
            11,
        ),
    ] {
        assert!(prepared.bind_prompt_call(&actual).is_err());
    }
    let cached = record(&call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        Some(47),
        11,
    ));
    assert!(
        cached
            .bind_prompt_call(&call(
                source,
                QwenSourceProfile::Qwen08,
                PromptWorkload::from_prompt_tokens(47).unwrap(),
                Some(48),
                11
            ))
            .is_err()
    );
    let cached = record(&call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(57).unwrap(),
        Some(62),
        11,
    ));
    assert!(
        cached
            .bind_prompt_call(&call(
                source,
                QwenSourceProfile::Qwen08,
                PromptWorkload::from_prompt_tokens(62).unwrap(),
                Some(62),
                11
            ))
            .is_err()
    );
}

#[test]
fn shared_semantics_retain_the_original_host_charge_after_cache_owner_drops() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "prepared-record-test").unwrap();
    let bytes = 4 * 1024 * 1024;
    arbiter.register(key.clone(), bytes).unwrap();
    let actual = call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    );
    let lease = arbiter
        .admit(
            &key,
            ResourceOwner::new("original-source", "prepared-record").unwrap(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes,
            }],
        )
        .unwrap();
    let source_owner = Arc::new(vec![1u8]);
    let witness = Arc::downgrade(&source_owner);
    let metadata = lease.into_host_workspace(source, source_owner).unwrap();
    let prepared = record(&actual).retain_prepared_metadata(metadata).unwrap();
    let bound = prepared
        .bind_prompt_call(&call(
            source,
            QwenSourceProfile::Qwen08,
            PromptWorkload::from_prompt_tokens(47).unwrap(),
            None,
            353,
        ))
        .unwrap();
    let live = bound.clone();
    drop(prepared);
    drop(bound);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, bytes);
    assert!(witness.upgrade().is_some());
    drop(live);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    assert!(witness.upgrade().is_none());
}

#[test]
fn cold_operator_adapter_checks_actual_common_semantics_and_original_row_issuer() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let first = call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    );
    let prepared = record(&first);
    assert!(prepared.operator(0).unwrap().is_none());
    assert!(prepared.operator(255).is_err());
    for index in 1..255 {
        let family = prepared.operator(index).unwrap().unwrap();
        let PlanOperation::SourceOperator { descriptor, .. } =
            &prepared.plan().nodes()[index].operation
        else {
            panic!("foreign mathematical operator")
        };
        assert_eq!(&family.mathematical_operator().unwrap(), &**descriptor);
        family.require_mathematical_operator(descriptor).unwrap();
        assert_eq!(family.call(), first.as_source_call());
    }
    let second = call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        353,
    );
    let rebound = prepared.bind_prompt_call(&second).unwrap();
    let descriptor = rebound.operator(1).unwrap().unwrap();
    assert_eq!(descriptor.call(), second.as_source_call());
    assert_ne!(descriptor.call(), first.as_source_call());
}

#[test]
fn prepared_operator_comparison_rejects_foreign_issuer_geometry_and_member_order() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let first = call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    );
    let actual = record(&first);
    let rms = actual.operator(1).unwrap().unwrap();
    let expected = rms.mathematical_operator().unwrap();
    let foreign = call(
        source,
        QwenSourceProfile::Qwen08,
        PromptWorkload::from_prompt_tokens(47).unwrap(),
        None,
        11,
    );
    let substituted = SourceOperator::new(
        foreign.as_source_call().request().source_call().unwrap(),
        expected.kind(),
        expected.numerical_contract(),
        expected.matrices().to_vec(),
        expected.parameters().to_vec(),
    )
    .unwrap();
    assert!(rms.require_mathematical_operator(&substituted).is_err());
    let SourceOperatorKind::Rms(mut geometry) = expected.kind() else {
        panic!("RMS absent")
    };
    geometry.epsilon *= 2.0;
    let substituted = SourceOperator::new(
        expected.call().clone(),
        SourceOperatorKind::Rms(geometry),
        expected.numerical_contract(),
        vec![],
        expected.parameters().to_vec(),
    )
    .unwrap();
    assert!(rms.require_mathematical_operator(&substituted).is_err());
    let projection = actual.operator(2).unwrap().unwrap();
    let expected = projection.mathematical_operator().unwrap();
    let mut members = expected.matrices().to_vec();
    members.reverse();
    let substituted = SourceOperator::new(
        expected.call().clone(),
        expected.kind(),
        expected.numerical_contract(),
        members,
        vec![],
    )
    .unwrap();
    assert!(
        projection
            .require_mathematical_operator(&substituted)
            .is_err()
    );
}
