//! Complete semantic fixtures; no trained source, root, native or device authority.
use super::*;
use crate::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

fn call(
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
    let name = |suffix: &str| layer.unwrap().name(suffix);
    let h = call.model_profile().hidden_size();
    let matrices = if let Vllm08Kind::Projection(p) = kind {
        p.members_for_profile(call.model_profile())
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
fn record(call: &ShortHfCall) -> RecordedVllm08Model {
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

#[test]
fn every_shape_shares_semantics_but_rebinds_all_nodes_and_producers() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        for workload in [
            ShortHfWorkload::Prompt47,
            ShortHfWorkload::Prompt57,
            ShortHfWorkload::Prompt62,
            ShortHfWorkload::Prompt67,
            ShortHfWorkload::Prompt72,
        ] {
            for base in std::iter::once(None).chain(workload.cached_bases().map(Some)) {
                let first = call(source, profile, workload, base, 11);
                let second = call(source, profile, workload, base, 353);
                let prepared = record(&first);
                let bound = prepared.bind_short_hf_call(&second).unwrap();
                let local_bytes = bound.binding_metadata_bytes().unwrap();
                assert!(local_bytes <= RecordedVllm08Model::BINDING_METADATA_BYTES);
                assert_eq!(
                    bound.require_binding_metadata(local_bytes).unwrap(),
                    local_bytes
                );
                assert!(bound.require_binding_metadata(local_bytes - 1).is_err());
                assert!(prepared.shares_prepared_semantics(&bound));
                assert_ne!(prepared.call(), bound.call());
                assert_eq!(bound.call(), second.as_vllm_call());
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
                assert_eq!(bound.plan().vllm_gdn_fragment(), Some(second.request()));
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
                    assert!(Arc::ptr_eq(&a.topology, &b.topology));
                    match (&a.operation, &b.operation) {
                        (
                            PlanOperation::TypedVllm08 { descriptor: a, .. },
                            PlanOperation::TypedVllm08 { descriptor: b, .. },
                        ) => {
                            assert!(Arc::ptr_eq(&a.semantics, &b.semantics));
                            assert_eq!(b.call(), second.as_vllm_call());
                            b.validate().unwrap();
                        }
                        (
                            PlanOperation::TypedVllmBf16Embedding { descriptor: a },
                            PlanOperation::TypedVllmBf16Embedding { descriptor: b },
                        ) => {
                            assert_eq!(a.teachers(), first.request().teachers());
                            assert_eq!(b.teachers(), second.request().teachers());
                        }
                        _ => panic!("foreign prepared operation"),
                    }
                }
                for index in 0..327 {
                    let tag = bound
                        .plan()
                        .vllm_value_provenance(ValueId(index))
                        .unwrap()
                        .unwrap();
                    tag.require_invocation(second.request()).unwrap();
                    assert!(tag.require_invocation(first.request()).is_err());
                    if let VllmValueOrigin::FullModel(origin) = tag.origin() {
                        assert_eq!(origin.call(), second.as_vllm_call());
                    }
                }
                // Equal token bytes minted separately still have independent issuers.
                let same_ids = call(source, profile, workload, base, 353);
                assert!(
                    bound
                        .plan()
                        .vllm_value_provenance(ValueId(326))
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
        FullModelProfile::Qwen08,
        ShortHfWorkload::Prompt47,
        None,
        11,
    ));
    for actual in [
        call(
            SourceIdentity::for_new_allocation().unwrap(),
            FullModelProfile::Qwen08,
            ShortHfWorkload::Prompt47,
            None,
            11,
        ),
        call(
            source,
            FullModelProfile::Qwen2B,
            ShortHfWorkload::Prompt47,
            None,
            11,
        ),
        call(
            source,
            FullModelProfile::Qwen08,
            ShortHfWorkload::Prompt57,
            None,
            11,
        ),
        call(
            source,
            FullModelProfile::Qwen08,
            ShortHfWorkload::Prompt47,
            Some(47),
            11,
        ),
    ] {
        assert!(prepared.bind_short_hf_call(&actual).is_err());
    }
    let cached = record(&call(
        source,
        FullModelProfile::Qwen08,
        ShortHfWorkload::Prompt47,
        Some(47),
        11,
    ));
    assert!(
        cached
            .bind_short_hf_call(&call(
                source,
                FullModelProfile::Qwen08,
                ShortHfWorkload::Prompt47,
                Some(48),
                11
            ))
            .is_err()
    );
    let cached = record(&call(
        source,
        FullModelProfile::Qwen08,
        ShortHfWorkload::Prompt57,
        Some(62),
        11,
    ));
    assert!(
        cached
            .bind_short_hf_call(&call(
                source,
                FullModelProfile::Qwen08,
                ShortHfWorkload::Prompt62,
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
        FullModelProfile::Qwen08,
        ShortHfWorkload::Prompt47,
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
        .bind_short_hf_call(&call(
            source,
            FullModelProfile::Qwen08,
            ShortHfWorkload::Prompt47,
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
