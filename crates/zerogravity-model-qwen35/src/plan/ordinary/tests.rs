use super::*;
use zerogravity_core::quant::QuantKind;

pub(super) fn matrix(
    source: SourceIdentity,
    profile: QwenSourceProfile,
    role: QwenMatrixRole,
) -> SourceTensor {
    let h = profile.hidden_size();
    let m = profile.intermediate_size();
    let (name, rows, columns) = match role {
        QwenMatrixRole::Embedding | QwenMatrixRole::Output => (
            "model.language_model.embed_tokens.weight".into(),
            248_320,
            h,
        ),
        QwenMatrixRole::FfnGate(i) => (format!("layer.{i}.gate"), m, h),
        QwenMatrixRole::FfnUp(i) => (format!("layer.{i}.up"), m, h),
        QwenMatrixRole::FfnDown(i) => (format!("layer.{i}.down"), h, m),
        QwenMatrixRole::GdnQkv(i) => (format!("layer.{i}.qkv"), 6144, h),
        QwenMatrixRole::GdnGate(i) => (format!("layer.{i}.z"), 2048, h),
        QwenMatrixRole::GdnAlpha(i) => (format!("layer.{i}.a"), 16, h),
        QwenMatrixRole::GdnBeta(i) => (format!("layer.{i}.b"), 16, h),
        QwenMatrixRole::GdnOutput(i) => (format!("layer.{i}.gdn_out"), h, 2048),
        QwenMatrixRole::Query(i) => (format!("layer.{i}.q"), 4096, h),
        QwenMatrixRole::Key(i) => (format!("layer.{i}.k"), 512, h),
        QwenMatrixRole::Value(i) => (format!("layer.{i}.v"), 512, h),
        QwenMatrixRole::AttentionOutput(i) => (format!("layer.{i}.attn_out"), h, 2048),
    };
    SourceTensor {
        source,
        name,
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        columns,
        rows,
    }
}

pub(super) fn parameter(
    source: SourceIdentity,
    profile: QwenSourceProfile,
    role: QwenParameterRole,
) -> SourceParameter {
    let h = profile.hidden_size();
    let (name, kind, shape) = match role {
        QwenParameterRole::FinalNorm => ("final_norm".into(), QuantKind::Bf16, vec![h as u64]),
        QwenParameterRole::InputNorm(i) => (
            format!("layer.{i}.input_norm"),
            QuantKind::Bf16,
            vec![h as u64],
        ),
        QwenParameterRole::FfnNorm(i) => (
            format!("layer.{i}.ffn_norm"),
            QuantKind::Bf16,
            vec![h as u64],
        ),
        QwenParameterRole::GdnConvolution(i) => {
            (format!("layer.{i}.conv"), QuantKind::Bf16, vec![6144, 1, 4])
        }
        QwenParameterRole::GdnNorm(i) => (format!("layer.{i}.gdn_norm"), QuantKind::F32, vec![128]),
        QwenParameterRole::GdnA(i) => (format!("layer.{i}.A_log"), QuantKind::F32, vec![16]),
        QwenParameterRole::GdnDt(i) => (format!("layer.{i}.dt"), QuantKind::Bf16, vec![16]),
        QwenParameterRole::QueryNorm(i) => {
            (format!("layer.{i}.q_norm"), QuantKind::Bf16, vec![256])
        }
        QwenParameterRole::KeyNorm(i) => (format!("layer.{i}.k_norm"), QuantKind::Bf16, vec![256]),
    };
    let element_bytes = if kind == QuantKind::Bf16 { 2 } else { 4 };
    let bytes = shape.iter().map(|n| *n as usize).product::<usize>() * element_bytes;
    SourceParameter {
        source,
        name,
        kind,
        shape,
        bytes,
    }
}

fn plan(
    source: SourceIdentity,
    profile: QwenSourceProfile,
    capacity: usize,
) -> Result<RecordedOrdinaryPlan> {
    record_ordinary_plan(
        source,
        profile,
        capacity,
        |role| Ok(matrix(source, profile, role)),
        |role| Ok(parameter(source, profile, role)),
    )
}

fn invocation(
    plan: &RecordedOrdinaryPlan,
    profile: QwenSourceProfile,
    base: usize,
    ids: Arc<[u32]>,
    emit_logits: bool,
) -> Result<BoundSourceInvocation> {
    plan.plan.bind_source_invocation(SourceInvocation::new(
        TextCallGeometry::new(
            plan.plan.source(),
            profile.vocabulary_size(),
            plan.plan.prefix_capacity(),
        )?,
        base,
        ids,
        emit_logits,
    )?)
}

#[test]
fn one_static_plan_serves_runtime_lengths_ids_and_emit_policy_for_both_profiles() {
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        let source = SourceIdentity::for_new_allocation().unwrap();
        for capacity in [512, 4096, 32_768] {
            let plan = plan(source, profile, capacity).unwrap();
            assert_eq!(plan.plan.prefix_capacity(), capacity);
            assert_eq!(ORDINARY_CUDA_MAX_CHUNK_ROWS, 64);
            let expected = domains()
                .unwrap()
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(
                plan.plan
                    .required_domains()
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>(),
                expected
            );
            assert_eq!(plan.plan.proposed_domains().len(), 42);
            assert!(
                plan.plan
                    .proposed_domains()
                    .iter()
                    .all(|version| version.ordinal == 1 && expected.contains(&version.slot))
            );
            assert_eq!(plan.boundaries.len(), 24);
            assert_eq!(plan.plan.source_call(), None);
            assert_eq!(plan.plan.logits(), Some(plan.logits));
            assert!(plan.plan.nodes().iter().all(|node| matches!(
                node.operation,
                PlanOperation::SourceOperatorTemplate { .. }
            )));
            let head = plan.plan.nodes().last().unwrap();
            let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &head.operation else {
                panic!("call-free head absent")
            };
            assert_eq!(descriptor.row_selection(), SourceRowSelection::LastActive);
            assert_eq!(descriptor.output_elements().unwrap(), [248_320]);

            let first = invocation(&plan, profile, 0, Arc::from([11, 353]), true).unwrap();
            let cached = invocation(&plan, profile, capacity - 1, Arc::from([7]), false).unwrap();
            assert!(Arc::ptr_eq(first.plan(), cached.plan()));
            assert!(Arc::ptr_eq(first.plan(), &plan.plan));
            assert_eq!(
                (
                    first.invocation().base(),
                    first.invocation().rows(),
                    first.invocation().frontier()
                ),
                (0, 2, 2)
            );
            assert_eq!(first.invocation().ids(), [11, 353]);
            assert!(first.invocation().emit_logits());
            assert_eq!(cached.invocation().frontier(), capacity);
            assert!(!cached.invocation().emit_logits());
            assert!(std::ptr::eq(first.active_nodes(), plan.plan.nodes()));
            assert!(std::ptr::eq(cached.active_nodes(), plan.plan.state_nodes()));
            assert_eq!(first.active_nodes().len(), cached.active_nodes().len() + 1);
        }
    }
}

#[test]
fn ordinary_plan_refuses_bad_context_chunk_frontier_ids_and_source_descriptors() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let profile = QwenSourceProfile::Qwen08;
    for capacity in [0, 511, 32_769] {
        assert!(plan(source, profile, capacity).is_err());
    }
    let plan = plan(source, profile, 512).unwrap();
    assert!(invocation(&plan, profile, 0, Arc::from([]), true).is_err());
    assert!(invocation(&plan, profile, 0, Arc::from(vec![11; 65]), true).is_err());
    assert!(invocation(&plan, profile, 500, Arc::from(vec![11; 13]), true).is_err());
    assert!(invocation(&plan, profile, 0, Arc::from([248_320]), true).is_err());

    let foreign = SourceIdentity::for_new_allocation().unwrap();
    assert!(
        record_ordinary_plan(
            source,
            profile,
            512,
            |role| Ok(matrix(foreign, profile, role)),
            |role| Ok(parameter(source, profile, role)),
        )
        .is_err()
    );
    assert!(
        record_ordinary_plan(
            source,
            profile,
            512,
            |role| Ok(matrix(source, profile, role)),
            |role| Ok(parameter(foreign, profile, role)),
        )
        .is_err()
    );
}
