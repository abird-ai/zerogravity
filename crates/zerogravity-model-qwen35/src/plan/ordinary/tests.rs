use super::*;
use zerogravity_core::encoding::StoredTensorEncoding;

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
        kind: StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
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
        QwenParameterRole::FinalNorm => (
            "final_norm".into(),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![h as u64],
        ),
        QwenParameterRole::InputNorm(i) => (
            format!("layer.{i}.input_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![h as u64],
        ),
        QwenParameterRole::FfnNorm(i) => (
            format!("layer.{i}.ffn_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![h as u64],
        ),
        QwenParameterRole::GdnConvolution(i) => (
            format!("layer.{i}.conv"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![6144, 1, 4],
        ),
        QwenParameterRole::GdnNorm(i) => (
            format!("layer.{i}.gdn_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32),
            vec![128],
        ),
        QwenParameterRole::GdnA(i) => (
            format!("layer.{i}.A_log"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32),
            vec![16],
        ),
        QwenParameterRole::GdnDt(i) => (
            format!("layer.{i}.dt"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![16],
        ),
        QwenParameterRole::QueryNorm(i) => (
            format!("layer.{i}.q_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![256],
        ),
        QwenParameterRole::KeyNorm(i) => (
            format!("layer.{i}.k_norm"),
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            vec![256],
        ),
    };
    let element_bytes =
        if kind == StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16) {
            2
        } else {
            4
        };
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
        SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        capacity,
        |role| Ok(matrix(source, profile, role)),
        |role| Ok(parameter(source, profile, role)),
    )
}

fn invocation(
    plan: &RecordedOrdinaryPlan,
    profile: QwenSourceProfile,
    mode: SourceCallMode,
    base: usize,
    ids: Arc<[u32]>,
    emit_logits: bool,
) -> Result<BoundSourceInvocation> {
    let bound = plan.plan.bind_source_invocation(SourceInvocation::new(
        TextCallGeometry::new(
            plan.plan.source(),
            profile.vocabulary_size(),
            plan.plan.prefix_capacity(),
        )?,
        base,
        ids,
        emit_logits,
    )?)?;
    bound.invocation().source_call(mode)?;
    Ok(bound)
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

            let first = invocation(
                &plan,
                profile,
                SourceCallMode::Prefill,
                0,
                Arc::from([11, 353]),
                true,
            )
            .unwrap();
            let cached = invocation(
                &plan,
                profile,
                SourceCallMode::Decode,
                capacity - 1,
                Arc::from([7]),
                false,
            )
            .unwrap();
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

            let materialized = first
                .materialize_source_plan(SourceCallMode::Prefill)
                .unwrap();
            assert!(materialized.shares_semantics(&plan.plan));
            assert!(materialized.source_call().is_some());
            assert_eq!(materialized.source_call().unwrap().rows().ids(), [11, 353]);
            assert!(
                materialized
                    .nodes()
                    .iter()
                    .all(|node| matches!(node.operation, PlanOperation::SourceOperator { .. }))
            );
            let head = materialized.nodes().last().unwrap();
            let PlanOperation::SourceOperator { descriptor, .. } = &head.operation else {
                unreachable!()
            };
            assert_eq!(
                descriptor.kind(),
                SourceOperatorKind::Projection {
                    selected_row: Some(1)
                }
            );
            let kv = materialized
                .nodes()
                .iter()
                .find_map(|node| match &node.operation {
                    PlanOperation::SourceOperator { descriptor, .. }
                        if matches!(descriptor.kind(), SourceOperatorKind::KvAppend { .. }) =>
                    {
                        Some(descriptor)
                    }
                    _ => None,
                })
                .unwrap();
            let SourceOperatorKind::KvAppend { packed_value, .. } = kv.kind() else {
                unreachable!()
            };
            assert_eq!(packed_value.rows, 2);
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
    assert!(
        invocation(
            &plan,
            profile,
            SourceCallMode::Prefill,
            0,
            Arc::from([]),
            true
        )
        .is_err()
    );
    assert!(
        invocation(
            &plan,
            profile,
            SourceCallMode::Prefill,
            0,
            Arc::from(vec![11; 65]),
            true
        )
        .is_err()
    );
    assert!(
        invocation(
            &plan,
            profile,
            SourceCallMode::Prefill,
            500,
            Arc::from(vec![11; 13]),
            true
        )
        .is_err()
    );
    assert!(
        invocation(
            &plan,
            profile,
            SourceCallMode::Prefill,
            0,
            Arc::from([248_320]),
            true
        )
        .is_err()
    );
    assert!(
        invocation(
            &plan,
            profile,
            SourceCallMode::Decode,
            0,
            Arc::from([11]),
            true
        )
        .is_err()
    );

    let one_prefill = invocation(
        &plan,
        profile,
        SourceCallMode::Prefill,
        7,
        Arc::from([11]),
        true,
    )
    .unwrap();
    let one_decode = invocation(
        &plan,
        profile,
        SourceCallMode::Decode,
        7,
        Arc::from([11]),
        true,
    )
    .unwrap();
    assert_eq!(
        one_prefill
            .materialize_source_plan(SourceCallMode::Prefill)
            .unwrap()
            .source_call()
            .unwrap()
            .mode(),
        SourceCallMode::Prefill
    );
    assert_eq!(
        one_decode
            .materialize_source_plan(SourceCallMode::Decode)
            .unwrap()
            .source_call()
            .unwrap()
            .mode(),
        SourceCallMode::Decode
    );

    let foreign = SourceIdentity::for_new_allocation().unwrap();
    assert!(
        record_ordinary_plan(
            source,
            profile,
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
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
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            512,
            |role| Ok(matrix(source, profile, role)),
            |role| Ok(parameter(foreign, profile, role)),
        )
        .is_err()
    );
}

#[test]
fn cpu_source_plan_reuses_family_traversal_with_distinct_gdn_arithmetic() {
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let numerical = SourceNumericalContract::TorchCpuBf16Avx2MixedSourceV1;
        let recorded = record_ordinary_plan(
            source,
            profile,
            numerical,
            514,
            |role| Ok(matrix(source, profile, role)),
            |role| Ok(parameter(source, profile, role)),
        )
        .unwrap();
        assert_eq!(recorded.plan.required_domains().len(), 42);
        assert_eq!(recorded.boundaries.len(), 24);
        assert_eq!(
            recorded.plan.numerical_profile(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        );
        let mut recurrent = 0;
        for node in recorded.plan.nodes() {
            let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
                panic!("original templates")
            };
            assert_eq!(descriptor.numerical_contract(), numerical);
            if let SourceOperatorKind::GatedDelta { geometry, .. } = descriptor.kind() {
                recurrent += 1;
                assert_eq!(
                    geometry.contract,
                    SourceGdnContract::TorchCpuAvx2L2SiluBf16V1
                );
            }
        }
        assert_eq!(recurrent, 18);
        for (base, rows, mode) in [
            (0, 47, SourceCallMode::Prefill),
            (47, 64, SourceCallMode::Prefill),
            (512, 1, SourceCallMode::Decode),
        ] {
            let invocation = SourceInvocation::new(
                TextCallGeometry::new(source, profile.vocabulary_size(), 514).unwrap(),
                base,
                vec![11; rows].into(),
                true,
            )
            .unwrap();
            let bound = recorded.plan.bind_source_invocation(invocation).unwrap();
            assert!(Arc::ptr_eq(bound.plan(), &recorded.plan));
            let plan = bound.materialize_source_plan(mode).unwrap();
            assert_eq!(plan.source_call().unwrap().rows().len(), rows);
            assert_eq!(plan.proposed_domains().len(), 42);
        }
    }
}

#[test]
fn cpu_source_cached_one_cannot_be_relabelled_as_chunk_prefill() {
    let cpu = SourceNumericalContract::TorchCpuBf16Avx2MixedSourceV1;
    assert!(require_source_phase(cpu, SourceCallMode::Prefill, 47, 1).is_err());
    assert!(require_source_phase(cpu, SourceCallMode::Decode, 47, 1).is_ok());
    assert!(require_source_phase(cpu, SourceCallMode::Prefill, 0, 1).is_ok());
    assert!(require_source_phase(cpu, SourceCallMode::Prefill, 47, 17).is_ok());
    // The distinct CUDA source contract retains its explicit chunk semantics.
    assert!(
        require_source_phase(
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            SourceCallMode::Prefill,
            47,
            1
        )
        .is_ok()
    );
}
