//! Source-free full-header fixtures; no trained27B/operator independence claim.
use super::*;
use crate::accelerator::program::ProgramNumericalProfile;
use crate::operator_slice::pure::PureSliceInput;
use zerogravity_core::resource_arbiter::AllocationClass;

fn input<'a>(source: &OriginalWindowOperators, bytes: &'a [u8]) -> PureSliceInput<'a> {
    PureSliceInput::from_bf16_bytes(
        source.source().source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        bytes,
    )
    .unwrap()
}

#[test]
fn all_bound_projection_roles_use_complete_actual_source_and_retained_results() {
    for tied in [false, true] {
        let fixture = Fixture::new();
        let loaded = source_with_options(&fixture, tied, 256, true, |_| {});
        let (source, admission) = windows(&fixture, &loaded);
        let operators = bind_windows(source.clone(), &loaded);
        for role in [
            QwenMatrixRole::GdnQkv(0),
            QwenMatrixRole::GdnGate(0),
            QwenMatrixRole::GdnAlpha(0),
            QwenMatrixRole::GdnBeta(0),
            QwenMatrixRole::GdnOutput(0),
            QwenMatrixRole::Query(3),
            QwenMatrixRole::Key(3),
            QwenMatrixRole::Value(3),
            QwenMatrixRole::AttentionOutput(3),
            QwenMatrixRole::FfnGate(0),
            QwenMatrixRole::FfnUp(0),
            QwenMatrixRole::FfnDown(0),
            QwenMatrixRole::Output,
        ] {
            let tensor = operators.matrix_descriptor(role).unwrap();
            let bytes = 0x3f80u16.to_le_bytes().repeat(tensor.columns);
            let before = reserved(&admission);
            let result = operators
                .execute_projection_slice(role, input(&operators, &bytes), None)
                .unwrap();
            let sign = if role == QwenMatrixRole::Output && !tied {
                -1.0
            } else {
                1.0
            };
            assert_eq!(result.input().unwrap(), vec![1.0; tensor.columns]);
            assert_eq!(
                result.output().unwrap(),
                vec![sign * tensor.columns as f32; tensor.rows]
            );
            assert_eq!(result.source_identity(), source.source_identity());
            assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
            drop(result);
            assert_eq!(reserved(&admission), before);
        }
        drop(operators);
        drop(source);
        assert_eq!(reserved(&admission), 0);
    }
}

#[test]
fn ordinary_hidden_roles_preserve_raw_gamma_and_complete_source_membership() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    for role in [
        QwenParameterRole::InputNorm(0),
        QwenParameterRole::InputNorm(2),
        QwenParameterRole::FfnNorm(0),
        QwenParameterRole::FinalNorm,
    ] {
        let result = operators
            .execute_ordinary_rms_slice(role, input(&operators, &bytes), None)
            .unwrap();
        let expected = if matches!(role, QwenParameterRole::InputNorm(_)) {
            2.0
        } else {
            1.25
        };
        assert_eq!(result.output().unwrap(), vec![expected; 256]);
    }
    let before = reserved(&admission);
    for role in [
        QwenParameterRole::GdnNorm(0),
        QwenParameterRole::GdnA(0),
        QwenParameterRole::GdnDt(0),
        QwenParameterRole::InputNorm(99),
    ] {
        assert!(
            operators
                .execute_ordinary_rms_slice(role, input(&operators, &bytes), None)
                .is_err()
        );
        assert_eq!(reserved(&admission), before);
    }
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn same_plan_projection_and_norm_nodes_do_not_create_state_or_external_producers() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let mut builder = PlanBuilder::new_operator_slice(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    let embedding = builder
        .embed_profiled(
            operators
                .matrix_descriptor(QwenMatrixRole::Embedding)
                .unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let normalized = operators
        .append_ordinary_rms_slice(&mut builder, QwenParameterRole::FfnNorm(0), &embedding)
        .unwrap();
    let up = operators
        .append_projection_slice(&mut builder, QwenMatrixRole::FfnUp(0), &normalized)
        .unwrap();
    let down = operators
        .append_projection_slice(&mut builder, QwenMatrixRole::FfnDown(0), &up)
        .unwrap();
    let head = operators
        .append_projection_slice(&mut builder, QwenMatrixRole::Output, &down)
        .unwrap();
    let plan = builder.finish_operator_slice(head).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(matches!(
        plan.plan().nodes()[4].operation,
        PlanOperation::TypedProject { .. }
    ));
    drop(operators);
    drop(source);
    drop(plan);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn checked_actual_vocabulary_output_and_wide_input_charge_grow_before_allocation() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let before = admission.arbiter.snapshot(&admission.resource).unwrap();
    let head = operators.workspace(248320, 248448).unwrap();
    let current = admission.arbiter.snapshot(&admission.resource).unwrap();
    assert_eq!(
        current.per_class[&AllocationClass::Output]
            - before
                .per_class
                .get(&AllocationClass::Output)
                .copied()
                .unwrap_or(0),
        248320 * 4
    );
    assert_eq!(head.bytes(), (248320 + 248448) * 4);
    drop(head);
    let wide = operators.workspace(17408, 1).unwrap();
    assert_eq!(wide.bytes(), 17408 * 4 + 960 * 1024);
    drop(wide);
    assert!(operators.workspace(usize::MAX, 1).is_err());
    assert!(operators.workspace(1, usize::MAX).is_err());
    assert_eq!(reserved(&admission), before.reserved);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn invalid_input_role_capacity_and_cancel_refuse_without_leaking_or_fencing() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let before = reserved(&admission);
    for role in [
        QwenMatrixRole::Embedding,
        QwenMatrixRole::GdnGate(3),
        QwenMatrixRole::Query(0),
    ] {
        assert!(
            operators
                .execute_projection_slice(role, input(&operators, &bytes), None)
                .is_err()
        );
    }
    assert!(
        operators
            .execute_projection_slice(
                QwenMatrixRole::FfnUp(0),
                input(&operators, &bytes[..2]),
                None
            )
            .is_err()
    );
    assert!(
        operators
            .execute_projection_slice(
                QwenMatrixRole::FfnUp(0),
                input(&operators, &bytes),
                Some(&AtomicBool::new(true))
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    for bad in [
        vec![0],
        0x7fc1u16.to_le_bytes().to_vec(),
        vec![0; 2 * 17409],
    ] {
        assert!(
            PureSliceInput::from_bf16_bytes(
                source.source_identity(),
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
                &bad
            )
            .is_err()
        );
    }
    let pressure = source.reserve_workspace(32 * 1024 * 1024).unwrap();
    let pressure2 = source.reserve_workspace(32 * 1024 * 1024).unwrap();
    let remaining = admission
        .arbiter
        .snapshot(&admission.resource)
        .unwrap()
        .remaining;
    let pressure3 = source
        .reserve_workspace((remaining - (2 * 1024 * 1024 - 1)) as usize)
        .unwrap();
    let held = reserved(&admission);
    assert!(
        operators
            .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
            .is_err()
    );
    assert_eq!(reserved(&admission), held);
    assert!(
        !admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .fenced
    );
    drop(pressure3);
    drop(pressure2);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn returned_borrowed_results_keep_source_and_original_tickets_after_callers_drop() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let source_metadata = source.metadata_reserved_bytes();
    let result = operators
        .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
        .unwrap();
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), source_metadata + 2 * 1024 * 1024);
    assert_eq!(result.output().unwrap(), vec![-256.0; 16]);
    assert_eq!(result.input().unwrap(), vec![1.0; 256]);
    drop(result);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn unaware_provider_refuses_new_roles_before_any_callback_or_admission() {
    struct Unaware(AtomicBool);
    impl crate::accelerator::MatVecKernel for Unaware {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            panic!("pure original CPU role cannot call an unaware provider");
        }
    }
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let provider = Unaware(AtomicBool::new(false));
    let plain = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
    let activation = plain
        .import_pure_input(input(&operators, &bytes), 256)
        .unwrap();
    let before = reserved(&admission);
    let mut unaware =
        crate::host_math::HostMath::for_window_operator_slice(&operators, None, Some(&provider));
    assert!(
        unaware
            .project_profiled(
                operators.matrix_descriptor(QwenMatrixRole::Output).unwrap(),
                &activation,
                crate::accelerator::program::ProjectionContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    assert!(!provider.0.load(std::sync::atomic::Ordering::Relaxed));
    assert_eq!(reserved(&admission), before);
    drop(activation);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn separately_opened_same_content_source_cannot_rebind_a_retained_input() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source_a, admission_a) = windows(&fixture, &loaded);
    let (source_b, admission_b) = windows(&fixture, &loaded);
    assert_ne!(source_a.source_identity(), source_b.source_identity());
    assert!(!source_a.same_issuer(&source_b));
    let operators_a = bind_windows(source_a.clone(), &loaded);
    let operators_b = bind_windows(source_b.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let math_a = crate::host_math::HostMath::for_window_operator_slice(&operators_a, None, None);
    let activation = math_a
        .import_pure_input(input(&operators_a, &bytes), 256)
        .unwrap();
    let before_b = reserved(&admission_b);
    let mut math_b =
        crate::host_math::HostMath::for_window_operator_slice(&operators_b, None, None);
    assert!(
        math_b
            .project_profiled(
                operators_b
                    .matrix_descriptor(QwenMatrixRole::Output)
                    .unwrap(),
                &activation,
                crate::accelerator::program::ProjectionContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    assert_eq!(reserved(&admission_b), before_b);
    drop(activation);
    drop(operators_a);
    drop(operators_b);
    drop(source_a);
    drop(source_b);
    assert_eq!(reserved(&admission_a), 0);
    assert_eq!(reserved(&admission_b), 0);
}

#[test]
fn valid_two_hidden_qkv_producer_cannot_record_one_hidden_norm_or_partial_node() {
    let fixture = Fixture::new();
    let mut cfg: Value = serde_json::from_slice(&config(false)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["linear_key_head_dim"] = json!(128);
    cfg["text_config"]["linear_value_head_dim"] = json!(128);
    let loaded = source_with_config(
        &fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |_, _, bits| bits,
        |_| {},
    );
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let matrix = operators
        .matrix_descriptor(QwenMatrixRole::GdnQkv(0))
        .unwrap();
    assert_eq!(matrix.rows, 512);
    assert_eq!(matrix.columns, 256);
    let mut builder = PlanBuilder::new_operator_slice(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    let embedding = builder
        .embed_profiled(
            operators
                .matrix_descriptor(QwenMatrixRole::Embedding)
                .unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let qkv = operators
        .append_projection_slice(&mut builder, QwenMatrixRole::GdnQkv(0), &embedding)
        .unwrap();
    assert!(matches!(operators.append_ordinary_rms_slice(&mut builder,
        QwenParameterRole::InputNorm(0), &qkv),
        Err(Error::Malformed(message)) if message.contains("exact hidden input")));
    let recorded = builder.finish_operator_slice(qkv).unwrap();
    assert_eq!(recorded.plan().nodes().len(), 2);
    assert!(recorded.plan().proposed_domains().is_empty());
    drop(recorded);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn completed_private_output_cancellation_drops_actual_source_backing_before_return() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let before = reserved(&admission);
    let output = operators
        .execute_projection_slice(QwenMatrixRole::Output, input(&operators, &bytes), None)
        .unwrap();
    assert_eq!(output.output().unwrap(), vec![-256.0; 16]);
    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
    let flag = AtomicBool::new(true);
    assert!(
        matches!(crate::operator_slice::pure::finish_pure_output(output, Some(&flag)),
        Err(Error::State(message)) if message.contains("cancelled before return"))
    );
    assert_eq!(reserved(&admission), before);
    assert!(
        !admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .fenced
    );
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
