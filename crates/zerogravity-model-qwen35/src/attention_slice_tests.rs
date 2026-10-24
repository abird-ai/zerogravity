//! Synthetic exact-role transport/ownership only; no trained hidden-state claim.
use super::*;
use crate::accelerator::program::{
    ActivationPrecision, PlanBuilder, ProgramNumericalProfile, ProjectionContract,
};
use crate::operator_slice::attention::AttentionSliceInput;

fn attention_source(fixture: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["head_dim"] = json!(256);
    // Deliberately same width as Q/K norms: role checks must reject GDN gamma.
    cfg["text_config"]["linear_value_head_dim"] = json!(256);
    source_with_config(
        fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, element, fallback| {
            let value = if name.ends_with(".self_attn.q_proj.weight") {
                [1.0f32, -2.0, 3.0, -4.0][element / (256 * 256)]
            } else if name.ends_with(".self_attn.k_proj.weight") {
                5.0
            } else if name.ends_with(".self_attn.v_proj.weight") {
                6.0
            } else {
                return fallback;
            };
            if element.is_multiple_of(256) {
                (value.to_bits() >> 16) as u16
            } else {
                0
            }
        },
        |_| {},
    )
}
fn raw_input() -> Vec<u8> {
    let mut bytes = vec![0; 512];
    bytes[..2].copy_from_slice(&0x3f80u16.to_le_bytes());
    bytes
}
fn input(source: crate::accelerator::SourceIdentity, bytes: &[u8]) -> AttentionSliceInput<'_> {
    AttentionSliceInput::from_bf16_bytes(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        bytes,
    )
    .unwrap()
}

#[test]
fn explicit_attention_input_uses_exact_rows_raw_norms_and_no_session() {
    let fixture = Fixture::new();
    let loaded = attention_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let (model, _, _) = bind(loaded);
    let bytes = raw_input();
    let expected = model
        .execute_attention_projection_norm_slice(3, input(model.source_identity(), &bytes))
        .unwrap();
    let before = reserved(&admission);
    let output = operators
        .execute_attention_projection_norm_slice(3, input(source.source_identity(), &bytes), None)
        .unwrap();
    assert_eq!(output.source_identity(), source.source_identity());
    assert_eq!(output.layer(), 3);
    assert_eq!(output.precision(), ActivationPrecision::Bf16WidenedF32);
    assert_eq!(output.numerical_profile(), expected.numerical_profile());
    same(expected.input().unwrap(), output.input().unwrap());
    same(
        expected.query_gate_projection().unwrap(),
        output.query_gate_projection().unwrap(),
    );
    same(
        expected.key_projection().unwrap(),
        output.key_projection().unwrap(),
    );
    same(
        expected.value_projection().unwrap(),
        output.value_projection().unwrap(),
    );
    same(expected.query().unwrap(), output.query().unwrap());
    same(expected.gate().unwrap(), output.gate().unwrap());
    same(
        expected.normalized_query().unwrap(),
        output.normalized_query().unwrap(),
    );
    same(
        expected.normalized_key().unwrap(),
        output.normalized_key().unwrap(),
    );
    // Independent integer row labels distinguish [head,q/gate,width] packing.
    assert_eq!(
        output.query_gate_projection().unwrap(),
        [
            vec![1.0; 256],
            vec![-2.0; 256],
            vec![3.0; 256],
            vec![-4.0; 256]
        ]
        .concat()
    );
    assert_eq!(
        output.query().unwrap(),
        [vec![1.0; 256], vec![3.0; 256]].concat()
    );
    assert_eq!(
        output.gate().unwrap(),
        [vec![-2.0; 256], vec![-4.0; 256]].concat()
    );
    assert_eq!(output.key_projection().unwrap(), &[5.0; 256]);
    assert_eq!(output.value_projection().unwrap(), &[6.0; 256]);
    assert_eq!(reserved(&admission), before + 8 * 1024 * 1024);
    assert_eq!(model.initialized_cpu_state_bytes(), 0);
    assert!(model.new_session().is_err());
    assert!(model.forward_plan(8, true, false).is_err());
    let query_norm = model
        .parameter_descriptor(QwenParameterRole::QueryNorm(3))
        .unwrap();
    assert_eq!(query_norm.kind, zerogravity_core::quant::QuantKind::F32);
    assert!(
        model
            .tensor_bytes(&query_norm.name)
            .unwrap()
            .chunks_exact(4)
            .all(|word| word == 0.25f32.to_le_bytes())
    );
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 40 * 1024 * 1024);
    same(
        expected.normalized_query().unwrap(),
        output.normalized_query().unwrap(),
    );
    drop(output);
    assert_eq!(reserved(&admission), 0);
    assert!(
        !admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .fenced
    );
}

#[test]
fn plan_append_has_no_input_invention_or_state_and_rejects_foreign_builder() {
    let fixture = Fixture::new();
    let (model, _, _) = bind(attention_source(&fixture));
    let profile = model.program_numerical_profile();
    let mut builder = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
    // Synthetic TypedEmbed is ONLY an existing same-builder transport fixture.
    let embedding = builder
        .embed_profiled(
            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    assert!(
        model
            .append_attention_projection_norm_slice(&mut builder, 0, &embedding)
            .is_err()
    );
    assert!(
        model
            .append_attention_projection_norm_slice(&mut builder, 4, &embedding)
            .is_err()
    );
    let values = model
        .append_attention_projection_norm_slice(&mut builder, 3, &embedding)
        .unwrap();
    let recorded = builder
        .finish_operator_slice(*values.normalized_query())
        .unwrap();
    assert_eq!(recorded.plan().nodes().len(), 10);
    assert_eq!(
        recorded
            .plan()
            .nodes()
            .iter()
            .filter(|node| matches!(node.operation, PlanOperation::TypedProject { .. }))
            .count(),
        3
    );
    assert_eq!(
        recorded
            .plan()
            .nodes()
            .iter()
            .filter(|node| matches!(node.operation, PlanOperation::TypedOrdinaryRms { .. }))
            .count(),
        2
    );
    let views = recorded
        .plan()
        .nodes()
        .iter()
        .filter_map(|node| match node.operation {
            PlanOperation::View(view) => Some(view),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(views.len(), 2);
    assert_eq!(
        (views[0].offset, views[1].offset, views[0].row_stride),
        (0, 256, 512)
    );
    assert_eq!(
        values.query_gate_projection().precision(),
        ActivationPrecision::Bf16WidenedF32
    );
    assert_eq!(
        values.key_projection().precision(),
        ActivationPrecision::Bf16WidenedF32
    );
    assert_eq!(
        values.value_projection().precision(),
        ActivationPrecision::Bf16WidenedF32
    );
    assert_eq!(
        values.query().precision(),
        ActivationPrecision::Bf16WidenedF32
    );
    assert_eq!(
        values.gate().precision(),
        ActivationPrecision::Bf16WidenedF32
    );
    assert_eq!(
        values.normalized_key().precision(),
        ActivationPrecision::Bf16WidenedF32
    );
    assert!(recorded.plan().required_domains().is_empty());
    assert!(recorded.plan().proposed_domains().is_empty());
    assert_eq!(recorded.plan().frontier_delta(), 0);
    let mut first = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
    let foreign = first
        .embed_profiled(
            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let mut second = PlanBuilder::new_operator_slice(model.source_identity(), profile).unwrap();
    assert!(
        model
            .append_attention_projection_norm_slice(&mut second, 3, &foreign)
            .is_err()
    );
    assert_eq!(
        first
            .finish_operator_slice(foreign)
            .unwrap()
            .plan()
            .nodes()
            .len(),
        1
    );
}

#[test]
fn raw_bf16_import_preserves_zero_subnormal_bits_and_checks_before_admission() {
    let fixture = Fixture::new();
    let loaded = attention_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
    let mut bytes = raw_input();
    for (index, bits) in [0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f]
        .into_iter()
        .enumerate()
    {
        bytes[2 * index..2 * index + 2].copy_from_slice(&bits.to_le_bytes());
    }
    let before = reserved(&admission);
    let imported = math
        .import_attention_input(input(source.source_identity(), &bytes), 256)
        .unwrap();
    for (word, value) in bytes.chunks_exact(2).zip(imported.as_slice().unwrap()) {
        assert_eq!(
            value.to_bits(),
            u32::from(u16::from_le_bytes([word[0], word[1]])) << 16
        );
    }
    assert_eq!(reserved(&admission), before + 1024 * 1024);
    assert!(imported.clone().into_vec().is_err());
    drop(imported);
    assert_eq!(reserved(&admission), before);
    for invalid in [
        Vec::new(),
        vec![0],
        vec![0; 2 * 5120 + 2],
        0x7f80u16.to_le_bytes().to_vec(),
        0xff80u16.to_le_bytes().to_vec(),
        0x7fc0u16.to_le_bytes().to_vec(),
    ] {
        assert!(
            AttentionSliceInput::from_bf16_bytes(
                source.source_identity(),
                operators.program_numerical_profile(),
                &invalid
            )
            .is_err()
        );
    }
    assert!(
        AttentionSliceInput::from_bf16_bytes(
            crate::accelerator::SourceIdentity::unbound(),
            operators.program_numerical_profile(),
            &bytes
        )
        .is_err()
    );
    assert!(
        AttentionSliceInput::from_bf16_bytes(
            source.source_identity(),
            ProgramNumericalProfile::GgufSourceV1,
            &bytes
        )
        .is_err()
    );
    let foreign = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
    assert!(
        math.import_attention_input(input(foreign, &bytes), 256)
            .is_err()
    );
    assert!(
        math.import_attention_input(input(source.source_identity(), &bytes[..510]), 256)
            .is_err()
    );
    assert!(
        math.import_attention_input(input(source.source_identity(), &bytes), 128)
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
}

#[test]
fn exact_role_provider_and_foreign_issuer_refusals_have_no_partial_output() {
    struct Unaware(AtomicBool);
    impl crate::accelerator::MatVecKernel for Unaware {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            panic!("typed attention never calls an unaware provider")
        }
    }
    let fixture = Fixture::new();
    let loaded = attention_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = raw_input();
    let mut math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
    let imported = math
        .import_attention_input(input(source.source_identity(), &bytes), 256)
        .unwrap();
    let before = reserved(&admission);
    let gdn_gamma = operators
        .parameter_descriptor(QwenParameterRole::GdnNorm(0))
        .unwrap();
    assert_eq!(
        OrdinaryRmsContract::TorchCpuBf16Avx2
            .validate(&gdn_gamma, 256)
            .unwrap(),
        256
    );
    assert!(
        math.ordinary_rms_profiled(gdn_gamma, &imported, OrdinaryRmsContract::TorchCpuBf16Avx2)
            .is_err()
    );
    assert!(
        operators
            .execute_projection_slice(
                QwenMatrixRole::Embedding,
                crate::operator_slice::pure::PureSliceInput::from_bf16_bytes(
                    source.source_identity(),
                    ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
                    &bytes,
                )
                .unwrap(),
                None,
            )
            .is_err()
    );
    assert!(
        math.ordinary_rms_profiled(
            operators
                .parameter_descriptor(QwenParameterRole::QueryNorm(3))
                .unwrap(),
            &imported,
            OrdinaryRmsContract::TorchCpuBf16Avx2
        )
        .is_err()
    ); // Two query heads required.
    let mut forged = operators
        .matrix_descriptor(QwenMatrixRole::Query(3))
        .unwrap();
    forged.rows -= 1;
    assert!(
        math.project_profiled(forged, &imported, ProjectionContract::TorchCpuBf16Avx2)
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    let provider = Unaware(AtomicBool::new(false));
    let unaware =
        crate::host_math::HostMath::for_window_operator_slice(&operators, None, Some(&provider));
    assert!(
        unaware
            .import_attention_input(input(source.source_identity(), &bytes), 256)
            .is_err()
    );
    assert!(!provider.0.load(Ordering::Relaxed));
    assert_eq!(reserved(&admission), before);
    let (other_source, other_admission) = windows(&fixture, &loaded);
    let other = bind_windows(other_source.clone(), &loaded);
    let mut foreign = crate::host_math::HostMath::for_window_operator_slice(&other, None, None);
    let other_before = reserved(&other_admission);
    assert!(
        foreign
            .project_profiled(
                other.matrix_descriptor(QwenMatrixRole::Query(3)).unwrap(),
                &imported,
                ProjectionContract::TorchCpuBf16Avx2
            )
            .is_err()
    );
    assert_eq!(reserved(&other_admission), other_before);
    drop(other);
    drop(other_source);
    assert_eq!(reserved(&other_admission), 0);
}

#[test]
fn cancel_and_real_capacity_refusal_release_partial_private_values() {
    let fixture = Fixture::new();
    let loaded = attention_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = raw_input();
    let before = reserved(&admission);
    let cancelled = AtomicBool::new(true);
    assert!(
        operators
            .execute_attention_projection_norm_slice(
                3,
                input(source.source_identity(), &bytes),
                Some(&cancelled)
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    cancelled.store(false, Ordering::Release);
    let mut math =
        crate::host_math::HostMath::for_window_operator_slice(&operators, Some(&cancelled), None);
    let imported = math
        .import_attention_input(input(source.source_identity(), &bytes), 256)
        .unwrap();
    assert_eq!(reserved(&admission), before + 1024 * 1024);
    cancelled.store(true, Ordering::Release);
    assert!(
        crate::forward::attention_projection_norm_slice(&operators, &mut math, 3, &imported)
            .is_err()
    );
    assert_eq!(reserved(&admission), before + 1024 * 1024);
    drop(imported);
    assert_eq!(reserved(&admission), before);
    let mut pressure = Vec::new();
    loop {
        let remaining = admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    assert!(
        operators
            .execute_attention_projection_norm_slice(
                3,
                input(source.source_identity(), &bytes),
                None
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), 128 * 1024 * 1024);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
