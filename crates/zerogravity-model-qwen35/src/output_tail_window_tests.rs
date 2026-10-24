//! Small genuine source-owner pipelines; not trained27B numerical qualification.
use super::*;
use crate::accelerator::program::{ActivationPrecision, PlanBuilder, ProgramNumericalProfile};
use crate::operator_slice::pure::PureSliceInput;

fn tail_source(fixture: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(false)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["intermediate_size"] = json!(16);
    source_with_config(
        fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, _, bits| {
            if name.ends_with("mlp.gate_proj.weight") || name.ends_with("mlp.up_proj.weight") {
                0x3b80 // 1/256
            } else if name.ends_with("mlp.down_proj.weight") {
                0x3d80 // 1/16
            } else {
                bits
            }
        },
        |_| {},
    )
}
fn input<'a>(operators: &OriginalWindowOperators, bytes: &'a [u8]) -> PureSliceInput<'a> {
    PureSliceInput::from_bf16_bytes(
        operators.source().source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        bytes,
    )
    .unwrap()
}

#[test]
fn output_tail_computes_all_stages_and_full_head_from_same_source() {
    let fixture = Fixture::new();
    let loaded = tail_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let before = reserved(&admission);
    let output = operators
        .execute_output_tail(input(&operators, &bytes), None)
        .unwrap();
    assert_eq!(output.layer(), 3);
    assert_eq!(output.input().unwrap(), &[1.0; 256]);
    assert_eq!(output.normalized_ffn().unwrap(), &[1.25; 256]);
    assert_eq!(output.gate().unwrap(), &[1.25; 16]);
    assert_eq!(output.up().unwrap(), &[1.25; 16]);
    assert_eq!(output.gated().unwrap(), &[1.21875; 16]);
    assert_eq!(output.branch().unwrap(), &[1.21875; 256]);
    assert_eq!(output.residual().unwrap(), &[2.21875; 256]);
    assert_eq!(output.normalized().unwrap(), &[1.25; 256]);
    assert_eq!(output.full_vocabulary().unwrap(), &[-320.0; 16]);
    assert_eq!(reserved(&admission), before + 9 * 1024 * 1024);
    let identity = source.source_identity();
    drop(operators);
    drop(source);
    assert_eq!(output.source_identity(), identity);
    assert_eq!(reserved(&admission), 41 * 1024 * 1024);
    assert_eq!(output.full_vocabulary().unwrap(), &[-320.0; 16]);
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
fn output_tail_records_one_existing_schedule_without_state_or_external_producer() {
    let fixture = Fixture::new();
    let loaded = tail_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let mut builder = PlanBuilder::new_operator_slice(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    let value = builder
        .embed_profiled(
            operators
                .matrix_descriptor(QwenMatrixRole::Embedding)
                .unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let values = operators.append_output_tail(&mut builder, &value).unwrap();
    let recorded = builder.finish_operator_slice(values.logits).unwrap();
    let plan = recorded.plan();
    assert_eq!(plan.nodes().len(), 9);
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::Silu { multiplied: true }))
            .count(),
        1
    );
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::Add))
            .count(),
        1
    );
    assert!(
        plan.precisions()
            .iter()
            .all(|p| *p == ActivationPrecision::Bf16WidenedF32)
    );
    assert!(plan.required_domains().is_empty());
    assert!(plan.proposed_domains().is_empty());
    assert!(plan.nodes().iter().all(|n| n.state.is_none()));
    drop(operators);
    drop(source);
    drop(recorded);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn output_tail_rejects_foreign_short_cancelled_and_pressure_inputs_without_leaks() {
    let fixture = Fixture::new();
    let loaded = tail_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let before = reserved(&admission);
    assert!(
        operators
            .execute_output_tail(input(&operators, &bytes), Some(&AtomicBool::new(true)))
            .is_err()
    );
    assert!(
        operators
            .execute_output_tail(input(&operators, &bytes[..bytes.len() - 2]), None)
            .is_err()
    );
    let (foreign, foreign_admission) = windows(&fixture, &loaded);
    let foreign_input = PureSliceInput::from_bf16_bytes(
        foreign.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        &bytes,
    )
    .unwrap();
    assert!(operators.execute_output_tail(foreign_input, None).is_err());
    assert_eq!(reserved(&admission), before);
    drop(foreign);
    assert_eq!(reserved(&foreign_admission), 0);
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
    let held = reserved(&admission);
    assert!(
        operators
            .execute_output_tail(input(&operators, &bytes), None)
            .is_err()
    );
    assert_eq!(reserved(&admission), held);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn output_tail_refuses_unqualified_scalar_silu_tail_before_private_admission() {
    let fixture = Fixture::new();
    let loaded = source_with_options(&fixture, false, 256, true, |_| {});
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let bytes = 0x3f80u16.to_le_bytes().repeat(256);
    let before = reserved(&admission);
    // This original fixture's intermediate_size8 needs Torch's different scalar path.
    assert!(
        operators
            .execute_output_tail(input(&operators, &bytes), None)
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}

#[test]
fn output_tail_pointwise_keeps_bf16_stage_casts_and_cancellation() {
    let fixture = Fixture::new();
    let loaded = tail_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let cancelled = AtomicBool::new(false);
    let mut math =
        crate::host_math::HostMath::for_window_operator_slice(&operators, Some(&cancelled), None);
    let gates = [
        -4.0f32, -2.0, -1.0, -0.5, -0.0, 0.0, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0,
    ];
    let bytes = gates
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let gate = math
        .import_pure_input(input(&operators, &bytes), 16)
        .unwrap();
    let primitive =
        zerogravity_cpu::original_hf::transcendentals::silu_bf16_widened_f32_contiguous16(
            source.source_identity(),
            &gates,
        )
        .unwrap();
    let mut staged_differs = false;
    for bits in [0x3eabu16, 0x3f33, 0x3f80, 0x3fa0, 0x3fc0, 0x4000, 0x40e0] {
        let bytes = bits.to_le_bytes().repeat(16);
        let up = math
            .import_pure_input(input(&operators, &bytes), 16)
            .unwrap();
        let actual = math.torch_cpu_bf16_silu_mul("tail", &gate, &up).unwrap();
        let multiplier = f32::from_bits(u32::from(bits) << 16);
        for (&value, &raw) in actual.as_slice().unwrap().iter().zip(&primitive) {
            let round = |v: f32| {
                f32::from_bits(
                    (v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1))) & 0xffff0000,
                )
            };
            assert_eq!(value.to_bits(), round(round(raw) * multiplier).to_bits());
            staged_differs |= value.to_bits() != round(raw * multiplier).to_bits();
        }
    }
    assert!(staged_differs, "fixture must distinguish fused arithmetic");
    let zeros = 0x8000u16.to_le_bytes().repeat(17408);
    let wide = math
        .import_pure_input(input(&operators, &zeros), 17408)
        .unwrap();
    let result = math
        .torch_cpu_bf16_silu_mul("max-width", &wide, &wide)
        .unwrap();
    assert_eq!(result.as_slice().unwrap().len(), 17408);
    assert!(result.as_slice().unwrap().iter().all(|v| v.to_bits() == 0));
    drop(result);
    drop(wide);
    let huge = 0x7f7fu16.to_le_bytes().repeat(16);
    let huge = math
        .import_pure_input(input(&operators, &huge), 16)
        .unwrap();
    let before = reserved(&admission);
    assert!(
        math.torch_cpu_bf16_silu_mul("overflow", &gate, &huge)
            .is_err()
    );
    assert!(math.torch_cpu_bf16_add("overflow", &huge, &huge).is_err());
    assert_eq!(reserved(&admission), before);
    drop(huge);
    let before = reserved(&admission);
    cancelled.store(true, Ordering::Release);
    assert!(
        math.torch_cpu_bf16_silu_mul("cancelled", &gate, &gate)
            .is_err()
    );
    assert!(math.torch_cpu_bf16_add("cancelled", &gate, &gate).is_err());
    assert_eq!(reserved(&admission), before);
    drop(gate);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
