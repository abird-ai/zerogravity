use super::*;
fn rows(source: SourceIdentity) -> TeacherSpan {
    TeacherSpan::new(source, 0, 128, 16, &[1, 2]).unwrap()
}
fn call(span: TeacherSpan) -> SourceCall {
    SourceCall::new(
        SourceRows::from_teacher_span(span, 16).unwrap(),
        SourceCallMode::Prefill,
    )
    .unwrap()
}
fn norm(source: SourceIdentity) -> SourceParameter {
    SourceParameter {
        source,
        name: "fixture.norm".into(),
        kind: QuantKind::F32,
        shape: vec![32],
        bytes: 128,
    }
}
fn input(call: SourceCall) -> SourceOperator {
    SourceOperator::new(
        call.clone(),
        SourceOperatorKind::RetainedRows {
            width: 32,
            origin: SourceRetainedOrigin::Normalized {
                epsilon_bits: 1e-6_f32.to_bits(),
            },
        },
        SourceNumericalContract::GgmlSourceF32V1,
        vec![],
        vec![norm(call.source())],
    )
    .unwrap()
}
fn projection(call: SourceCall) -> SourceOperator {
    SourceOperator::new(
        call.clone(),
        SourceOperatorKind::Projection { selected_row: None },
        SourceNumericalContract::GgmlQ4Repack8x8F32V1,
        vec![SourceTensor {
            source: call.source(),
            name: "fixture.matrix".into(),
            kind: QuantKind::Q4_0,
            axes: MatrixAxes::InputOutput,
            columns: 32,
            rows: 96,
        }],
        vec![],
    )
    .unwrap()
}
fn one(builder: &mut PlanBuilder, op: &SourceOperator, inputs: &[PlanValue]) -> Result<PlanValue> {
    Ok(builder.source_operator(op, inputs)?.remove(0))
}
#[test]
fn f32_source_chain_keeps_issuer_stage_precision_and_state_free_boundary() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = call(rows(source));
    let import = input(call.clone());
    let project = projection(call.clone());
    let mut builder =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1).unwrap();
    let retained = one(&mut builder, &import, &[]).unwrap();
    let foreign = projection(
        super::super::SourceCall::new(
            SourceRows::from_teacher_span(rows(source), 16).unwrap(),
            SourceCallMode::Prefill,
        )
        .unwrap(),
    );
    assert!(one(&mut builder, &foreign, &[retained]).is_err());
    let out = one(&mut builder, &project, &[retained]).unwrap();
    assert!(one(&mut builder, &project, &[out]).is_err());
    assert!(builder.view(&out, TensorView::vector(0, 192)).is_err());
    assert!(builder.silu("legacy", &out, None).is_err());
    let conv = SourceOperator::new(
        call.clone(),
        SourceOperatorKind::CausalConvSiluZero {
            channels: 96,
            kernel: 4,
        },
        SourceNumericalContract::GgmlSourceF32V1,
        vec![],
        vec![SourceParameter {
            source,
            name: "fixture.conv".into(),
            kind: QuantKind::F32,
            shape: vec![4, 96],
            bytes: 4 * 96 * 4,
        }],
    )
    .unwrap();
    assert!(one(&mut builder, &conv, &[retained]).is_err());
    let activated = one(&mut builder, &conv, &[out]).unwrap();
    let qk = SourceOperator::new(
        call,
        SourceOperatorKind::GroupedQkNormalize(SourceGroupedQkGeometry {
            heads: 1,
            width: 32,
            epsilon: 1e-6 / 32.0,
            scale: 1.0 / 32.0_f32.sqrt(),
        }),
        SourceNumericalContract::GgmlSourceF32V1,
        vec![],
        vec![],
    )
    .unwrap();
    assert!(one(&mut builder, &qk, &[out]).is_err());
    let final_ = one(&mut builder, &qk, &[activated]).unwrap();
    let record = builder.finish_operator_slice(final_).unwrap();
    assert!(record.plan().required_domains().is_empty());
    assert!(record.plan().proposed_domains().is_empty());
    assert_eq!(record.plan().frontier_delta(), 0);
    assert!(record.plan().logits().is_none());
    assert!(
        record
            .plan()
            .precisions()
            .iter()
            .all(|p| *p == ActivationPrecision::F32)
    );
    let mut bf16 =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::SourceCudaSm86Bf16V1)
            .unwrap();
    assert!(one(&mut bf16, &import, &[]).is_err());
    let mut full =
        PlanBuilder::new(source, 128, vec![StateSlot::Conv(ConvDomain(0))], false).unwrap();
    assert!(one(&mut full, &import, &[]).is_err());
    let mut other =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1).unwrap();
    assert!(one(&mut other, &project, &[retained]).is_err());
}
#[test]
fn codecs_gamma_and_f32_numerical_schedules_cannot_substitute() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let call = call(rows(source));
    let good = projection(call.clone());
    let mut matrix = good.matrices()[0].clone();
    matrix.kind = QuantKind::Q8_0;
    assert!(
        SourceOperator::new(
            call.clone(),
            good.kind(),
            good.numerical_contract(),
            vec![matrix],
            vec![]
        )
        .is_err()
    );
    assert!(
        SourceOperator::new(
            call.clone(),
            good.kind(),
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            good.matrices().to_vec(),
            vec![]
        )
        .is_err()
    );
    assert!(
        SourceOperator::new(
            call.clone(),
            SourceOperatorKind::Rms(SourceRmsGeometry {
                width: 32,
                epsilon: 1e-6,
                gamma: SourceGamma::OnePlus
            }),
            SourceNumericalContract::GgmlSourceF32V1,
            vec![],
            vec![norm(source)]
        )
        .is_err()
    );
    assert!(
        SourceOperator::new(
            call,
            good.kind(),
            SourceNumericalContract::OpenBlas033Zen16Q8F32V1,
            good.matrices().to_vec(),
            vec![]
        )
        .is_err()
    );
}
#[test]
fn retained_rms_metadata_cannot_retag_source_math_or_layout() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let span = rows(source);
    let layout = BulkF32Layout::dense([32, 2, 1, 1]).unwrap();
    let valid = RetainedRmsRows::new(
        span.clone(),
        layout,
        norm(source),
        1e-6,
        SourceNumericalContract::GgmlSourceF32V1,
    )
    .unwrap();
    let other = RetainedRmsRows::new(
        rows(source),
        layout,
        norm(source),
        1e-6,
        SourceNumericalContract::GgmlSourceF32V1,
    )
    .unwrap();
    assert_ne!(valid, other);
    assert!(
        RetainedRmsRows::new(
            span.clone(),
            layout,
            norm(SourceIdentity::for_new_allocation().unwrap()),
            1e-6,
            SourceNumericalContract::GgmlSourceF32V1
        )
        .is_err()
    );
    for epsilon in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        assert!(
            RetainedRmsRows::new(
                span.clone(),
                layout,
                norm(source),
                epsilon,
                SourceNumericalContract::GgmlSourceF32V1
            )
            .is_err()
        );
    }
    assert!(
        RetainedRmsRows::new(
            span.clone(),
            layout,
            norm(source),
            1e-6,
            SourceNumericalContract::ReferenceCudaSm86Bf16V1
        )
        .is_err()
    );
    assert!(
        RetainedRmsRows::new(
            span,
            BulkF32Layout::dense([32, 1, 1, 1]).unwrap(),
            norm(source),
            1e-6,
            SourceNumericalContract::GgmlSourceF32V1
        )
        .is_err()
    );
}
