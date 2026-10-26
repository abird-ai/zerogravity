//! Cold metadata refusal; source callbacks and device initialization are forbidden.
use super::ResidentProgram;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{
        BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Side,
        MathematicalBackend, MatrixAxes, PlanBuilder, ProgramNumericalProfile, SourceCall,
        SourceCallMode, SourceGamma, SourceGroupedQkGeometry, SourceNumericalContract,
        SourceOperator, SourceOperatorKind, SourceParameter, SourceRetainedOrigin,
        SourceRmsGeometry, SourceRows, SourceTensor, TeacherSpan,
    },
};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::source::{SourceTensorRef, TensorSource};
use zerogravity_core::{Error, Result};

struct ForbiddenSource;

impl TensorSource for ForbiddenSource {
    fn source_identity(&self) -> SourceIdentity {
        panic!("bulk refusal must precede source identity access");
    }

    fn tensor(&self, _name: &str) -> Result<SourceTensorRef<'_>> {
        panic!("bulk refusal must precede tensor access");
    }
}

#[test]
fn bulk_operator_slice_refuses_before_source_access() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let span = TeacherSpan::new(source, 0, 256, 512, &[11, 353])?;
    let left = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Left,
        "synthetic left",
        BulkF32Layout::dense([8, 4, 1, 1])?,
    )?;
    let right = BulkF32Operand::external(
        span,
        BulkF32Side::Right,
        "synthetic right",
        BulkF32Layout::dense([8, 2, 1, 1])?,
    )?;
    let matrix = BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1)?;
    let mut builder =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1)?;
    let left = builder.bulk_f32_input(matrix.left())?;
    let right = builder.bulk_f32_input(matrix.right())?;
    let output = builder.bulk_f32_matmul(&matrix, &left, &right)?;
    let slice = builder.finish_operator_slice(output)?;
    assert!(slice.plan().is_operator_slice());
    assert_eq!(slice.plan().nodes().len(), 3);
    assert!(matches!(
        ResidentProgram::preflight(&ForbiddenSource, slice.plan()),
        Err(Error::Unsupported(_))
    ));
    Ok(())
}

fn source_call(source: SourceIdentity) -> Result<SourceCall> {
    SourceCall::new(
        SourceRows::from_teacher_span(TeacherSpan::new(source, 0, 128, 16, &[1; 64])?, 16)?,
        SourceCallMode::Prefill,
    )
}
fn source_one(
    builder: &mut PlanBuilder,
    op: SourceOperator,
    inputs: &[zerogravity_core::accelerator::program::PlanValue],
) -> Result<zerogravity_core::accelerator::program::PlanValue> {
    Ok(builder.source_operator(&op, inputs)?.remove(0))
}
#[test]
fn quantized_projection_cut_refuses_before_source_callbacks() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let call = source_call(source)?;
    let mut b = PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1)?;
    let input = source_one(
        &mut b,
        SourceOperator::new(
            call.clone(),
            SourceOperatorKind::RetainedRows {
                width: 2048,
                origin: SourceRetainedOrigin::Gated,
            },
            SourceNumericalContract::OpenBlas033Zen16Q8F32V1,
            vec![],
            vec![],
        )?,
        &[],
    )?;
    let output = source_one(
        &mut b,
        SourceOperator::new(
            call,
            SourceOperatorKind::Projection { selected_row: None },
            SourceNumericalContract::OpenBlas033Zen16Q8F32V1,
            vec![SourceTensor {
                source,
                name: "fixture.projection".into(),
                kind: QuantKind::Q8_0,
                axes: MatrixAxes::InputOutput,
                columns: 2048,
                rows: 2048,
            }],
            vec![],
        )?,
        &[input],
    )?;
    let plan = b.finish_operator_slice(output)?;
    assert!(matches!(
        ResidentProgram::preflight(&ForbiddenSource, plan.plan()),
        Err(Error::Unsupported(_))
    ));
    Ok(())
}
#[test]
fn embedding_direct_rms_cut_refuses_before_source_callbacks() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let call = source_call(source)?;
    let mut b = PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1)?;
    let input = source_one(
        &mut b,
        SourceOperator::new(
            call.clone(),
            SourceOperatorKind::EmbeddingRows,
            SourceNumericalContract::GgmlSourceF32V1,
            vec![SourceTensor {
                source,
                name: "fixture.embedding".into(),
                kind: QuantKind::Q6K,
                axes: MatrixAxes::InputOutput,
                columns: 256,
                rows: 16,
            }],
            vec![],
        )?,
        &[],
    )?;
    let output = source_one(
        &mut b,
        SourceOperator::new(
            call,
            SourceOperatorKind::Rms(SourceRmsGeometry {
                width: 256,
                epsilon: 1e-6,
                gamma: SourceGamma::Direct,
            }),
            SourceNumericalContract::GgmlSourceF32V1,
            vec![],
            vec![SourceParameter {
                source,
                name: "fixture.gamma".into(),
                kind: QuantKind::F32,
                shape: vec![256],
                bytes: 1024,
            }],
        )?,
        &[input],
    )?;
    let plan = b.finish_operator_slice(output)?;
    assert!(matches!(
        ResidentProgram::preflight(&ForbiddenSource, plan.plan()),
        Err(Error::Unsupported(_))
    ));
    Ok(())
}
#[test]
fn quantized_conv_qk_chain_refuses_before_source_callbacks() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let call = source_call(source)?;
    let mut b = PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1)?;
    let input = source_one(
        &mut b,
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
            vec![SourceParameter {
                source,
                name: "fixture.gamma".into(),
                kind: QuantKind::F32,
                shape: vec![32],
                bytes: 128,
            }],
        )?,
        &[],
    )?;
    let projected = source_one(
        &mut b,
        SourceOperator::new(
            call.clone(),
            SourceOperatorKind::Projection { selected_row: None },
            SourceNumericalContract::GgmlQ4Repack8x8F32V1,
            vec![SourceTensor {
                source,
                name: "fixture.projection".into(),
                kind: QuantKind::Q4_0,
                axes: MatrixAxes::InputOutput,
                columns: 32,
                rows: 96,
            }],
            vec![],
        )?,
        &[input],
    )?;
    let conv = source_one(
        &mut b,
        SourceOperator::new(
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
        )?,
        &[projected],
    )?;
    let output = source_one(
        &mut b,
        SourceOperator::new(
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
        )?,
        &[conv],
    )?;
    let plan = b.finish_operator_slice(output)?;
    assert!(matches!(
        ResidentProgram::preflight(&ForbiddenSource, plan.plan()),
        Err(Error::Unsupported(_))
    ));
    Ok(())
}
