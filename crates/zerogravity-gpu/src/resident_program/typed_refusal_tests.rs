use super::*;
use zerogravity_core::accelerator::program::{
    ConvDomain, EmbeddingContract, MathematicalBackend, MatrixAxes, PlanBuilder,
    ProgramNumericalProfile, ProjectionContract, SourceNumericalContract, SourceOperatorKind,
    SourceOperatorTemplate, SourcePlanGeometry, SourceRowSelection, StateSlot,
};

struct UnreadSource;
impl TensorSource for UnreadSource {
    fn source_identity(&self) -> SourceIdentity {
        panic!("cold unsupported plan must not inspect source owner")
    }
    fn tensor(&self, _: &str) -> Result<zerogravity_core::source::SourceTensorRef<'_>> {
        panic!("cold unsupported plan must not inspect/upload source payload")
    }
}

#[test]
fn typed_original_slice_refuses_before_source_reads_or_resident_budget_work() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let tensor = SourceTensor {
        source,
        name: "original.embedding".into(),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        columns: 4,
        rows: 8,
    };
    let mut builder = PlanBuilder::new_operator_slice(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    let input = builder
        .embed_profiled(
            tensor.clone(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let output = builder
        .project_profiled(tensor, &input, ProjectionContract::TorchCpuBf16Avx2)
        .unwrap();
    let slice = builder.finish_operator_slice(output).unwrap();
    assert!(matches!(
        ResidentProgram::preflight(&UnreadSource, slice.plan()),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn call_free_source_template_refuses_before_legacy_source_or_batch_lowering() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let geometry = SourcePlanGeometry::new(source, 16, 512, 1).unwrap();
    let matrix = |name: &str, rows, columns| SourceTensor {
        source,
        name: name.into(),
        kind: QuantKind::Bf16,
        axes: MatrixAxes::OutputInput,
        columns,
        rows,
    };
    let slot = StateSlot::Conv(ConvDomain(0));
    let mut builder = PlanBuilder::new_source_template(geometry, vec![slot]).unwrap();
    let embedding = SourceOperatorTemplate::new(
        geometry,
        SourceRowSelection::AllActive,
        SourceOperatorKind::EmbeddingRows,
        SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        vec![matrix("embedding", 16, 4)],
        vec![],
    )
    .unwrap();
    let [input] = builder
        .record_source_template("embedding", &embedding, &[])
        .unwrap()
        .try_into()
        .unwrap();
    let carry = SourceOperatorTemplate::new(
        geometry,
        SourceRowSelection::AllActive,
        SourceOperatorKind::StateCarry { slot },
        SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        vec![],
        vec![],
    )
    .unwrap();
    assert!(
        builder
            .record_source_template("state", &carry, &[])
            .unwrap()
            .is_empty()
    );
    let head = SourceOperatorTemplate::new(
        geometry,
        SourceRowSelection::LastActive,
        SourceOperatorKind::Projection { selected_row: None },
        SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        vec![matrix("head", 8, 4)],
        vec![],
    )
    .unwrap();
    let [output] = builder
        .record_source_template("head", &head, &[input])
        .unwrap()
        .try_into()
        .unwrap();
    let plan = builder.finish_source_template(output).unwrap();
    assert!(matches!(
        ResidentProgram::require_legacy_contract(&plan),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        ResidentProgram::preflight(&UnreadSource, &plan),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn even_legacy_operator_fragment_cannot_impersonate_full_state_transition() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let tensor = SourceTensor {
        source,
        name: "embedding".into(),
        kind: QuantKind::F32,
        axes: MatrixAxes::InputOutput,
        columns: 4,
        rows: 8,
    };
    let mut builder =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1).unwrap();
    let output = builder.embed(tensor, PlanBuilder::inputs().token).unwrap();
    let slice = builder.finish_operator_slice(output).unwrap();
    assert!(matches!(
        ResidentProgram::preflight(&UnreadSource, slice.plan()),
        Err(Error::Unsupported(_))
    ));
}
