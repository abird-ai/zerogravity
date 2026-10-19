use super::*;
use zerogravity_core::accelerator::program::{
    EmbeddingContract, MathematicalBackend, MatrixAxes, PlanBuilder, ProgramNumericalProfile,
    ProjectionContract,
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
