//! Source-free native contract refusal, before source/device/provider access.
use super::ResidentProgram;
use zerogravity_core::accelerator::{SourceIdentity, program::*};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::source::{SourceTensorRef, TensorSource};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::plan::QwenPlanBuilder;
use zerogravity_model_qwen35::plan::{
    GdnProjection, QwenEmbedding, QwenGdnCall, QwenGdnProjection, QwenInputRows, QwenPrefixLayer,
    QwenSiluAndMul, ReferencePhase,
};

struct ForbiddenSource;
impl TensorSource for ForbiddenSource {
    fn source_identity(&self) -> SourceIdentity {
        panic!("native refusal must precede source identity");
    }
    fn tensor(&self, _name: &str) -> Result<SourceTensorRef<'_>> {
        panic!("native refusal must precede tensor access");
    }
}
#[test]
fn both_native_typed_operators_refuse_cold_before_source_access() -> Result<()> {
    for activate in [false, true] {
        let source = SourceIdentity::for_new_allocation()?;
        let teachers = QwenInputRows::new(source, 0, 512, &[11])?;
        let call = QwenGdnCall::new(
            QwenPrefixLayer::new(source, 0)?,
            ReferencePhase::InitialOne,
            teachers.clone(),
        )?;
        let mut b = QwenPlanBuilder::new_operator_slice(
            source,
            ProgramNumericalProfile::SourceCudaSm86Bf16V1,
        )?;
        let input = b.source_embedding(&QwenEmbedding::new(
            SourceTensor {
                source,
                name: "model.language_model.embed_tokens.weight".into(),
                kind: QuantKind::Bf16,
                axes: MatrixAxes::OutputInput,
                columns: 1024,
                rows: 248320,
            },
            teachers,
        )?)?;
        let gate = QwenGdnProjection::new(
            call.clone(),
            GdnProjection::GateUp,
            ["gate", "up"]
                .into_iter()
                .map(|role| SourceTensor {
                    source,
                    name: format!("model.language_model.layers.0.mlp.{role}_proj.weight"),
                    kind: QuantKind::Bf16,
                    axes: MatrixAxes::OutputInput,
                    columns: 1024,
                    rows: 3584,
                })
                .collect(),
        )?;
        let output = b.gdn_project(&gate, &input)?;
        let output = if activate {
            b.silu_and_mul(&QwenSiluAndMul::new(call)?, &output)?
        } else {
            output
        };
        let slice = b.finish_operator_slice(output)?;
        assert!(matches!(
            ResidentProgram::preflight(&ForbiddenSource, slice.plan()),
            Err(Error::Unsupported(_))
        ));
    }
    Ok(())
}
