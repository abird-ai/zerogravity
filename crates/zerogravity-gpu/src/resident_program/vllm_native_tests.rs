//! Source-free native contract refusal, before source/device/provider access.
use super::ResidentProgram;
use zerogravity_core::accelerator::{SourceIdentity, program::*};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::source::{SourceTensorRef, TensorSource};
use zerogravity_core::{Error, Result};

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
        let teachers = VllmBf16Teachers::new(source, 0, 512, &[11])?;
        let call = VllmBf16Invocation::new(
            GdnPrefixLayer::new(source, 0)?,
            VllmBf16Phase::InitialOne,
            teachers.clone(),
        )?;
        let mut b = PlanBuilder::new_operator_slice(
            source,
            ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
        )?;
        let input = b.vllm_bf16_source_embedding(&VllmBf16Embedding::new(
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
        let gate = VllmGdnProjection::new(
            call.clone(),
            VllmGdnProjectionKind::GateUp,
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
        let output = b.vllm_gdn_project(&gate, &input)?;
        let output = if activate {
            b.vllm_silu_and_mul(&VllmSiluAndMul::new(call)?, &output)?
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
