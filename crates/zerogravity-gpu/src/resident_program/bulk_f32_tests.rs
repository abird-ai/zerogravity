//! Cold metadata refusal; source callbacks and device initialization are forbidden.
use super::ResidentProgram;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{
        BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Side,
        GgufGdnProjection, GgufGdnUpstreamContract, GgufGdnUpstreamDescriptor, GgufPrefixContract,
        GgufPrefixDescriptor, MathematicalBackend, PlanBuilder, ProgramNumericalProfile,
        TeacherSpan, ZenSsmOutContract, ZenSsmOutDescriptor,
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

struct SyntheticQ8 {
    source: SourceIdentity,
    weights: Vec<u8>,
}
impl TensorSource for SyntheticQ8 {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        if name != "blk.0.ssm_out.weight" {
            return Err(Error::Malformed("unknown synthetic Q8 matrix".into()));
        }
        Ok(SourceTensorRef {
            kind: QuantKind::Q8_0,
            shape: &[2048, 2048],
            packed_bytes: &self.weights,
        })
    }
}

#[test]
fn zen_ssm_out_operator_slice_refuses_before_source_access() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let synthetic = SyntheticQ8 {
        source,
        weights: vec![0; 2048 * 2048 / 32 * 34],
    };
    let descriptor = ZenSsmOutDescriptor::new(
        TeacherSpan::new(source, 0, 256, 512, &[11; 64])?,
        &synthetic,
        "blk.0.ssm_out.weight",
        ZenSsmOutContract::OpenBlas033Zen16SsmOutV1,
    )?;
    drop(synthetic); // The refusal requires metadata only, never its payload.
    for projection in [false, true] {
        let mut builder =
            PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1)?;
        let input = builder.zen_ssm_out_input(&descriptor)?;
        let output = if projection {
            builder.zen_ssm_out_project(&descriptor, &input)?
        } else {
            input
        };
        let slice = builder.finish_operator_slice(output)?;
        assert!(slice.plan().is_operator_slice());
        assert_eq!(slice.plan().nodes().len(), if projection { 2 } else { 1 });
        assert!(matches!(
            ResidentProgram::preflight(&ForbiddenSource, slice.plan()),
            Err(Error::Unsupported(_))
        ));
    }
    Ok(())
}

struct SyntheticGgufPrefix {
    source: SourceIdentity,
    embedding: Vec<u8>,
    norm: Vec<u8>,
}
impl TensorSource for SyntheticGgufPrefix {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        match name {
            "token_embd.weight" => Ok(SourceTensorRef {
                kind: QuantKind::Q6K,
                shape: &[2048, 16],
                packed_bytes: &self.embedding,
            }),
            "blk.0.attn_norm.weight" => Ok(SourceTensorRef {
                kind: QuantKind::F32,
                shape: &[2048],
                packed_bytes: &self.norm,
            }),
            _ => Err(Error::Malformed(
                "unknown synthetic GGUF prefix tensor".into(),
            )),
        }
    }
}

#[test]
fn gguf_prefix_operator_slice_refuses_before_source_access() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let synthetic = SyntheticGgufPrefix {
        source,
        embedding: vec![0; 2048 * 16 / 256 * 210],
        norm: vec![0; 2048 * 4],
    };
    let descriptor = GgufPrefixDescriptor::new(
        TeacherSpan::new(source, 0, 8448, 16, &[11; 64])?,
        &synthetic,
        GgufPrefixContract::GgufSource2BFirst64V1,
        1e-6,
    )?;
    drop(synthetic);
    for normalized in [false, true] {
        let mut builder =
            PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1)?;
        let embedding = builder.gguf_prefix_embedding(&descriptor)?;
        let output = if normalized {
            builder.gguf_prefix_rms(&descriptor, &embedding)?
        } else {
            embedding
        };
        let slice = builder.finish_operator_slice(output)?;
        assert_eq!(slice.plan().nodes().len(), if normalized { 2 } else { 1 });
        assert!(matches!(
            ResidentProgram::preflight(&ForbiddenSource, slice.plan()),
            Err(Error::Unsupported(_))
        ));
    }
    Ok(())
}

struct SyntheticUpstream {
    prefix: SyntheticGgufPrefix,
    qkv: Vec<u8>,
    z: Vec<u8>,
    convolution: Vec<u8>,
}
impl TensorSource for SyntheticUpstream {
    fn source_identity(&self) -> SourceIdentity {
        self.prefix.source
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        let (kind, shape, packed_bytes): (_, &[u64], &[u8]) = match name {
            "blk.0.attn_qkv.weight" => (QuantKind::Q4_0, &[2048, 6144], &self.qkv),
            "blk.0.attn_gate.weight" => (QuantKind::Q4_0, &[2048, 2048], &self.z),
            "blk.0.ssm_conv1d.weight" => (QuantKind::F32, &[4, 6144], &self.convolution),
            _ => return self.prefix.tensor(name),
        };
        Ok(SourceTensorRef {
            kind,
            shape,
            packed_bytes,
        })
    }
}

#[test]
fn gguf_upstream_operator_slice_refuses_before_source_access() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let synthetic = SyntheticUpstream {
        prefix: SyntheticGgufPrefix {
            source,
            embedding: vec![0; 2048 * 16 / 256 * 210],
            norm: vec![0; 8192],
        },
        qkv: vec![0; 1152 * 6144],
        z: vec![0; 1152 * 2048],
        convolution: vec![0; 4 * 6144 * 4],
    };
    let prefix = GgufPrefixDescriptor::new(
        TeacherSpan::new(source, 0, 8448, 16, &[11; 64])?,
        &synthetic,
        GgufPrefixContract::GgufSource2BFirst64V1,
        1e-6,
    )?;
    let descriptor = GgufGdnUpstreamDescriptor::new(
        prefix,
        &synthetic,
        GgufGdnUpstreamContract::GgufQ4Repacked2BFirst64V1,
    )?;
    drop(synthetic);
    let mut builder =
        PlanBuilder::new_operator_slice(source, ProgramNumericalProfile::GgufSourceV1)?;
    let input = builder.gguf_gdn_input(&descriptor)?;
    let qkv = builder.gguf_gdn_project(&descriptor, GgufGdnProjection::Qkv, &input)?;
    let _z = builder.gguf_gdn_project(&descriptor, GgufGdnProjection::Z, &input)?;
    let convolution = builder.gguf_gdn_conv_silu(&descriptor, &qkv)?;
    let operands = builder.gguf_gdn_qk(&descriptor, &convolution)?;
    let recorded = builder.finish_operator_slice(operands)?;
    assert_eq!(recorded.plan().nodes().len(), 5);
    assert!(matches!(
        ResidentProgram::preflight(&ForbiddenSource, recorded.plan()),
        Err(Error::Unsupported(_))
    ));
    Ok(())
}
