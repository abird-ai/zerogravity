//! Explicit closed numerical contracts; none grants model/backend eligibility.
use super::{MatrixAxes, SourceTensor};
use crate::accelerator::SourceIdentity;
use crate::quant::QuantKind;
use crate::source::TensorSource;
use crate::{Error, Result};

/// Mathematical precision carried in F32 storage, independent of weight codec.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ActivationPrecision {
    #[default]
    F32,
    Bf16WidenedF32,
}
impl ActivationPrecision {
    pub fn validate_values(self, values: &[f32]) -> Result<()> {
        if values.is_empty()
            || values.iter().any(|value| {
                !value.is_finite() || self == Self::Bf16WidenedF32 && value.to_bits() & 0xffff != 0
            })
        {
            return Err(Error::Malformed(
                "activation differs from explicit precision".into(),
            ));
        }
        Ok(())
    }
}

/// Reserved program contract, not inference support or a backend-global dtype.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProgramNumericalProfile {
    #[default]
    GgufSourceV1,
    TorchCpuBf16Avx2MixedSourceV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingContract {
    Source,
    Bf16OutputInput,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionContract {
    Source,
    TorchCpuBf16Avx2,
}

pub(super) fn validate_bf16_matrix(tensor: &SourceTensor) -> Result<()> {
    if tensor.kind != QuantKind::Bf16 || tensor.axes != MatrixAxes::OutputInput {
        return Err(Error::Unsupported(
            "closed original projection requires BF16 output-first source".into(),
        ));
    }
    Ok(())
}

/// Unchanged small learned parameter metadata; payload stays with TensorSource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceParameter {
    pub source: SourceIdentity,
    pub name: String,
    pub kind: QuantKind,
    pub shape: Vec<u64>,
    pub bytes: usize,
}
impl SourceParameter {
    pub fn from_source(source: &dyn TensorSource, name: &str) -> Result<Self> {
        let identity = source.source_identity();
        if !identity.is_bound() || name.is_empty() {
            return Err(Error::State("unbound source parameter".into()));
        }
        let actual = source.tensor(name)?;
        actual.validate()?;
        if actual.shape.len() > 8 {
            return Err(Error::Limit("source parameter rank bound".into()));
        }
        Ok(Self {
            source: identity,
            name: name.into(),
            kind: actual.kind,
            shape: actual.shape.to_vec(),
            bytes: actual.packed_bytes.len(),
        })
    }
    pub fn validate_source(&self, source: &dyn TensorSource) -> Result<()> {
        if !self.source.is_bound()
            || self.source != source.source_identity()
            || self.name.is_empty()
            || self.shape.len() > 8
        {
            return Err(Error::State("source parameter owner/bounds differ".into()));
        }
        let actual = source.tensor(&self.name)?;
        actual.validate()?;
        if actual.kind != self.kind
            || actual.shape != self.shape.as_slice()
            || actual.packed_bytes.len() != self.bytes
        {
            return Err(Error::State(
                "source parameter differs from immutable owner".into(),
            ));
        }
        Ok(())
    }
}
