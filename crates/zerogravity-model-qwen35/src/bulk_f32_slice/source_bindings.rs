//! Closed Qwen GGUF roles and finite recipes; common graph contains only math.
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::*;
use zerogravity_core::quant::QuantKind;
use zerogravity_core::source::TensorSource;
use zerogravity_core::{Error, Result};

// Prior core type: GgufPrefixDescriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixContract {
    First64GgmlF32V1,
}

/// Metadata alone cannot supply the actual admitted prefix backing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixBinding {
    span: TeacherSpan,
    embedding: SourceTensor,
    norm: SourceParameter,
    contract: PrefixContract,
    layout: BulkF32Layout,
    epsilon_bits: u32,
}
impl PrefixBinding {
    pub fn new(
        span: TeacherSpan,
        source: &dyn TensorSource,
        contract: PrefixContract,
        epsilon: f32,
    ) -> Result<Self> {
        let value = Self {
            span,
            embedding: SourceTensor::from_source(
                source,
                "token_embd.weight",
                MatrixAxes::InputOutput,
            )?,
            norm: SourceParameter::from_source(source, "blk.0.attn_norm.weight")?,
            contract,
            layout: BulkF32Layout::dense([2048, 64, 1, 1])?,
            epsilon_bits: epsilon.to_bits(),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn span(&self) -> &TeacherSpan {
        &self.span
    }
    pub fn source(&self) -> SourceIdentity {
        self.span.source()
    }
    pub fn embedding(&self) -> &SourceTensor {
        &self.embedding
    }
    pub fn norm(&self) -> &SourceParameter {
        &self.norm
    }
    pub fn contract(&self) -> PrefixContract {
        self.contract
    }
    pub fn layout(&self) -> BulkF32Layout {
        self.layout
    }
    pub fn epsilon(&self) -> f32 {
        f32::from_bits(self.epsilon_bits)
    }
    pub fn validate(&self) -> Result<()> {
        if self.span.base() != 0
            || self.span.len() != 64
            || self
                .span
                .ids()
                .iter()
                .any(|&id| u64::from(id) >= self.embedding.rows as u64)
            || self.embedding.source != self.source()
            || self.embedding.name != "token_embd.weight"
            || self.embedding.kind != QuantKind::Q6K
            || self.embedding.axes != MatrixAxes::InputOutput
            || self.embedding.columns != 2048
            || !(1..=248320).contains(&self.embedding.rows)
            || self.norm.source != self.source()
            || self.norm.name != "blk.0.attn_norm.weight"
            || self.norm.kind != QuantKind::F32
            || self.norm.shape != [2048]
            || self.norm.bytes != 8192
            || self.epsilon_bits != 1e-6_f32.to_bits()
            || self.layout != BulkF32Layout::dense([2048, 64, 1, 1])?
        {
            return Err(Error::Unsupported(
                "closed original 2B first64 GGUF prefix".into(),
            ));
        }
        Ok(())
    }
}
// Prior core type: GgufGdnUpstreamDescriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnUpstreamContract {
    First64Q4RepackF32V1,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnProjection {
    Qkv,
    Z,
}

pub struct UpstreamValues<V> {
    pub qkv: V,
    pub z: V,
    pub convolution: V,
    pub operands: V,
}

/// Metadata binds ONE retained normalized issuer; runtime backing is separate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnUpstreamBinding {
    prefix: PrefixBinding,
    qkv: SourceTensor,
    z: SourceTensor,
    convolution: SourceParameter,
    contract: GdnUpstreamContract,
}
impl GdnUpstreamBinding {
    pub fn new(
        prefix: PrefixBinding,
        source: &dyn TensorSource,
        contract: GdnUpstreamContract,
    ) -> Result<Self> {
        let value = Self {
            prefix,
            qkv: SourceTensor::from_source(
                source,
                "blk.0.attn_qkv.weight",
                MatrixAxes::InputOutput,
            )?,
            z: SourceTensor::from_source(
                source,
                "blk.0.attn_gate.weight",
                MatrixAxes::InputOutput,
            )?,
            convolution: SourceParameter::from_source(source, "blk.0.ssm_conv1d.weight")?,
            contract,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn prefix(&self) -> &PrefixBinding {
        &self.prefix
    }
    pub fn span(&self) -> &TeacherSpan {
        self.prefix.span()
    }
    pub fn source(&self) -> SourceIdentity {
        self.prefix.source()
    }
    pub fn contract(&self) -> GdnUpstreamContract {
        self.contract
    }
    pub fn tensor(&self, projection: GdnProjection) -> &SourceTensor {
        match projection {
            GdnProjection::Qkv => &self.qkv,
            GdnProjection::Z => &self.z,
        }
    }
    pub fn convolution(&self) -> &SourceParameter {
        &self.convolution
    }
    pub fn validate(&self) -> Result<()> {
        self.prefix.validate()?;
        for (tensor, name, rows) in [
            (&self.qkv, "blk.0.attn_qkv.weight", 6144),
            (&self.z, "blk.0.attn_gate.weight", 2048),
        ] {
            if tensor.source != self.source()
                || tensor.name != name
                || tensor.kind != QuantKind::Q4_0
                || tensor.axes != MatrixAxes::InputOutput
                || tensor.columns != 2048
                || tensor.rows != rows
            {
                return Err(Error::Unsupported(
                    "closed GGUF original Q4_0 QKV/Z geometry".into(),
                ));
            }
        }
        if self.convolution.source != self.source()
            || self.convolution.name != "blk.0.ssm_conv1d.weight"
            || self.convolution.kind != QuantKind::F32
            || self.convolution.shape != [4, 6144]
            || self.convolution.bytes != 4 * 6144 * 4
        {
            return Err(Error::Unsupported(
                "closed GGUF original convolution geometry".into(),
            ));
        }
        Ok(())
    }
}
// Prior core type: ZenSsmOutDescriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnOutputContract {
    OpenBlas033Zen16F32V1,
}

/// Exact source matrix and ONE retained gated producer's teacher issuer.
/// Metadata alone cannot supply a runtime activation or its physical owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnOutputBinding {
    span: TeacherSpan,
    tensor: SourceTensor,
    contract: GdnOutputContract,
    layout: BulkF32Layout,
}
impl GdnOutputBinding {
    pub fn new(
        span: TeacherSpan,
        source: &dyn TensorSource,
        name: &str,
        contract: GdnOutputContract,
    ) -> Result<Self> {
        let value = Self {
            span,
            tensor: SourceTensor::from_source(source, name, MatrixAxes::InputOutput)?,
            contract,
            layout: BulkF32Layout::dense([2048, 64, 1, 1])?,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn span(&self) -> &TeacherSpan {
        &self.span
    }
    pub fn tensor(&self) -> &SourceTensor {
        &self.tensor
    }
    pub fn contract(&self) -> GdnOutputContract {
        self.contract
    }
    pub fn layout(&self) -> BulkF32Layout {
        self.layout
    }
    pub fn source(&self) -> SourceIdentity {
        self.span.source()
    }
    fn validate(&self) -> Result<()> {
        if self.span.len() != 64
            || self.tensor.source != self.source()
            || self.tensor.name != "blk.0.ssm_out.weight"
            || self.tensor.kind != QuantKind::Q8_0
            || self.tensor.axes != MatrixAxes::InputOutput
            || self.tensor.columns != 2048
            || self.tensor.rows != 2048
            || self.layout != BulkF32Layout::dense([2048, 64, 1, 1])?
        {
            return Err(Error::Unsupported(
                "closed Zen original layer0 Q8_0 ssm_out/64-teacher descriptor".into(),
            ));
        }
        Ok(())
    }
}

fn call(span: &TeacherSpan) -> Result<SourceCall> {
    let vocabulary = usize::try_from(span.vocabulary())
        .map_err(|_| Error::Limit("original teacher vocabulary".into()))?;
    SourceCall::new(
        SourceRows::from_teacher_span(span.clone(), vocabulary)?,
        SourceCallMode::Prefill,
    )
}
impl PrefixBinding {
    pub fn embedding_operator(&self) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            call(self.span())?,
            SourceOperatorKind::EmbeddingRows,
            SourceNumericalContract::GgmlSourceF32V1,
            vec![self.embedding.clone()],
            vec![],
        )
    }
    pub fn rms_operator(&self) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            call(self.span())?,
            SourceOperatorKind::Rms(SourceRmsGeometry {
                width: 2048,
                epsilon: self.epsilon(),
                gamma: SourceGamma::Direct,
            }),
            SourceNumericalContract::GgmlSourceF32V1,
            vec![],
            vec![self.norm.clone()],
        )
    }
    pub fn require_retained_rows(&self, binding: &RetainedRmsRows) -> Result<()> {
        self.validate()?;
        if !binding.matches_rms(
            &self.span,
            self.layout,
            &self.norm,
            self.epsilon(),
            SourceNumericalContract::GgmlSourceF32V1,
        ) {
            return Err(Error::State(
                "prefix exact retained RMS metadata/issuer".into(),
            ));
        }
        Ok(())
    }
    pub fn retained_rows(&self) -> Result<RetainedRmsRows> {
        self.validate()?;
        RetainedRmsRows::new(
            self.span.clone(),
            self.layout,
            self.norm.clone(),
            self.epsilon(),
            SourceNumericalContract::GgmlSourceF32V1,
        )
    }
}
impl GdnUpstreamBinding {
    pub fn input_operator(&self) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            call(self.span())?,
            SourceOperatorKind::RetainedRows {
                width: 2048,
                origin: SourceRetainedOrigin::Normalized {
                    epsilon_bits: self.prefix.epsilon().to_bits(),
                },
            },
            SourceNumericalContract::GgmlSourceF32V1,
            vec![],
            vec![self.prefix.norm().clone()],
        )
    }
    pub fn projection_operator(&self, projection: GdnProjection) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            call(self.span())?,
            SourceOperatorKind::Projection { selected_row: None },
            SourceNumericalContract::GgmlQ4Repack8x8F32V1,
            vec![self.tensor(projection).clone()],
            vec![],
        )
    }
    pub fn convolution_operator(&self) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            call(self.span())?,
            SourceOperatorKind::CausalConvSiluZero {
                channels: 6144,
                kernel: 4,
            },
            SourceNumericalContract::GgmlSourceF32V1,
            vec![],
            vec![self.convolution.clone()],
        )
    }
    pub fn qk_operator(&self) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            call(self.span())?,
            SourceOperatorKind::GroupedQkNormalize(SourceGroupedQkGeometry {
                heads: 16,
                width: 128,
                epsilon: self.prefix.epsilon() / 128.0,
                scale: 1.0 / 128.0_f32.sqrt(),
            }),
            SourceNumericalContract::GgmlSourceF32V1,
            vec![],
            vec![],
        )
    }
}
impl GdnOutputBinding {
    pub fn input_operator(&self) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            call(self.span())?,
            SourceOperatorKind::RetainedRows {
                width: 2048,
                origin: SourceRetainedOrigin::Gated,
            },
            SourceNumericalContract::OpenBlas033Zen16Q8F32V1,
            vec![],
            vec![],
        )
    }
    pub fn projection_operator(&self) -> Result<SourceOperator> {
        self.validate()?;
        SourceOperator::new(
            call(self.span())?,
            SourceOperatorKind::Projection { selected_row: None },
            SourceNumericalContract::OpenBlas033Zen16Q8F32V1,
            vec![self.tensor.clone()],
            vec![],
        )
    }
}
pub(crate) fn one<M: MathematicalBackend>(
    math: &mut M,
    descriptor: Result<SourceOperator>,
    inputs: &[M::Value],
) -> Result<M::Value> {
    let [value] = math
        .source_operator(&descriptor?, inputs)?
        .try_into()
        .map_err(|_| Error::State("one source F32 producer".into()))?;
    Ok(value)
}
#[cfg(test)]
#[path = "source_bindings/tests.rs"]
mod tests;
