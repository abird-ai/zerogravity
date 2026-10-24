//! Closed original GGUF prefix metadata; no session or forward permit.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufPrefixContract {
    GgufSource2BFirst64V1,
}

/// Metadata alone cannot supply the actual admitted prefix backing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufPrefixDescriptor {
    span: TeacherSpan,
    embedding: SourceTensor,
    norm: SourceParameter,
    contract: GgufPrefixContract,
    layout: BulkF32Layout,
    epsilon_bits: u32,
}
impl GgufPrefixDescriptor {
    pub fn new(
        span: TeacherSpan,
        source: &dyn TensorSource,
        contract: GgufPrefixContract,
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
    pub fn contract(&self) -> GgufPrefixContract {
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
fn check_builder(builder: &PlanBuilder, descriptor: &GgufPrefixDescriptor) -> Result<()> {
    descriptor.validate()?;
    if !builder.operator_slice
        || builder.profile != ProgramNumericalProfile::GgufSourceV1
        || builder.source != descriptor.source()
    {
        return Err(Error::Unsupported(
            "GGUF prefix requires state-free GGUF operator slice".into(),
        ));
    }
    Ok(())
}
pub(super) fn embedding_precision(
    builder: &PlanBuilder,
    descriptor: &GgufPrefixDescriptor,
) -> Result<ActivationPrecision> {
    check_builder(builder, descriptor)?;
    Ok(ActivationPrecision::F32)
}
pub(super) fn rms_precision(
    builder: &PlanBuilder,
    descriptor: &GgufPrefixDescriptor,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check_builder(builder, descriptor)?;
    let [input] = inputs else {
        return Err(Error::Malformed("GGUF prefix RMS arity".into()));
    };
    let producer = builder
        .nodes
        .iter()
        .find(|node| node.outputs.contains(&input.id));
    if input.precision != ActivationPrecision::F32
        || input.elements != descriptor.layout.backing_words()
        || !matches!(producer.map(|node| &node.operation), Some(PlanOperation::TypedGgufPrefixEmbedding { descriptor: original }) if original.as_ref() == descriptor)
    {
        return Err(Error::State(
            "GGUF prefix exact embedding producer/issuer".into(),
        ));
    }
    Ok(ActivationPrecision::F32)
}
pub(super) fn record_embedding(
    builder: &mut PlanBuilder,
    descriptor: &GgufPrefixDescriptor,
) -> Result<PlanValue> {
    embedding_precision(builder, descriptor)?;
    builder.one(
        "gguf-prefix-embedding",
        PlanOperation::TypedGgufPrefixEmbedding {
            descriptor: Box::new(descriptor.clone()),
        },
        &[],
        ValueShape::Vector(descriptor.layout.backing_words()),
        None,
    )
}
pub(super) fn record_rms(
    builder: &mut PlanBuilder,
    descriptor: &GgufPrefixDescriptor,
    input: &PlanValue,
) -> Result<PlanValue> {
    rms_precision(builder, descriptor, &[*input])?;
    builder.one(
        "gguf-prefix-inputnorm",
        PlanOperation::TypedGgufPrefixRms {
            descriptor: Box::new(descriptor.clone()),
        },
        &[*input],
        ValueShape::Vector(descriptor.layout.backing_words()),
        None,
    )
}

#[cfg(test)]
mod tests;
