//! Closed original Q8_0 ssm_out metadata; no model/state/backend permit.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZenSsmOutContract {
    OpenBlas033Zen16SsmOutV1,
}

/// Exact source matrix and ONE retained gated producer's teacher issuer.
/// Metadata alone cannot supply a runtime activation or its physical owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZenSsmOutDescriptor {
    span: TeacherSpan,
    tensor: SourceTensor,
    contract: ZenSsmOutContract,
    layout: BulkF32Layout,
}
impl ZenSsmOutDescriptor {
    pub fn new(
        span: TeacherSpan,
        source: &dyn TensorSource,
        name: &str,
        contract: ZenSsmOutContract,
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
    pub fn contract(&self) -> ZenSsmOutContract {
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

fn check_builder(builder: &PlanBuilder, descriptor: &ZenSsmOutDescriptor) -> Result<()> {
    descriptor.validate()?;
    if !builder.operator_slice
        || builder.profile != ProgramNumericalProfile::GgufSourceV1
        || builder.source != descriptor.source()
    {
        return Err(Error::Unsupported(
            "Zen ssm_out requires explicit state-free GGUF operator slice".into(),
        ));
    }
    Ok(())
}
pub(super) fn input_precision(
    builder: &PlanBuilder,
    descriptor: &ZenSsmOutDescriptor,
) -> Result<ActivationPrecision> {
    check_builder(builder, descriptor)?;
    Ok(ActivationPrecision::F32)
}
pub(super) fn projection_precision(
    builder: &PlanBuilder,
    descriptor: &ZenSsmOutDescriptor,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check_builder(builder, descriptor)?;
    let [input] = inputs else {
        return Err(Error::Malformed("Zen ssm_out input arity".into()));
    };
    let producer = builder
        .nodes
        .iter()
        .find(|node| node.outputs.contains(&input.id));
    if input.precision != ActivationPrecision::F32
        || input.elements != descriptor.layout.backing_words()
        || !matches!(producer.map(|node| &node.operation),
            Some(PlanOperation::TypedZenSsmOutInput { descriptor: original })
                if original.as_ref() == descriptor)
    {
        return Err(Error::State(
            "Zen ssm_out exact retained-gated producer/layout/issuer".into(),
        ));
    }
    Ok(ActivationPrecision::F32)
}
pub(super) fn record_input(
    builder: &mut PlanBuilder,
    descriptor: &ZenSsmOutDescriptor,
) -> Result<PlanValue> {
    input_precision(builder, descriptor)?;
    builder.one(
        "retained-bulk-gdn-gated",
        PlanOperation::TypedZenSsmOutInput {
            descriptor: Box::new(descriptor.clone()),
        },
        &[],
        ValueShape::Vector(descriptor.layout.backing_words()),
        None,
    )
}
pub(super) fn record_projection(
    builder: &mut PlanBuilder,
    descriptor: &ZenSsmOutDescriptor,
    input: &PlanValue,
) -> Result<PlanValue> {
    projection_precision(builder, descriptor, &[*input])?;
    builder.one(
        "closed-zen-q8-ssm-out",
        PlanOperation::TypedZenSsmOutProjection {
            descriptor: Box::new(descriptor.clone()),
        },
        &[*input],
        ValueShape::Vector(descriptor.layout.backing_words()),
        None,
    )
}

#[cfg(test)]
mod tests;
