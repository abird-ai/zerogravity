//! Original Q4_0 producers and source convolution/QK; no state publication.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufGdnUpstreamContract {
    GgufQ4Repacked2BFirst64V1,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufGdnProjection {
    Qkv,
    Z,
}

pub struct GgufGdnUpstreamValues<V> {
    pub qkv: V,
    pub z: V,
    pub convolution: V,
    pub operands: V,
}

/// Metadata binds ONE retained normalized issuer; runtime backing is separate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufGdnUpstreamDescriptor {
    prefix: GgufPrefixDescriptor,
    qkv: SourceTensor,
    z: SourceTensor,
    convolution: SourceParameter,
    contract: GgufGdnUpstreamContract,
}
impl GgufGdnUpstreamDescriptor {
    pub fn new(
        prefix: GgufPrefixDescriptor,
        source: &dyn TensorSource,
        contract: GgufGdnUpstreamContract,
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
    pub fn prefix(&self) -> &GgufPrefixDescriptor {
        &self.prefix
    }
    pub fn span(&self) -> &TeacherSpan {
        self.prefix.span()
    }
    pub fn source(&self) -> SourceIdentity {
        self.prefix.source()
    }
    pub fn contract(&self) -> GgufGdnUpstreamContract {
        self.contract
    }
    pub fn tensor(&self, projection: GgufGdnProjection) -> &SourceTensor {
        match projection {
            GgufGdnProjection::Qkv => &self.qkv,
            GgufGdnProjection::Z => &self.z,
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
fn check(builder: &PlanBuilder, descriptor: &GgufGdnUpstreamDescriptor) -> Result<()> {
    descriptor.validate()?;
    if !builder.operator_slice
        || builder.profile != ProgramNumericalProfile::GgufSourceV1
        || builder.source != descriptor.source()
    {
        return Err(Error::Unsupported(
            "GGUF upstream requires state-free original GGUF slice".into(),
        ));
    }
    Ok(())
}
pub(super) fn input_precision(
    builder: &PlanBuilder,
    descriptor: &GgufGdnUpstreamDescriptor,
) -> Result<ActivationPrecision> {
    check(builder, descriptor)?;
    Ok(ActivationPrecision::F32)
}
fn producer<'a>(
    builder: &'a PlanBuilder,
    inputs: &[PlanValue],
    elements: usize,
) -> Result<&'a PlanOperation> {
    let [input] = inputs else {
        return Err(Error::Malformed("GGUF upstream arity".into()));
    };
    if input.precision != ActivationPrecision::F32 || input.elements != elements {
        return Err(Error::State("GGUF upstream F32 input geometry".into()));
    }
    builder
        .nodes
        .iter()
        .find(|node| node.outputs.contains(&input.id))
        .map(|node| &node.operation)
        .ok_or_else(|| Error::State("GGUF upstream producer missing".into()))
}
pub(super) fn projection_precision(
    builder: &PlanBuilder,
    descriptor: &GgufGdnUpstreamDescriptor,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check(builder, descriptor)?;
    if !matches!(producer(builder, inputs, 2048 * 64)?, PlanOperation::TypedGgufGdnInput { descriptor: original } if original.as_ref() == descriptor)
    {
        return Err(Error::State(
            "GGUF upstream exact normalized producer/issuer".into(),
        ));
    }
    Ok(ActivationPrecision::F32)
}
pub(super) fn convolution_precision(
    builder: &PlanBuilder,
    descriptor: &GgufGdnUpstreamDescriptor,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check(builder, descriptor)?;
    if !matches!(producer(builder, inputs, 6144 * 64)?, PlanOperation::TypedGgufGdnProjection { descriptor: original, projection: GgufGdnProjection::Qkv } if original.as_ref() == descriptor)
    {
        return Err(Error::State(
            "GGUF upstream exact QKV producer/issuer".into(),
        ));
    }
    Ok(ActivationPrecision::F32)
}
pub(super) fn qk_precision(
    builder: &PlanBuilder,
    descriptor: &GgufGdnUpstreamDescriptor,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check(builder, descriptor)?;
    if !matches!(producer(builder, inputs, 6144 * 64)?, PlanOperation::TypedGgufGdnConvSilu { descriptor: original } if original.as_ref() == descriptor)
    {
        return Err(Error::State(
            "GGUF upstream exact convolution producer/issuer".into(),
        ));
    }
    Ok(ActivationPrecision::F32)
}
pub(super) fn record_input(
    builder: &mut PlanBuilder,
    descriptor: &GgufGdnUpstreamDescriptor,
) -> Result<PlanValue> {
    input_precision(builder, descriptor)?;
    builder.one(
        "retained-gguf-normalized",
        PlanOperation::TypedGgufGdnInput {
            descriptor: Box::new(descriptor.clone()),
        },
        &[],
        ValueShape::Vector(2048 * 64),
        None,
    )
}
pub(super) fn record_projection(
    builder: &mut PlanBuilder,
    descriptor: &GgufGdnUpstreamDescriptor,
    projection: GgufGdnProjection,
    input: &PlanValue,
) -> Result<PlanValue> {
    projection_precision(builder, descriptor, &[*input])?;
    builder.one(
        "source-q4-8x8-projection",
        PlanOperation::TypedGgufGdnProjection {
            descriptor: Box::new(descriptor.clone()),
            projection,
        },
        &[*input],
        ValueShape::Vector(descriptor.tensor(projection).rows * 64),
        None,
    )
}
pub(super) fn record_convolution(
    builder: &mut PlanBuilder,
    descriptor: &GgufGdnUpstreamDescriptor,
    input: &PlanValue,
) -> Result<PlanValue> {
    convolution_precision(builder, descriptor, &[*input])?;
    builder.one(
        "source-causal-convolution-silu",
        PlanOperation::TypedGgufGdnConvSilu {
            descriptor: Box::new(descriptor.clone()),
        },
        &[*input],
        ValueShape::Vector(6144 * 64),
        None,
    )
}
pub(super) fn record_qk(
    builder: &mut PlanBuilder,
    descriptor: &GgufGdnUpstreamDescriptor,
    input: &PlanValue,
) -> Result<PlanValue> {
    qk_precision(builder, descriptor, &[*input])?;
    builder.one(
        "source-grouped-qk-and-v",
        PlanOperation::TypedGgufGdnQk {
            descriptor: Box::new(descriptor.clone()),
        },
        &[*input],
        ValueShape::Vector(6144 * 64),
        None,
    )
}

#[cfg(test)]
mod tests;
