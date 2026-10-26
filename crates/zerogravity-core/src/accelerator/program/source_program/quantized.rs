//! Exact F32 source arithmetic; weight codec never chooses activation precision.
use super::*;

pub(super) fn validate(op: &SourceOperator) -> Result<()> {
    use SourceNumericalContract::*;
    use SourceOperatorKind::*;
    let bad = || Error::Unsupported("source F32 kind/codec/layout/numerical contract".into());
    if op.matrices().len() > 1
        || op.parameters().len() > 1
        || op.call().mode() != SourceCallMode::Prefill
    {
        return Err(bad());
    }
    for m in op.matrices() {
        if m.source != op.call().source()
            || m.name.is_empty()
            || m.axes != MatrixAxes::InputOutput
            || m.columns == 0
            || m.rows == 0
        {
            return Err(bad());
        }
        m.packed_bytes()?;
    }
    for p in op.parameters() {
        if p.source != op.call().source() || p.name.is_empty() {
            return Err(bad());
        }
    }
    let arity = match (op.numerical_contract(), op.kind()) {
        (GgmlSourceF32V1, EmbeddingRows) => {
            let m = op.matrices().first().ok_or_else(bad)?;
            if m.kind != QuantKind::Q6K || m.rows != op.call().rows().vocabulary() {
                return Err(bad());
            }
            (1, 0)
        }
        (GgmlSourceF32V1, Rms(g)) => {
            if g.gamma != SourceGamma::Direct || !g.epsilon.is_finite() || g.epsilon <= 0.0 {
                return Err(bad());
            }
            op.parameter(0, &[QuantKind::F32], &[g.width])?;
            (0, 1)
        }
        (
            GgmlSourceF32V1,
            RetainedRows {
                width,
                origin: SourceRetainedOrigin::Normalized { epsilon_bits },
            },
        ) => {
            let epsilon = f32::from_bits(epsilon_bits);
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(bad());
            }
            op.parameter(0, &[QuantKind::F32], &[width])?;
            (0, 1)
        }
        (
            OpenBlas033Zen16Q8F32V1,
            RetainedRows {
                width,
                origin: SourceRetainedOrigin::Gated,
            },
        ) => {
            if width != 2048 || op.call().rows().len() != 64 {
                return Err(bad());
            }
            (0, 0)
        }
        (GgmlQ4Repack8x8F32V1 | OpenBlas033Zen16Q8F32V1, Projection { selected_row: None }) => {
            let m = op.matrices().first().ok_or_else(bad)?;
            if op.numerical_contract() == GgmlQ4Repack8x8F32V1 {
                if m.kind != QuantKind::Q4_0
                    || !m.rows.is_multiple_of(8)
                    || !m.columns.is_multiple_of(32)
                {
                    return Err(bad());
                }
            } else if m.kind != QuantKind::Q8_0
                || m.columns != 2048
                || m.rows != 2048
                || op.call().rows().len() != 64
            {
                return Err(bad());
            }
            (1, 0)
        }
        (GgmlSourceF32V1, CausalConvSiluZero { channels, kernel }) => {
            if kernel != 4 {
                return Err(bad());
            }
            op.parameter(0, &[QuantKind::F32], &[kernel, channels])?;
            (0, 1)
        }
        (GgmlSourceF32V1, GroupedQkNormalize(g)) => {
            SourceOperator::product(g.heads, g.width)?;
            if !g.epsilon.is_finite()
                || g.epsilon <= 0.0
                || !g.scale.is_finite()
                || g.scale.to_bits() != (1.0 / (g.width as f32).sqrt()).to_bits()
            {
                return Err(bad());
            }
            (0, 0)
        }
        _ => return Err(bad()),
    };
    if (op.matrices().len(), op.parameters().len()) != arity {
        return Err(bad());
    }
    op.input_elements()?;
    op.output_elements()?;
    Ok(())
}

pub(super) fn precision(
    builder: &PlanBuilder,
    op: &SourceOperator,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    use SourceNumericalContract::*;
    use SourceOperatorKind::*;
    if !builder.operator_slice
        || builder.profile != ProgramNumericalProfile::GgufSourceV1
        || builder.source != op.call().source()
        || inputs.len() != op.input_elements()?.len()
    {
        return Err(Error::Unsupported(
            "source F32 state-free slice/source/arity contract".into(),
        ));
    }
    for (input, elements) in inputs.iter().zip(op.input_elements()?) {
        builder.validate_value(input)?;
        let tag = builder
            .source_value_provenance(input)?
            .ok_or_else(|| Error::State("source F32 producer absent".into()))?;
        if input.precision != ActivationPrecision::F32
            || input.shape != ValueShape::Vector(elements)
            || !input.is_contiguous()
            || tag.call() != op.call()
        {
            return Err(Error::State(
                "source F32 dense original producer/call".into(),
            ));
        }
        let PlanOperation::SourceOperator {
            descriptor: prior, ..
        } = &builder.nodes[tag.node_index()].operation
        else {
            return Err(Error::State("source F32 producer operation".into()));
        };
        let allowed = match (
            op.numerical_contract(),
            op.kind(),
            prior.numerical_contract(),
            prior.kind(),
        ) {
            (GgmlSourceF32V1, Rms(g), GgmlSourceF32V1, EmbeddingRows) => {
                prior.matrices()[0].columns == g.width
            }
            (
                GgmlQ4Repack8x8F32V1,
                Projection { .. },
                GgmlSourceF32V1,
                RetainedRows {
                    origin: SourceRetainedOrigin::Normalized { .. },
                    ..
                },
            ) => true,
            (
                OpenBlas033Zen16Q8F32V1,
                Projection { .. },
                OpenBlas033Zen16Q8F32V1,
                RetainedRows {
                    origin: SourceRetainedOrigin::Gated,
                    ..
                },
            ) => true,
            (
                GgmlSourceF32V1,
                CausalConvSiluZero { channels, .. },
                GgmlQ4Repack8x8F32V1,
                Projection { .. },
            ) => prior.matrices()[0].rows == channels,
            (
                GgmlSourceF32V1,
                GroupedQkNormalize(g),
                GgmlSourceF32V1,
                CausalConvSiluZero { channels, .. },
            ) => channels == 3 * g.heads * g.width,
            _ => false,
        };
        if !allowed {
            return Err(Error::State(
                "source F32 exact mathematical predecessor".into(),
            ));
        }
    }
    Ok(ActivationPrecision::F32)
}

#[cfg(test)]
mod tests;
