//! Family-neutral text operations. BF16 scalar is CPU diagnostic only; no native eligibility.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextElementwise {
    Scale(f32),
    /// Source buffer scalar rounded to BF16 before multiplication; CPU diagnostic only.
    ScaleBf16Scalar(f32),
    GeluTanh,
    Multiply,
    Add,
    TanhSoftcap(f32),
}
impl TextElementwise {
    pub fn validate(self, arity: usize) -> Result<()> {
        let expected = if matches!(self, Self::Multiply | Self::Add) {
            2
        } else {
            1
        };
        let scalar = match self {
            Self::Scale(x) | Self::ScaleBf16Scalar(x) => x.is_finite(),
            Self::TanhSoftcap(x) => x.is_finite() && x > 0.0,
            _ => true,
        };
        if arity != expected || !scalar {
            return Err(Error::Malformed(
                "analytic text elementwise arity/scalar".into(),
            ));
        }
        Ok(())
    }
}

/// Split-half rotary; proportional mode leaves zero-frequency pairs unchanged.
/// Frequencies use the FULL head width as denominator, including proportional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextRotary {
    pub head_width: usize,
    pub rotated_pairs: usize,
    pub theta: f32,
}
impl TextRotary {
    pub fn validate(self, elements: usize) -> Result<()> {
        if self.head_width == 0
            || !self.head_width.is_multiple_of(2)
            || self.rotated_pairs == 0
            || self.rotated_pairs > self.head_width / 2
            || elements == 0
            || !elements.is_multiple_of(self.head_width)
            || !self.theta.is_finite()
            || self.theta <= 1.0
        {
            return Err(Error::Malformed(
                "analytic split-half rotary geometry".into(),
            ));
        }
        Ok(())
    }
}

fn reference(builder: &PlanBuilder, inputs: &[PlanValue]) -> Result<()> {
    if builder.profile == ProgramNumericalProfile::Gemma4E2bBf16TextCandidateV1 {
        if inputs
            .iter()
            .all(|v| v.precision == ActivationPrecision::Bf16WidenedF32)
        {
            return Ok(());
        }
        return Err(Error::State("BF16 text candidate arithmetic input".into()));
    }
    if builder.profile != ProgramNumericalProfile::AnalyticF32TextV1
        || inputs
            .iter()
            .any(|value| value.precision != ActivationPrecision::F32)
    {
        return Err(Error::Unsupported(
            "analytic text requires its explicit F32 profile".into(),
        ));
    }
    Ok(())
}
pub(super) fn record_elementwise(
    builder: &mut PlanBuilder,
    name: &str,
    operation: TextElementwise,
    inputs: &[&PlanValue],
) -> Result<PlanValue> {
    operation.validate(inputs.len())?;
    if matches!(operation, TextElementwise::ScaleBf16Scalar(_))
        && builder.profile != ProgramNumericalProfile::Gemma4E2bBf16TextCandidateV1
    {
        return Err(Error::Unsupported(
            "BF16 text candidate has no qualified recorded/native profile".into(),
        ));
    }
    let values = inputs.iter().map(|value| **value).collect::<Vec<_>>();
    reference(builder, &values)?;
    let shape = values[0].shape;
    if values.iter().any(|value| value.shape != shape) {
        return Err(Error::Malformed(
            "analytic elementwise exact input shapes".into(),
        ));
    }
    builder.one(
        name,
        PlanOperation::TextElementwise(operation),
        &values,
        shape,
        None,
    )
}
pub(super) fn record_rotary(
    builder: &mut PlanBuilder,
    name: &str,
    input: &PlanValue,
    position: ParameterRef,
    geometry: TextRotary,
) -> Result<PlanValue> {
    reference(builder, &[*input])?;
    geometry.validate(input.elements)?;
    if position.0 != ParameterSlot::Position {
        return Err(Error::Malformed(
            "analytic rotary position parameter".into(),
        ));
    }
    builder.one(
        name,
        PlanOperation::TextRotary { geometry, position },
        &[*input],
        input.shape,
        None,
    )
}
pub(super) fn record_qk(
    builder: &mut PlanBuilder,
    name: &str,
    domain: KvDomain,
    query: &PlanValue,
    valid_tokens: ParameterRef,
    geometry: AttentionGeometry,
    window: Option<usize>,
) -> Result<QkValues<PlanValue>> {
    reference(builder, &[*query])?;
    let (queries, _) = geometry.validate()?;
    let state = builder.current(StateSlot::Kv(domain))?;
    if query.elements != queries
        || state.ordinal != 1
        || valid_tokens.0 != ParameterSlot::ValidTokens
        || window == Some(0)
    {
        return Err(Error::Malformed(
            "analytic QK appended KV/query/window".into(),
        ));
    }
    let shape = ValueShape::CausalRows {
        heads: geometry.query_heads,
        capacity: builder.prefix_capacity,
    };
    let values = builder.record(
        name,
        PlanOperation::TextWindowQk {
            state,
            geometry,
            valid_tokens,
            window,
        },
        &[*query],
        &[shape, shape],
        None,
    )?;
    Ok(QkValues {
        dots: values[0],
        scores: values[1],
    })
}
