//! Explicit analytic sparse contract; no trained/native eligibility.
use super::*;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextExpertTensor {
    pub source: SourceIdentity,
    pub name: String,
    pub kind: QuantKind,
    pub experts: usize,
    pub rows: usize,
    pub columns: usize,
}
#[derive(Clone, Debug)]
pub struct TextMoeContract {
    pub router: SourceTensor,
    pub gate_up: TextExpertTensor,
    pub down: TextExpertTensor,
    pub selected: usize,
    pub epsilon: f32,
}
pub struct TextMoeInputs<'a, V> {
    pub normalized: &'a V,
    pub router_input: &'a V,
    pub router_scale: SourceVector<'a>,
    pub expert_scale: SourceVector<'a>,
}
impl TextMoeContract {
    pub fn validate(
        &self,
        source: SourceIdentity,
        hidden: usize,
        router: SourceVector<'_>,
        expert: SourceVector<'_>,
    ) -> Result<()> {
        let e = self.gate_up.experts;
        let width = self.down.columns;
        if !source.is_bound()
            || hidden == 0
            || hidden > 16384
            || e == 0
            || e > 4096
            || width == 0
            || width > 16384
            || self.selected == 0
            || self.selected > e
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || self.router.source != source
            || self.router.columns != hidden
            || self.router.rows != e
            || self.router.axes != MatrixAxes::OutputInput
            || self.gate_up.source != source
            || self.down.source != source
            || self.down.experts != e
            || self.gate_up.rows != 2 * width
            || self.gate_up.columns != hidden
            || self.down.rows != hidden
            || router.source != source
            || router.values.len() != hidden
            || expert.source != source
            || expert.values.len() != e
            || router
                .values
                .iter()
                .chain(expert.values)
                .any(|v| !v.is_finite())
            || [&self.router.name, &self.gate_up.name, &self.down.name]
                .iter()
                .any(|n| n.is_empty())
            || [self.router.kind, self.gate_up.kind, self.down.kind]
                .iter()
                .any(|k| !matches!(k, QuantKind::Bf16 | QuantKind::F32))
        {
            return Err(Error::Malformed(
                "analytic MoE immutable source/geometry/parameters".into(),
            ));
        }
        Ok(())
    }
}
pub(super) fn record(
    builder: &mut PlanBuilder,
    name: &str,
    c: &TextMoeContract,
    input: TextMoeInputs<'_, PlanValue>,
) -> Result<PlanValue> {
    if builder.profile != ProgramNumericalProfile::AnalyticF32TextV1
        || input.normalized.precision != ActivationPrecision::F32
        || input.router_input.precision != ActivationPrecision::F32
        || input.normalized.shape != input.router_input.shape
    {
        return Err(Error::Unsupported(
            "analytic MoE requires explicit F32 profile/full inputs".into(),
        ));
    }
    c.validate(
        builder.source,
        input.normalized.elements,
        input.router_scale,
        input.expert_scale,
    )?;
    let router_scale = builder.constant(input.router_scale)?;
    let expert_scale = builder.constant(input.expert_scale)?;
    builder.one(
        name,
        PlanOperation::TextMoe {
            contract: Box::new(c.clone()),
            router_scale,
            expert_scale,
        },
        &[*input.normalized, *input.router_input],
        input.normalized.shape,
        None,
    )
}
