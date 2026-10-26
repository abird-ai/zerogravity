//! Same immutable source/SSA geometry; current input/attempt authority is separate.
use super::source_bindings::SourceBindings;
use super::*;
use crate::cuda::native::fixed_prompt_model::FixedPromptCallKind;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;

pub(crate) fn same_recipe(
    source: &VerifiedTensorSource,
    cold: &SourceBindings,
    actual: &SourceBindings,
) -> Result<()> {
    native_full::FullModelProgram::require_record(cold)?;
    native_full::FullModelProgram::require_record(actual)?;
    let a = FixedPromptCallKind::of(cold.call())?;
    let b = FixedPromptCallKind::of(actual.call())?;
    let geometry = matches!(
        (a, b),
        (FixedPromptCallKind::Prefill, FixedPromptCallKind::Prefill)
            | (
                FixedPromptCallKind::Decode { .. },
                FixedPromptCallKind::Decode { .. }
            )
    );
    let (ap, bp) = (cold.plan(), actual.plan());
    if !geometry
        || cold.call().source() != actual.call().source()
        || !source.same_issuer(cold.source())
        || !source.same_issuer(actual.source())
        || ap.numerical_profile() != bp.numerical_profile()
        || ap.values() != bp.values()
        || ap.nodes().len() != bp.nodes().len()
        || ap.precisions() != bp.precisions()
        || ap.required_domains() != bp.required_domains()
        || ap.proposed_domains() != bp.proposed_domains()
        || cold.selection().normalized() != actual.selection().normalized()
        || cold.selection().output() != actual.selection().output()
    {
        return Err(Error::State(
            "canonical source/physical geometry differs from cold storage".into(),
        ));
    }
    for (index, (a, b)) in ap.nodes().iter().zip(bp.nodes()).enumerate() {
        if a.inputs != b.inputs || a.outputs != b.outputs || a.state != b.state {
            return Err(Error::State(
                "canonical cold rebind changes actual SSA/state effects".into(),
            ));
        }
        cold.operator(index)?;
        actual.operator(index)?;
        match (&a.operation, &b.operation) {
            (
                PlanOperation::SourceOperator {
                    descriptor: left,
                    state: ls,
                },
                PlanOperation::SourceOperator {
                    descriptor: right,
                    state: rs,
                },
            ) => {
                if left.kind() != right.kind()
                    || left.numerical_contract() != right.numerical_contract()
                    || left.matrices() != right.matrices()
                    || left.parameters() != right.parameters()
                    || ls != rs
                {
                    return Err(Error::State(
                        "rebind changes immutable mathematical source recipe".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::State(
                    "rebind introduces an unrelated operation".into(),
                ));
            }
        }
    }
    Ok(())
}
