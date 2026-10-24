//! Same immutable source/SSA geometry; current input/attempt authority is separate.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind;
use zerogravity_core::accelerator::program::RecordedVllm08Model;
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

pub(crate) fn same_recipe(
    source: &OriginalWindowModelSource,
    cold: &RecordedVllm08Model,
    actual: &RecordedVllm08Model,
) -> Result<()> {
    native_full::FullModelProgram::require_record(source, cold)?;
    native_full::FullModelProgram::require_record(source, actual)?;
    let a = Canonical47CallKind::of(cold.call().request())?;
    let b = Canonical47CallKind::of(actual.call().request())?;
    let geometry = matches!(
        (a, b),
        (Canonical47CallKind::Prefill, Canonical47CallKind::Prefill)
            | (
                Canonical47CallKind::Decode { .. },
                Canonical47CallKind::Decode { .. }
            )
    );
    let (ap, bp) = (cold.plan(), actual.plan());
    if !geometry
        || cold.call().source() != actual.call().source()
        || cold.call().model_profile() != actual.call().model_profile()
        || ap.numerical_profile() != bp.numerical_profile()
        || ap.values() != bp.values()
        || ap.precisions() != bp.precisions()
        || ap.required_domains() != bp.required_domains()
        || ap.proposed_domains() != bp.proposed_domains()
        || cold.normalized() != actual.normalized()
        || cold.logits() != actual.logits()
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
        match (&a.operation, &b.operation) {
            (
                PlanOperation::TypedVllmBf16Embedding { descriptor: a },
                PlanOperation::TypedVllmBf16Embedding { descriptor: b },
            ) if index == 0 => {
                a.validate()?;
                b.validate()?;
                if a.tensor() != b.tensor() {
                    return Err(Error::State(
                        "canonical embedding source tensor differs".into(),
                    ));
                }
            }
            (
                PlanOperation::TypedVllm08 {
                    descriptor: a,
                    state: sa,
                },
                PlanOperation::TypedVllm08 {
                    descriptor: b,
                    state: sb,
                },
            ) => {
                a.validate()?;
                b.validate()?;
                if a.call() != cold.call()
                    || b.call() != actual.call()
                    || a.kind() != b.kind()
                    || a.layer() != b.layer()
                    || a.matrices() != b.matrices()
                    || a.parameters() != b.parameters()
                    || a.selected_row() != b.selected_row()
                    || sa != sb
                {
                    return Err(Error::State(
                        "canonical rebind changes exact source operator/site".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::State(
                    "canonical source rebind introduces another operation".into(),
                ));
            }
        }
    }
    Ok(())
}
