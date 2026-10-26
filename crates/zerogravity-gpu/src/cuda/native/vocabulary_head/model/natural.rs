//! Fresh exact Head producer binding over the existing single native record.
use super::*;
use crate::resident_program::natural_binding::{same_lease, same_recipe};
impl ModelHeadPrograms {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        phase: usize,
        values: Arc<ModelValues>,
        fused: Arc<FusedRmsPrepared>,
    ) -> Result<Arc<Self>> {
        let old = self
            .bindings
            .get(phase)
            .ok_or_else(|| Error::State("natural head original slot absent".into()))?;
        same_recipe(old.values.record(), values.record())?;
        same_lease(
            &old.output,
            values.lease(values.record().selection().output())?,
        )?;
        same_lease(old.input.lease(), fused.normalized())?;
        let record = values.record();
        let normalized_id = record
            .selection()
            .normalized()
            .ok_or_else(|| Error::State("natural normalized selection absent".into()))?;
        let head = &record.plan().nodes()[old.node];
        let normalized = record
            .plan()
            .source_value_provenance(normalized_id)?
            .ok_or_else(|| Error::State("natural final norm provenance absent".into()))?;
        if normalized.call() != record.call() || normalized.output_index() != 0 {
            return Err(Error::State(
                "natural final norm actual model origin".into(),
            ));
        }
        let Some(final_op) = record.operator(normalized.node_index())? else {
            return Err(Error::State("natural actual final norm descriptor".into()));
        };
        fused.require_full_model_operator(final_op)?;
        if head.inputs != [normalized_id] || head.outputs != [record.selection().output()] {
            return Err(Error::State("natural same Head SSA".into()));
        }
        let metadata = old.weight.source().reserve_workspace(64 * 1024)?;
        let input = HeadInputView::computed(old.weight.source().clone(), old.case, fused)?;
        let mut bindings = self.bindings.clone();
        bindings[phase] = BoundHead {
            input,
            values,
            weight: old.weight.clone(),
            output: old.output.clone(),
            node: old.node,
            case: old.case,
        };
        Ok(Arc::new(Self {
            programs: self.programs.clone(),
            bindings,
            _metadata: metadata,
        }))
    }
}
