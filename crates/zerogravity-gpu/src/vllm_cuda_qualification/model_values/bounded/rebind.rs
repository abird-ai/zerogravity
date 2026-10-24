//! Reuse only immutable allocation/lifetime analysis, never the actual record.
use super::*;

fn same_node(index: usize, old: &PlanNode, new: &PlanNode) -> Result<()> {
    if old.inputs != new.inputs || old.outputs != new.outputs || old.state != new.state {
        return Err(Error::State(
            "bounded fresh record changes SSA/state effects".into(),
        ));
    }
    let same = match (&old.operation, &new.operation) {
        (
            PlanOperation::TypedVllmBf16Embedding { descriptor: a },
            PlanOperation::TypedVllmBf16Embedding { descriptor: b },
        ) if index == 0 => a.tensor() == b.tensor(),
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
            a.kind() == b.kind()
                && a.layer() == b.layer()
                && a.matrices() == b.matrices()
                && a.parameters() == b.parameters()
                && a.input_elements() == b.input_elements()
                && a.output_elements() == b.output_elements()
                && a.selected_row() == b.selected_row()
                && sa == sb
        }
        _ => false,
    };
    if !same {
        return Err(Error::State(
            "bounded fresh record changes source operator/geometry".into(),
        ));
    }
    Ok(())
}
impl StorageRecipe {
    pub(crate) fn bind_call(
        scope: Arc<ShortHfModelBinding>,
        mode: StorageMode,
        limits: ValueStorageLimits,
        template: &ModelValues,
    ) -> Result<Arc<Self>> {
        let old = &template
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("bounded cold recipe absent".into()))?
            .recipe;
        require_new_record(scope.record(), old.scope.record())?;
        let a = old.scope.record();
        let b = scope.record();
        let (ap, bp) = (a.plan(), b.plan());
        if !Arc::ptr_eq(scope.source().source(), old.scope.source().source())
            || scope.call().model_profile() != old.scope.call().model_profile()
            || scope.call().span().len() != old.scope.call().span().len()
            || mode != old.mode
            || limits.max_nodes != 255
            || limits.max_values != 327
            || limits.reuse_policy != ValueStorageReusePolicy::ExactAlignedSize
            || limits.alignment != old.alignment
            || limits.max_slots > 327
            || old.slot_bytes.len() > limits.max_slots
            || old.total_bytes > limits.max_total_bytes
            || old.slot_bytes.iter().any(|&b| b > limits.max_slot_bytes)
            || ap.nodes().len() != 255
            || bp.nodes().len() != 255
            || ap.values() != bp.values()
            || ap.precisions() != bp.precisions()
            || ap.numerical_profile() != bp.numerical_profile()
            || ap.required_domains() != bp.required_domains()
            || ap.proposed_domains() != bp.proposed_domains()
            || a.logits() != b.logits()
            || a.normalized() != b.normalized()
        {
            return Err(Error::State(
                "bounded fresh source/physical recipe differs".into(),
            ));
        }
        // Both scopes already authenticated each current operator/provenance.
        // Ignore only the actual new call issuer/base/IDs, not any SSA/lowerer fact.
        for (index, (a, b)) in ap.nodes().iter().zip(bp.nodes()).enumerate() {
            same_node(index, a, b)?;
        }
        let metadata = scope
            .source()
            .source()
            .source()
            .reserve_workspace(64 * 1024)?;
        Ok(Arc::new(Self {
            scope,
            geometry: old.geometry.clone(),
            _binding_metadata: Some(metadata),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::{
        SourceIdentity,
        program::{
            FullModelProfile, KvDomain, ShortHfCall, ShortHfSpan, ShortHfWorkload, StateSlot,
            StateVersion, Vllm08Operator,
        },
    };
    fn node(source: SourceIdentity, base: usize) -> Result<PlanNode> {
        let call = ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::cached(source, ShortHfWorkload::Prompt47, base, 11)?,
        )?;
        Ok(PlanNode::new(
            "fresh FA".into(),
            PlanOperation::TypedVllm08 {
                descriptor: Box::new(Vllm08Operator::new(
                    call.as_vllm_call().clone(),
                    Some(call.as_vllm_call().layer(3)?),
                    Vllm08Kind::FlashAttention,
                    vec![],
                    vec![],
                )?),
                state: Some(StateVersion {
                    slot: StateSlot::Kv(KvDomain(3)),
                    ordinal: 1,
                }),
            },
            vec![ValueId(1)],
            vec![ValueId(2)],
            None,
        ))
    }
    #[test]
    fn fresh_feedback_issuer_and_base_do_not_change_the_cold_allocation_analysis() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let cold = node(source, 47)?;
        for base in 47..=53 {
            let current = node(source, base)?;
            same_node(33, &cold, &current)?;
            let (
                PlanOperation::TypedVllm08 { descriptor: a, .. },
                PlanOperation::TypedVllm08 { descriptor: b, .. },
            ) = (&cold.operation, &current.operation)
            else {
                unreachable!()
            };
            assert_ne!(a.call(), b.call());
        }
        let foreign = node(SourceIdentity::for_new_allocation()?, 47)?;
        assert!(same_node(33, &cold, &foreign).is_err());
        Ok(())
    }
    #[test]
    fn cold_analysis_does_not_accept_ssa_effect_or_private_state_read_edits() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let original = node(source, 47)?;
        same_node(33, &original, &original)?;
        for mutation in 0..4 {
            let mut changed = original.clone();
            match mutation {
                0 => changed.inputs[0] = ValueId(8),
                1 => changed.outputs[0] = ValueId(8),
                2 => {
                    if let PlanOperation::TypedVllm08 { state, .. } = &mut changed.operation {
                        *state = None;
                    }
                }
                _ => {
                    if let PlanOperation::TypedVllm08 { state, .. } = &mut changed.operation {
                        *state = Some(StateVersion {
                            slot: StateSlot::Kv(KvDomain(7)),
                            ordinal: 1,
                        });
                    }
                }
            }
            assert!(same_node(33, &original, &changed).is_err());
        }
        Ok(())
    }
}
