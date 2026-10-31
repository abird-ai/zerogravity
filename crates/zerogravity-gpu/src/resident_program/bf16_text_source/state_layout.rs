//! Physical state from typed producers, never model counts or tensor names.
use super::*;
use zerogravity_core::accelerator::program::SourceStateBinding;

fn producer(plan: &ForwardPlan, value: ValueId) -> Result<&PlanNode> {
    plan.nodes()
        .iter()
        .find(|node| node.outputs.contains(&value))
        .ok_or_else(|| Error::State("source state producer absent".into()))
}
fn prior(plan: &ForwardPlan, value: ValueId, slot: StateSlot) -> Result<()> {
    let node = producer(plan, value)?;
    match &node.operation {
        PlanOperation::TypedSourceStateRead { binding, state }
            if binding.slot() == slot && state.slot == slot && state.ordinal == 0 =>
        {
            Ok(())
        }
        _ => Err(Error::State(
            "source state original domain/version join".into(),
        )),
    }
}
pub(super) fn derive(
    recorded: &TextStepBindings,
    write: &PlanNode,
    binding: &SourceStateBinding,
) -> Result<StateLayout> {
    let plan = recorded.plan();
    let effect = write
        .state
        .ok_or_else(|| Error::State("source state effect absent".into()))?;
    if effect.read.slot != binding.slot()
        || effect.write.slot != binding.slot()
        || effect.read.ordinal != 0
        || effect.write.ordinal != 1
        || write.inputs.len() != binding.components().len()
    {
        return Err(Error::State("source state exact effect/components".into()));
    }
    for (id, n) in write.inputs.iter().zip(binding.components()) {
        let value = plan
            .values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("source state input absent".into()))?;
        if value.capacity_elements()? != *n
            || plan.precisions().get(id.0 as usize) != Some(&binding.precision())
        {
            return Err(Error::State(
                "source state explicit physical precision/extent".into(),
            ));
        }
    }
    let node = producer(plan, write.inputs[0])?;
    let (layout, components) = match (binding.slot(), &node.operation) {
        (StateSlot::Kv(slot), PlanOperation::TypedTorchAttention { geometry, position }) => {
            let (_, slab, active) = geometry.validate(*position)?;
            if *position != recorded.position()
                || node.outputs.get(2..4) != Some(write.inputs.as_slice())
                || node.inputs.len() != if *position == 0 { 4 } else { 6 }
            {
                return Err(Error::State("source KV exact attention producers".into()));
            }
            if *position > 0 {
                prior(plan, node.inputs[4], StateSlot::Kv(slot))?;
                let read = producer(plan, node.inputs[4])?;
                if read.outputs.as_slice() != &node.inputs[4..6] {
                    return Err(Error::State("source KV joined original planes".into()));
                }
                for id in &node.inputs[4..6] {
                    if plan.values()[id.0 as usize].capacity_elements()? != *position * slab {
                        return Err(Error::State("source KV actual active prefix".into()));
                    }
                }
            }
            (
                StateLayout::Kv {
                    slab,
                    capacity: recorded.capacity(),
                },
                vec![active * slab; 2],
            )
        }
        (StateSlot::Recurrent(slot), PlanOperation::TypedTorchGdn { contract, phase }) => {
            let (_, _, elements) = contract.validate(plan.source())?;
            if node.outputs.get(3) != write.inputs.first()
                || node.inputs.len() != 6
                || (*phase == TorchCpuGdnPhase::InitialOne) != (recorded.position() == 0)
            {
                return Err(Error::State("source recurrent actual GDN producer".into()));
            }
            prior(plan, node.inputs[5], StateSlot::Recurrent(slot))?;
            (StateLayout::Vector { elements }, vec![elements])
        }
        (StateSlot::Conv(slot), PlanOperation::TypedTorchFullConvolution) => {
            if node.inputs.len() != 3 || node.outputs.as_slice() != write.inputs.as_slice() {
                return Err(Error::State("source full4 exact producer".into()));
            }
            prior(plan, node.inputs[0], StateSlot::Conv(slot))?;
            // Full4 is computed from the old history/current projection before
            // GDN runs; the GDN node is a consumer, not this history's producer.
            let mut consumers = plan.nodes().iter().filter(|candidate| {
                matches!(candidate.operation, PlanOperation::TypedTorchGdn { .. })
                    && candidate.inputs.first() == node.inputs.get(2)
                    && candidate.inputs.get(4) == node.inputs.get(1)
            });
            let gdn = consumers
                .next()
                .ok_or_else(|| Error::State("source full4 requires genuine GDN consumer".into()))?;
            if consumers.next().is_some() {
                return Err(Error::State(
                    "source full4 GDN consumer is ambiguous".into(),
                ));
            }
            let PlanOperation::TypedTorchGdn { contract, .. } = &gdn.operation else {
                unreachable!()
            };
            let (channels, _, _) = contract.validate(plan.source())?;
            let minimal = producer(plan, node.inputs[1])?;
            if !matches!(minimal.operation, PlanOperation::Contiguous) || minimal.inputs.len() != 1
            {
                return Err(Error::State(
                    "source GDN contiguous previous history".into(),
                ));
            }
            let view = producer(plan, minimal.inputs[0])?;
            let PlanOperation::View(shape) = view.operation else {
                return Err(Error::State("source GDN previous full4 tail view".into()));
            };
            if view.inputs.as_slice() != &node.inputs[..1]
                || shape.offset != 1
                || shape.rows != channels
                || shape.width != 3
                || shape.row_stride != 4
            {
                return Err(Error::State(
                    "source GDN original full4/history join".into(),
                ));
            }
            (
                StateLayout::Vector {
                    elements: channels * 4,
                },
                vec![channels * 4],
            )
        }
        _ => {
            return Err(Error::Unsupported(
                "source state missing typed native producer".into(),
            ));
        }
    };
    if binding.components() != components {
        return Err(Error::State(
            "source state geometry differs from actual producer".into(),
        ));
    }
    Ok(layout)
}
