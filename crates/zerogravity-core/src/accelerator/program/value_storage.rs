//! Bounded SSA storage metadata; no allocation, alias, enqueue or Ready permit.
use super::{ActivationPrecision, ForwardPlan, ValueId};
use crate::device_events::DeviceAccess;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueStorageFormat {
    F32,
    Bf16,
}
impl ValueStorageFormat {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// Explicit physical selection; numerical precision does not select storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueStorageLayout {
    pub format: ValueStorageFormat,
    pub elements: usize,
}
impl ValueStorageLayout {
    pub fn bytes(self) -> Result<usize> {
        self.elements
            .checked_mul(self.format.byte_width())
            .filter(|_| self.elements > 0 && u32::try_from(self.elements).is_ok())
            .ok_or_else(|| Error::Limit("value storage element/byte extent".into()))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ValueStorageLimits {
    pub max_nodes: usize,
    pub max_values: usize,
    pub max_slots: usize,
    pub max_slot_bytes: usize,
    pub max_total_bytes: usize,
    pub alignment: usize,
    pub reuse_policy: ValueStorageReusePolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueStorageReusePolicy {
    /// Capacity analysis only for lowerers that permit bounded larger owners.
    BestFit,
    /// Preserve the exact aligned allocation class required by native handles.
    ExactAlignedSize,
}
impl ValueStorageLimits {
    fn validate(self) -> Result<()> {
        if self.max_nodes == 0
            || self.max_values == 0
            || self.max_slots == 0
            || self.max_slots > self.max_values
            || self.max_slot_bytes == 0
            || self.max_total_bytes == 0
            || !self.alignment.is_power_of_two()
        {
            return Err(Error::Malformed("value storage finite limits".into()));
        }
        Ok(())
    }
    fn aligned(self, bytes: usize) -> Result<usize> {
        bytes
            .checked_add(self.alignment - 1)
            .map(|n| n & !(self.alignment - 1))
            .filter(|&n| n > 0 && n <= self.max_slot_bytes)
            .ok_or_else(|| Error::Limit("value storage aligned slot limit".into()))
    }
}

/// `pinned` always receives a fresh dedicated slot, never reused in either
/// direction. `escaped` may reuse earlier dead storage, then survives to end.
/// Pin actor-owned outputs not yet lowered to checked reusable destinations.
/// Taps require an original ordered D2D copy after the producer's WHOLE operation
/// and before any slot reuse; their destinations survive through completion.
#[derive(Clone, Copy, Debug)]
pub enum ValueStorageRetention<'a> {
    RetainAll,
    Reuse {
        escaped: &'a [ValueId],
        pinned: &'a [ValueId],
        taps: &'a [ValueId],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueLifetime {
    pub producer: usize,
    /// Inclusive. Output storage never aliases an input consumed at this node.
    pub last_use: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueStorageBinding {
    pub slot: usize,
    /// A logical overwrite epoch, not an allocation or initialization generation.
    pub epoch: usize,
    pub bytes: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueStorageSlot {
    pub format: ValueStorageFormat,
    pub reserved_bytes: usize,
    pub dedicated: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueStorageTap {
    pub value: ValueId,
    pub after_node: usize,
    pub bytes: usize,
    pub reserved_bytes: usize,
}

/// Borrows the EXACT immutable source plan. An equal clone is not its issuer.
/// A backend still needs original leases, current producer seals, complete
/// operation order, admitted tap copies and actual completion/quarantine.
pub struct ValueStoragePlan<'a> {
    plan: &'a ForwardPlan,
    layouts: Vec<ValueStorageLayout>,
    lifetimes: Vec<ValueLifetime>,
    dedicated: Vec<bool>,
    bindings: Vec<ValueStorageBinding>,
    slots: Vec<ValueStorageSlot>,
    taps: Vec<ValueStorageTap>,
    total_bytes: usize,
    limits: ValueStorageLimits,
}

impl ForwardPlan {
    pub fn plan_value_storage<'a>(
        &'a self,
        layouts: &[ValueStorageLayout],
        retention: ValueStorageRetention<'_>,
        limits: ValueStorageLimits,
    ) -> Result<ValueStoragePlan<'a>> {
        ValueStoragePlan::build(self, layouts, retention, limits)
    }
}

impl<'a> ValueStoragePlan<'a> {
    fn build(
        plan: &'a ForwardPlan,
        layouts: &[ValueStorageLayout],
        retention: ValueStorageRetention<'_>,
        limits: ValueStorageLimits,
    ) -> Result<Self> {
        limits.validate()?;
        let n = plan.values().len();
        if !plan.source().is_bound()
            || n == 0
            || n > limits.max_values
            || plan.nodes().is_empty()
            || plan.nodes().len() > limits.max_nodes
            || layouts.len() != n
            || plan.precisions().len() != n
        {
            return Err(Error::Limit(
                "value storage source/plan/metadata limits".into(),
            ));
        }
        for ((shape, precision), layout) in plan.values().iter().zip(plan.precisions()).zip(layouts)
        {
            if layout.elements != shape.capacity_elements()?
                || layout.format == ValueStorageFormat::Bf16
                    && *precision != ActivationPrecision::Bf16WidenedF32
            {
                return Err(Error::State(
                    "value storage physical/precision layout".into(),
                ));
            }
            limits.aligned(layout.bytes()?)?;
        }
        let end = plan.nodes().len();
        let mut lifetimes = vec![None::<ValueLifetime>; n];
        for (node, operation) in plan.nodes().iter().enumerate() {
            for id in &operation.inputs {
                let life = lifetimes
                    .get_mut(id.0 as usize)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| Error::State("value storage read before producer".into()))?;
                life.last_use = node;
            }
            for id in &operation.outputs {
                let life = lifetimes
                    .get_mut(id.0 as usize)
                    .ok_or_else(|| Error::State("value storage output index".into()))?;
                if life.is_some() {
                    return Err(Error::State("value storage duplicate SSA producer".into()));
                }
                *life = Some(ValueLifetime {
                    producer: node,
                    last_use: node,
                });
            }
        }
        let mut lifetimes = lifetimes
            .into_iter()
            .map(|life| life.ok_or_else(|| Error::State("value storage missing producer".into())))
            .collect::<Result<Vec<_>>>()?;
        let mut tapped = vec![false; n];
        let mut dedicated = vec![false; n];
        match retention {
            ValueStorageRetention::RetainAll => {
                for life in &mut lifetimes {
                    life.last_use = end;
                }
                dedicated.fill(true);
            }
            ValueStorageRetention::Reuse {
                escaped,
                pinned,
                taps,
            } => {
                let mut selected = vec![false; n];
                for id in escaped {
                    select(&mut selected, *id)?;
                    lifetimes[id.0 as usize].last_use = end;
                }
                for id in pinned {
                    select(&mut selected, *id)?;
                    lifetimes[id.0 as usize].last_use = end;
                    dedicated[id.0 as usize] = true;
                }
                for id in taps {
                    select(&mut selected, *id)?;
                    tapped[id.0 as usize] = true;
                }
            }
        }
        // A published plan's logits are an implicit external consumer. The
        // unpublished full-model adapter must also escape its actual head ValueId.
        if let Some(logits) = plan.logits() {
            lifetimes
                .get_mut(logits.0 as usize)
                .ok_or_else(|| Error::State("value storage logits index".into()))?
                .last_use = end;
        }
        let mut result = Self {
            plan,
            layouts: layouts.to_vec(),
            lifetimes,
            dedicated,
            bindings: vec![
                ValueStorageBinding {
                    slot: 0,
                    epoch: 0,
                    bytes: 0
                };
                n
            ],
            slots: Vec::new(),
            taps: Vec::new(),
            total_bytes: 0,
            limits,
        };
        let mut slot_last = Vec::<(usize, usize)>::new();
        // Value IDs need not be ordered by producer; traverse the ACTUAL nodes.
        for operation in plan.nodes() {
            for id in &operation.outputs {
                let value = id.0 as usize;
                let layout = layouts[value];
                let life = result.lifetimes[value];
                let bytes = layout.bytes()?;
                let aligned = limits.aligned(bytes)?;
                let best = result
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(slot, storage)| {
                        !result.dedicated[value]
                            && !storage.dedicated
                            && storage.format == layout.format
                            && slot_last[*slot].0 < life.producer
                            && (limits.reuse_policy == ValueStorageReusePolicy::BestFit
                                || storage.reserved_bytes == aligned)
                    })
                    .min_by_key(|(slot, storage)| {
                        (
                            aligned.saturating_sub(storage.reserved_bytes),
                            storage.reserved_bytes.max(aligned),
                            *slot,
                        )
                    })
                    .map(|(slot, _)| slot);
                let (slot, epoch) = if let Some(slot) = best {
                    let growth = aligned.saturating_sub(result.slots[slot].reserved_bytes);
                    result.add_bytes(growth)?;
                    result.slots[slot].reserved_bytes =
                        result.slots[slot].reserved_bytes.max(aligned);
                    let epoch = slot_last[slot]
                        .1
                        .checked_add(1)
                        .ok_or_else(|| Error::Limit("value storage epoch overflow".into()))?;
                    slot_last[slot] = (life.last_use, epoch);
                    (slot, epoch)
                } else {
                    if result.slots.len() == limits.max_slots {
                        return Err(Error::Limit("value storage slot count".into()));
                    }
                    result.add_bytes(aligned)?;
                    let slot = result.slots.len();
                    result.slots.push(ValueStorageSlot {
                        format: layout.format,
                        reserved_bytes: aligned,
                        dedicated: result.dedicated[value],
                    });
                    slot_last.push((life.last_use, 0));
                    (slot, 0)
                };
                result.bindings[value] = ValueStorageBinding { slot, epoch, bytes };
                if tapped[value] {
                    result.add_bytes(aligned)?;
                    result.taps.push(ValueStorageTap {
                        value: *id,
                        after_node: life.producer,
                        bytes,
                        reserved_bytes: aligned,
                    });
                }
            }
        }
        result.validate()?;
        Ok(result)
    }
    fn add_bytes(&mut self, bytes: usize) -> Result<()> {
        self.total_bytes = self
            .total_bytes
            .checked_add(bytes)
            .filter(|&n| n <= self.limits.max_total_bytes)
            .ok_or_else(|| Error::Limit("value storage total admission".into()))?;
        Ok(())
    }
    pub fn require_plan(&self, plan: &ForwardPlan) -> Result<()> {
        if !std::ptr::eq(self.plan, plan) {
            return Err(Error::State("value storage foreign plan issuer".into()));
        }
        Ok(())
    }
    pub fn binding_at(
        &self,
        node: usize,
        value: ValueId,
        access: DeviceAccess,
    ) -> Result<ValueStorageBinding> {
        let operation = self
            .plan
            .nodes()
            .get(node)
            .ok_or_else(|| Error::State("value storage node index".into()))?;
        let ids = match access {
            DeviceAccess::Read => &operation.inputs,
            DeviceAccess::Write => &operation.outputs,
        };
        if !ids.contains(&value) {
            return Err(Error::State(
                "value storage not an actual node operand".into(),
            ));
        }
        self.bindings
            .get(value.0 as usize)
            .copied()
            .ok_or_else(|| Error::State("value storage operand index".into()))
    }
    pub fn lifetimes(&self) -> &[ValueLifetime] {
        &self.lifetimes
    }
    pub fn bindings(&self) -> &[ValueStorageBinding] {
        &self.bindings
    }
    pub fn slots(&self) -> &[ValueStorageSlot] {
        &self.slots
    }
    pub fn taps(&self) -> &[ValueStorageTap] {
        &self.taps
    }
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn validate(&self) -> Result<()> {
        for (node, operation) in self.plan.nodes().iter().enumerate() {
            for id in &operation.inputs {
                let life = self
                    .lifetimes
                    .get(id.0 as usize)
                    .ok_or_else(|| Error::State("value storage validation input".into()))?;
                if life.producer >= node || life.last_use < node {
                    return Err(Error::State("value storage truncated live input".into()));
                }
            }
            for id in &operation.outputs {
                let life = self
                    .lifetimes
                    .get(id.0 as usize)
                    .ok_or_else(|| Error::State("value storage validation output".into()))?;
                if life.producer != node || life.last_use < node {
                    return Err(Error::State("value storage producer lifetime".into()));
                }
            }
        }
        let mut total = 0usize;
        for slot in &self.slots {
            if self.limits.aligned(slot.reserved_bytes)? != slot.reserved_bytes {
                return Err(Error::State("value storage slot alignment".into()));
            }
            total = total
                .checked_add(slot.reserved_bytes)
                .ok_or_else(|| Error::Limit("value storage validation total".into()))?;
        }
        for (i, binding) in self.bindings.iter().enumerate() {
            let slot = self
                .slots
                .get(binding.slot)
                .ok_or_else(|| Error::State("value storage missing slot".into()))?;
            if binding.bytes != self.layouts[i].bytes()?
                || binding.bytes > slot.reserved_bytes
                || slot.format != self.layouts[i].format
                || slot.dedicated != self.dedicated[i]
                || self.dedicated[i] && binding.epoch != 0
                || self.limits.reuse_policy == ValueStorageReusePolicy::ExactAlignedSize
                    && self.limits.aligned(binding.bytes)? != slot.reserved_bytes
            {
                return Err(Error::State("value storage binding extent/format".into()));
            }
            for (j, other) in self.bindings[..i].iter().enumerate() {
                if binding.slot != other.slot {
                    continue;
                }
                let a = self.lifetimes[i];
                let b = self.lifetimes[j];
                if slot.dedicated
                    || a.producer <= b.last_use && b.producer <= a.last_use
                    || binding.epoch == other.epoch
                    || (a.producer > b.producer) != (binding.epoch > other.epoch)
                {
                    return Err(Error::State("value storage live SSA alias/epoch".into()));
                }
            }
        }
        let mut tapped = vec![false; self.layouts.len()];
        for tap in &self.taps {
            let value = tap.value.0 as usize;
            select(&mut tapped, tap.value)?;
            if tap.after_node != self.lifetimes[value].producer
                || tap.bytes != self.layouts[value].bytes()?
                || tap.reserved_bytes != self.limits.aligned(tap.bytes)?
            {
                return Err(Error::State("value storage tap order/extent".into()));
            }
            total = total
                .checked_add(tap.reserved_bytes)
                .ok_or_else(|| Error::Limit("value storage tap total".into()))?;
        }
        if total != self.total_bytes || total > self.limits.max_total_bytes {
            return Err(Error::State("value storage total differs".into()));
        }
        Ok(())
    }
}
fn select(selected: &mut [bool], value: ValueId) -> Result<()> {
    let flag = selected
        .get_mut(value.0 as usize)
        .ok_or_else(|| Error::State("value storage retained/tap index".into()))?;
    if std::mem::replace(flag, true) {
        return Err(Error::State("value storage duplicate retention/tap".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
