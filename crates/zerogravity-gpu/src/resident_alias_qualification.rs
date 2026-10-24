//! H19 physical allocation audit. Numerical and completion gates remain shared.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// Valid both cold and warm: skipped output tails may remain uninitialized.
/// Inspect admitted originals, never promote a value or submit device work.
pub(super) async fn qualify(resident: &Arc<ResidentModel>) -> Result<()> {
    resident.service.ensure_usable()?;
    let workspace = resident.workspace.lock().await;
    let program = &workspace.program;
    let plan = program.plan();
    let values = program.values();
    let fail = |detail: &str| Error::State(format!("H19 resident alias audit: {detail}"));
    let add = |a: usize, b: usize| {
        a.checked_add(b)
            .ok_or_else(|| Error::Limit("H19 allocation byte sum overflow".into()))
    };
    if values.len() != plan.values().len() {
        return Err(fail("physical/logical value counts differ"));
    }
    let mut owners = BTreeMap::new();
    let mut lease_ids = BTreeSet::new();
    let mut ledger_bytes = 0;
    for (lease, access) in &workspace.leases {
        let handle = lease.handle();
        if !lease_ids.insert(lease.id())
            || owners
                .insert(handle.clone().binding().memory.id(), (lease, *access))
                .is_some()
            || !Arc::ptr_eq(lease.service(), &resident.service)
            || lease.source_identity() != resident.service.source_identity()
            || lease.service_generation() != resident.service.generation()
            || handle.service != resident.service.client().service_id()
            // CubeCL may trim allocator alignment padding with offsets. The
            // admitted original is its logical range, not the backing page.
            || handle.size_in_used() != lease.bytes() as u64
        {
            return Err(fail("original allocation/lease is duplicated or not whole"));
        }
        ledger_bytes = add(ledger_bytes, lease.bytes())?;
    }
    let greedy = workspace
        .greedy
        .as_ref()
        .ok_or_else(|| fail("model greedy workspace absent"))?;
    let greedy_bytes = greedy.iter().try_fold(0, |n, handle| {
        add(n, lease_for(&workspace.leases, handle)?.bytes())
    })?;
    if ledger_bytes != add(program.budget().owned_bytes()?, greedy_bytes)? {
        return Err(fail("admitted originals differ from physical preflight"));
    }

    // Every lowerer output, including View, is packed dense storage. Walking
    // producers proves Contiguous only aliases an earlier immutable dense SSA
    // value; its logical View metadata does not describe physical strides.
    let mut produced = vec![false; values.len()];
    let mut value_originals = BTreeSet::new();
    let mut logical_bytes = 0;
    let mut physical_bytes = 0;
    let mut alias_bytes = 0;
    let mut aliases = 0;
    let mut initialized_values = 0;
    let mut initialized_aliases = 0;
    for node in plan.nodes() {
        for input in &node.inputs {
            if !produced.get(input.0 as usize).copied().unwrap_or(false) {
                return Err(fail("input is not produced by an earlier node"));
            }
        }
        let alias = matches!(node.operation, PlanOperation::Contiguous);
        if alias && (node.inputs.len() != 1 || node.outputs.len() != 1 || node.state.is_some()) {
            return Err(fail("Contiguous arity or state effect differs"));
        }
        for output in &node.outputs {
            let id = output.0 as usize;
            let buffer = values.get(id).ok_or_else(|| fail("output index"))?;
            let shape = plan.values().get(id).ok_or_else(|| fail("output shape"))?;
            let elements = shape.capacity_elements()?;
            let bytes = elements
                .checked_mul(4)
                .ok_or_else(|| Error::Limit("H19 value bytes overflow".into()))?;
            if produced[id] || elements == 0 || buffer.elements != elements {
                return Err(fail("output is duplicated, empty or incorrectly shaped"));
            }
            let (owner, access) = owners
                .get(&buffer.handle.clone().binding().memory.id())
                .ok_or_else(|| fail("value has no original admitted owner"))?;
            let original = owner.handle();
            if *access != DeviceAccess::Write
                || owner.bytes() != bytes
                || !same_handle(&original, &buffer.handle)
                || original.size() != buffer.handle.size()
                || original.stream != buffer.handle.stream
                || buffer.handle.size_in_used() != bytes as u64
            {
                return Err(fail("value is not the exact whole writable original"));
            }
            logical_bytes = add(logical_bytes, bytes)?;
            initialized_values += usize::from(owner.is_initialized());
            if alias {
                let input = node.inputs[0].0 as usize;
                let source = &values[input];
                if input >= id
                    || plan.values()[input] != *shape
                    || source.elements != elements
                    || !same_handle(&source.handle, &buffer.handle)
                    || source.handle.size() != buffer.handle.size()
                    || source.handle.stream != buffer.handle.stream
                    || !value_originals.contains(&owner.id())
                {
                    return Err(fail(
                        "Contiguous does not name its earlier whole dense input",
                    ));
                }
                aliases += 1;
                alias_bytes = add(alias_bytes, bytes)?;
                initialized_aliases += usize::from(owner.is_initialized());
            } else {
                if !value_originals.insert(owner.id()) {
                    return Err(fail("non-Contiguous output shares an allocation"));
                }
                physical_bytes = add(physical_bytes, bytes)?;
            }
            produced[id] = true;
        }
    }
    // The two recorded trained schedules differ only by diagnostic GDN packs.
    // Savings themselves are derived from each actual typed node/shape above.
    if ![12, 48].contains(&aliases)
        || produced.iter().any(|p| !p)
        || value_originals.len() + aliases != values.len()
        || physical_bytes != program.budget().value_bytes
        || logical_bytes.checked_sub(physical_bytes) != Some(alias_bytes)
    {
        return Err(fail(
            "trained alias count or unique value byte savings differ",
        ));
    }
    println!(
        "H19_RESIDENT_PACKED_ALIAS_PASS physical={:?} contiguous_nodes={aliases} eliminated_value_allocations={aliases} saved_value_bytes={alias_bytes} unoptimized_value_bytes={logical_bytes} actual_value_bytes={physical_bytes} original_value_allocations={} workspace_leases={} admitted_workspace_bytes={ledger_bytes} initialized_values={initialized_values} initialized_aliases={initialized_aliases} exact_whole_range=true earlier_dense_producers=true one_lease_per_original=true",
        resident.service.physical_key(),
        value_originals.len(),
        workspace.leases.len()
    );
    Ok(())
}
