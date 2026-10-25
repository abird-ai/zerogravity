//! Exact-size original owners; logical epochs confer no Ready/native authority.
use super::*;
use crate::resident_leases::ResidentInitialization;
use crate::vllm_cuda_input_rms::{FullModelRmsScratch, FusedRmsPrepared, RmsPrepared};
use crate::vllm_cuda_qualification::short_hf_model::ShortHfModelBinding;
use std::sync::{
    Mutex, OnceLock, Weak,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    PlanNode, PlanOperation, ValueLifetime, ValueStorageBinding, ValueStorageLimits,
    ValueStorageRetention, ValueStorageReusePolicy, Vllm08Kind, Vllm08Projection,
};

mod epoch;
pub(crate) mod inventory;
mod rebind;
pub(crate) mod workspace;
use epoch::Cursor;

const METADATA_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_NODE_ACCESSES: usize = 128;
pub(crate) const MAX_TOTAL_ACCESSES: usize = 4096;
pub(crate) const FINITE_CODE_GROUP_WIDTH: usize = 8;

/// Immutable group boundaries. Runtime exclusions never repack surviving values
/// into a different source/ABI key; their original individual checks remain.
pub(crate) fn finite_code_groups(dedicated: &[bool]) -> Result<Vec<Vec<ValueId>>> {
    if dedicated.len() > 512 {
        return Err(Error::Limit("finite code immutable member extent".into()));
    }
    Ok(dedicated
        .iter()
        .enumerate()
        .filter_map(|(id, &keep)| keep.then_some(ValueId(id as u32)))
        .collect::<Vec<_>>()
        .chunks(FINITE_CODE_GROUP_WIDTH)
        .filter(|group| group.len() >= 2)
        .map(<[ValueId]>::to_vec)
        .collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StorageMode {
    RetainAll,
    ReuseMlp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StorageInventory {
    pub(crate) logical_values: usize,
    pub(crate) physical_slots: usize,
    pub(crate) actor_slots: usize,
    pub(crate) reused_slots: usize,
    pub(crate) logical_slot_bytes: usize,
    pub(crate) aligned_slot_bytes: usize,
}

/// Cold immutable analysis; no current record, cursor, leases or run authority.
pub(crate) struct StorageGeometry {
    mode: StorageMode,
    layouts: Box<[ResidentBufferLayout]>,
    bindings: Box<[ValueStorageBinding]>,
    lives: Box<[ValueLifetime]>,
    dedicated: Box<[bool]>,
    external: Box<[bool]>,
    slot_bytes: Box<[usize]>,
    producers: Box<[usize]>,
    finite_dedicated: Box<[bool]>,
    total_bytes: usize,
    alignment: usize,
    _metadata: WindowWorkspace,
}
pub(crate) struct StorageRecipe {
    scope: Arc<ShortHfModelBinding>,
    geometry: Arc<StorageGeometry>,
    _binding_metadata: Option<WindowWorkspace>,
}
impl std::ops::Deref for StorageRecipe {
    type Target = StorageGeometry;
    fn deref(&self) -> &Self::Target {
        &self.geometry
    }
}
impl StorageRecipe {
    /// Only two proved MLP temporaries per layer may be reused. No caller IDs,
    /// hidden-access omission, late tap or future Head escape is inferred.
    pub(crate) fn new(
        scope: Arc<ShortHfModelBinding>,
        mode: StorageMode,
        limits: ValueStorageLimits,
    ) -> Result<Arc<Self>> {
        scope.call().validate()?;
        let record = scope.record();
        let plan = record.plan();
        if plan.values().len() != 327
            || plan.nodes().len() != 255
            || limits.max_nodes != 255
            || limits.max_values != 327
            || limits.max_slots > 327
            || limits.reuse_policy != ValueStorageReusePolicy::ExactAlignedSize
        {
            return Err(Error::State(
                "bounded full255/327 exact-size storage limits".into(),
            ));
        }
        // Admission precedes derived vectors, slot metadata and access envelopes.
        let metadata = scope
            .source()
            .source()
            .source()
            .reserve_workspace(METADATA_BYTES)?;
        let layouts = plan
            .values()
            .iter()
            .map(|shape| {
                ResidentBufferLayout::new(ResidentPhysicalType::Bf16, shape.capacity_elements()?)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut reusable = vec![false; 327];
        let mut external = vec![false; 327];
        external[0] = true;
        external[1] = true;
        let mut pairs = 0;
        for (index, node) in plan.nodes().iter().enumerate() {
            let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                continue;
            };
            if matches!(descriptor.kind(), Vllm08Kind::ResidualRms(_)) {
                for id in &node.outputs {
                    external[id.0 as usize] = true;
                }
            }
            if descriptor.kind() != Vllm08Kind::Projection(Vllm08Projection::GateUp) {
                continue;
            }
            let a = plan
                .nodes()
                .get(index + 1)
                .ok_or_else(|| Error::State("bounded actual activation absent".into()))?;
            let d = plan
                .nodes()
                .get(index + 2)
                .ok_or_else(|| Error::State("bounded actual Down absent".into()))?;
            let ([gate], [activated]) = (node.outputs.as_slice(), a.outputs.as_slice()) else {
                return Err(Error::State("bounded MLP producer arity".into()));
            };
            let typed = |n: &PlanNode, kind| {
                matches!(&n.operation,
                PlanOperation::TypedVllm08 { descriptor: op, .. }
                if op.kind() == kind && op.call() == record.call() && op.layer() == descriptor.layer())
            };
            if !typed(a, Vllm08Kind::SiluAndMul)
                || !typed(d, Vllm08Kind::Projection(Vllm08Projection::Down))
                || a.inputs != [*gate]
                || d.inputs != [*activated]
            {
                return Err(Error::State(
                    "bounded SAME layer GateUp/activation/Down chain".into(),
                ));
            }
            reusable[gate.0 as usize] = true;
            reusable[activated.0 as usize] = true;
            pairs += 1;
        }
        if pairs != 24 || external.iter().filter(|&&x| x).count() != 98 {
            return Err(Error::State(
                "bounded actual24 MLP and49 dedicated RMS producers".into(),
            ));
        }
        let pinned = reusable
            .iter()
            .enumerate()
            .filter_map(|(id, &reuse)| (!reuse).then_some(ValueId(id as u32)))
            .collect::<Vec<_>>();
        let retention = match mode {
            StorageMode::RetainAll => ValueStorageRetention::RetainAll,
            StorageMode::ReuseMlp => ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &pinned,
                taps: &[],
            },
        };
        let storage = ResidentBufferLayout::plan_values(plan, &layouts, retention, limits)?;
        storage.require_plan(plan)?;
        let mut slot_bytes = vec![0; storage.slots().len()];
        for binding in storage.bindings() {
            let old = &mut slot_bytes[binding.slot];
            if *old != 0 && *old != binding.bytes {
                // No new subview or original-lease initialized-state sharing.
                return Err(Error::Unsupported(
                    "bounded slot needs identical logical bytes".into(),
                ));
            }
            *old = binding.bytes;
        }
        let producers = storage.lifetimes().iter().map(|l| l.producer).collect();
        let finite_dedicated = storage
            .bindings()
            .iter()
            .zip(&layouts)
            .map(|(b, l)| {
                storage.slots()[b.slot].dedicated && l.kind() == ResidentPhysicalType::Bf16
            })
            .collect();
        let geometry = Arc::new(StorageGeometry {
            mode,
            layouts: layouts.into_boxed_slice(),
            bindings: storage.bindings().into(),
            lives: storage.lifetimes().into(),
            dedicated: storage.slots().iter().map(|s| s.dedicated).collect(),
            slot_bytes: slot_bytes.into_boxed_slice(),
            producers,
            finite_dedicated,
            external: external.into_boxed_slice(),
            total_bytes: storage.total_bytes(),
            alignment: limits.alignment,
            _metadata: metadata,
        });
        let result = Arc::new(Self {
            scope: scope.clone(),
            geometry,
            _binding_metadata: None,
        });
        Ok(result)
    }
    fn inventory(&self) -> StorageInventory {
        StorageInventory {
            logical_values: self.bindings.len(),
            physical_slots: self.slot_bytes.len(),
            actor_slots: self.external.iter().filter(|&&external| external).count(),
            reused_slots: self
                .dedicated
                .iter()
                .filter(|&&dedicated| !dedicated)
                .count(),
            logical_slot_bytes: self.slot_bytes.iter().sum(),
            aligned_slot_bytes: self.total_bytes,
        }
    }
    pub(crate) fn require_record(&self, record: &Arc<RecordedVllm08Model>) -> Result<()> {
        if !Arc::ptr_eq(record, self.scope.record()) {
            return Err(Error::State("bounded foreign current record issuer".into()));
        }
        Ok(())
    }
}

/// Allocate non-actor slots first; original RMS factories bind their own outputs.
pub(crate) struct StorageBuilder {
    buffers: Vec<Option<DeviceBuffer>>,
    slots: Vec<Option<ResidentLease>>,
    service: Arc<ResidentService>,
    pool: Option<Arc<PhysicalSlots>>,
    recipe: Arc<StorageRecipe>,
}
impl StorageBuilder {
    pub(crate) fn from_views(
        geometry: Arc<workspace::ValueWorkspaceGeometry>,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Self> {
        init.require_service(&service)?;
        let recipe = geometry.recipe.clone();
        if service.source_identity() != recipe.scope.call().span().source()
            || geometry.slots().len() != recipe.slot_bytes.len()
        {
            return Err(Error::State(
                "bounded exact shared geometry/source differs".into(),
            ));
        }
        init.retain_completion_owner(geometry.clone())?;
        let slots = geometry
            .slots()
            .iter()
            .cloned()
            .map(Some)
            .collect::<Vec<_>>();
        let pool = Arc::new(PhysicalSlots {
            slots: slots.clone(),
            active: Mutex::new(Weak::new()),
            workspace: Some(geometry),
            _source: recipe.scope.source().source().clone(),
        });
        let mut result = Self {
            buffers: vec![None; 327],
            slots,
            service,
            pool: Some(pool),
            recipe,
        };
        for value in 0..327 {
            if result.recipe.external[value] {
                continue;
            }
            let lease = result.slots[result.recipe.bindings[value].slot]
                .as_ref()
                .ok_or_else(|| Error::State("bounded shared original view absent".into()))?
                .clone();
            result.install(ValueId(value as u32), lease)?;
        }
        Ok(result)
    }
    pub(crate) fn allocate(
        recipe: Arc<StorageRecipe>,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Self> {
        init.require_service(&service)?;
        service.pinned_cuda_queue()?;
        if service.source_identity() != recipe.scope.call().span().source()
            || usize::try_from(service.client().properties().memory.alignment).ok()
                != Some(recipe.alignment)
            || recipe.total_bytes > service.capacity()
        {
            return Err(Error::State(
                "bounded original service/source/alignment/capacity".into(),
            ));
        }
        init.retain_completion_owner(recipe.clone())?;
        let mut external_slots = vec![false; recipe.slot_bytes.len()];
        for (value, binding) in recipe.bindings.iter().enumerate() {
            if recipe.external[value] {
                if !recipe.dedicated[binding.slot] || binding.epoch != 0 {
                    return Err(Error::State(
                        "bounded actor output must be dedicated".into(),
                    ));
                }
                external_slots[binding.slot] = true;
            }
        }
        let needed = recipe
            .slot_bytes
            .iter()
            .zip(&external_slots)
            .filter(|(_, external)| !**external)
            .try_fold(0usize, |n, (&bytes, _)| {
                n.checked_add(bytes)
                    .ok_or_else(|| Error::Limit("bounded slot extent overflow".into()))
            })?;
        if needed > service.remaining_capacity()? {
            return Err(Error::Limit(
                "bounded value admission before first allocation".into(),
            ));
        }
        let mut slots = vec![None; recipe.slot_bytes.len()];
        for (slot, &bytes) in recipe.slot_bytes.iter().enumerate() {
            if !external_slots[slot] {
                slots[slot] = Some(init.allocate_empty(bytes, BufferRole::Activation)?);
            }
        }
        let mut result = Self {
            buffers: vec![None; 327],
            slots,
            service,
            pool: None,
            recipe,
        };
        for value in 0..327 {
            if result.recipe.external[value] {
                continue;
            }
            let lease = result.slots[result.recipe.bindings[value].slot]
                .as_ref()
                .ok_or_else(|| Error::State("bounded allocated slot absent".into()))?
                .clone();
            result.install(ValueId(value as u32), lease)?;
        }
        Ok(result)
    }
    /// Fresh actual call only; no cold record's issuer/cursor is reused.
    pub(crate) fn bind_call(
        recipe: Arc<StorageRecipe>,
        template: &Arc<ModelValues>,
    ) -> Result<Self> {
        let old = template
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("bounded cold geometry absent".into()))?;
        template.service.ensure_usable()?;
        template.service.pinned_cuda_queue()?;
        if old
            .pool
            .active
            .try_lock()
            .map_err(|_| Error::State("bounded physical pool busy".into()))?
            .upgrade()
            .is_some()
        {
            return Err(Error::State(
                "bounded original flight still owns slots".into(),
            ));
        }
        let a = &old.recipe;
        require_new_record(recipe.scope.record(), &template.record)?;
        if !Arc::ptr_eq(recipe.scope.source().source(), a.scope.source().source())
            || recipe.scope.call().model_profile() != a.scope.call().model_profile()
            || !Arc::ptr_eq(&recipe.geometry, &a.geometry)
        {
            return Err(Error::State(
                "bounded fresh issuer must preserve exact cold physical recipe".into(),
            ));
        }
        let mut result = Self {
            buffers: vec![None; 327],
            slots: old.pool.slots.clone(),
            service: template.service.clone(),
            pool: Some(old.pool.clone()),
            recipe,
        };
        for value in 0..327 {
            if result.recipe.external[value] {
                continue;
            }
            let lease = result.slots[result.recipe.bindings[value].slot]
                .as_ref()
                .ok_or_else(|| Error::State("bounded original cold slot absent".into()))?
                .clone();
            result.install(ValueId(value as u32), lease)?;
        }
        Ok(result)
    }
    fn install(&mut self, id: ValueId, lease: ResidentLease) -> Result<()> {
        let value = id.0 as usize;
        let binding = self
            .recipe
            .bindings
            .get(value)
            .ok_or_else(|| Error::State("bounded value index".into()))?;
        if self.buffers[value].is_some()
            || !Arc::ptr_eq(lease.service(), &self.service)
            || lease.service_generation() != self.service.generation()
            || lease.source_identity() != self.service.source_identity()
            || lease.bytes() != binding.bytes
            || !matches!(
                lease.role(),
                BufferRole::Activation | BufferRole::Output | BufferRole::Scratch
            )
        {
            return Err(Error::State(
                "bounded exact original slot owner/source/generation/extent".into(),
            ));
        }
        if let Some(existing) = &self.slots[binding.slot] {
            require_same_lease(existing, &lease)?;
        } else {
            self.slots[binding.slot] = Some(lease.clone());
        }
        self.buffers[value] = Some(DeviceBuffer::from_lease(
            lease.clone(),
            self.recipe.layouts[value],
            lease.role(),
            DeviceAccess::Write,
        )?);
        Ok(())
    }
    pub(crate) fn lease(&self, id: ValueId) -> Result<&ResidentLease> {
        self.buffers
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::State("bounded earlier producer is not bound".into()))?
            .original_lease()
    }
    pub(crate) fn bind_ordinary(
        &mut self,
        actor: &Arc<RmsPrepared>,
        rms: &Arc<FullModelRmsScratch>,
    ) -> Result<()> {
        self.recipe.require_record(rms.record())?;
        actor.require_rms_scratch(rms, 1)?;
        self.install(ValueId(0), actor.embedding().clone())?;
        self.install(ValueId(1), actor.normalized().clone())
    }
    pub(crate) fn bind_dual(
        &mut self,
        node: usize,
        actor: &Arc<FusedRmsPrepared>,
        rms: &Arc<FullModelRmsScratch>,
    ) -> Result<()> {
        self.recipe.require_record(rms.record())?;
        actor.require_rms_scratch(rms, node)?;
        let actual = self
            .recipe
            .scope
            .record()
            .plan()
            .nodes()
            .get(node)
            .ok_or_else(|| Error::State("bounded dual source node absent".into()))?;
        let PlanOperation::TypedVllm08 { descriptor, .. } = &actual.operation else {
            return Err(Error::State("bounded dual typed producer absent".into()));
        };
        actor.require_full_model_operator(descriptor)?;
        let [normalized, residual] = actual.outputs.as_slice() else {
            return Err(Error::State("bounded exact two dual outputs".into()));
        };
        let ids = [*normalized, *residual];
        self.install(ids[0], actor.normalized().clone())?;
        self.install(ids[1], actor.residual().clone())
    }
    pub(crate) fn finish(
        self,
        status: ResidentLease,
    ) -> Result<(Arc<ModelValues>, Vec<DeviceBuffer>)> {
        let buffers = self
            .buffers
            .into_iter()
            .map(|b| b.ok_or_else(|| Error::State("bounded missing source producer".into())))
            .collect::<Result<Vec<_>>>()?;
        let mut owners = BTreeMap::new();
        let mut values = Vec::with_capacity(327);
        for (value, buffer) in buffers.iter().enumerate() {
            let lease = buffer.original_lease()?;
            let slot = self.recipe.bindings[value].slot;
            if let Some(prior_slot) = owners.insert(lease.id(), slot)
                && prior_slot != slot
            {
                return Err(Error::State(
                    "bounded distinct slots alias original owner".into(),
                ));
            }
            require_same_lease(
                lease,
                self.slots[slot]
                    .as_ref()
                    .ok_or_else(|| Error::State("bounded original slot absent".into()))?,
            )?;
            if lease.id() == status.id() {
                return Err(Error::State("bounded status aliases a value".into()));
            }
            values.push(lease.clone());
        }
        if !Arc::ptr_eq(status.service(), &self.service)
            || status.source_identity() != self.service.source_identity()
            || status.service_generation() != self.service.generation()
            || status.role() != BufferRole::Scratch
            || status.bytes() != 4
        {
            return Err(Error::State(
                "bounded original status source/service/extent".into(),
            ));
        }
        let pool = if let Some(pool) = self.pool {
            pool
        } else {
            let mut slots = self.slots;
            // Fresh real RMS actors may own fresh dedicated public outputs.
            // The geometry pool retains only its original non-actor owners.
            for (value, binding) in self.recipe.bindings.iter().enumerate() {
                if self.recipe.external[value] {
                    slots[binding.slot] = None;
                }
            }
            Arc::new(PhysicalSlots {
                slots,
                active: Mutex::new(Weak::new()),
                workspace: None,
                _source: self.recipe.scope.source().source().clone(),
            })
        };
        let pooled = values
            .iter()
            .enumerate()
            .filter_map(|(id, lease)| {
                (!self.recipe.dedicated[self.recipe.bindings[id].slot]).then_some(lease.id())
            })
            .collect();
        let storage = Arc::new(BoundedStorage {
            cursor: Mutex::new(Cursor::new(&self.recipe)),
            registered: AtomicBool::new(false),
            accesses: OnceLock::new(),
            pooled,
            pool,
            recipe: self.recipe.clone(),
        });
        Ok((
            Arc::new(ModelValues {
                record: self.recipe.scope.record().clone(),
                values,
                service: self.service,
                status,
                storage: Some(storage),
            }),
            buffers,
        ))
    }
}

fn require_same_lease(a: &ResidentLease, b: &ResidentLease) -> Result<()> {
    if a.id() != b.id()
        || a.allocation_generation() != b.allocation_generation()
        || a.bytes() != b.bytes()
        || a.role() != b.role()
        || a.service_generation() != b.service_generation()
        || a.source_identity() != b.source_identity()
        || !Arc::ptr_eq(a.service(), b.service())
    {
        return Err(Error::State("bounded exact physical lease identity".into()));
    }
    Ok(())
}

fn require_new_record<T>(record: &Arc<T>, cold: &Arc<T>) -> Result<()> {
    if Arc::ptr_eq(record, cold) {
        return Err(Error::State("bounded original record cannot reset".into()));
    }
    Ok(())
}

/// Entire concrete lowerer/copy/clear union, not a native allowlist or manifest.
/// The source-bound actual lowerer constructs this only after checking its roles.
pub(crate) struct NodeStorageAccess {
    storage: Weak<BoundedStorage>,
    record: Arc<RecordedVllm08Model>,
    node: usize,
    actual: Vec<(ResidentLease, DeviceAccess)>,
}

/// Minted only from the current record's complete concrete access census.
/// It says nothing about completion or value finiteness.
pub(crate) struct RetainedFiniteProof {
    record: Arc<RecordedVllm08Model>,
    members: Vec<(ValueId, usize)>,
}
impl RetainedFiniteProof {
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    pub(crate) fn members(&self) -> &[(ValueId, usize)] {
        &self.members
    }
    /// Sealed census-envelope constructor: per-value lease owners, dedicated
    /// flags, SSA producers and the complete per-node write census. Same
    /// eligibility equation as the N1 lowerer proof: a value defers only when
    /// its lease has exactly one owner and its only census write is by its own
    /// SSA producer node. Missing/hidden writers exclude, never group.
    pub(crate) fn from_envelope(
        record: Arc<RecordedVllm08Model>,
        owners: &[BufferLeaseId],
        dedicated: &[bool],
        producers: &[usize],
        writes: &[(usize, BufferLeaseId)],
    ) -> Result<Self> {
        if owners.len() != dedicated.len() || dedicated.len() != producers.len() {
            return Err(Error::State("finite cohort census envelope shape".into()));
        }
        let members =
            retained_finite_candidates(owners, dedicated, producers, writes)
                .into_iter()
                .map(|(value, producer)| {
                    Ok((
                        ValueId(u32::try_from(value).map_err(|_| {
                            Error::Limit("finite cohort value index overflow".into())
                        })?),
                        producer,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
        Ok(Self { record, members })
    }
}

fn retained_finite_candidates<K: Copy + Ord>(
    owners: &[K],
    dedicated: &[bool],
    producers: &[usize],
    writes: &[(usize, K)],
) -> Vec<(usize, usize)> {
    let mut aliases = BTreeMap::new();
    let mut actual = BTreeMap::<K, Vec<usize>>::new();
    for &owner in owners {
        *aliases.entry(owner).or_insert(0usize) += 1;
    }
    for &(node, owner) in writes {
        actual.entry(owner).or_default().push(node);
    }
    owners
        .iter()
        .enumerate()
        .filter_map(|(value, owner)| {
            let producer = *producers.get(value)?;
            (dedicated.get(value) == Some(&true)
                && aliases[owner] == 1
                && actual.get(owner).map(Vec::as_slice) == Some(&[producer]))
            .then_some((value, producer))
        })
        .collect()
}

impl NodeStorageAccess {
    pub(crate) fn bind(
        values: &Arc<ModelValues>,
        node: usize,
        actual: &[(ResidentLease, DeviceAccess)],
    ) -> Result<Self> {
        let storage = values
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("bounded lowerer on retained-all old binder".into()))?;
        let expected = values
            .record
            .plan()
            .nodes()
            .get(node)
            .ok_or_else(|| Error::State("bounded lowerer source node".into()))?;
        if actual.is_empty() || actual.len() > MAX_NODE_ACCESSES {
            return Err(Error::Limit("bounded concrete node access count".into()));
        }
        let mut unique = BTreeMap::new();
        for (lease, access) in actual {
            if !Arc::ptr_eq(lease.service(), values.service())
                || lease.service_generation() != values.service().generation()
                || lease.source_identity() != values.record.plan().source()
                || lease.id() == values.status.id()
            {
                return Err(Error::State(
                    "bounded lowerer lease/source/service/status".into(),
                ));
            }
            if let Some((prior, old_access)) = unique.get_mut(&lease.id()) {
                require_same_lease(prior, lease)?;
                if *access == DeviceAccess::Write {
                    *old_access = *access;
                }
            } else {
                unique.insert(lease.id(), (lease.clone(), *access));
            }
        }
        for (ids, required) in [
            (&expected.inputs, DeviceAccess::Read),
            (&expected.outputs, DeviceAccess::Write),
        ] {
            for id in ids {
                let lease = values.lease(*id)?;
                let Some((actual, access)) = unique.get(&lease.id()) else {
                    return Err(Error::State(format!(
                        "bounded lowerer omitted actual public operand node={node} value={} access={required:?}",
                        id.0
                    )));
                };
                require_same_lease(lease, actual)?;
                if *access != required {
                    return Err(Error::State("bounded lowerer public access differs".into()));
                }
            }
        }
        for (lease, access) in unique.values() {
            // Detect hidden accesses into a pooled owner beyond this node's SSA.
            let pooled = storage.pooled.contains(&lease.id());
            let declared = match access {
                DeviceAccess::Read => &expected.inputs,
                DeviceAccess::Write => &expected.outputs,
            };
            if pooled
                && !declared
                    .iter()
                    .any(|id| values.values[id.0 as usize].id() == lease.id())
            {
                return Err(Error::Unsupported(
                    "bounded hidden native access requires dedicated owner".into(),
                ));
            }
        }
        Ok(Self {
            storage: Arc::downgrade(storage),
            record: values.record.clone(),
            node,
            actual: unique.into_values().collect(),
        })
    }
}

pub(super) struct BoundedStorage {
    cursor: Mutex<Cursor>,
    registered: AtomicBool,
    accesses: OnceLock<Vec<NodeStorageAccess>>,
    pooled: std::collections::BTreeSet<BufferLeaseId>,
    pool: Arc<PhysicalSlots>,
    recipe: Arc<StorageRecipe>,
}
struct PhysicalSlots {
    slots: Vec<Option<ResidentLease>>,
    active: Mutex<Weak<ValueStorageRun>>,
    workspace: Option<Arc<workspace::ValueWorkspaceGeometry>>,
    _source: Arc<zerogravity_model_qwen35::original_binding::OriginalWindowModelSource>,
}
/// Only the original armed flight strongly owns a busy run after enqueue.
pub(crate) struct ValueStorageRun {
    storage: Arc<BoundedStorage>,
}
impl ValueStorageRun {
    pub(crate) fn require_finished(&self) -> Result<()> {
        self.storage
            .cursor
            .try_lock()
            .map_err(|_| Error::State("bounded cursor busy".into()))?
            .require_finished()
    }
}
impl ModelValues {
    pub(crate) fn workspace_geometry(
        &self,
    ) -> Option<&Arc<crate::resident_leases::backing_pool::ResidentWorkspaceGeometry>> {
        self.storage
            .as_ref()?
            .pool
            .workspace
            .as_ref()
            .map(|g| &g.owner)
    }
    /// No retained/write authority. Current grouping uses these same immutable
    /// boundaries and independently requires the complete concrete lowerer proof.
    pub(crate) fn finite_code_groups(&self) -> Result<Vec<Vec<ValueId>>> {
        let s = self
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("finite code requires bounded geometry".into()))?;
        s.recipe.require_record(&self.record)?;
        if s.recipe.finite_dedicated.len() != self.values.len() || self.values.len() > 512 {
            return Err(Error::Limit("finite code geometry extent".into()));
        }
        finite_code_groups(&s.recipe.finite_dedicated)
    }
    pub(crate) fn finite_code_candidates(&self) -> Result<Vec<Vec<u32>>> {
        self.finite_code_groups()?
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|id| {
                        let bytes = self.lease(id)?.bytes();
                        if bytes == 0 || !bytes.is_multiple_of(2) {
                            return Err(Error::State("finite code BF16 extent".into()));
                        }
                        u32::try_from(bytes / 2)
                            .map_err(|_| Error::Limit("finite code elements".into()))
                    })
                    .collect()
            })
            .collect()
    }
    pub(crate) fn retained_finite_proof(&self) -> Result<RetainedFiniteProof> {
        let s = self.storage.as_ref().ok_or_else(|| {
            Error::Unsupported("finite cohorts require concrete bounded storage".into())
        })?;
        s.recipe.require_record(&self.record)?;
        let nodes = s
            .accesses
            .get()
            .ok_or_else(|| Error::State("finite cohorts require every concrete lowerer".into()))?;
        if nodes.len() != self.record.plan().nodes().len()
            || nodes.iter().enumerate().any(|(i, n)| {
                n.node != i
                    || !Arc::ptr_eq(&n.record, &self.record)
                    || !Weak::ptr_eq(&n.storage, &Arc::downgrade(s))
            })
        {
            return Err(Error::State("finite cohort current lowerer issuer".into()));
        }
        if s.recipe.producers.len() != self.values.len()
            || s.recipe.finite_dedicated.len() != self.values.len()
        {
            return Err(Error::State(
                "finite cohort immutable producer geometry differs".into(),
            ));
        }
        let owners = self
            .values
            .iter()
            .map(ResidentLease::id)
            .collect::<Vec<_>>();
        let writes = nodes
            .iter()
            .flat_map(|n| {
                n.actual
                    .iter()
                    .filter(|(_, access)| *access == DeviceAccess::Write)
                    .map(move |(lease, _)| (n.node, lease.id()))
            })
            .collect::<Vec<_>>();
        RetainedFiniteProof::from_envelope(
            self.record.clone(),
            &owners,
            &s.recipe.finite_dedicated,
            &s.recipe.producers,
            &writes,
        )
    }

    /// Original registered run, after every actual node/epoch and before Ready.
    pub(crate) fn require_finished_storage(&self) -> Result<()> {
        let s = self
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("finite cohort original storage absent".into()))?;
        self.service.ensure_usable()?;
        let run = s
            .pool
            .active
            .try_lock()
            .map_err(|_| Error::State("finite cohort pool busy".into()))?
            .upgrade()
            .ok_or_else(|| Error::State("finite cohort armed run absent".into()))?;
        if !Arc::ptr_eq(&run.storage, s) || !s.registered.load(Ordering::Acquire) {
            return Err(Error::State("finite cohort belongs to another run".into()));
        }
        s.cursor
            .try_lock()
            .map_err(|_| Error::State("finite cohort cursor busy".into()))?
            .require_finished()
    }
    pub(crate) fn storage_inventory(&self) -> Result<StorageInventory> {
        Ok(self
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("not opt-in bounded storage".into()))?
            .recipe
            .inventory())
    }
    pub(crate) fn short_hf_storage_scope(&self) -> Result<&Arc<ShortHfModelBinding>> {
        Ok(&self
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("not source-bound short-HF storage".into()))?
            .recipe
            .scope)
    }
    pub(crate) fn retained_value_readback(
        &self,
        id: ValueId,
    ) -> Result<crate::resident_leases::ResidentReadback> {
        let lease = self.lease(id)?;
        if let Some(s) = &self.storage {
            let binding = &s.recipe.bindings[id.0 as usize];
            if !s.recipe.dedicated[binding.slot] {
                return Err(Error::Unsupported(
                    "overwritten MLP value requires an admitted producer tap".into(),
                ));
            }
        }
        lease.readback(0, lease.bytes())
    }
    pub(crate) fn seal_storage_accesses(&self, nodes: Vec<NodeStorageAccess>) -> Result<()> {
        let s = self
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("bounded envelopes require opt-in values".into()))?;
        let count = nodes.iter().try_fold(0usize, |n, node| {
            n.checked_add(node.actual.len())
                .ok_or_else(|| Error::Limit("bounded lowerer count overflow".into()))
        })?;
        if nodes.len() != 255
            || count > MAX_TOTAL_ACCESSES
            || nodes.iter().enumerate().any(|(i, n)| {
                n.node != i
                    || !Arc::ptr_eq(&n.record, &self.record)
                    || !Weak::ptr_eq(&n.storage, &Arc::downgrade(s))
            })
        {
            return Err(Error::State(
                "bounded complete255 source-issued lowerer envelopes".into(),
            ));
        }
        s.accesses
            .set(nodes)
            .map_err(|_| Error::State("bounded lowerer envelopes already sealed".into()))
    }
    pub(crate) fn require_storage_flight(
        &self,
        accesses: &[(ResidentLease, DeviceAccess)],
    ) -> Result<()> {
        let Some(s) = &self.storage else {
            return Ok(());
        };
        let nodes = s.accesses.get().ok_or_else(|| {
            Error::Unsupported("bounded complete lowerer coverage unproved".into())
        })?;
        let mut physical = BTreeMap::new();
        for (lease, access) in accesses {
            if let Some((prior, old_access)) = physical.get_mut(&lease.id()) {
                require_same_lease(prior, lease)?;
                if *access == DeviceAccess::Write {
                    *old_access = *access;
                }
            } else {
                physical.insert(lease.id(), (lease.clone(), *access));
            }
        }
        for (lease, required) in nodes
            .iter()
            .flat_map(|n| &n.actual)
            .map(|(l, access)| (l, *access))
            .chain(
                self.values
                    .iter()
                    .chain([&self.status])
                    .map(|l| (l, DeviceAccess::Write)),
            )
        {
            let actual = physical
                .get(&lease.id())
                .filter(|(_, access)| {
                    required == DeviceAccess::Read || *access == DeviceAccess::Write
                })
                .ok_or_else(|| {
                    Error::State(
                        "bounded actual original flight omits physical owner/access".into(),
                    )
                })?;
            require_same_lease(lease, &actual.0)?;
        }
        Ok(())
    }
    pub(crate) fn begin_storage(&self) -> Result<()> {
        if let Some(s) = &self.storage {
            self.service.ensure_usable()?;
            let run = s
                .pool
                .active
                .try_lock()
                .map_err(|_| Error::State("bounded original pool busy".into()))?
                .upgrade()
                .ok_or_else(|| {
                    Error::State("bounded lacks original armed-flight registration".into())
                })?;
            if !Arc::ptr_eq(&run.storage, s) || !s.registered.load(Ordering::Acquire) {
                return Err(Error::State(
                    "bounded old record cannot borrow another run's slot epoch".into(),
                ));
            }
            s.cursor
                .lock()
                .map_err(|_| Error::State("bounded cursor poisoned".into()))?
                .begin()?;
        }
        Ok(())
    }
    pub(crate) fn begin_storage_node(&self, index: usize, node: &PlanNode) -> Result<()> {
        if let Some(s) = &self.storage {
            let expected = self
                .record
                .plan()
                .nodes()
                .get(index)
                .ok_or_else(|| Error::State("bounded source node absent".into()))?;
            if !std::ptr::eq(expected, node) {
                return Err(Error::State(
                    "bounded exact original dispatch node issuer".into(),
                ));
            }
            s.cursor
                .lock()
                .map_err(|_| Error::State("bounded cursor poisoned".into()))?
                .before(&s.recipe, index, node)?;
        }
        Ok(())
    }
    pub(crate) fn finish_storage_node(&self, index: usize, actual_next: usize) -> Result<()> {
        if let Some(s) = &self.storage {
            if actual_next != index + 1 {
                return Err(Error::State(
                    "storage must follow original full-node success".into(),
                ));
            }
            let node = self
                .record
                .plan()
                .nodes()
                .get(index)
                .ok_or_else(|| Error::State("bounded source node absent".into()))?;
            s.cursor
                .lock()
                .map_err(|_| Error::State("bounded cursor poisoned".into()))?
                .after(&s.recipe, index, node)?;
        }
        Ok(())
    }
    /// Same immutable current record and original armed run/cursor only.
    /// Shared private primitive storage still needs its own exact role/order
    /// checks and inclusion in every concrete lowerer/flight access union.
    pub(crate) fn require_storage_node(&self, node: usize) -> Result<()> {
        let s = self
            .storage
            .as_ref()
            .ok_or_else(|| Error::State("pooled primitive requires opt-in storage".into()))?;
        let run = s
            .pool
            .active
            .try_lock()
            .map_err(|_| Error::State("bounded physical pool busy".into()))?
            .upgrade()
            .ok_or_else(|| Error::State("pooled primitive lacks original armed run".into()))?;
        if !Arc::ptr_eq(&run.storage, s) || !s.registered.load(Ordering::Acquire) {
            return Err(Error::State(
                "pooled primitive belongs to another actual record".into(),
            ));
        }
        s.cursor
            .try_lock()
            .map_err(|_| Error::State("bounded node cursor busy".into()))?
            .require_active(node)
    }
    pub(crate) fn finish_storage(&self) -> Result<()> {
        if let Some(s) = &self.storage {
            s.cursor
                .lock()
                .map_err(|_| Error::State("bounded cursor poisoned".into()))?
                .finish()?;
        }
        Ok(())
    }
    pub(crate) fn require_all_value_readbacks(&self) -> Result<()> {
        if self
            .storage
            .as_ref()
            .is_some_and(|s| s.recipe.mode != StorageMode::RetainAll)
        {
            return Err(Error::Unsupported(
                "pooled values require dedicated late observations; all327 readback refused".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn start_storage_run(
        &self,
        registration: &mut crate::resident_inference::RmsScratchRegistration<'_, '_>,
        rms: &Arc<FullModelRmsScratch>,
    ) -> Result<Option<Arc<ValueStorageRun>>> {
        let Some(s) = &self.storage else {
            return Ok(None);
        };
        s.recipe.require_record(rms.record())?;
        registration.require_scope(rms)?;
        self.service.ensure_usable()?;
        let mut all = BTreeMap::new();
        for (lease, access) in s
            .accesses
            .get()
            .ok_or_else(|| Error::Unsupported("bounded complete lowerer envelopes absent".into()))?
            .iter()
            .flat_map(|node| &node.actual)
            .cloned()
            .chain(self.accesses())
        {
            let entry = all.entry(lease.id()).or_insert((lease, access));
            if access == DeviceAccess::Write {
                entry.1 = access;
            }
        }
        let required = all.into_values().collect::<Vec<_>>();
        registration.require_value_storage_accesses(&required)?;
        let mut active = s
            .pool
            .active
            .try_lock()
            .map_err(|_| Error::State("bounded physical slots busy".into()))?;
        if active.upgrade().is_some() {
            return Err(Error::State(
                "bounded original completion still pending".into(),
            ));
        }
        if s.registered.swap(true, Ordering::AcqRel) {
            return Err(Error::State(
                "bounded actual record already registered".into(),
            ));
        }
        let run = Arc::new(ValueStorageRun { storage: s.clone() });
        *active = Arc::downgrade(&run);
        registration.retain_value_storage(run.clone())?;
        Ok(Some(run))
    }
}

#[cfg(test)]
mod finite_retention_tests {
    use super::*;
    #[test]
    fn compiler_context_hidden_writes_and_aliases_never_repack_immutable_groups() -> Result<()> {
        let dedicated = [true; 17];
        let groups = finite_code_groups(&dedicated)?;
        assert_eq!(groups.len(), 2);
        let owners = (100..117).collect::<Vec<_>>();
        let producers = (0..17).collect::<Vec<_>>();
        let declared = producers
            .iter()
            .copied()
            .zip(owners.iter().copied())
            .collect::<Vec<_>>();
        for mask in 0usize..256 {
            let mut writes = declared.clone();
            for (index, &owner) in owners.iter().enumerate().take(8) {
                if mask & (1 << index) == 0 {
                    writes.push((255, owner));
                }
            }
            let proof = retained_finite_candidates(&owners, &dedicated, &producers, &writes);
            assert_eq!(proof.len(), 9 + mask.count_ones() as usize);
            let active = groups
                .iter()
                .filter(|group| {
                    group
                        .iter()
                        .all(|id| proof.contains(&(id.0 as usize, id.0 as usize)))
                })
                .collect::<Vec<_>>();
            assert_eq!(active.len(), if mask == 255 { 2 } else { 1 });
            assert_eq!(active.last().unwrap().as_slice(), &groups[1]);
        }
        // A real owner alias across group boundaries rejects both groups;
        // ordinal/value identity alone cannot preserve either deferred launch.
        let mut aliased = owners.clone();
        aliased[8] = aliased[7];
        let proof = retained_finite_candidates(&aliased, &dedicated, &producers, &declared);
        assert!(groups.iter().all(|group| {
            group
                .iter()
                .any(|id| !proof.contains(&(id.0 as usize, id.0 as usize)))
        }));
        // Complete candidate storage fits the existing64KiB temporary admission,
        // including IDs/sizes, both vector tables and maximum source staging.
        let max = finite_code_groups(&[true; 512])?;
        assert_eq!(max.len(), 64);
        let temporary = 2 * 512 * std::mem::size_of::<ValueId>()
            + 2 * max.len() * std::mem::size_of::<Vec<ValueId>>()
            + 3 * 4096;
        assert!(temporary <= 64 * 1024);
        println!(
            "CUDA_READINESS_FINITE_GROUPS subsets=256 max_groups=64 temporary={temporary} admitted=65536"
        );
        Ok(())
    }

    #[test]
    fn only_unique_dedicated_exact_concrete_producers_may_defer() {
        let owners = [10, 20, 30];
        let producers = [0, 1, 2];
        let writes = [(0, 10), (1, 20), (2, 30), (2, 99)];
        assert_eq!(
            retained_finite_candidates(&owners, &[true; 3], &producers, &writes),
            vec![(0, 0), (1, 1), (2, 2)]
        );
        assert_eq!(
            retained_finite_candidates(&owners, &[true, false, true], &producers, &writes),
            vec![(0, 0), (2, 2)]
        );
        // Distinct logical epochs sharing one owner stay immediate, even when
        // the claimed source lifetimes themselves do not overlap.
        assert_eq!(
            retained_finite_candidates(&[10, 10, 30], &[true; 3], &producers, &writes),
            vec![(2, 2)]
        );
        let mut hidden = writes.to_vec();
        hidden.push((7, 20));
        assert_eq!(
            retained_finite_candidates(&owners, &[true; 3], &producers, &hidden),
            vec![(0, 0), (2, 2)]
        );
        assert_eq!(
            retained_finite_candidates(&owners, &[true; 3], &[0, 8, 2], &writes),
            vec![(0, 0), (2, 2)]
        );
        assert_eq!(
            retained_finite_candidates(&owners, &[true; 3], &producers, &writes[..1]),
            vec![(0, 0)]
        );
        let mut duplicate = writes.to_vec();
        duplicate.push((0, 10));
        assert_eq!(
            retained_finite_candidates(&owners, &[true; 3], &producers, &duplicate),
            vec![(1, 1), (2, 2)]
        );
    }
}

#[cfg(test)]
mod tests;
