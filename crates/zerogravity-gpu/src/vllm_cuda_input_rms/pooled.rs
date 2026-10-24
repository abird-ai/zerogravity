//! Opt-in short-HF private scratch. No native/math/current-root permission.
use super::{BoundModelRmsSource, Columns, Rows};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{FullModelProfile, PlanOperation, RecordedVllm08Model, Vllm08Kind},
    device_events::{BufferLeaseId, BufferRole, DeviceAccess},
};
use zerogravity_model_qwen35::original_binding::BoundShortHfCall;

#[path = "pooled/layout.rs"]
mod layout;
pub(super) use layout::ScratchRole;
use layout::{OrderedActors, declared_rows};
#[path = "pooled/workspace.rs"]
mod workspace;
pub(crate) use workspace::{RmsWorkspaceOutput, RmsWorkspaceViews};

struct Storage {
    source: BoundShortHfCall,
    service: Arc<ResidentService>,
    generation: u64,
    rows: usize,
    columns: Columns,
    leases: BTreeMap<ScratchRole, ResidentLease>,
    definitions: Arc<super::definitions::RmsDefinitions>,
    workspace: Option<Arc<RmsWorkspaceViews>>,
    active: Mutex<Weak<RmsScratchRun>>,
    epoch: AtomicU64,
    _metadata: Arc<WindowWorkspace>,
}

/// Per-actual-record producer seal over the same nine physical owners.
/// Actors/body retain this storage; only the armed flight owns a busy run.
pub(crate) struct FullModelRmsScratch {
    storage: Arc<Storage>,
    source: BoundShortHfCall,
    record: Arc<RecordedVllm08Model>,
    bindings: BTreeMap<usize, BoundModelRmsSource>,
    ordered_nodes: Arc<[usize]>,
    consumed: AtomicBool,
    _metadata: WindowWorkspace,
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("short-HF RMS pool cancelled".into()));
    }
    Ok(())
}
fn bindings(
    source: &BoundShortHfCall,
    record: &RecordedVllm08Model,
) -> Result<BTreeMap<usize, BoundModelRmsSource>> {
    source.call().validate()?;
    let call = source.call().as_vllm_call();
    if record.call() != call
        || record.plan().source() != call.source()
        || record.plan().nodes().len() != 255
        || record.plan().values().len() != 327
        || record.boundaries().len() != 24
        || record.plan().required_domains().len() != 42
        || source.source().model_profile() != call.model_profile()
        || source.source().source().owner().config().rms_norm_eps() != 1e-6
    {
        return Err(Error::State(
            "pooled RMS actual short-HF source/full record".into(),
        ));
    }
    let mut result = BTreeMap::new();
    for (index, node) in record.plan().nodes().iter().enumerate() {
        let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
            continue;
        };
        if descriptor.call() != call {
            return Err(Error::State("pooled RMS foreign call".into()));
        }
        if matches!(
            descriptor.kind(),
            Vllm08Kind::InputRms | Vllm08Kind::ResidualRms(_)
        ) {
            if (descriptor.kind() == Vllm08Kind::InputRms
                && (index != 1 || node.inputs.len() != 1 || node.outputs.len() != 1))
                || (descriptor.kind() != Vllm08Kind::InputRms
                    && (node.inputs.len() != 2 || node.outputs.len() != 2))
            {
                return Err(Error::State("pooled RMS original producer arity".into()));
            }
            for id in node.inputs.iter().chain(&node.outputs) {
                record
                    .plan()
                    .vllm_value_provenance(*id)?
                    .ok_or_else(|| Error::State("pooled RMS actual producer absent".into()))?
                    .require_invocation(call.request())?;
            }
            result.insert(
                index,
                BoundModelRmsSource::bind_short_hf(source, descriptor.as_ref().clone())?,
            );
        }
    }
    if result.len() != 49 || !result.contains_key(&1) || !result.contains_key(&253) {
        return Err(Error::State(
            "pooled RMS actual ordinary+48 dual sites".into(),
        ));
    }
    Ok(result)
}
impl FullModelRmsScratch {
    pub(super) fn row_spec(rows: usize, profile: FullModelProfile) -> Result<Rows> {
        declared_rows(rows, profile)
    }
    fn allocation_roles(
        rows: usize,
        profile: FullModelProfile,
    ) -> Result<Vec<(ScratchRole, usize)>> {
        declared_rows(rows, profile)?;
        let columns = profile.hidden_size();
        ScratchRole::ALL
            .into_iter()
            .map(|role| Ok((role, role.bytes(rows, columns)?)))
            .collect()
    }
    pub(crate) fn allocation_layout(rows: usize, profile: FullModelProfile) -> Result<Vec<usize>> {
        Ok(Self::allocation_roles(rows, profile)?
            .into_iter()
            .map(|(_, bytes)| bytes)
            .collect())
    }
    /// Allocate once before native activation, on the original initialization.
    pub(crate) fn allocate(
        source: BoundShortHfCall,
        record: Arc<RecordedVllm08Model>,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::allocate_inner(source, record, service, Some(init), None, cancelled)
    }
    pub(crate) fn bind_views(
        source: BoundShortHfCall,
        record: Arc<RecordedVllm08Model>,
        service: Arc<ResidentService>,
        workspace: Arc<RmsWorkspaceViews>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::allocate_inner(source, record, service, None, Some(workspace), cancelled)
    }
    fn allocate_inner(
        source: BoundShortHfCall,
        record: Arc<RecordedVllm08Model>,
        service: Arc<ResidentService>,
        mut init: Option<&mut ResidentInitialization<'_>>,
        workspace: Option<Arc<RmsWorkspaceViews>>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        if let Some(init) = &init {
            init.require_service(&service)?;
        }
        if init.is_none() && workspace.is_none() {
            return Err(Error::State(
                "RMS storage needs actual init or sealed views".into(),
            ));
        }
        source.call().validate()?;
        if let Some(views) = &workspace {
            views.require(&source, &record, &service)?;
        }
        let rows = source.call().span().len();
        // Refuse unsupported arithmetic before allocating a private device class.
        declared_rows(rows, source.call().model_profile())?;
        if service.source_identity() != source.source().source().source_identity() {
            return Err(Error::State("pooled RMS source/service issuer".into()));
        }
        let bindings = bindings(&source, &record)?;
        let columns = match source.call().model_profile() {
            FullModelProfile::Qwen08 => Columns::Vllm08,
            FullModelProfile::Qwen2B => Columns::Qwen2b,
        };
        let definitions = super::definitions::RmsDefinitions::prepare(
            source.source().source().source(),
            declared_rows(rows, source.call().model_profile())?,
            columns,
        )?;
        let layout = Self::allocation_roles(rows, source.call().model_profile())?;
        let bytes = layout.iter().try_fold(0usize, |n, (_, bytes)| {
            n.checked_add(*bytes)
                .ok_or_else(|| Error::Limit("pooled RMS total bytes".into()))
        })?;
        if workspace.is_none() && service.remaining_capacity()? < bytes {
            return Err(Error::Limit("pooled RMS capacity before allocation".into()));
        }
        let metadata = Arc::new(
            source
                .source()
                .source()
                .source()
                .reserve_workspace(64 * 1024)?,
        );
        if let Some(init) = &mut init {
            init.retain_completion_owner(metadata.clone())?;
            init.retain_completion_owner(source.source().source().clone())?;
            init.retain_completion_owner(record.clone())?;
            init.retain_completion_owner(Arc::new(source.clone()))?;
        }
        let mut leases = BTreeMap::new();
        for (index, (role, bytes)) in layout.into_iter().enumerate() {
            not_cancelled(cancelled)?;
            let lease = if let Some(views) = &workspace {
                views.scratch(index, bytes)?
            } else {
                init.as_mut()
                    .ok_or_else(|| Error::State("RMS actual allocation guard absent".into()))?
                    .allocate_empty(bytes, BufferRole::Scratch)?
            };
            leases.insert(role, lease);
        }
        let storage = Arc::new(Storage {
            generation: service.generation(),
            source: source.clone(),
            service,
            rows,
            columns,
            leases,
            definitions,
            workspace,
            active: Mutex::new(Weak::new()),
            epoch: AtomicU64::new(0),
            _metadata: metadata,
        });
        let scope = Arc::new(Self {
            storage,
            _metadata: source
                .source()
                .source()
                .source()
                .reserve_workspace(64 * 1024)?,
            source,
            record,
            ordered_nodes: bindings.keys().copied().collect(),
            bindings,
            consumed: AtomicBool::new(false),
        });
        if let Some(init) = &mut init {
            init.retain_completion_owner(scope.clone())?;
        }
        Ok(scope)
    }
    /// Fresh actual record only. The original current-attempt/cursor still owns execution.
    pub(crate) fn bind_call(
        self: &Arc<Self>,
        source: BoundShortHfCall,
        record: Arc<RecordedVllm08Model>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        self.storage.service.ensure_usable()?;
        let bindings = bindings(&source, &record)?;
        declared_rows(source.call().span().len(), source.call().model_profile())?;
        if !source
            .source()
            .source()
            .source()
            .same_issuer(self.storage.source.source().source().source())
            || source.call().model_profile() != self.storage.source.call().model_profile()
            || source.call().span().len() != self.storage.rows
            || Arc::ptr_eq(&record, &self.record)
            || bindings
                .keys()
                .copied()
                .ne(self.ordered_nodes.iter().copied())
        {
            return Err(Error::State(
                "pooled RMS fresh source/record/geometry issuer".into(),
            ));
        }
        Ok(Arc::new(Self {
            storage: self.storage.clone(),
            source: source.clone(),
            record,
            bindings,
            ordered_nodes: self.ordered_nodes.clone(),
            consumed: AtomicBool::new(false),
            _metadata: source
                .source()
                .source()
                .source()
                .reserve_workspace(64 * 1024)?,
        }))
    }
    pub(crate) fn retention_witness(self: &Arc<Self>) -> Result<RmsScratchWitness> {
        let run = self
            .storage
            .active
            .try_lock()
            .map_err(|_| Error::State("pooled RMS witness busy".into()))?
            .clone();
        Ok(RmsScratchWitness {
            storage: Arc::downgrade(&self.storage),
            scope: Arc::downgrade(self),
            run,
        })
    }
    pub(crate) fn ordinary_binding(&self) -> Result<BoundModelRmsSource> {
        Ok(self.binding(1)?.clone())
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.storage.service
    }
    pub(super) fn rows(&self) -> Result<Rows> {
        declared_rows(self.storage.rows, self.source.call().model_profile())
    }
    pub(crate) fn readiness_temporary_host_bytes(cohort: bool) -> Result<usize> {
        super::definitions::RmsDefinitions::readiness_temporary_host_bytes(cohort)
    }
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
        let definitions = self.definitions();
        let mut count = 0;
        for kernel in definitions.kernels() {
            kernel.prepare_code(queue)?;
            count += 1;
        }
        Ok(count)
    }
    pub(super) fn definitions(&self) -> Arc<super::definitions::RmsDefinitions> {
        self.storage.definitions.clone()
    }
    pub(super) fn columns(&self) -> Columns {
        self.storage.columns
    }
    pub(super) fn source(&self) -> &BoundShortHfCall {
        &self.source
    }
    pub(super) fn binding(&self, node: usize) -> Result<&BoundModelRmsSource> {
        self.bindings
            .get(&node)
            .ok_or_else(|| Error::State("pooled RMS actual source site absent".into()))
    }
    pub(super) fn actor(self: &Arc<Self>, node: usize) -> Result<Arc<PooledRmsActor>> {
        self.binding(node)?;
        Ok(Arc::new(PooledRmsActor {
            scope: self.clone(),
            node,
            consumed: AtomicBool::new(false),
        }))
    }
    /// Unique real lease union, never an accounting-only reduction.
    pub(crate) fn merge_accesses(
        &self,
        accesses: impl IntoIterator<Item = (ResidentLease, DeviceAccess)>,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.service().ensure_usable()?;
        let mut result: BTreeMap<BufferLeaseId, (ResidentLease, DeviceAccess)> = BTreeMap::new();
        for (lease, access) in accesses {
            if !Arc::ptr_eq(lease.service(), self.service())
                || lease.service_generation() != self.storage.generation
                || lease.source_identity() != self.service().source_identity()
            {
                return Err(Error::State(
                    "pooled RMS access source/service/generation".into(),
                ));
            }
            if let Some((old, prior)) = result.get_mut(&lease.id()) {
                if old.allocation_generation() != lease.allocation_generation()
                    || old.bytes() != lease.bytes()
                    || old.role() != lease.role()
                {
                    return Err(Error::State(
                        "pooled RMS reused unrelated lease identity".into(),
                    ));
                }
                if access == DeviceAccess::Write {
                    *prior = access;
                }
            } else {
                result.insert(lease.id(), (lease, access));
            }
        }
        Ok(result.into_values().collect())
    }
    pub(crate) fn require_submitted(
        &self,
        service: &Arc<ResidentService>,
        flight: &crate::resident_leases::ResidentSubmission<'_>,
    ) -> Result<()> {
        service.ensure_usable()?;
        if !Arc::ptr_eq(service, self.service()) || service.generation() != self.storage.generation
        {
            return Err(Error::State(
                "pooled RMS original armed service differs".into(),
            ));
        }
        flight.require_accesses(
            self.storage
                .leases
                .values()
                .map(|lease| (lease, DeviceAccess::Write)),
        )
    }
    pub(crate) fn require_active_run(&self) -> Result<()> {
        let run = self
            .storage
            .active
            .try_lock()
            .map_err(|_| Error::State("pooled RMS active source run busy".into()))?
            .upgrade()
            .ok_or_else(|| {
                Error::State("pooled RMS source actor lacks original registered flight".into())
            })?;
        if !std::ptr::eq(run.scope.as_ref(), self)
            || run.epoch != self.storage.epoch.load(Ordering::Acquire)
        {
            return Err(Error::State(
                "pooled RMS source actor has stale/foreign record epoch".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn start_run(
        self: &Arc<Self>,
        registration: &mut crate::resident_inference::RmsScratchRegistration<'_, '_>,
    ) -> Result<Arc<RmsScratchRun>> {
        // Only the original workspace adapter can mint this registration.
        registration.require_scope(self)?;
        // Fenced/quarantined services refuse before a nonblocking busy claim.
        self.service().ensure_usable()?;
        let mut active = self
            .storage
            .active
            .try_lock()
            .map_err(|_| Error::State("pooled RMS active claim busy".into()))?;
        if active.upgrade().is_some() {
            return Err(Error::State(
                "pooled RMS original flight still owns workspace".into(),
            ));
        }
        if self.consumed.swap(true, Ordering::AcqRel) {
            return Err(Error::State("pooled RMS record executes once".into()));
        }
        let epoch = self
            .storage
            .epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| Error::Limit("pooled RMS epoch exhausted".into()))?
            + 1;
        let run = Arc::new(RmsScratchRun {
            scope: self.clone(),
            epoch,
            cursor: Mutex::new(OrderedActors::default()),
        });
        *active = Arc::downgrade(&run);
        registration.retain(run.clone())?;
        Ok(run)
    }
}

pub(crate) struct RmsScratchRun {
    scope: Arc<FullModelRmsScratch>,
    epoch: u64,
    cursor: Mutex<OrderedActors>,
}
impl RmsScratchRun {
    pub(crate) fn require_finished(&self) -> Result<()> {
        self.cursor
            .try_lock()
            .map_err(|_| Error::State("pooled RMS actor still borrowed".into()))?
            .require_finished(self.scope.bindings.len())
    }
}

/// Prepared actors hold storage/producer seals, never strong busy-run authority.
pub(super) struct PooledRmsActor {
    scope: Arc<FullModelRmsScratch>,
    node: usize,
    consumed: AtomicBool,
}
impl PooledRmsActor {
    pub(super) fn definitions(&self) -> Arc<super::definitions::RmsDefinitions> {
        self.scope.definitions()
    }
    pub(super) fn has_workspace_outputs(&self) -> bool {
        self.scope.storage.workspace.is_some()
    }
    pub(super) fn workspace_output(
        &self,
        role: RmsWorkspaceOutput,
        bytes: usize,
    ) -> Result<Option<ResidentLease>> {
        self.scope
            .storage
            .workspace
            .as_ref()
            .map(|views| views.output(self.node, role, bytes))
            .transpose()
    }
    pub(super) fn lease(&self, role: ScratchRole, bytes: usize) -> Result<ResidentLease> {
        let lease = self
            .scope
            .storage
            .leases
            .get(&role)
            .ok_or_else(|| Error::State("pooled RMS scratch role absent".into()))?;
        if lease.bytes() != bytes || lease.role() != BufferRole::Scratch {
            return Err(Error::State("pooled RMS exact private extent/role".into()));
        }
        Ok(lease.clone())
    }
    pub(super) fn require_storage<'a>(
        &self,
        leases: impl IntoIterator<Item = &'a ResidentLease>,
    ) -> Result<()> {
        let leases = leases.into_iter().collect::<Vec<_>>();
        layout::require_distinct(leases.iter().map(|lease| lease.id()))?;
        for lease in leases {
            if !Arc::ptr_eq(lease.service(), self.scope.service())
                || lease.service_generation() != self.scope.storage.generation
                || lease.source_identity() != self.scope.service().source_identity()
            {
                return Err(Error::State(
                    "pooled RMS actor foreign input/output storage".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn require_scope(
        &self,
        scope: &Arc<FullModelRmsScratch>,
        node: usize,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.scope, scope) || self.node != node {
            return Err(Error::State(
                "pooled RMS actual actor/source/record differs".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn rebind(
        &self,
        scope: &Arc<FullModelRmsScratch>,
        node: usize,
    ) -> Result<Arc<Self>> {
        let old = self.scope.binding(self.node)?.operator();
        let new = scope.binding(node)?.operator();
        if self.node != node
            || !Arc::ptr_eq(&self.scope.storage, &scope.storage)
            || old.kind() != new.kind()
            || old.layer() != new.layer()
            || old.parameters() != new.parameters()
            || old.call().model_profile() != new.call().model_profile()
            || old.call().rows() != new.call().rows()
        {
            return Err(Error::State(
                "pooled RMS fresh actual record/site/shared physical storage differs".into(),
            ));
        }
        scope.actor(node)
    }
    pub(super) fn begin(&self) -> Result<ActorBorrow<'_>> {
        self.scope.service().ensure_usable()?;
        let run = self
            .scope
            .storage
            .active
            .try_lock()
            .map_err(|_| Error::State("pooled RMS active claim busy".into()))?
            .upgrade()
            .ok_or_else(|| {
                Error::State("pooled RMS lacks original workspace/armed flight".into())
            })?;
        if !Arc::ptr_eq(&run.scope, &self.scope)
            || run.epoch != self.scope.storage.epoch.load(Ordering::Acquire)
        {
            return Err(Error::State("pooled RMS stale actor/run epoch".into()));
        }
        // No guard borrows a temporary Arc: the run is kept by original flight.
        if self.consumed.swap(true, Ordering::AcqRel) {
            return Err(Error::State("pooled RMS actor executes once".into()));
        }
        let mut cursor = run
            .cursor
            .try_lock()
            .map_err(|_| Error::State("pooled RMS concurrent actor refused".into()))?;
        cursor.begin(&self.scope.ordered_nodes, self.node)?;
        drop(cursor);
        Ok(ActorBorrow {
            actor: self,
            run,
            completed: false,
        })
    }
}

pub(super) struct ActorBorrow<'a> {
    actor: &'a PooledRmsActor,
    run: Arc<RmsScratchRun>,
    completed: bool,
}
impl ActorBorrow<'_> {
    pub(super) fn complete(mut self) -> Result<()> {
        let mut cursor = self
            .run
            .cursor
            .try_lock()
            .map_err(|_| Error::State("pooled RMS completion borrow busy".into()))?;
        cursor.require_current(&self.actor.scope.ordered_nodes, self.actor.node)?;
        cursor.completed();
        self.completed = true;
        Ok(())
    }
}
impl Drop for ActorBorrow<'_> {
    fn drop(&mut self) {
        if !self.completed
            && let Ok(mut cursor) = self.run.cursor.lock()
        {
            cursor.fail();
        }
    }
}

/// Pure Weak inspection; no service-usability query, enqueue or release authority.
pub(crate) struct RmsScratchWitness {
    storage: Weak<Storage>,
    scope: Weak<FullModelRmsScratch>,
    run: Weak<RmsScratchRun>,
}
pub(crate) struct RmsScratchRetention {
    pub(crate) run_retained: bool,
    pub(crate) epoch: u64,
    pub(crate) private_ids: [BufferLeaseId; 9],
    pub(crate) private_bytes: usize,
    pub(crate) private_initialized: [bool; 9],
    pub(crate) service_generation: u64,
}
impl RmsScratchWitness {
    pub(crate) fn inspect(&self) -> Result<RmsScratchRetention> {
        let storage = self
            .storage
            .upgrade()
            .ok_or_else(|| Error::State("lost pooled RMS physical storage".into()))?;
        let scope = self
            .scope
            .upgrade()
            .ok_or_else(|| Error::State("lost pooled RMS source/record scope".into()))?;
        if !Arc::ptr_eq(&scope.storage, &storage)
            || storage.generation != storage.service.generation()
            || storage.service.source_identity() != scope.record.call().source()
            || !scope
                .source
                .source()
                .source()
                .source()
                .same_issuer(storage.source.source().source().source())
        {
            return Err(Error::State(
                "retained RMS physical/source/service/record differs".into(),
            ));
        }
        let run = self.run.upgrade();
        if run.as_ref().is_some_and(|run| {
            !Arc::ptr_eq(&run.scope, &scope) || run.epoch != storage.epoch.load(Ordering::Acquire)
        }) {
            return Err(Error::State(
                "retained RMS run belongs to another original record".into(),
            ));
        }
        let mut ids = std::collections::BTreeSet::new();
        let mut bytes = 0usize;
        for role in ScratchRole::ALL {
            let lease = &storage.leases[&role];
            if !ids.insert(lease.id())
                || lease.role() != BufferRole::Scratch
                || lease.bytes() != role.bytes(storage.rows, storage.columns.count())?
                || !Arc::ptr_eq(lease.service(), &storage.service)
                || lease.service_generation() != storage.generation
            {
                return Err(Error::State(
                    "retained RMS nine genuine private owners differ".into(),
                ));
            }
            bytes = bytes
                .checked_add(lease.bytes())
                .ok_or_else(|| Error::Limit("retained RMS bytes".into()))?;
        }
        Ok(RmsScratchRetention {
            run_retained: run.is_some(),
            epoch: storage.epoch.load(Ordering::Acquire),
            private_ids: ScratchRole::ALL.map(|role| storage.leases[&role].id()),
            private_bytes: bytes,
            private_initialized: ScratchRole::ALL
                .map(|role| storage.leases[&role].is_initialized()),
            service_generation: storage.generation,
        })
    }
}

#[cfg(test)]
#[path = "pooled/tests.rs"]
mod tests;
