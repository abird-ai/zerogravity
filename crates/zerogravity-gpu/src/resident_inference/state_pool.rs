//! Shared .8B/2B root checkout: fresh logical leases, unchanged domain binding.
use super::*;
use crate::resident_leases::backing_pool::ResidentPool;
use crate::resident_leases::backing_pool::{ResidentPagePool, ResidentPlaneLayout};
#[cfg(feature = "cuda-cublas")]
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda-cublas")]
use zerogravity_core::device_events::BufferLeaseId;

#[derive(Clone, Debug)]
#[allow(
    dead_code,
    reason = "hybrid inventory is consumed by the next ordinary paged-session selection slice"
)]
pub(super) struct HybridStateInventoryPlan {
    pub vector_layout: Vec<ResidentPlaneLayout>,
    pub staging_layout: Vec<ResidentPlaneLayout>,
    pub page: ResidentPlaneLayout,
    pub roots: usize,
    pub staging_sets: usize,
    pub page_tokens: usize,
    pub max_append_tokens: usize,
    pub max_pages: usize,
    pub pages_per_committed_root: usize,
    pub candidate_pages_per_active_session: usize,
    pub root_budget_bytes: usize,
    pub vector_max_bytes: usize,
    pub staging_max_bytes: usize,
    pub page_max_bytes: usize,
    pub slack_bytes: usize,
}

#[allow(
    dead_code,
    reason = "hybrid inventory is consumed by the next ordinary paged-session selection slice"
)]
impl HybridStateInventoryPlan {
    pub(super) fn analyze(
        states: &[StateRequirement],
        state_bytes: usize,
        roots: usize,
        staging_sets: usize,
        page_tokens: usize,
        max_append_tokens: usize,
        alignment: usize,
    ) -> Result<Self> {
        if states.is_empty()
            || state_bytes == 0
            || roots < 2
            || staging_sets == 0
            || staging_sets >= roots
            || page_tokens == 0
            || max_append_tokens == 0
            || !alignment.is_power_of_two()
        {
            return Err(Error::Limit("resident hybrid inventory policy".into()));
        }
        let mut vector_layout = Vec::new();
        let mut staging_layout = Vec::new();
        let mut vector_bytes = 0usize;
        let mut kv_bytes = 0usize;
        let mut page = None::<ResidentPlaneLayout>;
        let mut kv_domains = 0usize;
        let mut pages_per_committed_root = 0usize;
        for requirement in states {
            let (planes, physical) = requirement.plane_layout()?;
            let plane_bytes = physical.bytes()?;
            match requirement.layout {
                StateLayout::Vector { .. } => {
                    if planes != 1 || !plane_bytes.is_multiple_of(alignment) {
                        return Err(Error::State(
                            "resident hybrid vector plane geometry differs".into(),
                        ));
                    }
                    vector_bytes = vector_bytes
                        .checked_add(plane_bytes)
                        .ok_or_else(|| Error::Limit("resident hybrid vector bytes".into()))?;
                    vector_layout.push(ResidentPlaneLayout {
                        bytes: plane_bytes,
                        role: BufferRole::RecurrentState,
                        alignment,
                    });
                }
                StateLayout::Kv { slab, capacity } => {
                    if planes != 2 || page_tokens > capacity || max_append_tokens > capacity {
                        return Err(Error::State(
                            "resident hybrid KV plane geometry differs".into(),
                        ));
                    }
                    let row_bytes = slab
                        .checked_mul(requirement.physical.byte_width())
                        .ok_or_else(|| Error::Limit("resident hybrid KV row bytes".into()))?;
                    let page_bytes = row_bytes
                        .checked_mul(page_tokens)
                        .ok_or_else(|| Error::Limit("resident hybrid KV page bytes".into()))?;
                    if row_bytes == 0
                        || page_bytes == 0
                        || !page_bytes.is_multiple_of(alignment)
                        || row_bytes.checked_mul(capacity) != Some(plane_bytes)
                    {
                        return Err(Error::State(
                            "resident hybrid KV row/page geometry differs".into(),
                        ));
                    }
                    let candidate = ResidentPlaneLayout {
                        bytes: page_bytes,
                        role: BufferRole::RecurrentState,
                        alignment,
                    };
                    if page.is_some_and(|existing| existing != candidate) {
                        return Err(Error::Unsupported(
                            "resident hybrid heterogeneous KV page geometry".into(),
                        ));
                    }
                    page = Some(candidate);
                    kv_domains = kv_domains
                        .checked_add(1)
                        .ok_or_else(|| Error::Limit("resident hybrid KV domain count".into()))?;
                    let pages_per_plane = capacity.div_ceil(page_tokens);
                    pages_per_committed_root = pages_per_committed_root
                        .checked_add(pages_per_plane.checked_mul(2).ok_or_else(|| {
                            Error::Limit("resident hybrid KV root page count".into())
                        })?)
                        .ok_or_else(|| Error::Limit("resident hybrid root page count".into()))?;
                    kv_bytes = kv_bytes
                        .checked_add(requirement.bytes)
                        .ok_or_else(|| Error::Limit("resident hybrid KV bytes".into()))?;
                    staging_layout.extend((0..2).map(|_| ResidentPlaneLayout {
                        bytes: plane_bytes,
                        role: BufferRole::RecurrentState,
                        alignment,
                    }));
                }
            }
        }
        let page = page.ok_or_else(|| Error::State("resident hybrid KV state absent".into()))?;
        if vector_layout.is_empty() || staging_layout.is_empty() {
            return Err(Error::State(
                "resident hybrid requires vector and KV state".into(),
            ));
        }
        let accounted = vector_bytes
            .checked_add(kv_bytes)
            .ok_or_else(|| Error::Limit("resident hybrid state bytes".into()))?;
        if accounted != state_bytes {
            return Err(Error::State(
                "resident hybrid state requirement byte sum differs".into(),
            ));
        }
        let root_budget_bytes = state_bytes
            .checked_mul(roots)
            .ok_or_else(|| Error::Limit("resident hybrid root budget".into()))?;
        let vector_max_bytes = vector_bytes
            .checked_mul(roots)
            .ok_or_else(|| Error::Limit("resident hybrid vector maximum".into()))?;
        let staging_max_bytes = kv_bytes
            .checked_mul(staging_sets)
            .ok_or_else(|| Error::Limit("resident hybrid staging maximum".into()))?;
        let page_budget = root_budget_bytes
            .checked_sub(vector_max_bytes)
            .and_then(|n| n.checked_sub(staging_max_bytes))
            .ok_or_else(|| Error::Limit("resident hybrid page budget".into()))?;
        let envelope_pages = page_budget / page.bytes;
        let candidate_pages_per_domain = (page_tokens - 1)
            .checked_add(max_append_tokens)
            .ok_or_else(|| Error::Limit("resident hybrid append span".into()))?
            .div_ceil(page_tokens)
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("resident hybrid candidate pages/domain".into()))?;
        let candidate_pages_per_active_session = kv_domains
            .checked_mul(candidate_pages_per_domain)
            .ok_or_else(|| Error::Limit("resident hybrid candidate pages/session".into()))?;
        // `staging_sets` is the maximum number of simultaneously active current
        // roots. Snapshots, displaced roots and unpublished candidates share their
        // immutable committed prefix pages; an additional logical root can consume
        // only its divergent COW tail/appended pages. Preserve every logical root
        // slot from the original `2*max_sessions+2` lifecycle envelope without
        // preallocating a second full KV slab for each shared-prefix owner.
        let committed_pages = pages_per_committed_root
            .checked_mul(staging_sets)
            .ok_or_else(|| Error::Limit("resident hybrid committed page bound".into()))?;
        let divergent_slots = roots
            .checked_sub(staging_sets)
            .ok_or_else(|| Error::State("resident hybrid divergent slot underflow".into()))?;
        let reachable_pages = candidate_pages_per_active_session
            .checked_mul(divergent_slots)
            .and_then(|n| n.checked_add(committed_pages))
            .ok_or_else(|| Error::Limit("resident hybrid reachable page bound".into()))?;
        let max_pages = envelope_pages.min(reachable_pages);
        let page_max_bytes = max_pages
            .checked_mul(page.bytes)
            .ok_or_else(|| Error::Limit("resident hybrid page maximum".into()))?;
        let minimum_active_pages = pages_per_committed_root
            .checked_add(candidate_pages_per_active_session)
            .and_then(|n| n.checked_mul(staging_sets))
            .ok_or_else(|| Error::Limit("resident hybrid active page bound".into()))?;
        if max_pages < minimum_active_pages {
            return Err(Error::Limit(
                "resident hybrid page inventory cannot fork every active session".into(),
            ));
        }
        let used = vector_max_bytes
            .checked_add(staging_max_bytes)
            .and_then(|n| n.checked_add(page_max_bytes))
            .ok_or_else(|| Error::Limit("resident hybrid admitted maximum".into()))?;
        let slack_bytes = root_budget_bytes
            .checked_sub(used)
            .ok_or_else(|| Error::State("resident hybrid exceeded root budget".into()))?;
        Ok(Self {
            vector_layout,
            staging_layout,
            page,
            roots,
            staging_sets,
            page_tokens,
            max_append_tokens,
            max_pages,
            pages_per_committed_root,
            candidate_pages_per_active_session,
            root_budget_bytes,
            vector_max_bytes,
            staging_max_bytes,
            page_max_bytes,
            slack_bytes,
        })
    }

    pub(super) fn resident_bytes(&self) -> Result<usize> {
        self.vector_max_bytes
            .checked_add(self.staging_max_bytes)
            .and_then(|n| n.checked_add(self.page_max_bytes))
            .ok_or_else(|| Error::Limit("resident hybrid admitted byte sum".into()))
    }
}

#[cfg(feature = "cuda-cublas")]
pub(super) fn hybrid_state_reserve_bytes(
    states: &[StateRequirement],
    state_bytes: usize,
    roots: usize,
    staging_sets: usize,
    page_tokens: usize,
    max_append_tokens: usize,
) -> Result<usize> {
    // Alignment changes backing validity/metadata, not any state byte extent.
    // Physical installation repeats this analysis with the actual allocator
    // alignment before creating a backing.
    let plan = HybridStateInventoryPlan::analyze(
        states,
        state_bytes,
        roots,
        staging_sets,
        page_tokens,
        max_append_tokens,
        1,
    )?;
    ResidentPool::metadata_bytes(plan.vector_layout.len(), plan.roots)?;
    ResidentPool::metadata_bytes(plan.staging_layout.len(), plan.staging_sets)?;
    ResidentPagePool::metadata_bytes(plan.max_pages)?;
    plan.resident_bytes()
}

#[allow(
    dead_code,
    reason = "hybrid storage is installed by the next ordinary source-call selection slice"
)]
pub(super) enum PooledStateStorage {
    Contiguous(Arc<ResidentPool>),
    Hybrid(Arc<HybridPooledStates>),
}

pub(super) struct PooledStates {
    pub storage: PooledStateStorage,
    #[cfg(all(
        any(
            feature = "qualification",
            feature = "hip-source",
            feature = "cuda-cublas"
        ),
        target_os = "linux"
    ))]
    pub source: Option<zerogravity_artifacts::source_weights::VerifiedTensorSource>,
}

#[allow(
    dead_code,
    reason = "hybrid storage is installed by the next ordinary source-call selection slice"
)]
pub(super) struct HybridPooledStates {
    pub plan: HybridStateInventoryPlan,
    pub vector_pool: Arc<ResidentPool>,
    pub staging_pool: Arc<ResidentPool>,
    pub page_pool: Arc<ResidentPagePool>,
    pub numerical: zerogravity_core::accelerator::program::SourceNumericalContract,
}

#[cfg(feature = "cuda-cublas")]
enum HybridCandidatePart {
    Vector {
        prior: Domain,
        candidate: Domain,
    },
    Kv {
        prior: paged_state::ResidentPagedKvDomain,
        staging: paged_state::ResidentPagedKvStaging,
        append: paged_state::ResidentPagedKvAppend,
    },
}

#[cfg(feature = "cuda-cublas")]
pub(crate) struct HybridCandidateBundle {
    parts: Vec<HybridCandidatePart>,
    bindings: Vec<StateBinding>,
}

#[cfg(feature = "cuda-cublas")]
impl HybridCandidateBundle {
    pub(crate) fn bindings(&self) -> &[StateBinding] {
        &self.bindings
    }

    pub(crate) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut unique = BTreeMap::<BufferLeaseId, (ResidentLease, DeviceAccess)>::new();
        for part in &self.parts {
            match part {
                HybridCandidatePart::Vector { prior, candidate } => {
                    for lease in &prior.leases {
                        merge_access(&mut unique, lease.clone(), DeviceAccess::Read)?;
                    }
                    for lease in &candidate.leases {
                        merge_access(&mut unique, lease.clone(), DeviceAccess::Write)?;
                    }
                }
                HybridCandidatePart::Kv {
                    prior,
                    staging,
                    append,
                } => {
                    for (lease, access) in
                        prior.gather_accesses(prior.tokens()?, staging.keys(), staging.values())?
                    {
                        merge_access(&mut unique, lease, access)?;
                    }
                    for (lease, access) in append.accesses()? {
                        merge_access(&mut unique, lease, access)?;
                    }
                }
            }
        }
        Ok(unique.into_values().collect())
    }

    pub(crate) fn write_outputs(&self) -> Result<Vec<ResidentLease>> {
        let mut outputs = Vec::new();
        for part in &self.parts {
            match part {
                HybridCandidatePart::Vector { candidate, .. } => {
                    outputs.extend(candidate.leases.iter().cloned());
                }
                HybridCandidatePart::Kv { append, .. } => {
                    outputs.extend(append.outputs()?);
                }
            }
        }
        Ok(outputs)
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn enqueue_gather_cuda(
        &self,
        flight: &crate::resident_leases::ResidentSubmission<'_>,
    ) -> Result<usize> {
        let mut dispatches = 0usize;
        for part in &self.parts {
            if let HybridCandidatePart::Kv { prior, staging, .. } = part {
                dispatches = dispatches
                    .checked_add(prior.enqueue_gather_cuda(
                        flight,
                        prior.tokens()?,
                        staging.keys(),
                        staging.values(),
                    )?)
                    .ok_or_else(|| Error::Limit("resident hybrid gather dispatches".into()))?;
            }
        }
        Ok(dispatches)
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn enqueue_scatter_cuda(
        &self,
        flight: &crate::resident_leases::ResidentSubmission<'_>,
    ) -> Result<usize> {
        let mut dispatches = 0usize;
        for part in &self.parts {
            if let HybridCandidatePart::Kv {
                prior,
                staging,
                append,
            } = part
            {
                let materialized =
                    append.enqueue_cuda_from_staging(flight, staging, prior.tokens()?)?;
                dispatches = dispatches
                    .checked_add(materialized.zero_dispatches)
                    .and_then(|n| n.checked_add(materialized.copy_dispatches))
                    .and_then(|n| n.checked_add(materialized.append_transfers))
                    .ok_or_else(|| Error::Limit("resident hybrid scatter dispatches".into()))?;
            }
        }
        Ok(dispatches)
    }

    pub(crate) fn into_domains(self) -> Result<Vec<Domain>> {
        let mut domains = Vec::with_capacity(self.parts.len());
        for part in self.parts {
            match part {
                HybridCandidatePart::Vector { candidate, .. } => domains.push(candidate),
                HybridCandidatePart::Kv {
                    prior,
                    append,
                    staging: _,
                } => domains.push(Domain::from_paged_kv(append.into_candidate(&prior)?)?),
            }
        }
        Ok(domains)
    }
}

#[cfg(feature = "cuda-cublas")]
fn merge_access(
    unique: &mut BTreeMap<BufferLeaseId, (ResidentLease, DeviceAccess)>,
    lease: ResidentLease,
    access: DeviceAccess,
) -> Result<()> {
    match unique.entry(lease.id()) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert((lease, access));
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            let (existing, existing_access) = entry.get_mut();
            if existing.allocation_generation() != lease.allocation_generation()
                || existing.bytes() != lease.bytes()
                || existing.role() != lease.role()
                || !Arc::ptr_eq(existing.service(), lease.service())
            {
                return Err(Error::State(
                    "resident hybrid access lease identity aliases".into(),
                ));
            }
            if access == DeviceAccess::Write {
                *existing_access = DeviceAccess::Write;
            }
        }
    }
    Ok(())
}

#[allow(
    dead_code,
    reason = "hybrid storage is installed by the next ordinary source-call selection slice"
)]
impl HybridPooledStates {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        plan: HybridStateInventoryPlan,
        vector_metadata: zerogravity_core::resource_arbiter::HostWorkspace,
        staging_metadata: zerogravity_core::resource_arbiter::HostWorkspace,
        page_metadata: zerogravity_core::resource_arbiter::HostWorkspace,
        numerical: zerogravity_core::accelerator::program::SourceNumericalContract,
    ) -> Result<Arc<Self>> {
        let vector_pool = ResidentPool::prepare(
            service.clone(),
            vector_metadata,
            &plan.vector_layout,
            plan.roots,
        )?;
        let staging_pool = ResidentPool::prepare(
            service.clone(),
            staging_metadata,
            &plan.staging_layout,
            plan.staging_sets,
        )?;
        let page_pool =
            ResidentPagePool::prepare(service, page_metadata, plan.page, plan.max_pages)?;
        Ok(Arc::new(Self {
            plan,
            vector_pool,
            staging_pool,
            page_pool,
            numerical,
        }))
    }

    #[cfg(feature = "cuda-cublas")]
    pub(crate) fn candidate(
        &self,
        states: &[StateRequirement],
        prior: &[Domain],
        tokens: usize,
        position: usize,
    ) -> Result<HybridCandidateBundle> {
        if tokens > self.plan.max_append_tokens {
            return Err(Error::Limit(
                "resident hybrid append exceeds admitted token bound".into(),
            ));
        }
        if tokens == 0 || states.len() != prior.len() {
            return Err(Error::State(
                "resident hybrid candidate state/token coverage differs".into(),
            ));
        }
        let vector_leases = self.vector_pool.checkout()?;
        let staging_leases = self.staging_pool.checkout()?;
        let mut vectors = vector_leases.iter();
        let mut staging = staging_leases.iter();
        let mut parts = Vec::with_capacity(states.len());
        let mut bindings = Vec::with_capacity(states.len());

        for (requirement, current) in states.iter().zip(prior) {
            if requirement.slot != current.slot {
                return Err(Error::State(
                    "resident hybrid candidate slot ordering differs".into(),
                ));
            }
            match requirement.layout {
                StateLayout::Vector { .. } => {
                    if current.paged_kv.is_some() {
                        return Err(Error::State(
                            "resident hybrid vector current root is paged".into(),
                        ));
                    }
                    let candidate = make_domain(requirement, &mut |layout| {
                        let lease = vectors.next().ok_or_else(|| {
                            Error::State("resident hybrid candidate vector plane missing".into())
                        })?;
                        if lease.bytes() != layout.bytes()?
                            || lease.role() != BufferRole::RecurrentState
                            || lease.is_initialized()
                        {
                            return Err(Error::State(
                                "resident hybrid candidate vector plane differs".into(),
                            ));
                        }
                        Ok(lease.clone())
                    })?;
                    bindings.push(StateBinding {
                        slot: requirement.slot,
                        prior: current.buffers.clone(),
                        candidate: candidate.buffers.clone(),
                        prior_tokens: position,
                        mode: crate::resident_program::StateBindingMode::Distinct,
                    });
                    parts.push(HybridCandidatePart::Vector {
                        prior: current.clone(),
                        candidate,
                    });
                }
                StateLayout::Kv { .. } => {
                    let current = current.paged_kv()?.clone();
                    if current.tokens()? != position {
                        return Err(Error::State(
                            "resident hybrid candidate KV frontier differs".into(),
                        ));
                    }
                    let key = staging.next().ok_or_else(|| {
                        Error::State("resident hybrid candidate key staging missing".into())
                    })?;
                    let value = staging.next().ok_or_else(|| {
                        Error::State("resident hybrid candidate value staging missing".into())
                    })?;
                    let staged =
                        current.staging(key.clone(), value.clone(), DeviceAccess::Write)?;
                    let buffers = staged.buffers();
                    bindings.push(StateBinding {
                        slot: requirement.slot,
                        prior: buffers.clone(),
                        candidate: buffers,
                        prior_tokens: position,
                        mode: crate::resident_program::StateBindingMode::KvInPlace,
                    });
                    let append = current.fork_for_append(tokens)?;
                    parts.push(HybridCandidatePart::Kv {
                        prior: current,
                        staging: staged,
                        append,
                    });
                }
            }
        }
        if vectors.next().is_some() || staging.next().is_some() {
            return Err(Error::State(
                "resident hybrid candidate pool has extra planes".into(),
            ));
        }
        Ok(HybridCandidateBundle { parts, bindings })
    }
}

pub(crate) async fn allocate_hybrid_initial_state<F>(
    model: &ResidentModel,
    pools: &Arc<HybridPooledStates>,
    cancelled: Option<&AtomicBool>,
    retain: F,
) -> Result<Vec<Domain>>
where
    F: FnOnce(&mut crate::resident_leases::ResidentInitialization<'_>) -> Result<()>,
{
    check_cancel(cancelled)?;
    model.service.ensure_usable()?;
    let vector_leases = pools.vector_pool.checkout()?;
    let mut vectors = vector_leases.iter();
    let mut domains = Vec::with_capacity(model.states.len());
    for requirement in &model.states {
        match requirement.layout {
            StateLayout::Vector { .. } => {
                domains.push(make_domain(requirement, &mut |layout| {
                    let lease = vectors.next().ok_or_else(|| {
                        Error::State("resident hybrid vector root plane missing".into())
                    })?;
                    if layout.bytes()? != lease.bytes()
                        || lease.role() != BufferRole::RecurrentState
                        || lease.is_initialized()
                        || !Arc::ptr_eq(lease.service(), &model.service)
                    {
                        return Err(Error::State(
                            "resident hybrid vector root plane differs".into(),
                        ));
                    }
                    Ok(lease.clone())
                })?);
            }
            StateLayout::Kv { .. } => {
                domains.push(Domain::from_paged_kv(
                    paged_state::ResidentPagedKvDomain::empty_for_contract(
                        requirement,
                        pools.page_pool.clone(),
                        pools.plan.page_tokens,
                        pools.numerical,
                    )?,
                )?);
            }
        }
    }
    if vectors.next().is_some() {
        return Err(Error::State(
            "resident hybrid vector root has extra planes".into(),
        ));
    }
    model.validate_domains(&domains, false)?;
    let vector_root_bytes = pools
        .plan
        .vector_layout
        .iter()
        .try_fold(0usize, |sum, plane| {
            sum.checked_add(plane.bytes)
                .ok_or_else(|| Error::Limit("resident hybrid vector root bytes".into()))
        })?;
    let _allocation = model.allocation.lock().await;
    let ticket = model.service.queue().submit(
        model.service.budget(),
        model.service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::RecurrentState,
            bytes: vector_root_bytes as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    check_cancel(cancelled)?;
    slot.begin_submission()?;
    let mut init = model.service.begin_initialization(0)?;
    retain(&mut init)?;
    for lease in &vector_leases {
        init.upload_zeroed_into(lease)?;
    }
    drop(init.complete_sync().await?);
    slot.finish()?;
    check_cancel(cancelled)?;
    model.validate_domains(&domains, true)?;
    Ok(domains)
}

fn check_cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("resident root checkout cancelled".into()));
    }
    Ok(())
}
pub(crate) async fn allocate_pooled_state<F>(
    model: &ResidentModel,
    pool: &Arc<ResidentPool>,
    zero: bool,
    cancelled: Option<&AtomicBool>,
    retain: F,
) -> Result<Vec<Domain>>
where
    F: FnOnce(&mut crate::resident_leases::ResidentInitialization<'_>) -> Result<()>,
{
    check_cancel(cancelled)?;
    model.service.ensure_usable()?;
    // Nonblocking checkout occurs BEFORE any fair slot/workspace is acquired.
    let leases = pool.checkout()?;
    let mut next = leases.iter();
    let mut domains = Vec::with_capacity(model.states.len());
    for requirement in &model.states {
        domains.push(make_domain(requirement, &mut |layout| {
            let lease = next
                .next()
                .ok_or_else(|| Error::State("resident root pool plane missing".into()))?;
            if layout.bytes()? != lease.bytes()
                || lease.role() != BufferRole::RecurrentState
                || lease.is_initialized()
                || !Arc::ptr_eq(lease.service(), &model.service)
            {
                return Err(Error::State(
                    "resident root pool exact current plane".into(),
                ));
            }
            Ok(lease.clone())
        })?);
    }
    if next.next().is_some() {
        return Err(Error::State("resident root pool extra plane".into()));
    }
    model.validate_domains(&domains, false)?;
    if zero {
        let _allocation = model.allocation.lock().await;
        let ticket = model.service.queue().submit(
            model.service.budget(),
            model.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::RecurrentState,
                bytes: model.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        check_cancel(cancelled)?;
        slot.begin_submission()?;
        let mut init = model.service.begin_initialization(0)?;
        retain(&mut init)?;
        for lease in &leases {
            init.upload_zeroed_into(lease)?;
        }
        drop(init.complete_sync().await?);
        slot.finish()?;
    }
    check_cancel(cancelled)?;
    model.validate_domains(&domains, zero)?;
    Ok(domains)
}

impl ResidentModel {
    #[cfg(any(
        feature = "cuda-native-qualification",
        feature = "cuda-cublas",
        feature = "gemma-bf16-qualification",
        feature = "hip-source"
    ))]
    pub(crate) fn root_pool_layout(&self) -> Result<Vec<ResidentPlaneLayout>> {
        let alignment = usize::try_from(self.service.client().properties().memory.alignment)
            .map_err(|_| Error::Limit("resident root allocator alignment".into()))?;
        let mut layout = Vec::new();
        let mut total = 0usize;
        for requirement in &self.states {
            let (count, plane) = requirement.plane_layout()?;
            let bytes = plane.bytes()?;
            total = count
                .checked_mul(bytes)
                .and_then(|n| total.checked_add(n))
                .ok_or_else(|| Error::Limit("resident root layout sum".into()))?;
            layout.extend((0..count).map(|_| ResidentPlaneLayout {
                bytes,
                role: BufferRole::RecurrentState,
                alignment,
            }));
        }
        if total != self.state_bytes {
            return Err(Error::State(
                "resident root source layout payload differs".into(),
            ));
        }
        Ok(layout)
    }
    #[cfg(all(
        any(
            feature = "gemma-bf16-qualification",
            feature = "hip-source",
            feature = "cuda-cublas"
        ),
        target_os = "linux"
    ))]
    pub(crate) fn install_window_root_pool(
        &self,
        source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
        roots: usize,
    ) -> Result<()> {
        if source.source_identity() != self.service.source_identity()
            || self.state_pool.get().is_some()
        {
            return Err(Error::State(
                "resident root pool source/reinitialization".into(),
            ));
        }
        let layout = self.root_pool_layout()?;
        let metadata = source
            .reserve_workspace(ResidentPool::metadata_bytes(layout.len(), roots)?)?
            .into_host_workspace()?;
        let pool = ResidentPool::prepare(self.service.clone(), metadata, &layout, roots)?;
        self.state_pool
            .set(PooledStates {
                storage: PooledStateStorage::Contiguous(pool),
                source: Some(source),
            })
            .map_err(|_| Error::State("resident root pool rebound".into()))
    }

    #[cfg(all(feature = "cuda-cublas", target_os = "linux"))]
    #[allow(
        dead_code,
        reason = "hybrid source-call selection lands after physical pool acceptance"
    )]
    pub(crate) fn install_window_hybrid_state_pools(
        &self,
        source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
        roots: usize,
        staging_sets: usize,
        page_tokens: usize,
        max_append_tokens: usize,
        numerical: zerogravity_core::accelerator::program::SourceNumericalContract,
    ) -> Result<()> {
        if source.source_identity() != self.service.source_identity()
            || self.state_pool.get().is_some()
        {
            return Err(Error::State(
                "resident hybrid pool source/reinitialization".into(),
            ));
        }
        let alignment = usize::try_from(self.service.client().properties().memory.alignment)
            .map_err(|_| Error::Limit("resident hybrid allocator alignment".into()))?;
        let plan = HybridStateInventoryPlan::analyze(
            &self.states,
            self.state_bytes,
            roots,
            staging_sets,
            page_tokens,
            max_append_tokens,
            alignment,
        )?;
        let vector_metadata = source
            .reserve_workspace(ResidentPool::metadata_bytes(
                plan.vector_layout.len(),
                plan.roots,
            )?)?
            .into_host_workspace()?;
        let staging_metadata = source
            .reserve_workspace(ResidentPool::metadata_bytes(
                plan.staging_layout.len(),
                plan.staging_sets,
            )?)?
            .into_host_workspace()?;
        let page_metadata = source
            .reserve_workspace(ResidentPagePool::metadata_bytes(plan.max_pages)?)?
            .into_host_workspace()?;
        let pools = HybridPooledStates::prepare(
            self.service.clone(),
            plan,
            vector_metadata,
            staging_metadata,
            page_metadata,
            numerical,
        )?;
        self.state_pool
            .set(PooledStates {
                storage: PooledStateStorage::Hybrid(pools),
                source: Some(source),
            })
            .map_err(|_| Error::State("resident hybrid pool rebound".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::program::{KvDomain, RecurrentDomain};

    fn hybrid_requirements() -> Vec<StateRequirement> {
        vec![
            StateRequirement {
                slot: StateSlot::Recurrent(RecurrentDomain(0)),
                layout: StateLayout::Vector { elements: 128 },
                physical: ResidentPhysicalType::F32,
                bytes: 128 * 4,
            },
            StateRequirement {
                slot: StateSlot::Kv(KvDomain(1)),
                layout: StateLayout::Kv {
                    slab: 128,
                    capacity: 8,
                },
                physical: ResidentPhysicalType::F32,
                bytes: 2 * 128 * 8 * 4,
            },
            StateRequirement {
                slot: StateSlot::Kv(KvDomain(2)),
                layout: StateLayout::Kv {
                    slab: 128,
                    capacity: 8,
                },
                physical: ResidentPhysicalType::F32,
                bytes: 2 * 128 * 8 * 4,
            },
        ]
    }

    #[test]
    fn hybrid_inventory_never_exceeds_original_root_envelope() {
        let states = hybrid_requirements();
        let state_bytes = states.iter().map(|state| state.bytes).sum::<usize>();
        let plan =
            HybridStateInventoryPlan::analyze(&states, state_bytes, 4, 1, 4, 4, 512).unwrap();
        assert_eq!(plan.vector_layout.len(), 1);
        assert_eq!(plan.staging_layout.len(), 4);
        assert_eq!(plan.page.bytes, 2048);
        assert_eq!(plan.max_pages, 24);
        assert_eq!(plan.pages_per_committed_root, 8);
        assert_eq!(plan.candidate_pages_per_active_session, 8);
        assert_eq!(plan.root_budget_bytes, state_bytes * 4);
        assert_eq!(
            plan.vector_max_bytes + plan.staging_max_bytes + plan.page_max_bytes,
            plan.root_budget_bytes
        );
        assert_eq!(plan.slack_bytes, 0);
    }

    #[test]
    fn hybrid_inventory_does_not_preallocate_unreachable_cow_pages() {
        let states = hybrid_requirements();
        let state_bytes = states.iter().map(|state| state.bytes).sum::<usize>();
        let plan =
            HybridStateInventoryPlan::analyze(&states, state_bytes, 4, 1, 4, 1, 512).unwrap();
        assert_eq!(plan.pages_per_committed_root, 8);
        assert_eq!(plan.candidate_pages_per_active_session, 4);
        assert_eq!(plan.max_pages, 20);
        assert_eq!(
            plan.resident_bytes().unwrap() + plan.slack_bytes,
            plan.root_budget_bytes
        );
        assert!(plan.slack_bytes > 0);
    }

    #[test]
    fn hybrid_inventory_refuses_root_budget_without_candidate_page_headroom() {
        let states = hybrid_requirements();
        let state_bytes = states.iter().map(|state| state.bytes).sum::<usize>();
        assert!(matches!(
            HybridStateInventoryPlan::analyze(&states, state_bytes, 2, 1, 4, 4, 512),
            Err(Error::Limit(reason))
                if reason == "resident hybrid page inventory cannot fork every active session"
        ));
    }

    #[test]
    fn hybrid_inventory_rejects_heterogeneous_page_geometry() {
        let mut states = hybrid_requirements();
        states[2] = StateRequirement {
            slot: StateSlot::Kv(KvDomain(2)),
            layout: StateLayout::Kv {
                slab: 64,
                capacity: 8,
            },
            physical: ResidentPhysicalType::F32,
            bytes: 2 * 64 * 8 * 4,
        };
        let state_bytes = states.iter().map(|state| state.bytes).sum::<usize>();
        assert!(matches!(
            HybridStateInventoryPlan::analyze(&states, state_bytes, 4, 1, 4, 4, 512),
            Err(Error::Unsupported(reason))
                if reason == "resident hybrid heterogeneous KV page geometry"
        ));
    }

    #[test]
    fn hybrid_inventory_rejects_append_bound_above_state_capacity() {
        let states = hybrid_requirements();
        let state_bytes = states.iter().map(|state| state.bytes).sum::<usize>();
        assert!(matches!(
            HybridStateInventoryPlan::analyze(&states, state_bytes, 4, 1, 4, 9, 512),
            Err(Error::State(reason))
                if reason == "resident hybrid KV plane geometry differs"
        ));
    }

    #[cfg(all(
        feature = "cuda-native-qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    #[test]
    #[ignore = "physical CUDA hybrid vector/staging/page pool envelope"]
    fn physical_hybrid_pool_matches_original_root_envelope() -> Result<()> {
        use crate::cuda::test_support::CudaDiagnostic;
        use zerogravity_core::accelerator::program::SourceNumericalContract;
        use zerogravity_core::resource_arbiter::{
            AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        };

        fn metadata(
            source: zerogravity_core::accelerator::SourceIdentity,
            label: &str,
            bytes: usize,
        ) -> Result<zerogravity_core::resource_arbiter::HostWorkspace> {
            let key = PhysicalDeviceKey::new("host", label)?;
            let arbiter = ResourceArbiter::process_shared();
            arbiter.register_matching(
                key.clone(),
                u64::try_from(bytes)
                    .map_err(|_| Error::Limit("hybrid pool metadata extent".into()))?,
            )?;
            arbiter
                .admit(
                    &key,
                    ResourceOwner::new("hybrid-pool-qualification", label)?,
                    &[AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: bytes as u64,
                    }],
                )?
                .into_host_workspace(source, Arc::new(()))
        }

        let diagnostic = CudaDiagnostic::new("hybrid-state-pools")?;
        let service = diagnostic.service.clone();
        let alignment = usize::try_from(service.client().properties().memory.alignment)
            .map_err(|_| Error::Limit("hybrid pool allocator alignment".into()))?;
        let states = hybrid_requirements();
        let state_bytes = states.iter().map(|state| state.bytes).sum::<usize>();
        let plan = HybridStateInventoryPlan::analyze(&states, state_bytes, 4, 1, 4, 4, alignment)?;
        let expected_bytes = plan.root_budget_bytes;
        let vector_metadata = metadata(
            service.source_identity(),
            "zerogravity-hybrid-vector-metadata",
            ResidentPool::metadata_bytes(plan.vector_layout.len(), plan.roots)?,
        )?;
        let staging_metadata = metadata(
            service.source_identity(),
            "zerogravity-hybrid-staging-metadata",
            ResidentPool::metadata_bytes(plan.staging_layout.len(), plan.staging_sets)?,
        )?;
        let page_metadata = metadata(
            service.source_identity(),
            "zerogravity-hybrid-page-metadata",
            ResidentPagePool::metadata_bytes(plan.max_pages)?,
        )?;
        let pools = HybridPooledStates::prepare(
            service.clone(),
            plan,
            vector_metadata,
            staging_metadata,
            page_metadata,
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        )?;

        let mut vector_roots = Vec::new();
        for _ in 0..pools.plan.roots {
            vector_roots.push(pools.vector_pool.checkout()?);
        }
        let staging = pools.staging_pool.checkout()?;
        let mut pages = Vec::new();
        for _ in 0..pools.plan.max_pages {
            pages.push(pools.page_pool.checkout()?);
        }
        if service.resident_bytes()? != expected_bytes
            || vector_roots.iter().any(|root| {
                root.len() != pools.plan.vector_layout.len()
                    || root.iter().any(ResidentLease::is_initialized)
            })
            || staging.len() != pools.plan.staging_layout.len()
            || staging.iter().any(ResidentLease::is_initialized)
            || pages.iter().any(ResidentLease::is_initialized)
        {
            return Err(Error::State(
                "hybrid physical pool simultaneous envelope differs".into(),
            ));
        }
        let empty = paged_state::ResidentPagedKvDomain::empty_for_contract(
            &states[1],
            pools.page_pool.clone(),
            pools.plan.page_tokens,
            pools.numerical,
        )?;
        if empty.tokens()? != 0 || !empty.leases()?.is_empty() {
            return Err(Error::State(
                "hybrid physical pool empty paged root differs".into(),
            ));
        }
        drop(empty);
        drop(vector_roots);
        drop(staging);
        drop(pages);
        let vector = pools.vector_pool.snapshot()?;
        let staging = pools.staging_pool.snapshot()?;
        let pages = pools.page_pool.snapshot()?;
        if vector.allocated_sets != pools.plan.roots
            || vector.idle_sets != pools.plan.roots
            || staging.allocated_sets != pools.plan.staging_sets
            || staging.idle_sets != pools.plan.staging_sets
            || pages.allocated_pages != pools.plan.max_pages
            || pages.idle_pages != pools.plan.max_pages
            || service.resident_bytes()? != expected_bytes
        {
            return Err(Error::State(
                "hybrid physical pool idle ownership differs".into(),
            ));
        }
        let max_pages = pools.plan.max_pages;
        let vector_sets = pools.plan.roots;
        let staging_sets = pools.plan.staging_sets;
        drop(pools);
        if service.resident_bytes()? != 0 {
            return Err(Error::State(
                "hybrid physical pool backing did not retire".into(),
            ));
        }
        println!(
            "P12_HYBRID_POOL_PASS old_root_budget_bytes={expected_bytes} vector_sets={vector_sets} staging_sets={staging_sets} max_pages={max_pages} simultaneous_exact=true empty_paged_root=true idle_reuse=true final_resident_bytes=0"
        );
        Ok(())
    }

    #[cfg(all(
        feature = "cuda-native-qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    #[test]
    #[ignore = "physical CUDA current-root-aware hybrid candidate one-flight composition"]
    fn physical_hybrid_candidate_one_flight() -> Result<()> {
        use crate::cuda::{
            ordinary_text::{
                Access as OrdinaryAccess, BufferRole as OrdinaryRole, Geometry, LaunchBinding,
                Program, Step,
            },
            test_support::CudaDiagnostic,
        };
        use cubecl::CubeElement;
        use zerogravity_core::accelerator::program::SourceNumericalContract;
        use zerogravity_core::resource_arbiter::{
            AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        };

        fn metadata(
            source: zerogravity_core::accelerator::SourceIdentity,
            label: &str,
            bytes: usize,
        ) -> Result<zerogravity_core::resource_arbiter::HostWorkspace> {
            let key = PhysicalDeviceKey::new("host", label)?;
            let arbiter = ResourceArbiter::process_shared();
            arbiter.register_matching(
                key.clone(),
                u64::try_from(bytes)
                    .map_err(|_| Error::Limit("hybrid candidate metadata extent".into()))?,
            )?;
            arbiter
                .admit(
                    &key,
                    ResourceOwner::new("hybrid-candidate-qualification", label)?,
                    &[AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: bytes as u64,
                    }],
                )?
                .into_host_workspace(source, Arc::new(()))
        }

        fn bf16_bytes(words: &[u16]) -> Vec<u8> {
            words.iter().flat_map(|word| word.to_le_bytes()).collect()
        }

        futures_lite::future::block_on(async {
            let diagnostic = CudaDiagnostic::new("hybrid-candidate-one-flight")?;
            let service = diagnostic.service.clone();
            let client = service.client();
            let alignment = usize::try_from(client.properties().memory.alignment)
                .map_err(|_| Error::Limit("hybrid candidate allocator alignment".into()))?;
            let geometry = Geometry::qwen35_08();
            let kv_width = geometry.kv_width()?;
            const CAPACITY: usize = 8;
            const BASE: usize = 3;
            const ROWS: usize = 1;
            const PAGE_TOKENS: usize = 4;
            const PACKED_WIDTH: usize = 5120;
            const VALUE_OFFSET: usize = 4608;
            let states = vec![
                StateRequirement {
                    slot: StateSlot::Recurrent(RecurrentDomain(0)),
                    layout: StateLayout::Vector { elements: 512 },
                    physical: ResidentPhysicalType::Bf16,
                    bytes: 512 * 2,
                },
                StateRequirement {
                    slot: StateSlot::Kv(KvDomain(1)),
                    layout: StateLayout::Kv {
                        slab: kv_width,
                        capacity: CAPACITY,
                    },
                    physical: ResidentPhysicalType::Bf16,
                    bytes: 2 * kv_width * CAPACITY * 2,
                },
                StateRequirement {
                    slot: StateSlot::Kv(KvDomain(2)),
                    layout: StateLayout::Kv {
                        slab: kv_width,
                        capacity: CAPACITY,
                    },
                    physical: ResidentPhysicalType::Bf16,
                    bytes: 2 * kv_width * CAPACITY * 2,
                },
            ];
            let state_bytes = states.iter().map(|state| state.bytes).sum::<usize>();
            let plan = HybridStateInventoryPlan::analyze(
                &states,
                state_bytes,
                4,
                1,
                PAGE_TOKENS,
                ROWS,
                alignment,
            )?;
            let pools = HybridPooledStates::prepare(
                service.clone(),
                plan,
                metadata(
                    service.source_identity(),
                    "zerogravity-hybrid-candidate-vector-metadata",
                    ResidentPool::metadata_bytes(1, 4)?,
                )?,
                metadata(
                    service.source_identity(),
                    "zerogravity-hybrid-candidate-staging-metadata",
                    ResidentPool::metadata_bytes(4, 1)?,
                )?,
                metadata(
                    service.source_identity(),
                    "zerogravity-hybrid-candidate-page-metadata",
                    ResidentPagePool::metadata_bytes(24)?,
                )?,
                SourceNumericalContract::ReferenceCudaSm86Bf16V1,
            )?;

            let vector_root = pools.vector_pool.checkout()?;
            let mut vector_iter = vector_root.iter();
            let current_vector = make_domain(&states[0], &mut |layout| {
                let lease = vector_iter.next().ok_or_else(|| {
                    Error::State("hybrid candidate current vector missing".into())
                })?;
                if lease.bytes() != layout.bytes()? {
                    return Err(Error::State(
                        "hybrid candidate current vector extent differs".into(),
                    ));
                }
                Ok(lease.clone())
            })?;
            if vector_iter.next().is_some() {
                return Err(Error::State(
                    "hybrid candidate current vector has extra planes".into(),
                ));
            }
            let current_vector_accesses = current_vector
                .leases
                .iter()
                .cloned()
                .map(|lease| (lease, DeviceAccess::Write))
                .collect::<Vec<_>>();
            let vector_flight = service.submit(0, &current_vector_accesses)?;
            for lease in &current_vector.leases {
                client.write(
                    &lease.handle(),
                    Bytes::from_bytes_vec(vec![0u8; lease.bytes()]),
                );
            }
            let vector_reads = current_vector
                .leases
                .iter()
                .map(|lease| lease.readback(0, lease.bytes()))
                .collect::<Result<Vec<_>>>()?;
            let (vector_ready, vector_data) = vector_flight
                .complete_with_readback(&current_vector.leases, &vector_reads, |data| {
                    if data
                        .iter()
                        .flat_map(|bytes| bytes.iter())
                        .any(|byte| *byte != 0)
                    {
                        return Err(Error::State(
                            "hybrid candidate current vector initialization differs".into(),
                        ));
                    }
                    Ok(())
                })
                .await?;
            drop(vector_ready);
            drop(vector_data);
            drop(vector_reads);
            drop(current_vector_accesses);
            drop(vector_root);

            async fn materialize(
                service: &Arc<ResidentService>,
                current: &paged_state::ResidentPagedKvDomain,
                keys: &[u8],
                values: &[u8],
            ) -> Result<paged_state::ResidentPagedKvDomain> {
                let append = current.fork_for_append(BASE)?;
                let accesses = append.accesses()?;
                let flight = service.submit(0, &accesses)?;
                append.enqueue_cuda(&flight, keys, values)?;
                let outputs = append.outputs()?;
                let reads = outputs
                    .iter()
                    .map(|lease| lease.readback(0, lease.bytes()))
                    .collect::<Result<Vec<_>>>()?;
                let (ready, data) = flight
                    .complete_with_readback(&outputs, &reads, |_| Ok(()))
                    .await?;
                drop(ready);
                drop(data);
                drop(reads);
                drop(outputs);
                drop(accesses);
                append.into_candidate(current)
            }

            let row_bytes = kv_width * 2;
            let row = |token: usize, salt: usize| {
                (0..kv_width)
                    .map(|lane| 0x3c00u16 + ((token * 29 + lane * 7 + salt) % 0x0200) as u16)
                    .collect::<Vec<_>>()
            };
            let mut current = vec![current_vector];
            let mut expected = Vec::new();
            for (domain_index, requirement) in states[1..].iter().enumerate() {
                let empty = paged_state::ResidentPagedKvDomain::empty_for_contract(
                    requirement,
                    pools.page_pool.clone(),
                    PAGE_TOKENS,
                    pools.numerical,
                )?;
                let mut keys = Vec::new();
                let mut values = Vec::new();
                for token in 0..BASE {
                    keys.extend(bf16_bytes(&row(token, 11 + domain_index * 17)));
                    values.extend(bf16_bytes(&row(token, 113 + domain_index * 19)));
                }
                let ready = materialize(&service, &empty, &keys, &values).await?;
                expected.push((keys, values));
                current.push(Domain::from_paged_kv(ready)?);
            }

            let bundle = pools.candidate(&states, &current, ROWS, BASE)?;
            if bundle.bindings().len() != states.len()
                || bundle.bindings()[0].mode != crate::resident_program::StateBindingMode::Distinct
                || bundle.bindings()[1..].iter().any(|binding| {
                    binding.mode != crate::resident_program::StateBindingMode::KvInPlace
                })
            {
                return Err(Error::State("hybrid candidate binding modes differ".into()));
            }
            let accesses = bundle.accesses()?;
            let flight = service.submit(0, &accesses)?;
            let gather_dispatches = bundle.enqueue_gather_cuda(&flight)?;
            let promoted = bundle.write_outputs()?;
            if promoted.len() != 5 {
                return Err(Error::State(
                    "hybrid candidate promoted output count differs".into(),
                ));
            }
            client.write(
                &promoted[0].handle(),
                Bytes::from_bytes_vec(vec![0u8; promoted[0].bytes()]),
            );

            let program = Program::new(geometry)?;
            let step = Step::new(ROWS, BASE, CAPACITY)?;
            let status = client.create_from_slice(u32::as_bytes(&[0u32]));
            for (domain_index, binding) in bundle.bindings()[1..].iter().enumerate() {
                let StateBuffers::Kv { keys, values } = &binding.candidate else {
                    return Err(Error::State(
                        "hybrid candidate KV staging binding differs".into(),
                    ));
                };
                let new_key = row(BASE, 11 + domain_index * 17);
                let new_value = row(BASE, 113 + domain_index * 19);
                let mut packed = vec![0u16; PACKED_WIDTH];
                packed[VALUE_OFFSET..VALUE_OFFSET + kv_width].copy_from_slice(&new_value);
                let key = client.create_from_slice(&bf16_bytes(&new_key));
                let packed = client.create_from_slice(&bf16_bytes(&packed));
                program
                    .kv_append(step)?
                    .prepare_owned(
                        client,
                        vec![
                            LaunchBinding::new(OrdinaryRole::Key, OrdinaryAccess::Read, key),
                            LaunchBinding::new(
                                OrdinaryRole::PackedAttention,
                                OrdinaryAccess::Read,
                                packed,
                            ),
                            LaunchBinding::new(
                                OrdinaryRole::KeyCache,
                                OrdinaryAccess::ReadWrite,
                                keys.handle.clone(),
                            ),
                            LaunchBinding::new(
                                OrdinaryRole::ValueCache,
                                OrdinaryAccess::ReadWrite,
                                values.handle.clone(),
                            ),
                        ],
                        LaunchBinding::new(
                            OrdinaryRole::Status,
                            OrdinaryAccess::ReadWrite,
                            status.clone(),
                        ),
                    )?
                    .enqueue_owned(client, &status)?;
                expected[domain_index]
                    .0
                    .extend_from_slice(&bf16_bytes(&new_key));
                expected[domain_index]
                    .1
                    .extend_from_slice(&bf16_bytes(&new_value));
            }
            let scatter_dispatches = bundle.enqueue_scatter_cuda(&flight)?;
            let readback = promoted[0].readback(0, promoted[0].bytes())?;
            let (ready, vector_data) = flight
                .complete_with_readback(&promoted, &[readback], |data| {
                    if data.len() != 1 || data[0].iter().any(|byte| *byte != 0) {
                        return Err(Error::State(
                            "hybrid candidate vector completion differs".into(),
                        ));
                    }
                    Ok(())
                })
                .await?;
            let status_bytes = client
                .read_one(status)
                .map_err(|error| Error::State(format!("hybrid candidate status: {error}")))?;
            if status_bytes.len() != 4
                || u32::from_le_bytes(status_bytes[..4].try_into().unwrap()) != 0
            {
                return Err(Error::State(
                    "hybrid candidate ordinary KV status differs".into(),
                ));
            }
            drop(vector_data);
            let candidate = bundle.into_domains()?;
            let canonical = canonical_write_outputs(&current, &candidate)?;
            if canonical.len() != ready.outputs().len()
                || canonical
                    .iter()
                    .zip(ready.outputs())
                    .any(|(expected, actual)| expected.id() != actual.id())
            {
                return Err(Error::State(
                    "hybrid candidate Ready/canonical output ordering differs".into(),
                ));
            }
            if candidate[0]
                .leases
                .iter()
                .any(|lease| !lease.is_initialized())
                || candidate[1..]
                    .iter()
                    .any(|domain| domain.paged_kv().is_err())
            {
                return Err(Error::State(
                    "hybrid candidate canonical domain readiness differs".into(),
                ));
            }
            drop(accesses);

            let staging = pools.staging_pool.checkout()?;
            let mut staging_iter = staging.iter();
            let mut verify_accesses =
                BTreeMap::<BufferLeaseId, (ResidentLease, DeviceAccess)>::new();
            let mut verify_pairs = Vec::new();
            for domain in &candidate[1..] {
                let paged = domain.paged_kv()?;
                if paged.tokens()? != BASE + ROWS {
                    return Err(Error::State(
                        "hybrid candidate paged frontier differs".into(),
                    ));
                }
                let key = staging_iter
                    .next()
                    .ok_or_else(|| Error::State("hybrid verify key staging missing".into()))?;
                let value = staging_iter
                    .next()
                    .ok_or_else(|| Error::State("hybrid verify value staging missing".into()))?;
                for (lease, access) in paged.gather_accesses(BASE + ROWS, key, value)? {
                    merge_access(&mut verify_accesses, lease, access)?;
                }
                verify_pairs.push((paged.clone(), key.clone(), value.clone()));
            }
            if staging_iter.next().is_some() {
                return Err(Error::State(
                    "hybrid verify staging has extra planes".into(),
                ));
            }
            let verify_accesses = verify_accesses.into_values().collect::<Vec<_>>();
            let verify_flight = service.submit(0, &verify_accesses)?;
            for (paged, key, value) in &verify_pairs {
                paged.enqueue_gather_cuda(&verify_flight, BASE + ROWS, key, value)?;
            }
            let verify_outputs = staging.clone();
            let verify_reads = verify_outputs
                .iter()
                .map(|lease| lease.readback(0, lease.bytes()))
                .collect::<Result<Vec<_>>>()?;
            let (_, gathered) = verify_flight
                .complete_with_readback(&verify_outputs, &verify_reads, |_| Ok(()))
                .await?;
            for (domain_index, pair) in gathered.chunks_exact(2).enumerate() {
                let active = (BASE + ROWS) * row_bytes;
                if pair[0][..active] != expected[domain_index].0
                    || pair[1][..active] != expected[domain_index].1
                    || pair[0][active..].iter().any(|byte| *byte != 0)
                    || pair[1][active..].iter().any(|byte| *byte != 0)
                {
                    return Err(Error::State(
                        "hybrid candidate gathered continuation differs".into(),
                    ));
                }
            }

            drop(gathered);
            drop(verify_reads);
            drop(verify_outputs);
            drop(verify_accesses);
            drop(verify_pairs);
            drop(staging);
            drop(ready);
            drop(promoted);
            drop(canonical);
            drop(candidate);
            drop(current);
            let old_budget = pools.plan.root_budget_bytes;
            drop(pools);
            if service.resident_bytes()? != 0 {
                return Err(Error::State(
                    "hybrid candidate pool backing did not retire".into(),
                ));
            }
            println!(
                "P12_HYBRID_CANDIDATE_PASS base={BASE} rows={ROWS} kv_domains=2 gather_dispatches={gather_dispatches} scatter_dispatches={scatter_dispatches} ready_outputs=5 canonical_ready_order=true paged_frontier=4 vector_ready=true continuation_exact=true inactive_tail_zero=true old_root_budget_bytes={old_budget} final_resident_bytes=0"
            );
            Ok(())
        })
    }
}
