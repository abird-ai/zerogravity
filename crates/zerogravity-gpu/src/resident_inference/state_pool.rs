//! Shared .8B/2B root checkout: fresh logical leases, unchanged domain binding.
use super::*;
use crate::resident_leases::backing_pool::ResidentPagePool;
#[cfg(any(
    feature = "cuda-native-qualification",
    feature = "cuda-cublas",
    feature = "gemma-bf16-qualification",
    feature = "hip-source"
))]
use crate::resident_leases::backing_pool::ResidentPlaneLayout;
use crate::resident_leases::backing_pool::ResidentPool;
use std::sync::atomic::{AtomicBool, Ordering};

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
                    if planes != 2 || page_tokens > capacity {
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
        let max_pages = page_budget / page.bytes;
        let page_max_bytes = max_pages
            .checked_mul(page.bytes)
            .ok_or_else(|| Error::Limit("resident hybrid page maximum".into()))?;
        let candidate_pages_per_domain = (page_tokens - 1)
            .checked_add(max_append_tokens)
            .ok_or_else(|| Error::Limit("resident hybrid append span".into()))?
            .div_ceil(page_tokens)
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("resident hybrid candidate pages/domain".into()))?;
        let candidate_pages_per_active_session = kv_domains
            .checked_mul(candidate_pages_per_domain)
            .ok_or_else(|| Error::Limit("resident hybrid candidate pages/session".into()))?;
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
}
