//! Resident physical-page adapter for the model-neutral paged-state table.
//!
//! This does not alter ResidentSession state or publish roots. It only binds the
//! existing charged ResidentPagePool/ResidentLease owner to the core immutable
//! block-table/COW planner so later KV integration cannot bypass the established
//! generation, reader, completion and quarantine authority.

use super::*;
use crate::resident_leases::backing_pool::{ResidentPagePool, ResidentPagePoolSnapshot};
use zerogravity_core::paged_state::{
    PageAppendPlan, PageCompatibility, PageGeometry, PagedBlockTable,
};

#[derive(Clone)]
pub(crate) struct ResidentPagedTable {
    pool: Arc<ResidentPagePool>,
    table: PagedBlockTable<ResidentLease>,
}

impl ResidentPagedTable {
    pub(crate) fn empty(
        pool: Arc<ResidentPagePool>,
        compatibility: PageCompatibility,
        geometry: PageGeometry,
    ) -> Result<Self> {
        if pool.page_bytes() != geometry.page_bytes()
            || pool.role() != BufferRole::RecurrentState
            || !geometry.unit_bytes().is_multiple_of(pool.alignment())
        {
            return Err(Error::State(
                "resident paged table physical role/extent/alignment differs".into(),
            ));
        }
        Ok(Self {
            pool,
            table: PagedBlockTable::empty(compatibility, geometry),
        })
    }

    pub(crate) fn compatibility(&self) -> PageCompatibility {
        self.table.compatibility()
    }

    pub(crate) fn geometry(&self) -> PageGeometry {
        self.table.geometry()
    }

    pub(crate) fn logical_bytes(&self) -> usize {
        self.table.logical_bytes()
    }

    pub(crate) fn page_count(&self) -> usize {
        self.table.page_count()
    }

    pub(crate) fn page(&self, index: usize) -> Option<(&ResidentLease, usize)> {
        self.table.page(index)
    }

    pub(crate) fn pool_snapshot(&self) -> Result<ResidentPagePoolSnapshot> {
        self.pool.snapshot()
    }

    pub(crate) fn share_prefix(&self, bytes: usize) -> Result<Self> {
        Ok(Self {
            pool: self.pool.clone(),
            table: self.table.share_prefix(
                self.table.compatibility(),
                self.table.geometry(),
                bytes,
            )?,
        })
    }

    /// Private candidate only. The returned plan pins every source/target page
    /// needed by physical COW/write work; neither object is a Ready witness.
    pub(crate) fn fork_for_append(
        &self,
        bytes: usize,
    ) -> Result<(Self, PageAppendPlan<ResidentLease>)> {
        let (table, plan) = self.table.fork_for_append(bytes, || self.pool.checkout())?;
        Ok((
            Self {
                pool: self.pool.clone(),
                table,
            },
            plan,
        ))
    }
}

#[cfg(all(
    test,
    feature = "cuda-native-qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
mod tests {
    use super::*;
    use crate::cuda::test_support::CudaDiagnostic;
    use crate::resident_leases::backing_pool::{ResidentPagePool, ResidentPlaneLayout};
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use zerogravity_core::device_events::{BufferRole, DeviceAccess};
    use zerogravity_core::resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    };

    #[test]
    #[ignore = "physical CUDA paged-state pool/COW ownership and unknown-completion quarantine"]
    fn physical_page_pool_cow_pressure_reader_and_quarantine() -> Result<()> {
        futures_lite::future::block_on(async {
            let diagnostic = CudaDiagnostic::new("paged-state-pool")?;
            let service = diagnostic.service.clone();
            let alignment = usize::try_from(service.client().properties().memory.alignment)
                .map_err(|_| Error::Limit("paged-state allocator alignment".into()))?;
            let page_bytes = alignment
                .checked_mul(4)
                .ok_or_else(|| Error::Limit("paged-state page extent overflow".into()))?;
            if page_bytes as u64 > service.client().properties().memory.max_page_size {
                return Err(Error::Unsupported(
                    "paged-state test page exceeds backend page capability".into(),
                ));
            }
            let max_pages = 3usize;
            let metadata_bytes = ResidentPagePool::metadata_bytes(max_pages)?;
            let host_key =
                PhysicalDeviceKey::new("host", "zerogravity-paged-state-pool-qualification")?;
            let host_limit = u64::try_from(metadata_bytes)
                .ok()
                .and_then(|n| n.checked_mul(2))
                .filter(|n| *n != 0)
                .ok_or_else(|| Error::Limit("paged-state host metadata limit".into()))?;
            let arbiter = ResourceArbiter::process_shared();
            arbiter.register_matching(host_key.clone(), host_limit)?;
            let metadata = arbiter
                .admit(
                    &host_key,
                    ResourceOwner::new("paged-state-pool", "block-table-metadata")?,
                    &[AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: metadata_bytes as u64,
                    }],
                )?
                .into_host_workspace(service.source_identity(), Arc::new(()))?;
            let pool = ResidentPagePool::prepare(
                service.clone(),
                metadata,
                ResidentPlaneLayout {
                    bytes: page_bytes,
                    role: BufferRole::RecurrentState,
                    alignment,
                },
                max_pages,
            )?;
            if pool.page_bytes() != page_bytes
                || pool.alignment() != alignment
                || pool.role() != BufferRole::RecurrentState
            {
                return Err(Error::State(
                    "paged-state physical pool geometry changed".into(),
                ));
            }

            let compatibility = PageCompatibility::new(
                service.source_identity(),
                service.generation(),
                1,
                page_bytes as u64,
            )?;
            let geometry = PageGeometry::new(page_bytes, alignment)?;
            let empty = ResidentPagedTable::empty(pool.clone(), compatibility, geometry)?;
            let (first, first_plan) = empty.fork_for_append(page_bytes - alignment)?;
            if first.page_count() != 1
                || first.logical_bytes() != page_bytes - alignment
                || first_plan.copies().len() != 0
                || first_plan.writes().len() != 1
            {
                return Err(Error::State(
                    "paged-state initial partial page plan differs".into(),
                ));
            }
            let first_generation = first
                .page(0)
                .ok_or_else(|| Error::State("paged-state first page absent".into()))?
                .0
                .allocation_generation();

            let (second, second_plan) = first.fork_for_append(2 * alignment)?;
            if second.page_count() != 2
                || second.logical_bytes() != page_bytes + alignment
                || second_plan.copies().len() != 1
                || second_plan.copies()[0].bytes() != page_bytes - alignment
                || second_plan.writes().len() != 2
            {
                return Err(Error::State(
                    "paged-state partial-tail COW plan differs".into(),
                ));
            }
            let generations = (0..second.page_count())
                .map(|index| {
                    second
                        .page(index)
                        .map(|(lease, _)| lease.allocation_generation())
                        .ok_or_else(|| Error::State("paged-state candidate page absent".into()))
                })
                .collect::<Result<BTreeSet<_>>>()?;
            if generations.contains(&first_generation) || generations.len() != 2 {
                return Err(Error::State(
                    "paged-state COW reused old logical page generation".into(),
                ));
            }
            let full = second.pool_snapshot()?;
            if full.allocated_pages != 3
                || full.idle_pages != 0
                || full.backing_bytes != max_pages * page_bytes
            {
                return Err(Error::State(
                    "paged-state finite physical pressure accounting differs".into(),
                ));
            }
            let before_pressure = (second.logical_bytes(), second.page_count());
            if second.fork_for_append(page_bytes).is_ok()
                || (second.logical_bytes(), second.page_count()) != before_pressure
            {
                return Err(Error::State(
                    "paged-state exhaustion mutated prior block table".into(),
                ));
            }

            let reader = second.clone();
            drop(first);
            drop(first_plan);
            drop(second);
            drop(second_plan);
            let pinned = pool.snapshot()?;
            if pinned.allocated_pages != 3 || pinned.idle_pages != 1 {
                return Err(Error::State(
                    "paged-state cloned reader did not pin candidate pages".into(),
                ));
            }
            drop(reader);
            let idle = pool.snapshot()?;
            if idle.allocated_pages != 3 || idle.idle_pages != 3 {
                return Err(Error::State(
                    "paged-state retired owners did not return all pages".into(),
                ));
            }

            let reused = pool.checkout()?;
            let reused_generation = reused.allocation_generation();
            if reused_generation == first_generation || generations.contains(&reused_generation) {
                return Err(Error::State(
                    "paged-state physical reuse repeated logical generation".into(),
                ));
            }
            let ticket = service.queue().submit(
                service.budget(),
                service.owner().clone(),
                &[AllocationRequest {
                    kind: AllocationClass::RecurrentState,
                    bytes: page_bytes as u64,
                }],
                None,
            )?;
            let mut slot = ticket.wait().await?;
            slot.begin_submission()?;
            let mut initialization = service.begin_initialization(0)?;
            initialization.upload_zeroed_into(&reused)?;
            drop(initialization.complete_sync().await?);
            slot.finish()?;
            if !reused.is_initialized() {
                return Err(Error::State(
                    "paged-state reused page initialization did not complete".into(),
                ));
            }

            let readback = reused.readback(0, alignment)?;
            let read = service.submit(0, &[(reused.clone(), DeviceAccess::Read)])?;
            let during_read = pool.snapshot()?;
            if during_read.idle_pages != 2 {
                return Err(Error::State(
                    "paged-state in-flight reader returned page early".into(),
                ));
            }
            let bytes = read
                .complete_read_only_readback(&[readback], |cohort| {
                    if cohort.len() != 1
                        || cohort[0].len() != alignment
                        || cohort[0].iter().any(|byte| *byte != 0)
                    {
                        return Err(Error::State(
                            "paged-state initialized reader bytes differ".into(),
                        ));
                    }
                    Ok(())
                })
                .await?;
            if bytes.len() != 1 {
                return Err(Error::State(
                    "paged-state readback cohort size differs".into(),
                ));
            }
            drop(reused);
            if pool.snapshot()?.idle_pages != 3 {
                return Err(Error::State(
                    "paged-state completed reader did not release page".into(),
                ));
            }

            let uncertain = pool.checkout()?;
            let unknown = service.submit(0, &[(uncertain.clone(), DeviceAccess::Write)])?;
            unknown.require_accesses([(&uncertain, DeviceAccess::Write)])?;
            service.client().write(
                &uncertain.handle(),
                cubecl::bytes::Bytes::from_bytes_vec(vec![0x5a; page_bytes]),
            );
            drop(uncertain);
            drop(unknown);
            let fenced = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
            let quarantined =
                crate::physical_quarantine::quarantined_handles_for(service.client())?;
            let after_unknown = pool.snapshot()?;
            if service.ensure_usable().is_ok()
                || !fenced.fenced
                || quarantined == 0
                || after_unknown.idle_pages != 2
            {
                return Err(Error::State(
                    "paged-state unknown completion did not retain/fence page".into(),
                ));
            }
            service
                .client()
                .sync()
                .await
                .map_err(|error| Error::State(format!("paged-state exit quiescence: {error}")))?;
            if service.ensure_usable().is_ok()
                || !ResourceArbiter::process_shared()
                    .snapshot(&service.physical_key())?
                    .fenced
                || crate::physical_quarantine::quarantined_handles_for(service.client())? == 0
                || pool.snapshot()?.idle_pages != 2
            {
                return Err(Error::State(
                    "paged-state diagnostic sync cleared unknown ownership".into(),
                ));
            }

            println!(
                "P12_PAGED_STATE_POOL_PASS page_bytes={page_bytes} max_pages={max_pages} allocated={} idle_before_unknown=3 idle_after_unknown={} COW_tail=true generation_fresh=true reader_pinned=true exhaustion_atomic=true submitted_unknown_fenced=true diagnostic_sync_does_not_reclaim=true quarantined_handles={quarantined}",
                after_unknown.allocated_pages, after_unknown.idle_pages
            );
            Ok(())
        })
    }
}
