//! Resident physical-page adapter for the model-neutral paged-state table.
//!
//! This does not alter ResidentSession state or publish roots. It only binds the
//! existing charged ResidentPagePool/ResidentLease owner to the core immutable
//! block-table/COW planner so later KV integration cannot bypass the established
//! generation, reader, completion and quarantine authority.

use super::*;
use crate::resident_leases::backing_pool::{ResidentPagePool, ResidentPagePoolSnapshot};
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
use crate::{cuda::word_transport::RuntimeWordTransport, resident_leases::ResidentSubmission};
use std::collections::BTreeMap;
use zerogravity_core::paged_state::{
    PageAppendPlan, PageCompatibility, PageGeometry, PagedBlockTable,
};

#[derive(Clone)]
pub(crate) struct ResidentPagedTable {
    pool: Arc<ResidentPagePool>,
    table: PagedBlockTable<ResidentLease>,
}

pub(crate) struct ResidentPagedAppend {
    candidate: ResidentPagedTable,
    plan: PageAppendPlan<ResidentLease>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PagedMaterialization {
    pub(crate) zero_uploads: usize,
    pub(crate) copy_dispatches: usize,
    pub(crate) append_uploads: usize,
    pub(crate) append_bytes: usize,
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
    pub(crate) fn fork_for_append(&self, bytes: usize) -> Result<ResidentPagedAppend> {
        let (table, plan) = self.table.fork_for_append(bytes, || self.pool.checkout())?;
        Ok(ResidentPagedAppend {
            candidate: Self {
                pool: self.pool.clone(),
                table,
            },
            plan,
        })
    }
}

impl ResidentPagedAppend {
    pub(crate) fn candidate(&self) -> &ResidentPagedTable {
        &self.candidate
    }

    pub(crate) fn plan(&self) -> &PageAppendPlan<ResidentLease> {
        &self.plan
    }

    pub(crate) fn into_candidate(self) -> ResidentPagedTable {
        self.candidate
    }

    fn target_pages(
        &self,
    ) -> Result<BTreeMap<zerogravity_core::device_events::BufferLeaseId, ResidentLease>> {
        let mut targets = BTreeMap::new();
        for write in self.plan.writes() {
            let lease = write.target();
            if lease.bytes() != self.candidate.geometry().page_bytes()
                || lease.role() != BufferRole::RecurrentState
                || lease.is_initialized()
            {
                return Err(Error::State(
                    "paged append target is not one fresh exact resident page".into(),
                ));
            }
            if let Some(existing) = targets.insert(lease.id(), lease.clone())
                && !same_resident_lease(&existing, lease)
            {
                return Err(Error::State(
                    "paged append target identity aliases another owner".into(),
                ));
            }
        }
        for copy in self.plan.copies() {
            let target = copy.target();
            if !targets
                .get(&target.id())
                .is_some_and(|lease| same_resident_lease(lease, target))
            {
                return Err(Error::State(
                    "paged COW target is absent from append writes".into(),
                ));
            }
        }
        if targets.is_empty() {
            return Err(Error::State(
                "paged append has no fresh target pages".into(),
            ));
        }
        Ok(targets)
    }

    pub(crate) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let targets = self.target_pages()?;
        let mut accesses = BTreeMap::new();
        for copy in self.plan.copies() {
            let source = copy.source();
            if !source.is_initialized()
                || source.role() != BufferRole::RecurrentState
                || source.bytes() != self.candidate.geometry().page_bytes()
                || copy.bytes() == 0
                || copy.bytes() > source.bytes()
                || !Arc::ptr_eq(source.service(), copy.target().service())
            {
                return Err(Error::State(
                    "paged COW source/target ownership or extent differs".into(),
                ));
            }
            if accesses
                .insert(source.id(), (source.clone(), DeviceAccess::Read))
                .is_some()
            {
                return Err(Error::State("paged COW source repeated in one plan".into()));
            }
        }
        for (_, target) in targets {
            if accesses
                .insert(target.id(), (target, DeviceAccess::Write))
                .is_some()
            {
                return Err(Error::State(
                    "paged append target aliases a COW source".into(),
                ));
            }
        }
        Ok(accesses.into_values().collect())
    }

    pub(crate) fn outputs(&self) -> Result<Vec<ResidentLease>> {
        Ok(self.target_pages()?.into_values().collect())
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn enqueue_cuda(
        &self,
        flight: &ResidentSubmission<'_>,
        append: &[u8],
    ) -> Result<PagedMaterialization> {
        let expected = self
            .plan
            .final_bytes()
            .checked_sub(self.plan.prior_bytes())
            .ok_or_else(|| Error::State("paged append plan frontier regressed".into()))?;
        if append.len() != expected || self.candidate.logical_bytes() != self.plan.final_bytes() {
            return Err(Error::State(
                "paged append payload/table extent differs from plan".into(),
            ));
        }
        let accesses = self.accesses()?;
        flight.require_accesses(accesses.iter().map(|(lease, access)| (lease, *access)))?;
        let targets = self.target_pages()?;
        let page_bytes = self.candidate.geometry().page_bytes();
        let service = targets
            .values()
            .next()
            .expect("nonempty target pages")
            .service();

        // Correctness-first materialization: clear every fresh page before any
        // copy/write. A future direct paged kernel may replace this with bounded
        // tail clearing only after its own inactive-tail qualification.
        for target in targets.values() {
            service.client().write(
                &target.handle(),
                Bytes::from_bytes_vec(vec![0u8; page_bytes]),
            );
        }

        let mut copy_dispatches = 0usize;
        for copy in self.plan.copies() {
            if !copy.bytes().is_multiple_of(4) {
                return Err(Error::Unsupported(
                    "CUDA paged COW copy requires word-aligned extent".into(),
                ));
            }
            let words = copy.bytes() / 4;
            let transport =
                RuntimeWordTransport::copy(copy.bytes(), copy.bytes(), 1, words, words, words)?;
            let source = prefix_handle(copy.source(), copy.bytes())?;
            let target = prefix_handle(copy.target(), copy.bytes())?;
            transport
                .kernel
                .prepare(service.client(), transport.grid, vec![source, target])?
                .launch()?;
            copy_dispatches += 1;
        }

        let mut cursor = 0usize;
        for write in self.plan.writes() {
            let end = cursor
                .checked_add(write.bytes())
                .ok_or_else(|| Error::Limit("paged append payload cursor overflow".into()))?;
            let payload = append
                .get(cursor..end)
                .ok_or_else(|| Error::State("paged append write exceeds payload".into()))?;
            let handle = range_handle(write.target(), write.offset(), write.bytes())?;
            service
                .client()
                .write(&handle, Bytes::from_bytes_vec(payload.to_vec()));
            cursor = end;
        }
        if cursor != append.len() {
            return Err(Error::State(
                "paged append plan did not consume complete payload".into(),
            ));
        }

        Ok(PagedMaterialization {
            zero_uploads: targets.len(),
            copy_dispatches,
            append_uploads: self.plan.writes().len(),
            append_bytes: append.len(),
        })
    }
}

fn same_resident_lease(a: &ResidentLease, b: &ResidentLease) -> bool {
    a.id() == b.id()
        && a.allocation_generation() == b.allocation_generation()
        && a.bytes() == b.bytes()
        && Arc::ptr_eq(a.service(), b.service())
}

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
fn prefix_handle(lease: &ResidentLease, bytes: usize) -> Result<Handle> {
    range_handle(lease, 0, bytes)
}

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
fn range_handle(lease: &ResidentLease, offset: usize, bytes: usize) -> Result<Handle> {
    let end = offset
        .checked_add(bytes)
        .filter(|end| *end <= lease.bytes())
        .ok_or_else(|| Error::State("paged resident range outside page".into()))?;
    if bytes == 0 {
        return Err(Error::State("paged resident range is empty".into()));
    }
    let mut handle = lease.handle();
    if offset != 0 {
        handle = handle.offset_start(offset as u64);
    }
    let tail = lease.bytes() - end;
    if tail != 0 {
        handle = handle.offset_end(tail as u64);
    }
    if handle.size_in_used() != bytes as u64 {
        return Err(Error::State(
            "paged resident range handle extent differs".into(),
        ));
    }
    Ok(handle)
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
            let first = empty.fork_for_append(page_bytes - alignment)?;
            if first.candidate().page_count() != 1
                || first.candidate().logical_bytes() != page_bytes - alignment
                || first.plan().copies().len() != 0
                || first.plan().writes().len() != 1
            {
                return Err(Error::State(
                    "paged-state initial partial page plan differs".into(),
                ));
            }
            let first_generation = first
                .candidate()
                .page(0)
                .ok_or_else(|| Error::State("paged-state first page absent".into()))?
                .0
                .allocation_generation();

            let second = first.candidate().fork_for_append(2 * alignment)?;
            if second.candidate().page_count() != 2
                || second.candidate().logical_bytes() != page_bytes + alignment
                || second.plan().copies().len() != 1
                || second.plan().copies()[0].bytes() != page_bytes - alignment
                || second.plan().writes().len() != 2
            {
                return Err(Error::State(
                    "paged-state partial-tail COW plan differs".into(),
                ));
            }
            let generations = (0..second.candidate().page_count())
                .map(|index| {
                    second
                        .candidate()
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
            let full = second.candidate().pool_snapshot()?;
            if full.allocated_pages != 3
                || full.idle_pages != 0
                || full.backing_bytes != max_pages * page_bytes
            {
                return Err(Error::State(
                    "paged-state finite physical pressure accounting differs".into(),
                ));
            }
            let before_pressure = (
                second.candidate().logical_bytes(),
                second.candidate().page_count(),
            );
            if second.candidate().fork_for_append(page_bytes).is_ok()
                || (
                    second.candidate().logical_bytes(),
                    second.candidate().page_count(),
                ) != before_pressure
            {
                return Err(Error::State(
                    "paged-state exhaustion mutated prior block table".into(),
                ));
            }

            let reader = second.candidate().clone();
            drop(first);
            drop(second);
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

    #[test]
    #[ignore = "physical CUDA paged-state COW copy/write materialization exactness"]
    fn physical_page_append_materializes_copy_write_and_zero_tail() -> Result<()> {
        futures_lite::future::block_on(async {
            let diagnostic = CudaDiagnostic::new("paged-state-materialize")?;
            let service = diagnostic.service.clone();
            let alignment = usize::try_from(service.client().properties().memory.alignment)
                .map_err(|_| Error::Limit("paged materialization alignment".into()))?;
            let page_bytes = alignment
                .checked_mul(4)
                .ok_or_else(|| Error::Limit("paged materialization page extent".into()))?;
            if page_bytes as u64 > service.client().properties().memory.max_page_size {
                return Err(Error::Unsupported(
                    "paged materialization page exceeds backend capability".into(),
                ));
            }
            let max_pages = 3usize;
            let metadata_bytes = ResidentPagePool::metadata_bytes(max_pages)?;
            let host_key = PhysicalDeviceKey::new(
                "host",
                "zerogravity-paged-state-materialize-qualification",
            )?;
            let arbiter = ResourceArbiter::process_shared();
            arbiter.register_matching(
                host_key.clone(),
                u64::try_from(metadata_bytes)
                    .map_err(|_| Error::Limit("paged materialization metadata extent".into()))?,
            )?;
            let metadata = arbiter
                .admit(
                    &host_key,
                    ResourceOwner::new("paged-state-materialize", "block-table-metadata")?,
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
            let compatibility = PageCompatibility::new(
                service.source_identity(),
                service.generation(),
                2,
                page_bytes as u64,
            )?;
            let geometry = PageGeometry::new(page_bytes, alignment)?;
            let empty = ResidentPagedTable::empty(pool.clone(), compatibility, geometry)?;

            let first_payload = (0..page_bytes - alignment)
                .map(|index| (index.wrapping_mul(17).wrapping_add(3) % 251) as u8)
                .collect::<Vec<_>>();
            let first = empty.fork_for_append(first_payload.len())?;
            let first_accesses = first.accesses()?;
            let first_flight = service.submit(0, &first_accesses)?;
            let first_materialized = first.enqueue_cuda(&first_flight, &first_payload)?;
            if first_materialized
                != (PagedMaterialization {
                    zero_uploads: 1,
                    copy_dispatches: 0,
                    append_uploads: 1,
                    append_bytes: first_payload.len(),
                })
            {
                return Err(Error::State(
                    "paged first materialization accounting differs".into(),
                ));
            }
            let first_outputs = first.outputs()?;
            let first_readback = first
                .candidate()
                .page(0)
                .ok_or_else(|| Error::State("paged first candidate page absent".into()))?
                .0
                .readback(0, page_bytes)?;
            let mut first_expected = first_payload.clone();
            first_expected.resize(page_bytes, 0);
            let (first_ready, first_bytes) = first_flight
                .complete_with_readback(&first_outputs, &[first_readback], |cohort| {
                    if cohort.len() != 1 || &cohort[0][..] != first_expected.as_slice() {
                        return Err(Error::State(
                            "paged first materialized page/tail differs".into(),
                        ));
                    }
                    Ok(())
                })
                .await?;
            if first_bytes.len() != 1 || first_outputs.iter().any(|lease| !lease.is_initialized()) {
                return Err(Error::State(
                    "paged first materialization did not become physically Ready".into(),
                ));
            }
            drop(first_ready);
            drop(first_bytes);
            drop(first_outputs);
            drop(first_accesses);
            let first_table = first.into_candidate();

            let second_payload = (0..2 * alignment)
                .map(|index| (index.wrapping_mul(29).wrapping_add(11) % 253) as u8)
                .collect::<Vec<_>>();
            let second = first_table.fork_for_append(second_payload.len())?;
            let second_accesses = second.accesses()?;
            let second_flight = service.submit(1, &second_accesses)?;
            let second_materialized = second.enqueue_cuda(&second_flight, &second_payload)?;
            if second_materialized
                != (PagedMaterialization {
                    zero_uploads: 2,
                    copy_dispatches: 1,
                    append_uploads: 2,
                    append_bytes: second_payload.len(),
                })
            {
                return Err(Error::State(
                    "paged COW materialization accounting differs".into(),
                ));
            }
            let second_outputs = second.outputs()?;
            let second_readbacks = (0..second.candidate().page_count())
                .map(|index| {
                    second
                        .candidate()
                        .page(index)
                        .ok_or_else(|| Error::State("paged COW candidate page absent".into()))?
                        .0
                        .readback(0, page_bytes)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut expected0 = first_payload.clone();
            expected0.extend_from_slice(&second_payload[..alignment]);
            let mut expected1 = second_payload[alignment..].to_vec();
            expected1.resize(page_bytes, 0);
            let (second_ready, second_bytes) = second_flight
                .complete_with_readback(&second_outputs, &second_readbacks, |cohort| {
                    if cohort.len() != 2
                        || &cohort[0][..] != expected0.as_slice()
                        || &cohort[1][..] != expected1.as_slice()
                    {
                        return Err(Error::State(
                            "paged COW copy/write/tail materialization differs".into(),
                        ));
                    }
                    Ok(())
                })
                .await?;
            if second
                .candidate()
                .page(0)
                .is_none_or(|(_, valid)| valid != page_bytes)
                || second
                    .candidate()
                    .page(1)
                    .is_none_or(|(_, valid)| valid != alignment)
                || second_outputs.iter().any(|lease| !lease.is_initialized())
            {
                return Err(Error::State(
                    "paged COW valid extents/Ready state differ".into(),
                ));
            }
            drop(second_ready);
            drop(second_bytes);
            drop(second_outputs);
            drop(second_readbacks);
            drop(second_accesses);
            let second_table = second.into_candidate();

            let original = first_table
                .page(0)
                .ok_or_else(|| Error::State("paged original COW source absent".into()))?
                .0
                .clone();
            let original_readback = original.readback(0, page_bytes)?;
            let original_flight = service.submit(1, &[(original.clone(), DeviceAccess::Read)])?;
            original_flight
                .complete_read_only_readback(&[original_readback], |cohort| {
                    if cohort.len() != 1 || &cohort[0][..] != first_expected.as_slice() {
                        return Err(Error::State(
                            "paged COW mutated immutable source page".into(),
                        ));
                    }
                    Ok(())
                })
                .await?;
            drop(original);

            drop(first_table);
            drop(second_table);
            let final_pool = pool.snapshot()?;
            if final_pool.allocated_pages != max_pages || final_pool.idle_pages != max_pages {
                return Err(Error::State(
                    "paged materialization did not retire all page owners".into(),
                ));
            }
            println!(
                "P12_PAGED_STATE_MATERIALIZE_PASS page_bytes={page_bytes} unit_bytes={alignment} first_valid={} cow_copy={} append={} final_valid=[{},{}] source_immutable=true inactive_tail_zero=true all_pages_retired=true",
                first_payload.len(),
                page_bytes - alignment,
                second_payload.len(),
                page_bytes,
                alignment
            );
            Ok(())
        })
    }
}
