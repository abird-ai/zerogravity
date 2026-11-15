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

#[derive(Clone)]
pub(crate) struct ResidentPagedKvDomain {
    slot: StateSlot,
    slab: usize,
    capacity: usize,
    physical: ResidentPhysicalType,
    row_bytes: usize,
    page_tokens: usize,
    keys: ResidentPagedTable,
    values: ResidentPagedTable,
}

pub(crate) struct ResidentPagedKvAppend {
    slot: StateSlot,
    keys: ResidentPagedAppend,
    values: ResidentPagedAppend,
    tokens: usize,
    row_bytes: usize,
}

#[derive(Clone)]
pub(crate) struct ResidentPagedKvStaging {
    requirement: StateRequirement,
    keys: ResidentLease,
    values: ResidentLease,
    buffers: StateBuffers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PagedMaterialization {
    pub(crate) zero_uploads: usize,
    pub(crate) copy_dispatches: usize,
    pub(crate) append_transfers: usize,
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

    pub(crate) fn into_candidate(self) -> Result<ResidentPagedTable> {
        if self.outputs()?.iter().any(|lease| !lease.is_initialized()) {
            return Err(Error::State(
                "paged append candidate is not physically Ready".into(),
            ));
        }
        Ok(self.candidate)
    }

    fn target_pages(
        &self,
    ) -> Result<BTreeMap<zerogravity_core::device_events::BufferLeaseId, ResidentLease>> {
        let mut targets = BTreeMap::new();
        for write in self.plan.writes() {
            let lease = write.target();
            if lease.bytes() != self.candidate.geometry().page_bytes()
                || lease.role() != BufferRole::RecurrentState
            {
                return Err(Error::State(
                    "paged append target is not one exact resident page".into(),
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
            if target.is_initialized() {
                return Err(Error::State(
                    "paged append target was already initialized before submit".into(),
                ));
            }
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
            append_transfers: self.plan.writes().len(),
            append_bytes: append.len(),
        })
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn enqueue_cuda_from_contiguous(
        &self,
        flight: &ResidentSubmission<'_>,
        source: &ResidentLease,
        source_offset: usize,
    ) -> Result<PagedMaterialization> {
        let append_bytes = self
            .plan
            .final_bytes()
            .checked_sub(self.plan.prior_bytes())
            .ok_or_else(|| Error::State("paged scatter plan frontier regressed".into()))?;
        let source_end = source_offset
            .checked_add(append_bytes)
            .filter(|end| *end <= source.bytes())
            .ok_or_else(|| Error::State("paged scatter source range differs".into()))?;
        if source.role() != BufferRole::RecurrentState
            || !Arc::ptr_eq(
                source.service(),
                self.target_pages()?
                    .values()
                    .next()
                    .expect("paged scatter targets")
                    .service(),
            )
            || !source_offset.is_multiple_of(4)
            || !append_bytes.is_multiple_of(4)
            || source_end <= source_offset
        {
            return Err(Error::State(
                "paged scatter source ownership/alignment differs".into(),
            ));
        }
        let accesses = self.accesses()?;
        flight.require_accesses(accesses.iter().map(|(lease, access)| (lease, *access)))?;
        flight.require_accesses([(source, DeviceAccess::Read)])?;
        let targets = self.target_pages()?;
        let page_bytes = self.candidate.geometry().page_bytes();
        let service = targets
            .values()
            .next()
            .expect("nonempty paged scatter targets")
            .service();
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
                    "CUDA paged scatter COW requires word-aligned extent".into(),
                ));
            }
            let words = copy.bytes() / 4;
            let transport =
                RuntimeWordTransport::copy(copy.bytes(), copy.bytes(), 1, words, words, words)?;
            transport
                .kernel
                .prepare(
                    service.client(),
                    transport.grid,
                    vec![
                        prefix_handle(copy.source(), copy.bytes())?,
                        prefix_handle(copy.target(), copy.bytes())?,
                    ],
                )?
                .launch()?;
            copy_dispatches += 1;
        }

        let mut cursor = 0usize;
        for write in self.plan.writes() {
            if !write.bytes().is_multiple_of(4) {
                return Err(Error::Unsupported(
                    "CUDA paged scatter append requires word-aligned extent".into(),
                ));
            }
            let words = write.bytes() / 4;
            let transport =
                RuntimeWordTransport::copy(write.bytes(), write.bytes(), 1, words, words, words)?;
            transport
                .kernel
                .prepare(
                    service.client(),
                    transport.grid,
                    vec![
                        range_handle(source, source_offset + cursor, write.bytes())?,
                        range_handle(write.target(), write.offset(), write.bytes())?,
                    ],
                )?
                .launch()?;
            cursor = cursor
                .checked_add(write.bytes())
                .ok_or_else(|| Error::Limit("paged scatter cursor overflow".into()))?;
        }
        if cursor != append_bytes {
            return Err(Error::State(
                "paged scatter did not cover complete append extent".into(),
            ));
        }
        Ok(PagedMaterialization {
            zero_uploads: targets.len(),
            copy_dispatches,
            append_transfers: self.plan.writes().len(),
            append_bytes,
        })
    }
}

impl ResidentPagedKvDomain {
    pub(crate) fn empty(
        requirement: &StateRequirement,
        pool: Arc<ResidentPagePool>,
        page_tokens: usize,
        numerical: u64,
        schema: u64,
    ) -> Result<Self> {
        let StateSlot::Kv(domain) = requirement.slot else {
            return Err(Error::State(
                "paged KV domain requires a KV state slot".into(),
            ));
        };
        let StateLayout::Kv { slab, capacity } = requirement.layout else {
            return Err(Error::State(
                "paged KV domain requires StateLayout::Kv".into(),
            ));
        };
        let (planes, plane) = requirement.plane_layout()?;
        let plane_bytes = plane.bytes()?;
        let row_bytes = slab
            .checked_mul(requirement.physical.byte_width())
            .ok_or_else(|| Error::Limit("paged KV row byte extent overflow".into()))?;
        let expected_total = plane_bytes
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("paged KV total byte extent overflow".into()))?;
        if planes != 2
            || page_tokens == 0
            || page_tokens > capacity
            || row_bytes == 0
            || row_bytes.checked_mul(capacity) != Some(plane_bytes)
            || requirement.bytes != expected_total
        {
            return Err(Error::State(
                "paged KV state requirement geometry differs".into(),
            ));
        }
        let page_bytes = row_bytes
            .checked_mul(page_tokens)
            .ok_or_else(|| Error::Limit("paged KV page byte extent overflow".into()))?;
        if pool.page_bytes() != page_bytes
            || pool.role() != BufferRole::RecurrentState
            || !row_bytes.is_multiple_of(pool.alignment())
        {
            return Err(Error::State(
                "paged KV pool/page-token geometry differs".into(),
            ));
        }
        let domain_key = u64::try_from(domain.0)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Limit("paged KV domain identity overflow".into()))?;
        let key_layout = schema
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Limit("paged KV key schema identity overflow".into()))?;
        let value_layout = schema
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| Error::Limit("paged KV value schema identity overflow".into()))?;
        let geometry = PageGeometry::new(page_bytes, row_bytes)?;
        let source = pool.service().source_identity();
        let keys = ResidentPagedTable::empty(
            pool.clone(),
            PageCompatibility::new(source, numerical, domain_key, key_layout)?,
            geometry,
        )?;
        let values = ResidentPagedTable::empty(
            pool,
            PageCompatibility::new(source, numerical, domain_key, value_layout)?,
            geometry,
        )?;
        Ok(Self {
            slot: requirement.slot,
            slab,
            capacity,
            physical: requirement.physical,
            row_bytes,
            page_tokens,
            keys,
            values,
        })
    }

    pub(crate) fn slot(&self) -> StateSlot {
        self.slot
    }

    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub(crate) fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    pub(crate) fn tokens(&self) -> Result<usize> {
        let key_bytes = self.keys.logical_bytes();
        if key_bytes != self.values.logical_bytes() || !key_bytes.is_multiple_of(self.row_bytes) {
            return Err(Error::State(
                "paged KV key/value logical frontiers differ".into(),
            ));
        }
        Ok(key_bytes / self.row_bytes)
    }

    pub(crate) fn fork_for_append(&self, tokens: usize) -> Result<ResidentPagedKvAppend> {
        if tokens == 0
            || self
                .tokens()?
                .checked_add(tokens)
                .is_none_or(|n| n > self.capacity)
        {
            return Err(Error::Limit(
                "paged KV append exceeds admitted token capacity".into(),
            ));
        }
        let bytes = tokens
            .checked_mul(self.row_bytes)
            .ok_or_else(|| Error::Limit("paged KV append byte extent overflow".into()))?;
        let keys = self.keys.fork_for_append(bytes)?;
        let values = match self.values.fork_for_append(bytes) {
            Ok(values) => values,
            Err(error) => {
                drop(keys);
                return Err(error);
            }
        };
        Ok(ResidentPagedKvAppend {
            slot: self.slot,
            keys,
            values,
            tokens,
            row_bytes: self.row_bytes,
        })
    }

    pub(crate) fn gather_accesses(
        &self,
        tokens: usize,
        key_staging: &ResidentLease,
        value_staging: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.require_staging(tokens, key_staging, value_staging)?;
        let mut accesses = BTreeMap::new();
        for table in [&self.keys, &self.values] {
            for index in 0..table.page_count() {
                let (page, _) = table
                    .page(index)
                    .ok_or_else(|| Error::State("paged KV gather page absent".into()))?;
                if !page.is_initialized()
                    || accesses
                        .insert(page.id(), (page.clone(), DeviceAccess::Read))
                        .is_some()
                {
                    return Err(Error::State(
                        "paged KV gather page ownership/identity differs".into(),
                    ));
                }
            }
        }
        for staging in [key_staging, value_staging] {
            if accesses
                .insert(staging.id(), (staging.clone(), DeviceAccess::Write))
                .is_some()
            {
                return Err(Error::State(
                    "paged KV staging aliases a committed page".into(),
                ));
            }
        }
        Ok(accesses.into_values().collect())
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn enqueue_gather_cuda(
        &self,
        flight: &ResidentSubmission<'_>,
        tokens: usize,
        key_staging: &ResidentLease,
        value_staging: &ResidentLease,
    ) -> Result<usize> {
        let accesses = self.gather_accesses(tokens, key_staging, value_staging)?;
        flight.require_accesses(accesses.iter().map(|(lease, access)| (lease, *access)))?;
        let plane_bytes = self
            .row_bytes
            .checked_mul(self.capacity)
            .ok_or_else(|| Error::Limit("paged KV staging byte extent overflow".into()))?;
        let service = self.keys.pool.service();
        for staging in [key_staging, value_staging] {
            service.client().write(
                &staging.handle(),
                Bytes::from_bytes_vec(vec![0u8; plane_bytes]),
            );
        }
        let mut dispatches = 0usize;
        for (table, staging) in [(&self.keys, key_staging), (&self.values, value_staging)] {
            let mut offset = 0usize;
            for index in 0..table.page_count() {
                let (page, valid) = table
                    .page(index)
                    .ok_or_else(|| Error::State("paged KV gather page absent".into()))?;
                if !valid.is_multiple_of(4) {
                    return Err(Error::Unsupported(
                        "CUDA paged KV gather requires word-aligned page extent".into(),
                    ));
                }
                let words = valid / 4;
                let transport = RuntimeWordTransport::copy(valid, valid, 1, words, words, words)?;
                transport
                    .kernel
                    .prepare(
                        service.client(),
                        transport.grid,
                        vec![
                            prefix_handle(page, valid)?,
                            range_handle(staging, offset, valid)?,
                        ],
                    )?
                    .launch()?;
                dispatches += 1;
                offset = offset
                    .checked_add(valid)
                    .ok_or_else(|| Error::Limit("paged KV gather cursor overflow".into()))?;
            }
            if offset
                != tokens
                    .checked_mul(self.row_bytes)
                    .ok_or_else(|| Error::Limit("paged KV gather frontier overflow".into()))?
            {
                return Err(Error::State(
                    "paged KV gather copied a different logical frontier".into(),
                ));
            }
        }
        Ok(dispatches)
    }

    fn require_staging(
        &self,
        tokens: usize,
        key_staging: &ResidentLease,
        value_staging: &ResidentLease,
    ) -> Result<()> {
        if tokens != self.tokens()? || tokens > self.capacity {
            return Err(Error::State(
                "paged KV gather frontier differs from committed domain".into(),
            ));
        }
        let plane_bytes = self
            .slab
            .checked_mul(self.capacity)
            .and_then(|n| n.checked_mul(self.physical.byte_width()))
            .ok_or_else(|| Error::Limit("paged KV staging geometry overflow".into()))?;
        if [key_staging, value_staging].iter().any(|lease| {
            lease.bytes() != plane_bytes
                || lease.role() != BufferRole::RecurrentState
                || lease.is_initialized()
                || !Arc::ptr_eq(lease.service(), self.keys.pool.service())
        }) || key_staging.id() == value_staging.id()
        {
            return Err(Error::State(
                "paged KV contiguous staging ownership/layout differs".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn staging(
        &self,
        key: ResidentLease,
        value: ResidentLease,
        access: DeviceAccess,
    ) -> Result<ResidentPagedKvStaging> {
        let requirement = StateRequirement {
            slot: self.slot,
            layout: StateLayout::Kv {
                slab: self.slab,
                capacity: self.capacity,
            },
            physical: self.physical,
            bytes: self
                .row_bytes
                .checked_mul(self.capacity)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| Error::Limit("paged KV staging requirement bytes".into()))?,
        };
        ResidentPagedKvStaging::new(requirement, key, value, access)
    }
}

impl ResidentPagedKvAppend {
    pub(crate) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut accesses = BTreeMap::new();
        for append in [&self.keys, &self.values] {
            for (lease, access) in append.accesses()? {
                if accesses.insert(lease.id(), (lease, access)).is_some() {
                    return Err(Error::State(
                        "paged KV append key/value page identities alias".into(),
                    ));
                }
            }
        }
        Ok(accesses.into_values().collect())
    }

    pub(crate) fn outputs(&self) -> Result<Vec<ResidentLease>> {
        Ok(self
            .keys
            .outputs()?
            .into_iter()
            .chain(self.values.outputs()?)
            .collect())
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn enqueue_cuda(
        &self,
        flight: &ResidentSubmission<'_>,
        keys: &[u8],
        values: &[u8],
    ) -> Result<PagedMaterialization> {
        let expected = self
            .tokens
            .checked_mul(self.row_bytes)
            .ok_or_else(|| Error::Limit("paged KV append payload extent overflow".into()))?;
        if keys.len() != expected || values.len() != expected {
            return Err(Error::State(
                "paged KV key/value append payload extent differs".into(),
            ));
        }
        let key = self.keys.enqueue_cuda(flight, keys)?;
        let value = self.values.enqueue_cuda(flight, values)?;
        Ok(PagedMaterialization {
            zero_uploads: key.zero_uploads + value.zero_uploads,
            copy_dispatches: key.copy_dispatches + value.copy_dispatches,
            append_transfers: key.append_transfers + value.append_transfers,
            append_bytes: key.append_bytes + value.append_bytes,
        })
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn enqueue_cuda_from_staging(
        &self,
        flight: &ResidentSubmission<'_>,
        staging: &ResidentPagedKvStaging,
        prior_tokens: usize,
    ) -> Result<PagedMaterialization> {
        if staging.requirement.slot != self.slot || staging.row_bytes()? != self.row_bytes {
            return Err(Error::State(
                "paged KV scatter staging geometry differs".into(),
            ));
        }
        let source_offset = prior_tokens
            .checked_mul(self.row_bytes)
            .ok_or_else(|| Error::Limit("paged KV scatter source offset overflow".into()))?;
        let key = self
            .keys
            .enqueue_cuda_from_contiguous(flight, staging.keys(), source_offset)?;
        let value =
            self.values
                .enqueue_cuda_from_contiguous(flight, staging.values(), source_offset)?;
        Ok(PagedMaterialization {
            zero_uploads: key.zero_uploads + value.zero_uploads,
            copy_dispatches: key.copy_dispatches + value.copy_dispatches,
            append_transfers: key.append_transfers + value.append_transfers,
            append_bytes: key.append_bytes + value.append_bytes,
        })
    }

    pub(crate) fn into_candidate(
        self,
        previous: &ResidentPagedKvDomain,
    ) -> Result<ResidentPagedKvDomain> {
        let Self {
            slot,
            keys,
            values,
            tokens,
            row_bytes,
        } = self;
        if slot != previous.slot || row_bytes != previous.row_bytes {
            return Err(Error::State("paged KV append row geometry changed".into()));
        }
        let keys = keys.into_candidate()?;
        let values = values.into_candidate()?;
        let result = ResidentPagedKvDomain {
            slot: previous.slot,
            slab: previous.slab,
            capacity: previous.capacity,
            physical: previous.physical,
            row_bytes: previous.row_bytes,
            page_tokens: previous.page_tokens,
            keys,
            values,
        };
        if result.tokens()? != previous.tokens()? + tokens {
            return Err(Error::State(
                "paged KV committed frontier differs after append".into(),
            ));
        }
        Ok(result)
    }
}

impl ResidentPagedKvStaging {
    pub(crate) fn new(
        requirement: StateRequirement,
        keys: ResidentLease,
        values: ResidentLease,
        access: DeviceAccess,
    ) -> Result<Self> {
        let StateLayout::Kv { .. } = requirement.layout else {
            return Err(Error::State(
                "paged KV staging requires StateLayout::Kv".into(),
            ));
        };
        let (planes, physical) = requirement.plane_layout()?;
        if planes != 2
            || keys.id() == values.id()
            || !Arc::ptr_eq(keys.service(), values.service())
            || keys.bytes() != physical.bytes()?
            || values.bytes() != physical.bytes()?
        {
            return Err(Error::State(
                "paged KV staging physical owners/layout differ".into(),
            ));
        }
        let buffers = StateBuffers::Kv {
            keys: DeviceBuffer::from_lease(
                keys.clone(),
                physical,
                BufferRole::RecurrentState,
                access,
            )?,
            values: DeviceBuffer::from_lease(
                values.clone(),
                physical,
                BufferRole::RecurrentState,
                access,
            )?,
        };
        Ok(Self {
            requirement,
            keys,
            values,
            buffers,
        })
    }

    pub(crate) fn buffers(&self) -> StateBuffers {
        self.buffers.clone()
    }

    pub(crate) fn keys(&self) -> &ResidentLease {
        &self.keys
    }

    pub(crate) fn values(&self) -> &ResidentLease {
        &self.values
    }

    pub(crate) fn leases(&self) -> [ResidentLease; 2] {
        [self.keys.clone(), self.values.clone()]
    }

    pub(crate) fn row_bytes(&self) -> Result<usize> {
        let StateLayout::Kv { slab, .. } = self.requirement.layout else {
            unreachable!("validated paged KV staging")
        };
        slab.checked_mul(self.requirement.physical.byte_width())
            .ok_or_else(|| Error::Limit("paged KV staging row bytes overflow".into()))
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
    use zerogravity_core::accelerator::program::KvDomain;
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
                    append_transfers: 1,
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
            let first_table = first.into_candidate()?;

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
                    append_transfers: 2,
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
            let second_table = second.into_candidate()?;

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

    async fn materialize_kv_append(
        service: &Arc<ResidentService>,
        domain: &ResidentPagedKvDomain,
        key_bytes: &[u8],
        value_bytes: &[u8],
    ) -> Result<(ResidentPagedKvDomain, PagedMaterialization)> {
        if key_bytes.len() != value_bytes.len()
            || !key_bytes.len().is_multiple_of(domain.row_bytes())
        {
            return Err(Error::State(
                "paged KV qualifier append payload geometry".into(),
            ));
        }
        let tokens = key_bytes.len() / domain.row_bytes();
        let append = domain.fork_for_append(tokens)?;
        let accesses = append.accesses()?;
        let flight = service.submit(0, &accesses)?;
        let materialized = append.enqueue_cuda(&flight, key_bytes, value_bytes)?;
        let outputs = append.outputs()?;
        let readbacks = outputs
            .iter()
            .map(|lease| lease.readback(0, lease.bytes()))
            .collect::<Result<Vec<_>>>()?;
        let (ready, bytes) = flight
            .complete_with_readback(&outputs, &readbacks, |cohort| {
                if cohort.len() != outputs.len()
                    || cohort
                        .iter()
                        .zip(&outputs)
                        .any(|(bytes, lease)| bytes.len() != lease.bytes())
                {
                    return Err(Error::State(
                        "paged KV qualifier append completion coverage".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        drop(ready);
        drop(bytes);
        drop(readbacks);
        drop(outputs);
        drop(accesses);
        Ok((append.into_candidate(domain)?, materialized))
    }

    async fn gather_kv_exact(
        service: &Arc<ResidentService>,
        domain: &ResidentPagedKvDomain,
        expected_keys: &[u8],
        expected_values: &[u8],
    ) -> Result<usize> {
        if expected_keys.len() != expected_values.len()
            || expected_keys.len() != domain.tokens()? * domain.row_bytes()
        {
            return Err(Error::State(
                "paged KV qualifier gather expected frontier".into(),
            ));
        }
        let plane_bytes = domain
            .row_bytes()
            .checked_mul(domain.capacity)
            .ok_or_else(|| Error::Limit("paged KV qualifier staging extent".into()))?;
        let key_staging =
            service.allocate_uninitialized(plane_bytes, BufferRole::RecurrentState)?;
        let value_staging =
            service.allocate_uninitialized(plane_bytes, BufferRole::RecurrentState)?;
        let accesses = domain.gather_accesses(domain.tokens()?, &key_staging, &value_staging)?;
        let flight = service.submit(0, &accesses)?;
        let dispatches =
            domain.enqueue_gather_cuda(&flight, domain.tokens()?, &key_staging, &value_staging)?;
        let outputs = vec![key_staging.clone(), value_staging.clone()];
        let readbacks = vec![
            key_staging.readback(0, plane_bytes)?,
            value_staging.readback(0, plane_bytes)?,
        ];
        let mut expected_key_plane = expected_keys.to_vec();
        let mut expected_value_plane = expected_values.to_vec();
        expected_key_plane.resize(plane_bytes, 0);
        expected_value_plane.resize(plane_bytes, 0);
        let (ready, data) = flight
            .complete_with_readback(&outputs, &readbacks, |cohort| {
                if cohort.len() != 2
                    || &cohort[0][..] != expected_key_plane.as_slice()
                    || &cohort[1][..] != expected_value_plane.as_slice()
                {
                    return Err(Error::State(
                        "paged KV gather differs from contiguous reference".into(),
                    ));
                }
                Ok(())
            })
            .await?;
        drop(ready);
        drop(data);
        drop(readbacks);
        drop(outputs);
        drop(accesses);
        drop(key_staging);
        drop(value_staging);
        Ok(dispatches)
    }

    #[test]
    #[ignore = "physical CUDA paged KV page-boundary gather-to-contiguous parity"]
    fn physical_paged_kv_boundaries_gather_exactly() -> Result<()> {
        futures_lite::future::block_on(async {
            let diagnostic = CudaDiagnostic::new("paged-kv-boundaries")?;
            let service = diagnostic.service.clone();
            let alignment = usize::try_from(service.client().properties().memory.alignment)
                .map_err(|_| Error::Limit("paged KV allocator alignment".into()))?;
            if !alignment.is_multiple_of(2) {
                return Err(Error::Unsupported(
                    "paged KV BF16 row requires even allocator alignment".into(),
                ));
            }
            const PAGE_TOKENS: usize = 4;
            const CAPACITY: usize = 7;
            let slab = alignment / 2;
            let row_bytes = slab * ResidentPhysicalType::Bf16.byte_width();
            let page_bytes = PAGE_TOKENS * row_bytes;
            if page_bytes as u64 > service.client().properties().memory.max_page_size {
                return Err(Error::Unsupported(
                    "paged KV qualifier page exceeds backend capability".into(),
                ));
            }
            let requirement = StateRequirement {
                slot: StateSlot::Kv(KvDomain(3)),
                layout: StateLayout::Kv {
                    slab,
                    capacity: CAPACITY,
                },
                physical: ResidentPhysicalType::Bf16,
                bytes: 2 * slab * CAPACITY * ResidentPhysicalType::Bf16.byte_width(),
            };
            requirement.plane_layout()?;

            let max_pages = 4usize;
            let metadata_bytes = ResidentPagePool::metadata_bytes(max_pages)?;
            let host_key =
                PhysicalDeviceKey::new("host", "zerogravity-paged-kv-boundaries-qualification")?;
            let arbiter = ResourceArbiter::process_shared();
            arbiter.register_matching(
                host_key.clone(),
                u64::try_from(metadata_bytes)
                    .map_err(|_| Error::Limit("paged KV metadata extent".into()))?,
            )?;
            let metadata = arbiter
                .admit(
                    &host_key,
                    ResourceOwner::new("paged-kv-boundaries", "block-table-metadata")?,
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
            let mut domain = ResidentPagedKvDomain::empty(
                &requirement,
                pool.clone(),
                PAGE_TOKENS,
                service.generation(),
                0x51,
            )?;
            if domain.slot() != requirement.slot
                || domain.row_bytes() != row_bytes
                || domain.page_tokens() != PAGE_TOKENS
                || domain.tokens()? != 0
            {
                return Err(Error::State(
                    "paged KV typed requirement binding differs".into(),
                ));
            }

            let token_row = |token: usize, salt: usize| {
                (0..row_bytes)
                    .map(|byte| {
                        ((token * 37).wrapping_add(byte * 13).wrapping_add(salt) % 251) as u8
                    })
                    .collect::<Vec<_>>()
            };
            let mut expected_keys = Vec::new();
            let mut expected_values = Vec::new();
            for token in 0..3 {
                expected_keys.extend(token_row(token, 7));
                expected_values.extend(token_row(token, 113));
            }
            let (next, first_materialized) =
                materialize_kv_append(&service, &domain, &expected_keys, &expected_values).await?;
            if first_materialized.zero_uploads != 2
                || first_materialized.copy_dispatches != 0
                || first_materialized.append_bytes != 2 * 3 * row_bytes
            {
                return Err(Error::State(
                    "paged KV page_size-1 materialization accounting differs".into(),
                ));
            }
            domain = next;
            let gather3 =
                gather_kv_exact(&service, &domain, &expected_keys, &expected_values).await?;

            let prior3 = domain.clone();
            let key4 = token_row(3, 7);
            let value4 = token_row(3, 113);
            let (next, boundary_materialized) =
                materialize_kv_append(&service, &domain, &key4, &value4).await?;
            if boundary_materialized.copy_dispatches != 2
                || boundary_materialized.zero_uploads != 2
                || boundary_materialized.append_bytes != 2 * row_bytes
            {
                return Err(Error::State(
                    "paged KV page boundary COW accounting differs".into(),
                ));
            }
            domain = next;
            expected_keys.extend_from_slice(&key4);
            expected_values.extend_from_slice(&value4);
            let gather4 =
                gather_kv_exact(&service, &domain, &expected_keys, &expected_values).await?;
            let prior3_keys = expected_keys[..3 * row_bytes].to_vec();
            let prior3_values = expected_values[..3 * row_bytes].to_vec();
            gather_kv_exact(&service, &prior3, &prior3_keys, &prior3_values).await?;
            drop(prior3);

            let prior4 = domain.clone();
            let key5 = token_row(4, 7);
            let value5 = token_row(4, 113);
            let (next, beyond_materialized) =
                materialize_kv_append(&service, &domain, &key5, &value5).await?;
            if beyond_materialized.copy_dispatches != 0
                || beyond_materialized.zero_uploads != 2
                || beyond_materialized.append_bytes != 2 * row_bytes
            {
                return Err(Error::State(
                    "paged KV page_size+1 materialization accounting differs".into(),
                ));
            }
            domain = next;
            expected_keys.extend_from_slice(&key5);
            expected_values.extend_from_slice(&value5);
            let gather5 =
                gather_kv_exact(&service, &domain, &expected_keys, &expected_values).await?;
            gather_kv_exact(
                &service,
                &prior4,
                &expected_keys[..4 * row_bytes],
                &expected_values[..4 * row_bytes],
            )
            .await?;
            drop(prior4);

            if domain.tokens()? != 5 || domain.fork_for_append(3).is_ok() {
                return Err(Error::State(
                    "paged KV frontier/capacity gate differs".into(),
                ));
            }
            drop(domain);
            let final_pool = pool.snapshot()?;
            if final_pool.allocated_pages != max_pages || final_pool.idle_pages != max_pages {
                return Err(Error::State(
                    "paged KV boundary qualifier did not retire all pages".into(),
                ));
            }
            println!(
                "P12_PAGED_KV_BOUNDARY_PASS page_tokens={PAGE_TOKENS} row_bytes={row_bytes} boundaries=3,4,5 gather_dispatches={gather3},{gather4},{gather5} cow_at_boundary=2 full_page_share=true prior_immutable=true inactive_tail_zero=true capacity_refusal=true all_pages_retired=true"
            );
            Ok(())
        })
    }

    #[test]
    #[ignore = "physical CUDA paged KV gather-contiguous-scatter bridge"]
    fn physical_paged_kv_gather_scatter_bridge_is_gpu_only() -> Result<()> {
        futures_lite::future::block_on(async {
            let diagnostic = CudaDiagnostic::new("paged-kv-gather-scatter")?;
            let service = diagnostic.service.clone();
            let alignment = usize::try_from(service.client().properties().memory.alignment)
                .map_err(|_| Error::Limit("paged KV bridge alignment".into()))?;
            if !alignment.is_multiple_of(2) {
                return Err(Error::Unsupported(
                    "paged KV bridge BF16 row requires even alignment".into(),
                ));
            }
            const PAGE_TOKENS: usize = 4;
            const CAPACITY: usize = 7;
            let slab = alignment / 2;
            let row_bytes = alignment;
            let page_bytes = PAGE_TOKENS * row_bytes;
            let plane_bytes = CAPACITY * row_bytes;
            let requirement = StateRequirement {
                slot: StateSlot::Kv(KvDomain(5)),
                layout: StateLayout::Kv {
                    slab,
                    capacity: CAPACITY,
                },
                physical: ResidentPhysicalType::Bf16,
                bytes: 2 * plane_bytes,
            };
            requirement.plane_layout()?;
            let max_pages = 4usize;
            let metadata_bytes = ResidentPagePool::metadata_bytes(max_pages)?;
            let host_key =
                PhysicalDeviceKey::new("host", "zerogravity-paged-kv-bridge-qualification")?;
            let arbiter = ResourceArbiter::process_shared();
            arbiter.register_matching(
                host_key.clone(),
                u64::try_from(metadata_bytes)
                    .map_err(|_| Error::Limit("paged KV bridge metadata".into()))?,
            )?;
            let metadata = arbiter
                .admit(
                    &host_key,
                    ResourceOwner::new("paged-kv-bridge", "block-table-metadata")?,
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
            let empty = ResidentPagedKvDomain::empty(
                &requirement,
                pool.clone(),
                PAGE_TOKENS,
                service.generation(),
                0x61,
            )?;
            let row = |token: usize, salt: usize| {
                (0..row_bytes)
                    .map(|byte| ((token * 31 + byte * 7 + salt) % 251) as u8)
                    .collect::<Vec<_>>()
            };
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for token in 0..3 {
                keys.extend(row(token, 3));
                values.extend(row(token, 109));
            }
            let (domain3, _) = materialize_kv_append(&service, &empty, &keys, &values).await?;
            let key4 = row(3, 3);
            let value4 = row(3, 109);
            let append = domain3.fork_for_append(1)?;

            let prior_key =
                service.allocate_uninitialized(plane_bytes, BufferRole::RecurrentState)?;
            let prior_value =
                service.allocate_uninitialized(plane_bytes, BufferRole::RecurrentState)?;
            let candidate_key =
                service.allocate_uninitialized(plane_bytes, BufferRole::RecurrentState)?;
            let candidate_value =
                service.allocate_uninitialized(plane_bytes, BufferRole::RecurrentState)?;
            let prior_staging =
                domain3.staging(prior_key.clone(), prior_value.clone(), DeviceAccess::Write)?;
            let candidate_staging = domain3.staging(
                candidate_key.clone(),
                candidate_value.clone(),
                DeviceAccess::Write,
            )?;
            let StateBuffers::Kv {
                keys: candidate_key_buffer,
                values: candidate_value_buffer,
            } = candidate_staging.buffers()
            else {
                return Err(Error::State("paged KV bridge staging ABI differs".into()));
            };
            if candidate_key_buffer.original_lease()?.id() != candidate_key.id()
                || candidate_value_buffer.original_lease()?.id() != candidate_value.id()
            {
                return Err(Error::State(
                    "paged KV bridge did not expose original contiguous ABI".into(),
                ));
            }

            let mut accesses = BTreeMap::new();
            let mut insert = |lease: ResidentLease, access: DeviceAccess| -> Result<()> {
                match accesses.entry(lease.id()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((lease, access));
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        if !same_resident_lease(&entry.get().0, &lease) {
                            return Err(Error::State(
                                "paged KV bridge access identity alias".into(),
                            ));
                        }
                        if access == DeviceAccess::Write {
                            entry.get_mut().1 = DeviceAccess::Write;
                        }
                    }
                }
                Ok(())
            };
            for (lease, access) in
                domain3.gather_accesses(3, prior_staging.keys(), prior_staging.values())?
            {
                insert(lease, access)?;
            }
            for (lease, access) in append.accesses()? {
                insert(lease, access)?;
            }
            for lease in candidate_staging.leases() {
                insert(lease, DeviceAccess::Write)?;
            }
            let accesses = accesses.into_values().collect::<Vec<_>>();
            let flight = service.submit(0, &accesses)?;
            let gather_dispatches = domain3.enqueue_gather_cuda(
                &flight,
                3,
                prior_staging.keys(),
                prior_staging.values(),
            )?;

            for lease in candidate_staging.leases() {
                service.client().write(
                    &lease.handle(),
                    Bytes::from_bytes_vec(vec![0u8; plane_bytes]),
                );
            }
            let prefix_bytes = 3 * row_bytes;
            for (source, target) in [
                (prior_staging.keys(), candidate_staging.keys()),
                (prior_staging.values(), candidate_staging.values()),
            ] {
                let words = prefix_bytes / 4;
                let transport =
                    RuntimeWordTransport::copy(prefix_bytes, prefix_bytes, 1, words, words, words)?;
                transport
                    .kernel
                    .prepare(
                        service.client(),
                        transport.grid,
                        vec![
                            prefix_handle(source, prefix_bytes)?,
                            prefix_handle(target, prefix_bytes)?,
                        ],
                    )?
                    .launch()?;
            }
            service.client().write(
                &range_handle(candidate_staging.keys(), prefix_bytes, row_bytes)?,
                Bytes::from_bytes_vec(key4.clone()),
            );
            service.client().write(
                &range_handle(candidate_staging.values(), prefix_bytes, row_bytes)?,
                Bytes::from_bytes_vec(value4.clone()),
            );
            let scatter = append.enqueue_cuda_from_staging(&flight, &candidate_staging, 3)?;
            if gather_dispatches != 2
                || scatter.copy_dispatches != 2
                || scatter.append_transfers != 2
                || scatter.append_bytes != 2 * row_bytes
            {
                return Err(Error::State(
                    "paged KV bridge gather/scatter accounting differs".into(),
                ));
            }
            let outputs = append.outputs()?;
            let readbacks = outputs
                .iter()
                .map(|lease| lease.readback(0, lease.bytes()))
                .collect::<Result<Vec<_>>>()?;
            let (ready, data) = flight
                .complete_with_readback(&outputs, &readbacks, |cohort| {
                    if cohort.len() != outputs.len() {
                        return Err(Error::State("paged KV bridge completion coverage".into()));
                    }
                    Ok(())
                })
                .await?;
            drop(ready);
            drop(data);
            drop(readbacks);
            drop(outputs);
            drop(accesses);
            let domain4 = append.into_candidate(&domain3)?;
            keys.extend_from_slice(&key4);
            values.extend_from_slice(&value4);
            gather_kv_exact(&service, &domain4, &keys, &values).await?;
            gather_kv_exact(
                &service,
                &domain3,
                &keys[..3 * row_bytes],
                &values[..3 * row_bytes],
            )
            .await?;
            drop(prior_staging);
            drop(candidate_staging);
            drop(prior_key);
            drop(prior_value);
            drop(candidate_key);
            drop(candidate_value);
            drop(domain3);
            drop(domain4);
            drop(empty);
            let final_pool = pool.snapshot()?;
            if final_pool.allocated_pages != max_pages || final_pool.idle_pages != max_pages {
                return Err(Error::State(
                    "paged KV bridge did not retire page pool".into(),
                ));
            }
            println!(
                "P12_PAGED_KV_BRIDGE_PASS prior=3 candidate=4 page_tokens={PAGE_TOKENS} row_bytes={row_bytes} gather_dispatches={gather_dispatches} prefix_copy_dispatches=2 scatter_cow_dispatches={} scatter_append_transfers={} gpu_only_after_input=true prior_immutable=true contiguous_abi_exact=true all_pages_retired=true",
                scatter.copy_dispatches, scatter.append_transfers
            );
            Ok(())
        })
    }
}
