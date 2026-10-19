//! Persistent physical leases. Ready retires an event, never its live outputs.
//! The engine/binder owns session epochs, immutable roots and fair worker slots.
//! Every raw handle must stay paired with its owning lease while it is used.
use crate::physical_queue::{PhysicalQueue, queue_for};
use crate::physical_service::{PhysicalServiceBudget, PhysicalWork};
use cubecl::{Device, bytes::Bytes, client::Client, server::Handle};
use std::collections::BTreeSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::device_events::{
    BufferLeaseId, BufferRole, DeviceAccess, DeviceEventId, DeviceEventLedger,
};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceOwner};
use zerogravity_core::session_root::RootBinding;

static NEXT_SERVICE_GENERATION: AtomicU64 = AtomicU64::new(1);

fn issue(counter: &AtomicU64) -> Result<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::Limit("resident generation exhausted".into()))
}

/// One prepaid envelope. Every snapshot/lease holds this same service Arc.
pub(crate) struct ResidentService {
    client: Client,
    budget: PhysicalServiceBudget,
    queue: Arc<PhysicalQueue>,
    ledger: Mutex<DeviceEventLedger>,
    source: SourceIdentity,
    owner: ResourceOwner,
    generation: u64,
    next_allocation: AtomicU64,
    uploaded_value_bytes: AtomicU64,
    cap: usize,
    poisoned: AtomicBool,
    retained_forever: AtomicBool,
}

impl ResidentService {
    pub fn new(
        device: Device,
        source: SourceIdentity,
        owner: ResourceOwner,
        provider_cap: usize,
        device_cap: u64,
    ) -> Result<Arc<Self>> {
        if !source.is_bound() || owner.model.is_empty() || owner.session.is_empty() {
            return Err(Error::State(
                "resident service requires bound source/owner".into(),
            ));
        }
        let generation = issue(&NEXT_SERVICE_GENERATION)?;
        let client = device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let budget = PhysicalServiceBudget::attach(
            &client,
            device_cap,
            u64::try_from(provider_cap)
                .map_err(|_| Error::Limit("resident cap overflow".into()))?,
            owner.clone(),
        )?;
        let queue = queue_for(budget.physical_key())?;
        Ok(Arc::new(Self {
            client,
            budget,
            queue,
            ledger: Mutex::new(DeviceEventLedger::new(generation, provider_cap)?),
            source,
            owner,
            generation,
            next_allocation: AtomicU64::new(1),
            uploaded_value_bytes: AtomicU64::new(0),
            cap: provider_cap,
            poisoned: AtomicBool::new(false),
            retained_forever: AtomicBool::new(false),
        }))
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
    pub fn budget(&self) -> &PhysicalServiceBudget {
        &self.budget
    }
    pub fn queue(&self) -> &Arc<PhysicalQueue> {
        &self.queue
    }
    pub fn owner(&self) -> &ResourceOwner {
        &self.owner
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    #[cfg(feature = "qualification")]
    pub fn capacity(&self) -> usize {
        self.cap
    }
    /// Explicit buffer H2D bytes; excludes dispatch/control parameter uniforms.
    #[cfg(feature = "qualification")]
    pub fn uploaded_value_bytes(&self) -> u64 {
        self.uploaded_value_bytes.load(Ordering::Acquire)
    }
    pub fn physical_key(&self) -> PhysicalDeviceKey {
        self.budget.physical_key()
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub fn physical_remaining(&self) -> Result<u64> {
        self.budget.remaining()
    }

    pub fn resident_bytes(&self) -> Result<usize> {
        self.ledger
            .lock()
            .map(|l| l.resident_bytes())
            .map_err(|_| Error::State("resident lease ledger poisoned".into()))
    }
    pub fn remaining_capacity(&self) -> Result<usize> {
        self.cap
            .checked_sub(self.resident_bytes()?)
            .ok_or_else(|| Error::State("resident capacity accounting invalid".into()))
    }
    #[cfg(feature = "qualification")]
    pub fn pending_events(&self) -> Result<usize> {
        self.ledger
            .lock()
            .map(|l| l.pending_events())
            .map_err(|_| Error::State("resident event ledger poisoned".into()))
    }
    pub fn ensure_usable(&self) -> Result<()> {
        if self.poisoned.load(Ordering::Acquire) || self.budget.fence_flag().load(Ordering::Acquire)
        {
            return Err(Error::State("resident service fenced".into()));
        }
        self.budget
            .prepaid_bytes_for(&self.physical_key(), &self.owner)?;
        crate::physical_quarantine::ensure_service_safe(&self.client)
    }

    /// Linearize logical publication against EVERY fence on this physical card.
    /// Caller may only swap prepared payloads/assign root metadata in `publish`.
    /// Displaced leases and captured GPU owners must drop after this returns.
    pub fn publish_if_usable<T>(&self, publish: impl FnOnce() -> Result<T>) -> Result<T> {
        self.ensure_usable()?;
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().with_unfenced(
            &self.physical_key(),
            || {
                if self.poisoned.load(Ordering::Acquire)
                    || self.budget.fence_flag().load(Ordering::Acquire)
                {
                    return Err(Error::State(
                        "resident service fenced before publication".into(),
                    ));
                }
                publish()
            },
        )
    }

    // Keep exactly one service Arc permanently after uncertainty. The existing
    // quarantine owns actual handles; this keepalive owns their coarse charge
    // and poisoned ledger. It introduces no second resource/fence authority.
    fn fence_and_retain(self: &Arc<Self>, leases: &[ResidentLease]) {
        self.poisoned.store(true, Ordering::Release);
        // If no active PhysicalWork owns the envelope, create/drop one before
        // quarantining handles so an unexpected release failure also fences it.
        if let Ok(work) = self.budget.begin(&self.client) {
            drop(work);
        }
        self.budget.fence_flag().store(true, Ordering::Release);
        let handles = leases
            .iter()
            .filter_map(|lease| {
                if lease.0.quarantined.swap(true, Ordering::AcqRel) {
                    None
                } else {
                    Some(lease.0.handle().clone())
                }
            })
            .collect::<Vec<_>>();
        crate::physical_quarantine::quarantine_uncertain(self.client.clone(), handles);
        if !self.retained_forever.swap(true, Ordering::AcqRel) {
            std::mem::forget(self.clone());
        }
    }

    /// Local admission subdivides the existing envelope; no global re-charge.
    /// Empty memory has no valid contents until a successful Write submission.
    pub fn allocate_uninitialized(
        self: &Arc<Self>,
        bytes: usize,
        role: BufferRole,
    ) -> Result<ResidentLease> {
        self.ensure_usable()?;
        let allocation_generation = issue(&self.next_allocation)?;
        let id = self
            .ledger
            .lock()
            .map_err(|_| Error::State("resident lease ledger poisoned".into()))?
            .admit(bytes, role)?;
        let mut admission = PendingAdmission {
            service: self.clone(),
            id,
            armed: true,
        };
        let handle = self.client.empty(bytes);
        let lease = ResidentLease(Arc::new(LeaseInner {
            service: self.clone(),
            id,
            handle: Some(handle),
            bytes,
            role,
            allocation_generation,
            initialized: AtomicBool::new(false),
            quarantined: AtomicBool::new(false),
        }));
        admission.armed = false;
        Ok(lease)
    }

    /// Start before a prepared-program allocator creates ANY handle. Caller
    /// owns the selected, begun fair worker slot. This guard admits and pins
    /// every allocation before create, and retains it before any H2D write.
    pub fn begin_initialization<'a>(
        self: &'a Arc<Self>,
        expected_root: u64,
    ) -> Result<ResidentInitialization<'a>> {
        self.ensure_usable()?;
        let work = self.budget.begin(&self.client)?;
        Ok(ResidentInitialization {
            service: self,
            expected_root,
            events: Vec::new(),
            leases: Vec::new(),
            uploaded: BTreeSet::new(),
            work: Some(work),
            armed: true,
        })
    }

    fn check_lease(self: &Arc<Self>, lease: &ResidentLease) -> Result<()> {
        if !Arc::ptr_eq(self, &lease.0.service)
            || self.generation != lease.service_generation()
            || self.source != lease.source_identity()
            || lease.0.handle().service != self.client.service_id()
        {
            return Err(Error::State(
                "resident lease source/service/generation mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Caller holds the selected fair worker slot and begins it immediately
    /// before this method. Accesses are whole-lease, unique, and retain every
    /// actual source/input/scratch/candidate handle until physical retirement.
    pub fn submit<'a>(
        self: &'a Arc<Self>,
        expected_root: u64,
        accesses: &[(ResidentLease, DeviceAccess)],
    ) -> Result<ResidentSubmission<'a>> {
        self.submit_bound(expected_root, None, accesses)
    }

    /// One physical event pins every row's actual buffers. Preserve complete
    /// source/session/epoch/root bindings before any kernel can be launched.
    pub fn submit_cohort<'a>(
        self: &'a Arc<Self>,
        roots: &[RootBinding],
        accesses: &[(ResidentLease, DeviceAccess)],
    ) -> Result<ResidentSubmission<'a>> {
        let cohort = ResidentCohortRoots::new(self.source, roots)?;
        self.submit_bound(roots[0].root(), Some(cohort), accesses)
    }

    fn submit_bound<'a>(
        self: &'a Arc<Self>,
        expected_root: u64,
        cohort: Option<ResidentCohortRoots>,
        accesses: &[(ResidentLease, DeviceAccess)],
    ) -> Result<ResidentSubmission<'a>> {
        self.ensure_usable()?;
        let mut unique = BTreeSet::new();
        for (lease, access) in accesses {
            self.check_lease(lease)?;
            if !unique.insert(lease.id())
                || (*access == DeviceAccess::Read && !lease.is_initialized())
                || (*access == DeviceAccess::Write
                    && lease.0.role == BufferRole::ImmutableWeights
                    && lease.is_initialized())
            {
                return Err(Error::State(
                    "duplicate/uninitialized/immutable resident access".into(),
                ));
            }
        }
        let event = self
            .ledger
            .lock()
            .map_err(|_| Error::State("resident event ledger poisoned".into()))?
            .submit(
                expected_root,
                &accesses
                    .iter()
                    .map(|(l, a)| (l.id(), *a))
                    .collect::<Vec<_>>(),
            )?;
        let work = match self.budget.begin(&self.client) {
            Ok(work) => work,
            Err(error) => {
                if let Ok(mut ledger) = self.ledger.lock() {
                    let _ = ledger.report_completion_unknown(event);
                }
                self.fence_and_retain(&accesses.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>());
                return Err(error);
            }
        };
        Ok(ResidentSubmission {
            service: self,
            event,
            expected_root,
            cohort,
            accesses: accesses.to_vec(),
            work: Some(work),
            armed: true,
        })
    }
}

/// Immutable metadata for one submitted cohort; never a publication authority.
struct ResidentCohortRoots {
    bindings: Vec<RootBinding>,
}
impl ResidentCohortRoots {
    fn new(source: SourceIdentity, bindings: &[RootBinding]) -> Result<Self> {
        if !source.is_bound() || !matches!(bindings.len(), 2 | 4 | 8) {
            return Err(Error::Limit(
                "resident completion cohort source/size".into(),
            ));
        }
        let frontier = bindings[0].frontier();
        let mut sessions = BTreeSet::new();
        for binding in bindings {
            if binding.source() != source
                || binding.frontier() != frontier
                || !sessions.insert(binding.session())
            {
                return Err(Error::State(
                    "resident completion cohort source/frontier/session mismatch".into(),
                ));
            }
        }
        Ok(Self {
            bindings: bindings.to_vec(),
        })
    }

    fn require_exact(&self, bindings: impl Iterator<Item = RootBinding>) -> Result<()> {
        if self.bindings.iter().copied().ne(bindings) {
            return Err(Error::State(
                "resident completion cohort roots changed or reordered".into(),
            ));
        }
        Ok(())
    }
}

struct PendingAdmission {
    service: Arc<ResidentService>,
    id: BufferLeaseId,
    armed: bool,
}
impl Drop for PendingAdmission {
    fn drop(&mut self) {
        if self.armed {
            let released = self
                .service
                .ledger
                .lock()
                .ok()
                .is_some_and(|mut l| l.release_lease(self.id).is_ok());
            if !released {
                self.service.fence_and_retain(&[]);
            }
        }
    }
}

struct LeaseInner {
    // Rust drops fields in declaration order after Drop: release the physical
    // handle BEFORE the final service Arc can release its prepaid envelope.
    handle: Option<Handle>,
    id: BufferLeaseId,
    bytes: usize,
    role: BufferRole,
    allocation_generation: u64,
    initialized: AtomicBool,
    quarantined: AtomicBool,
    service: Arc<ResidentService>,
}
impl LeaseInner {
    fn handle(&self) -> &Handle {
        self.handle.as_ref().expect("live resident handle")
    }
}
impl Drop for LeaseInner {
    fn drop(&mut self) {
        let released = if self.service.ensure_usable().is_ok() {
            self.service.ledger.lock().ok().is_some_and(|mut ledger| {
                if ledger.release_lease(self.id).is_ok() {
                    // Keep admission locked until the original handle has
                    // dropped, preventing a local allocation-overlap gap.
                    drop(self.handle.take());
                    true
                } else {
                    false
                }
            })
        } else {
            false
        };
        if !released {
            self.service.fence_and_retain(&[]);
            if !self.quarantined.swap(true, Ordering::AcqRel) {
                crate::physical_quarantine::quarantine_uncertain(
                    self.service.client.clone(),
                    vec![self.handle().clone()],
                );
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct ResidentLease(Arc<LeaseInner>);
impl ResidentLease {
    pub fn id(&self) -> BufferLeaseId {
        self.0.id
    }
    pub fn bytes(&self) -> usize {
        self.0.bytes
    }
    pub fn handle(&self) -> Handle {
        self.0.handle().clone()
    }
    pub fn service(&self) -> &Arc<ResidentService> {
        &self.0.service
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.0.service.source
    }
    pub fn service_generation(&self) -> u64 {
        self.0.service.generation
    }
    pub fn allocation_generation(&self) -> u64 {
        self.0.allocation_generation
    }
    pub fn is_initialized(&self) -> bool {
        self.0.initialized.load(Ordering::Acquire)
    }
    pub fn readback(&self, offset: usize, bytes: usize) -> Result<ResidentReadback> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|n| n > self.bytes()) {
            return Err(Error::Malformed(
                "resident readback view outside allocation".into(),
            ));
        }
        Ok(ResidentReadback {
            lease: self.clone(),
            offset,
            bytes,
        })
    }
}

pub(crate) struct ResidentReadback {
    lease: ResidentLease,
    offset: usize,
    bytes: usize,
}
impl ResidentReadback {
    fn handle(&self) -> Handle {
        self.lease
            .handle()
            .offset_start(self.offset as u64)
            .offset_end((self.lease.bytes() - self.offset - self.bytes) as u64)
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[must_use]
pub(crate) struct ResidentSubmission<'a> {
    service: &'a Arc<ResidentService>,
    event: DeviceEventId,
    expected_root: u64,
    cohort: Option<ResidentCohortRoots>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    work: Option<PhysicalWork<'a>>,
    armed: bool,
}
impl ResidentSubmission<'_> {
    #[cfg(feature = "hardware-abort-probe")]
    pub fn handles(&self) -> Vec<Handle> {
        self.accesses.iter().map(|(l, _)| l.handle()).collect()
    }
    fn check_outputs(&self, outputs: &[ResidentLease]) -> Result<()> {
        if outputs.is_empty() {
            return Err(Error::State("resident Ready requires outputs".into()));
        }
        let mut unique = BTreeSet::new();
        for output in outputs {
            self.service.check_lease(output)?;
            if !unique.insert(output.id())
                || !self
                    .accesses
                    .iter()
                    .any(|(l, a)| l.id() == output.id() && *a == DeviceAccess::Write)
            {
                return Err(Error::State(
                    "resident promotion requires unique submitted writes".into(),
                ));
            }
        }
        Ok(())
    }
    fn finish_ready(mut self, outputs: &[ResidentLease]) -> Result<ResidentReady> {
        self.service.ensure_usable()?;
        {
            let mut ledger = self
                .service
                .ledger
                .lock()
                .map_err(|_| Error::State("resident Ready ledger poisoned".into()))?;
            ledger.report_physical_ready(self.event)?;
            ledger.retire_physical_ready(self.event)?; // Retires event pins ONLY.
        }
        self.work
            .take()
            .ok_or_else(|| Error::State("resident physical guard absent".into()))?
            .complete()?;
        for (lease, access) in &self.accesses {
            if *access == DeviceAccess::Write {
                lease.0.initialized.store(true, Ordering::Release);
            }
        }
        self.armed = false;
        Ok(ResidentReady {
            service: self.service.clone(),
            events: vec![self.event],
            expected_root: self.expected_root,
            root_binding: None,
            outputs: outputs.to_vec(),
        })
    }
    /// Actual readback and trusted numerical/status validation occur HERE,
    /// before the private Ready constructor can retire any event pin.
    pub async fn complete_with_readback<F>(
        self,
        outputs: &[ResidentLease],
        cohort: &[ResidentReadback],
        validate: F,
    ) -> Result<(ResidentReady, Vec<Bytes>)>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        if self.cohort.is_some() {
            return Err(Error::State(
                "resident cohort requires all-row completion".into(),
            ));
        }
        self.check_outputs(outputs)?;
        let bytes = self.readback_checked(cohort, validate).await?;
        let ready = self.finish_ready(outputs)?;
        Ok((ready, bytes))
    }

    /// Split a single trusted physical completion into exact original row
    /// proofs. No row can promote a foreign, shared or non-written output.
    pub async fn complete_cohort_with_readback<F>(
        self,
        rows: &[(RootBinding, Vec<ResidentLease>)],
        cohort: &[ResidentReadback],
        validate: F,
    ) -> Result<(Vec<ResidentReady>, Vec<Bytes>)>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        self.cohort
            .as_ref()
            .ok_or_else(|| Error::State("resident submission has no bound cohort".into()))?
            .require_exact(rows.iter().map(|(root, _)| *root))?;
        if rows.iter().any(|(_, outputs)| outputs.is_empty()) {
            return Err(Error::State("resident cohort row has no outputs".into()));
        }
        let outputs = rows
            .iter()
            .flat_map(|(_, outputs)| outputs.iter().cloned())
            .collect::<Vec<_>>();
        self.check_outputs(&outputs)?;
        let bytes = self.readback_checked(cohort, validate).await?;
        let completion = self.finish_ready(&outputs)?;
        let ready = rows
            .iter()
            .map(|(binding, outputs)| ResidentReady {
                service: completion.service.clone(),
                events: completion.events.clone(),
                expected_root: binding.root(),
                root_binding: Some(*binding),
                outputs: outputs.clone(),
            })
            .collect();
        Ok((ready, bytes))
    }
    async fn readback_checked<F>(
        &self,
        cohort: &[ResidentReadback],
        validate: F,
    ) -> Result<Vec<Bytes>>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        if cohort.is_empty() {
            return Err(Error::State(
                "resident completion requires physical readback".into(),
            ));
        }
        for item in cohort {
            self.service.check_lease(&item.lease)?;
            if !self.accesses.iter().any(|(l, _)| l.id() == item.lease.id()) {
                return Err(Error::State("readback lease absent from submission".into()));
            }
        }
        let bytes = self
            .service
            .client
            .read_async(cohort.iter().map(|item| item.handle()).collect())
            .await
            .map_err(|e| Error::State(format!("resident physical readback failed: {e}")))?;
        if bytes.len() != cohort.len() || bytes.iter().zip(cohort).any(|(b, c)| b.len() != c.bytes)
        {
            return Err(Error::State(
                "resident physical readback cohort incomplete".into(),
            ));
        }
        validate(&bytes)?;
        Ok(bytes)
    }
    /// Snapshot/export uses only existing initialized Read leases. It never
    /// fabricates a Write or promotes/initializes any new state allocation.
    pub async fn complete_read_only_readback<F>(
        self,
        cohort: &[ResidentReadback],
        validate: F,
    ) -> Result<Vec<Bytes>>
    where
        F: FnOnce(&[Bytes]) -> Result<()>,
    {
        if self.cohort.is_some()
            || self
                .accesses
                .iter()
                .any(|(lease, access)| *access != DeviceAccess::Read || !lease.is_initialized())
        {
            return Err(Error::State(
                "read-only completion requires initialized Reads".into(),
            ));
        }
        let bytes = self.readback_checked(cohort, validate).await?;
        drop(self.finish_ready(&[])?);
        Ok(bytes)
    }
}
impl Drop for ResidentSubmission<'_> {
    fn drop(&mut self) {
        if self.armed {
            if let Ok(mut ledger) = self.service.ledger.lock() {
                let _ = ledger.report_completion_unknown(self.event);
            }
            self.service.fence_and_retain(
                &self
                    .accesses
                    .iter()
                    .map(|(l, _)| l.clone())
                    .collect::<Vec<_>>(),
            );
            drop(self.work.take());
        }
    }
}

/// Persistent physically Ready outputs. Root/epoch/session/domain validation
/// and publication remain with the sole engine coordinator, never this type.
#[must_use]
pub(crate) struct ResidentReady {
    service: Arc<ResidentService>,
    events: Vec<DeviceEventId>,
    expected_root: u64,
    root_binding: Option<RootBinding>,
    outputs: Vec<ResidentLease>,
}
impl ResidentReady {
    pub fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub fn event_ids(&self) -> &[DeviceEventId] {
        &self.events
    }
    pub fn expected_root(&self) -> u64 {
        self.expected_root
    }
    pub fn root_binding(&self) -> Option<RootBinding> {
        self.root_binding
    }
    pub fn outputs(&self) -> &[ResidentLease] {
        &self.outputs
    }
}

/// Allocation callback owner for model-load/compiled-program preparation.
/// Source and scratch allocations share one existing prepaid physical guard.
#[must_use]
pub(crate) struct ResidentInitialization<'a> {
    service: &'a Arc<ResidentService>,
    expected_root: u64,
    events: Vec<DeviceEventId>,
    leases: Vec<ResidentLease>,
    uploaded: BTreeSet<BufferLeaseId>,
    work: Option<PhysicalWork<'a>>,
    armed: bool,
}
impl ResidentInitialization<'_> {
    pub fn allocate_empty(&mut self, bytes: usize, role: BufferRole) -> Result<ResidentLease> {
        self.service.ensure_usable()?;
        let allocation_generation = issue(&self.service.next_allocation)?;
        let (id, event) = {
            let mut ledger = self
                .service
                .ledger
                .lock()
                .map_err(|_| Error::State("resident initialization ledger poisoned".into()))?;
            let id = ledger.admit(bytes, role)?;
            match ledger.submit(self.expected_root, &[(id, DeviceAccess::Write)]) {
                Ok(event) => (id, event),
                Err(error) => {
                    ledger.release_lease(id)?;
                    return Err(error);
                }
            }
        };
        // Retain the event before create can panic; no write/copy has occurred.
        self.events.push(event);
        let handle = self.service.client.empty(bytes);
        let lease = ResidentLease(Arc::new(LeaseInner {
            handle: Some(handle),
            id,
            bytes,
            role,
            allocation_generation,
            initialized: AtomicBool::new(false),
            quarantined: AtomicBool::new(false),
            service: self.service.clone(),
        }));
        self.leases.push(lease.clone());
        Ok(lease)
    }
    pub fn upload_bytes(&mut self, bytes: &[u8], role: BufferRole) -> Result<ResidentLease> {
        let lease = self.allocate_empty(bytes.len(), role)?;
        // The original handle is already retained by this guard BEFORE H2D.
        self.service
            .client
            .write(&lease.handle(), Bytes::from_bytes_vec(bytes.to_vec()));
        self.service
            .uploaded_value_bytes
            .fetch_add(bytes.len() as u64, Ordering::AcqRel);
        self.uploaded.insert(lease.id());
        Ok(lease)
    }
    pub async fn complete_sync(mut self) -> Result<ResidentReady> {
        if self.events.is_empty() {
            return Err(Error::State("empty resident initialization".into()));
        }
        self.service
            .client
            .sync()
            .await
            .map_err(|e| Error::State(format!("resident preparation barrier failed: {e}")))?;
        self.service.ensure_usable()?;
        {
            let mut ledger =
                self.service.ledger.lock().map_err(|_| {
                    Error::State("resident preparation Ready ledger poisoned".into())
                })?;
            for event in &self.events {
                ledger.report_physical_ready(*event)?;
                ledger.retire_physical_ready(*event)?;
            }
        }
        self.work
            .take()
            .ok_or_else(|| Error::State("resident preparation guard absent".into()))?
            .complete()?;
        for lease in &self.leases {
            if self.uploaded.contains(&lease.id()) {
                lease.0.initialized.store(true, Ordering::Release);
            }
        }
        self.armed = false;
        Ok(ResidentReady {
            service: self.service.clone(),
            events: self.events.clone(),
            expected_root: self.expected_root,
            root_binding: None,
            outputs: self.leases.clone(),
        })
    }
}
impl Drop for ResidentInitialization<'_> {
    fn drop(&mut self) {
        if self.armed {
            if let Ok(mut ledger) = self.service.ledger.lock() {
                for event in &self.events {
                    let _ = ledger.report_completion_unknown(*event);
                }
            }
            self.service.fence_and_retain(&self.leases);
            drop(self.work.take());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::session_root::SessionAuthority;

    fn sessions(source: SourceIdentity, count: usize) -> Vec<SessionAuthority> {
        (0..count)
            .map(|_| SessionAuthority::new(source).unwrap())
            .collect()
    }

    #[test]
    fn completion_cohort_accepts_only_finite_native_row_counts() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let owners = sessions(source, 9);
        let bindings = owners
            .iter()
            .map(SessionAuthority::binding)
            .collect::<Vec<_>>();
        for count in [2, 4, 8] {
            let cohort = ResidentCohortRoots::new(source, &bindings[..count]).unwrap();
            cohort
                .require_exact(bindings[..count].iter().copied())
                .unwrap();
        }
        for count in [0, 1, 3, 5, 6, 7, 9] {
            assert!(ResidentCohortRoots::new(source, &bindings[..count]).is_err());
        }
        assert!(ResidentCohortRoots::new(SourceIdentity::unbound(), &bindings[..2]).is_err());
    }

    #[test]
    fn completion_cohort_rejects_foreign_duplicate_and_differing_frontier() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut owners = sessions(source, 2);
        let foreign = SessionAuthority::new(SourceIdentity::for_new_allocation().unwrap()).unwrap();
        let initial = owners[0].binding();
        assert!(ResidentCohortRoots::new(source, &[initial, initial]).is_err());
        assert!(ResidentCohortRoots::new(source, &[initial, foreign.binding()]).is_err());
        let advance = owners[1].begin_advance(1).unwrap();
        owners[1].prepare_publish(advance).unwrap().commit();
        assert!(ResidentCohortRoots::new(source, &[initial, owners[1].binding()]).is_err());
        assert_eq!(owners[0].binding(), initial);
    }

    #[test]
    fn completion_cohort_preserves_independent_root_epochs_and_exact_order() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut owners = sessions(source, 2);
        let advance = owners[0].begin_advance(1).unwrap();
        owners[0].prepare_publish(advance).unwrap().commit();
        let advance = owners[1].begin_advance(5).unwrap();
        owners[1].prepare_publish(advance).unwrap().commit();
        let reset = owners[1].begin_reset().unwrap();
        owners[1].prepare_publish(reset).unwrap().commit();
        let advance = owners[1].begin_advance(1).unwrap();
        owners[1].prepare_publish(advance).unwrap().commit();
        let bindings = [owners[0].binding(), owners[1].binding()];
        assert_ne!(bindings[0].root(), bindings[1].root());
        assert_ne!(bindings[0].session_epoch(), bindings[1].session_epoch());
        assert_eq!(bindings[0].frontier(), bindings[1].frontier());
        let cohort = ResidentCohortRoots::new(source, &bindings).unwrap();
        cohort.require_exact(bindings.into_iter()).unwrap();
        assert!(cohort.require_exact(bindings.into_iter().rev()).is_err());
        assert!(cohort.require_exact(bindings[..1].iter().copied()).is_err());
        assert!(cohort.require_exact([bindings[0]; 2].into_iter()).is_err());
        let replace = owners[1].begin_validated_replacement(1).unwrap();
        owners[1].prepare_publish(replace).unwrap().commit();
        assert_eq!(owners[1].binding().frontier(), bindings[1].frontier());
        assert!(
            cohort
                .require_exact([bindings[0], owners[1].binding()].into_iter())
                .is_err()
        );
        // Capture and validation never publish any root themselves.
        assert_eq!(owners[0].binding(), bindings[0]);
    }
}
