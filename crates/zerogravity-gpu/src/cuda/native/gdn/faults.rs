//! Actual original-ledger refusal and armed-flight abandonment; no recovery.
use super::*;
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use std::sync::Weak;
use zerogravity_core::resource_arbiter::PhysicalDeviceKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnFault {
    AdmissionPressure,
    AbandonAfterLaunch,
}

/// Host observations only. UNKNOWN_RETAINED is neither completion nor Ready;
/// a fresh process must exit without any synchronize/unload/recovery attempt.
#[derive(Clone, Copy, Debug)]
pub struct GdnFaultOutcome {
    pub fault: GdnFault,
    pub status: &'static str,
    pub submitted_calls: usize,
    pub gdn_launches: usize,
    pub compared_words: usize,
    /// Mathematical producer completion only; preparation barriers precede it.
    pub ready_completions: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_retained: bool,
    pub owner_retained: bool,
    pub call_retained: bool,
    pub prepared_retained: bool,
    pub service_retained: bool,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
}

/// SAME preparation/launch path as qualify_gdn. `cancelled` is checked before
/// even workspace/fixture admission; ROOT retains its own host baseline.
/// AbandonAfterLaunch intentionally poisons the original service until exit.
pub async fn qualify_gdn_fault(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [GdnFixture; 5],
    device_cap: u64,
    fault: GdnFault,
    cancelled: Option<&AtomicBool>,
) -> Result<GdnFaultOutcome> {
    match qualify_gdn_inner(
        device,
        source,
        fixtures,
        device_cap,
        Mode::Fault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::Fault(observed) => Ok(observed),
        Outcome::Compared(_) => Err(Error::State("GDN fault returned comparison outcome".into())),
    }
}

pub(super) struct PressureWitness {
    key: PhysicalDeviceKey,
    source: Weak<VerifiedTensorSource>,
    owner: Weak<GdnOwner>,
    service: Weak<ResidentService>,
}

pub(super) fn admission_pressure(
    source: &Arc<VerifiedTensorSource>,
    owner: &Arc<GdnOwner>,
    service: &Arc<ResidentService>,
) -> Result<PressureWitness> {
    let key = service.physical_key();
    let before = ResourceArbiter::process_shared().snapshot(&key)?;
    if before.fenced
        || before.reserved != PROVIDER_CAP as u64
        || before.reservations != 1
        || before.per_class.len() != 1
        || before.per_class.get(&AllocationClass::ProviderEnvelope) != Some(&(PROVIDER_CAP as u64))
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
    {
        return Err(Error::State(
            "GDN pressure requires fresh original envelope".into(),
        ));
    }
    // Actual SAME ledger must refuse BEFORE client.empty; no fake lease/handle.
    let refusal = service.allocate_uninitialized(PROVIDER_CAP + 1, BufferRole::Scratch);
    if !matches!(refusal, Err(Error::Limit(_))) {
        return Err(Error::State(
            "GDN original managed admission did not refuse pressure".into(),
        ));
    }
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "GDN pressure allocated/uploaded/retained work".into(),
        ));
    }
    service.ensure_usable()?;
    Ok(PressureWitness {
        key,
        source: Arc::downgrade(source),
        owner: Arc::downgrade(owner),
        service: Arc::downgrade(service),
    })
}

pub(super) fn pressure_retired(witness: PressureWitness) -> Result<GdnFaultOutcome> {
    let snapshot = ResourceArbiter::process_shared().snapshot(&witness.key)?;
    if witness.source.upgrade().is_some()
        || witness.owner.upgrade().is_some()
        || witness.service.upgrade().is_some()
        || snapshot.reserved != 0
        || snapshot.reservations != 0
        || !snapshot.per_class.is_empty()
        || snapshot.fenced
    {
        return Err(Error::State(
            "GDN pressure original envelope not retired unfenced".into(),
        ));
    }
    Ok(GdnFaultOutcome {
        fault: GdnFault::AdmissionPressure,
        status: "ADMISSION_REFUSED_BEFORE_H2D",
        submitted_calls: 0,
        gdn_launches: 0,
        compared_words: 0,
        ready_completions: 0,
        resident_bytes: 0,
        pending_events: 0,
        uploaded_value_bytes: 0,
        quarantined_handles: 0,
        quarantined_completion_owners: 0,
        source_retained: false,
        owner_retained: false,
        call_retained: false,
        prepared_retained: false,
        service_retained: false,
        physical_reserved_bytes: snapshot.reserved,
        physical_fenced: snapshot.fenced,
    })
}

pub(super) struct UnknownWitness {
    key: PhysicalDeviceKey,
    source: Weak<VerifiedTensorSource>,
    owner: Weak<GdnOwner>,
    call: Weak<Call>,
    prepared: Weak<GdnPrepared>,
    service: Weak<ResidentService>,
    resident_bytes: usize,
    uploaded_value_bytes: u64,
}
impl UnknownWitness {
    pub(super) fn new(
        source: &Arc<VerifiedTensorSource>,
        owner: &Arc<GdnOwner>,
        call: &Arc<Call>,
        service: &Arc<ResidentService>,
    ) -> Result<Self> {
        let key = service.physical_key();
        let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
        let resident_bytes = service.resident_bytes()?;
        if call.phase != GdnPhase::InitialOne
            || call.snapshot.is_some()
            || !call.reads.is_empty()
            || service.pending_events()? != 0
            || resident_bytes == 0
            || quarantined_handles_for(service.client())? != 0
            || quarantined_completion_owners_for(service.client())? != 0
            || snapshot.fenced
            || snapshot.reserved != PROVIDER_CAP as u64
            || snapshot.reservations != 1
            || snapshot.per_class.len() != 1
            || snapshot.per_class.get(&AllocationClass::ProviderEnvelope)
                != Some(&(PROVIDER_CAP as u64))
            || !Arc::ptr_eq(source, &owner._source)
        {
            return Err(Error::State(
                "GDN abandonment requires fresh first original call".into(),
            ));
        }
        service.ensure_usable()?;
        Ok(Self {
            key,
            source: Arc::downgrade(source),
            owner: Arc::downgrade(owner),
            call: Arc::downgrade(call),
            prepared: Arc::downgrade(&call.gdn),
            service: Arc::downgrade(service),
            resident_bytes,
            uploaded_value_bytes: service.uploaded_value_bytes(),
        })
    }

    /// Only Weak upgrades and host ledger/quarantine inspection after all
    /// local strong owners drop. No client.sync, readback, launch or unload.
    pub(super) fn inspect(self, launches: usize) -> Result<GdnFaultOutcome> {
        let source = self.source.upgrade();
        let owner = self.owner.upgrade();
        let call = self.call.upgrade();
        let prepared = self.prepared.upgrade();
        let service = self.service.upgrade();
        let retained = [
            source.is_some(),
            owner.is_some(),
            call.is_some(),
            prepared.is_some(),
            service.is_some(),
        ];
        let service =
            service.ok_or_else(|| Error::State("GDN quarantine lost original service".into()))?;
        let resident_bytes = service.resident_bytes()?;
        let pending_events = service.pending_events()?;
        let uploaded_value_bytes = service.uploaded_value_bytes();
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        let snapshot = ResourceArbiter::process_shared().snapshot(&self.key)?;
        if retained.contains(&false)
            || launches != 12
            || service.ensure_usable().is_ok()
            || resident_bytes != self.resident_bytes
            || pending_events != 1
            || uploaded_value_bytes != self.uploaded_value_bytes
            || handles == 0
            || owners < 3
            || !snapshot.fenced
            || snapshot.reserved != PROVIDER_CAP as u64
            || snapshot.reservations != 1
            || snapshot.per_class.len() != 1
            || snapshot.per_class.get(&AllocationClass::ProviderEnvelope)
                != Some(&(PROVIDER_CAP as u64))
        {
            return Err(Error::State(
                "GDN original unknown completion did not retain/fence all owners".into(),
            ));
        }
        Ok(GdnFaultOutcome {
            fault: GdnFault::AbandonAfterLaunch,
            status: "UNKNOWN_RETAINED",
            submitted_calls: 1,
            gdn_launches: launches,
            compared_words: 0,
            ready_completions: 0,
            resident_bytes,
            pending_events,
            uploaded_value_bytes,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            source_retained: retained[0],
            owner_retained: retained[1],
            call_retained: retained[2],
            prepared_retained: retained[3],
            service_retained: retained[4],
            physical_reserved_bytes: snapshot.reserved,
            physical_fenced: snapshot.fenced,
        })
    }
}
