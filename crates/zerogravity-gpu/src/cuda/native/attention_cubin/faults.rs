//! SAME original attention leaf inner; no alternate queue, recovery or Ready.
use super::super::{
    PROVIDER_CAP,
    projection_cubin::{CubinQualificationInputs, resident::PreparedImages},
};
use super::{
    descriptor::AttentionBindings,
    manifests::AttentionCase,
    qualify::{self, Job, Mode, Outcome, Owner},
    resident::{AttentionPrograms, AttentionRetentionWitness},
    weights::AttentionWeights,
};
use crate::{
    physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for},
    resident_leases::ResidentService,
};
use cubecl::Device;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Weak, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::{
    Error, Result,
    device_events::{BufferLeaseId, BufferRole},
    resource_arbiter::{AllocationClass, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CubinAttentionFault {
    #[default]
    Pressure,
    UnknownAfterCopy,
    CancelBeforeCopy,
    UnknownAfterMultiply,
    CancelBeforeMultiply,
}
#[derive(Clone, Debug, Default)]
pub struct CubinAttentionFaultOutcome {
    pub fault: CubinAttentionFault,
    pub status: &'static str,
    pub submitted_calls: usize,
    pub native_launches: usize,
    pub device_copies: usize,
    pub compared_words: usize,
    pub readbacks_executed: usize,
    pub ready_completions: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_retained: bool,
    pub owner_retained: bool,
    pub weights_retained: bool,
    pub job_retained: bool,
    pub programs_retained: bool,
    pub images_retained: bool,
    pub service_retained: bool,
    pub readback_cohort_retained: bool,
    pub readback_descriptors: usize,
    pub submitted_physical_leases: usize,
    pub prepared_physical_leases: usize,
    pub image_owners_retained: usize,
    pub kernel_owners_retained: usize,
    pub native_charge_bytes: usize,
    pub host_workspace_charge_bytes: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub host_final_zero_established: bool,
}
pub async fn qualify_attention_cubin_fault(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<qualify::CubinAttentionFixture>,
    cos_sin: Vec<u8>,
    inputs: CubinQualificationInputs,
    fault: CubinAttentionFault,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinAttentionFaultOutcome> {
    if !matches!(
        fault,
        CubinAttentionFault::Pressure
            | CubinAttentionFault::UnknownAfterCopy
            | CubinAttentionFault::CancelBeforeCopy
    ) {
        return Err(Error::Unsupported(
            "old attention fault scope cannot run gate cuts".into(),
        ));
    }
    match qualify::inner(
        device,
        source,
        fixtures,
        cos_sin,
        inputs,
        Mode::Fault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::Fault(r) => Ok(r),
        Outcome::Compared(_) => Err(Error::State("attention fault returned comparisons".into())),
    }
}
/// Same original scope/flight/quarantine; actual sigmoid/multiply cuts only.
pub async fn qualify_attention_gated_cubin_fault(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<qualify::CubinAttentionGatedFixture>,
    cos_sin: Vec<u8>,
    inputs: CubinQualificationInputs,
    fault: CubinAttentionFault,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinAttentionFaultOutcome> {
    super::super::projection_cubin::not_cancelled(cancelled)?;
    if fixtures.capacity() > 8
        || !matches!(
            fault,
            CubinAttentionFault::Pressure
                | CubinAttentionFault::UnknownAfterMultiply
                | CubinAttentionFault::CancelBeforeMultiply
        )
    {
        return Err(Error::Unsupported(
            "closed computed-gate fault scope/capacity".into(),
        ));
    }
    let fixtures = fixtures
        .into_iter()
        .map(qualify::CubinAttentionGatedFixture::into_common)
        .collect();
    match qualify::inner(
        device,
        source,
        fixtures,
        cos_sin,
        inputs,
        Mode::GateFault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::Fault(result) => Ok(result),
        Outcome::Compared(_) => Err(Error::State("gate fault returned comparisons".into())),
    }
}
fn envelope(s: &DeviceBudgetSnapshot) -> bool {
    s.reserved == PROVIDER_CAP as u64
        && s.reservations == 1
        && s.per_class.len() == 1
        && s.per_class.get(&AllocationClass::ProviderEnvelope) == Some(&(PROVIDER_CAP as u64))
}
pub(super) struct PressureWitness {
    key: PhysicalDeviceKey,
    source: Weak<VerifiedTensorSource>,
    owner: Weak<Owner>,
    service: Weak<ResidentService>,
}
pub(super) fn pressure(
    source: &Arc<VerifiedTensorSource>,
    owner: &Arc<Owner>,
    service: &Arc<ResidentService>,
) -> Result<PressureWitness> {
    let key = service.physical_key();
    let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
    if snapshot.fenced
        || !envelope(&snapshot)
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "attention pressure requires fresh original envelope".into(),
        ));
    }
    if !matches!(
        service.allocate_uninitialized(PROVIDER_CAP + 1, BufferRole::Scratch),
        Err(Error::Limit(_))
    ) {
        return Err(Error::State(
            "attention same managed ledger failed to refuse pressure".into(),
        ));
    }
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "attention pressure allocated/uploaded/native work".into(),
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
impl PressureWitness {
    pub(super) fn retired(self) -> Result<CubinAttentionFaultOutcome> {
        let s = ResourceArbiter::process_shared().snapshot(&self.key)?;
        if self.source.upgrade().is_some()
            || self.owner.upgrade().is_some()
            || self.service.upgrade().is_some()
            || s.reserved != 0
            || s.reservations != 0
            || s.fenced
            || !s.per_class.is_empty()
        {
            return Err(Error::State(
                "attention pressure retained original owners/envelope".into(),
            ));
        }
        Ok(CubinAttentionFaultOutcome {
            fault: CubinAttentionFault::Pressure,
            status: "ADMISSION_REFUSED_BEFORE_H2D",
            ..Default::default()
        })
    }
}
type ReadIdentity = (BufferLeaseId, usize, usize);
fn read_cohort(job: &Job) -> Vec<ReadIdentity> {
    job.reads
        .iter()
        .map(|r| (r.lease_id(), r.offset(), r.bytes()))
        .collect()
}
fn validate_cohort(bindings: &AttentionBindings, reads: &[ReadIdentity]) -> Result<()> {
    let ids = bindings.cohort().into_iter().collect::<BTreeSet<_>>();
    if reads.is_empty() || reads.iter().any(|(id, _, n)| !ids.contains(id) || *n == 0) {
        return Err(Error::State(
            "attention readback must retain original physical cohort".into(),
        ));
    }
    Ok(())
}
fn validate_copy_cut(fault: CubinAttentionFault, launches: usize, copies: usize) -> Result<()> {
    let expected = match fault {
        CubinAttentionFault::UnknownAfterCopy => (4, 1),
        CubinAttentionFault::CancelBeforeCopy => (4, 0),
        CubinAttentionFault::UnknownAfterMultiply => (6, 1),
        CubinAttentionFault::CancelBeforeMultiply => (5, 1),
        _ => return Err(Error::State("attention unknown fault scope".into())),
    };
    if (launches, copies) != expected {
        return Err(Error::State(
            "attention exact native/copy cut boundary".into(),
        ));
    }
    Ok(())
}

pub(super) struct UnknownWitness {
    key: PhysicalDeviceKey,
    source: Weak<VerifiedTensorSource>,
    owner: Weak<Owner>,
    weights: Weak<AttentionWeights>,
    job: Weak<Job>,
    programs: Weak<AttentionPrograms>,
    images: Weak<PreparedImages>,
    service: Weak<ResidentService>,
    native: AttentionRetentionWitness,
    reads: Vec<ReadIdentity>,
    first_ids: BTreeSet<BufferLeaseId>,
    all_ids: BTreeSet<BufferLeaseId>,
    resident_bytes: usize,
    uploaded: u64,
}
impl UnknownWitness {
    pub(super) fn new(
        source: &Arc<VerifiedTensorSource>,
        owner: &Arc<Owner>,
        weights: &Arc<AttentionWeights>,
        jobs: &[Arc<Job>],
        programs: &Arc<AttentionPrograms>,
        images: &Arc<PreparedImages>,
        service: &Arc<ResidentService>,
    ) -> Result<Self> {
        if jobs.len() != 5
            || owner.fixtures.len() != 5
            || jobs[0].bindings.descriptor().case() != AttentionCase::InitialOne
        {
            return Err(Error::State(
                "attention fault exact original first of five cases".into(),
            ));
        }
        let mut allocations = BTreeMap::new();
        for job in jobs {
            for (l, _) in &job.accesses {
                if !Arc::ptr_eq(l.service(), service)
                    || l.source_identity() != source.source_identity()
                {
                    return Err(Error::State(
                        "attention original prepared physical source/service".into(),
                    ));
                }
                if let Some((bytes, initialized)) =
                    allocations.insert(l.id(), (l.bytes(), l.is_initialized()))
                    && (bytes != l.bytes() || initialized != l.is_initialized())
                {
                    return Err(Error::State(
                        "attention shared original allocation identity changed".into(),
                    ));
                }
            }
        }
        let managed = allocations
            .values()
            .try_fold(0usize, |n, (b, _)| n.checked_add(*b))
            .ok_or_else(|| Error::Limit("attention retained managed bytes overflow".into()))?;
        let uploaded = allocations
            .values()
            .filter(|(_, initialized)| *initialized)
            .try_fold(0u64, |n, (b, _)| n.checked_add(*b as u64))
            .ok_or_else(|| Error::Limit("attention actual uploaded bytes overflow".into()))?;
        let key = service.physical_key();
        let s = ResourceArbiter::process_shared().snapshot(&key)?;
        let resident_bytes = service.resident_bytes()?;
        if s.fenced
            || !envelope(&s)
            || resident_bytes != managed + 64 * 1024 * 1024
            || service.pending_events()? != 0
            || service.uploaded_value_bytes() != uploaded
            || quarantined_handles_for(service.client())? != 0
            || quarantined_completion_owners_for(service.client())? != 0
        {
            return Err(Error::State(
                "attention unknown requires exact original prepared graph".into(),
            ));
        }
        let reads = read_cohort(&jobs[0]);
        validate_cohort(&jobs[0].bindings, &reads)?;
        service.ensure_usable()?;
        Ok(Self {
            key,
            source: Arc::downgrade(source),
            owner: Arc::downgrade(owner),
            weights: Arc::downgrade(weights),
            job: Arc::downgrade(&jobs[0]),
            programs: Arc::downgrade(programs),
            images: Arc::downgrade(images),
            service: Arc::downgrade(service),
            native: programs.retention_witness()?,
            reads,
            first_ids: jobs[0].bindings.cohort().into_iter().collect(),
            all_ids: allocations.keys().copied().collect(),
            resident_bytes,
            uploaded,
        })
    }
    /// No native call, synchronization, unload or recovery. All caller-owned
    /// strong references already dropped; temporary host Weak upgrades only.
    pub(super) fn inspect(
        self,
        fault: CubinAttentionFault,
        launches: usize,
        copies: usize,
    ) -> Result<CubinAttentionFaultOutcome> {
        let source = self
            .source
            .upgrade()
            .ok_or_else(|| Error::State("attention source lost from original quarantine".into()))?;
        let owner = self.owner.upgrade().ok_or_else(|| {
            Error::State("attention leaf owner lost from original quarantine".into())
        })?;
        let weights = self.weights.upgrade().ok_or_else(|| {
            Error::State("attention original source W windows lost from quarantine".into())
        })?;
        let job = self.job.upgrade().ok_or_else(|| {
            Error::State("attention original job/readbacks lost from quarantine".into())
        })?;
        let programs = self.programs.upgrade().ok_or_else(|| {
            Error::State("attention original programs lost from quarantine".into())
        })?;
        let images = self
            .images
            .upgrade()
            .ok_or_else(|| Error::State("attention original images lost from quarantine".into()))?;
        let service = self
            .service
            .upgrade()
            .ok_or_else(|| Error::State("attention service lost from quarantine".into()))?;
        let native = self.native.inspect()?;
        let s = ResourceArbiter::process_shared().snapshot(&self.key)?;
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        validate_copy_cut(fault, launches, copies)?;
        validate_cohort(&job.bindings, &self.reads)?;
        if !Arc::ptr_eq(&source, &owner.source)
            || read_cohort(&job) != self.reads
            || job.bindings.cohort().into_iter().collect::<BTreeSet<_>>() != self.first_ids
            || native.image_owners != job.bindings.descriptor().scope().images()
            || native.kernel_owners != job.bindings.descriptor().scope().records()
            || native.native_charge_bytes != 64 * 1024 * 1024
            || native.image_admission_bytes == 0
            || service.ensure_usable().is_ok()
            || service.resident_bytes()? != self.resident_bytes
            || service.pending_events()? != 1
            || service.uploaded_value_bytes() != self.uploaded
            || handles < self.first_ids.len()
            || owners < 4
            || !s.fenced
            || !envelope(&s)
        {
            return Err(Error::State(
                "attention original native/copy quarantine graph differs".into(),
            ));
        }
        // This read is metadata only; the same source W payload, not a copied
        // witness owner, remains held by the original completion-owner graph.
        let host_workspace_charge_bytes =
            owner.host_bytes() + weights.host_bytes() + native.image_admission_bytes;
        drop((programs, images, weights));
        Ok(CubinAttentionFaultOutcome {
            fault,
            status: if matches!(
                fault,
                CubinAttentionFault::UnknownAfterCopy | CubinAttentionFault::UnknownAfterMultiply
            ) {
                "UNKNOWN_RETAINED"
            } else {
                "CANCELLED_UNKNOWN_RETAINED"
            },
            submitted_calls: 1,
            native_launches: launches,
            device_copies: copies,
            compared_words: 0,
            readbacks_executed: 0,
            ready_completions: 0,
            resident_bytes: self.resident_bytes,
            pending_events: 1,
            uploaded_value_bytes: self.uploaded,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            source_retained: true,
            owner_retained: true,
            weights_retained: true,
            job_retained: true,
            programs_retained: true,
            images_retained: true,
            service_retained: true,
            readback_cohort_retained: true,
            readback_descriptors: self.reads.len(),
            submitted_physical_leases: self.first_ids.len(),
            prepared_physical_leases: self.all_ids.len(),
            image_owners_retained: native.image_owners,
            kernel_owners_retained: native.kernel_owners,
            native_charge_bytes: native.native_charge_bytes,
            host_workspace_charge_bytes,
            physical_reserved_bytes: s.reserved,
            physical_fenced: s.fenced,
            host_final_zero_established: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_copy_and_late_cancel_are_distinct_actual_boundaries() {
        assert!(validate_copy_cut(CubinAttentionFault::UnknownAfterCopy, 4, 1).is_ok());
        assert!(validate_copy_cut(CubinAttentionFault::CancelBeforeCopy, 4, 0).is_ok());
        for fault in [
            CubinAttentionFault::UnknownAfterCopy,
            CubinAttentionFault::CancelBeforeCopy,
        ] {
            for n in [0, 1, 3, 5, 30] {
                assert!(validate_copy_cut(fault, n, 0).is_err());
            }
        }
        assert!(validate_copy_cut(CubinAttentionFault::UnknownAfterCopy, 4, 0).is_err());
        assert!(validate_copy_cut(CubinAttentionFault::CancelBeforeCopy, 4, 1).is_err());
        assert!(validate_copy_cut(CubinAttentionFault::Pressure, 4, 0).is_err());
    }
    #[test]
    fn computed_gate_cuts_follow_real_bf16_store_and_preserve_original_copy_cut() {
        assert!(validate_copy_cut(CubinAttentionFault::UnknownAfterMultiply, 6, 1).is_ok());
        assert!(validate_copy_cut(CubinAttentionFault::CancelBeforeMultiply, 5, 1).is_ok());
        for fault in [
            CubinAttentionFault::UnknownAfterMultiply,
            CubinAttentionFault::CancelBeforeMultiply,
        ] {
            for (launches, copies) in [(0, 0), (4, 1), (5, 0), (6, 0), (40, 3)] {
                assert!(validate_copy_cut(fault, launches, copies).is_err());
            }
        }
    }
}
