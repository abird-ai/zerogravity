//! Same real head enqueue/admission path; unknown witnesses hold Weak only.
use super::super::projection_cubin::resident::{NativeRetentionWitness, PreparedImages};
use super::*;
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use std::sync::Weak;
use zerogravity_core::resource_arbiter::PhysicalDeviceKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyHeadFault {
    Pressure,
    UnknownAfterHead,
    LateCancelAfterHead,
}
#[derive(Clone, Copy, Debug)]
pub struct VocabularyHeadFaultOutcome {
    pub fault: VocabularyHeadFault,
    pub scope: VocabularyHeadScope,
    pub status: &'static str,
    pub native_launches: usize,
    pub rms_stages: usize,
    pub producer_ready: usize,
    pub readbacks: usize,
    pub comparisons: usize,
    pub initialized_weight: bool,
    pub weight_copies: usize,
    pub weight_checkpoints: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_retained: bool,
    pub fixture_owner_retained: bool,
    pub job_retained: bool,
    pub weight_retained: bool,
    pub programs_retained: bool,
    pub fused_retained: bool,
    pub service_retained: bool,
    pub readback_cohort_retained: bool,
    pub descriptors: usize,
    pub managed_leases: usize,
    pub image_owners: usize,
    pub kernel_owners: usize,
    pub module_set_retained: bool,
    pub native_charge_bytes: usize,
    pub physical_reserved: u64,
    pub physical_fenced: bool,
    pub host_final_zero_established: bool,
}
pub async fn qualify_vocabulary_head_fault(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    fixtures: VocabularyHeadFixtures,
    inputs: CubinQualificationInputs,
    scope: VocabularyHeadScope,
    fault: VocabularyHeadFault,
    cancelled: Option<&AtomicBool>,
) -> Result<VocabularyHeadFaultOutcome> {
    match qualify::inner(
        device,
        source,
        fixtures,
        inputs,
        scope,
        Mode::Fault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::Fault(outcome) => Ok(outcome),
        Outcome::Compared(_) => Err(Error::State("head fault returned comparisons".into())),
    }
}
pub(super) struct PressureWitness {
    key: PhysicalDeviceKey,
}
pub(super) fn pressure(service: &Arc<ResidentService>) -> Result<PressureWitness> {
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "head admission pressure service not fresh".into(),
        ));
    }
    if !matches!(
        service.allocate_uninitialized(HEAD_CAP + 1, BufferRole::Scratch),
        Err(Error::Limit(_))
    ) {
        return Err(Error::State(
            "same head managed ledger failed cap+1 refusal".into(),
        ));
    }
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
    {
        return Err(Error::State(
            "head pressure allocated/copied/submitted work".into(),
        ));
    }
    service.ensure_usable()?;
    Ok(PressureWitness {
        key: service.physical_key(),
    })
}
pub(super) fn pressure_retired(
    witness: PressureWitness,
    scope: Scope,
) -> Result<VocabularyHeadFaultOutcome> {
    let snapshot = ResourceArbiter::process_shared().snapshot(&witness.key)?;
    if snapshot.reserved != 0
        || snapshot.reservations != 0
        || snapshot.fenced
        || !snapshot.per_class.is_empty()
    {
        return Err(Error::State("head pressure envelope did not retire".into()));
    }
    Ok(VocabularyHeadFaultOutcome {
        fault: VocabularyHeadFault::Pressure,
        scope,
        status: "ADMISSION_REFUSED_BEFORE_H2D_NATIVE",
        native_launches: 0,
        rms_stages: 0,
        producer_ready: 0,
        readbacks: 0,
        comparisons: 0,
        initialized_weight: false,
        weight_copies: 0,
        weight_checkpoints: 0,
        resident_bytes: 0,
        pending_events: 0,
        uploaded_bytes: 0,
        quarantined_handles: 0,
        quarantined_completion_owners: 0,
        source_retained: false,
        fixture_owner_retained: false,
        job_retained: false,
        weight_retained: false,
        programs_retained: false,
        fused_retained: false,
        service_retained: false,
        readback_cohort_retained: false,
        descriptors: 0,
        managed_leases: 0,
        image_owners: 0,
        kernel_owners: 0,
        module_set_retained: false,
        native_charge_bytes: 0,
        physical_reserved: 0,
        physical_fenced: false,
        host_final_zero_established: false,
    })
}
type ReadbackIdentity = (BufferLeaseId, usize, usize);
fn readbacks(job: &Job) -> Vec<ReadbackIdentity> {
    job.reads
        .iter()
        .map(|r| (r.lease_id(), r.offset(), r.bytes()))
        .collect()
}
pub(super) struct UnknownWitness {
    owner: Weak<HeadOwner>,
    source: Weak<OriginalWindowModelSource>,
    job: Weak<Job>,
    weight: Weak<TiedWeight>,
    programs: Weak<HeadPrograms>,
    images: Weak<PreparedImages>,
    fused: Option<Weak<FusedRmsPrepared>>,
    service: Weak<ResidentService>,
    native: NativeRetentionWitness,
    descriptors: Vec<ReadbackIdentity>,
    ids: BTreeSet<BufferLeaseId>,
    resident: usize,
    uploaded: u64,
    scope: Scope,
    fault: VocabularyHeadFault,
}
impl UnknownWitness {
    pub(super) fn new(
        owner: &Arc<HeadOwner>,
        job: &Arc<Job>,
        programs: &Arc<HeadPrograms>,
        weight: &Arc<TiedWeight>,
        service: &Arc<ResidentService>,
        scope: Scope,
        fault: VocabularyHeadFault,
    ) -> Result<Self> {
        if fault == VocabularyHeadFault::Pressure
            || owner.cohort.fixtures.len() != 1
            || job.case != VocabularyHeadCase::Initial11
            || job.fused.is_some() != (scope == Scope::ComputedFinalNorm)
            || job.output.is_initialized()
            || !weight.lease().is_initialized()
        {
            return Err(Error::State(
                "head fault requires one actual initial11 prepared cohort".into(),
            ));
        }
        let ids = job
            .accesses
            .iter()
            .map(|(l, _)| l.id())
            .collect::<BTreeSet<_>>();
        let expected = if scope == Scope::ComputedFinalNorm {
            16
        } else {
            3
        };
        let descriptors = readbacks(job);
        if ids.len() != expected
            || ids.len() != job.accesses.len()
            || descriptors.len()
                != if scope == Scope::ComputedFinalNorm {
                    8
                } else {
                    2
                }
            || descriptors.iter().any(|(id, offset, bytes)| {
                !job.accesses.iter().any(|(l, _)| {
                    l.id() == *id
                        && offset
                            .checked_add(*bytes)
                            .is_some_and(|end| end <= l.bytes())
                })
            })
            || job.accesses.iter().any(|(l, a)| {
                !Arc::ptr_eq(l.service(), service)
                    || l.source_identity() != owner.source.source_identity()
                    || (*a == DeviceAccess::Read && !l.is_initialized())
                    || (*a == DeviceAccess::Write && l.is_initialized())
            })
        {
            return Err(Error::State(
                "head actual original lease/access/readback cohort".into(),
            ));
        }
        let managed = job.accesses.iter().try_fold(0usize, |sum, (l, _)| {
            sum.checked_add(l.bytes())
                .ok_or_else(|| Error::Limit("head witness bytes overflow".into()))
        })?;
        let uploaded = job
            .accesses
            .iter()
            .filter(|(_, a)| *a == DeviceAccess::Read)
            .try_fold(0u64, |sum, (l, _)| {
                sum.checked_add(l.bytes() as u64)
                    .ok_or_else(|| Error::Limit("head uploaded bytes overflow".into()))
            })?;
        if service.resident_bytes()? != managed + NATIVE_BYTES
            || service.pending_events()? != 0
            || service.uploaded_value_bytes() != uploaded
        {
            return Err(Error::State(
                "head actual source/managed/native byte closure".into(),
            ));
        }
        Ok(Self {
            owner: Arc::downgrade(owner),
            source: Arc::downgrade(&owner.source),
            job: Arc::downgrade(job),
            weight: Arc::downgrade(weight),
            programs: Arc::downgrade(programs),
            images: Arc::downgrade(programs.images()),
            fused: job.fused.as_ref().map(Arc::downgrade),
            service: Arc::downgrade(service),
            native: programs.retention_witness()?,
            descriptors,
            ids,
            resident: managed + NATIVE_BYTES,
            uploaded,
            scope,
            fault,
        })
    }
    pub(super) fn inspect(self, launches: usize) -> Result<VocabularyHeadFaultOutcome> {
        let owner = self.owner.upgrade();
        let source = self.source.upgrade();
        let job = self.job.upgrade();
        let weight = self.weight.upgrade();
        let programs = self.programs.upgrade();
        let images = self.images.upgrade();
        let service = self.service.upgrade();
        let flags = [
            owner.is_some(),
            source.is_some(),
            job.is_some(),
            weight.is_some(),
            programs.is_some(),
            images.is_some(),
            service.is_some(),
        ];
        let fused = self.fused.as_ref().and_then(Weak::upgrade);
        let owner = owner.ok_or_else(|| Error::State("head unknown lost fixture owner".into()))?;
        let source =
            source.ok_or_else(|| Error::State("head unknown lost original source".into()))?;
        let job = job.ok_or_else(|| Error::State("head unknown lost original job".into()))?;
        let weight = weight.ok_or_else(|| Error::State("head unknown lost full tied W".into()))?;
        let service =
            service.ok_or_else(|| Error::State("head unknown lost original service".into()))?;
        if flags.iter().any(|alive| !*alive)
            || (self.scope == Scope::ComputedFinalNorm) != fused.is_some()
            || !Arc::ptr_eq(&source, &owner.source)
            || !weight.source().same_issuer(source.source())
            || !Arc::ptr_eq(weight.lease().service(), &service)
            || readbacks(&job) != self.descriptors
            || job
                .accesses
                .iter()
                .map(|(l, _)| l.id())
                .collect::<BTreeSet<_>>()
                != self.ids
            || job.output.is_initialized()
            || !weight.lease().is_initialized()
            || service.ensure_usable().is_ok()
            || service.resident_bytes()? != self.resident
            || service.pending_events()? != 1
            || service.uploaded_value_bytes() != self.uploaded
        {
            return Err(Error::State(
                "head unknown original source/lease/phase/descriptor graph lost".into(),
            ));
        }
        if let Some(fused) = fused.as_ref()
            && job
                .fused
                .as_ref()
                .is_none_or(|original| !Arc::ptr_eq(original, fused))
        {
            return Err(Error::State("head actual final RMS producer lost".into()));
        }
        let native = self.native.inspect()?;
        let pin = cubecl::cuda::PinnedBf16Cubin::Bf16Gemv2TSm86.derivation();
        if native.image.bytes != pin.image_bytes
            || native.image.sha256 != pin.image_sha256
            || native.image.binary_sm != 86
            || native.image.derived_from != Some(cubecl::cuda::PinnedBf16Cubin::Bf16Gemv2TSm86)
            || native.image.captured_from.is_some()
            || native.image_owners != 1
            || native.kernel_owners != 1
            || native.native_charge_bytes != NATIVE_BYTES
        {
            return Err(Error::State(
                "head unknown actual native image/set/charge differs".into(),
            ));
        }
        let snapshot = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        if launches != 1
            || handles != self.ids.len()
            || owners < 4
            || snapshot.reserved != HEAD_CAP as u64
            || snapshot.reservations != 1
            || !snapshot.fenced
            || snapshot.per_class.len() != 1
            || snapshot.per_class.get(&AllocationClass::ProviderEnvelope)
                != Some(&(HEAD_CAP as u64))
        {
            return Err(Error::State(
                "head unknown original physical quarantine missing".into(),
            ));
        }
        let (copies, checkpoints) = weight.copy_counts()?;
        // NativeRetentionWitness::inspect just proved its actual Weak set is
        // alive; the upgraded shared image owner also retains that same set.
        Ok(VocabularyHeadFaultOutcome {
            fault: self.fault,
            scope: self.scope,
            status: if self.fault == VocabularyHeadFault::LateCancelAfterHead {
                "CANCELLED_UNKNOWN_RETAINED"
            } else {
                "UNKNOWN_RETAINED"
            },
            native_launches: launches,
            rms_stages: if self.scope == Scope::ComputedFinalNorm {
                10
            } else {
                0
            },
            producer_ready: 0,
            readbacks: 0,
            comparisons: 0,
            initialized_weight: weight.lease().is_initialized(),
            weight_copies: copies,
            weight_checkpoints: checkpoints,
            resident_bytes: self.resident,
            pending_events: service.pending_events()?,
            uploaded_bytes: self.uploaded,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            source_retained: flags[1],
            fixture_owner_retained: flags[0],
            job_retained: flags[2],
            weight_retained: flags[3],
            programs_retained: flags[4],
            fused_retained: fused.is_some(),
            service_retained: flags[6],
            readback_cohort_retained: true,
            descriptors: self.descriptors.len(),
            managed_leases: self.ids.len(),
            image_owners: native.image_owners,
            kernel_owners: native.kernel_owners,
            module_set_retained: flags[5],
            native_charge_bytes: native.native_charge_bytes,
            physical_reserved: snapshot.reserved,
            physical_fenced: snapshot.fenced,
            host_final_zero_established: false,
        })
    }
}
