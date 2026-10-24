//! Original activation admission and unknown completion; no alternate execution.
use super::*;
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use crate::vllm_cuda_qualification::projection_cubin::resident::NativeImageIdentity;
use std::sync::Weak;
use zerogravity_core::{device_events::BufferLeaseId, resource_arbiter::PhysicalDeviceKey};

const NATIVE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubinActivationFault {
    Pressure,
    Unknown,
}

/// UNKNOWN_RETAINED grants no completion, Ready or native retirement authority.
#[derive(Clone, Copy, Debug)]
pub struct CubinActivationFaultOutcome {
    pub fault: CubinActivationFault,
    pub status: &'static str,
    pub submitted_calls: usize,
    pub native_launches: usize,
    pub compared_words: usize,
    pub ready_completions: usize,
    pub readbacks_executed: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_retained: bool,
    pub leaf_owner_retained: bool,
    pub job_retained: bool,
    pub programs_retained: bool,
    pub shared_images_retained: bool,
    pub service_retained: bool,
    pub readback_cohort_retained: bool,
    pub readback_descriptors: usize,
    pub prepared_physical_leases: usize,
    pub image_owners_retained: usize,
    pub kernel_owners_retained: usize,
    pub module_set_retained: bool,
    pub native_admission_retained: bool,
    pub native_charge_bytes: usize,
    pub host_workspace_charge_bytes: usize,
    pub host_final_zero_established: bool,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
}

/// Fixed original five-job preparation and first genuine Rows1 native launch.
pub async fn qualify_activation_cubin_fault(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinActivationFixture>,
    inputs: CubinActivationInputs,
    fault: CubinActivationFault,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinActivationFaultOutcome> {
    match qualify_activation_inner(
        device,
        source,
        fixtures,
        inputs,
        Mode::Fault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::Fault(observed) => Ok(*observed),
        Outcome::Compared(_) => Err(Error::State("activation fault returned comparisons".into())),
    }
}

fn original_envelope(snapshot: &zerogravity_core::resource_arbiter::DeviceBudgetSnapshot) -> bool {
    snapshot.reserved == PROVIDER_CAP as u64
        && snapshot.reservations == 1
        && snapshot.per_class.len() == 1
        && snapshot.per_class.get(&AllocationClass::ProviderEnvelope)
            == Some(&(PROVIDER_CAP as u64))
}

pub(super) struct PressureWitness {
    key: PhysicalDeviceKey,
    source: Weak<VerifiedTensorSource>,
    owner: Weak<ActivationOwner>,
    service: Weak<ResidentService>,
}
pub(super) fn admission_pressure(
    source: &Arc<VerifiedTensorSource>,
    owner: &Arc<ActivationOwner>,
    service: &Arc<ResidentService>,
) -> Result<PressureWitness> {
    let key = service.physical_key();
    let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
    if snapshot.fenced
        || !original_envelope(&snapshot)
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "activation pressure requires fresh original envelope".into(),
        ));
    }
    // SAME managed ledger refuses before client.empty; no physical handle exists.
    if !matches!(
        service.allocate_uninitialized(PROVIDER_CAP + 1, BufferRole::Scratch),
        Err(Error::Limit(_))
    ) {
        return Err(Error::State(
            "activation original ledger did not refuse pressure".into(),
        ));
    }
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "activation pressure allocated/uploaded/quarantined work".into(),
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
pub(super) fn pressure_retired(witness: PressureWitness) -> Result<CubinActivationFaultOutcome> {
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
            "activation pressure retained original local owners/envelope".into(),
        ));
    }
    Ok(CubinActivationFaultOutcome {
        fault: CubinActivationFault::Pressure,
        status: "ADMISSION_REFUSED_BEFORE_H2D",
        submitted_calls: 0,
        native_launches: 0,
        compared_words: 0,
        ready_completions: 0,
        readbacks_executed: 0,
        resident_bytes: 0,
        pending_events: 0,
        uploaded_value_bytes: 0,
        quarantined_handles: 0,
        quarantined_completion_owners: 0,
        source_retained: false,
        leaf_owner_retained: false,
        job_retained: false,
        programs_retained: false,
        shared_images_retained: false,
        service_retained: false,
        readback_cohort_retained: false,
        readback_descriptors: 0,
        prepared_physical_leases: 0,
        image_owners_retained: 0,
        kernel_owners_retained: 0,
        module_set_retained: false,
        native_admission_retained: false,
        native_charge_bytes: 0,
        host_workspace_charge_bytes: 0,
        host_final_zero_established: false,
        physical_reserved_bytes: snapshot.reserved,
        physical_fenced: snapshot.fenced,
    })
}

type ReadbackIdentity = (BufferLeaseId, usize, usize);
fn readbacks(job: &Job) -> Result<[ReadbackIdentity; 2]> {
    let [first, second] = job.reads.as_slice() else {
        return Err(Error::State(
            "activation exact two original descriptors".into(),
        ));
    };
    Ok([first, second].map(|read| (read.lease_id(), read.offset(), read.bytes())))
}
fn validate_layout(
    activation: Activation,
    fixtures: usize,
    bytes: [usize; 2],
    access: [DeviceAccess; 2],
    initialized: [bool; 2],
    descriptors: [(usize, usize); 2],
) -> Result<()> {
    if activation != Activation::Rows1
        || fixtures != 5
        || bytes != [activation.input_bytes(), activation.output_bytes()]
        || access != [DeviceAccess::Read, DeviceAccess::Write]
        || initialized != [true, false]
        || descriptors
            != [
                (0, activation.input_bytes()),
                (0, activation.output_bytes()),
            ]
    {
        return Err(Error::State(
            "activation first original Rows1 lease/descriptor layout".into(),
        ));
    }
    Ok(())
}
fn validate_job(
    job: &Job,
    fixtures: usize,
    source: &VerifiedTensorSource,
    service: &Arc<ResidentService>,
) -> Result<()> {
    let descriptors = readbacks(job)?;
    if job.accesses.len() != 2 {
        return Err(Error::State("activation exact original X/Y flight".into()));
    }
    validate_layout(
        job.activation,
        fixtures,
        [job.input.bytes(), job.output.bytes()],
        [job.accesses[0].1, job.accesses[1].1],
        [job.input.is_initialized(), job.output.is_initialized()],
        [
            (descriptors[0].1, descriptors[0].2),
            (descriptors[1].1, descriptors[1].2),
        ],
    )?;
    if job.input.id() == job.output.id()
        || job.accesses[0].0.id() != job.input.id()
        || job.accesses[1].0.id() != job.output.id()
        || descriptors[0].0 != job.input.id()
        || descriptors[1].0 != job.output.id()
        || job.accesses.iter().any(|(lease, _)| {
            !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != source.source_identity()
        })
    {
        return Err(Error::State(
            "activation original X/Y identities/source/service".into(),
        ));
    }
    Ok(())
}
fn validate_image(image: &NativeImageIdentity) -> Result<()> {
    let kind = cubecl::cuda::PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86;
    let expected = kind.capture();
    if image.bytes != expected.image_bytes
        || image.sha256 != expected.image_sha256
        || image.binary_sm != expected.binary_sm
        || image.captured_from != Some(kind)
        || image.derived_from.is_some()
    {
        return Err(Error::State(
            "activation retained exact selected resource/category".into(),
        ));
    }
    Ok(())
}

pub(super) struct UnknownWitness {
    key: PhysicalDeviceKey,
    source: Weak<VerifiedTensorSource>,
    owner: Weak<ActivationOwner>,
    job: Weak<Job>,
    programs: Weak<ActivationPrograms>,
    native: resident::ActivationRetentionWitness,
    service: Weak<ResidentService>,
    resident_bytes: usize,
    uploaded_value_bytes: u64,
    descriptors: [ReadbackIdentity; 2],
    prepared_ids: BTreeSet<BufferLeaseId>,
}
impl UnknownWitness {
    pub(super) fn new(
        source: &Arc<VerifiedTensorSource>,
        owner: &Arc<ActivationOwner>,
        jobs: &[Arc<Job>],
        programs: &Arc<ActivationPrograms>,
        service: &Arc<ResidentService>,
    ) -> Result<Self> {
        validate_fixtures(&owner.fixtures)?;
        if jobs.len() != owner.fixtures.len() || !Arc::ptr_eq(source, &owner._source) {
            return Err(Error::State(
                "activation original five-job/source cohort".into(),
            ));
        }
        let job = jobs
            .first()
            .ok_or_else(|| Error::State("activation first job absent".into()))?;
        validate_job(job, owner.fixtures.len(), source, service)?;
        let mut prepared_ids = BTreeSet::new();
        let mut managed_bytes = 0usize;
        let mut uploaded_bytes = 0u64;
        for (job, fixture) in jobs.iter().zip(&owner.fixtures) {
            if job.activation != fixture.activation
                || job.accesses.len() != 2
                || job.input.bytes() != fixture.activation.input_bytes()
                || job.output.bytes() != fixture.activation.output_bytes()
                || job.accesses[0].0.id() != job.input.id()
                || job.accesses[0].1 != DeviceAccess::Read
                || job.accesses[1].0.id() != job.output.id()
                || job.accesses[1].1 != DeviceAccess::Write
                || !job.input.is_initialized()
                || job.output.is_initialized()
                || job.accesses.iter().any(|(lease, _)| {
                    !Arc::ptr_eq(lease.service(), service)
                        || lease.source_identity() != source.source_identity()
                })
                || !prepared_ids.insert(job.input.id())
                || !prepared_ids.insert(job.output.id())
            {
                return Err(Error::State(
                    "activation actual original prepared lease cohort".into(),
                ));
            }
            managed_bytes = managed_bytes
                .checked_add(job.input.bytes())
                .and_then(|n| n.checked_add(job.output.bytes()))
                .ok_or_else(|| Error::Limit("activation prepared bytes overflow".into()))?;
            uploaded_bytes = uploaded_bytes
                .checked_add(job.input.bytes() as u64)
                .ok_or_else(|| Error::Limit("activation uploaded bytes overflow".into()))?;
        }
        let key = service.physical_key();
        let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
        let resident_bytes = service.resident_bytes()?;
        if snapshot.fenced
            || !original_envelope(&snapshot)
            || service.pending_events()? != 0
            || managed_bytes.checked_add(NATIVE_BYTES) != Some(resident_bytes)
            || service.uploaded_value_bytes() != uploaded_bytes
            || quarantined_handles_for(service.client())? != 0
            || quarantined_completion_owners_for(service.client())? != 0
        {
            return Err(Error::State(
                "activation unknown requires fresh actual prepared graph".into(),
            ));
        }
        service.ensure_usable()?;
        Ok(Self {
            key,
            source: Arc::downgrade(source),
            owner: Arc::downgrade(owner),
            job: Arc::downgrade(job),
            programs: Arc::downgrade(programs),
            native: programs.retention_witness()?,
            service: Arc::downgrade(service),
            resident_bytes,
            uploaded_value_bytes: uploaded_bytes,
            descriptors: readbacks(job)?,
            prepared_ids,
        })
    }
    /// Host-only Weak upgrades after ALL caller-owned strong references drop.
    pub(super) fn inspect(self, launches: usize) -> Result<CubinActivationFaultOutcome> {
        let source = self.source.upgrade();
        let owner = self.owner.upgrade();
        let job = self.job.upgrade();
        let programs = self.programs.upgrade();
        let service = self.service.upgrade();
        let retained = [
            source.is_some(),
            owner.is_some(),
            job.is_some(),
            programs.is_some(),
            service.is_some(),
        ];
        let native = self.native.inspect()?;
        validate_image(&native.image)?;
        let source =
            source.ok_or_else(|| Error::State("activation source lost from quarantine".into()))?;
        let owner =
            owner.ok_or_else(|| Error::State("activation owner lost from quarantine".into()))?;
        let job =
            job.ok_or_else(|| Error::State("activation first job lost from quarantine".into()))?;
        let service = service
            .ok_or_else(|| Error::State("activation service lost from quarantine".into()))?;
        validate_job(&job, owner.fixtures.len(), &source, &service)?;
        if !Arc::ptr_eq(&source, &owner._source)
            || readbacks(&job)? != self.descriptors
            || !self.prepared_ids.contains(&job.input.id())
            || !self.prepared_ids.contains(&job.output.id())
        {
            return Err(Error::State(
                "activation retained original descriptor/source cohort".into(),
            ));
        }
        let resident_bytes = service.resident_bytes()?;
        let pending_events = service.pending_events()?;
        let uploaded_value_bytes = service.uploaded_value_bytes();
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        let snapshot = ResourceArbiter::process_shared().snapshot(&self.key)?;
        if retained.contains(&false)
            || launches != 1
            || native.image_owners != 1
            || native.kernel_owners != 3
            || native.native_charge_bytes != NATIVE_BYTES
            || native.image_admission_bytes == 0
            || service.ensure_usable().is_ok()
            || resident_bytes != self.resident_bytes
            || pending_events != 1
            || uploaded_value_bytes != self.uploaded_value_bytes
            || handles < self.prepared_ids.len()
            || owners < 3
            || !snapshot.fenced
            || !original_envelope(&snapshot)
        {
            return Err(Error::State(
                "activation unknown did not retain actual original graph".into(),
            ));
        }
        let host_workspace_charge_bytes = owner
            ._host
            .bytes()
            .checked_add(native.image_admission_bytes)
            .ok_or_else(|| Error::Limit("activation retained host charge overflow".into()))?;
        Ok(CubinActivationFaultOutcome {
            fault: CubinActivationFault::Unknown,
            status: "UNKNOWN_RETAINED",
            submitted_calls: 1,
            native_launches: launches,
            compared_words: 0,
            ready_completions: 0,
            readbacks_executed: 0,
            resident_bytes,
            pending_events,
            uploaded_value_bytes,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            source_retained: retained[0],
            leaf_owner_retained: retained[1],
            job_retained: retained[2],
            programs_retained: retained[3],
            shared_images_retained: true,
            service_retained: retained[4],
            readback_cohort_retained: true,
            readback_descriptors: self.descriptors.len(),
            prepared_physical_leases: self.prepared_ids.len(),
            image_owners_retained: native.image_owners,
            kernel_owners_retained: native.kernel_owners,
            module_set_retained: true,
            native_admission_retained: true,
            native_charge_bytes: native.native_charge_bytes,
            host_workspace_charge_bytes,
            host_final_zero_established: false,
            physical_reserved_bytes: snapshot.reserved,
            physical_fenced: snapshot.fenced,
        })
    }
}

#[cfg(test)]
mod tests;
