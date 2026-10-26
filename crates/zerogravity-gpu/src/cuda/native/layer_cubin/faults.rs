//! Same connected path; original managed refusal and uncertain completion only.
use super::*;
use crate::cuda::rms::{FusedRmsRetentionState, FusedRmsRetentionWitness};
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use std::sync::Weak;
use zerogravity_core::{accelerator::SourceIdentity, resource_arbiter::PhysicalDeviceKey};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostnormCubinFault {
    AdmissionPressure,
    UnknownAfterFused,
    CancelBeforeFused,
}

pub(super) fn validate_scope(scope: LayerScope) -> Result<()> {
    if !scope.postnorm() {
        return Err(Error::Malformed(
            "faults require the closed postnorm scope".into(),
        ));
    }
    Ok(())
}

/// UNKNOWN_RETAINED establishes no producer completion or Ready. Exit the fresh
/// process without readback, synchronize, native retirement or recovery.
#[derive(Clone, Copy, Debug)]
pub struct PostnormCubinFaultOutcome {
    pub fault: PostnormCubinFault,
    pub status: &'static str,
    pub submitted_calls: usize,
    pub rms_stages: usize,
    pub native_launches: usize,
    pub gdn_launches: usize,
    pub postnorm_stages: usize,
    pub compared_words: usize,
    pub ready_completions: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_retained: bool,
    pub owner_retained: bool,
    pub call_retained: bool,
    pub rms_retained: bool,
    pub gdn_retained: bool,
    pub fused_retained: bool,
    pub fused_source_retained: bool,
    pub programs_retained: bool,
    pub service_retained: bool,
    pub readback_cohort_retained: bool,
    pub readback_descriptors: usize,
    pub image_owners_retained: usize,
    pub kernel_owners_retained: usize,
    pub module_set_retained: bool,
    pub native_admission_retained: bool,
    pub native_charge_bytes: usize,
    pub fused_host_workspace_charge_bytes: usize,
    pub host_final_zero_established: bool,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
}

/// Fixed original five-call postnorm scope, using the same inner preparation
/// and producer flight. True pre-cancel precedes fixtures, admission and CUDA.
pub async fn qualify_layer_postnorm_cubin_fault(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [LayerFixture; 5],
    inputs: CubinQualificationInputs,
    fault: PostnormCubinFault,
    cancelled: Option<&AtomicBool>,
) -> Result<PostnormCubinFaultOutcome> {
    match qualify_layer_inner(
        device,
        source,
        LayerFixtures::FivePhasesPostNorm(fixtures),
        inputs,
        Mode::Fault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::Fault(observed) => Ok(observed),
        Outcome::Compared(_) | Outcome::MlpFault(_) => {
            Err(Error::State("postnorm fault returned comparisons".into()))
        }
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
    owner: Weak<LayerOwner>,
    service: Weak<ResidentService>,
}

pub(super) fn admission_pressure(
    owner: &Arc<LayerOwner>,
    service: &Arc<ResidentService>,
) -> Result<PressureWitness> {
    let key = service.physical_key();
    let before = ResourceArbiter::process_shared().snapshot(&key)?;
    if before.fenced
        || !original_envelope(&before)
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "postnorm pressure requires fresh original envelope".into(),
        ));
    }
    if !matches!(
        service.allocate_uninitialized(PROVIDER_CAP + 1, BufferRole::Scratch),
        Err(Error::Limit(_))
    ) {
        return Err(Error::State(
            "postnorm managed pressure did not refuse".into(),
        ));
    }
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "postnorm pressure allocated or uploaded work".into(),
        ));
    }
    service.ensure_usable()?;
    Ok(PressureWitness {
        key,
        owner: Arc::downgrade(owner),
        service: Arc::downgrade(service),
    })
}

impl PressureWitness {
    pub(super) fn inspect(self) -> Result<PostnormCubinFaultOutcome> {
        let snapshot = ResourceArbiter::process_shared().snapshot(&self.key)?;
        if self.owner.upgrade().is_some()
            || self.service.upgrade().is_some()
            || snapshot.reserved != 0
            || snapshot.reservations != 0
            || !snapshot.per_class.is_empty()
            || snapshot.fenced
        {
            return Err(Error::State(
                "postnorm pressure envelope did not retire unfenced".into(),
            ));
        }
        Ok(PostnormCubinFaultOutcome {
            fault: PostnormCubinFault::AdmissionPressure,
            status: "ADMISSION_REFUSED_BEFORE_H2D",
            submitted_calls: 0,
            rms_stages: 0,
            native_launches: 0,
            gdn_launches: 0,
            postnorm_stages: 0,
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
            rms_retained: false,
            gdn_retained: false,
            fused_retained: false,
            fused_source_retained: false,
            programs_retained: false,
            service_retained: false,
            readback_cohort_retained: false,
            readback_descriptors: 0,
            image_owners_retained: 0,
            kernel_owners_retained: 0,
            module_set_retained: false,
            native_admission_retained: false,
            native_charge_bytes: 0,
            fused_host_workspace_charge_bytes: 0,
            host_final_zero_established: false,
            physical_reserved_bytes: snapshot.reserved,
            physical_fenced: snapshot.fenced,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct SubmittedStages {
    pub(super) rms: usize,
    pub(super) native: usize,
    pub(super) gdn: usize,
    pub(super) postnorm: usize,
}

fn validate_stages(fault: PostnormCubinFault, stages: SubmittedStages) -> Result<()> {
    let postnorm = match fault {
        PostnormCubinFault::UnknownAfterFused => 10,
        PostnormCubinFault::CancelBeforeFused => 0,
        PostnormCubinFault::AdmissionPressure => {
            return Err(Error::State(
                "pressure cannot submit producer stages".into(),
            ));
        }
    };
    if stages.rms != 10 || stages.native != 4 || stages.gdn != 12 || stages.postnorm != postnorm {
        return Err(Error::State(
            "postnorm fault submitted stage boundary differs".into(),
        ));
    }
    Ok(())
}

fn validate_readback_layout(layout: &[(usize, usize)]) -> Result<()> {
    if layout.len() != 30
        || layout[..7]
            != [
                (0, 2048),
                (0, 2048),
                (0, 2048),
                (0, 2048),
                (0, 4096),
                (0, 2048),
                (0, 2048),
            ]
        || layout.iter().any(|(_, bytes)| *bytes == 0)
    {
        return Err(Error::State(
            "postnorm first original readback cohort differs".into(),
        ));
    }
    Ok(())
}

type ReadbackIdentity = (BufferLeaseId, usize, usize);
fn readback_identity(read: &ResidentReadback) -> ReadbackIdentity {
    (read.lease_id(), read.offset(), read.bytes())
}

pub(super) struct UnknownWitness {
    fault: PostnormCubinFault,
    key: PhysicalDeviceKey,
    source: SourceIdentity,
    owner: Weak<LayerOwner>,
    call: Weak<Call>,
    rms: Weak<RmsPrepared>,
    gdn: Weak<GdnPrepared>,
    fused: Weak<FusedRmsPrepared>,
    fused_source: FusedRmsRetentionWitness,
    fused_state: FusedRmsRetentionState,
    programs: Weak<super::super::projection_cubin::resident::ProjectionPrograms>,
    native: super::super::projection_cubin::resident::NativeRetentionWitness,
    service: Weak<ResidentService>,
    readbacks: Vec<ReadbackIdentity>,
    accesses: Vec<(BufferLeaseId, DeviceAccess)>,
    resident_bytes: usize,
    uploaded_value_bytes: u64,
    image_owners: usize,
    kernel_owners: usize,
}

impl UnknownWitness {
    pub(super) fn new(
        fault: PostnormCubinFault,
        owner: &Arc<LayerOwner>,
        call: &Arc<Call>,
        programs: &Arc<super::super::projection_cubin::resident::ProjectionPrograms>,
        service: &Arc<ResidentService>,
    ) -> Result<Self> {
        if fault == PostnormCubinFault::AdmissionPressure
            || !owner.fixtures.scope().postnorm()
            || call.phase != GdnPhase::InitialOne
            || call.snapshot.is_some()
        {
            return Err(Error::State(
                "unknown requires first original postnorm call".into(),
            ));
        }
        let fused = call
            .postnorm
            .as_ref()
            .ok_or_else(|| Error::State("unknown postnorm owner absent".into()))?;
        let fused_source = fused.retention_witness();
        let fused_state = fused_source.inspect()?;
        let projected = call
            .projected
            .as_ref()
            .ok_or_else(|| Error::State("unknown original projection absent".into()))?;
        if fused_state.source != owner._source.source_identity()
            || fused_state.source != service.source_identity()
            || fused_state.rows != 1
            || fused_state.input_ids != [projected.id(), call.rms.embedding().id()]
            || fused_state.output_initialized != [false; 3]
            || fused_state.host_workspace_bytes == 0
        {
            return Err(Error::State(
                "unknown fused original input/source bindings differ".into(),
            ));
        }
        for (id, access) in [
            (projected.id(), DeviceAccess::Write),
            (call.rms.embedding().id(), DeviceAccess::Read),
            (fused_state.output_ids[0], DeviceAccess::Write),
            (fused_state.output_ids[1], DeviceAccess::Write),
            (fused_state.output_ids[2], DeviceAccess::Write),
        ] {
            if !call.accesses.iter().any(|(lease, mode)| {
                lease.id() == id
                    && *mode == access
                    && Arc::ptr_eq(lease.service(), service)
                    && lease.source_identity() == fused_state.source
            }) {
                return Err(Error::State(
                    "unknown original fused flight access differs".into(),
                ));
            }
        }
        let readbacks = call.reads.iter().map(readback_identity).collect::<Vec<_>>();
        validate_readback_layout(
            &readbacks
                .iter()
                .map(|&(_, offset, bytes)| (offset, bytes))
                .collect::<Vec<_>>(),
        )?;
        let mut declared = Vec::from(call.rms.readbacks()?);
        declared.extend(fused.readbacks()?);
        for record in &owner.fixtures.as_slice()[0].expected {
            declared.push(call.readback(record.output)?);
        }
        if readbacks != declared.iter().map(readback_identity).collect::<Vec<_>>()
            || readbacks[3..7]
                .iter()
                .map(|(id, _, _)| *id)
                .collect::<Vec<_>>()
                != fused_state.readback_ids
            || readbacks[3..7]
                .iter()
                .map(|(_, _, bytes)| *bytes)
                .collect::<Vec<_>>()
                != fused_state.readback_bytes
        {
            return Err(Error::State(
                "unknown actual readback lease identities differ".into(),
            ));
        }
        let native = programs.retention_witness()?;
        let native_state = native.inspect()?;
        let key = service.physical_key();
        let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
        let resident_bytes = service.resident_bytes()?;
        if resident_bytes <= FOREIGN_BYTES
            || service.uploaded_value_bytes() == 0
            || service.pending_events()? != 0
            || quarantined_handles_for(service.client())? != 0
            || quarantined_completion_owners_for(service.client())? != 0
            || snapshot.fenced
            || !original_envelope(&snapshot)
            || native_state.native_charge_bytes != FOREIGN_BYTES
            || native_state.image_owners != 6
            || native_state.kernel_owners != 11
        {
            return Err(Error::State(
                "unknown requires fresh original prepared postnorm graph".into(),
            ));
        }
        service.ensure_usable()?;
        Ok(Self {
            fault,
            key,
            source: fused_state.source,
            owner: Arc::downgrade(owner),
            call: Arc::downgrade(call),
            rms: Arc::downgrade(&call.rms),
            gdn: Arc::downgrade(&call.gdn),
            fused: Arc::downgrade(fused),
            fused_source,
            fused_state,
            programs: Arc::downgrade(programs),
            native,
            service: Arc::downgrade(service),
            readbacks,
            accesses: call
                .accesses
                .iter()
                .map(|(lease, access)| (lease.id(), *access))
                .collect(),
            resident_bytes,
            uploaded_value_bytes: service.uploaded_value_bytes(),
            image_owners: native_state.image_owners,
            kernel_owners: native_state.kernel_owners,
        })
    }

    /// Only Weak upgrades and original host ledgers after every local owner
    /// drops. Other calls' handles also quarantine because the service is fenced;
    /// their private CPU source owners need not be retained.
    pub(super) fn inspect(self, stages: SubmittedStages) -> Result<PostnormCubinFaultOutcome> {
        validate_stages(self.fault, stages)?;
        let owner = self.owner.upgrade();
        let call = self.call.upgrade();
        let rms = self.rms.upgrade();
        let gdn = self.gdn.upgrade();
        let fused = self.fused.upgrade();
        let programs = self.programs.upgrade();
        let service = self.service.upgrade();
        let retained = [
            owner.is_some(),
            call.is_some(),
            rms.is_some(),
            gdn.is_some(),
            fused.is_some(),
            programs.is_some(),
            service.is_some(),
        ];
        let owner = owner
            .ok_or_else(|| Error::State("postnorm quarantine lost original source owner".into()))?;
        let call =
            call.ok_or_else(|| Error::State("postnorm quarantine lost original call".into()))?;
        let service = service
            .ok_or_else(|| Error::State("postnorm quarantine lost original service".into()))?;
        let fused_state = self.fused_source.inspect()?;
        let native = self.native.inspect()?;
        let resident_bytes = service.resident_bytes()?;
        let pending_events = service.pending_events()?;
        let uploaded_value_bytes = service.uploaded_value_bytes();
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        let snapshot = ResourceArbiter::process_shared().snapshot(&self.key)?;
        if retained.contains(&false)
            || owner._source.source_identity() != self.source
            || fused_state != self.fused_state
            || call.reads.iter().map(readback_identity).collect::<Vec<_>>() != self.readbacks
            || call
                .accesses
                .iter()
                .map(|(lease, access)| (lease.id(), *access))
                .collect::<Vec<_>>()
                != self.accesses
            || native.image_owners != self.image_owners
            || native.kernel_owners != self.kernel_owners
            || native.native_charge_bytes != FOREIGN_BYTES
            || native.image_admission_bytes == 0
            || service.ensure_usable().is_ok()
            || resident_bytes != self.resident_bytes
            || pending_events != 1
            || uploaded_value_bytes != self.uploaded_value_bytes
            || handles == 0
            || owners < 5
            || !snapshot.fenced
            || !original_envelope(&snapshot)
        {
            return Err(Error::State(
                "postnorm unknown did not retain original source/flight graph".into(),
            ));
        }
        Ok(PostnormCubinFaultOutcome {
            fault: self.fault,
            status: if self.fault == PostnormCubinFault::CancelBeforeFused {
                "CANCELLED_UNKNOWN_RETAINED"
            } else {
                "UNKNOWN_RETAINED"
            },
            submitted_calls: 1,
            rms_stages: stages.rms,
            native_launches: stages.native,
            gdn_launches: stages.gdn,
            postnorm_stages: stages.postnorm,
            compared_words: 0,
            ready_completions: 0,
            resident_bytes,
            pending_events,
            uploaded_value_bytes,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            source_retained: owner._source.source_identity() == self.source,
            owner_retained: retained[0],
            call_retained: retained[1],
            rms_retained: retained[2],
            gdn_retained: retained[3],
            fused_retained: retained[4],
            fused_source_retained: true,
            programs_retained: retained[5],
            service_retained: retained[6],
            readback_cohort_retained: true,
            readback_descriptors: self.readbacks.len(),
            image_owners_retained: native.image_owners,
            kernel_owners_retained: native.kernel_owners,
            module_set_retained: true,
            native_admission_retained: true,
            native_charge_bytes: native.native_charge_bytes,
            fused_host_workspace_charge_bytes: fused_state.host_workspace_bytes,
            host_final_zero_established: false,
            physical_reserved_bytes: snapshot.reserved,
            physical_fenced: snapshot.fenced,
        })
    }
}

#[cfg(test)]
#[path = "faults/tests.rs"]
mod tests;
