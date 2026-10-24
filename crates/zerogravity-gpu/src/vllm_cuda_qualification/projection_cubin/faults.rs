//! Actual original-ledger refusal and first native enqueue abandonment.
use super::*;
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use std::sync::Weak;
use zerogravity_core::resource_arbiter::PhysicalDeviceKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubinProjectionFault {
    AdmissionPressure,
    UnknownAfterNative,
}

/// Host observations only. UNKNOWN_RETAINED is not physical completion/Ready.
/// Exit the fresh process without synchronize, native unload or recovery.
#[derive(Clone, Copy, Debug)]
pub struct CubinProjectionFaultOutcome {
    pub fault: CubinProjectionFault,
    pub status: &'static str,
    pub submitted_calls: usize,
    pub native_launches: usize,
    pub compared_words: usize,
    /// Mathematical producer completion only; preparation barriers precede it.
    pub ready_completions: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_retained: bool,
    pub leaf_owner_retained: bool,
    pub job_retained: bool,
    pub programs_retained: bool,
    pub service_retained: bool,
    pub image_owners_retained: usize,
    pub kernel_owners_retained: usize,
    pub module_set_retained: bool,
    pub native_admission_retained: bool,
    pub native_charge_bytes: usize,
    pub host_workspace_charge_bytes: usize,
    /// ROOT separately inspects its actual host ledger after external owners drop.
    pub host_final_zero_established: bool,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
}

/// SAME native path, closed to one original Qkvz1 or captured Out2 leaf.
/// A true cancelled flag refuses before fixtures, host workspace or CUDA.
pub async fn qualify_projection_cubin_fault(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinProjectionFixture>,
    inputs: CubinQualificationInputs,
    fault: CubinProjectionFault,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinProjectionFaultOutcome> {
    match qualify_projection_inner(
        device,
        source,
        fixtures,
        inputs,
        Mode::Fault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::Fault(observed) => Ok(observed),
        Outcome::Compared(_) => Err(Error::State(
            "projection fault returned comparison outcome".into(),
        )),
    }
}

#[derive(Clone, Copy)]
pub(super) enum FaultCut {
    OriginalQkvz1,
    CapturedOut2,
}
impl FaultCut {
    pub(super) fn for_projection(projection: Projection) -> Result<Self> {
        match projection {
            Projection::Qkvz1 => Ok(Self::OriginalQkvz1),
            Projection::Out2 => Ok(Self::CapturedOut2),
            _ => Err(Error::Malformed(
                "native faults require one original Qkvz1 or captured Out2".into(),
            )),
        }
    }
    fn projection(self) -> Projection {
        match self {
            Self::OriginalQkvz1 => Projection::Qkvz1,
            Self::CapturedOut2 => Projection::Out2,
        }
    }
    fn image(self) -> ProjectionImage {
        match self {
            Self::OriginalQkvz1 => ProjectionImage::Bf16Gemv2TSm86,
            Self::CapturedOut2 => ProjectionImage::CutlassWmmaSm80,
        }
    }
    fn windows(self) -> &'static [(&'static str, usize, usize)] {
        match self {
            Self::OriginalQkvz1 => &[
                (
                    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                    6144,
                    1024,
                ),
                (
                    "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                    2048,
                    1024,
                ),
            ],
            Self::CapturedOut2 => &[(
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                1024,
                2048,
            )],
        }
    }
    pub(super) fn validate_geometry(
        self,
        weight_index: usize,
        window_count: usize,
        buffers: [usize; 3],
    ) -> Result<()> {
        let projection = self.projection();
        if weight_index != projection.weight_index()
            || window_count != self.windows().len()
            || buffers
                != [
                    projection.weight_bytes(),
                    projection.input_bytes(),
                    projection.output_bytes(),
                ]
        {
            return Err(Error::State(
                "native fault original W role/windows/operand extents".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn validate_image(self, image: &resident::NativeImageIdentity) -> Result<()> {
        let expected = self.image();
        let provenance = match expected.provenance() {
            ProjectionImageSource::Derived(kind) => {
                image.derived_from == Some(kind) && image.captured_from.is_none()
            }
            ProjectionImageSource::Captured(kind) => {
                image.captured_from == Some(kind) && image.derived_from.is_none()
            }
        };
        if image.bytes != expected.bytes()
            || image.sha256 != expected.sha256()
            || image.binary_sm != expected.binary_sm()
            || !provenance
        {
            return Err(Error::State(
                "native fault retained image identity/provenance differs".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn validate_fault_fixtures(fixtures: &[CubinProjectionFixture]) -> Result<()> {
    if fixtures.len() != 1 {
        return Err(Error::Malformed(
            "native faults require exactly one closed projection".into(),
        ));
    }
    FaultCut::for_projection(fixtures[0].projection)?;
    Ok(())
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
    owner: Weak<LeafOwner>,
    service: Weak<ResidentService>,
}
pub(super) fn admission_pressure(
    source: &Arc<VerifiedTensorSource>,
    owner: &Arc<LeafOwner>,
    service: &Arc<ResidentService>,
) -> Result<PressureWitness> {
    let key = service.physical_key();
    let before = ResourceArbiter::process_shared().snapshot(&key)?;
    if before.fenced
        || !original_envelope(&before)
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
    {
        return Err(Error::State(
            "native pressure requires a fresh original envelope".into(),
        ));
    }
    // The SAME real ledger rejects before client.empty; no lease/handle exists.
    if !matches!(
        service.allocate_uninitialized(PROVIDER_CAP + 1, BufferRole::Scratch),
        Err(Error::Limit(_))
    ) {
        return Err(Error::State(
            "original native managed admission did not refuse pressure".into(),
        ));
    }
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "native pressure allocated/uploaded/quarantined work".into(),
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
pub(super) fn pressure_retired(witness: PressureWitness) -> Result<CubinProjectionFaultOutcome> {
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
            "native pressure did not retire its original owners/envelope".into(),
        ));
    }
    Ok(CubinProjectionFaultOutcome {
        fault: CubinProjectionFault::AdmissionPressure,
        status: "ADMISSION_REFUSED_BEFORE_H2D",
        submitted_calls: 0,
        native_launches: 0,
        compared_words: 0,
        ready_completions: 0,
        resident_bytes: 0,
        pending_events: 0,
        uploaded_value_bytes: 0,
        quarantined_handles: 0,
        quarantined_completion_owners: 0,
        source_retained: false,
        leaf_owner_retained: false,
        job_retained: false,
        programs_retained: false,
        service_retained: false,
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

pub(super) struct UnknownWitness {
    cut: FaultCut,
    key: PhysicalDeviceKey,
    source: Weak<VerifiedTensorSource>,
    owner: Weak<LeafOwner>,
    job: Weak<Job>,
    programs: Weak<resident::ProjectionPrograms>,
    native: resident::NativeRetentionWitness,
    service: Weak<ResidentService>,
    resident_bytes: usize,
    uploaded_value_bytes: u64,
}
impl UnknownWitness {
    pub(super) fn new(
        source: &Arc<VerifiedTensorSource>,
        owner: &Arc<LeafOwner>,
        job: &Arc<Job>,
        programs: &Arc<resident::ProjectionPrograms>,
        service: &Arc<ResidentService>,
    ) -> Result<Self> {
        let key = service.physical_key();
        let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
        let resident_bytes = service.resident_bytes()?;
        let cut = FaultCut::for_projection(job.projection)?;
        validate_fault_fixtures(&owner.fixtures)?;
        if owner.weights.len() != 1 || job.accesses.len() != 3 {
            return Err(Error::State(
                "native fault exact original weight/lease set".into(),
            ));
        }
        let (&weight_index, raw) = owner
            .weights
            .first_key_value()
            .ok_or_else(|| Error::State("native fault original W absent".into()))?;
        cut.validate_geometry(
            weight_index,
            owner._windows.len(),
            [
                job.accesses[0].0.bytes(),
                job.input.bytes(),
                job.output.bytes(),
            ],
        )?;
        for (window, &(name, rows, columns)) in owner._windows.iter().zip(cut.windows()) {
            let matrix = window.matrix();
            if matrix.source_identity() != source.source_identity()
                || matrix.tensor().name != name
                || matrix.tensor().kind != QuantKind::Bf16
                || matrix.axes() != zerogravity_core::accelerator::program::MatrixAxes::OutputInput
                || matrix.rows() != rows
                || matrix.columns() != columns
                || window.row_range() != (0..rows)
                || window.as_slice().len() != rows * columns * 2
            {
                return Err(Error::State(
                    "native fault exact retained original W windows".into(),
                ));
            }
        }
        if raw.len() != job.projection.weight_bytes()
            || owner.fixtures[0].projection != job.projection
            || job.accesses.iter().any(|(lease, _)| {
                !Arc::ptr_eq(lease.service(), service)
                    || lease.source_identity() != source.source_identity()
            })
            || job.accesses[0].1 != DeviceAccess::Read
            || !job.accesses[0].0.is_initialized()
            || job.accesses[1].1 != DeviceAccess::Read
            || job.accesses[1].0.id() != job.input.id()
            || !job.input.is_initialized()
            || job.accesses[2].1 != DeviceAccess::Write
            || job.accesses[2].0.id() != job.output.id()
            || job
                .accesses
                .iter()
                .map(|(lease, _)| lease.id())
                .collect::<BTreeSet<_>>()
                .len()
                != 3
        {
            return Err(Error::State(
                "native fault exact original W/X/Y flight bindings".into(),
            ));
        }
        if !job.reads.is_empty()
            || owner.fixtures.len() != 1
            || !Arc::ptr_eq(source, &owner._source)
            || service.pending_events()? != 0
            || resident_bytes <= FOREIGN_BYTES
            || service.uploaded_value_bytes() == 0
            || quarantined_handles_for(service.client())? != 0
            || quarantined_completion_owners_for(service.client())? != 0
            || snapshot.fenced
            || !original_envelope(&snapshot)
        {
            return Err(Error::State(
                "native unknown probe requires a fresh prepared closed fault cut".into(),
            ));
        }
        service.ensure_usable()?;
        Ok(Self {
            cut,
            key,
            source: Arc::downgrade(source),
            owner: Arc::downgrade(owner),
            job: Arc::downgrade(job),
            programs: Arc::downgrade(programs),
            native: programs.retention_witness()?,
            service: Arc::downgrade(service),
            resident_bytes,
            uploaded_value_bytes: service.uploaded_value_bytes(),
        })
    }
    /// Only Weak upgrades and original host ledgers/quarantine are inspected.
    pub(super) fn inspect(self, launches: usize) -> Result<CubinProjectionFaultOutcome> {
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
        self.cut.validate_image(&native.image)?;
        let service = service
            .ok_or_else(|| Error::State("native quarantine lost original service".into()))?;
        let owner = owner
            .ok_or_else(|| Error::State("native quarantine lost original source owner".into()))?;
        let resident_bytes = service.resident_bytes()?;
        let pending_events = service.pending_events()?;
        let uploaded_value_bytes = service.uploaded_value_bytes();
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        let snapshot = ResourceArbiter::process_shared().snapshot(&self.key)?;
        if retained.contains(&false)
            || launches != 1
            || native.image_owners != 1
            || native.kernel_owners != 1
            || native.native_charge_bytes != FOREIGN_BYTES
            || native.image_admission_bytes == 0
            || service.ensure_usable().is_ok()
            || resident_bytes != self.resident_bytes
            || pending_events != 1
            || uploaded_value_bytes != self.uploaded_value_bytes
            || handles == 0
            || owners < 3
            || !snapshot.fenced
            || !original_envelope(&snapshot)
        {
            return Err(Error::State(
                "native unknown completion did not retain/fence original owners".into(),
            ));
        }
        Ok(CubinProjectionFaultOutcome {
            fault: CubinProjectionFault::UnknownAfterNative,
            status: "UNKNOWN_RETAINED",
            submitted_calls: 1,
            native_launches: launches,
            compared_words: 0,
            ready_completions: 0,
            resident_bytes,
            pending_events,
            uploaded_value_bytes,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            source_retained: retained[0],
            leaf_owner_retained: retained[1],
            job_retained: retained[2],
            programs_retained: retained[3],
            service_retained: retained[4],
            image_owners_retained: native.image_owners,
            kernel_owners_retained: native.kernel_owners,
            module_set_retained: true,
            native_admission_retained: true,
            native_charge_bytes: native.native_charge_bytes,
            host_workspace_charge_bytes: owner
                ._host
                .iter()
                .map(WindowWorkspace::bytes)
                .sum::<usize>()
                + native.image_admission_bytes,
            host_final_zero_established: false,
            physical_reserved_bytes: snapshot.reserved,
            physical_fenced: snapshot.fenced,
        })
    }
}
