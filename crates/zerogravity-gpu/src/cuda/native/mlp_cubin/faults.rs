//! One original Down64 clear+kernel cut, using the positive runner and leases.
use super::*;
use crate::cuda::native::projection_cubin::resident::NativeImageIdentity;
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use std::sync::Weak;
use zerogravity_core::{device_events::BufferLeaseId, resource_arbiter::PhysicalDeviceKey};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubinMlpFault {
    Pressure,
    Unknown,
}

/// Unknown completion is not a Ready, comparison or native retirement permit.
#[derive(Clone, Copy, Debug)]
pub struct CubinMlpFaultOutcome {
    pub fault: CubinMlpFault,
    pub status: &'static str,
    pub submitted_calls: usize,
    pub native_launches: usize,
    pub workspace_clears: usize,
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
    pub weight_window_retained: bool,
    pub job_retained: bool,
    pub workspace_retained: bool,
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

pub async fn qualify_mlp_cubin_fault(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: Vec<CubinMlpFixture>,
    inputs: CubinQualificationInputs,
    fault: CubinMlpFault,
    cancelled: Option<&AtomicBool>,
) -> Result<CubinMlpFaultOutcome> {
    match qualify_mlp_inner(
        device,
        source,
        fixtures,
        inputs,
        Mode::Fault(fault),
        cancelled,
    )
    .await?
    {
        Outcome::Fault(outcome) => Ok(*outcome),
        Outcome::Compared(_) => Err(Error::State("MLP fault returned comparisons".into())),
    }
}

pub(super) fn validate_scope(fixtures: &[CubinMlpFixture]) -> Result<()> {
    if fixtures.len() != 1 || fixtures[0].projection != MlpProjection::Down64 {
        return Err(Error::Malformed(
            "MLP faults require one original Down64 leaf".into(),
        ));
    }
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
    window: Weak<TensorWindow>,
    service: Weak<ResidentService>,
}
pub(super) fn admission_pressure(
    source: &Arc<VerifiedTensorSource>,
    owner: &Arc<LeafOwner>,
    service: &Arc<ResidentService>,
) -> Result<PressureWitness> {
    validate_scope(&owner.fixtures)?;
    let window = owner
        ._windows
        .first()
        .filter(|_| owner._windows.len() == 1)
        .ok_or_else(|| Error::State("MLP pressure original DownW window".into()))?;
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
            "MLP pressure requires fresh original envelope".into(),
        ));
    }
    if !matches!(
        service.allocate_uninitialized(PROVIDER_CAP + 1, BufferRole::Scratch),
        Err(Error::Limit(_))
    ) {
        return Err(Error::State(
            "MLP original managed ledger did not refuse pressure".into(),
        ));
    }
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "MLP pressure allocated/uploaded/quarantined work".into(),
        ));
    }
    service.ensure_usable()?;
    Ok(PressureWitness {
        key,
        source: Arc::downgrade(source),
        owner: Arc::downgrade(owner),
        window: Arc::downgrade(window),
        service: Arc::downgrade(service),
    })
}
pub(super) fn pressure_retired(witness: PressureWitness) -> Result<CubinMlpFaultOutcome> {
    let snapshot = ResourceArbiter::process_shared().snapshot(&witness.key)?;
    if witness.source.upgrade().is_some()
        || witness.owner.upgrade().is_some()
        || witness.window.upgrade().is_some()
        || witness.service.upgrade().is_some()
        || snapshot.reserved != 0
        || snapshot.reservations != 0
        || !snapshot.per_class.is_empty()
        || snapshot.fenced
    {
        return Err(Error::State(
            "MLP pressure original owners/envelope not retired".into(),
        ));
    }
    Ok(CubinMlpFaultOutcome {
        fault: CubinMlpFault::Pressure,
        status: "ADMISSION_REFUSED_BEFORE_H2D",
        submitted_calls: 0,
        native_launches: 0,
        workspace_clears: 0,
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
        weight_window_retained: false,
        job_retained: false,
        workspace_retained: false,
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
fn readbacks(job: &Job) -> [ReadbackIdentity; 2] {
    job.reads
        .each_ref()
        .map(|r| (r.lease_id(), r.offset(), r.bytes()))
}
fn validate_layout(
    projection: MlpProjection,
    fixtures: usize,
    bytes: [usize; 4],
    access: [DeviceAccess; 4],
    initialized: [bool; 4],
    descriptors: [(usize, usize); 2],
) -> Result<()> {
    if projection != MlpProjection::Down64
        || fixtures != 1
        || bytes
            != [
                projection.weight_bytes(),
                projection.input_bytes(),
                projection.output_bytes(),
                WORKSPACE_BYTES,
            ]
        || access
            != [
                DeviceAccess::Read,
                DeviceAccess::Read,
                DeviceAccess::Write,
                DeviceAccess::Write,
            ]
        || initialized != [true, true, false, false]
        || descriptors
            != [
                (0, projection.input_bytes()),
                (0, projection.output_bytes()),
            ]
    {
        return Err(Error::State(
            "MLP exact unready Down64 W/X/Y/workspace/descriptor cut".into(),
        ));
    }
    Ok(())
}
fn validate_window(
    owner: &LeafOwner,
    source: &VerifiedTensorSource,
    window: &Arc<TensorWindow>,
) -> Result<()> {
    let matrix = window.matrix();
    let projection = MlpProjection::Down64;
    let raw = owner
        .weights
        .get(&projection.weight_index())
        .filter(|_| owner.weights.len() == 1)
        .ok_or_else(|| Error::State("MLP exact original DownW packing".into()))?;
    if owner._windows.len() != 1
        || !Arc::ptr_eq(window, &owner._windows[0])
        || matrix.source_identity() != source.source_identity()
        || matrix.tensor().name != "model.language_model.layers.0.mlp.down_proj.weight"
        || matrix.tensor().kind != QuantKind::Bf16
        || matrix.axes() != MatrixAxes::OutputInput
        || matrix.rows() != 1024
        || matrix.columns() != 3584
        || window.row_range() != (0..1024)
        || raw.len() != projection.weight_bytes()
        || window.as_slice() != raw.as_slice()
    {
        return Err(Error::State(
            "MLP retained authentic complete DownW source window".into(),
        ));
    }
    Ok(())
}
fn validate_job(
    job: &Job,
    source: &VerifiedTensorSource,
    service: &Arc<ResidentService>,
) -> Result<()> {
    let [weight, input, output, workspace] = job.accesses.as_slice() else {
        return Err(Error::State("MLP exact original four-lease flight".into()));
    };
    let descriptors = readbacks(job);
    let bindings = [weight, input, output, workspace];
    validate_layout(
        job.projection,
        1,
        bindings.map(|(lease, _)| lease.bytes()),
        bindings.map(|(_, access)| *access),
        bindings.map(|(lease, _)| lease.is_initialized()),
        [
            (descriptors[0].1, descriptors[0].2),
            (descriptors[1].1, descriptors[1].2),
        ],
    )?;
    let distinct = bindings
        .iter()
        .map(|(lease, _)| lease.id())
        .collect::<BTreeSet<_>>();
    if distinct.len() != 4
        || input.0.id() != job.input.id()
        || output.0.id() != job.output.id()
        || descriptors[0].0 != job.input.id()
        || descriptors[1].0 != job.output.id()
        || bindings.iter().any(|(lease, _)| {
            !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != source.source_identity()
        })
    {
        return Err(Error::State(
            "MLP original distinct W/X/Y/workspace source/service IDs".into(),
        ));
    }
    Ok(())
}
fn validate_image(image: &NativeImageIdentity) -> Result<()> {
    let kind = cubecl::cuda::PinnedBf16Cubin::AmpereBf16Gemm128x64Sm80;
    let expected = kind.derivation();
    if image.bytes != expected.image_bytes
        || image.sha256 != expected.image_sha256
        || image.binary_sm != expected.binary_sm
        || image.derived_from != Some(kind)
        || image.captured_from.is_some()
    {
        return Err(Error::State(
            "MLP retained exact derived Down64 image".into(),
        ));
    }
    Ok(())
}

pub(super) struct UnknownWitness {
    key: PhysicalDeviceKey,
    source: Weak<VerifiedTensorSource>,
    owner: Weak<LeafOwner>,
    window: Weak<TensorWindow>,
    job: Weak<Job>,
    programs: Weak<MlpPrograms>,
    native: resident::MlpRetentionWitness,
    service: Weak<ResidentService>,
    workspace: BufferLeaseId,
    descriptors: [ReadbackIdentity; 2],
    resident_bytes: usize,
    uploaded_value_bytes: u64,
    lease_ids: BTreeSet<BufferLeaseId>,
}
impl UnknownWitness {
    pub(super) fn new(
        source: &Arc<VerifiedTensorSource>,
        owner: &Arc<LeafOwner>,
        job: &Arc<Job>,
        workspace: &ResidentLease,
        programs: &Arc<MlpPrograms>,
        service: &Arc<ResidentService>,
    ) -> Result<Self> {
        validate_scope(&owner.fixtures)?;
        validate_job(job, source, service)?;
        let window = owner
            ._windows
            .first()
            .ok_or_else(|| Error::State("MLP original DownW window absent".into()))?;
        validate_window(owner, source, window)?;
        if !Arc::ptr_eq(source, &owner._source)
            || job.accesses[3].0.id() != workspace.id()
            || !Arc::ptr_eq(workspace.service(), service)
        {
            return Err(Error::State(
                "MLP original source/workspace owner identity".into(),
            ));
        }
        let managed = job.accesses.iter().try_fold(0usize, |n, (lease, _)| {
            n.checked_add(lease.bytes())
                .ok_or_else(|| Error::Limit("MLP fault bytes overflow".into()))
        })?;
        let uploaded = job.accesses[0].0.bytes() as u64 + job.input.bytes() as u64;
        let key = service.physical_key();
        let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
        let resident_bytes = service.resident_bytes()?;
        if snapshot.fenced
            || !original_envelope(&snapshot)
            || service.pending_events()? != 0
            || managed.checked_add(FOREIGN_BYTES) != Some(resident_bytes)
            || service.uploaded_value_bytes() != uploaded
            || quarantined_handles_for(service.client())? != 0
            || quarantined_completion_owners_for(service.client())? != 0
        {
            return Err(Error::State(
                "MLP unknown requires original prepared Down64 cut".into(),
            ));
        }
        service.ensure_usable()?;
        Ok(Self {
            key,
            source: Arc::downgrade(source),
            owner: Arc::downgrade(owner),
            window: Arc::downgrade(window),
            job: Arc::downgrade(job),
            programs: Arc::downgrade(programs),
            native: programs.retention_witness()?,
            service: Arc::downgrade(service),
            workspace: workspace.id(),
            descriptors: readbacks(job),
            resident_bytes,
            uploaded_value_bytes: uploaded,
            lease_ids: job.accesses.iter().map(|(lease, _)| lease.id()).collect(),
        })
    }
    /// Weak upgrades/original host ledgers only, after ALL caller owners dropped.
    pub(super) fn inspect(self, submitted: resident::MlpSubmitted) -> Result<CubinMlpFaultOutcome> {
        let source = self.source.upgrade();
        let owner = self.owner.upgrade();
        let window = self.window.upgrade();
        let job = self.job.upgrade();
        let programs = self.programs.upgrade();
        let service = self.service.upgrade();
        let retained = [
            source.is_some(),
            owner.is_some(),
            window.is_some(),
            job.is_some(),
            programs.is_some(),
            service.is_some(),
        ];
        let native = self.native.inspect()?;
        validate_image(&native.image)?;
        let source = source.ok_or_else(|| Error::State("MLP quarantine source lost".into()))?;
        let owner = owner.ok_or_else(|| Error::State("MLP quarantine source owner lost".into()))?;
        let window =
            window.ok_or_else(|| Error::State("MLP quarantine DownW window lost".into()))?;
        let job = job.ok_or_else(|| Error::State("MLP quarantine Job lost".into()))?;
        let service = service.ok_or_else(|| Error::State("MLP quarantine service lost".into()))?;
        validate_job(&job, &source, &service)?;
        validate_window(&owner, &source, &window)?;
        if !Arc::ptr_eq(&source, &owner._source)
            || job.accesses[3].0.id() != self.workspace
            || readbacks(&job) != self.descriptors
            || job
                .accesses
                .iter()
                .map(|(lease, _)| lease.id())
                .collect::<BTreeSet<_>>()
                != self.lease_ids
        {
            return Err(Error::State(
                "MLP retained original W/X/Y/workspace/descriptor cohort".into(),
            ));
        }
        let resident_bytes = service.resident_bytes()?;
        let pending_events = service.pending_events()?;
        let uploaded_value_bytes = service.uploaded_value_bytes();
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        let snapshot = ResourceArbiter::process_shared().snapshot(&self.key)?;
        if retained.contains(&false)
            || submitted.kernels != 1
            || submitted.clears != 1
            || native.image_owners != 1
            || native.kernel_owners != 1
            || native.native_charge_bytes != FOREIGN_BYTES
            || native.image_admission_bytes == 0
            || service.ensure_usable().is_ok()
            || resident_bytes != self.resident_bytes
            || pending_events != 1
            || uploaded_value_bytes != self.uploaded_value_bytes
            || handles < self.lease_ids.len()
            || owners < 3
            || !snapshot.fenced
            || !original_envelope(&snapshot)
        {
            return Err(Error::State(
                "MLP unknown did not retain clear/kernel original graph".into(),
            ));
        }
        let host_workspace_charge_bytes =
            owner
                ._host
                .iter()
                .try_fold(native.image_admission_bytes, |n, charge| {
                    n.checked_add(charge.bytes())
                        .ok_or_else(|| Error::Limit("MLP retained host charge overflow".into()))
                })?;
        Ok(CubinMlpFaultOutcome {
            fault: CubinMlpFault::Unknown,
            status: "UNKNOWN_RETAINED",
            submitted_calls: 1,
            native_launches: submitted.kernels,
            workspace_clears: submitted.clears,
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
            weight_window_retained: retained[2],
            job_retained: retained[3],
            workspace_retained: true,
            programs_retained: retained[4],
            shared_images_retained: true,
            service_retained: retained[5],
            readback_cohort_retained: true,
            readback_descriptors: self.descriptors.len(),
            prepared_physical_leases: self.lease_ids.len(),
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
