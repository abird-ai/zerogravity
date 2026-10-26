//! SAME positive preparation/first native call; Weak-only uncertain inspection.
use super::super::projection_cubin::resident::NativeRetentionWitness;
use super::qualification::{PreparedJobs, Qwen2bProjectionFixture, SourceOwner};
use super::*;
use crate::cuda::native::RetainedSource;
use cubecl::Device;
use std::sync::Weak;
use zerogravity_core::device_events::BufferLeaseId;
use zerogravity_core::resource_arbiter::{ResourceArbiter, ResourceOwner};
const MIB: usize = 1024 * 1024;
pub(super) const UNKNOWN: &str = "deliberate original2B projection unknown after actual QKVZ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen2bProjectionFault {
    Pressure,
    UnknownAfterQkvz,
    CancelBeforeBa,
}
#[derive(Debug)]
pub struct Qwen2bProjectionFaultOutcome {
    pub fault: Qwen2bProjectionFault,
    pub actual_native_launches: usize,
    pub actual_prepared_buffers: usize,
    pub actual_submitted_buffers: usize,
    pub uploaded_value_bytes: u64,
    pub pending_events: usize,
    pub resident_bytes: usize,
    pub quarantined_handles: usize,
    pub quarantined_owners: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub image_owners: usize,
    pub kernel_records: usize,
    pub native_charge_bytes: usize,
    pub original_source_retained: bool,
    pub original_prepared_retained: bool,
    pub all_outputs_unpromoted: bool,
    pub ready_outputs: usize,
    pub readback_calls: usize,
}
fn lease_ids(jobs: &PreparedJobs, weights: &ProjectionWeights) -> Vec<BufferLeaseId> {
    weights
        .accesses()
        .into_iter()
        .map(|(lease, _)| lease.id())
        .chain(std::iter::once(jobs.workspace.id()))
        .chain(jobs.jobs.iter().flat_map(|job| {
            std::iter::once(job.input.id()).chain(job.outputs.iter().map(ResidentLease::id))
        }))
        .collect()
}
pub(super) struct Probe {
    source: Weak<dyn std::any::Any + Send + Sync>,
    owner: Weak<SourceOwner>,
    jobs: Weak<PreparedJobs>,
    programs: Weak<ProjectionPrograms>,
    weights: Weak<ProjectionWeights>,
    service: Weak<ResidentService>,
    native_graph: NativeRetentionWitness,
    ids: Vec<BufferLeaseId>,
    submitted_ids: Vec<BufferLeaseId>,
    resident: usize,
    uploaded: u64,
    pub(super) native: usize,
}
impl Probe {
    pub(super) fn new(
        owner: &Arc<SourceOwner>,
        jobs: &Arc<PreparedJobs>,
        programs: &Arc<ProjectionPrograms>,
        weights: &Arc<ProjectionWeights>,
        service: &Arc<ResidentService>,
    ) -> Result<Self> {
        let kernels = programs
            .launches
            .values()
            .flatten()
            .map(|launch| launch.kernel.clone())
            .collect::<Vec<_>>();
        let native_graph = programs.images.retention_witness(&kernels)?;
        Ok(Self {
            source: owner.bound.weak_owner(),
            owner: Arc::downgrade(owner),
            jobs: Arc::downgrade(jobs),
            programs: Arc::downgrade(programs),
            weights: Arc::downgrade(weights),
            service: Arc::downgrade(service),
            native_graph,
            ids: lease_ids(jobs, weights),
            submitted_ids: jobs.jobs[0]
                .accesses
                .iter()
                .map(|(lease, _)| lease.id())
                .collect(),
            resident: service.resident_bytes()?,
            uploaded: service.uploaded_value_bytes(),
            native: 0,
        })
    }
    fn inspect(self, fault: Qwen2bProjectionFault) -> Result<Qwen2bProjectionFaultOutcome> {
        // Every actual run() local/armed flight/fair slot has dropped first.
        let _source = self
            .source
            .upgrade()
            .ok_or_else(|| Error::State("2B uncertain native source lost".into()))?;
        let _owner = self
            .owner
            .upgrade()
            .ok_or_else(|| Error::State("2B uncertain original host/window owner lost".into()))?;
        let jobs = self
            .jobs
            .upgrade()
            .ok_or_else(|| Error::State("2B uncertain actual prepared cohort lost".into()))?;
        let _programs = self
            .programs
            .upgrade()
            .ok_or_else(|| Error::State("2B uncertain native program owner lost".into()))?;
        let weights = self
            .weights
            .upgrade()
            .ok_or_else(|| Error::State("2B uncertain original packed W owner lost".into()))?;
        let service = self
            .service
            .upgrade()
            .ok_or_else(|| Error::State("2B uncertain service owner lost".into()))?;
        let native = self.native_graph.inspect()?;
        let physical = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        let outputs_unpromoted = jobs
            .jobs
            .iter()
            .flat_map(|job| &job.outputs)
            .all(|lease| !lease.is_initialized());
        let submitted_ids = jobs.jobs[0]
            .accesses
            .iter()
            .map(|(lease, _)| lease.id())
            .collect::<Vec<_>>();
        if self.native != 1
            || lease_ids(&jobs, &weights) != self.ids
            || self.ids.len() != 18
            || self.ids.iter().copied().collect::<BTreeSet<_>>().len() != 18
            || submitted_ids != self.submitted_ids
            || submitted_ids.len() != 5
            || submitted_ids.iter().copied().collect::<BTreeSet<_>>().len() != 5
            || submitted_ids.iter().any(|id| !self.ids.contains(id))
            || service.resident_bytes()? != self.resident
            || service.uploaded_value_bytes() != self.uploaded
            || service.pending_events()? != 1
            // Registry holds this flight's handles. The retained jobs owner
            // separately keeps all eighteen prepared buffers alive.
            || handles != submitted_ids.len()
            || owners < 3
            || !outputs_unpromoted
            || !physical.fenced
            || physical.reserved != 128 * MIB as u64
            || native.image_owners != 4
            || native.kernel_owners != 7
            || native.native_charge_bytes != 64 * MIB
        {
            return Err(Error::State(
                "original2B actual native uncertain cohort/fence differs".into(),
            ));
        }
        Ok(Qwen2bProjectionFaultOutcome {
            fault,
            actual_native_launches: self.native,
            actual_prepared_buffers: self.ids.len(),
            actual_submitted_buffers: submitted_ids.len(),
            uploaded_value_bytes: self.uploaded,
            pending_events: 1,
            resident_bytes: self.resident,
            quarantined_handles: handles,
            quarantined_owners: owners,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: true,
            image_owners: native.image_owners,
            kernel_records: native.kernel_owners,
            native_charge_bytes: native.native_charge_bytes,
            original_source_retained: true,
            original_prepared_retained: true,
            all_outputs_unpromoted: true,
            ready_outputs: 0,
            readback_calls: 0,
        })
    }
}
async fn pressure(
    device: Device,
    bound: Arc<RetainedSource>,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<Qwen2bProjectionFaultOutcome> {
    not_cancelled(cancelled)?;
    let source = bound.weak_owner();
    let service = ResidentService::new_pinned_cuda(
        device,
        bound.source_identity(),
        ResourceOwner::new("qwen2b-cuda-cubin", "original-projection-pressure")?,
        128 * MIB,
        device_cap,
    )?;
    let key = service.physical_key();
    let refusal = service.allocate_uninitialized(128 * MIB + 1, BufferRole::Scratch);
    if !matches!(refusal,Err(Error::Limit(ref why)) if why=="resident plus opaque provider budget exceeded")
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || service.ensure_usable().is_err()
    {
        return Err(Error::State(
            "2B projection cap+1 must refuse before original handles/H2D/native".into(),
        ));
    }
    drop((refusal, bound, service));
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    if source.upgrade().is_some() || physical.reserved != 0 || physical.fenced {
        return Err(Error::State(
            "2B projection pressure source/envelope not retired".into(),
        ));
    }
    Ok(Qwen2bProjectionFaultOutcome {
        fault: Qwen2bProjectionFault::Pressure,
        actual_native_launches: 0,
        actual_prepared_buffers: 0,
        actual_submitted_buffers: 0,
        uploaded_value_bytes: 0,
        pending_events: 0,
        resident_bytes: 0,
        quarantined_handles: 0,
        quarantined_owners: 0,
        physical_reserved_bytes: 0,
        physical_fenced: false,
        image_owners: 0,
        kernel_records: 0,
        native_charge_bytes: 0,
        original_source_retained: false,
        original_prepared_retained: false,
        all_outputs_unpromoted: true,
        ready_outputs: 0,
        readback_calls: 0,
    })
}
pub async fn qualify_qwen2b_projection_fault(
    device: Device,
    bound: Arc<RetainedSource>,
    fixtures: Vec<Qwen2bProjectionFixture>,
    inputs: CubinQualificationInputs,
    fault: Qwen2bProjectionFault,
    cancelled: Option<&AtomicBool>,
) -> Result<Qwen2bProjectionFaultOutcome> {
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA2B projection fault".into(),
        ));
    }
    if fault == Qwen2bProjectionFault::Pressure {
        let result = pressure(device, bound, inputs.device_cap, cancelled).await;
        drop((fixtures, inputs));
        return result;
    }
    let mut probe = None;
    let result = super::qualification::run(
        device,
        bound,
        fixtures,
        inputs,
        cancelled,
        super::qualification::RunHooks {
            fault: Some(fault),
            probe: &mut probe,
            validate_actual: |_: super::qualification::Qwen2bProjectionReadback<'_>| {
                Err(Error::State(
                    "uncertain2B projection reached trusted callback".into(),
                ))
            },
        },
    )
    .await;
    let intended = matches!((&result,fault),(Err(Error::State(why)),Qwen2bProjectionFault::UnknownAfterQkvz) if why==UNKNOWN)
        || matches!((&result,fault),(Err(Error::State(why)),Qwen2bProjectionFault::CancelBeforeBa) if why=="closed original cubin qualification cancelled");
    if !intended {
        return match result {
            Err(error) => Err(error),
            Ok(_) => Err(Error::State(
                "actual2B projection uncertain cut not reached".into(),
            )),
        };
    }
    probe
        .ok_or_else(|| Error::State("2B actual native weak witness absent".into()))?
        .inspect(fault)
}
