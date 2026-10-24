//! No readback/Ready/sync/recovery after actual solve uncertainty.
use super::resident::Prepared;
use super::*;
use cubecl::Device;
use std::sync::Weak;
use zerogravity_core::device_events::BufferLeaseId;
use zerogravity_core::resource_arbiter::{ResourceArbiter, ResourceOwner};
pub(super) const UNKNOWN: &str = "deliberate original47 solve leaf unknown after actual solve";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solve47Fault {
    Pressure,
    UnknownAfterSolve,
    CancelBeforeRecompute,
}
#[derive(Debug)]
pub struct Solve47FaultOutcome {
    pub fault: Solve47Fault,
    pub actual_zero_dispatches: usize,
    pub actual_solve_dispatches: usize,
    pub actual_recompute_dispatches: usize,
    pub prepared_buffers: usize,
    pub submitted_buffers: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub resident_bytes: usize,
    pub quarantined_handles: usize,
    pub quarantined_owners: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub source_and_prepared_retained: bool,
    pub all_outputs_unpromoted: bool,
    pub ready_outputs: usize,
    pub readback_calls: usize,
}
pub(super) struct Probe {
    model: Weak<OriginalWindowModelSource>,
    prepared: Weak<Prepared>,
    service: Weak<ResidentService>,
    ids: Vec<BufferLeaseId>,
    reads: Vec<(BufferLeaseId, usize, usize)>,
    resident: usize,
    pub(super) dispatches: usize,
}
impl Probe {
    pub(super) fn new(prepared: &Arc<Prepared>) -> Result<Self> {
        Ok(Self {
            model: Arc::downgrade(prepared.owner.bound.source()),
            prepared: Arc::downgrade(prepared),
            service: Arc::downgrade(&prepared.service),
            ids: prepared.leases().iter().map(ResidentLease::id).collect(),
            reads: prepared.jobs[0]
                .reads
                .iter()
                .map(|r| (r.lease_id(), r.offset(), r.bytes()))
                .collect(),
            resident: prepared.service.resident_bytes()?,
            dispatches: 0,
        })
    }
    fn inspect(self, fault: Solve47Fault) -> Result<Solve47FaultOutcome> {
        // Only Weak values crossed run(); all actual locals/armed flight/slot dropped.
        let model = self
            .model
            .upgrade()
            .ok_or_else(|| Error::State("solve47 source lost after cut".into()))?;
        let prepared = self
            .prepared
            .upgrade()
            .ok_or_else(|| Error::State("solve47 actual code/input/output owner lost".into()))?;
        let service = self
            .service
            .upgrade()
            .ok_or_else(|| Error::State("solve47 service lost".into()))?;
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        let physical = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
        let all_unpromoted = prepared
            .jobs
            .iter()
            .flat_map(|j| &j.outputs)
            .all(|l| !l.is_initialized());
        let ids = prepared
            .leases()
            .iter()
            .map(ResidentLease::id)
            .collect::<Vec<_>>();
        let reads = prepared.jobs[0]
            .reads
            .iter()
            .map(|r| (r.lease_id(), r.offset(), r.bytes()))
            .collect::<Vec<_>>();
        if self.dispatches != 2
            || ids != self.ids
            || ids.len() != 19
            || ids.iter().copied().collect::<BTreeSet<_>>().len() != 19
            || reads != self.reads
            || reads.len() != 10
            || prepared.jobs[0].accesses.len() != 10
            || handles != 10
            || owners < 1
            || !Arc::ptr_eq(&model, prepared.owner.bound.source())
            || !Arc::ptr_eq(&service, &prepared.service)
            || service.resident_bytes()? != self.resident
            || service.uploaded_value_bytes() != INPUT_BYTES as u64
            || service.pending_events()? != 1
            || !all_unpromoted
            || !physical.fenced
            || physical.reserved != PROVIDER as u64
        {
            return Err(Error::State(
                "solve47 original quarantine full prepared19/submitted10 graph mismatch".into(),
            ));
        }
        Ok(Solve47FaultOutcome {
            fault,
            actual_zero_dispatches: 1,
            actual_solve_dispatches: 1,
            actual_recompute_dispatches: 0,
            prepared_buffers: 19,
            submitted_buffers: 10,
            pending_events: 1,
            uploaded_value_bytes: INPUT_BYTES as u64,
            resident_bytes: self.resident,
            quarantined_handles: handles,
            quarantined_owners: owners,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: true,
            source_and_prepared_retained: true,
            all_outputs_unpromoted: true,
            ready_outputs: 0,
            readback_calls: 0,
        })
    }
}
pub async fn qualify_solve47_fault(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixture: Option<Solve47Fixture>,
    fault: Solve47Fault,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<Solve47FaultOutcome> {
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release CUDA required for original47 solve fault".into(),
        ));
    }
    if fault == Solve47Fault::Pressure {
        if fixture.is_some() {
            return Err(Error::State(
                "solve47 pressure has no fixture/H2D prerequisite".into(),
            ));
        }
        let weak = Arc::downgrade(&model);
        let bound = model.bind_full_model_source()?;
        if bound.model_profile() != FullModelProfile::Qwen08
            || model.source().staging_chunk_bytes() != MIB
        {
            return Err(Error::Unsupported(
                "closed .8B source/1MiB solve47 pressure".into(),
            ));
        }
        let service = ResidentService::new_pinned_cuda(
            device,
            model.source_identity(),
            ResourceOwner::new("vllm47-source", "solve-leaf-cap-pressure")?,
            PROVIDER,
            device_cap,
        )?;
        let key = service.physical_key();
        let refusal = service.allocate_uninitialized(PROVIDER + 1, BufferRole::Scratch);
        if !matches!(refusal,Err(Error::Limit(ref why)) if why=="resident plus opaque provider budget exceeded")
            || service.resident_bytes()? != 0
            || service.pending_events()? != 0
            || service.uploaded_value_bytes() != 0
            || service.ensure_usable().is_err()
        {
            return Err(Error::State(
                "solve47 cap+1 before allocations/H2D/math".into(),
            ));
        }
        drop((refusal, bound, model, service));
        let physical = ResourceArbiter::process_shared().snapshot(&key)?;
        if weak.upgrade().is_some() || physical.reserved != 0 || physical.fenced {
            return Err(Error::State(
                "solve47 pressure source/physical final0".into(),
            ));
        }
        return Ok(Solve47FaultOutcome {
            fault,
            actual_zero_dispatches: 0,
            actual_solve_dispatches: 0,
            actual_recompute_dispatches: 0,
            prepared_buffers: 0,
            submitted_buffers: 0,
            pending_events: 0,
            uploaded_value_bytes: 0,
            resident_bytes: 0,
            quarantined_handles: 0,
            quarantined_owners: 0,
            physical_reserved_bytes: 0,
            physical_fenced: false,
            source_and_prepared_retained: false,
            all_outputs_unpromoted: true,
            ready_outputs: 0,
            readback_calls: 0,
        });
    }
    let fixture = fixture
        .ok_or_else(|| Error::State("authentic declared solve47 input leaf absent".into()))?;
    let mut probe = None;
    let result = super::qualification::run(
        device,
        model,
        fixture,
        device_cap,
        super::qualification::RunHooks {
            fault: Some(fault),
            probe: &mut probe,
            validate: |_: Solve47Readback<'_>| {
                Err(Error::State(
                    "solve47 uncertain path reached readback".into(),
                ))
            },
        },
        cancelled,
    )
    .await;
    let intended = matches!((&fault,&result),(Solve47Fault::UnknownAfterSolve,Err(Error::State(why))) if why==UNKNOWN)
        || matches!((&fault,&result),(Solve47Fault::CancelBeforeRecompute,Err(Error::State(why))) if why==CANCELLED);
    if !intended {
        return match result {
            Err(error) => Err(error),
            Ok(_) => Err(Error::State("solve47 cut absent".into())),
        };
    }
    probe
        .ok_or_else(|| Error::State("solve47 original submitted witness absent".into()))?
        .inspect(fault)
}
