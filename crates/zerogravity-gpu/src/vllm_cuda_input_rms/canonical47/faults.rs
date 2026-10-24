//! Deliberate diagnostic cuts; original armed flight is the quarantine owner.
use super::*;
use crate::resident_leases::ResidentService;
use cubecl::Device;
use std::sync::Weak;
use zerogravity_core::device_events::{BufferLeaseId, BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

const PROVIDER: usize = 128 * 1024 * 1024;
const UNKNOWN: &str = "deliberate canonical47 RMS unknown after actual Mean";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canonical47RmsFault {
    Pressure,
    UnknownAfterMean,
    CancelAfterMean,
}
#[derive(Clone, Debug)]
pub struct Canonical47RmsFaultOutcome {
    pub fault: Canonical47RmsFault,
    pub actual_stage_dispatches: usize,
    pub uploaded_value_bytes: u64,
    pub prepared_buffers: usize,
    pub quarantined_handles: usize,
    pub quarantined_owners: usize,
    pub pending_events: usize,
    pub resident_bytes: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub original_source_retained: bool,
    pub original_prepared_retained: bool,
    pub all_write_outputs_unpromoted: bool,
    pub ready_outputs: usize,
    pub readback_calls: usize,
}
struct Probe {
    model: Weak<OriginalWindowModelSource>,
    service: Weak<ResidentService>,
    prepared: Weak<RmsPrepared>,
    ids: [BufferLeaseId; 12],
    resident: usize,
    uploaded: u64,
    stages: usize,
}
async fn cut(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fault: Canonical47RmsFault,
    device_cap: u64,
    probe: &mut Option<Probe>,
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    not_cancelled(cancelled)?;
    let model_weak = Arc::downgrade(&model);
    let source = Canonical47RmsSource::bind(model)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source().source_identity(),
        ResourceOwner::new("vllm08-cuda-source", "ordinary-rms-uncertain-cut")?,
        PROVIDER,
        device_cap,
    )?;
    let requests = [AllocationRequest {
        kind: AllocationClass::Other,
        bytes: PROVIDER as u64,
    }];
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let prepared =
        RmsPrepared::prepare_canonical47(service.clone(), source, &PROMPT, 1, cancelled).await?;
    slot.finish()?;
    if prepared.accesses().len() != 12 || prepared.outputs().len() != 10 {
        return Err(Error::State(
            "canonical47 RMS actual prepared cohort12/10".into(),
        ));
    }
    let before = Probe {
        model: model_weak,
        service: Arc::downgrade(&service),
        prepared: Arc::downgrade(&prepared),
        ids: std::array::from_fn(|i| prepared.accesses()[i].0.id()),
        resident: service.resident_bytes()?,
        uploaded: service.uploaded_value_bytes(),
        stages: 0,
    };
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let mut flight = service.submit(1, prepared.accesses())?;
    // The same positive-path owner retains actual source/windows/specs/leases.
    flight.retain_completion_owner(prepared.completion_owner())?;
    *probe = Some(before);
    let actual = prepared.enqueue_canonical47_through_mean()?;
    probe.as_mut().unwrap().stages = actual;
    match fault {
        Canonical47RmsFault::UnknownAfterMean => Err(Error::State(UNKNOWN.into())),
        Canonical47RmsFault::CancelAfterMean => {
            let flag = AtomicBool::new(true);
            not_cancelled(Some(&flag))
        }
        Canonical47RmsFault::Pressure => {
            Err(Error::State("pressure cannot reach RMS enqueue".into()))
        }
    }
    // Armed flight/fair slot and ALL strong locals drop before outer inspection.
}
fn intended(fault: Canonical47RmsFault, result: &Result<()>) -> bool {
    matches!((fault,result),
        (Canonical47RmsFault::UnknownAfterMean,Err(Error::State(why))) if why==UNKNOWN)
        || matches!((fault,result),
        (Canonical47RmsFault::CancelAfterMean,Err(Error::State(why))) if why=="canonical47 RMS cancelled")
}
async fn pressure(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<Canonical47RmsFaultOutcome> {
    let weak = Arc::downgrade(&model);
    let source = Canonical47RmsSource::bind(model)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source().source_identity(),
        ResourceOwner::new("vllm08-cuda-source", "ordinary-rms-cap-pressure")?,
        PROVIDER,
        device_cap,
    )?;
    let key = service.physical_key();
    not_cancelled(cancelled)?;
    let refusal = service.allocate_uninitialized(PROVIDER + 1, BufferRole::Scratch);
    if !matches!(refusal,Err(Error::Limit(ref why)) if why=="resident plus opaque provider budget exceeded")
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || service.ensure_usable().is_err()
    {
        return Err(Error::State(
            "canonical47 original cap+1 refusal before handles/H2D/math".into(),
        ));
    }
    drop((refusal, source, service));
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    if weak.upgrade().is_some() || physical.reserved != 0 || physical.fenced {
        return Err(Error::State(
            "canonical47 original pressure source/envelope did not retire".into(),
        ));
    }
    Ok(Canonical47RmsFaultOutcome {
        fault: Canonical47RmsFault::Pressure,
        actual_stage_dispatches: 0,
        uploaded_value_bytes: 0,
        prepared_buffers: 0,
        quarantined_handles: 0,
        quarantined_owners: 0,
        pending_events: 0,
        resident_bytes: 0,
        physical_reserved_bytes: 0,
        physical_fenced: false,
        original_source_retained: false,
        original_prepared_retained: false,
        all_write_outputs_unpromoted: true,
        ready_outputs: 0,
        readback_calls: 0,
    })
}
/// Never completes, reads, recovers or unloads after an uncertain cut.
pub async fn qualify_canonical47_rms_fault(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fault: Canonical47RmsFault,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<Canonical47RmsFaultOutcome> {
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required forcanonical47 RMS fault".into(),
        ));
    }
    if fault == Canonical47RmsFault::Pressure {
        return pressure(device, model, device_cap, cancelled).await;
    }
    let mut probe = None;
    let result = cut(device, model, fault, device_cap, &mut probe, cancelled).await;
    if !intended(fault, &result) {
        return match result {
            Err(error) => Err(error),
            Ok(()) => Err(Error::State("canonical47 RMS cut was not reached".into())),
        };
    }
    let probe = probe
        .ok_or_else(|| Error::State("canonical47 RMS actual submission witness absent".into()))?;
    let source = probe
        .model
        .upgrade()
        .ok_or_else(|| Error::State("canonical47 quarantine lost original model/source".into()))?;
    let service = probe
        .service
        .upgrade()
        .ok_or_else(|| Error::State("canonical47 quarantine lost original service".into()))?;
    let prepared = probe
        .prepared
        .upgrade()
        .ok_or_else(|| Error::State("canonical47 quarantine lost actual prepared owner".into()))?;
    let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
    let owners = crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
    let physical = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
    let actual_ids: [BufferLeaseId; 12] = std::array::from_fn(|i| prepared.accesses()[i].0.id());
    let writes_unpromoted = prepared
        .outputs()
        .iter()
        .all(|lease| !lease.is_initialized());
    if probe.stages != 5
        || actual_ids != probe.ids
        || service.resident_bytes()? != probe.resident
        || service.uploaded_value_bytes() != probe.uploaded
        || probe.uploaded != 98304
        || service.pending_events()? != 1
        || handles < 12
        || owners < 1
        || !physical.fenced
        || physical.reserved != PROVIDER as u64
        || service.ensure_usable().is_ok()
        || !writes_unpromoted
        || source.source_identity() != service.source_identity()
        || prepared.accesses().iter().any(|(lease, access)| {
            !Arc::ptr_eq(lease.service(), &service)
                || lease.source_identity() != source.source_identity()
                || (*access == DeviceAccess::Read && !lease.is_initialized())
        })
    {
        return Err(Error::State(
            "canonical47 original uncertain actual source/cohort/fence changed".into(),
        ));
    }
    Ok(Canonical47RmsFaultOutcome {
        fault,
        actual_stage_dispatches: probe.stages,
        uploaded_value_bytes: probe.uploaded,
        prepared_buffers: 12,
        quarantined_handles: handles,
        quarantined_owners: owners,
        pending_events: 1,
        resident_bytes: probe.resident,
        physical_reserved_bytes: physical.reserved,
        physical_fenced: physical.fenced,
        original_source_retained: true,
        original_prepared_retained: true,
        all_write_outputs_unpromoted: writes_unpromoted,
        ready_outputs: 0,
        readback_calls: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deliberate_cut_acceptance_requires_exact_original_cause() {
        assert!(intended(
            Canonical47RmsFault::UnknownAfterMean,
            &Err(Error::State(UNKNOWN.into()))
        ));
        assert!(intended(
            Canonical47RmsFault::CancelAfterMean,
            &Err(Error::State("canonical47 RMS cancelled".into()))
        ));
        for fault in [
            Canonical47RmsFault::UnknownAfterMean,
            Canonical47RmsFault::CancelAfterMean,
            Canonical47RmsFault::Pressure,
        ] {
            assert!(!intended(fault, &Ok(())));
            assert!(!intended(
                fault,
                &Err(Error::State("unrelated after cut".into()))
            ));
            assert!(!intended(fault, &Err(Error::Limit(UNKNOWN.into()))));
        }
        assert!(!intended(
            Canonical47RmsFault::UnknownAfterMean,
            &Err(Error::State("canonical47 RMS cancelled".into()))
        ));
    }
}
