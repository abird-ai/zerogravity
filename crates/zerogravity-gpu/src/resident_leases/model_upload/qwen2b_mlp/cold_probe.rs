//! Real immutable W transport/source proof; no MLP kernel or computed activation.
use super::*;
use crate::resident_leases::{
    Qwen2bMlpSourceVerification as SourceVerification, Qwen2bMlpWeightCohort as Cohort,
    Qwen2bMlpWeights as Weights,
};
use cubecl::Device;
use std::sync::{Weak, atomic::AtomicUsize};
use zerogravity_core::device_events::BufferLeaseId;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;
const MIB: usize = 1024 * 1024;
const BYTES: usize = 75_497_472;
const LEADING: usize = 16_384;
const UNKNOWN: &str = "deliberate cold2B W unknown after completed source read";
const REFUSED: &str = "deliberate cold2B W trusted validator refusal";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen2bMlpWeightMode {
    Complete,
    Pressure,
    LegacyUploadRefusal,
    UnknownAfterFirstRead,
    UnknownAfterMiddleRead,
    UnknownAfterLastRead,
    CancelAfterFirstRead,
    CancelAfterMiddleRead,
    CancelAfterLastRead,
    ValidatorRefusal,
    EmptyComputeOutputs,
}
impl Qwen2bMlpWeightMode {
    fn cut(self) -> Option<(usize, bool)> {
        match self {
            Self::UnknownAfterFirstRead => Some((1, false)),
            Self::UnknownAfterMiddleRead => Some((37, false)),
            Self::UnknownAfterLastRead => Some((73, false)),
            Self::CancelAfterFirstRead => Some((1, true)),
            Self::CancelAfterMiddleRead => Some((37, true)),
            Self::CancelAfterLastRead => Some((73, true)),
            _ => None,
        }
    }
    fn is_fault(self) -> bool {
        !matches!(
            self,
            Self::Complete | Self::Pressure | Self::LegacyUploadRefusal
        )
    }
}
#[derive(Debug)]
pub struct Qwen2bMlpWeightOutcome {
    pub mode: Qwen2bMlpWeightMode,
    pub status: &'static str,
    pub rows: usize,
    pub copies_enqueued: usize,
    pub copy_checkpoints: usize,
    pub initialized_weights: bool,
    pub serial_reads_completed: usize,
    pub source_bytes_compared: usize,
    pub leading_readback_bytes: usize,
    pub trusted_validator_entered: bool,
    pub read_only_flight_retired: bool,
    pub diagnostic_ready_outputs: usize,
    pub native_launches: usize,
    pub prepared_buffers: usize,
    pub submitted_buffers: usize,
    pub resident_bytes: usize,
    pub uploaded_value_bytes: u64,
    pub pending_events: usize,
    pub quarantined_handles: usize,
    pub quarantined_owners: usize,
    pub source_record_weights_plan_retained: bool,
    pub record_admission_retained: bool,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub root_published: bool,
    pub native_mlp_qualified: bool,
}
pub(super) struct ReadProbe {
    mode: Qwen2bMlpWeightMode,
    completed: AtomicUsize,
    bytes: AtomicUsize,
    validator: AtomicBool,
}
impl ReadProbe {
    fn new(mode: Qwen2bMlpWeightMode) -> Self {
        Self {
            mode,
            completed: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            validator: AtomicBool::new(false),
        }
    }
    pub(super) fn after_completed_read(&self, count: usize, bytes: usize) -> Result<()> {
        self.completed.store(count, Ordering::Release);
        self.bytes.fetch_add(bytes, Ordering::AcqRel);
        if let Some((cut, cancel)) = self.mode.cut()
            && count == cut
        {
            // Actual D2H+source comparison finished. No client failure is invented.
            if cancel {
                let flag = AtomicBool::new(true);
                not_cancelled(Some(&flag))?;
            }
            return Err(Error::State(UNKNOWN.into()));
        }
        Ok(())
    }
}
struct ColdOwners {
    cohort: Arc<Cohort>,
    record_admission: WindowWorkspace,
    _metadata: WindowWorkspace,
}
struct Prepared {
    // Actual W/descriptors drop before record/metadata admissions in owner.
    weights: Arc<Weights>,
    reads: Vec<ResidentReadback>,
    owner: Arc<ColdOwners>,
}
struct Inputs {
    // Stack-only move. Record drops before its caller's actual admission.
    record: Arc<RecordedVllm08Model>,
    source: BoundFullModelSource,
    record_admission: WindowWorkspace,
}
impl Prepared {
    fn compare_leading(&self, raw: &[Bytes], cancelled: Option<&AtomicBool>) -> Result<()> {
        if raw.len() != 2 {
            return Err(Error::State("two actual leading W rows".into()));
        }
        for (weight, actual) in [&self.weights.gate_up, &self.weights.down].iter().zip(raw) {
            let matrix = weight
                .owner
                .matrices
                .first()
                .ok_or_else(|| Error::State("actual W first member".into()))?;
            let (window, _, bytes) =
                weight.checked_verification_window(matrix, 0..1, 0, cancelled)?;
            if actual.len() != bytes || &actual[..] != window.as_slice() {
                return Err(Error::State(
                    "actual leading W row differs fresh source".into(),
                ));
            }
        }
        not_cancelled(cancelled)
    }
}
struct Probe {
    source: Weak<OriginalWindowModelSource>,
    prepared: Weak<Prepared>,
    plan: Weak<SourceVerification>,
    service: Weak<ResidentService>,
    ids: Vec<BufferLeaseId>,
    reads: Vec<(BufferLeaseId, usize, usize)>,
    rows: usize,
}
impl Probe {
    fn new(prepared: &Arc<Prepared>, plan: &Arc<Qwen2bMlpSourceVerification>) -> Self {
        Self {
            source: Arc::downgrade(prepared.owner.cohort.source.source()),
            prepared: Arc::downgrade(prepared),
            plan: Arc::downgrade(plan),
            service: Arc::downgrade(&prepared.weights.service),
            ids: prepared
                .weights
                .accesses()
                .iter()
                .map(|(v, _)| v.id())
                .collect(),
            reads: prepared
                .reads
                .iter()
                .map(|v| (v.lease_id(), v.offset(), v.bytes()))
                .collect(),
            rows: prepared.owner.cohort.record.call().rows(),
        }
    }
    fn inspect(self, mode: Qwen2bMlpWeightMode) -> Result<Qwen2bMlpWeightOutcome> {
        // All caller/run/armed-flight strong owners have dropped; only Weak crosses run.
        let source = self
            .source
            .upgrade()
            .ok_or_else(|| Error::State("cold2B actual source lost".into()))?;
        let prepared = self
            .prepared
            .upgrade()
            .ok_or_else(|| Error::State("cold2B real W/record owner lost".into()))?;
        let plan = self
            .plan
            .upgrade()
            .ok_or_else(|| Error::State("cold2B exact source plan lost".into()))?;
        let service = self
            .service
            .upgrade()
            .ok_or_else(|| Error::State("cold2B service lost".into()))?;
        let state = plan
            .cold_probe
            .as_ref()
            .ok_or_else(|| Error::State("cold2B read cut metadata absent".into()))?;
        let count = state.completed.load(Ordering::Acquire);
        let compared = state.bytes.load(Ordering::Acquire);
        let validator = state.validator.load(Ordering::Acquire);
        let expected = mode.cut().map(|(n, _)| n).unwrap_or(
            if mode == Qwen2bMlpWeightMode::ValidatorRefusal {
                73
            } else {
                0
            },
        );
        let ids = prepared
            .weights
            .accesses()
            .iter()
            .map(|(v, _)| v.id())
            .collect::<Vec<_>>();
        let reads = prepared
            .reads
            .iter()
            .map(|v| (v.lease_id(), v.offset(), v.bytes()))
            .collect::<Vec<_>>();
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        let physical = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
        if count != expected
            || validator != (mode == Qwen2bMlpWeightMode::ValidatorRefusal)
            || ids != self.ids
            || ids.len() != 2
            || ids[0] == ids[1]
            || reads != self.reads
            || reads.len() != 2
            || !Arc::ptr_eq(&source, prepared.owner.cohort.source.source())
            || !Arc::ptr_eq(&plan.weights, &prepared.weights)
            || !Arc::ptr_eq(&service, &prepared.weights.service)
            || !prepared
                .owner
                .record_admission
                .source()
                .same_issuer(source.source())
            || prepared.owner.record_admission.bytes() != 32 * MIB
            || prepared
                .weights
                .accesses()
                .iter()
                .any(|(v, a)| !v.is_initialized() || *a != DeviceAccess::Read)
            || service.resident_bytes()? != BYTES
            || service.uploaded_value_bytes() != BYTES as u64
            || service.pending_events()? != 1
            || handles != 2
            || owners < 1
            || !physical.fenced
            || physical.reserved != BYTES as u64
        {
            return Err(Error::State(
                "cold2B whole actual source/record/73upload/W/plan quarantine graph mismatch"
                    .into(),
            ));
        }
        if compared != compared_at(expected)? {
            return Err(Error::State(
                "cold2B exact source-byte prefix differs".into(),
            ));
        }
        Ok(outcome(
            mode,
            self.rows,
            "DELIBERATE_READ_FLIGHT_REFUSAL_RETAINED",
            Snapshot {
                count,
                compared,
                leading: if mode == Qwen2bMlpWeightMode::EmptyComputeOutputs {
                    0
                } else {
                    LEADING
                },
                validator,
                resident: BYTES,
                pending: 1,
                handles,
                owners,
                retained: true,
                reserved: physical.reserved,
                fenced: true,
                ..Snapshot::default()
            },
        ))
    }
}
#[derive(Default)]
struct Snapshot {
    count: usize,
    compared: usize,
    leading: usize,
    validator: bool,
    retired: bool,
    resident: usize,
    pending: usize,
    handles: usize,
    owners: usize,
    retained: bool,
    reserved: u64,
    fenced: bool,
}
fn compared_at(count: usize) -> Result<usize> {
    if count > 73 {
        return Err(Error::State("finite cold2B read prefix".into()));
    }
    Ok(if count <= 48 {
        count * MIB
    } else if count < 73 {
        48 * MIB + (count - 48) * 85 * 12288
    } else {
        BYTES
    })
}
fn outcome(
    mode: Qwen2bMlpWeightMode,
    rows: usize,
    status: &'static str,
    s: Snapshot,
) -> Qwen2bMlpWeightOutcome {
    let allocated = !matches!(
        mode,
        Qwen2bMlpWeightMode::Pressure | Qwen2bMlpWeightMode::LegacyUploadRefusal
    );
    Qwen2bMlpWeightOutcome {
        mode,
        status,
        rows,
        copies_enqueued: if allocated { 73 } else { 0 },
        copy_checkpoints: if allocated { 73 } else { 0 },
        initialized_weights: allocated,
        serial_reads_completed: s.count,
        source_bytes_compared: s.compared,
        leading_readback_bytes: s.leading,
        trusted_validator_entered: s.validator,
        read_only_flight_retired: s.retired,
        diagnostic_ready_outputs: 0,
        native_launches: 0,
        prepared_buffers: if allocated { 2 } else { 0 },
        submitted_buffers: if allocated { 2 } else { 0 },
        resident_bytes: s.resident,
        uploaded_value_bytes: if allocated { BYTES as u64 } else { 0 },
        pending_events: s.pending,
        quarantined_handles: s.handles,
        quarantined_owners: s.owners,
        source_record_weights_plan_retained: s.retained,
        record_admission_retained: s.retained,
        physical_reserved_bytes: s.reserved,
        physical_fenced: s.fenced,
        root_published: false,
        native_mlp_qualified: false,
    }
}
/// Existing source/record/ticket may preexist. True flag precedes ALL new qualifier work.
/// Helper reserves SAME source32MiB before recording, not a detached surrogate admission.
pub async fn qualify_qwen2b_mlp_weights(
    device: Device,
    source: BoundFullModelSource,
    record: Arc<RecordedVllm08Model>,
    record_admission: WindowWorkspace,
    mode: Qwen2bMlpWeightMode,
    device_cap: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<Qwen2bMlpWeightOutcome> {
    let inputs = Inputs {
        record,
        source,
        record_admission,
    };
    not_cancelled(cancelled)?;
    let mut probe = None;
    let result = run(device, inputs, (mode, device_cap, cancelled), &mut probe).await;
    if !mode.is_fault() {
        return result;
    }
    let intended = match &result {
        Err(Error::State(why)) if mode == Qwen2bMlpWeightMode::EmptyComputeOutputs => {
            why == "resident Ready requires outputs"
        }
        Err(Error::State(why)) if mode == Qwen2bMlpWeightMode::ValidatorRefusal => why == REFUSED,
        Err(Error::State(why)) if mode.cut().is_some_and(|(_, cancel)| cancel) => {
            why == "original full-model learned upload cancelled"
        }
        Err(Error::State(why)) => why == UNKNOWN,
        _ => false,
    };
    if !intended {
        return match result {
            Err(error) => Err(error),
            Ok(_) => Err(Error::State("cold2B deliberate cut absent".into())),
        };
    }
    probe
        .ok_or_else(|| Error::State("cold2B Weak-only submitted probe absent".into()))?
        .inspect(mode)
}
async fn run(
    device: Device,
    inputs: Inputs,
    controls: (Qwen2bMlpWeightMode, u64, Option<&AtomicBool>),
    probe: &mut Option<Probe>,
) -> Result<Qwen2bMlpWeightOutcome> {
    let (mode, device_cap, cancelled) = controls;
    not_cancelled(cancelled)?;
    if inputs.record_admission.bytes() != 32 * MIB
        || !inputs
            .record_admission
            .source()
            .same_issuer(inputs.source.source().source())
    {
        return Err(Error::State(
            "same source32MiB record allocation admission required".into(),
        ));
    }
    let record_admission = inputs.record_admission;
    let cohort = Cohort::bind(
        inputs.source,
        inputs.record,
        Qwen2bGdnArithmetic::OwnedSolve4Recompute3,
    )?;
    let rows = cohort.record().call().rows();
    if mode == Qwen2bMlpWeightMode::LegacyUploadRefusal {
        let result = super::super::require_original08_upload_source(
            cohort.source.source(),
            cohort.operator(8)?,
        );
        if !matches!(result,Err(Error::Unsupported(ref why)) if why=="closed original .8B24 source/config/rotary geometry")
        {
            return Err(Error::State(
                "legacy .8B upload did not reject exact actual2B source geometry".into(),
            ));
        }
        return Ok(outcome(
            mode,
            rows,
            "LEGACY_08_UPLOAD_REJECTED_ACTUAL_2B_SOURCE",
            Snapshot::default(),
        ));
    }
    if cohort.arithmetic() != Qwen2bGdnArithmetic::OwnedSolve4Recompute3 {
        return Err(Error::State("cold2B finite arithmetic source seal".into()));
    }
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "cold2B source W diagnostic requires release CUDA".into(),
        ));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        cohort.source.source().source_identity(),
        ResourceOwner::new("original2B-MLP-source", "finite-cold-weights-only")?,
        BYTES,
        device_cap,
    )?;
    let key = service.physical_key();
    if service.client().properties().memory.alignment != 512 {
        return Err(Error::Unsupported(
            "cold2B exact512B allocation alignment".into(),
        ));
    }
    if mode == Qwen2bMlpWeightMode::Pressure {
        let refusal = service.allocate_uninitialized(BYTES + 1, BufferRole::ImmutableWeights);
        if !matches!(refusal,Err(Error::Limit(ref why)) if why=="resident plus opaque provider budget exceeded")
            || service.resident_bytes()? != 0
            || service.pending_events()? != 0
            || service.uploaded_value_bytes() != 0
            || crate::physical_quarantine::quarantined_handles_for(service.client())? != 0
            || crate::physical_quarantine::quarantined_completion_owners_for(service.client())? != 0
            || service.ensure_usable().is_err()
        {
            return Err(Error::State(
                "cold2B cap+1 must refuse before W/source windows/H2D".into(),
            ));
        }
        drop((refusal, cohort, record_admission, service));
        let physical = ResourceArbiter::process_shared().snapshot(&key)?;
        if physical.reserved != 0 || physical.fenced {
            return Err(Error::State("cold2B pressure final physical0".into()));
        }
        return Ok(outcome(
            mode,
            rows,
            "CAP_PLUS_ONE_REFUSED_BEFORE_ALLOCATION",
            Snapshot::default(),
        ));
    }
    let host = cohort.source.source().source().reserve_workspace(MIB)?;
    let owner = Arc::new(ColdOwners {
        cohort: cohort.clone(),
        record_admission,
        _metadata: host,
    });
    let requests = [AllocationRequest {
        kind: AllocationClass::ImmutableWeights,
        bytes: BYTES as u64,
    }];
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let weights = Weights::upload(cohort, service.clone(), owner.clone(), cancelled).await?;
    slot.finish()?;
    weights.require_record(owner.cohort.record(), &service)?;
    if weights.immutable_bytes() != BYTES {
        return Err(Error::State(
            "both complete2B W groups actual byte extent".into(),
        ));
    }
    let reads = vec![
        weights.lease(8)?.readback(0, 4096)?,
        weights.lease(10)?.readback(0, 12288)?,
    ];
    let prepared = Arc::new(Prepared {
        owner,
        weights,
        reads,
    });
    let mut plan = prepared.weights.verification_plan(prepared.clone())?;
    Arc::get_mut(&mut plan)
        .ok_or_else(|| Error::State("new unique cold source verification plan".into()))?
        .cold_probe = Some(Arc::new(ReadProbe::new(mode)));
    if mode.is_fault() {
        *probe = Some(Probe::new(&prepared, &plan));
    }
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut flight = service.submit(1, &prepared.weights.accesses())?;
    plan.retain_before_enqueue(&mut flight)?;
    let state = plan
        .cold_probe
        .as_ref()
        .ok_or_else(|| Error::State("cold2B plan metadata absent".into()))?;
    if mode == Qwen2bMlpWeightMode::EmptyComputeOutputs {
        let _ = plan
            .complete_compute(flight, &[], &prepared.reads, cancelled, |_| {
                Err(Error::State(
                    "empty compute unexpectedly reached validator".into(),
                ))
            })
            .await?;
        return Err(Error::State("empty compute unexpectedly completed".into()));
    }
    let raw = plan
        .complete_retirement(flight, &prepared.reads, cancelled, |raw| {
            state.validator.store(true, Ordering::Release);
            if mode == Qwen2bMlpWeightMode::ValidatorRefusal {
                return Err(Error::State(REFUSED.into()));
            }
            prepared.compare_leading(raw, cancelled)
        })
        .await?;
    slot.finish()?;
    let count = state.completed.load(Ordering::Acquire);
    let compared = state.bytes.load(Ordering::Acquire);
    if count != 73 || compared != BYTES || raw.iter().map(|v| v.len()).sum::<usize>() != LEADING {
        return Err(Error::State(
            "actual all73 cold W source reads/fullbytes/leading cohort required".into(),
        ));
    }
    drop((raw, plan, prepared));
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != BYTES as u64
    {
        return Err(Error::State(
            "cold2B successful actual W lease/event retirement0".into(),
        ));
    }
    drop(service);
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.reserved != 0 || physical.fenced {
        return Err(Error::State("cold2B successful physical0/unfenced".into()));
    }
    Ok(outcome(
        mode,
        rows,
        "ACTUAL_FULL_SOURCE_WEIGHTS_READ_ONLY_VERIFIED",
        Snapshot {
            count,
            compared,
            leading: LEADING,
            validator: true,
            retired: true,
            ..Snapshot::default()
        },
    ))
}
#[cfg(test)]
mod tests;
