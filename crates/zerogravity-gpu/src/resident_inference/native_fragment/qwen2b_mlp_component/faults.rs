//! Deliberate original-flight cuts; completed comparisons are not GPU failures.
use super::*;
use crate::resident_leases::Qwen2bMlpComputeReadCut as ReadCut;
use std::sync::{Mutex as StdMutex, Weak};
use zerogravity_core::device_events::BufferLeaseId;
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen2bMlpComponentFault {
    AdmissionPressure,
    UnknownAfterGateUp,
    CancelBeforeActivation,
    UnknownAfterFirstSourceRead,
    UnknownAfterMiddleSourceRead,
    UnknownAfterLastSourceRead,
    CancelAfterFirstSourceRead,
    CancelAfterMiddleSourceRead,
    CancelAfterLastSourceRead,
    ValidatorRefusal,
    UnknownAfterDownMain,
    CancelBeforeDownReducer,
}
impl Qwen2bMlpComponentFault {
    pub(super) fn read_cut(self) -> Option<ReadCut> {
        match self {
            Self::UnknownAfterFirstSourceRead => Some(ReadCut::UnknownFirst),
            Self::UnknownAfterMiddleSourceRead => Some(ReadCut::UnknownMiddle),
            Self::UnknownAfterLastSourceRead => Some(ReadCut::UnknownLast),
            Self::CancelAfterFirstSourceRead => Some(ReadCut::CancelFirst),
            Self::CancelAfterMiddleSourceRead => Some(ReadCut::CancelMiddle),
            Self::CancelAfterLastSourceRead => Some(ReadCut::CancelLast),
            _ => None,
        }
    }
    fn source_reads(self) -> usize {
        match self {
            Self::UnknownAfterFirstSourceRead | Self::CancelAfterFirstSourceRead => 1,
            Self::UnknownAfterMiddleSourceRead | Self::CancelAfterMiddleSourceRead => 37,
            Self::UnknownAfterLastSourceRead
            | Self::CancelAfterLastSourceRead
            | Self::ValidatorRefusal => 73,
            _ => 0,
        }
    }
    fn is_source_cancel(self) -> bool {
        matches!(
            self,
            Self::CancelAfterFirstSourceRead
                | Self::CancelAfterMiddleSourceRead
                | Self::CancelAfterLastSourceRead
        )
    }
}
#[derive(Clone, Debug)]
pub struct Qwen2bMlpComponentFaultOutcome {
    pub arithmetic: Qwen2bGdnArithmetic,
    pub fault: Qwen2bMlpComponentFault,
    pub status: &'static str,
    pub rows: usize,
    pub native_launches: usize,
    pub submitted_buffers: usize,
    pub retained_graph_buffers: usize,
    pub readback_descriptors: usize,
    pub main_readback_completed: bool,
    pub completed_source_reads: usize,
    pub source_bytes_compared: usize,
    pub trusted_validator_entered: bool,
    pub expected_callback_entered: bool,
    pub compute_ready_completions: usize,
    pub resident_bytes: usize,
    pub uploaded_value_bytes: u64,
    pub pending_events: usize,
    pub quarantined_handles: usize,
    pub quarantined_owners: usize,
    pub source_record_ticket_body_plan_native_retained: bool,
    pub out_dual_mlp_graph_retained: bool,
    pub down1_output_unpromoted: bool,
    pub down1_enqueues: usize,
    pub original_root: Option<RootBinding>,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub parent_model_accepted: bool,
    pub root_published: bool,
}
struct Witness {
    source: Weak<OriginalWindowModelSource>,
    service: Weak<ResidentService>,
    body: Weak<ComponentProgram>,
    cohort: Weak<Vec<ResidentReadback>>,
    plan: Weak<Qwen2bMlpSourceVerification>,
    descriptors: Vec<(BufferLeaseId, usize, usize)>,
    submitted: Vec<BufferLeaseId>,
    base: RootBinding,
    key: PhysicalDeviceKey,
}
#[derive(Default)]
pub(super) struct Probe(StdMutex<Option<Witness>>);
impl Probe {
    pub(super) fn arm(
        &self,
        service: &Arc<ResidentService>,
        body: &Arc<ComponentProgram>,
        cohort: &Arc<Vec<ResidentReadback>>,
        plan: &Arc<Qwen2bMlpSourceVerification>,
        accesses: &[(ResidentLease, DeviceAccess)],
        base: RootBinding,
    ) -> Result<()> {
        let mut slot = self
            .0
            .lock()
            .map_err(|_| Error::State("new2B MLP probe lock".into()))?;
        if slot.is_some() {
            return Err(Error::State("new2B MLP probe armed twice".into()));
        }
        *slot = Some(Witness {
            source: Arc::downgrade(body.recipe().source().source()),
            service: Arc::downgrade(service),
            body: Arc::downgrade(body),
            cohort: Arc::downgrade(cohort),
            plan: Arc::downgrade(plan),
            descriptors: cohort
                .iter()
                .map(|r| (r.lease_id(), r.offset(), r.bytes()))
                .collect(),
            submitted: accesses.iter().map(|(l, _)| l.id()).collect(),
            base,
            key: service.physical_key(),
        });
        Ok(())
    }
    fn inspect(
        &self,
        fault: Qwen2bMlpComponentFault,
        arithmetic: Qwen2bGdnArithmetic,
    ) -> Result<Qwen2bMlpComponentFaultOutcome> {
        let lock = self
            .0
            .lock()
            .map_err(|_| Error::State("new2B MLP probe lock".into()))?;
        let w = lock
            .as_ref()
            .ok_or_else(|| Error::State("no actual prepared2B MLP flight".into()))?;
        let body = w
            .body
            .upgrade()
            .ok_or_else(|| Error::State("2B whole computed graph lost".into()))?;
        let service = w
            .service
            .upgrade()
            .ok_or_else(|| Error::State("2B original service lost".into()))?;
        let source = w
            .source
            .upgrade()
            .ok_or_else(|| Error::State("2B actual source owner lost".into()))?;
        let cohort = w
            .cohort
            .upgrade()
            .ok_or_else(|| Error::State("2B exact readback graph lost".into()))?;
        let plan = w
            .plan
            .upgrade()
            .ok_or_else(|| Error::State("2B same installed source plan lost".into()))?;
        let descriptors = cohort
            .iter()
            .map(|r| (r.lease_id(), r.offset(), r.bytes()))
            .collect::<Vec<_>>();
        let retained = body
            .retained_accesses()?
            .iter()
            .map(|(l, _)| l.id())
            .chain(descriptors.iter().map(|(id, _, _)| *id))
            .collect::<BTreeSet<_>>();
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        let physical = ResourceArbiter::process_shared().snapshot(&w.key)?;
        let p = plan.compute_progress()?;
        let expected = fault.source_reads();
        let expected_bytes = if expected <= 48 {
            expected * 1024 * 1024
        } else {
            75_497_472
        };
        let native_cut = matches!(
            fault,
            Qwen2bMlpComponentFault::UnknownAfterGateUp
                | Qwen2bMlpComponentFault::CancelBeforeActivation
                | Qwen2bMlpComponentFault::UnknownAfterDownMain
                | Qwen2bMlpComponentFault::CancelBeforeDownReducer
        );
        let down_cut = matches!(
            fault,
            Qwen2bMlpComponentFault::UnknownAfterDownMain
                | Qwen2bMlpComponentFault::CancelBeforeDownReducer
        );
        if down_cut
            && (body.recipe().record().call().rows() != 1
                || body.qwen2b_native_launches()? != 6
                || body.qwen2b_down1_enqueues()? != 1
                || !body.qwen2b_down1_output_unpromoted()?)
        {
            return Err(Error::State(
                "2B exact Down main cut/Y future output".into(),
            ));
        }
        if descriptors != w.descriptors
            || w.submitted.iter().any(|id| !retained.contains(id))
            || handles != w.submitted.len()
            || owners < 3
            || service.pending_events()? != 1
            || !physical.fenced
            || physical.reserved != PROVIDER_BYTES as u64
            || service.ensure_usable().is_ok()
            || !Arc::ptr_eq(&source, body.recipe().source().source())
            || !body.qwen2b_mlp_plan_retained(&plan)?
            || !body.post_attention_graph_retained()?
            || !body.qwen2b_mlp_graph_retained()?
            || !body.outputs_unpromoted()?
            || p.completed_reads != expected
            || p.compared_source_bytes != expected_bytes
            || p.trusted_validator_entered != (fault == Qwen2bMlpComponentFault::ValidatorRefusal)
            || (native_cut && !down_cut && !body.cut_reached())
            || (!native_cut && body.finish_nodes().is_err())
        {
            return Err(Error::State(
                "2B actual compute graph/ticket/source/plan quarantine mismatch".into(),
            ));
        }
        Ok(Qwen2bMlpComponentFaultOutcome {
            arithmetic,
            fault,
            status: "DELIBERATE_COMPUTE_FLIGHT_REFUSAL_RETAINED",
            rows: body.recipe().record().call().rows(),
            native_launches: body.qwen2b_native_launches()?,
            submitted_buffers: w.submitted.len(),
            retained_graph_buffers: retained.len(),
            readback_descriptors: descriptors.len(),
            main_readback_completed: expected > 0,
            completed_source_reads: p.completed_reads,
            source_bytes_compared: p.compared_source_bytes,
            trusted_validator_entered: p.trusted_validator_entered,
            expected_callback_entered: false,
            compute_ready_completions: 0,
            resident_bytes: service.resident_bytes()?,
            uploaded_value_bytes: service.uploaded_value_bytes(),
            pending_events: 1,
            quarantined_handles: handles,
            quarantined_owners: owners,
            source_record_ticket_body_plan_native_retained: true,
            out_dual_mlp_graph_retained: true,
            down1_output_unpromoted: if body.recipe().record().call().rows() == 1 {
                body.qwen2b_down1_output_unpromoted()?
            } else {
                false
            },
            down1_enqueues: if body.recipe().record().call().rows() == 1 {
                body.qwen2b_down1_enqueues()?
            } else {
                0
            },
            original_root: Some(w.base),
            physical_reserved_bytes: physical.reserved,
            physical_fenced: true,
            parent_model_accepted: false,
            root_published: false,
        })
    }
}
pub(super) fn pressure(
    service: Arc<ResidentService>,
    arithmetic: Qwen2bGdnArithmetic,
) -> Result<Qwen2bMlpComponentFaultOutcome> {
    let key = service.physical_key();
    if !matches!(service.allocate_uninitialized(PROVIDER_BYTES+1,BufferRole::Scratch),
        Err(Error::Limit(ref s)) if s=="resident plus opaque provider budget exceeded")
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
    {
        return Err(Error::State(
            "2B MLP actual cap+1 before allocation/H2D".into(),
        ));
    }
    drop(service);
    let p = ResourceArbiter::process_shared().snapshot(&key)?;
    if p.fenced || p.reserved != 0 {
        return Err(Error::State("2B pressure finalphysical0".into()));
    }
    Ok(Qwen2bMlpComponentFaultOutcome {
        arithmetic,
        fault: Qwen2bMlpComponentFault::AdmissionPressure,
        status: "PRESSURE_REFUSED_BEFORE_ALLOCATION",
        rows: 0,
        native_launches: 0,
        submitted_buffers: 0,
        retained_graph_buffers: 0,
        readback_descriptors: 0,
        main_readback_completed: false,
        completed_source_reads: 0,
        source_bytes_compared: 0,
        trusted_validator_entered: false,
        expected_callback_entered: false,
        compute_ready_completions: 0,
        resident_bytes: 0,
        uploaded_value_bytes: 0,
        pending_events: 0,
        quarantined_handles: 0,
        quarantined_owners: 0,
        source_record_ticket_body_plan_native_retained: false,
        out_dual_mlp_graph_retained: false,
        down1_output_unpromoted: false,
        down1_enqueues: 0,
        original_root: None,
        physical_reserved_bytes: 0,
        physical_fenced: false,
        parent_model_accepted: false,
        root_published: false,
    })
}
fn intended(error: &Error, fault: Qwen2bMlpComponentFault) -> bool {
    let Error::State(message) = error else {
        return false;
    };
    match fault {
        Qwen2bMlpComponentFault::UnknownAfterGateUp => {
            message == "deliberate qwen2b MLP unknown after actual GateUp"
        }
        Qwen2bMlpComponentFault::CancelBeforeActivation => message == "native plan cancelled",
        Qwen2bMlpComponentFault::UnknownAfterDownMain => {
            message == "deliberate qwen2b MLP unknown after actual Down main"
        }
        Qwen2bMlpComponentFault::CancelBeforeDownReducer => {
            message == "closed original cubin qualification cancelled"
        }
        Qwen2bMlpComponentFault::ValidatorRefusal => {
            message == "deliberate qwen2b MLP trusted validator refusal"
        }
        _ if fault.is_source_cancel() => message == "original full-model learned upload cancelled",
        _ if fault.read_cut().is_some() => {
            message == "deliberate qwen2b compute unknown after completed source read"
        }
        _ => false,
    }
}
pub async fn qualify_qwen2b_mlp_component_fault(
    device: Device,
    source: Qwen2bMlpComponentSource,
    arithmetic: Qwen2bGdnArithmetic,
    inputs: Qwen2bMlpQualificationInputs,
    fault: Qwen2bMlpComponentFault,
    cancelled: Option<&AtomicBool>,
) -> Result<Qwen2bMlpComponentFaultOutcome> {
    let probe = Probe::default();
    let result = run(
        device,
        source,
        inputs,
        cancelled,
        |_: MlpComponentReadback<'_>| {
            Err(Error::State(
                "2B MLP fault must never call expected reader".into(),
            ))
        },
        RunMode {
            arithmetic,
            fault: Some(fault),
            probe: Some(&probe),
        },
    )
    .await;
    // Inner future/caller graph gone. Probe contains ONLY Weak and copied scalar metadata.
    match result {
        Ok(Outcome::Pressure(out)) => Ok(out),
        Ok(Outcome::Compared(_)) => Err(Error::State("2B MLP deliberate cut absent".into())),
        Err(error) if intended(&error, fault) => probe.inspect(fault, arithmetic),
        Err(error) => Err(error),
    }
}
