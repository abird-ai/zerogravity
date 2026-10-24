//! Actual cap+1 and two original node-loop uncertainty cuts; Weak-only witness.
use super::*;
use std::sync::{Mutex as StdMutex, Weak};
use zerogravity_core::device_events::BufferLeaseId;
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canonical47GdnComponentFault {
    AdmissionPressure,
    UnknownAfterConvolution,
    CancelBeforeRecurrent,
}
#[derive(Clone, Debug)]
pub struct Canonical47GdnComponentFaultOutcome {
    pub arithmetic: Canonical47GdnArithmetic,
    pub parent_model_accepted: bool,
    pub fault: Canonical47GdnComponentFault,
    pub status: &'static str,
    pub native_launches: usize,
    pub submitted_buffers: usize,
    pub retained_graph_buffers: usize,
    pub readback_descriptors: usize,
    pub original_root: Option<RootBinding>,
    pub compared_words: usize,
    pub ready_completions: usize,
    pub resident_bytes: usize,
    pub uploaded_value_bytes: u64,
    pub pending_events: usize,
    pub quarantined_handles: usize,
    pub quarantined_owners: usize,
    pub source_and_body_retained: bool,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
}
struct Witness {
    service: Weak<ResidentService>,
    body: Weak<ComponentProgram>,
    source: Weak<OriginalWindowModelSource>,
    cohort: Weak<Vec<ResidentReadback>>,
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
        accesses: &[(ResidentLease, DeviceAccess)],
        base: RootBinding,
    ) -> Result<()> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| Error::State("canonical47 fault witness lock".into()))?;
        if guard.is_some() {
            return Err(Error::State("canonical47 fault witness armed twice".into()));
        }
        *guard = Some(Witness {
            service: Arc::downgrade(service),
            body: Arc::downgrade(body),
            source: Arc::downgrade(body.recipe().source().source()),
            cohort: Arc::downgrade(cohort),
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
        fault: Canonical47GdnComponentFault,
        arithmetic: Canonical47GdnArithmetic,
    ) -> Result<Canonical47GdnComponentFaultOutcome> {
        let guard = self
            .0
            .lock()
            .map_err(|_| Error::State("canonical47 fault witness lock".into()))?;
        let w = guard.as_ref().ok_or_else(|| {
            Error::State("canonical47 failure before real component flight".into())
        })?;
        let service = w
            .service
            .upgrade()
            .ok_or_else(|| Error::State("canonical47 uncertain service released".into()))?;
        let body = w
            .body
            .upgrade()
            .ok_or_else(|| Error::State("canonical47 uncertain body released".into()))?;
        let source = w
            .source
            .upgrade()
            .ok_or_else(|| Error::State("canonical47 uncertain source released".into()))?;
        let cohort = w
            .cohort
            .upgrade()
            .ok_or_else(|| Error::State("canonical47 actual D2H descriptors released".into()))?;
        let descriptors = cohort
            .iter()
            .map(|r| (r.lease_id(), r.offset(), r.bytes()))
            .collect::<Vec<_>>();
        let accesses = body.retained_accesses()?;
        let retained = accesses
            .iter()
            .map(|(l, _)| l.id())
            .chain(descriptors.iter().map(|(id, _, _)| *id))
            .collect::<BTreeSet<_>>();
        let registry = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        let physical = ResourceArbiter::process_shared().snapshot(&w.key)?;
        if !body.cut_reached()
            || descriptors != w.descriptors
            || !Arc::ptr_eq(&source, body.recipe().source().source())
            || w.submitted.iter().any(|id| !retained.contains(id))
            || registry != w.submitted.len()
            || owners < 2
            || service.pending_events()? != 1
            || !physical.fenced
            || physical.reserved != PROVIDER_BYTES as u64
            || service.ensure_usable().is_ok()
            || !body.outputs_unpromoted()?
        {
            return Err(Error::State(
                "canonical47 exact submitted/retained graph or original uncertainty envelope differs"
                    .into(),
            ));
        }
        Ok(Canonical47GdnComponentFaultOutcome {
            arithmetic,
            parent_model_accepted: false,
            fault,
            status: "UNKNOWN_RETAINED",
            native_launches: body.native_launches(),
            submitted_buffers: w.submitted.len(),
            retained_graph_buffers: retained.len(),
            readback_descriptors: descriptors.len(),
            original_root: Some(w.base),
            compared_words: 0,
            ready_completions: 0,
            resident_bytes: service.resident_bytes()?,
            uploaded_value_bytes: service.uploaded_value_bytes(),
            pending_events: 1,
            quarantined_handles: registry,
            quarantined_owners: owners,
            source_and_body_retained: true,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: true,
        })
    }
}
pub(super) fn pressure(
    service: Arc<ResidentService>,
    arithmetic: Canonical47GdnArithmetic,
) -> Result<Canonical47GdnComponentFaultOutcome> {
    let key = service.physical_key();
    if !matches!(service.allocate_uninitialized(PROVIDER_BYTES+1,BufferRole::Scratch),
        Err(Error::Limit(ref s)) if s=="resident plus opaque provider budget exceeded")
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
    {
        return Err(Error::State(
            "canonical47 pressure was not refused before original provider handles/H2D".into(),
        ));
    }
    drop(service);
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "canonical47 cold pressure envelope retained".into(),
        ));
    }
    Ok(Canonical47GdnComponentFaultOutcome {
        arithmetic,
        parent_model_accepted: false,
        fault: Canonical47GdnComponentFault::AdmissionPressure,
        status: "PRESSURE_REFUSED_BEFORE_ALLOCATION",
        native_launches: 0,
        submitted_buffers: 0,
        retained_graph_buffers: 0,
        readback_descriptors: 0,
        original_root: None,
        compared_words: 0,
        ready_completions: 0,
        resident_bytes: 0,
        uploaded_value_bytes: 0,
        pending_events: 0,
        quarantined_handles: 0,
        quarantined_owners: 0,
        source_and_body_retained: false,
        physical_reserved_bytes: 0,
        physical_fenced: false,
    })
}
pub async fn qualify_canonical47_gdn_component_fault(
    device: Device,
    source: BoundFullModelSource,
    record: Arc<RecordedVllm08Model>,
    arithmetic: Canonical47GdnArithmetic,
    inputs: CubinQualificationInputs,
    fault: Canonical47GdnComponentFault,
    cancelled: Option<&AtomicBool>,
) -> Result<Canonical47GdnComponentFaultOutcome> {
    let probe = Probe::default();
    let result = run(
        device,
        source,
        record,
        inputs,
        cancelled,
        |_: Canonical47GdnComponentReadback<'_>| {
            Err(Error::State(
                "canonical47 uncertain path must never compare".into(),
            ))
        },
        RunMode {
            arithmetic,
            fault: Some(fault),
            probe: Some(&probe),
        },
    )
    .await;
    // Future/inner locals have dropped. Probe owns no strong graph/source keepalive.
    match result {
        Ok(Outcome::Pressure(out)) => Ok(out),
        Ok(Outcome::Compared(_)) => Err(Error::State(
            "canonical47 fault completed positive math".into(),
        )),
        Err(error) => {
            let reached = probe
                .0
                .lock()
                .map_err(|_| Error::State("canonical47 probe lock".into()))?
                .as_ref()
                .and_then(|w| w.body.upgrade())
                .is_some_and(|b| b.cut_reached());
            let intended = match (&error, fault) {
                (Error::State(message), Canonical47GdnComponentFault::UnknownAfterConvolution) => {
                    message == "deliberate canonical47 unknown after actual Conv"
                }
                (Error::State(message), Canonical47GdnComponentFault::CancelBeforeRecurrent) => {
                    message == "closed vLLM GDN phase cancelled"
                }
                _ => false,
            };
            if reached && intended {
                probe.inspect(fault, arithmetic)
            } else {
                Err(error)
            }
        }
    }
}
