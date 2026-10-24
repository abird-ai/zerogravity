//! Actual cap+1 and two original node-loop uncertainty cuts; Weak-only witness.
use super::*;
use std::sync::{Mutex as StdMutex, Weak};
use zerogravity_core::device_events::BufferLeaseId;
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpComponentFault {
    AdmissionPressure,
    UnknownAfterGateUp,
    CancelBeforeActivation,
}
#[derive(Clone, Debug)]
pub struct MlpComponentFaultOutcome {
    pub arithmetic: PostAttentionArithmetic,
    pub parent_model_accepted: bool,
    pub fault: MlpComponentFault,
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
    pub out_and_dual_graph_retained: bool,
    pub mlp_graph_retained: bool,
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
            .map_err(|_| Error::State("MLP fault witness lock".into()))?;
        if guard.is_some() {
            return Err(Error::State("MLP fault witness armed twice".into()));
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
        fault: MlpComponentFault,
        arithmetic: PostAttentionArithmetic,
    ) -> Result<MlpComponentFaultOutcome> {
        let guard = self
            .0
            .lock()
            .map_err(|_| Error::State("MLP fault witness lock".into()))?;
        let w = guard
            .as_ref()
            .ok_or_else(|| Error::State("MLP failure before real component flight".into()))?;
        let service = w
            .service
            .upgrade()
            .ok_or_else(|| Error::State("MLP uncertain service released".into()))?;
        let body = w
            .body
            .upgrade()
            .ok_or_else(|| Error::State("MLP uncertain body released".into()))?;
        let source = w
            .source
            .upgrade()
            .ok_or_else(|| Error::State("MLP uncertain source released".into()))?;
        let cohort = w
            .cohort
            .upgrade()
            .ok_or_else(|| Error::State("MLP actual D2H descriptors released".into()))?;
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
            || !body.post_attention_graph_retained()?
            || !body.mlp_graph_retained()?
        {
            return Err(Error::State(
                "MLP exact submitted/retained graph or original uncertainty envelope differs"
                    .into(),
            ));
        }
        Ok(MlpComponentFaultOutcome {
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
            out_and_dual_graph_retained: true,
            mlp_graph_retained: true,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: true,
        })
    }
}
pub(super) fn pressure(
    service: Arc<ResidentService>,
    arithmetic: PostAttentionArithmetic,
) -> Result<MlpComponentFaultOutcome> {
    let key = service.physical_key();
    if !matches!(service.allocate_uninitialized(PROVIDER_BYTES+1,BufferRole::Scratch),
        Err(Error::Limit(ref s)) if s=="resident plus opaque provider budget exceeded")
        || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
    {
        return Err(Error::State(
            "MLP pressure was not refused before original provider handles/H2D".into(),
        ));
    }
    drop(service);
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State("MLP cold pressure envelope retained".into()));
    }
    Ok(MlpComponentFaultOutcome {
        arithmetic,
        parent_model_accepted: false,
        fault: MlpComponentFault::AdmissionPressure,
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
        out_and_dual_graph_retained: false,
        mlp_graph_retained: false,
        physical_reserved_bytes: 0,
        physical_fenced: false,
    })
}
fn intended_fault(error: &Error, fault: MlpComponentFault) -> bool {
    match (error, fault) {
        (Error::State(message), MlpComponentFault::UnknownAfterGateUp) => {
            message == "deliberate canonical47 MLP unknown after actual GateUp"
        }
        (Error::State(message), MlpComponentFault::CancelBeforeActivation) => {
            message == "native plan cancelled"
        }
        _ => false,
    }
}

pub async fn qualify_mlp_component_fault(
    device: Device,
    source: BoundFullModelSource,
    record: Arc<RecordedVllm08Model>,
    arithmetic: PostAttentionArithmetic,
    inputs: Mlp47QualificationInputs,
    fault: MlpComponentFault,
    cancelled: Option<&AtomicBool>,
) -> Result<MlpComponentFaultOutcome> {
    let probe = Probe::default();
    let result = run(
        device,
        source,
        record,
        inputs,
        cancelled,
        |_: MlpComponentReadback<'_>| {
            Err(Error::State("MLP uncertain path must never compare".into()))
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
        Ok(Outcome::Compared(_)) => Err(Error::State("MLP fault completed positive math".into())),
        Err(error) => {
            let reached = probe
                .0
                .lock()
                .map_err(|_| Error::State("MLP probe lock".into()))?
                .as_ref()
                .and_then(|w| w.body.upgrade())
                .is_some_and(|b| b.cut_reached());
            let intended = intended_fault(&error, fault);
            if reached && intended {
                probe.inspect(fault, arithmetic)
            } else {
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod cause_tests {
    use super::*;
    #[test]
    fn actual_cut_flag_does_not_accept_an_unrelated_error_cause() {
        let unknown = Error::State("deliberate canonical47 MLP unknown after actual GateUp".into());
        let cancel = Error::State("native plan cancelled".into());
        assert!(intended_fault(
            &unknown,
            MlpComponentFault::UnknownAfterGateUp
        ));
        assert!(intended_fault(
            &cancel,
            MlpComponentFault::CancelBeforeActivation
        ));
        assert!(!intended_fault(
            &unknown,
            MlpComponentFault::CancelBeforeActivation
        ));
        assert!(!intended_fault(
            &cancel,
            MlpComponentFault::UnknownAfterGateUp
        ));
        for error in [
            Error::State("foreign root changed".into()),
            Error::Limit("source admission".into()),
        ] {
            assert!(!intended_fault(
                &error,
                MlpComponentFault::UnknownAfterGateUp
            ));
            assert!(!intended_fault(
                &error,
                MlpComponentFault::CancelBeforeActivation
            ));
        }
    }
}
