//! SAME initial fragment, actual original admission/cancellation/quarantine cuts.
use super::*;
use crate::resident_leases::ResidentReadback;
use crate::resident_program::native::NativeProgram;
use std::sync::{Mutex as StdMutex, Weak};
use zerogravity_core::device_events::BufferLeaseId;
use zerogravity_core::resource_arbiter::PhysicalDeviceKey;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnFragmentFault {
    AdmissionPressure,
    UnknownAfterConvolution,
    CancelBeforeRecurrent,
    UnknownAfterPostAttentionRms,
    CancelBeforeMlpDown,
}
#[derive(Clone, Debug)]
pub struct GdnFragmentFaultOutcome {
    pub fault: GdnFragmentFault,
    pub status: &'static str,
    pub ready_completions: usize,
    pub compared_words: usize,
    pub rms_stages: usize,
    pub native_launches: usize,
    pub convolution_dispatches: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub original_root: Option<RootBinding>,
    pub original_root_domains: usize,
    pub original_state_buffers: usize,
    pub retained_graph_leases: usize,
    pub readback_descriptors: usize,
    pub source_backing_retained: bool,
    pub body_retained: bool,
    pub all_accesses_retained: bool,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
}
pub(super) struct Witness {
    service: Weak<ResidentService>,
    source: SourceOwnerWitness,
    body: Weak<NativeProgram>,
    readbacks: Weak<Vec<ResidentReadback>>,
    descriptors: Vec<(BufferLeaseId, usize, usize)>,
    accesses: Vec<BufferLeaseId>,
    root_buffers: usize,
    root: RootBinding,
    key: PhysicalDeviceKey,
    cap: usize,
}
pub(super) struct ProbeContext<'a> {
    pub service: &'a Arc<ResidentService>,
    pub source: &'a VerifiedTensorSource,
    pub body: &'a Arc<NativeProgram>,
    pub readbacks: &'a Arc<Vec<ResidentReadback>>,
    pub accesses: &'a [(ResidentLease, DeviceAccess)],
    pub base: RootBinding,
    pub root_buffers: usize,
}
#[derive(Default)]
pub(super) struct Probe(StdMutex<Option<Witness>>);
impl Probe {
    pub(super) fn arm(&self, context: ProbeContext<'_>) -> Result<()> {
        let ProbeContext {
            service,
            source,
            body,
            readbacks,
            accesses,
            base,
            root_buffers,
        } = context;
        let mut guard = self
            .0
            .lock()
            .map_err(|_| Error::State("native fragment witness poisoned".into()))?;
        if guard.is_some() {
            return Err(Error::State("native fragment witness armed twice".into()));
        }
        *guard = Some(Witness {
            service: Arc::downgrade(service),
            source: source.weak_owner(),
            body: Arc::downgrade(body),
            readbacks: Arc::downgrade(readbacks),
            descriptors: readbacks
                .iter()
                .map(|r| (r.lease_id(), r.offset(), r.bytes()))
                .collect(),
            accesses: accesses.iter().map(|(l, _)| l.id()).collect(),
            root_buffers,
            root: base,
            key: service.physical_key(),
            cap: service.capacity(),
        });
        Ok(())
    }
    pub(super) fn cut_observed(&self) -> Result<bool> {
        let guard = self
            .0
            .lock()
            .map_err(|_| Error::State("native fault probe poisoned".into()))?;
        Ok(guard
            .as_ref()
            .and_then(|w| w.body.upgrade())
            .is_some_and(|b| b.fault_cut_observed()))
    }
    pub(super) fn inspect(&self, fault: GdnFragmentFault) -> Result<GdnFragmentFaultOutcome> {
        let guard = self
            .0
            .lock()
            .map_err(|_| Error::State("native fragment witness poisoned".into()))?;
        let w = guard.as_ref().ok_or_else(|| {
            Error::State("fault failed before actual initial fragment enqueue".into())
        })?;
        if !w.source.retained() {
            return Err(Error::State(
                "uncertain actual source backing released".into(),
            ));
        }
        let service = w
            .service
            .upgrade()
            .ok_or_else(|| Error::State("uncertain original service released".into()))?;
        let body = w
            .body
            .upgrade()
            .ok_or_else(|| Error::State("uncertain actual graph released".into()))?;

        let readbacks = w
            .readbacks
            .upgrade()
            .ok_or_else(|| Error::State("uncertain actual readback cohort released".into()))?;
        let descriptors = readbacks
            .iter()
            .map(|r| (r.lease_id(), r.offset(), r.bytes()))
            .collect::<Vec<_>>();
        let graph = body.retained_accesses()?;
        let known = graph
            .iter()
            .map(|(l, _)| l.id())
            .chain(descriptors.iter().map(|(id, _, _)| *id))
            .collect::<BTreeSet<_>>();
        let all_accesses_retained = w.accesses.iter().all(|id| known.contains(id));
        let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&w.key)?;
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        if !body.fault_cut_observed()
            || !w.source.same_issuer(body.source())
            || descriptors != w.descriptors
            || !all_accesses_retained
            || w.root_buffers != 96
            || service.pending_events()? != 1
            || !physical.fenced
            || physical.reserved != w.cap as u64
            || handles < w.accesses.len()
            || owners < 2
            || service.ensure_usable().is_ok()
        {
            return Err(Error::State("actual native fragment uncertainty did not retain exact original graph/all42 roots/cohort/envelope".into()));
        }
        Ok(GdnFragmentFaultOutcome {
            fault,
            status: "UNKNOWN_RETAINED",
            ready_completions: 0,
            compared_words: 0,
            rms_stages: body.rms_stages(),
            native_launches: body.native_launches(),
            convolution_dispatches: body.convolution_dispatches(),
            resident_bytes: service.resident_bytes()?,
            pending_events: service.pending_events()?,
            uploaded_value_bytes: service.uploaded_value_bytes(),
            original_root: Some(w.root),
            original_root_domains: 42,
            original_state_buffers: w.root_buffers,
            retained_graph_leases: w.accesses.len(),
            readback_descriptors: descriptors.len(),
            source_backing_retained: true,
            body_retained: true,
            all_accesses_retained,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: physical.fenced,
        })
    }
}
pub(super) enum Mode {
    Compare,
    Fault {
        fault: GdnFragmentFault,
        probe: Arc<Probe>,
    },
}
impl Mode {
    pub(super) fn fault(&self) -> Option<GdnFragmentFault> {
        match self {
            Self::Compare => None,
            Self::Fault { fault, .. } => Some(*fault),
        }
    }
}
pub(super) enum Outcome {
    Compared(GdnFragmentReceipt),
    Fault(GdnFragmentFaultOutcome),
}
/// Fresh diagnostic process; UNKNOWN never synchronizes/unloads/recoveries.
pub async fn qualify_gdn_fragment_fault(
    device: Device,
    record: Arc<SourceBindings>,
    inputs: CubinQualificationInputs,
    fault: GdnFragmentFault,
    cancelled: Option<&AtomicBool>,
) -> Result<GdnFragmentFaultOutcome> {
    not_cancelled(cancelled)?;
    if record.selection().completed_layers() != 0
        || matches!(
            fault,
            GdnFragmentFault::UnknownAfterPostAttentionRms | GdnFragmentFault::CancelBeforeMlpDown
        )
    {
        return Err(Error::Unsupported(
            "expanded prefix faults need their own actual graph cohort acceptance".into(),
        ));
    }
    qualify_fault_inner(
        device,
        record,
        NativeFragmentImages::Projection(inputs),
        fault,
        cancelled,
    )
    .await
}
pub async fn qualify_gdn_mlp_fragment_fault(
    device: Device,
    record: Arc<SourceBindings>,
    inputs: LayerMlpCubinInputs,
    fault: GdnFragmentFault,
    cancelled: Option<&AtomicBool>,
) -> Result<GdnFragmentFaultOutcome> {
    not_cancelled(cancelled)?;
    if !matches!(record.selection().completed_layers(), 1 | 3) {
        return Err(Error::Unsupported(
            "complete initial layer0/prefix0..2 fault record".into(),
        ));
    }
    qualify_fault_inner(
        device,
        record,
        NativeFragmentImages::Connected(inputs),
        fault,
        cancelled,
    )
    .await
}
async fn qualify_fault_inner(
    device: Device,
    record: Arc<SourceBindings>,
    inputs: NativeFragmentImages,
    fault: GdnFragmentFault,
    cancelled: Option<&AtomicBool>,
) -> Result<GdnFragmentFaultOutcome> {
    let probe = Arc::new(Probe::default());
    let result = qualify_inner(
        device,
        record,
        inputs,
        cancelled,
        |_| Err(Error::State("fault must never compare/readback".into())),
        Mode::Fault {
            fault,
            probe: probe.clone(),
        },
    )
    .await;
    match result {
        Ok(Outcome::Fault(out)) => Ok(out),
        Ok(Outcome::Compared(_)) => Err(Error::State("fault returned positive completion".into())),
        Err(error) => {
            let cut = {
                let guard = probe
                    .0
                    .lock()
                    .map_err(|_| Error::State("native fault probe poisoned".into()))?;
                guard
                    .as_ref()
                    .and_then(|w| w.body.upgrade())
                    .is_some_and(|b| b.fault_cut_observed())
            };
            if cut {
                probe.inspect(fault)
            } else {
                Err(error)
            }
        }
    }
}
pub(super) fn pressure(
    service: Arc<ResidentService>,
    fault: GdnFragmentFault,
) -> Result<GdnFragmentFaultOutcome> {
    let key = service.physical_key();
    if !matches!(
        service.allocate_uninitialized(
            service
                .capacity()
                .checked_add(1)
                .ok_or_else(|| Error::Limit("native pressure overflow".into()))?,
            BufferRole::Scratch
        ),
        Err(Error::Limit(_))
    ) || service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
    {
        return Err(Error::State(
            "native actual cap+1 pressure did not refuse before allocation/H2D".into(),
        ));
    }
    drop(service);
    let physical =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.fenced || physical.reserved != 0 {
        return Err(Error::State(
            "native clean pressure original envelope not retired".into(),
        ));
    }
    Ok(GdnFragmentFaultOutcome {
        fault,
        status: "REFUSED_BEFORE_ALLOCATION",
        ready_completions: 0,
        compared_words: 0,
        rms_stages: 0,
        native_launches: 0,
        convolution_dispatches: 0,
        resident_bytes: 0,
        pending_events: 0,
        uploaded_value_bytes: 0,
        original_root: None,
        original_root_domains: 0,
        original_state_buffers: 0,
        retained_graph_leases: 0,
        readback_descriptors: 0,
        source_backing_retained: false,
        body_retained: false,
        all_accesses_retained: false,
        quarantined_handles: 0,
        quarantined_completion_owners: 0,
        physical_reserved_bytes: physical.reserved,
        physical_fenced: physical.fenced,
    })
}
