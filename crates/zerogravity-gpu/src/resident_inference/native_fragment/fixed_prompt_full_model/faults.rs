use super::*;
use crate::cuda::native::fixed_prompt_model_native::FixedPromptNativeRetention;
use std::sync::{Mutex as StdMutex, Weak};
use zerogravity_artifacts::source_weights::SourceOwnerWitness;
use zerogravity_core::device_events::BufferLeaseId;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedPromptFullModelFault {
    AdmissionPressure,
    UnknownAfterFirstConvolution,
    CancelBeforeFirstRecurrent,
    UnknownAfterLastFa,
    CancelBeforeHead,
    UnknownAfterHead,
}
impl FixedPromptFullModelFault {
    fn code(self) -> usize {
        match self {
            Self::AdmissionPressure => 0,
            Self::UnknownAfterFirstConvolution => 17,
            Self::CancelBeforeFirstRecurrent => 18,
            Self::UnknownAfterLastFa => 19,
            Self::CancelBeforeHead => 20,
            Self::UnknownAfterHead => 21,
        }
    }
}
#[derive(Clone, Debug)]
pub struct FixedPromptFullModelFaultOutcome {
    pub fault: FixedPromptFullModelFault,
    pub status: &'static str,
    pub native_launches: usize,
    pub convolution_dispatches: usize,
    pub attention_native_enqueues: [[usize; 4]; 6],
    pub head_native_enqueues: usize,
    pub prepared_state_buffers: usize,
    pub retained_plan_values: usize,
    pub retained_source_weight_groups: usize,
    pub prepared_graph_leases: usize,
    pub submitted_current_leases: usize,
    pub readback_descriptors: usize,
    pub source_retained: bool,
    pub all_prepared_owners_retained: bool,
    pub all_readback_descriptors_retained: bool,
    pub all_future_uninitialized: bool,
    pub head_output_unpromoted: bool,
    pub native_image_owners: usize,
    pub native_kernel_owners: usize,
    pub native_charge_bytes: usize,
    pub image_admission_bytes: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_value_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub ready_completions: usize,
    pub readback_commands: usize,
    pub expected_callbacks: usize,
    pub original_root: Option<RootBinding>,
    pub model_root_published: bool,
    pub parent_model_accepted: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Snapshot {
    leases: BTreeMap<BufferLeaseId, (usize, u64, bool)>,
    descriptors: Vec<(BufferLeaseId, usize, usize)>,
    future: BTreeSet<BufferLeaseId>,
    native: FixedPromptNativeRetention,
    resident: usize,
    uploaded: u64,
}
fn snapshot(owners: &PreparedOwners, service: &ResidentService) -> Result<Snapshot> {
    let mut leases = BTreeMap::<BufferLeaseId, ResidentLease>::new();
    for (lease, _) in owners.cold.body.accesses()? {
        leases.insert(lease.id(), lease.clone());
    }
    for domain in owners.cold.initial.iter().chain(&owners.cold.candidate) {
        for lease in &domain.leases {
            leases.insert(lease.id(), lease.clone());
        }
    }
    if leases.values().any(|lease| {
        !std::ptr::eq(lease.service().as_ref(), service)
            || lease.service_generation() != service.generation()
            || lease.source_identity() != owners.cut.scope().record().call().source()
    }) {
        return Err(Error::State(
            "whole metadata original graph service/source generation".into(),
        ));
    }
    let descriptors = owners
        .cohort
        .iter()
        .map(|r| (r.lease_id(), r.offset(), r.bytes()))
        .collect::<Vec<_>>();
    if descriptors.len() != budget::DESCRIPTORS
        || descriptors.iter().any(|(id, offset, bytes)| {
            leases.get(id).is_none_or(|lease| {
                *bytes == 0
                    || offset
                        .checked_add(*bytes)
                        .is_none_or(|end| end > lease.bytes())
            })
        })
    {
        return Err(Error::State(
            "whole retained exact actual descriptor ranges".into(),
        ));
    }
    let mut future = (1..327)
        .map(|id| {
            owners
                .cold
                .body
                .values()
                .lease(ValueId(id))
                .map(ResidentLease::id)
        })
        .collect::<Result<BTreeSet<_>>>()?;
    future.extend(
        owners
            .cold
            .candidate
            .iter()
            .flat_map(|domain| domain.leases.iter().map(ResidentLease::id)),
    );
    if future.is_empty()
        || future
            .iter()
            .any(|id| leases.get(id).is_none_or(ResidentLease::is_initialized))
        || !owners.native.whole_outputs_unpromoted()?
        || !owners.native.whole_graph_retained()?
        || !owners
            .head
            .retained_values(&owners.cut, owners.cold.body.values())?
    {
        return Err(Error::State(
            "whole unknown future owner initialized or source head lost".into(),
        ));
    }
    let weight_accesses = owners.cold.weights.accesses();
    if owners.attention.keys().copied().collect::<Vec<_>>() != [3, 7, 11, 15, 19, 23]
        || !Arc::ptr_eq(owners.native.head()?, &owners.head)
        || !Arc::ptr_eq(owners.head.tied(), &owners.cold.tied)
        || !Arc::ptr_eq(owners.native.weights(), &owners.cold.weights)
        || owners.attention.iter().any(|(layer, binding)| {
            owners
                .native
                .attention_bindings()
                .ok()
                .and_then(|map| map.get(layer))
                .is_none_or(|actual| !Arc::ptr_eq(actual, binding))
        })
    {
        return Err(Error::State(
            "whole exact cold/native operand owner Arcs differ".into(),
        ));
    }
    if weight_accesses.len() != 114
        || weight_accesses.iter().any(|(lease, access)| {
            *access != DeviceAccess::Read
                || !lease.is_initialized()
                || lease.role() != BufferRole::ImmutableWeights
        })
    {
        return Err(Error::State(
            "whole retained initialized source W owner metadata".into(),
        ));
    }
    let native = owners.native.retention_snapshot()?;
    if native.images != 12
        || native.kernels != 16
        || native.native_charge_bytes != 64 * 1024 * 1024
        || leases.len() > 1662
    {
        return Err(Error::State(
            "whole exact one-set graph metadata bound".into(),
        ));
    }
    Ok(Snapshot {
        leases: leases
            .into_iter()
            .map(|(id, lease)| {
                (
                    id,
                    (
                        lease.bytes(),
                        lease.allocation_generation(),
                        lease.is_initialized(),
                    ),
                )
            })
            .collect(),
        descriptors,
        future,
        native,
        resident: service.resident_bytes()?,
        uploaded: service.uploaded_value_bytes(),
    })
}
struct Witness {
    owners: Weak<PreparedOwners>,
    service: Weak<ResidentService>,
    source: SourceOwnerWitness,
    snapshot: Snapshot,
    current: BTreeSet<BufferLeaseId>,
    base: RootBinding,
}
#[derive(Default)]
pub(super) struct Probe(StdMutex<Option<Witness>>);
impl Probe {
    fn arm(
        &self,
        owners: &Arc<PreparedOwners>,
        service: &Arc<ResidentService>,
        accesses: &[(ResidentLease, DeviceAccess)],
        base: RootBinding,
    ) -> Result<()> {
        let mut witness = self
            .0
            .lock()
            .map_err(|_| Error::State("whole probe lock poisoned".into()))?;
        let before = snapshot(owners, service)?;
        let current = accesses
            .iter()
            .map(|(lease, _)| lease.id())
            .collect::<BTreeSet<_>>();
        if witness.is_some()
            || owners.base != base
            || current.len() != accesses.len()
            || current.is_empty()
            || current.iter().any(|id| !before.leases.contains_key(id))
            || service.pending_events()? != 0
        {
            return Err(Error::State(
                "whole exact original preflight/current graph".into(),
            ));
        }
        *witness = Some(Witness {
            owners: Arc::downgrade(owners),
            service: Arc::downgrade(service),
            source: owners.cut.scope().source().weak_owner(),
            snapshot: before,
            current,
            base,
        });
        Ok(())
    }
    fn inspect(
        &self,
        fault: FixedPromptFullModelFault,
    ) -> Result<FixedPromptFullModelFaultOutcome> {
        let lock = self
            .0
            .lock()
            .map_err(|_| Error::State("whole probe lock poisoned".into()))?;
        let witness = lock
            .as_ref()
            .ok_or_else(|| Error::State("whole intended submitted fault not armed".into()))?;
        // No outer strong owner remains; only original quarantine may retain this graph.
        if !witness.source.retained() {
            return Err(Error::State("whole quarantine lost source".into()));
        }
        let source = &witness.source;
        let owners = witness
            .owners
            .upgrade()
            .ok_or_else(|| Error::State("whole quarantine lost prepared graph".into()))?;
        let service = witness
            .service
            .upgrade()
            .ok_or_else(|| Error::State("whole quarantine lost service".into()))?;

        let after = snapshot(&owners, &service)?;
        if witness.snapshot != after
            || !source.same_issuer(owners.cut.scope().source())
            || !owners.cold.body.whole_fault_cut_seen()
            || service.pending_events()? != 1
            || service.ensure_usable().is_ok()
        {
            return Err(Error::State(
                "whole actual unknown graph changed/lost original fence".into(),
            ));
        }
        let counts = owners.native.whole_counts()?;
        let head = owners.head.enqueues();
        let native = owners.cold.body.native_launches();
        let conv = owners.cold.body.convolution_dispatches();
        let expected = match fault {
            FixedPromptFullModelFault::UnknownAfterFirstConvolution
            | FixedPromptFullModelFault::CancelBeforeFirstRecurrent => (3, 6, [[0; 4]; 6], 0),
            FixedPromptFullModelFault::UnknownAfterLastFa => (209, 108, [[1; 4]; 6], 0),
            FixedPromptFullModelFault::CancelBeforeHead => (216, 108, [[1; 4]; 6], 0),
            FixedPromptFullModelFault::UnknownAfterHead => (217, 108, [[1; 4]; 6], 1),
            FixedPromptFullModelFault::AdmissionPressure => {
                return Err(Error::State(
                    "pressure cannot enter submitted whole witness".into(),
                ));
            }
        };
        if (native, conv, counts, head) != expected {
            return Err(Error::State(
                "whole actual named source cut counters differ".into(),
            ));
        }
        let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?;
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let completion_owners =
            crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        if !physical.fenced
            || physical.reserved != service.capacity() as u64
            || handles < witness.current.len()
            || completion_owners == 0
            || witness
                .current
                .iter()
                .any(|id| !after.leases.contains_key(id))
        {
            return Err(Error::State(
                "whole original physical quarantine absent".into(),
            ));
        }
        let head_output_unpromoted = owners.head.output_unpromoted()?;
        Ok(FixedPromptFullModelFaultOutcome {
            fault,
            status: "UNKNOWN_RETAINED",
            native_launches: native,
            convolution_dispatches: conv,
            attention_native_enqueues: counts,
            head_native_enqueues: head,
            prepared_state_buffers: owners
                .cold
                .initial
                .iter()
                .chain(&owners.cold.candidate)
                .map(|domain| domain.leases.len())
                .sum(),
            retained_plan_values: owners.cut.values().len(),
            retained_source_weight_groups: owners.cold.weights.accesses().len(),
            prepared_graph_leases: after.leases.len(),
            submitted_current_leases: witness.current.len(),
            readback_descriptors: after.descriptors.len(),
            source_retained: source.same_issuer(owners.cut.scope().source()),
            all_prepared_owners_retained: witness.snapshot.leases == after.leases,
            all_readback_descriptors_retained: witness.snapshot.descriptors == after.descriptors,
            all_future_uninitialized: witness.snapshot.future == after.future,
            head_output_unpromoted,
            native_image_owners: after.native.images,
            native_kernel_owners: after.native.kernels,
            native_charge_bytes: after.native.native_charge_bytes,
            image_admission_bytes: after.native.image_admission_bytes,
            resident_bytes: after.resident,
            pending_events: service.pending_events()?,
            uploaded_value_bytes: after.uploaded,
            quarantined_handles: handles,
            quarantined_completion_owners: completion_owners,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: physical.fenced,
            ready_completions: 0,
            readback_commands: 0,
            expected_callbacks: 0,
            original_root: Some(witness.base),
            model_root_published: false,
            parent_model_accepted: false,
        })
    }
}
pub(super) enum Mode {
    Compare,
    Fault {
        fault: FixedPromptFullModelFault,
        probe: Arc<Probe>,
    },
}
impl Mode {
    pub(super) fn is_pressure(&self) -> bool {
        matches!(
            self,
            Self::Fault {
                fault: FixedPromptFullModelFault::AdmissionPressure,
                ..
            }
        )
    }
    pub(super) fn code(&self) -> usize {
        match self {
            Self::Compare => 0,
            Self::Fault { fault, .. } => fault.code(),
        }
    }
    pub(super) fn arm(
        &self,
        owners: &Arc<PreparedOwners>,
        service: &Arc<ResidentService>,
        accesses: &[(ResidentLease, DeviceAccess)],
        base: RootBinding,
    ) -> Result<()> {
        match self {
            Self::Compare => Ok(()),
            Self::Fault { probe, .. } => probe.arm(owners, service, accesses, base),
        }
    }
}
pub(super) enum Outcome {
    Compared(Box<FixedPromptFullModelReceipt>),
    Fault(Box<FixedPromptFullModelFaultOutcome>),
}
fn intended_error(error: &Error, fault: FixedPromptFullModelFault) -> bool {
    let Error::State(message) = error else {
        return false;
    };
    match fault {
        FixedPromptFullModelFault::UnknownAfterFirstConvolution => {
            message == "deliberate full24 unknown after actual source Conv and canonical copy"
        }
        FixedPromptFullModelFault::CancelBeforeFirstRecurrent
        | FixedPromptFullModelFault::CancelBeforeHead => {
            message == "closed original cubin qualification cancelled"
        }
        FixedPromptFullModelFault::UnknownAfterLastFa => {
            message == "deliberate canonical47 whole unknown after last FA"
        }
        FixedPromptFullModelFault::UnknownAfterHead => {
            message == "deliberate canonical47 whole unknown after Head"
        }
        FixedPromptFullModelFault::AdmissionPressure => false,
    }
}
pub async fn qualify_fixed_prompt_full_model_fault(
    device: Device,
    source: VerifiedTensorSource,
    record: Arc<SourceBindings>,
    arithmetic: FixedPromptGdnArithmetic,
    inputs: FixedPromptQkKvQualificationInputs,
    fault: FixedPromptFullModelFault,
    cancelled: Option<&AtomicBool>,
) -> Result<FixedPromptFullModelFaultOutcome> {
    not_cancelled(cancelled)?;
    let probe = Arc::new(Probe::default());
    let result = qualify_inner(
        device,
        source,
        record,
        arithmetic,
        inputs,
        cancelled,
        Mode::Fault {
            fault,
            probe: probe.clone(),
        },
        |_| Err(Error::State("whole fault entered expected callback".into())),
    )
    .await;
    match result {
        Ok(Outcome::Fault(value)) => Ok(*value),
        Ok(Outcome::Compared(_)) => {
            Err(Error::State("whole fault completed positive flight".into()))
        }
        Err(error) => {
            if intended_error(&error, fault) {
                probe.inspect(fault)
            } else {
                Err(error)
            }
        }
    }
}
pub(super) fn pressure(service: Arc<ResidentService>) -> Result<FixedPromptFullModelFaultOutcome> {
    let active = super::super::faults::pressure(service, GdnFragmentFault::AdmissionPressure)?;
    Ok(FixedPromptFullModelFaultOutcome {
        fault: FixedPromptFullModelFault::AdmissionPressure,
        status: active.status,
        native_launches: 0,
        convolution_dispatches: 0,
        attention_native_enqueues: [[0; 4]; 6],
        head_native_enqueues: 0,
        prepared_state_buffers: 0,
        retained_plan_values: 0,
        retained_source_weight_groups: 0,
        prepared_graph_leases: 0,
        submitted_current_leases: 0,
        readback_descriptors: 0,
        source_retained: false,
        all_prepared_owners_retained: false,
        all_readback_descriptors_retained: false,
        all_future_uninitialized: true,
        head_output_unpromoted: false,
        native_image_owners: 0,
        native_kernel_owners: 0,
        native_charge_bytes: 0,
        image_admission_bytes: 0,
        resident_bytes: active.resident_bytes,
        pending_events: active.pending_events,
        uploaded_value_bytes: active.uploaded_value_bytes,
        quarantined_handles: active.quarantined_handles,
        quarantined_completion_owners: active.quarantined_completion_owners,
        physical_reserved_bytes: active.physical_reserved_bytes,
        physical_fenced: active.physical_fenced,
        ready_completions: 0,
        readback_commands: 0,
        expected_callbacks: 0,
        original_root: None,
        model_root_published: false,
        parent_model_accepted: false,
    })
}
