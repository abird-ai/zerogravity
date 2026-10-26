//! Cuts in the SAME complete prepared graph/original node loop.
use super::*;
use std::sync::{Mutex as StdMutex, Weak};
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::device_events::BufferLeaseId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceModelFault {
    AdmissionPressure,
    UnknownAfterConvolution,
    CancelBeforeRecurrent,
}
#[derive(Clone, Debug)]
pub struct SourceModelFaultOutcome {
    pub fault: SourceModelFault,
    pub status: &'static str,
    pub ready_completions: usize,
    pub readback_commands: usize,
    pub completed_phases: usize,
    pub native_launches: usize,
    pub convolution_dispatches: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub prepared_state_buffers: usize,
    pub retained_plan_values: usize,
    pub retained_source_weight_groups: usize,
    pub submitted_current_leases: usize,
    pub all_prepared_owners_retained: bool,
    pub all_future_uninitialized: bool,
    pub native_image_owners: usize,
    pub native_head_kernel_owners: usize,
    pub native_charge_bytes: usize,
    pub image_admission_bytes: usize,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
}
type Descriptor = (BufferLeaseId, usize, usize);
struct Witness {
    owners: Weak<FullOwners>,
    service: Weak<ResidentService>,
    current: Vec<BufferLeaseId>,
    prepared: BTreeSet<BufferLeaseId>,
    cohorts: Vec<Vec<Descriptor>>,
    resident: usize,
}
#[derive(Default)]
pub(super) struct Probe(StdMutex<Option<Witness>>);
fn prepared_ids(owners: &FullOwners) -> Result<BTreeSet<BufferLeaseId>> {
    let mut ids = BTreeSet::new();
    for body in &owners.bodies {
        ids.extend(body.accesses()?.iter().map(|(l, _)| l.id()));
    }
    ids.extend(
        owners
            .initial
            .iter()
            .flatten()
            .chain(owners.candidates.iter().flatten())
            .flat_map(|d| d.leases.iter().map(ResidentLease::id)),
    );
    Ok(ids)
}
fn cohorts(owners: &FullOwners) -> Vec<Vec<Descriptor>> {
    owners
        .frames
        .iter()
        .map(|f| {
            f.reads
                .iter()
                .map(|r| (r.lease_id(), r.offset(), r.bytes()))
                .collect()
        })
        .collect()
}
impl Probe {
    pub(super) fn arm(
        &self,
        owners: &Arc<FullOwners>,
        service: &Arc<ResidentService>,
        current: &[(ResidentLease, DeviceAccess)],
    ) -> Result<()> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| Error::State("full24 fault probe poisoned".into()))?;
        if guard.is_some() {
            return Err(Error::State("full24 fault probe already armed".into()));
        }
        *guard = Some(Witness {
            owners: Arc::downgrade(owners),
            service: Arc::downgrade(service),
            current: current.iter().map(|(l, _)| l.id()).collect(),
            prepared: prepared_ids(owners)?,
            cohorts: cohorts(owners),
            resident: service.resident_bytes()?,
        });
        Ok(())
    }
    pub(super) fn cut_observed(&self) -> Result<bool> {
        let guard = self
            .0
            .lock()
            .map_err(|_| Error::State("full24 fault probe poisoned".into()))?;
        Ok(guard
            .as_ref()
            .and_then(|w| w.owners.upgrade())
            .is_some_and(|o| o.bodies[0].fault_cut_observed()))
    }
    pub(super) fn inspect(&self, fault: SourceModelFault) -> Result<SourceModelFaultOutcome> {
        let guard = self
            .0
            .lock()
            .map_err(|_| Error::State("full24 fault probe poisoned".into()))?;
        let w = guard
            .as_ref()
            .ok_or_else(|| Error::State("full24 failed before actual fault cut".into()))?;
        let owners = w
            .owners
            .upgrade()
            .ok_or_else(|| Error::State("full24 original submitted graph released".into()))?;
        let service = w
            .service
            .upgrade()
            .ok_or_else(|| Error::State("full24 unknown original service released".into()))?;
        let body = &owners.bodies[0];
        let actual = prepared_ids(&owners)?;
        let native = owners.library.inspect_native_retention()?;
        let future_uninitialized = owners.bodies.iter().skip(1).all(|body| {
            (1..327).all(|i| {
                body.values()
                    .lease(ValueId(i))
                    .is_ok_and(|l| !l.is_initialized())
            })
        }) && owners
            .candidates
            .iter()
            .skip(1)
            .flatten()
            .flat_map(|d| &d.leases)
            .all(|l| !l.is_initialized());
        let state_buffers = owners
            .initial
            .iter()
            .flatten()
            .chain(owners.candidates.iter().flatten())
            .map(|d| d.leases.len())
            .sum();
        let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?;
        let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
        let held = crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
        if !body.fault_cut_observed()
            || body.convolution_dispatches() == 0
            || actual != w.prepared
            || cohorts(&owners) != w.cohorts
            || !future_uninitialized
            || state_buffers != 7 * 48
            || w.current.iter().any(|id| !actual.contains(id))
            || service.resident_bytes()? != w.resident
            || service.pending_events()? != 1
            || service.ensure_usable().is_ok()
            || !physical.fenced
            || physical.reserved != service.capacity() as u64
            || handles < w.current.len()
            || held == 0
        {
            return Err(Error::State("full24 unknown did not retain exact submitted/prepared graph, future initialization boundary or physical fence".into()));
        }
        Ok(SourceModelFaultOutcome {
            fault,
            status: "UNKNOWN_RETAINED",
            ready_completions: 0,
            readback_commands: 0,
            completed_phases: 0,
            native_launches: body.native_launches(),
            convolution_dispatches: body.convolution_dispatches(),
            resident_bytes: w.resident,
            pending_events: service.pending_events()?,
            prepared_state_buffers: state_buffers,
            retained_plan_values: owners.bodies.len() * 327,
            retained_source_weight_groups: owners.weights.accesses().len() + 1,
            submitted_current_leases: w.current.len(),
            all_prepared_owners_retained: true,
            all_future_uninitialized: future_uninitialized,
            native_image_owners: native.images,
            native_head_kernel_owners: native.head_kernels,
            native_charge_bytes: native.native_charge_bytes,
            image_admission_bytes: native.image_admission_bytes,
            quarantined_handles: handles,
            quarantined_completion_owners: held,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: physical.fenced,
        })
    }
}
pub(super) enum Mode {
    Compare,
    Fault {
        fault: SourceModelFault,
        probe: Arc<Probe>,
    },
}
impl Mode {
    pub(super) fn is_pressure(&self) -> bool {
        matches!(
            self,
            Self::Fault {
                fault: SourceModelFault::AdmissionPressure,
                ..
            }
        )
    }
    pub(super) fn fault_mode(&self) -> usize {
        match self {
            Self::Compare => 0,
            Self::Fault {
                fault: SourceModelFault::UnknownAfterConvolution,
                ..
            } => 1,
            Self::Fault {
                fault: SourceModelFault::CancelBeforeRecurrent,
                ..
            } => 2,
            _ => 0,
        }
    }
    pub(super) fn arm(
        &self,
        owners: &Arc<FullOwners>,
        service: &Arc<ResidentService>,
        accesses: &[(ResidentLease, DeviceAccess)],
    ) -> Result<()> {
        match self {
            Self::Compare => Ok(()),
            Self::Fault { probe, .. } => probe.arm(owners, service, accesses),
        }
    }
}
pub(super) enum Outcome {
    Natural(super::natural_cold::PreparedNaturalCold),
    Compared(SourceModelReceipt),
    Fault(SourceModelFaultOutcome),
}
pub async fn qualify_source_model_fault(
    device: Device,
    scopes: SourceModelScopes,
    inputs: SourceModelInputs,
    rotary: Option<RotaryOutcome>,
    fault: SourceModelFault,
    cancelled: Option<&AtomicBool>,
) -> Result<SourceModelFaultOutcome> {
    let (source, records) = scopes.into_parts();
    let probe = Arc::new(Probe::default());
    let result = qualify_inner(
        device,
        source,
        records,
        inputs,
        Execution {
            rotary,
            cancelled,
            natural: false,
            reusable: false,
            mode: Mode::Fault {
                fault,
                probe: probe.clone(),
            },
        },
        |_| {
            Err(Error::State(
                "full24 fault must never readback/compare".into(),
            ))
        },
    )
    .await;
    match result {
        Ok(Outcome::Fault(value)) => Ok(value),
        Ok(Outcome::Natural(_)) => Err(Error::State(
            "forced fault qualifier returned natural preparation".into(),
        )),
        Ok(Outcome::Compared(_)) => Err(Error::State(
            "full24 fault returned positive completion".into(),
        )),
        Err(error) => {
            let intended = matches!((&error, fault),
                (Error::State(message), SourceModelFault::UnknownAfterConvolution)
                    if message == "deliberate full24 unknown after actual source Conv and canonical copy")
                || matches!((&error, fault),
                (Error::State(message), SourceModelFault::CancelBeforeRecurrent)
                    if message == "closed original cubin qualification cancelled");
            if intended && probe.cut_observed()? {
                probe.inspect(fault)
            } else {
                Err(error)
            }
        }
    }
}
pub(super) fn pressure(service: Arc<ResidentService>) -> Result<SourceModelFaultOutcome> {
    let active = super::super::faults::pressure(service, GdnFragmentFault::AdmissionPressure)?;
    Ok(SourceModelFaultOutcome {
        fault: SourceModelFault::AdmissionPressure,
        status: active.status,
        ready_completions: 0,
        readback_commands: 0,
        completed_phases: 0,
        native_launches: 0,
        convolution_dispatches: 0,
        resident_bytes: active.resident_bytes,
        pending_events: active.pending_events,
        prepared_state_buffers: 0,
        retained_plan_values: 0,
        retained_source_weight_groups: 0,
        submitted_current_leases: 0,
        all_prepared_owners_retained: false,
        all_future_uninitialized: true,
        native_image_owners: 0,
        native_head_kernel_owners: 0,
        native_charge_bytes: 0,
        image_admission_bytes: 0,
        quarantined_handles: 0,
        quarantined_completion_owners: 0,
        physical_reserved_bytes: active.physical_reserved_bytes,
        physical_fenced: active.physical_fenced,
    })
}
