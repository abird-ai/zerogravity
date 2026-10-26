//! Same sequence inner path; only the final phase deliberately loses completion.
use super::*;
use std::sync::{Mutex as StdMutex, Weak};
use zerogravity_core::device_events::BufferLeaseId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnSequenceFault {
    AdmissionPressure,
    UnknownAfterConvolution,
    CancelBeforeRecurrent,
    UnknownAfterPostAttentionRms,
    CancelBeforeMlpDown,
}
impl GdnSequenceFault {
    fn fragment(self) -> GdnFragmentFault {
        match self {
            Self::AdmissionPressure => GdnFragmentFault::AdmissionPressure,
            Self::UnknownAfterConvolution => GdnFragmentFault::UnknownAfterConvolution,
            Self::CancelBeforeRecurrent => GdnFragmentFault::CancelBeforeRecurrent,
            Self::UnknownAfterPostAttentionRms => GdnFragmentFault::UnknownAfterPostAttentionRms,
            Self::CancelBeforeMlpDown => GdnFragmentFault::CancelBeforeMlpDown,
        }
    }
}
#[derive(Clone, Debug)]
pub struct GdnSequenceFaultOutcome {
    pub fault: GdnSequenceFault,
    /// Earlier phases really completed and were independently compared.
    pub completed_phases: usize,
    pub attempted_phases: usize,
    pub original_previous_ready_retained: bool,
    pub all_prepared_phases_retained: bool,
    pub prepared_state_buffers: usize,
    /// Only the faulting phase: zero Ready/comparisons, original fence/pending1.
    pub active: GdnFragmentFaultOutcome,
}
type Descriptor = (BufferLeaseId, usize, usize);
struct Witness {
    owners: Weak<SequenceOwners>,
    previous: Weak<CompletedPrefixPhase>,
    phase_index: usize,
    cohorts: Vec<Vec<Descriptor>>,
    states: Vec<(BufferLeaseId, u64, u64)>,
}
#[derive(Default)]
pub(super) struct SequenceProbe {
    active: faults::Probe,
    witness: StdMutex<Option<Witness>>,
}
pub(super) struct Arm<'a> {
    pub service: &'a Arc<ResidentService>,
    pub source: &'a VerifiedTensorSource,
    pub owners: &'a Arc<SequenceOwners>,
    pub index: usize,
    pub previous: Option<&'a Arc<CompletedPrefixPhase>>,
    pub accesses: &'a [(ResidentLease, DeviceAccess)],
    pub base: RootBinding,
}
fn descriptors(owners: &SequenceOwners) -> Vec<Vec<Descriptor>> {
    owners
        .frames
        .iter()
        .map(|f| {
            f.cohort
                .iter()
                .map(|r| (r.lease_id(), r.offset(), r.bytes()))
                .collect()
        })
        .collect()
}
fn state_ids(owners: &SequenceOwners) -> Vec<(BufferLeaseId, u64, u64)> {
    owners
        .initial
        .iter()
        .chain(owners.candidates.iter().flatten())
        .flat_map(|d| &d.leases)
        .map(|l| (l.id(), l.allocation_generation(), l.service_generation()))
        .collect()
}
impl SequenceProbe {
    fn arm(&self, arm: Arm<'_>) -> Result<()> {
        let Arm {
            service,
            source,
            owners,
            index,
            previous,
            accesses,
            base,
        } = arm;
        let previous = previous
            .ok_or_else(|| Error::State("late-phase fault needs real preceding Ready".into()))?;
        if index + 1 != owners.bodies.len() || previous.completed_phases() != index {
            return Err(Error::State(
                "sequence fault only after genuine earlier completions".into(),
            ));
        }
        let root_buffers = owners.candidates[index - 1]
            .iter()
            .chain(&owners.candidates[index])
            .map(|d| d.leases.len())
            .sum();
        self.active.arm(faults::ProbeContext {
            service,
            source,
            body: &owners.bodies[index],
            readbacks: &owners.frames[index].cohort,
            accesses,
            base,
            root_buffers,
        })?;
        let mut guard = self
            .witness
            .lock()
            .map_err(|_| Error::State("sequence witness poisoned".into()))?;
        if guard.is_some() {
            return Err(Error::State("sequence witness already armed".into()));
        }
        *guard = Some(Witness {
            owners: Arc::downgrade(owners),
            previous: Arc::downgrade(previous),
            phase_index: index,
            cohorts: descriptors(owners),
            states: state_ids(owners),
        });
        Ok(())
    }
    fn inspect(&self, fault: GdnSequenceFault) -> Result<GdnSequenceFaultOutcome> {
        // Called only after the inner future and ALL its normal locals drop.
        let active = self.active.inspect(fault.fragment())?;
        let guard = self
            .witness
            .lock()
            .map_err(|_| Error::State("sequence witness poisoned".into()))?;
        let witness = guard
            .as_ref()
            .ok_or_else(|| Error::State("sequence actual owner witness absent".into()))?;
        let owners = witness
            .owners
            .upgrade()
            .ok_or_else(|| Error::State("uncertain prepared future graph released".into()))?;
        let previous = witness
            .previous
            .upgrade()
            .ok_or_else(|| Error::State("actual preceding physical Ready released".into()))?;
        let ready = previous.ready();
        if witness.phase_index + 1 != owners.bodies.len()
            || previous.completed_phases() != witness.phase_index
            || descriptors(&owners) != witness.cohorts
            || state_ids(&owners) != witness.states
            || !Arc::ptr_eq(ready.service(), owners.initial[0].leases[0].service())
            || ready.root_binding().is_some()
            || ready.event_ids().is_empty()
            || owners
                .bodies
                .iter()
                .any(|body| !body.source().same_issuer(&owners._source))
            || owners.initial.len() != 42
            || owners.candidates.iter().any(|d| d.len() != 42)
        {
            return Err(Error::State(
                "uncertain sequence failed exact original Ready/owners/cohorts/all42 retention"
                    .into(),
            ));
        }
        Ok(GdnSequenceFaultOutcome {
            fault,
            completed_phases: witness.phase_index,
            attempted_phases: owners.bodies.len(),
            original_previous_ready_retained: true,
            all_prepared_phases_retained: true,
            prepared_state_buffers: witness.states.len(),
            active,
        })
    }
}
pub(super) enum SequenceMode {
    Compare,
    RestoreBeforeContinuation,
    Fault {
        fault: GdnSequenceFault,
        probe: Arc<SequenceProbe>,
    },
}
impl SequenceMode {
    pub(super) fn is_restore(&self) -> bool {
        matches!(self, Self::RestoreBeforeContinuation)
    }
    pub(super) fn is_pressure(&self) -> bool {
        matches!(
            self,
            Self::Fault {
                fault: GdnSequenceFault::AdmissionPressure,
                ..
            }
        )
    }
    pub(super) fn active_fault(&self, index: usize, phases: usize) -> Option<GdnFragmentFault> {
        match self {
            Self::Fault { fault, .. } if index + 1 == phases => Some(fault.fragment()),
            _ => None,
        }
    }
    pub(super) fn arm(&self, arm: Arm<'_>) -> Result<()> {
        match self {
            Self::Fault { probe, .. } => probe.arm(arm),
            Self::Compare | Self::RestoreBeforeContinuation => Err(Error::State(
                "positive sequence cannot arm fault witness".into(),
            )),
        }
    }
}
pub(super) enum SequenceOutcome {
    Compared(GdnSequenceReceipt),
    Fault(GdnSequenceFaultOutcome),
    Restored(GdnSequenceRestoreReceipt),
}
/// Earlier phase callbacks are mandatory independent comparisons; the final
/// faulting phase must never reach this callback, D2H, Ready or native unload.
pub async fn qualify_gdn_sequence_fault<F>(
    device: Device,
    records: Vec<Arc<SourceBindings>>,
    inputs: LayerMlpCubinInputs,
    fault: GdnSequenceFault,
    cancelled: Option<&AtomicBool>,
    mut validate: F,
) -> Result<GdnSequenceFaultOutcome>
where
    F: FnMut(usize, GdnFragmentReadback<'_>) -> Result<()>,
{
    not_cancelled(cancelled)?;
    let phases = records.len();
    let probe = Arc::new(SequenceProbe::default());
    let result = qualify_sequence_inner(
        device,
        records,
        inputs,
        cancelled,
        |index, read| {
            if index + 1 == phases {
                return Err(Error::State(
                    "uncertain final phase must not compare".into(),
                ));
            }
            validate(index, read)
        },
        SequenceMode::Fault {
            fault,
            probe: probe.clone(),
        },
    )
    .await;
    match result {
        Ok(SequenceOutcome::Fault(value)) => Ok(value),
        Ok(SequenceOutcome::Compared(_) | SequenceOutcome::Restored(_)) => {
            Err(Error::State("sequence fault unexpectedly completed".into()))
        }
        Err(_) if probe.active.cut_observed()? => probe.inspect(fault),
        Err(error) => Err(error),
    }
}
