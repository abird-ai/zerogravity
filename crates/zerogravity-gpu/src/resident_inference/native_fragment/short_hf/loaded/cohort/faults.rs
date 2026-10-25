//! Fresh-process diagnostics use the real submitted flight's unknown quarantine.
use super::*;
use crate::resident_program::native_cohort::FullModelCohort;
pub use crate::resident_program::native_cohort::ShortHfCohortFault;
use std::sync::Mutex;
pub(super) struct Probe {
    pub(super) fault: ShortHfCohortFault,
    body: Mutex<Weak<FullModelCohort>>,
}
impl Probe {
    pub(super) fn bind(&self, body: &Arc<FullModelCohort>) -> Result<()> {
        let mut held = self
            .body
            .lock()
            .map_err(|_| Error::State("B4 diagnostic owner lock".into()))?;
        if held.upgrade().is_some() {
            return Err(Error::State("B4 diagnostic body already bound".into()));
        }
        *held = Arc::downgrade(body);
        Ok(())
    }
}
pub struct ShortHfCohortFaultOutcome {
    pub fault: ShortHfCohortFault,
    pub roots: Vec<RootBinding>,
    body: Weak<FullModelCohort>,
    cold: Weak<ColdModel>,
    cohort: Weak<ColdCohort>,
}
#[derive(Debug)]
pub struct ShortHfCohortFaultRetention {
    pub submitted_nodes: usize,
    pub native_launches: usize,
    pub projection_launches: usize,
    pub device_copies: usize,
    pub sampler_dispatches: usize,
    /// Grouped typed-transport launches submitted up to the cut cursor
    /// (GDN kinds 0..4 + KV import/export); positive evidence that the
    /// grouped transports ran only up to the submitted node.
    pub grouped_transport_launches: usize,
    pub retained_actual_leases: usize,
    pub quarantined_handles: usize,
    pub quarantined_owners: usize,
    pub candidate_planes_unpromoted: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub physical_reserved_bytes: u64,
    pub service_usable: bool,
    pub physical_fenced: bool,
}
impl ShortHfCohortFaultOutcome {
    /// Called AFTER dropping facade/Loaded/sessions. Only actual quarantine may
    /// retain the weak current body, prior roots, native context and source.
    pub fn retained_after_external_drop(&self) -> Result<ShortHfCohortFaultRetention> {
        let body = self
            .body
            .upgrade()
            .ok_or_else(|| Error::State("B4 submitted unknown body was lost".into()))?;
        let cold = self
            .cold
            .upgrade()
            .ok_or_else(|| Error::State("B4 submitted cold/source graph was lost".into()))?;
        let cohort = self
            .cohort
            .upgrade()
            .ok_or_else(|| Error::State("B4 submitted cohort/source graph was lost".into()))?;
        let service = body.status().service();
        if !Arc::ptr_eq(service, cold.native.cohort_workspace().service())
            || cohort.capacity.pending() != 1
        {
            return Err(Error::State(
                "B4 exact source/native/reservation retention".into(),
            ));
        }
        let [
            submitted_nodes,
            native_launches,
            projection_launches,
            device_copies,
            sampler_dispatches,
        ] = body.fault_counters();
        let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?;
        let mut candidate_planes_unpromoted = 0;
        for row in body.state_bindings() {
            for binding in row {
                for buffer in binding.candidate.buffers() {
                    if !buffer.original_lease()?.is_initialized() {
                        candidate_planes_unpromoted += 1;
                    }
                }
            }
        }
        Ok(ShortHfCohortFaultRetention {
            submitted_nodes,
            native_launches,
            projection_launches,
            device_copies,
            sampler_dispatches,
            grouped_transport_launches: body.grouped_transport_launches(),
            retained_actual_leases: body.accesses().len(),
            quarantined_handles: crate::physical_quarantine::quarantined_handles_for(
                service.client(),
            )?,
            quarantined_owners: crate::physical_quarantine::quarantined_completion_owners_for(
                service.client(),
            )?,
            candidate_planes_unpromoted,
            resident_bytes: service.resident_bytes()?,
            pending_events: service.pending_events()?,
            physical_reserved_bytes: physical.reserved,
            physical_fenced: physical.fenced,
            service_usable: service.ensure_usable().is_ok(),
        })
    }
}
impl ShortHfLoaded {
    pub async fn probe_feedback_cohort_fault(
        self: &Arc<Self>,
        sessions: &[&ShortHfSession],
        cancelled: &[&AtomicBool],
        fault: ShortHfCohortFault,
    ) -> Result<ShortHfCohortFaultOutcome> {
        let inputs = DecodeCohortInputs::issue(sessions, cancelled)?;
        let roots = sessions.iter().map(|s| s.binding()).collect::<Vec<_>>();
        let probe = Arc::new(Probe {
            fault,
            body: Mutex::new(Weak::new()),
        });
        let result = self
            .prepare_cohort_inner(inputs, sessions, cancelled, None, Some(probe.clone()))
            .await;
        match result {
            Err(Error::State(message))
                if message == "B4 deliberate submitted unknown completion cut" => {}
            Err(error) => return Err(error),
            Ok(_) => {
                return Err(Error::State(
                    "B4 actual submitted fault cut was not reached".into(),
                ));
            }
        }
        if sessions.iter().zip(&roots).any(|(s, r)| s.binding() != *r)
            || self.service().ensure_usable().is_ok()
        {
            return Err(Error::State(
                "B4 unknown cut published or failed to fence".into(),
            ));
        }
        let body = probe
            .body
            .lock()
            .map_err(|_| Error::State("B4 diagnostic owner lock".into()))?
            .clone();
        Ok(ShortHfCohortFaultOutcome {
            fault,
            roots,
            body,
            cold: Arc::downgrade(&self.cold),
            cohort: Arc::downgrade(
                self.cohort
                    .as_ref()
                    .ok_or_else(|| Error::State("B4 admitted cohort absent".into()))?,
            ),
        })
    }
}
