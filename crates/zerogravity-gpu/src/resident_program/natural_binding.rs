//! Closed geometry/storage checks; no call, state, completion or teacher issuer.
use super::*;
use crate::resident_leases::ResidentLease;
use crate::resident_program::source_bindings::SourceBindings;
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode};

pub(crate) fn slot(call: &SourceCall) -> Result<usize> {
    if call.rows().capacity() != 512 || call.rows().vocabulary() != 248320 {
        return Err(Error::Unsupported(
            "natural actual source call capacity/vocabulary".into(),
        ));
    }
    slot_geometry(call.mode(), call.rows().base(), call.rows().len())
}
fn slot_geometry(phase: SourceCallMode, base: usize, rows: usize) -> Result<usize> {
    use SourceCallMode::*;
    match (phase, base, rows) {
        (Prefill, 0, 1) => Ok(0),
        (Decode, 1, 1) => Ok(1),
        (Prefill, 0, 64) => Ok(2),
        (Prefill, 64, 2) => Ok(3),
        (Decode, 66, 1) => Ok(4),
        _ => Err(Error::Unsupported(
            "natural call outside original five geometries".into(),
        )),
    }
}
pub(crate) fn same_lease(a: &ResidentLease, b: &ResidentLease) -> Result<()> {
    let ar = BufferRange::new(&a.handle())?;
    let br = BufferRange::new(&b.handle())?;
    if !Arc::ptr_eq(a.service(), b.service())
        || a.id() != b.id()
        || a.allocation_generation() != b.allocation_generation()
        || a.service_generation() != b.service_generation()
        || a.bytes() != b.bytes()
        || a.role() != b.role()
        || a.source_identity() != b.source_identity()
        || ar.service != br.service
        || ar.memory != br.memory
        || ar.bytes != br.bytes
    {
        return Err(Error::State(
            "natural binding changes original storage lease".into(),
        ));
    }
    a.service().ensure_usable()
}
/// Compare the actual sealed recipes, excluding ONLY the true new call issuer.
/// Mathematical roles, source parameters, SSA/state effects and layouts stay exact.
pub(crate) fn same_recipe(a: &SourceBindings, b: &SourceBindings) -> Result<()> {
    if a.plan().nodes().len() != 255
        || b.plan().nodes().len() != 255
        || a.plan().values().len() != 327
        || b.plan().values().len() != 327
    {
        return Err(Error::State(
            "natural exact full24 original recipe bounds".into(),
        ));
    }
    let (ap, bp) = (a.plan(), b.plan());
    if slot(a.call())? != slot(b.call())?
        || a.call().source() != b.call().source()
        || !a.source().same_issuer(b.source())
        || ap.numerical_profile() != bp.numerical_profile()
        || ap.values() != bp.values()
        || ap.precisions() != bp.precisions()
        || ap.required_domains() != bp.required_domains()
        || ap.proposed_domains() != bp.proposed_domains()
        || a.selection().output() != b.selection().output()
        || a.selection().normalized() != b.selection().normalized()
        || ap.nodes().len() != bp.nodes().len()
    {
        return Err(Error::State(
            "natural recipe differs from cold geometry/source".into(),
        ));
    }
    for (index, (old, new)) in ap.nodes().iter().zip(bp.nodes()).enumerate() {
        if old.inputs != new.inputs || old.outputs != new.outputs || old.state != new.state {
            return Err(Error::State(
                "natural call changes sealed SSA/state effect".into(),
            ));
        }
        a.operator(index)?;
        b.operator(index)?;
        match (&old.operation, &new.operation) {
            (
                PlanOperation::SourceOperator {
                    descriptor: left,
                    state: ls,
                },
                PlanOperation::SourceOperator {
                    descriptor: right,
                    state: rs,
                },
            ) => {
                if left.call() != a.call()
                    || right.call() != b.call()
                    || left.kind() != right.kind()
                    || left.numerical_contract() != right.numerical_contract()
                    || left.matrices() != right.matrices()
                    || left.parameters() != right.parameters()
                    || ls != rs
                {
                    return Err(Error::State(
                        "rebind changes immutable mathematical source recipe".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::State(
                    "rebind introduces an unrelated operation".into(),
                ));
            }
        }
    }
    Ok(())
}

use crate::resident_inference::native_fragment::natural_publication::{
    NaturalCallInput, NaturalCallState,
};
use crate::resident_leases::TiedWeight;
use crate::resident_program::native_full::FullModelProgram;
use crate::resident_program::text_positions::TextPositionsPrepared;
use std::sync::atomic::{AtomicBool, Ordering};
/// Exact cold storage, no current Session/call binding or publisher. The shared
/// one-use claim survives failed/repeated wrappers and cannot be reset.
pub(crate) struct ColdProgramSlot {
    body: Arc<FullModelProgram>,
    values: Vec<DeviceBuffer>,
    preflight: ResidentPreflight,
    claimed: SlotClaim,
}
impl ColdProgramSlot {
    pub(crate) fn new(
        body: Arc<FullModelProgram>,
        values: Vec<DeviceBuffer>,
        preflight: ResidentPreflight,
    ) -> Self {
        Self {
            body,
            values,
            preflight,
            claimed: SlotClaim::default(),
        }
    }
    pub(crate) fn is_claimed(&self) -> bool {
        self.claimed.is_claimed()
    }
    pub(crate) fn record(&self) -> &Arc<SourceBindings> {
        self.body.record()
    }
    /// The original workspace lock covers binding through real completion.
    /// Every call has a new state submission claim and fresh output packet.
    pub(crate) fn bind_reusable(
        &self,
        input: Arc<NaturalCallInput>,
        state: Arc<NaturalCallState>,
        positions: Arc<TextPositionsPrepared>,
        table: Arc<TiedWeight>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ResidentProgram> {
        if self.claimed.is_claimed() {
            return Err(Error::State(
                "reusable template was consumed by closed-five binding".into(),
            ));
        }
        state.require_reusable_submission_claimed()?;
        same_recipe(self.record(), input.record())?;
        let source = input.source().clone();
        let state_ids = state
            .states()
            .iter()
            .flat_map(|state| state.leases().into_iter().map(|(lease, _)| lease.id()))
            .collect();
        let body =
            self.body
                .bind_reusable(input, state, &self.values, positions, table, cancelled)?;
        let mut program = ResidentProgram::from_natural_body(
            source,
            body,
            self.values.clone(),
            self.preflight.clone(),
        )?;
        program.finalize_full_model_preflight(&state_ids)?;
        Ok(program)
    }
    pub(crate) fn bind(
        &self,
        input: Arc<NaturalCallInput>,
        state: Arc<NaturalCallState>,
        positions: Arc<TextPositionsPrepared>,
        table: Arc<TiedWeight>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ResidentProgram> {
        if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Err(Error::State("natural cold bind cancelled".into()));
        }
        same_recipe(self.record(), input.record())?;
        self.claimed.claim()?;
        let source = input.source().clone();
        let state_ids = state
            .states()
            .iter()
            .flat_map(|state| state.leases().into_iter().map(|(lease, _)| lease.id()))
            .collect();
        let body =
            self.body
                .bind_natural(input, state, &self.values, positions, table, cancelled)?;
        let mut program = ResidentProgram::from_natural_body(
            source,
            body,
            self.values.clone(),
            self.preflight.clone(),
        )?;
        program.finalize_full_model_preflight(&state_ids)?;
        Ok(program)
    }
}

#[derive(Default)]
struct SlotClaim(AtomicBool);
impl SlotClaim {
    fn is_claimed(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    fn claim(&self) -> Result<()> {
        self.0
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| Error::State("natural original physical slot already claimed".into()))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_slot_claim_is_shared_and_survives_failed_bind_without_reset() {
        let claim = Arc::new(SlotClaim::default());
        assert!(!claim.is_claimed());
        let racers = (0..8)
            .map(|_| {
                let claim = claim.clone();
                std::thread::spawn(move || claim.claim().is_ok())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            racers
                .into_iter()
                .map(|job| job.join().unwrap())
                .filter(|won| *won)
                .count(),
            1
        );
        assert!(claim.is_claimed());
        assert!(claim.claim().is_err());
    }
    #[test]
    fn fixed_storage_slots_do_not_derive_authority_from_row_count() {
        use SourceCallMode::*;
        for (index, (phase, base, rows)) in [
            (Prefill, 0, 1),
            (Decode, 1, 1),
            (Prefill, 0, 64),
            (Prefill, 64, 2),
            (Decode, 66, 1),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(slot_geometry(phase, base, rows).unwrap(), index);
        }
        for (phase, base, rows) in [
            (Prefill, 1, 1),
            (Decode, 0, 1),
            (Decode, 64, 1),
            (Prefill, 64, 1),
            (Prefill, 0, 63),
        ] {
            assert!(slot_geometry(phase, base, rows).is_err());
        }
    }
}
