//! Original source-issued prior/candidate owners for one physical cohort.
use super::*;
use std::collections::BTreeSet;

pub(crate) struct DecodeCohortStates {
    inputs: Arc<DecodeCohortInputs>,
    rows: Box<[Arc<ShortHfCallState>]>,
    // Preparing, submitted, unknown and completed Pending all retain this slot.
    reservation: DecodeCohortReservation,
}

impl DecodeCohortStates {
    /// None is solely a proved root-pool capacity refusal before submission.
    /// Source, stale, numerical and uncertain-owner failures remain errors.
    pub(crate) async fn prepare(
        inputs: Arc<DecodeCohortInputs>,
        sessions: &[&ShortHfSession],
        cancelled: &[&AtomicBool],
        capacity: &DecodeCohortCapacity,
    ) -> Result<Option<Arc<Self>>> {
        inputs.require_current(sessions)?;
        if cancelled.len() != sessions.len() {
            return Err(Error::Malformed("HF cohort exact cancellation rows".into()));
        }
        let loaded = inputs
            .loaded
            .upgrade()
            .ok_or_else(|| Error::State("HF cohort loaded source retired".into()))?;
        let bound =
            DecodeCohortRootBound::derive(loaded.active.load(Ordering::Acquire), sessions.len())?;
        let root_layout = sessions[0]
            .original
            .domains
            .iter()
            .flat_map(|domain| &domain.leases)
            .map(ResidentLease::bytes)
            .collect::<Vec<_>>();
        if bound.bytes(&root_layout)? != bound.total * loaded.model.state_bytes {
            return Err(Error::State(
                "B4 all actual root planes differ from source inventory".into(),
            ));
        }
        if loaded.resources().root_count < bound.total {
            return Ok(None);
        }
        let Some(reservation) = capacity.try_reserve(loaded.service())? else {
            return Ok(None);
        };
        let mut rows = Vec::with_capacity(sessions.len());
        for ((session, input), flag) in sessions.iter().zip(inputs.rows()).zip(cancelled) {
            request_not_cancelled(Some(flag))?;
            rows.push(
                session
                    .original
                    .bind_fresh_short_hf_state(input.clone(), &loaded.root_pool, Some(flag))
                    .await?,
            );
        }
        let result = Arc::new(Self {
            inputs,
            rows: rows.into_boxed_slice(),
            reservation,
        });
        result.require_current(sessions)?;
        Ok(Some(result))
    }

    pub(crate) fn require_current(&self, sessions: &[&ShortHfSession]) -> Result<()> {
        self.inputs.require_current(sessions)?;
        if self.rows.len() != sessions.len() {
            return Err(Error::State("HF cohort current state row count".into()));
        }
        let mut prior = BTreeSet::new();
        for session in sessions {
            for lease in session
                .original
                .domains
                .iter()
                .flat_map(|domain| &domain.leases)
            {
                if !prior.insert(lease.id()) {
                    return Err(Error::State(
                        "HF cohort rows share mutable prior storage".into(),
                    ));
                }
            }
        }
        let mut candidates = BTreeSet::new();
        for ((state, session), input) in self.rows.iter().zip(sessions).zip(self.inputs.rows()) {
            if !Arc::ptr_eq(state.input(), input) {
                return Err(Error::State("HF cohort state/source row reordered".into()));
            }
            state.require_current(&session.original)?;
            for lease in state.outputs() {
                if prior.contains(&lease.id()) || !candidates.insert(lease.id()) {
                    return Err(Error::State(
                        "HF cohort candidate aliases a current/sibling root".into(),
                    ));
                }
            }
        }
        if prior.len() != 48 * sessions.len() || candidates.len() != prior.len() {
            return Err(Error::State(
                "HF cohort exact48 planes per independent root".into(),
            ));
        }
        Ok(())
    }

    /// Call under the original fair-ticket/workspace guard before any binding.
    pub(crate) fn claim_submission(&self, sessions: &[&ShortHfSession]) -> Result<()> {
        self.require_current(sessions)?;
        for state in &self.rows {
            state.claim_submission()?;
        }
        Ok(())
    }
    pub(crate) fn rows(&self) -> &[Arc<ShortHfCallState>] {
        &self.rows
    }
    pub(crate) fn into_reservation(self) -> DecodeCohortReservation {
        self.reservation
    }
}
