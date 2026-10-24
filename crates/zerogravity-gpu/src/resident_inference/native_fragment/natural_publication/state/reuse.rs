//! Fresh private candidates; no indexed-pool claim or physical storage reset.
use super::*;
use std::collections::BTreeSet;
use zerogravity_core::device_events::BufferLeaseId;

const METADATA: usize = 256 * 1024;

struct SubmissionClaim(AtomicBool);
impl SubmissionClaim {
    fn consume(&self) -> Result<()> {
        self.0
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| Error::State("reusable natural candidate already submitted".into()))
    }
}
pub(super) struct ReusableState {
    submission: SubmissionClaim,
    _metadata: WindowWorkspace,
}
impl NaturalCallState {
    pub(crate) fn require_reusable_state(&self) -> Result<()> {
        if self._reuse.is_none() {
            return Err(Error::State(
                "reusable path cannot consume indexed closed-five candidates".into(),
            ));
        }
        Ok(())
    }
    /// Shared across every Arc clone. A failed bind/launch cannot reset it.
    pub(crate) fn claim_reusable_submission(&self) -> Result<()> {
        if let Some(reuse) = &self._reuse {
            reuse.submission.consume()?;
        }
        Ok(())
    }
    pub(crate) fn require_reusable_submission_claimed(&self) -> Result<()> {
        if !self
            ._reuse
            .as_ref()
            .is_some_and(|reuse| reuse.submission.0.load(Ordering::Acquire))
        {
            return Err(Error::State(
                "reusable body lacks consumed original candidate submission claim".into(),
            ));
        }
        Ok(())
    }
}

struct CandidateLease {
    id: BufferLeaseId,
    role: BufferRole,
    initialized: bool,
}

fn require_fresh(prior: &[BufferLeaseId], candidate: &[CandidateLease]) -> Result<()> {
    let prior = prior.iter().copied().collect::<BTreeSet<_>>();
    let mut ids = BTreeSet::new();
    if prior.len() != 48
        || candidate.len() != 48
        || candidate.iter().any(|lease| {
            lease.role != BufferRole::RecurrentState
                || lease.initialized
                || prior.contains(&lease.id)
                || !ids.insert(lease.id)
        })
    {
        return Err(Error::State(
            "reusable natural candidates require48 fresh distinct uninitialized state leases"
                .into(),
        ));
    }
    Ok(())
}

impl ResidentSession {
    /// Original allocator admits only this call's canonical candidates. No GPU
    /// command is needed until the original full-plan traversal writes them.
    pub(crate) async fn bind_fresh_natural_state(
        &self,
        input: Arc<NaturalCallInput>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<NaturalCallState>> {
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        self.model.validate_domains(&self.domains, true)?;
        let metadata = input.source().source().reserve_workspace(METADATA)?;
        let candidate = self.model.allocate_state(false).await?;
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        self.model.validate_domains(&candidate, false)?;
        let prior = self
            .domains
            .iter()
            .flat_map(|d| &d.leases)
            .map(ResidentLease::id)
            .collect::<Vec<_>>();
        let leases = candidate
            .iter()
            .flat_map(|d| &d.leases)
            .map(|lease| CandidateLease {
                id: lease.id(),
                role: lease.role(),
                initialized: lease.is_initialized(),
            })
            .collect::<Vec<_>>();
        require_fresh(&prior, &leases)?;
        let mut states = Vec::with_capacity(24);
        for layer in 0..24 {
            states.push(self.bind_full_model_state(
                input.attempt(),
                input.record(),
                input.call().layer(layer)?,
                &candidate,
                input.phase_binding().clone(),
            )?);
        }
        // New claim, bound exactly once; no pool index/claim is reset or reused.
        let claim = Arc::new(OnceLock::new());
        claim
            .set(input.clone())
            .map_err(|_| Error::State("fresh natural candidate claim".into()))?;
        let state = Arc::new(NaturalCallState {
            input,
            states,
            candidate,
            claim,
            _reuse: Some(ReusableState {
                submission: SubmissionClaim(AtomicBool::new(false)),
                _metadata: metadata,
            }),
        });
        state.require_current(self)?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::device_events::DeviceEventLedger;

    fn cohort() -> (Vec<BufferLeaseId>, Vec<CandidateLease>) {
        let mut ledger = DeviceEventLedger::new(1, 96).unwrap();
        let prior = (0..48)
            .map(|_| ledger.admit(1, BufferRole::RecurrentState).unwrap())
            .collect();
        let candidate = (0..48)
            .map(|_| CandidateLease {
                id: ledger.admit(1, BufferRole::RecurrentState).unwrap(),
                role: BufferRole::RecurrentState,
                initialized: false,
            })
            .collect();
        (prior, candidate)
    }

    #[test]
    fn clones_and_racing_submissions_consume_one_shared_candidate_claim() {
        let claim = Arc::new(SubmissionClaim(AtomicBool::new(false)));
        let threads = (0..8)
            .map(|_| {
                let claim = claim.clone();
                std::thread::spawn(move || claim.consume().is_ok())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .filter(|won| *won)
                .count(),
            1
        );
        assert!(claim.consume().is_err());
    }
    #[test]
    fn failed_work_and_no_publication_do_not_rearm_submission() {
        let claim = SubmissionClaim(AtomicBool::new(false));
        claim.consume().unwrap();
        let bind_result: Result<()> = Err(Error::State("binding refused before enqueue".into()));
        assert!(bind_result.is_err());
        assert!(claim.consume().is_err());
        let fresh = SubmissionClaim(AtomicBool::new(false));
        fresh.consume().unwrap();
        assert!(fresh.consume().is_err());
    }

    #[test]
    fn fresh_candidate_cannot_reuse_current_or_another_candidate_lease() {
        let (prior, mut candidate) = cohort();
        require_fresh(&prior, &candidate).unwrap();
        let original = candidate[1].id;
        candidate[1].id = prior[0];
        assert!(require_fresh(&prior, &candidate).is_err());
        candidate[1].id = candidate[0].id;
        assert!(require_fresh(&prior, &candidate).is_err());
        candidate[1].id = original;
        require_fresh(&prior, &candidate).unwrap();
    }

    #[test]
    fn role_initialization_and_complete_physical_cohort_are_required() {
        let (prior, mut candidate) = cohort();
        candidate[0].initialized = true;
        assert!(require_fresh(&prior, &candidate).is_err());
        candidate[0].initialized = false;
        candidate[0].role = BufferRole::Activation;
        assert!(require_fresh(&prior, &candidate).is_err());
        candidate[0].role = BufferRole::RecurrentState;
        assert!(require_fresh(&prior[..47], &candidate).is_err());
        assert!(require_fresh(&prior, &candidate[..47]).is_err());
    }
}
