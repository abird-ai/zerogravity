//! Fresh actual canonical candidates and a submission claim shared by all aliases.
use super::*;
use crate::resident_inference::native_fragment::model_state::BoundModelState;

pub(crate) struct FixedPromptCallState {
    input: Arc<FixedPromptCallInput>,
    states: Vec<Arc<BoundModelState>>,
    candidate: Vec<Domain>,
    submitted: AtomicBool,
    _metadata: WindowWorkspace,
}

impl FixedPromptCallState {
    pub(crate) fn input(&self) -> &Arc<FixedPromptCallInput> {
        &self.input
    }
    pub(crate) fn states(&self) -> &[Arc<BoundModelState>] {
        &self.states
    }
    pub(super) fn candidate(&self) -> &[Domain] {
        &self.candidate
    }
    pub(super) fn outputs(&self) -> Vec<ResidentLease> {
        self.candidate
            .iter()
            .flat_map(|d| d.leases.iter().cloned())
            .collect()
    }
    pub(crate) fn require_current(&self, session: &ResidentSession) -> Result<()> {
        self.input.require_session(session)?;
        if self.states.len() != 24 {
            return Err(Error::State("canonical24 current bound states".into()));
        }
        session.model.validate_domains(&self.candidate, false)?;
        for (layer, state) in self.states.iter().enumerate() {
            state.require_fixed_prompt_published(self.input.scope())?;
            if state.call() != self.input.call()
                || state.layer() != layer
                || state.base() != self.input.attempt().base()
            {
                return Err(Error::State(
                    "canonical state source/layer/current-root differs".into(),
                ));
            }
        }
        Ok(())
    }
    // Called under the original fair-ticket→workspace lock, before binding/enqueue.
    pub(super) fn claim_submission(&self) -> Result<()> {
        self.submitted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| Error::State("canonical candidate already submitted".into()))
    }
    pub(crate) fn require_submission_claimed(&self) -> Result<()> {
        if self.submitted.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(Error::State(
                "canonical body lacks consumed candidate claim".into(),
            ))
        }
    }
}

impl ResidentSession {
    pub(crate) async fn bind_fresh_fixed_prompt_state(
        &self,
        input: Arc<FixedPromptCallInput>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<FixedPromptCallState>> {
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        let metadata = input.source().reserve_workspace(256 * 1024)?;
        let candidate = self.model.allocate_state(false).await?;
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        self.model.validate_domains(&candidate, false)?;
        let prior = self
            .domains
            .iter()
            .flat_map(|d| &d.leases)
            .map(ResidentLease::id)
            .collect::<BTreeSet<_>>();
        let mut ids = BTreeSet::new();
        for lease in candidate.iter().flat_map(|d| &d.leases) {
            if lease.role() != BufferRole::RecurrentState
                || lease.is_initialized()
                || prior.contains(&lease.id())
                || !ids.insert(lease.id())
            {
                return Err(Error::State(
                    "canonical candidate must be fresh, uninitialized and independent".into(),
                ));
            }
        }
        if ids.len() != 48 {
            return Err(Error::State("canonical exact48 candidate leases".into()));
        }
        let mut states = Vec::with_capacity(24);
        for layer in 0..24 {
            states.push(self.bind_fixed_prompt_published_state(
                input.attempt(),
                input.scope(),
                layer,
                &candidate,
                input.phase().clone(),
            )?);
        }
        let result = Arc::new(FixedPromptCallState {
            input,
            states,
            candidate,
            submitted: AtomicBool::new(false),
            _metadata: metadata,
        });
        result.require_current(self)?;
        Ok(result)
    }
}
