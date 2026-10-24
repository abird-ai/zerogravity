//! Fresh actual short-HF candidates and a submission claim shared by all aliases.
use super::*;
use crate::resident_inference::native_fragment::model_state::BoundModelState;
use crate::resident_leases::backing_pool::{ResidentPlaneLayout, ResidentPool};

pub(super) fn root_pool_layout(model: &ResidentModel) -> Result<Vec<ResidentPlaneLayout>> {
    let layout = model.root_pool_layout()?;
    if model.states.len() != 42 || layout.len() != 48 {
        return Err(Error::State("short-HF exact42 domains/48 planes".into()));
    }
    Ok(layout)
}
pub(super) async fn allocate_pooled_state<F>(
    model: &Arc<ResidentModel>,
    pool: &Arc<ResidentPool>,
    zero: bool,
    cancelled: Option<&AtomicBool>,
    retain: F,
) -> Result<Vec<Domain>>
where
    F: FnOnce(&mut crate::resident_leases::ResidentInitialization<'_>) -> Result<()>,
{
    crate::resident_inference::state_pool::allocate_pooled_state(
        model, pool, zero, cancelled, retain,
    )
    .await
}

pub(crate) struct ShortHfCallState {
    input: Arc<ShortHfCallInput>,
    states: Vec<Arc<BoundModelState>>,
    candidate: Vec<Domain>,
    submitted: AtomicBool,
    _metadata: WindowWorkspace,
}

impl ShortHfCallState {
    pub(crate) fn input(&self) -> &Arc<ShortHfCallInput> {
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
            return Err(Error::State("short-HF24 current bound states".into()));
        }
        session.model.validate_domains(&self.candidate, false)?;
        for (layer, state) in self.states.iter().enumerate() {
            state.require_short_hf(self.input.scope())?;
            if state.call() != self.input.call()
                || state.layer() != self.input.call().layer(layer)?
                || state.base() != self.input.attempt().base()
            {
                return Err(Error::State(
                    "short-HF state source/layer/current-root differs".into(),
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
            .map_err(|_| Error::State("short-HF candidate already submitted".into()))
    }
    /// Derive this row's runtime bindings from its actual current session and
    /// private candidates. Callers cannot substitute a sibling's StateSlot map.
    pub(crate) fn runtime_bindings(
        &self,
        session: &ResidentSession,
    ) -> Result<Vec<crate::resident_program::StateBinding>> {
        self.require_current(session)?;
        self.require_submission_claimed()?;
        crate::resident_inference::bind_domains(
            &session.domains,
            &self.candidate,
            self.input.base(),
        )
    }
    pub(crate) fn require_submission_claimed(&self) -> Result<()> {
        if self.submitted.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(Error::State(
                "short-HF body lacks consumed candidate claim".into(),
            ))
        }
    }
}

impl ResidentSession {
    pub(crate) async fn bind_fresh_short_hf_state(
        &self,
        input: Arc<ShortHfCallInput>,
        pool: &Arc<ResidentPool>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<ShortHfCallState>> {
        not_cancelled(cancelled)?;
        input.require_session(self)?;
        let metadata = input
            .source()
            .source()
            .source()
            .reserve_workspace(256 * 1024)?;
        let candidate =
            allocate_pooled_state(&self.model, pool, false, cancelled, |_| Ok(())).await?;
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
                    "short-HF candidate must be fresh, uninitialized and independent".into(),
                ));
            }
        }
        if ids.len() != 48 {
            return Err(Error::State("short-HF exact48 candidate leases".into()));
        }
        let mut states = Vec::with_capacity(24);
        for layer in 0..24 {
            states.push(self.bind_short_hf_state(
                input.attempt(),
                input.scope(),
                input.call().layer(layer)?,
                &candidate,
                input.phase().clone(),
            )?);
        }
        let result = Arc::new(ShortHfCallState {
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
