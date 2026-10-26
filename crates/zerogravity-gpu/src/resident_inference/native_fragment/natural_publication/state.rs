//! Five preallocated canonical roots, bound only to original current attempts.
use super::*;
use crate::resident_inference::native_fragment::model_state::BoundModelState;
mod reuse;
use reuse::ReusableState;

pub struct NaturalStatePool {
    model: Weak<ResidentModel>,
    service: Arc<ResidentService>,
    candidates: Vec<Vec<Domain>>,
    claims: [Arc<OnceLock<Arc<NaturalCallInput>>>; 5],
    _metadata: WindowWorkspace,
}
impl NaturalStatePool {
    /// Adopt the actual cold-prepared canonical storage. No managed allocation,
    /// upload, module creation or initialization occurs after native activation.
    pub(in crate::resident_inference) fn from_prepared(
        source: &VerifiedTensorSource,
        model: &Arc<ResidentModel>,
        candidates: Vec<Vec<Domain>>,
    ) -> Result<Arc<Self>> {
        if source.source_identity() != model.service.source_identity()
            || model.capacity != 512
            || model.vocab != 248320
            || model.states.len() != 42
        {
            return Err(Error::Unsupported(
                "natural original full24 canonical state pool".into(),
            ));
        }
        let metadata = source.reserve_workspace(256 * 1024)?;
        if candidates.len() != 5 || candidates.capacity() > 5 {
            return Err(Error::State(
                "natural pool requires exactly five cold canonical roots".into(),
            ));
        }
        let mut ids = std::collections::BTreeSet::new();
        for domains in &candidates {
            model.validate_domains(domains, false)?;
            if domains.iter().map(|d| d.leases.len()).sum::<usize>() != 48 {
                return Err(Error::State(
                    "natural exact48 physical canonical buffers".into(),
                ));
            }
            for lease in domains.iter().flat_map(|d| &d.leases) {
                if lease.role() != BufferRole::RecurrentState
                    || lease.is_initialized()
                    || !ids.insert(lease.id())
                {
                    return Err(Error::State("natural cold candidates must be distinct uninitialized original state leases".into()));
                }
            }
        }
        Ok(Arc::new(Self {
            model: Arc::downgrade(model),
            service: model.service.clone(),
            candidates,
            claims: std::array::from_fn(|_| Arc::new(OnceLock::new())),
            _metadata: metadata,
        }))
    }
}
pub(crate) struct NaturalCallState {
    input: Arc<NaturalCallInput>,
    states: Vec<Arc<BoundModelState>>,
    candidate: Vec<Domain>,
    claim: Arc<OnceLock<Arc<NaturalCallInput>>>,
    _reuse: Option<ReusableState>,
}
impl NaturalCallState {
    pub(crate) fn input(&self) -> &Arc<NaturalCallInput> {
        &self.input
    }
    pub(crate) fn states(&self) -> &[Arc<BoundModelState>] {
        &self.states
    }
    pub(crate) fn require_current(&self, session: &ResidentSession) -> Result<()> {
        self.input.require_session(session)?;
        if !self
            .claim
            .get()
            .is_some_and(|input| Arc::ptr_eq(input, &self.input))
            || self.states.len() != 24
        {
            return Err(Error::State(
                "natural candidate slot lacks exact original one-call claim".into(),
            ));
        }
        session.model.validate_domains(&self.candidate, false)?;
        for (layer, state) in self.states.iter().enumerate() {
            if state.call() != self.input.call()
                || state.layer() != layer
                || state.base() != self.input.attempt().base()
            {
                return Err(Error::State(
                    "natural actual state layer/source/current-root differs".into(),
                ));
            }
        }
        Ok(())
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
}
impl ResidentSession {
    pub(crate) fn bind_natural_state(
        &self,
        input: Arc<NaturalCallInput>,
        pool: &NaturalStatePool,
    ) -> Result<Arc<NaturalCallState>> {
        input.require_session(self)?;
        same_model(&pool.model, self)?;
        if !Arc::ptr_eq(&pool.service, &self.model.service) {
            return Err(Error::State(
                "natural canonical pool service differs".into(),
            ));
        }
        let index = input.slot().index();
        if pool.claims[index].get().is_some() {
            return Err(Error::State(
                "natural canonical slot already claimed".into(),
            ));
        }
        let candidate = pool.candidates[index].clone();
        let mut states = Vec::with_capacity(24);
        for layer in 0..24 {
            states.push(self.bind_full_model_state(
                input.attempt(),
                input.record(),
                layer,
                &candidate,
                input.phase_binding().clone(),
            )?);
        }
        pool.claims[index]
            .set(input.clone())
            .map_err(|_| Error::State("natural competing canonical slot claim".into()))?;
        let state = Arc::new(NaturalCallState {
            input,
            states,
            candidate,
            claim: pool.claims[index].clone(),
            _reuse: None,
        });
        state.require_current(self)?;
        Ok(state)
    }
}
