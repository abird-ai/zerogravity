//! Private physically completed prefix versions inside ONE unpublished attempt.
//! No root binding, event, Ready or source identity is issued here.
use super::*;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode};

mod component;
mod fixed_prompt_model_prefix;
mod fixed_prompt_published;
mod fixed_prompt_whole;
mod published_call;
mod text;

/// Immutable request spans only; the original session authority stays at base.
pub(super) struct GdnAttemptSchedule {
    base: RootBinding,
    attempt: RootAttempt,
    invocations: Vec<SourceCall>,
    completed_layers: usize,
}
impl GdnAttemptSchedule {
    /// Source calls come from model-authenticated prepared plans. This schedule
    /// checks physical coordinates; it grants no model/source eligibility.
    /// Distinct per-call publication path: ONE original current RootAttempt,
    /// never the existing unpublished aggregate forced-teacher schedule.
    pub(super) fn new_published_call(
        source: &VerifiedTensorSource,
        attempt: RootAttempt,
        call: SourceCall,
    ) -> Result<Arc<Self>> {
        let base = attempt.base();
        if base.source() != source.source_identity() {
            return Err(Error::State(
                "natural published call source issuer differs".into(),
            ));
        }
        published_call::require_published_call(base, &attempt, &call)?;
        Ok(Arc::new(Self {
            base,
            attempt,
            invocations: vec![call],
            completed_layers: 24,
        }))
    }
    pub(super) fn new(
        base: RootBinding,
        attempt: RootAttempt,
        invocations: Vec<SourceCall>,
        completed_layers: usize,
    ) -> Result<Arc<Self>> {
        if !matches!(completed_layers, 1 | 3) {
            return Err(Error::Unsupported(
                "original prefix remains exactly1 or3 layers".into(),
            ));
        }
        Self::new_inner(base, attempt, invocations, completed_layers)
    }
    /// Distinct full24 source scope; the existing prefix constructor is unchanged.
    /// Original SessionAuthority/RootAttempt ownership and physical Ready protocol
    /// are reused. Neither this schedule nor its completed phases publish a root.
    pub(super) fn new_model(
        source: &VerifiedTensorSource,
        base: RootBinding,
        attempt: RootAttempt,
        calls: Vec<SourceCall>,
    ) -> Result<Arc<Self>> {
        if calls.capacity() > 3 || base.source() != source.source_identity() {
            return Err(Error::Unsupported(
                "closed full24 original source/request scope".into(),
            ));
        }
        for call in &calls {
            if call.source() != source.source_identity() {
                return Err(Error::State(
                    "full24 original request issuer differs".into(),
                ));
            }
        }
        let invocations = calls;
        Self::new_inner(base, attempt, invocations, 24)
    }
    fn new_inner(
        base: RootBinding,
        attempt: RootAttempt,
        invocations: Vec<SourceCall>,
        completed_layers: usize,
    ) -> Result<Arc<Self>> {
        if base.frontier() != 0 || attempt.base() != base || invocations.capacity() > 3 {
            return Err(Error::Unsupported(
                "original initial-prefix attempt/scope/request capacity".into(),
            ));
        }
        let expected: &[(SourceCallMode, usize, usize)] = match invocations.as_slice() {
            [_, _] => &[
                (SourceCallMode::Prefill, 0, 1),
                (SourceCallMode::Decode, 1, 1),
            ],
            [_, _, _] => &[
                (SourceCallMode::Prefill, 0, 64),
                (SourceCallMode::Prefill, 64, 2),
                (SourceCallMode::Decode, 66, 1),
            ],
            _ => {
                return Err(Error::Unsupported(
                    "closed two original request phase lists".into(),
                ));
            }
        };
        for (invocation, &(mode, start, rows)) in invocations.iter().zip(expected) {
            if invocation.source() != base.source()
                || invocation.mode() != mode
                || invocation.rows().len() != rows
                || invocation.rows().base() != start
                || invocation.rows().capacity() != 512
                || invocation.rows().vocabulary() != 248320
            {
                return Err(Error::State(
                    "exact original teacher/source/phase request schedule".into(),
                ));
            }
        }
        let last = invocations.last().unwrap();
        let end = last
            .rows()
            .base()
            .checked_add(last.rows().len())
            .ok_or_else(|| Error::Limit("prefix transaction span overflow".into()))?;
        if attempt.frontier() != end as u64 {
            return Err(Error::State(
                "original root attempt must cover exactly the complete request span".into(),
            ));
        }
        Ok(Arc::new(Self {
            base,
            attempt,
            invocations,
            completed_layers,
        }))
    }
    pub(super) fn invocation(&self, index: usize) -> Result<&SourceCall> {
        self.invocations
            .get(index)
            .ok_or_else(|| Error::State("prefix phase index out of request".into()))
    }
    fn require_current(&self, current: RootBinding) -> Result<()> {
        if current != self.base || self.attempt.base() != self.base {
            return Err(Error::State(
                "original session/root changed during unpublished prefix request".into(),
            ));
        }
        Ok(())
    }
    fn require_session(&self, session: &ResidentSession) -> Result<()> {
        self.require_current(session.binding())?;
        if session.model.service.source_identity() != self.base.source() {
            return Err(Error::State(
                "original prefix physical source differs".into(),
            ));
        }
        session.model.service.ensure_usable()
    }
}

/// Real physical outputs, not a session root/snapshot or full-model frontier.
/// Candidate payloads drop before their genuine completion/source owners.
enum CompletedBody {
    Prefix {
        _body: Arc<crate::resident_program::native::NativeProgram>,
    },
    Full {
        _body: Arc<crate::resident_program::native_full::FullModelProgram>,
    },
}
pub(super) struct CompletedPrefixPhase {
    domains: Vec<Domain>,
    ready: ResidentReady,
    _body: CompletedBody,
    _prior: Option<Arc<CompletedPrefixPhase>>,
    schedule: Arc<GdnAttemptSchedule>,
    phase_index: usize,
}
impl CompletedPrefixPhase {
    pub(super) fn ready(&self) -> &ResidentReady {
        &self.ready
    }
    pub(super) fn completed_phases(&self) -> usize {
        self.phase_index + 1
    }
    /// Invoke only with the actual trusted complete_with_readback result.
    /// There is deliberately no constructor from raw bytes/captured state.
    pub(super) fn after_completion(
        session: &ResidentSession,
        schedule: Arc<GdnAttemptSchedule>,
        phase_index: usize,
        domains: Vec<Domain>,
        ready: ResidentReady,
        body: Arc<crate::resident_program::native::NativeProgram>,
        prior: Option<Arc<Self>>,
    ) -> Result<Arc<Self>> {
        schedule.require_session(session)?;
        let invocation = schedule.invocation(phase_index)?;
        if body.invocation_for_layer(0)? != *invocation
            || body.completed_layers() != schedule.completed_layers
        {
            return Err(Error::State(
                "actual prepared prefix invocation differs from phase".into(),
            ));
        }
        Self::after_prepared_completion(
            session,
            schedule,
            phase_index,
            domains,
            ready,
            CompletedBody::Prefix { _body: body },
            prior,
        )
    }
    pub(super) fn after_full_completion(
        session: &ResidentSession,
        schedule: Arc<GdnAttemptSchedule>,
        phase_index: usize,
        domains: Vec<Domain>,
        ready: ResidentReady,
        body: Arc<crate::resident_program::native_full::FullModelProgram>,
        prior: Option<Arc<Self>>,
    ) -> Result<Arc<Self>> {
        schedule.require_session(session)?;
        if schedule.completed_layers != 24
            || body.record().call() != schedule.invocation(phase_index)?
        {
            return Err(Error::State(
                "full24 actual prepared body differs from original completed request phase".into(),
            ));
        }
        Self::after_prepared_completion(
            session,
            schedule,
            phase_index,
            domains,
            ready,
            CompletedBody::Full { _body: body },
            prior,
        )
    }
    fn after_prepared_completion(
        session: &ResidentSession,
        schedule: Arc<GdnAttemptSchedule>,
        phase_index: usize,
        domains: Vec<Domain>,
        ready: ResidentReady,
        body: CompletedBody,
        prior: Option<Arc<Self>>,
    ) -> Result<Arc<Self>> {
        schedule.require_session(session)?;
        let invocation = schedule.invocation(phase_index)?;
        match (phase_index, &prior) {
            (0, None) => {}
            (0, Some(_)) | (_, None) => {
                return Err(Error::State(
                    "exact previous physically completed prefix required".into(),
                ));
            }
            (_, Some(prior)) => {
                prior.require_next(
                    session,
                    &schedule,
                    phase_index,
                    invocation,
                    schedule.completed_layers,
                )?;
            }
        }
        session.model.validate_domains(&domains, true)?;
        if !Arc::ptr_eq(ready.service(), &session.model.service)
            || ready.expected_root() != schedule.base.root()
            || ready.root_binding().is_some()
            || ready.event_ids().is_empty()
        {
            return Err(Error::State(
                "prefix continuation requires genuine same-service unbound physical completion"
                    .into(),
            ));
        }
        let outputs = ready.outputs();
        for lease in domains.iter().flat_map(|domain| &domain.leases) {
            if !outputs.iter().any(|output| {
                output.id() == lease.id()
                    && output.allocation_generation() == lease.allocation_generation()
                    && output.service_generation() == lease.service_generation()
                    && same_handle(&output.handle(), &lease.handle())
            }) {
                return Err(Error::State(
                    "actual prefix candidate absent from original physical Ready outputs".into(),
                ));
            }
        }
        Ok(Arc::new(Self {
            domains,
            ready,
            _body: body,
            _prior: prior,
            schedule,
            phase_index,
        }))
    }
    pub(super) fn require_next<'a>(
        &'a self,
        session: &ResidentSession,
        schedule: &Arc<GdnAttemptSchedule>,
        index: usize,
        invocation: &SourceCall,
        completed_layers: usize,
    ) -> Result<&'a [Domain]> {
        schedule.require_session(session)?;
        if !Arc::ptr_eq(schedule, &self.schedule)
            || self.phase_index.checked_add(1) != Some(index)
            || schedule.invocation(index)? != invocation
            || completed_layers != schedule.completed_layers
            || !Arc::ptr_eq(self.ready.service(), &session.model.service)
            || self.ready.expected_root() != schedule.base.root()
        {
            return Err(Error::State(
                "exact previous actual prefix phase/source/teacher/scope owner required".into(),
            ));
        }
        session.model.validate_domains(&self.domains, true)?;
        Ok(&self.domains)
    }
}

/// Exact future input ownership only. Preparation may inspect these genuine
/// leases while uninitialized; first enqueue requires the actual prior Ready.
pub(crate) struct PlannedPrefixPrior {
    domains: Vec<Domain>,
    completed: std::sync::OnceLock<Arc<CompletedPrefixPhase>>,
    schedule: Arc<GdnAttemptSchedule>,
    phase_index: usize,
    service_generation: u64,
}
impl PlannedPrefixPrior {
    pub(super) fn new(
        session: &ResidentSession,
        schedule: Arc<GdnAttemptSchedule>,
        phase_index: usize,
        domains: Vec<Domain>,
    ) -> Result<Arc<Self>> {
        schedule.require_session(session)?;
        schedule.invocation(phase_index)?;
        if phase_index == 0 {
            return Err(Error::State(
                "initial phase uses the original initialized session root".into(),
            ));
        }
        session.model.validate_domains(&domains, false)?;
        // These are the preceding phase's original private candidates, never
        // original committed-root storage or a sibling partially cropped view.
        let initial = session.domains.iter().flat_map(|domain| &domain.leases);
        let initial_ids = initial.map(ResidentLease::id).collect::<BTreeSet<_>>();
        if domains
            .iter()
            .flat_map(|domain| &domain.leases)
            .any(|lease| initial_ids.contains(&lease.id()))
        {
            return Err(Error::State(
                "planned continuation aliases original committed root".into(),
            ));
        }
        Ok(Arc::new(Self {
            domains,
            completed: std::sync::OnceLock::new(),
            schedule,
            phase_index,
            service_generation: session.model.service.generation(),
        }))
    }
    pub(super) fn bind_actual_completion(
        &self,
        session: &ResidentSession,
        prior: Arc<CompletedPrefixPhase>,
    ) -> Result<()> {
        let actual = prior.require_next(
            session,
            &self.schedule,
            self.phase_index,
            self.schedule.invocation(self.phase_index)?,
            self.schedule.completed_layers,
        )?;
        if actual.len() != self.domains.len()
            || actual.iter().zip(&self.domains).any(|(a, b)| {
                a.slot != b.slot
                    || a.generations != b.generations
                    || a.leases.len() != b.leases.len()
                    || a.leases
                        .iter()
                        .zip(&b.leases)
                        .any(|(a, b)| a.id() != b.id() || !same_handle(&a.handle(), &b.handle()))
            })
        {
            return Err(Error::State(
                "actual prior completion differs from preallocated canonical candidates".into(),
            ));
        }
        self.completed
            .set(prior)
            .map_err(|_| Error::State("prefix prior completion already bound".into()))
    }
    pub(crate) fn require_prepared_lease(&self, lease: &ResidentLease) -> Result<()> {
        let present = self
            .domains
            .iter()
            .flat_map(|domain| &domain.leases)
            .any(|planned| {
                planned.id() == lease.id()
                    && planned.allocation_generation() == lease.allocation_generation()
                    && planned.service_generation() == lease.service_generation()
                    && same_handle(&planned.handle(), &lease.handle())
            });
        if !present {
            return Err(Error::State(
                "uninitialized adapter prior lacks exact planned producer lease".into(),
            ));
        }
        Ok(())
    }
    /// Called before ANY adapter transport, not after enqueue or readback.
    pub(crate) fn require_launch_ready(&self, service: &Arc<ResidentService>) -> Result<()> {
        service.ensure_usable()?;
        let completed = self.completed.get().ok_or_else(|| {
            Error::State("planned prefix prior has no genuine completed producer".into())
        })?;
        if !Arc::ptr_eq(completed.ready.service(), service)
            || completed.ready.expected_root() != self.schedule.base.root()
            || service.generation() != self.service_generation
            || self
                .domains
                .iter()
                .flat_map(|domain| &domain.leases)
                .any(|lease| !lease.is_initialized())
        {
            return Err(Error::State(
                "prefix prior physical service/root/initialization differs".into(),
            ));
        }
        Ok(())
    }
}

/// Borrows the original session through enqueue/completion; restore/reset need
/// &mut ResidentSession and cannot intervene after this fresh authority check.
/// This is neither a root/Ready nor a persistent completion owner.
pub(crate) struct PrefixPhaseEnqueue<'a> {
    session: &'a ResidentSession,
    schedule: &'a Arc<GdnAttemptSchedule>,
    previous: Option<&'a CompletedPrefixPhase>,
    phase_index: usize,
    service_generation: u64,
}

/// Preparation binding only; no permission to read a future unready candidate.
pub(in crate::resident_inference) struct AttemptPhaseBinding {
    schedule: Arc<GdnAttemptSchedule>,
    phase_index: usize,
    prior: Option<Arc<PlannedPrefixPrior>>,
}
impl AttemptPhaseBinding {
    pub(super) fn new(
        schedule: Arc<GdnAttemptSchedule>,
        phase_index: usize,
        prior: Option<Arc<PlannedPrefixPrior>>,
    ) -> Result<Arc<Self>> {
        schedule.invocation(phase_index)?;
        match (phase_index, &prior) {
            (0, None) => {}
            (0, Some(_)) | (_, None) => {
                return Err(Error::State(
                    "exact phase preparation predecessor required".into(),
                ));
            }
            (_, Some(prior))
                if prior.phase_index == phase_index && Arc::ptr_eq(&prior.schedule, &schedule) => {}
            _ => return Err(Error::State("foreign planned prefix predecessor".into())),
        }
        Ok(Arc::new(Self {
            schedule,
            phase_index,
            prior,
        }))
    }
    pub(in crate::resident_inference) fn require_binding(
        &self,
        session: &ResidentSession,
        attempt: &RootAttempt,
        invocation: &SourceCall,
        layer: usize,
    ) -> Result<()> {
        self.schedule.require_session(session)?;
        let expected = self.schedule.invocation(self.phase_index)?;
        if attempt.base() != self.schedule.attempt.base()
            || attempt.frontier() != self.schedule.attempt.frontier()
            || layer >= self.schedule.completed_layers
            || invocation != expected
        {
            return Err(Error::State(
                "prepared phase differs from original request attempt/source/span".into(),
            ));
        }
        Ok(())
    }
    pub(in crate::resident_inference) fn require_full_model_binding(
        &self,
        session: &ResidentSession,
        attempt: &RootAttempt,
        call: &SourceCall,
    ) -> Result<()> {
        if self.schedule.completed_layers != 24 {
            return Err(Error::Unsupported(
                "original prefix permit cannot bind full24 state".into(),
            ));
        }
        self.require_binding(session, attempt, call, 0)
    }
    pub(crate) fn require_full_model_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
        call: &SourceCall,
    ) -> Result<()> {
        if self.schedule.completed_layers != 24 {
            return Err(Error::Unsupported(
                "original prefix permit cannot enqueue full24 state".into(),
            ));
        }
        self.require_permit(permit, service, call, 0)
    }
    pub(in crate::resident_inference) fn same_full_model_request(&self, other: &Self) -> bool {
        self.schedule.completed_layers == 24
            && other.schedule.completed_layers == 24
            && Arc::ptr_eq(&self.schedule, &other.schedule)
    }
    pub(in crate::resident_inference) fn prior_domains<'a>(
        &'a self,
        session: &'a ResidentSession,
    ) -> &'a [Domain] {
        self.prior
            .as_ref()
            .map_or(&session.domains, |prior| prior.domains.as_slice())
    }
    pub(in crate::resident_inference) fn prior_is_ready(&self) -> bool {
        self.prior.is_none()
    }
    pub(crate) fn require_prepared_prior(&self, lease: &ResidentLease) -> Result<()> {
        if let Some(prior) = &self.prior {
            return prior.require_prepared_lease(lease);
        }
        if !lease.is_initialized() {
            return Err(Error::State(
                "original initial root prior is not physically initialized".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require_launch_ready(&self, service: &Arc<ResidentService>) -> Result<()> {
        if let Some(prior) = &self.prior {
            prior.require_launch_ready(service)?;
        }
        Ok(())
    }
    pub(crate) fn require_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
        invocation: &SourceCall,
        layer: usize,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.schedule, permit.schedule) || self.phase_index != permit.phase_index {
            return Err(Error::State(
                "enqueue permit differs from prepared original attempt phase".into(),
            ));
        }
        permit.require_invocation(service, invocation, layer, self.schedule.completed_layers)
    }
}
impl<'a> PrefixPhaseEnqueue<'a> {
    pub(crate) fn session(&self) -> &ResidentSession {
        self.session
    }
    pub(super) fn new(
        session: &'a ResidentSession,
        schedule: &'a Arc<GdnAttemptSchedule>,
        phase_index: usize,
        previous: Option<&'a Arc<CompletedPrefixPhase>>,
    ) -> Result<Self> {
        schedule.require_session(session)?;
        let invocation = schedule.invocation(phase_index)?;
        let previous = previous.map(Arc::as_ref);
        match (phase_index, previous) {
            (0, None) => session.model.validate_domains(&session.domains, true)?,
            (0, Some(_)) | (_, None) => {
                return Err(Error::State(
                    "exact previous completed prefix phase required before enqueue".into(),
                ));
            }
            (_, Some(prior)) => {
                prior.require_next(
                    session,
                    schedule,
                    phase_index,
                    invocation,
                    schedule.completed_layers,
                )?;
            }
        }
        Ok(Self {
            session,
            schedule,
            previous,
            phase_index,
            service_generation: session.model.service.generation(),
        })
    }
    pub(crate) fn require_invocation(
        &self,
        service: &Arc<ResidentService>,
        invocation: &SourceCall,
        layer: usize,
        completed_layers: usize,
    ) -> Result<()> {
        self.schedule.require_session(self.session)?;
        let expected = self.schedule.invocation(self.phase_index)?;
        if !Arc::ptr_eq(service, &self.session.model.service)
            || service.generation() != self.service_generation
            || completed_layers != self.schedule.completed_layers
            || layer >= completed_layers
            || invocation != expected
        {
            return Err(Error::State(
                "original attempt/source/generation/teacher phase revoked or differs".into(),
            ));
        }
        if let Some(previous) = self.previous {
            previous.require_next(
                self.session,
                self.schedule,
                self.phase_index,
                expected,
                completed_layers,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "continuation/tests.rs"]
mod tests;
