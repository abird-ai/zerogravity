//! One real current-root canonical call; no new authority or aggregate attempt.
use super::*;
use crate::vllm_cuda_qualification::canonical47_model::{
    Canonical47CallKind, Canonical47ModelBinding,
};
use zerogravity_core::accelerator::program::Vllm08Call;

fn require_coordinates(base: RootBinding, attempt: &RootAttempt, call: &Vllm08Call) -> Result<()> {
    let kind = Canonical47CallKind::of(call.request())?;
    let (start, rows) = match kind {
        Canonical47CallKind::Prefill => (0, 47),
        Canonical47CallKind::Decode { base } => (base, 1),
    };
    if attempt.base() != base
        || call.source() != base.source()
        || base.frontier() != start as u64
        || attempt.frontier() != (start + rows) as u64
    {
        return Err(Error::State(
            "canonical published call differs from original current RootAttempt".into(),
        ));
    }
    Ok(())
}

impl GdnAttemptSchedule {
    pub(in crate::resident_inference) fn new_canonical47_published(
        scope: &Arc<Canonical47ModelBinding>,
        attempt: RootAttempt,
    ) -> Result<Arc<Self>> {
        let call = scope.record().call();
        scope.source().source().require_vllm08_full_source()?;
        if scope.source().source().source_identity() != call.source() {
            return Err(Error::State(
                "canonical published original source issuer".into(),
            ));
        }
        require_coordinates(attempt.base(), &attempt, call)?;
        Ok(Arc::new(Self {
            base: attempt.base(),
            attempt,
            invocations: vec![call.request().clone()],
            completed_layers: 24,
        }))
    }
}

impl AttemptPhaseBinding {
    fn require_canonical47_published_shape(&self, call: &Vllm08Call) -> Result<()> {
        if self.phase_index != 0
            || self.prior.is_some()
            || self.schedule.completed_layers != 24
            || self.schedule.invocations.len() != 1
            || self.schedule.invocation(0)? != call.request()
        {
            return Err(Error::State(
                "canonical published exact one-call schedule".into(),
            ));
        }
        require_coordinates(self.schedule.base, &self.schedule.attempt, call)
    }

    pub(in crate::resident_inference) fn require_canonical47_published_binding(
        &self,
        session: &ResidentSession,
        attempt: &RootAttempt,
        call: &Vllm08Call,
    ) -> Result<()> {
        self.require_canonical47_published_shape(call)?;
        self.require_binding(session, attempt, call.request())
    }

    pub(crate) fn require_canonical47_published_permit(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        service: &Arc<ResidentService>,
        call: &Vllm08Call,
    ) -> Result<()> {
        self.require_canonical47_published_shape(call)?;
        self.require_permit(permit, service, call.request())
    }

    pub(in crate::resident_inference) fn same_canonical47_published_request(
        &self,
        other: &Self,
    ) -> bool {
        Arc::ptr_eq(&self.schedule, &other.schedule)
            && self.phase_index == 0
            && other.phase_index == 0
            && self.prior.is_none()
            && other.prior.is_none()
            && self.schedule.completed_layers == 24
            && self.schedule.invocations.len() == 1
    }

    pub(in crate::resident_inference) fn require_retained_canonical47_published(
        &self,
        base: RootBinding,
        call: &Vllm08Call,
    ) -> Result<()> {
        self.require_canonical47_published_shape(call)?;
        if self.schedule.base != base {
            return Err(Error::State(
                "canonical retained actual root differs".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::{
        SourceIdentity,
        program::{GdnPrefixLayer, VllmBf16Mode, VllmBf16Teachers},
    };
    use zerogravity_core::session_root::SessionAuthority;

    fn decode(source: SourceIdentity, base: usize) -> Vllm08Call {
        Vllm08Call::from_span(
            VllmBf16Invocation::from_span(
                GdnPrefixLayer::new(source, 0).unwrap(),
                VllmBf16Mode::Decode,
                VllmBf16Teachers::new(source, base, 512, &[11]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn actual_advance_covers_each_finite_cached_frontier_without_issuing_a_root() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut authority = SessionAuthority::new(source).unwrap();
        let initial = authority.begin_advance(47).unwrap();
        authority.prepare_publish(initial).unwrap().commit();
        for base in 47..=53 {
            let before = authority.binding();
            let attempt = authority.begin_advance(1).unwrap();
            require_coordinates(before, &attempt, &decode(source, base)).unwrap();
            assert_eq!(authority.binding(), before);
            authority.prepare_publish(attempt).unwrap().commit();
        }
        assert_eq!(authority.binding().frontier(), 54);
    }

    #[test]
    fn foreign_attempt_wrong_advance_and_outside_span_are_refused() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut a = SessionAuthority::new(source).unwrap();
        let mut b = SessionAuthority::new(source).unwrap();
        for authority in [&mut a, &mut b] {
            let attempt = authority.begin_advance(47).unwrap();
            authority.prepare_publish(attempt).unwrap().commit();
        }
        assert!(
            require_coordinates(
                a.binding(),
                &b.begin_advance(1).unwrap(),
                &decode(source, 47)
            )
            .is_err()
        );
        assert!(
            require_coordinates(
                a.binding(),
                &a.begin_advance(2).unwrap(),
                &decode(source, 47)
            )
            .is_err()
        );
        assert!(
            require_coordinates(
                a.binding(),
                &a.begin_advance(1).unwrap(),
                &decode(source, 46)
            )
            .is_err()
        );
        assert!(
            require_coordinates(
                a.binding(),
                &a.begin_advance(1).unwrap(),
                &decode(source, 54)
            )
            .is_err()
        );
        let foreign = SourceIdentity::for_new_allocation().unwrap();
        assert!(
            require_coordinates(
                a.binding(),
                &a.begin_advance(1).unwrap(),
                &decode(foreign, 47)
            )
            .is_err()
        );
    }

    #[test]
    fn restore_revokes_old_attempt_and_allows_a_fresh_real_attempt() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut authority = SessionAuthority::new(source).unwrap();
        let initial = authority.begin_advance(47).unwrap();
        authority.prepare_publish(initial).unwrap().commit();
        let saved = authority.binding();
        let stale = authority.begin_advance(1).unwrap();
        let advance = authority.begin_advance(1).unwrap();
        authority.prepare_publish(advance).unwrap().commit();
        let restore = authority.begin_restore(saved).unwrap();
        authority.prepare_publish(restore).unwrap().commit();
        assert!(require_coordinates(authority.binding(), &stale, &decode(source, 47)).is_err());
        require_coordinates(
            authority.binding(),
            &authority.begin_advance(1).unwrap(),
            &decode(source, 47),
        )
        .unwrap();
    }
}
