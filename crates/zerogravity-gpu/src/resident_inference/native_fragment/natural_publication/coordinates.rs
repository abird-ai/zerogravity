//! Closed original natural request coordinates; no source or Root issuer.
use zerogravity_core::accelerator::program::{Vllm08Call, VllmBf16Phase};
use zerogravity_core::session_root::{RootAttempt, RootBinding};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NaturalSlot {
    InitialOne,
    FeedbackAtOne,
    Prefix64,
    FinalTwo,
    FeedbackAt66,
}
impl NaturalSlot {
    pub(crate) const ALL: [Self; 5] = [
        Self::InitialOne,
        Self::FeedbackAtOne,
        Self::Prefix64,
        Self::FinalTwo,
        Self::FeedbackAt66,
    ];
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::InitialOne => 0,
            Self::FeedbackAtOne => 1,
            Self::Prefix64 => 2,
            Self::FinalTwo => 3,
            Self::FeedbackAt66 => 4,
        }
    }
    pub(crate) const fn phase(self) -> VllmBf16Phase {
        match self {
            Self::InitialOne => VllmBf16Phase::InitialOne,
            Self::FeedbackAtOne | Self::FeedbackAt66 => VllmBf16Phase::CachedOne,
            Self::Prefix64 => VllmBf16Phase::Initial64,
            Self::FinalTwo => VllmBf16Phase::ContinuedTwo,
        }
    }
    pub(crate) const fn base(self) -> usize {
        match self {
            Self::InitialOne | Self::Prefix64 => 0,
            Self::FeedbackAtOne => 1,
            Self::FinalTwo => 64,
            Self::FeedbackAt66 => 66,
        }
    }
    pub(crate) const fn rows(self) -> usize {
        match self {
            Self::Prefix64 => 64,
            Self::FinalTwo => 2,
            _ => 1,
        }
    }
    pub(crate) const fn feedback(self) -> bool {
        matches!(self, Self::FeedbackAtOne | Self::FeedbackAt66)
    }
    pub(crate) const fn emits(self) -> bool {
        !matches!(self, Self::Prefix64)
    }
    pub(crate) fn require_current(
        self,
        current: RootBinding,
        attempt: &RootAttempt,
        call: &Vllm08Call,
    ) -> Result<()> {
        call.validate()?;
        if current != attempt.base()
            || current.source() != call.source()
            || current.frontier() != self.base() as u64
            || call.request().legacy_phase()? != self.phase()
            || call.request().teachers().base() != self.base()
            || call.rows() != self.rows()
            || attempt.frontier() != (self.base() + self.rows()) as u64
        {
            return Err(Error::State(
                "natural full24 original current-root/call span differs or was revoked".into(),
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
        program::{GdnPrefixLayer, VllmBf16Invocation, VllmBf16Teachers},
    };
    use zerogravity_core::session_root::SessionAuthority;
    fn call(slot: NaturalSlot, source: SourceIdentity) -> Vllm08Call {
        Vllm08Call::new(
            VllmBf16Invocation::new(
                GdnPrefixLayer::new(source, 0).unwrap(),
                slot.phase(),
                VllmBf16Teachers::new(source, slot.base(), 512, &vec![11; slot.rows()]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn exact_original_roots_progress_by_one_transaction_and_real_teacher_span() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        for slots in [
            [NaturalSlot::InitialOne, NaturalSlot::FeedbackAtOne].as_slice(),
            [
                NaturalSlot::Prefix64,
                NaturalSlot::FinalTwo,
                NaturalSlot::FeedbackAt66,
            ]
            .as_slice(),
        ] {
            let mut authority = SessionAuthority::new(source).unwrap();
            for (ordinal, &slot) in slots.iter().enumerate() {
                let attempt = authority.begin_advance(slot.rows() as u64).unwrap();
                slot.require_current(authority.binding(), &attempt, &call(slot, source))
                    .unwrap();
                let before = authority.binding();
                let after = authority.prepare_publish(attempt).unwrap().commit();
                assert_eq!(after.root(), before.root() + 1);
                assert_eq!(after.frontier(), (slot.base() + slot.rows()) as u64);
                assert_eq!(after.root(), ordinal as u64 + 1);
            }
        }
    }
    #[test]
    fn cancelled_or_uncommitted_attempt_never_advances_and_competing_commit_revokes_it() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut authority = SessionAuthority::new(source).unwrap();
        let before = authority.binding();
        drop(authority.begin_advance(1).unwrap());
        assert_eq!(authority.binding(), before);
        let stale = authority.begin_advance(1).unwrap();
        let winner = authority.begin_advance(1).unwrap();
        authority.prepare_publish(winner).unwrap().commit();
        assert!(
            NaturalSlot::InitialOne
                .require_current(
                    authority.binding(),
                    &stale,
                    &call(NaturalSlot::InitialOne, source)
                )
                .is_err()
        );
        assert!(authority.prepare_publish(stale).is_err());
    }
    #[test]
    fn restore_revokes_old_attempt_even_when_frontier_returns_to_same_coordinate() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut authority = SessionAuthority::new(source).unwrap();
        let initial = authority.binding();
        let old = authority.begin_advance(1).unwrap();
        let advance = authority.begin_advance(1).unwrap();
        authority.prepare_publish(advance).unwrap().commit();
        let restore = authority.begin_restore(initial).unwrap();
        authority.prepare_publish(restore).unwrap().commit();
        assert_eq!(authority.binding().frontier(), 0);
        assert_ne!(authority.binding(), initial);
        assert!(
            NaturalSlot::InitialOne
                .require_current(
                    authority.binding(),
                    &old,
                    &call(NaturalSlot::InitialOne, source)
                )
                .is_err()
        );
        let fresh = authority.begin_advance(1).unwrap();
        NaturalSlot::InitialOne
            .require_current(
                authority.binding(),
                &fresh,
                &call(NaturalSlot::InitialOne, source),
            )
            .unwrap();
    }
    #[test]
    fn source_session_span_and_emission_boundaries_are_not_interchangeable() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let a = SessionAuthority::new(source).unwrap();
        let b = SessionAuthority::new(source).unwrap();
        let attempt = a.begin_advance(1).unwrap();
        assert!(
            NaturalSlot::InitialOne
                .require_current(
                    b.binding(),
                    &attempt,
                    &call(NaturalSlot::InitialOne, source)
                )
                .is_err()
        );
        let foreign = SourceIdentity::for_new_allocation().unwrap();
        assert!(
            NaturalSlot::InitialOne
                .require_current(
                    a.binding(),
                    &attempt,
                    &call(NaturalSlot::InitialOne, foreign)
                )
                .is_err()
        );
        assert!(
            NaturalSlot::Prefix64
                .require_current(
                    a.binding(),
                    &attempt,
                    &call(NaturalSlot::InitialOne, source)
                )
                .is_err()
        );
        assert_eq!(
            NaturalSlot::ALL.map(NaturalSlot::emits),
            [true, true, false, true, true]
        );
        assert_eq!(
            NaturalSlot::ALL.map(NaturalSlot::feedback),
            [false, true, false, false, true]
        );
        assert_eq!(NaturalSlot::ALL.map(NaturalSlot::index), [0, 1, 2, 3, 4]);
    }
}
