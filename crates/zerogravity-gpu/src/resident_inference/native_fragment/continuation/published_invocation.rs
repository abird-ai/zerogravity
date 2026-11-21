//! Pure coordinates for ONE original published-root call, without root issuance.
use zerogravity_core::accelerator::program::{InvocationPhase, TextInvocation};
use zerogravity_core::session_root::{RootAttempt, RootBinding};
use zerogravity_core::{Error, Result};

pub(super) fn require_published_invocation(
    base: RootBinding,
    attempt: &RootAttempt,
    call: &TextInvocation,
) -> Result<()> {
    let (start, rows) = (call.tokens().base(), call.tokens().len());
    let exact = matches!(
        (call.phase(), start, rows),
        (InvocationPhase::Prefill, 0, 1)
            | (InvocationPhase::Decode, 1 | 66, 1)
            | (InvocationPhase::Prefill, 0, 64)
            | (InvocationPhase::Prefill, 64, 2)
    );
    let end = start
        .checked_add(rows)
        .ok_or_else(|| Error::Limit("natural text invocation span overflow".into()))?;
    if !exact
        || call.tokens().capacity() != 512
        || call.tokens().vocabulary() != 248320
        || base.source() != call.source()
        || attempt.base() != base
        || base.frontier() != start as u64
        || attempt.frontier() != end as u64
    {
        return Err(Error::State(
            "natural one-call schedule differs from original published root/request".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::{TextCallGeometry, TokenSpan};
    use zerogravity_core::session_root::SessionAuthority;
    fn call(
        source: SourceIdentity,
        mode: InvocationPhase,
        base: usize,
        rows: usize,
    ) -> TextInvocation {
        TextInvocation::new(
            TokenSpan::new(
                TextCallGeometry::new(source, 248320, 512).unwrap(),
                base,
                std::sync::Arc::from(vec![7; rows]),
            )
            .unwrap(),
            mode,
        )
        .unwrap()
    }
    #[test]
    fn original_published_root_allows_each_closed_call_without_aggregate_forced_attempt() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        for steps in [
            &[
                (InvocationPhase::Prefill, 0, 1),
                (InvocationPhase::Decode, 1, 1),
            ][..],
            &[
                (InvocationPhase::Prefill, 0, 64),
                (InvocationPhase::Prefill, 64, 2),
                (InvocationPhase::Decode, 66, 1),
            ][..],
        ] {
            let mut authority = SessionAuthority::new(source).unwrap();
            for &(phase, base, rows) in steps {
                let attempt = authority.begin_advance(rows as u64).unwrap();
                require_published_invocation(
                    authority.binding(),
                    &attempt,
                    &call(source, phase, base, rows),
                )
                .unwrap();
                authority.prepare_publish(attempt).unwrap().commit();
            }
        }
    }
    #[test]
    fn foreign_root_session_source_or_span_never_becomes_a_schedule() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let a = SessionAuthority::new(source).unwrap();
        let b = SessionAuthority::new(source).unwrap();
        let c = call(source, InvocationPhase::Prefill, 0, 1);
        assert!(
            require_published_invocation(a.binding(), &b.begin_advance(1).unwrap(), &c).is_err()
        );
        assert!(
            require_published_invocation(a.binding(), &a.begin_advance(2).unwrap(), &c).is_err()
        );
        let other = SourceIdentity::for_new_allocation().unwrap();
        assert!(
            require_published_invocation(
                a.binding(),
                &a.begin_advance(1).unwrap(),
                &call(other, InvocationPhase::Prefill, 0, 1)
            )
            .is_err()
        );
    }
    #[test]
    fn real_restore_epoch_rejects_old_attempt_and_accepts_fresh_same_frontier() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut authority = SessionAuthority::new(source).unwrap();
        let base = authority.binding();
        let stale = authority.begin_advance(1).unwrap();
        let advance = authority.begin_advance(1).unwrap();
        authority.prepare_publish(advance).unwrap().commit();
        let restore = authority.begin_restore(base).unwrap();
        authority.prepare_publish(restore).unwrap().commit();
        let c = call(source, InvocationPhase::Prefill, 0, 1);
        assert!(require_published_invocation(authority.binding(), &stale, &c).is_err());
        let fresh = authority.begin_advance(1).unwrap();
        require_published_invocation(authority.binding(), &fresh, &c).unwrap();
    }
    #[test]
    fn all_validation_is_read_only_and_dropping_cancelled_proposal_does_not_publish() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let authority = SessionAuthority::new(source).unwrap();
        let before = authority.binding();
        let attempt = authority.begin_advance(64).unwrap();
        require_published_invocation(
            before,
            &attempt,
            &call(source, InvocationPhase::Prefill, 0, 64),
        )
        .unwrap();
        drop(attempt);
        assert_eq!(authority.binding(), before);
    }
}
