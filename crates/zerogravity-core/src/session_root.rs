//! Logical session root publication, shared by host and resident execution.
//!
//! This module owns identities and the publication frontier only. State payloads,
//! domain validation, physical completion, memory leases and quarantine remain
//! with their existing owners. In particular, a reserved attempt is NOT proof
//! that any device work has completed.

use crate::accelerator::SourceIdentity;
use crate::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

/// Process-local session identity. Zero is never issued; identities never wrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(u64);

impl SessionId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Identity of an actually published logical root, not a proposed GPU result.
///
/// Private fields prevent callers from synthesizing a snapshot binding from a
/// proposed frontier. Source identity is local allocation identity, not a model
/// checksum. A portable checkpoint must independently validate its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootBinding {
    source: SourceIdentity,
    session: SessionId,
    root: u64,
    session_epoch: u64,
    frontier: u64,
}

impl RootBinding {
    pub const fn source(self) -> SourceIdentity {
        self.source
    }
    pub const fn session(self) -> SessionId {
        self.session
    }
    pub const fn root(self) -> u64 {
        self.root
    }
    pub const fn session_epoch(self) -> u64 {
        self.session_epoch
    }
    pub const fn frontier(self) -> u64 {
        self.frontier
    }
}

/// A checked reservation, which neither mutates nor pins the current root.
/// Multiple attempts may exist; publication makes all competing attempts stale.
/// The proposed binding is deliberately not exposed as a `RootBinding`.
#[derive(Clone, Debug)]
#[must_use = "an attempt alone does not publish state"]
pub struct RootAttempt {
    base: RootBinding,
    next: RootBinding,
}

impl RootAttempt {
    pub const fn base(&self) -> RootBinding {
        self.base
    }
    pub const fn frontier(&self) -> u64 {
        self.next.frontier
    }
}

/// Sole logical root authority for one session. Do not clone or mirror this
/// owner in a backend: CPU and GPU publication must use this same instance.
#[derive(Debug)]
pub struct SessionAuthority {
    binding: RootBinding,
}

impl SessionAuthority {
    pub fn new(source: SourceIdentity) -> Result<Self> {
        if !source.is_bound() {
            return Err(Error::State(
                "session requires a bound source identity".into(),
            ));
        }
        let session = NEXT_SESSION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| Error::Limit("session IDs exhausted".into()))?;
        Ok(Self {
            binding: RootBinding {
                source,
                session: SessionId(session),
                root: 0,
                session_epoch: 0,
                frontier: 0,
            },
        })
    }

    pub const fn binding(&self) -> RootBinding {
        self.binding
    }

    /// Reserve one logical transaction before allocating or launching its
    /// candidate state. One multi-token prefill still advances root by ONE.
    pub fn begin_advance(&self, tokens: u64) -> Result<RootAttempt> {
        if tokens == 0 {
            return Err(Error::State("session advance must contain tokens".into()));
        }
        let frontier = self
            .binding
            .frontier
            .checked_add(tokens)
            .ok_or_else(|| Error::Limit("session frontier exhausted".into()))?;
        self.reserve(frontier, self.binding.session_epoch)
    }

    /// Restore a previously published snapshot from this source and session.
    /// A rewind invalidates attempts from the abandoned timeline, even when the
    /// restored frontier happens to equal the current frontier.
    pub fn begin_restore(&self, snapshot: RootBinding) -> Result<RootAttempt> {
        if snapshot.source != self.binding.source
            || snapshot.session != self.binding.session
            || snapshot.root > self.binding.root
            || snapshot.session_epoch > self.binding.session_epoch
            || (snapshot.root == self.binding.root && snapshot != self.binding)
        {
            return Err(Error::State(
                "foreign or unpublished session snapshot".into(),
            ));
        }
        self.reserve_replacement(snapshot.frontier)
    }

    /// Reset replaces all state with a private initial candidate, while old
    /// immutable snapshots and physically pending work retain their owners.
    pub fn begin_reset(&self) -> Result<RootAttempt> {
        self.reserve_replacement(0)
    }

    /// Trusted checkpoint owner only: reserve a replacement AFTER independently
    /// validating the complete portable payload, source checksum, schemas and
    /// frontier bounds. Portable data cannot supply a local `RootBinding`.
    /// Import binds it to this session and advances the local root and epoch.
    #[cfg(any(test, feature = "diagnostic-cpu-checkpoint"))]
    pub fn begin_import(&self, frontier: u64) -> Result<RootAttempt> {
        self.reserve_replacement(frontier)
    }

    /// Preserve the direct CPU API's same-session draft restore behavior.
    /// The owner must first validate the draft source, session and full payload.
    /// This is a replacement, not promotion of the draft's stale base attempt.
    pub fn begin_validated_replacement(&self, frontier: u64) -> Result<RootAttempt> {
        self.reserve_replacement(frontier)
    }

    /// Check replacement counters before allocating a private checkpoint
    /// payload. This creates no import attempt and validates no portable data.
    #[cfg(feature = "diagnostic-cpu-checkpoint")]
    pub fn preflight_replacement(&self) -> Result<()> {
        self.reserve_replacement(self.binding.frontier).map(|_| ())
    }

    fn reserve_replacement(&self, frontier: u64) -> Result<RootAttempt> {
        let epoch = self
            .binding
            .session_epoch
            .checked_add(1)
            .ok_or_else(|| Error::Limit("session epoch exhausted".into()))?;
        self.reserve(frontier, epoch)
    }

    fn reserve(&self, frontier: u64, session_epoch: u64) -> Result<RootAttempt> {
        let root = self
            .binding
            .root
            .checked_add(1)
            .ok_or_else(|| Error::Limit("session root ID exhausted".into()))?;
        Ok(RootAttempt {
            base: self.binding,
            next: RootBinding {
                root,
                session_epoch,
                frontier,
                ..self.binding
            },
        })
    }

    /// Final logical validation, AFTER the owner has validated the complete
    /// private payload and, for a GPU proposal, actual physical readiness.
    ///
    /// The returned guard holds the sole authority mutably, so a competing
    /// publication cannot intervene between this check and `commit`. Finish
    /// every other fallible check before replacing the owner's state payload;
    /// then assign that payload and immediately call the infallible commit.
    pub fn prepare_publish(&mut self, attempt: RootAttempt) -> Result<RootTransition<'_>> {
        if attempt.base != self.binding {
            return Err(Error::State("stale or foreign session root attempt".into()));
        }
        Ok(RootTransition {
            authority: self,
            next: attempt.next,
        })
    }
}

/// Exclusive, validated publication guard. Dropping it leaves the root intact.
/// Its scalar getters support payload metadata without leaking a binding for a
/// root which might never be committed. This is not a device-ready proof.
#[must_use = "dropping a transition does not publish state"]
pub struct RootTransition<'a> {
    authority: &'a mut SessionAuthority,
    next: RootBinding,
}

impl RootTransition<'_> {
    pub const fn next_root(&self) -> u64 {
        self.next.root
    }
    pub const fn next_epoch(&self) -> u64 {
        self.next.session_epoch
    }
    pub const fn frontier(&self) -> u64 {
        self.next.frontier
    }

    /// Single logical publication point; no allocations or fallible cleanup.
    pub fn commit(self) -> RootBinding {
        self.authority.binding = self.next;
        self.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> SessionAuthority {
        SessionAuthority::new(SourceIdentity::from_issued_id(1)).unwrap()
    }

    fn advance(session: &mut SessionAuthority, tokens: u64) -> RootBinding {
        let attempt = session.begin_advance(tokens).unwrap();
        session.prepare_publish(attempt).unwrap().commit()
    }

    #[test]
    fn initial_identity_and_transaction_root_count() {
        assert!(SessionAuthority::new(SourceIdentity::unbound()).is_err());
        let mut a = session();
        let b = session();
        assert_ne!(a.binding().session(), b.binding().session());
        assert_ne!(a.binding().session().get(), 0);
        assert_eq!(a.binding().root(), 0);
        assert_eq!(a.binding().frontier(), 0);
        assert_eq!(a.binding().session_epoch(), 0);
        let root = advance(&mut a, 4);
        assert_eq!(root.root(), 1);
        assert_eq!(root.frontier(), 4);
        assert_eq!(root.session_epoch(), 0);
        assert_eq!(advance(&mut a, 1).root(), 2);
    }

    #[test]
    fn competing_proposals_and_foreign_sessions_cannot_publish() {
        let mut a = session();
        let mut b = session();
        let first = a.begin_advance(1).unwrap();
        let cloned = first.clone();
        let second = a.begin_advance(7).unwrap();
        let foreign = a.begin_advance(1).unwrap();
        let unchanged = b.binding();
        assert!(b.prepare_publish(foreign).is_err());
        assert_eq!(b.binding(), unchanged);
        a.prepare_publish(first).unwrap().commit();
        let current = a.binding();
        assert!(a.prepare_publish(second).is_err());
        assert!(a.prepare_publish(cloned).is_err());
        assert_eq!(a.binding(), current);
    }

    #[test]
    fn abandoning_attempts_and_guards_does_not_publish() {
        let mut a = session();
        let initial = a.binding();
        let _abandoned = a.begin_advance(100).unwrap();
        let attempt = a.begin_advance(3).unwrap();
        assert_eq!(attempt.base(), initial);
        {
            let guard = a.prepare_publish(attempt).unwrap();
            assert_eq!(guard.next_root(), 1);
            assert_eq!(guard.next_epoch(), 0);
            assert_eq!(guard.frontier(), 3);
        }
        assert_eq!(a.binding(), initial);
        assert!(a.begin_advance(0).is_err());
        assert_eq!(a.binding(), initial);
    }

    #[test]
    fn restore_rewinds_and_reset_invalidates_old_timeline() {
        let mut a = session();
        let initial = a.binding();
        let saved = advance(&mut a, 4);
        let future = advance(&mut a, 6);
        let stale = a.begin_advance(1).unwrap();
        let restore = a.begin_restore(saved).unwrap();
        let restored = a.prepare_publish(restore).unwrap().commit();
        assert_eq!(
            (
                restored.root(),
                restored.frontier(),
                restored.session_epoch()
            ),
            (3, 4, 1)
        );
        assert!(a.prepare_publish(stale).is_err());
        // A snapshot from a previously committed, now abandoned timeline is
        // still a valid immutable snapshot, even ahead of today's frontier.
        let restore = a.begin_restore(future).unwrap();
        let restored = a.prepare_publish(restore).unwrap().commit();
        assert_eq!(
            (
                restored.root(),
                restored.frontier(),
                restored.session_epoch()
            ),
            (4, 10, 2)
        );
        let reset = a.begin_reset().unwrap();
        let reset = a.prepare_publish(reset).unwrap().commit();
        assert_eq!(
            (reset.root(), reset.frontier(), reset.session_epoch()),
            (5, 0, 3)
        );
        let restore = a.begin_restore(initial).unwrap();
        assert_eq!(a.prepare_publish(restore).unwrap().commit().root(), 6);
    }

    #[test]
    fn source_epoch_and_entire_base_binding_are_checked() {
        let mut a = session();
        let initial = a.binding();
        for mutate in 0..5 {
            let mut attempt = a.begin_advance(1).unwrap();
            match mutate {
                0 => attempt.base.source = SourceIdentity::from_issued_id(2),
                1 => attempt.base.session = SessionId(u64::MAX),
                2 => attempt.base.root += 1,
                3 => attempt.base.session_epoch += 1,
                _ => attempt.base.frontier += 1,
            }
            assert!(a.prepare_publish(attempt).is_err());
            assert_eq!(a.binding(), initial);
        }
        let foreign = session().binding();
        assert!(a.begin_restore(foreign).is_err());
        let mut forged = initial;
        forged.source = SourceIdentity::from_issued_id(2);
        assert!(a.begin_restore(forged).is_err());
        forged = initial;
        forged.frontier = 1;
        assert!(a.begin_restore(forged).is_err());
        assert_eq!(a.binding(), initial);
    }

    #[test]
    fn exhausted_counters_fail_before_mutation() {
        let mut a = session();
        a.binding.root = u64::MAX;
        let initial = a.binding();
        assert!(a.begin_advance(1).is_err());
        assert!(a.begin_reset().is_err());
        assert!(a.begin_restore(initial).is_err());
        assert!(a.begin_import(0).is_err());
        assert_eq!(a.binding(), initial);

        a.binding.root = 7;
        a.binding.frontier = u64::MAX;
        let initial = a.binding();
        assert!(a.begin_advance(1).is_err());
        assert_eq!(a.binding(), initial);

        a.binding.session_epoch = u64::MAX;
        let initial = a.binding();
        assert!(a.begin_reset().is_err());
        assert!(a.begin_restore(initial).is_err());
        assert!(a.begin_import(5).is_err());
        assert_eq!(a.binding(), initial);
        // Ordinary progress does not consume a timeline epoch.
        a.binding.frontier = 0;
        assert_eq!(advance(&mut a, 1).session_epoch(), u64::MAX);
    }

    #[test]
    fn checkpoint_import_uses_local_identity_and_invalidates_attempts() {
        let mut a = session();
        let local = a.binding();
        let old_attempt = a.begin_advance(1).unwrap();
        let import = a.begin_import(19).unwrap();
        assert_eq!(a.binding(), local);
        let imported = a.prepare_publish(import).unwrap().commit();
        assert_eq!(imported.source(), local.source());
        assert_eq!(imported.session(), local.session());
        assert_eq!(
            (
                imported.root(),
                imported.session_epoch(),
                imported.frontier()
            ),
            (1, 1, 19)
        );
        assert!(a.prepare_publish(old_attempt).is_err());
        let import = a.begin_import(2).unwrap();
        let rewound = a.prepare_publish(import).unwrap().commit();
        assert_eq!(
            (rewound.root(), rewound.session_epoch(), rewound.frontier()),
            (2, 2, 2)
        );
    }
}
