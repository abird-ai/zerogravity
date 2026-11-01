//! Minimal real single-writer logical state publication primitive.
//!
//! This is CPU reference state, not an async GPU/peer completion protocol.
//! All prepared domains are private until the coordinator publishes one root
//! through the same SessionAuthority used by ordinary host/resident sessions.

use crate::accelerator::SourceIdentity;
use crate::session_root::{RootAttempt, RootBinding, SessionAuthority};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainVersion {
    pub schema: u64,
    /// Immutable bytes allow a prior committed root to survive a failed draft.
    pub bytes: Arc<[u8]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateRoot {
    pub id: u64,
    pub frontier: u64,
    pub domains: BTreeMap<String, DomainVersion>,
}

#[derive(Debug)]
pub struct PreparedDomain {
    pub name: String,
    pub schema: u64,
    pub bytes: Vec<u8>,
    pub frontier: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct CommittedOutput {
    pub root_id: u64,
    pub epoch: u64,
    pub sequence: u64,
    pub payload: Vec<u8>,
}

pub struct TaskCoordinator {
    authority: SessionAuthority,
    /// Complete payload plus copies of committed root/frontier metadata.
    root: StateRoot,
    output_sequence: u64,
}

impl TaskCoordinator {
    pub fn new(
        epoch: u64,
        required_domains: impl IntoIterator<Item = (String, u64)>,
    ) -> Result<Self> {
        Self::with_source(epoch, SourceIdentity::unbound(), required_domains)
    }

    fn with_source(
        epoch: u64,
        source: SourceIdentity,
        required_domains: impl IntoIterator<Item = (String, u64)>,
    ) -> Result<Self> {
        let mut domains = BTreeMap::new();
        for (name, schema) in required_domains {
            if name.is_empty()
                || domains
                    .insert(
                        name,
                        DomainVersion {
                            schema,
                            bytes: Arc::new([]),
                        },
                    )
                    .is_some()
            {
                return Err(Error::State("duplicate or empty state domain".into()));
            }
        }
        if domains.is_empty() {
            return Err(Error::State("no required domains".into()));
        }
        let authority = SessionAuthority::new_reference(source, epoch)?;
        let binding = authority.binding();
        Ok(Self {
            authority,
            root: StateRoot {
                id: binding.root(),
                frontier: binding.frontier(),
                domains,
            },
            output_sequence: 0,
        })
    }

    /// A model-specific execution path must use this *same* TaskCoordinator,
    /// with one owned immutable source identity. It is not a new scheduler
    /// or session authority. The source-bound publication route rejects
    /// all unbound/foreign sources before considering payloads.
    pub fn new_source_bound(
        epoch: u64,
        source: SourceIdentity,
        required_domains: impl IntoIterator<Item = (String, u64)>,
    ) -> Result<Self> {
        if !source.is_bound() {
            return Err(Error::State("task coordinator source is unbound".into()));
        }
        Self::with_source(epoch, source, required_domains)
    }

    pub const fn binding(&self) -> RootBinding {
        self.authority.binding()
    }

    pub const fn epoch(&self) -> u64 {
        self.binding().session_epoch()
    }

    /// Same process-local SessionId namespace as ordinary host/resident owners.
    pub const fn coordinator_id(&self) -> u64 {
        self.binding().session().get()
    }

    pub fn source_identity(&self) -> Option<SourceIdentity> {
        let source = self.binding().source();
        source.is_bound().then_some(source)
    }

    /// Reserve before computing a private candidate; no state is published.
    /// One multi-unit transaction advances the root once, not once per unit.
    pub fn begin_advance(&self, units: u64) -> Result<RootAttempt> {
        self.authority.begin_advance(units)
    }

    pub fn root(&self) -> &StateRoot {
        &self.root
    }

    /// Source-bound publication for actual verified CPU proposals. The source
    /// comes from the actual producer; the attempt must belong to this owner.
    /// The trusted producer still validates domain bytes and, for device work,
    /// actual physical completion. A RootAttempt is never a Ready witness.
    pub fn publish_source_bound(
        &mut self,
        source: SourceIdentity,
        attempt: RootAttempt,
        prepared: Vec<PreparedDomain>,
        payload: Vec<u8>,
    ) -> Result<CommittedOutput> {
        if !source.is_bound() || self.source_identity() != Some(source) {
            return Err(Error::State("task publication changed model source".into()));
        }
        self.publish_prepared(attempt, prepared, payload)
    }

    /// Source-less reference publication only, including the numeric-root
    /// device-event bridge. This cannot authorize source-bound Ready work.
    /// No partial mutation occurs on any invalid candidate.
    pub fn publish(
        &mut self,
        expected_root: u64,
        candidate_frontier: u64,
        prepared: Vec<PreparedDomain>,
        payload: Vec<u8>,
    ) -> Result<CommittedOutput> {
        if self.source_identity().is_some() {
            return Err(Error::State(
                "source-bound coordinator requires source-bound publication".into(),
            ));
        }
        let binding = self.binding();
        if expected_root != binding.root() {
            return Err(Error::State(
                "stale root: session concurrently advanced".into(),
            ));
        }
        let units = candidate_frontier
            .checked_sub(binding.frontier())
            .filter(|units| *units > 0)
            .ok_or_else(|| Error::State("frontier did not advance".into()))?;
        let attempt = self.begin_advance(units)?;
        self.publish_prepared(attempt, prepared, payload)
    }

    fn publish_prepared(
        &mut self,
        attempt: RootAttempt,
        prepared: Vec<PreparedDomain>,
        payload: Vec<u8>,
    ) -> Result<CommittedOutput> {
        let binding = self.binding();
        if attempt.base() != binding {
            return Err(Error::State("stale or foreign session root attempt".into()));
        }
        let candidate_frontier = attempt.frontier();
        if candidate_frontier <= binding.frontier() {
            return Err(Error::State("frontier did not advance".into()));
        }
        if prepared.len() != self.root.domains.len() {
            return Err(Error::State("incomplete domain preparation".into()));
        }
        let mut names = BTreeSet::new();
        let mut new_domains = BTreeMap::new();
        for candidate in prepared {
            let Some(old) = self.root.domains.get(&candidate.name) else {
                return Err(Error::State("unregistered state domain".into()));
            };
            if !names.insert(candidate.name.clone()) {
                return Err(Error::State("duplicate prepared state domain".into()));
            }
            if old.schema != candidate.schema || candidate.frontier != candidate_frontier {
                return Err(Error::State("schema or causal frontier mismatch".into()));
            }
            new_domains.insert(
                candidate.name,
                DomainVersion {
                    schema: candidate.schema,
                    bytes: candidate.bytes.into(),
                },
            );
        }
        let sequence = self
            .output_sequence
            .checked_add(1)
            .ok_or_else(|| Error::State("output sequence exhausted".into()))?;
        let transition = self.authority.prepare_publish(attempt)?;
        // All validation/allocation is complete. Install the full payload and
        // commit without fallible work, callbacks or awaits between them.
        // Prior Arc-backed snapshots remain valid after replacement.
        self.root = StateRoot {
            id: transition.next_root(),
            frontier: transition.frontier(),
            domains: new_domains,
        };
        let committed = transition.commit();
        self.output_sequence = sequence;
        Ok(CommittedOutput {
            root_id: committed.root(),
            epoch: committed.session_epoch(),
            sequence,
            payload,
        })
    }
}

#[cfg(test)]
mod tests;
