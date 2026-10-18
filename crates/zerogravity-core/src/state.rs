//! Minimal real single-writer logical state publication primitive.
//!
//! This is CPU reference state, not an async GPU/peer completion protocol.
//! All prepared domains are private until the coordinator publishes one root.

use crate::accelerator::SourceIdentity;
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_COORDINATOR: AtomicU64 = AtomicU64::new(1);

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
    pub epoch: u64,
    /// Process-local identity for this logical owner, distinct from a source
    /// allocation: two independent sessions can share ONE model source.
    coordinator_id: u64,
    /// Only source-qualified coordinators admit source-bound publications.
    /// The original generic CPU coordination API remains unchanged.
    source: Option<SourceIdentity>,
    root: StateRoot,
    output_sequence: u64,
}

impl TaskCoordinator {
    pub fn new(
        epoch: u64,
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
        let coordinator_id = NEXT_COORDINATOR
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit("task coordinator identity exhausted".into()))?;
        Ok(Self {
            epoch,
            coordinator_id,
            source: None,
            root: StateRoot {
                id: 0,
                frontier: 0,
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
        let mut coordinator = Self::new(epoch, required_domains)?;
        coordinator.source = Some(source);
        Ok(coordinator)
    }

    pub fn coordinator_id(&self) -> u64 {
        self.coordinator_id
    }

    pub fn source_identity(&self) -> Option<SourceIdentity> {
        self.source
    }

    pub fn root(&self) -> &StateRoot {
        &self.root
    }

    /// Source-bound publication for actual verified CPU proposals. The
    /// trusted owner must still independently validate the full payload
    /// and actual device completion before invoking this.
    pub fn publish_source_bound(
        &mut self,
        source: SourceIdentity,
        expected_root: u64,
        candidate_frontier: u64,
        prepared: Vec<PreparedDomain>,
        payload: Vec<u8>,
    ) -> Result<CommittedOutput> {
        if !source.is_bound() || self.source != Some(source) {
            return Err(Error::State("task publication changed model source".into()));
        }
        self.publish_prepared(expected_root, candidate_frontier, prepared, payload)
    }

    /// All-or-nothing logical root update. The prepared inputs must already be
    /// produced by a checked CPU worker (or later, a device Ready event bridge).
    /// No partial mutation occurs on any invalid candidate.
    pub fn publish(
        &mut self,
        expected_root: u64,
        candidate_frontier: u64,
        prepared: Vec<PreparedDomain>,
        payload: Vec<u8>,
    ) -> Result<CommittedOutput> {
        if self.source.is_some() {
            return Err(Error::State(
                "source-bound coordinator requires source-bound publication".into(),
            ));
        }
        self.publish_prepared(expected_root, candidate_frontier, prepared, payload)
    }

    fn publish_prepared(
        &mut self,
        expected_root: u64,
        candidate_frontier: u64,
        prepared: Vec<PreparedDomain>,
        payload: Vec<u8>,
    ) -> Result<CommittedOutput> {
        if expected_root != self.root.id {
            return Err(Error::State(
                "stale root: session concurrently advanced".into(),
            ));
        }
        if candidate_frontier <= self.root.frontier {
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
        let next_id = self
            .root
            .id
            .checked_add(1)
            .ok_or_else(|| Error::State("root ID exhausted".into()))?;
        let sequence = self
            .output_sequence
            .checked_add(1)
            .ok_or_else(|| Error::State("output sequence exhausted".into()))?;
        // Single publication point. The old root's Arc-backed buffers stay valid
        // for readers even after this assignment.
        self.root = StateRoot {
            id: next_id,
            frontier: candidate_frontier,
            domains: new_domains,
        };
        self.output_sequence = sequence;
        Ok(CommittedOutput {
            root_id: next_id,
            epoch: self.epoch,
            sequence,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn prepared(frontier: u64) -> Vec<PreparedDomain> {
        vec![
            PreparedDomain {
                name: "kv".into(),
                schema: 1,
                bytes: vec![1, 2],
                frontier,
            },
            PreparedDomain {
                name: "recurrence".into(),
                schema: 2,
                bytes: vec![3],
                frontier,
            },
            PreparedDomain {
                name: "rng".into(),
                schema: 3,
                bytes: vec![4],
                frontier,
            },
        ]
    }
    fn coordinator() -> TaskCoordinator {
        TaskCoordinator::new(
            7,
            [
                ("kv".into(), 1),
                ("recurrence".into(), 2),
                ("rng".into(), 3),
            ],
        )
        .unwrap()
    }
    #[test]
    fn atomic_three_domain_publication() {
        let mut c = coordinator();
        let before = c.root().clone();
        let mut bad = prepared(1);
        bad[1].schema = 888;
        assert!(c.publish(0, 1, bad, b"uncommitted".to_vec()).is_err());
        assert_eq!(c.root(), &before);
        let output = c.publish(0, 1, prepared(1), b"committed".to_vec()).unwrap();
        assert_eq!((output.epoch, output.sequence, output.root_id), (7, 1, 1));
        assert_eq!(output.payload, b"committed");
        assert_eq!(before.domains["kv"].bytes.len(), 0);
        assert_eq!(c.root().domains["kv"].bytes.as_ref(), &[1, 2]);
    }
    #[test]
    fn rejects_duplicate_missing_wrong_frontier_and_stale() {
        let mut c = coordinator();
        assert!(
            c.publish(
                0,
                1,
                prepared(1)[..2]
                    .iter()
                    .map(|p| PreparedDomain {
                        name: p.name.clone(),
                        schema: p.schema,
                        bytes: p.bytes.clone(),
                        frontier: p.frontier
                    })
                    .collect(),
                vec![]
            )
            .is_err()
        );
        let mut dup = prepared(1);
        dup[2].name = "kv".into();
        assert!(c.publish(0, 1, dup, vec![]).is_err());
        let mut mixed = prepared(1);
        mixed[2].frontier = 2;
        assert!(c.publish(0, 1, mixed, vec![]).is_err());
        c.publish(0, 1, prepared(1), vec![]).unwrap();
        assert!(c.publish(0, 2, prepared(2), vec![]).is_err());
        assert_eq!(c.root().id, 1);
    }
}

#[cfg(test)]
mod bound_source_tests {
    use super::*;

    fn source() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }

    fn payload(frontier: u64) -> Vec<PreparedDomain> {
        vec![PreparedDomain {
            name: "source.gdn.recurrent.0".into(),
            schema: 0x51,
            bytes: vec![frontier as u8],
            frontier,
        }]
    }

    #[test]
    fn original_and_new_source_bound_routes_share_one_atomic_publication_owner() {
        let a = source();
        let b = source();
        let mut c =
            TaskCoordinator::new_source_bound(99, a, [("source.gdn.recurrent.0".into(), 0x51)])
                .unwrap();
        assert_eq!(c.source_identity(), Some(a));
        assert_ne!(c.coordinator_id(), 0);
        assert_eq!(c.root().id, 0);
        let before = c.root().clone();
        assert!(c.publish(0, 1, payload(1), vec![1]).is_err());
        assert!(
            c.publish_source_bound(b, 0, 1, payload(1), vec![1])
                .is_err()
        );
        assert!(
            c.publish_source_bound(SourceIdentity::unbound(), 0, 1, payload(1), vec![])
                .is_err()
        );
        assert_eq!(c.root(), &before);
        let next = c
            .publish_source_bound(a, 0, 1, payload(1), vec![42])
            .unwrap();
        assert_eq!(next.payload, vec![42]);
        assert_eq!(next.epoch, 99);
        assert_eq!((next.root_id, next.sequence, c.root().frontier), (1, 1, 1));
        assert!(c.publish_source_bound(a, 0, 2, payload(2), vec![]).is_err());
        assert_eq!(c.root().id, 1);
        assert!(
            TaskCoordinator::new_source_bound(0, SourceIdentity::unbound(), [("x".into(), 1)])
                .is_err()
        );
        let plain = TaskCoordinator::new(8, [("x".into(), 1)]).unwrap();
        assert!(plain.source_identity().is_none());
        assert_ne!(plain.coordinator_id(), c.coordinator_id());
    }
}
