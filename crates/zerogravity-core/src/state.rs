//! Minimal real single-writer logical state publication primitive.
//!
//! This is CPU reference state, not an async GPU/peer completion protocol.
//! All prepared domains are private until the coordinator publishes one root.

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
    pub epoch: u64,
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
        Ok(Self {
            epoch,
            root: StateRoot {
                id: 0,
                frontier: 0,
                domains,
            },
            output_sequence: 0,
        })
    }

    pub fn root(&self) -> &StateRoot {
        &self.root
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
