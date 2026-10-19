//! Source- and session-bound batch-native row contract.
//!
//! Only trusted SessionAuthority can issue RootBinding; no caller can forge
//! a batch row from an uncommitted/foreign root. This is *admission*, not a
//! second token loop, buffer lease or device Ready. The physical resident
//! owner must still bind real nonaliasing handles and publish each row only
//! after a qualified terminal GPU event.

use std::collections::BTreeSet;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::session_root::RootBinding;
use zerogravity_core::{Error, Result};

const MAX_ROWS: usize = 8;
const MAX_ELEMENTS: usize = 1_048_576;
const MAX_PREFIX: u64 = 512;

/// Immutable logical rows in one proposed native GPU operator batch.
#[derive(Clone, Debug)]
pub struct ResidentBatchRows {
    source: SourceIdentity,
    bindings: Vec<RootBinding>,
    frontier: u64,
    row_elements: usize,
    activation_bytes: usize,
}
impl ResidentBatchRows {
    /// Register one *committed* root per independent session, with an exact
    /// common source/frontier and an explicit dense activation row width.
    pub fn new(bindings: &[RootBinding], row_elements: usize) -> Result<Self> {
        let source = bindings
            .first()
            .ok_or_else(|| Error::Malformed("empty resident batch".into()))?
            .source();
        if !source.is_bound()
            || bindings.len() > MAX_ROWS
            || row_elements == 0
            || row_elements > MAX_ELEMENTS
        {
            return Err(Error::Limit("resident GPU batch size/source/rows".into()));
        }
        let frontier = bindings[0].frontier();
        if frontier >= MAX_PREFIX {
            return Err(Error::Limit(
                "resident batch model context exhausted".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for binding in bindings {
            if binding.source() != source
                || binding.frontier() != frontier
                || !seen.insert(binding.session().get())
            {
                return Err(Error::State(
                    "resident batch has foreign source, differing frontier, or duplicate session"
                        .into(),
                ));
            }
        }
        let total = bindings
            .len()
            .checked_mul(row_elements)
            .filter(|n| *n <= MAX_ELEMENTS)
            .ok_or_else(|| Error::Limit("resident batch dense activation count".into()))?;
        let activation_bytes = total
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("resident batch activation byte count".into()))?;
        Ok(Self {
            source,
            bindings: bindings.to_vec(),
            frontier,
            row_elements,
            activation_bytes,
        })
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    pub fn frontier(&self) -> u64 {
        self.frontier
    }
    pub fn len(&self) -> usize {
        self.bindings.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
    pub fn activation_bytes(&self) -> usize {
        self.activation_bytes
    }
    pub fn row_elements(&self) -> usize {
        self.row_elements
    }
    pub fn row_byte_offset(&self, index: usize) -> Result<usize> {
        if index >= self.bindings.len() {
            return Err(Error::Malformed("batch row outside source cohort".into()));
        }
        index
            .checked_mul(self.row_elements)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Limit("batch row offset overflow".into()))
    }
    /// After an asynchronous GPU completion, the trusted resident owner
    /// revalidates every original root BEFORE independently publishing
    /// each row's result. This check does not substitute device Ready.
    pub fn require_current(&self, bindings: &[RootBinding]) -> Result<()> {
        if bindings != self.bindings {
            return Err(Error::State(
                "stale/reordered/foreign resident batch roots".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::session_root::SessionAuthority;
    fn source() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    #[test]
    fn source_sessions_and_offsets_are_unique_and_nonoverlapping() {
        let id = source();
        let left = SessionAuthority::new(id).unwrap();
        let mut right = SessionAuthority::new(id).unwrap();
        let rows = ResidentBatchRows::new(&[left.binding(), right.binding()], 64).unwrap();
        assert_eq!(rows.source(), id);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.row_byte_offset(0).unwrap(), 0);
        assert_eq!(rows.row_byte_offset(1).unwrap(), 256);
        assert_eq!(rows.activation_bytes(), 512);
        assert!(rows.row_byte_offset(2).is_err());
        assert!(
            rows.require_current(&[left.binding(), right.binding()])
                .is_ok()
        );
        let attempt = right.begin_advance(1).unwrap();
        right.prepare_publish(attempt).unwrap().commit();
        assert!(
            rows.require_current(&[left.binding(), right.binding()])
                .is_err()
        );
        assert!(
            rows.require_current(&[right.binding(), left.binding()])
                .is_err()
        );
        assert_eq!(rows.frontier(), 0);
        // Capture holds only *committed* prior roots, never candidate roots.
    }
    #[test]
    fn foreign_and_duplicate_lanes_refuse_without_any_session_mutation() {
        let a = source();
        let b = source();
        let first = SessionAuthority::new(a).unwrap();
        let other = SessionAuthority::new(a).unwrap();
        let foreign = SessionAuthority::new(b).unwrap();
        assert!(ResidentBatchRows::new(&[], 64).is_err());
        assert!(ResidentBatchRows::new(&[first.binding(), first.binding()], 64).is_err());
        assert!(ResidentBatchRows::new(&[first.binding(), foreign.binding()], 64).is_err());
        assert!(ResidentBatchRows::new(&[first.binding(), other.binding()], 0).is_err());
        assert!(ResidentBatchRows::new(&[first.binding(), other.binding()], usize::MAX).is_err());
        assert!(ResidentBatchRows::new(&[first.binding(), other.binding()], 1_048_576).is_err());
        assert_eq!(first.binding().root(), 0);
        assert_eq!(other.binding().root(), 0);
    }
    #[test]
    fn same_source_different_frontier_and_exhausted_capacity_refuse() {
        let id = source();
        let a = SessionAuthority::new(id).unwrap();
        let mut b = SessionAuthority::new(id).unwrap();
        let attempt = b.begin_advance(1).unwrap();
        b.prepare_publish(attempt).unwrap().commit();
        assert!(ResidentBatchRows::new(&[a.binding(), b.binding()], 32).is_err());
        let mut c = SessionAuthority::new(id).unwrap();
        let advance = c.begin_advance(512).unwrap();
        c.prepare_publish(advance).unwrap().commit();
        assert!(ResidentBatchRows::new(&[c.binding()], 32).is_err());
    }
}
