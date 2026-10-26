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
    frontier: Option<u64>,
    row_elements: usize,
    activation_bytes: usize,
    word_bytes: usize,
}
impl ResidentBatchRows {
    /// Register one *committed* root per independent session, with an exact
    /// common source and an explicit dense activation row width. Each row
    /// retains its own committed causal frontier.
    pub fn new(bindings: &[RootBinding], row_elements: usize) -> Result<Self> {
        Self::new_layout(bindings, row_elements, 4)
    }
    /// Exact packed BF16 row metadata. Numerical/profile and physical lowering
    /// still require the actual prepared source operation; this grants neither.
    #[cfg(any(test, all(feature = "cuda-native-qualification", target_os = "linux")))]
    pub(crate) fn bf16(bindings: &[RootBinding], row_elements: usize) -> Result<Self> {
        Self::new_layout(bindings, row_elements, 2)
    }
    fn new_layout(
        bindings: &[RootBinding],
        row_elements: usize,
        word_bytes: usize,
    ) -> Result<Self> {
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
        let mut seen = BTreeSet::new();
        for binding in bindings {
            if binding.frontier() >= MAX_PREFIX {
                return Err(Error::Limit(
                    "resident batch model context exhausted".into(),
                ));
            }
            if binding.source() != source || !seen.insert(binding.session().get()) {
                return Err(Error::State(
                    "resident batch has foreign source or duplicate session".into(),
                ));
            }
        }
        let first = bindings[0].frontier();
        let frontier = bindings
            .iter()
            .all(|binding| binding.frontier() == first)
            .then_some(first);
        let total = bindings
            .len()
            .checked_mul(row_elements)
            .filter(|n| *n <= MAX_ELEMENTS)
            .ok_or_else(|| Error::Limit("resident batch dense activation count".into()))?;
        let activation_bytes = total
            .checked_mul(word_bytes)
            .ok_or_else(|| Error::Limit("resident batch activation byte count".into()))?;
        Ok(Self {
            source,
            bindings: bindings.to_vec(),
            frontier,
            row_elements,
            activation_bytes,
            word_bytes,
        })
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    /// None when the exact per-row frontiers differ.
    pub fn frontier(&self) -> Option<u64> {
        self.frontier
    }
    pub fn frontiers(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.bindings.iter().map(|binding| binding.frontier())
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
            .and_then(|n| n.checked_mul(self.word_bytes))
            .ok_or_else(|| Error::Limit("batch row offset overflow".into()))
    }
    /// Physical row codec only; never a numerical contract or execution permit.
    #[cfg(any(test, all(feature = "cuda-native-qualification", target_os = "linux")))]
    pub(crate) fn word_bytes(&self) -> usize {
        self.word_bytes
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
        assert_eq!(rows.frontier(), Some(0));
        // Capture holds only *committed* prior roots, never candidate roots.
    }
    #[test]
    fn bf16_rows_preserve_independent_roots_and_exact_two_byte_extents() {
        let id = source();
        let owners = (0..8)
            .map(|_| SessionAuthority::new(id).unwrap())
            .collect::<Vec<_>>();
        let roots = owners
            .iter()
            .map(SessionAuthority::binding)
            .collect::<Vec<_>>();
        for width in [1, 2, 4, 8] {
            let bf16 = ResidentBatchRows::bf16(&roots[..width], 2048).unwrap();
            let f32 = ResidentBatchRows::new(&roots[..width], 2048).unwrap();
            assert_eq!(bf16.activation_bytes(), width * 2048 * 2);
            assert_eq!(bf16.word_bytes(), 2);
            assert_eq!(f32.word_bytes(), 4);
            assert_eq!(f32.activation_bytes(), bf16.activation_bytes() * 2);
            for row in 0..width {
                assert_eq!(bf16.row_byte_offset(row).unwrap(), row * 4096);
            }
            bf16.require_current(&roots[..width]).unwrap();
            assert!(bf16.row_byte_offset(width).is_err());
        }
        assert!(ResidentBatchRows::bf16(&[roots[0], roots[0]], 2048).is_err());
        assert!(ResidentBatchRows::bf16(&roots, usize::MAX).is_err());
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
    fn differing_frontiers_preserve_exact_roots_and_exhausted_capacity_refuses() {
        let id = source();
        let a = SessionAuthority::new(id).unwrap();
        let mut b = SessionAuthority::new(id).unwrap();
        let attempt = b.begin_advance(1).unwrap();
        b.prepare_publish(attempt).unwrap().commit();
        let original = [a.binding(), b.binding()];
        let rows = ResidentBatchRows::new(&original, 32).unwrap();
        assert_eq!(rows.frontier(), None);
        assert_eq!(rows.frontiers().collect::<Vec<_>>(), [0, 1]);
        rows.require_current(&original).unwrap();
        assert!(rows.require_current(&[original[1], original[0]]).is_err());
        let mut c = SessionAuthority::new(id).unwrap();
        let advance = c.begin_advance(512).unwrap();
        c.prepare_publish(advance).unwrap().commit();
        assert!(ResidentBatchRows::new(&[c.binding()], 32).is_err());
        assert!(ResidentBatchRows::new(&[a.binding(), c.binding()], 32).is_err());
    }

    #[test]
    fn mixed_native_widths_keep_every_frontier_and_checked_dense_offsets() {
        let id = source();
        let positions = [255, 256, 0, 31, 32, 1, 510, 511];
        let mut owners = (0..8)
            .map(|_| SessionAuthority::new(id).unwrap())
            .collect::<Vec<_>>();
        for (owner, position) in owners.iter_mut().zip(positions) {
            if position != 0 {
                let attempt = owner.begin_advance(position).unwrap();
                owner.prepare_publish(attempt).unwrap().commit();
            }
        }
        let bindings = owners
            .iter()
            .map(SessionAuthority::binding)
            .collect::<Vec<_>>();
        for width in [2, 4, 8] {
            let elements = MAX_ELEMENTS / width;
            let rows = ResidentBatchRows::new(&bindings[..width], elements).unwrap();
            assert_eq!(rows.frontier(), None);
            assert_eq!(rows.frontiers().collect::<Vec<_>>(), positions[..width]);
            assert_eq!(rows.activation_bytes(), MAX_ELEMENTS * 4);
            for row in 0..width {
                assert_eq!(rows.row_byte_offset(row).unwrap(), row * elements * 4);
            }
            assert!(rows.row_byte_offset(width).is_err());
            assert!(ResidentBatchRows::new(&bindings[..width], elements + 1).is_err());
            let mut reordered = bindings[..width].to_vec();
            reordered.rotate_left(1);
            assert!(rows.require_current(&reordered).is_err());
            rows.require_current(&bindings[..width]).unwrap();
        }
    }
}
