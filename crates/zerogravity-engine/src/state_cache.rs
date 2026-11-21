//! Bounded semantic prefix lookup for complete-domain state snapshots.
//!
//! This module never decides that a payload is physically restorable. Callers
//! supply an exact semantic compatibility fingerprint and retain the payload's
//! source/schema/completion authority. A cache hit is therefore only a candidate
//! for the caller's existing full-domain validation/restore path.

use crate::{Error, Result};
use std::{
    collections::VecDeque,
    num::{NonZeroU64, NonZeroUsize},
    sync::Arc,
};
use zerogravity_core::accelerator::SourceIdentity;

/// Explicit authorization domain for cross-request prefix reuse.
///
/// The engine never derives this from model/source equality. A trusted caller
/// must opt in by supplying a nonzero scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrefixSharingScope(NonZeroU64);

impl PrefixSharingScope {
    pub fn new(value: u64) -> Result<Self> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(|| Error::State("prefix sharing scope must be nonzero".into()))
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Exact process-local compatibility required before a cached prefix may be tried.
///
/// Grouped fingerprints keep the engine model-neutral while forcing callers to
/// bind every relevant semantic class. They are opaque issued/fingerprinted values,
/// never physical pointers or user-controlled hashes accepted without validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCompatibility {
    source: SourceIdentity,
    execution: u64,
    input: u64,
    position: u64,
    state: u64,
    sharing: PrefixSharingScope,
}

impl PrefixCompatibility {
    pub fn new(
        source: SourceIdentity,
        execution: u64,
        input: u64,
        position: u64,
        state: u64,
        sharing: PrefixSharingScope,
    ) -> Result<Self> {
        if !source.is_bound() || execution == 0 || input == 0 || position == 0 || state == 0 {
            return Err(Error::State(
                "prefix cache requires bound nonzero semantic identities".into(),
            ));
        }
        Ok(Self {
            source,
            execution,
            input,
            position,
            state,
            sharing,
        })
    }

    pub const fn source(self) -> SourceIdentity {
        self.source
    }
}

struct PrefixEntry<U, T> {
    compatibility: PrefixCompatibility,
    units: Arc<[U]>,
    payload: Arc<T>,
    weight: usize,
}

/// One immutable cache hit. frontier is the number of matched semantic units.
pub struct PrefixHit<T> {
    pub frontier: usize,
    pub payload: Arc<T>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheEviction {
    pub entries: usize,
    pub weight: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheSnapshot {
    pub entries: usize,
    pub weight: usize,
    pub max_entries: usize,
    pub max_weight: usize,
}

/// One required state domain's exact prefix checkpoint policy.
///
/// Every zero-based frontier that is a multiple of `checkpoint_alignment`
/// through `retained_through` is restorable. The backend owns the actual
/// checkpoint payloads; this type only reconciles their semantic frontiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixDomainCheckpoint {
    retained_through: usize,
    checkpoint_alignment: NonZeroUsize,
}

impl PrefixDomainCheckpoint {
    pub fn new(retained_through: usize, checkpoint_alignment: usize) -> Result<Self> {
        let checkpoint_alignment = NonZeroUsize::new(checkpoint_alignment)
            .ok_or_else(|| Error::Limit("prefix checkpoint alignment must be positive".into()))?;
        Ok(Self {
            retained_through,
            checkpoint_alignment,
        })
    }

    pub const fn retained_through(self) -> usize {
        self.retained_through
    }

    pub const fn checkpoint_alignment(self) -> usize {
        self.checkpoint_alignment.get()
    }
}

/// Largest exact frontier legal for every required domain and the request.
///
/// This is allocation-free and monotone in `requested`: it first caps by the
/// least retained domain frontier, then rounds down to the least common
/// multiple of every domain's checkpoint alignment. If that common alignment
/// exceeds the cap, only the empty frontier is common.
pub fn reconcile_prefix_frontier(
    requested: usize,
    domains: &[PrefixDomainCheckpoint],
) -> Result<usize> {
    let mut cap = requested;
    for domain in domains {
        cap = cap.min(domain.retained_through());
    }
    if domains.is_empty() {
        return Err(Error::State(
            "prefix reconciliation requires at least one state domain".into(),
        ));
    }
    if cap == 0 {
        return Ok(0);
    }

    let mut common_alignment = 1usize;
    for domain in domains {
        let alignment = domain.checkpoint_alignment();
        if alignment > cap {
            return Ok(0);
        }
        let divisor = gcd(common_alignment, alignment);
        let reduced = common_alignment / divisor;
        if reduced > cap / alignment {
            return Ok(0);
        }
        common_alignment = reduced * alignment;
    }
    Ok(cap - cap % common_alignment)
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Small bounded cache for complete-domain prefix candidates.
///
/// Insertion order is the eviction order. Lookup is a bounded linear scan with no
/// allocation; the longest exact prefix wins and the newest entry wins an equal
/// frontier. This keeps the first implementation simple and predictable while the
/// physical payload remains owned and validated by the backend.
pub struct PrefixCache<U, T> {
    entries: VecDeque<PrefixEntry<U, T>>,
    max_entries: usize,
    max_weight: usize,
    weight: usize,
}

impl<U: Eq, T> PrefixCache<U, T> {
    pub fn new(max_entries: usize, max_weight: usize) -> Result<Self> {
        if max_entries == 0 || max_weight == 0 {
            return Err(Error::Limit(
                "prefix cache requires positive entry and weight bounds".into(),
            ));
        }
        Ok(Self {
            entries: VecDeque::with_capacity(max_entries),
            max_entries,
            max_weight,
            weight: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn weight(&self) -> usize {
        self.weight
    }

    pub fn snapshot(&self) -> PrefixCacheSnapshot {
        PrefixCacheSnapshot {
            entries: self.entries.len(),
            weight: self.weight,
            max_entries: self.max_entries,
            max_weight: self.max_weight,
        }
    }

    pub fn insert(
        &mut self,
        compatibility: PrefixCompatibility,
        units: Arc<[U]>,
        payload: Arc<T>,
        weight: usize,
    ) -> Result<()> {
        if units.is_empty() || weight == 0 || weight > self.max_weight {
            return Err(Error::Limit(
                "prefix cache entry exceeds semantic/weight bounds".into(),
            ));
        }

        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.compatibility == compatibility && entry.units == units)
        {
            let removed = self
                .entries
                .remove(index)
                .ok_or_else(|| Error::State("prefix cache duplicate removal".into()))?;
            self.weight = self
                .weight
                .checked_sub(removed.weight)
                .ok_or_else(|| Error::State("prefix cache weight underflow".into()))?;
        }

        while self.entries.len() >= self.max_entries
            || self
                .weight
                .checked_add(weight)
                .is_none_or(|total| total > self.max_weight)
        {
            let removed = self
                .entries
                .pop_front()
                .ok_or_else(|| Error::State("prefix cache eviction without entry".into()))?;
            self.weight = self
                .weight
                .checked_sub(removed.weight)
                .ok_or_else(|| Error::State("prefix cache weight underflow".into()))?;
        }

        self.weight = self
            .weight
            .checked_add(weight)
            .ok_or_else(|| Error::Limit("prefix cache weight overflow".into()))?;
        self.entries.push_back(PrefixEntry {
            compatibility,
            units,
            payload,
            weight,
        });
        Ok(())
    }

    pub fn lookup(
        &self,
        compatibility: PrefixCompatibility,
        request: &[U],
    ) -> Option<PrefixHit<T>> {
        let mut best = None::<&PrefixEntry<U, T>>;
        for entry in self.entries.iter().rev() {
            if entry.compatibility != compatibility
                || entry.units.len() > request.len()
                || !request.starts_with(&entry.units)
            {
                continue;
            }
            if best.is_none_or(|current| entry.units.len() > current.units.len()) {
                best = Some(entry);
            }
        }
        best.map(|entry| PrefixHit {
            frontier: entry.units.len(),
            payload: entry.payload.clone(),
        })
    }

    pub fn invalidate_scope(&mut self, sharing: PrefixSharingScope) -> PrefixCacheEviction {
        let mut eviction = PrefixCacheEviction::default();
        self.entries.retain(|entry| {
            if entry.compatibility.sharing == sharing {
                eviction.entries += 1;
                eviction.weight += entry.weight;
                false
            } else {
                true
            }
        });
        self.weight = self
            .weight
            .checked_sub(eviction.weight)
            .expect("prefix cache invalidation weight");
        eviction
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.weight = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compatibility(source: SourceIdentity, sharing: u64) -> PrefixCompatibility {
        PrefixCompatibility::new(
            source,
            11,
            12,
            13,
            14,
            PrefixSharingScope::new(sharing).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn longest_exact_prefix_and_semantic_scope_win_without_allocation() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let exact = compatibility(source, 20);
        let foreign = compatibility(source, 21);
        let mut cache = PrefixCache::<u32, &'static str>::new(8, 64).unwrap();
        cache
            .insert(exact, Arc::from([1, 2]), Arc::new("two"), 8)
            .unwrap();
        cache
            .insert(exact, Arc::from([1, 2, 3]), Arc::new("three"), 8)
            .unwrap();
        cache
            .insert(foreign, Arc::from([1, 2, 3, 4]), Arc::new("foreign"), 8)
            .unwrap();

        let hit = cache.lookup(exact, &[1, 2, 3, 9]).unwrap();
        assert_eq!(hit.frontier, 3);
        assert_eq!(*hit.payload, "three");
        assert!(cache.lookup(exact, &[1, 8, 3]).is_none());
        assert_eq!(
            *cache.lookup(foreign, &[1, 2, 3, 4]).unwrap().payload,
            "foreign"
        );
    }

    #[test]
    fn duplicate_refresh_and_fifo_bounds_are_exact() {
        let scope = compatibility(SourceIdentity::for_new_allocation().unwrap(), 20);
        let mut cache = PrefixCache::<u32, &'static str>::new(2, 10).unwrap();
        cache
            .insert(scope, Arc::from([1]), Arc::new("old"), 4)
            .unwrap();
        cache
            .insert(scope, Arc::from([1]), Arc::new("new"), 5)
            .unwrap();
        assert_eq!((cache.len(), cache.weight()), (1, 5));
        assert_eq!(*cache.lookup(scope, &[1]).unwrap().payload, "new");

        cache
            .insert(scope, Arc::from([2]), Arc::new("two"), 5)
            .unwrap();
        cache
            .insert(scope, Arc::from([3]), Arc::new("three"), 5)
            .unwrap();
        assert_eq!((cache.len(), cache.weight()), (2, 10));
        assert_eq!(
            cache.snapshot(),
            PrefixCacheSnapshot {
                entries: 2,
                weight: 10,
                max_entries: 2,
                max_weight: 10,
            }
        );
        assert!(cache.lookup(scope, &[1]).is_none());
        assert_eq!(*cache.lookup(scope, &[2]).unwrap().payload, "two");
        assert_eq!(*cache.lookup(scope, &[3]).unwrap().payload, "three");
        assert!(
            cache
                .insert(scope, Arc::from([4]), Arc::new("too-heavy"), 11)
                .is_err()
        );

        let evicted = cache.invalidate_scope(PrefixSharingScope::new(20).unwrap());
        assert_eq!(
            evicted,
            PrefixCacheEviction {
                entries: 2,
                weight: 10,
            }
        );
        assert!(cache.is_empty());
        assert_eq!(cache.weight(), 0);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.weight(), 0);
    }

    #[test]
    fn common_frontier_intersects_retention_and_alignment_without_allocation() {
        let domains = [
            PrefixDomainCheckpoint::new(37, 1).unwrap(),
            PrefixDomainCheckpoint::new(35, 4).unwrap(),
            PrefixDomainCheckpoint::new(33, 8).unwrap(),
        ];
        assert_eq!(reconcile_prefix_frontier(40, &domains).unwrap(), 32);
        assert_eq!(reconcile_prefix_frontier(19, &domains).unwrap(), 16);
        assert_eq!(reconcile_prefix_frontier(7, &domains).unwrap(), 0);
    }

    #[test]
    fn common_frontier_handles_coprime_rules_and_refuses_invalid_policy() {
        let domains = [
            PrefixDomainCheckpoint::new(65, 6).unwrap(),
            PrefixDomainCheckpoint::new(65, 10).unwrap(),
        ];
        assert_eq!(reconcile_prefix_frontier(65, &domains).unwrap(), 60);
        assert_eq!(reconcile_prefix_frontier(29, &domains).unwrap(), 0);
        assert!(PrefixDomainCheckpoint::new(1, 0).is_err());
        assert!(reconcile_prefix_frontier(1, &[]).is_err());
    }
}
