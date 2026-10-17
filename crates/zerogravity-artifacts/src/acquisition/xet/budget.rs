//! Invocation-local counters shared by typed targets and the original HTTP body
//! owner. These are limits on observed attempts/body/control work, not an OS
//! memory allocator or physical TLS/kernel traffic accounting.
use super::XetOptions;
use crate::{Error, Result};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Seed {
    attempts: u64,
    wire: u64,
    metadata: u64,
    decoded: u64,
    leaves: u64,
    refreshes: u64,
    accepted: Accepted,
}
#[derive(Debug)]
pub(crate) struct Budget {
    limits: XetOptions,
    metadata_limit: u64,
    attempts: AtomicU64,
    wire: AtomicU64,
    metadata: AtomicU64,
    decoded: AtomicU64,
    leaves: AtomicU64,
    refreshes: AtomicU64,
    accepted: Mutex<Accepted>,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Accepted {
    terms: usize,
    groups: usize,
    ranges: usize,
}
fn exhausted() -> Error {
    Error::Limit("Xet invocation work budget exhausted".into())
}
fn checked_increment(counter: &AtomicU64, maximum: u64) -> Result<()> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(1).filter(|next| *next <= maximum)
        })
        .map(|_| ())
        .map_err(|_| exhausted())
}
fn observed(counter: &AtomicU64, bytes: u64, maximum: u64) -> Result<()> {
    let old = counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(bytes)
        })
        .map_err(|_| exhausted())?;
    if old + bytes > maximum {
        Err(exhausted())
    } else {
        Ok(())
    }
}
impl Budget {
    pub(crate) fn check_hub_origin(&self, origin: &str) -> Result<()> {
        if origin != self.limits.endpoints.hub {
            return Err(Error::State("Xet Hub scope differs".into()));
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn for_test(limits: &XetOptions, metadata_limit: usize) -> Result<Self> {
        Self::new(limits, metadata_limit, Seed::default())
    }
    pub(super) fn new(limits: &XetOptions, metadata_limit: usize, seed: Seed) -> Result<Self> {
        limits.validate()?;
        if metadata_limit == 0
            || metadata_limit > 8 * 1024 * 1024
            || seed.attempts > limits.max_requests as u64
            || seed.wire > limits.max_wire_bytes
            || seed.metadata > metadata_limit as u64
            || seed.decoded > limits.max_chunks as u64
            || seed.leaves > limits.max_chunks as u64
            || seed.refreshes > limits.max_refreshes as u64
            || seed.accepted.terms > limits.max_terms
            || seed.accepted.groups > limits.max_fetch_groups
            || seed.accepted.ranges > limits.max_ranges
        {
            return Err(exhausted());
        }
        Ok(Self {
            limits: limits.clone(),
            metadata_limit: metadata_limit as u64,
            attempts: AtomicU64::new(seed.attempts),
            wire: AtomicU64::new(seed.wire),
            metadata: AtomicU64::new(seed.metadata),
            decoded: AtomicU64::new(seed.decoded),
            leaves: AtomicU64::new(seed.leaves),
            refreshes: AtomicU64::new(seed.refreshes),
            accepted: Mutex::new(seed.accepted),
        })
    }
    pub(crate) fn begin_attempt(&self, cancel: &AtomicBool) -> Result<()> {
        super::check_cancel(cancel)?;
        self.ensure_live()?;
        checked_increment(&self.attempts, self.limits.max_requests as u64)
    }
    pub(super) fn ensure_live(&self) -> Result<()> {
        if self.wire.load(Ordering::Acquire) > self.limits.max_wire_bytes
            || self.metadata.load(Ordering::Acquire) > self.metadata_limit
        {
            return Err(exhausted());
        }
        Ok(())
    }
    pub(crate) fn charge_body(&self, bytes: u64, metadata: bool) -> Result<()> {
        observed(&self.wire, bytes, self.limits.max_wire_bytes)?;
        if metadata {
            self.charge_metadata(bytes)?;
        }
        Ok(())
    }
    pub(crate) fn charge_metadata(&self, bytes: u64) -> Result<()> {
        observed(&self.metadata, bytes, self.metadata_limit)
    }
    pub(crate) fn read_limit(&self, requested: usize, metadata: bool) -> Result<usize> {
        self.ensure_live()?;
        let mut left = self
            .limits
            .max_wire_bytes
            .saturating_sub(self.wire.load(Ordering::Acquire));
        if metadata {
            left = left.min(
                self.metadata_limit
                    .saturating_sub(self.metadata.load(Ordering::Acquire)),
            );
        }
        Ok(requested.min(left.saturating_add(1).min(usize::MAX as u64) as usize))
    }
    pub(super) fn decode_chunk(&self, cancel: &AtomicBool) -> Result<()> {
        super::check_cancel(cancel)?;
        checked_increment(&self.decoded, self.limits.max_chunks as u64)
    }
    /// Cumulative accepted reconstruction entries, including files/reissues.
    /// This does not purport to count all parser CPU work: malformed parsing is
    /// independently bounded by observed metadata bytes and per-document caps.
    pub(super) fn reconstruction(
        &self,
        terms: usize,
        groups: usize,
        ranges: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        super::check_cancel(cancel)?;
        let mut accepted = self
            .accepted
            .lock()
            .map_err(|_| Error::State("Xet control counter owner failed".into()))?;
        let terms = accepted
            .terms
            .checked_add(terms)
            .filter(|n| *n <= self.limits.max_terms)
            .ok_or_else(exhausted)?;
        let groups = accepted
            .groups
            .checked_add(groups)
            .filter(|n| *n <= self.limits.max_fetch_groups)
            .ok_or_else(exhausted)?;
        let ranges = accepted
            .ranges
            .checked_add(ranges)
            .filter(|n| *n <= self.limits.max_ranges)
            .ok_or_else(exhausted)?;
        *accepted = Accepted {
            terms,
            groups,
            ranges,
        };
        Ok(())
    }
    pub(super) fn file_leaf(&self, cancel: &AtomicBool) -> Result<()> {
        super::check_cancel(cancel)?;
        checked_increment(&self.leaves, self.limits.max_chunks as u64)
    }
    pub(super) fn refresh(&self, cancel: &AtomicBool) -> Result<()> {
        super::check_cancel(cancel)?;
        checked_increment(&self.refreshes, self.limits.max_refreshes as u64)
    }
    pub(super) fn seed(&self) -> Result<Seed> {
        self.ensure_live()?;
        let accepted = *self
            .accepted
            .lock()
            .map_err(|_| Error::State("Xet control counter owner failed".into()))?;
        Ok(Seed {
            attempts: self.attempts.load(Ordering::Acquire),
            wire: self.wire.load(Ordering::Acquire),
            metadata: self.metadata.load(Ordering::Acquire),
            decoded: self.decoded.load(Ordering::Acquire),
            leaves: self.leaves.load(Ordering::Acquire),
            refreshes: self.refreshes.load(Ordering::Acquire),
            accepted,
        })
    }
    #[cfg(test)]
    pub(crate) fn observed_wire_bytes(&self) -> u64 {
        self.wire.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod seed_tests {
    use super::*;
    #[test]
    fn resolution_seed_preserves_every_consumed_budget_and_refuses_current_limits() {
        let limits = XetOptions {
            max_requests: 1,
            max_wire_bytes: 3,
            max_chunks: 1,
            max_refreshes: 1,
            max_terms: 1,
            max_fetch_groups: 1,
            max_ranges: 1,
            ..XetOptions::default()
        };
        let flag = AtomicBool::new(false);
        let original = Budget::new(&limits, 3, Seed::default()).unwrap();
        original.begin_attempt(&flag).unwrap();
        original.charge_body(3, true).unwrap();
        original.decode_chunk(&flag).unwrap();
        original.file_leaf(&flag).unwrap();
        original.refresh(&flag).unwrap();
        original.reconstruction(1, 1, 1, &flag).unwrap();
        let seed = original.seed().unwrap();
        let recovered = Budget::new(&limits, 3, seed).unwrap();
        assert!(recovered.begin_attempt(&flag).is_err());
        assert!(recovered.decode_chunk(&flag).is_err());
        assert!(recovered.file_leaf(&flag).is_err());
        assert!(recovered.refresh(&flag).is_err());
        assert!(recovered.reconstruction(1, 0, 0, &flag).is_err());
        assert!(recovered.reconstruction(0, 1, 0, &flag).is_err());
        assert!(recovered.reconstruction(0, 0, 1, &flag).is_err());
        assert!(recovered.charge_body(1, false).is_err());
        assert!(Budget::new(&limits, 2, seed).is_err());
        let mut tighter = limits.clone();
        tighter.max_refreshes = 0;
        assert!(Budget::new(&tighter, 3, seed).is_err());
        for field in 0..6 {
            let mut excessive = Seed::default();
            match field {
                0 => excessive.decoded = 2,
                1 => excessive.leaves = 2,
                2 => excessive.refreshes = 2,
                3 => excessive.accepted.terms = 2,
                4 => excessive.accepted.groups = 2,
                _ => excessive.accepted.ranges = 2,
            }
            assert!(Budget::new(&limits, 3, excessive).is_err());
        }
    }
    #[test]
    fn poisoned_accepted_owner_never_silently_resets_seed() {
        let budget = Budget::new(&XetOptions::default(), 1024, Seed::default()).unwrap();
        let _ = std::panic::catch_unwind(|| {
            let _guard = budget.accepted.lock().unwrap();
            panic!("test-only poison");
        });
        assert!(matches!(budget.seed(), Err(Error::State(_))));
    }
}
