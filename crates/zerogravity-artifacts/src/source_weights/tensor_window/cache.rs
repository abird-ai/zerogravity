//! Small authenticated windows, paid from unused source metadata capacity.
//! No source owner backedge; outstanding evicted readers retain their budget.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

const ENTRIES: usize = 512;
const MAX_WINDOW: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowCacheStats {
    pub budget_bytes: usize,
    pub live_payload_bytes: usize,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Key(pub usize, pub u64, pub usize);
struct Budget {
    limit: usize,
    used: AtomicUsize,
}
struct Charge {
    budget: Arc<Budget>,
    bytes: usize,
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
impl Budget {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Charge> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|n| *n <= self.limit)
            })
            .ok()?;
        Some(Charge {
            budget: self.clone(),
            bytes,
        })
    }
}
pub(super) struct Bytes {
    data: Vec<u8>,
    _charge: Charge,
}
pub(super) enum Storage {
    Staged(StagedSlice),
    Cached(Arc<Bytes>, u64),
}
impl Storage {
    pub(super) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Staged(s) => s.as_slice(),
            Self::Cached(b, _) => &b.data,
        }
    }
    pub(super) fn source_offset(&self) -> u64 {
        match self {
            Self::Staged(s) => s.source_offset(),
            Self::Cached(_, n) => *n,
        }
    }
    pub(super) fn charged_bytes(&self) -> usize {
        match self {
            Self::Staged(s) => s.charged_bytes(),
            Self::Cached(b, _) => b._charge.bytes,
        }
    }
}
struct Entry {
    key: Key,
    data: Arc<Bytes>,
    touched: u64,
}
pub(super) struct Cache {
    entries: Vec<Entry>,
    budget: Option<Arc<Budget>>,
    prepaid: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}
impl Cache {
    pub(super) fn max_window_bytes(&self) -> usize {
        if self.budget.is_some() { MAX_WINDOW } else { 0 }
    }
    pub(super) fn new(prepaid: usize) -> Result<Self> {
        let metadata = ENTRIES * std::mem::size_of::<Entry>() + 4096;
        let mut entries = Vec::new();
        let budget = if prepaid > metadata {
            entries
                .try_reserve_exact(ENTRIES)
                .map_err(|_| Error::Limit("source cache metadata allocation".into()))?;
            if entries.capacity() != ENTRIES {
                return Err(Error::Limit("source cache metadata capacity".into()));
            }
            Some(Arc::new(Budget {
                limit: prepaid - metadata,
                used: AtomicUsize::new(0),
            }))
        } else {
            None
        };
        Ok(Self {
            entries,
            budget,
            prepaid,
            clock: 0,
            hits: 0,
            misses: 0,
        })
    }
    pub(super) fn get(&mut self, key: Key) -> Option<Storage> {
        if key.2 > MAX_WINDOW || self.budget.is_none() {
            return None;
        }
        self.clock = self.clock.saturating_add(1);
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            e.touched = self.clock;
            self.hits = self.hits.saturating_add(1);
            Some(Storage::Cached(e.data.clone(), key.1))
        } else {
            self.misses = self.misses.saturating_add(1);
            None
        }
    }
    pub(super) fn invalidate_shard(&mut self, shard: usize) {
        self.entries.retain(|e| e.key.0 != shard);
    }
    fn evict(&mut self) -> bool {
        let Some((i, _)) = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.touched)
        else {
            return false;
        };
        self.entries.swap_remove(i);
        true
    }
    pub(super) fn insert(&mut self, key: Key, data: &[u8]) {
        if data.is_empty()
            || data.len() != key.2
            || data.len() > MAX_WINDOW
            || self.entries.iter().any(|e| e.key == key)
        {
            return;
        }
        let Some(budget) = self.budget.clone() else {
            return;
        };
        let bytes = data.len() + std::mem::size_of::<Bytes>() + 2 * std::mem::size_of::<usize>();
        if self.entries.len() == ENTRIES {
            self.evict();
        }
        let charge = loop {
            if let Some(charge) = budget.reserve(bytes) {
                break charge;
            }
            if !self.evict() {
                return;
            }
        };
        let mut copy = Vec::new();
        if copy.try_reserve_exact(data.len()).is_err() || copy.capacity() != data.len() {
            return;
        }
        copy.extend_from_slice(data);
        self.clock = self.clock.saturating_add(1);
        self.entries.push(Entry {
            key,
            data: Arc::new(Bytes {
                data: copy,
                _charge: charge,
            }),
            touched: self.clock,
        });
    }
    pub(super) fn stats(&self) -> WindowCacheStats {
        WindowCacheStats {
            budget_bytes: self.prepaid,
            live_payload_bytes: self
                .budget
                .as_ref()
                .map_or(0, |b| b.used.load(Ordering::Acquire)),
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eviction_never_reclaims_a_live_reader_or_spends_its_capacity() {
        let mut c = Cache::new(128 * 1024).unwrap();
        let data = vec![0x5a; 64 * 1024];
        c.insert(Key(0, 1, data.len()), &data);
        let held = c.get(Key(0, 1, data.len())).unwrap();
        let used = c.stats().live_payload_bytes;
        c.insert(Key(0, 2, data.len()), &data);
        assert_eq!(c.stats().entries, 0);
        assert_eq!(c.stats().live_payload_bytes, used);
        assert_eq!(held.as_slice(), data);
        drop(held);
        assert_eq!(c.stats().live_payload_bytes, 0);
        c.insert(Key(0, 2, data.len()), &data);
        assert!(c.get(Key(0, 2, data.len())).is_some());
        assert!(c.get(Key(1, 2, data.len())).is_none());
        assert!(c.get(Key(0, 1, data.len())).is_none());
        assert!(c.stats().live_payload_bytes <= c.stats().budget_bytes);
    }
    #[test]
    fn disabled_and_large_streamed_windows_allocate_no_cache_payload() {
        let mut c = Cache::new(0).unwrap();
        c.insert(Key(0, 0, 1), &[1]);
        assert_eq!(c.stats().live_payload_bytes, 0);
        let mut c = Cache::new(8 * 1024 * 1024).unwrap();
        let large = vec![1; MAX_WINDOW + 1];
        c.insert(Key(0, 0, large.len()), &large);
        assert_eq!(c.stats().live_payload_bytes, 0);
        assert!(c.get(Key(0, 0, large.len())).is_none());
    }
}
