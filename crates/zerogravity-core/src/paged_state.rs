//! Model-neutral immutable paged-state mapping and copy-on-write planning.
//!
//! This module owns logical block-table geometry only. It does not allocate device
//! memory, report physical completion, publish a `StateRoot`, or grant source/model
//! execution permission. Callers supply page handles whose existing owner is
//! responsible for physical admission, generation/ABA protection, completion,
//! cancellation and quarantine.
//!
//! A committed table is immutable. `fork_for_append` creates a private candidate:
//! full pages are shared, a partially valid tail is always copied before append,
//! and every fresh page is allocated before the table is changed. Dropping the
//! candidate/plan leaves the prior table untouched and releases only caller-owned
//! fresh handles through their normal RAII path.

use crate::accelerator::SourceIdentity;
use crate::{Error, Result};

/// Exact semantic compatibility required before state pages may be shared.
///
/// These are process-local contract fingerprints, not content hashes. The source
/// identity is the original immutable loaded allocation; the remaining nonzero
/// keys are issued by the selected numerical/state-layout owners. A higher-level
/// prefix cache still needs tokenizer/template/processor/adapter/tenant identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageCompatibility {
    source: SourceIdentity,
    numerical: u64,
    domain: u64,
    layout: u64,
}

impl PageCompatibility {
    pub fn new(source: SourceIdentity, numerical: u64, domain: u64, layout: u64) -> Result<Self> {
        if !source.is_bound() || numerical == 0 || domain == 0 || layout == 0 {
            return Err(Error::State(
                "paged state requires bound source/numerical/domain/layout identity".into(),
            ));
        }
        Ok(Self {
            source,
            numerical,
            domain,
            layout,
        })
    }

    pub const fn source(self) -> SourceIdentity {
        self.source
    }

    pub const fn numerical(self) -> u64 {
        self.numerical
    }

    pub const fn domain(self) -> u64 {
        self.domain
    }

    pub const fn layout(self) -> u64 {
        self.layout
    }
}

/// Physical page extent plus the smallest legal semantic append unit.
///
/// A KV adapter uses one token-row byte extent as `unit_bytes`; a page must be
/// an exact multiple of that unit. This prevents a generic byte table from
/// silently splitting one model-defined state element at a page boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageGeometry {
    page_bytes: usize,
    unit_bytes: usize,
}

impl PageGeometry {
    pub fn new(page_bytes: usize, unit_bytes: usize) -> Result<Self> {
        if page_bytes == 0 || unit_bytes == 0 || !page_bytes.is_multiple_of(unit_bytes) {
            return Err(Error::Limit(
                "paged state page extent must be a positive multiple of its unit".into(),
            ));
        }
        Ok(Self {
            page_bytes,
            unit_bytes,
        })
    }

    pub const fn page_bytes(self) -> usize {
        self.page_bytes
    }

    pub const fn unit_bytes(self) -> usize {
        self.unit_bytes
    }
}

/// One immutable logical-to-physical page mapping.
#[derive(Clone, Debug)]
struct PageEntry<H> {
    handle: H,
    valid: usize,
}

/// A source page which must be copied into a private target before candidate use.
///
/// Both handles are retained by this descriptor, so the source and target remain
/// owned until the physical copy owner has accepted them.
#[derive(Clone, Debug)]
pub struct PageCopy<H> {
    source: H,
    target: H,
    bytes: usize,
}

impl<H> PageCopy<H> {
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    pub const fn source(&self) -> &H {
        &self.source
    }

    pub const fn target(&self) -> &H {
        &self.target
    }
}

/// One append range within a private target page.
///
/// The descriptor pins the target handle until the physical write owner accepts
/// it. Offset and bytes are always within the configured page extent.
#[derive(Clone, Debug)]
pub struct PageWrite<H> {
    target: H,
    offset: usize,
    bytes: usize,
}

impl<H> PageWrite<H> {
    pub const fn target(&self) -> &H {
        &self.target
    }

    pub const fn offset(&self) -> usize {
        self.offset
    }

    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Physical work required to materialize one private append candidate.
///
/// This is a plan/pin set, never a Ready witness. Callers must perform every copy
/// and write under their original device leases/events and publish only after
/// actual completion plus the existing root/domain validation.
#[derive(Clone, Debug)]
pub struct PageAppendPlan<H> {
    prior_bytes: usize,
    final_bytes: usize,
    copies: Vec<PageCopy<H>>,
    writes: Vec<PageWrite<H>>,
}

impl<H> PageAppendPlan<H> {
    pub const fn prior_bytes(&self) -> usize {
        self.prior_bytes
    }

    pub const fn final_bytes(&self) -> usize {
        self.final_bytes
    }

    pub fn copies(&self) -> &[PageCopy<H>] {
        &self.copies
    }

    pub fn writes(&self) -> &[PageWrite<H>] {
        &self.writes
    }
}

/// Immutable logical block table for one exact state domain.
///
/// `H` is the existing physical page owner (for example, a resident device
/// lease). Cloning a table shares those owners; this module never manufactures
/// or recycles physical handles itself.
#[derive(Clone, Debug)]
pub struct PagedBlockTable<H> {
    compatibility: PageCompatibility,
    geometry: PageGeometry,
    logical_bytes: usize,
    pages: Vec<PageEntry<H>>,
}

impl<H> PagedBlockTable<H> {
    pub fn empty(compatibility: PageCompatibility, geometry: PageGeometry) -> Self {
        Self {
            compatibility,
            geometry,
            logical_bytes: 0,
            pages: Vec::new(),
        }
    }

    pub const fn compatibility(&self) -> PageCompatibility {
        self.compatibility
    }
    /// Backing of a cloned table's entries, excluding allocations owned by H.
    pub fn cloned_page_bytes(&self) -> usize {
        std::mem::size_of_val(self.pages.as_slice())
    }

    pub const fn page_bytes(&self) -> usize {
        self.geometry.page_bytes
    }

    pub const fn geometry(&self) -> PageGeometry {
        self.geometry
    }

    pub const fn logical_bytes(&self) -> usize {
        self.logical_bytes
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    pub fn page(&self, logical: usize) -> Option<(&H, usize)> {
        self.pages
            .get(logical)
            .map(|entry| (&entry.handle, entry.valid))
    }

    /// Share an immutable prefix after checking the complete page compatibility.
    ///
    /// A prefix ending inside a page narrows only the logical valid extent; the
    /// physical page remains immutable and retained. Any later append COWs that
    /// tail before writing.
    pub fn share_prefix(
        &self,
        compatibility: PageCompatibility,
        geometry: PageGeometry,
        bytes: usize,
    ) -> Result<Self>
    where
        H: Clone,
    {
        self.require_compatible(compatibility, geometry)?;
        if bytes > self.logical_bytes || !bytes.is_multiple_of(self.geometry.unit_bytes) {
            return Err(Error::State(
                "paged state prefix exceeds extent or splits its semantic unit".into(),
            ));
        }
        if bytes == 0 {
            return Ok(Self::empty(self.compatibility, self.geometry));
        }

        let count = pages_for(bytes, self.geometry.page_bytes)?;
        let mut pages = self.pages[..count].to_vec();
        let tail = bytes % self.geometry.page_bytes;
        if tail != 0 {
            pages.last_mut().expect("nonzero prefix has a page").valid = tail;
        }
        let result = Self {
            compatibility: self.compatibility,
            geometry: self.geometry,
            logical_bytes: bytes,
            pages,
        };
        result.validate()?;
        Ok(result)
    }

    /// Create one private append candidate without mutating the committed table.
    ///
    /// The allocator is called for every new physical page before any candidate
    /// table mutation. A partial committed tail always receives a private target
    /// and a copy descriptor. Full committed pages are shared unchanged.
    pub fn fork_for_append<F>(
        &self,
        append_bytes: usize,
        mut allocate: F,
    ) -> Result<(Self, PageAppendPlan<H>)>
    where
        H: Clone,
        F: FnMut() -> Result<H>,
    {
        if append_bytes == 0 || !append_bytes.is_multiple_of(self.geometry.unit_bytes) {
            return Err(Error::State(
                "paged state append must contain whole semantic units".into(),
            ));
        }
        self.validate()?;
        let final_bytes = self
            .logical_bytes
            .checked_add(append_bytes)
            .ok_or_else(|| Error::Limit("paged state logical extent overflow".into()))?;

        let tail_valid = self.pages.last().map_or(0, |entry| entry.valid);
        let partial_tail = tail_valid != 0 && tail_valid < self.geometry.page_bytes;
        let tail_write = if partial_tail {
            append_bytes.min(self.geometry.page_bytes - tail_valid)
        } else {
            0
        };
        let remaining = append_bytes - tail_write;
        let fresh_pages = usize::from(partial_tail)
            .checked_add(pages_for(remaining, self.geometry.page_bytes)?)
            .ok_or_else(|| Error::Limit("paged state fresh page count overflow".into()))?;

        // Allocation is transactional with respect to the table: if any physical
        // owner refuses, every earlier fresh handle drops and `self` is unchanged.
        let mut fresh = Vec::with_capacity(fresh_pages);
        for _ in 0..fresh_pages {
            fresh.push(allocate()?);
        }
        let mut fresh = fresh.into_iter();

        let mut pages = self.pages.clone();
        let mut copies = Vec::with_capacity(usize::from(partial_tail));
        let mut writes = Vec::with_capacity(pages_for(append_bytes, self.geometry.page_bytes)? + 1);
        let mut left = append_bytes;

        if partial_tail {
            let entry = pages.last_mut().expect("partial tail has a page");
            let source = entry.handle.clone();
            let target = fresh.next().expect("preallocated COW target");
            copies.push(PageCopy {
                source,
                target: target.clone(),
                bytes: tail_valid,
            });
            let bytes = left.min(self.geometry.page_bytes - tail_valid);
            writes.push(PageWrite {
                target: target.clone(),
                offset: tail_valid,
                bytes,
            });
            entry.handle = target;
            entry.valid = tail_valid + bytes;
            left -= bytes;
        }

        while left != 0 {
            let target = fresh.next().expect("preallocated append page");
            let bytes = left.min(self.geometry.page_bytes);
            writes.push(PageWrite {
                target: target.clone(),
                offset: 0,
                bytes,
            });
            pages.push(PageEntry {
                handle: target,
                valid: bytes,
            });
            left -= bytes;
        }
        debug_assert!(fresh.next().is_none());

        let candidate = Self {
            compatibility: self.compatibility,
            geometry: self.geometry,
            logical_bytes: final_bytes,
            pages,
        };
        candidate.validate()?;
        Ok((
            candidate,
            PageAppendPlan {
                prior_bytes: self.logical_bytes,
                final_bytes,
                copies,
                writes,
            },
        ))
    }

    fn require_compatible(
        &self,
        compatibility: PageCompatibility,
        geometry: PageGeometry,
    ) -> Result<()> {
        if compatibility != self.compatibility || geometry != self.geometry {
            return Err(Error::State(
                "paged state source/profile/domain/layout/page geometry differs".into(),
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.logical_bytes == 0 {
            if !self.pages.is_empty() {
                return Err(Error::State(
                    "empty paged state retains logical pages".into(),
                ));
            }
            return Ok(());
        }
        if !self.logical_bytes.is_multiple_of(self.geometry.unit_bytes) {
            return Err(Error::State(
                "paged state logical extent splits its semantic unit".into(),
            ));
        }
        let expected = pages_for(self.logical_bytes, self.geometry.page_bytes)?;
        if self.pages.len() != expected {
            return Err(Error::State(
                "paged state block-table coverage differs".into(),
            ));
        }
        for (index, entry) in self.pages.iter().enumerate() {
            if entry.valid == 0
                || entry.valid > self.geometry.page_bytes
                || !entry.valid.is_multiple_of(self.geometry.unit_bytes)
                || (index + 1 != self.pages.len() && entry.valid != self.geometry.page_bytes)
            {
                return Err(Error::State(
                    "paged state has invalid full/tail extent".into(),
                ));
            }
        }
        let tail = self.logical_bytes % self.geometry.page_bytes;
        let expected_tail = if tail == 0 {
            self.geometry.page_bytes
        } else {
            tail
        };
        if self
            .pages
            .last()
            .is_none_or(|entry| entry.valid != expected_tail)
        {
            return Err(Error::State(
                "paged state terminal valid extent differs".into(),
            ));
        }
        Ok(())
    }
}

fn pages_for(bytes: usize, page_bytes: usize) -> Result<usize> {
    if page_bytes == 0 {
        return Err(Error::Limit(
            "paged state page extent must be positive".into(),
        ));
    }
    Ok(bytes / page_bytes + usize::from(!bytes.is_multiple_of(page_bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, Weak};

    #[derive(Debug)]
    struct MockPage {
        id: usize,
        live: Arc<AtomicUsize>,
    }
    impl Drop for MockPage {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::AcqRel);
        }
    }
    #[derive(Clone, Debug)]
    struct MockHandle(Arc<MockPage>);
    impl MockHandle {
        fn id(&self) -> usize {
            self.0.id
        }
    }
    struct MockPool {
        limit: usize,
        next: AtomicUsize,
        live: Arc<AtomicUsize>,
        issued: Mutex<Vec<Weak<MockPage>>>,
    }
    impl MockPool {
        fn new(limit: usize) -> Self {
            Self {
                limit,
                next: AtomicUsize::new(1),
                live: Arc::new(AtomicUsize::new(0)),
                issued: Mutex::new(Vec::new()),
            }
        }
        fn alloc(&self) -> Result<MockHandle> {
            let live = self.live.load(Ordering::Acquire);
            if live >= self.limit {
                return Err(Error::Limit("mock physical page pool exhausted".into()));
            }
            let id = self.next.fetch_add(1, Ordering::AcqRel);
            self.live.fetch_add(1, Ordering::AcqRel);
            let page = Arc::new(MockPage {
                id,
                live: self.live.clone(),
            });
            self.issued.lock().unwrap().push(Arc::downgrade(&page));
            Ok(MockHandle(page))
        }
        fn live(&self) -> usize {
            self.live.load(Ordering::Acquire)
        }
    }

    fn key() -> PageCompatibility {
        PageCompatibility::new(SourceIdentity::for_new_allocation().unwrap(), 11, 22, 33).unwrap()
    }
    fn geometry(page_bytes: usize) -> PageGeometry {
        PageGeometry::new(page_bytes, 1).unwrap()
    }

    #[test]
    fn compatibility_is_complete_and_zero_or_unbound_identity_refuses() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        assert!(PageCompatibility::new(SourceIdentity::unbound(), 1, 2, 3).is_err());
        for keys in [(0, 2, 3), (1, 0, 3), (1, 2, 0)] {
            assert!(PageCompatibility::new(source, keys.0, keys.1, keys.2).is_err());
        }
        let k = PageCompatibility::new(source, 7, 8, 9).unwrap();
        assert_eq!(k.source(), source);
        assert_eq!((k.numerical(), k.domain(), k.layout()), (7, 8, 9));
        assert!(PageGeometry::new(0, 1).is_err());
        assert!(PageGeometry::new(8, 0).is_err());
        assert!(PageGeometry::new(10, 4).is_err());
    }

    #[test]
    fn full_pages_share_but_partial_tail_cows_before_append() {
        let pool = MockPool::new(8);
        let empty = PagedBlockTable::empty(key(), geometry(8));
        let (root, first) = empty.fork_for_append(14, || pool.alloc()).unwrap();
        assert!(first.copies().is_empty());
        assert_eq!(
            first
                .writes()
                .iter()
                .map(|w| (w.offset(), w.bytes()))
                .collect::<Vec<_>>(),
            [(0, 8), (0, 6)]
        );
        assert_eq!(root.logical_bytes(), 14);
        assert_eq!(root.page_count(), 2);
        let full_id = root.page(0).unwrap().0.id();
        let tail_id = root.page(1).unwrap().0.id();

        let (candidate, append) = root.fork_for_append(6, || pool.alloc()).unwrap();
        assert_eq!(append.prior_bytes(), 14);
        assert_eq!(append.final_bytes(), 20);
        assert_eq!(append.copies().len(), 1);
        assert_eq!(append.copies()[0].source().id(), tail_id);
        assert_eq!(append.copies()[0].bytes(), 6);
        assert_ne!(append.copies()[0].target().id(), tail_id);
        assert_eq!(
            append
                .writes()
                .iter()
                .map(|w| (w.offset(), w.bytes()))
                .collect::<Vec<_>>(),
            [(6, 2), (0, 4)]
        );

        assert_eq!(root.page(0).unwrap().0.id(), full_id);
        assert_eq!(candidate.page(0).unwrap().0.id(), full_id);
        assert_eq!(root.page(1).unwrap().0.id(), tail_id);
        assert_eq!(root.page(1).unwrap().1, 6);
        assert_ne!(candidate.page(1).unwrap().0.id(), tail_id);
        assert_eq!(candidate.page(1).unwrap().1, 8);
        assert_eq!(candidate.page(2).unwrap().1, 4);
    }

    #[test]
    fn full_tail_needs_no_copy_and_prefix_tail_is_cowed_at_its_valid_extent() {
        let pool = MockPool::new(8);
        let empty = PagedBlockTable::empty(key(), geometry(8));
        let (root, _) = empty.fork_for_append(24, || pool.alloc()).unwrap();
        let (continued, plan) = root.fork_for_append(8, || pool.alloc()).unwrap();
        assert!(plan.copies().is_empty());
        assert_eq!(continued.page_count(), 4);
        assert_eq!(
            continued.page(0).unwrap().0.id(),
            root.page(0).unwrap().0.id()
        );

        let prefix = root
            .share_prefix(root.compatibility(), root.geometry(), 18)
            .unwrap();
        assert_eq!(prefix.page_count(), 3);
        assert_eq!(prefix.page(2).unwrap().1, 2);
        let old_tail = prefix.page(2).unwrap().0.id();
        let (fork, append) = prefix.fork_for_append(3, || pool.alloc()).unwrap();
        assert_eq!(append.copies().len(), 1);
        assert_eq!(append.copies()[0].bytes(), 2);
        assert_eq!(append.copies()[0].source().id(), old_tail);
        assert_ne!(fork.page(2).unwrap().0.id(), old_tail);
        assert_eq!(fork.page(2).unwrap().1, 5);
    }

    #[test]
    fn incompatible_prefix_identity_or_extent_never_shares_pages() {
        let pool = MockPool::new(4);
        let empty = PagedBlockTable::empty(key(), geometry(8));
        let (root, _) = empty.fork_for_append(12, || pool.alloc()).unwrap();
        let source = SourceIdentity::for_new_allocation().unwrap();
        for wrong in [
            PageCompatibility::new(source, 11, 22, 33).unwrap(),
            PageCompatibility::new(root.compatibility().source(), 12, 22, 33).unwrap(),
            PageCompatibility::new(root.compatibility().source(), 11, 23, 33).unwrap(),
            PageCompatibility::new(root.compatibility().source(), 11, 22, 34).unwrap(),
        ] {
            assert!(root.share_prefix(wrong, geometry(8), 8).is_err());
        }
        assert!(
            root.share_prefix(root.compatibility(), geometry(16), 8)
                .is_err()
        );
        assert!(
            root.share_prefix(root.compatibility(), geometry(8), 13)
                .is_err()
        );
        assert_eq!(root.logical_bytes(), 12);
    }

    #[test]
    fn allocation_exhaustion_is_atomic_and_prior_table_remains_unchanged() {
        let pool = MockPool::new(2);
        let empty = PagedBlockTable::empty(key(), geometry(8));
        let (root, plan) = empty.fork_for_append(6, || pool.alloc()).unwrap();
        drop(plan);
        assert_eq!(pool.live(), 1);
        let before_id = root.page(0).unwrap().0.id();
        let before_live = pool.live();

        // COW of the 6-byte tail plus another page requires two fresh pages, but
        // only one is available. The first fresh handle must roll back on refusal.
        assert!(root.fork_for_append(10, || pool.alloc()).is_err());
        assert_eq!(root.logical_bytes(), 6);
        assert_eq!(root.page_count(), 1);
        assert_eq!(root.page(0).unwrap().0.id(), before_id);
        assert_eq!(pool.live(), before_live);
    }

    #[test]
    fn copy_and_write_plan_pins_owners_until_physical_consumer_retires() {
        let pool = MockPool::new(4);
        let empty = PagedBlockTable::empty(key(), geometry(8));
        let (root, first) = empty.fork_for_append(6, || pool.alloc()).unwrap();
        drop(first);
        let (candidate, plan) = root.fork_for_append(1, || pool.alloc()).unwrap();
        assert_eq!(pool.live(), 2);

        drop(root);
        drop(candidate);
        // Source and target are still retained by the physical copy/write plan.
        assert_eq!(pool.live(), 2);
        drop(plan);
        assert_eq!(pool.live(), 0);
    }

    #[test]
    fn prefix_and_append_cover_only_full_pages_plus_one_valid_tail() {
        let pool = MockPool::new(16);
        let empty = PagedBlockTable::empty(key(), geometry(7));
        let (root, plan) = empty.fork_for_append(25, || pool.alloc()).unwrap();
        assert_eq!(plan.copies().len(), 0);
        assert_eq!(root.page_count(), 4);
        assert_eq!(
            (0..root.page_count())
                .map(|i| root.page(i).unwrap().1)
                .collect::<Vec<_>>(),
            [7, 7, 7, 4]
        );

        let prefix = root
            .share_prefix(root.compatibility(), root.geometry(), 15)
            .unwrap();
        assert_eq!(
            (0..prefix.page_count())
                .map(|i| prefix.page(i).unwrap().1)
                .collect::<Vec<_>>(),
            [7, 7, 1]
        );
        let ids = (0..prefix.page_count())
            .map(|i| prefix.page(i).unwrap().0.id())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), prefix.page_count());
        assert_eq!(
            prefix
                .share_prefix(prefix.compatibility(), prefix.geometry(), 0)
                .unwrap()
                .page_count(),
            0
        );
    }

    #[test]
    fn semantic_unit_alignment_rejects_split_prefixes_and_appends() {
        let pool = MockPool::new(8);
        let geometry = PageGeometry::new(24, 6).unwrap();
        assert_eq!(geometry.page_bytes(), 24);
        assert_eq!(geometry.unit_bytes(), 6);
        let empty = PagedBlockTable::empty(key(), geometry);
        assert!(empty.fork_for_append(5, || pool.alloc()).is_err());
        let (root, _) = empty.fork_for_append(30, || pool.alloc()).unwrap();
        assert_eq!(root.page(0).unwrap().1, 24);
        assert_eq!(root.page(1).unwrap().1, 6);
        assert!(
            root.share_prefix(root.compatibility(), root.geometry(), 25)
                .is_err()
        );
        let prefix = root
            .share_prefix(root.compatibility(), root.geometry(), 18)
            .unwrap();
        assert_eq!(prefix.page_count(), 1);
        assert_eq!(prefix.page(0).unwrap().1, 18);
    }
}
