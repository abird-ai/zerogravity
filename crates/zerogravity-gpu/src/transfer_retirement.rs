//! Bounded host-owner retention on an existing ordered device flight.
//! Marks prove retirement only; they never initialize a lease or publish Ready.
use std::collections::VecDeque;
use zerogravity_core::{Error, Result};

pub(crate) trait RetirementMark: Send + Sync + 'static {
    fn ready(&mut self) -> Result<bool>;
    fn wait(&mut self) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RetirementCounts {
    pub(crate) issued: usize,
    pub(crate) retired: usize,
    pub(crate) queries: usize,
    pub(crate) waits: usize,
    pub(crate) pending: usize,
}

type Quarantine<M, O> = Box<dyn FnOnce(Vec<(M, O)>) + Send + Sync>;

pub(crate) struct TransferRetirementRing<M: RetirementMark, O: Send + Sync + 'static> {
    capacity: usize,
    pending: VecDeque<(M, O)>,
    reusable: Vec<M>,
    counts: RetirementCounts,
    quarantine: Option<Quarantine<M, O>>,
}

impl<M: RetirementMark, O: Send + Sync + 'static> TransferRetirementRing<M, O> {
    /// `quarantine` retains the original service, full physical/charged owners
    /// and these entries on any unresolved Drop. No pending entry is released.
    pub(crate) fn new(
        capacity: usize,
        reusable: Vec<M>,
        quarantine: impl FnOnce(Vec<(M, O)>) + Send + Sync + 'static,
    ) -> Result<Self> {
        if !(1..=8).contains(&capacity) || reusable.len() > capacity {
            return Err(Error::Limit("bounded transfer retirement capacity".into()));
        }
        Ok(Self {
            capacity,
            pending: VecDeque::with_capacity(capacity),
            reusable,
            counts: RetirementCounts::default(),
            quarantine: Some(Box::new(quarantine)),
        })
    }

    /// Called BEFORE allocating/reading the next host source owner. A single
    /// stream orders consumption before the next device overwrite; this bound
    /// separately limits retained host storage and outstanding marks.
    pub(crate) fn make_room(&mut self) -> Result<()> {
        self.require_active()?;
        self.reap_ready()?;
        if self.pending.len() == self.capacity {
            self.wait_oldest()?;
        }
        Ok(())
    }

    pub(crate) fn take_reusable(&mut self) -> Option<M> {
        self.reusable.pop()
    }

    /// The caller installed `owner` on its actual armed flight before enqueue.
    /// A newly recorded mark then transfers that same owner into this ring.
    pub(crate) fn push(&mut self, mark: M, owner: O) -> Result<()> {
        if self.quarantine.is_none() {
            std::mem::forget((mark, owner));
            return Err(Error::State(
                "transfer retirement ring already quarantined".into(),
            ));
        }
        if self.pending.len() >= self.capacity {
            // This is a caller bug after actual enqueue: preserve the incoming
            // mark too, rather than dropping an unknown transfer on refusal.
            let mut all: Vec<_> = self.pending.drain(..).collect();
            all.push((mark, owner));
            if let Some(quarantine) = self.quarantine.take() {
                quarantine(all);
            } else {
                std::mem::forget(all);
            }
            return Err(Error::State("transfer retirement ring exhausted".into()));
        }
        self.pending.push_back((mark, owner));
        self.counts.issued += 1;
        Ok(())
    }

    fn require_active(&self) -> Result<()> {
        if self.quarantine.is_none() {
            return Err(Error::State(
                "transfer retirement ring already quarantined".into(),
            ));
        }
        Ok(())
    }

    fn reap_ready(&mut self) -> Result<()> {
        self.require_active()?;
        while let Some((mark, _)) = self.pending.front_mut() {
            self.counts.queries += 1;
            if !mark.ready()? {
                break;
            }
            let (mark, owner) = self.pending.pop_front().unwrap();
            self.counts.retired += 1;
            self.reusable.push(mark);
            drop(owner);
        }
        Ok(())
    }

    fn wait_oldest(&mut self) -> Result<()> {
        let Some((mark, _)) = self.pending.front_mut() else {
            return Ok(());
        };
        self.counts.waits += 1;
        mark.wait()?;
        // A completed wait must still produce an actual successful query.
        self.counts.queries += 1;
        if !mark.ready()? {
            return Err(Error::State(
                "transfer wait without actual retirement".into(),
            ));
        }
        let (mark, owner) = self.pending.pop_front().unwrap();
        self.counts.retired += 1;
        self.reusable.push(mark);
        drop(owner);
        Ok(())
    }

    /// Explicit debug/fault boundary; production does not drain each page.
    pub(crate) fn drain(&mut self) -> Result<()> {
        self.reap_ready()?;
        while !self.pending.is_empty() {
            self.wait_oldest()?;
        }
        Ok(())
    }

    /// Invoked inside the ORIGINAL successful completion validator. Its
    /// terminal same-stream barrier orders every mark; query each one before
    /// Ready. Pending/error refuses completion and keeps all remaining owners.
    pub(crate) fn confirm_after_completion(&mut self) -> Result<()> {
        self.reap_ready()?;
        if !self.pending.is_empty() || self.counts.retired != self.counts.issued {
            return Err(Error::State(
                "terminal completion precedes transfer retirement".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn counts(&self) -> RetirementCounts {
        RetirementCounts {
            pending: self.pending.len(),
            ..self.counts
        }
    }

    pub(crate) fn owners(&self) -> impl Iterator<Item = &O> {
        self.pending.iter().map(|(_, owner)| owner)
    }

    pub(crate) fn take_retired_marks(&mut self) -> Result<Vec<M>> {
        self.require_active()?;
        if !self.pending.is_empty() || self.counts.retired != self.counts.issued {
            return Err(Error::State(
                "unretired transfer marks cannot return to pool".into(),
            ));
        }
        Ok(std::mem::take(&mut self.reusable))
    }
}

impl<M: RetirementMark, O: Send + Sync + 'static> Drop for TransferRetirementRing<M, O> {
    fn drop(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let pending = self.pending.drain(..).collect();
        if let Some(quarantine) = self.quarantine.take() {
            quarantine(pending);
        } else {
            // Already failed closed. Never free an unresolved host owner.
            std::mem::forget(pending);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    struct Mark {
        ready: Arc<AtomicBool>,
        error: bool,
        wait_completes: bool,
    }
    impl RetirementMark for Mark {
        fn ready(&mut self) -> Result<bool> {
            if self.error {
                return Err(Error::State("actual event error".into()));
            }
            Ok(self.ready.load(Ordering::Acquire))
        }
        fn wait(&mut self) -> Result<()> {
            if self.error {
                return Err(Error::State("actual event wait error".into()));
            }
            self.ready.store(self.wait_completes, Ordering::Release);
            Ok(())
        }
    }
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn mark(ready: bool, wait_completes: bool) -> Mark {
        Mark {
            ready: Arc::new(AtomicBool::new(ready)),
            error: false,
            wait_completes,
        }
    }

    #[test]
    fn full_ring_retires_oldest_before_next_source_allocation() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut ring = TransferRetirementRing::new(2, vec![], |_| panic!("healthy")).unwrap();
        for _ in 0..2 {
            ring.push(mark(false, true), Owner(drops.clone())).unwrap();
        }
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(ring.owners().count(), 2);
        ring.make_room().unwrap();
        assert_eq!(
            ring.counts(),
            RetirementCounts {
                issued: 2,
                retired: 1,
                queries: 2,
                waits: 1,
                pending: 1
            }
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let mut reusable = ring.take_reusable().unwrap();
        assert!(reusable.ready().unwrap());
        ring.drain().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(ring.take_retired_marks().unwrap().len(), 1);
    }

    #[test]
    fn terminal_pending_and_event_errors_preserve_exact_charged_owners() {
        for error in [false, true] {
            let held = Arc::new(Mutex::new(Vec::new()));
            let held_out = held.clone();
            let drops = Arc::new(AtomicUsize::new(0));
            let mut ring =
                TransferRetirementRing::new(2, vec![], move |v| held_out.lock().unwrap().extend(v))
                    .unwrap();
            let mut m = mark(false, false);
            m.error = error;
            ring.push(m, Owner(drops.clone())).unwrap();
            assert!(ring.confirm_after_completion().is_err());
            assert!(ring.take_retired_marks().is_err());
            drop(ring);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert_eq!(held.lock().unwrap().len(), 1);
            held.lock().unwrap().clear();
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn wait_without_query_success_cannot_credit_retirement() {
        let held = Arc::new(Mutex::new(Vec::new()));
        let copy = held.clone();
        let drops = Arc::new(AtomicUsize::new(0));
        let mut ring =
            TransferRetirementRing::new(1, vec![], move |v| copy.lock().unwrap().extend(v))
                .unwrap();
        ring.push(mark(false, false), Owner(drops.clone())).unwrap();
        assert!(ring.make_room().is_err());
        assert_eq!(ring.counts().retired, 0);
        drop(ring);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        held.lock().unwrap().clear();
    }

    #[test]
    fn overflow_preserves_newly_issued_owner_too() {
        let held = Arc::new(Mutex::new(Vec::new()));
        let copy = held.clone();
        let drops = Arc::new(AtomicUsize::new(0));
        let mut ring =
            TransferRetirementRing::new(1, vec![], move |v| copy.lock().unwrap().extend(v))
                .unwrap();
        ring.push(mark(false, false), Owner(drops.clone())).unwrap();
        assert!(ring.push(mark(false, false), Owner(drops.clone())).is_err());
        assert!(ring.make_room().is_err());
        assert!(ring.confirm_after_completion().is_err());
        assert!(ring.take_retired_marks().is_err());
        drop(ring);
        assert_eq!(held.lock().unwrap().len(), 2);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        held.lock().unwrap().clear();
    }

    #[test]
    fn completion_queries_every_mark_then_reuses_bounded_owners() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut ring = TransferRetirementRing::new(2, vec![], |_| panic!("healthy")).unwrap();
        for _ in 0..2 {
            ring.push(mark(true, true), Owner(drops.clone())).unwrap();
        }
        ring.confirm_after_completion().unwrap();
        assert_eq!(
            ring.counts(),
            RetirementCounts {
                issued: 2,
                retired: 2,
                queries: 2,
                waits: 0,
                pending: 0
            }
        );
        assert_eq!(ring.take_retired_marks().unwrap().len(), 2);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}
