//! Scheduler observations; never source, completion or publication authority.
use crate::model_instance::execution::Completion;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[derive(Default)]
pub struct WorkStats {
    pub(crate) retirement_failed: AtomicBool,
    pub(crate) attempted: AtomicU64,
    pub(crate) completed: AtomicU64,
    pub(crate) gpu_forwards: AtomicU64,
    pub(crate) failed: AtomicU64,
    pub(crate) cancelled: AtomicU64,
    pub(crate) active: AtomicUsize,
    pub(crate) peak: AtomicUsize,
    pub(crate) groups: AtomicU64,
    pub(crate) co_scheduled: AtomicU64,
    pub(crate) cohort_publications: AtomicU64,
    pub(crate) cohort_rows: AtomicU64,
    pub(crate) kernel_dispatches: AtomicU64,
    pub(crate) model_tokens: AtomicU64,
    pub(crate) committed_roots: AtomicU64,
    pub(crate) fused_batches: AtomicU64,
    pub(crate) fused_state_only_batches: AtomicU64,
    pub(crate) fused_state_only_rows: AtomicU64,
    pub(crate) fused_greedy_batches: AtomicU64,
    pub(crate) native_bands: [AtomicU64; 3],
    pub(crate) native_admission_refusals: AtomicU64,
    pub(crate) fused_rows: AtomicU64,
    pub(crate) native_projections: AtomicU64,
    pub(crate) native_sampler: AtomicU64,
    pub(crate) discarded_cohorts: AtomicU64,
    pub(crate) output_tokens: AtomicU64,
    pub(crate) parked: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub attempted: u64,
    pub completed: u64,
    pub gpu_forwards: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub active: usize,
    pub peak: usize,
    pub groups: u64,
    pub co_scheduled: u64,
    pub cohort_publications: u64,
    pub cohort_rows: u64,
    pub kernel_dispatches: u64,
    pub model_tokens: u64,
    pub committed_roots: u64,
    pub fused_batches: u64,
    pub fused_state_only_batches: u64,
    pub fused_state_only_rows: u64,
    pub fused_greedy_batches: u64,
    pub native_bands: [u64; 3],
    pub native_admission_refusals: u64,
    pub fused_rows: u64,
    pub native_projections: u64,
    pub native_sampler: u64,
    pub discarded_cohorts: u64,
    pub output_tokens: u64,
    pub parked: usize,
}
impl WorkStats {
    pub fn snapshot(&self) -> Counters {
        Counters {
            attempted: self.attempted.load(Ordering::Acquire),
            completed: self.completed.load(Ordering::Acquire),
            gpu_forwards: self.gpu_forwards.load(Ordering::Acquire),
            failed: self.failed.load(Ordering::Acquire),
            cancelled: self.cancelled.load(Ordering::Acquire),
            active: self.active.load(Ordering::Acquire),
            peak: self.peak.load(Ordering::Acquire),
            groups: self.groups.load(Ordering::Acquire),
            co_scheduled: self.co_scheduled.load(Ordering::Acquire),
            cohort_publications: self.cohort_publications.load(Ordering::Acquire),
            cohort_rows: self.cohort_rows.load(Ordering::Acquire),
            kernel_dispatches: self.kernel_dispatches.load(Ordering::Acquire),
            model_tokens: self.model_tokens.load(Ordering::Acquire),
            committed_roots: self.committed_roots.load(Ordering::Acquire),
            fused_batches: self.fused_batches.load(Ordering::Acquire),
            fused_state_only_batches: self.fused_state_only_batches.load(Ordering::Acquire),
            fused_state_only_rows: self.fused_state_only_rows.load(Ordering::Acquire),
            fused_greedy_batches: self.fused_greedy_batches.load(Ordering::Acquire),
            native_bands: std::array::from_fn(|i| self.native_bands[i].load(Ordering::Relaxed)),
            native_admission_refusals: self.native_admission_refusals.load(Ordering::Acquire),
            fused_rows: self.fused_rows.load(Ordering::Acquire),
            native_projections: self.native_projections.load(Ordering::Acquire),
            native_sampler: self.native_sampler.load(Ordering::Acquire),
            discarded_cohorts: self.discarded_cohorts.load(Ordering::Acquire),
            output_tokens: self.output_tokens.load(Ordering::Acquire),
            parked: self.parked.load(Ordering::Acquire),
        }
    }
    pub(crate) fn record(&self, c: Completion) {
        self.gpu_forwards.fetch_add(c.forwards, Ordering::Relaxed);
        self.kernel_dispatches
            .fetch_add(c.dispatches, Ordering::Relaxed);
        self.model_tokens
            .fetch_add(c.model_tokens, Ordering::Relaxed);
        self.committed_roots
            .fetch_add(c.committed_roots, Ordering::Relaxed);
        if c.native_rows != 0 {
            self.fused_batches.fetch_add(1, Ordering::Relaxed);
            self.fused_rows
                .fetch_add(c.native_rows as u64, Ordering::Relaxed);
            self.native_projections
                .fetch_add(c.native_projections, Ordering::Relaxed);
            self.native_sampler
                .fetch_add(c.native_sampler, Ordering::Relaxed);
            if c.state_only {
                self.fused_state_only_batches
                    .fetch_add(1, Ordering::Relaxed);
                self.fused_state_only_rows
                    .fetch_add(c.native_rows as u64, Ordering::Relaxed);
            } else {
                self.fused_greedy_batches.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(i) = [2, 4, 8].iter().position(|&n| n == c.native_rows) {
                self.native_bands[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
