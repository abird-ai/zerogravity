//! Per-server transport counters, separate from engine physical diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServerMetrics {
    pub accepted_requests: u64,
    pub active_connections: usize,
}

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[derive(Default)]
pub(crate) struct Counters {
    accepted: AtomicU64,
    active: AtomicUsize,
}
impl Counters {
    pub(crate) fn accept_request(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn snapshot(&self) -> ServerMetrics {
        ServerMetrics {
            accepted_requests: self.accepted.load(Ordering::Relaxed),
            active_connections: self.active.load(Ordering::Acquire),
        }
    }
    pub(crate) fn connection(&self) -> ActiveConnection<'_> {
        self.active.fetch_add(1, Ordering::AcqRel);
        ActiveConnection(self)
    }
}
pub(crate) struct ActiveConnection<'a>(&'a Counters);
impl Drop for ActiveConnection<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}
