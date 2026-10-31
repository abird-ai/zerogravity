//! Per-server transport counters, separate from engine physical diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServerMetrics {
    pub accepted_requests: u64,
    pub active_connections: usize,
}
