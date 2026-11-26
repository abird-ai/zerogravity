//! Registration-scoped observations of already committed text output.
//! No extra per-token clock, allocation, request registry or transport grant.
use std::{
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

fn nanos(from: Instant, to: Instant) -> Option<u64> {
    to.checked_duration_since(from)
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
}
fn rate(tokens: u64, ns: u64) -> Option<f64> {
    (tokens != 0 && ns != 0).then(|| tokens as f64 * 1e9 / ns as f64)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct RequestPerformance {
    /// Engine acquire to first committed non-EOS token, not wire delivery.
    pub ttft_ns: Option<u64>,
    pub output_tokens: u64,
    pub decode_intervals: u64,
    pub decode_ns: u64,
}
impl RequestPerformance {
    pub fn ttft_ms(self) -> Option<f64> {
        self.ttft_ns.map(|ns| ns as f64 / 1e6)
    }
    pub fn decode_tps(self) -> Option<f64> {
        rate(self.decode_intervals, self.decode_ns)
    }
}

/// Exact registration lifetime. Counters may advance between observations.
/// Decode aggregates include valid observed intervals from active and ended
/// requests; TTFT includes requests that reached their first output token.
/// Completion is the original terminal receipt, not delivery to a socket.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ModelPerformance {
    pub loaded_ms: u64,
    pub output_tokens: u64,
    pub completed_requests: u64,
    pub ttft_samples: u64,
    pub ttft_total_ns: u64,
    pub decode_intervals: u64,
    pub decode_total_ns: u64,
    pub last_completed: Option<RequestPerformance>,
}
impl ModelPerformance {
    pub fn mean_ttft_ms(self) -> Option<f64> {
        (self.ttft_samples != 0).then(|| self.ttft_total_ns as f64 / self.ttft_samples as f64 / 1e6)
    }
    pub fn decode_tps(self) -> Option<f64> {
        rate(self.decode_intervals, self.decode_total_ns)
    }
}

/// Owned by the existing ModelRead. No request text or model handle.
pub(super) struct RequestTiming {
    start: Instant,
    last: Option<Instant>,
    sample: RequestPerformance,
    completed: bool,
}
impl RequestTiming {
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            last: None,
            sample: RequestPerformance::default(),
            completed: false,
        }
    }
}

#[derive(Default)]
struct Finished {
    count: u64,
    last: Option<RequestPerformance>,
}

/// The original route registration owns these fixed-size counters. The
/// registry's existing sizeof-based reservation prices the added fields.
pub(super) struct RegistrationPerformance {
    loaded: Instant,
    output_tokens: AtomicU64,
    ttft_samples: AtomicU64,
    ttft_total_ns: AtomicU64,
    decode_intervals: AtomicU64,
    decode_total_ns: AtomicU64,
    finished: Mutex<Finished>,
}
impl RegistrationPerformance {
    pub fn new(loaded: Instant) -> Self {
        Self {
            loaded,
            output_tokens: AtomicU64::new(0),
            ttft_samples: AtomicU64::new(0),
            ttft_total_ns: AtomicU64::new(0),
            decode_intervals: AtomicU64::new(0),
            decode_total_ns: AtomicU64::new(0),
            finished: Mutex::new(Finished::default()),
        }
    }
    pub fn observe(&self, request: &mut RequestTiming, committed: Instant) {
        if request.completed {
            return;
        }
        let sample = &mut request.sample;
        if let Some(last) = request.last {
            if let Some(ns) = nanos(last, committed) {
                sample.decode_intervals += 1;
                sample.decode_ns = sample.decode_ns.saturating_add(ns);
                self.decode_total_ns.fetch_add(ns, Relaxed);
                self.decode_intervals.fetch_add(1, Relaxed);
            }
        } else if let Some(ns) = nanos(request.start, committed) {
            sample.ttft_ns = Some(ns);
            self.ttft_total_ns.fetch_add(ns, Relaxed);
            self.ttft_samples.fetch_add(1, Relaxed);
        }
        request.last = Some(committed);
        sample.output_tokens += 1;
        self.output_tokens.fetch_add(1, Relaxed);
    }
    pub fn complete(&self, request: &mut RequestTiming) {
        if std::mem::replace(&mut request.completed, true) {
            return;
        }
        let mut finished = self
            .finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        finished.count = finished.count.saturating_add(1);
        finished.last = Some(request.sample);
    }
    pub fn snapshot(&self, now: Instant) -> ModelPerformance {
        let finished = self
            .finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ModelPerformance {
            loaded_ms: now
                .checked_duration_since(self.loaded)
                .map_or(0, |d| d.as_millis().min(u128::from(u64::MAX)) as u64),
            output_tokens: self.output_tokens.load(Relaxed),
            completed_requests: finished.count,
            ttft_samples: self.ttft_samples.load(Relaxed),
            ttft_total_ns: self.ttft_total_ns.load(Relaxed),
            decode_intervals: self.decode_intervals.load(Relaxed),
            decode_total_ns: self.decode_total_ns.load(Relaxed),
            last_completed: finished.last,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn interleaved_requests_keep_their_own_clocks_and_samples() {
        let start = Instant::now();
        let p = RegistrationPerformance::new(start);
        let at = |ms| start + Duration::from_millis(ms);
        assert_eq!(p.snapshot(at(0)).decode_tps(), None);
        assert_eq!(p.snapshot(at(0)).mean_ttft_ms(), None);
        let mut a = RequestTiming::new(at(10));
        let mut b = RequestTiming::new(at(60));
        p.observe(&mut a, at(110));
        p.observe(&mut b, at(260));
        p.observe(&mut a, at(310));
        p.observe(&mut b, at(360));
        let observed = p.snapshot(at(400));
        assert_eq!(observed.output_tokens, 4);
        assert_eq!(observed.ttft_samples, 2);
        assert_eq!(observed.mean_ttft_ms(), Some(150.0));
        assert_eq!(observed.decode_intervals, 2);
        assert_eq!(observed.decode_tps(), Some(2.0 * 1e9 / 300_000_000.0));
        assert_eq!(observed.completed_requests, 0);
        p.complete(&mut a);
        p.complete(&mut a);
        // An incomplete/cancelled run contributes committed observations
        // but never a successful completion.
        let _ = b;
        let observed = p.snapshot(at(500));
        assert_eq!(observed.loaded_ms, 500);
        assert_eq!(observed.completed_requests, 1);
        let last = observed.last_completed.unwrap();
        assert_eq!(last.ttft_ms(), Some(100.0));
        assert_eq!(last.decode_tps(), Some(5.0));
        let mut empty = RequestTiming::new(at(600));
        p.complete(&mut empty);
        let observed = p.snapshot(at(700));
        assert_eq!(observed.ttft_samples, 2);
        assert_eq!(observed.last_completed.unwrap().decode_tps(), None);
        assert_eq!(observed.last_completed.unwrap().ttft_ms(), None);
    }
}
