//! Previous switch: ZG_SHORT_HF_PERF.
//! Opt-in host wall only; no device queries, synchronization or success authority.
use std::{io::Write, sync::OnceLock, time::Instant};

pub(crate) struct Phases {
    scope: &'static str,
    base: u64,
    rows: usize,
    started: Option<Instant>,
    last: Option<Instant>,
    phases: [(&'static str, u64); 16],
    len: usize,
    succeeded: bool,
}
impl Phases {
    pub(crate) fn new(scope: &'static str, base: u64, rows: usize) -> Self {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        let enabled =
            *ENABLED.get_or_init(|| std::env::var("ZG_QWEN35_HOST_TIMING").is_ok_and(|v| v == "1"));
        let started = enabled.then(Instant::now);
        Self {
            scope,
            base,
            rows,
            started,
            last: started,
            phases: [("", 0); 16],
            len: 0,
            succeeded: false,
        }
    }
    pub(crate) fn mark(&mut self, label: &'static str) {
        if let Some(last) = self.last {
            let now = Instant::now();
            if self.len < self.phases.len() {
                self.phases[self.len] = (
                    label,
                    now.duration_since(last).as_nanos().min(u64::MAX as u128) as u64,
                );
                self.len += 1;
            }
            self.last = Some(now);
        }
    }
    pub(crate) fn succeed(&mut self) {
        self.succeeded = true;
    }
}
impl Drop for Phases {
    fn drop(&mut self) {
        let Some(started) = self.started else {
            return;
        };
        let mut out = std::io::stderr().lock();
        let _ = write!(
            out,
            "{{\"qwen35_host_wall\":\"{}\",\"cached_tokens\":{},\"input_tokens\":{},\"returned_ok\":{},\"total_ns\":{}",
            self.scope,
            self.base,
            self.rows,
            self.succeeded,
            started.elapsed().as_nanos()
        );
        for (label, ns) in &self.phases[..self.len] {
            let _ = write!(out, ",\"{label}_ns\":{ns}");
        }
        let _ = writeln!(out, "}}");
    }
}
