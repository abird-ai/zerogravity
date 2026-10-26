//! Opt-in host wall diagnostics. Submission wall is never GPU execution time.
use std::time::Instant;
pub(super) const READ: usize = 0;
pub(super) const VALIDATE: usize = 1;
pub(super) const UPLOAD: usize = 2;
pub(super) const KERNEL: usize = 3;
pub(super) const CHECKPOINT: usize = 4;
pub(super) struct HeadTiming {
    start: Option<Instant>,
    ns: [u128; 5],
    resident: bool,
    rows: usize,
    audit: bool,
    pub(super) uploaded: usize,
    pub(super) native: usize,
    pub(super) checkpoints: usize,
    pub(super) complete: bool,
    pub(super) retirement: crate::transfer_retirement::RetirementCounts,
}
impl HeadTiming {
    pub(super) fn new(resident: bool, rows: usize, audit: bool) -> Self {
        Self {
            start: (std::env::var_os("ZG_QWEN35_HOST_TIMING").is_some_and(|v| v == "1"))
                .then(Instant::now),
            ns: [0; 5],
            resident,
            rows,
            audit,
            uploaded: 0,
            native: 0,
            checkpoints: 0,
            complete: false,
            retirement: Default::default(),
        }
    }
    pub(super) fn start(&self) -> Option<Instant> {
        self.start.map(|_| Instant::now())
    }
    pub(super) fn record(&mut self, phase: usize, start: Option<Instant>) {
        if let Some(start) = start {
            self.ns[phase] += start.elapsed().as_nanos();
        }
    }
}
impl Drop for HeadTiming {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            eprintln!(
                "text_head_perf mode={} rows={} audit={} complete={} whole_ns={} read_rows_ns={} validate_ns={} h2d_submit_ns={} kernel_submit_ns={} checkpoint_ns={} h2d_bytes={} native_launches={} checkpoints={} retirement_issued={} retirement_retired={} retirement_queries={} retirement_waits={} retirement_pending={}",
                if self.resident {
                    "resident"
                } else {
                    "streamed"
                },
                self.rows,
                self.audit,
                self.complete,
                start.elapsed().as_nanos(),
                self.ns[READ],
                self.ns[VALIDATE],
                self.ns[UPLOAD],
                self.ns[KERNEL],
                self.ns[CHECKPOINT],
                self.uploaded,
                self.native,
                self.checkpoints,
                self.retirement.issued,
                self.retirement.retired,
                self.retirement.queries,
                self.retirement.waits,
                self.retirement.pending
            );
        }
    }
}
