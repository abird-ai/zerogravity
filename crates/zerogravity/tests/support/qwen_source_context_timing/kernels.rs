//! Bounded serialized kernel timings on the original runtime; no launch authority.
use super::{Result, need};
use cubecl_runtime::logging::{
    Duration, LaunchObservation, LaunchObserver, TimingMethod, TimingRequest,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};
use zerogravity::gpu::SourceCallFrameKind;

#[derive(Default)]
struct Counts {
    issued: usize,
    timed: usize,
    ns: u128,
}
#[derive(Default)]
struct Measurements {
    entries: BTreeMap<&'static str, Counts>,
    invalid: bool,
}
#[derive(Default)]
struct Observer(Mutex<Measurements>);
impl LaunchObserver for Observer {
    fn launched(&self, kernel: &'static str) {
        let mut data = self.0.lock().unwrap();
        if !data.entries.contains_key(kernel) && data.entries.len() == 16 {
            data.invalid = true;
            return;
        }
        data.entries.entry(kernel).or_default().issued += 1;
    }
    fn timing(&self) -> TimingRequest {
        TimingRequest::Resolved
    }
    fn timed(&self, kernel: &'static str, duration: Duration, method: TimingMethod) {
        let mut data = self.0.lock().unwrap();
        // The runtime reports an unreadable event as 3600s, never as a fast sample.
        if !matches!(method, TimingMethod::Device) || duration.as_secs() >= 3600 {
            data.invalid = true;
            return;
        }
        let Some(entry) = data.entries.get_mut(kernel) else {
            data.invalid = true;
            return;
        };
        entry.timed += 1;
        entry.ns += duration.as_nanos();
    }
}

pub(super) struct Trace<'a> {
    starts: &'a BTreeSet<u32>,
    ends: &'a BTreeSet<u32>,
    observer: Arc<Observer>,
    watching: Option<LaunchObservation>,
    windows: usize,
}
impl<'a> Trace<'a> {
    pub(super) fn new(starts: &'a BTreeSet<u32>, ends: &'a BTreeSet<u32>) -> Self {
        Self {
            starts,
            ends,
            observer: Arc::new(Observer::default()),
            watching: None,
            windows: 0,
        }
    }
    pub(super) fn completed(&mut self, kind: SourceCallFrameKind) -> Result<()> {
        let SourceCallFrameKind::Value(id) = kind else {
            return Ok(());
        };
        // Producer callbacks run outside launch observation, after its output readback.
        if self.ends.contains(&id.0) {
            need(
                self.watching.take().is_some(),
                "attention timing window opened",
            )?;
            self.windows += 1;
        }
        if self.starts.contains(&id.0) {
            need(
                self.watching.is_none(),
                "attention timing windows do not overlap",
            )?;
            self.watching = Some(LaunchObservation::new(self.observer.clone()));
        }
        Ok(())
    }
    pub(super) fn report(&self, context: usize, base: usize, rows: usize) -> Result<()> {
        need(
            self.watching.is_none()
                && self.windows == self.ends.len()
                && self.windows > 0
                && self.starts.len() == self.ends.len(),
            "all original attention timing windows closed",
        )?;
        let data = self.observer.0.lock().unwrap();
        need(
            !data.invalid
                && data
                    .entries
                    .keys()
                    .any(|name| name.contains("source_attention")),
            "valid original attention device kernel timings",
        )?;
        for (kernel, count) in &data.entries {
            need(
                count.issued > 0 && count.issued == count.timed && count.ns > 0,
                "each issued attention kernel has a device timing",
            )?;
            println!(
                "SOURCE_CONTEXT_KERNEL_INTERVAL context={context} base={base} rows={rows} kernel={kernel} launches={} device_ns={} serialized=true sdk=false performance=false",
                count.timed, count.ns
            );
        }
        Ok(())
    }
}
