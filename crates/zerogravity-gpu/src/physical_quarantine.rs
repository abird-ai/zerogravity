//! P07-F: process-lifetime quarantine for uncertain physical GPU work.
//!
//! CubeCL's managed handles are tied to a *device service* and may remain
//! referenced by a kernel after an async caller drops its future. On an
//! unverified completion, retain ALL source, dynamic and output handles in
//! a process-owned quarantine, even if the provider itself is dropped.
//!
//! No other provider may submit work to a quarantined service: its memory
//! correctness cannot be established without a trusted physical barrier.
//! We intentionally do NOT expose a "clear on timeout" or unsafe
//! "just retry" escape hatch. Reclamation requires a future explicitly
//! proved device-service teardown/recovery path; today, process exit is the
//! conservative release boundary.
//!
//! This module does not implement a CUDA/ROCm/Vulkan physical recovery path.

use cubecl::{client::Client, server::Handle};
use std::sync::{Mutex, OnceLock};
use zerogravity_core::Error;
use zerogravity_core::Result;

/// The Client retains the original service AND stream identity alongside the
/// buffers, which are never transferred to another device for reclamation.
struct PhysicalQuarantine {
    client: Client,
    handles: Vec<Handle>,
}

/// Exact same owned-value retention policy can be exercised on host-only
/// DropSpy objects; production stores cloned CubeCL physical handles.
struct Quarantine<T> {
    items: Mutex<Vec<T>>,
}
impl<T> Quarantine<T> {
    fn new() -> Self {
        Self {
            items: Mutex::new(Vec::new()),
        }
    }
    fn retain(&self, value: T) {
        self.items
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(value);
    }
    fn with_items<R>(&self, function: impl FnOnce(&[T]) -> R) -> Result<R> {
        let entries = self
            .items
            .lock()
            .map_err(|_| Error::State("GPU physical quarantine registry poisoned".into()))?;
        Ok(function(&entries))
    }
}

static UNCERTAIN: OnceLock<Quarantine<PhysicalQuarantine>> = OnceLock::new();
fn registry() -> &'static Quarantine<PhysicalQuarantine> {
    UNCERTAIN.get_or_init(Quarantine::new)
}

/// A device service is unusable once ANY of its outstanding work has unknown
/// physical completion. Checked BEFORE allocating/submitting on a new
/// provider as well as before source cache reuse.
pub fn ensure_service_safe(client: &Client) -> Result<()> {
    // Fence the submitting service AND any service whose handle was
    // accidentally mixed into a cancelled launch. Retention wins over reuse.
    let fenced = registry().with_items(|records| {
        records.iter().any(|r| {
            r.client.service_id() == client.service_id()
                || r.handles.iter().any(|h| h.service == client.service_id())
        })
    })?;
    if fenced {
        return Err(Error::State(
            "GPU device service has uncertain in-flight work; restart/trusted recovery required"
                .into(),
        ));
    }
    Ok(())
}

/// Called only from a failing/aborted in-flight RAII Drop. The handles stay
/// alive even when the original source provider and task are destroyed.
/// A poisoned global mutex still owns its entries; recover its guard to
/// conservatively add more keepalives without losing physical resources.
pub fn quarantine_uncertain(client: Client, handles: Vec<Handle>) {
    if handles.is_empty() {
        return;
    }
    // Retain every handle even in the presence of a mistaken mixed-service
    // submission. The global fence covers every involved service.
    registry().retain(PhysicalQuarantine { client, handles });
}

/// Debug/diagnostic only: number of physically retained handle references,
/// NOT completed kernels and NOT a request to release physical memory.
pub fn quarantined_handles_for(client: &Client) -> Result<usize> {
    registry().with_items(|records| {
        records
            .iter()
            .filter(|r| {
                r.client.service_id() == client.service_id()
                    || r.handles.iter().any(|h| h.service == client.service_id())
            })
            .map(|r| r.handles.len())
            .sum()
    })
}

#[cfg(test)]
mod tests {
    use super::Quarantine;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    /// Same retention semantics as the process-owned physical table, tested
    /// without constructing a fake CUDA/Vulkan Handle or claiming GPU work.
    struct DropCount(Arc<AtomicUsize>);
    impl Drop for DropCount {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn uncertain_handle_retention_outlives_request_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        let quarantine = Quarantine::new();
        {
            let resources = vec![DropCount(drops.clone()), DropCount(drops.clone())];
            quarantine.retain(resources);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(
            quarantine.with_items(|records| records[0].len()).unwrap(),
            2
        );
        drop(quarantine);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}
