//! Pull-only callbacks. No registrations in default CPU/offline/cache commands.
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity::Result;

pub(super) struct CancellationSignals {
    ids: Vec<signal_hook::SigId>,
}
impl CancellationSignals {
    pub(super) fn register(cancel: Arc<AtomicBool>) -> Result<Self> {
        let mut registered = Self {
            ids: Vec::with_capacity(2),
        };
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            registered
                .ids
                .push(signal_hook::flag::register(signal, cancel.clone())?);
        }
        Ok(registered)
    }
}
impl Drop for CancellationSignals {
    fn drop(&mut self) {
        for id in self.ids.drain(..) {
            signal_hook::low_level::unregister(id);
        }
        // signal-hook retains its OS dispatcher. Only our callbacks are
        // removed; this standalone single-command CLI exits after returning.
        // No restoration of arbitrary prior OS dispositions is claimed.
    }
}
