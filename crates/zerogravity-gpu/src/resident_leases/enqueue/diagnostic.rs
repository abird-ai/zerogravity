//! Opt-in device-clock envelopes, never completion or publication authority.
use super::super::{ResidentReady, ResidentService, ResidentSubmission};
use super::EnqueuePlacement;
use cubecl::client::{Client, ProfileWindow};
use cubecl_common::profile::{ProfileDuration, TimingMethod};
use std::{
    cell::RefCell,
    sync::{Arc, Mutex, OnceLock, Weak},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{Error, Result, device_events::DeviceEventId};

pub(crate) const ADMISSION_BYTES: usize = 64 * 1024;
const WHOLE: usize = 0;
const HEAD: usize = 1;
thread_local! {
    static ACTIVE: RefCell<Option<Weak<Measurements>>> = const { RefCell::new(None) };
}
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZG_GPU_ENQUEUE_PROFILE").is_ok_and(|v| v == "1"))
}
fn failure(message: &str) -> Error {
    Error::State(format!("enqueue device observer: {message}"))
}

#[derive(Default)]
struct Slots {
    seen: [bool; 2],
    durations: [Option<ProfileDuration>; 2],
    failed: bool,
    consumed: bool,
}
impl Slots {
    fn begin(&mut self, slot: usize) -> Result<()> {
        if slot >= self.seen.len() || self.failed || self.consumed || self.seen[slot] {
            return Err(failure("duplicate/invalid/finished window"));
        }
        self.seen[slot] = true;
        Ok(())
    }
    fn take(&mut self) -> Result<[ProfileDuration; 2]> {
        if self.failed
            || self.consumed
            || self.seen != [true; 2]
            || self.durations.iter().any(Option::is_none)
        {
            return Err(failure(
                "two complete original windows required exactly once",
            ));
        }
        self.consumed = true;
        Ok([
            self.durations[WHOLE].take().unwrap(),
            self.durations[HEAD].take().unwrap(),
        ])
    }
}

/// Retained by the original flight before either event; closed futures survive
/// unknown completion with that same flight, including partial Head windows.
pub(crate) struct Measurements {
    service: Weak<ResidentService>,
    client: Client,
    event: DeviceEventId,
    root: u64,
    stream: cubecl_core::environment::stream::StreamId,
    slots: Mutex<Slots>,
    _admission: WindowWorkspace,
}
impl Measurements {
    fn fail(&self) {
        if let Ok(mut slots) = self.slots.lock() {
            slots.failed = true;
        }
    }
    fn window(self: &Arc<Self>, slot: usize) -> Result<Window> {
        self.slots
            .lock()
            .map_err(|_| failure("window ledger lock"))?
            .begin(slot)?;
        let token = self.client.profile_start().map_err(|e| {
            self.fail();
            failure(&format!("start: {e}"))
        })?;
        let observer = self.clone();
        let client = self.client.clone();
        let window = Window {
            token: OpenToken::new(
                token,
                Box::new(move |token| {
                    observer.fail();
                    // Preserve an original enqueue error/panic even if transport
                    // abandonment itself panics while unwinding.
                    abandon_safely(|| {
                        client.profile_abandon(token);
                    });
                }),
            ),
            observer: self.clone(),
            slot,
        };
        if token.stream_id != self.stream {
            return Err(failure("actual profile stream differs"));
        }
        Ok(window)
    }
    pub(crate) async fn resolve_after_ready(
        &self,
        ready: &ResidentReady,
        base: usize,
        rows: usize,
    ) -> Result<()> {
        let service = self
            .service
            .upgrade()
            .ok_or_else(|| failure("original service retired"))?;
        if !Arc::ptr_eq(ready.service(), &service)
            || ready.event_ids() != [self.event]
            || ready.expected_root() != self.root
        {
            return Err(failure("measurement needs SAME original physical Ready"));
        }
        service.ensure_usable()?;
        let durations = self
            .slots
            .lock()
            .map_err(|_| failure("window ledger lock"))?
            .take()?;
        let mut ns = [0u128; 2];
        // This caller is after original completion, outside the actor scope.
        // Resolving while enqueueing would wait on the actor's own progress.
        for (i, duration) in durations.into_iter().enumerate() {
            ns[i] = resolve_device(duration).await?;
        }
        service.ensure_usable()?;
        eprintln!(
            "{{\"original_flight_gpu_envelope\":true,\"base\":{base},\"rows\":{rows},\"root\":{},\"whole_ns\":{},\"head_ns\":{},\"whole_windows\":1,\"head_windows\":1,\"device_clock\":true,\"includes_submission_bubbles\":true}}",
            self.root, ns[WHOLE], ns[HEAD]
        );
        Ok(())
    }
}
async fn resolve_device(duration: ProfileDuration) -> Result<u128> {
    if duration.timing_method() != TimingMethod::Device {
        return Err(failure("system timing is not a device measurement"));
    }
    let ticks = duration
        .resolve()
        .await
        .ok_or_else(|| failure("device timing absent"))?;
    valid_duration(ticks.duration())
}
fn abandon_safely(abandon: impl FnOnce()) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(abandon));
}
fn valid_duration(duration: std::time::Duration) -> Result<u128> {
    // Existing EventProfiler reports unreadable events as 3600s, never zero.
    if duration >= std::time::Duration::from_secs(3600) {
        return Err(failure("unreadable device timing sentinel"));
    }
    Ok(duration.as_nanos())
}

/// End consumes the token on either returned result; a panic during end still
/// leaves Drop able to abandon an open token. Successful end never abandons.
struct OpenToken<T: Copy, A: FnMut(T)> {
    token: Option<T>,
    abandon: A,
}
impl<T: Copy, A: FnMut(T)> OpenToken<T, A> {
    fn new(token: T, abandon: A) -> Self {
        Self {
            token: Some(token),
            abandon,
        }
    }
    fn finish<D>(&mut self, end: impl FnOnce(T) -> Result<D>) -> Result<D> {
        let token = self.token.ok_or_else(|| failure("window already ended"))?;
        let result = end(token);
        self.token = None;
        result
    }
}
impl<T: Copy, A: FnMut(T)> Drop for OpenToken<T, A> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            (self.abandon)(token);
        }
    }
}
type Abandon = Box<dyn FnMut(ProfileWindow)>;
struct Window {
    token: OpenToken<ProfileWindow, Abandon>,
    observer: Arc<Measurements>,
    slot: usize,
}
impl Window {
    fn finish(mut self) -> Result<()> {
        let duration = self
            .token
            .finish(|token| {
                self.observer
                    .client
                    .profile_end(token)
                    .map_err(|e| failure(&format!("end: {e}")))
            })
            .inspect_err(|_| self.observer.fail())?;
        if duration.timing_method() != TimingMethod::Device {
            self.observer.fail();
            return Err(failure("device profiling unavailable"));
        }
        let mut slots = self
            .observer
            .slots
            .lock()
            .map_err(|_| failure("window ledger lock"))?;
        if slots.failed || slots.durations[self.slot].is_some() {
            return Err(failure("window changed before close"));
        }
        slots.durations[self.slot] = Some(duration);
        Ok(())
    }
}
struct Active;
impl Active {
    fn enter(observer: &Arc<Measurements>) -> Result<Self> {
        ACTIVE.with(|active| {
            let mut active = active
                .try_borrow_mut()
                .map_err(|_| failure("nested observer borrow"))?;
            if active.is_some() {
                return Err(failure("nested original-flight observer"));
            }
            *active = Some(Arc::downgrade(observer));
            Ok(Self)
        })
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            *active.borrow_mut() = None;
        });
    }
}

pub(crate) fn head<T>(enqueue: impl FnOnce() -> Result<T>) -> Result<T> {
    let observer = ACTIVE.with(|active| active.borrow().clone());
    let Some(observer) = observer else {
        return enqueue();
    };
    let observer = observer
        .upgrade()
        .ok_or_else(|| failure("active original-flight observer lost"))?;
    let window = observer.window(HEAD)?;
    let output = enqueue()?;
    window.finish()?;
    Ok(output)
}
impl ResidentSubmission<'_> {
    pub(crate) fn observe_ordered_enqueue<T: Send + 'static>(
        &mut self,
        placement: EnqueuePlacement,
        admission: Option<WindowWorkspace>,
        enqueue: impl FnOnce(&mut Self) -> Result<T> + Send,
    ) -> Result<(T, Option<Arc<Measurements>>)> {
        let Some(admission) = admission else {
            return enqueue(self).map(|out| (out, None));
        };
        self.require_accesses(std::iter::empty())?;
        if placement != EnqueuePlacement::DeviceActor
            || admission.bytes() != ADMISSION_BYTES
            || admission.source().source_identity() != self.service.source_identity()
        {
            return Err(failure("actual actor/source/host admission differs"));
        }
        let queue = self.service.pinned_cuda_queue()?;
        if queue.service_id() != self.service.client().service_id() {
            return Err(failure("actual original native/client service differs"));
        }
        // Only new_pinned_cuda installs Some: it stores the Client returned by
        // pin_cold with this SAME queue. Neither private field is later replaced.
        // Native launch and profile resolve that pinned logical stream through
        // the same server stream registry. Check the returned stamp as well.
        let observer = Arc::new(Measurements {
            service: Arc::downgrade(self.service),
            client: self.service.client().clone(),
            event: self.event,
            root: self.expected_root,
            stream: queue.logical_stream_id(),
            slots: Mutex::new(Slots::default()),
            _admission: admission,
        });
        self.retain_completion_owner(observer.clone())?;
        let out = self.ordered_enqueue(|flight| {
            let _active = Active::enter(&observer)?;
            let window = observer.window(WHOLE)?;
            let out = enqueue(flight)?;
            window.finish()?;
            Ok(out)
        })?;
        Ok((out, Some(observer)))
    }
}

#[cfg(test)]
mod tests;
