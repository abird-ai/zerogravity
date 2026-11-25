//! Source-committed presentation is not transport delivery, ACK or playback.
pub use crate::router::ResponseEvent;
use crate::router::{RequestEvents, ServiceError, event_channel::mpsc, pending::ServingCharge};
use std::time::Duration;
pub use zerogravity_core::state::CommittedOutput;
pub use zerogravity_core::task::OutputMilestone;

/// Text events come from the original successful session publication. No
/// synthetic `CommittedOutput`/root is minted to adapt the old skeleton API.
pub trait OutputStream: Send {
    fn next(&mut self) -> crate::Result<Option<ResponseEvent>>;
    fn cancel(&mut self) -> crate::Result<()>;

    /// Receive at most one event; timeout bounds receiver waiting, not
    /// synchronous validation/admission at a pipeline stage boundary.
    /// Callers can interleave cancellation with polling. No new event queue.
    fn poll_next(&mut self, _timeout: Duration) -> Result<OutputPoll, ServiceError> {
        Err(ServiceError::new(
            400,
            "unsupported",
            "bounded polling unavailable on this stream",
        ))
    }
}
pub enum OutputPoll {
    Pending,
    Event(ResponseEvent),
    Finished,
}
pub(crate) struct TextOutputStream {
    events: RequestEvents,
    finished: bool,
    _charge: ServingCharge,
}
impl TextOutputStream {
    pub(crate) fn new(events: RequestEvents, charge: ServingCharge) -> Self {
        Self {
            events,
            finished: false,
            _charge: charge,
        }
    }
}
impl OutputStream for TextOutputStream {
    fn next(&mut self) -> crate::Result<Option<ResponseEvent>> {
        next_blocking(self)
    }
    fn poll_next(&mut self, timeout: Duration) -> Result<OutputPoll, ServiceError> {
        if self.finished {
            return Ok(OutputPoll::Finished);
        }
        let result = poll_event(&self.events, timeout);
        self.finished = match &result {
            Ok(OutputPoll::Event(event)) => {
                matches!(event, ResponseEvent::Done { .. } | ResponseEvent::Error(_))
            }
            Ok(OutputPoll::Pending) => false,
            _ => true,
        };
        result
    }
    fn cancel(&mut self) -> crate::Result<()> {
        if !self.finished {
            self.events.cancel();
            self.finished = true;
        }
        Ok(())
    }
}

/// Shared original event receiver. Zero timeout is nonblocking. Clamp
/// individual waits to avoid unbounded waits/Instant overflow in callers.
pub(crate) fn poll_event(
    events: &RequestEvents,
    timeout: Duration,
) -> Result<OutputPoll, ServiceError> {
    match events
        .events
        .recv_timeout(timeout.min(Duration::from_secs(30)))
    {
        Ok(event) => Ok(OutputPoll::Event(event)),
        Err(mpsc::RecvTimeoutError::Timeout) => Ok(OutputPoll::Pending),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(ServiceError::new(
            500,
            "worker_disconnected",
            "text worker disconnected before terminal output",
        )),
    }
}

pub(crate) fn next_blocking(stream: &mut dyn OutputStream) -> crate::Result<Option<ResponseEvent>> {
    loop {
        match stream
            .poll_next(Duration::from_secs(30))
            .map_err(crate::runtime::task_error)?
        {
            OutputPoll::Event(event) => return Ok(Some(event)),
            OutputPoll::Finished => return Ok(None),
            OutputPoll::Pending => continue,
        }
    }
}
