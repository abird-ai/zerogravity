//! Source-committed presentation is not transport delivery, ACK or playback.
pub use crate::router::ResponseEvent;
use crate::router::{RequestEvents, event_channel::mpsc, pending::ServingCharge};
use std::time::Duration;
pub use zerogravity_core::state::CommittedOutput;
pub use zerogravity_core::task::OutputMilestone;

/// Text events come from the original successful session publication. No
/// synthetic `CommittedOutput`/root is minted to adapt the old skeleton API.
pub trait OutputStream: Send {
    fn next(&mut self) -> crate::Result<Option<ResponseEvent>>;
    fn cancel(&mut self) -> crate::Result<()>;
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
        if self.finished {
            return Ok(None);
        }
        loop {
            match self.events.events.recv_timeout(Duration::from_secs(30)) {
                Ok(event) => {
                    self.finished =
                        matches!(&event, ResponseEvent::Done { .. } | ResponseEvent::Error(_));
                    return Ok(Some(event));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.finished = true;
                    return Err(crate::Error::State(
                        "text worker disconnected before terminal output".into(),
                    ));
                }
            }
        }
    }
    fn cancel(&mut self) -> crate::Result<()> {
        self.events.cancel();
        self.finished = true;
        Ok(())
    }
}
