//! Committed output is not transport delivery, ACK or playback.
pub use zerogravity_core::state::CommittedOutput;
pub use zerogravity_core::task::OutputMilestone;
pub trait OutputStream: Send {
    fn next(&mut self) -> crate::Result<Option<CommittedOutput>>;
    fn cancel(&mut self) -> crate::Result<()>;
}
