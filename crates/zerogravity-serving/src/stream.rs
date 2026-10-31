//! Transport framing does not advance the logical model root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamFormat {
    Json,
    ServerSentEvents,
}
pub use zerogravity_engine::output::OutputMilestone;
