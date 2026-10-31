//! Task-stage descriptions; no worker, interpreter or state publisher.
mod media;
pub use crate::targets::TaskKind;
pub use media::{ConvertedTicks, IngressMilestone, MediaClockId, MediaPosition};
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBase {
    pub numerator: NonZeroU64,
    pub denominator: NonZeroU64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskFrontier {
    Tokens(u64),
    Frames { index: u64, time_base: TimeBase },
    Iteration(u64),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageIntent {
    TextPrefill,
    TextDecode,
    Encoder,
    AcousticWindow,
    Codebook,
    Denoise,
    Decision,
    /// Future opt-in parameter-version update, not inference autograd.
    ParameterUpdate,
}
pub trait TaskDriver: Send + Sync {
    fn next_stage(&self, frontier: TaskFrontier) -> crate::Result<Option<StageIntent>>;
}
/// These observations are not interchangeable publication authorities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMilestone {
    Prepared,
    Committed,
    Delivered,
    Acknowledged,
    Played,
}
