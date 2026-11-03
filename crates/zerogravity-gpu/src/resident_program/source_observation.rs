//! Read-only completed source-call frames. No operand, root or execution authority.
use zerogravity_core::{
    Result,
    accelerator::program::{StateSlot, ValueId},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceCallFrameKind {
    Value(ValueId),
    State {
        slot: StateSlot,
        plane: usize,
        prior: bool,
    },
}
#[derive(Clone, Copy, Debug)]
pub struct SourceCallFrame {
    pub base: usize,
    pub rows: usize,
    pub kind: SourceCallFrameKind,
    pub offset_words: usize,
    pub total_words: usize,
}
pub trait SourceCallObserver: Send {
    fn wants(&self, kind: SourceCallFrameKind) -> bool;
    fn completed(&mut self, frame: SourceCallFrame, actual: &[u8]) -> Result<()>;
}
