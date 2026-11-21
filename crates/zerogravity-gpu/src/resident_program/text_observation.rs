//! Read-only completed text-invocation frames. No operand, root or execution authority.
use zerogravity_core::{
    Result,
    accelerator::program::{StateSlot, ValueId},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextFrameKind {
    Value(ValueId),
    State {
        slot: StateSlot,
        plane: usize,
        prior: bool,
    },
}
#[derive(Clone, Copy, Debug)]
pub struct TextFrame {
    pub base: usize,
    pub rows: usize,
    pub kind: TextFrameKind,
    pub offset_words: usize,
    pub total_words: usize,
}
pub trait TextObserver: Send {
    fn wants(&self, kind: TextFrameKind) -> bool;
    fn completed(&mut self, frame: TextFrame, actual: &[u8]) -> Result<()>;
}
