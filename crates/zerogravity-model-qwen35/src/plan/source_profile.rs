//! Exact original source-family geometry only; not workload/optimization policy.
//! SourceIdentity and ProgramNumericalProfile/provider semantics stay independent.
//! No source/runtime/state authority or learned-source substitution.
use super::QwenProjection;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenSourceProfile {
    Qwen08,
    Qwen2B,
}
impl QwenSourceProfile {
    pub fn hidden_size(self) -> usize {
        match self {
            Self::Qwen08 => 1024,
            Self::Qwen2B => 2048,
        }
    }
    pub fn intermediate_size(self) -> usize {
        match self {
            Self::Qwen08 => 3584,
            Self::Qwen2B => 6144,
        }
    }
    pub fn projection_input(self, kind: QwenProjection) -> usize {
        match kind {
            QwenProjection::GdnOut | QwenProjection::AttentionOut => 2048,
            QwenProjection::Down => self.intermediate_size(),
            _ => self.hidden_size(),
        }
    }
    pub fn projection_output(self, kind: QwenProjection) -> usize {
        match kind {
            QwenProjection::Qkvz => 8192,
            QwenProjection::Ba => 32,
            QwenProjection::AttentionQkv => 5120,
            QwenProjection::GateUp => self.intermediate_size() * 2,
            QwenProjection::Head => 248320,
            _ => self.hidden_size(),
        }
    }
}
