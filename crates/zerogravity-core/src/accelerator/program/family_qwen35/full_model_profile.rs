//! Exact original source-family geometry only; not workload/optimization policy.
//! SourceIdentity and ProgramNumericalProfile/provider semantics stay independent.
//! No source/runtime/state authority or learned-source substitution.
use super::Vllm08Projection;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullModelProfile {
    Qwen08,
    Qwen2B,
}
impl FullModelProfile {
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
    pub fn projection_input(self, kind: Vllm08Projection) -> usize {
        match kind {
            Vllm08Projection::GdnOut | Vllm08Projection::AttentionOut => 2048,
            Vllm08Projection::Down => self.intermediate_size(),
            _ => self.hidden_size(),
        }
    }
    pub fn projection_output(self, kind: Vllm08Projection) -> usize {
        match kind {
            Vllm08Projection::Qkvz => 8192,
            Vllm08Projection::Ba => 32,
            Vllm08Projection::AttentionQkv => 5120,
            Vllm08Projection::GateUp => self.intermediate_size() * 2,
            Vllm08Projection::Head => 248320,
            _ => self.hidden_size(),
        }
    }
}
