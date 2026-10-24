//! Explicit CPU diagnostic contract and borrowed producer observations.
use super::*;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceTextArithmetic {
    AnalyticF32,
    /// Eager cast schedule candidate. GEMM/BMM/libm parity remains unqualified.
    Bf16EagerCandidate,
}
impl ReferenceTextArithmetic {
    pub fn precision(self) -> ActivationPrecision {
        match self {
            Self::AnalyticF32 => ActivationPrecision::F32,
            Self::Bf16EagerCandidate => ActivationPrecision::Bf16WidenedF32,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceTextOperation {
    Embedding,
    Projection,
    Rms,
    Elementwise,
    Rotary,
    Qk,
    Softmax,
    Pv,
    Pack,
    Boundary,
}
pub struct ReferenceTextObservation<'a> {
    pub source: SourceIdentity,
    pub position: usize,
    pub layer: Option<usize>,
    pub arithmetic: ReferenceTextArithmetic,
    pub operation: ReferenceTextOperation,
    pub label: &'a str,
    pub values: &'a [f32],
}
/// Expected-only output consumer. It cannot alter a producer or obtain a state writer.
/// Implementations must admit retained copies before allocation; streaming is preferable.
pub trait ReferenceTextObserver {
    fn output(&self, observation: ReferenceTextObservation<'_>) -> Result<()>;
}
