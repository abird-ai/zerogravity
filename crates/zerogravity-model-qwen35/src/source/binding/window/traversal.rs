//! Source-profile state and observation seams shared by recording and execution.
//! No fixture schedule, root publication or backend eligibility.

#[derive(Clone)]
pub(crate) enum SourceLayerState<V> {
    Gdn {
        history: V,
        full_history: Option<V>,
        state: V,
    },
    Attention {
        keys: Option<V>,
        values: Option<V>,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct SourceOutput {
    pub(crate) layers: usize,
    pub(crate) logits: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceStage {
    InputNorm,
    Qkv,
    Z,
    B,
    A,
    Core,
    NormGate,
    Gated,
    GdnOut,
    PostAttention,
    PostNorm,
    FfnGate,
    FfnUp,
    FfnGated,
    FfnDown,
}

pub(crate) trait SourceObserver<V> {
    fn embedding(&mut self, _: &V) {}
    fn stage(&mut self, _: usize, _: SourceStage, _: &V) {}
    fn layer(&mut self, _: usize, _: &V) {}
    fn final_norm(&mut self, _: &V) {}
}

impl<V> SourceObserver<V> for () {}
