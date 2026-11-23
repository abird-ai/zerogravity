//! Model-neutral physical attention provider selection.
//!
//! The model plan describes attention semantics. This value chooses which
//! already-implemented physical provider executes prefill and decode. It owns no
//! model, state, queue or admission authority.

use zerogravity_core::accelerator::program::InvocationPhase;

/// Physical implementation identity. Add a variant only when that provider has
/// a real prepared implementation in ZeroGravity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionProvider {
    /// ZeroGravity-authored backend kernel.
    Builtin,
    /// Statically compiled FlashAttention 2 BF16/head-256 CUDA forward kernel.
    #[cfg(feature = "cuda-flash-attn")]
    FlashAttentionFa2,
}

/// Phase-specific provider choice resolved during cold model preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionProviderPlan {
    prefill: AttentionProvider,
    decode: AttentionProvider,
}

impl AttentionProviderPlan {
    pub const fn new(prefill: AttentionProvider, decode: AttentionProvider) -> Self {
        Self { prefill, decode }
    }

    pub const fn builtin() -> Self {
        Self::new(AttentionProvider::Builtin, AttentionProvider::Builtin)
    }

    /// Use FA2 only for dense prefill; decode stays on the builtin provider
    /// until a decode-oriented provider demonstrates a real win.
    #[cfg(feature = "cuda-flash-attn")]
    pub const fn flash_attention_fa2_prefill() -> Self {
        Self::new(
            AttentionProvider::FlashAttentionFa2,
            AttentionProvider::Builtin,
        )
    }

    pub const fn provider(self, phase: InvocationPhase) -> AttentionProvider {
        match phase {
            InvocationPhase::Prefill => self.prefill,
            InvocationPhase::Decode => self.decode,
        }
    }

    pub const fn prefill(self) -> AttentionProvider {
        self.prefill
    }

    pub const fn decode(self) -> AttentionProvider {
        self.decode
    }
}
