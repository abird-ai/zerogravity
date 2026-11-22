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
