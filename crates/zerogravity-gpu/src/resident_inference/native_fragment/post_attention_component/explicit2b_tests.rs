//! Actual recipe join, not synthetic source/lease/completion qualification.
use super::*;
use crate::cuda::gdn::{FixedPromptGdnArithmetic, Qwen2bGdnArithmetic};

#[test]
fn callback_choice_requires_same_actual_body_explicit_seal() {
    let owned = Qwen2bGdnArithmetic::OwnedSolve4Recompute3;
    let explicit = PostAttentionArithmetic::Qwen2bExplicit(owned);
    explicit
        .require_binding_parts(
            ComponentKind::Qwen2b,
            ComponentCut::PostAttention,
            Some(owned),
        )
        .unwrap();
    assert!(
        explicit
            .require_binding_parts(ComponentKind::Qwen2b, ComponentCut::PostAttention, None)
            .is_err()
    );
    assert!(
        PostAttentionArithmetic::Qwen2bOwned
            .require_binding_parts(
                ComponentKind::Qwen2b,
                ComponentCut::PostAttention,
                Some(owned)
            )
            .is_err()
    );
    for cut in [ComponentCut::Gated, ComponentCut::Mlp] {
        assert!(
            explicit
                .require_binding_parts(ComponentKind::Qwen2b, cut, Some(owned))
                .is_err()
        );
    }
    let other = ComponentKind::FixedPrompt(FixedPromptGdnArithmetic::OwnedFixedPromptSolve4);
    assert!(
        explicit
            .require_binding_parts(other, ComponentCut::PostAttention, Some(owned))
            .is_err()
    );
}
#[test]
fn old_none_and_47_bindings_remain_distinct_from_explicit2b() {
    let old = PostAttentionArithmetic::Qwen2bOwned;
    old.require_binding_parts(ComponentKind::Qwen2b, ComponentCut::PostAttention, None)
        .unwrap();
    assert_eq!(old.explicit_qwen2b(), None);
    let arithmetic = FixedPromptGdnArithmetic::OwnedFixedPromptSolve4;
    let old47 = PostAttentionArithmetic::FixedPrompt(arithmetic);
    old47
        .require_binding_parts(
            ComponentKind::FixedPrompt(arithmetic),
            ComponentCut::PostAttention,
            None,
        )
        .unwrap();
    assert_eq!(old47.explicit_qwen2b(), None);
    assert!(
        old47
            .require_binding_parts(ComponentKind::Qwen2b, ComponentCut::PostAttention, None)
            .is_err()
    );
}
