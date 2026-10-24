//! Metadata rejection tests; original source and physical tests remain separate.
use super::*;
use crate::vllm_cuda_gdn::{Canonical47GdnArithmetic, Qwen2bGdnArithmetic};
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{GdnPrefixLayer, TeacherSpan, VllmBf16Invocation, VllmBf16Teachers},
};

fn request(
    rows: usize,
    base: usize,
    capacity: usize,
    layer: usize,
    mode: VllmBf16Mode,
) -> VllmBf16Invocation {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let ids = vec![11; rows];
    let teachers = if rows == 1 {
        VllmBf16Teachers::new(source, base, capacity, &ids).unwrap()
    } else {
        VllmBf16Teachers::from_span(TeacherSpan::new(source, base, capacity, 248320, &ids).unwrap())
            .unwrap()
    };
    VllmBf16Invocation::from_span(GdnPrefixLayer::new(source, layer).unwrap(), mode, teachers)
        .unwrap()
}
#[test]
fn explicit_tuple_extends_only_initial2b_gated_to_post_attention() {
    let owned = Qwen2bGdnArithmetic::OwnedSolve4Recompute3;
    assert_eq!(ComponentKind::Qwen2b.profile(), FullModelProfile::Qwen2B);
    assert_eq!(ComponentKind::Qwen2b.columns(), 2048);
    assert_eq!(ComponentCut::PostAttention.node_end(), 8);
    assert_eq!(ComponentCut::PostAttention.value_end(), 10);
    for rows in [1, 64] {
        for cut in [ComponentCut::Gated, ComponentCut::PostAttention] {
            require_qwen2b_arithmetic(ComponentKind::Qwen2b, cut, rows, owned).unwrap();
        }
        assert!(
            require_qwen2b_arithmetic(ComponentKind::Qwen2b, ComponentCut::Mlp, rows, owned)
                .is_err()
        );
    }
    for rows in [0, 2, 47, 63, 65, 512] {
        assert!(
            require_qwen2b_arithmetic(
                ComponentKind::Qwen2b,
                ComponentCut::PostAttention,
                rows,
                owned
            )
            .is_err()
        );
    }
    assert!(
        require_qwen2b_arithmetic(
            ComponentKind::Canonical47(Canonical47GdnArithmetic::OwnedCanonical47Solve4),
            ComponentCut::PostAttention,
            47,
            owned,
        )
        .is_err()
    );
}
#[test]
fn exact_request_still_requires_initial_layer0_base0_capacity512() {
    for rows in [1, 64] {
        ComponentKind::Qwen2b
            .require_request(&request(rows, 0, 512, 0, VllmBf16Mode::Prefill))
            .unwrap();
    }
    for wrong in [
        request(47, 0, 512, 0, VllmBf16Mode::Prefill),
        request(1, 1, 512, 0, VllmBf16Mode::Prefill),
        request(64, 0, 1024, 0, VllmBf16Mode::Prefill),
        request(64, 0, 512, 1, VllmBf16Mode::Prefill),
        request(1, 64, 512, 0, VllmBf16Mode::Decode),
    ] {
        assert!(ComponentKind::Qwen2b.require_request(&wrong).is_err());
    }
}
#[test]
fn same_ids_do_not_replace_the_original_teacher_issuer() {
    let original = request(64, 0, 512, 0, VllmBf16Mode::Prefill);
    let replacement = VllmBf16Invocation::from_span(
        original.layer(),
        original.mode(),
        VllmBf16Teachers::from_span(
            TeacherSpan::new(
                original.layer().source(),
                0,
                512,
                248320,
                original.teachers().ids(),
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(original.teachers().ids(), replacement.teachers().ids());
    assert_ne!(original.teachers(), replacement.teachers());
    assert_ne!(original, replacement);
}
