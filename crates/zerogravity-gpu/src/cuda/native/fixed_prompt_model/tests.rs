use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{TextCallGeometry, TokenSpan};

fn call(
    mode: InvocationPhase,
    base: usize,
    rows: usize,
    capacity: usize,
) -> Result<TextInvocation> {
    TextInvocation::new(
        TokenSpan::new(
            TextCallGeometry::new(SourceIdentity::for_new_allocation()?, 248320, capacity)?,
            base,
            vec![353; rows].into(),
        )?,
        mode,
    )
}
#[test]
fn physical_fixed_prompt_rows_and_cached_frontiers_are_exact() -> Result<()> {
    assert_eq!(
        FixedPromptInvocationKind::of(&call(InvocationPhase::Prefill, 0, 47, 512)?)?,
        FixedPromptInvocationKind::Prefill
    );
    for base in 47..=53 {
        assert_eq!(
            FixedPromptInvocationKind::of(&call(InvocationPhase::Decode, base, 1, 512)?)?,
            FixedPromptInvocationKind::Decode { base }
        );
    }
    for (mode, base, rows, capacity) in [
        (InvocationPhase::Prefill, 0, 48, 512),
        (InvocationPhase::Prefill, 1, 47, 512),
        (InvocationPhase::Prefill, 0, 47, 513),
        (InvocationPhase::Decode, 46, 1, 512),
        (InvocationPhase::Decode, 54, 1, 512),
    ] {
        assert!(FixedPromptInvocationKind::of(&call(mode, base, rows, capacity)?).is_err())
    }
    assert!(call(InvocationPhase::Decode, 0, 1, 512).is_err());
    assert!(call(InvocationPhase::Decode, 47, 2, 512).is_err());
    Ok(())
}
#[test]
fn physical_row_metadata_does_not_replace_actual_teacher_issuer() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let make = || {
        TextInvocation::new(
            TokenSpan::new(
                TextCallGeometry::new(source, 248320, 512)?,
                0,
                vec![353; 47].into(),
            )?,
            InvocationPhase::Prefill,
        )
    };
    let a = make()?;
    let b = make()?;
    assert_eq!(a.source(), b.source());
    assert_eq!(a.rows().ids(), b.rows().ids());
    assert_ne!(a.rows(), b.rows());
    assert_ne!(a, b);
    assert_eq!(
        FixedPromptInvocationKind::of(&a)?,
        FixedPromptInvocationKind::of(&b)?
    );
    assert_eq!(a, a.clone());
    Ok(())
}

#[test]
fn only_owned_prefill_gdn_reaches_the_prepared_join() {
    use crate::cuda::gdn::FixedPromptGdnArithmetic as A;
    assert!(
        FixedPromptInvocationKind::Prefill
            .require_prefill_gdn(A::OwnedFixedPromptSolve4)
            .is_ok()
    );
    assert!(
        FixedPromptInvocationKind::Prefill
            .require_prefill_gdn(A::OriginalPlainSolve2)
            .is_err()
    );
    for base in 47..=53 {
        assert!(
            FixedPromptInvocationKind::Decode { base }
                .require_prefill_gdn(A::OwnedFixedPromptSolve4)
                .is_err()
        );
    }
}
