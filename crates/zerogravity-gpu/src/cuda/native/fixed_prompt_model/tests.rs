use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{SourceRows, TextCallGeometry};

fn call(mode: SourceCallMode, base: usize, rows: usize, capacity: usize) -> Result<SourceCall> {
    SourceCall::new(
        SourceRows::new(
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
        FixedPromptCallKind::of(&call(SourceCallMode::Prefill, 0, 47, 512)?)?,
        FixedPromptCallKind::Prefill
    );
    for base in 47..=53 {
        assert_eq!(
            FixedPromptCallKind::of(&call(SourceCallMode::Decode, base, 1, 512)?)?,
            FixedPromptCallKind::Decode { base }
        );
    }
    for (mode, base, rows, capacity) in [
        (SourceCallMode::Prefill, 0, 48, 512),
        (SourceCallMode::Prefill, 1, 47, 512),
        (SourceCallMode::Prefill, 0, 47, 513),
        (SourceCallMode::Decode, 46, 1, 512),
        (SourceCallMode::Decode, 54, 1, 512),
    ] {
        assert!(FixedPromptCallKind::of(&call(mode, base, rows, capacity)?).is_err())
    }
    assert!(call(SourceCallMode::Decode, 0, 1, 512).is_err());
    assert!(call(SourceCallMode::Decode, 47, 2, 512).is_err());
    Ok(())
}
#[test]
fn physical_row_metadata_does_not_replace_actual_teacher_issuer() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let make = || {
        SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, 248320, 512)?,
                0,
                vec![353; 47].into(),
            )?,
            SourceCallMode::Prefill,
        )
    };
    let a = make()?;
    let b = make()?;
    assert_eq!(a.source(), b.source());
    assert_eq!(a.rows().ids(), b.rows().ids());
    assert_ne!(a.rows(), b.rows());
    assert_ne!(a, b);
    assert_eq!(FixedPromptCallKind::of(&a)?, FixedPromptCallKind::of(&b)?);
    assert_eq!(a, a.clone());
    Ok(())
}

#[test]
fn only_owned_prefill_gdn_reaches_the_prepared_join() {
    use crate::cuda::gdn::FixedPromptGdnArithmetic as A;
    assert!(
        FixedPromptCallKind::Prefill
            .require_prefill_gdn(A::OwnedFixedPromptSolve4)
            .is_ok()
    );
    assert!(
        FixedPromptCallKind::Prefill
            .require_prefill_gdn(A::OriginalPlainSolve2)
            .is_err()
    );
    for base in 47..=53 {
        assert!(
            FixedPromptCallKind::Decode { base }
                .require_prefill_gdn(A::OwnedFixedPromptSolve4)
                .is_err()
        );
    }
}
