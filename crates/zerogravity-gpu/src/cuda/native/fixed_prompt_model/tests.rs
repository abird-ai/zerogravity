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
    let a = call(SourceCallMode::Prefill, 0, 47, 512)?;
    let b = call(SourceCallMode::Prefill, 0, 47, 512)?;
    assert_ne!(a, b);
    assert_eq!(FixedPromptCallKind::of(&a)?, FixedPromptCallKind::of(&b)?);
    assert_eq!(a, a.clone());
    Ok(())
}
