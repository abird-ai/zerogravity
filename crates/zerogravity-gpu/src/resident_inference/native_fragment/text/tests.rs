//! Pure metadata refusals only; no committed GPU packet or model qualification.
use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::accelerator::program::{SourceRows, TextCallGeometry};
use zerogravity_core::session_root::SessionAuthority;
fn call(source: SourceIdentity, base: usize, mode: SourceCallMode, id: u32) -> SourceCall {
    let rows = SourceRows::new(
        TextCallGeometry::new(source, 248320, 512).unwrap(),
        base,
        Arc::from([id]),
    )
    .unwrap();
    SourceCall::new(rows, mode).unwrap()
}
#[test]
fn prepared_call_cannot_replace_current_feedback_coordinates_or_id() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    let mut authority = SessionAuthority::new(source)?;
    let advance = authority.begin_advance(47)?;
    authority.prepare_publish(advance)?.commit();
    let root = authority.binding();
    let actual = call(source, 47, SourceCallMode::Decode, 353);
    require_feedback_coordinates(&actual, root, 47, 353)?;
    for wrong in [
        call(source, 46, SourceCallMode::Decode, 353),
        call(source, 48, SourceCallMode::Decode, 353),
        call(source, 47, SourceCallMode::Decode, 354),
        call(source, 47, SourceCallMode::Prefill, 353),
        call(
            SourceIdentity::for_new_allocation()?,
            47,
            SourceCallMode::Decode,
            353,
        ),
    ] {
        assert!(require_feedback_coordinates(&wrong, root, 47, 353).is_err());
    }
    assert!(require_feedback_coordinates(&actual, root, 48, 353).is_err());
    assert_eq!(authority.binding(), root);
    Ok(())
}
