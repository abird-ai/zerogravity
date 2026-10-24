use super::*;

#[test]
fn native_geometry_is_exact_without_padding() -> Result<()> {
    for rows in [1, 2, 4, 8] {
        for width in [1, 31, 32, 33, 65_536] {
            assert_eq!(geometry(rows, width)?, rows * width);
        }
    }
    for (rows, width) in [
        (0, 1),
        (3, 1),
        (16, 1),
        (8, 0),
        (1, 65_537),
        (8, usize::MAX),
    ] {
        assert!(geometry(rows, width).is_err());
    }
    Ok(())
}

#[test]
fn checked_spans_never_hide_short_misaligned_or_overflowed_views() -> Result<()> {
    assert_eq!(checked_span(80, 16, 16, 12)?, 16..64);
    for (size, start, trim, words) in [
        (80, 1, 15, 1),
        (80, 16, 81, 1),
        (80, 64, 20, 1),
        (80, 16, 16, 13),
        (80, 16, 16, 0),
        (80, 16, 16, usize::MAX),
        (u64::MAX, u64::MAX - 1, 0, 1),
    ] {
        assert!(checked_span(size, start, trim, words).is_err());
    }
    Ok(())
}
