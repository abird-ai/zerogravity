use super::*;

#[test]
fn packed_domains_keep_actual_rows_instead_of_flattening_to_b1() -> Result<()> {
    for rows in [2, 4, 8] {
        for width in [1, 31, 32, 33, 65_536] {
            assert_eq!(ordered_width(rows, rows * width)?, Some(width));
        }
    }
    // This packed B8 shape exceeds the B1 width limit, but every genuine row
    // remains in the accepted domain. No padding/request is introduced.
    assert_eq!(ordered_width(8, 8 * 65_536)?, Some(65_536));
    assert_eq!(ordered_width(1, 8 * 65_536)?, None);
    Ok(())
}

#[test]
fn only_known_geometry_uses_original_route_and_partition_changes_refuse() -> Result<()> {
    for rows in [1, 2, 4, 8] {
        assert_eq!(ordered_width(rows, 0)?, None);
        assert_eq!(ordered_width(rows, rows * 65_537)?, None);
    }
    for (rows, words) in [(0, 32), (3, 96), (16, 512), (2, 65), (4, 129), (8, 257)] {
        assert!(ordered_width(rows, words).is_err());
    }
    Ok(())
}
