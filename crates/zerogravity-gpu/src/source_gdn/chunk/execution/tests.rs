use super::*;

#[test]
fn chunk_geometry_tracks_active_output_and_full_private_state() {
    for heads in [16, 32, 48] {
        for rows in 1..=64 {
            let g = Geometry::new(rows, heads).unwrap();
            assert_eq!(
                g.input_words(),
                [
                    rows * heads * 128,
                    rows * heads * 128,
                    rows * heads * 128,
                    rows * heads,
                    rows * heads,
                    heads * 16384
                ]
            );
            assert_eq!(g.output_words()[17], heads * 16384);
            assert_eq!(g.output_words()[18], rows * heads * 128);
            assert_eq!(g.output_words()[0], heads * 64 * 128);
            assert_eq!(g.scratch_words(), [heads * 64; 2]);
        }
    }
    for (rows, heads) in [
        (0, 16),
        (65, 16),
        (1, 0),
        (1, 15),
        (1, 49),
        (usize::MAX, 16),
        (1, usize::MAX),
    ] {
        assert!(Geometry::new(rows, heads).is_err());
    }
}
