use super::*;

#[test]
fn convolution_capture_uses_logical_channel_token_order() {
    let phase = GdnPhase::ContinuedTwo;
    let mut physical = vec![0; LayerOutput::Convolution.bytes(phase)];
    let mut logical = physical.clone();
    for (physical_word, logical_word, value) in [
        (0, 0, 1_u16),
        (6144, 1, 2),
        (1, 2, 3),
        (6145, 3, 4),
        (6143, 12286, 5),
        (12287, 12287, 6),
    ] {
        physical[physical_word * 2..physical_word * 2 + 2].copy_from_slice(&value.to_le_bytes());
        logical[logical_word * 2..logical_word * 2 + 2].copy_from_slice(&value.to_le_bytes());
    }
    assert_ne!(physical, logical);
    assert_eq!(
        LayerOutput::Convolution
            .first_difference(phase, &physical, &logical)
            .unwrap(),
        None
    );
    logical[2] ^= 1;
    assert_eq!(
        LayerOutput::Convolution
            .first_difference(phase, &physical, &logical)
            .unwrap(),
        Some(1)
    );
    assert!(
        LayerOutput::Convolution
            .first_difference(phase, &physical[..16], &logical)
            .is_err()
    );
}
