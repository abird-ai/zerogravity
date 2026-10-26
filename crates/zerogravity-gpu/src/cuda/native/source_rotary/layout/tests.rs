use super::*;

#[test]
fn requested_spans_preserve_source_extents_and_boundaries() {
    for (kind, positions) in [
        (SourceKind::FixedPrompt, 67),
        (SourceKind::Text, 79),
        (SourceKind::Text, 512),
        (SourceKind::Text, 8192),
    ] {
        let layout = Layout::new(kind, positions).unwrap();
        layout.require_span(0, positions).unwrap();
        layout.require_span(positions - 1, 1).unwrap();
        for (base, rows) in [
            (0, 0),
            (positions, 1),
            (positions - 1, 2),
            (usize::MAX, 1),
            (1, usize::MAX),
        ] {
            assert!(layout.require_span(base, rows).is_err());
        }
        assert_eq!(layout.bytes(), positions * 128);
        assert_eq!(layout.prefix_reservation_bytes().unwrap(), positions * 384);
        let sum = RotaryOutput::ALL
            .into_iter()
            .map(|r| layout.prefix_bytes(r))
            .sum::<usize>();
        assert_eq!(sum, 648 + 772 * positions);
        assert!(layout.readback_bytes().unwrap() >= sum);
        for role in RotaryOutput::ALL {
            assert!(layout.prefix_bytes(role) <= role.bytes());
        }
    }
}

#[test]
fn source_limit_and_route_identity_are_independent_of_case_lengths() {
    let full = RotaryOutput::Cache.bytes() / 128;
    Layout::new(SourceKind::Text, full)
        .unwrap()
        .require_span(full - 1, 1)
        .unwrap();
    for positions in [0, full + 1, usize::MAX] {
        assert!(Layout::new(SourceKind::Text, positions).is_err());
    }
    let fixed = Layout::new(SourceKind::FixedPrompt, 67).unwrap();
    let text = Layout::new(SourceKind::Text, 67).unwrap();
    assert_eq!(fixed.bytes(), text.bytes());
    assert_ne!(fixed.kind, text.kind);
}
