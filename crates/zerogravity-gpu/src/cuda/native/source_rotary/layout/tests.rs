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
        let readback = sum.max(1024 * 1024);
        assert_eq!(layout.readback_bytes().unwrap(), readback);
        assert_eq!(
            layout.retained_host_bytes().unwrap(),
            64 * 1024 + readback + positions * 384
        );
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
    fixed.require_kind(SourceKind::FixedPrompt).unwrap();
    text.require_kind(SourceKind::Text).unwrap();
    assert!(fixed.require_kind(SourceKind::Text).is_err());
    assert!(text.require_kind(SourceKind::FixedPrompt).is_err());
}

#[test]
fn fixed_and_text_costs_are_derived_from_the_requested_owned_span() {
    let fixed = Layout::new(SourceKind::FixedPrompt, 67).unwrap();
    assert_eq!(fixed.bytes(), 8_576);
    assert_eq!(fixed.readback_bytes().unwrap(), 1024 * 1024);
    assert_eq!(fixed.prefix_reservation_bytes().unwrap(), 25_728);
    assert_eq!(fixed.retained_host_bytes().unwrap(), 1_139_840);
    assert!(fixed.retained_host_bytes().unwrap() >= 1024 * 1024);

    let text = Layout::new(SourceKind::Text, 512).unwrap();
    assert_eq!(text.bytes(), 65_536);
    assert_eq!(text.readback_bytes().unwrap(), 1024 * 1024);
    assert_eq!(text.prefix_reservation_bytes().unwrap(), 196_608);
    assert_eq!(text.retained_host_bytes().unwrap(), 1_310_720);
    assert!(text.retained_host_bytes().unwrap() >= 1024 * 1024);
}

#[test]
fn production_routes_own_distinct_api_identity() {
    let fixed = Layout::new(SourceKind::FixedPrompt, 67).unwrap();
    let text = Layout::new(SourceKind::Text, 67).unwrap();
    assert_eq!(fixed.production_owner(), "original-hf08");
    assert_eq!(text.production_owner(), "qwen35-text");
    assert_eq!(
        fixed.production_ready_status(),
        "SOURCE_ROTARY_FIXED_PREFIX_READY"
    );
    assert_eq!(
        text.production_ready_status(),
        "SOURCE_ROTARY_TEXT_PREFIX_READY"
    );
    assert_ne!(fixed.production_owner(), text.production_owner());
    assert_ne!(
        fixed.production_ready_status(),
        text.production_ready_status()
    );
}
