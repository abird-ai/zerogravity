//! Closed diagnostic geometry/input validation; no device/model access.
use super::*;

#[test]
fn only_two_extended_capacities_and_complete_causal_crossings() {
    for bad in [0, 256, 512, 513, 767, 769, 1023, 1025, 2048, usize::MAX] {
        assert!(ContextSpec::bounded(bad).is_err());
    }
    for capacity in [768, 1024] {
        let spec = ContextSpec::bounded(capacity).unwrap();
        assert_eq!(spec.milestones.last(), Some(&capacity));
        assert!(spec.milestones.windows(2).all(|pair| pair[0] < pair[1]));
        for boundary in (256..capacity).step_by(256) {
            assert!(spec.milestones.contains(&(boundary - 1)));
            assert!(spec.milestones.contains(&boundary));
            assert!(spec.milestones.contains(&(boundary + 1)));
        }
        assert!(spec.milestones.contains(&spec.replay_start));
        assert!(spec.milestones.contains(&spec.replay_end));
        assert_eq!(spec.replay_end - spec.replay_start, 2);
        assert!(spec.replay_end < capacity);
    }
    let legacy = ContextSpec::legacy();
    assert_eq!(
        (legacy.capacity, legacy.replay_start, legacy.replay_end),
        (512, 254, 257)
    );
    assert_eq!(legacy.milestones, &MILESTONES);
}

#[test]
fn complete_raw_ids_refuse_bad_tail_and_geometry_before_device_use() {
    for capacity in [768, 1024] {
        let spec = ContextSpec::bounded(capacity).unwrap();
        let mut ids = vec![11; capacity];
        spec.validate_ids(&ids, 248_320).unwrap();
        assert!(spec.validate_ids(&ids[..capacity - 1], 248_320).is_err());
        assert!(spec.validate_ids(&ids, 248_319).is_err());
        ids[capacity - 1] = 248_319;
        spec.validate_ids(&ids, 248_320).unwrap();
        ids[capacity - 1] = 248_320;
        assert!(spec.validate_ids(&ids, 248_320).is_err());
        ids[capacity - 1] = u32::MAX;
        assert!(spec.validate_ids(&ids, 248_320).is_err());
    }
}
