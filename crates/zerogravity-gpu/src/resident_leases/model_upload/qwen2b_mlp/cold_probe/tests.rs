//! Actual private cut accounting; no fabricated service/event or Ready.
use super::*;
#[test]
fn exact_three_completed_read_positions_are_closed_and_total_tail_included() {
    assert_eq!(compared_at(0).unwrap(), 0);
    assert_eq!(compared_at(1).unwrap(), 1048576);
    assert_eq!(compared_at(37).unwrap(), 38797312);
    assert_eq!(compared_at(48).unwrap(), 50331648);
    assert_eq!(compared_at(72).unwrap(), 75399168);
    assert_eq!(compared_at(73).unwrap(), 75497472);
    assert!(compared_at(74).is_err());
}
#[test]
fn read_cut_refuses_at_exact_completed_byte_prefix_without_validator_entry() {
    for mode in [
        Qwen2bMlpWeightMode::UnknownAfterFirstRead,
        Qwen2bMlpWeightMode::UnknownAfterMiddleRead,
        Qwen2bMlpWeightMode::UnknownAfterLastRead,
        Qwen2bMlpWeightMode::CancelAfterFirstRead,
        Qwen2bMlpWeightMode::CancelAfterMiddleRead,
        Qwen2bMlpWeightMode::CancelAfterLastRead,
    ] {
        let probe = ReadProbe::new(mode);
        let (cut, cancel) = mode.cut().unwrap();
        for index in 1..=cut {
            let bytes = compared_at(index).unwrap() - compared_at(index - 1).unwrap();
            let result = probe.after_completed_read(index, bytes);
            if index < cut {
                result.unwrap();
            } else {
                assert!(
                    matches!(result,Err(Error::State(ref why)) if why==if cancel {"original full-model learned upload cancelled"}else{UNKNOWN})
                );
            }
        }
        assert_eq!(probe.completed.load(Ordering::Acquire), cut);
        assert_eq!(
            probe.bytes.load(Ordering::Acquire),
            compared_at(cut).unwrap()
        );
        assert!(!probe.validator.load(Ordering::Acquire));
    }
}
#[test]
fn complete_and_validator_modes_never_inject_an_earlier_read_failure() {
    for mode in [
        Qwen2bMlpWeightMode::Complete,
        Qwen2bMlpWeightMode::ValidatorRefusal,
    ] {
        let probe = ReadProbe::new(mode);
        for index in 1..=73 {
            probe
                .after_completed_read(
                    index,
                    compared_at(index).unwrap() - compared_at(index - 1).unwrap(),
                )
                .unwrap();
        }
        assert_eq!(probe.bytes.load(Ordering::Acquire), 75497472);
        assert!(!probe.validator.load(Ordering::Acquire));
    }
}
