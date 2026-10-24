use super::*;

#[test]
fn faults_never_admit_an_earlier_connected_scope() {
    assert!(validate_scope(LayerScope::FivePhasesThroughPostNorm).is_ok());
    assert!(validate_scope(LayerScope::FivePhases).is_err());
    assert!(validate_scope(LayerScope::FirstSequenceThroughGated).is_err());
}

fn first_readbacks() -> Vec<(usize, usize)> {
    let mut values = vec![(0, 2048); 30];
    values[4].1 = 4096;
    values
}

#[test]
fn first_fused_cohort_refuses_wrong_count_offset_and_bf16_sum_extent() {
    assert!(validate_readback_layout(&first_readbacks()).is_ok());
    for mode in 0..5 {
        let mut values = first_readbacks();
        match mode {
            0 => {
                values.pop();
            }
            1 => values.push((0, 2048)),
            2 => values[3].0 = 2,
            3 => values[4].1 = 2048,
            _ => values[6].1 = 0,
        }
        assert!(validate_readback_layout(&values).is_err(), "case{mode}");
    }
}

#[test]
fn unknown_and_late_cancel_require_distinct_actual_fused_boundaries() {
    let completed_enqueue = SubmittedStages {
        rms: 10,
        native: 4,
        gdn: 12,
        postnorm: 10,
    };
    let cancelled_enqueue = SubmittedStages {
        postnorm: 0,
        ..completed_enqueue
    };
    assert!(validate_stages(PostnormCubinFault::UnknownAfterFused, completed_enqueue).is_ok());
    assert!(validate_stages(PostnormCubinFault::CancelBeforeFused, cancelled_enqueue).is_ok());
    assert!(validate_stages(PostnormCubinFault::UnknownAfterFused, cancelled_enqueue).is_err());
    assert!(validate_stages(PostnormCubinFault::CancelBeforeFused, completed_enqueue).is_err());
    for stages in [
        SubmittedStages {
            rms: 0,
            ..completed_enqueue
        },
        SubmittedStages {
            native: 3,
            ..completed_enqueue
        },
        SubmittedStages {
            gdn: 0,
            ..completed_enqueue
        },
        SubmittedStages {
            postnorm: 9,
            ..completed_enqueue
        },
    ] {
        assert!(validate_stages(PostnormCubinFault::UnknownAfterFused, stages).is_err());
    }
    assert!(validate_stages(PostnormCubinFault::AdmissionPressure, cancelled_enqueue).is_err());
}
