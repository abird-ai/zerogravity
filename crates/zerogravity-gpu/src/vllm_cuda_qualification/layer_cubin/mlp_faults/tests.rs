use super::*;

#[test]
fn connected_fault_scope_cannot_retag_old_postnorm_or_earlier_cuts() {
    for scope in [
        LayerScope::FirstSequenceThroughGated,
        LayerScope::FivePhases,
        LayerScope::FivePhasesThroughPostNorm,
    ] {
        assert!(validate_mode(scope, Mode::MlpFault(LayerMlpCubinFault::UnknownAfterMlp)).is_err());
        assert_eq!(scope.provider_cap(), 128 * 1024 * 1024);
    }
    assert!(
        validate_mode(
            LayerScope::FivePhasesThroughMlp,
            Mode::Fault(PostnormCubinFault::UnknownAfterFused)
        )
        .is_err()
    );
    assert!(
        validate_mode(
            LayerScope::FivePhasesThroughPostNorm,
            Mode::Fault(PostnormCubinFault::UnknownAfterFused)
        )
        .is_ok()
    );
    assert!(
        validate_mode(
            LayerScope::FivePhasesThroughMlp,
            Mode::MlpFault(LayerMlpCubinFault::UnknownAfterMlp)
        )
        .is_ok()
    );
    assert_eq!(
        LayerScope::FivePhasesThroughMlp.provider_cap(),
        144 * 1024 * 1024
    );
}

fn after_mlp() -> SubmittedStages {
    SubmittedStages {
        rms: 10,
        native: 8,
        gdn: 12,
        postnorm: 10,
        mlp_calls: 2,
        activation: 1,
        clears: 0,
    }
}
#[test]
fn actual_partial_and_complete_mlp_boundaries_are_distinct() {
    let complete = after_mlp();
    let partial = SubmittedStages {
        native: 6,
        mlp_calls: 1,
        ..complete
    };
    assert!(validate_stages(LayerMlpCubinFault::UnknownAfterMlp, complete).is_ok());
    assert!(validate_stages(LayerMlpCubinFault::CancelBeforeDown, partial).is_ok());
    assert!(validate_stages(LayerMlpCubinFault::UnknownAfterMlp, partial).is_err());
    assert!(validate_stages(LayerMlpCubinFault::CancelBeforeDown, complete).is_err());
    assert!(validate_stages(LayerMlpCubinFault::AdmissionPressure, complete).is_err());
    for value in [
        SubmittedStages {
            activation: 0,
            ..complete
        },
        SubmittedStages {
            clears: 1,
            ..complete
        },
        SubmittedStages {
            gdn: 11,
            ..complete
        },
        SubmittedStages {
            postnorm: 0,
            ..complete
        },
        SubmittedStages {
            native: 7,
            ..complete
        },
        SubmittedStages { rms: 0, ..complete },
    ] {
        assert!(validate_stages(LayerMlpCubinFault::UnknownAfterMlp, value).is_err());
    }
}

#[test]
fn actual_shared_native_owners_cannot_accept_a_missing_activation_or_record() {
    assert!(validate_native([6, 7, 7], [11, 7, 3], [FOREIGN_BYTES; 3]).is_ok());
    for images in [[6, 6, 7], [6, 7, 6], [5, 7, 7]] {
        assert!(validate_native(images, [11, 7, 3], [FOREIGN_BYTES; 3]).is_err());
    }
    for records in [[11, 7, 2], [11, 6, 3], [10, 7, 3], [11, 8, 3]] {
        assert!(validate_native([6, 7, 7], records, [FOREIGN_BYTES; 3]).is_err());
    }
    assert!(
        validate_native(
            [6, 7, 7],
            [11, 7, 3],
            [FOREIGN_BYTES, FOREIGN_BYTES - 1, FOREIGN_BYTES]
        )
        .is_err()
    );
}

#[test]
fn readback_cohort_requires_real_mlp_records_and_unrounded_f32_sum() {
    let mut layout = vec![(0, 2048); 33];
    layout[4].1 = 4096;
    assert!(validate_readbacks(&layout).is_ok());
    for variant in 0..5 {
        let mut value = layout.clone();
        match variant {
            0 => {
                value.pop();
            }
            1 => value.push((0, 2048)),
            2 => value[4].1 = 2048,
            3 => value[0].0 = 2,
            _ => value[32].1 = 0,
        }
        assert!(validate_readbacks(&value).is_err());
    }
}
