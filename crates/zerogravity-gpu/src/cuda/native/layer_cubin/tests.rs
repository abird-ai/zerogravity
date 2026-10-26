use super::*;

fn fixtures() -> [LayerFixture; 5] {
    let ids: [Vec<u32>; 5] = [
        vec![11],
        vec![353],
        INITIAL64_IDS.to_vec(),
        vec![25, 43112],
        vec![353],
    ];
    std::array::from_fn(|index| LayerFixture {
        teacher_ids: ids[index].clone(),
        expected: LayerOutput::required(PHASES[index])
            .into_iter()
            .map(|output| LayerExpectedRecord {
                output,
                bytes: vec![0; output.bytes(PHASES[index])],
            })
            .collect(),
    })
}

#[test]
fn original_five_phase_teacher_partitions_and_record_sets_are_closed() {
    let mut values = fixtures();
    assert!(validate_fixtures(&values, LayerScope::FivePhases).is_ok());
    values[3].teacher_ids.swap(0, 1);
    assert!(validate_fixtures(&values, LayerScope::FivePhases).is_err());
    values[3].teacher_ids.swap(0, 1);
    values[2].expected.pop();
    assert!(validate_fixtures(&values, LayerScope::FivePhases).is_err());
}

#[test]
fn duplicate_wrong_shape_and_nonfinite_expected_records_refuse() {
    for mode in 0..3 {
        let mut values = fixtures();
        match mode {
            0 => values[0].expected[1].output = values[0].expected[0].output,
            1 => {
                values[1].expected[0].bytes.pop();
            }
            _ => {
                let record = values[4]
                    .expected
                    .iter_mut()
                    .find(|r| r.output == LayerOutput::RecurrentState)
                    .unwrap();
                record.bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
        }
        assert!(
            validate_fixtures(&values, LayerScope::FivePhases).is_err(),
            "case{mode}"
        );
    }
}

fn first_sequence_fixtures() -> [LayerFixture; 2] {
    let [initial, decode, _, _, _] = fixtures();
    [initial, decode].map(|mut fixture| {
        fixture
            .expected
            .retain(|record| record.output != LayerOutput::Projected);
        fixture
    })
}

#[test]
fn first_sequence_requires_actual_initial_then_computed_cache_decode_without_projection() {
    let scope = LayerScope::FirstSequenceThroughGated;
    let mut values = first_sequence_fixtures();
    assert!(validate_fixtures(&values, scope).is_ok());
    assert_eq!(scope.phases(), &PHASES[..2]);
    assert_eq!(values[0].expected.len(), 19);
    assert_eq!(values[1].expected.len(), 8);
    values[1].teacher_ids[0] = 11;
    assert!(validate_fixtures(&values, scope).is_err());
    values[1].teacher_ids[0] = 353;
    values[0].expected.push(LayerExpectedRecord {
        output: LayerOutput::Projected,
        bytes: vec![0; 2048],
    });
    assert!(validate_fixtures(&values, scope).is_err());
    assert!(validate_fixtures(&values, LayerScope::FivePhases).is_err());
}

#[test]
fn first_sequence_keeps_two_derived_images_and_no_splitk_workspace() {
    let scope = LayerScope::FirstSequenceThroughGated;
    let projections = scope
        .phases()
        .iter()
        .copied()
        .flat_map(|phase| scope.projections(phase).unwrap())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(projections, vec![Projection::Qkvz1, Projection::Ba1]);
    assert_eq!(
        super::super::projection_cubin::required_images(&projections)
            .unwrap()
            .len(),
        2
    );
    assert!(projections.iter().all(|p| {
        p.launches()
            .iter()
            .all(|launch| !launch.bindings().contains(&ProjectionBinding::Workspace))
    }));
    let full = LayerScope::FivePhases;
    let projections = full
        .phases()
        .iter()
        .copied()
        .flat_map(|phase| full.projections(phase).unwrap())
        .collect::<Vec<_>>();
    let images = super::super::projection_cubin::required_images(&projections).unwrap();
    assert_eq!(images.len(), 6);
    assert!(images.contains(&super::super::cubin_manifests::ProjectionImage::CutlassWmmaSm80));
}

fn postnorm_fixtures() -> [LayerFixture; 5] {
    let mut values = fixtures();
    for (fixture, phase) in values.iter_mut().zip(PHASES) {
        for output in [
            LayerOutput::PostNormSumF32,
            LayerOutput::PostNormalized,
            LayerOutput::PostResidual,
        ] {
            fixture.expected.push(LayerExpectedRecord {
                output,
                bytes: vec![0; output.bytes(phase)],
            });
        }
    }
    values
}

#[test]
fn postnorm_scope_requires_all_three_outputs_without_widening_original_scopes() {
    let scope = LayerScope::FivePhasesThroughPostNorm;
    let mut values = postnorm_fixtures();
    assert!(validate_fixtures(&values, scope).is_ok());
    assert_eq!(scope.phases(), &PHASES);
    assert_eq!(
        values.each_ref().map(|fixture| fixture.expected.len()),
        [23, 12, 23, 23, 12]
    );
    assert!(scope.postnorm());
    assert!(!LayerScope::FivePhases.postnorm());
    assert!(!LayerScope::FirstSequenceThroughGated.postnorm());
    assert!(validate_fixtures(&fixtures(), scope).is_err());
    assert!(validate_fixtures(&values, LayerScope::FivePhases).is_err());
    assert!(validate_fixtures(&values[..2], LayerScope::FirstSequenceThroughGated).is_err());
    values[0]
        .expected
        .retain(|record| record.output != LayerOutput::PostNormSumF32);
    assert!(validate_fixtures(&values, scope).is_err());
}

#[test]
fn postnorm_sum_and_bf16_outputs_have_distinct_extents_and_finite_guards() {
    for phase in PHASES {
        let words = phase.tokens() * 1024;
        assert_eq!(LayerOutput::PostNormSumF32.word_bytes(), 4);
        assert_eq!(LayerOutput::PostNormSumF32.bytes(phase), words * 4);
        for output in [LayerOutput::PostNormalized, LayerOutput::PostResidual] {
            assert_eq!(output.word_bytes(), 2);
            assert_eq!(output.bytes(phase), words * 2);
        }
    }
    for (output, invalid) in [
        (LayerOutput::PostNormSumF32, f32::NAN.to_le_bytes().to_vec()),
        (
            LayerOutput::PostNormalized,
            0x7f80u16.to_le_bytes().to_vec(),
        ),
        (LayerOutput::PostResidual, 0x7fc0u16.to_le_bytes().to_vec()),
    ] {
        let mut values = postnorm_fixtures();
        let record = values[3]
            .expected
            .iter_mut()
            .find(|record| record.output == output)
            .unwrap();
        record.bytes[..invalid.len()].copy_from_slice(&invalid);
        assert!(validate_fixtures(&values, LayerScope::FivePhasesThroughPostNorm).is_err());
    }
    let mut values = postnorm_fixtures();
    let sum = values[2]
        .expected
        .iter_mut()
        .find(|record| record.output == LayerOutput::PostNormSumF32)
        .unwrap();
    sum.bytes.truncate(PHASES[2].tokens() * 1024 * 2);
    assert!(validate_fixtures(&values, LayerScope::FivePhasesThroughPostNorm).is_err());
}

#[test]
fn postnorm_scope_reuses_exact_original_projection_and_image_sets() {
    let projections = |scope: LayerScope| {
        scope
            .phases()
            .iter()
            .copied()
            .flat_map(|phase| scope.projections(phase).unwrap())
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(
        projections(LayerScope::FivePhasesThroughPostNorm),
        projections(LayerScope::FivePhases)
    );
    assert_eq!(LayerScope::FivePhasesThroughPostNorm.projection_count(), 3);
    assert!(validate_fixtures(&fixtures(), LayerScope::FivePhases).is_ok());
    assert!(
        validate_fixtures(
            &first_sequence_fixtures(),
            LayerScope::FirstSequenceThroughGated
        )
        .is_ok()
    );
}

fn mlp_fixtures() -> [LayerFixture; 5] {
    let mut values = postnorm_fixtures();
    for (fixture, phase) in values.iter_mut().zip(PHASES) {
        for output in [
            LayerOutput::GateUp,
            LayerOutput::Activation,
            LayerOutput::MlpOutput,
        ] {
            fixture.expected.push(LayerExpectedRecord {
                output,
                bytes: vec![0; output.bytes(phase)],
            });
        }
    }
    values
}

#[test]
fn mlp_scope_requires_exact_dual_residual_and_three_new_outputs_only() {
    let scope = LayerScope::FivePhasesThroughMlp;
    let values = mlp_fixtures();
    assert!(validate_fixtures(&values, scope).is_ok());
    assert_eq!(
        values.each_ref().map(|fixture| fixture.expected.len()),
        [26, 15, 26, 26, 15]
    );
    assert!(validate_fixtures(&postnorm_fixtures(), scope).is_err());
    assert!(validate_fixtures(&values, LayerScope::FivePhasesThroughPostNorm).is_err());
    for output in [
        LayerOutput::PostResidual,
        LayerOutput::GateUp,
        LayerOutput::Activation,
        LayerOutput::MlpOutput,
    ] {
        let mut incomplete = mlp_fixtures();
        incomplete[3]
            .expected
            .retain(|record| record.output != output);
        assert!(validate_fixtures(&incomplete, scope).is_err());
    }
    for phase in PHASES {
        for (output, columns) in [
            (LayerOutput::GateUp, 7168),
            (LayerOutput::Activation, 3584),
            (LayerOutput::MlpOutput, 1024),
        ] {
            assert_eq!(output.word_bytes(), 2);
            assert_eq!(output.bytes(phase), phase.tokens() * columns * 2);
        }
    }
}

#[test]
fn connected_mlp_has_separate_caps_and_preserves_old_weight_and_phase_scopes() {
    let mlp = LayerScope::FivePhasesThroughMlp;
    assert_eq!(mlp.provider_cap(), 144 * 1024 * 1024);
    assert_eq!(mlp.weight_count(), 5);
    assert!(mlp.postnorm());
    assert!(mlp.mlp());
    assert_eq!(mlp.phases(), &PHASES);
    for scope in [
        LayerScope::FirstSequenceThroughGated,
        LayerScope::FivePhases,
        LayerScope::FivePhasesThroughPostNorm,
    ] {
        assert_eq!(scope.provider_cap(), PROVIDER_CAP);
        assert!(!scope.mlp());
        assert_eq!(scope.weight_count(), scope.projection_count());
    }
    let values = mlp_fixtures();
    let mut bf16 = 0;
    let mut f32 = 0;
    for fixture in &values {
        for record in &fixture.expected {
            if record.output.word_bytes() == 4 {
                f32 += record.bytes.len() / 4;
            } else {
                bf16 += record.bytes.len() / 2;
            }
        }
    }
    assert_eq!((bf16, f32), (4_071_072, 2_171_024));
}

#[test]
fn owned47_leaf_cannot_select_legacy_opaque_manifests() {
    let phase = GdnPhase::FixedPrompt;
    assert!(matches!(
        Projection::for_phase(phase),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        MlpProjection::for_phase(phase),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        Activation::for_phase(phase),
        Err(Error::Unsupported(_))
    ));
    assert!(LayerScope::FivePhasesThroughMlp.projections(phase).is_err());
}
