use super::*;

fn fixture(projection: MlpProjection) -> CubinMlpFixture {
    CubinMlpFixture {
        projection,
        input: vec![0; projection.input_bytes()],
        expected: vec![0; projection.output_bytes()],
    }
}

#[test]
fn source_recipe_retains_gate_then_up_and_one_original_down_matrix() {
    assert_eq!(
        weight_parts(MlpRole::GateUp),
        &[("gate_proj", 3584, 1024), ("up_proj", 3584, 1024)]
    );
    assert_eq!(weight_parts(MlpRole::Down), &[("down_proj", 1024, 3584)]);
    for projection in [MlpProjection::GateUp1, MlpProjection::Down1] {
        assert_eq!(
            weight_parts(projection.role())
                .iter()
                .map(|(_, rows, columns)| rows * columns * 2)
                .sum::<usize>(),
            projection.weight_bytes()
        );
    }
}

#[test]
fn exact_original_ten_leaf_schedule_has_five_images_and_required_workspace() {
    let phases = [
        VllmGdnPhase::InitialOne,
        VllmGdnPhase::CachedDecode,
        VllmGdnPhase::Initial64,
        VllmGdnPhase::ContinuedTwo,
        VllmGdnPhase::CachedDecode,
    ];
    let projections = phases
        .into_iter()
        .flat_map(|phase| MlpProjection::for_phase(phase).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(projections.len(), 10);
    assert_eq!(
        projections
            .iter()
            .map(|p| p.output_bytes() / 2)
            .sum::<usize>(),
        565_248
    );
    assert_eq!(required_images(&projections).len(), 5);
    assert!(!uses_workspace(MlpProjection::GateUp1));
    assert!(!uses_workspace(MlpProjection::GateUp2));
    assert!(!uses_workspace(MlpProjection::GateUp64));
    for projection in [
        MlpProjection::Down1,
        MlpProjection::Down2,
        MlpProjection::Down64,
    ] {
        assert!(uses_workspace(projection));
    }
}

#[test]
fn leaf_refuses_incomplete_nonfinite_and_unadmitted_capacity() {
    assert!(validate_fixtures(&[fixture(MlpProjection::GateUp1)]).is_ok());
    assert!(validate_fixtures(&[]).is_err());
    let too_many = (0..11)
        .map(|_| fixture(MlpProjection::Down1))
        .collect::<Vec<_>>();
    assert!(validate_fixtures(&too_many).is_err());
    for mode in 0..4 {
        let mut value = fixture(MlpProjection::Down1);
        match mode {
            0 => {
                value.input.pop();
            }
            1 => value.expected.push(0),
            2 => value.input[..2].copy_from_slice(&0xff80u16.to_le_bytes()),
            _ => value.expected[..2].copy_from_slice(&0x7fc1u16.to_le_bytes()),
        }
        assert!(validate_fixtures(&[value]).is_err(), "fault{mode}");
    }
    let mut value = fixture(MlpProjection::Down1);
    value.expected = Vec::with_capacity(HOST_BYTES + 1);
    value
        .expected
        .resize(MlpProjection::Down1.output_bytes(), 0);
    assert!(validate_fixtures(&[value]).is_err());
}
