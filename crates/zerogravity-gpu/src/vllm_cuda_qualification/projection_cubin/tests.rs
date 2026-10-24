use super::*;

#[test]
fn canonical47_qkkv_union_is_exact_and_does_not_raise_generic_limit() {
    let mut images = super::super::canonical47_mlp::required_images();
    images.extend([
        ProjectionImage::AttentionAmpere256x128Sm80,
        ProjectionImage::AttentionQkSm86,
        ProjectionImage::AttentionKvSm86,
    ]);
    assert_eq!(images.len(), 7);
    require_canonical47_qkkv_image_sources(&images).unwrap();
    assert!(require_image_sources(&images).is_err());
    for image in &images {
        let mut wrong = images.clone();
        wrong.remove(image);
        assert!(require_canonical47_qkkv_image_sources(&wrong).is_err());
        wrong.insert(ProjectionImage::Bf16Gemv2TSm86);
        assert_eq!(wrong.len(), 7);
        assert!(require_canonical47_qkkv_image_sources(&wrong).is_err());
    }
    let mut extra = images;
    extra.insert(ProjectionImage::Bf16Gemv2TSm86);
    assert!(require_canonical47_qkkv_image_sources(&extra).is_err());
    assert!(require_canonical47_qkkv_image_sources(&BTreeSet::new()).is_err());
}

fn fixture(projection: Projection) -> CubinProjectionFixture {
    CubinProjectionFixture {
        projection,
        input: vec![0; projection.input_bytes()],
        expected: vec![0; projection.output_bytes()],
    }
}

#[test]
fn closed_captured_image_completes_only_the_bounded_original_operation_set() {
    let independent = [
        Projection::Qkvz1,
        Projection::Ba1,
        Projection::Qkvz64,
        Projection::Out64,
    ];
    assert_eq!(required_images(&independent).unwrap().len(), 4);
    for projection in [
        Projection::Out1,
        Projection::Ba64,
        Projection::Qkvz2,
        Projection::Ba2,
        Projection::Out2,
    ] {
        let images = required_images(&[projection]).unwrap();
        assert!(images.contains(&ProjectionImage::CutlassWmmaSm80));
    }
    let all = [
        Projection::Qkvz1,
        Projection::Ba1,
        Projection::Out1,
        Projection::Qkvz64,
        Projection::Ba64,
        Projection::Out64,
        Projection::Qkvz2,
        Projection::Ba2,
        Projection::Out2,
    ];
    assert_eq!(required_images(&all).unwrap().len(), 6);
    assert!(required_images(&[]).is_err());
}

#[test]
fn leaf_fixtures_enforce_geometry_finiteness_and_capacity() {
    let valid = fixture(Projection::Ba1);
    assert!(validate_fixtures(&[valid]).is_ok());
    assert!(validate_fixtures(&[]).is_err());
    let many = (0..16)
        .map(|_| fixture(Projection::Ba1))
        .collect::<Vec<_>>();
    assert!(validate_fixtures(&many).is_err());
    for mode in 0..4 {
        let mut bad = fixture(Projection::Ba1);
        match mode {
            0 => {
                bad.input.pop();
            }
            1 => bad.expected.extend_from_slice(&0u16.to_le_bytes()),
            2 => bad.expected[..2].copy_from_slice(&0x7fc1u16.to_le_bytes()),
            _ => bad.input[..2].copy_from_slice(&0xff80u16.to_le_bytes()),
        }
        assert!(validate_fixtures(&[bad]).is_err(), "case{mode}");
    }
    let mut huge = fixture(Projection::Ba1);
    huge.expected = Vec::with_capacity(HOST_BYTES + 1);
    huge.expected.resize(Projection::Ba1.output_bytes(), 0);
    assert!(
        validate_fixtures(&[huge]).is_err(),
        "unadmitted spare capacity"
    );
}

#[test]
fn native_faults_accept_only_one_original_qkvz1_or_captured_out2() {
    assert!(faults::validate_fault_fixtures(&[fixture(Projection::Qkvz1)]).is_ok());
    assert!(faults::validate_fault_fixtures(&[fixture(Projection::Out2)]).is_ok());
    assert!(faults::validate_fault_fixtures(&[]).is_err());
    assert!(faults::validate_fault_fixtures(&[fixture(Projection::Ba1)]).is_err());
    assert!(
        faults::validate_fault_fixtures(&[fixture(Projection::Qkvz1), fixture(Projection::Qkvz1)])
            .is_err()
    );
}

#[test]
fn fault_witness_geometry_refuses_crossed_weights_windows_and_operands() {
    for (projection, index, windows, buffers) in [
        (Projection::Qkvz1, 0, 2, [16_777_216, 2048, 16384]),
        (Projection::Out2, 2, 1, [4_194_304, 8192, 4096]),
    ] {
        let cut = faults::FaultCut::for_projection(projection).unwrap();
        assert!(cut.validate_geometry(index, windows, buffers).is_ok());
        assert!(
            cut.validate_geometry((index + 1) % 3, windows, buffers)
                .is_err()
        );
        assert!(cut.validate_geometry(index, 3 - windows, buffers).is_err());
        for operand in 0..3 {
            let mut wrong = buffers;
            wrong[operand] -= 2;
            assert!(cut.validate_geometry(index, windows, wrong).is_err());
        }
    }
    for other in [
        Projection::Ba1,
        Projection::Out1,
        Projection::Qkvz64,
        Projection::Ba64,
        Projection::Out64,
        Projection::Qkvz2,
        Projection::Ba2,
    ] {
        assert!(faults::FaultCut::for_projection(other).is_err());
    }
}

#[test]
fn fault_witness_rejects_another_image_hash_geometry_or_provenance_category() {
    for (projection, image) in [
        (Projection::Qkvz1, ProjectionImage::Bf16Gemv2TSm86),
        (Projection::Out2, ProjectionImage::CutlassWmmaSm80),
    ] {
        let (derived_from, captured_from) = match image.provenance() {
            ProjectionImageSource::Derived(kind) => (Some(kind), None),
            ProjectionImageSource::Captured(kind) => (None, Some(kind)),
        };
        let identity = resident::NativeImageIdentity {
            bytes: image.bytes(),
            sha256: image.sha256(),
            binary_sm: image.binary_sm(),
            derived_from,
            captured_from,
        };
        let cut = faults::FaultCut::for_projection(projection).unwrap();
        assert!(cut.validate_image(&identity).is_ok());
        for mode in 0..4 {
            let mut wrong = identity;
            match mode {
                0 => wrong.sha256[0] ^= 1,
                1 => wrong.bytes -= 1,
                2 => wrong.binary_sm = if wrong.binary_sm == 80 { 86 } else { 80 },
                _ => {
                    wrong.derived_from = None;
                    wrong.captured_from = None;
                }
            }
            assert!(cut.validate_image(&wrong).is_err());
        }
        let opposite = if projection == Projection::Qkvz1 {
            Projection::Out2
        } else {
            Projection::Qkvz1
        };
        assert!(
            faults::FaultCut::for_projection(opposite)
                .unwrap()
                .validate_image(&identity)
                .is_err()
        );
    }
}

#[test]
fn canonical47_fa_union_is_exact_without_widening_old_image_validators() {
    let mut exact = super::super::canonical47_mlp::required_images();
    exact.extend([
        ProjectionImage::AttentionAmpere256x128Sm80,
        ProjectionImage::AttentionQkSm86,
        ProjectionImage::AttentionKvSm86,
        ProjectionImage::AttentionFaCausalCombineSm80,
    ]);
    assert_eq!(exact.len(), 8);
    require_canonical47_fa_image_sources(&exact).unwrap();
    assert!(require_image_sources(&exact).is_err());
    assert!(require_canonical47_qkkv_image_sources(&exact).is_err());
    for image in &exact {
        let mut wrong = exact.clone();
        wrong.remove(image);
        assert!(require_canonical47_fa_image_sources(&wrong).is_err());
        wrong.insert(ProjectionImage::Bf16Gemv2TSm86);
        assert_eq!(wrong.len(), 8);
        assert!(require_canonical47_fa_image_sources(&wrong).is_err());
    }
    let mut extra = exact;
    extra.insert(ProjectionImage::Bf16Gemv2TSm86);
    assert!(require_canonical47_fa_image_sources(&extra).is_err());
    assert!(require_canonical47_fa_image_sources(&BTreeSet::new()).is_err());
}

#[test]
fn canonical47_layer3_images_are_exact_without_expanding_fa_or_generic() {
    let mut images = super::super::canonical47_mlp::required_images();
    images.extend([
        ProjectionImage::AttentionAmpere256x128Sm80,
        ProjectionImage::AttentionQkSm86,
        ProjectionImage::AttentionKvSm86,
        ProjectionImage::AttentionFaCausalCombineSm80,
        ProjectionImage::AttentionSigmoidSm86,
        ProjectionImage::AttentionMultiplySm86,
    ]);
    assert_eq!(images.len(), 10);
    require_canonical47_layer3_image_sources(&images).unwrap();
    assert!(require_image_sources(&images).is_err());
    assert!(require_canonical47_fa_image_sources(&images).is_err());
    assert!(require_canonical47_qkkv_image_sources(&images).is_err());
    for image in &images {
        let mut wrong = images.clone();
        wrong.remove(image);
        assert!(require_canonical47_layer3_image_sources(&wrong).is_err());
        wrong.insert(ProjectionImage::Bf16Gemv2TSm86);
        assert_eq!(wrong.len(), 10);
        assert!(require_canonical47_layer3_image_sources(&wrong).is_err());
    }
    let mut wrong = images;
    wrong.insert(ProjectionImage::Bf16Gemv2TSm86);
    assert!(require_canonical47_layer3_image_sources(&wrong).is_err());
}
