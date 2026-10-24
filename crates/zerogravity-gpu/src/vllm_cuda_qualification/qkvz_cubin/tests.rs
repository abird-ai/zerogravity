use super::*;
use cubecl::cuda::{CubinIo, CubinLaunchAttributes};

fn fixtures() -> [QkvzCubinFixture; 3] {
    TEACHERS.map(|teacher_id| QkvzCubinFixture {
        teacher_id,
        normalized: vec![0; INPUT_BYTES],
        qkvz: vec![0; OUTPUT_BYTES],
    })
}

#[test]
fn full_image_capacity_must_fit_original_admission() {
    assert!(validate_image_capacity(IMAGE_BYTES, IMAGE_BYTES, IMAGE_BYTES).is_ok());
    assert!(validate_image_capacity(IMAGE_BYTES, HOST_BYTES, HOST_BYTES).is_ok());
    for (bytes, capacity, admission) in [
        (IMAGE_BYTES - 1, IMAGE_BYTES, HOST_BYTES),
        (IMAGE_BYTES, IMAGE_BYTES - 1, HOST_BYTES),
        (IMAGE_BYTES, IMAGE_BYTES + 1, IMAGE_BYTES),
        (IMAGE_BYTES, HOST_BYTES + 1, usize::MAX),
        (0, 0, HOST_BYTES),
    ] {
        assert!(validate_image_capacity(bytes, capacity, admission).is_err());
    }
}

#[test]
fn closed_teachers_and_complete_finite_expectations_only() {
    assert!(validate_fixtures(&fixtures()).is_ok());
    for kind in 0..5 {
        let mut values = fixtures();
        match kind {
            0 => values.swap(0, 1),
            1 => {
                values[1].normalized.pop();
            }
            2 => values[2].qkvz.extend_from_slice(&0u16.to_le_bytes()),
            3 => values[0].qkvz[..2].copy_from_slice(&0x7fc1u16.to_le_bytes()),
            _ => values[1].normalized[..2].copy_from_slice(&0x7f80u16.to_le_bytes()),
        }
        assert!(validate_fixtures(&values).is_err(), "case{kind}");
    }
    let mut values = fixtures();
    values[0].qkvz = Vec::with_capacity(1024 * 1024 + 1);
    values[0].qkvz.resize(OUTPUT_BYTES, 0);
    assert!(validate_fixtures(&values).is_err(), "owned spare capacity");
}

#[test]
fn comparison_detects_one_bit_and_nonfinite_without_ready_claim() {
    let expected = vec![0; INPUT_BYTES];
    assert!(compare_output(0, &expected, &expected).is_ok());
    let mut changed = expected.clone();
    changed[INPUT_BYTES - 1] = 1;
    assert!(compare_output(0, &changed, &expected).is_err());
    changed[..2].copy_from_slice(&0xff80u16.to_le_bytes());
    assert!(compare_output(0, &changed, &expected).is_err());
    assert!(compare_output(0, &expected[..INPUT_BYTES - 2], &expected).is_err());
}

#[test]
fn n1_actual_abi_has_original_scalars_and_only_w_x_y_bindings() {
    let manifest = closed_manifest().unwrap();
    assert_eq!(manifest.binary_sm, 86);
    assert_eq!(manifest.grid, [1024, 1, 1]);
    assert_eq!(manifest.block, [128, 1, 1]);
    assert_eq!(manifest.dynamic_shared_bytes, 0);
    assert_eq!(manifest.attributes, CubinLaunchAttributes::None);
    assert_eq!(manifest.parameters.len(), 3);
    let actual = manifest
        .parameters
        .iter()
        .map(|p| (p.offset, p.template.len()))
        .collect::<Vec<_>>();
    assert_eq!(actual, [(0, 152), (152, 4), (156, 4)]);
    assert_eq!(manifest.parameters[1].template, 1f32.to_le_bytes());
    assert_eq!(manifest.parameters[2].template, 0f32.to_le_bytes());
    assert_eq!(
        manifest
            .bindings
            .iter()
            .map(|b| (b.bytes, b.io))
            .collect::<Vec<_>>(),
        [
            (WEIGHT_BYTES as u64, CubinIo::Read),
            (INPUT_BYTES as u64, CubinIo::Read),
            (OUTPUT_BYTES as u64, CubinIo::Write),
        ]
    );
    assert_eq!(
        manifest
            .fixups
            .iter()
            .map(|f| (
                f.parameter,
                f.byte_offset,
                f.binding,
                f.binding_offset,
                f.extent,
                f.io
            ))
            .collect::<Vec<_>>(),
        [
            (0, 0, 0, 0, WEIGHT_BYTES as u64, CubinIo::Read),
            (0, 16, 1, 0, INPUT_BYTES as u64, CubinIo::Read),
            (0, 32, 2, 0, OUTPUT_BYTES as u64, CubinIo::Write),
            (0, 48, 2, 0, OUTPUT_BYTES as u64, CubinIo::Write),
        ]
    );
    for fixup in &manifest.fixups {
        assert_eq!(
            &manifest.parameters[fixup.parameter].template
                [fixup.byte_offset..fixup.byte_offset + 8],
            &[0; 8]
        );
    }
}
