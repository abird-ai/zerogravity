//! Scope/metadata guards only; no fabricated CUDA handle or inference result.
use super::*;

#[test]
fn unknown_scope_is_exactly_one_down64_leaf() {
    // Layout/finite checks remain in the same positive runner before this guard.
    let fixture = |projection| CubinMlpFixture {
        projection,
        input: Vec::new(),
        expected: Vec::new(),
    };
    assert!(validate_scope(&[fixture(MlpProjection::Down64)]).is_ok());
    assert!(validate_scope(&[]).is_err());
    assert!(
        validate_scope(&[
            fixture(MlpProjection::Down64),
            fixture(MlpProjection::Down64)
        ])
        .is_err()
    );
    for projection in [
        MlpProjection::GateUp1,
        MlpProjection::Down1,
        MlpProjection::GateUp64,
        MlpProjection::GateUp2,
        MlpProjection::Down2,
    ] {
        assert!(validate_scope(&[fixture(projection)]).is_err());
    }
}

#[test]
fn down64_cut_requires_original_access_and_both_unready_writes() {
    let p = MlpProjection::Down64;
    let bytes = [
        p.weight_bytes(),
        p.input_bytes(),
        p.output_bytes(),
        WORKSPACE_BYTES,
    ];
    let access = [
        DeviceAccess::Read,
        DeviceAccess::Read,
        DeviceAccess::Write,
        DeviceAccess::Write,
    ];
    let descriptors = [(0, p.input_bytes()), (0, p.output_bytes())];
    assert!(validate_layout(p, 1, bytes, access, [true, true, false, false], descriptors).is_ok());
    for count in [0, 2, 10] {
        assert!(
            validate_layout(
                p,
                count,
                bytes,
                access,
                [true, true, false, false],
                descriptors
            )
            .is_err()
        );
    }
    for index in 0..4 {
        let mut bad_bytes = bytes;
        bad_bytes[index] += 1;
        assert!(
            validate_layout(
                p,
                1,
                bad_bytes,
                access,
                [true, true, false, false],
                descriptors
            )
            .is_err()
        );
        let mut bad_access = access;
        bad_access[index] = if index < 2 {
            DeviceAccess::Write
        } else {
            DeviceAccess::Read
        };
        assert!(
            validate_layout(
                p,
                1,
                bytes,
                bad_access,
                [true, true, false, false],
                descriptors
            )
            .is_err()
        );
        let mut initialized = [true, true, false, false];
        initialized[index] = !initialized[index];
        assert!(validate_layout(p, 1, bytes, access, initialized, descriptors).is_err());
    }
    for descriptors in [
        [(1, p.input_bytes()), (0, p.output_bytes())],
        [(0, p.input_bytes()), (0, p.output_bytes() - 1)],
    ] {
        assert!(
            validate_layout(p, 1, bytes, access, [true, true, false, false], descriptors).is_err()
        );
    }
}

#[test]
fn down64_native_owner_is_one_exact_derived_image() {
    let kind = cubecl::cuda::PinnedBf16Cubin::AmpereBf16Gemm128x64Sm80;
    let record = kind.derivation();
    let original = NativeImageIdentity {
        bytes: record.image_bytes,
        sha256: record.image_sha256,
        binary_sm: record.binary_sm,
        derived_from: Some(kind),
        captured_from: None,
    };
    validate_image(&original).unwrap();
    for fault in 0..6 {
        let mut bad = original;
        match fault {
            0 => bad.bytes -= 1,
            1 => bad.sha256[31] ^= 1,
            2 => bad.binary_sm = 86,
            3 => bad.derived_from = None,
            4 => bad.derived_from = Some(cubecl::cuda::PinnedBf16Cubin::Bf16Gemv2TSm86),
            _ => {
                bad.captured_from = Some(cubecl::cuda::PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80)
            }
        }
        assert!(validate_image(&bad).is_err(), "fault{fault}");
    }
}
