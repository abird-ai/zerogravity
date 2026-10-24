//! Metadata/original host-ledger guards only; no CUDA client or fabricated handle.
use super::*;

#[test]
fn first_rows1_layout_keeps_exact_unready_output_and_two_descriptors() {
    let a = Activation::Rows1;
    let bytes = [a.input_bytes(), a.output_bytes()];
    let access = [DeviceAccess::Read, DeviceAccess::Write];
    let descriptors = [(0, bytes[0]), (0, bytes[1])];
    assert!(validate_layout(a, 5, bytes, access, [true, false], descriptors).is_ok());
    for activation in [Activation::Rows2, Activation::Rows64] {
        assert!(validate_layout(activation, 5, bytes, access, [true, false], descriptors).is_err());
    }
    for count in [0, 1, 4, 6] {
        assert!(validate_layout(a, count, bytes, access, [true, false], descriptors).is_err());
    }
    for changed in [[bytes[0] - 1, bytes[1]], [bytes[0], bytes[1] + 1]] {
        assert!(validate_layout(a, 5, changed, access, [true, false], descriptors).is_err());
    }
    for changed in [
        [DeviceAccess::Write, DeviceAccess::Write],
        [DeviceAccess::Read, DeviceAccess::Read],
    ] {
        assert!(validate_layout(a, 5, bytes, changed, [true, false], descriptors).is_err());
    }
    for initialized in [[false, false], [true, true]] {
        assert!(validate_layout(a, 5, bytes, access, initialized, descriptors).is_err());
    }
    for changed in [
        [(1, bytes[0]), descriptors[1]],
        [descriptors[0], (0, bytes[1] - 1)],
    ] {
        assert!(validate_layout(a, 5, bytes, access, [true, false], changed).is_err());
    }
}

#[test]
fn retained_activation_cannot_be_derived_cutlass_or_another_resource() {
    let kind = cubecl::cuda::PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86;
    let record = kind.capture();
    let original = NativeImageIdentity {
        bytes: record.image_bytes,
        sha256: record.image_sha256,
        binary_sm: record.binary_sm,
        derived_from: None,
        captured_from: Some(kind),
    };
    validate_image(&original).unwrap();
    for fault in 0..6 {
        let mut bad = original;
        match fault {
            0 => bad.bytes -= 1,
            1 => bad.sha256[0] ^= 1,
            2 => bad.binary_sm = 80,
            3 => bad.captured_from = None,
            4 => {
                bad.captured_from = Some(cubecl::cuda::PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80)
            }
            _ => bad.derived_from = Some(cubecl::cuda::PinnedBf16Cubin::Bf16Gemv2TSm86),
        }
        assert!(validate_image(&bad).is_err(), "fault{fault}");
    }
}

#[test]
fn exact_original_envelope_excludes_empty_and_additional_reservations() {
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "activation-fault-ledger-test").unwrap();
    arbiter
        .register(key.clone(), 2 * PROVIDER_CAP as u64)
        .unwrap();
    assert!(!original_envelope(&arbiter.snapshot(&key).unwrap()));
    let owner = ResourceOwner::new("activation", "original-host-ledger-test").unwrap();
    let envelope = arbiter
        .admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::ProviderEnvelope,
                bytes: PROVIDER_CAP as u64,
            }],
        )
        .unwrap();
    assert!(original_envelope(&arbiter.snapshot(&key).unwrap()));
    let extra = arbiter
        .admit(
            &key,
            owner,
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: 1,
            }],
        )
        .unwrap();
    assert!(!original_envelope(&arbiter.snapshot(&key).unwrap()));
    drop(extra);
    assert!(original_envelope(&arbiter.snapshot(&key).unwrap()));
    drop(envelope);
    assert!(!original_envelope(&arbiter.snapshot(&key).unwrap()));
}
