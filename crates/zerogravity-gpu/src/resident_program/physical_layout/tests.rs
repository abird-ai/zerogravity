use super::*;

#[test]
fn dense_physical_words_have_explicit_byte_width_and_subranges() {
    for n in [1, 3, 1024, 2048, 65536] {
        let f32 = ResidentBufferLayout::new(ResidentPhysicalType::F32, n).unwrap();
        let bf16 = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, n).unwrap();
        assert_eq!(f32.bytes().unwrap(), n * 4);
        assert_eq!(bf16.bytes().unwrap(), n * 2);
        assert_eq!(f32.element_range(0..n).unwrap(), 0..n * 4);
        assert_eq!(bf16.element_range(0..n).unwrap(), 0..n * 2);
        if n > 1 {
            assert_eq!(bf16.element_range(1..n).unwrap(), 2..n * 2);
        }
    }
}

#[test]
fn legacy_f32_byte_domain_keeps_original_zero_and_u32_rules() {
    for n in [0, 1, 1024, u32::MAX as usize, usize::MAX] {
        let original = n.checked_mul(4).filter(|_| u32::try_from(n).is_ok());
        assert_eq!(ResidentBufferLayout::legacy_f32(n).bytes().ok(), original);
    }
}

#[test]
fn typed_extent_refuses_empty_index_overflow_and_outside_subranges() {
    for kind in [ResidentPhysicalType::F32, ResidentPhysicalType::Bf16] {
        for n in [0, usize::MAX, usize::MAX / 2 + 1] {
            assert!(ResidentBufferLayout::new(kind, n).is_err());
        }
        if usize::BITS > 32 {
            assert!(ResidentBufferLayout::new(kind, u32::MAX as usize + 1).is_err());
        }
        let layout = ResidentBufferLayout::new(kind, 8).unwrap();
        for range in [0..0, 8..8, 8..9, 7..9, std::ops::Range { start: 3, end: 2 }] {
            assert!(layout.element_range(range).is_err());
        }
    }
}

#[test]
fn same_bytes_cannot_reinterpret_bf16_as_f32_or_resize_elements() {
    let bf16 = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, 8).unwrap();
    let f32 = ResidentBufferLayout::new(ResidentPhysicalType::F32, 4).unwrap();
    assert_eq!(bf16.bytes().unwrap(), f32.bytes().unwrap());
    assert!(bf16.require(f32).is_err());
    assert!(f32.require(bf16).is_err());
    assert!(
        bf16.require(ResidentBufferLayout::new(ResidentPhysicalType::Bf16, 7).unwrap())
            .is_err()
    );
    assert!(bf16.require(bf16).is_ok());
}

#[test]
fn typed_regions_reject_misalignment_padding_short_reads_and_overflow() {
    for kind in [ResidentPhysicalType::F32, ResidentPhysicalType::Bf16] {
        let layout = ResidentBufferLayout::new(kind, 3).unwrap();
        let bytes = layout.bytes().unwrap() as u64;
        let width = kind.byte_width() as u64;
        assert!(layout.validate_region(0, bytes).is_ok());
        assert!(layout.validate_region(width, bytes).is_ok());
        for (offset, extent) in [
            (1, bytes),
            (0, bytes - 1),
            (0, bytes + width),
            (u64::MAX - width + 1, bytes),
        ] {
            assert!(layout.validate_region(offset, extent).is_err());
        }
    }
}

#[test]
fn typed_roles_do_not_admit_weight_control_or_bf16_canonical_state() {
    for kind in [ResidentPhysicalType::F32, ResidentPhysicalType::Bf16] {
        let layout = ResidentBufferLayout::new(kind, 1024).unwrap();
        for access in [DeviceAccess::Read, DeviceAccess::Write] {
            for role in [
                BufferRole::Activation,
                BufferRole::Output,
                BufferRole::Scratch,
            ] {
                assert!(layout.validate_role(role, access).is_ok());
            }
            assert!(
                layout
                    .validate_role(BufferRole::ImmutableWeights, access)
                    .is_err()
            );
            assert_eq!(
                layout
                    .validate_role(BufferRole::RecurrentState, access)
                    .is_ok(),
                kind == ResidentPhysicalType::F32
            );
        }
    }
}
