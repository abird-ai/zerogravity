use super::*;

#[test]
fn actual_five_phases_keep_selected_geometry_and_packed_bf16_extents() {
    let phases = [
        VllmGdnPhase::InitialOne,
        VllmGdnPhase::CachedDecode,
        VllmGdnPhase::Initial64,
        VllmGdnPhase::ContinuedTwo,
        VllmGdnPhase::CachedDecode,
    ];
    for (phase, rows) in phases.into_iter().zip([1, 1, 64, 2, 1]) {
        let operation = Activation::for_phase(phase).unwrap();
        assert_eq!(operation.rows(), rows);
        assert_eq!(
            operation.bindings(),
            &[ProjectionBinding::X, ProjectionBinding::Y]
        );
        let manifest = operation.manifest();
        assert_eq!(manifest.grid, [rows as u32, 1, 1]);
        assert_eq!(manifest.block, [448, 1, 1]);
        assert_eq!(manifest.dynamic_shared_bytes, 0);
        assert_eq!(manifest.attributes, CubinLaunchAttributes::None);
        assert_eq!(manifest.binary_sm, 86);
        assert_eq!(manifest.bindings.len(), 2);
        assert_eq!(manifest.bindings[0].bytes, (rows * 7168 * 2) as u64);
        assert_eq!(manifest.bindings[1].bytes, (rows * 3584 * 2) as u64);
        assert_eq!(manifest.bindings[0].io, CubinIo::Read);
        assert_eq!(manifest.bindings[1].io, CubinIo::Write);
    }
}

#[test]
fn native_output_first_pointer_abi_and_all_scalar_bits_are_preserved() {
    for operation in [Activation::Rows1, Activation::Rows64, Activation::Rows2] {
        let manifest = operation.manifest();
        let layout: Vec<_> = manifest
            .parameters
            .iter()
            .map(|p| (p.offset, p.template.len()))
            .collect();
        assert_eq!(layout, [(0, 8), (8, 8), (16, 4), (20, 4), (24, 4), (28, 4)]);
        assert_eq!(manifest.parameters[0].template, [0; 8]);
        assert_eq!(manifest.parameters[1].template, [0; 8]);
        assert_eq!(manifest.parameters[2].template, 3584u32.to_le_bytes());
        assert_eq!(manifest.parameters[3].template, 0u32.to_le_bytes());
        assert_eq!(
            manifest.parameters[4].template,
            1f32.to_bits().to_le_bytes()
        );
        assert_eq!(manifest.parameters[5].template, 0u32.to_le_bytes());
        assert_eq!(manifest.fixups.len(), 2);
        let output = &manifest.fixups[0];
        assert_eq!(
            (
                output.parameter,
                output.byte_offset,
                output.binding,
                output.binding_offset
            ),
            (0, 0, 1, 0)
        );
        assert_eq!(output.extent, operation.output_bytes() as u64);
        assert_eq!(output.io, CubinIo::Write);
        let input = &manifest.fixups[1];
        assert_eq!(
            (
                input.parameter,
                input.byte_offset,
                input.binding,
                input.binding_offset
            ),
            (1, 0, 0, 0)
        );
        assert_eq!(input.extent, operation.input_bytes() as u64);
        assert_eq!(input.io, CubinIo::Read);
        assert_eq!((input.alignment, output.alignment), (16, 16));
    }
}

#[test]
fn selected_image_metadata_keeps_exact_cuda129_version_header() {
    assert_eq!(IMAGE_BYTES, 901_664);
    assert_eq!(
        &IMAGE_HEADER[..16],
        b"\x7fELF\x02\x01\x01\x33\x07\0\0\0\0\0\0\0"
    );
    assert_eq!(&IMAGE_HEADER[16..24], &[2, 0, 190, 0, 129, 0, 0, 0]);
    assert_eq!(IMAGE_SHA256.len(), 32);
    let phoff = u64::from_le_bytes(IMAGE_HEADER[32..40].try_into().unwrap());
    let phsize = u16::from_le_bytes(IMAGE_HEADER[54..56].try_into().unwrap());
    let phcount = u16::from_le_bytes(IMAGE_HEADER[56..58].try_into().unwrap());
    assert_eq!(
        phoff + u64::from(phsize) * u64::from(phcount),
        IMAGE_BYTES as u64
    );
}
#[test]
fn separately_captured47_abi_preserves_the_legacy_phase_refusal() {
    assert!(Activation::for_phase(VllmGdnPhase::Initial47).is_err());
    let m = super::canonical47_manifest();
    assert_eq!(m.grid, [47, 1, 1]);
    assert_eq!(m.block, [448, 1, 1]);
    assert_eq!(m.bindings[0].bytes, 673_792);
    assert_eq!(m.bindings[1].bytes, 336_896);
    assert_eq!(
        m.parameters
            .iter()
            .flat_map(|p| p.template.iter())
            .copied()
            .collect::<Vec<_>>(),
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63,
            0, 0, 0, 0
        ]
    );
}
