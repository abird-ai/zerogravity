use super::*;

const OPERATIONS: [Projection; 9] = [
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

#[test]
fn actual_phase_groups_have_eleven_templates_and_nineteen_launches() {
    assert_eq!(
        OPERATIONS.iter().map(|p| p.launches().len()).sum::<usize>(),
        11
    );
    let phases = [
        GdnPhase::InitialOne,
        GdnPhase::CachedDecode,
        GdnPhase::Initial64,
        GdnPhase::ContinuedTwo,
        GdnPhase::CachedDecode,
    ];
    assert_eq!(
        phases
            .into_iter()
            .flat_map(|phase| Projection::for_phase(phase).unwrap())
            .map(|p| p.launches().len())
            .sum::<usize>(),
        19
    );
    for operation in OPERATIONS {
        let launches = operation.launches();
        for launch in launches {
            let metadata = launch.image();
            let roles = launch.bindings();
            let manifest = launch.into_manifest();
            assert_eq!(manifest.binary_sm, metadata.binary_sm());
            for (binding, role) in manifest.bindings.iter().zip(roles) {
                let bytes = match role {
                    ProjectionBinding::W => operation.weight_bytes(),
                    ProjectionBinding::X => operation.input_bytes(),
                    ProjectionBinding::Y => operation.output_bytes(),
                    ProjectionBinding::Workspace => WORKSPACE_BYTES,
                };
                assert_eq!(binding.bytes, bytes as u64);
            }
            for fixup in &manifest.fixups {
                assert_eq!(fixup.binding_offset, 0);
                assert_eq!(fixup.alignment, 16);
                assert_eq!(
                    &manifest.parameters[fixup.parameter].template
                        [fixup.byte_offset..fixup.byte_offset + 8],
                    &[0; 8]
                );
                assert!(fixup.extent <= manifest.bindings[fixup.binding].bytes);
            }
        }
    }
}

#[test]
fn splitk_uses_same_full_managed_workspace_and_exact_logical_dependency() {
    for (operation, partial) in [(Projection::Out1, 6144), (Projection::Ba64, 65536)] {
        let launches = operation.launches();
        assert_eq!(launches.len(), 2);
        for (index, launch) in launches.into_iter().enumerate() {
            let roles = launch.bindings();
            let workspace = roles
                .iter()
                .position(|role| *role == ProjectionBinding::Workspace)
                .unwrap();
            let manifest = launch.into_manifest();
            assert_eq!(manifest.bindings[workspace].bytes, 8_519_680);
            assert_eq!(
                manifest.bindings[workspace].io,
                if index == 0 {
                    CubinIo::Write
                } else {
                    CubinIo::Read
                }
            );
            let fixup = manifest
                .fixups
                .iter()
                .find(|f| f.binding == workspace)
                .unwrap();
            assert_eq!(fixup.extent, partial);
            if index == 0 {
                assert_eq!(
                    &manifest.parameters[0].template[256..260],
                    &u32::MAX.to_le_bytes()
                );
                assert_eq!(&manifest.parameters[0].template[260..264], &[0; 4]);
            } else {
                assert_eq!(&manifest.parameters[0].template[44..48], &[0; 4]);
                for offset in [96, 104] {
                    let parameter = manifest
                        .parameters
                        .iter()
                        .find(|p| p.offset == offset)
                        .unwrap();
                    assert_eq!(parameter.template, [0; 8]);
                    assert!(
                        !manifest
                            .fixups
                            .iter()
                            .any(|f| manifest.parameters[f.parameter].offset + f.byte_offset
                                == offset)
                    );
                }
            }
        }
    }
    for operation in [Projection::Qkvz2, Projection::Ba2, Projection::Out2] {
        let manifest = operation.launches().remove(0).into_manifest();
        assert_eq!(
            &manifest.parameters[0].template[256..260],
            &u32::MAX.to_le_bytes()
        );
        assert_eq!(&manifest.parameters[0].template[260..264], &[0; 4]);
    }
}

#[test]
fn captured_original_image_and_five_derived_images_keep_distinct_provenance() {
    let image = ProjectionImage::CutlassWmmaSm80;
    let ProjectionImageSource::Captured(pinned) = image.provenance() else {
        panic!("captured input must not claim carrier derivation")
    };
    let captured = pinned.capture();
    assert!(captured.preserved_input_resource_equal);
    assert!(!captured.immutable_carrier_membership_proven);
    assert!(!captured.runtime_derivation_proven);
    assert_eq!(captured.image_bytes, image.bytes());
    assert_eq!(captured.image_sha256, image.sha256());
    assert_eq!(captured.binary_sm, image.binary_sm());
    for image in [
        ProjectionImage::Bf16Gemv2TSm86,
        ProjectionImage::Bf16GemvxSm86,
        ProjectionImage::Bf16SplitKReduceSm86,
        ProjectionImage::AmpereBf16Gemm128x64Sm80,
        ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
    ] {
        let ProjectionImageSource::Derived(pinned) = image.provenance() else {
            panic!("existing derived proof must remain unchanged")
        };
        let source = pinned.derivation();
        assert_eq!(source.image_bytes, image.bytes());
        assert_eq!(source.image_sha256, image.sha256());
        assert_eq!(source.binary_sm, image.binary_sm());
    }
}
