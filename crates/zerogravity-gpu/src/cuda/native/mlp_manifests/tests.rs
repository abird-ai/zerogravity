use super::*;
use std::collections::BTreeSet;

const PROJECTIONS: [MlpProjection; 6] = [
    MlpProjection::GateUp1,
    MlpProjection::Down1,
    MlpProjection::GateUp64,
    MlpProjection::Down64,
    MlpProjection::GateUp2,
    MlpProjection::Down2,
];

fn kernels(projection: MlpProjection) -> Vec<MlpLaunch> {
    projection
        .operations()
        .into_iter()
        .filter_map(|operation| match operation {
            MlpOperation::Kernel(launch) => Some(launch),
            MlpOperation::Clear(_) => None,
        })
        .collect()
}

fn bytes(manifest: &CubinKernelManifest) -> Vec<u8> {
    let extent = manifest
        .parameters
        .iter()
        .map(|p| p.offset + p.template.len())
        .max()
        .unwrap();
    let mut out = vec![0; extent];
    for parameter in &manifest.parameters {
        out[parameter.offset..parameter.offset + parameter.template.len()]
            .copy_from_slice(&parameter.template);
    }
    out
}

#[test]
fn actual_five_calls_have_thirteen_kernels_two_clears_seven_templates() {
    let phases = [
        GdnPhase::InitialOne,
        GdnPhase::CachedDecode,
        GdnPhase::Initial64,
        GdnPhase::ContinuedTwo,
        GdnPhase::CachedDecode,
    ];
    let mut templates = BTreeSet::new();
    let (mut launches, mut clears) = (0, 0);
    for projection in phases
        .into_iter()
        .flat_map(|phase| MlpProjection::for_phase(phase).unwrap())
    {
        for operation in projection.operations() {
            match operation {
                MlpOperation::Clear(_) => clears += 1,
                MlpOperation::Kernel(launch) => {
                    launches += 1;
                    let manifest = launch.into_manifest();
                    templates.insert((manifest.entry.clone(), bytes(&manifest)));
                }
            }
        }
    }
    assert_eq!((launches, clears, templates.len()), (13, 2, 7));
}

#[test]
fn each_closed_projection_keeps_real_weight_input_output_extents() {
    for projection in PROJECTIONS {
        for launch in kernels(projection) {
            let roles = launch.bindings();
            let image = launch.image();
            let manifest = launch.into_manifest();
            assert_eq!(manifest.binary_sm, image.binary_sm());
            assert_eq!(manifest.attributes, CubinLaunchAttributes::None);
            assert_eq!(manifest.bindings.len(), roles.len());
            for (binding, role) in manifest.bindings.iter().zip(roles) {
                let (expected, io) = match role {
                    ProjectionBinding::W => (projection.weight_bytes(), CubinIo::Read),
                    ProjectionBinding::X => (projection.input_bytes(), CubinIo::Read),
                    ProjectionBinding::Y => (projection.output_bytes(), CubinIo::Write),
                    ProjectionBinding::Workspace => (8_519_680, binding.io),
                };
                assert_eq!(binding.bytes, expected as u64);
                assert_eq!(binding.io, io);
                assert_eq!(binding.alignment, 16);
            }
            let mut covered = BTreeSet::new();
            for fixup in &manifest.fixups {
                let parameter = &manifest.parameters[fixup.parameter];
                assert_eq!(fixup.binding_offset, 0);
                assert_eq!(fixup.alignment, 16);
                assert_eq!(
                    &parameter.template[fixup.byte_offset..fixup.byte_offset + 8],
                    &[0; 8]
                );
                assert!(fixup.extent <= manifest.bindings[fixup.binding].bytes);
                assert!(covered.insert((fixup.parameter, fixup.byte_offset)));
            }
        }
    }
}

#[test]
fn serial_splitk_requires_actual_clear_before_same_kernel_readwrite() {
    for (projection, expected) in [(MlpProjection::Down64, 32), (MlpProjection::Down2, 64)] {
        let mut operations = projection.operations().into_iter();
        let Some(MlpOperation::Clear(clear)) = operations.next() else {
            panic!("original native clear must precede serial split-K")
        };
        assert_eq!(clear.binding(), ProjectionBinding::Workspace);
        assert_eq!(clear.binding_bytes(), 8_519_680);
        assert_eq!(clear.byte_offset(), 0);
        assert_eq!(clear.bytes(), expected);
        assert_eq!(clear.value(), 0);
        assert_eq!(clear.io(), CubinIo::Write);
        let Some(MlpOperation::Kernel(launch)) = operations.next() else {
            panic!("actual serial kernel follows the real clear")
        };
        assert!(operations.next().is_none());
        let workspace = launch
            .bindings()
            .iter()
            .position(|v| *v == ProjectionBinding::Workspace)
            .unwrap();
        let manifest = launch.into_manifest();
        assert_eq!(manifest.bindings[workspace].io, CubinIo::ReadWrite);
        let semaphore = manifest
            .fixups
            .iter()
            .find(|f| f.binding == workspace)
            .unwrap();
        assert_eq!(semaphore.byte_offset, 120);
        assert_eq!(semaphore.extent, expected as u64);
        assert_eq!(semaphore.io, CubinIo::ReadWrite);
        let raw = bytes(&manifest);
        assert_eq!(&raw[132..136], &0x100cu32.to_le_bytes());
        assert_eq!(&raw[152..156], &1f32.to_bits().to_le_bytes());
        assert_eq!(&raw[156..164], &[0; 8]);
        assert_eq!(&raw[176..184], &[0, 0, 0, 16, 0, 0, 0, 16]);
        assert_eq!(manifest.grid[2], 2);
    }
    for projection in [
        MlpProjection::GateUp1,
        MlpProjection::Down1,
        MlpProjection::GateUp64,
        MlpProjection::GateUp2,
    ] {
        assert!(
            projection
                .operations()
                .iter()
                .all(|op| matches!(op, MlpOperation::Kernel(_)))
        );
    }
}

#[test]
fn parallel_splitk_retains_full_workspace_and_bf16_partial_dependency() {
    let launches = kernels(MlpProjection::Down1);
    assert_eq!(launches.len(), 2);
    for (ordinal, launch) in launches.into_iter().enumerate() {
        let workspace = launch
            .bindings()
            .iter()
            .position(|v| *v == ProjectionBinding::Workspace)
            .unwrap();
        let manifest = launch.into_manifest();
        let expected = if ordinal == 0 {
            CubinIo::Write
        } else {
            CubinIo::Read
        };
        assert_eq!(manifest.bindings[workspace].bytes, 8_519_680);
        assert_eq!(manifest.bindings[workspace].io, expected);
        let partial = manifest
            .fixups
            .iter()
            .find(|f| f.binding == workspace)
            .unwrap();
        assert_eq!(partial.extent, 6144);
        assert_eq!(partial.io, expected);
        let raw = bytes(&manifest);
        if ordinal == 0 {
            assert_eq!(&raw[20..24], &3u32.to_le_bytes());
            assert_eq!(&raw[36..40], &1216u32.to_le_bytes());
            assert_eq!(&raw[256..260], &u32::MAX.to_le_bytes());
            assert_eq!(&raw[260..264], &[0; 4]);
        } else {
            assert_eq!(&raw[40..44], &3u32.to_le_bytes());
            assert_eq!(&raw[44..48], &[0; 4]);
            assert_eq!(&raw[96..112], &[0; 16]);
            assert!(!manifest.fixups.iter().any(|f| {
                let at = manifest.parameters[f.parameter].offset + f.byte_offset;
                (96..112).contains(&at)
            }));
        }
    }
}

#[test]
fn native_geometry_and_entries_are_distinct_even_when_images_are_shared() {
    let expected = [
        (
            MlpProjection::GateUp64,
            ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
            "ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_tn",
            [28, 1, 1],
            32768,
            176,
        ),
        (
            MlpProjection::Down64,
            ProjectionImage::AmpereBf16Gemm128x64Sm80,
            "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn",
            [8, 1, 2],
            24576,
            192,
        ),
        (
            MlpProjection::Down2,
            ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
            "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn",
            [16, 1, 2],
            49152,
            192,
        ),
    ];
    for (projection, image, entry, grid, shared, extent) in expected {
        let launch = kernels(projection).remove(0);
        assert_eq!(launch.image(), image);
        let manifest = launch.into_manifest();
        assert_eq!(manifest.entry, entry);
        assert_eq!(manifest.grid, grid);
        assert_eq!(manifest.block, [128, 1, 1]);
        assert_eq!(manifest.dynamic_shared_bytes, shared);
        assert_eq!(bytes(&manifest).len(), extent);
    }
    let manifest = kernels(MlpProjection::GateUp2).remove(0).into_manifest();
    let raw = bytes(&manifest);
    assert_eq!(manifest.grid, [8, 56, 1]);
    assert_eq!(&raw[28..32], &[0; 4]);
    assert_eq!(&raw[20..24], &1u32.to_le_bytes());
    assert_eq!(&raw[256..260], &u32::MAX.to_le_bytes());
}
