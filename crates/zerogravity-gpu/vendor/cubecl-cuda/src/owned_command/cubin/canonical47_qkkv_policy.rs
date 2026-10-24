//! Finite47 QK/KV union; exact images and complete source-derived ABI records.
use super::{CubinKernelManifest, VerifiedCubin, failure};
use cubecl_core::server::ServerError;
const ABIS: Option<[[u8; 32]; 2]> = Some([
    [
        0x54, 0x5c, 0xd3, 0xdb, 0x46, 0x43, 0x42, 0x0a, 0xd2, 0x56, 0x8e, 0x71, 0x88, 0x28, 0x3d,
        0x02, 0xab, 0x30, 0x39, 0x53, 0x14, 0xa9, 0xe9, 0x19, 0x65, 0xab, 0x3c, 0x58, 0xbf, 0x9e,
        0x3d, 0x78,
    ],
    [
        0x1f, 0x64, 0x61, 0x48, 0x04, 0xfd, 0x43, 0x48, 0xed, 0xfd, 0xbf, 0xbf, 0xb6, 0x75, 0xb1,
        0x75, 0xb3, 0xca, 0xc8, 0xe3, 0xc2, 0xd4, 0xa2, 0x09, 0x4a, 0xfe, 0xaa, 0x5c, 0x39, 0xc6,
        0x2a, 0x79,
    ],
]);
const IMAGES: [([u8; 32], usize, u8); 2] = [
    (
        [
            0x90, 0x49, 0xad, 0x21, 0xc9, 0x64, 0x12, 0x6f, 0xe3, 0x2a, 0x39, 0x4f, 0x6e, 0x53,
            0x35, 0xe9, 0xec, 0x09, 0x01, 0xcb, 0x03, 0x15, 0x59, 0xdb, 0x3c, 0xab, 0x0e, 0x63,
            0xd9, 0x3b, 0x46, 0x64,
        ],
        17584,
        86,
    ),
    (
        [
            0x07, 0x81, 0xfb, 0xbc, 0xae, 0x3a, 0x3f, 0x26, 0xcf, 0x97, 0x20, 0x5a, 0xf8, 0x06,
            0xe6, 0xe9, 0x00, 0x14, 0x13, 0x82, 0x7e, 0x66, 0xe2, 0x98, 0xd4, 0xa0, 0x74, 0x3c,
            0xde, 0x06, 0x70, 0x35,
        ],
        1813600,
        86,
    ),
];
pub(super) fn require(
    image: &VerifiedCubin,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    if super::canonical47_qkv_policy::require(image, manifest).is_ok() {
        return Ok(());
    }
    match (image.derived_from(), image.captured_from()) {
        (Some(pin), None) => {
            pin.require_image(image.byte_len(), image.sha256(), image.binary_sm())?
        }
        (None, Some(pin)) => {
            pin.require_image(image.byte_len(), image.sha256(), image.binary_sm())?
        }
        _ => {
            return Err(failure(
                "canonical47 QK/KV exactly one original sealed image provenance",
            ));
        }
    }
    require_record(
        image.sha256(),
        image.byte_len(),
        image.binary_sm(),
        manifest,
        ABIS,
    )
}
fn require_record(
    sha: [u8; 32],
    bytes: usize,
    sm: u8,
    manifest: &CubinKernelManifest,
    abis: Option<[[u8; 32]; 2]>,
) -> Result<(), ServerError> {
    let abis = abis
        .ok_or_else(|| failure("canonical47 QK/KV exact complete ABI pins pending ROOT proof"))?;
    let digest = super::attention_policy::digest(manifest);
    if !IMAGES
        .iter()
        .zip(abis)
        .any(|(image, abi)| *image == (sha, bytes, sm) && digest == abi)
    {
        return Err(failure(
            "canonical47 QK/KV full original image/entry/scalars/geometry/extents ABI differs",
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_command::{
        CubinBinding, CubinIo, CubinLaunchAttributes, CubinParameter, CubinPointerFixup,
    };

    #[test]
    fn complete_record_refuses_each_field_mutation_and_absent_pin() {
        // Unit-only fixture: no production ABI, image provenance or native permit.
        let m = CubinKernelManifest {
            entry: "qkkv-policy-fixture".into(),
            binary_sm: 86,
            grid: [47, 10, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 16,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![CubinBinding {
                bytes: 1560,
                alignment: 8,
                io: CubinIo::Read,
            }],
            parameters: vec![CubinParameter {
                offset: 64,
                template: vec![0; 8],
            }],
            fixups: vec![CubinPointerFixup {
                parameter: 0,
                byte_offset: 0,
                binding: 0,
                binding_offset: 0,
                extent: 1560,
                alignment: 8,
                io: CubinIo::Read,
            }],
        };
        let (sha, bytes, sm) = IMAGES[0];
        assert!(require_record(sha, bytes, sm, &m, None).is_err());
        assert!(require_record(sha, bytes, sm, &m, ABIS).is_err());
        let fixture_abis = Some([super::super::attention_policy::digest(&m), [0; 32]]);
        require_record(sha, bytes, sm, &m, fixture_abis).unwrap();
        for mutation in 0..17 {
            let mut wrong = m.clone();
            match mutation {
                0 => wrong.entry.push('x'),
                1 => wrong.binary_sm += 1,
                2 => wrong.grid[0] = 64,
                3 => wrong.block[0] += 1,
                4 => wrong.dynamic_shared_bytes += 16,
                5 => wrong.attributes = CubinLaunchAttributes::Unsupported,
                6 => wrong.bindings[0].bytes += 8,
                7 => wrong.bindings[0].alignment += 8,
                8 => wrong.bindings[0].io = CubinIo::ReadWrite,
                9 => wrong.parameters[0].offset += 8,
                10 => wrong.parameters[0].template[7] ^= 1,
                11 => wrong.fixups[0].parameter += 1,
                12 => wrong.fixups[0].byte_offset += 1,
                13 => wrong.fixups[0].binding += 1,
                14 => wrong.fixups[0].binding_offset += 8,
                15 => wrong.fixups[0].extent += 8,
                _ => wrong.fixups[0].io = CubinIo::Write,
            }
            assert!(
                require_record(sha, bytes, sm, &wrong, fixture_abis).is_err(),
                "mutation{mutation}"
            );
        }
        let mut wrong = sha;
        wrong[0] ^= 1;
        assert!(require_record(wrong, bytes, sm, &m, fixture_abis).is_err());
        assert!(require_record(sha, bytes + 1, sm, &m, fixture_abis).is_err());
        assert!(require_record(sha, bytes, sm + 1, &m, fixture_abis).is_err());
    }

    #[test]
    fn exact_union_capacity_does_not_expand_older_scopes() {
        use super::super::policy::CubinSetPolicy as P;
        assert!(P::Canonical47QkKv.limits(8, 11, 11, false, false).is_ok());
        for (modules, functions, records, new_module, new_function) in [
            (8, 11, 12, false, false),
            (8, 11, 11, true, false),
            (8, 11, 11, false, true),
        ] {
            assert!(
                P::Canonical47QkKv
                    .limits(modules, functions, records, new_module, new_function)
                    .is_err()
            );
        }
        assert!(P::Canonical47Qkv.limits(6, 9, 10, false, false).is_err());
        assert!(P::Qwen2bMlp.limits(5, 7, 7, false, false).is_err());
    }
    #[test]
    fn new_scope_cannot_inherit_a_legacy_layout_or_hash_only_permit() {
        assert!(ABIS.is_some());
        assert_eq!(IMAGES.iter().map(|i| i.1).sum::<usize>(), 1831184);
        assert_ne!(IMAGES[0].0, IMAGES[1].0);
        assert!(IMAGES.iter().all(|i| i.2 == 86));
    }
}
