use super::*;
use crate::owned_command::{
    CubinBinding, CubinIo, CubinLaunchAttributes, CubinParameter, CubinPointerFixup,
};
use std::collections::BTreeSet;

// Layout-policy tests only. No fake source issuer, CUDA context or arithmetic.
fn manifest(rule: &EntryRule) -> CubinKernelManifest {
    CubinKernelManifest {
        entry: rule.entry.into(),
        binary_sm: rule.image.binary_sm(),
        grid: [1, 1, 1],
        block: [32, 1, 1],
        dynamic_shared_bytes: rule.shared,
        attributes: CubinLaunchAttributes::None,
        bindings: vec![CubinBinding {
            bytes: 16,
            alignment: 16,
            io: CubinIo::Write,
        }],
        parameters: rule
            .params
            .iter()
            .map(|&(offset, size)| CubinParameter {
                offset,
                template: vec![0; size],
            })
            .collect(),
        fixups: vec![CubinPointerFixup {
            parameter: 0,
            byte_offset: 0,
            binding: 0,
            binding_offset: 0,
            extent: 16,
            alignment: 16,
            io: CubinIo::Write,
        }],
    }
}

#[test]
fn original_limits_stay_six_eight_nineteen() {
    let scope = CubinSetPolicy::ProjectionSix;
    assert!(scope.limits(6, 8, 18, false, false).is_ok());
    assert!(scope.limits(5, 7, 18, true, true).is_ok());
    for (modules, functions, records, new_module, new_function) in [
        (6, 8, 18, true, false),
        (6, 8, 18, false, true),
        (6, 8, 19, false, false),
        (7, 8, 0, false, false),
        (6, 9, 0, false, false),
    ] {
        assert!(
            scope
                .limits(modules, functions, records, new_module, new_function)
                .is_err()
        );
    }
}

#[test]
fn connected_limits_admit_exact_frontier_but_never_an_extra_record() {
    let scope = CubinSetPolicy::ConnectedQwen08;
    assert!(scope.limits(6, 11, 20, true, true).is_ok());
    assert!(scope.limits(7, 12, 20, false, false).is_ok());
    for (modules, functions, records, new_module, new_function) in [
        (7, 12, 20, true, false),
        (7, 12, 20, false, true),
        (7, 12, 21, false, false),
        (8, 12, 0, false, false),
        (7, 13, 0, false, false),
    ] {
        assert!(
            scope
                .limits(modules, functions, records, new_module, new_function)
                .is_err()
        );
    }
    assert!(
        CubinSetPolicy::ProjectionSix
            .limits(6, 11, 20, true, true)
            .is_err()
    );
}

#[test]
fn exact_original_union_has_seven_image_kinds_and_twelve_entries() {
    let images: BTreeSet<_> = RULES.iter().map(|r| r.image as u8).collect();
    let entries: BTreeSet<_> = RULES.iter().map(|r| (r.image as u8, r.entry)).collect();
    assert_eq!((images.len(), entries.len(), RULES.len()), (7, 12, 12));
    for rule in RULES {
        let actual = manifest(rule);
        actual.validate().unwrap();
        require_connected_manifest(rule.image, &actual).unwrap();
    }
}

#[test]
fn familiar_entry_under_wrong_sealed_image_refuses() {
    for rule in RULES {
        let actual = manifest(rule);
        for wrong in [
            ConnectedImage::Gemv2T,
            ConnectedImage::Gemvx,
            ConnectedImage::Cutlass,
            ConnectedImage::Reduce,
            ConnectedImage::Ampere128,
            ConnectedImage::Ampere64,
            ConnectedImage::Activation,
        ] {
            if wrong != rule.image {
                assert!(require_connected_manifest(wrong, &actual).is_err());
            }
        }
    }
}

#[test]
fn unknown_full_entry_and_shared_near_variants_refuse() {
    for rule in RULES {
        let mut actual = manifest(rule);
        actual.entry.push('_');
        assert!(require_connected_manifest(rule.image, &actual).is_err());
        let mut actual = manifest(rule);
        actual.dynamic_shared_bytes += 1;
        assert!(require_connected_manifest(rule.image, &actual).is_err());
        let mut actual = manifest(rule);
        actual.binary_sm = if actual.binary_sm == 80 { 86 } else { 80 };
        assert!(require_connected_manifest(rule.image, &actual).is_err());
    }
}

#[test]
fn reflected_parameter_offsets_sizes_counts_are_exact() {
    for rule in RULES {
        let mut actual = manifest(rule);
        actual.parameters[0].offset += 8;
        assert!(require_connected_manifest(rule.image, &actual).is_err());
        let mut actual = manifest(rule);
        actual.parameters[0].template.push(0);
        assert!(require_connected_manifest(rule.image, &actual).is_err());
        let mut actual = manifest(rule);
        actual.parameters.pop();
        assert!(require_connected_manifest(rule.image, &actual).is_err());
    }
}

#[test]
fn exact_seven_provenance_kinds_exclude_raw_intervals_or_merged_claims() {
    use PinnedBf16Cubin as D;
    use PinnedCapturedBf16Cubin as C;
    for (derived, captured, expected) in [
        (Some(D::Bf16Gemv2TSm86), None, ConnectedImage::Gemv2T),
        (Some(D::Bf16GemvxSm86), None, ConnectedImage::Gemvx),
        (Some(D::Bf16SplitKReduceSm86), None, ConnectedImage::Reduce),
        (
            Some(D::AmpereBf16Gemm128x64Sm80),
            None,
            ConnectedImage::Ampere128,
        ),
        (
            Some(D::AmpereBf16Gemm64x64Stages64x5Sm80),
            None,
            ConnectedImage::Ampere64,
        ),
        (None, Some(C::CutlassWmmaBf16Sm80), ConnectedImage::Cutlass),
        (
            None,
            Some(C::SiluAndMulBf16Sm86),
            ConnectedImage::Activation,
        ),
    ] {
        assert_eq!(
            ConnectedImage::from_provenance(derived, captured).unwrap(),
            expected
        );
    }
    assert!(ConnectedImage::from_provenance(None, None).is_err());
    assert!(
        ConnectedImage::from_provenance(Some(D::Bf16Gemv2TSm86), Some(C::SiluAndMulBf16Sm86))
            .is_err()
    );
}
