//! Exact known .8B mathematical ABI union; no model/rotary/Ready authority.
use super::{
    CubinKernelManifest, PinnedBf16Cubin, PinnedCapturedBf16Cubin, VerifiedCubin, failure,
};
use cubecl_core::server::ServerError;

pub(super) const MODULES: usize = 14;
pub(super) const FUNCTIONS: usize = 22;
// Separate factories prepare21 GDN/MLP/ACT +26 attention/gate +one head.
// Identical values across factories still consume separate actor records.
pub(super) const RECORDS: usize = 48;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageKind {
    Derived(PinnedBf16Cubin),
    Captured(PinnedCapturedBf16Cubin),
}
struct Rule {
    kind: ImageKind,
    bytes: usize,
    sm: u8,
    sha: [u8; 32],
    abi: [u8; 32],
}
const fn literal<const N: usize>(hex: &str) -> [u8; N] {
    const fn digit(n: u8) -> u8 {
        match n {
            b'0'..=b'9' => n - b'0',
            b'a'..=b'f' => n - b'a' + 10,
            _ => panic!("fixed source hex"),
        }
    }
    assert!(hex.len() == N * 2);
    let mut output = [0; N];
    let raw = hex.as_bytes();
    let mut i = 0;
    while i < N {
        output[i] = digit(raw[2 * i]) << 4 | digit(raw[2 * i + 1]);
        i += 1;
    }
    output
}
const RULES: &[Rule] = &[
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::Bf16GemvxSm86),
        bytes: 5273248,
        sm: 86,
        sha: literal("02cd5eee5a1f93addca2135cde7dfd71b05e499547e227a43806ee953eb2fa99"),
        abi: literal("d9de581fc1883887393db01770d551940a2be7e9cf9f42209f67fe8a1d63dda5"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::Bf16SplitKReduceSm86),
        bytes: 4252960,
        sm: 86,
        sha: literal("074807d671d9156ccc56b5a8cbe82f1ba8fae21c8169e674b31b4b8603ba42ec"),
        abi: literal("b0487cd229b7229a1b9e7e0d8f976ae932622bbb0ab8ec0c555e1bea760ea914"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::Bf16SplitKReduceSm86),
        bytes: 4252960,
        sm: 86,
        sha: literal("074807d671d9156ccc56b5a8cbe82f1ba8fae21c8169e674b31b4b8603ba42ec"),
        abi: literal("c688c3c56911c6313280fceb5481b09238f277812ae2c2d194487a85f13796df"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::Bf16SplitKReduceSm86),
        bytes: 4252960,
        sm: 86,
        sha: literal("074807d671d9156ccc56b5a8cbe82f1ba8fae21c8169e674b31b4b8603ba42ec"),
        abi: literal("d3b4cc9d41e830fdec228740fce0c519c4348b8251e86ac1042c1c307018def0"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionKvSm86),
        bytes: 1813600,
        sm: 86,
        sha: literal("0781fbbcae3a3f26cf97205af806e6e9001413827e66e298d4a0743cde067035"),
        abi: literal("040f80ec3e2fe76c1f892119da6d3bc1aaae6f41919a282f47055d6f10e26f4d"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionKvSm86),
        bytes: 1813600,
        sm: 86,
        sha: literal("0781fbbcae3a3f26cf97205af806e6e9001413827e66e298d4a0743cde067035"),
        abi: literal("3539646b0364b22d21b04c6c953aa157d68fdfe1e6402c1e174d7cfae6cf3c77"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionKvSm86),
        bytes: 1813600,
        sm: 86,
        sha: literal("0781fbbcae3a3f26cf97205af806e6e9001413827e66e298d4a0743cde067035"),
        abi: literal("f15fab025664a6bf9d9d3283ce0d4587f951db238eb266345c615f23d7798459"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionSigmoidSm86),
        bytes: 2679648,
        sm: 86,
        sha: literal("08c00ee67fa7452b5370a7b2fc2b611afdbe219b5343cebf935aeef6e872ef8b"),
        abi: literal("2c10f6b35639058fceee2a67a289b4c2e3f20d1d90e57dd799d334a714462bde"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionSigmoidSm86),
        bytes: 2679648,
        sm: 86,
        sha: literal("08c00ee67fa7452b5370a7b2fc2b611afdbe219b5343cebf935aeef6e872ef8b"),
        abi: literal("95b7da110621a7c4db153e97ce17b85255ae1a51e4f82da8fd134be6da3e449b"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionSigmoidSm86),
        bytes: 2679648,
        sm: 86,
        sha: literal("08c00ee67fa7452b5370a7b2fc2b611afdbe219b5343cebf935aeef6e872ef8b"),
        abi: literal("cc6c05db356a92bfec063e79359c6b8674dc88937b345ff468202617f91b82d2"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionAmpere256x128Sm80),
        bytes: 6352608,
        sm: 80,
        sha: literal("6ab9468c2d21b9b24c66fbb4380efca47a9057ddcc6adb4eb83997460986a83e"),
        abi: literal("957b6002a6730bad4d41a681d281a28e2755d1271ae8cc9b97e9ed4218675b0d"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86),
        bytes: 901664,
        sm: 86,
        sha: literal("7ec9414283b09942de90011eb5bdbaeb63b352cd0dc500f2c7ba3854f660fadf"),
        abi: literal("06d64266a3681314bc1de37deef6709e1cddbbd4e4c7e524b9d498c4c4807098"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86),
        bytes: 901664,
        sm: 86,
        sha: literal("7ec9414283b09942de90011eb5bdbaeb63b352cd0dc500f2c7ba3854f660fadf"),
        abi: literal("494d183797b1e09ce086cbb24d5d7c0177cf6c0ef2db833157ca630f75c7663b"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86),
        bytes: 901664,
        sm: 86,
        sha: literal("7ec9414283b09942de90011eb5bdbaeb63b352cd0dc500f2c7ba3854f660fadf"),
        abi: literal("85b3207019b96157e2a9055d492af79296855fbf1b8cc41ab8b7cca7d69b7d98"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionQkSm86),
        bytes: 17584,
        sm: 86,
        sha: literal("9049ad21c964126fe32a394f6e5335e9ec0901cb031559db3cab0e63d93b4664"),
        abi: literal("72c0085e1fd0b55c3f7e526aef09129d6842624cedce3cca12a2f098052920c1"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionQkSm86),
        bytes: 17584,
        sm: 86,
        sha: literal("9049ad21c964126fe32a394f6e5335e9ec0901cb031559db3cab0e63d93b4664"),
        abi: literal("ae37066796152a16d8fe7a9c05baba8f4601703e639bc8cc62fb69f0b3f1c17d"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionQkSm86),
        bytes: 17584,
        sm: 86,
        sha: literal("9049ad21c964126fe32a394f6e5335e9ec0901cb031559db3cab0e63d93b4664"),
        abi: literal("be91b95609fa619300decdc4b90338bf8c0d03a01b01e004cd1fa4b9ec1e9359"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionFaSplitSm80),
        bytes: 6317664,
        sm: 80,
        sha: literal("915016a9eaee3a391e29bbfcf00ee8de82e4da53c4865d080c04505d0d736345"),
        abi: literal("352ac51e5a5ca605a72435b685fb86344d52703060c6aba8138969fda7d72f2e"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionFaSplitSm80),
        bytes: 6317664,
        sm: 80,
        sha: literal("915016a9eaee3a391e29bbfcf00ee8de82e4da53c4865d080c04505d0d736345"),
        abi: literal("997aa113efbaeb94f0610799ec72d7df0c8d5cef94b0636bdbdbea679490b840"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionFaSplitSm80),
        bytes: 6317664,
        sm: 80,
        sha: literal("915016a9eaee3a391e29bbfcf00ee8de82e4da53c4865d080c04505d0d736345"),
        abi: literal("de28f1a77576fdd70c2d3032d5c93c450f0319d8adaea5b6042318391be8ab9a"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionMultiplySm86),
        bytes: 3397824,
        sm: 86,
        sha: literal("9277f5b0fa8d3a02004e2c8f02c1991c0bef281b8aa8aebab886797daa13502a"),
        abi: literal("42cab20f8a31630c03ee0e8112c4c16cff2bba5ad275b6c50237fb105007d4ba"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionMultiplySm86),
        bytes: 3397824,
        sm: 86,
        sha: literal("9277f5b0fa8d3a02004e2c8f02c1991c0bef281b8aa8aebab886797daa13502a"),
        abi: literal("46d1e24320b8a42180fbdc6a0cd5bbda2e65d22ae5474bd867b0dadd68b0c207"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionMultiplySm86),
        bytes: 3397824,
        sm: 86,
        sha: literal("9277f5b0fa8d3a02004e2c8f02c1991c0bef281b8aa8aebab886797daa13502a"),
        abi: literal("ef3ff56cf047b6c4cdb9f839388dca67abf3523c9b492fb66075ceab22de43ec"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::Bf16Gemv2TSm86),
        bytes: 2818208,
        sm: 86,
        sha: literal("94fc9c63d0562b88395c3d5001397aa1a3a3f572450a00db627750b817c7de56"),
        abi: literal("6887754e4c6b7ca266d06fb143d5fc1035a799d49379af2282bc6896d0063fda"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::Bf16Gemv2TSm86),
        bytes: 2818208,
        sm: 86,
        sha: literal("94fc9c63d0562b88395c3d5001397aa1a3a3f572450a00db627750b817c7de56"),
        abi: literal("e2f3cd4ce6b76cc18354618b706543a49cf9af2b2d2bee2acad7c04e9394145f"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::Bf16Gemv2TSm86),
        bytes: 2818208,
        sm: 86,
        sha: literal("94fc9c63d0562b88395c3d5001397aa1a3a3f572450a00db627750b817c7de56"),
        abi: literal("eb2842731a0c76a14a034f78d6c6dbcc26a6da02370db6a4264ba440e064faee"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::Bf16Gemv2TSm86),
        bytes: 2818208,
        sm: 86,
        sha: literal("94fc9c63d0562b88395c3d5001397aa1a3a3f572450a00db627750b817c7de56"),
        abi: literal("fbf3622a9be90850f1db1edc5370b1101062e13e35b76f398984a560a90b57c3"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
        bytes: 1428640,
        sm: 80,
        sha: literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d"),
        abi: literal("041e03e35d1cde6507ba2d9e7e46d0d1797f3363684b1665ca45945230b94ed1"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
        bytes: 1428640,
        sm: 80,
        sha: literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d"),
        abi: literal("8dbca83d0d7e227f0cbff2766488454e25b4563a3ed70154ed8f6ca3061229b8"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
        bytes: 1428640,
        sm: 80,
        sha: literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d"),
        abi: literal("a2b1593b03e9ce548e9f359a2e7b896a038a590d93b829363dc0987181bb9411"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
        bytes: 1428640,
        sm: 80,
        sha: literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d"),
        abi: literal("a5bb15e846ea770450e4eadb316065011158bdae3a13e4fc0493ff34a86f3b56"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
        bytes: 1428640,
        sm: 80,
        sha: literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d"),
        abi: literal("bc4a70439ee9f39307c59846c75decd14dd9814bc2c4177061b190a698fc6404"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
        bytes: 1428640,
        sm: 80,
        sha: literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d"),
        abi: literal("d8304984b584551a530f8c79d3b3dca359d443067039d2bd0b4ecfc725437702"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
        bytes: 1428640,
        sm: 80,
        sha: literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d"),
        abi: literal("d8d2c86a12c5b43abd8dee203c07a66383d07d8fb464012461cf43edead47d5d"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
        bytes: 1428640,
        sm: 80,
        sha: literal("b0ddb21d5b657acf1bb6cfaaff3444be7db85d98135048d3b220e5e3ce46420d"),
        abi: literal("f30c38268a85e5819fc954466ed380e7696ab97da76dded735ff15274165ceaa"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionFaCausalCombineSm80),
        bytes: 3596000,
        sm: 80,
        sha: literal("b4c2cb3cdcf9d456d5a0c905e468d9fe01a79df46544c8f447b0471ae2b4f6d1"),
        abi: literal("2545736a332e8dbf2ab3c1efbea435de12de90956965bc7d1cd6491a85936b6f"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionFaCausalCombineSm80),
        bytes: 3596000,
        sm: 80,
        sha: literal("b4c2cb3cdcf9d456d5a0c905e468d9fe01a79df46544c8f447b0471ae2b4f6d1"),
        abi: literal("52e3f25b6fba7762163bfcdc564b22da771fc73c5bfa2bc477f9741e76cbad29"),
    },
    Rule {
        kind: ImageKind::Captured(PinnedCapturedBf16Cubin::AttentionFaCausalCombineSm80),
        bytes: 3596000,
        sm: 80,
        sha: literal("b4c2cb3cdcf9d456d5a0c905e468d9fe01a79df46544c8f447b0471ae2b4f6d1"),
        abi: literal("59122529c11beb1943b5abc69eaedaa4bfe9f5f8826d7b2876189d83dac2173b"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::AmpereBf16Gemm128x64Sm80),
        bytes: 6601808,
        sm: 80,
        sha: literal("c1ca8db71001178d04f1b83d7c55883107fa122927e10d41020672678fff2c86"),
        abi: literal("2f38a29f18b1f1ab7400bb318a1184cfe37ba7ebfaae868e9b8c32798e081d70"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::AmpereBf16Gemm128x64Sm80),
        bytes: 6601808,
        sm: 80,
        sha: literal("c1ca8db71001178d04f1b83d7c55883107fa122927e10d41020672678fff2c86"),
        abi: literal("eefe175a1090107717cb19b7652ecacbc0d2de061ea85dfc44dbe057cf06e392"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::AmpereBf16Gemm64x64Stages64x5Sm80),
        bytes: 6422224,
        sm: 80,
        sha: literal("c3361a22db27ce017fec90a46f66c508cad5d668d606070b214b0351b3cf9588"),
        abi: literal("4455466a25a63b161105dc9a2ad84845e25dafbb9324c4c4fd82ddf6848f97a7"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::AmpereBf16Gemm64x64Stages64x5Sm80),
        bytes: 6422224,
        sm: 80,
        sha: literal("c3361a22db27ce017fec90a46f66c508cad5d668d606070b214b0351b3cf9588"),
        abi: literal("982a7da961ec64834b09b8a645a3231967f9cf1d4fbbbd82f150ffda447996b5"),
    },
    Rule {
        kind: ImageKind::Derived(PinnedBf16Cubin::AmpereBf16Gemm64x64Stages64x5Sm80),
        bytes: 6422224,
        sm: 80,
        sha: literal("c3361a22db27ce017fec90a46f66c508cad5d668d606070b214b0351b3cf9588"),
        abi: literal("b13f03f034908a684086040a2d62cbf111fe9d41f2a3b49af8f4459251008cd8"),
    },
];

pub(super) fn require(
    image: &VerifiedCubin,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    let kind = match (image.derived_from(), image.captured_from()) {
        (Some(pin), None) => ImageKind::Derived(pin),
        (None, Some(pin)) => ImageKind::Captured(pin),
        _ => {
            return Err(failure(
                "full-model exactly one original executable provenance category",
            ));
        }
    };
    match kind {
        ImageKind::Derived(pin) => {
            pin.require_image(image.byte_len(), image.sha256(), image.binary_sm())?
        }
        ImageKind::Captured(pin) => {
            pin.require_image(image.byte_len(), image.sha256(), image.binary_sm())?
        }
    }
    require_metadata(
        kind,
        image.byte_len(),
        image.sha256(),
        image.binary_sm(),
        manifest,
    )
}
fn require_metadata(
    kind: ImageKind,
    bytes: usize,
    sha: [u8; 32],
    sm: u8,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    let digest = super::attention_policy::digest(manifest);
    if !RULES.iter().any(|rule| {
        rule.kind == kind
            && rule.bytes == bytes
            && rule.sha == sha
            && rule.sm == sm
            && rule.abi == digest
    }) {
        return Err(failure(
            "full-model exact original image/provenance/whole mathematical ABI union",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_known_inventory_does_not_change_or_upgrade_old_set_policies() {
        use super::super::policy::CubinSetPolicy;
        let images = RULES
            .iter()
            .map(|rule| rule.sha)
            .collect::<std::collections::BTreeSet<_>>();
        let derived = RULES
            .iter()
            .filter(|rule| matches!(rule.kind, ImageKind::Derived(_)))
            .map(|rule| rule.sha)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!((images.len(), derived.len()), (14, 5));
        assert!(RULES.len() <= RECORDS);
        CubinSetPolicy::FullQwen08
            .limits(13, 21, 47, true, true)
            .unwrap();
        assert!(
            CubinSetPolicy::FullQwen08
                .limits(14, 22, 48, false, false)
                .is_err()
        );
        assert!(
            CubinSetPolicy::FullQwen08
                .limits(14, 21, 47, true, true)
                .is_err()
        );
        assert!(
            CubinSetPolicy::FullQwen08
                .limits(13, 22, 47, true, true)
                .is_err()
        );
        CubinSetPolicy::ProjectionSix
            .limits(5, 7, 18, true, true)
            .unwrap();
        assert!(
            CubinSetPolicy::ProjectionSix
                .limits(6, 8, 19, false, false)
                .is_err()
        );
        CubinSetPolicy::ConnectedQwen08
            .limits(6, 11, 20, true, true)
            .unwrap();
        assert!(
            CubinSetPolicy::ConnectedQwen08
                .limits(7, 12, 21, false, false)
                .is_err()
        );
        CubinSetPolicy::FirstAttentionCaptured
            .limits(8, 12, 19, true, true)
            .unwrap();
        assert!(
            CubinSetPolicy::FirstAttentionCaptured
                .limits(9, 13, 20, false, false)
                .is_err()
        );
        CubinSetPolicy::FirstAttentionComputedGate
            .limits(10, 14, 25, true, true)
            .unwrap();
        assert!(
            CubinSetPolicy::FirstAttentionComputedGate
                .limits(11, 15, 26, false, false)
                .is_err()
        );
    }
    #[test]
    fn same_entry_head_cannot_change_scalar_shape_pointer_layout_or_image_category() {
        let pin = PinnedBf16Cubin::Bf16Gemv2TSm86;
        let source = pin.derivation();
        let m = super::super::head_policy::Bf16VocabularyHead::manifest();
        let check = |m: &CubinKernelManifest| {
            require_metadata(
                ImageKind::Derived(pin),
                source.image_bytes,
                source.image_sha256,
                source.binary_sm,
                m,
            )
        };
        check(&m).unwrap();
        for fault in 0..8 {
            let mut bad = m.clone();
            match fault {
                0 => bad.grid[0] -= 1,
                1 => bad.parameters[0].template[68] ^= 1,
                2 => bad.parameters[2].template[0] = 1,
                3 => bad.bindings[0].bytes -= 16,
                4 => bad.fixups[0].binding = 1,
                5 => bad.fixups[0].extent -= 16,
                6 => bad.fixups[3].io = super::super::CubinIo::Read,
                _ => bad.dynamic_shared_bytes = 16,
            }
            assert!(check(&bad).is_err(), "fault{fault}");
        }
        assert!(
            require_metadata(
                ImageKind::Captured(PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
                source.image_bytes,
                source.image_sha256,
                source.binary_sm,
                &m
            )
            .is_err()
        );
        let mut sha = source.image_sha256;
        sha[0] ^= 1;
        assert!(
            require_metadata(
                ImageKind::Derived(pin),
                source.image_bytes,
                sha,
                source.binary_sm,
                &m
            )
            .is_err()
        );
    }
}
