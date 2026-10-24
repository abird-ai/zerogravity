//! Actual original CUDA rotary resources only; no DSO/carrier derivation claim.
use super::{CapturedBf16CubinRecord, PinnedCapturedBf16Cubin};
pub(super) const fn capture(kind: PinnedCapturedBf16Cubin) -> CapturedBf16CubinRecord {
    let (bytes, sha) = match kind {
        PinnedCapturedBf16Cubin::RotaryArangeSm86 => (
            638536,
            [
                162, 33, 79, 6, 235, 22, 152, 215, 155, 144, 115, 213, 170, 11, 150, 104, 125, 182,
                72, 161, 223, 203, 151, 47, 253, 138, 205, 85, 240, 70, 49, 72,
            ],
        ),
        PinnedCapturedBf16Cubin::RotaryDivideSm86 => (
            4066320,
            [
                107, 219, 107, 129, 14, 2, 64, 207, 155, 67, 238, 162, 117, 166, 42, 86, 154, 27,
                65, 58, 9, 128, 33, 100, 112, 115, 160, 35, 253, 114, 87, 239,
            ],
        ),
        PinnedCapturedBf16Cubin::RotaryFillSm86 => (
            1942400,
            [
                132, 32, 23, 248, 123, 174, 108, 223, 70, 72, 122, 83, 214, 129, 215, 102, 188,
                221, 146, 78, 193, 13, 141, 140, 118, 3, 17, 243, 83, 18, 109, 197,
            ],
        ),
        PinnedCapturedBf16Cubin::RotaryPowSm86 => (
            16041664,
            [
                58, 241, 255, 123, 101, 130, 32, 202, 68, 226, 194, 25, 38, 117, 94, 116, 87, 52,
                57, 19, 58, 154, 130, 121, 34, 154, 231, 169, 191, 106, 29, 70,
            ],
        ),
        PinnedCapturedBf16Cubin::RotaryReciprocalSm86 => (
            4246208,
            [
                231, 78, 80, 220, 186, 206, 203, 185, 81, 63, 216, 111, 181, 34, 217, 200, 190, 60,
                99, 128, 103, 56, 148, 230, 31, 24, 19, 138, 163, 80, 197, 16,
            ],
        ),
        PinnedCapturedBf16Cubin::RotaryCosSm86 => (
            850848,
            [
                116, 241, 110, 225, 193, 160, 47, 144, 35, 239, 86, 78, 102, 117, 21, 178, 142, 25,
                230, 46, 236, 52, 202, 46, 136, 171, 234, 200, 248, 85, 71, 132,
            ],
        ),
        PinnedCapturedBf16Cubin::RotarySinSm86 => (
            848032,
            [
                76, 128, 112, 7, 107, 80, 0, 18, 254, 170, 240, 34, 231, 156, 176, 143, 60, 215,
                17, 75, 41, 133, 22, 4, 82, 64, 47, 160, 64, 197, 96, 52,
            ],
        ),
        PinnedCapturedBf16Cubin::RotaryCatSm86 => (
            2103392,
            [
                22, 180, 143, 49, 87, 235, 235, 173, 249, 140, 255, 92, 29, 73, 193, 68, 129, 176,
                50, 157, 185, 52, 18, 66, 184, 154, 99, 80, 253, 116, 11, 237,
            ],
        ),
        PinnedCapturedBf16Cubin::RotaryBf16CastSm86 => (
            4557464,
            [
                245, 118, 81, 186, 175, 114, 90, 188, 97, 71, 170, 92, 59, 108, 221, 95, 54, 189,
                33, 208, 148, 209, 224, 113, 37, 206, 74, 241, 207, 163, 155, 212,
            ],
        ),
        _ => panic!("closed original rotary captured kind"),
    };
    CapturedBf16CubinRecord {
        image_bytes: bytes,
        image_sha256: sha,
        binary_sm: 86,
        capture_metadata_sha256: "e37017e70ceb5dd880442f93f547a2b9f4177f165a7c6d0358445e45b5d35509",
        capture_data_sha256: "adba0df1a13d5b1b5d4132efa0971e24c6581f392ab03c879b3b3d100c8f7389",
        reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
        reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
        reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
        reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
        reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
        preserved_input_resource_equal: false,
        immutable_carrier_membership_proven: false,
        runtime_derivation_proven: false,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nine_original_resources_remain_captured_and_refuse_changed_payloads() {
        for k in [
            PinnedCapturedBf16Cubin::RotaryArangeSm86,
            PinnedCapturedBf16Cubin::RotaryDivideSm86,
            PinnedCapturedBf16Cubin::RotaryFillSm86,
            PinnedCapturedBf16Cubin::RotaryPowSm86,
            PinnedCapturedBf16Cubin::RotaryReciprocalSm86,
            PinnedCapturedBf16Cubin::RotaryCosSm86,
            PinnedCapturedBf16Cubin::RotarySinSm86,
            PinnedCapturedBf16Cubin::RotaryCatSm86,
            PinnedCapturedBf16Cubin::RotaryBf16CastSm86,
        ] {
            let r = capture(k);
            assert!(k.require_image(r.image_bytes, r.image_sha256, 86).is_ok());
            assert!(
                !r.preserved_input_resource_equal
                    && !r.immutable_carrier_membership_proven
                    && !r.runtime_derivation_proven
            );
            assert!(
                k.require_image(r.image_bytes + 1, r.image_sha256, 86)
                    .is_err()
            );
            let mut bad = r.image_sha256;
            bad[0] ^= 1;
            assert!(k.require_image(r.image_bytes, bad, 86).is_err());
            assert!(k.require_image(r.image_bytes, r.image_sha256, 80).is_err());
        }
    }
}
