//! Exact selected canonical47 BA image; no preceding-input/carrier/derivation claim.
use super::CapturedBf16CubinRecord;
pub(super) const CAPTURE: CapturedBf16CubinRecord = CapturedBf16CubinRecord {
    image_bytes: 7_789_728,
    image_sha256: [
        0x66, 0xc2, 0xe6, 0x1f, 0x0c, 0xc1, 0xd1, 0xbf, 0x90, 0xee, 0xb8, 0xea, 0x32, 0x45, 0xe2,
        0x84, 0x9b, 0x0c, 0x49, 0x46, 0x6c, 0x48, 0x6c, 0x8d, 0x1d, 0xaa, 0x21, 0x91, 0x66, 0xaf,
        0xdc, 0x30,
    ],
    binary_sm: 80,
    capture_metadata_sha256: "2bd4f2741090d0cde2b2de3e24e8d276b00323f70138e03724d3d2815b55119b",
    capture_data_sha256: "9a21b1ab20cbb1b1275b25ba67592fb1fa6514d731be62366c3b7dc73b197ca6",
    reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
    reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
    reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
    reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
    preserved_input_resource_equal: false,
    immutable_carrier_membership_proven: false,
    runtime_derivation_proven: false,
};
#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_command::bf16_gemm::PinnedCapturedBf16Cubin;
    #[test]
    fn canonical47_selected_resource_has_exact_identity_and_truthful_provenance() {
        let kind = PinnedCapturedBf16Cubin::Canonical47CutlassBf16Sm80;
        let r = kind.capture();
        assert_eq!(r, CAPTURE);
        assert!(!r.preserved_input_resource_equal);
        assert!(!r.immutable_carrier_membership_proven);
        assert!(!r.runtime_derivation_proven);
        assert!(
            kind.require_image(r.image_bytes, r.image_sha256, r.binary_sm)
                .is_ok()
        );
        for bit in 0..256 {
            let mut digest = r.image_sha256;
            digest[bit / 8] ^= 1 << (bit % 8);
            assert!(kind.require_image(r.image_bytes, digest, 80).is_err());
        }
        for bytes in [0, r.image_bytes - 1, r.image_bytes + 1, usize::MAX] {
            assert!(kind.require_image(bytes, r.image_sha256, 80).is_err());
        }
        for sm in [0, 79, 86, 90] {
            assert!(
                kind.require_image(r.image_bytes, r.image_sha256, sm)
                    .is_err()
            );
        }
        let old = PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80.capture();
        assert_ne!(old.image_sha256, r.image_sha256);
    }
}
