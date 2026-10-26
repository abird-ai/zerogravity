//! Closed original captures; provenance remains distinct from derivation.
use super::*;
use crate::owned_command::cubin::VerifiedCubin;
use std::any::Any;
mod fixed_prompt;
mod rotary;
mod text;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinnedCapturedBf16Cubin {
    TextFaCausalCombineSm80,
    TextFaSplitCombineSm80,
    TextCutlassBf16Sm80,
    Cohort2BAmpere64SlicedSm80,
    FixedPromptCutlassBf16Sm80,
    RotaryArangeSm86,
    RotaryDivideSm86,
    RotaryFillSm86,
    RotaryPowSm86,
    RotaryReciprocalSm86,
    RotaryCosSm86,
    RotarySinSm86,
    RotaryCatSm86,
    RotaryBf16CastSm86,
    CutlassWmmaBf16Sm80,
    SiluAndMulBf16Sm86,
    AttentionAmpere256x128Sm80,
    AttentionQkSm86,
    AttentionKvSm86,
    AttentionFaSplitSm80,
    AttentionFaCausalCombineSm80,
    AttentionSigmoidSm86,
    AttentionMultiplySm86,
}

/// Reference-capture metadata, not caller authority or an immutable DSO claim.
/// Public copies cannot change the constructor's closed record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapturedBf16CubinRecord {
    pub image_bytes: usize,
    pub image_sha256: [u8; 32],
    pub binary_sm: u8,
    pub capture_metadata_sha256: &'static str,
    pub capture_data_sha256: &'static str,
    pub reference_hf_source_sha256: &'static str,
    pub reference_runtime_image: &'static str,
    pub reference_runtime_manifest: &'static str,
    pub reference_vllm_commit: &'static str,
    pub reference_driver_sha256: &'static str,
    pub preserved_input_resource_equal: bool,
    pub immutable_carrier_membership_proven: bool,
    pub runtime_derivation_proven: bool,
}

impl PinnedCapturedBf16Cubin {
    pub const fn capture(self) -> CapturedBf16CubinRecord {
        match self {
            Self::TextFaCausalCombineSm80 => text::SHORTHFFACAUSALCOMBINESM80,
            Self::TextFaSplitCombineSm80 => text::SHORTHFFASPLITCOMBINESM80,
            Self::TextCutlassBf16Sm80 => text::SHORTHFCUTLASSBF16SM80,
            Self::Cohort2BAmpere64SlicedSm80 => text::COHORT2BAMPERE64SLICEDSM80,
            Self::FixedPromptCutlassBf16Sm80 => fixed_prompt::CAPTURE,
            Self::RotaryArangeSm86
            | Self::RotaryDivideSm86
            | Self::RotaryFillSm86
            | Self::RotaryPowSm86
            | Self::RotaryReciprocalSm86
            | Self::RotaryCosSm86
            | Self::RotarySinSm86
            | Self::RotaryCatSm86
            | Self::RotaryBf16CastSm86 => rotary::capture(self),
            Self::CutlassWmmaBf16Sm80 => CapturedBf16CubinRecord {
                image_bytes: 1_428_640,
                image_sha256: [
                    0xb0, 0xdd, 0xb2, 0x1d, 0x5b, 0x65, 0x7a, 0xcf, 0x1b, 0xb6, 0xcf, 0xaa, 0xff,
                    0x34, 0x44, 0xbe, 0x7d, 0xb8, 0x5d, 0x98, 0x13, 0x50, 0x48, 0xd3, 0xb2, 0x20,
                    0xe5, 0xe3, 0xce, 0x46, 0x42, 0x0d,
                ],
                binary_sm: 80,
                // Actual startup5; resource indices are deliberately unbound.
                capture_metadata_sha256: "74f397d1fa3c87ad00054e92aa282f800482f196fcdb7d719ba0eee12dfec9f0",
                capture_data_sha256: "7e12692559372b4febd9c8a72d9a0910aa0f4ac25d9449c27023b2df7a38c48c",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                preserved_input_resource_equal: true,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
            Self::SiluAndMulBf16Sm86 => CapturedBf16CubinRecord {
                image_bytes: 901_664,
                image_sha256: [
                    0x7e, 0xc9, 0x41, 0x42, 0x83, 0xb0, 0x99, 0x42, 0xde, 0x90, 0x01, 0x1e, 0xb5,
                    0xbd, 0xba, 0xeb, 0x63, 0xb3, 0x52, 0xcd, 0x0d, 0xc5, 0x00, 0xf2, 0xc7, 0xba,
                    0x38, 0x54, 0xf6, 0x60, 0xfa, 0xdf,
                ],
                binary_sm: 86,
                capture_metadata_sha256: "6ab755d663c4aeb70f53705fd0100fb97d9dc8f8e4c663978865e0423be5b622",
                capture_data_sha256: "e185bc7b6d18fd1d77ebfdb983eb9b8ab9a474880e14986b617b1bd36d635a7d",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                // Actual selected resource; preceding loader input unobserved.
                preserved_input_resource_equal: false,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
            Self::AttentionAmpere256x128Sm80 => CapturedBf16CubinRecord {
                image_bytes: 6352608,
                image_sha256: [
                    0x6a, 0xb9, 0x46, 0x8c, 0x2d, 0x21, 0xb9, 0xb2, 0x4c, 0x66, 0xfb, 0xb4, 0x38,
                    0x0e, 0xfc, 0xa4, 0x7a, 0x90, 0x57, 0xdd, 0xcc, 0x6a, 0xdb, 0x4e, 0xb8, 0x39,
                    0x97, 0x46, 0x09, 0x86, 0xa8, 0x3e,
                ],
                binary_sm: 80,
                capture_metadata_sha256: "9ce23fcd1e28ac81c32ffeca77f3439bae7d55ebcbb3ecb5b3712a3aad8c963f",
                capture_data_sha256: "f2958a8ddef7462e1eb2130712b681ff24223ac8d3fadd279c9bf85cf15103cb",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                // Actual selected resource; preceding loader input unobserved.
                preserved_input_resource_equal: false,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
            Self::AttentionSigmoidSm86 => CapturedBf16CubinRecord {
                image_bytes: 2679648,
                image_sha256: [
                    0x08, 0xc0, 0x0e, 0xe6, 0x7f, 0xa7, 0x45, 0x2b, 0x53, 0x70, 0xa7, 0xb2, 0xfc,
                    0x2b, 0x61, 0x1a, 0xfd, 0xbe, 0x21, 0x9b, 0x53, 0x43, 0xce, 0xbf, 0x93, 0x5a,
                    0xee, 0xf6, 0xe8, 0x72, 0xef, 0x8b,
                ],
                binary_sm: 86,
                capture_metadata_sha256: "445accd4b8e8982428509143816789013fb16fcfc1fd709a0af29e125af0c8fc",
                capture_data_sha256: "9f3785d46a4c2fcdc4b049c46b5726746db17a99acea5f3b4bfddeefc133b473",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                // Actual selected resource; preceding loader input unobserved.
                preserved_input_resource_equal: false,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
            Self::AttentionMultiplySm86 => CapturedBf16CubinRecord {
                image_bytes: 3397824,
                image_sha256: [
                    0x92, 0x77, 0xf5, 0xb0, 0xfa, 0x8d, 0x3a, 0x02, 0x00, 0x4e, 0x2c, 0x8f, 0x02,
                    0xc1, 0x99, 0x1c, 0x0b, 0xef, 0x28, 0x1b, 0x8a, 0xa8, 0xae, 0xba, 0xb8, 0x86,
                    0x79, 0x7d, 0xaa, 0x13, 0x50, 0x2a,
                ],
                binary_sm: 86,
                capture_metadata_sha256: "445accd4b8e8982428509143816789013fb16fcfc1fd709a0af29e125af0c8fc",
                capture_data_sha256: "9f3785d46a4c2fcdc4b049c46b5726746db17a99acea5f3b4bfddeefc133b473",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                // Actual selected resource; preceding loader input unobserved.
                preserved_input_resource_equal: false,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
            Self::AttentionQkSm86 => CapturedBf16CubinRecord {
                image_bytes: 17584,
                image_sha256: [
                    0x90, 0x49, 0xad, 0x21, 0xc9, 0x64, 0x12, 0x6f, 0xe3, 0x2a, 0x39, 0x4f, 0x6e,
                    0x53, 0x35, 0xe9, 0xec, 0x09, 0x01, 0xcb, 0x03, 0x15, 0x59, 0xdb, 0x3c, 0xab,
                    0x0e, 0x63, 0xd9, 0x3b, 0x46, 0x64,
                ],
                binary_sm: 86,
                capture_metadata_sha256: "9ce23fcd1e28ac81c32ffeca77f3439bae7d55ebcbb3ecb5b3712a3aad8c963f",
                capture_data_sha256: "f2958a8ddef7462e1eb2130712b681ff24223ac8d3fadd279c9bf85cf15103cb",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                // Actual selected resource; preceding loader input unobserved.
                preserved_input_resource_equal: false,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
            Self::AttentionKvSm86 => CapturedBf16CubinRecord {
                image_bytes: 1813600,
                image_sha256: [
                    0x07, 0x81, 0xfb, 0xbc, 0xae, 0x3a, 0x3f, 0x26, 0xcf, 0x97, 0x20, 0x5a, 0xf8,
                    0x06, 0xe6, 0xe9, 0x00, 0x14, 0x13, 0x82, 0x7e, 0x66, 0xe2, 0x98, 0xd4, 0xa0,
                    0x74, 0x3c, 0xde, 0x06, 0x70, 0x35,
                ],
                binary_sm: 86,
                capture_metadata_sha256: "9ce23fcd1e28ac81c32ffeca77f3439bae7d55ebcbb3ecb5b3712a3aad8c963f",
                capture_data_sha256: "f2958a8ddef7462e1eb2130712b681ff24223ac8d3fadd279c9bf85cf15103cb",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                // Actual selected resource; preceding loader input unobserved.
                preserved_input_resource_equal: false,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
            Self::AttentionFaSplitSm80 => CapturedBf16CubinRecord {
                image_bytes: 6317664,
                image_sha256: [
                    0x91, 0x50, 0x16, 0xa9, 0xea, 0xee, 0x3a, 0x39, 0x1e, 0x29, 0xbb, 0xfc, 0xf0,
                    0x0e, 0xe8, 0xde, 0x82, 0xe4, 0xda, 0x53, 0xc4, 0x86, 0x5d, 0x08, 0x0c, 0x04,
                    0x50, 0x5d, 0x0d, 0x73, 0x63, 0x45,
                ],
                binary_sm: 80,
                capture_metadata_sha256: "9ce23fcd1e28ac81c32ffeca77f3439bae7d55ebcbb3ecb5b3712a3aad8c963f",
                capture_data_sha256: "f2958a8ddef7462e1eb2130712b681ff24223ac8d3fadd279c9bf85cf15103cb",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                // Actual selected resource; preceding loader input unobserved.
                preserved_input_resource_equal: false,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
            Self::AttentionFaCausalCombineSm80 => CapturedBf16CubinRecord {
                image_bytes: 3596000,
                image_sha256: [
                    0xb4, 0xc2, 0xcb, 0x3c, 0xdc, 0xf9, 0xd4, 0x56, 0xd5, 0xa0, 0xc9, 0x05, 0xe4,
                    0x68, 0xd9, 0xfe, 0x01, 0xa7, 0x9d, 0xf4, 0x65, 0x44, 0xc8, 0xf4, 0x47, 0xb0,
                    0x47, 0x1a, 0xe2, 0xb4, 0xf6, 0xd1,
                ],
                binary_sm: 80,
                capture_metadata_sha256: "9ce23fcd1e28ac81c32ffeca77f3439bae7d55ebcbb3ecb5b3712a3aad8c963f",
                capture_data_sha256: "f2958a8ddef7462e1eb2130712b681ff24223ac8d3fadd279c9bf85cf15103cb",
                reference_hf_source_sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                reference_runtime_image: "f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
                reference_runtime_manifest: "eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
                reference_vllm_commit: "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                reference_driver_sha256: "dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f",
                // Actual selected resource; preceding loader input unobserved.
                preserved_input_resource_equal: false,
                immutable_carrier_membership_proven: false,
                runtime_derivation_proven: false,
            },
        }
    }

    pub(crate) fn require_image(
        self,
        bytes: usize,
        digest: [u8; 32],
        sm: u8,
    ) -> Result<(), ServerError> {
        let expected = self.capture();
        if bytes != expected.image_bytes
            || digest != expected.image_sha256
            || sm != expected.binary_sm
        {
            return Err(failure(
                "captured cubin differs from closed original capture",
            ));
        }
        Ok(())
    }
}

impl Bf16BlasSources {
    /// Exact authenticated captured image only; no expected-hash parameter,
    /// fake provider interval, DSO membership or runtime derivation claim.
    ///
    /// # Safety
    /// Caller admitted the Vec's FULL capacity before allocation. This owner
    /// retains original host-image and SAME service's native-module admission
    /// until real unload/quarantine. Original HF source ownership remains in
    /// that composite owner; no queue/arbiter/admission substitution is allowed.
    pub unsafe fn captured_cubin(
        self: &Arc<Self>,
        admission_owner: Arc<dyn Any + Send + Sync>,
        kind: PinnedCapturedBf16Cubin,
        image: Vec<u8>,
    ) -> Result<Arc<VerifiedCubin>, ServerError> {
        self.require_queue(&self.queue)?;
        let sealed = VerifiedCubin::seal_captured(self.clone(), admission_owner, kind, image)?;
        self.require_queue(&self.queue)?;
        Ok(sealed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_resource_does_not_inherit_cutlass_loader_input_equality() {
        let kind = PinnedCapturedBf16Cubin::SiluAndMulBf16Sm86;
        let record = kind.capture();
        assert_eq!((record.image_bytes, record.binary_sm), (901_664, 86));
        assert!(!record.preserved_input_resource_equal);
        assert!(!record.immutable_carrier_membership_proven);
        assert!(!record.runtime_derivation_proven);
        assert_ne!(
            record.image_sha256,
            PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80
                .capture()
                .image_sha256
        );
        assert!(
            kind.require_image(record.image_bytes, record.image_sha256, 86)
                .is_ok()
        );
        for bit in 0..256 {
            let mut digest = record.image_sha256;
            digest[bit / 8] ^= 1 << (bit % 8);
            assert!(kind.require_image(record.image_bytes, digest, 86).is_err());
        }
        for bytes in [
            0,
            record.image_bytes - 1,
            record.image_bytes + 1,
            usize::MAX,
        ] {
            assert!(kind.require_image(bytes, record.image_sha256, 86).is_err());
        }
        for sm in [0, 80, 90] {
            assert!(
                kind.require_image(record.image_bytes, record.image_sha256, sm)
                    .is_err()
            );
        }
    }

    #[test]
    fn captured_record_has_no_derived_source_claim_or_derived_hash_alias() {
        let kind = PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80;
        let record = kind.capture();
        assert!(record.preserved_input_resource_equal);
        assert!(!record.immutable_carrier_membership_proven);
        assert!(!record.runtime_derivation_proven);
        assert_eq!(record.image_bytes, 1_428_640);
        assert_eq!(record.binary_sm, 80);
        for derived in [
            PinnedBf16Cubin::AmpereBf16Gemm128x64Sm80,
            PinnedBf16Cubin::AmpereBf16Gemm64x64Stages64x5Sm80,
            PinnedBf16Cubin::Bf16SplitKReduceSm86,
            PinnedBf16Cubin::Bf16Gemv2TSm86,
            PinnedBf16Cubin::Bf16GemvxSm86,
        ] {
            assert_ne!(record.image_sha256, derived.derivation().image_sha256);
        }
    }

    #[test]
    fn captured_registry_refuses_every_digest_bit_extent_and_wrong_sm() {
        let kind = PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80;
        let record = kind.capture();
        assert!(
            kind.require_image(record.image_bytes, record.image_sha256, 80)
                .is_ok()
        );
        for bit in 0..256 {
            let mut digest = record.image_sha256;
            digest[bit / 8] ^= 1 << (bit % 8);
            assert!(kind.require_image(record.image_bytes, digest, 80).is_err());
        }
        for bytes in [
            0,
            record.image_bytes - 1,
            record.image_bytes + 1,
            usize::MAX,
        ] {
            assert!(kind.require_image(bytes, record.image_sha256, 80).is_err());
        }
        for sm in [0, 79, 86, 90] {
            assert!(
                kind.require_image(record.image_bytes, record.image_sha256, sm)
                    .is_err()
            );
        }
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    #[test]
    fn attention_resources_are_closed_and_never_claim_derived_or_preserved_input() {
        let kinds = [
            PinnedCapturedBf16Cubin::AttentionAmpere256x128Sm80,
            PinnedCapturedBf16Cubin::AttentionQkSm86,
            PinnedCapturedBf16Cubin::AttentionKvSm86,
            PinnedCapturedBf16Cubin::AttentionFaSplitSm80,
            PinnedCapturedBf16Cubin::AttentionFaCausalCombineSm80,
        ];
        for kind in kinds {
            let r = kind.capture();
            assert!(!r.preserved_input_resource_equal);
            assert!(!r.immutable_carrier_membership_proven);
            assert!(!r.runtime_derivation_proven);
            assert_eq!(
                r.capture_metadata_sha256,
                "9ce23fcd1e28ac81c32ffeca77f3439bae7d55ebcbb3ecb5b3712a3aad8c963f"
            );
            assert!(
                kind.require_image(r.image_bytes, r.image_sha256, r.binary_sm)
                    .is_ok()
            );
            for bit in 0..256 {
                let mut bad = r.image_sha256;
                bad[bit / 8] ^= 1 << (bit % 8);
                assert!(kind.require_image(r.image_bytes, bad, r.binary_sm).is_err());
            }
            for bytes in [0, r.image_bytes - 1, r.image_bytes + 1, usize::MAX] {
                assert!(
                    kind.require_image(bytes, r.image_sha256, r.binary_sm)
                        .is_err()
                );
            }
            assert!(
                kind.require_image(
                    r.image_bytes,
                    r.image_sha256,
                    if r.binary_sm == 80 { 86 } else { 80 }
                )
                .is_err()
            );
        }
    }
}

#[cfg(test)]
mod gate_capture_tests {
    use super::*;
    #[test]
    fn gate_resources_are_distinct_closed_captures_without_carrier_or_derivation_claim() {
        for kind in [
            PinnedCapturedBf16Cubin::AttentionSigmoidSm86,
            PinnedCapturedBf16Cubin::AttentionMultiplySm86,
        ] {
            let r = kind.capture();
            assert_eq!(r.binary_sm, 86);
            assert!(!r.preserved_input_resource_equal);
            assert!(!r.immutable_carrier_membership_proven);
            assert!(!r.runtime_derivation_proven);
            assert_eq!(
                r.capture_metadata_sha256,
                "445accd4b8e8982428509143816789013fb16fcfc1fd709a0af29e125af0c8fc"
            );
            assert!(
                kind.require_image(r.image_bytes, r.image_sha256, 86)
                    .is_ok()
            );
            for bit in 0..256 {
                let mut digest = r.image_sha256;
                digest[bit / 8] ^= 1 << (bit % 8);
                assert!(kind.require_image(r.image_bytes, digest, 86).is_err());
            }
            for bytes in [0, r.image_bytes - 1, r.image_bytes + 1, usize::MAX] {
                assert!(kind.require_image(bytes, r.image_sha256, 86).is_err());
            }
            assert!(
                kind.require_image(r.image_bytes, r.image_sha256, 80)
                    .is_err()
            );
        }
        let sigmoid = PinnedCapturedBf16Cubin::AttentionSigmoidSm86.capture();
        let multiply = PinnedCapturedBf16Cubin::AttentionMultiplySm86.capture();
        assert_ne!(sigmoid.image_sha256, multiply.image_sha256);
        assert!(
            PinnedCapturedBf16Cubin::AttentionSigmoidSm86
                .require_image(multiply.image_bytes, multiply.image_sha256, 86)
                .is_err()
        );
    }
}
