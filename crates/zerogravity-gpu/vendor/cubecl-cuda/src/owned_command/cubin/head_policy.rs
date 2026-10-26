//! Exact original vocabulary-head ABI and source image, no model authority.
//! root-final-head-driver-arguments-1 data42,760B SHA922d9b10…4bac8.
use super::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup, PinnedBf16Cubin, VerifiedCubin, failure,
};

const WEIGHT_BYTES: u64 = 508_559_360;
const INPUT_BYTES: u64 = 2048;
const OUTPUT_BYTES: u64 = 496_640;
pub(crate) mod resident;
const IMAGE: PinnedBf16Cubin = PinnedBf16Cubin::Bf16Gemv2TSm86;
const ENTRY: &str = "_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_";

const fn literal<const N: usize>(hex: &str) -> [u8; N] {
    const fn nibble(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => panic!("invalid pinned literal"),
        }
    }
    assert!(hex.len() == N * 2);
    let source = hex.as_bytes();
    let mut result = [0; N];
    let mut index = 0;
    while index < N {
        result[index] = nibble(source[2 * index]) << 4 | nibble(source[2 * index + 1]);
        index += 1;
    }
    result
}

// ALL5 actual160B records become exactly this after replacing only pointer
// words0/16/32/48; canonical SHAdf131f854eb4cd2d752d1192e120babf1059a2b1d62eaa517351b63fb1eb4f3b.
const TEMPLATE: [u8; 160] = literal(
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004000000ca03000004000001000000010000000100000001000000000000000000000000000000000000000000000000000000000000000000803f000000000000000000000000000000000000000000000000000000000000803f00000000",
);

/// No shape/rows/template/image/launch knobs. Five original phase-selected
/// inputs all use the SAME single-row projection and complete vocabulary.
fn manifest() -> CubinKernelManifest {
    CubinKernelManifest {
        entry: ENTRY.into(),
        binary_sm: 86,
        grid: [31_040, 1, 1],
        block: [128, 1, 1],
        dynamic_shared_bytes: 0,
        attributes: CubinLaunchAttributes::None,
        bindings: vec![
            CubinBinding {
                bytes: WEIGHT_BYTES,
                alignment: 16,
                io: CubinIo::Read,
            },
            CubinBinding {
                bytes: INPUT_BYTES,
                alignment: 16,
                io: CubinIo::Read,
            },
            CubinBinding {
                bytes: OUTPUT_BYTES,
                alignment: 16,
                io: CubinIo::Write,
            },
        ],
        parameters: vec![
            CubinParameter {
                offset: 0,
                template: TEMPLATE[..152].to_vec(),
            },
            CubinParameter {
                offset: 152,
                template: TEMPLATE[152..156].to_vec(),
            },
            CubinParameter {
                offset: 156,
                template: TEMPLATE[156..].to_vec(),
            },
        ],
        fixups: [
            (0, 0, WEIGHT_BYTES, CubinIo::Read),
            (16, 1, INPUT_BYTES, CubinIo::Read),
            (32, 2, OUTPUT_BYTES, CubinIo::Write),
            (48, 2, OUTPUT_BYTES, CubinIo::Write),
        ]
        .into_iter()
        .map(|(byte_offset, binding, extent, io)| CubinPointerFixup {
            parameter: 0,
            byte_offset,
            binding,
            binding_offset: 0,
            extent,
            alignment: 16,
            io,
        })
        .collect(),
    }
}

/// Closed data-only row selection, never a teacher phase or model permit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyHeadSelection {
    WholeSelected,
    LastOfTwo,
    LastOfSixtyFour,
    // Provenance: LastOf47.
    FixedPromptLast,
    LastOf57,
    LastOf62,
    LastOf67,
    LastOf72,
}
pub(crate) fn selection_allowed(
    policy: super::policy::CubinSetPolicy,
    selection: VocabularyHeadSelection,
) -> bool {
    use super::policy::CubinSetPolicy as P;
    if matches!(policy, P::Text08 | P::Text08Cohort4) {
        return matches!(
            selection,
            VocabularyHeadSelection::WholeSelected
                | VocabularyHeadSelection::FixedPromptLast
                | VocabularyHeadSelection::LastOf57
                | VocabularyHeadSelection::LastOf62
                | VocabularyHeadSelection::LastOf67
                | VocabularyHeadSelection::LastOf72
        );
    }
    match selection {
        VocabularyHeadSelection::LastOf57
        | VocabularyHeadSelection::LastOf62
        | VocabularyHeadSelection::LastOf67
        | VocabularyHeadSelection::LastOf72 => false,
        VocabularyHeadSelection::FixedPromptLast => {
            matches!(policy, P::FixedPromptFull | P::FixedPromptNatural)
        }
        VocabularyHeadSelection::WholeSelected if policy == P::FixedPromptNatural => true,
        _ => matches!(policy, P::VocabularyHead | P::FullQwen08),
    }
}
impl VocabularyHeadSelection {
    pub(crate) fn full_bytes(self, index: usize) -> Result<u64, cubecl_core::server::ServerError> {
        match index {
            0 => Ok(WEIGHT_BYTES),
            1 => Ok(match self {
                Self::WholeSelected => 2048,
                Self::LastOfTwo => 4096,
                Self::LastOfSixtyFour => 131_072,
                Self::FixedPromptLast => 96_256,
                Self::LastOf57 => 57 * INPUT_BYTES,
                Self::LastOf62 => 62 * INPUT_BYTES,
                Self::LastOf67 => 67 * INPUT_BYTES,
                Self::LastOf72 => 72 * INPUT_BYTES,
            }),
            2 => Ok(OUTPUT_BYTES),
            _ => Err(failure("head exact three full resources")),
        }
    }
    pub(crate) fn checked_addresses(
        self,
        resources: &[(u64, u64)],
    ) -> Result<[u64; 3], cubecl_core::server::ServerError> {
        if resources.len() != 3 {
            return Err(failure("head exact three native resource ranges"));
        }
        let mut ends = [0; 3];
        for (index, &(pointer, bytes)) in resources.iter().enumerate() {
            if bytes != self.full_bytes(index)? || pointer == 0 || pointer % 16 != 0 {
                return Err(failure("head full native resource size/alignment"));
            }
            ends[index] = pointer
                .checked_add(bytes)
                .ok_or_else(|| failure("head native resource end overflow"))?;
        }
        for a in 0..3 {
            for b in a + 1..3 {
                if resources[a].0 < ends[b] && resources[b].0 < ends[a] {
                    return Err(failure("head original full native resources alias"));
                }
            }
        }
        let offset = self
            .full_bytes(1)?
            .checked_sub(INPUT_BYTES)
            .ok_or_else(|| failure("head selected range underflow"))?;
        let selected = resources[1]
            .0
            .checked_add(offset)
            .ok_or_else(|| failure("head selected native start overflow"))?;
        if selected.checked_add(INPUT_BYTES) != Some(ends[1]) {
            return Err(failure("head last-row native extent differs"));
        }
        Ok([resources[0].0, selected, resources[2].0])
    }
}

/// Closed metadata only; exact source image and actual leases remain mandatory.
pub struct Bf16VocabularyHead;
impl Bf16VocabularyHead {
    pub fn manifest() -> CubinKernelManifest {
        manifest()
    }
}

pub(crate) fn require(
    image: &VerifiedCubin,
    m: &CubinKernelManifest,
) -> Result<(), cubecl_core::server::ServerError> {
    require_metadata(
        image.derived_from(),
        image.captured_from(),
        image.byte_len(),
        image.sha256(),
        image.binary_sm(),
        m,
    )
}
fn require_metadata(
    derived: Option<PinnedBf16Cubin>,
    captured: Option<super::PinnedCapturedBf16Cubin>,
    bytes: usize,
    sha: [u8; 32],
    binary_sm: u8,
    m: &CubinKernelManifest,
) -> Result<(), cubecl_core::server::ServerError> {
    let pin = IMAGE.derivation();
    if derived != Some(IMAGE)
        || captured.is_some()
        || bytes != pin.image_bytes
        || sha != pin.image_sha256
        || binary_sm != pin.binary_sm
        || *m != manifest()
    {
        return Err(failure(
            "closed original vocabulary head image and entire160B ABI",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn last47_view_and_policy_cannot_alias_legacy_shapes_or_prefixes() {
        use super::super::policy::CubinSetPolicy as P;
        let valid = [
            (0x1000_0000, WEIGHT_BYTES),
            (0x4000_0000, 96_256),
            (0x5000_0000, OUTPUT_BYTES),
        ];
        let addresses = VocabularyHeadSelection::FixedPromptLast
            .checked_addresses(&valid)
            .unwrap();
        assert_eq!(addresses[1], 0x4000_0000 + 94_208);
        for wrong_bytes in [2048, 4096, 94_208, 98_304, 131_072] {
            let mut wrong = valid;
            wrong[1].1 = wrong_bytes;
            assert!(
                VocabularyHeadSelection::FixedPromptLast
                    .checked_addresses(&wrong)
                    .is_err()
            );
        }
        assert!(selection_allowed(
            P::FixedPromptFull,
            VocabularyHeadSelection::FixedPromptLast
        ));
        for old in [
            P::VocabularyHead,
            P::FullQwen08,
            P::FixedPromptLayer3,
            P::FixedPromptFa,
        ] {
            assert!(!selection_allowed(old, VocabularyHeadSelection::FixedPromptLast));
        }
        for old in [
            VocabularyHeadSelection::WholeSelected,
            VocabularyHeadSelection::LastOfTwo,
            VocabularyHeadSelection::LastOfSixtyFour,
        ] {
            assert!(!selection_allowed(P::FixedPromptFull, old));
            assert!(selection_allowed(P::VocabularyHead, old));
            assert!(selection_allowed(P::FullQwen08, old));
        }
    }
    #[test]
    fn selected_native_view_requires_original_full_ranges_and_nonalias() {
        for (selection, bytes) in [
            (VocabularyHeadSelection::WholeSelected, 2048),
            (VocabularyHeadSelection::LastOfTwo, 4096),
            (VocabularyHeadSelection::LastOfSixtyFour, 131_072),
            (VocabularyHeadSelection::FixedPromptLast, 96_256),
        ] {
            let ranges = [
                (0x1000_0000, WEIGHT_BYTES),
                (0x4000_0000, bytes),
                (0x5000_0000, OUTPUT_BYTES),
            ];
            let pointers = selection.checked_addresses(&ranges).unwrap();
            assert_eq!(
                pointers,
                [ranges[0].0, ranges[1].0 + bytes - 2048, ranges[2].0]
            );
            let mut wrong = ranges;
            wrong[1].1 = 2048;
            if bytes != 2048 {
                assert!(selection.checked_addresses(&wrong).is_err());
            }
            let mut alias = ranges;
            alias[1].0 = ranges[0].0 + 16;
            assert!(selection.checked_addresses(&alias).is_err());
            let mut overflow = ranges;
            overflow[2].0 = u64::MAX - 15;
            assert!(selection.checked_addresses(&overflow).is_err());
            let mut unaligned = ranges;
            unaligned[1].0 += 1;
            assert!(selection.checked_addresses(&unaligned).is_err());
            let mut output_alias = ranges;
            output_alias[2].0 = ranges[1].0;
            assert!(selection.checked_addresses(&output_alias).is_err());
        }
        assert!(
            VocabularyHeadSelection::WholeSelected
                .checked_addresses(&[])
                .is_err()
        );
    }
    #[test]
    fn head_policy_has_one_record_capacity_without_changing_old_policy() {
        use super::super::policy::CubinSetPolicy;
        CubinSetPolicy::VocabularyHead
            .limits(0, 0, 0, true, true)
            .unwrap();
        assert!(
            CubinSetPolicy::VocabularyHead
                .limits(1, 1, 1, false, false)
                .is_err()
        );
        assert!(
            CubinSetPolicy::VocabularyHead
                .limits(1, 1, 0, true, false)
                .is_err()
        );
        assert!(
            CubinSetPolicy::VocabularyHead
                .limits(1, 1, 0, false, true)
                .is_err()
        );
        CubinSetPolicy::ProjectionSix
            .limits(5, 7, 18, true, true)
            .unwrap();
        CubinSetPolicy::ConnectedQwen08
            .limits(6, 11, 20, true, true)
            .unwrap();
        CubinSetPolicy::FirstAttentionCaptured
            .limits(8, 12, 19, true, true)
            .unwrap();
    }
    #[test]
    fn refuses_wrong_provenance_or_any_altered_head_field() {
        let pin = IMAGE.derivation();
        let check = |m: &CubinKernelManifest| {
            require_metadata(
                Some(IMAGE),
                None,
                pin.image_bytes,
                pin.image_sha256,
                pin.binary_sm,
                m,
            )
        };
        check(&manifest()).unwrap();
        assert!(
            require_metadata(
                None,
                None,
                pin.image_bytes,
                pin.image_sha256,
                pin.binary_sm,
                &manifest()
            )
            .is_err()
        );
        assert!(
            require_metadata(
                Some(IMAGE),
                Some(super::super::PinnedCapturedBf16Cubin::CutlassWmmaBf16Sm80),
                pin.image_bytes,
                pin.image_sha256,
                pin.binary_sm,
                &manifest()
            )
            .is_err()
        );
        let mut hash = pin.image_sha256;
        hash[0] ^= 1;
        assert!(
            require_metadata(
                Some(IMAGE),
                None,
                pin.image_bytes,
                hash,
                pin.binary_sm,
                &manifest()
            )
            .is_err()
        );
        for mutation in 0..8 {
            let mut m = manifest();
            match mutation {
                0 => m.grid[0] -= 1,
                1 => m.bindings[0].bytes -= 16,
                2 => m.parameters[0].template[68] ^= 1,
                3 => m.parameters[2].template[0] = 1,
                4 => m.fixups[3].io = CubinIo::Read,
                5 => m.fixups[0].extent -= 16,
                6 => m.dynamic_shared_bytes = 16,
                7 => m.attributes = CubinLaunchAttributes::Unsupported,
                _ => unreachable!(),
            }
            assert!(check(&m).is_err());
        }
    }

    #[test]
    fn text_resident_views_keep_exact_complete_owners_and_policy() {
        use super::super::policy::CubinSetPolicy as P;
        for (s, rows) in [
            (VocabularyHeadSelection::WholeSelected, 1),
            (VocabularyHeadSelection::FixedPromptLast, 47),
            (VocabularyHeadSelection::LastOf57, 57),
            (VocabularyHeadSelection::LastOf62, 62),
            (VocabularyHeadSelection::LastOf67, 67),
            (VocabularyHeadSelection::LastOf72, 72),
        ] {
            assert!(selection_allowed(P::Text08, s));
            assert!(!selection_allowed(P::Text2B, s));
            let ranges = [
                (0x1000_0000, WEIGHT_BYTES),
                (0x4000_0000, rows * INPUT_BYTES),
                (0x5000_0000, OUTPUT_BYTES),
            ];
            assert_eq!(
                s.checked_addresses(&ranges).unwrap(),
                [
                    ranges[0].0,
                    ranges[1].0 + (rows - 1) * INPUT_BYTES,
                    ranges[2].0
                ]
            );
            for index in 0..3 {
                let mut bad = ranges;
                bad[index].1 -= 16;
                assert!(s.checked_addresses(&bad).is_err());
            }
            let mut alias = ranges;
            alias[2].0 = ranges[1].0;
            assert!(s.checked_addresses(&alias).is_err());
            if matches!(
                s,
                VocabularyHeadSelection::LastOf57
                    | VocabularyHeadSelection::LastOf62
                    | VocabularyHeadSelection::LastOf67
                    | VocabularyHeadSelection::LastOf72
            ) {
                for old in [
                    P::VocabularyHead,
                    P::FullQwen08,
                    P::FixedPromptFull,
                    P::FixedPromptNatural,
                ] {
                    assert!(!selection_allowed(old, s));
                }
            }
        }
        assert!(!selection_allowed(
            P::Text08,
            VocabularyHeadSelection::LastOfTwo
        ));
        assert!(!selection_allowed(
            P::Text08,
            VocabularyHeadSelection::LastOfSixtyFour
        ));
    }
    #[test]
    fn full_vocabulary_is_a_separate_closed_geometry() {
        let m = manifest();
        assert_eq!(m.grid, [31_040, 1, 1]);
        assert_eq!(m.block, [128, 1, 1]);
        assert_eq!(m.dynamic_shared_bytes, 0);
        assert_eq!(m.attributes, CubinLaunchAttributes::None);
        assert_eq!(
            m.bindings.iter().map(|b| b.bytes).collect::<Vec<_>>(),
            [248_320 * 1024 * 2, 1024 * 2, 248_320 * 2]
        );
        assert_eq!(IMAGE, PinnedBf16Cubin::Bf16Gemv2TSm86);
        assert_eq!(
            u32::from_le_bytes(TEMPLATE[68..72].try_into().unwrap()),
            248_320
        );
    }

    #[test]
    fn exact_output_alias_is_one_write_binding_without_workspace() {
        let m = manifest();
        assert_eq!(m.bindings.len(), 3);
        assert_eq!(
            m.fixups
                .iter()
                .map(|f| (f.byte_offset, f.binding, f.extent, f.io))
                .collect::<Vec<_>>(),
            [
                (0, 0, WEIGHT_BYTES, CubinIo::Read),
                (16, 1, INPUT_BYTES, CubinIo::Read),
                (32, 2, OUTPUT_BYTES, CubinIo::Write),
                (48, 2, OUTPUT_BYTES, CubinIo::Write)
            ]
        );
        assert_eq!(m.bindings[2].io, CubinIo::Write);
    }

    #[test]
    fn exact_reflected_layout_and_all_nonpointer_words_are_preserved() {
        let m = manifest();
        assert_eq!(
            m.parameters
                .iter()
                .map(|p| (p.offset, p.template.len()))
                .collect::<Vec<_>>(),
            [(0, 152), (152, 4), (156, 4)]
        );
        assert_eq!(
            m.parameters
                .into_iter()
                .flat_map(|p| p.template)
                .collect::<Vec<_>>(),
            TEMPLATE
        );
        let nonzero = TEMPLATE
            .chunks_exact(4)
            .enumerate()
            .filter_map(|(i, b)| {
                let v = u32::from_le_bytes(b.try_into().unwrap());
                (v != 0).then_some((i * 4, v))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            nonzero,
            [
                (64, 1024),
                (68, 248320),
                (72, 1024),
                (76, 1),
                (80, 1),
                (84, 1),
                (88, 1),
                (120, 1f32.to_bits()),
                (152, 1f32.to_bits())
            ]
        );
    }
}

// Separate finite streamed source windows; old head manifest remains exact.
pub(crate) mod streamed;
