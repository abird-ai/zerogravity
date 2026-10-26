//! Finite original FixedPrompt FA packet; independent proof gates remain closed.
use super::super::cubin_manifests::ProjectionImage;
use super::manifests::{AttentionLaunch, AttentionRole};
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup,
};
use zerogravity_core::{Error, Result};

// Independent exact47 selected-code proofs remain ROOT obligations.
const MEMORY_PROOF: Option<()> = Some(());
const PADDING_AND_DUMMY_SEMANTICS: Option<()> = Some(());
const SELECTED_RESOURCES: Option<()> = Some(());

const ENTRY: &str = "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi256ELi64ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_traitsILi256ELi64ELi64ELi4ES3_EELb1ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE";
#[cfg(test)]
const POINTER_OFFSETS: [usize; 9] = [0, 8, 16, 112, 160, 224, 232, 248, 352];
#[cfg(test)]
const OBSERVED_TAIL: [u8; 6] = [0xa4, 0x20, 0xa0, 0x73, 0, 0];

// Original capture2 event103 declared aggregate: zero only nine typed pointers.
// Preserve captured padding466..471; no unused-byte canonicalization is claimed.
const TEMPLATE_HEX: &str = "0000000000000000000000000000000000000000000000000000000000000000008008000000000000800800000000000008000000000000000400000000000000040000000000000001000000000000000200000000000000020000000000000800000002000000040000000000000000000000000000000000000000000000000000000000000000080000000000000001000000000000000000000000000000000000000000000000000000000000010000002f0000002f0000000000000000010000800000008000000000010000000000002f0000000000803d3baab83d000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000200200000000803fff0000000000803f0000803d2f0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000101010000000000000000000000000000000000000000000100a420a0730000";
const fn template() -> [u8; 472] {
    const fn nibble(x: u8) -> u8 {
        match x {
            b'0'..=b'9' => x - b'0',
            b'a'..=b'f' => x - b'a' + 10,
            _ => panic!("invalid fixed FA literal"),
        }
    }
    let text = TEMPLATE_HEX.as_bytes();
    assert!(text.len() == 944);
    let mut out = [0; 472];
    let mut i = 0;
    while i < out.len() {
        out[i] = nibble(text[i * 2]) * 16 + nibble(text[i * 2 + 1]);
        i += 1;
    }
    out
}
const TEMPLATE: [u8; 472] = template();
fn binding(bytes: u64, alignment: u64, io: CubinIo) -> CubinBinding {
    CubinBinding {
        bytes,
        alignment,
        io,
    }
}
fn fixup(
    byte_offset: usize,
    binding: usize,
    binding_offset: u64,
    extent: u64,
    alignment: u64,
    io: CubinIo,
) -> CubinPointerFixup {
    CubinPointerFixup {
        parameter: 0,
        byte_offset,
        binding,
        binding_offset,
        extent,
        alignment,
        io,
    }
}
fn finite_packet() -> AttentionLaunch {
    AttentionLaunch {
        image: ProjectionImage::AttentionFaCausalCombineSm80,
        roles: &[
            AttentionRole::Q,
            AttentionRole::OriginalKvState,
            AttentionRole::AttentionOutput,
            AttentionRole::LseF32,
            AttentionRole::DummyCuKI32,
            AttentionRole::CuQI32,
            AttentionRole::SeqUsedI32,
            AttentionRole::BlockTableI32,
        ],
        manifest: CubinKernelManifest {
            entry: ENTRY.into(),
            binary_sm: 80,
            grid: [1, 1, 8],
            block: [128, 1, 1],
            dynamic_shared_bytes: 98304,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![
                binding(192512, 16, CubinIo::Read),
                binding(5570560, 16, CubinIo::Read),
                binding(192512, 16, CubinIo::Write),
                binding(1504, 16, CubinIo::Write),
                binding(8, 4, CubinIo::Read),
                binding(8, 4, CubinIo::Read),
                binding(4, 4, CubinIo::Read),
                binding(16, 4, CubinIo::Read),
            ],
            parameters: vec![CubinParameter {
                offset: 0,
                template: TEMPLATE.to_vec(),
            }],
            fixups: vec![
                fixup(0, 0, 0, 192512, 16, CubinIo::Read),
                fixup(8, 1, 0, 5570560, 16, CubinIo::Read),
                fixup(16, 1, 512, 5570048, 16, CubinIo::Read),
                fixup(112, 2, 0, 192512, 16, CubinIo::Write),
                fixup(160, 3, 0, 1504, 4, CubinIo::Write),
                fixup(232, 4, 0, 8, 4, CubinIo::Read),
                fixup(224, 5, 0, 8, 4, CubinIo::Read),
                fixup(248, 6, 0, 4, 4, CubinIo::Read),
                fixup(352, 7, 0, 16, 4, CubinIo::Read),
            ],
        },
    }
}
fn require_proofs(memory: Option<()>, padding: Option<()>, resources: Option<()>) -> Result<()> {
    if memory.is_none() || padding.is_none() || resources.is_none() {
        return Err(Error::Unsupported(
            "fixed_prompt FA selected memory/padding/dummy/resource proof pending ROOT".into(),
        ));
    }
    Ok(())
}
pub(super) fn packet() -> Result<AttentionLaunch> {
    require_proofs(
        MEMORY_PROOF,
        PADDING_AND_DUMMY_SEMANTICS,
        SELECTED_RESOURCES,
    )?;
    Ok(finite_packet())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "ROOT metadata collector; no native or numerical permission"]
    fn actual_finite_fa_packet_metadata_for_root_review() {
        let packet = finite_packet();
        println!(
            "FA47_PACKET image={:?} manifest={:?}",
            packet.image, packet.manifest
        );
    }
    fn u32_at(offset: usize) -> u32 {
        u32::from_le_bytes(TEMPLATE[offset..offset + 4].try_into().unwrap())
    }
    fn u64_at(offset: usize) -> u64 {
        u64::from_le_bytes(TEMPLATE[offset..offset + 8].try_into().unwrap())
    }
    #[test]
    fn exact47_requires_every_independently_qualified_proof() {
        assert!(packet().is_ok());
        for (memory, padding, resources) in [
            (None, Some(()), Some(())),
            (Some(()), None, Some(())),
            (Some(()), Some(()), None),
        ] {
            assert!(require_proofs(memory, padding, resources).is_err());
        }
    }
    #[test]
    fn only_typed_pointer_slots_are_zeroed_and_observed_padding_survives() {
        for offset in POINTER_OFFSETS {
            assert_eq!(u64_at(offset), 0);
        }
        assert_eq!(&TEMPLATE[466..472], &OBSERVED_TAIL);
        assert_eq!(&TEMPLATE[440..444], &[1, 1, 1, 0]);
        assert_eq!(&TEMPLATE[464..466], &[1, 0]);
        assert_eq!(u32_at(216), 0x3d800000);
        assert_eq!(u32_at(220), 0x3db8aa3b);
    }
    #[test]
    fn call0_controls_are47_causal_unswapped_direct_output() {
        for (offset, value) in [
            (96, 8),
            (100, 2),
            (104, 4),
            (176, 1),
            (180, 47),
            (184, 47),
            (188, 0),
            (192, 256),
            (196, 128),
            (200, 128),
            (204, 256),
            (208, 0),
            (212, 47),
            (368, 544),
            (388, 47),
            (392, 0),
            (444, 0),
        ] {
            assert_eq!(u32_at(offset), value);
        }
        for offset in [120, 168, 240, 256, 264] {
            assert_eq!(u64_at(offset), 0);
        }
        assert_eq!(u64_at(360), 4);
    }
    #[test]
    fn shared_cache_two_views_are_one_owner_and_every_fixup_stays_bounded() {
        let packet = finite_packet();
        assert_eq!(packet.roles.len(), 8);
        assert_eq!(packet.manifest.bindings.len(), 8);
        assert_eq!(packet.manifest.fixups.len(), 9);
        for f in &packet.manifest.fixups {
            let b = &packet.manifest.bindings[f.binding];
            assert_eq!(f.parameter, 0);
            assert_eq!(f.binding_offset + f.extent, b.bytes);
            assert_eq!(f.io, b.io);
            assert!(f.alignment <= b.alignment);
            assert_eq!(&TEMPLATE[f.byte_offset..f.byte_offset + 8], &[0; 8]);
        }
        let k = &packet.manifest.fixups[1];
        let v = &packet.manifest.fixups[2];
        assert_eq!(k.binding, v.binding);
        assert_eq!(v.binding_offset, 512);
        assert_eq!(packet.manifest.bindings[2].bytes, 47 * 2048 * 2);
        assert_eq!(packet.manifest.bindings[3].bytes, 8 * 47 * 4);
    }
}
