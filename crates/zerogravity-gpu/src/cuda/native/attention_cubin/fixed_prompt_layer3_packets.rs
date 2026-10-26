//! Finite original47 gate packets; same selected code, new exact scalar/bounds.
use super::manifests::{AttentionCase, AttentionLaunch};
use crate::cuda::native::cubin_manifests::ProjectionImage;
use cubecl::cuda::CubinKernelManifest;
use zerogravity_core::{Error, Result};
const MEMORY_PROOF: Option<()> = Some(());
pub(crate) fn require_memory_qualification() -> Result<()> {
    require_memory_proof(MEMORY_PROOF)
}
fn require_memory_proof(proof: Option<()>) -> Result<()> {
    proof.ok_or_else(|| {
        Error::Unsupported(
            "fixed_prompt layer3 Sig/Mul exact memory/control proof pending ROOT".into(),
        )
    })
}
fn finite_packets() -> [AttentionLaunch; 2] {
    let mut packets = [
        super::gate_manifests::sigmoid(AttentionCase::Initial64),
        super::gate_manifests::multiply(AttentionCase::Initial64),
    ];
    for packet in &mut packets {
        packet.manifest.grid = [94, 1, 1];
        packet.manifest.parameters[0].template = (96256i32).to_le_bytes().to_vec();
        for binding in &mut packet.manifest.bindings {
            binding.bytes = 192512;
        }
        for fixup in &mut packet.manifest.fixups {
            fixup.extent = 192512;
        }
    }
    packets
}
pub(crate) fn packets() -> Result<[(ProjectionImage, CubinKernelManifest); 2]> {
    require_memory_qualification()?;
    Ok(finite_packets().map(|packet| (packet.image, packet.manifest)))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact47_no_tail_geometry_preserves_reflected_functor_and_pointer_fields() {
        let packets = finite_packets();
        for (packet, n) in packets.into_iter().zip([2, 3]) {
            assert_eq!(packet.manifest.grid, [94, 1, 1]);
            assert_eq!(94 * 128 * 8, 96256);
            assert_eq!(packet.manifest.block, [128, 1, 1]);
            assert_eq!(packet.manifest.dynamic_shared_bytes, 0);
            assert_eq!(packet.manifest.parameters.len(), 3);
            assert_eq!(packet.manifest.parameters[0].offset, 0);
            assert_eq!(
                packet.manifest.parameters[0].template,
                96256i32.to_le_bytes()
            );
            assert_eq!(packet.manifest.parameters[1].offset, 4);
            assert_eq!(packet.manifest.parameters[1].template, [0]);
            assert_eq!(packet.manifest.parameters[2].offset, 8);
            assert_eq!(packet.manifest.parameters[2].template, vec![0; n * 8]);
            assert_eq!(packet.manifest.bindings.len(), n);
            for (i, fixup) in packet.manifest.fixups.iter().enumerate() {
                assert_eq!(
                    (fixup.parameter, fixup.byte_offset, fixup.binding),
                    (2, i * 8, i)
                );
                assert_eq!(fixup.binding_offset, 0);
                assert_eq!(fixup.extent, 192512);
                assert_eq!(fixup.alignment, 8);
                assert_eq!(fixup.io, packet.manifest.bindings[i].io);
            }
        }
        assert!(super::packets().is_ok());
        assert!(require_memory_proof(None).is_err());
    }
    #[test]
    #[ignore = "ROOT compiled metadata collector only; no native permission"]
    fn actual_layer3_gate_packets_for_root_comparison() {
        for (i, p) in finite_packets().into_iter().enumerate() {
            println!(
                "LAYER3_GATE47 index={i} image={:?} manifest={:?}",
                p.image, p.manifest
            );
        }
    }
}
