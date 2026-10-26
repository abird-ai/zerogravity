//! Exact47 source recipe over existing compiled QK/KV code; not legacy Initial64.
use super::super::cubin_manifests::ProjectionImage;
use super::manifests::{self, AttentionRole};
use cubecl::cuda::CubinKernelManifest;
use zerogravity_core::{Error, Result};

// ROOT materializes only after exact47 selected address/packet/source proof.
const MEMORY_PROOF: Option<()> = Some(());
pub(crate) fn require_memory_qualification() -> Result<()> {
    MEMORY_PROOF.ok_or_else(|| {
        Error::Unsupported("fixed_prompt QK/KV memory semantics pending ROOT proof".into())
    })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Role {
    Packed,
    Q,
    K,
    Gate,
    QGamma,
    KGamma,
    Rotary,
    Positions,
    Cache,
    Slots,
    KScale,
    VScale,
    O,
    Lse,
    CuQ,
    DummyCuK,
    SeqUsed,
    Table,
}
pub(crate) struct Packet {
    pub(crate) image: ProjectionImage,
    pub(crate) roles: Vec<Role>,
    pub(crate) manifest: CubinKernelManifest,
}
fn role(role: AttentionRole) -> Result<Role> {
    Ok(match role {
        AttentionRole::PackedQkv => Role::Packed,
        AttentionRole::Q => Role::Q,
        AttentionRole::K => Role::K,
        AttentionRole::Gate => Role::Gate,
        AttentionRole::QGamma => Role::QGamma,
        AttentionRole::KGamma => Role::KGamma,
        AttentionRole::CosSinSourcePrefix67 => Role::Rotary,
        AttentionRole::PositionsI64Pitch65 => Role::Positions,
        AttentionRole::OriginalKvState => Role::Cache,
        AttentionRole::SlotsI64 => Role::Slots,
        AttentionRole::KScale => Role::KScale,
        AttentionRole::VScale => Role::VScale,
        AttentionRole::AttentionOutput => Role::O,
        AttentionRole::LseF32 => Role::Lse,
        AttentionRole::CuQI32 => Role::CuQ,
        AttentionRole::DummyCuKI32 => Role::DummyCuK,
        AttentionRole::SeqUsedI32 => Role::SeqUsed,
        AttentionRole::BlockTableI32 => Role::Table,
        _ => {
            return Err(Error::State(
                "fixed_prompt two-stage packet contains unrelated role".into(),
            ));
        }
    })
}
fn finite_packets() -> Result<[Packet; 2]> {
    let mut qk = manifests::launch_12();
    let mut kv = manifests::launch_13();
    if qk.image != ProjectionImage::AttentionQkSm86
        || kv.image != ProjectionImage::AttentionKvSm86
        || qk.manifest.grid != [64, 10, 1]
        || kv.manifest.grid != [64, 1, 1]
    {
        return Err(Error::State(
            "fixed_prompt exact preexisting compiled QK/KV recipe identity changed".into(),
        ));
    }
    // Existing original parameter templates are byte-preserved. M is the
    // program grid; all actual47 scalar strides/hidden zero fields match.
    qk.manifest.grid = [47, 10, 1];
    kv.manifest.grid = [47, 1, 1];
    for (roles, manifest) in [(qk.roles, &mut qk.manifest), (kv.roles, &mut kv.manifest)] {
        if roles.len() != manifest.bindings.len() {
            return Err(Error::State(
                "fixed_prompt exact packet role/binding arity".into(),
            ));
        }
        for (actual, binding) in roles.iter().zip(&mut manifest.bindings) {
            binding.bytes = match role(*actual)? {
                Role::Packed => 481280,
                Role::Q | Role::Gate => 192512,
                Role::K => 48128,
                Role::Slots => 376,
                _ => binding.bytes,
            };
        }
        for fixup in &mut manifest.fixups {
            let binding = manifest
                .bindings
                .get(fixup.binding)
                .ok_or_else(|| Error::State("fixed_prompt exact pointer binding index".into()))?;
            fixup.extent = binding
                .bytes
                .checked_sub(fixup.binding_offset)
                .ok_or_else(|| {
                    Error::State("fixed_prompt pointer offset beyond original lease".into())
                })?;
        }
    }
    Ok([
        Packet {
            image: qk.image,
            roles: qk.roles.iter().map(|v| role(*v)).collect::<Result<_>>()?,
            manifest: qk.manifest,
        },
        Packet {
            image: kv.image,
            roles: kv.roles.iter().map(|v| role(*v)).collect::<Result<_>>()?,
            manifest: kv.manifest,
        },
    ])
}
pub(crate) fn packets() -> Result<[Packet; 2]> {
    require_memory_qualification()?;
    finite_packets()
}

pub(crate) fn packets_with_fa(include_fa: bool) -> Result<Vec<Packet>> {
    let mut all = packets()?.into_iter().collect::<Vec<_>>();
    if include_fa {
        let fa = super::fixed_prompt_fa_packet::packet()?;
        all.push(Packet {
            image: fa.image,
            roles: fa.roles.iter().map(|v| role(*v)).collect::<Result<_>>()?,
            manifest: fa.manifest,
        });
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "ROOT metadata collector; no native or numerical acceptance"]
    fn actual_finite_qkkv_packet_metadata_for_root_review() {
        for (index, packet) in finite_packets().unwrap().into_iter().enumerate() {
            println!(
                "QKKV47_PACKET index={index} image={:?} manifest={:?}",
                packet.image, packet.manifest
            );
        }
    }
    #[test]
    fn exact47_qk_kv_preserves_all_original_parameters_and_new_row_bounds() {
        let [qk, kv] = finite_packets().unwrap();
        assert_eq!(
            qk.manifest.parameters,
            manifests::launch_12().manifest.parameters
        );
        assert_eq!(
            kv.manifest.parameters,
            manifests::launch_13().manifest.parameters
        );
        assert_eq!(qk.manifest.grid, [47, 10, 1]);
        assert_eq!(kv.manifest.grid, [47, 1, 1]);
        assert_eq!(qk.manifest.bindings[0].bytes, 481280);
        assert_eq!(qk.manifest.bindings[1].bytes, 192512);
        assert_eq!(qk.manifest.bindings[2].bytes, 48128);
        assert_eq!(kv.manifest.bindings[0].bytes, 48128);
        assert_eq!(kv.manifest.bindings[3].bytes, 376);
        assert!(
            qk.manifest
                .fixups
                .iter()
                .chain(&kv.manifest.fixups)
                .all(|v| v.extent > 0)
        );
    }
    #[test]
    fn exact47_pointer_spans_stay_inside_complete_owned_leases() {
        for packet in packets().unwrap() {
            for fixup in packet.manifest.fixups {
                let binding = &packet.manifest.bindings[fixup.binding];
                assert_eq!(fixup.byte_offset, 0);
                assert_eq!(fixup.binding_offset + fixup.extent, binding.bytes);
                assert!(fixup.alignment <= binding.alignment);
                assert_eq!(fixup.io, binding.io);
            }
        }
    }
}
