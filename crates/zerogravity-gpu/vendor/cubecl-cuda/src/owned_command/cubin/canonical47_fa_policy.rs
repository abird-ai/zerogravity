//! Finite47 FA union; full ABI gate independent of image identity.
use super::{CubinKernelManifest, PinnedCapturedBf16Cubin, VerifiedCubin, failure};
use cubecl_core::server::ServerError;
const ABI: Option<[u8; 32]> = Some([
    0x65, 0xca, 0x08, 0x5a, 0x20, 0xba, 0xf3, 0xf2, 0xf0, 0x51, 0xbe, 0x59, 0xa3, 0xdc, 0x85, 0xef,
    0xc1, 0x99, 0x50, 0x05, 0x9d, 0xd7, 0x12, 0xd9, 0x7d, 0x1d, 0x74, 0x25, 0x10, 0xe6, 0x5c, 0xbe,
]);
pub(super) fn require(
    image: &VerifiedCubin,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    if super::canonical47_qkkv_policy::require(image, manifest).is_ok() {
        return Ok(());
    }
    if image.derived_from().is_some()
        || image.captured_from() != Some(PinnedCapturedBf16Cubin::AttentionFaCausalCombineSm80)
    {
        return Err(failure(
            "canonical47 FA exact captured selected-resource provenance",
        ));
    }
    PinnedCapturedBf16Cubin::AttentionFaCausalCombineSm80.require_image(
        image.byte_len(),
        image.sha256(),
        image.binary_sm(),
    )?;
    require_record(manifest, ABI)
}
fn require_record(
    manifest: &CubinKernelManifest,
    abi: Option<[u8; 32]>,
) -> Result<(), ServerError> {
    let abi = abi.ok_or_else(|| {
        failure("canonical47 FA complete ABI pending independent ROOT comparison")
    })?;
    if super::attention_policy::digest(manifest) != abi {
        return Err(failure(
            "canonical47 FA complete entry/scalars/padding/geometry/extents ABI differs",
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
    fn complete_abi_rejects_mutations_and_absent_authority() {
        let m = CubinKernelManifest {
            entry: "FA-policy-fixture".into(),
            binary_sm: 80,
            grid: [1, 1, 8],
            block: [128, 1, 1],
            dynamic_shared_bytes: 98304,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![CubinBinding {
                bytes: 1504,
                alignment: 16,
                io: CubinIo::Write,
            }],
            parameters: vec![CubinParameter {
                offset: 0,
                template: vec![0; 472],
            }],
            fixups: vec![CubinPointerFixup {
                parameter: 0,
                byte_offset: 160,
                binding: 0,
                binding_offset: 0,
                extent: 1504,
                alignment: 4,
                io: CubinIo::Write,
            }],
        };
        assert!(ABI.is_some());
        assert!(require_record(&m, None).is_err());
        assert!(require_record(&m, ABI).is_err());
        let fixture = Some(super::super::attention_policy::digest(&m));
        require_record(&m, fixture).unwrap();
        for mutation in 0..18 {
            let mut wrong = m.clone();
            match mutation {
                0 => wrong.entry.push('x'),
                1 => wrong.binary_sm += 1,
                2 => wrong.grid[2] += 1,
                3 => wrong.block[0] += 1,
                4 => wrong.dynamic_shared_bytes += 16,
                5 => wrong.attributes = CubinLaunchAttributes::Unsupported,
                6 => wrong.bindings[0].bytes += 4,
                7 => wrong.bindings[0].alignment += 4,
                8 => wrong.bindings[0].io = CubinIo::ReadWrite,
                9 => wrong.parameters[0].offset += 8,
                10 => wrong.parameters[0].template[470] ^= 1,
                11 => wrong.fixups[0].parameter += 1,
                12 => wrong.fixups[0].byte_offset += 8,
                13 => wrong.fixups[0].binding += 1,
                14 => wrong.fixups[0].binding_offset += 4,
                15 => wrong.fixups[0].extent += 4,
                16 => wrong.fixups[0].alignment += 4,
                _ => wrong.fixups[0].io = CubinIo::Read,
            }
            assert!(
                require_record(&wrong, fixture).is_err(),
                "mutation {mutation}"
            );
        }
    }
    #[test]
    fn exact_capacity_preserves_old_policies() {
        use super::super::policy::CubinSetPolicy as P;
        assert!(P::Canonical47Fa.limits(9, 12, 12, false, false).is_ok());
        for (m, f, r, nm, nf) in [
            (9, 12, 13, false, false),
            (9, 12, 12, true, false),
            (9, 12, 12, false, true),
            (10, 12, 12, false, false),
        ] {
            assert!(P::Canonical47Fa.limits(m, f, r, nm, nf).is_err());
        }
        assert!(P::Canonical47QkKv.limits(9, 12, 12, false, false).is_err());
        assert!(P::Canonical47Qkv.limits(9, 12, 12, false, false).is_err());
    }
}
