//! Distinct full layer3 union; exact new gate ABIs, old FA set unchanged.
use super::{CubinKernelManifest, PinnedCapturedBf16Cubin, VerifiedCubin, failure};
use cubecl_core::server::ServerError;
const ABIS: Option<[[u8; 32]; 2]> = Some([
    [
        115, 123, 221, 53, 246, 55, 230, 211, 180, 167, 78, 172, 114, 52, 159, 176, 160, 139, 233,
        213, 61, 86, 176, 168, 88, 46, 224, 191, 210, 137, 155, 247,
    ],
    [
        19, 164, 202, 167, 208, 145, 244, 56, 130, 215, 51, 169, 231, 149, 118, 138, 41, 123, 0,
        201, 250, 240, 80, 155, 20, 66, 237, 188, 139, 224, 157, 13,
    ],
]);
pub(super) fn require(
    image: &VerifiedCubin,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    if super::canonical47_fa_policy::require(image, manifest).is_ok() {
        return Ok(());
    }
    let (index, pin) = match image.captured_from() {
        Some(PinnedCapturedBf16Cubin::AttentionSigmoidSm86) => {
            (0, PinnedCapturedBf16Cubin::AttentionSigmoidSm86)
        }
        Some(PinnedCapturedBf16Cubin::AttentionMultiplySm86) => {
            (1, PinnedCapturedBf16Cubin::AttentionMultiplySm86)
        }
        _ => {
            return Err(failure(
                "canonical47 layer3 exact Sig/Mul selected-resource provenance",
            ));
        }
    };
    if image.derived_from().is_some() {
        return Err(failure(
            "canonical47 layer3 captured image cannot claim derived provenance",
        ));
    }
    pin.require_image(image.byte_len(), image.sha256(), image.binary_sm())?;
    require_record(index, manifest, ABIS)
}
fn require_record(
    index: usize,
    manifest: &CubinKernelManifest,
    abis: Option<[[u8; 32]; 2]>,
) -> Result<(), ServerError> {
    let abis =
        abis.ok_or_else(|| failure("canonical47 layer3 complete Sig/Mul ABI pins pending ROOT"))?;
    if abis.get(index) != Some(&super::attention_policy::digest(manifest)) {
        return Err(failure("canonical47 layer3 complete gate ABI differs"));
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
    fn gate_full_abi_refuses_every_field_and_crossed_index() {
        let m = CubinKernelManifest {
            entry: "layer3fixture".into(),
            binary_sm: 86,
            grid: [94, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![CubinBinding {
                bytes: 192512,
                alignment: 8,
                io: CubinIo::Write,
            }],
            parameters: vec![
                CubinParameter {
                    offset: 0,
                    template: 96256i32.to_le_bytes().to_vec(),
                },
                CubinParameter {
                    offset: 4,
                    template: vec![0],
                },
                CubinParameter {
                    offset: 8,
                    template: vec![0; 8],
                },
            ],
            fixups: vec![CubinPointerFixup {
                parameter: 2,
                byte_offset: 0,
                binding: 0,
                binding_offset: 0,
                extent: 192512,
                alignment: 8,
                io: CubinIo::Write,
            }],
        };
        assert!(require_record(0, &m, None).is_err());
        let fixture = Some([super::super::attention_policy::digest(&m), [0; 32]]);
        require_record(0, &m, fixture).unwrap();
        assert!(require_record(1, &m, fixture).is_err());
        assert!(require_record(2, &m, fixture).is_err());
        for mutation in 0..18 {
            let mut wrong = m.clone();
            match mutation {
                0 => wrong.entry.push('x'),
                1 => wrong.binary_sm += 1,
                2 => wrong.grid[0] += 1,
                3 => wrong.block[0] += 1,
                4 => wrong.dynamic_shared_bytes += 4,
                5 => wrong.attributes = CubinLaunchAttributes::Unsupported,
                6 => wrong.bindings[0].bytes += 4,
                7 => wrong.bindings[0].alignment += 4,
                8 => wrong.bindings[0].io = CubinIo::Read,
                9 => wrong.parameters[0].offset += 1,
                10 => wrong.parameters[0].template[0] ^= 1,
                11 => wrong.fixups[0].parameter += 1,
                12 => wrong.fixups[0].byte_offset += 1,
                13 => wrong.fixups[0].binding += 1,
                14 => wrong.fixups[0].binding_offset += 8,
                15 => wrong.fixups[0].extent += 8,
                16 => wrong.fixups[0].alignment += 4,
                _ => wrong.fixups[0].io = CubinIo::Read,
            }
            assert!(
                require_record(0, &wrong, fixture).is_err(),
                "mutation{mutation}"
            );
        }
    }
    #[test]
    fn policy_capacity_does_not_widen_fa_or_kv() {
        use super::super::policy::CubinSetPolicy as P;
        assert!(
            P::Canonical47Layer3
                .limits(11, 14, 14, false, false)
                .is_ok()
        );
        for (m, f, r, nm, nf) in [
            (11, 14, 15, false, false),
            (11, 14, 14, true, false),
            (11, 14, 14, false, true),
            (12, 14, 14, false, false),
        ] {
            assert!(P::Canonical47Layer3.limits(m, f, r, nm, nf).is_err());
        }
        assert!(P::Canonical47Fa.limits(11, 14, 14, false, false).is_err());
        assert!(P::Canonical47QkKv.limits(11, 14, 14, false, false).is_err());
    }
}
