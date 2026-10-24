//! Exact old attention20 +six separately rounded gate records only.
use super::{CubinKernelManifest, PinnedCapturedBf16Cubin, VerifiedCubin, failure};
use cubecl_core::server::ServerError;
pub(super) fn require(image: &VerifiedCubin, m: &CubinKernelManifest) -> Result<(), ServerError> {
    if super::attention_policy::require(image, m).is_ok() {
        return Ok(());
    }
    let kind = image
        .captured_from()
        .filter(|kind| {
            matches!(
                kind,
                PinnedCapturedBf16Cubin::AttentionSigmoidSm86
                    | PinnedCapturedBf16Cubin::AttentionMultiplySm86
            )
        })
        .ok_or_else(|| failure("attention-gate exact captured resource provenance"))?;
    if image.derived_from().is_some() {
        return Err(failure(
            "gate captured resource cannot masquerade as derived",
        ));
    }
    kind.require_image(image.byte_len(), image.sha256(), image.binary_sm())?;
    require_record(image.sha256(), image.byte_len(), image.binary_sm(), m)
}
fn require_record(
    sha: [u8; 32],
    bytes: usize,
    sm: u8,
    m: &CubinKernelManifest,
) -> Result<(), ServerError> {
    let digest = super::attention_policy::digest(m);
    if !RULES
        .iter()
        .any(|r| r.0 == sha && r.1 == bytes && r.2 == sm && r.3 == digest)
    {
        return Err(failure("exact complete two-kernel BF16 gate ABI"));
    }
    Ok(())
}
const RULES: &[([u8; 32], usize, u8, [u8; 32])] = &[
    (
        [
            0x08, 0xc0, 0x0e, 0xe6, 0x7f, 0xa7, 0x45, 0x2b, 0x53, 0x70, 0xa7, 0xb2, 0xfc, 0x2b,
            0x61, 0x1a, 0xfd, 0xbe, 0x21, 0x9b, 0x53, 0x43, 0xce, 0xbf, 0x93, 0x5a, 0xee, 0xf6,
            0xe8, 0x72, 0xef, 0x8b,
        ],
        2679648,
        86,
        [
            0xcc, 0x6c, 0x05, 0xdb, 0x35, 0x6a, 0x92, 0xbf, 0xec, 0x06, 0x3e, 0x79, 0x35, 0x9c,
            0x6b, 0x86, 0x74, 0xdc, 0x88, 0x93, 0x7b, 0x34, 0x5f, 0xf4, 0x68, 0x20, 0x26, 0x17,
            0xf9, 0x1b, 0x82, 0xd2,
        ],
    ), //sigmoid1
    (
        [
            0x08, 0xc0, 0x0e, 0xe6, 0x7f, 0xa7, 0x45, 0x2b, 0x53, 0x70, 0xa7, 0xb2, 0xfc, 0x2b,
            0x61, 0x1a, 0xfd, 0xbe, 0x21, 0x9b, 0x53, 0x43, 0xce, 0xbf, 0x93, 0x5a, 0xee, 0xf6,
            0xe8, 0x72, 0xef, 0x8b,
        ],
        2679648,
        86,
        [
            0x2c, 0x10, 0xf6, 0xb3, 0x56, 0x39, 0x05, 0x8f, 0xce, 0xee, 0x2a, 0x67, 0xa2, 0x89,
            0xb4, 0xc2, 0xe3, 0xf2, 0x0d, 0x1d, 0x90, 0xe5, 0x7d, 0xd7, 0x99, 0xd3, 0x34, 0xa7,
            0x14, 0x46, 0x2b, 0xde,
        ],
    ), //sigmoid2
    (
        [
            0x08, 0xc0, 0x0e, 0xe6, 0x7f, 0xa7, 0x45, 0x2b, 0x53, 0x70, 0xa7, 0xb2, 0xfc, 0x2b,
            0x61, 0x1a, 0xfd, 0xbe, 0x21, 0x9b, 0x53, 0x43, 0xce, 0xbf, 0x93, 0x5a, 0xee, 0xf6,
            0xe8, 0x72, 0xef, 0x8b,
        ],
        2679648,
        86,
        [
            0x95, 0xb7, 0xda, 0x11, 0x06, 0x21, 0xa7, 0xc4, 0xdb, 0x15, 0x3e, 0x97, 0xce, 0x17,
            0xb8, 0x52, 0x55, 0xae, 0x1a, 0x51, 0xe4, 0xf8, 0x2d, 0xa8, 0xfd, 0x13, 0x4b, 0xe6,
            0xda, 0x3e, 0x44, 0x9b,
        ],
    ), //sigmoid64
    (
        [
            0x92, 0x77, 0xf5, 0xb0, 0xfa, 0x8d, 0x3a, 0x02, 0x00, 0x4e, 0x2c, 0x8f, 0x02, 0xc1,
            0x99, 0x1c, 0x0b, 0xef, 0x28, 0x1b, 0x8a, 0xa8, 0xae, 0xba, 0xb8, 0x86, 0x79, 0x7d,
            0xaa, 0x13, 0x50, 0x2a,
        ],
        3397824,
        86,
        [
            0x46, 0xd1, 0xe2, 0x43, 0x20, 0xb8, 0xa4, 0x21, 0x80, 0xfb, 0xdc, 0x6a, 0x0c, 0xd5,
            0xbb, 0xda, 0x2e, 0x65, 0xd2, 0x2a, 0xe5, 0x47, 0x4b, 0xd8, 0x67, 0xb0, 0xda, 0xdd,
            0x68, 0xb0, 0xc2, 0x07,
        ],
    ), //multiply1
    (
        [
            0x92, 0x77, 0xf5, 0xb0, 0xfa, 0x8d, 0x3a, 0x02, 0x00, 0x4e, 0x2c, 0x8f, 0x02, 0xc1,
            0x99, 0x1c, 0x0b, 0xef, 0x28, 0x1b, 0x8a, 0xa8, 0xae, 0xba, 0xb8, 0x86, 0x79, 0x7d,
            0xaa, 0x13, 0x50, 0x2a,
        ],
        3397824,
        86,
        [
            0x42, 0xca, 0xb2, 0x0f, 0x8a, 0x31, 0x63, 0x0c, 0x03, 0xee, 0x0e, 0x81, 0x12, 0xc4,
            0xc1, 0x6c, 0xff, 0x2b, 0xba, 0x5a, 0xd2, 0x75, 0xb6, 0xc5, 0x02, 0x37, 0xfb, 0x10,
            0x50, 0x07, 0xd4, 0xba,
        ],
    ), //multiply2
    (
        [
            0x92, 0x77, 0xf5, 0xb0, 0xfa, 0x8d, 0x3a, 0x02, 0x00, 0x4e, 0x2c, 0x8f, 0x02, 0xc1,
            0x99, 0x1c, 0x0b, 0xef, 0x28, 0x1b, 0x8a, 0xa8, 0xae, 0xba, 0xb8, 0x86, 0x79, 0x7d,
            0xaa, 0x13, 0x50, 0x2a,
        ],
        3397824,
        86,
        [
            0xef, 0x3f, 0xf5, 0x6c, 0xf0, 0x47, 0xb6, 0xc4, 0xcd, 0xb9, 0xf8, 0x39, 0x38, 0x8d,
            0xca, 0x67, 0xab, 0xf3, 0x52, 0x3c, 0x9b, 0x49, 0x2f, 0xb6, 0x60, 0x75, 0xce, 0xab,
            0x22, 0xde, 0x43, 0xec,
        ],
    ), //multiply64
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_command::cubin::{
        CubinBinding, CubinIo, CubinLaunchAttributes, CubinParameter, CubinPointerFixup,
        CubinSetPolicy,
    };
    fn fixture(multiply: bool, rows: usize) -> CubinKernelManifest {
        let count = if multiply { 3 } else { 2 };
        CubinKernelManifest{
            entry:if multiply{"_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIN3c108BFloat16ES4_S4_NS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLm3EEEEviT0_T1_"}else{"_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_19sigmoid_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE0_clEvENKUlvE2_clEvEUlN3c108BFloat16EE_St5arrayIPcLm2EEEEviT0_T1_"}.into(),
            binary_sm:86,grid:[(rows*2)as u32,1,1],block:[128,1,1],dynamic_shared_bytes:0,attributes:CubinLaunchAttributes::None,
            bindings:(0..count).map(|i|CubinBinding{bytes:(rows*4096)as u64,alignment:8,io:if i==0{CubinIo::Write}else{CubinIo::Read}}).collect(),
            parameters:vec![CubinParameter{offset:0,template:((rows*2048)as u32).to_le_bytes().to_vec()},CubinParameter{offset:4,template:vec![0]},CubinParameter{offset:8,template:vec![0;count*8]}],
            fixups:(0..count).map(|i|CubinPointerFixup{parameter:2,byte_offset:i*8,binding:i,binding_offset:0,extent:(rows*4096)as u64,alignment:8,io:if i==0{CubinIo::Write}else{CubinIo::Read}}).collect(),
        }
    }
    #[test]
    fn six_complete_abis_refuse_scalar_pointer_geometry_io_and_image_substitution() {
        for multiply in [false, true] {
            for (row_index, rows) in [1, 2, 64].into_iter().enumerate() {
                let rule = RULES[usize::from(multiply) * 3 + row_index];
                let m = fixture(multiply, rows);
                assert!(m.validate().is_ok());
                assert!(require_record(rule.0, rule.1, rule.2, &m).is_ok());
                for change in 0..12 {
                    let mut bad = m.clone();
                    match change {
                        0 => bad.entry.push('x'),
                        1 => bad.binary_sm = 80,
                        2 => bad.grid[0] += 1,
                        3 => bad.block[0] = 256,
                        4 => bad.dynamic_shared_bytes = 4,
                        5 => bad.parameters[0].template[0] ^= 1,
                        6 => bad.parameters[1].template[0] = 3,
                        7 => bad.parameters[2].template[0] = 1,
                        8 => bad.bindings[0].io = CubinIo::Read,
                        9 => bad.fixups[0].binding = 1,
                        10 => bad.fixups[0].extent -= 2,
                        _ => bad.attributes = CubinLaunchAttributes::Unsupported,
                    }
                    assert!(
                        require_record(rule.0, rule.1, rule.2, &bad).is_err(),
                        "gate ABI change{change}"
                    );
                }
                let foreign = RULES[if multiply { 0 } else { 3 }];
                assert!(require_record(foreign.0, foreign.1, foreign.2, &m).is_err());
            }
        }
    }
    #[test]
    fn computed_gate_capacity_is_new_immutable_scope_and_cannot_upgrade_old_sets() {
        assert!(
            CubinSetPolicy::FirstAttentionComputedGate
                .limits(10, 14, 25, true, true)
                .is_ok()
        );
        for (modules, functions, records, new_module, new_function) in [
            (11, 14, 25, true, false),
            (10, 15, 25, false, true),
            (10, 14, 26, false, false),
            (12, 0, 0, false, false),
        ] {
            assert!(
                CubinSetPolicy::FirstAttentionComputedGate
                    .limits(modules, functions, records, new_module, new_function)
                    .is_err()
            );
        }
        for policy in [
            CubinSetPolicy::ProjectionSix,
            CubinSetPolicy::ConnectedQwen08,
            CubinSetPolicy::FirstAttentionCaptured,
        ] {
            assert!(policy.limits(11, 15, 25, false, false).is_err());
        }
        assert!(
            CubinSetPolicy::FirstAttentionCaptured
                .limits(8, 12, 19, true, true)
                .is_ok()
        );
    }
}
