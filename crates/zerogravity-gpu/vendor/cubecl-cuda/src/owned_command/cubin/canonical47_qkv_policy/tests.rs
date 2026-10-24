//! Complete ABI mutation guard; fixture grants no production pin or image seal.
use super::*;
use crate::owned_command::{
    CubinBinding, CubinIo, CubinLaunchAttributes, CubinParameter, CubinPointerFixup,
};
fn fixture() -> CubinKernelManifest {
    CubinKernelManifest {
        entry: "policy-only-fixture".into(),
        binary_sm: 80,
        grid: [20, 1, 2],
        block: [256, 1, 1],
        dynamic_shared_bytes: 24576,
        attributes: CubinLaunchAttributes::None,
        bindings: vec![CubinBinding {
            bytes: 96256,
            alignment: 16,
            io: CubinIo::Read,
        }],
        parameters: vec![CubinParameter {
            offset: 0,
            template: vec![0; 192],
        }],
        fixups: vec![CubinPointerFixup {
            parameter: 0,
            byte_offset: 8,
            binding: 0,
            binding_offset: 0,
            extent: 96256,
            alignment: 16,
            io: CubinIo::Read,
        }],
    }
}
#[test]
fn no_unaccepted_production_pin_and_complete_mutation_refusal() {
    let m = fixture();
    let (sha, bytes, sm) = IMAGES[0];
    assert!(require_record(sha, bytes, sm, &m, None).is_err());
    let abis = Some([super::super::attention_policy::digest(&m), [0; 32]]);
    require_record(sha, bytes, sm, &m, abis).unwrap();
    for mutation in 0..12 {
        let mut wrong = m.clone();
        match mutation {
            0 => wrong.entry.push('x'),
            1 => wrong.binary_sm += 1,
            2 => wrong.grid[2] += 1,
            3 => wrong.block[0] += 1,
            4 => wrong.dynamic_shared_bytes += 16,
            5 => wrong.parameters[0].template[108] ^= 1,
            6 => wrong.parameters[0].offset += 8,
            7 => wrong.bindings[0].bytes += 2,
            8 => wrong.bindings[0].io = CubinIo::ReadWrite,
            9 => wrong.fixups[0].binding_offset += 16,
            10 => wrong.fixups[0].extent += 2,
            _ => wrong.fixups[0].alignment += 16,
        }
        assert!(
            require_record(sha, bytes, sm, &wrong, abis).is_err(),
            "mutation{mutation}"
        );
    }
    let mut wrong = sha;
    wrong[0] ^= 1;
    assert!(require_record(wrong, bytes, sm, &m, abis).is_err());
    assert!(require_record(sha, bytes + 1, sm, &m, abis).is_err());
    assert!(require_record(sha, bytes, sm + 1, &m, abis).is_err());
}
#[test]
fn new_set_limits_are_separate_and_old_limits_stay_closed() {
    use super::super::policy::CubinSetPolicy as P;
    assert!(P::Canonical47Qkv.limits(6, 9, 9, false, false).is_ok());
    for args in [
        (6, 9, 10, false, false),
        (6, 9, 9, true, false),
        (6, 9, 9, false, true),
    ] {
        assert!(
            P::Canonical47Qkv
                .limits(args.0, args.1, args.2, args.3, args.4)
                .is_err()
        );
    }
    assert!(P::Canonical47Mlp.limits(5, 7, 7, false, false).is_ok());
    assert!(P::Canonical47Mlp.limits(5, 7, 8, false, false).is_err());
    assert!(P::ProjectionSix.limits(6, 8, 18, false, false).is_ok());
    assert!(P::ProjectionSix.limits(6, 8, 19, false, false).is_err());
}
