use super::{descriptor::*, manifests::*};
use cubecl::cuda::CubinIo;
use std::collections::BTreeSet;
use zerogravity_core::{accelerator::SourceIdentity, device_events::DeviceAccess};

#[test]
fn all_source_records_have_exact_closed_cohort_without_aliasing_roles() {
    let mut records = BTreeSet::new();
    let mut images = BTreeSet::new();
    let mut functions = BTreeSet::new();
    for case in AttentionCase::ALL {
        for stage in STAGES {
            for launch in launches(case, stage) {
                assert_eq!(launch.roles.len(), launch.manifest.bindings.len());
                assert_eq!(
                    launch.roles.iter().copied().collect::<BTreeSet<_>>().len(),
                    launch.roles.len()
                );
                images.insert(launch.image);
                functions.insert(launch.manifest.entry.clone());
                records.insert(super::manifest_identity(
                    &launch.manifest,
                    launch.image.sha256(),
                ));
                for f in &launch.manifest.fixups {
                    let b = &launch.manifest.bindings[f.binding];
                    assert!(
                        f.binding_offset
                            .checked_add(f.extent)
                            .is_some_and(|end| end <= b.bytes)
                    );
                    let p = &launch.manifest.parameters[f.parameter];
                    assert!(
                        p.template[f.byte_offset..f.byte_offset + 8]
                            .iter()
                            .all(|&v| v == 0)
                    );
                }
            }
        }
    }
    assert_eq!((images.len(), functions.len(), records.len()), (9, 13, 20));
    assert_eq!(images, required_images());
    assert!(require_closed_cases(&AttentionCase::ALL).is_ok());
    assert!(require_closed_cases(&AttentionCase::ALL[..4]).is_err());
    let mut wrong = AttentionCase::ALL;
    wrong.swap(0, 1);
    assert!(require_closed_cases(&wrong).is_err());
}

#[test]
fn actual_controls_preserve_original_block_and_pitch_without_remapping() {
    assert!(
        AttentionDescriptor::new(SourceIdentity::unbound(), AttentionCase::InitialOne).is_err()
    );
    let source = SourceIdentity::for_new_allocation().unwrap();
    for case in AttentionCase::ALL {
        let d = AttentionDescriptor::new(source, case).unwrap();
        let c = d.controls();
        assert_eq!(c.slot_count, case.rows());
        assert_eq!(c.seq_used as usize, case.base() + case.rows());
        assert_eq!(c.block_table, [case.block() as i32, 0, 0, 0]);
        assert_eq!(c.cu_q, [0, case.rows() as i32]);
        assert_eq!(c.dummy_cu_k, [0, 0]);
        for axis in 0..3 {
            for i in 0..case.rows() {
                assert_eq!(c.positions[axis * 65 + i], (case.base() + i) as i64);
            }
        }
        for i in 0..case.rows() {
            let slot = c.slots[i] as usize;
            assert_eq!(slot, case.block() * 544 + case.base() + i);
            assert!(slot / 544 < 5);
            assert!(slot % 544 < 67);
        }
        let roles = d.required_roles().unwrap();
        assert_eq!(
            roles[&AttentionRole::OriginalKvState],
            (KV_BYTES, DeviceAccess::Write)
        );
        assert!(roles.contains_key(&AttentionRole::OutInput));
        assert_ne!(AttentionRole::OutInput, AttentionRole::AttentionOutput);
    }
}

#[test]
fn native_n1_copy_and_split_partials_have_source_extents_and_distinct_outputs() {
    for case in AttentionCase::ALL {
        let fa = launches(case, AttentionStage::Fa);
        assert_eq!(
            fa.len(),
            if case == AttentionCase::CachedOneAt66 {
                2
            } else {
                1
            }
        );
        if case.rows() == 1 {
            assert!(
                fa.iter()
                    .all(|x| x.roles.contains(&AttentionRole::PrivateO))
            );
            assert!(
                fa.iter()
                    .all(|x| !x.roles.contains(&AttentionRole::AttentionOutput))
            );
        } else {
            assert!(fa[0].roles.contains(&AttentionRole::AttentionOutput));
        }
        if case == AttentionCase::CachedOneAt66 {
            for role in [AttentionRole::PartialOF32, AttentionRole::PartialLseF32] {
                let a = fa[0].roles.iter().position(|r| *r == role).unwrap();
                let b = fa[1].roles.iter().position(|r| *r == role).unwrap();
                assert_eq!(fa[0].manifest.bindings[a].io, CubinIo::Write);
                assert_eq!(fa[1].manifest.bindings[b].io, CubinIo::Read);
            }
        }
    }
}

#[test]
fn qk_positions_are_fixed65_pitch_and_interleaved_kv_is_one_binding() {
    for case in AttentionCase::ALL {
        let qk = launches(case, AttentionStage::Qk).remove(0);
        let p = qk
            .manifest
            .parameters
            .iter()
            .find(|p| p.offset == 96)
            .unwrap();
        assert_eq!(p.template, 65u32.to_le_bytes());
        let kv = launches(case, AttentionStage::Kv).remove(0);
        let cache = kv
            .roles
            .iter()
            .position(|r| *r == AttentionRole::OriginalKvState)
            .unwrap();
        let fields = kv
            .manifest
            .fixups
            .iter()
            .filter(|f| f.binding == cache)
            .collect::<Vec<_>>();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].binding_offset, 0);
        assert_eq!(fields[1].binding_offset, 512);
        assert_eq!(kv.manifest.bindings[cache].bytes, KV_BYTES as u64);
    }
}

#[test]
fn leaf_scope_refuses_missing_records_wrong_positions_and_nonfinite_operands() {
    use super::qualify::{
        CubinAttentionExpected, CubinAttentionFixture, CubinAttentionOutput, validate_fixtures,
    };
    fn fixtures() -> Vec<CubinAttentionFixture> {
        AttentionCase::ALL
            .into_iter()
            .map(|case| CubinAttentionFixture {
                case,
                normalized: vec![0; case.rows() * 2048],
                gated: vec![0; case.rows() * 4096],
                positions: (0..3)
                    .flat_map(|_| (0..case.rows()).map(|i| (case.base() + i) as i64))
                    .collect(),
                expected: [
                    CubinAttentionOutput::PackedQkv,
                    CubinAttentionOutput::Q,
                    CubinAttentionOutput::K,
                    CubinAttentionOutput::Gate,
                    CubinAttentionOutput::AttentionOutput,
                    CubinAttentionOutput::Projected,
                    CubinAttentionOutput::KvActive,
                ]
                .into_iter()
                .map(|output| CubinAttentionExpected {
                    output,
                    bytes: vec![0; output.bytes(case)],
                })
                .collect(),
            })
            .collect()
    }
    let cos = vec![0; 8576];
    assert!(validate_fixtures(&fixtures(), &cos).is_ok());
    for kind in 0..7 {
        let mut f = fixtures();
        match kind {
            0 => {
                f.pop();
            }
            1 => f.swap(0, 1),
            2 => f[2].positions[65] = 900,
            3 => {
                f[0].expected.pop();
            }
            4 => f[0].expected[0].output = CubinAttentionOutput::K,
            5 => f[0].normalized[..2].copy_from_slice(&0x7fc0u16.to_le_bytes()),
            _ => {
                f[0].gated.pop();
            }
        }
        assert!(validate_fixtures(&f, &cos).is_err(), "case{kind}");
    }
    assert!(validate_fixtures(&fixtures(), &cos[..8574]).is_err());
}

#[test]
fn computed_gate_is_exact11_15_26_and_keeps_old9_13_20_scope() {
    let mut records = BTreeSet::new();
    let mut images = BTreeSet::new();
    let mut functions = BTreeSet::new();
    let mut launches_count = 0;
    let scope = AttentionScope::ComputedGate;
    assert_eq!(
        scope.stages(),
        &[
            AttentionStage::Qkv,
            AttentionStage::Qk,
            AttentionStage::Kv,
            AttentionStage::Fa,
            AttentionStage::Sigmoid,
            AttentionStage::Multiply,
            AttentionStage::Out
        ]
    );
    for case in AttentionCase::ALL {
        for &stage in scope.stages() {
            for launch in launches(case, stage) {
                launches_count += 1;
                images.insert(launch.image);
                functions.insert(launch.manifest.entry.clone());
                records.insert(super::manifest_identity(
                    &launch.manifest,
                    launch.image.sha256(),
                ));
            }
        }
    }
    assert_eq!(
        (images.len(), functions.len(), records.len(), launches_count),
        (11, 15, 26, 40)
    );
    assert_eq!(images, super::required_gate_images());
    assert_eq!(AttentionScope::CapturedOutInput.records(), 20);
    assert_eq!(AttentionScope::CapturedOutInput.images(), 9);
    assert!(!STAGES.contains(&AttentionStage::Sigmoid));
    assert!(!STAGES.contains(&AttentionStage::Multiply));
}
#[test]
fn gate_uses_real_distinct_rounded_producer_and_true_out_consumer() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    for case in AttentionCase::ALL {
        let s = launches(case, AttentionStage::Sigmoid).remove(0);
        let m = launches(case, AttentionStage::Multiply).remove(0);
        assert_eq!(s.roles, &[AttentionRole::SigmoidGate, AttentionRole::Gate]);
        assert_eq!(
            m.roles,
            &[
                AttentionRole::OutInput,
                AttentionRole::AttentionOutput,
                AttentionRole::SigmoidGate
            ]
        );
        for launch in [&s, &m] {
            assert_eq!(
                launch.manifest.parameters[0].template,
                ((case.rows() * 2048) as u32).to_le_bytes()
            );
            assert_eq!(launch.manifest.parameters[1].offset, 4);
            assert_eq!(launch.manifest.parameters[1].template, [0]);
            assert_eq!(launch.manifest.parameters[2].offset, 8);
            assert_eq!(launch.manifest.grid, [(case.rows() * 2) as u32, 1, 1]);
            assert_eq!(launch.manifest.block, [128, 1, 1]);
            assert_eq!(launch.manifest.dynamic_shared_bytes, 0);
            for (i, b) in launch.manifest.bindings.iter().enumerate() {
                assert_eq!(b.bytes, (case.rows() * 4096) as u64);
                assert_eq!(
                    b.io,
                    if i == 0 {
                        CubinIo::Write
                    } else {
                        CubinIo::Read
                    }
                );
                let f = &launch.manifest.fixups[i];
                assert_eq!((f.parameter, f.byte_offset, f.binding), (2, i * 8, i));
                assert_eq!((f.binding_offset, f.extent, f.alignment), (0, b.bytes, 8));
            }
        }
        let old = AttentionDescriptor::new(source, case)
            .unwrap()
            .required_roles()
            .unwrap();
        let new = AttentionDescriptor::with_scope(source, case, AttentionScope::ComputedGate)
            .unwrap()
            .required_roles()
            .unwrap();
        assert_eq!(old[&AttentionRole::OutInput].1, DeviceAccess::Read);
        assert_eq!(new[&AttentionRole::OutInput].1, DeviceAccess::Write);
        assert_eq!(new[&AttentionRole::SigmoidGate].1, DeviceAccess::Write);
        assert_eq!(new.len(), old.len() + 1);
    }
}
#[test]
fn connected_fixture_cannot_inject_or_omit_gate_outputs() {
    use super::qualify::{
        CubinAttentionExpected as E, CubinAttentionFixture as F, CubinAttentionOutput as O,
        validate_fixtures_scoped,
    };
    let make = || {
        AttentionCase::ALL
            .into_iter()
            .map(|case| F {
                case,
                normalized: vec![0; case.rows() * 2048],
                gated: Vec::new(),
                positions: (0..3)
                    .flat_map(|_| (0..case.rows()).map(|i| (case.base() + i) as i64))
                    .collect(),
                expected: [
                    O::PackedQkv,
                    O::Q,
                    O::K,
                    O::Gate,
                    O::AttentionOutput,
                    O::SigmoidGate,
                    O::Gated,
                    O::Projected,
                    O::KvActive,
                ]
                .into_iter()
                .map(|output| E {
                    output,
                    bytes: vec![0; output.bytes(case)],
                })
                .collect(),
            })
            .collect::<Vec<_>>()
    };
    let cos = vec![0; 8576];
    assert!(validate_fixtures_scoped(&make(), &cos, AttentionScope::ComputedGate).is_ok());
    for kind in 0..5 {
        let mut f = make();
        match kind {
            0 => f[0].gated = vec![0; 4096],
            1 => {
                f[0].expected.pop();
            }
            2 => f[0].expected[5].output = O::Gated,
            3 => {
                f[0].expected[5].bytes.pop();
            }
            _ => f[0].positions[0] = 1,
        };
        assert!(validate_fixtures_scoped(&f, &cos, AttentionScope::ComputedGate).is_err());
    }
    assert!(validate_fixtures_scoped(&make(), &cos, AttentionScope::CapturedOutInput).is_err());
}
