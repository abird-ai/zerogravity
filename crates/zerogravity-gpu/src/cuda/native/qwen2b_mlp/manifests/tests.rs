use super::*;
fn word(raw: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(raw[off..off + 4].try_into().unwrap())
}
#[test]
fn finite_source_producer_consumer_extents_join_both_phases() {
    for rows in [1, 64] {
        let gate = launches(rows, 8).unwrap().remove(0).manifest;
        let act = activation(rows).unwrap();
        let down = launches(rows, 10).unwrap().remove(0).manifest;
        assert_eq!(gate.bindings[0].bytes, 50_331_648);
        assert_eq!(down.bindings[0].bytes, 25_165_824);
        assert_eq!(gate.bindings[2].bytes, act.bindings[0].bytes);
        assert_eq!(act.bindings[1].bytes, down.bindings[1].bytes);
        assert_eq!(down.bindings[2].bytes, gate.bindings[1].bytes);
        assert_eq!(gate.bindings[2].bytes, 2 * act.bindings[1].bytes);
        assert_eq!(act.block, [768, 1, 1]);
        assert_eq!(word(&act.parameters[2].template, 0), 6144);
    }
    for rows in [0, 2, 47, 65] {
        assert!(launches(rows, 8).is_err());
        assert!(launches(rows, 10).is_err());
        assert!(activation(rows).is_err());
        assert!(super::super::required_images(rows).is_err());
    }
    for index in [0, 7, 9, 11] {
        assert!(launches(1, index).is_err());
    }
}
#[test]
fn down1_main_and_reducer_share_only_three_written_bf16_planes() {
    let ds = launches(1, 10).unwrap();
    assert_eq!(ds.len(), 2);
    let main = &ds[0].manifest;
    let reduce = &ds[1].manifest;
    assert_eq!(main.grid, [8, 16, 3]);
    assert_eq!(main.dynamic_shared_bytes, 16896);
    assert_eq!(main.bindings[3].bytes, WORKSPACE_BYTES as u64);
    assert_eq!(main.fixups[2].binding, 2); // beta0 C aliases Y, no C read.
    assert_eq!(main.fixups[2].io, CubinIo::Write);
    assert_eq!(main.fixups[3].binding, 3);
    assert_eq!(main.fixups[3].extent, 12_288);
    assert_eq!(reduce.fixups[0].extent, main.fixups[3].extent);
    assert_eq!(reduce.bindings[1].bytes, WORKSPACE_BYTES as u64);
    assert_eq!(reduce.fixups[0].io, CubinIo::Read);
    assert_eq!(reduce.bindings[0].bytes, 4096);
    assert_eq!(word(&packets::P1, 0), 1);
    assert_eq!(word(&packets::P1, 8), 6144);
    assert_eq!(word(&packets::P2, 36), 2048);
    assert_eq!(word(&packets::P2, 40), 3);
    assert_eq!(word(&packets::P2, 44), 0);
    assert_eq!(reduce.parameters[0].template.len(), 60);
    assert_eq!(reduce.parameters[1].offset, 64);
    assert!(
        reduce
            .parameters
            .iter()
            .all(|p| p.offset + p.template.len() <= 60 || p.offset >= 64)
    );
    assert!(!main.bindings.iter().any(|b| b.io == CubinIo::ReadWrite));
}
#[test]
fn down64_has_no_workspace_or_reducer_and_retains_its_actual_k() {
    let ds = launches(64, 10).unwrap();
    assert_eq!(ds.len(), 1);
    let down = &ds[0].manifest;
    assert_eq!(down.grid, [16, 1, 1]);
    assert_eq!(down.dynamic_shared_bytes, 24576);
    assert_eq!(down.bindings.len(), 3);
    assert_eq!(word(&packets::P4, 112), 6144);
    assert_eq!(down.fixups.len(), 4);
    assert!(down.fixups.iter().all(|f| f.binding < 3));
    assert_eq!(
        launches(64, 8).unwrap()[0].manifest.dynamic_shared_bytes,
        81920
    );
}
#[test]
fn legacy_activation_dimensions_are_separate_and_unchanged() {
    let old = super::super::super::activation_manifest::fixed_prompt_manifest();
    let new = activation(64).unwrap();
    assert_eq!(old.block, [448, 1, 1]);
    assert_eq!(word(&old.parameters[2].template, 0), 3584);
    assert_eq!(new.block, [768, 1, 1]);
    assert_eq!(word(&new.parameters[2].template, 0), 6144);
    assert_eq!(old.entry, new.entry);
    assert_eq!(old.binary_sm, new.binary_sm);
    assert_eq!(old.parameters.len(), new.parameters.len());
}
