use super::*;

#[test]
fn finite_source_windows_partition_every_vocabulary_row_once_without_padding() {
    for (layout, hidden, rows, windows) in [
        (Vllm08WeightLayout::Vocabulary08, 1024, 512, 485),
        (Vllm08WeightLayout::Vocabulary2b, 2048, 256, 970),
    ] {
        assert_eq!(geometry(layout).unwrap(), (hidden, rows));
        let mut covered = vec![false; VOCABULARY];
        for ordinal in 0..windows {
            let window = Window {
                layout,
                input_rows: 72,
                ordinal,
            };
            let range = window.output_range().unwrap();
            assert_eq!(range.start as usize, ordinal * rows * 2);
            assert_eq!(rows * hidden * 2, WINDOW_BYTES as usize);
            for row in range.start / 2..range.end / 2 {
                assert!(!covered[row as usize]);
                covered[row as usize] = true;
            }
        }
        assert!(covered.into_iter().all(|x| x));
        assert!(
            Window {
                layout,
                input_rows: 1,
                ordinal: windows
            }
            .validate()
            .is_err()
        );
    }
}
#[test]
fn last_selected_row_uses_full_real_owner_and_exact_output_window() {
    for (layout, hidden, rows) in [
        (Vllm08WeightLayout::Vocabulary08, 1024, 512),
        (Vllm08WeightLayout::Vocabulary2b, 2048, 256),
    ] {
        for input_rows in [1, 47, 57, 62, 67, 72] {
            for ordinal in [0, VOCABULARY / rows - 1] {
                let s = Window {
                    layout,
                    input_rows,
                    ordinal,
                };
                let owners = [
                    (0x1000000, WINDOW_BYTES),
                    (0x2000000, (input_rows * hidden * 2) as u64),
                    (0x3000000, OUTPUT_BYTES),
                ];
                let pointers = s.checked_addresses(&owners).unwrap();
                assert_eq!(
                    pointers,
                    [
                        owners[0].0,
                        owners[1].0 + ((input_rows - 1) * hidden * 2) as u64,
                        owners[2].0 + (ordinal * rows * 2) as u64
                    ]
                );
                for index in 0..3 {
                    let mut bad = owners;
                    bad[index].1 -= 16;
                    assert!(s.checked_addresses(&bad).is_err());
                    let mut bad = owners;
                    bad[index].0 += 1;
                    assert!(s.checked_addresses(&bad).is_err());
                }
                let mut bad = owners;
                bad[2].0 = bad[0].0;
                assert!(s.checked_addresses(&bad).is_err());
            }
        }
    }
}
#[test]
fn hostile_geometry_and_native_address_ends_refuse() {
    for rows in [0, 2, 46, 48, 64, 73, usize::MAX] {
        assert!(
            Window {
                layout: Vllm08WeightLayout::Vocabulary08,
                input_rows: rows,
                ordinal: 0
            }
            .validate()
            .is_err()
        );
    }
    assert!(geometry(Vllm08WeightLayout::GdnQkvz).is_err());
    let s = Window {
        layout: Vllm08WeightLayout::Vocabulary08,
        input_rows: 47,
        ordinal: 0,
    };
    assert!(s.checked_addresses(&[]).is_err());
    assert!(
        s.checked_addresses(&[
            (0x1000000, WINDOW_BYTES),
            (0x2000000, 47 * 2048),
            (u64::MAX - 15, OUTPUT_BYTES)
        ])
        .is_err()
    );
}
#[test]
fn only_original_n_and_grid_change_for_the_08_finite_tile() {
    let tiled = manifest(Vllm08WeightLayout::Vocabulary08).unwrap();
    let old = super::super::manifest();
    assert_eq!(old.grid, [31040, 1, 1]);
    assert_eq!(tiled.grid, [64, 1, 1]);
    for (offset, (&actual, &original)) in tiled.parameters[0]
        .template
        .iter()
        .zip(&old.parameters[0].template)
        .enumerate()
    {
        if !(68..72).contains(&offset) {
            assert_eq!(actual, original, "unexpected scalar change at{offset}");
        }
    }
    assert_eq!(tiled.parameters[1..], old.parameters[1..]);
    assert_eq!(
        tiled
            .parameters
            .iter()
            .map(|p| (p.offset, p.template.len()))
            .collect::<Vec<_>>(),
        [(0, 152), (152, 4), (156, 4)]
    );
    assert_eq!(tiled.entry, old.entry);
    assert_eq!(tiled.block, old.block);
    assert_eq!(tiled.dynamic_shared_bytes, old.dynamic_shared_bytes);
}
#[test]
fn captured_2b_packet_has_independent_width_and_permission() {
    let m = manifest(Vllm08WeightLayout::Vocabulary2b).unwrap();
    assert_eq!(
        matching_layout(&m).unwrap(),
        Vllm08WeightLayout::Vocabulary2b
    );
    assert_eq!(m.grid, [32, 1, 1]);
    assert_ne!(m, manifest(Vllm08WeightLayout::Vocabulary08).unwrap());
    for offset in [64, 72] {
        assert_eq!(
            u32::from_le_bytes(
                m.parameters[0].template[offset..offset + 4]
                    .try_into()
                    .unwrap()
            ),
            2048
        );
        let mut wrong = m.clone();
        wrong.parameters[0].template[offset..offset + 4].copy_from_slice(&1024u32.to_le_bytes());
        assert!(matching_layout(&wrong).is_err());
    }
}
#[test]
fn every_old_native_policy_stays_closed_to_streamed_windows() {
    use super::super::super::policy::CubinSetPolicy as P;
    for policy in [
        P::SourceRotary,
        P::ProjectionSix,
        P::FullQwen08,
        P::ConnectedQwen08,
        P::Canonical47Mlp,
        P::Qwen2bMlp,
        P::Canonical47Qkv,
        P::Canonical47QkKv,
        P::Canonical47Fa,
        P::Canonical47Layer3,
        P::Canonical47Full,
        P::Canonical47Natural,
        P::FirstAttentionCaptured,
        P::FirstAttentionComputedGate,
        P::VocabularyHead,
    ] {
        assert!(!policy_allowed(policy));
    }
}

#[test]
fn entire_manifest_and_all_160_literal_bytes_are_closed() {
    let m = manifest(Vllm08WeightLayout::Vocabulary08).unwrap();
    matching_layout(&m).unwrap();
    for parameter in 0..3 {
        for byte in 0..m.parameters[parameter].template.len() {
            let mut changed = m.clone();
            changed.parameters[parameter].template[byte] ^= 1;
            assert!(matching_layout(&changed).is_err());
        }
    }
    for fixup in 0..4 {
        let mut changed = m.clone();
        changed.fixups[fixup].binding_offset += 16;
        assert!(matching_layout(&changed).is_err());
        let mut changed = m.clone();
        changed.fixups[fixup].extent -= 1;
        assert!(matching_layout(&changed).is_err());
    }
    for binding in 0..3 {
        let mut changed = m.clone();
        changed.bindings[binding].bytes += 16;
        assert!(matching_layout(&changed).is_err());
    }
    let mut wrong = m.clone();
    wrong.grid[0] += 1;
    assert!(matching_layout(&wrong).is_err());
    let mut wrong = m.clone();
    wrong.block[0] /= 2;
    assert!(matching_layout(&wrong).is_err());
    let mut wrong = m;
    wrong.dynamic_shared_bytes = 2560;
    assert!(matching_layout(&wrong).is_err());
}
