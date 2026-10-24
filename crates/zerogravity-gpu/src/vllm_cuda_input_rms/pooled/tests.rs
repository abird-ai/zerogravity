//! Pure storage/order fixtures; no model payload, client or constructor forgery.
use super::*;

#[test]
fn pooling_uses_only_finite_source_row_and_width_specifications() {
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        assert_eq!(declared_rows(1, profile).unwrap(), Rows::One);
        for rows in [47, 57, 62, 67, 72] {
            let value = declared_rows(rows, profile).unwrap();
            assert_eq!(value.count(), rows);
            assert!(Rows::new(rows).is_err());
        }
        for rows in [0, 2, 46, 48, 56, 58, 61, 63, 64, 66, 68, 71, 73, usize::MAX] {
            assert!(matches!(
                declared_rows(rows, profile),
                Err(Error::Unsupported(_))
            ));
        }
    }
    assert_eq!(
        declared_rows(47, FullModelProfile::Qwen08).unwrap(),
        Rows::FortySeven
    );
}

#[test]
fn private_storage_has_nine_distinct_roles_and_checked_extents() {
    let roles = ScratchRole::ALL
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(roles.len(), 9);
    for columns in [1024, 2048] {
        for rows in [1, 47, 57, 62, 67, 72] {
            let bytes = ScratchRole::ALL
                .into_iter()
                .map(|role| role.bytes(rows, columns).unwrap())
                .sum::<usize>();
            assert_eq!(bytes, 8 * columns + 16 * rows * columns + 12 * rows);
        }
    }
    assert!(ScratchRole::InputF32.bytes(usize::MAX, 2048).is_err());
    assert!(ScratchRole::Weighted.bytes(72, usize::MAX).is_err());
}

#[test]
fn actual_actor_order_refuses_replay_skips_concurrent_borrow_and_partial_finish() {
    let nodes = [1, 7, 11, 253];
    let mut cursor = OrderedActors::default();
    assert!(cursor.require_finished(nodes.len()).is_err());
    assert!(cursor.begin(&nodes, 7).is_err());
    cursor.begin(&nodes, 1).unwrap();
    assert!(cursor.begin(&nodes, 1).is_err());
    assert!(cursor.begin(&nodes, 7).is_err());
    assert!(cursor.require_current(&nodes, 7).is_err());
    assert!(cursor.require_finished(nodes.len()).is_err());
    cursor.require_current(&nodes, 1).unwrap();
    cursor.completed();
    assert!(cursor.begin(&nodes, 1).is_err());
    for node in &nodes[1..] {
        cursor.begin(&nodes, *node).unwrap();
        cursor.require_current(&nodes, *node).unwrap();
        cursor.completed();
    }
    cursor.require_finished(nodes.len()).unwrap();
    assert!(cursor.begin(&nodes, 253).is_err());
}

#[test]
fn failed_actor_cannot_release_private_epoch_as_success_or_resume() {
    let nodes = [1, 7];
    let mut cursor = OrderedActors::default();
    cursor.begin(&nodes, 1).unwrap();
    cursor.fail();
    assert!(cursor.require_finished(0).is_err());
    assert!(cursor.begin(&nodes, 1).is_err());
    assert!(cursor.begin(&nodes, 7).is_err());
}

#[test]
fn per_actor_alias_refusal_uses_real_distinct_ids_from_the_same_ledger() {
    use zerogravity_core::device_events::DeviceEventLedger;
    let mut ledger = DeviceEventLedger::new(1, 14 * 64).unwrap();
    let ids = (0..14)
        .map(|_| ledger.admit(64, BufferRole::Scratch).unwrap())
        .collect::<Vec<_>>();
    layout::require_distinct(ids.iter().copied()).unwrap();
    let mut alias = ids.clone();
    alias[13] = alias[0];
    assert!(layout::require_distinct(alias).is_err());
    layout::require_distinct(ids).unwrap();
}

#[test]
fn cold_actor_inventory_uses_actual_allocated_role_layouts() -> Result<()> {
    use crate::vllm_cuda_input_rms::{FusedRmsPrepared, RmsPrepared};
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        let h = profile.hidden_size();
        for rows in [1, 47, 57, 62, 67, 72] {
            let shared = FullModelRmsScratch::allocation_layout(rows, profile)?;
            assert_eq!(shared.len(), 9);
            assert_eq!(
                shared.iter().sum::<usize>(),
                8 * h + 16 * rows * h + 12 * rows
            );
            assert_eq!(
                RmsPrepared::short_hf_private_layout(rows, profile)?,
                [2 * h]
            );
            assert_eq!(
                FusedRmsPrepared::short_hf_private_layout(rows, profile)?,
                [2 * h, 4 * rows * h]
            );
        }
        for rows in [0, 2, 46, 48, 64, 73, usize::MAX] {
            assert!(FullModelRmsScratch::allocation_layout(rows, profile).is_err());
            assert!(RmsPrepared::short_hf_private_layout(rows, profile).is_err());
            assert!(FusedRmsPrepared::short_hf_private_layout(rows, profile).is_err());
        }
    }
    Ok(())
}
