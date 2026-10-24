use super::*;
use crate::vllm_cuda_input_rms::{Stage, specification_for};
use cubecl_runtime::kernel::BufferIOAttr::{ReadOnly as R, WriteOnly as W};

#[test]
fn original2b_source_pins_cannot_retag_other_width_or_opening() -> Result<()> {
    require_pins(CONFIG, INDEX, SOURCE_BYTES, CHUNK)?;
    for wrong in [
        0,
        SOURCE_BYTES - 1,
        SOURCE_BYTES + 1,
        1_746_942_600,
        u64::MAX,
    ] {
        assert!(require_pins(CONFIG, INDEX, wrong, CHUNK).is_err());
    }
    for wrong in [0, CHUNK - 1, CHUNK + 1, 4 * CHUNK, usize::MAX] {
        assert!(require_pins(CONFIG, INDEX, SOURCE_BYTES, wrong).is_err());
    }
    for wrong in [
        "",
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        INDEX,
    ] {
        assert!(require_pins(wrong, INDEX, SOURCE_BYTES, CHUNK).is_err());
    }
    for wrong in [
        "",
        "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
        CONFIG,
    ] {
        assert!(require_pins(CONFIG, wrong, SOURCE_BYTES, CHUNK).is_err());
    }
    Ok(())
}
#[test]
fn explicit_physical_width_changes_all_pointer_extents() -> Result<()> {
    for rows in [Rows::One, Rows::Two, Rows::SixtyFour] {
        let n = rows.count() * 2048;
        assert_eq!(
            Stage::GammaWiden
                .buffers_for(rows, Columns::Qwen2b)
                .as_ref(),
            [(4096, R), (8192, W)]
        );
        assert_eq!(
            Stage::InputWiden
                .buffers_for(rows, Columns::Qwen2b)
                .as_ref(),
            [(n * 2, R), (n * 4, W)]
        );
        assert_eq!(
            Stage::Mean.buffers_for(rows, Columns::Qwen2b).as_ref(),
            [(n * 4, R), (rows.count() * 4, W)]
        );
        assert_eq!(
            Stage::Normalize.buffers_for(rows, Columns::Qwen2b).as_ref(),
            [(n * 4, R), (rows.count() * 4, R), (n * 4, W)]
        );
        assert_eq!(
            Stage::Weight.buffers_for(rows, Columns::Qwen2b).as_ref(),
            [(n * 4, R), (8192, R), (n * 4, W)]
        );
        assert_eq!(
            Stage::Bf16Store.buffers_for(rows, Columns::Qwen2b).as_ref(),
            [(n * 4, R), (n * 2, W)]
        );
        assert_ne!(
            Stage::InputWiden.buffers_for(rows, Columns::Qwen2b),
            Stage::InputWiden.buffers(rows)
        );
        for stage in Stage::ORDER {
            assert!(
                specification_for(stage, rows, Columns::Qwen2b)?
                    .grid
                    .into_iter()
                    .all(|n| n > 0)
            );
        }
    }
    assert_eq!(
        specification_for(Stage::Mean, Rows::One, Columns::Qwen2b)?.grid,
        [1, 1, 1]
    );
    assert_eq!(
        specification_for(Stage::Mean, Rows::Two, Columns::Qwen2b)?.grid,
        [1, 1, 1]
    );
    assert_eq!(
        specification_for(Stage::Mean, Rows::SixtyFour, Columns::Qwen2b)?.grid,
        [4, 1, 1]
    );
    Ok(())
}
#[test]
fn captured_dual_leaf_rejects_wrong_width_nonfinite_and_bad_ids() {
    let good = Qwen2bDualRmsLeaf {
        branch: vec![0; 4096],
        residual: vec![0; 4096],
    };
    assert!(qualification::validate_fixture(&[11], &good, 1).is_ok());
    assert!(qualification::validate_fixture(&[248320], &good, 1).is_err());
    assert!(qualification::validate_fixture(&[], &good, 1).is_err());
    assert!(qualification::validate_fixture(&[11], &good, 0).is_err());
    assert!(qualification::validate_fixture(&[11], &good, usize::MAX).is_err());
    for bytes in [0, 2048, 4095, 4097, 8192] {
        let bad = Qwen2bDualRmsLeaf {
            branch: vec![0; bytes],
            residual: vec![0; 4096],
        };
        assert!(qualification::validate_fixture(&[11], &bad, 1).is_err());
    }
    for bits in [0x7f80u16, 0xff80, 0x7fc0] {
        let mut bad = Qwen2bDualRmsLeaf {
            branch: vec![0; 4096],
            residual: vec![0; 4096],
        };
        bad.branch[..2].copy_from_slice(&bits.to_le_bytes());
        assert!(qualification::validate_fixture(&[11], &bad, 1).is_err());
    }
    // No extra restriction on finite signed zero or BF16 subnormal leaves.
    let mut finite = good;
    finite.branch[..2].copy_from_slice(&0x8000u16.to_le_bytes());
    finite.residual[..2].copy_from_slice(&1u16.to_le_bytes());
    assert!(qualification::validate_fixture(&[11], &finite, 1).is_ok());
}
#[test]
fn cancellation_precedes_the_source_and_provider_frontier() {
    let flag = AtomicBool::new(true);
    assert!(not_cancelled(Some(&flag)).is_err());
    flag.store(false, Ordering::Release);
    assert!(not_cancelled(Some(&flag)).is_ok());
    assert!(not_cancelled(None).is_ok());
}
