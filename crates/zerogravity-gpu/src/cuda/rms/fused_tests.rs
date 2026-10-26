use super::*;

#[test]
fn only_original_observed_row_extents_have_fixed_fused_specs() -> Result<()> {
    for invalid in [0, 3, 32, 47, 63, 65, 1024, usize::MAX] {
        assert!(Rows::new(invalid).is_err());
    }
    for (count, grid) in [(1, [4, 1, 1]), (2, [8, 1, 1]), (64, [256, 1, 1])] {
        let rows = Rows::new(count)?;
        assert_eq!(
            fused_specification(Stage::InputWiden, rows, Columns::Qwen08)?.grid,
            grid
        );
    }
    Ok(())
}

#[test]
fn fused_add_binds_two_original_inputs_and_separate_output_widths() {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    for rows in [Rows::One, Rows::Two, Rows::SixtyFour] {
        let n = rows.count() * 1024;
        assert_eq!(
            add_bindings(rows, Columns::Qwen08).as_ref(),
            [(n * 2, R), (n * 2, R), (n * 4, W), (n * 2, W)]
        );
        assert_eq!(
            Buffer::stage(Stage::InputWiden),
            [
                Buffer::Attention,
                Buffer::PriorResidual,
                Buffer::SumF32,
                Buffer::ResidualBf16
            ]
        );
    }
}

#[test]
fn normalization_uses_unrounded_sum_and_keeps_original_reduction_grids() -> Result<()> {
    assert_eq!(
        Buffer::stage(Stage::Square),
        [Buffer::SumF32, Buffer::Squares]
    );
    assert_eq!(
        Buffer::stage(Stage::Normalize),
        [Buffer::SumF32, Buffer::Inverse, Buffer::NormalizedF32]
    );
    for (rows, grid) in [
        (Rows::One, [1, 1, 1]),
        (Rows::Two, [1, 1, 1]),
        (Rows::SixtyFour, [4, 1, 1]),
    ] {
        assert_eq!(
            fused_specification(Stage::Mean, rows, Columns::Qwen08)?.grid,
            grid
        );
    }
    Ok(())
}

#[test]
fn owned_budget_includes_both_outputs_and_excludes_borrowed_inputs() {
    use std::collections::BTreeSet;
    for rows in [Rows::One, Rows::Two, Rows::SixtyFour] {
        let layout = Buffer::owned(rows, Columns::Qwen08);
        assert_eq!(layout.len(), 12);
        assert!(
            layout
                .iter()
                .all(|(role, _)| !matches!(role, Buffer::Attention | Buffer::PriorResidual))
        );
        let mut roles = BTreeSet::new();
        for (role, bytes) in layout {
            assert!(bytes > 0);
            assert!(roles.insert(role as u8));
        }
        let bytes = layout.iter().map(|(_, bytes)| bytes).sum::<usize>();
        assert_eq!(bytes, 10240 + rows.count() * (20 * 1024 + 12));
        for output in [Buffer::NormalizedBf16, Buffer::ResidualBf16] {
            assert!(layout.contains(&(output, rows.count() * 2048)));
        }
    }
}

#[test]
fn dual2b_real_physical_outputs_and_unrounded_sum_use_same_stages() -> Result<()> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    for rows in [Rows::One, Rows::Two, Rows::SixtyFour] {
        let n = rows.count() * 2048;
        assert_eq!(
            add_bindings(rows, Columns::Qwen2b).as_ref(),
            [(n * 2, R), (n * 2, R), (n * 4, W), (n * 2, W)]
        );
        let layout = Buffer::owned(rows, Columns::Qwen2b);
        assert_eq!(
            layout.iter().map(|(_, bytes)| bytes).sum::<usize>(),
            20480 + rows.count() * (20 * 2048 + 12)
        );
        assert!(layout.contains(&(Buffer::SumF32, n * 4)));
        assert!(layout.contains(&(Buffer::NormalizedBf16, n * 2)));
        assert!(layout.contains(&(Buffer::ResidualBf16, n * 2)));
        assert_eq!(
            Buffer::stage(Stage::Square),
            [Buffer::SumF32, Buffer::Squares]
        );
        assert_eq!(
            Buffer::stage(Stage::Normalize),
            [Buffer::SumF32, Buffer::Inverse, Buffer::NormalizedF32]
        );
        assert_eq!(
            fused_specification(Stage::InputWiden, rows, Columns::Qwen2b)?.grid,
            [(n.div_ceil(256)) as u32, 1, 1]
        );
    }
    Ok(())
}

#[test]
fn fixed_prompt_dual_keeps_true_dense_outputs_and_same_unrounded_stages() -> Result<()> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    let rows = Rows::FixedPrompt;
    let n = 47 * 1024;
    assert_eq!(
        add_bindings(rows, Columns::Qwen08).as_ref(),
        [(n * 2, R), (n * 2, R), (n * 4, W), (n * 2, W)]
    );
    let layout = Buffer::owned(rows, Columns::Qwen08);
    assert_eq!(layout.iter().map(|(_, bytes)| bytes).sum::<usize>(), 973364);
    assert!(layout.contains(&(Buffer::SumF32, n * 4)));
    assert!(layout.contains(&(Buffer::NormalizedBf16, n * 2)));
    assert!(layout.contains(&(Buffer::ResidualBf16, n * 2)));
    assert_eq!(
        fused_specification(Stage::InputWiden, rows, Columns::Qwen08)?.grid,
        [188, 1, 1]
    );
    assert_eq!(
        fused_specification(Stage::Mean, rows, Columns::Qwen08)?.grid,
        [3, 1, 1]
    );
    assert_eq!(
        Buffer::stage(Stage::Square),
        [Buffer::SumF32, Buffer::Squares]
    );
    assert_eq!(
        Buffer::stage(Stage::Normalize),
        [Buffer::SumF32, Buffer::Inverse, Buffer::NormalizedF32]
    );
    for stage in Stage::ORDER {
        assert!(fused_specification(stage, rows, Columns::Qwen2b).is_err());
    }
    Ok(())
}

#[test]
fn all_legacy_fused_add_sources_and_layouts_are_identical() -> Result<()> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    use cubecl_runtime::kernel::KernelMetadata;
    for columns in [Columns::Qwen08, Columns::Qwen2b] {
        for rows in [Rows::One, Rows::Two, Rows::SixtyFour] {
            let n = rows.count() * columns.count();
            let original: Arc<str> = format!(
                "#define ZG_ROWS {}\n#define ZG_COLUMNS {}\n{ADD_SOURCE}",
                rows.count(),
                columns.count()
            )
            .into();
            let old = PinnedKernel::new(
                "zg_vllm_post_rms_add",
                original,
                256,
                Arc::from([(n * 2, R), (n * 2, R), (n * 4, W), (n * 2, W)]),
            )?;
            let current = fused_specification(Stage::InputWiden, rows, columns)?;
            assert_eq!(current.grid, [n.div_ceil(256) as u32, 1, 1]);
            assert_eq!(current.kernel.id(), old.id());
        }
    }
    Ok(())
}

#[test]
fn text_dual_keeps_unrounded_sum_and_exact_two_public_stores() -> Result<()> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    for count in [47, 57, 62, 67, 72] {
        let rows = Rows::Text {
            prompt_tokens: count,
        };
        for columns in [Columns::Qwen08, Columns::Qwen2b] {
            let dense = count * columns.count();
            assert_eq!(
                add_bindings(rows, columns).as_ref(),
                [
                    (dense * 2, R),
                    (dense * 2, R),
                    (dense * 4, W),
                    (dense * 2, W)
                ]
            );
            assert_eq!(
                fused_specification(Stage::InputWiden, rows, columns)?.grid,
                [dense.div_ceil(256) as u32, 1, 1]
            );
            assert_eq!(
                fused_specification(Stage::Mean, rows, columns)?.grid,
                [count.div_ceil(16) as u32, 1, 1]
            );
            assert_eq!(
                Buffer::stage(Stage::Square),
                [Buffer::SumF32, Buffer::Squares]
            );
            assert_eq!(
                Buffer::stage(Stage::Normalize),
                [Buffer::SumF32, Buffer::Inverse, Buffer::NormalizedF32]
            );
            for stage in Stage::ORDER {
                fused_specification(stage, rows, columns)?;
            }
        }
    }
    Ok(())
}
