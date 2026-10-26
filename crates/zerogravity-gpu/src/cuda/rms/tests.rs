use super::*;

#[test]
fn only_observed_rms_geometries_are_constructible() {
    for value in [0, 3, 16, 47, 63, 65, 1024, usize::MAX] {
        assert!(Rows::new(value).is_err());
    }
    for value in [1, 2, 64] {
        assert_eq!(Rows::new(value).unwrap().count(), value);
    }
}

#[test]
fn source_mean_block_geometry_and_physical_stores_remain_distinct() -> Result<()> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    for (rows, block, grid) in [
        (Rows::One, (256, 1), [1, 1, 1]),
        (Rows::Two, (256, 2), [1, 1, 1]),
        (Rows::SixtyFour, (32, 16), [4, 1, 1]),
    ] {
        assert_eq!(rows.mean_block(), block);
        assert_eq!(specification(Stage::Mean, rows)?.grid, grid);
        let elements = rows.count() * 1024;
        assert_eq!(
            Stage::Mean.buffers(rows).as_ref(),
            [(elements * 4, R), (rows.count() * 4, W)]
        );
        assert_eq!(
            Stage::Normalize.buffers(rows).as_ref(),
            [(elements * 4, R), (rows.count() * 4, R), (elements * 4, W)]
        );
        assert_eq!(
            Stage::Weight.buffers(rows).as_ref(),
            [(elements * 4, R), (4096, R), (elements * 4, W)]
        );
        assert_eq!(
            Stage::Bf16Store.buffers(rows).as_ref(),
            [(elements * 4, R), (elements * 2, W)]
        );
        for stage in Stage::ORDER {
            assert!(specification(stage, rows)?.grid.into_iter().all(|n| n > 0));
        }
    }
    Ok(())
}

#[test]
fn fixed_prompt_dense_extents_and_masked_final_warp_are_exact() -> Result<()> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    use std::collections::BTreeSet;
    let rows = Rows::FixedPrompt;
    assert!(Rows::new(47).is_err());
    assert_eq!(rows.mean_block(), (32, 16));
    assert_eq!(specification(Stage::Mean, rows)?.grid, [3, 1, 1]);
    assert_eq!(
        Stage::Mean.buffers(rows).as_ref(),
        [(47 * 1024 * 4, R), (47 * 4, W)]
    );
    assert_eq!(
        Stage::Bf16Store.buffers(rows).as_ref(),
        [(47 * 1024 * 4, R), (47 * 1024 * 2, W)]
    );
    let mut loads = BTreeSet::new();
    let mut stores = BTreeSet::new();
    let mut invalid_threads = 0;
    let mut participants = 0;
    for block in 0..3 {
        for thread in 0..512 {
            participants += 1;
            let x = thread % 32;
            let row = block * 16 + thread / 32;
            if row < 47 {
                for vector in (x..256).step_by(32) {
                    for lane in 0..4 {
                        assert!(loads.insert(row * 1024 + vector * 4 + lane));
                    }
                }
                if x == 0 {
                    assert!(stores.insert(row));
                }
            } else {
                assert_eq!(row, 47);
                invalid_threads += 1;
            }
        }
    }
    assert_eq!(participants, 3 * 512);
    assert_eq!(invalid_threads, 32);
    assert_eq!(loads.len(), 47 * 1024);
    assert_eq!(loads.last(), Some(&(47 * 1024 - 1)));
    assert_eq!(stores, (0..47).collect());
    for stage in Stage::ORDER {
        assert!(specification_for(stage, rows, Columns::Qwen2b).is_err());
    }
    Ok(())
}

#[test]
fn all_legacy_rendered_sources_layouts_and_kernel_ids_are_identical() -> Result<()> {
    use cubecl_runtime::kernel::KernelMetadata;
    for columns in [Columns::Qwen08, Columns::Qwen2b] {
        for rows in [Rows::One, Rows::Two, Rows::SixtyFour] {
            let count = columns.count();
            let n = rows.count();
            let factor = if columns == Columns::Qwen08 {
                "0x1p-10f"
            } else {
                "0x1p-11f"
            };
            let (width, height) = match (count, n) {
                (1024, 1) => (256u32, 1u32),
                (2048, 1) => (512, 1),
                (_, 2) => (256, 2),
                (_, 64) => (32, 16),
                _ => unreachable!(),
            };
            let original: Arc<str> = format!(
                "#define ZG_ROWS {n}\n#define ZG_COLUMNS {count}\n#define ZG_MEAN_FACTOR {factor}\n#define ZG_MEAN_WIDTH {width}\n#define ZG_MEAN_HEIGHT {height}\n{SOURCE}",
            ).into();
            assert_eq!(rendered_source(rows, columns)?.as_ref(), original.as_ref());
            assert!(!original.contains("row < ZG_ROWS"));
            for stage in Stage::ORDER {
                let words = match stage {
                    Stage::GammaWiden | Stage::GammaShift => count,
                    Stage::Epsilon | Stage::Inverse => n,
                    _ => n * count,
                };
                let (threads, grid) = if stage == Stage::Mean {
                    (width * height, [n as u32 / height, 1, 1])
                } else {
                    (256, [words.div_ceil(256) as u32, 1, 1])
                };
                use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
                let dense_f32 = n * count * 4;
                let dense_bf16 = n * count * 2;
                let scalar_f32 = n * 4;
                let bindings: Arc<[(usize, BufferIOAttr)]> = match stage {
                    Stage::GammaWiden => Arc::from([(count * 2, R), (count * 4, W)]),
                    Stage::GammaShift => Arc::from([(count * 4, R), (count * 4, W)]),
                    Stage::InputWiden => Arc::from([(dense_bf16, R), (dense_f32, W)]),
                    Stage::Square => Arc::from([(dense_f32, R), (dense_f32, W)]),
                    Stage::Mean => Arc::from([(dense_f32, R), (scalar_f32, W)]),
                    Stage::Epsilon | Stage::Inverse => {
                        Arc::from([(scalar_f32, R), (scalar_f32, W)])
                    }
                    Stage::Normalize => {
                        Arc::from([(dense_f32, R), (scalar_f32, R), (dense_f32, W)])
                    }
                    Stage::Weight => Arc::from([(dense_f32, R), (count * 4, R), (dense_f32, W)]),
                    Stage::Bf16Store => Arc::from([(dense_f32, R), (dense_bf16, W)]),
                };
                assert_eq!(stage.buffers_for(rows, columns).as_ref(), bindings.as_ref());
                let old = PinnedKernel::new(stage.entry(), original.clone(), threads, bindings)?;
                let current = specification_for(stage, rows, columns)?;
                assert_eq!(current.grid, grid);
                assert_eq!(current.kernel.id(), old.id());
            }
        }
    }
    Ok(())
}

#[test]
fn text_mean_visits_true_rows_once_and_masks_only_invalid_tail() -> Result<()> {
    use std::collections::BTreeSet;
    for count in [47, 57, 62, 67, 72] {
        assert!(Rows::new(count).is_err());
        let rows = Rows::Text {
            prompt_tokens: count,
        };
        assert_eq!(rows.count(), count);
        for columns in [Columns::Qwen08, Columns::Qwen2b] {
            let width = columns.count();
            let (block_x, block_y) = columns.mean_block(rows);
            assert_eq!((block_x, block_y), (32, 16));
            let grid = count.div_ceil(16);
            assert_eq!(
                specification_for(Stage::Mean, rows, columns)?.grid,
                [grid as u32, 1, 1]
            );
            let source = rendered_source(rows, columns)?;
            assert_eq!(source.matches("if (row < ZG_ROWS)").count(), 1);
            assert_eq!(source.matches("row < ZG_ROWS && x == 0").count(), 1);
            assert_eq!(
                source.matches("__syncthreads()").count(),
                SOURCE.matches("__syncthreads()").count()
            );
            assert!(
                source.contains("__fadd_rn(__fadd_rn(__fadd_rn(accum0, accum1), accum2), accum3)")
            );
            assert!(source.contains("__shfl_down_sync(0xffffffffu, value, offset)"));
            let mut loads = BTreeSet::new();
            let mut stores = BTreeSet::new();
            let mut invalid = 0;
            for b in 0..grid {
                for t in 0..512 {
                    let x = t % 32;
                    let row = 16 * b + t / 32;
                    if row < count {
                        for vector in (x..width / 4).step_by(32) {
                            for lane in 0..4 {
                                assert!(loads.insert(row * width + vector * 4 + lane));
                            }
                        }
                        if x == 0 {
                            assert!(stores.insert(row));
                        }
                    } else {
                        invalid += 1;
                    }
                }
            }
            assert_eq!(loads.len(), count * width);
            assert_eq!(loads.first(), Some(&0));
            assert_eq!(loads.last(), Some(&(count * width - 1)));
            assert_eq!(stores, (0..count).collect());
            assert_eq!(invalid, (grid * 16 - count) * 32);
            // The y/global split predicates remain false for both source widths.
            let values = width / 32;
            assert!(values < 256);
            for stage in Stage::ORDER {
                let buffers = stage.buffers_for(rows, columns);
                assert!(buffers.iter().all(|(bytes, _)| *bytes > 0));
                specification_for(stage, rows, columns)?;
            }
        }
    }
    Ok(())
}
