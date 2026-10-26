//! Whole-call controls and extents; no original capture or numerical proof.
use super::*;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_model_qwen35::plan::PromptSpan;
use zerogravity_model_qwen35::plan::PromptWorkload;
use zerogravity_model_qwen35::plan::{PromptCall, QwenSourceProfile};

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}
#[test]
fn one_prefill_has_true_chunk_indices_scratch_and_final_offsets() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        for rows in [47, 57, 62, 67, 72] {
            let workload = PromptWorkload::from_prompt_tokens(rows)?;
            let call = PromptCall::new(
                profile,
                PromptSpan::prefill(source, workload, &vec![11; rows])?,
            )?;
            let geometry = TextGdnGeometry::for_call(
                &call.as_source_call().request().source_call()?,
                GdnKernelRecipe::fixture(
                    if profile == QwenSourceProfile::Qwen08 {
                        0
                    } else {
                        1
                    },
                    rows,
                    false,
                )?,
            )?;
            assert_eq!(geometry.phase.tokens(), rows);
            assert_eq!(geometry.phase.chunks(), rows.div_ceil(64));
            assert!(!geometry.phase.has_initial_state());
            assert_eq!(
                geometry.phase.stages(),
                [
                    GdnStage::Conv,
                    GdnStage::Prep,
                    GdnStage::Cumsum,
                    GdnStage::Kkt,
                    GdnStage::Solve,
                    GdnStage::Recompute,
                    GdnStage::H,
                    GdnStage::Output,
                    GdnStage::Gated
                ]
            );
            let controls = metadata(geometry.phase, geometry.slot)?
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(i32s(&controls[&GdnBuffer::QueryStarts]), [0, rows as i32]);
            assert_eq!(
                i32s(&controls[&GdnBuffer::ChunkIndices]),
                if rows <= 64 {
                    vec![0, 0]
                } else {
                    vec![0, 0, 0, 1]
                }
            );
            let offsets = controls[&GdnBuffer::ChunkOffsets]
                .chunks_exact(8)
                .map(|w| i64::from_le_bytes(w.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(offsets, [0, rows.div_ceil(64) as i64]);
            assert_eq!(controls[&GdnBuffer::HasInitial], [0]);
            let batch = i32s(&controls[&GdnBuffer::ConvBatch]);
            let offsets = i32s(&controls[&GdnBuffer::ConvChunkOffsets]);
            let conv = rows.div_ceil(8);
            assert_eq!(&batch[..conv], vec![0; conv]);
            assert!(batch[conv..].iter().all(|x| *x == -1));
            assert_eq!(&offsets[..conv], (0..conv as i32).collect::<Vec<_>>());
            assert!(offsets[conv..].iter().all(|x| *x == -1));
            assert_eq!(
                GdnBuffer::HChunks.bytes(geometry.phase),
                rows.div_ceil(64) * 524288
            );
            assert_eq!(GdnBuffer::Kkt.bytes(geometry.phase), rows * 16 * 64 * 4);
            assert_eq!(GdnBuffer::Inverse.bytes(geometry.phase), rows * 16 * 64 * 2);
            assert_eq!(GdnBuffer::FinalState.bytes(geometry.phase), 1048576);
            for (&role, bytes) in &controls {
                assert_eq!(role.bytes(geometry.phase), bytes.len());
            }
            assert!(metadata(geometry.phase, GdnCacheSlot::Four).is_err());
        }
    }
    Ok(())
}
#[test]
fn observed_source_tuples_are_distinct_and_never_open_legacy_factories() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        for rows in [47, 57, 62, 67, 72] {
            let workload = PromptWorkload::from_prompt_tokens(rows)?;
            let call = PromptCall::new(
                profile,
                PromptSpan::prefill(source, workload, &vec![11; rows])?,
            )?;
            let g = TextGdnGeometry::for_call(
                &call.as_source_call().request().source_call()?,
                GdnKernelRecipe::fixture(
                    if profile == QwenSourceProfile::Qwen08 {
                        0
                    } else {
                        1
                    },
                    rows,
                    false,
                )?,
            )?;
            for &stage in g.phase.stages() {
                assert!(kernel_with_typed_outputs(g.phase, stage).is_err());
                assert!(assets::select_qwen2b(g.phase, stage).is_err());
                {
                    let actual = select(g.recipe, g.phase, stage)?;
                    let expected = (actual.grid, actual.source);
                    let spec = kernel_from_asset(
                        g.phase,
                        stage,
                        true,
                        actual,
                        arithmetic::GdnArithmeticFamily::Text(g.recipe),
                    )?;
                    assert_eq!(spec.grid, expected.0);
                    assert_eq!(
                        spec.bindings.len(),
                        match stage {
                            GdnStage::Conv => 9,
                            GdnStage::Prep => 10,
                            GdnStage::Cumsum | GdnStage::Solve => 4,
                            GdnStage::Kkt => 6,
                            GdnStage::Recompute => 9,
                            GdnStage::H => 10,
                            GdnStage::Output => 8,
                            GdnStage::Gated => 5,
                            _ => unreachable!(),
                        }
                    );
                    if profile == QwenSourceProfile::Qwen08
                        && rows == 47
                        && stage == GdnStage::Cumsum
                    {
                        assert_ne!(
                            expected.1,
                            assets::select(GdnPhase::FixedPrompt, stage)?.source
                        );
                    }
                }
            }
        }
    }
    for profile in [QwenSourceProfile::Qwen08, QwenSourceProfile::Qwen2B] {
        for stage in [GdnStage::Conv, GdnStage::Decode, GdnStage::Gated] {
            let phase = GdnPhase::CachedDecode;
            let recipe = GdnKernelRecipe::fixture(
                if profile == QwenSourceProfile::Qwen08 {
                    0
                } else {
                    1
                },
                1,
                true,
            )?;
            let actual = select(recipe, phase, stage)?;
            let expected = actual.grid;
            let spec = kernel_from_asset(
                phase,
                stage,
                true,
                actual,
                arithmetic::GdnArithmeticFamily::Text(recipe),
            )?;
            assert_eq!(
                spec.bindings.len(),
                match stage {
                    GdnStage::Conv | GdnStage::Gated => 5,
                    _ => 9,
                }
            );
            assert_eq!(spec.grid, expected);
        }
    }
    assert_ne!(
        GdnFamilyIdentity::text(GdnKernelRecipe::fixture(0, 47, false)?),
        GdnFamilyIdentity::text(GdnKernelRecipe::fixture(1, 47, false)?)
    );
    assert_ne!(
        GdnFamilyIdentity::text(GdnKernelRecipe::fixture(0, 47, false)?),
        GdnFamilyIdentity::original08()
    );
    assert_ne!(
        GdnFamilyIdentity::text(GdnKernelRecipe::fixture(1, 47, false)?),
        GdnFamilyIdentity::original2b()
    );
    Ok(())
}

#[test]
fn parameterized_conv_uses_source_extents_dynamic_grid_and_one_packed_abi() -> Result<()> {
    use zerogravity_core::accelerator::program::SourceConvGeometry;

    let geometry = SourceConvGeometry {
        qkv_channels: 6144,
        z_channels: 2048,
        coefficient_channels: 32,
        kernel: 4,
    };
    let mut code_identity = None;
    for rows in [47, 48, 49, 50, 57, 62, 67, 72] {
        let definition =
            parameterized_conv::prepare(GdnPhase::TextPrefill(rows), GdnCacheSlot::One, geometry)?;
        assert_eq!(definition.asset.entry, parameterized_conv::ENTRY);
        assert_eq!(definition.asset.source, parameterized_conv::SOURCE);
        assert_eq!(definition.asset.grid, [rows.div_ceil(8) as u32, 24, 1]);
        assert_eq!(definition.scalars.words(), [5, 8192, 6144]);
        assert_eq!(definition.scalars.schema().byte_size(), 24);
        assert_eq!(
            definition.scalars.packed_bytes(),
            [
                5_u64.to_le_bytes(),
                8192_u64.to_le_bytes(),
                6144_u64.to_le_bytes(),
            ]
            .concat()
        );
        code_identity.get_or_insert((definition.asset.entry, definition.asset.source));
        assert_eq!(
            code_identity,
            Some((definition.asset.entry, definition.asset.source))
        );
    }
    assert!(parameterized_conv::SOURCE.contains("ZgGdnConvScalarWords zg_scalars"));
    assert!(
        parameterized_conv::SOURCE
            .contains("\"r\"(static_cast<unsigned int>(zg_scalars.words[0]))")
    );
    assert!(parameterized_conv::SOURCE.contains("\"l\"(zg_scalars.words[1])"));
    assert!(parameterized_conv::SOURCE.contains("\"l\"(zg_scalars.words[2])"));
    assert!(!parameterized_conv::SOURCE.contains("zg_scalar9"));
    assert!(!parameterized_conv::SOURCE.contains("zg_vllm_a905e193ffad4e19"));
    assert!(!parameterized_conv::SOURCE.contains(" 492;"));
    assert!(!parameterized_conv::SOURCE.contains(" 157;"));
    Ok(())
}

#[test]
fn parameterized_conv_scalar_values_are_part_of_arithmetic_not_code_identity() -> Result<()> {
    use crate::cuda::kernel::{StaticScalarBlock, StaticScalarValue};
    use cubecl_runtime::kernel::KernelMetadata;
    use zerogravity_core::accelerator::program::SourceConvGeometry;

    let phase = GdnPhase::TextPrefill(47);
    let geometry = SourceConvGeometry {
        qkv_channels: 6144,
        z_channels: 2048,
        coefficient_channels: 32,
        kernel: 4,
    };
    let recipe = GdnKernelRecipe::fixture(0, 47, false)?;
    let definition = parameterized_conv::prepare(phase, GdnCacheSlot::One, geometry)?;
    let changed = StaticScalarBlock::new(
        definition.scalars.schema(),
        &[
            StaticScalarValue::U32(4),
            StaticScalarValue::U64(8192),
            StaticScalarValue::U64(6144),
        ],
    )?;
    let actual = kernel_from_asset_with_scalars(
        phase,
        GdnStage::Conv,
        true,
        definition.asset,
        arithmetic::GdnArithmeticFamily::Text(recipe),
        definition.scalars,
    )?;
    let changed_definition = parameterized_conv::prepare(phase, GdnCacheSlot::One, geometry)?;
    let changed = kernel_from_asset_with_scalars(
        phase,
        GdnStage::Conv,
        true,
        changed_definition.asset,
        arithmetic::GdnArithmeticFamily::Text(recipe),
        changed,
    )?;
    assert_eq!(actual.kernel.id(), changed.kernel.id());
    assert_ne!(actual.arithmetic, changed.arithmetic);
    assert_ne!(actual.bindings, changed.bindings);
    Ok(())
}

#[test]
fn parameterized_conv_refuses_wrong_source_layout_controls_and_overflow() -> Result<()> {
    use zerogravity_core::accelerator::program::SourceConvGeometry;

    let phase = GdnPhase::TextPrefill(48);
    let geometry = SourceConvGeometry {
        qkv_channels: 6144,
        z_channels: 2048,
        coefficient_channels: 32,
        kernel: 4,
    };
    let retained = parameterized_conv::ConvLayout {
        cache_arena_bytes: GdnBuffer::CacheArena.bytes(phase),
        input_bytes: GdnBuffer::PackedQkvz.bytes(phase),
        output_bytes: GdnBuffer::ConvOutput.bytes(phase),
        conv_batch_bytes: GdnBuffer::ConvBatch.bytes(phase),
        conv_chunk_offsets_bytes: GdnBuffer::ConvChunkOffsets.bytes(phase),
        cache_indices_bytes: GdnBuffer::CacheIndices.bytes(phase),
        has_initial_bytes: GdnBuffer::HasInitial.bytes(phase),
        query_starts_bytes: GdnBuffer::QueryStarts.bytes(phase),
    };
    let mut bad = retained;
    bad.cache_arena_bytes -= GdnBuffer::CacheArena.bytes(phase) / CACHE_SLOTS;
    assert!(
        parameterized_conv::prepare_with_layout(phase, GdnCacheSlot::One, geometry, bad).is_err()
    );
    bad = retained;
    bad.input_bytes -= 2;
    assert!(
        parameterized_conv::prepare_with_layout(phase, GdnCacheSlot::One, geometry, bad).is_err()
    );
    bad = retained;
    bad.output_bytes += 2;
    assert!(
        parameterized_conv::prepare_with_layout(phase, GdnCacheSlot::One, geometry, bad).is_err()
    );
    bad = retained;
    bad.conv_batch_bytes = 4;
    bad.conv_chunk_offsets_bytes = 4;
    assert!(
        parameterized_conv::prepare_with_layout(phase, GdnCacheSlot::One, geometry, bad).is_err()
    );
    let mut wrong_geometry = geometry;
    wrong_geometry.z_channels = 2047;
    assert!(
        parameterized_conv::prepare_with_layout(
            phase,
            GdnCacheSlot::One,
            wrong_geometry,
            retained,
        )
        .is_err()
    );
    assert!(
        parameterized_conv::prepare_with_layout(
            GdnPhase::TextPrefill(usize::MAX),
            GdnCacheSlot::One,
            geometry,
            retained,
        )
        .is_err()
    );
    assert!(
        parameterized_conv::prepare_with_layout(
            GdnPhase::CachedDecode,
            GdnCacheSlot::One,
            geometry,
            retained,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn pooled_private_lease_ownership_never_implies_a_retained_layer_observation() {
    for role in [
        GdnBuffer::Q,
        GdnBuffer::K,
        GdnBuffer::V,
        GdnBuffer::GRaw,
        GdnBuffer::Beta,
        GdnBuffer::GCumulative,
        GdnBuffer::Kkt,
        GdnBuffer::Inverse,
        GdnBuffer::W,
        GdnBuffer::U,
        GdnBuffer::HChunks,
        GdnBuffer::InitialState,
        GdnBuffer::FinalState,
        GdnBuffer::VNew,
        GdnBuffer::Core,
        GdnBuffer::AContiguous,
        GdnBuffer::BContiguous,
        GdnBuffer::CacheArena,
    ] {
        assert!(!retained_history(role), "{role:?}");
    }
    for role in [
        GdnBuffer::PackedQkvz,
        GdnBuffer::PackedBa,
        GdnBuffer::ConvOutput,
        GdnBuffer::ZCompact,
        GdnBuffer::Gated,
        GdnBuffer::Rstd,
        GdnBuffer::ConvWeights,
        GdnBuffer::GatedGamma,
        GdnBuffer::QueryStarts,
        GdnBuffer::ChunkIndices,
        GdnBuffer::ChunkOffsets,
    ] {
        assert!(retained_history(role), "{role:?}");
    }
}

#[test]
fn exact_reference_fingerprint_refuses_changed_count_phase_or_unregistered_recipe() -> Result<()> {
    let recipe = GdnKernelRecipe::fixture(0, 47, false)?;
    assert!(GdnKernelRecipe::bind(recipe.recipe_sha256(), 48, false).is_err());
    assert!(GdnKernelRecipe::bind(recipe.recipe_sha256(), 47, true).is_err());
    assert!(GdnKernelRecipe::bind("changed source tuple", 47, false).is_err());
    assert!(GdnKernelRecipe::fixture(0, 48, false).is_err());
    assert!(GdnKernelRecipe::fixture(0, 49, false).is_err());
    assert!(GdnKernelRecipe::fixture(0, 50, false).is_err());
    assert!(GdnKernelRecipe::fixture(0, 47, true).is_err());
    Ok(())
}
