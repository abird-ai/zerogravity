//! Whole-call controls and extents; no original capture or numerical proof.
use super::*;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{ShortHfSpan, ShortHfWorkload},
};

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}
#[test]
fn one_prefill_has_true_chunk_indices_scratch_and_final_offsets() -> Result<()> {
    let source = SourceIdentity::for_new_allocation()?;
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        for rows in [47, 57, 62, 67, 72] {
            let workload = ShortHfWorkload::from_prompt_rows(rows)?;
            let call = ShortHfCall::new(
                profile,
                ShortHfSpan::prefill(source, workload, &vec![11; rows])?,
            )?;
            let geometry = ShortHfGdnGeometry::for_call(&call)?;
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
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        for rows in [47, 57, 62, 67, 72] {
            let workload = ShortHfWorkload::from_prompt_rows(rows)?;
            let call = ShortHfCall::new(
                profile,
                ShortHfSpan::prefill(source, workload, &vec![11; rows])?,
            )?;
            let g = ShortHfGdnGeometry::for_call(&call)?;
            for &stage in g.phase.stages() {
                assert!(kernel_with_typed_outputs(g.phase, stage).is_err());
                assert!(assets::select_qwen2b(g.phase, stage).is_err());
                {
                    let actual = select(profile, g.phase, stage)?;
                    let expected = (actual.grid, actual.source);
                    let spec = kernel_from_asset(
                        g.phase,
                        stage,
                        true,
                        actual,
                        arithmetic::GdnArithmeticFamily::ShortHf(profile),
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
                    if profile == FullModelProfile::Qwen08
                        && rows == 47
                        && stage == GdnStage::Cumsum
                    {
                        assert_ne!(
                            expected.1,
                            assets::select(VllmGdnPhase::Initial47, stage)?.source
                        );
                    }
                }
            }
        }
    }
    for profile in [FullModelProfile::Qwen08, FullModelProfile::Qwen2B] {
        for stage in [GdnStage::Conv, GdnStage::Decode, GdnStage::Gated] {
            let phase = VllmGdnPhase::CachedDecode;
            let actual = select(profile, phase, stage)?;
            let expected = actual.grid;
            let spec = kernel_from_asset(
                phase,
                stage,
                true,
                actual,
                arithmetic::GdnArithmeticFamily::ShortHf(profile),
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
        GdnFamilyIdentity::short_hf(FullModelProfile::Qwen08),
        GdnFamilyIdentity::short_hf(FullModelProfile::Qwen2B)
    );
    assert_ne!(
        GdnFamilyIdentity::short_hf(FullModelProfile::Qwen08),
        GdnFamilyIdentity::original08()
    );
    assert_ne!(
        GdnFamilyIdentity::short_hf(FullModelProfile::Qwen2B),
        GdnFamilyIdentity::original2b()
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
