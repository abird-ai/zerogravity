use super::*;
use cubecl_runtime::kernel::KernelMetadata;

#[test]
fn every_closed_stage_has_exact_extent_access_and_source_identity() {
    for phase in [
        GdnPhase::InitialOne,
        GdnPhase::Initial64,
        GdnPhase::ContinuedTwo,
        GdnPhase::CachedDecode,
    ] {
        for &stage in phase.stages() {
            let spec = kernel(phase, stage).unwrap();
            assert!(!spec.bindings.is_empty());
            assert!(spec.grid.into_iter().all(|n| n > 0));
            for binding in spec.bindings {
                assert!(binding.physical_bytes > 0);
                assert!(binding.offset + binding.physical_bytes <= binding.role.bytes(phase));
                assert_ne!(binding.io, BufferIOAttr::Dead);
            }
        }
    }
    assert!(kernel(GdnPhase::Initial64, GdnStage::Decode).is_err());
    assert!(kernel(GdnPhase::CachedDecode, GdnStage::Prep).is_err());
    assert_ne!(
        kernel(GdnPhase::InitialOne, GdnStage::Cumsum)
            .unwrap()
            .kernel
            .id(),
        kernel(GdnPhase::Initial64, GdnStage::Cumsum)
            .unwrap()
            .kernel
            .id()
    );
    assert_ne!(
        kernel(GdnPhase::InitialOne, GdnStage::Gated)
            .unwrap()
            .kernel
            .id(),
        kernel(GdnPhase::ContinuedTwo, GdnStage::Gated)
            .unwrap()
            .kernel
            .id()
    );
    let h = assets::select(GdnPhase::Initial64, GdnStage::H).unwrap();
    assert_eq!(h.shared, 49412);
    assert_eq!(h.threads, 128);
    assert_eq!(h.grid, [2, 16, 1]);
    assert_eq!(
        h.ptx_sha256,
        "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c"
    );
}

#[test]
fn actual_aliases_and_intermediate_producer_roles_are_explicit() {
    use GdnBuffer::*;
    let decode = kernel(GdnPhase::CachedDecode, GdnStage::Conv).unwrap();
    assert_eq!(decode.bindings[0], decode.bindings[4]);
    assert_eq!(decode.bindings[0].role, PackedQkvz);
    let decode = kernel(GdnPhase::CachedDecode, GdnStage::Decode).unwrap();
    assert_eq!(decode.bindings[6], decode.bindings[7]);
    assert_eq!(decode.bindings[6].offset, CONV_STATE_BYTES);
    assert_eq!(
        decode.bindings[6].physical_bytes,
        CacheArena.bytes(GdnPhase::CachedDecode) - CONV_STATE_BYTES
    );
    assert_eq!(
        (decode.bindings[1].role, decode.bindings[1].offset),
        (PackedBa, 32)
    );
    assert_eq!(
        (decode.bindings[2].role, decode.bindings[2].offset),
        (PackedBa, 0)
    );
    let gated = kernel(GdnPhase::InitialOne, GdnStage::Gated).unwrap();
    assert_eq!(
        (gated.bindings[0].role, gated.bindings[1].role),
        (Core, Gated)
    );
    assert_eq!(
        (
            gated.bindings[3].role,
            gated.bindings[3].offset,
            gated.bindings[3].physical_bytes
        ),
        (PackedQkvz, 12288, 4096)
    );
    for phase in [GdnPhase::Initial64, GdnPhase::ContinuedTwo] {
        let prep = kernel(phase, GdnStage::Prep).unwrap();
        assert_eq!(
            (prep.bindings[1].role, prep.bindings[2].role),
            (AContiguous, BContiguous)
        );
        let h = kernel(phase, GdnStage::H).unwrap();
        assert_eq!(
            (h.bindings[1].role, h.bindings[2].role, h.bindings[3].role),
            (U, W, VNew)
        );
        assert_eq!(
            (h.bindings[6].role, h.bindings[7].role),
            (InitialState, FinalState)
        );
        let output = kernel(phase, GdnStage::Output).unwrap();
        assert_eq!(output.bindings[2].role, VNew);
        let gated = kernel(phase, GdnStage::Gated).unwrap();
        assert_eq!(gated.bindings[3].role, ZCompact);
    }
}

#[test]
fn private_cache_preserves_pitch_and_only_nonnull_actual_indices() {
    assert_eq!(CACHE_SLOT_BYTES, 1114112);
    assert_eq!(GdnBuffer::CacheArena.bytes(GdnPhase::Initial64), 5570560);
    for index in [0, 2, 3, 5, u32::MAX] {
        assert!(GdnCacheSlot::new(index).is_err());
    }
    for (phase, slot) in [
        (GdnPhase::InitialOne, GdnCacheSlot::One),
        (GdnPhase::Initial64, GdnCacheSlot::Four),
        (GdnPhase::ContinuedTwo, GdnCacheSlot::Four),
        (GdnPhase::CachedDecode, GdnCacheSlot::Four),
    ] {
        assert!(
            slot.recurrent_offset() + RECURRENT_STATE_BYTES <= GdnBuffer::CacheArena.bytes(phase)
        );
        let metadata = metadata(phase, slot).unwrap();
        for (role, bytes) in &metadata {
            assert_eq!(bytes.len(), role.bytes(phase));
        }
        assert_eq!(metadata[0].1, (slot.index() as i32).to_le_bytes());
        if phase != GdnPhase::CachedDecode {
            let field = |role| {
                &metadata
                    .iter()
                    .find(|(actual, _)| *actual == role)
                    .unwrap()
                    .1
            };
            assert_eq!(
                field(GdnBuffer::HasInitial),
                &[u8::from(phase.has_initial_state())]
            );
            assert_eq!(
                &field(GdnBuffer::QueryStarts)[4..8],
                &(phase.tokens() as i32).to_le_bytes()
            );
            let offsets = field(GdnBuffer::ConvChunkOffsets);
            let chunks = phase.tokens().div_ceil(8);
            assert_eq!(
                &offsets[(chunks - 1) * 4..chunks * 4],
                &((chunks - 1) as i32).to_le_bytes()
            );
            assert_eq!(
                &offsets[chunks * 4..chunks * 4 + 4],
                &(-1_i32).to_le_bytes()
            );
        }
    }
    assert!(metadata(GdnPhase::Initial64, GdnCacheSlot::One).is_err());
    assert!(metadata(GdnPhase::InitialOne, GdnCacheSlot::Four).is_err());
    let required = required_buffers(GdnPhase::CachedDecode).unwrap();
    assert!(!required.iter().any(|(role, _)| *role == GdnBuffer::HChunks));
    assert_eq!(
        required
            .iter()
            .filter(|(role, _)| *role == GdnBuffer::CacheArena)
            .count(),
        1
    );
}

// Append to the existing gdn/tests.rs; pure source metadata only.
#[test]
fn cached_typed_bindings_are_actual_public_outputs_and_legacy_is_unchanged() {
    let phase = GdnPhase::CachedDecode;
    for stage in [GdnStage::Conv, GdnStage::Decode, GdnStage::Gated] {
        let old = kernel(phase, stage).unwrap();
        let typed = kernel_with_typed_outputs(phase, stage).unwrap();
        assert_eq!(old.grid, typed.grid);
        assert_eq!(old.bindings.len(), typed.bindings.len());
        for (original, actual) in old.bindings.iter().zip(&typed.bindings) {
            if original.role != GdnBuffer::PackedQkvz {
                assert_eq!(original, actual);
                continue;
            }
            assert_eq!(original.io, actual.io);
            assert_eq!(actual.offset, 0);
            if stage == GdnStage::Gated {
                assert_eq!(original.offset, 12_288);
                assert_eq!(actual.role, GdnBuffer::ZCompact);
                assert_eq!(actual.physical_bytes, 4_096);
            } else {
                assert_eq!(original.physical_bytes, 16_384);
                assert_eq!(actual.role, GdnBuffer::ConvOutput);
                assert_eq!(actual.physical_bytes, 12_288);
            }
        }
    }
}
#[test]
fn selected_cached_ptx_thread_addresses_exactly_fit_the_public_conv_operand() {
    let conv = assets::select(GdnPhase::CachedDecode, GdnStage::Conv).unwrap();
    assert_eq!(conv.grid, [1, 24, 1]);
    assert_eq!(conv.threads, 128);
    assert_eq!(
        conv.ptx_sha256,
        "a1e81ca0dfff5c4f241061a3baddcc25e7407ad42a2f99f039703199fa5469aa"
    );
    let mut max_read = 0;
    let mut max_write = 0;
    for y in 0..conv.grid[1] as usize {
        for thread in 0..conv.threads as usize {
            let pair = (y << 8) | ((thread << 1) & 254);
            max_read = max_read.max(2 * pair + 4);
            max_write = max_write.max(2 * pair + 4);
            // PTX p5 permits scalar X history reads only when bfe(tid,5,2)==2.
            if ((thread >> 5) & 3) == 2 {
                let first = (y << 8) | ((thread << 3) & 248);
                for offset in (0..16).step_by(2) {
                    max_read = max_read.max(2 * first + offset + 2);
                }
            }
        }
    }
    assert_eq!((max_read, max_write), (12_288, 12_288));
    let decode = assets::select(GdnPhase::CachedDecode, GdnStage::Decode).unwrap();
    assert_eq!(decode.grid, [4, 16, 1]);
    assert_eq!(decode.threads, 32);
    assert_eq!(
        decode.ptx_sha256,
        "e15170544e78ee3ab6b19670b4cc80e9a74db549314bb57f8a897ac855283fd5"
    );
    let mut end = [0; 3];
    for x in 0..4 {
        for y in 0..16 {
            for thread in 0..32 {
                let qk = 2 * ((y << 7) | (thread << 2));
                end[0] = end[0].max(qk + 8);
                end[1] = end[1].max(qk + 4096 + 8);
                end[2] = end[2].max(2 * (4096 + (y << 7) + (x << 5) + thread) + 2);
            }
        }
    }
    assert_eq!(end, [4096, 8192, 12_288]);
}

#[test]
fn fixed_prompt_is_an_exact_initial_slot_one_leaf_without_decode_or_prior_state() {
    let phase = GdnPhase::FixedPrompt;
    assert_eq!(phase.tokens(), 47);
    assert!(!phase.has_initial_state());
    assert_eq!(phase.stages().len(), 9);
    assert!(metadata(phase, GdnCacheSlot::Four).is_err());
    assert!(kernel(phase, GdnStage::Decode).is_err());
    let values = metadata(phase, GdnCacheSlot::One).unwrap();
    let field = |role| &values.iter().find(|(actual, _)| *actual == role).unwrap().1;
    assert_eq!(field(GdnBuffer::CacheIndices), &1_i32.to_le_bytes());
    assert_eq!(field(GdnBuffer::HasInitial), &[0]);
    assert_eq!(field(GdnBuffer::QueryStarts), &[0, 0, 0, 0, 47, 0, 0, 0]);
    for index in 0..6 {
        assert_eq!(
            &field(GdnBuffer::ConvBatch)[index * 4..index * 4 + 4],
            &0_i32.to_le_bytes()
        );
        assert_eq!(
            &field(GdnBuffer::ConvChunkOffsets)[index * 4..index * 4 + 4],
            &(index as i32).to_le_bytes()
        );
    }
    assert_eq!(
        &field(GdnBuffer::ConvBatch)[24..28],
        &(-1_i32).to_le_bytes()
    );
    assert_eq!(
        &field(GdnBuffer::ConvChunkOffsets)[24..28],
        &(-1_i32).to_le_bytes()
    );
    assert_eq!(GdnBuffer::PackedQkvz.bytes(phase), 770_048);
    assert_eq!(GdnBuffer::ConvOutput.bytes(phase), 577_536);
    assert_eq!(GdnBuffer::CacheArena.bytes(phase), 5_570_560);
}

#[test]
fn fixed_prompt_uses_observed_owned_sources_and_exact_unpadded_binding_extents() {
    let phase = GdnPhase::FixedPrompt;
    for (stage, threads, shared, grid, pointers) in [
        (GdnStage::Conv, 128, 0, [6, 24, 1], 9),
        (GdnStage::Prep, 128, 0, [3, 32, 1], 10),
        (GdnStage::Cumsum, 64, 8, [1, 16, 1], 4),
        (GdnStage::Kkt, 256, 16_384, [1, 16, 1], 6),
        (GdnStage::Solve, 128, 10_240, [1, 16, 1], 4),
        (GdnStage::Recompute, 256, 40_960, [1, 16, 1], 9),
        (GdnStage::H, 128, 41_220, [4, 16, 1], 10),
        (GdnStage::Output, 128, 12_288, [4, 1, 16], 8),
        (GdnStage::Gated, 32, 0, [188, 1, 1], 5),
    ] {
        let asset = assets::select(phase, stage).unwrap();
        let old = assets::select(GdnPhase::Initial64, stage).unwrap();
        let spec = kernel_with_typed_outputs(phase, stage).unwrap();
        assert_eq!(
            (asset.threads, asset.shared, asset.grid),
            (threads, shared, grid)
        );
        assert_eq!(spec.bindings.len(), pointers);
        for binding in &spec.bindings {
            assert_eq!(binding.offset, 0);
            assert_eq!(binding.physical_bytes, binding.role.bytes(phase));
            assert_ne!(binding.io, BufferIOAttr::Dead);
        }
        assert_ne!(
            spec.kernel.id(),
            kernel_with_typed_outputs(GdnPhase::Initial64, stage)
                .unwrap()
                .kernel
                .id()
        );
        if matches!(
            stage,
            GdnStage::Solve | GdnStage::Recompute | GdnStage::Output
        ) {
            assert_eq!(asset.source, old.source);
        } else {
            assert_ne!(asset.source, old.source);
        }
    }
    let h = kernel_with_typed_outputs(phase, GdnStage::H).unwrap();
    assert_eq!(h.bindings[1].role, GdnBuffer::U);
    assert_eq!(h.bindings[2].role, GdnBuffer::W);
    assert_eq!(h.bindings[3].role, GdnBuffer::VNew);
    assert_eq!(h.bindings[6].role, GdnBuffer::InitialState);
    assert_eq!(h.bindings[7].role, GdnBuffer::FinalState);
    assert_eq!(h.bindings[6].physical_bytes, RECURRENT_STATE_BYTES);
    assert_eq!(h.bindings[7].physical_bytes, RECURRENT_STATE_BYTES);
}
