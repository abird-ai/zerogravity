use super::*;
#[test]
fn fixed_prompt_split_counts_use_actual_partition_and_adapter_copies() {
    let phase = GdnPhase::FixedPrompt;
    // Actual packing A/B/Z plus Conv; original model adapter adds ConvIn/Out.
    let convolution = StagePartition::Convolution.dispatches(phase);
    let recurrent = StagePartition::Recurrent.dispatches(phase);
    assert_eq!((convolution, recurrent), (4, 11));
    assert_eq!((convolution + 2, recurrent + 2), (6, 13));
    assert_eq!((convolution + 2) * 18, 108);
    assert_eq!(convolution + recurrent + 4, 19);
}

#[test]
fn packing_copies_source_words_without_reduction_or_cast() {
    let phase = GdnPhase::Initial64;
    for (kind, role, offset) in [
        (CopyKind::A, GdnBuffer::AContiguous, 8),
        (CopyKind::B, GdnBuffer::BContiguous, 0),
    ] {
        let g = copy_geometry(phase, GdnCacheSlot::Four, kind).unwrap();
        assert_eq!(g.source, Some(GdnBuffer::PackedBa));
        assert_eq!(g.destination, role);
        assert_eq!(
            (
                g.rows,
                g.words,
                g.source_stride,
                g.destination_stride,
                g.source_offset
            ),
            (64, 8, 16, 8, offset)
        );
        assert_eq!(
            (g.rows - 1) * g.source_stride + g.source_offset + g.words,
            1008 + offset + 8
        );
    }
    let z = copy_geometry(phase, GdnCacheSlot::Four, CopyKind::Z).unwrap();
    assert_eq!(
        (z.source, z.destination),
        (Some(GdnBuffer::PackedQkvz), GdnBuffer::ZCompact)
    );
    assert_eq!(
        (z.rows, z.words, z.source_stride, z.source_offset),
        (64, 1024, 4096, 3072)
    );
    assert_eq!(
        (z.rows - 1) * z.source_stride + z.source_offset + z.words,
        GdnBuffer::PackedQkvz.bytes(phase) / 4
    );
    assert!(copy_geometry(GdnPhase::InitialOne, GdnCacheSlot::One, CopyKind::A).is_err());
}

#[test]
fn prior_and_final_cache_transport_keeps_original_state_pitch() {
    let slot = GdnCacheSlot::Four;
    let gather = copy_geometry(GdnPhase::ContinuedTwo, slot, CopyKind::Gather).unwrap();
    let scatter = copy_geometry(GdnPhase::ContinuedTwo, slot, CopyKind::Scatter).unwrap();
    assert_eq!(gather.source, Some(GdnBuffer::CacheArena));
    assert_eq!(gather.destination, GdnBuffer::InitialState);
    assert_eq!(gather.source_offset, slot.recurrent_offset() / 4);
    assert_eq!(scatter.destination_offset, gather.source_offset);
    assert_eq!(gather.words, 262144);
    assert_eq!(scatter.words, 262144);
    assert!(
        gather.source_offset + gather.words
            <= GdnBuffer::CacheArena.bytes(GdnPhase::ContinuedTwo) / 4
    );
    let initial = copy_geometry(GdnPhase::Initial64, slot, CopyKind::Zero).unwrap();
    assert_eq!(initial.source, None);
    assert_eq!(initial.destination, GdnBuffer::InitialState);
    assert_eq!(initial.words, 262144);
    assert!(copy_geometry(GdnPhase::CachedDecode, slot, CopyKind::Gather).is_err());
    assert!(copy_geometry(GdnPhase::ContinuedTwo, slot, CopyKind::Zero).is_err());
}

#[test]
fn solver_inverse_keeps_original_zeros_like_upper_blocks() {
    for (phase, slot, bytes) in [
        (GdnPhase::InitialOne, GdnCacheSlot::One, 2048),
        (GdnPhase::Initial64, GdnCacheSlot::Four, 131072),
        (GdnPhase::ContinuedTwo, GdnCacheSlot::Four, 4096),
    ] {
        let inverse = copy_geometry(phase, slot, CopyKind::ZeroInverse).unwrap();
        assert_eq!(inverse.source, None);
        assert_eq!(inverse.destination, GdnBuffer::Inverse);
        assert_eq!((inverse.rows, inverse.words * 4), (1, bytes));
        assert_eq!(inverse.destination_offset, 0);
    }
    assert!(
        copy_geometry(
            GdnPhase::CachedDecode,
            GdnCacheSlot::One,
            CopyKind::ZeroInverse
        )
        .is_err()
    );
}

// Append inside gdn/resident/tests.rs; metadata only, no fake GPU.
#[test]
fn cached_prefix_and_z_copies_preserve_distinct_actual_ssa_operands() {
    for slot in [GdnCacheSlot::One, GdnCacheSlot::Four] {
        let prefix = copy_geometry(GdnPhase::CachedDecode, slot, CopyKind::QkvPrefix).unwrap();
        assert_eq!(prefix.source, Some(GdnBuffer::PackedQkvz));
        assert_eq!(prefix.destination, GdnBuffer::ConvOutput);
        assert_eq!(
            (
                prefix.rows,
                prefix.words,
                prefix.source_offset,
                prefix.destination_offset
            ),
            (1, 3072, 0, 0)
        );
        let z = copy_geometry(GdnPhase::CachedDecode, slot, CopyKind::Z).unwrap();
        assert_eq!(z.source, Some(GdnBuffer::PackedQkvz));
        assert_eq!(z.destination, GdnBuffer::ZCompact);
        assert_eq!(
            (z.rows, z.words, z.source_offset, z.destination_offset),
            (1, 1024, 3072, 0)
        );
        assert_eq!(
            prefix.words * 4,
            GdnBuffer::ConvOutput.bytes(GdnPhase::CachedDecode)
        );
        assert_eq!(
            z.words * 4,
            GdnBuffer::ZCompact.bytes(GdnPhase::CachedDecode)
        );
        assert_eq!(
            (z.source_offset + z.words) * 4,
            GdnBuffer::PackedQkvz.bytes(GdnPhase::CachedDecode)
        );
    }
    for (phase, slot) in [
        (GdnPhase::InitialOne, GdnCacheSlot::One),
        (GdnPhase::Initial64, GdnCacheSlot::Four),
        (GdnPhase::ContinuedTwo, GdnCacheSlot::Four),
    ] {
        assert!(copy_geometry(phase, slot, CopyKind::QkvPrefix).is_err());
    }
}

#[test]
fn fixed_prompt_packs_only_real_rows_and_zeroes_initial_state_without_a_gather() {
    let phase = GdnPhase::FixedPrompt;
    let slot = GdnCacheSlot::One;
    for (kind, role, source_offset) in [
        (CopyKind::A, GdnBuffer::AContiguous, 8),
        (CopyKind::B, GdnBuffer::BContiguous, 0),
    ] {
        let geometry = copy_geometry(phase, slot, kind).unwrap();
        assert_eq!(geometry.destination, role);
        assert_eq!(geometry.rows, 47);
        assert_eq!(geometry.source_offset, source_offset);
        assert_eq!(geometry.source_stride, 16);
        assert_eq!(geometry.destination_stride, 8);
        assert_eq!(
            (geometry.rows - 1) * geometry.destination_stride + geometry.words,
            376
        );
        assert!(copy_geometry(phase, GdnCacheSlot::Four, kind).is_err());
    }
    let z = copy_geometry(phase, slot, CopyKind::Z).unwrap();
    assert_eq!(
        (z.rows, z.words, z.source_stride, z.source_offset),
        (47, 1024, 4096, 3072)
    );
    assert_eq!(
        ((z.rows - 1) * z.source_stride + z.source_offset + z.words) * 4,
        GdnBuffer::PackedQkvz.bytes(phase)
    );
    let initial = copy_geometry(phase, slot, CopyKind::Zero).unwrap();
    assert_eq!(initial.source, None);
    assert_eq!(initial.destination, GdnBuffer::InitialState);
    assert_eq!(
        (initial.words * 4, initial.destination_offset),
        (RECURRENT_STATE_BYTES, 0)
    );
    assert!(copy_geometry(phase, slot, CopyKind::Gather).is_err());
    assert!(copy_geometry(phase, GdnCacheSlot::Four, CopyKind::Zero).is_err());
    let scatter = copy_geometry(phase, slot, CopyKind::Scatter).unwrap();
    assert_eq!(scatter.source, Some(GdnBuffer::FinalState));
    assert_eq!(scatter.destination, GdnBuffer::CacheArena);
    assert_eq!(scatter.words * 4, RECURRENT_STATE_BYTES);
    assert_eq!(scatter.destination_offset * 4, slot.recurrent_offset());
    let inverse = copy_geometry(phase, slot, CopyKind::ZeroInverse).unwrap();
    assert_eq!(
        (inverse.destination, inverse.words * 4),
        (GdnBuffer::Inverse, 96_256)
    );
}
