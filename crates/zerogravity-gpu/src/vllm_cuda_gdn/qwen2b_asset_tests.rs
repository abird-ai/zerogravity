use super::*;
use crate::vllm_cuda_gdn as gdn;
use cubecl_runtime::kernel::KernelMetadata;

#[test]
fn complete_five_phase_schedule_has_exact_ref3_code_geometry() {
    let schedule = [
        VllmGdnPhase::InitialOne,
        VllmGdnPhase::CachedDecode,
        VllmGdnPhase::Initial64,
        VllmGdnPhase::ContinuedTwo,
        VllmGdnPhase::CachedDecode,
    ];
    let mut instances = 0;
    for phase in schedule {
        for &stage in phase.stages() {
            let asset = select(phase, stage).unwrap();
            let spec = gdn::kernel_qwen2b(phase, stage, false).unwrap();
            assert_eq!(spec.grid, asset.grid);
            assert!(!spec.bindings.is_empty());
            assert!(asset.source.contains(asset.entry));
            assert!(asset.source.contains(asset.ptx_sha256));
            for value in &spec.bindings {
                assert!(
                    value.offset.checked_add(value.physical_bytes).unwrap()
                        <= value.role.bytes(phase)
                );
                assert_ne!(value.io, BufferIOAttr::Dead);
            }
            if stage == GdnStage::Cumsum {
                assert_eq!(
                    (asset.threads, asset.shared, asset.grid),
                    (64, 8, [1, 16, 1])
                );
                assert_eq!(
                    asset.ptx_sha256,
                    "b4c97f73509b35ae7f3281b5621b44ca354e08b65a496575e95670537a0783ab"
                );
            }
            if stage == GdnStage::Recompute {
                assert_eq!(
                    (asset.threads, asset.shared, asset.grid),
                    (256, 32768, [1, 16, 1])
                );
                assert_eq!(
                    asset.ptx_sha256,
                    "d9c9fff7acb0fc04ad9b7ef177f8889ead5c5ce2df8b5d5a81a85958ae191763"
                );
            }
            instances += 1;
        }
    }
    assert_eq!(instances, 33);
    for stage in VllmGdnPhase::Initial47.stages() {
        assert!(select(VllmGdnPhase::Initial47, *stage).is_err());
    }
    assert!(select(VllmGdnPhase::CachedDecode, GdnStage::Prep).is_err());
    assert!(select(VllmGdnPhase::Initial64, GdnStage::Decode).is_err());
}

#[test]
fn source_family_changes_selected_code_and_preserves_original_bindings() {
    for phase in [
        VllmGdnPhase::InitialOne,
        VllmGdnPhase::CachedDecode,
        VllmGdnPhase::Initial64,
        VllmGdnPhase::ContinuedTwo,
    ] {
        for &stage in phase.stages() {
            let old = super::super::select(phase, stage).unwrap();
            let new = select(phase, stage).unwrap();
            for typed in [false, true] {
                let a = gdn::kernel_impl(phase, stage, typed).unwrap();
                let b = gdn::kernel_qwen2b(phase, stage, typed).unwrap();
                assert_eq!(a.bindings, b.bindings);
                assert_eq!(a.grid, b.grid);
                if matches!(
                    stage,
                    GdnStage::Conv | GdnStage::Cumsum | GdnStage::Recompute
                ) {
                    assert_ne!(a.kernel.id(), b.kernel.id());
                } else {
                    assert_eq!(a.kernel.id(), b.kernel.id());
                }
            }
            if matches!(
                stage,
                GdnStage::Conv | GdnStage::Cumsum | GdnStage::Recompute
            ) {
                assert_ne!(new.source, old.source);
            } else {
                assert_eq!(new.source, old.source);
                assert_eq!(new.ptx_sha256, old.ptx_sha256);
                assert_eq!(
                    (new.entry, new.threads, new.shared),
                    (old.entry, old.threads, old.shared)
                );
            }
        }
    }
}

#[test]
fn original_default329_and_typed_cached_address_extents_remain_distinct() {
    use GdnBuffer::*;
    for phase in [
        VllmGdnPhase::InitialOne,
        VllmGdnPhase::CachedDecode,
        VllmGdnPhase::Initial64,
        VllmGdnPhase::ContinuedTwo,
    ] {
        let old = super::super::select(phase, GdnStage::Conv).unwrap();
        let new = select(phase, GdnStage::Conv).unwrap();
        assert!(old.source.contains(", 329;"));
        assert!(new.source.contains(", 159;"));
        assert!(!new.source.contains(", 329;"));
        // Original five-slot authority stays stricter than the observed159.
        assert_eq!(CacheArena.bytes(phase), 5_570_560);
        assert!(
            metadata(
                phase,
                if phase == VllmGdnPhase::InitialOne {
                    GdnCacheSlot::Four
                } else if matches!(phase, VllmGdnPhase::Initial64 | VllmGdnPhase::ContinuedTwo) {
                    GdnCacheSlot::One
                } else {
                    continue;
                }
            )
            .is_err()
        );
    }
    for stage in [GdnStage::Conv, GdnStage::Decode] {
        let spec = gdn::kernel_qwen2b(VllmGdnPhase::CachedDecode, stage, true).unwrap();
        assert_eq!(spec.bindings[0].role, ConvOutput);
        assert_eq!(spec.bindings[0].physical_bytes, 12288);
    }
    let gated = gdn::kernel_qwen2b(VllmGdnPhase::CachedDecode, GdnStage::Gated, true).unwrap();
    assert_eq!(gated.bindings[3].role, ZCompact);
    assert_eq!(gated.bindings[3].physical_bytes, 4096);
    assert_eq!(gated.bindings[3].offset, 0);
}
