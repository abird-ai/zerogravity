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
            let workload = PromptWorkload::new(rows, 512)?;
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
            let workload = PromptWorkload::new(rows, 512)?;
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
        assert_eq!(
            definition.asset.ptx_sha256,
            parameterized_conv::ORIGINAL_PTX_SHA256
        );
        assert_ne!(
            definition.asset.ptx_sha256,
            parameterized_conv::CANDIDATE_SOURCE_SHA256
        );
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
    assert!(parameterized_conv::SOURCE.contains(&format!(
        "Original PTX SHA256: {}",
        parameterized_conv::ORIGINAL_PTX_SHA256
    )));
    assert!(parameterized_conv::SOURCE.contains(&format!(
        "Reversible candidate SHA256: {}",
        parameterized_conv::CANDIDATE_SOURCE_SHA256
    )));
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
fn parameterized_conv_host_packet_keeps_nine_pointers_then_one_static_block() -> Result<()> {
    use cubecl::server::Handle;
    use cubecl_common::device::{DeviceId, ServiceId};
    use cubecl_core::environment::stream::StreamId;
    use cubecl_runtime::server::KernelResource;
    use zerogravity_core::accelerator::program::SourceConvGeometry;

    struct HostPacketTest;
    let phase = GdnPhase::TextPrefill(47);
    let definition = parameterized_conv::prepare(
        phase,
        GdnCacheSlot::One,
        SourceConvGeometry {
            qkv_channels: 6144,
            z_channels: 2048,
            coefficient_channels: 32,
            kernel: 4,
        },
    )?;
    let scalars = definition.scalars;
    let recipe = GdnKernelRecipe::fixture(0, 47, false)?;
    let spec = kernel_from_asset_with_scalars(
        phase,
        GdnStage::Conv,
        true,
        definition.asset,
        arithmetic::GdnArithmeticFamily::Text(recipe),
        scalars,
    )?;
    let service = ServiceId::of::<HostPacketTest>(DeviceId::new(0, 0));
    let stream = StreamId::current();
    let handles = spec
        .bindings
        .iter()
        .map(|binding| Handle::new(service, stream, binding.physical_bytes as u64))
        .collect::<Vec<_>>();
    let packet = spec.kernel.host_launch_arguments(&handles, scalars)?;

    assert_eq!(spec.kernel.binding_count(), 9);
    assert_eq!(packet.resources.len(), 9);
    assert_eq!(
        packet.declared_io,
        spec.bindings
            .iter()
            .map(|binding| binding.io)
            .collect::<Vec<_>>()
    );
    for (index, (resource, handle)) in packet.resources.iter().zip(&handles).enumerate() {
        let KernelResource::Buffer(binding) = resource else {
            panic!("Conv argument {index} is not a pointer buffer");
        };
        assert_eq!(binding.memory.id(), handle.memory.descriptor().id);
        assert_eq!(binding.service, service);
        assert_eq!(binding.stream, stream);
        assert_eq!(binding.size, spec.bindings[index].physical_bytes as u64);
    }
    assert_eq!(packet.info.data, [5, 8192, 6144]);
    assert_eq!(packet.info.dynamic_metadata_offset, packet.info.data.len());
    assert_eq!(
        packet
            .info
            .data
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
        scalars.packed_bytes()
    );
    assert!(
        spec.kernel
            .host_launch_arguments(&handles[..8], scalars)
            .is_err()
    );

    let formal = "void zg_gdn_conv_prefill_parameterized_v1(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, void* zg_arg5, void* zg_arg6, void* zg_arg7, void* zg_arg8, ZgGdnConvScalarWords zg_scalars)";
    assert!(parameterized_conv::SOURCE.contains(formal));
    assert!(
        parameterized_conv::SOURCE
            .contains("static_assert(sizeof(ZgGdnConvScalarWords) == 24, \"GDN Conv scalar ABI\")")
    );
    assert!(parameterized_conv::SOURCE.contains(
        "static_assert(alignof(ZgGdnConvScalarWords) == 8, \"GDN Conv scalar alignment\")"
    ));
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
    assert!(
        parameterized_conv::prepare_with_layout(phase, GdnCacheSlot::Four, geometry, retained,)
            .is_err()
    );
    Ok(())
}

#[test]
fn parameterized_prefill_geometry_scalars_and_pointer_contracts_are_exact() -> Result<()> {
    use cubecl_runtime::kernel::BufferIOAttr::{ReadOnly as R, WriteOnly as W};

    for rows in [47, 48, 49, 50, 57, 62, 67, 72] {
        let phase = GdnPhase::TextPrefill(rows);
        let prep = parameterized_prefill::prepare(phase, GdnCacheSlot::One, GdnStage::Prep)?;
        assert_eq!(prep.asset.entry, parameterized_prefill::PREP_ENTRY);
        assert_eq!(prep.asset.source, parameterized_prefill::PREP_SOURCE);
        assert_eq!(prep.asset.grid, [rows.div_ceil(16) as u32, 32, 1]);
        assert_eq!(prep.asset.shared, 0);
        assert_eq!(prep.scalars.schema().byte_size(), 56);
        assert_eq!(
            prep.scalars.words(),
            [6144, 16, 16, 2048, 2048, 2048, rows as u64]
        );
        assert_eq!(
            prep.scalars.packed_bytes(),
            [6144_u64, 16, 16, 2048, 2048, 2048, rows as u64]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>()
        );

        for (stage, entry, source, shared) in [
            (
                GdnStage::Cumsum,
                parameterized_prefill::CUMSUM_ENTRY,
                parameterized_prefill::CUMSUM_SOURCE,
                8,
            ),
            (
                GdnStage::Solve,
                parameterized_prefill::SOLVE_ENTRY,
                parameterized_prefill::SOLVE_SOURCE,
                10240,
            ),
        ] {
            let definition = parameterized_prefill::prepare(phase, GdnCacheSlot::One, stage)?;
            assert_eq!(definition.asset.entry, entry);
            assert_eq!(definition.asset.source, source);
            assert_eq!(definition.asset.grid, [rows.div_ceil(64) as u32, 16, 1]);
            assert_eq!(definition.asset.shared, shared);
            assert!(definition.scalars.words().is_empty());
        }
    }

    let phase = GdnPhase::TextPrefill(47);
    let recipe = GdnKernelRecipe::fixture(0, 47, false)?;
    for (stage, expected) in [
        (
            GdnStage::Prep,
            vec![
                (GdnBuffer::ConvOutput, 47 * 6144 * 2, R),
                (GdnBuffer::AContiguous, 47 * 16 * 2, R),
                (GdnBuffer::BContiguous, 47 * 16 * 2, R),
                (GdnBuffer::ALog, 16 * 4, R),
                (GdnBuffer::DtBias, 16 * 2, R),
                (GdnBuffer::Q, 47 * 2048 * 2, W),
                (GdnBuffer::K, 47 * 2048 * 2, W),
                (GdnBuffer::V, 47 * 2048 * 2, W),
                (GdnBuffer::GRaw, 47 * 16 * 4, W),
                (GdnBuffer::Beta, 47 * 16 * 4, W),
            ],
        ),
        (
            GdnStage::Cumsum,
            vec![
                (GdnBuffer::GRaw, 47 * 16 * 4, R),
                (GdnBuffer::GCumulative, 47 * 16 * 4, W),
                (GdnBuffer::QueryStarts, 8, R),
                (GdnBuffer::ChunkIndices, 8, R),
            ],
        ),
        (
            GdnStage::Solve,
            vec![
                (GdnBuffer::Kkt, 47 * 16 * 64 * 4, R),
                (GdnBuffer::Inverse, 47 * 16 * 64 * 2, W),
                (GdnBuffer::QueryStarts, 8, R),
                (GdnBuffer::ChunkIndices, 8, R),
            ],
        ),
    ] {
        let definition = parameterized_prefill::prepare(phase, GdnCacheSlot::One, stage)?;
        let spec = kernel_from_asset_with_scalars(
            phase,
            stage,
            true,
            definition.asset,
            arithmetic::GdnArithmeticFamily::Text(recipe),
            definition.scalars,
        )?;
        assert_eq!(
            spec.bindings
                .iter()
                .map(|binding| (binding.role, binding.physical_bytes, binding.io))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(spec.bindings.iter().all(|binding| binding.offset == 0));
    }
    Ok(())
}

#[test]
fn parameterized_prefill_host_packets_keep_pointers_then_only_preps_packed_block() -> Result<()> {
    use cubecl::server::Handle;
    use cubecl_common::device::{DeviceId, ServiceId};
    use cubecl_core::environment::stream::StreamId;
    use cubecl_runtime::server::KernelResource;

    struct HostPacketTest;
    let phase = GdnPhase::TextPrefill(47);
    let recipe = GdnKernelRecipe::fixture(0, 47, false)?;
    let service = ServiceId::of::<HostPacketTest>(DeviceId::new(0, 0));
    let stream = StreamId::current();
    for stage in [GdnStage::Prep, GdnStage::Cumsum, GdnStage::Solve] {
        let definition = parameterized_prefill::prepare(phase, GdnCacheSlot::One, stage)?;
        let scalars = definition.scalars;
        let spec = kernel_from_asset_with_scalars(
            phase,
            stage,
            true,
            definition.asset,
            arithmetic::GdnArithmeticFamily::Text(recipe),
            scalars,
        )?;
        let handles = spec
            .bindings
            .iter()
            .map(|binding| Handle::new(service, stream, binding.physical_bytes as u64))
            .collect::<Vec<_>>();
        let packet = spec.kernel.host_launch_arguments(&handles, scalars)?;
        assert_eq!(
            packet.resources.len(),
            if stage == GdnStage::Prep { 10 } else { 4 }
        );
        assert_eq!(
            packet.declared_io,
            spec.bindings
                .iter()
                .map(|binding| binding.io)
                .collect::<Vec<_>>()
        );
        for (index, (resource, handle)) in packet.resources.iter().zip(&handles).enumerate() {
            let KernelResource::Buffer(binding) = resource else {
                panic!("prefill argument {index} is not a pointer buffer");
            };
            assert_eq!(binding.memory.id(), handle.memory.descriptor().id);
            assert_eq!(binding.service, service);
            assert_eq!(binding.stream, stream);
            assert_eq!(binding.size, spec.bindings[index].physical_bytes as u64);
        }
        let expected = if stage == GdnStage::Prep {
            vec![6144, 16, 16, 2048, 2048, 2048, 47]
        } else {
            Vec::new()
        };
        assert_eq!(packet.info.data, expected);
        assert_eq!(packet.info.dynamic_metadata_offset, packet.info.data.len());
        assert!(
            spec.kernel
                .host_launch_arguments(&handles[..handles.len() - 1], scalars)
                .is_err()
        );
    }
    assert!(parameterized_prefill::PREP_SOURCE.contains(
        "void zg_gdn_prep_prefill_parameterized_v1(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3, void* zg_arg4, void* zg_arg5, void* zg_arg6, void* zg_arg7, void* zg_arg8, void* zg_arg9, ZgGdnPrepScalarWords zg_scalars)"
    ));
    assert!(
        parameterized_prefill::PREP_SOURCE
            .contains("static_assert(sizeof(ZgGdnPrepScalarWords) == 56, \"GDN Prep scalar ABI\")")
    );
    for source in [
        parameterized_prefill::CUMSUM_SOURCE,
        parameterized_prefill::SOLVE_SOURCE,
    ] {
        assert!(!source.contains("zg_scalars"));
        assert!(!source.contains("void* zg_arg4"));
    }
    Ok(())
}

#[test]
fn parameterized_prefill_refuses_bad_phase_slot_extents_controls_and_integer_ranges() -> Result<()>
{
    let phase = GdnPhase::TextPrefill(48);
    let prep = parameterized_prefill::PrepLayout {
        mixed_qkv_bytes: 48 * 6144 * 2,
        a_bytes: 48 * 16 * 2,
        b_bytes: 48 * 16 * 2,
        a_log_bytes: 16 * 4,
        dt_bias_bytes: 16 * 2,
        q_bytes: 48 * 2048 * 2,
        k_bytes: 48 * 2048 * 2,
        v_bytes: 48 * 2048 * 2,
        g_bytes: 48 * 16 * 4,
        beta_bytes: 48 * 16 * 4,
    };
    assert!(
        parameterized_prefill::prepare_prep_with_layout(phase, GdnCacheSlot::One, prep).is_ok()
    );
    for bad in [
        parameterized_prefill::PrepLayout {
            mixed_qkv_bytes: prep.mixed_qkv_bytes - 2,
            ..prep
        },
        parameterized_prefill::PrepLayout {
            a_bytes: prep.a_bytes + 2,
            ..prep
        },
        parameterized_prefill::PrepLayout {
            a_log_bytes: prep.a_log_bytes - 4,
            ..prep
        },
        parameterized_prefill::PrepLayout {
            q_bytes: prep.q_bytes - 2,
            ..prep
        },
        parameterized_prefill::PrepLayout {
            beta_bytes: prep.beta_bytes + 4,
            ..prep
        },
    ] {
        assert!(
            parameterized_prefill::prepare_prep_with_layout(phase, GdnCacheSlot::One, bad).is_err()
        );
    }

    for (stage, input_bytes, output_bytes) in [
        (GdnStage::Cumsum, 48 * 16 * 4, 48 * 16 * 4),
        (GdnStage::Solve, 48 * 16 * 64 * 4, 48 * 16 * 64 * 2),
    ] {
        let layout = parameterized_prefill::SequenceLayout {
            input_bytes,
            output_bytes,
            query_starts_bytes: 8,
            chunk_indices_bytes: 8,
        };
        assert!(
            parameterized_prefill::prepare_sequence_with_layout(
                phase,
                GdnCacheSlot::One,
                stage,
                layout,
            )
            .is_ok()
        );
        for bad in [
            parameterized_prefill::SequenceLayout {
                input_bytes: input_bytes - 4,
                ..layout
            },
            parameterized_prefill::SequenceLayout {
                output_bytes: output_bytes + 2,
                ..layout
            },
            parameterized_prefill::SequenceLayout {
                query_starts_bytes: 4,
                ..layout
            },
            parameterized_prefill::SequenceLayout {
                chunk_indices_bytes: 16,
                ..layout
            },
        ] {
            assert!(
                parameterized_prefill::prepare_sequence_with_layout(
                    phase,
                    GdnCacheSlot::One,
                    stage,
                    bad,
                )
                .is_err()
            );
        }
    }

    for stage in [GdnStage::Prep, GdnStage::Cumsum, GdnStage::Solve] {
        assert!(
            parameterized_prefill::prepare(GdnPhase::CachedDecode, GdnCacheSlot::One, stage)
                .is_err()
        );
        assert!(
            parameterized_prefill::prepare(GdnPhase::TextPrefill(0), GdnCacheSlot::One, stage)
                .is_err()
        );
        assert!(parameterized_prefill::prepare(phase, GdnCacheSlot::Four, stage).is_err());
        assert!(
            parameterized_prefill::prepare(
                GdnPhase::TextPrefill(i32::MAX as usize + 1),
                GdnCacheSlot::One,
                stage,
            )
            .is_err()
        );
        assert!(
            parameterized_prefill::prepare(
                GdnPhase::TextPrefill(usize::MAX),
                GdnCacheSlot::One,
                stage,
            )
            .is_err()
        );
    }
    assert!(parameterized_prefill::prepare(phase, GdnCacheSlot::One, GdnStage::Kkt).is_err());
    Ok(())
}

#[test]
fn parameterized_prefill_code_identity_is_new_stable_and_not_a_fixture_selector() -> Result<()> {
    use cubecl_runtime::kernel::KernelMetadata;

    let mut ids = std::collections::BTreeMap::new();
    for rows in [47, 57] {
        let phase = GdnPhase::TextPrefill(rows);
        let recipe = GdnKernelRecipe::fixture(0, rows, false)?;
        for stage in [GdnStage::Prep, GdnStage::Cumsum, GdnStage::Solve] {
            let definition = parameterized_prefill::prepare(phase, GdnCacheSlot::One, stage)?;
            let spec = kernel_from_asset_with_scalars(
                phase,
                stage,
                true,
                definition.asset,
                arithmetic::GdnArithmeticFamily::Text(recipe),
                definition.scalars,
            )?;
            assert_eq!(spec.kernel.id(), spec.kernel.clone().id());
            ids.insert((stage as u8, rows), spec.kernel.id());
            assert_ne!(
                select(recipe, phase, stage)?.entry,
                observed::select(recipe, phase, stage)?.entry
            );
        }
    }
    // Source/schema stay shared; exact physical pointer extents remain in each
    // compiled key and therefore distinguish different launch instances.
    for stage in [GdnStage::Prep, GdnStage::Cumsum, GdnStage::Solve] {
        assert_ne!(ids[&(stage as u8, 47)], ids[&(stage as u8, 57)]);
    }
    assert_ne!(
        ids[&(GdnStage::Prep as u8, 47)],
        ids[&(GdnStage::Cumsum as u8, 47)]
    );
    assert_ne!(
        ids[&(GdnStage::Cumsum as u8, 47)],
        ids[&(GdnStage::Solve as u8, 47)]
    );
    let recipe47 = GdnKernelRecipe::fixture(0, 47, false)?;
    assert!(select(recipe47, GdnPhase::TextPrefill(48), GdnStage::Prep).is_err());
    assert!(
        parameterized_prefill::prepare(
            GdnPhase::TextPrefill(48),
            GdnCacheSlot::One,
            GdnStage::Prep,
        )
        .is_ok()
    );
    assert_ne!(
        parameterized_prefill::PREP_ORIGINAL_PTX_SHA256,
        parameterized_prefill::PREP_CANDIDATE_SOURCE_SHA256
    );
    assert_ne!(
        parameterized_prefill::CUMSUM_ORIGINAL_PTX_SHA256,
        parameterized_prefill::CUMSUM_CANDIDATE_SOURCE_SHA256
    );
    assert!(
        parameterized_prefill::SOLVE_CANDIDATE_SOURCE_SHA256S
            .iter()
            .all(|candidate| *candidate != parameterized_prefill::SOLVE_ORIGINAL_PTX_SHA256)
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
