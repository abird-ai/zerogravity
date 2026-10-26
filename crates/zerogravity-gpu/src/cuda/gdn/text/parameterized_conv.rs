//! One checked text-prefill convolution definition; no capture-cell dispatch.
use super::*;
use crate::cuda::kernel::{
    StaticScalarBlock, StaticScalarSchema, StaticScalarType, StaticScalarValue,
};
use zerogravity_core::accelerator::program::SourceConvGeometry;

pub(super) const ENTRY: &str = "zg_gdn_conv_prefill_parameterized_v1";
pub(super) const SOURCE: &str = include_str!("conv_parameterized.cu");
// Authenticated reversible three-formal candidate; the final packed ABI has a
// distinct entry and complete SOURCE identity.
pub(super) const CANDIDATE_SHA256: &str =
    "2165f48fc19f67b7d6570086cf179feb5d00717f5a816772ee32db716df655c9";
#[cfg(test)]
pub(super) const PACKED_SOURCE_SHA256: &str =
    "eddbbab8b514d4d3278d8b3f057281508a2aa5171a967c94ecb434b61df7f678";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ConvLayout {
    pub cache_arena_bytes: usize,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub conv_batch_bytes: usize,
    pub conv_chunk_offsets_bytes: usize,
    pub cache_indices_bytes: usize,
    pub has_initial_bytes: usize,
    pub query_starts_bytes: usize,
}
impl ConvLayout {
    fn retained(phase: GdnPhase) -> Self {
        Self {
            cache_arena_bytes: GdnBuffer::CacheArena.bytes(phase),
            input_bytes: GdnBuffer::PackedQkvz.bytes(phase),
            output_bytes: GdnBuffer::ConvOutput.bytes(phase),
            conv_batch_bytes: GdnBuffer::ConvBatch.bytes(phase),
            conv_chunk_offsets_bytes: GdnBuffer::ConvChunkOffsets.bytes(phase),
            cache_indices_bytes: GdnBuffer::CacheIndices.bytes(phase),
            has_initial_bytes: GdnBuffer::HasInitial.bytes(phase),
            query_starts_bytes: GdnBuffer::QueryStarts.bytes(phase),
        }
    }
}

pub(super) struct Definition {
    pub asset: assets::Asset,
    pub scalars: StaticScalarBlock,
}

pub(super) fn prepare(
    phase: GdnPhase,
    slot: GdnCacheSlot,
    geometry: SourceConvGeometry,
) -> Result<Definition> {
    prepare_with_layout(phase, slot, geometry, ConvLayout::retained(phase))
}

pub(super) fn prepare_with_layout(
    phase: GdnPhase,
    slot: GdnCacheSlot,
    geometry: SourceConvGeometry,
    layout: ConvLayout,
) -> Result<Definition> {
    let GdnPhase::TextPrefill(tokens) = phase else {
        return Err(Error::Unsupported(
            "parameterized GDN Conv is text prefill only".into(),
        ));
    };
    if tokens == 0
        || geometry.qkv_channels != 6144
        || geometry.z_channels != 2048
        || geometry.coefficient_channels != 32
        || geometry.kernel != 4
        || geometry.qkv_channels % 256 != 0
        || layout.cache_indices_bytes != 4
        || layout.has_initial_bytes != 1
        || layout.query_starts_bytes != 8
        || layout.conv_batch_bytes % 4 != 0
        || layout.conv_chunk_offsets_bytes != layout.conv_batch_bytes
        || layout.cache_arena_bytes % CACHE_SLOT_BYTES != 0
    {
        return Err(Error::Unsupported(
            "parameterized GDN Conv source/control geometry".into(),
        ));
    }

    let input_stride = geometry
        .qkv_channels
        .checked_add(geometry.z_channels)
        .ok_or_else(|| Error::Limit("GDN Conv input stride overflow".into()))?;
    let output_stride = geometry.qkv_channels;
    let bf16_bytes = 2usize;
    let expected_input = tokens
        .checked_mul(input_stride)
        .and_then(|n| n.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Limit("GDN Conv input extent overflow".into()))?;
    let expected_output = tokens
        .checked_mul(output_stride)
        .and_then(|n| n.checked_mul(bf16_bytes))
        .ok_or_else(|| Error::Limit("GDN Conv output extent overflow".into()))?;
    if layout.input_bytes != expected_input || layout.output_bytes != expected_output {
        return Err(Error::State(
            "parameterized GDN Conv source versus retained buffer extent".into(),
        ));
    }

    let cache_lines = layout.cache_arena_bytes / CACHE_SLOT_BYTES;
    if cache_lines != CACHE_SLOTS
        || slot.index() >= cache_lines
        || slot
            .convolution_offset()
            .checked_add(CONV_STATE_BYTES)
            .is_none_or(|end| end > layout.cache_arena_bytes)
    {
        return Err(Error::State(
            "parameterized GDN Conv retained cache slot/control range".into(),
        ));
    }

    let grid_x = tokens
        .checked_add(7)
        .ok_or_else(|| Error::Limit("GDN Conv grid overflow".into()))?
        / 8;
    if grid_x > layout.conv_batch_bytes / 4 {
        return Err(Error::State(
            "parameterized GDN Conv control table range".into(),
        ));
    }
    let grid = [
        u32::try_from(grid_x).map_err(|_| Error::Limit("GDN Conv grid u32 overflow".into()))?,
        u32::try_from(geometry.qkv_channels / 256)
            .map_err(|_| Error::Limit("GDN Conv channel grid u32 overflow".into()))?,
        1,
    ];
    let schema = StaticScalarSchema::new(
        &[
            (StaticScalarType::U32, 0),
            (StaticScalarType::U64, 8),
            (StaticScalarType::U64, 16),
        ],
        24,
        8,
    )?;
    let scalars = StaticScalarBlock::new(
        schema,
        &[
            StaticScalarValue::U32(
                u32::try_from(cache_lines)
                    .map_err(|_| Error::Limit("GDN Conv cache capacity u32 overflow".into()))?
                    as u64,
            ),
            StaticScalarValue::U64(
                u64::try_from(input_stride)
                    .map_err(|_| Error::Limit("GDN Conv input stride u64 overflow".into()))?,
            ),
            StaticScalarValue::U64(
                u64::try_from(output_stride)
                    .map_err(|_| Error::Limit("GDN Conv output stride u64 overflow".into()))?,
            ),
        ],
    )?;
    Ok(Definition {
        asset: assets::Asset {
            entry: ENTRY,
            source: SOURCE,
            threads: 128,
            grid,
            shared: 0,
            ptx_sha256: CANDIDATE_SHA256,
        },
        scalars,
    })
}
