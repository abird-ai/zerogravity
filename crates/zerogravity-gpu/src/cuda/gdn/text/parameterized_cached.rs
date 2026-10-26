//! Checked cached Conv/Decode definitions over the retained private cache arena.
use super::*;
use crate::cuda::kernel::{
    StaticScalarBlock, StaticScalarSchema, StaticScalarType, StaticScalarValue,
};
use cubecl_runtime::kernel::BufferIOAttr;

pub(super) const CONV_ENTRY: &str = "zg_gdn_conv_cached_parameterized_v1";
pub(super) const DECODE_ENTRY: &str = "zg_gdn_decode_cached_parameterized_v1";
pub(super) const CONV_SOURCE: &str = include_str!("cached_conv_parameterized.cu");
pub(super) const DECODE_SOURCE: &str = include_str!("cached_decode_parameterized.cu");
pub(super) const CONV_ORIGINAL_PTX_SHA256: &str =
    "a1e81ca0dfff5c4f241061a3baddcc25e7407ad42a2f99f039703199fa5469aa";
pub(super) const DECODE_ORIGINAL_PTX_SHA256: &str =
    "e15170544e78ee3ab6b19670b4cc80e9a74db549314bb57f8a897ac855283fd5";
pub(super) const DECODE_SCALE_BITS: u32 = 1_035_273_459;

#[cfg(test)]
pub(super) const CONV_CANDIDATE_SOURCE_SHA256: &str =
    "d70f130acd441e5d5f028bdf7c63f9cb602921acd320832f40edd3bffd536b61";
#[cfg(test)]
pub(super) const DECODE_CANDIDATE_SOURCE_SHA256: &str =
    "52f1469ebbba967be188411a2e214711c0e1257b5f35275284f1a7f22f284def";
#[cfg(test)]
pub(super) const CONV_PACKED_SOURCE_SHA256: &str =
    "1c627519f34a759b910617a7e964f0e41e4cf02c3236dfc3e86f64d3d2425053";
#[cfg(test)]
pub(super) const DECODE_PACKED_SOURCE_SHA256: &str =
    "e0bd77f07a1ead8b9ef25ce26769aabccd3ac2839fef8eeaee0386c19906ee05";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CachedLayout {
    pub arena_bytes: usize,
    pub slot_pitch: usize,
    pub conv_offset: usize,
    pub conv_bytes: usize,
    pub recurrent_offset: usize,
    pub recurrent_bytes: usize,
    pub cache_index_bytes: [u8; 4],
}
impl CachedLayout {
    fn retained(phase: GdnPhase, slot: GdnCacheSlot) -> Self {
        Self {
            arena_bytes: GdnBuffer::CacheArena.bytes(phase),
            slot_pitch: CACHE_SLOT_BYTES,
            conv_offset: 0,
            conv_bytes: CONV_STATE_BYTES,
            recurrent_offset: CONV_STATE_BYTES,
            recurrent_bytes: RECURRENT_STATE_BYTES,
            cache_index_bytes: (slot.index() as i32).to_le_bytes(),
        }
    }

    fn validate(self, phase: GdnPhase, slot: GdnCacheSlot) -> Result<u32> {
        if phase != GdnPhase::CachedDecode
            || self != Self::retained(phase, slot)
            || self.slot_pitch == 0
            || !self.arena_bytes.is_multiple_of(self.slot_pitch)
        {
            return Err(Error::State(
                "parameterized cached GDN retained arena layout".into(),
            ));
        }
        let cache_lines = self.arena_bytes / self.slot_pitch;
        let cache_index = i32::from_le_bytes(self.cache_index_bytes);
        let index = usize::try_from(cache_index)
            .map_err(|_| Error::State("parameterized cached GDN negative CacheIndices".into()))?;
        let slot_base = index
            .checked_mul(self.slot_pitch)
            .ok_or_else(|| Error::Limit("parameterized cached GDN slot offset overflow".into()))?;
        let slot_end = slot_base
            .checked_add(self.slot_pitch)
            .ok_or_else(|| Error::Limit("parameterized cached GDN slot end overflow".into()))?;
        let conv_end = slot_base
            .checked_add(self.conv_offset)
            .and_then(|n| n.checked_add(self.conv_bytes));
        let recurrent_end = slot_base
            .checked_add(self.recurrent_offset)
            .and_then(|n| n.checked_add(self.recurrent_bytes));
        let decode_view_end = self
            .recurrent_offset
            .checked_add(slot_base)
            .and_then(|n| n.checked_add(self.recurrent_bytes));
        if cache_lines != CACHE_SLOTS
            || index != slot.index()
            || index >= cache_lines
            || self.conv_offset != 0
            || self.conv_bytes != CONV_STATE_BYTES
            || self.recurrent_offset != CONV_STATE_BYTES
            || self.recurrent_bytes != RECURRENT_STATE_BYTES
            || slot.convolution_offset() != slot_base + self.conv_offset
            || slot.recurrent_offset() != slot_base + self.recurrent_offset
            || conv_end.is_none_or(|end| end > slot_end || end > self.arena_bytes)
            || recurrent_end.is_none_or(|end| end > slot_end || end > self.arena_bytes)
            || decode_view_end.is_none_or(|end| end > self.arena_bytes)
        {
            return Err(Error::State(
                "parameterized cached GDN CacheIndices/slot extents".into(),
            ));
        }
        u32::try_from(cache_lines)
            .map_err(|_| Error::Limit("parameterized cached GDN cache lines overflow".into()))
    }
}

pub(super) struct Definition {
    pub asset: assets::Asset,
    pub scalars: StaticScalarBlock,
}

pub(super) fn prepare(phase: GdnPhase, slot: GdnCacheSlot, stage: GdnStage) -> Result<Definition> {
    prepare_checked(phase, slot, stage, CachedLayout::retained(phase, slot))
}

#[cfg(test)]
pub(super) fn prepare_with_layout(
    phase: GdnPhase,
    slot: GdnCacheSlot,
    stage: GdnStage,
    layout: CachedLayout,
) -> Result<Definition> {
    prepare_checked(phase, slot, stage, layout)
}

fn prepare_checked(
    phase: GdnPhase,
    slot: GdnCacheSlot,
    stage: GdnStage,
    layout: CachedLayout,
) -> Result<Definition> {
    slot.validate(phase)?;
    let cache_lines = layout.validate(phase, slot)?;
    match stage {
        GdnStage::Conv => {
            let schema = StaticScalarSchema::new(
                &[
                    (StaticScalarType::U32, 0),
                    (StaticScalarType::U32, 8),
                    (StaticScalarType::U64, 16),
                    (StaticScalarType::U64, 24),
                ],
                32,
                8,
            )?;
            Ok(Definition {
                asset: assets::Asset {
                    entry: CONV_ENTRY,
                    source: CONV_SOURCE,
                    threads: 128,
                    grid: [1, 24, 1],
                    shared: 0,
                    ptx_sha256: CONV_ORIGINAL_PTX_SHA256,
                },
                scalars: StaticScalarBlock::new(
                    schema,
                    &[
                        StaticScalarValue::U32(1),
                        StaticScalarValue::U32(u64::from(cache_lines)),
                        StaticScalarValue::U64(1),
                        StaticScalarValue::U64(1),
                    ],
                )?,
            })
        }
        GdnStage::Decode => Ok(Definition {
            asset: assets::Asset {
                entry: DECODE_ENTRY,
                source: DECODE_SOURCE,
                threads: 32,
                grid: [4, 16, 1],
                shared: 64,
                ptx_sha256: DECODE_ORIGINAL_PTX_SHA256,
            },
            scalars: StaticScalarBlock::new(
                StaticScalarSchema::new(&[(StaticScalarType::F32Bits, 0)], 8, 8)?,
                &[StaticScalarValue::F32Bits(DECODE_SCALE_BITS)],
            )?,
        }),
        _ => Err(Error::Unsupported(
            "parameterized cached GDN Conv/Decode stage".into(),
        )),
    }
}

pub(super) fn require_bindings(
    phase: GdnPhase,
    stage: GdnStage,
    scalars: StaticScalarBlock,
    bindings: &[BindingSpec],
) -> Result<()> {
    use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
    use GdnBuffer::*;
    let expected = match stage {
        GdnStage::Conv => vec![
            (ConvOutput, 0, ConvOutput.bytes(phase), RW),
            (ConvWeights, 0, ConvWeights.bytes(phase), R),
            (CacheArena, 0, CacheArena.bytes(phase), RW),
            (CacheIndices, 0, CacheIndices.bytes(phase), R),
            (ConvOutput, 0, ConvOutput.bytes(phase), RW),
        ],
        GdnStage::Decode => vec![
            (ConvOutput, 0, ConvOutput.bytes(phase), R),
            (PackedBa, 32, 32, R),
            (PackedBa, 0, 32, R),
            (ALog, 0, ALog.bytes(phase), R),
            (DtBias, 0, DtBias.bytes(phase), R),
            (Core, 0, Core.bytes(phase), W),
            (
                CacheArena,
                CONV_STATE_BYTES,
                CacheArena.bytes(phase) - CONV_STATE_BYTES,
                RW,
            ),
            (
                CacheArena,
                CONV_STATE_BYTES,
                CacheArena.bytes(phase) - CONV_STATE_BYTES,
                RW,
            ),
            (CacheIndices, 0, CacheIndices.bytes(phase), R),
        ],
        _ => {
            return Err(Error::Unsupported(
                "parameterized cached GDN binding stage".into(),
            ));
        }
    };
    if bindings.len() != expected.len()
        || !bindings.iter().zip(&expected).all(|(actual, expected)| {
            (
                actual.role,
                actual.offset,
                actual.physical_bytes,
                actual.io,
                actual.static_scalars,
            ) == (expected.0, expected.1, expected.2, expected.3, scalars)
        })
        || stage == GdnStage::Conv && bindings[0] != bindings[4]
        || stage == GdnStage::Decode && bindings[6] != bindings[7]
    {
        return Err(Error::State(
            "parameterized cached GDN pointer/alias/access contract".into(),
        ));
    }
    Ok(())
}
