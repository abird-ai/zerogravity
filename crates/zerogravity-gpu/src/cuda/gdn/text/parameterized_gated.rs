//! Checked prefill and cached gated-normalization definitions.
use super::*;
use crate::cuda::kernel::{
    StaticScalarBlock, StaticScalarSchema, StaticScalarType, StaticScalarValue,
};
use cubecl_runtime::kernel::BufferIOAttr;

pub(super) const PREFILL_ENTRY: &str = "zg_gdn_gated_prefill_parameterized_v1";
pub(super) const CACHED_ENTRY: &str = "zg_gdn_gated_cached_parameterized_v1";
pub(super) const PREFILL_SOURCE: &str = include_str!("gated_parameterized.cu");
pub(super) const CACHED_SOURCE: &str = include_str!("cached_gated_parameterized.cu");
pub(super) const PREFILL_ORIGINAL_PTX_SHA256: &str =
    "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea";
pub(super) const CACHED_ORIGINAL_PTX_SHA256: &str =
    "a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb";
pub(super) const EPS_BITS: u32 = 897_988_541;

#[cfg(test)]
pub(super) const PREFILL_CANDIDATE_SOURCE_SHA256: &str =
    "833429f0e7ac1aaeb47fb60feb502de2a7fc346f2ebe703685956071fcac06cd";
#[cfg(test)]
pub(super) const CACHED_CANDIDATE_SOURCE_SHA256: &str =
    "f6984295f4d3ba7e8ab76555747f84a0ab18fc03c073c1f5cf3cadbb04b033e9";
#[cfg(test)]
pub(super) const PREFILL_PACKED_SOURCE_SHA256: &str =
    "072a33753de0fe0c40046761bb9ca8461533bda077b97525b7e695f2f9b76162";
#[cfg(test)]
pub(super) const CACHED_PACKED_SOURCE_SHA256: &str =
    "81f3cd72080efd6f1f83410f35213eb12ca91e7867c64af6cf3abdf1aaeda9fe";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GatedLayout {
    pub x_bytes: usize,
    pub y_bytes: usize,
    pub weight_bytes: usize,
    pub z_bytes: usize,
    pub rstd_bytes: usize,
}
impl GatedLayout {
    fn retained(phase: GdnPhase) -> Result<Self> {
        if !matches!(phase, GdnPhase::TextPrefill(_) | GdnPhase::CachedDecode) {
            return Err(Error::Unsupported(
                "parameterized GDN Gated text phase".into(),
            ));
        }
        let rows = phase
            .tokens()
            .checked_mul(16)
            .ok_or_else(|| Error::Limit("parameterized GDN Gated M overflow".into()))?;
        Ok(Self {
            x_bytes: extent(rows, 128, 2, "X")?,
            y_bytes: extent(rows, 128, 2, "Y")?,
            weight_bytes: extent(1, 128, 2, "weight")?,
            z_bytes: extent(rows, 128, 2, "Z")?,
            rstd_bytes: extent(rows, 1, 4, "Rstd")?,
        })
    }
}

pub(super) struct Definition {
    pub asset: assets::Asset,
    pub scalars: StaticScalarBlock,
}

pub(super) fn prepare(phase: GdnPhase, slot: GdnCacheSlot) -> Result<Definition> {
    prepare_checked(phase, slot, GatedLayout::retained(phase)?)
}

#[cfg(test)]
pub(super) fn prepare_with_layout(
    phase: GdnPhase,
    slot: GdnCacheSlot,
    layout: GatedLayout,
) -> Result<Definition> {
    prepare_checked(phase, slot, layout)
}

fn extent(rows: usize, width: usize, element_bytes: usize, role: &str) -> Result<usize> {
    rows.checked_mul(width)
        .and_then(|n| n.checked_mul(element_bytes))
        .ok_or_else(|| Error::Limit(format!("parameterized GDN Gated {role} extent overflow")))
}

fn prepare_checked(phase: GdnPhase, slot: GdnCacheSlot, layout: GatedLayout) -> Result<Definition> {
    if !matches!(phase, GdnPhase::TextPrefill(_) | GdnPhase::CachedDecode) {
        return Err(Error::Unsupported(
            "parameterized GDN Gated text phase".into(),
        ));
    }
    slot.validate(phase)?;
    let tokens = phase.tokens();
    let rows = tokens
        .checked_mul(16)
        .ok_or_else(|| Error::Limit("parameterized GDN Gated M overflow".into()))?;
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| Error::Limit("parameterized GDN Gated M u32 overflow".into()))?;
    if tokens == 0 || i32::try_from(rows).is_err() {
        return Err(Error::Limit(
            "parameterized GDN Gated signed row extent".into(),
        ));
    }
    let expected = GatedLayout::retained(phase)?;
    if layout != expected {
        return Err(Error::State(
            "parameterized GDN Gated retained pointer extents".into(),
        ));
    }
    let schema = StaticScalarSchema::new(
        &[
            (StaticScalarType::U32, 0),
            (StaticScalarType::U32, 8),
            (StaticScalarType::U32, 16),
            (StaticScalarType::U32, 24),
            (StaticScalarType::F32Bits, 32),
        ],
        40,
        8,
    )?;
    let scalars = StaticScalarBlock::new(
        schema,
        &[
            StaticScalarValue::U32(128),
            StaticScalarValue::U32(128),
            StaticScalarValue::U32(128),
            StaticScalarValue::U32(u64::from(rows_u32)),
            StaticScalarValue::F32Bits(EPS_BITS),
        ],
    )?;
    let (entry, source, ptx_sha256) = match phase {
        GdnPhase::TextPrefill(_) => (PREFILL_ENTRY, PREFILL_SOURCE, PREFILL_ORIGINAL_PTX_SHA256),
        GdnPhase::CachedDecode => (CACHED_ENTRY, CACHED_SOURCE, CACHED_ORIGINAL_PTX_SHA256),
        _ => unreachable!(),
    };
    Ok(Definition {
        asset: assets::Asset {
            entry,
            source,
            threads: 32,
            grid: [rows_u32 / 4, 1, 1],
            shared: 0,
            ptx_sha256,
        },
        scalars,
    })
}

pub(super) fn require_bindings(
    phase: GdnPhase,
    scalars: StaticScalarBlock,
    bindings: &[BindingSpec],
) -> Result<()> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    use GdnBuffer::*;
    let expected = [
        (Core, 0, Core.bytes(phase), R),
        (Gated, 0, Gated.bytes(phase), W),
        (GatedGamma, 0, GatedGamma.bytes(phase), R),
        (ZCompact, 0, ZCompact.bytes(phase), R),
        (Rstd, 0, Rstd.bytes(phase), W),
    ];
    if bindings.len() != expected.len()
        || !bindings.iter().zip(expected).all(|(actual, expected)| {
            (
                actual.role,
                actual.offset,
                actual.physical_bytes,
                actual.io,
                actual.static_scalars,
            ) == (expected.0, expected.1, expected.2, expected.3, scalars)
        })
    {
        return Err(Error::State(
            "parameterized GDN Gated pointer/access contract".into(),
        ));
    }
    Ok(())
}
