//! Checked parameterized Prep and pointer-only Cumsum/Solve definitions.
use super::*;
use crate::cuda::kernel::{
    StaticScalarBlock, StaticScalarSchema, StaticScalarType, StaticScalarValue,
};

pub(super) const PREP_ENTRY: &str = "zg_gdn_prep_prefill_parameterized_v1";
pub(super) const CUMSUM_ENTRY: &str = "zg_gdn_cumsum_prefill_parameterized_v1";
pub(super) const SOLVE_ENTRY: &str = "zg_gdn_solve_prefill_parameterized_v1";
pub(super) const PREP_SOURCE: &str = include_str!("prep_parameterized.cu");
pub(super) const CUMSUM_SOURCE: &str = include_str!("cumsum_parameterized.cu");
pub(super) const SOLVE_SOURCE: &str = include_str!("solve_parameterized.cu");

pub(super) const PREP_ORIGINAL_PTX_SHA256: &str =
    "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670";
pub(super) const CUMSUM_ORIGINAL_PTX_SHA256: &str =
    "9ee24bac2a541a6feab663f2509ba00e112be3cfab4a50a532fe58a101f2eae7";
pub(super) const SOLVE_ORIGINAL_PTX_SHA256: &str =
    "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135";

#[cfg(test)]
pub(super) const PREP_CANDIDATE_SOURCE_SHA256: &str =
    "e71a9758bc2d18a1d9301b3294650c096d738bec80d7630e56f12299d298fc09";
#[cfg(test)]
pub(super) const CUMSUM_CANDIDATE_SOURCE_SHA256: &str =
    "dfb31a1a569028c2775791b7d9e75230fac31bd8009ec977aebd6c595e05a311";
#[cfg(test)]
pub(super) const SOLVE_CANDIDATE_SOURCE_SHA256S: [&str; 3] = [
    "ffd4eeb0f041223092c6a198d70eb2eb1d3cf4fa8aff3166f85a42d83f05a9b6",
    "00860d3b61c543fa9f9a5223d6a4a62aa16dd6e01eaa56105c538263c017d6e1",
    "7178fd11687f132ffc6941a31027339aa6c997c95e16082694f8925e32be821e",
];
#[cfg(test)]
pub(super) const PREP_PACKED_SOURCE_SHA256: &str =
    "cc756c1e7fd6f95f2714d4148840175d2785977786a8bdb22dd42643a7968f30";
#[cfg(test)]
pub(super) const CUMSUM_POINTER_SOURCE_SHA256: &str =
    "f325e2227619fa5d9c55b2a4a339f060e1660e06408c405e744376f4674ca9ce";
#[cfg(test)]
pub(super) const SOLVE_POINTER_SOURCE_SHA256: &str =
    "6905c74feddef8e628e99869afc46a93829661fab97ffe0dfbeaa0c53a38dbf7";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PrepLayout {
    pub mixed_qkv_bytes: usize,
    pub a_bytes: usize,
    pub b_bytes: usize,
    pub a_log_bytes: usize,
    pub dt_bias_bytes: usize,
    pub q_bytes: usize,
    pub k_bytes: usize,
    pub v_bytes: usize,
    pub g_bytes: usize,
    pub beta_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SequenceLayout {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub query_starts_bytes: usize,
    pub chunk_indices_bytes: usize,
}

pub(super) struct Definition {
    pub asset: assets::Asset,
    pub scalars: StaticScalarBlock,
}

#[derive(Clone, Copy)]
enum GridKind {
    Prep,
    Chunk,
}

struct CheckedGeometry {
    tokens: usize,
    chunks: usize,
    grid: [u32; 3],
}

/// Shared checked text-prefill geometry. The i32 bound covers QueryStarts and
/// ChunkIndices contents; u32 conversion covers the physical launch packet.
fn checked_geometry(
    phase: GdnPhase,
    slot: GdnCacheSlot,
    kind: GridKind,
) -> Result<CheckedGeometry> {
    let GdnPhase::TextPrefill(tokens) = phase else {
        return Err(Error::Unsupported(
            "parameterized GDN stage is text prefill only".into(),
        ));
    };
    slot.validate(phase)?;
    if tokens == 0 || i32::try_from(tokens).is_err() {
        return Err(Error::Limit(
            "parameterized GDN text/control i32 extent".into(),
        ));
    }
    let chunks = tokens.div_ceil(64);
    let (tile, y) = match kind {
        GridKind::Prep => (16, 32),
        GridKind::Chunk => (64, 16),
    };
    let x = u32::try_from(tokens.div_ceil(tile))
        .map_err(|_| Error::Limit("parameterized GDN grid u32 overflow".into()))?;
    Ok(CheckedGeometry {
        tokens,
        chunks,
        grid: [x, y, 1],
    })
}

fn extent(rows: usize, width: usize, element_bytes: usize, role: &str) -> Result<usize> {
    rows.checked_mul(width)
        .and_then(|n| n.checked_mul(element_bytes))
        .ok_or_else(|| Error::Limit(format!("parameterized GDN {role} extent overflow")))
}

fn expected_prep(tokens: usize) -> Result<PrepLayout> {
    Ok(PrepLayout {
        mixed_qkv_bytes: extent(tokens, 6144, 2, "Prep mixed QKV")?,
        a_bytes: extent(tokens, 16, 2, "Prep A")?,
        b_bytes: extent(tokens, 16, 2, "Prep B")?,
        a_log_bytes: extent(16, 1, 4, "Prep A-log")?,
        dt_bias_bytes: extent(16, 1, 2, "Prep dt-bias")?,
        q_bytes: extent(tokens, 2048, 2, "Prep Q")?,
        k_bytes: extent(tokens, 2048, 2, "Prep K")?,
        v_bytes: extent(tokens, 2048, 2, "Prep V")?,
        g_bytes: extent(tokens, 16, 4, "Prep G")?,
        beta_bytes: extent(tokens, 16, 4, "Prep beta")?,
    })
}

fn expected_sequence(tokens: usize, chunks: usize, stage: GdnStage) -> Result<SequenceLayout> {
    let (input_width, input_element, output_width, output_element) = match stage {
        GdnStage::Cumsum => (16, 4, 16, 4),
        GdnStage::Solve => (16 * 64, 4, 16 * 64, 2),
        _ => {
            return Err(Error::Unsupported(
                "parameterized GDN sequence stage".into(),
            ));
        }
    };
    Ok(SequenceLayout {
        input_bytes: extent(tokens, input_width, input_element, "sequence input")?,
        output_bytes: extent(tokens, output_width, output_element, "sequence output")?,
        query_starts_bytes: 2 * std::mem::size_of::<i32>(),
        chunk_indices_bytes: extent(chunks, 2, std::mem::size_of::<i32>(), "chunk indices")?,
    })
}

pub(super) fn prepare(phase: GdnPhase, slot: GdnCacheSlot, stage: GdnStage) -> Result<Definition> {
    match stage {
        GdnStage::Prep => {
            let geometry = checked_geometry(phase, slot, GridKind::Prep)?;
            let layout = expected_prep(geometry.tokens)?;
            prepare_prep_checked(geometry, layout)
        }
        GdnStage::Cumsum | GdnStage::Solve => {
            let geometry = checked_geometry(phase, slot, GridKind::Chunk)?;
            let layout = expected_sequence(geometry.tokens, geometry.chunks, stage)?;
            prepare_sequence_checked(stage, geometry, layout)
        }
        _ => Err(Error::Unsupported(
            "parameterized GDN Prep/Cumsum/Solve stage".into(),
        )),
    }
}

#[cfg(test)]
pub(super) fn prepare_prep_with_layout(
    phase: GdnPhase,
    slot: GdnCacheSlot,
    layout: PrepLayout,
) -> Result<Definition> {
    let geometry = checked_geometry(phase, slot, GridKind::Prep)?;
    prepare_prep_checked(geometry, layout)
}

fn prepare_prep_checked(geometry: CheckedGeometry, layout: PrepLayout) -> Result<Definition> {
    if layout != expected_prep(geometry.tokens)? {
        return Err(Error::State(
            "parameterized GDN Prep source versus retained pointer extents".into(),
        ));
    }
    let schema = StaticScalarSchema::new(
        &[
            (StaticScalarType::U32, 0),
            (StaticScalarType::U32, 8),
            (StaticScalarType::U32, 16),
            (StaticScalarType::U32, 24),
            (StaticScalarType::U32, 32),
            (StaticScalarType::U32, 40),
            (StaticScalarType::U32, 48),
        ],
        56,
        8,
    )?;
    let tokens = u32::try_from(geometry.tokens)
        .map_err(|_| Error::Limit("parameterized GDN Prep L u32 overflow".into()))?;
    let scalars = StaticScalarBlock::new(
        schema,
        &[
            StaticScalarValue::U32(6144),
            StaticScalarValue::U32(16),
            StaticScalarValue::U32(16),
            StaticScalarValue::U32(2048),
            StaticScalarValue::U32(2048),
            StaticScalarValue::U32(2048),
            StaticScalarValue::U32(u64::from(tokens)),
        ],
    )?;
    Ok(Definition {
        asset: assets::Asset {
            entry: PREP_ENTRY,
            source: PREP_SOURCE,
            threads: 128,
            grid: geometry.grid,
            shared: 0,
            ptx_sha256: PREP_ORIGINAL_PTX_SHA256,
        },
        scalars,
    })
}

#[cfg(test)]
pub(super) fn prepare_sequence_with_layout(
    phase: GdnPhase,
    slot: GdnCacheSlot,
    stage: GdnStage,
    layout: SequenceLayout,
) -> Result<Definition> {
    let geometry = checked_geometry(phase, slot, GridKind::Chunk)?;
    prepare_sequence_checked(stage, geometry, layout)
}

fn prepare_sequence_checked(
    stage: GdnStage,
    geometry: CheckedGeometry,
    layout: SequenceLayout,
) -> Result<Definition> {
    if layout != expected_sequence(geometry.tokens, geometry.chunks, stage)? {
        return Err(Error::State(
            "parameterized GDN sequence source versus retained pointer/control extents".into(),
        ));
    }
    let (entry, source, shared, ptx_sha256) = match stage {
        GdnStage::Cumsum => (CUMSUM_ENTRY, CUMSUM_SOURCE, 8, CUMSUM_ORIGINAL_PTX_SHA256),
        GdnStage::Solve => (SOLVE_ENTRY, SOLVE_SOURCE, 10240, SOLVE_ORIGINAL_PTX_SHA256),
        _ => {
            return Err(Error::Unsupported(
                "parameterized GDN sequence stage".into(),
            ));
        }
    };
    Ok(Definition {
        asset: assets::Asset {
            entry,
            source,
            threads: 128,
            grid: geometry.grid,
            shared,
            ptx_sha256,
        },
        scalars: StaticScalarBlock::empty(),
    })
}
