//! Closed source-derived logical admission; not VRAM/RSS/allocator telemetry.
use super::*;
use crate::cuda::gdn::{GdnBuffer as B, GdnPhase};
use crate::resident_program::native_full::FullModelProgram;
use crate::resident_program::source_bindings::SourceBindings;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::accelerator::program::{SourceCallMode, SourceOperatorKind};
const MIB: usize = 1024 * 1024;
fn add(sum: &mut usize, n: usize) -> Result<()> {
    *sum = sum
        .checked_add(n)
        .ok_or_else(|| Error::Limit("full24 logical capacity overflow".into()))?;
    Ok(())
}
pub(crate) fn provider_bound(
    source: &VerifiedTensorSource,
    records: &[Arc<SourceBindings>],
) -> Result<usize> {
    if source.staging_chunk_bytes() != MIB {
        return Err(Error::Unsupported(
            "full24 requires the original closed1MiB authenticated source scan geometry".into(),
        ));
    }
    if records.len() != 5 {
        return Err(Error::Unsupported("closed five full24 source calls".into()));
    }
    let expected = [
        (SourceCallMode::Prefill, 0, 1),
        (SourceCallMode::Decode, 1, 1),
        (SourceCallMode::Prefill, 0, 64),
        (SourceCallMode::Prefill, 64, 2),
        (SourceCallMode::Decode, 66, 1),
    ];
    for (record, (phase, base, rows)) in records.iter().zip(expected) {
        FullModelProgram::require_record(record)?;
        let call = record.call();
        if !record.source().same_issuer(source)
            || call.mode() != phase
            || call.rows().base() != base
            || call.rows().len() != rows
        {
            return Err(Error::State(
                "full24 exact original five-phase request order".into(),
            ));
        }
    }
    let mut weights = 508_559_360usize;
    let mut groups = 0usize;
    for index in 0..records[0].plan().nodes().len() {
        let Some(descriptor) = &records[0].operator(index)? else {
            continue;
        };
        if matches!(
            descriptor.kind(),
            SourceOperatorKind::Projection { selected_row: None }
        ) {
            groups += 1;
            for matrix in descriptor.matrices() {
                add(
                    &mut weights,
                    matrix
                        .rows
                        .checked_mul(matrix.columns)
                        .and_then(|n| n.checked_mul(2))
                        .ok_or_else(|| Error::Limit("full24 checked source W bytes".into()))?,
                )?;
            }
        }
    }
    if groups != 114 || weights != 1_503_789_056 {
        return Err(Error::State(
            "full24 exact original finite W geometry".into(),
        ));
    }
    let mut bound = weights;
    // Two genuine Session roots +five private complete candidates.
    add(&mut bound, 7 * 32_784_384)?;
    // Exactly one native arena per source layer and request history.
    add(&mut bound, 2 * 24 * 5_570_560)?;
    for record in records {
        let rows = record.call().rows().len();
        for value in record.plan().values() {
            add(
                &mut bound,
                value
                    .capacity_elements()?
                    .checked_mul(2)
                    .ok_or_else(|| Error::Limit("full24 public BF16 bytes".into()))?,
            )?;
        }
        add(&mut bound, 49 * (10_240 + 16_396 * rows))?;
        let phase = match (record.call().mode(), record.call().rows().base(), rows) {
            (SourceCallMode::Prefill, 0, 1) => GdnPhase::InitialOne,
            (SourceCallMode::Prefill, 0, 64) => GdnPhase::Initial64,
            (SourceCallMode::Prefill, 64, 2) => GdnPhase::ContinuedTwo,
            (SourceCallMode::Decode, 1 | 66, 1) => GdnPhase::CachedDecode,
            _ => {
                return Err(Error::State(
                    "full24 exact reference physical phase differs".into(),
                ));
            }
        };
        // Conservative all known private GDN roles, even when a selected phase
        // omits one. Public outputs/parameters/arena are counted separately.
        let private = [
            B::AContiguous,
            B::BContiguous,
            B::Q,
            B::K,
            B::V,
            B::GRaw,
            B::Beta,
            B::GCumulative,
            B::Kkt,
            B::Inverse,
            B::W,
            B::U,
            B::HChunks,
            B::InitialState,
            B::FinalState,
            B::VNew,
            B::Core,
            B::Rstd,
            B::CacheIndices,
            B::HasInitial,
            B::QueryStarts,
            B::ConvBatch,
            B::ConvChunkOffsets,
            B::ChunkIndices,
            B::ChunkOffsets,
        ]
        .into_iter()
        .try_fold(0usize, |mut sum, role| {
            add(&mut sum, role.bytes(phase))?;
            Ok::<_, Error>(sum)
        })?;
        add(&mut bound, 18 * private)?;
        add(&mut bound, 24 * 4 + 4 + 2916 + 12 + 3 * 65 * 8)?;
    }
    add(
        &mut bound,
        attention_cubin::full_model_private_bound(source.source_identity())?,
    )?;
    // Learned source parameters conservatively include duplicate raw gamma.
    add(&mut bound, 1_002_176)?;
    add(&mut bound, 8_519_680 + 8576 + 64 * MIB)?;
    let bound = bound
        .checked_add(64 * MIB - 1)
        .map(|n| n / (64 * MIB) * (64 * MIB))
        .ok_or_else(|| Error::Limit("full24 rounded logical capacity".into()))?;
    if bound > 5 * 1024 * MIB {
        return Err(Error::Limit(
            "full24 source-derived managed envelope exceeds unchanged5GiB outer device cap".into(),
        ));
    }
    Ok(bound)
}

/// Only the new complete forced-full24 qualifier scope. Host tickets bound
/// owned source/metadata/readback allocations; this is not host RSS capacity.
pub fn qwen08_model_host_bound() -> Result<usize> {
    let mut retained = 0usize;
    // Existing original source metadata/opening +independent expected reader.
    add(&mut retained, 64 * MIB + 128 * MIB + 4 * MIB)?;
    add(&mut retained, 14 * 16 * MIB)?; // original per-image source admissions
    add(&mut retained, 5 * 4 * MIB)?; // original full Body metadata
    // Each phase pre-admits complete public+prior/candidate+RMS readback bytes.
    for rows in [1usize, 1, 64, 2, 1] {
        let public = 1_670_272 * rows + 496_640;
        let readback =
            public + 2 * 32_784_384 + 48 * (2048 + rows * 8192) + 2048 + rows * 4096 + MIB;
        add(&mut retained, readback.div_ceil(32 * MIB) * 32 * MIB)?;
        // 1 ordinary +48 dual source owners; their existing exact ticket math.
        add(&mut retained, 49 * 2 * MIB)?;
        let physical = 10_240 + 20_492 * rows;
        add(
            &mut retained,
            physical * 3 + (2048 + rows * 2048) * 3 + 64 * 1024,
        )?;
        add(
            &mut retained,
            48 * (physical * 3 + (2048 + rows * 8192) * 3 + 64 * 1024),
        )?;
        add(&mut retained, 49 * 64 * 1024)?; // source-bound RMS descriptors
        let phase = match rows {
            64 => GdnPhase::Initial64,
            2 => GdnPhase::ContinuedTwo,
            _ => GdnPhase::CachedDecode,
        };
        // InitialOne uses the larger 9-stage set; cached1 source-code bound is
        // covered by its own actual set, so count InitialOne separately below.
        add(
            &mut retained,
            18 * crate::cuda::gdn::model_metadata_bound(phase)?,
        )?;
    }
    // Replace the first cached-code term with the actual initial1 source set.
    let cached = crate::cuda::gdn::model_metadata_bound(GdnPhase::CachedDecode)?;
    let initial = crate::cuda::gdn::model_metadata_bound(GdnPhase::InitialOne)?;
    retained = retained
        .checked_sub(18 * cached)
        .ok_or_else(|| Error::Limit("full24 initial source metadata replacement".into()))?;
    add(&mut retained, 18 * initial)?;
    // Upload coverage metadata, shared raw/converted parameters and actual
    // per-phase transport/control/greedy owners. No fullhost learned W pack.
    add(
        &mut retained,
        115 * 64 * 1024 + 18 * 320 * 1024 + 6 * 64 * 1024,
    )?;
    add(
        &mut retained,
        (90 + 30 + 30 + 5 + 5) * 64 * 1024 + 2 * MIB + 64 * 1024,
    )?;
    // Actual reader retains complete authenticated chunks +offset Vec.
    //245 RMS gamma +69 embedding +72 GDN +12QK windows, all<=49152B<1MiB.
    // Two chunks per arbitrary alignment; no repeated-read sharing assumed.
    let per_chunk = MIB
        .checked_add(std::mem::size_of::<(usize, std::ops::Range<usize>)>())
        .ok_or_else(|| Error::Limit("full24 source staging offset accounting".into()))?;
    add(&mut retained, (245 + 69 + 72 + 12) * 2 * per_chunk)?;
    add(&mut retained, MIB + 64 * 1024 + 3 * 8576)?; // genuine computed rotary result
    // Sequential bounds: ONE zero-cache ticket (released after real H2D),
    // bounded scan/H2D source window, and final32MiB independent W readback.
    let sequential = (3 * 5_570_560 + 64 * 1024).max(32 * MIB) + 16 * MIB;
    let full = retained
        .checked_add(sequential)
        .and_then(|n| n.checked_add(64 * MIB - 1))
        .map(|n| n / (64 * MIB) * (64 * MIB))
        .ok_or_else(|| Error::Limit("full24 checked host ticket envelope".into()))?;
    Ok(full)
}

/// Separate reusable checkpoint: four retained ORIGINAL Session roots, one
/// obsolete prior retained by the workspace, and one current fresh candidate, plus ten independent completed/diagnostic packets.
/// Ordinary ledger admission still charges every additional live owner.
pub(crate) fn reusable_provider_bound(
    source: &VerifiedTensorSource,
    records: &[Arc<SourceBindings>],
) -> Result<usize> {
    with_reusable_capacity(provider_bound(source, records)?)
}
fn with_reusable_capacity(base: usize) -> Result<usize> {
    const ROOTS: usize = 6;
    const ROOT_BYTES: usize = 32_784_384;
    const PACKETS: usize = 10;
    const ORIGINAL_ALIGNMENT: usize = 512;
    let bytes = ROOTS
        .checked_mul(ROOT_BYTES)
        .and_then(|n| n.checked_add(PACKETS * ORIGINAL_ALIGNMENT))
        .and_then(|n| base.checked_add(n))
        .and_then(|n| n.checked_add(64 * MIB - 1))
        .map(|n| n / (64 * MIB) * (64 * MIB))
        .ok_or_else(|| Error::Limit("reusable canonical/packet capacity overflow".into()))?;
    if bytes > 5 * 1024 * MIB {
        return Err(Error::Limit(
            "reusable model exceeds unchanged5GiB physical envelope".into(),
        ));
    }
    Ok(bytes)
}
#[cfg(test)]
mod reusable_tests {
    use super::*;
    #[test]
    fn reusable_capacity_prepays_retained_roots_and_distinct_packet_alignment() {
        assert_eq!(
            with_reusable_capacity(1024 * MIB).unwrap(),
            (1024 + 192) * MIB
        );
    }
    #[test]
    fn reusable_capacity_does_not_relax_original_limit_or_overflow() {
        assert!(with_reusable_capacity(5 * 1024 * MIB).is_err());
        assert!(with_reusable_capacity(usize::MAX).is_err());
    }
}
