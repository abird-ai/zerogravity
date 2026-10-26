//! Resident-program bit transport, finite certificates and greedy reduction.
//!
//! No model loop, allocator, source identity, session, event or root authority.
//! Offsets/strides are runtime WORD counts, never byte counts. A binder must
//! validate views and retain every original handle through real completion.
//! Input/output touched ranges must not alias within any dispatch. Reductions
//! ping-pong between distinct buffers. Never clear status while work can use it.
use cubecl::prelude::*;
use zerogravity_core::Error;
use zerogravity_core::Result;

pub(crate) const MAX_WORD_OPS: usize = 1_000_000;
pub(crate) const ARGMAX_RECORD_WORDS: usize = 3;
pub(crate) const VIEW_OUT_OF_BOUNDS: u32 = 1;
pub(crate) const NONFINITE_VALUE: u32 = 2;
pub(crate) const INVALID_REDUCTION: u32 = 4;

/// Count invocation words and validate nonoverlapping destination rows. Input
/// rows may overlap or broadcast. Binder separately validates runtime offsets
/// against actual source/destination word capacities and rejects aliasing.
pub(crate) fn checked_copy_words(
    rows: usize,
    width: usize,
    source_stride: usize,
    destination_stride: usize,
) -> Result<usize> {
    let words = rows
        .checked_mul(width)
        .filter(|n| *n > 0 && *n <= MAX_WORD_OPS)
        .ok_or_else(|| Error::Limit("resident word-copy dispatch geometry invalid".into()))?;
    if rows > 1 && destination_stride < width {
        return Err(Error::Malformed(
            "resident copy destination rows overlap".into(),
        ));
    }
    for stride in [source_stride, destination_stride] {
        let span = (rows - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(width))
            .ok_or_else(|| Error::Limit("resident word-view span overflow".into()))?;
        if u32::try_from(span).is_err() || u32::try_from(stride).is_err() {
            return Err(Error::Limit(
                "resident word-view exceeds shader index domain".into(),
            ));
        }
    }
    Ok(words)
}

/// Fixed-cap serial tiles yield portable, deterministic staged reduction.
/// The result needs 3 words/partial. Reapply until one record remains.
pub(crate) fn argmax_partial_count(count: usize, tile: usize) -> Result<usize> {
    if count == 0 || count > MAX_WORD_OPS || !(32..=1024).contains(&tile) {
        return Err(Error::Limit(
            "resident greedy reduction geometry invalid".into(),
        ));
    }
    Ok(count.div_ceil(tile))
}

/// Exact finite f32::total_cmp order. +0 outranks -0, matching main.rs.
/// Equal keys use lowest ID; no floating conversion/arithmetic can flush a
/// subnormal or collapse the two source zero signs.
#[cube]
pub(crate) fn finite_sortable_key(bits: u32) -> u32 {
    let mut key = bits ^ 0x80000000u32;
    if (bits & 0x80000000u32) != 0u32 {
        key = !bits;
    }
    key
}

#[cube]
fn finite_numeric_sortable_key(bits: u32) -> u32 {
    let mut key = finite_sortable_key(bits);
    if (bits & 0x7fffffffu32) == 0u32 {
        key = 0x80000000u32;
    }
    key
}

/// Exactly one initialized status word is required for every helper. Dispatch
/// before all step writers, then keep its handle/epoch private to the attempt.
#[cube(launch_unchecked)]
pub(crate) fn clear_step_status(status: &[Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        status[0].store(0u32);
    }
}

/// Bitwise copy/pack of typed strided rows, preserving all 32 source bits,
/// including signed zero, subnormals and uninterpreted NaN payloads. UInt64
/// address calculations prevent wraparound BEFORE physical bounds checking.
/// Typed finite Views may certify the copied bits in this same dispatch;
/// raw token/scratch transport must keep that compile-time option disabled.
#[cube(launch_unchecked)]
pub(crate) fn copy_strided_words(
    source: &[u32],
    destination: &mut [u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    destination_offset: u32,
    source_stride: u32,
    destination_stride: u32,
    #[comptime] rows: usize,
    #[comptime] width: usize,
    #[comptime] certify_finite: bool,
) {
    let i = ABSOLUTE_POS;
    let divisor = width.max(1usize);
    if width == 0usize || rows == 0usize || rows > MAX_WORD_OPS / divisor {
        if i == 0usize {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    } else if i < rows * width {
        if rows > 1usize && destination_stride < u32::cast_from(width) {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        } else {
            let row = u64::cast_from(i / width);
            let lane = u64::cast_from(i % width);
            let src = u64::cast_from(source_offset) + row * u64::cast_from(source_stride) + lane;
            let dst = u64::cast_from(destination_offset)
                + row * u64::cast_from(destination_stride)
                + lane;
            if src < u64::cast_from(source.len()) && dst < u64::cast_from(destination.len()) {
                let bits = source[usize::cast_from(src)];
                destination[usize::cast_from(dst)] = bits;
                if certify_finite && (bits & 0x7f800000u32) == 0x7f800000u32 {
                    status[0].fetch_or(NONFINITE_VALUE);
                }
            } else {
                status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
            }
        }
    }
}

/// Certify only typed FINAL outputs whose contract requires finiteness. Do not
/// pass expf/inverse/scaled quantizer intermediates that legally contain Inf
/// or NaN. Certificates OR into the same attempt-private initialized status.
#[cube(launch_unchecked)]
pub(crate) fn certify_finite_strided(
    source: &[u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    source_stride: u32,
    #[comptime] rows: usize,
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    let divisor = width.max(1usize);
    if width == 0usize || rows == 0usize || rows > MAX_WORD_OPS / divisor {
        if i == 0usize {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    } else if i < rows * width {
        let index = u64::cast_from(source_offset)
            + u64::cast_from(i / width) * u64::cast_from(source_stride)
            + u64::cast_from(i % width);
        if index < u64::cast_from(source.len()) {
            if (source[usize::cast_from(index)] & 0x7f800000u32) == 0x7f800000u32 {
                status[0].fetch_or(NONFINITE_VALUE);
            }
        } else {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    }
}

/// Stage one: one record [original token ID, original max bits, status] per
/// tile. Nonfinite logits always fail the attempt, even if not the maximum.
#[allow(clippy::manual_range_contains)] // CubeCL cannot expand RangeInclusive::contains.
#[cube]
fn argmax_logits_ordered(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    destination_offset: u32,
    #[comptime] count: usize,
    #[comptime] tile: usize,
    #[comptime] numeric_zero_ties: bool,
) {
    let group = ABSOLUTE_POS;
    if count == 0usize || count > MAX_WORD_OPS || tile < 32usize || tile > 1024usize {
        if group == 0usize {
            status[0].fetch_or(INVALID_REDUCTION);
        }
    } else if group < count.div_ceil(tile) {
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = 0u32;
        for lane in 0..tile {
            let id = group * tile + lane;
            if id < count {
                let src = u64::cast_from(source_offset) + u64::cast_from(id);
                if src < u64::cast_from(source.len()) {
                    let bits = source[usize::cast_from(src)];
                    if (bits & 0x7f800000u32) == 0x7f800000u32 {
                        flags |= NONFINITE_VALUE;
                    } else {
                        let mut key = finite_sortable_key(bits);
                        if comptime!(numeric_zero_ties) {
                            key = finite_numeric_sortable_key(bits);
                        }
                        let token = u32::cast_from(id);
                        if !found || key > best_key || (key == best_key && token < best_id) {
                            found = true;
                            best_id = token;
                            best_bits = bits;
                            best_key = key;
                        }
                    }
                } else {
                    flags |= VIEW_OUT_OF_BOUNDS;
                }
            }
        }
        if !found {
            flags |= INVALID_REDUCTION;
        }
        let dst = u64::cast_from(destination_offset) + u64::cast_from(group) * 3u64;
        if dst + 2u64 < u64::cast_from(partials.len()) {
            let d = usize::cast_from(dst);
            partials[d] = best_id;
            partials[d + 1] = best_bits;
            partials[d + 2] = flags;
        } else {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        if flags != 0u32 {
            status[0].fetch_or(flags);
        }
    }
}

/// Later stages reduce exactly the same records, preserving source IDs/bits.
/// Use disjoint ping-pong buffers and dispatch until the output has one record.
#[allow(clippy::manual_range_contains)] // CubeCL cannot expand RangeInclusive::contains.
#[cube]
fn argmax_partials_ordered(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    destination_offset: u32,
    #[comptime] count: usize,
    #[comptime] vocabulary: usize,
    #[comptime] tile: usize,
    #[comptime] numeric_zero_ties: bool,
) {
    let group = ABSOLUTE_POS;
    if count == 0usize
        || count > MAX_WORD_OPS
        || vocabulary == 0usize
        || vocabulary > MAX_WORD_OPS
        || tile < 32usize
        || tile > 1024usize
    {
        if group == 0usize {
            status[0].fetch_or(INVALID_REDUCTION);
        }
    } else if group < count.div_ceil(tile) {
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = 0u32;
        for lane in 0..tile {
            let item = group * tile + lane;
            if item < count {
                let src = u64::cast_from(source_offset) + u64::cast_from(item) * 3u64;
                if src + 2u64 < u64::cast_from(source.len()) {
                    let s = usize::cast_from(src);
                    let token = source[s];
                    let bits = source[s + 1];
                    flags |= source[s + 2];
                    if token >= u32::cast_from(vocabulary) {
                        flags |= INVALID_REDUCTION;
                    } else if (bits & 0x7f800000u32) == 0x7f800000u32 {
                        flags |= NONFINITE_VALUE;
                    } else {
                        let mut key = finite_sortable_key(bits);
                        if comptime!(numeric_zero_ties) {
                            key = finite_numeric_sortable_key(bits);
                        }
                        if !found || key > best_key || (key == best_key && token < best_id) {
                            found = true;
                            best_id = token;
                            best_bits = bits;
                            best_key = key;
                        }
                    }
                } else {
                    flags |= VIEW_OUT_OF_BOUNDS;
                }
            }
        }
        if !found {
            flags |= INVALID_REDUCTION;
        }
        let dst = u64::cast_from(destination_offset) + u64::cast_from(group) * 3u64;
        if dst + 2u64 < u64::cast_from(partials.len()) {
            let d = usize::cast_from(dst);
            partials[d] = best_id;
            partials[d + 1] = best_bits;
            partials[d + 2] = flags;
        } else {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        if flags != 0u32 {
            status[0].fetch_or(flags);
        }
    }
}

/// Compact final record is 12 bytes: [token ID, original max bits, status].
/// The SAME output handle may bind its first u32 as next embedding's token;
/// no host token upload is required. Any failed attempt writes invalid ID
/// u32::MAX, preventing an unchecked next source-row selection.
#[cube(launch_unchecked)]
pub(crate) fn finish_argmax(
    source: &[u32],
    output: &mut [u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    #[comptime] vocabulary: usize,
) {
    if ABSOLUTE_POS == 0usize {
        let mut flags = status[0].load();
        let mut token = 0xffffffffu32;
        let mut bits = 0u32;
        let src = u64::cast_from(source_offset);
        if src + 2u64 < u64::cast_from(source.len()) {
            let s = usize::cast_from(src);
            token = source[s];
            bits = source[s + 1];
            flags |= source[s + 2];
            if vocabulary == 0usize
                || vocabulary > MAX_WORD_OPS
                || token >= u32::cast_from(vocabulary)
            {
                flags |= INVALID_REDUCTION;
            }
            if (bits & 0x7f800000u32) == 0x7f800000u32 {
                flags |= NONFINITE_VALUE;
            }
        } else {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        if flags != 0u32 {
            token = 0xffffffffu32;
            status[0].fetch_or(flags);
        }
        if output.len() >= 3usize {
            output[0] = token;
            output[1] = bits;
            output[2] = flags;
        } else {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn argmax_logits(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    destination_offset: u32,
    #[comptime] count: usize,
    #[comptime] tile: usize,
) {
    argmax_logits_ordered(
        source,
        partials,
        status,
        source_offset,
        destination_offset,
        count,
        tile,
        false,
    );
}

#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    target_os = "linux"
))]
#[cube(launch_unchecked)]
pub(crate) fn argmax_logits_numeric_zero(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    destination_offset: u32,
    #[comptime] count: usize,
    #[comptime] tile: usize,
) {
    argmax_logits_ordered(
        source,
        partials,
        status,
        source_offset,
        destination_offset,
        count,
        tile,
        true,
    );
}

#[cube(launch_unchecked)]
pub(crate) fn argmax_partials(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    destination_offset: u32,
    #[comptime] count: usize,
    #[comptime] vocabulary: usize,
    #[comptime] tile: usize,
) {
    argmax_partials_ordered(
        source,
        partials,
        status,
        source_offset,
        destination_offset,
        count,
        vocabulary,
        tile,
        false,
    );
}

#[cfg(all(
    any(
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ),
    target_os = "linux"
))]
#[cube(launch_unchecked)]
pub(crate) fn argmax_partials_numeric_zero(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    source_offset: u32,
    destination_offset: u32,
    #[comptime] count: usize,
    #[comptime] vocabulary: usize,
    #[comptime] tile: usize,
) {
    argmax_partials_ordered(
        source,
        partials,
        status,
        source_offset,
        destination_offset,
        count,
        vocabulary,
        tile,
        true,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_source_key_ties_zeros_without_flushing_subnormal_order() {
        assert_eq!(
            finite_numeric_sortable_key(0),
            finite_numeric_sortable_key(0x80000000)
        );
        assert!(finite_numeric_sortable_key(0x00010000) > finite_numeric_sortable_key(0));
        assert!(finite_numeric_sortable_key(0x80010000) < finite_numeric_sortable_key(0));
        for a in [
            0, 0x80000000, 0x00010000, 0x80010000, 0x3f800000, 0xbf800000, 0x7f7f0000, 0xff7f0000,
        ] {
            for b in [
                0, 0x80000000, 0x00010000, 0x80010000, 0x3f800000, 0xbf800000, 0x7f7f0000,
                0xff7f0000,
            ] {
                let actual = finite_numeric_sortable_key(a) > finite_numeric_sortable_key(b);
                assert_eq!(
                    actual,
                    zerogravity_core::accelerator::program::Bf16TextGreedy::prefer(
                        17,
                        (1, a),
                        (2, b)
                    )
                    .unwrap()
                        && finite_numeric_sortable_key(a) != finite_numeric_sortable_key(b)
                );
            }
        }
        assert!(finite_sortable_key(0) > finite_sortable_key(0x80000000));
    }

    #[test]
    fn finite_integer_key_matches_native_total_order_including_both_zero_signs() {
        let boundary = [
            0u32, 1, 2, 0x003fffff, 0x007fffff, 0x00800000, 0x00800001, 0x3f000000, 0x3f800000,
            0x3f800001, 0x7f7ffffe, 0x7f7fffff,
        ];
        for a in boundary {
            for b in boundary {
                for sa in [0u32, 0x80000000] {
                    for sb in [0u32, 0x80000000] {
                        let a = a | sa;
                        let b = b | sb;
                        assert_eq!(
                            finite_sortable_key(a).cmp(&finite_sortable_key(b)),
                            f32::from_bits(a).total_cmp(&f32::from_bits(b))
                        );
                    }
                }
            }
        }
        let mut rng = 0x5135_791d_662b_8a47u64;
        for _ in 0..100_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let a = rng as u32;
            let b = (rng >> 32) as u32;
            if f32::from_bits(a).is_finite() && f32::from_bits(b).is_finite() {
                assert_eq!(
                    finite_sortable_key(a).cmp(&finite_sortable_key(b)),
                    f32::from_bits(a).total_cmp(&f32::from_bits(b))
                );
            }
        }
        assert!(finite_sortable_key(0) > finite_sortable_key(0x80000000));
    }

    #[test]
    fn geometry_refuses_overlap_zero_and_excessive_dispatches() {
        assert_eq!(checked_copy_words(4, 8, 0, 16).unwrap(), 32);
        assert!(checked_copy_words(4, 8, 16, 7).is_err());
        assert!(checked_copy_words(0, 8, 8, 8).is_err());
        assert!(checked_copy_words(4, 0, 8, 8).is_err());
        assert!(checked_copy_words(1, MAX_WORD_OPS + 1, 0, 0).is_err());
        assert!(checked_copy_words(4, 8, usize::MAX, 8).is_err());
        assert_eq!(argmax_partial_count(248_320, 256).unwrap(), 970);
        assert_eq!(argmax_partial_count(970, 256).unwrap(), 4);
        assert_eq!(argmax_partial_count(4, 256).unwrap(), 1);
        assert!(argmax_partial_count(0, 256).is_err());
        assert!(argmax_partial_count(1024, 0).is_err());
    }
}

/// Packed BF16 transport through u32 storage: no numeric BF16 feature/cast.
/// Even invalid IDs/bounds initialize every output word before status refusal.
#[cfg(all(feature = "qualification", target_os = "linux"))]
#[cube(launch_unchecked)]
pub(crate) fn gather_bf16_matrix_words(
    source: &[u32],
    ids: &[u32],
    output: &mut [u32],
    status: &mut [u32],
    rows: u32,
    row_words: u32,
) {
    let i = ABSOLUTE_POS;
    // Independent status writers initialize every real status slot,including
    // malformed device bounds. No lane-zero race or status-clear epoch.
    if i < status.len() {
        let mut flags = 1u32;
        if row_words > 0u32 && i < ids.len() && ids.len() == status.len() {
            let width = usize::cast_from(row_words);
            let id = ids[i];
            if id < rows
                && source.len() >= width
                && output.len() >= width
                && output.len().is_multiple_of(width)
                && output.len() / width == ids.len()
                && usize::cast_from(id) <= (source.len() - width) / width
            {
                flags = 0u32;
            }
        }
        status[i] = flags;
    }
    if i < output.len() {
        let mut bits = 0u32;
        if row_words > 0u32 {
            let width = usize::cast_from(row_words);
            let r = i / width;
            let lane = i % width;
            if r < ids.len() && r < status.len() {
                let id = ids[r];
                // Division proves multiplication cannot wrap before indexing.
                if id < rows
                    && source.len() >= width
                    && usize::cast_from(id) <= (source.len() - width) / width
                {
                    bits = source[usize::cast_from(id) * width + lane];
                }
            }
        }
        output[i] = bits;
    }
}

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
#[path = "program_kernels/cohort_copy.rs"]
pub(crate) mod cohort_copy;

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
#[path = "program_kernels/readiness.rs"]
mod readiness;
#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) use readiness::{prepare_state_certificates, prepare_text_code};
