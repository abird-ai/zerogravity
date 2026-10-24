//! Finite BF16 numeric argmax; integer loads/keys preserve all source bits.
//! Separate contract from the existing F32 total_cmp reducer.
use crate::program_kernels::{INVALID_REDUCTION, NONFINITE_VALUE, VIEW_OUT_OF_BOUNDS};
use cubecl::prelude::*;

pub(super) const VOCABULARY: usize = 248_320;
pub(super) const TILE: usize = 1024;
pub(super) const BLOCKS: usize = VOCABULARY.div_ceil(TILE);
pub(super) const PARTIAL_WORDS: usize = BLOCKS * 3;

/// Numeric finite order. Zero signs share a key; selected payload is unchanged.
#[cube]
pub(super) fn numeric_key(bits: u32) -> u32 {
    let mut canonical = bits;
    if (bits & 0x7fffu32) == 0u32 {
        canonical = 0u32;
    }
    let mut key = canonical ^ 0x8000u32;
    if (canonical & 0x8000u32) != 0u32 {
        key = (!canonical) & 0xffffu32;
    }
    key
}

#[cube]
pub(super) fn precedes(key: u32, token: u32, best_key: u32, best_id: u32) -> bool {
    key > best_key || (key == best_key && token < best_id)
}

/// One exact [original ID, bit-widened original BF16, flags] per1024 IDs.
/// Two BF16 lanes are loaded as one U32 transport word, never as an F32 value.
#[cube(launch_unchecked)]
pub(super) fn block_winners(source: &[u32], partials: &mut [u32], status: &[Atomic<u32>]) {
    let group = ABSOLUTE_POS;
    if group < BLOCKS {
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = 0u32;
        for lane in 0..TILE {
            let id = group * TILE + lane;
            if id < VOCABULARY {
                let word = id / 2usize;
                if word < source.len() {
                    let shift = u32::cast_from(id % 2usize) * 16u32;
                    let bits = (source[word] >> shift) & 0xffffu32;
                    if (bits & 0x7f80u32) == 0x7f80u32 {
                        flags |= NONFINITE_VALUE;
                    } else {
                        let key = numeric_key(bits);
                        let token = u32::cast_from(id);
                        if !found || precedes(key, token, best_key, best_id) {
                            found = true;
                            best_id = token;
                            best_bits = bits << 16u32;
                            best_key = key;
                        }
                    }
                } else {
                    flags |= VIEW_OUT_OF_BOUNDS;
                }
            }
        }
        if source.len() != VOCABULARY / 2usize || partials.len() != PARTIAL_WORDS {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        if !found {
            flags |= INVALID_REDUCTION;
        }
        let dst = group * 3usize;
        if dst + 2usize < partials.len() {
            partials[dst] = best_id;
            partials[dst + 1usize] = best_bits;
            partials[dst + 2usize] = flags;
        } else {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        if flags != 0u32 {
            status[0].fetch_or(flags);
        }
    }
}

/// Numeric second stage +compact packet. Any failed writer yields invalid ID.
/// Shared status is accumulated and never cleared by this consumer.
#[cube(launch_unchecked)]
pub(super) fn finish_winners(partials: &[u32], packet: &mut [u32], status: &[Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = status[0].load();
        if partials.len() != PARTIAL_WORDS || packet.len() != 3usize {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        for block in 0..BLOCKS {
            let src = block * 3usize;
            if src + 2usize < partials.len() {
                let token = partials[src];
                let bits = partials[src + 1usize];
                flags |= partials[src + 2usize];
                if token >= u32::cast_from(VOCABULARY)
                    || token / u32::cast_from(TILE) != u32::cast_from(block)
                    || (bits & 0xffffu32) != 0u32
                {
                    flags |= INVALID_REDUCTION;
                } else if (bits & 0x7f800000u32) == 0x7f800000u32 {
                    flags |= NONFINITE_VALUE;
                } else {
                    let key = numeric_key(bits >> 16u32);
                    if !found || precedes(key, token, best_key, best_id) {
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
        if !found {
            flags |= INVALID_REDUCTION;
        }
        if flags != 0u32 {
            best_id = 0xffffffffu32;
            status[0].fetch_or(flags);
        }
        if packet.len() >= 3usize {
            packet[0] = best_id;
            packet[1] = best_bits;
            packet[2] = flags;
        } else {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    }
}

/// One physical full[B,V] consumer; separate request axis, same numeric keys.
#[cube(launch_unchecked)]
pub(super) fn cohort_block_winners(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] rows: usize,
) {
    let global = ABSOLUTE_POS;
    let row = global / BLOCKS;
    let group = global % BLOCKS;
    if row < rows {
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = 0u32;
        for lane in 0..TILE {
            let id = group * TILE + lane;
            if id < VOCABULARY {
                let word = row * (VOCABULARY / 2usize) + id / 2usize;
                if word < source.len() {
                    let shift = u32::cast_from(id % 2usize) * 16u32;
                    let bits = (source[word] >> shift) & 0xffffu32;
                    if (bits & 0x7f80u32) == 0x7f80u32 {
                        flags |= NONFINITE_VALUE;
                    } else {
                        let key = numeric_key(bits);
                        let token = u32::cast_from(id);
                        if !found || precedes(key, token, best_key, best_id) {
                            found = true;
                            best_id = token;
                            best_bits = bits << 16u32;
                            best_key = key;
                        }
                    }
                } else {
                    flags |= VIEW_OUT_OF_BOUNDS;
                }
            }
        }
        if source.len() != rows * (VOCABULARY / 2usize)
            || partials.len() != rows * PARTIAL_WORDS
            || (rows != 4usize && rows != 8usize)
        {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        if !found {
            flags |= INVALID_REDUCTION;
        }
        let dst = global * 3usize;
        if dst + 2usize < partials.len() {
            partials[dst] = best_id;
            partials[dst + 1usize] = best_bits;
            partials[dst + 2usize] = flags;
        } else {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        if flags != 0u32 {
            status[0].fetch_or(flags);
        }
    }
}

/// One compact original-bit packet per independent request.
#[cube(launch_unchecked)]
pub(super) fn cohort_finish_winners(
    partials: &[u32],
    packet: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] rows: usize,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = status[0].load();
        if partials.len() != rows * PARTIAL_WORDS
            || packet.len() != rows * 3usize
            || (rows != 4usize && rows != 8usize)
        {
            flags |= VIEW_OUT_OF_BOUNDS;
        }
        for block in 0..BLOCKS {
            let src = (row * BLOCKS + block) * 3usize;
            if src + 2usize < partials.len() {
                let token = partials[src];
                let bits = partials[src + 1usize];
                flags |= partials[src + 2usize];
                if token >= u32::cast_from(VOCABULARY)
                    || token / u32::cast_from(TILE) != u32::cast_from(block)
                    || (bits & 0xffffu32) != 0u32
                {
                    flags |= INVALID_REDUCTION;
                } else if (bits & 0x7f800000u32) == 0x7f800000u32 {
                    flags |= NONFINITE_VALUE;
                } else {
                    let key = numeric_key(bits >> 16u32);
                    if !found || precedes(key, token, best_key, best_id) {
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
        if !found {
            flags |= INVALID_REDUCTION;
        }
        if flags != 0u32 {
            best_id = 0xffffffffu32;
            status[0].fetch_or(flags);
        }
        let dst = row * 3usize;
        if dst + 2usize < packet.len() {
            packet[dst] = best_id;
            packet[dst + 1usize] = best_bits;
            packet[dst + 2usize] = flags;
        } else {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    }
}
