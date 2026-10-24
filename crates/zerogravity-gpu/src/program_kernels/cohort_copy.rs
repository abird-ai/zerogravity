//! Integer transport for the three source-proven B4 FA copy bijections.
//! Paired BF16 lanes or individual F32 words; no floating arithmetic.
use super::VIEW_OUT_OF_BOUNDS;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn copy_words(
    source: &[u32],
    destination: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] mode: u32,
) {
    let expected = if mode == 2u32 { 32usize } else { 4096usize };
    let i = ABSOLUTE_POS;
    if mode > 2u32 || source.len() != expected || destination.len() != expected {
        if i == 0usize {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    } else if i < expected {
        let index = if mode == 2u32 {
            // F32 LSE: original OffsetCalculator sizes[4,4,2],
            // byte strides dst[4,16,64], src[16,4,64].
            (i / 16usize) * 16usize + (i % 4usize) * 4usize + (i % 16usize) / 4usize
        } else {
            let batch = i / 1024usize;
            let head = (i % 1024usize) / 128usize;
            let lane = i % 128usize;
            let original_head = if mode == 0u32 {
                // Dense[HK,G] -> private[G,HK]. Each word holds two D lanes.
                (head % 2usize) * 4usize + head / 2usize
            } else {
                (head % 4usize) * 2usize + head / 4usize
            };
            batch * 1024usize + original_head * 128usize + lane
        };
        destination[i] = source[index];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn clear_prefix(
    destination: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] words: usize,
) {
    let i = ABSOLUTE_POS;
    if words == 0usize || words > destination.len() || words > 40usize {
        if i == 0usize {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    } else if i < words {
        destination[i] = 0u32;
    }
}

/// Source embedding is pure BF16 row selection, two lanes per word.
#[cube(launch_unchecked)]
pub(crate) fn gather_rows(
    table: &[u32],
    ids: &[u32],
    output: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] rows: usize,
    #[comptime] width_words: usize,
) {
    let i = ABSOLUTE_POS;
    if (width_words != 512usize && width_words != 1024usize)
        || (rows != 4usize && rows != 188usize)
        || table.len() != 248320usize * width_words
        || ids.len() != rows
        || output.len() != rows * width_words
    {
        if i == 0usize {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    } else if i < rows * width_words {
        let id = ids[i / width_words];
        if id < 248320u32 {
            output[i] = table[usize::cast_from(id) * width_words + i % width_words];
        } else {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    }
}

/// SAME BF16 gather consuming four original immutable completed token packets.
#[cube(launch_unchecked)]
pub(crate) fn gather_feedback_rows(
    table: &[u32],
    a: &[u32],
    b: &[u32],
    c: &[u32],
    d: &[u32],
    output: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] width_words: usize,
) {
    let i = ABSOLUTE_POS;
    if (width_words != 512usize && width_words != 1024usize)
        || table.len() != 248320usize * width_words
        || output.len() != 4usize * width_words
        || a.len() != 3usize
        || b.len() != 3usize
        || c.len() != 3usize
        || d.len() != 3usize
    {
        if i == 0usize {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    } else if i < 4usize * width_words {
        let row = i / width_words;
        let id = if row == 0usize {
            a[0]
        } else if row == 1usize {
            b[0]
        } else if row == 2usize {
            c[0]
        } else {
            d[0]
        };
        let valid = if row == 0usize {
            a[2]
        } else if row == 1usize {
            b[2]
        } else if row == 2usize {
            c[2]
        } else {
            d[2]
        };
        if id < 248320u32 && valid == 0u32 {
            output[i] = table[usize::cast_from(id) * width_words + i % width_words];
        } else {
            status[0].fetch_or(VIEW_OUT_OF_BOUNDS);
        }
    }
}
