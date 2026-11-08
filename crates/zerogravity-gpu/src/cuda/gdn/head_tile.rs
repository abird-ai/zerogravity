//! Head-axis transport around the fixed-width GDN leaf. No arithmetic or state publication.
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};

pub(crate) const TILE_HEADS: usize = 16;
pub(crate) const HEAD_WIDTH: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadGroups {
    pub key: usize,
    pub value: usize,
}
impl HeadGroups {
    pub(crate) fn new(key: usize, value: usize) -> Result<Self> {
        if key == 0
            || key > 64
            || value == 0
            || value > 256
            || !value.is_multiple_of(key)
            || !value.is_multiple_of(TILE_HEADS)
        {
            return Err(Error::Unsupported("CUDA GDN grouped head extent".into()));
        }
        Ok(Self { key, value })
    }
    pub(crate) fn tiles(self) -> usize {
        self.value / TILE_HEADS
    }
    pub(crate) fn direct(self) -> bool {
        self.key == TILE_HEADS && self.value == TILE_HEADS
    }
    pub(crate) fn channels(self) -> usize {
        (2 * self.key + self.value) * HEAD_WIDTH
    }
    pub(crate) fn repeat(self) -> usize {
        self.value / self.key
    }
}

/// Word-aligned head blocks. BF16 callers use pairs; F32 callers use whole words.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HeadCopy {
    pub rows: usize,
    pub heads: usize,
    pub first_head: usize,
    pub repeat: usize,
    pub words_per_head: usize,
    pub full_stride: usize,
    pub full_offset: usize,
    pub tile_stride: usize,
    pub tile_offset: usize,
}
impl HeadCopy {
    fn bounds(self) -> Result<(usize, usize, usize)> {
        if self.rows == 0
            || self.heads == 0
            || self.repeat == 0
            || self.words_per_head == 0
            || [
                self.rows,
                self.heads,
                self.first_head,
                self.repeat,
                self.words_per_head,
                self.full_stride,
                self.full_offset,
                self.tile_stride,
                self.tile_offset,
            ]
            .iter()
            .any(|n| *n > u32::MAX as usize)
        {
            return Err(Error::Malformed("CUDA head copy empty extent".into()));
        }
        let last = self
            .first_head
            .checked_add(self.heads - 1)
            .ok_or_else(|| Error::Limit("CUDA head copy head extent".into()))?;
        let extent = |stride: usize, offset: usize, heads: usize| {
            let tail = heads
                .checked_mul(self.words_per_head)
                .and_then(|n| n.checked_add(offset))
                .filter(|n| *n <= stride)
                .ok_or_else(|| Error::Limit("CUDA head copy row extent".into()))?;
            (self.rows - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(tail))
                .filter(|n| *n <= u32::MAX as usize / 4)
                .ok_or_else(|| Error::Limit("CUDA head copy address extent".into()))
        };
        let full = extent(self.full_stride, self.full_offset, last / self.repeat + 1)?;
        let tile = extent(self.tile_stride, self.tile_offset, self.heads)?;
        let words = self
            .rows
            .checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.words_per_head))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or_else(|| Error::Limit("CUDA head copy launch extent".into()))?;
        Ok((full, tile, words))
    }

    pub(crate) fn launch(
        self,
        client: &Client,
        full: &Handle,
        tile: &Handle,
        scatter: bool,
    ) -> Result<usize> {
        let (full_words, tile_words, words) = self.bounds()?;
        if full.service != client.service_id()
            || tile.service != client.service_id()
            || full.size_in_used() < (full_words * 4) as u64
            || tile.size_in_used() < (tile_words * 4) as u64
        {
            return Err(Error::State("CUDA head copy service/extent".into()));
        }
        if crate::resident_program::BufferRange::new(full)?
            .overlaps(&crate::resident_program::BufferRange::new(tile)?)
        {
            return Err(Error::State("CUDA head copy overlapping owners".into()));
        }
        let source = if scatter { tile } else { full };
        let destination = if scatter { full } else { tile };
        // SAFETY: checked word extents; enclosing flight retains both non-aliasing owners.
        unsafe {
            copy_heads::launch_unchecked(
                client,
                CubeCount::Static((words as u32).div_ceil(128), 1, 1),
                CubeDim::new_1d(128),
                BufferArg::from_raw_parts(source.clone(), source.size_in_used() as usize / 4),
                BufferArg::from_raw_parts(
                    destination.clone(),
                    destination.size_in_used() as usize / 4,
                ),
                words as u32,
                self.heads as u32,
                self.first_head as u32,
                self.repeat as u32,
                self.words_per_head as u32,
                self.full_stride as u32,
                self.full_offset as u32,
                self.tile_stride as u32,
                self.tile_offset as u32,
                scatter,
            );
        }
        Ok(1)
    }
}

#[cube(launch_unchecked)]
fn copy_heads(
    source: &[u32],
    destination: &mut [u32],
    words: u32,
    heads: u32,
    first: u32,
    repeat: u32,
    width: u32,
    full_stride: u32,
    full_offset: u32,
    tile_stride: u32,
    tile_offset: u32,
    #[comptime] scatter: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(words) {
        let at = u32::cast_from(i);
        let row = at / (heads * width);
        let local = at % (heads * width);
        let head = first + local / width;
        let full = row * full_stride + full_offset + (head / repeat) * width + local % width;
        let tile = row * tile_stride + tile_offset + local;
        if scatter {
            // Repeated Q/K heads have exactly one writer, including across tiles.
            if head.is_multiple_of(repeat) {
                destination[usize::cast_from(full)] = source[usize::cast_from(tile)];
            }
        } else {
            destination[usize::cast_from(tile)] = source[usize::cast_from(full)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_head_tiles_cover_each_source_head_once_on_scatter() {
        for (keys, values) in [(16, 16), (16, 48), (8, 32), (1, 16), (32, 64)] {
            let g = HeadGroups::new(keys, values).unwrap();
            let mut writers = vec![0; keys];
            for head in 0..values {
                if head.is_multiple_of(g.repeat()) {
                    writers[head / g.repeat()] += 1;
                }
            }
            assert_eq!(writers, vec![1; keys]);
            assert_eq!(g.tiles() * TILE_HEADS, values);
        }
        assert!(HeadGroups::new(0, 16).is_err());
        assert!(HeadGroups::new(16, 24).is_err());
    }
    #[test]
    fn strided_copy_bounds_include_last_tile_and_reject_short_rows() {
        let copy = HeadCopy {
            rows: 64,
            heads: 16,
            first_head: 32,
            repeat: 3,
            words_per_head: 64,
            full_stride: 5120,
            full_offset: 1024,
            tile_stride: 3072,
            tile_offset: 1024,
        };
        assert_eq!(
            copy.bounds().unwrap(),
            (63 * 5120 + 2048, 63 * 3072 + 2048, 64 * 16 * 64)
        );
        assert!(
            HeadCopy {
                full_stride: 2047,
                ..copy
            }
            .bounds()
            .is_err()
        );
        assert!(HeadCopy { repeat: 0, ..copy }.bounds().is_err());
    }
}
