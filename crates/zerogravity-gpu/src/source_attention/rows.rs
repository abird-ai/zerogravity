//! Full-call source attention using the original QK/softmax/PV kernels.
//! Independent query tiles share one complete immutable KV extent. Every query
//! retains source reduction order and the full PV contraction, including future zeros.
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};
mod prepared;
pub(crate) use prepared::kernels;

const QUERY_TILE_ROWS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    heads: usize,
    kv_heads: usize,
    capacity: usize,
}
impl Geometry {
    pub(crate) fn new(heads: usize, kv_heads: usize, capacity: usize) -> Result<Self> {
        if !(1..=24).contains(&heads)
            || !(1..=4).contains(&kv_heads)
            || !heads.is_multiple_of(kv_heads)
            || !(1..=32768).contains(&capacity)
        {
            return Err(Error::Unsupported(
                "source attention complete-call geometry".into(),
            ));
        }
        Ok(Self {
            heads,
            kv_heads,
            capacity,
        })
    }
    pub(crate) fn scratch_words(self) -> [usize; 7] {
        let groups = self.heads * QUERY_TILE_ROWS;
        let scores = groups * self.capacity;
        [scores, scores, scores, scores, groups, groups, groups]
    }
    fn tokens(self, base: usize, rows: usize) -> Result<usize> {
        if !(1..=64).contains(&rows) {
            return Err(Error::Limit("source attention call rows".into()));
        }
        base.checked_add(rows)
            .filter(|n| *n <= self.capacity)
            .ok_or_else(|| Error::Limit("source attention call frontier".into()))
    }
}

/// Exact actual views. Caller admits and retains all handles through completion.
/// Scratch is reusable across query tiles; it is not a returned observation tape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &Client,
    g: Geometry,
    base: usize,
    rows: usize,
    query: &Handle,
    keys: &Handle,
    values: &Handle,
    output: &Handle,
    scratch: [&Handle; 7],
    status: &Handle,
    zero: &Handle,
) -> Result<usize> {
    use crate::resident_program::BufferRange;
    let tokens = g.tokens(base, rows)?;
    let q = g.heads * 256;
    let kv = g.kv_heads * 256;
    let mut writes = Vec::with_capacity(9);
    for (handle, words) in scratch
        .into_iter()
        .zip(g.scratch_words())
        .chain([(output, rows * q), (status, 1)])
    {
        let range = BufferRange::new(handle)?;
        range.require_exact_len(words * 4)?;
        if handle.service != client.service_id()
            || writes
                .iter()
                .any(|prior: &BufferRange| prior.overlaps(&range))
        {
            return Err(Error::State(
                "source attention complete-call distinct writes".into(),
            ));
        }
        writes.push(range);
    }
    for (handle, words) in [
        (query, rows * q),
        (keys, tokens * kv),
        (values, tokens * kv),
        (zero, 1),
    ] {
        let range = BufferRange::new(handle)?;
        range.require_exact_len(words * 4)?;
        if handle.service != client.service_id()
            || writes.iter().any(|write| write.overlaps(&range))
        {
            return Err(Error::State(
                "source attention complete-call readonly inputs".into(),
            ));
        }
    }
    let arg = |handle: &Handle, words| unsafe { BufferArg::from_raw_parts(handle.clone(), words) };
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    let dim = CubeDim::new_1d(32);
    for row in (0..rows).step_by(QUERY_TILE_ROWS) {
        let batch = (rows - row).min(QUERY_TILE_ROWS);
        let groups = batch * g.heads;
        let scores = groups * g.capacity;
        let words = [scores, scores, scores, scores, groups, groups, groups];
        let capacity_words = g.scratch_words();
        let work: [Handle; 7] = core::array::from_fn(|i| {
            scratch[i]
                .clone()
                .offset_end(((capacity_words[i] - words[i]) * 4) as u64)
        });
        let view = |handle: &Handle| {
            handle
                .clone()
                .offset_start((row * q * 4) as u64)
                .offset_end(((rows - row - batch) * q * 4) as u64)
        };
        let query = view(query);
        let output = view(output);
        unsafe {
            // Only actual KV tokens need dot-product blocks. Mask fills capacity tails.
            super::kernels::qk::launch_unchecked(
                client,
                CubeCount::Static(tokens as u32, groups as u32, 1),
                CubeDim::new_1d(64),
                arg(&query, batch * q),
                arg(keys, tokens * kv),
                arg(&work[0], scores),
                arg(&work[1], scores),
                arg(status, 1),
                arg(zero, 1),
                256usize,
                g.heads,
                g.kv_heads,
                g.capacity,
                0.0625f32,
                tokens as u32,
                0u32,
            );
            mask::launch_unchecked(
                client,
                grid(scores),
                dim,
                arg(&work[1], scores),
                arg(&work[0], scores),
                (base + row + 1) as u32,
                tokens as u32,
                g.heads,
                g.capacity,
            );
            super::kernels::reduce::launch_unchecked(
                client,
                CubeCount::Static(groups as u32, 1, 1),
                dim,
                arg(&work[1], scores),
                arg(&work[4], groups),
                arg(status, 1),
                tokens as u32,
                true,
                g.capacity,
            );
            super::kernels::exponentials::launch_unchecked(
                client,
                grid(scores),
                dim,
                arg(&work[1], scores),
                arg(&work[4], groups),
                arg(&work[2], scores),
                arg(status, 1),
                tokens as u32,
                g.heads * QUERY_TILE_ROWS,
                g.capacity,
            );
            super::kernels::reduce::launch_unchecked(
                client,
                CubeCount::Static(groups as u32, 1, 1),
                dim,
                arg(&work[2], scores),
                arg(&work[5], groups),
                arg(status, 1),
                tokens as u32,
                false,
                g.capacity,
            );
            super::kernels::inverse::launch_unchecked(
                client,
                grid(groups),
                dim,
                arg(&work[5], groups),
                arg(&work[6], groups),
                arg(status, 1),
                g.heads * QUERY_TILE_ROWS,
            );
            super::kernels::probabilities::launch_unchecked(
                client,
                grid(scores),
                dim,
                arg(&work[2], scores),
                arg(&work[6], groups),
                arg(&work[3], scores),
                arg(status, 1),
                tokens as u32,
                g.heads * QUERY_TILE_ROWS,
                g.capacity,
            );
            super::kernels::pv::launch_unchecked(
                client,
                grid(batch * q),
                dim,
                arg(&work[3], scores),
                arg(values, tokens * kv),
                arg(&output, batch * q),
                arg(status, 1),
                arg(zero, 1),
                256usize,
                g.heads,
                g.kv_heads,
                g.capacity,
                tokens as u32,
                // Original full-call GEMM schedule, never the physical tile width.
                rows as u32,
                tokens as u32,
            );
        }
    }
    Ok(rows.div_ceil(QUERY_TILE_ROWS) * 8)
}

#[cube(launch_unchecked)]
pub(crate) fn mask(
    scores: &mut [f32],
    dots: &mut [f32],
    visible: u32,
    active: u32,
    #[comptime] heads: usize,
    #[comptime] capacity: usize,
) {
    let i = ABSOLUTE_POS;
    if i < scores.len() {
        let token = i % capacity;
        let row = i / (heads * capacity);
        if token >= usize::cast_from(active) {
            dots[i] = 0.0f32;
            scores[i] = f32::reinterpret(0xff800000u32);
        } else if token >= usize::cast_from(visible) + row {
            scores[i] = f32::reinterpret(0xff800000u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cold_geometry_and_actual_call_extents_are_distinct() {
        let g = Geometry::new(8, 2, 514).unwrap();
        assert_eq!(
            g.scratch_words(),
            [65792, 65792, 65792, 65792, 128, 128, 128]
        );
        assert_eq!(g.tokens(0, 47).unwrap(), 47);
        assert_eq!(g.tokens(47, 64).unwrap(), 111);
        assert_eq!(g.tokens(513, 1).unwrap(), 514);
        for (base, rows) in [(0, 0), (0, 65), (514, 1), (usize::MAX, 2)] {
            assert!(g.tokens(base, rows).is_err());
        }
        for (heads, kv, capacity) in [
            (0, 2, 514),
            (25, 1, 514),
            (8, 3, 514),
            (8, 0, 514),
            (8, 2, 0),
            (8, 2, 32769),
        ] {
            assert!(Geometry::new(heads, kv, capacity).is_err());
        }
    }
}
