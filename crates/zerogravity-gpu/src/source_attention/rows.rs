//! Full-call source attention using the original QK/softmax/PV kernels.
//! Query rows execute in order against one complete immutable KV extent. PV
//! retains the whole call contraction, including causally masked future words.
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};
mod prepared;
pub(crate) use prepared::kernels;

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
        let scores = self.heads * self.capacity;
        [
            scores, scores, scores, scores, self.heads, self.heads, self.heads,
        ]
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
/// Scratch is reusable across query rows; it is not a returned observation tape.
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
    let scores = g.heads * g.capacity;
    for row in 0..rows {
        let view = |handle: &Handle| {
            handle
                .clone()
                .offset_start((row * q * 4) as u64)
                .offset_end(((rows - row - 1) * q * 4) as u64)
        };
        let query = view(query);
        let output = view(output);
        unsafe {
            super::kernels::qk::launch_unchecked(
                client,
                CubeCount::Static(scores as u32, 1, 1),
                CubeDim::new_1d(64),
                arg(&query, q),
                arg(keys, tokens * kv),
                arg(scratch[0], scores),
                arg(scratch[1], scores),
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
                arg(scratch[1], scores),
                (base + row + 1) as u32,
                g.capacity,
            );
            super::kernels::reduce::launch_unchecked(
                client,
                CubeCount::Static(g.heads as u32, 1, 1),
                dim,
                arg(scratch[1], scores),
                arg(scratch[4], g.heads),
                arg(status, 1),
                tokens as u32,
                true,
                g.capacity,
            );
            super::kernels::exponentials::launch_unchecked(
                client,
                grid(scores),
                dim,
                arg(scratch[1], scores),
                arg(scratch[4], g.heads),
                arg(scratch[2], scores),
                arg(status, 1),
                tokens as u32,
                g.heads,
                g.capacity,
            );
            super::kernels::reduce::launch_unchecked(
                client,
                CubeCount::Static(g.heads as u32, 1, 1),
                dim,
                arg(scratch[2], scores),
                arg(scratch[5], g.heads),
                arg(status, 1),
                tokens as u32,
                false,
                g.capacity,
            );
            super::kernels::inverse::launch_unchecked(
                client,
                grid(g.heads),
                dim,
                arg(scratch[5], g.heads),
                arg(scratch[6], g.heads),
                arg(status, 1),
                g.heads,
            );
            super::kernels::probabilities::launch_unchecked(
                client,
                grid(scores),
                dim,
                arg(scratch[2], scores),
                arg(scratch[6], g.heads),
                arg(scratch[3], scores),
                arg(status, 1),
                tokens as u32,
                g.heads,
                g.capacity,
            );
            super::kernels::pv::launch_unchecked(
                client,
                grid(q),
                dim,
                arg(scratch[3], scores),
                arg(values, tokens * kv),
                arg(&output, q),
                arg(status, 1),
                arg(zero, 1),
                256usize,
                g.heads,
                g.kv_heads,
                g.capacity,
                tokens as u32,
                rows as u32,
                tokens as u32,
            );
        }
    }
    Ok(rows * 8)
}

#[cube(launch_unchecked)]
pub(crate) fn mask(scores: &mut [f32], visible: u32, #[comptime] capacity: usize) {
    let i = ABSOLUTE_POS;
    if i < scores.len() && i % capacity >= usize::cast_from(visible) {
        scores[i] = f32::reinterpret(0xff800000u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cold_geometry_and_actual_call_extents_are_distinct() {
        let g = Geometry::new(8, 2, 514).unwrap();
        assert_eq!(g.scratch_words(), [4112, 4112, 4112, 4112, 8, 8, 8]);
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
