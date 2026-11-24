//! Shared padded GDN launch primitive; caller owns admission, flight and publication.
use super::*;
use cubecl::{client::Client, server::Handle};
use zerogravity_core::{Error, Result};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    rows: usize,
    heads: usize,
}
impl Geometry {
    pub(crate) fn new(rows: usize, heads: usize) -> Result<Self> {
        if !(1..=64).contains(&rows) || !(1..=48).contains(&heads) || !heads.is_multiple_of(16) {
            return Err(Error::Unsupported(
                "source chunk GDN row/head geometry".into(),
            ));
        }
        Ok(Self { rows, heads })
    }
    pub(crate) fn input_words(self) -> [usize; 6] {
        let v = self.rows * self.heads * 128;
        [
            v,
            v,
            v,
            self.rows * self.heads,
            self.rows * self.heads,
            self.heads * 16384,
        ]
    }
    pub(crate) fn output_words(self) -> [usize; 19] {
        let h = self.heads;
        [
            h * 8192,
            h * 8192,
            h * 8192,
            h * 8192,
            h * 64,
            h * 4096,
            h * 4096,
            h * 4096,
            h * 8192,
            h * 8192,
            h * 8192,
            h * 8192,
            h * 8192,
            h,
            h * 8192,
            h * 8192,
            h * 8192,
            h * 16384,
            self.rows * h * 128,
        ]
    }
    pub(crate) fn scratch_words(self) -> [usize; 2] {
        [self.heads * 64; 2]
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &Client,
    g: Geometry,
    guarded_triangular: bool,
    i: [&Handle; 6],
    o: [&Handle; 19],
    s: [&Handle; 2],
    normal: &Handle,
    rare: &Handle,
    status_handle: &Handle,
) -> Result<usize> {
    use crate::resident_program::BufferRange;
    let mut writes = Vec::with_capacity(22);
    for (handle, bytes) in o
        .iter()
        .copied()
        .zip(g.output_words().map(|n| n * 4))
        .chain(s.iter().copied().zip(g.scratch_words().map(|n| n * 4)))
        .chain([(status_handle, 4)])
    {
        let range = BufferRange::new(handle)?;
        range.require_exact_len(bytes)?;
        if handle.service != client.service_id()
            || writes.iter().any(|old: &BufferRange| old.overlaps(&range))
        {
            return Err(Error::State(
                "source chunk GDN distinct same-service writes".into(),
            ));
        }
        writes.push(range);
    }
    for (handle, bytes) in i
        .iter()
        .copied()
        .zip(g.input_words().map(|n| n * 4))
        .chain([(normal, 256 * 4), (rare, 128 * 8)])
    {
        let range = BufferRange::new(handle)?;
        range.require_exact_len(bytes)?;
        if handle.service != client.service_id()
            || writes.iter().any(|write| write.overlaps(&range))
        {
            return Err(Error::State(
                "source chunk GDN immutable same-service operands".into(),
            ));
        }
    }
    let h = g.heads;
    let rows = g.rows as u32;
    let dim = CubeDim::new_1d(32);
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    let arg = |handle: &Handle, n: usize, _word_bytes: usize| -> BufferArg {
        // Exact extents/service/alias checked above, before the first launch.
        // The submitting caller retains these original handles through completion.
        unsafe { BufferArg::from_raw_parts(handle.clone(), n) }
    };
    let f = |handle: &Handle, n: usize| arg(handle, n, 4);
    let status_arg = || f(status_handle, 1);
    unsafe {
        normalize::launch_unchecked(
            client,
            grid(h * 64),
            dim,
            f(i[0], g.rows * h * 128),
            f(i[1], g.rows * h * 128),
            f(o[0], h * 8192),
            f(o[1], h * 8192),
            status_arg(),
            rows,
            h,
        );
        beta_vectors::launch_unchecked(
            client,
            grid(h * 8192),
            dim,
            f(o[1], h * 8192),
            f(i[2], g.rows * h * 128),
            f(i[3], g.rows * h),
            f(o[2], h * 8192),
            f(o[3], h * 8192),
            status_arg(),
            rows,
            h,
        );
        cumulative::launch_unchecked(
            client,
            grid(h),
            dim,
            f(i[4], g.rows * h),
            arg(rare, 128, 8),
            f(o[4], h * 64),
            status_arg(),
            rows,
            h,
        );
        decays::launch_unchecked(
            client,
            grid(h * 4096),
            dim,
            f(o[4], h * 64),
            f(normal, 256),
            arg(rare, 128, 8),
            f(o[5], h * 4096),
            f(s[0], h * 64),
            f(s[1], h * 64),
            f(o[13], h),
            status_arg(),
            h,
        );
        gram::launch_unchecked(
            client,
            grid(h * 4096),
            dim,
            f(o[0], h * 8192),
            f(o[1], h * 8192),
            f(o[2], h * 8192),
            f(o[5], h * 4096),
            f(o[6], h * 4096),
            f(o[7], h * 4096),
            status_arg(),
            h,
        );
        decay_vectors::launch_unchecked(
            client,
            grid(h * 8192),
            dim,
            f(o[0], h * 8192),
            f(o[1], h * 8192),
            f(o[2], h * 8192),
            f(s[0], h * 64),
            f(s[1], h * 64),
            f(o[8], h * 8192),
            f(o[11], h * 8192),
            f(o[12], h * 8192),
            status_arg(),
            h,
        );
        let triangular = if guarded_triangular {
            triangular_guarded::launch_unchecked
        } else {
            triangular::launch_unchecked
        };
        triangular(
            client,
            grid(h * 128),
            dim,
            f(o[6], h * 4096),
            f(o[3], h * 8192),
            f(o[8], h * 8192),
            f(o[9], h * 8192),
            f(o[10], h * 8192),
            status_arg(),
            h,
        );
        prior_products::launch_unchecked(
            client,
            grid(h * 8192),
            dim,
            f(o[10], h * 8192),
            f(o[11], h * 8192),
            f(i[5], h * 16384),
            f(o[9], h * 8192),
            f(o[14], h * 8192),
            f(o[15], h * 8192),
            status_arg(),
            h,
        );
        core_product::launch_unchecked(
            client,
            grid(h * 8192),
            dim,
            f(o[7], h * 4096),
            f(o[14], h * 8192),
            f(o[15], h * 8192),
            f(o[16], h * 8192),
            f(o[18], g.rows * h * 128),
            status_arg(),
            rows,
            h,
        );
        state_product::launch_unchecked(
            client,
            grid(h * 16384),
            dim,
            f(o[12], h * 8192),
            f(o[14], h * 8192),
            f(i[5], h * 16384),
            f(o[13], h),
            f(o[17], h * 16384),
            status_arg(),
            h,
        );
    }
    Ok(10)
}
