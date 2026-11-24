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

use crate::dense::{Command, DensePrecision, Gemm, Transpose, UnitLowerSolve};

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
    launch_with_dense(
        client,
        g,
        guarded_triangular,
        i,
        o,
        s,
        normal,
        rare,
        status_handle,
        false,
        None::<fn(Command<'_>) -> Result<()>>,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_with_dense(
    client: &Client,
    g: Geometry,
    guarded_triangular: bool,
    i: [&Handle; 6],
    o: [&Handle; 19],
    s: [&Handle; 2],
    normal: &Handle,
    rare: &Handle,
    status_handle: &Handle,
    native_solve: bool,
    mut dense: Option<impl FnMut(Command<'_>) -> Result<()>>,
) -> Result<usize> {
    use crate::resident_program::BufferRange;
    if native_solve && dense.is_none() {
        return Err(Error::State(
            "native triangular solve requires dense provider".into(),
        ));
    }
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
    let product = |m, n, k, left, right| Gemm {
        m,
        n,
        k,
        batches: h,
        left,
        right,
        precision: DensePrecision::F32,
    };
    let nn = Transpose::None;
    let tr = Transpose::Transposed;
    let mut count = 10;
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
        if let Some(dense) = &mut dense {
            dense(Command::Gemm {
                geometry: product(64, 64, 128, nn, tr),
                left: o[2],
                right: o[1],
                output: o[6],
            })?;
            dense(Command::Gemm {
                geometry: product(64, 64, 128, nn, tr),
                left: o[0],
                right: o[1],
                output: o[7],
            })?;
            gram_epilogue::launch_unchecked(
                client,
                grid(h * 4096),
                dim,
                f(o[6], h * 4096),
                f(o[7], h * 4096),
                f(o[5], h * 4096),
                status_arg(),
            );
            count += 2;
        } else {
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
        }
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
        if native_solve {
            let dense = dense.as_mut().expect("validated dense provider");
            let geometry = UnitLowerSolve {
                order: 64,
                rhs_columns: 128,
                batches: h,
            };
            crate::program_kernels::checked_copy_words(h, 8192, 8192, 8192)?;
            for (input, rhs) in [(o[3], o[9]), (o[8], o[10])] {
                crate::program_kernels::copy_strided_words::launch_unchecked(
                    client,
                    grid(h * 8192),
                    dim,
                    f(input, h * 8192),
                    f(rhs, h * 8192),
                    status_arg(),
                    0u32,
                    0u32,
                    8192u32,
                    8192u32,
                    h,
                    8192usize,
                    true,
                );
                dense(Command::Solve {
                    geometry,
                    lower: o[6],
                    rhs,
                })?;
                crate::program_kernels::certify_finite_strided::launch_unchecked(
                    client,
                    grid(h * 8192),
                    dim,
                    f(rhs, h * 8192),
                    status_arg(),
                    0u32,
                    8192u32,
                    h,
                    8192usize,
                );
            }
            count += 5;
        } else {
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
        }
        if let Some(dense) = &mut dense {
            dense(Command::Gemm {
                geometry: product(64, 128, 128, nn, nn),
                left: o[10],
                right: i[5],
                output: o[14],
            })?;
            dense(Command::Gemm {
                geometry: product(64, 128, 128, nn, nn),
                left: o[11],
                right: i[5],
                output: o[15],
            })?;
            prior_epilogue::launch_unchecked(
                client,
                grid(h * 8192),
                dim,
                f(o[9], h * 8192),
                f(o[14], h * 8192),
                f(o[15], h * 8192),
                status_arg(),
            );
            count += 2;
        } else {
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
        }
        if let Some(dense) = &mut dense {
            dense(Command::Gemm {
                geometry: product(64, 128, 64, nn, nn),
                left: o[7],
                right: o[14],
                output: o[16],
            })?;
            core_epilogue::launch_unchecked(
                client,
                grid(h * 8192),
                dim,
                f(o[15], h * 8192),
                f(o[16], h * 8192),
                f(o[18], g.rows * h * 128),
                status_arg(),
                rows,
                h,
            );
            count += 1;
        } else {
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
        }
        if let Some(dense) = &mut dense {
            dense(Command::Gemm {
                geometry: product(128, 128, 64, tr, nn),
                left: o[12],
                right: o[14],
                output: o[17],
            })?;
            state_epilogue::launch_unchecked(
                client,
                grid(h * 16384),
                dim,
                f(i[5], h * 16384),
                f(o[13], h),
                f(o[17], h * 16384),
                status_arg(),
            );
            count += 1;
        } else {
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
    }
    Ok(count)
}
