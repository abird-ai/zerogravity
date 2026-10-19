//! Batch-native original source RMSNorm, residual and SiLU kernels.
//!
//! No new mathematics: one physical launch per original operator stage
//! for B independent rows instead of B launches, preserving every
//! source-mandated FP32 staging boundary. Qualification only: no
//! ResidentSession/BufferLease batching is implied by these operands.

use zerogravity_core::{Error, Result};

const MAX_BATCH: usize = 8;
const MAX_ELEMENTS: usize = 65_536;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub batch: usize,
    pub row: usize,
    pub group_width: usize,
}
impl Shape {
    pub fn validate(self) -> Result<usize> {
        if !(1..=MAX_BATCH).contains(&self.batch)
            || self.row == 0
            || self.row > 8192
            || self.group_width == 0
            || !self.group_width.is_multiple_of(32)
            || !self.row.is_multiple_of(self.group_width)
        {
            return Err(Error::Limit("batched source elementwise geometry".into()));
        }
        self.batch
            .checked_mul(self.row)
            .filter(|n| *n <= MAX_ELEMENTS && u32::try_from(*n).is_ok())
            .ok_or_else(|| Error::Limit("batched source elementwise byte capacity".into()))
    }
    pub fn groups(self) -> Result<usize> {
        let n = self.validate()?;
        Ok(n / self.group_width)
    }
}

#[cfg(feature = "qualification")]
pub(crate) fn qualify(device: cubecl::Device) -> Result<()> {
    use crate::source_accumulation::SourceAccumulation;
    use crate::{elementwise, rms_norm, silu, source_accumulation};
    use cubecl::prelude::*;

    let client = device.client();
    source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;

    fn grid(n: usize) -> CubeCount {
        CubeCount::Static((n as u32).div_ceil(32), 1, 1)
    }
    fn threads() -> CubeDim {
        CubeDim::new_1d(32)
    }
    fn arg(handle: &cubecl::server::Handle, n: usize) -> BufferArg {
        // Qualification holds all handles across terminal readback.
        unsafe { BufferArg::from_raw_parts(handle.clone(), n) }
    }
    fn read(client: &cubecl::client::Client, handle: cubecl::server::Handle) -> Result<Vec<f32>> {
        let raw = client
            .read_one(handle)
            .map_err(|e| Error::State(format!("batched GPU terminal read: {e}")))?;
        Ok(f32::from_bytes(&raw).to_vec())
    }
    fn check(actual: &[f32], expected: &[f32], name: &str, batch: usize, row: usize) -> Result<()> {
        if actual.len() != expected.len() {
            return Err(Error::State(format!(
                "{name} B{batch}: output count differs"
            )));
        }
        for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            if got.to_bits() != want.to_bits() {
                return Err(Error::State(format!(
                    "{name} B{batch} request {} lane {}: {:08x}!={:08x}",
                    i / row,
                    i % row,
                    got.to_bits(),
                    want.to_bits()
                )));
            }
        }
        Ok(())
    }
    fn run_norm(
        client: &cubecl::client::Client,
        input: &[f32],
        weights: &[f32],
        shape: Shape,
    ) -> Result<Vec<f32>> {
        let n = shape.validate()?;
        let groups = shape.groups()?;
        if input.len() != n || weights.len() != shape.group_width {
            return Err(Error::Malformed(
                "batched source RMS input/gamma shape".into(),
            ));
        }
        let x = client.create_from_slice(f32::as_bytes(input));
        let gamma = client.create_from_slice(f32::as_bytes(weights));
        let squares = client.empty(n * 4);
        let inv = client.empty(groups * 2 * 4);
        let output = client.empty(n * 4);
        unsafe {
            rms_norm::rms_squares::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(&x, n),
                arg(&squares, n),
                n,
            );
            rms_norm::rms_inverse::launch_unchecked(
                client,
                grid(groups),
                threads(),
                arg(&squares, n),
                arg(&inv, groups * 2),
                0.000001f32,
                shape.group_width,
                groups,
            );
            rms_norm::rms_round_inverse::launch_unchecked(
                client,
                grid(groups),
                threads(),
                arg(&inv, groups * 2),
                groups,
            );
            rms_norm::rms_apply::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(&x, n),
                arg(&gamma, shape.group_width),
                arg(&inv, groups * 2),
                arg(&output, n),
                shape.group_width,
                groups,
            );
        }
        read(client, output)
    }
    fn run_add(client: &cubecl::client::Client, left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
        let n = left.len();
        if n == 0 || n != right.len() || n > MAX_ELEMENTS {
            return Err(Error::Malformed("batched source residual shapes".into()));
        }
        let lhs = client.create_from_slice(f32::as_bytes(left));
        let rhs = client.create_from_slice(f32::as_bytes(right));
        let output = client.empty(n * 4);
        unsafe {
            elementwise::residual_add::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(&lhs, n),
                arg(&rhs, n),
                arg(&output, n),
                n,
            );
        }
        read(client, output)
    }
    fn run_silu(client: &cubecl::client::Client, inputs: &[f32]) -> Result<Vec<f32>> {
        let n = inputs.len();
        if n == 0 || n > MAX_ELEMENTS {
            return Err(Error::Malformed("batched source SiLU size".into()));
        }
        let input = client.create_from_slice(f32::as_bytes(inputs));
        let scratch = client.empty(n * silu::SCRATCH * 4);
        let output = client.empty(n * 4);
        // Each original source kernel stage is a separate launch for
        // all B rows, preserving the B1 FP32 materialization schedule.
        for stage in 0..12 {
            unsafe {
                silu::silu_step::launch_unchecked(
                    client,
                    grid(n),
                    threads(),
                    arg(&input, n),
                    arg(&scratch, n * silu::SCRATCH),
                    arg(&output, n),
                    n,
                    stage,
                );
            }
        }
        read(client, output)
    }

    for batch in [2, 4, 8] {
        let shape = Shape {
            batch,
            row: 64,
            group_width: 64,
        };
        let n = shape.validate()?;
        let x = (0..n)
            .map(|i| ((i * 97 + i / 64 * 113 + 7) % 509) as f32 / 64.0 - 3.5)
            .collect::<Vec<_>>();
        let y = (0..n)
            .map(|i| ((i * 41 + 23) % 367) as f32 / 49.0 - 3.0)
            .collect::<Vec<_>>();
        let gamma = (0..64)
            .map(|i| 0.3 + (i % 13) as f32 / 17.0)
            .collect::<Vec<_>>();

        let norm = run_norm(&client, &x, &gamma, shape)?;
        let added = run_add(&client, &x, &y)?;
        let activated = run_silu(&client, &x)?;
        for row in 0..batch {
            let start = row * 64;
            let end = start + 64;
            let b1 = Shape {
                batch: 1,
                row: 64,
                group_width: 64,
            };
            let norm_one = run_norm(&client, &x[start..end], &gamma, b1)?;
            let add_one = run_add(&client, &x[start..end], &y[start..end])?;
            let silu_one = run_silu(&client, &x[start..end])?;
            check(&norm[start..end], &norm_one, "RMSNorm", batch, 64)?;
            check(&added[start..end], &add_one, "ResidualAdd", batch, 64)?;
            check(&activated[start..end], &silu_one, "SiLU", batch, 64)?;
        }
        println!(
            "P12_ELEMENTWISE_BATCH_GPU_PASS B={batch} row=64 \
             norm_bits={n} add_bits={n} silu_bits={n} \
             fused_stage_launches=17 B1_stage_launches={}",
            17 * batch
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_original_elementwise_shapes_and_groups() {
        let good = Shape {
            batch: 8,
            row: 64,
            group_width: 32,
        };
        assert_eq!(good.validate().unwrap(), 512);
        assert_eq!(good.groups().unwrap(), 16);
        let one = Shape {
            batch: 1,
            row: 64,
            group_width: 64,
        };
        assert_eq!(one.groups().unwrap(), 1);
        for invalid in [
            Shape { batch: 0, ..one },
            Shape { batch: 9, ..one },
            Shape { row: 0, ..one },
            Shape { row: 8193, ..one },
            Shape {
                group_width: 0,
                ..one
            },
            Shape {
                group_width: 31,
                ..one
            },
            Shape {
                group_width: 96,
                ..one
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
}
