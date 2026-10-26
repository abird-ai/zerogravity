//! Batch-native source GDN recurrent update on independent request rows.
//!
//! Reuses exactly the resident single-token decay/dot/reduce/beta/update/
//! scale kernels. One physical launch per numerical stage handles B rows,
//! with qkv in canonical [Q for all requests][K all][V all] order.
//! Candidate recurrent state stays row-private. No StateRoot publication.

use zerogravity_core::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) struct Shape {
    batch: usize,
    heads: usize,
    width: usize,
}
impl Shape {
    fn geometry(self) -> Result<(usize, usize)> {
        if !(1..=8).contains(&self.batch)
            || self.heads == 0
            || self.heads > 16
            || self.width == 0
            || self.width > 256
            || !self.width.is_multiple_of(32)
        {
            return Err(Error::Limit("batched GDN recurrence geometry".into()));
        }
        let values = self
            .batch
            .checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.width))
            .ok_or_else(|| Error::Limit("batched GDN value count".into()))?;
        let elements = values
            .checked_mul(self.width)
            .filter(|n| *n <= 1_048_576)
            .ok_or_else(|| Error::Limit("batched GDN state byte budget".into()))?;
        Ok((values, elements))
    }
    fn validate(self, qkv: &[f32], state: &[f32], coeff: &[f32]) -> Result<()> {
        let (values, elements) = self.geometry()?;
        if qkv.len() != values * 3
            || state.len() != elements
            || coeff.len() != self.batch * self.heads * 3
            || qkv.iter().chain(state).chain(coeff).any(|v| !v.is_finite())
        {
            return Err(Error::Malformed("batched GDN source buffers/finite".into()));
        }
        Ok(())
    }
}
#[cfg(feature = "qualification")]
pub fn qualify(device: cubecl::Device) -> Result<()> {
    use crate::gdn_recurrent;
    use cubecl::prelude::*;

    let client = device.client();
    fn grid(n: usize) -> CubeCount {
        CubeCount::Static((n as u32).div_ceil(32), 1, 1)
    }
    fn threads() -> CubeDim {
        CubeDim::new_1d(32)
    }
    fn buffer(handle: &cubecl::server::Handle, n: usize) -> BufferArg {
        // All source input/scratch/state handles are held across complete
        // terminal readback. Caller must still own actual ResourceLeases.
        unsafe { BufferArg::from_raw_parts(handle.clone(), n) }
    }
    fn read(client: &cubecl::client::Client, h: cubecl::server::Handle) -> Result<Vec<f32>> {
        let data = client
            .read_one(h)
            .map_err(|e| Error::State(format!("batched GDN physical terminal readback: {e}")))?;
        Ok(f32::from_bytes(&data).to_vec())
    }
    struct DotOperands<'a> {
        state: &'a cubecl::server::Handle,
        qkv: &'a cubecl::server::Handle,
        lanes: &'a cubecl::server::Handle,
        reduced: &'a cubecl::server::Handle,
        output: &'a cubecl::server::Handle,
    }
    fn dot(
        client: &cubecl::client::Client,
        handles: DotOperands<'_>,
        width: usize,
        values: usize,
        vector_offset: usize,
    ) {
        let DotOperands {
            state,
            qkv,
            lanes,
            reduced,
            output,
        } = handles;
        unsafe {
            for block in 0..width / 32 {
                gdn_recurrent::dot_lanes::launch_unchecked(
                    client,
                    grid(values * 32),
                    threads(),
                    buffer(state, values * width),
                    buffer(qkv, values * 3),
                    buffer(lanes, values * 32),
                    width,
                    values,
                    vector_offset,
                    block,
                );
            }
            for level in 0..5 {
                let source = if level % 2 == 0 { lanes } else { reduced };
                let dest = if level == 4 {
                    output
                } else if level % 2 == 0 {
                    reduced
                } else {
                    lanes
                };
                let src_len = if level % 2 == 0 {
                    values * 32
                } else {
                    values * 16
                };
                let dst_len = if level == 4 {
                    values
                } else if level % 2 == 0 {
                    values * 16
                } else {
                    values * 32
                };
                gdn_recurrent::dot_reduce::launch_unchecked(
                    client,
                    grid(values * (16 >> level)),
                    threads(),
                    buffer(source, src_len),
                    buffer(dest, dst_len),
                    values,
                    level,
                );
            }
        }
    }
    fn execute(
        client: &cubecl::client::Client,
        shape: Shape,
        qkv: &[f32],
        state: &[f32],
        coeff: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        shape.validate(qkv, state, coeff)?;
        let (values, elements) = shape.geometry()?;
        let input = client.create_from_slice(f32::as_bytes(qkv));
        let previous = client.create_from_slice(f32::as_bytes(state));
        let coeff = client.create_from_slice(f32::as_bytes(coeff));
        let decayed = client.empty(elements * 4);
        let lanes = client.empty(values * 32 * 4);
        let reduced = client.empty(values * 16 * 4);
        let delta = client.empty(values * 4);
        let proposal = client.empty(elements * 4);
        let output = client.empty(values * 4);
        unsafe {
            gdn_recurrent::decay::launch_unchecked(
                client,
                grid(elements),
                threads(),
                buffer(&previous, elements),
                buffer(&coeff, shape.batch * shape.heads * 3),
                buffer(&decayed, elements),
                shape.width,
                elements,
            );
        }
        dot(
            client,
            DotOperands {
                state: &decayed,
                qkv: &input,
                lanes: &lanes,
                reduced: &reduced,
                output: &delta,
            },
            shape.width,
            values,
            values,
        );
        unsafe {
            gdn_recurrent::residual::launch_unchecked(
                client,
                grid(values),
                threads(),
                buffer(&input, values * 3),
                buffer(&delta, values),
                values,
            );
            gdn_recurrent::beta::launch_unchecked(
                client,
                grid(values),
                threads(),
                buffer(&coeff, shape.batch * shape.heads * 3),
                buffer(&delta, values),
                shape.width,
                values,
            );
            gdn_recurrent::update::launch_unchecked(
                client,
                grid(elements),
                threads(),
                buffer(&decayed, elements),
                buffer(&input, values * 3),
                buffer(&delta, values),
                buffer(&proposal, elements),
                shape.width,
                values,
                elements,
            );
        }
        dot(
            client,
            DotOperands {
                state: &proposal,
                qkv: &input,
                lanes: &lanes,
                reduced: &reduced,
                output: &output,
            },
            shape.width,
            values,
            0,
        );
        unsafe {
            gdn_recurrent::scale::launch_unchecked(
                client,
                grid(values),
                threads(),
                buffer(&coeff, shape.batch * shape.heads * 3),
                buffer(&output, values),
                shape.width,
                values,
            );
        }
        let output = read(client, output)?;
        let state = read(client, proposal)?;
        // 1 decay + (1 dot + 5 reduce) * 2 + 3 update ops + 1 scale.
        Ok((output, state, 17))
    }
    fn compare(actual: &[f32], reference: &[f32], name: &str, b: usize, lane: usize) -> Result<()> {
        if actual.len() != reference.len() {
            return Err(Error::State(format!(
                "batched {name} B{b} row count mismatch"
            )));
        }
        for (i, (&got, &want)) in actual.iter().zip(reference).enumerate() {
            if got.to_bits() != want.to_bits() {
                return Err(Error::State(format!(
                    "batched {name} B{b} lane {lane} element {i}: {:08x}!={:08x}",
                    got.to_bits(),
                    want.to_bits()
                )));
            }
        }
        Ok(())
    }
    const WIDTH: usize = 32;
    const HEADS: usize = 2;
    for batch in [2, 4, 8] {
        let shape = Shape {
            batch,
            heads: HEADS,
            width: WIDTH,
        };
        let (values, elements) = shape.geometry()?;
        let one_values = HEADS * WIDTH;
        let one_elements = HEADS * WIDTH * WIDTH;
        let qkv = (0..values * 3)
            .map(|i| ((i * 37 + i / (values) * 71 + 17) % 311) as f32 / 128.0 - 1.2)
            .collect::<Vec<_>>();
        let state = (0..elements)
            .map(|i| ((i * 19 + i / one_elements * 13) % 191) as f32 / 1024.0 - 0.09)
            .collect::<Vec<_>>();
        let coefficients = (0..batch * HEADS)
            .flat_map(|i| {
                [
                    0.42 + (i % 7) as f32 / 40.0,
                    0.50 + (i % 3) as f32 / 20.0,
                    0.125 + (i % 5) as f32 / 70.0,
                ]
            })
            .collect::<Vec<_>>();
        let (output, next, physical) = execute(&client, shape, &qkv, &state, &coefficients)?;
        assert_eq!(physical, 17);
        for request in 0..batch {
            let off = request * one_values;
            let mut isolated_qkv = Vec::with_capacity(one_values * 3);
            for role in 0..3 {
                isolated_qkv
                    .extend_from_slice(&qkv[role * values + off..role * values + off + one_values]);
            }
            let isolated_state = &state[request * one_elements..(request + 1) * one_elements];
            let isolated_coeff = &coefficients[request * HEADS * 3..(request + 1) * HEADS * 3];
            let (reference_output, reference_state, b1) = execute(
                &client,
                Shape {
                    batch: 1,
                    heads: HEADS,
                    width: WIDTH,
                },
                &isolated_qkv,
                isolated_state,
                isolated_coeff,
            )?;
            assert_eq!(b1, physical);
            compare(
                &output[off..off + one_values],
                &reference_output,
                "GDN output",
                batch,
                request,
            )?;
            compare(
                &next[request * one_elements..(request + 1) * one_elements],
                &reference_state,
                "GDN state",
                batch,
                request,
            )?;
        }
        println!(
            "P12_GDN_BATCH_GPU_PASS B={batch} recurrent_heads={HEADS} width={WIDTH} \
             exact_output_bits={values} exact_state_bits={elements} \
             fused_stage_launches=17 B1_stage_launches={}",
            physical * batch
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_batch_recurrence_geometry_is_source_shape_exact() {
        let valid = Shape {
            batch: 2,
            heads: 2,
            width: 32,
        };
        assert!(
            valid
                .validate(&vec![1.0; 384], &vec![0.0; 4096], &[0.5; 12])
                .is_ok()
        );
        for wrong in [
            Shape { batch: 0, ..valid },
            Shape { batch: 9, ..valid },
            Shape { heads: 0, ..valid },
            Shape { heads: 17, ..valid },
            Shape { width: 0, ..valid },
            Shape { width: 31, ..valid },
            Shape {
                width: 257,
                ..valid
            },
        ] {
            assert!(wrong.geometry().is_err());
        }
        assert!(
            valid
                .validate(&[1.0; 3], &vec![0.0; 4096], &[0.5; 12])
                .is_err()
        );
        assert!(
            valid
                .validate(&vec![1.0; 384], &vec![0.0; 4095], &[0.5; 12])
                .is_err()
        );
        assert!(
            valid
                .validate(&vec![1.0; 384], &vec![0.0; 4096], &[0.5; 11])
                .is_err()
        );
        let mut corrupt = vec![0.0; 4096];
        corrupt[3] = f32::INFINITY;
        assert!(
            valid
                .validate(&vec![1.0; 384], &corrupt, &[0.5; 12])
                .is_err()
        );
    }
}
