//! Physically packed multi-request source QK -> softmax -> AV.
//!
//! All original attention math kernels are reused, with B independent
//! sessions presented as disjoint heads. One GPU pack converts private
//! [request,token,kv_head,width] KV into the source's canonical
//! [token,request,kv_head,width] layout. All numerical stages launch
//! once over B heads, not once per request. This is an *operator chain*
//! fixture, not a published multi-session model/StateRoot transaction.

use zerogravity_core::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) struct Shape {
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    width: usize,
    capacity: usize,
    valid: usize,
}
impl Shape {
    fn validate(self) -> Result<(usize, usize, usize)> {
        if !(1..=8).contains(&self.batch)
            || self.query_heads == 0
            || self.query_heads > 16
            || self.kv_heads == 0
            || self.kv_heads > 16
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.width == 0
            || self.width > 256
            || !self.width.is_multiple_of(32)
            || self.capacity == 0
            || self.capacity > 512
            || !self.capacity.is_multiple_of(32)
            || self.valid == 0
            || self.valid > self.capacity
        {
            return Err(Error::Limit("batched source attention geometry".into()));
        }
        let q = self
            .batch
            .checked_mul(self.query_heads)
            .and_then(|v| v.checked_mul(self.width))
            .ok_or_else(|| Error::Limit("batch Q capacity overflow".into()))?;
        let k = self
            .batch
            .checked_mul(self.capacity)
            .and_then(|v| v.checked_mul(self.kv_heads))
            .and_then(|v| v.checked_mul(self.width))
            .ok_or_else(|| Error::Limit("batch KV capacity overflow".into()))?;
        let score = self
            .batch
            .checked_mul(self.query_heads)
            .and_then(|v| v.checked_mul(self.capacity))
            .ok_or_else(|| Error::Limit("batch QK capacity overflow".into()))?;
        if q.checked_add(k).is_none_or(|v| v > 1_048_576) || score > 1_048_576 {
            return Err(Error::Limit("batch attention diagnostic bytes".into()));
        }
        Ok((q, k, score))
    }
}

#[cfg(feature = "qualification")]
use cubecl::prelude::*;

#[cfg(feature = "qualification")]
#[cube(launch_unchecked)]
pub(crate) fn pack_kv(
    source_k: &[f32],
    source_v: &[f32],
    packed_k: &mut [f32],
    packed_v: &mut [f32],
    #[comptime] batch: usize,
    #[comptime] tokens: usize,
    #[comptime] slab: usize,
) {
    let i = ABSOLUTE_POS;
    if i < batch * tokens * slab {
        let request = i / (tokens * slab);
        let within = i % (tokens * slab);
        let token = within / slab;
        let lane = within % slab;
        let dst = (token * batch + request) * slab + lane;
        packed_k[dst] = source_k[i];
        packed_v[dst] = source_v[i];
    }
}
#[cfg(feature = "qualification")]
#[cube(launch_unchecked)]
pub(crate) fn concatenate(
    first: &[f32],
    second: &[f32],
    output: &mut [f32],
    #[comptime] left: usize,
    #[comptime] right: usize,
) {
    let i = ABSOLUTE_POS;
    if i < left + right {
        output[i] = if i < left { first[i] } else { second[i - left] };
    }
}

#[cfg(feature = "qualification")]
pub(crate) fn qualify(device: cubecl::Device) -> Result<()> {
    use crate::{
        attention_qk as qk, attention_softmax as sm, attention_value as av, resident_bindings,
    };
    fn grid(n: usize) -> CubeCount {
        CubeCount::Static((n as u32).div_ceil(32), 1, 1)
    }
    fn threads() -> CubeDim {
        CubeDim::new_1d(32)
    }
    fn arg(h: &cubecl::server::Handle, n: usize) -> BufferArg {
        // Source and all scratch/candidate owners stay alive through
        // actual physical readback; production must use BufferLeases.
        unsafe { BufferArg::from_raw_parts(h.clone(), n) }
    }
    let client = device.client();
    fn flatten(
        client: &cubecl::client::Client,
        source_q: &[f32],
        source_k: &[f32],
        source_v: &[f32],
        shape: Shape,
    ) -> Result<(
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    )> {
        let (q, k, _) = shape.validate()?;
        if source_q.len() != q || source_k.len() != k || source_v.len() != k {
            return Err(Error::Malformed(
                "batch source Q/K/V buffer geometry".into(),
            ));
        }
        let q_handle = client.create_from_slice(f32::as_bytes(source_q));
        let raw_k = client.create_from_slice(f32::as_bytes(source_k));
        let raw_v = client.create_from_slice(f32::as_bytes(source_v));
        let packed_k = client.empty(k * 4);
        let packed_v = client.empty(k * 4);
        let slab = shape.kv_heads * shape.width;
        unsafe {
            pack_kv::launch_unchecked(
                client,
                grid(k),
                threads(),
                arg(&raw_k, k),
                arg(&raw_v, k),
                arg(&packed_k, k),
                arg(&packed_v, k),
                shape.batch,
                shape.capacity,
                slab,
            );
        }
        Ok((q_handle, packed_k, packed_v))
    }
    fn concat(
        client: &cubecl::client::Client,
        left: &cubecl::server::Handle,
        left_count: usize,
        right: &cubecl::server::Handle,
        right_count: usize,
    ) -> cubecl::server::Handle {
        let output = client.empty((left_count + right_count) * 4);
        unsafe {
            concatenate::launch_unchecked(
                client,
                grid(left_count + right_count),
                threads(),
                arg(left, left_count),
                arg(right, right_count),
                arg(&output, left_count + right_count),
                left_count,
                right_count,
            );
        }
        output
    }
    fn reduce(
        client: &cubecl::client::Client,
        lanes: &cubecl::server::Handle,
        reduced: &cubecl::server::Handle,
        result: &cubecl::server::Handle,
        columns: usize,
    ) {
        for level in 0..5 {
            let input = if level % 2 == 0 { lanes } else { reduced };
            let out = if level == 4 {
                result
            } else if level % 2 == 0 {
                reduced
            } else {
                lanes
            };
            let n = if level % 2 == 0 {
                columns * 32
            } else {
                columns * 16
            };
            let m = if level == 4 {
                columns
            } else if level % 2 == 0 {
                columns * 16
            } else {
                columns * 32
            };
            unsafe {
                qk::reduce::launch_unchecked(
                    client,
                    grid(columns * (16 >> level)),
                    threads(),
                    arg(input, n),
                    arg(out, m),
                    columns,
                    level,
                );
            }
        }
    }
    fn softmax(
        client: &cubecl::client::Client,
        masked: &cubecl::server::Handle,
        heads: usize,
        capacity: usize,
        valid: usize,
    ) -> cubecl::server::Handle {
        let n = heads * capacity;
        let groups = capacity.div_ceil(8);
        let scratch = client.empty(n * sm::SCRATCH * 4);
        let candidate = client.empty(n * 4);
        let sums = client.empty(n * 8);
        let subtracted = client.empty(n * 8);
        let maxes = client.empty(heads * 4);
        let partial = client.empty(heads * groups * 4 * 4);
        let reduced = client.empty(heads * groups * 2 * 4);
        let exps = client.empty(n * 4);
        let total = client.empty(heads * 8);
        let inverse64 = client.empty(heads * 8);
        let inverse32 = client.empty(heads * 4);
        let result = client.empty(n * 4);
        unsafe {
            sm::maxima::launch_unchecked(
                client,
                grid(heads),
                threads(),
                arg(masked, n),
                arg(&maxes, heads),
                heads,
                capacity,
            );
            for stage in 0..2 {
                sm::prepare::launch_unchecked(
                    client,
                    grid(n),
                    threads(),
                    arg(masked, n),
                    arg(&maxes, heads),
                    arg(&scratch, n * sm::SCRATCH),
                    n,
                    capacity,
                    stage,
                );
            }
            for step in 0..8 {
                if step == 1 || step == 3 {
                    sm::prepare::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(masked, n),
                        arg(&maxes, heads),
                        arg(&scratch, n * sm::SCRATCH),
                        n,
                        capacity,
                        if step == 1 { 2usize } else { 3usize },
                    );
                }
                for stage in 0..3 {
                    sm::polynomial::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&scratch, n * sm::SCRATCH),
                        arg(&candidate, n),
                        arg(&sums, n),
                        arg(&subtracted, n),
                        n,
                        step,
                        stage,
                    );
                }
            }
            sm::exponentials::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(&scratch, n * sm::SCRATCH),
                arg(&exps, n),
                n,
            );
            resident_bindings::zero_future::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(&exps, n),
                valid as u32,
                heads,
                capacity,
            );
            for level in 0..3 {
                let (from, to, from_len, to_len) = if level == 0 {
                    (&exps, &partial, n, heads * groups * 4)
                } else if level == 1 {
                    (&partial, &reduced, heads * groups * 4, heads * groups * 2)
                } else {
                    (&reduced, &partial, heads * groups * 2, heads * groups * 4)
                };
                sm::reduce::launch_unchecked(
                    client,
                    grid(heads * groups * (4 >> level)),
                    threads(),
                    arg(from, from_len),
                    arg(to, to_len),
                    heads,
                    capacity,
                    level,
                );
            }
            sm::totals::launch_unchecked(
                client,
                grid(heads),
                threads(),
                arg(&partial, heads * groups * 4),
                arg(&total, heads),
                heads,
                groups,
            );
            sm::reciprocal::launch_unchecked(
                client,
                grid(heads),
                threads(),
                arg(&total, heads),
                arg(&inverse64, heads),
                heads,
            );
            sm::inverses::launch_unchecked(
                client,
                grid(heads),
                threads(),
                arg(&inverse64, heads),
                arg(&inverse32, heads),
                heads,
            );
            sm::probabilities::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(&exps, n),
                arg(&inverse32, heads),
                arg(&result, n),
                n,
                capacity,
            );
        }
        result
    }
    fn execute(
        client: &cubecl::client::Client,
        shape: Shape,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<Vec<f32>> {
        let (q_elements, kv_elements, score_elements) = shape.validate()?;
        let heads = shape.batch * shape.query_heads;
        let kvheads = shape.batch * shape.kv_heads;
        let (query, keys, values) = flatten(client, q, k, v, shape)?;
        let packet = concat(client, &query, q_elements, &keys, kv_elements);
        let columns = score_elements;
        let lanes = client.empty(columns * 32 * 4);
        let temp = client.empty(columns * 32 * 4);
        let sums = client.empty(columns * 32 * 8);
        let subtracted = client.empty(columns * 32 * 8);
        let dots = client.empty(columns * 2 * 4);
        for block in 0..shape.width / 32 {
            let (previous, current) = if block % 2 == 0 {
                (&temp, &lanes)
            } else {
                (&lanes, &temp)
            };
            for stage in 0..3 {
                unsafe {
                    qk::accumulate::launch_unchecked(
                        client,
                        grid(columns * 32),
                        threads(),
                        arg(&packet, q_elements + kv_elements),
                        arg(previous, columns * 32),
                        arg(current, columns * 32),
                        arg(&sums, columns * 32),
                        arg(&subtracted, columns * 32),
                        shape.width,
                        shape.capacity,
                        heads,
                        kvheads,
                        block,
                        stage,
                    );
                }
            }
        }
        let last = if (shape.width / 32 - 1).is_multiple_of(2) {
            &lanes
        } else {
            &temp
        };
        let other = if (shape.width / 32 - 1).is_multiple_of(2) {
            &temp
        } else {
            &lanes
        };
        reduce(client, last, other, &dots, columns);
        unsafe {
            qk::scale::launch_unchecked(
                client,
                grid(columns),
                threads(),
                arg(&dots, columns * 2),
                1.0f32 / (shape.width as f32).sqrt(),
                columns,
            );
        }
        let masked = client.empty(columns * 4);
        unsafe {
            resident_bindings::masked_scores::launch_unchecked(
                client,
                grid(columns),
                threads(),
                arg(&dots, columns * 2),
                arg(&masked, columns),
                shape.valid as u32,
                heads,
                shape.capacity,
            );
        }
        let probabilities = softmax(client, &masked, heads, shape.capacity, shape.valid);
        let avpacket = concat(client, &probabilities, score_elements, &values, kv_elements);
        let channels = heads * shape.width;
        let av_lanes = client.empty(channels * 32 * 4);
        let av_temp = client.empty(channels * 32 * 4);
        let av_sums = client.empty(channels * 32 * 8);
        let av_sub = client.empty(channels * 32 * 8);
        let result = client.empty(channels * 4);
        let blocks = shape.valid.div_ceil(32);
        for block in 0..blocks {
            let (previous, current) = if block % 2 == 0 {
                (&av_temp, &av_lanes)
            } else {
                (&av_lanes, &av_temp)
            };
            for stage in 0..3 {
                unsafe {
                    av::accumulate::launch_unchecked(
                        client,
                        grid(channels * 32),
                        threads(),
                        arg(&avpacket, score_elements + kv_elements),
                        arg(previous, channels * 32),
                        arg(current, channels * 32),
                        arg(&av_sums, channels * 32),
                        arg(&av_sub, channels * 32),
                        block as u32,
                        shape.width,
                        shape.capacity,
                        heads,
                        kvheads,
                        stage,
                    );
                }
            }
        }
        let last = if (blocks - 1).is_multiple_of(2) {
            &av_lanes
        } else {
            &av_temp
        };
        let other = if (blocks - 1).is_multiple_of(2) {
            &av_temp
        } else {
            &av_lanes
        };
        reduce(client, last, other, &result, channels);
        let bytes = client
            .read_one(result)
            .map_err(|e| Error::State(format!("GPU batched attention readback: {e}")))?;
        Ok(f32::from_bytes(&bytes).to_vec())
    }
    const QH: usize = 4;
    const KH: usize = 2;
    const WIDTH: usize = 32;
    const CAP: usize = 32;
    for batch in [2, 4, 8] {
        for valid in [1, 17, 32] {
            let shape = Shape {
                batch,
                query_heads: QH,
                kv_heads: KH,
                width: WIDTH,
                capacity: CAP,
                valid,
            };
            let (q, k, _) = shape.validate()?;
            // Q across contiguous request/head lanes.
            let query = (0..q)
                .map(|i| ((i * 47 + i / (QH * WIDTH) * 13 + 71) % 523) as f32 / 101.0 - 2.1)
                .collect::<Vec<_>>();
            // K/V originally resident [B,token,kv_head,width].
            let keys = (0..k)
                .map(|i| ((i * 31 + i / (CAP * KH * WIDTH) * 29 + 3) % 467) as f32 / 111.0 - 1.9)
                .collect::<Vec<_>>();
            let values = (0..k)
                .map(|i| ((i * 71 + i / (CAP * KH * WIDTH) * 17 + 91) % 601) as f32 / 131.0 - 2.0)
                .collect::<Vec<_>>();
            let actual = execute(&client, shape, &query, &keys, &values)?;
            assert_eq!(actual.len(), batch * QH * WIDTH);
            let mut compared = 0;
            for request in 0..batch {
                let q_start = request * QH * WIDTH;
                let kv_start = request * CAP * KH * WIDTH;
                let reference = execute(
                    &client,
                    Shape { batch: 1, ..shape },
                    &query[q_start..q_start + QH * WIDTH],
                    &keys[kv_start..kv_start + CAP * KH * WIDTH],
                    &values[kv_start..kv_start + CAP * KH * WIDTH],
                )?;
                for (i, (&got, &want)) in actual[q_start..q_start + QH * WIDTH]
                    .iter()
                    .zip(&reference)
                    .enumerate()
                {
                    if got.to_bits() != want.to_bits() {
                        return Err(Error::State(format!(
                            "B{batch} valid{valid} attention request{request} \
                             lane{i}: {:08x}!={:08x}",
                            got.to_bits(),
                            want.to_bits()
                        )));
                    }
                    compared += 1;
                }
            }
            println!(
                "P12_ATTENTION_BATCH_GPU_PASS B={batch} valid={valid} \
                 qheads={QH} kvheads={KH} width={WIDTH} exact_output_bits={compared} \
                 qk_softmax_av_per_stage_dispatches=one_for_all_rows"
            );
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_batch_attention_geometry_refuses_cross_row_mismatch() {
        let valid = Shape {
            batch: 2,
            query_heads: 4,
            kv_heads: 2,
            width: 32,
            capacity: 32,
            valid: 17,
        };
        assert_eq!(valid.validate().unwrap(), (256, 4096, 256));
        for s in [
            Shape { batch: 0, ..valid },
            Shape { batch: 9, ..valid },
            Shape {
                query_heads: 3,
                ..valid
            },
            Shape {
                kv_heads: 0,
                ..valid
            },
            Shape { width: 31, ..valid },
            Shape {
                capacity: 33,
                ..valid
            },
            Shape { valid: 0, ..valid },
            Shape { valid: 33, ..valid },
        ] {
            assert!(s.validate().is_err());
        }
    }
}
