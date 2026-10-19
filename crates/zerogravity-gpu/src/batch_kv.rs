//! Batched immutable-origin, private-candidate causal KV update.
//!
//! One CubeCL launch handles B independent sessions. Every candidate bit
//! is written; previous state is read-only. The source's accepted B1
//! append_kv kernel remains the one-row reference (including padding).
//! Device buffer leases/StateRoot publication remain with ResidentService.

use zerogravity_core::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) struct Shape {
    batch: usize,
    slab: usize,
    capacity: usize,
    position: usize,
}
impl Shape {
    fn validate(self, prior: usize, token: usize, output: usize) -> Result<usize> {
        if !(1..=8).contains(&self.batch)
            || self.slab == 0
            || self.slab > 8192
            || self.capacity == 0
            || self.capacity > 512
            || self.position >= self.capacity
        {
            return Err(Error::Limit("batched KV source geometry/capacity".into()));
        }
        let state = self
            .batch
            .checked_mul(self.capacity)
            .and_then(|n| n.checked_mul(self.slab))
            .filter(|n| *n <= 1_048_576)
            .ok_or_else(|| Error::Limit("batched KV source byte limit".into()))?;
        let new_tokens = self
            .batch
            .checked_mul(self.slab)
            .ok_or_else(|| Error::Limit("batched KV source token width".into()))?;
        if prior != state || output != state || token != new_tokens {
            return Err(Error::Malformed(
                "batched KV source bound cache/token shape".into(),
            ));
        }
        Ok(state)
    }
}

#[cfg(feature = "qualification")]
use cubecl::prelude::*;

#[cfg(feature = "qualification")]
#[cube(launch_unchecked)]
pub(crate) fn append_kv_batched(
    old_k: &[u32],
    old_v: &[u32],
    new_k: &[u32],
    new_v: &[u32],
    candidate_k: &mut [u32],
    candidate_v: &mut [u32],
    position: u32,
    #[comptime] slab: usize,
    #[comptime] capacity: usize,
    #[comptime] batch: usize,
) {
    let i = ABSOLUTE_POS;
    if i < batch * capacity * slab {
        let request = i / (capacity * slab);
        let local = i % (capacity * slab);
        let token = local / slab;
        let lane = local % slab;
        let mut k = 0u32;
        let mut v = 0u32;
        if token < position as usize {
            k = old_k[i];
            v = old_v[i];
        } else if token == position as usize {
            k = new_k[request * slab + lane];
            v = new_v[request * slab + lane];
        }
        candidate_k[i] = k;
        candidate_v[i] = v;
    }
}
#[cfg(feature = "qualification")]
pub(crate) fn qualify(device: cubecl::Device) -> Result<()> {
    use crate::resident_bindings::append_kv;
    fn grid(n: usize) -> CubeCount {
        CubeCount::Static((n as u32).div_ceil(32), 1, 1)
    }
    fn threads() -> CubeDim {
        CubeDim::new_1d(32)
    }
    fn arg(h: &cubecl::server::Handle, n: usize) -> BufferArg {
        unsafe { BufferArg::from_raw_parts(h.clone(), n) }
    }
    let client = device.client();
    const SLAB: usize = 64;
    const CAP: usize = 32;
    for batch in [2, 4, 8] {
        for position in [0usize, 7, 31] {
            let shape = Shape {
                batch,
                slab: SLAB,
                capacity: CAP,
                position,
            };
            let count = shape.validate(batch * CAP * SLAB, batch * SLAB, batch * CAP * SLAB)?;
            let prior_k = (0..count)
                .map(|i| ((i * 37 + 19) % u32::MAX as usize) as u32)
                .collect::<Vec<_>>();
            let prior_v = (0..count)
                .map(|i| ((i * 11 + 819) % u32::MAX as usize) as u32)
                .collect::<Vec<_>>();
            let tokens_k = (0..batch * SLAB)
                .map(|i| (i * 3 + 101) as u32)
                .collect::<Vec<_>>();
            let tokens_v = (0..batch * SLAB)
                .map(|i| (i * 7 + 403) as u32)
                .collect::<Vec<_>>();
            let pk = client.create_from_slice(u32::as_bytes(&prior_k));
            let pv = client.create_from_slice(u32::as_bytes(&prior_v));
            let nk = client.create_from_slice(u32::as_bytes(&tokens_k));
            let nv = client.create_from_slice(u32::as_bytes(&tokens_v));
            let out_k = client.empty(count * 4);
            let out_v = client.empty(count * 4);
            unsafe {
                append_kv_batched::launch_unchecked(
                    &client,
                    grid(count),
                    threads(),
                    arg(&pk, count),
                    arg(&pv, count),
                    arg(&nk, batch * SLAB),
                    arg(&nv, batch * SLAB),
                    arg(&out_k, count),
                    arg(&out_v, count),
                    position as u32,
                    SLAB,
                    CAP,
                    batch,
                );
            }
            let output_k = client
                .read_one(out_k)
                .map_err(|e| Error::State(format!("batched KV key readback: {e}")))?;
            let output_v = client
                .read_one(out_v)
                .map_err(|e| Error::State(format!("batched KV value readback: {e}")))?;
            let output_k = u32::from_bytes(&output_k);
            let output_v = u32::from_bytes(&output_v);
            for request in 0..batch {
                let row = request * CAP * SLAB;
                let old_k =
                    client.create_from_slice(u32::as_bytes(&prior_k[row..row + CAP * SLAB]));
                let old_v =
                    client.create_from_slice(u32::as_bytes(&prior_v[row..row + CAP * SLAB]));
                let new_k = client.create_from_slice(u32::as_bytes(
                    &tokens_k[request * SLAB..(request + 1) * SLAB],
                ));
                let new_v = client.create_from_slice(u32::as_bytes(
                    &tokens_v[request * SLAB..(request + 1) * SLAB],
                ));
                let one_k = client.empty(CAP * SLAB * 4);
                let one_v = client.empty(CAP * SLAB * 4);
                unsafe {
                    append_kv::launch_unchecked(
                        &client,
                        grid(CAP * SLAB),
                        threads(),
                        arg(&old_k, CAP * SLAB),
                        arg(&old_v, CAP * SLAB),
                        arg(&new_k, SLAB),
                        arg(&new_v, SLAB),
                        arg(&one_k, CAP * SLAB),
                        arg(&one_v, CAP * SLAB),
                        position as u32,
                        SLAB,
                        CAP,
                    );
                }
                let gold_k = client
                    .read_one(one_k)
                    .map_err(|e| Error::State(format!("B1 KV key readback: {e}")))?;
                let gold_v = client
                    .read_one(one_v)
                    .map_err(|e| Error::State(format!("B1 KV value readback: {e}")))?;
                let gold_k = u32::from_bytes(&gold_k);
                let gold_v = u32::from_bytes(&gold_v);
                for i in 0..CAP * SLAB {
                    if output_k[row + i] != gold_k[i] || output_v[row + i] != gold_v[i] {
                        return Err(Error::State(format!(
                            "B{batch} position{position} KV request{request} cell{i} differs from B1"
                        )));
                    }
                }
            }
            println!(
                "P12_KV_BATCH_GPU_PASS B={batch} position={position} \
                 exact_key_bits={count} exact_value_bits={count} fused_kv_launches=1"
            );
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn causal_kv_batch_shape_and_padding_refusals() {
        assert_eq!(
            Shape {
                batch: 2,
                slab: 64,
                capacity: 32,
                position: 7
            }
            .validate(4096, 128, 4096)
            .unwrap(),
            4096
        );
        let shape = Shape {
            batch: 2,
            slab: 64,
            capacity: 32,
            position: 7,
        };
        for (a, b, c) in [(4095, 128, 4096), (4096, 127, 4096), (4096, 128, 4095)] {
            assert!(shape.validate(a, b, c).is_err());
        }
        for bad in [
            Shape { batch: 0, ..shape },
            Shape { batch: 9, ..shape },
            Shape { slab: 0, ..shape },
            Shape {
                slab: 8193,
                ..shape
            },
            Shape {
                capacity: 0,
                ..shape
            },
            Shape {
                capacity: 513,
                ..shape
            },
            Shape {
                position: 32,
                ..shape
            },
        ] {
            assert!(bad.validate(4096, 128, 4096).is_err());
        }
    }
}
