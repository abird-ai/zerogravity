//! P07-B first source-matched mixed quantization GPU regions.
//!
//! Actual source checkpoint codecs needed by Qwen3.5 2B:
//!   Q4_1 -> dynamic Q8_1, signed zero-point correction
//!   Q8_0 -> dynamic Q8_0
//!   Q6_K -> signed dynamic Q8_K with 8 integer SIMD lanes
//!   F32 -> 4 x 8 FMA accumulators with source horizontal reduction
//!
//! Intentionally one GPU invocation, confirmed host readback per region.
//! This is a correctness-first specialization; not a persistent whole-model
//! residency backend or high-throughput graph scheduler.
//! Q4_0 remains implemented by the independent qualified q4_q8 module.

use crate::source_accumulation;
#[cfg(feature = "qualification")]
use crate::source_accumulation::SourceAccumulation;
#[cfg(feature = "qualification")]
use cubecl::Device;
use cubecl::prelude::*;
#[cfg(feature = "qualification")]
use std::sync::atomic::{AtomicUsize, Ordering};
use zerogravity_artifacts::quant::QuantKind;
#[cfg(any(test, feature = "qualification"))]
use zerogravity_artifacts::quant::{f16_to_f32, f32_to_f16_rne};
#[cfg(any(test, feature = "qualification"))]
use zerogravity_core::Error;
#[cfg(any(test, feature = "qualification"))]
use zerogravity_core::Result;
#[cfg(any(test, feature = "qualification"))]
use zerogravity_core::accelerator::MatVecRegion;
#[cfg(feature = "qualification")]
use zerogravity_core::accelerator::{MatVecKernel, RegionExecution};

pub(crate) struct Prepared {
    pub(crate) kind: QuantKind,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) blocks: usize,
    #[cfg(any(test, feature = "qualification"))]
    pub(crate) source: Vec<u8>,
    // One source dot scale per quant block, including dynamic Q8 scale.
    pub(crate) scales: Vec<f32>,
    // Q4_1 zero-point contribution, one per quant block.
    pub(crate) offsets: Vec<f32>,
    // Signed dynamic Q8 codes, stored as i32 for portable CubeCL indexing.
    pub(crate) activation: Vec<i32>,
    // Original f32 source matrix only for F32 specialization.
    #[cfg(any(test, feature = "qualification"))]
    pub(crate) weights_f32: Vec<f32>,
    pub(crate) input_f32: Vec<f32>,
}

/// Pure host encoding/preparation, not a host dot or foreign runtime.
#[cfg(any(test, feature = "qualification"))]
pub(crate) fn prepare(region: MatVecRegion<'_>) -> Result<Prepared> {
    let kind = region.kind;
    let (block_width, source_bytes, row_multiple) = match kind {
        QuantKind::Q4_1 => (32, 20, 1),
        QuantKind::Q8_0 => (32, 34, 1),
        QuantKind::Q6K => (256, 210, 1),
        QuantKind::F32 => (32, 128, 1),
        _ => {
            return Err(Error::Unsupported(
                "P07-B supports Q4_1, Q8_0, Q6_K and aligned F32".into(),
            ));
        }
    };
    if region.rows == 0
        || region.columns == 0
        || !region.rows.is_multiple_of(row_multiple)
        || !region.columns.is_multiple_of(block_width)
        || region.activation.len() != region.columns
        || !region.activation.iter().all(|v| v.is_finite())
    {
        return Err(Error::Malformed(
            "P07-B source weight/activation geometry invalid".into(),
        ));
    }
    let blocks = region.columns / block_width;
    let count = region
        .rows
        .checked_mul(blocks)
        .ok_or_else(|| Error::Limit("P07-B matrix blocks overflow".into()))?;
    let nbytes = count
        .checked_mul(source_bytes)
        .ok_or_else(|| Error::Limit("P07-B source bytes overflow".into()))?;
    if region.packed_weights.len() != nbytes {
        return Err(Error::Malformed(
            "P07-B original GGUF source byte length invalid".into(),
        ));
    }
    // F32 source has no dynamic quantization, only the verified 4x8 reduction
    // schedule. Restrict to complete 32-element SIMD groups, as present in
    // the actual Qwen checkpoint matrices, avoiding unvalidated scalar tails.
    if kind == QuantKind::F32 {
        let mut f32_weights = Vec::with_capacity(region.rows * region.columns);
        for chunk in region.packed_weights.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().expect("validated f32 bytes"));
            if !val.is_finite() {
                return Err(Error::Malformed("nonfinite F32 GGUF matrix".into()));
            }
            f32_weights.push(val);
        }
        return Ok(Prepared {
            kind,
            rows: region.rows,
            cols: region.columns,
            blocks,
            source: Vec::new(),
            scales: Vec::new(),
            offsets: Vec::new(),
            activation: Vec::new(),
            weights_f32: f32_weights,
            input_f32: region.activation.to_vec(),
        });
    }
    let mut activation = Vec::with_capacity(region.columns);
    let mut input_scale = Vec::with_capacity(blocks);
    let mut input_sum = Vec::with_capacity(blocks);
    for x in region.activation.chunks_exact(block_width) {
        if kind == QuantKind::Q6K {
            let mut largest = 0f32;
            let mut signed_largest = 0f32;
            for &v in x {
                if v.abs() > largest {
                    largest = v.abs();
                    signed_largest = v;
                }
            }
            let inv = if largest == 0.0 {
                0.0
            } else {
                -127.0 / signed_largest
            };
            let scale = if largest == 0.0 { 0.0 } else { 1.0 / inv };
            input_scale.push(scale);
            input_sum.push(0.0);
            for &v in x {
                activation.push((v * inv).round_ties_even().clamp(-128.0, 127.0) as i32);
            }
        } else {
            let largest = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let exact_d = largest / 127.0;
            let inv = if largest == 0.0 { 0.0 } else { 127.0 / largest };
            let mut sum = 0i32;
            for &v in x {
                let q = (v * inv).round_ties_even().clamp(-128.0, 127.0) as i32;
                activation.push(q);
                sum += q;
            }
            input_scale.push(f16_to_f32(f32_to_f16_rne(exact_d)));
            input_sum.push(f16_to_f32(f32_to_f16_rne(sum as f32 * exact_d)));
        }
    }
    let mut scales = Vec::with_capacity(count);
    let mut offsets = Vec::with_capacity(if kind == QuantKind::Q4_1 { count } else { 0 });
    for (idx, block) in region.packed_weights.chunks_exact(source_bytes).enumerate() {
        let wscale = if kind == QuantKind::Q6K {
            f16_to_f32(u16::from_le_bytes([block[208], block[209]]))
        } else {
            f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
        };
        if !wscale.is_finite() {
            return Err(Error::Malformed(
                "P07-B source weight scale nonfinite".into(),
            ));
        }
        scales.push(wscale * input_scale[idx % blocks]);
        if kind == QuantKind::Q4_1 {
            let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            if !m.is_finite() {
                return Err(Error::Malformed("P07-B source Q4_1 bias nonfinite".into()));
            }
            offsets.push(m * input_sum[idx % blocks]);
        }
    }
    Ok(Prepared {
        kind,
        rows: region.rows,
        cols: region.columns,
        blocks,
        source: region.packed_weights.to_vec(),
        scales,
        offsets,
        activation,
        weights_f32: Vec::new(),
        input_f32: Vec::new(),
    })
}

#[cube]
fn signed_byte(value: u32) -> i32 {
    let mut result = i32::cast_from(value);
    if value >= 128u32 {
        result -= 256i32;
    }
    result
}

/// Kernel lanes preserve original GGML eight-way float reduction order.
#[cube]
fn eight_lane_reduce(lanes: &Array<f32>) -> f32 {
    let r0 = lanes[0] + lanes[4];
    let r1 = lanes[1] + lanes[5];
    let r2 = lanes[2] + lanes[6];
    let r3 = lanes[3] + lanes[7];
    (r0 + r2) + (r1 + r3)
}

#[cube(launch_unchecked)]
pub(crate) fn q4_1_q8_1_gemv(
    packed: &[u8],
    scales: &[f32],
    offsets: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] rows: usize,
    #[comptime] wide_accumulate: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut lanes = Array::<f32>::new(8usize);
        for i in 0..8usize {
            lanes[i] = 0.0;
        }
        let mut summs = 0.0f32;
        for block in 0..blocks {
            let bid = row * blocks + block;
            summs += offsets[bid];
            for lane in 0..8usize {
                let mut dot = 0i32;
                for sub in 0..4usize {
                    let at = lane * 4 + sub;
                    let byte = packed[bid * 20 + 4 + (at % 16)];
                    let nib = u32::cast_from(byte);
                    let w = if at < 16usize {
                        i32::cast_from(nib & 15u32)
                    } else {
                        i32::cast_from(nib >> 4u32)
                    };
                    dot += w * q8[block * 32 + at];
                }
                lanes[lane] = source_accumulation::source_mul_add(
                    f32::cast_from(dot),
                    scales[bid],
                    lanes[lane],
                    wide_accumulate,
                );
            }
        }
        output[row] = eight_lane_reduce(&lanes) + summs;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn q8_0_q8_0_gemv(
    packed: &[u8],
    scales: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] rows: usize,
    #[comptime] wide_accumulate: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut lanes = Array::<f32>::new(8usize);
        for i in 0..8usize {
            lanes[i] = 0.0;
        }
        for block in 0..blocks {
            let bid = row * blocks + block;
            for lane in 0..8usize {
                let mut dot = 0i32;
                for sub in 0..4usize {
                    let at = lane * 4 + sub;
                    let w = signed_byte(u32::cast_from(packed[bid * 34 + 2 + at]));
                    dot += w * q8[block * 32 + at];
                }
                lanes[lane] = source_accumulation::source_mul_add(
                    f32::cast_from(dot),
                    scales[bid],
                    lanes[lane],
                    wide_accumulate,
                );
            }
        }
        output[row] = eight_lane_reduce(&lanes);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn q6_k_q8_k_gemv(
    packed: &[u8],
    scales: &[f32],
    q8: &[i32],
    output: &mut [f32],
    #[comptime] blocks: usize,
    #[comptime] rows: usize,
    #[comptime] wide_accumulate: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut lanes = Array::<f32>::new(8usize);
        for i in 0..8usize {
            lanes[i] = 0.0;
        }
        for block in 0..blocks {
            let bid = row * blocks + block;
            let base = bid * 210;
            let mut int_acc = Array::<i32>::new(8usize);
            for i in 0..8usize {
                int_acc[i] = 0;
            }
            for half in 0..2usize {
                for at32 in 0..32usize {
                    let high = u32::cast_from(packed[base + 128 + half * 32 + at32]);
                    for group in 0..4usize {
                        let address = at32 + (group % 2) * 32;
                        let nib = u32::cast_from(packed[base + half * 64 + address]);
                        let low = if group < 2usize {
                            nib & 15u32
                        } else {
                            nib >> 4u32
                        };
                        let w = i32::cast_from(
                            low | (((high >> (2u32 * u32::cast_from(group))) & 3u32) << 4u32),
                        );
                        let subscale = signed_byte(u32::cast_from(
                            packed[base + 192 + half * 8 + group * 2 + at32 / 16],
                        ));
                        let at = half * 128 + group * 32 + at32;
                        let target = at32 / 4;
                        int_acc[target] += w * subscale * q8[block * 256 + at];
                    }
                }
            }
            for lane in 0..8usize {
                let mut offset = 0i32;
                for sub in 0..2usize {
                    let group = lane * 2 + sub;
                    let subscale = signed_byte(u32::cast_from(packed[base + 192 + group]));
                    let mut sum = 0i32;
                    for qidx in 0..16usize {
                        sum += q8[block * 256 + group * 16 + qidx];
                    }
                    offset += sum * subscale;
                }
                int_acc[lane] -= 32i32 * offset;
                lanes[lane] = source_accumulation::source_mul_add(
                    f32::cast_from(int_acc[lane]),
                    scales[bid],
                    lanes[lane],
                    wide_accumulate,
                );
            }
        }
        output[row] = eight_lane_reduce(&lanes);
    }
}

/// One canonical Qwen F32 32-lane source reduction shared by B1 GEMV
/// and the native B>1 projection. No second numeric F32 definition.
#[cube]
pub(crate) fn source_f32_ggml_dot(
    weights: &[f32],
    x: &[f32],
    row: usize,
    input_offset: usize,
    #[comptime] cols: usize,
    #[comptime] wide_accumulate: bool,
) -> f32 {
    let mut lanes = Array::<f32>::new(32usize);
    for i in 0..32usize {
        lanes[i] = 0.0;
    }
    for base in 0..cols / 32 {
        for lane in 0..32usize {
            let i = base * 32 + lane;
            lanes[lane] = source_accumulation::source_mul_add(
                weights[row * cols + i],
                x[input_offset + i],
                lanes[lane],
                wide_accumulate,
            );
        }
    }
    let mut reduced = Array::<f32>::new(8usize);
    for i in 0..8usize {
        reduced[i] = (lanes[i] + lanes[i + 16]) + (lanes[8 + i] + lanes[24 + i]);
    }
    let p0 = reduced[0] + reduced[4];
    let p1 = reduced[1] + reduced[5];
    let p2 = reduced[2] + reduced[6];
    let p3 = reduced[3] + reduced[7];
    (p0 + p1) + (p2 + p3)
}

/// Accepted B1 source math uses exactly the same arithmetic as B>1.
#[cube(launch_unchecked)]
pub(crate) fn f32_ggml_dot_gemv(
    weights: &[f32],
    x: &[f32],
    output: &mut [f32],
    #[comptime] cols: usize,
    #[comptime] rows: usize,
    #[comptime] wide_accumulate: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        output[row] = source_f32_ggml_dot(weights, x, row, 0usize, cols, wide_accumulate);
    }
}

/// Scoped actual device region provider, selected by exact tensor name.
/// Only one tensor is handled by this stage; all others legitimately fall
/// back to source-qualified Rust CPU. A failed GPU operation never falls back.
#[cfg(feature = "qualification")]
pub struct CubeMixedProvider {
    device: Device,
    tensor: String,
    max_rows: usize,
    completed: AtomicUsize,
}
#[cfg(feature = "qualification")]
impl CubeMixedProvider {
    pub fn new(device: Device, tensor: impl Into<String>, max_rows: usize) -> Self {
        Self {
            device,
            tensor: tensor.into(),
            max_rows,
            completed: AtomicUsize::new(0),
        }
    }
    pub fn completed_regions(&self) -> usize {
        self.completed.load(Ordering::Relaxed)
    }
}
#[cfg(feature = "qualification")]
impl MatVecKernel for CubeMixedProvider {
    fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
        if region.tensor_name != self.tensor
            || region.rows > self.max_rows
            || !matches!(
                region.kind,
                QuantKind::Q4_1 | QuantKind::Q8_0 | QuantKind::Q6K | QuantKind::F32
            )
        {
            return Ok(RegionExecution::Unsupported);
        }
        let prepared = prepare(region)?;
        let client = self.device.client();
        source_accumulation::ensure_capability(&client, SourceAccumulation::WideF64)?;
        let output = client.empty(prepared.rows * 4);
        let launch_rows = CubeCount::Static((prepared.rows as u32).div_ceil(32), 1, 1);
        let dims = CubeDim::new_1d(32);
        match prepared.kind {
            QuantKind::F32 => {
                let weights = client.create_from_slice(f32::as_bytes(&prepared.weights_f32));
                let x = client.create_from_slice(f32::as_bytes(&prepared.input_f32));
                unsafe {
                    f32_ggml_dot_gemv::launch_unchecked(
                        &client,
                        launch_rows,
                        dims,
                        BufferArg::from_raw_parts(weights, prepared.weights_f32.len()),
                        BufferArg::from_raw_parts(x, prepared.input_f32.len()),
                        BufferArg::from_raw_parts(output.clone(), prepared.rows),
                        prepared.cols,
                        prepared.rows,
                        true,
                    );
                }
            }
            QuantKind::Q4_1 => {
                let weights = client.create_from_slice(&prepared.source);
                let scales = client.create_from_slice(f32::as_bytes(&prepared.scales));
                let offsets = client.create_from_slice(f32::as_bytes(&prepared.offsets));
                let q8 = client.create_from_slice(i32::as_bytes(&prepared.activation));
                unsafe {
                    q4_1_q8_1_gemv::launch_unchecked(
                        &client,
                        launch_rows,
                        dims,
                        BufferArg::from_raw_parts(weights, prepared.source.len()),
                        BufferArg::from_raw_parts(scales, prepared.scales.len()),
                        BufferArg::from_raw_parts(offsets, prepared.offsets.len()),
                        BufferArg::from_raw_parts(q8, prepared.activation.len()),
                        BufferArg::from_raw_parts(output.clone(), prepared.rows),
                        prepared.blocks,
                        prepared.rows,
                        true,
                    );
                }
            }
            QuantKind::Q8_0 => {
                let weights = client.create_from_slice(&prepared.source);
                let scales = client.create_from_slice(f32::as_bytes(&prepared.scales));
                let q8 = client.create_from_slice(i32::as_bytes(&prepared.activation));
                unsafe {
                    q8_0_q8_0_gemv::launch_unchecked(
                        &client,
                        launch_rows,
                        dims,
                        BufferArg::from_raw_parts(weights, prepared.source.len()),
                        BufferArg::from_raw_parts(scales, prepared.scales.len()),
                        BufferArg::from_raw_parts(q8, prepared.activation.len()),
                        BufferArg::from_raw_parts(output.clone(), prepared.rows),
                        prepared.blocks,
                        prepared.rows,
                        true,
                    );
                }
            }
            QuantKind::Q6K => {
                let weights = client.create_from_slice(&prepared.source);
                let scales = client.create_from_slice(f32::as_bytes(&prepared.scales));
                let q8 = client.create_from_slice(i32::as_bytes(&prepared.activation));
                unsafe {
                    q6_k_q8_k_gemv::launch_unchecked(
                        &client,
                        launch_rows,
                        dims,
                        BufferArg::from_raw_parts(weights, prepared.source.len()),
                        BufferArg::from_raw_parts(scales, prepared.scales.len()),
                        BufferArg::from_raw_parts(q8, prepared.activation.len()),
                        BufferArg::from_raw_parts(output.clone(), prepared.rows),
                        prepared.blocks,
                        prepared.rows,
                        true,
                    );
                }
            }
            _ => unreachable!("all accepted source kinds have an exact GPU kernel"),
        }
        // Device readback is the synchronous completion/visibility fence;
        // there is no "submitted but complete" path in this interface.
        let raw = client
            .read_one(output)
            .map_err(|e| Error::State(format!("P07-B GPU wait/readback failed: {e}")))?;
        if raw.len() != prepared.rows * 4 {
            return Err(Error::State("P07-B incomplete GPU output bytes".into()));
        }
        let result = raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("output bytes validated")))
            .collect::<Vec<_>>();
        if !result.iter().all(|v| v.is_finite()) {
            return Err(Error::State("P07-B GPU returned nonfinite output".into()));
        }
        self.completed.fetch_add(1, Ordering::Relaxed);
        Ok(RegionExecution::Completed(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU-only interpretation of the *same* integer/FP32 source kernel,
    /// used to qualify unambiguous GPU instruction ordering against native
    /// Rust GGUF matvec. This is not a real device execution or fallback.
    #[cfg(feature = "qualification")]
    fn emulate(prepared: &Prepared) -> Vec<f32> {
        let mut out = vec![0.0f32; prepared.rows];
        for (row, slot) in out.iter_mut().enumerate() {
            match prepared.kind {
                QuantKind::Q4_1 => {
                    let mut lanes = [0f32; 8];
                    let mut summs = 0f32;
                    for block in 0..prepared.blocks {
                        let id = row * prepared.blocks + block;
                        summs += prepared.offsets[id];
                        for (lane, acc) in lanes.iter_mut().enumerate() {
                            let mut dot = 0i32;
                            for sub in 0..4 {
                                let at = lane * 4 + sub;
                                let b = prepared.source[id * 20 + 4 + (at % 16)];
                                let w = if at < 16 {
                                    i32::from(b & 15)
                                } else {
                                    i32::from(b >> 4)
                                };
                                dot += w * prepared.activation[block * 32 + at];
                            }
                            *acc = (dot as f32).mul_add(prepared.scales[id], *acc);
                        }
                    }
                    *slot = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
                        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
                        + summs;
                }
                QuantKind::Q8_0 => {
                    let mut lanes = [0f32; 8];
                    for block in 0..prepared.blocks {
                        let id = row * prepared.blocks + block;
                        for (lane, acc) in lanes.iter_mut().enumerate() {
                            let mut dot = 0i32;
                            for sub in 0..4 {
                                let at = lane * 4 + sub;
                                dot += i32::from(prepared.source[id * 34 + 2 + at] as i8)
                                    * prepared.activation[block * 32 + at];
                            }
                            *acc = (dot as f32).mul_add(prepared.scales[id], *acc);
                        }
                    }
                    *slot = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
                        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
                }
                QuantKind::Q6K => {
                    let mut lanes = [0f32; 8];
                    for block in 0..prepared.blocks {
                        let id = row * prepared.blocks + block;
                        let raw = &prepared.source[id * 210..(id + 1) * 210];
                        let mut ints = [0i32; 8];
                        for half in 0..2 {
                            for at32 in 0..32 {
                                let hi = raw[128 + half * 32 + at32];
                                for group in 0..4 {
                                    let address = at32 + (group % 2) * 32;
                                    let nib = raw[half * 64 + address];
                                    let lo = if group < 2 { nib & 15 } else { nib >> 4 };
                                    let w = i32::from(lo | (((hi >> (2 * group)) & 3) << 4));
                                    let subscale = i32::from(
                                        raw[192 + half * 8 + group * 2 + at32 / 16] as i8,
                                    );
                                    let at = half * 128 + group * 32 + at32;
                                    ints[at32 / 4] +=
                                        w * subscale * prepared.activation[block * 256 + at];
                                }
                            }
                        }
                        for lane in 0..8 {
                            let mut offset = 0i32;
                            for sub in 0..2 {
                                let group = lane * 2 + sub;
                                let scale = i32::from(raw[192 + group] as i8);
                                let sum = prepared.activation
                                    [block * 256 + group * 16..block * 256 + (group + 1) * 16]
                                    .iter()
                                    .sum::<i32>();
                                offset += sum * scale;
                            }
                            ints[lane] -= 32 * offset;
                            lanes[lane] =
                                (ints[lane] as f32).mul_add(prepared.scales[id], lanes[lane]);
                        }
                    }
                    *slot = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
                        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
                }
                QuantKind::F32 => {
                    let mut lanes = [0f32; 32];
                    for at in 0..prepared.cols {
                        let lane = at % 32;
                        lanes[lane] = prepared.weights_f32[row * prepared.cols + at]
                            .mul_add(prepared.input_f32[at], lanes[lane]);
                    }
                    let mut reduced = [0f32; 8];
                    for lane in 0..8 {
                        reduced[lane] =
                            (lanes[lane] + lanes[lane + 16]) + (lanes[lane + 8] + lanes[lane + 24]);
                    }
                    let p0 = reduced[0] + reduced[4];
                    let p1 = reduced[1] + reduced[5];
                    let p2 = reduced[2] + reduced[6];
                    let p3 = reduced[3] + reduced[7];
                    *slot = (p0 + p1) + (p2 + p3);
                }
                _ => panic!("only source-qualified mixed kinds"),
            }
        }
        out
    }

    #[test]
    fn mixed_kernel_preparation_rejects_bad_geometry_and_nonfinite_source() {
        let cols = 32usize;
        let rows = 1usize;
        let mut block = [0u8; 20];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        let input = [1.0f32; 32];
        let original = MatVecRegion {
            source_identity: zerogravity_core::accelerator::SourceIdentity::unbound(),
            tensor_name: "blk.0.ffn_down.weight",
            kind: QuantKind::Q4_1,
            columns: cols,
            rows,
            packed_weights: &block,
            activation: &input,
        };
        let generated = prepare(original).unwrap();
        assert_eq!(generated.source, block);
        assert!(generated.weights_f32.is_empty());
        assert_eq!(generated.scales.len(), 1);
        assert_eq!(generated.offsets.len(), 1);
        let mut wrong = original;
        wrong.columns = 31;
        assert!(prepare(wrong).is_err());
        wrong = original;
        wrong.rows = 2;
        assert!(prepare(wrong).is_err());
        let nan = [f32::NAN; 32];
        wrong = original;
        wrong.activation = &nan;
        assert!(prepare(wrong).is_err());
        let mut nonfinite_block = block;
        nonfinite_block[0..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        wrong = original;
        wrong.packed_weights = &nonfinite_block;
        assert!(prepare(wrong).is_err());
    }

    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires locally pinned Qwen3.5 2B model GGUF"]
    fn source_mixed_closed_region_in_real_model_is_bitwise_and_atomic() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use zerogravity_model_qwen35::QwenDenseCpu;
        struct HostModeledMixed {
            name: String,
            completed: AtomicUsize,
        }
        impl MatVecKernel for HostModeledMixed {
            fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
                if region.tensor_name != self.name {
                    return Ok(RegionExecution::Unsupported);
                }
                self.completed.fetch_add(1, Ordering::Relaxed);
                Ok(RegionExecution::Completed(emulate(&prepare(region)?)))
            }
        }

        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("target/models/qwen35-2b");
        let gguf = source.join("model.gguf");
        let config = source.join("config.json");
        let tokens = [11u32];
        let mut cpu = QwenDenseCpu::open(&gguf, &config).unwrap();
        let baseline = cpu.prefill_transaction(&tokens).unwrap();
        let expected_root = cpu.state_fingerprint();
        for name in [
            "blk.0.ffn_down.weight",
            "blk.0.ssm_out.weight",
            "blk.11.attn_q.weight",
            "blk.0.ssm_alpha.weight",
        ] {
            let kernel = Arc::new(HostModeledMixed {
                name: name.to_owned(),
                completed: AtomicUsize::new(0),
            });
            let mut model = QwenDenseCpu::open_with_kernel(&gguf, &config, kernel.clone()).unwrap();
            let actual = model.prefill_transaction(&tokens).unwrap();
            assert_eq!(
                kernel.completed.load(Ordering::Relaxed),
                1,
                "{name}: one and only one completed source region required"
            );
            assert_eq!(
                actual, baseline,
                "{name}: source-correct modeled completion changed complete trained logits"
            );
            assert_eq!(
                model.state_fingerprint(),
                expected_root,
                "{name}: published StateRoot changed"
            );
            assert_eq!(model.position(), 1);
            println!(
                "P07_B_SOURCE_FULL_MODEL_CLOSED_REGION_PARITY tensor={name} full_vocab={} source_roots=1",
                actual.len()
            );
        }
    }

    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires local pinned Qwen3.5 2B model GGUF"]
    fn trained_source_codec_and_mixed_gemv_match_native_cpu_bitwise() {
        use zerogravity_cpu::weights::CpuWeights;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("target/models/qwen35-2b/model.gguf");
        let weights = CpuWeights::open(&path).unwrap();
        // Every listed source is an actual matrix in the mixed-quant GGUF.
        let source_tensors = [
            "blk.0.ffn_down.weight",  // Q4_1 × Q8_1
            "blk.0.ssm_out.weight",   // Q8_0 × Q8_0
            "blk.11.attn_q.weight",   // Q6_K × Q8_K
            "blk.0.ssm_alpha.weight", // F32 AVX2 4x8 FMA dot
        ];
        for (index, name) in source_tensors.into_iter().enumerate() {
            let tensor = weights.tensor(name).unwrap();
            let cols = tensor.shape[0] as usize;
            let rows = tensor.shape[1] as usize;
            // Use real source checkpoint embeddings when dimensions match
            // and deterministic mixed-sign probes otherwise. The latter
            // catch Q8_K signed-max and low-amplitude quantization edge cases.
            let input = (0..cols)
                .map(|i| (((i * 113 + index * 31) % 997) as f32 - 498.0) / 371.0)
                .collect::<Vec<_>>();
            let expected = weights.matvec(name, &input).unwrap();
            struct OfflineSourceProvider;
            impl MatVecKernel for OfflineSourceProvider {
                fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
                    let prepared = prepare(region)?;
                    Ok(RegionExecution::Completed(emulate(&prepared)))
                }
            }
            let actual = weights
                .matvec_with(name, &input, Some(&OfflineSourceProvider))
                .unwrap();
            assert_eq!(expected.len(), actual.len());
            let first = expected
                .iter()
                .zip(actual.iter())
                .position(|(&a, &b)| a.to_bits() != b.to_bits());
            assert!(
                first.is_none(),
                "{name}: first source SIMD/FMA difference at {first:?}: expected={:?} actual={:?}",
                first.map(|i| expected[i]),
                first.map(|i| actual[i])
            );
            println!(
                "P07_B_TRAINED_MIXED_BIT_EXACT kind={} tensor={name} rows={rows} cols={cols}",
                tensor.kind.name()
            );
        }
    }
}
