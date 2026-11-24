//! Shared lowering of the existing explicit CPU BF16 dot contract.
// MIT License
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
use cubecl::prelude::*;

// Adjacent workgroups reuse the same weight row across independent prompt rows.
#[cube]
fn output_index(group: usize, requests: usize, #[comptime] rows: usize) -> usize {
    (group % requests) * rows + group / requests
}

#[cube]
fn rounded_index(
    output: usize,
    #[comptime] rows: usize,
    #[comptime] output_row_stride: usize,
) -> usize {
    if comptime!(rows == output_row_stride) {
        output
    } else {
        (output / rows) * output_row_stride + output % rows
    }
}

/// One packed BF16 word for this lane. The alternate pair-load path lets
/// adjacent lanes share the exact same u32 through a plane shuffle: only the
/// even lane performs the global load, while both lanes keep their original
/// low/high-half arithmetic. shared_pair=false compiles to the canonical
/// independent load with no runtime branch.
#[cube]
fn packed_word(values: &[u32], word: usize, lane: usize, #[comptime] shared_pair: bool) -> u32 {
    if comptime!(shared_pair) {
        let loaded = if lane % 2usize == 0usize {
            values[word]
        } else {
            0u32
        };
        let peer = plane_shuffle_xor(loaded, 1u32);
        if lane % 2usize == 0usize {
            loaded
        } else {
            peer
        }
    } else {
        values[word]
    }
}

/// Exact CPU BF16 dot/reduction shared by diagnostic and certified stores.
/// shared_pair changes only how the already-packed u32 reaches adjacent lanes;
/// each lane extracts the same original BF16 and keeps the same FMA and
/// reduction order.
#[cube]
fn project_value(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    requests: usize,
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] shared_pair: bool,
    #[comptime] input_certified: bool,
) -> (usize, f32, u32, u32) {
    let output = output_index(usize::cast_from(CUBE_POS_X), requests, rows);
    let request = output / rows;
    let row = output % rows;
    let lane = usize::cast_from(UNIT_POS);
    let mut sums = Shared::<[f32]>::new_slice(64usize);
    let mut flags = Shared::<[u32]>::new_slice(64usize);
    // Dynamic immutable +0 defeats CubeCL's literal FAdd(x,0) simplifier.
    // CPU initializes all64 partials with +0 and adds its empty-tail +0 last.
    let z = zero[0];
    let mut accumulator = z;
    let mut invalid = 0u32;
    if u32::reinterpret(z) != 0u32 {
        invalid |= 4u32;
    }
    for chunk in 0..columns / 64usize {
        let column = chunk * 64usize + lane;
        let wi = row * columns + column;
        let xi = request * columns + column;
        let packed_weight = packed_word(weights, wi / 2usize, lane, shared_pair);
        let packed_input = packed_word(input, xi / 2usize, lane, shared_pair);
        let wb = (packed_weight >> (u32::cast_from(wi % 2usize) * 16u32)) & 0xffffu32;
        let xb = (packed_input >> (u32::cast_from(xi % 2usize) * 16u32)) & 0xffffu32;
        if (wb & 0x7f80u32) == 0x7f80u32 {
            invalid |= 1u32;
        }
        if comptime!(!input_certified) && (xb & 0x7f80u32) == 0x7f80u32 {
            invalid |= 2u32;
        }
        accumulator = fma(
            f32::reinterpret(wb << 16u32),
            f32::reinterpret(xb << 16u32),
            accumulator,
        );
    }
    sums[lane] = accumulator;
    flags[lane] = invalid;
    // Fold integer status alongside the unchanged float tree/barriers.
    sync_cube();
    if lane < 32usize {
        sums[lane] = sums[lane] + sums[lane + 32usize];
        flags[lane] = flags[lane] | flags[lane + 32usize];
    }
    sync_cube();
    if lane < 16usize {
        sums[lane] = sums[lane] + sums[lane + 16usize];
        flags[lane] = flags[lane] | flags[lane + 16usize];
    }
    sync_cube();
    if lane < 8usize {
        sums[lane] = sums[lane] + sums[lane + 8usize];
        flags[lane] = flags[lane] | flags[lane + 8usize];
    }
    sync_cube();

    let mut result = 0.0f32;
    let mut rounded_bits = 0u32;
    let mut combined = 0u32;
    if lane == 0usize {
        let r0 = sums[0] + sums[4];
        let r1 = sums[1] + sums[5];
        let r2 = sums[2] + sums[6];
        let r3 = sums[3] + sums[7];
        result = ((r0 + r2) + (r1 + r3)) + z;
        let bits = u32::reinterpret(result);
        rounded_bits = (bits + (0x7fffu32 + ((bits >> 16u32) & 1u32))) & 0xffff0000u32;
        for at in 0..8usize {
            combined |= flags[at];
        }
        if (bits & 0x7f800000u32) == 0x7f800000u32 {
            combined |= 8u32;
        }
        if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
            combined |= 16u32;
        }
    }
    (output, result, rounded_bits, combined)
}

/// Exact wave32 sibling. One thread carries the two original CPU partials
/// lane and lane+32; each partial keeps its own FMA sequence, then the
/// first +32 reduction is performed locally before the unchanged 32->16->8
/// shared reduction. This preserves the CPU64 arithmetic tree while requiring
/// only one 32-lane hardware wave per output row. Current gfx1150 Qwen keeps
/// the 64-thread kernel: output16 serialized project time was 577.7ms here vs
/// 561.1ms selected, and whole-model wall time regressed.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn project_cpu_bf16_wave32(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    accumulation: &mut [f32],
    rounded: &mut [f32],
    status: &mut [u32],
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] output_row_stride: usize,
) {
    let group = usize::cast_from(CUBE_POS_X);
    let requests = accumulation.len() / rows;
    let output = output_index(group, requests, rows);
    let request = output / rows;
    let row = output % rows;
    let lane = usize::cast_from(UNIT_POS);

    let z = zero[0];
    let mut low = z;
    let mut high = z;
    let mut low_invalid = 0u32;
    let mut high_invalid = 0u32;
    if u32::reinterpret(z) != 0u32 {
        low_invalid |= 4u32;
        high_invalid |= 4u32;
    }

    for chunk in 0..columns / 64usize {
        let low_column = chunk * 64usize + lane;
        let high_column = low_column + 32usize;

        let low_wi = row * columns + low_column;
        let high_wi = row * columns + high_column;
        let low_xi = request * columns + low_column;
        let high_xi = request * columns + high_column;

        let low_wb =
            (weights[low_wi / 2usize] >> (u32::cast_from(low_wi % 2usize) * 16u32)) & 0xffffu32;
        let high_wb =
            (weights[high_wi / 2usize] >> (u32::cast_from(high_wi % 2usize) * 16u32)) & 0xffffu32;
        let low_xb =
            (input[low_xi / 2usize] >> (u32::cast_from(low_xi % 2usize) * 16u32)) & 0xffffu32;
        let high_xb =
            (input[high_xi / 2usize] >> (u32::cast_from(high_xi % 2usize) * 16u32)) & 0xffffu32;

        if (low_wb & 0x7f80u32) == 0x7f80u32 {
            low_invalid |= 1u32;
        }
        if (high_wb & 0x7f80u32) == 0x7f80u32 {
            high_invalid |= 1u32;
        }
        if (low_xb & 0x7f80u32) == 0x7f80u32 {
            low_invalid |= 2u32;
        }
        if (high_xb & 0x7f80u32) == 0x7f80u32 {
            high_invalid |= 2u32;
        }

        low = fma(
            f32::reinterpret(low_wb << 16u32),
            f32::reinterpret(low_xb << 16u32),
            low,
        );
        high = fma(
            f32::reinterpret(high_wb << 16u32),
            f32::reinterpret(high_xb << 16u32),
            high,
        );
    }

    // Exactly the original first reduction stage: partial[lane] + partial[lane+32].
    let mut sums = Shared::<[f32]>::new_slice(32usize);
    let mut flags = Shared::<[u32]>::new_slice(32usize);
    sums[lane] = low + high;
    flags[lane] = low_invalid | high_invalid;
    sync_cube();

    if lane < 16usize {
        sums[lane] = sums[lane] + sums[lane + 16usize];
        flags[lane] = flags[lane] | flags[lane + 16usize];
    }
    sync_cube();
    if lane < 8usize {
        sums[lane] = sums[lane] + sums[lane + 8usize];
        flags[lane] = flags[lane] | flags[lane + 8usize];
    }
    sync_cube();

    if lane == 0usize {
        let r0 = sums[0] + sums[4];
        let r1 = sums[1] + sums[5];
        let r2 = sums[2] + sums[6];
        let r3 = sums[3] + sums[7];
        let result = ((r0 + r2) + (r1 + r3)) + z;
        let bits = u32::reinterpret(result);
        let rounded_bits = (bits + (0x7fffu32 + ((bits >> 16u32) & 1u32))) & 0xffff0000u32;
        let mut combined = 0u32;
        for at in 0..8usize {
            combined |= flags[at];
        }
        if (bits & 0x7f800000u32) == 0x7f800000u32 {
            combined |= 8u32;
        }
        if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
            combined |= 16u32;
        }
        accumulation[output] = result;
        rounded[rounded_index(output, rows, output_row_stride)] = f32::reinterpret(rounded_bits);
        status[output] = combined;
    }
}

/// One [request,original output row] per64-thread workgroup. The lane index
/// is exactly CPU register*8+lane; no subgroup reduction or TensorCore math.
#[cube(launch_unchecked)]
pub(crate) fn project_cpu_bf16(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    accumulation: &mut [f32],
    rounded: &mut [f32],
    status: &mut [u32],
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] output_row_stride: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    let (output, result, rounded_bits, combined) = project_value(
        weights,
        input,
        zero,
        accumulation.len() / rows,
        rows,
        columns,
        false,
        false,
    );
    if lane == 0usize {
        accumulation[output] = result;
        rounded[rounded_index(output, rows, output_row_stride)] = f32::reinterpret(rounded_bits);
        status[output] = combined;
    }
}

/// Exact packed-load sibling. Adjacent lanes share each packed u32 with a
/// plane XOR shuffle, halving the global packed-word loads in principle while
/// preserving the canonical64 independent partials and reduction tree.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn project_cpu_bf16_pair_load(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    accumulation: &mut [f32],
    rounded: &mut [f32],
    status: &mut [u32],
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] output_row_stride: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    let (output, result, rounded_bits, combined) = project_value(
        weights,
        input,
        zero,
        accumulation.len() / rows,
        rows,
        columns,
        true,
        false,
    );
    if lane == 0usize {
        accumulation[output] = result;
        rounded[rounded_index(output, rows, output_row_stride)] = f32::reinterpret(rounded_bits);
        status[output] = combined;
    }
}

/// Exact input-certified sibling. The pack phase has already published the
/// packed-input nonfinite bit into shared status, so the hot GEMM keeps only
/// weight/result checks while preserving every arithmetic operation.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn project_cpu_bf16_input_certified(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    accumulation: &mut [f32],
    rounded: &mut [f32],
    status: &mut [u32],
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] output_row_stride: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    let (output, result, rounded_bits, combined) = project_value(
        weights,
        input,
        zero,
        accumulation.len() / rows,
        rows,
        columns,
        false,
        true,
    );
    if lane == 0usize {
        accumulation[output] = result;
        rounded[rounded_index(output, rows, output_row_stride)] = f32::reinterpret(rounded_bits);
        status[output] = combined;
    }
}

/// Singleton specialization that computes two adjacent output rows in one
/// workgroup. Each lane loads the packed activation once per K64 chunk and reuses
/// that exact BF16 value across two independent row accumulators. Every row keeps
/// the scalar kernel's FMA order, reduction tree, rounding and status contract.
/// Current gfx1150 Qwen keeps rows1: the dominant3584x1024 shape flipped sign
/// between repeats, contraction lost materially, and other expansion gains were small.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn project_cpu_bf16_rows2(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    accumulation: &mut [f32],
    rounded: &mut [f32],
    status: &mut [u32],
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] output_row_stride: usize,
) {
    let group = usize::cast_from(CUBE_POS_X);
    let requests = accumulation.len() / rows;
    let request = group % requests;
    let first_row = (group / requests) * 2usize;
    let lane = usize::cast_from(UNIT_POS);

    let z = zero[0];
    let mut zero_flag = 0u32;
    if u32::reinterpret(z) != 0u32 {
        zero_flag |= 4u32;
    }

    let mut accumulators = Array::<f32>::new(2usize);
    let mut invalids = Array::<u32>::new(2usize);
    for local in 0..2usize {
        accumulators[local] = z;
        invalids[local] = zero_flag;
    }

    for chunk in 0..columns / 64usize {
        let column = chunk * 64usize + lane;
        let xi = request * columns + column;
        let xb = (input[xi / 2usize] >> (u32::cast_from(xi % 2usize) * 16u32)) & 0xffffu32;
        let input_invalid = (xb & 0x7f80u32) == 0x7f80u32;
        let x = f32::reinterpret(xb << 16u32);

        for local in 0..2usize {
            let row = first_row + local;
            let wi = row * columns + column;
            let wb = (weights[wi / 2usize] >> (u32::cast_from(wi % 2usize) * 16u32)) & 0xffffu32;
            if (wb & 0x7f80u32) == 0x7f80u32 {
                invalids[local] |= 1u32;
            }
            if input_invalid {
                invalids[local] |= 2u32;
            }
            accumulators[local] = fma(f32::reinterpret(wb << 16u32), x, accumulators[local]);
        }
    }

    let mut sums = Shared::<[f32]>::new_slice(128usize);
    let mut flags = Shared::<[u32]>::new_slice(128usize);
    for local in 0..2usize {
        let at = local * 64usize + lane;
        sums[at] = accumulators[local];
        flags[at] = invalids[local];
    }
    sync_cube();

    if lane < 32usize {
        for local in 0..2usize {
            let at = local * 64usize + lane;
            sums[at] = sums[at] + sums[at + 32usize];
            flags[at] = flags[at] | flags[at + 32usize];
        }
    }
    sync_cube();
    if lane < 16usize {
        for local in 0..2usize {
            let at = local * 64usize + lane;
            sums[at] = sums[at] + sums[at + 16usize];
            flags[at] = flags[at] | flags[at + 16usize];
        }
    }
    sync_cube();
    if lane < 8usize {
        for local in 0..2usize {
            let at = local * 64usize + lane;
            sums[at] = sums[at] + sums[at + 8usize];
            flags[at] = flags[at] | flags[at + 8usize];
        }
    }
    sync_cube();

    if lane == 0usize {
        for local in 0..2usize {
            let at = local * 64usize;
            let r0 = sums[at] + sums[at + 4usize];
            let r1 = sums[at + 1usize] + sums[at + 5usize];
            let r2 = sums[at + 2usize] + sums[at + 6usize];
            let r3 = sums[at + 3usize] + sums[at + 7usize];
            let result = ((r0 + r2) + (r1 + r3)) + z;
            let bits = u32::reinterpret(result);
            let rounded_bits = (bits + (0x7fffu32 + ((bits >> 16u32) & 1u32))) & 0xffff0000u32;
            let mut combined = 0u32;
            for lane_flag in 0..8usize {
                combined |= flags[at + lane_flag];
            }
            if (bits & 0x7f800000u32) == 0x7f800000u32 {
                combined |= 8u32;
            }
            if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
                combined |= 16u32;
            }

            let output = request * rows + first_row + local;
            accumulation[output] = result;
            rounded[rounded_index(output, rows, output_row_stride)] =
                f32::reinterpret(rounded_bits);
            status[output] = combined;
        }
    }
}

/// Retained singleton specialization that computes four adjacent output rows in one
/// workgroup. Each lane loads the packed activation once per K64 chunk and reuses
/// that exact BF16 value across four independent row accumulators. Every row keeps
/// the scalar kernel's FMA order, reduction tree, rounding and status contract.
/// Current gfx1150 Qwen keeps the scalar row kernel because rows4 loses wall time.
#[allow(dead_code)]
#[cube(launch_unchecked)]
pub(crate) fn project_cpu_bf16_rows4(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    accumulation: &mut [f32],
    rounded: &mut [f32],
    status: &mut [u32],
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] output_row_stride: usize,
) {
    let group = usize::cast_from(CUBE_POS_X);
    let requests = accumulation.len() / rows;
    let request = group % requests;
    let first_row = (group / requests) * 4usize;
    let lane = usize::cast_from(UNIT_POS);

    let z = zero[0];
    let mut zero_flag = 0u32;
    if u32::reinterpret(z) != 0u32 {
        zero_flag |= 4u32;
    }

    let mut accumulators = Array::<f32>::new(4usize);
    let mut invalids = Array::<u32>::new(4usize);
    for local in 0..4usize {
        accumulators[local] = z;
        invalids[local] = zero_flag;
    }

    for chunk in 0..columns / 64usize {
        let column = chunk * 64usize + lane;
        let xi = request * columns + column;
        let xb = (input[xi / 2usize] >> (u32::cast_from(xi % 2usize) * 16u32)) & 0xffffu32;
        let input_invalid = (xb & 0x7f80u32) == 0x7f80u32;
        let x = f32::reinterpret(xb << 16u32);

        for local in 0..4usize {
            let row = first_row + local;
            let wi = row * columns + column;
            let wb = (weights[wi / 2usize] >> (u32::cast_from(wi % 2usize) * 16u32)) & 0xffffu32;
            if (wb & 0x7f80u32) == 0x7f80u32 {
                invalids[local] |= 1u32;
            }
            if input_invalid {
                invalids[local] |= 2u32;
            }
            accumulators[local] = fma(f32::reinterpret(wb << 16u32), x, accumulators[local]);
        }
    }

    let mut sums = Shared::<[f32]>::new_slice(256usize);
    let mut flags = Shared::<[u32]>::new_slice(256usize);
    for local in 0..4usize {
        let at = local * 64usize + lane;
        sums[at] = accumulators[local];
        flags[at] = invalids[local];
    }
    sync_cube();

    if lane < 32usize {
        for local in 0..4usize {
            let at = local * 64usize + lane;
            sums[at] = sums[at] + sums[at + 32usize];
            flags[at] = flags[at] | flags[at + 32usize];
        }
    }
    sync_cube();
    if lane < 16usize {
        for local in 0..4usize {
            let at = local * 64usize + lane;
            sums[at] = sums[at] + sums[at + 16usize];
            flags[at] = flags[at] | flags[at + 16usize];
        }
    }
    sync_cube();
    if lane < 8usize {
        for local in 0..4usize {
            let at = local * 64usize + lane;
            sums[at] = sums[at] + sums[at + 8usize];
            flags[at] = flags[at] | flags[at + 8usize];
        }
    }
    sync_cube();

    if lane == 0usize {
        for local in 0..4usize {
            let at = local * 64usize;
            let r0 = sums[at] + sums[at + 4usize];
            let r1 = sums[at + 1usize] + sums[at + 5usize];
            let r2 = sums[at + 2usize] + sums[at + 6usize];
            let r3 = sums[at + 3usize] + sums[at + 7usize];
            let result = ((r0 + r2) + (r1 + r3)) + z;
            let bits = u32::reinterpret(result);
            let rounded_bits = (bits + (0x7fffu32 + ((bits >> 16u32) & 1u32))) & 0xffff0000u32;
            let mut combined = 0u32;
            for lane_flag in 0..8usize {
                combined |= flags[at + lane_flag];
            }
            if (bits & 0x7f800000u32) == 0x7f800000u32 {
                combined |= 8u32;
            }
            if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
                combined |= 16u32;
            }

            let output = request * rows + first_row + local;
            accumulation[output] = result;
            rounded[rounded_index(output, rows, output_row_stride)] =
                f32::reinterpret(rounded_bits);
            status[output] = combined;
        }
    }
}

/// Production sibling: preserve the exact projection arithmetic while certifying
/// validity directly into the shared flight status instead of a per-output vector.
#[cube(launch_unchecked)]
pub(crate) fn project_cpu_bf16_certified(
    weights: &[u32],
    input: &[u32],
    zero: &[f32],
    accumulation: &mut [f32],
    rounded: &mut [f32],
    status: &[Atomic<u32>],
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] output_row_stride: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    let (output, result, rounded_bits, combined) = project_value(
        weights,
        input,
        zero,
        accumulation.len() / rows,
        rows,
        columns,
        false,
        false,
    );
    if lane == 0usize {
        accumulation[output] = result;
        rounded[rounded_index(output, rows, output_row_stride)] = f32::reinterpret(rounded_bits);
        if combined != 0u32 {
            status[0].fetch_or(combined);
        }
    }
}

#[cfg(test)]
mod tests;
