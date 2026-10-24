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
) {
    let output = usize::cast_from(CUBE_POS_X);
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
        let wb = (weights[wi / 2usize] >> (u32::cast_from(wi % 2usize) * 16u32)) & 0xffffu32;
        let xb = (input[xi / 2usize] >> (u32::cast_from(xi % 2usize) * 16u32)) & 0xffffu32;
        if (wb & 0x7f80u32) == 0x7f80u32 {
            invalid |= 1u32;
        }
        if (xb & 0x7f80u32) == 0x7f80u32 {
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
    sync_cube();
    if lane < 32usize {
        let collapsed = sums[lane] + sums[lane + 32usize];
        sums[lane] = collapsed;
    }
    sync_cube();
    if lane < 16usize {
        let collapsed = sums[lane] + sums[lane + 16usize];
        sums[lane] = collapsed;
    }
    sync_cube();
    if lane < 8usize {
        let collapsed = sums[lane] + sums[lane + 8usize];
        sums[lane] = collapsed;
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
        for at in 0..64usize {
            combined |= flags[at];
        }
        if (bits & 0x7f800000u32) == 0x7f800000u32 {
            combined |= 8u32;
        }
        if (rounded_bits & 0x7f800000u32) == 0x7f800000u32 {
            combined |= 16u32;
        }
        accumulation[output] = result;
        rounded[output] = f32::reinterpret(rounded_bits);
        status[output] = combined;
    }
}
