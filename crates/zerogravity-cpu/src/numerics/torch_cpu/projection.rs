//! Pinned Torch CPU AVX2 BF16 output-first projection; original dot reduction.

// Numerical topology follows PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb,
// ReducedPrecisionFloatGemvFastPathKernel.cpp and functional_base.h (AVX2).
// That reduction carries this ggml attribution; no foreign runtime is called.
//
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
pub(crate) fn dot_bf16_cpu_avx2(row: &[u8], x: &[f32]) -> f32 {
    let weight = |i: usize| {
        let at = 2 * i;
        f32::from_bits(u32::from(u16::from_le_bytes([row[at], row[at + 1]])) << 16)
    };
    let reduce = |v: [f32; 8]| ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
    let aligned = x.len() / 64 * 64;
    let vector_end = x.len() / 16 * 16;
    let mut registers = [[0.0f32; 8]; 8];
    for chunk in (0..aligned).step_by(64) {
        for (register, lanes) in registers.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let i = chunk + register * 8 + lane;
                *sum = weight(i).mul_add(x[i], *sum);
            }
        }
    }
    for offset in [4, 2, 1] {
        let (low, high) = registers.split_at_mut(offset);
        for (lanes, other) in low.iter_mut().zip(high.iter()) {
            for (sum, &value) in lanes.iter_mut().zip(other) {
                *sum += value;
            }
        }
    }
    let mut result = reduce(registers[0]);
    let mut tail = [0.0f32; 8];
    for chunk in (aligned..vector_end).step_by(16) {
        for (lane, sum) in tail.iter_mut().enumerate() {
            let i = chunk + lane;
            *sum = weight(i).mul_add(x[i], *sum);
            *sum = weight(i + 8).mul_add(x[i + 8], *sum);
        }
    }
    result += reduce(tail);
    for (i, &activation) in x.iter().enumerate().skip(vector_end) {
        result += weight(i) * activation;
    }
    result
}

/// Token-major source rows against an original `[output,input]` BF16 tensor.
/// Private BF16 results only; caller admits storage and retains source ownership.
/// The original F.linear transposed-weight path uses this same dot for each row.
/// This does not select a model, physical backend or recurrent execution phase.
pub fn project_bf16_rows(
    source: crate::accelerator::SourceIdentity,
    weights: zerogravity_core::source::SourceTensorRef<'_>,
    input: &[f32],
    rows: usize,
) -> crate::Result<Vec<f32>> {
    use crate::Error;
    use zerogravity_core::{
        encoding::{ScalarDType, StoredTensorEncoding},
        numerics::ActivationPrecision,
    };
    if !source.is_bound() || !(1..=64).contains(&rows) {
        return Err(Error::Malformed(
            "BF16 projection source/row geometry".into(),
        ));
    }
    weights.validate()?;
    if weights.kind != StoredTensorEncoding::Scalar(ScalarDType::Bf16) || weights.shape.len() != 2 {
        return Err(Error::Unsupported(
            "BF16 projection original output-first tensor".into(),
        ));
    }
    let dimension = |n: u64| {
        usize::try_from(n)
            .ok()
            .filter(|n| *n > 0 && *n <= u32::MAX as usize)
            .ok_or_else(|| Error::Limit("BF16 projection dimension".into()))
    };
    let outputs = dimension(weights.shape[0])?;
    let columns = dimension(weights.shape[1])?;
    if columns.checked_mul(rows) != Some(input.len()) {
        return Err(Error::Malformed(
            "BF16 projection complete input rows".into(),
        ));
    }
    ActivationPrecision::Bf16WidenedF32.validate_values(input)?;
    if weights
        .packed_bytes
        .chunks_exact(2)
        .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
    {
        return Err(Error::Malformed(
            "BF16 projection finite source weights".into(),
        ));
    }
    let count = outputs
        .checked_mul(rows)
        .filter(|n| n.checked_mul(4).is_some())
        .ok_or_else(|| Error::Limit("BF16 projection output extent".into()))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("BF16 projection output allocation".into()))?;
    if result.capacity() != count {
        return Err(Error::Limit("BF16 projection output capacity".into()));
    }
    for row in input.chunks_exact(columns) {
        for weight in weights.packed_bytes.chunks_exact(columns * 2) {
            result.push(super::norm::round_bf16(dot_bf16_cpu_avx2(weight, row))?);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod rows_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_bf16_vector_and_scalar_tails_match_pinned_actual_torch_golden_bits() {
        // Synthetic B1 F.linear, Torch cf30153..., CPU AVX2, no GPU.
        // x=1; w_i=(i%7-3)*2^-10, first=16384, last=-16384 if width>1.
        for (width, expected) in [
            (1, 0x4680_0000u32),
            (2, 0),
            (7, 0),
            (8, 0x3b80_0000),
            (9, 0),
            (15, 0x3b80_0000),
            (16, 0xbb40_0000),
            (17, 0xbb40_0000),
            (31, 0xbb40_0000),
            (32, 0xbb40_0000),
            (33, 0xbb40_0000),
            (47, 0xbb40_0000),
            (48, 0xbb00_0000),
            (49, 0),
            (63, 0),
            (64, 0x3b00_0000),
            (65, 0xba80_0000),
            (79, 0xba80_0000),
            (80, 0xbb40_0000),
            (95, 0xbb80_0000),
            (96, 0xbb40_0000),
            (127, 0x3b00_0000),
            (128, 0xbb40_0000),
            (129, 0xbb40_0000),
            (255, 0xbb00_0000),
            (256, 0xbb40_0000),
            (257, 0xbb40_0000),
            (1024, 0xbb40_0000),
        ] {
            let mut weight = (0..width)
                .map(|i| ((i % 7) as f32 - 3.0) * 2f32.powi(-10))
                .collect::<Vec<_>>();
            weight[0] = 16384.0;
            if width > 1 {
                weight[width - 1] = -16384.0;
            }
            let packed = weight
                .iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<_>>();
            let value = dot_bf16_cpu_avx2(&packed, &vec![1.0; width]);
            let word = value.to_bits();
            let rounded = word.wrapping_add(0x7fff + ((word >> 16) & 1)) & 0xffff_0000;
            assert_eq!(rounded, expected, "synthetic width={width}");
        }
    }
}
