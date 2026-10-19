//! Closed pinned Torch CPU BF16 sigmoid for ONE contiguous iterator span.
//!
//! Torch cf30153c4c131c8164ee7798e5022d810682e2cb UnaryOpsKernel.cpp36–52
//! and Loops.h: two BF16 vectors (32 elements) per iteration, scalar tail.
//! Vector lanes reuse the parent's pinned SLEEF exp. Scalar lanes follow the
//! installed Ubuntu glibc2.39-0ubuntu8.9 expf FMA target, ELF0x7dc40, libm SHA
//! fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0.
//! Scalar algorithm/table adapted under MIT from Arm optimized-routines
//! f67e7daa5457646fad9f75580263dba654c988aa math/{expf,exp2f_data}.c;
//! installed F64 FMA contraction/order is explicit. No host exp/libm call.
//! No general exp/F32 sigmoid, strided tensor, coefficient/model eligibility.
//!
//! Adapted portions Copyright (c) 2017-2025, Arm Limited.
//! Data portions Copyright (c) 2017-2018, Arm Limited.
//! SPDX-License-Identifier: MIT

/*
MIT License

Copyright (c) 1999-2022, Arm Limited.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use super::{MAX_ELEMENTS, sleef_exp_finite_lane};
use crate::accelerator::SourceIdentity;
use crate::{Error, Result};

const VECTOR_BLOCK: usize = 32;
const EXP2_TABLE: [u64; 32] = [
    0x3ff0000000000000,
    0x3fefd9b0d3158574,
    0x3fefb5586cf9890f,
    0x3fef9301d0125b51,
    0x3fef72b83c7d517b,
    0x3fef54873168b9aa,
    0x3fef387a6e756238,
    0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb,
    0x3feedea64c123422,
    0x3feece086061892d,
    0x3feebfdad5362a27,
    0x3feeb42b569d4f82,
    0x3feeab07dd485429,
    0x3feea47eb03a5585,
    0x3feea09e667f3bcd,
    0x3fee9f75e8ec5f74,
    0x3feea11473eb0187,
    0x3feea589994cce13,
    0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5,
    0x3feec49182a3f090,
    0x3feed503b23e255d,
    0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47,
    0x3fef199bdd85529c,
    0x3fef3720dcef9069,
    0x3fef5818dcfba487,
    0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da,
    0x3fefd0765b6e4540,
];

// Private scalar exp intermediate for finite exact-BF16 inputs only. This is
// not a general expf API: errno/fenv and raw underflow intermediates are outside
// the public sigmoid result contract. The source's may-underflow branch cannot
// affect 1/(1+exp(x)) once the FP32 add1 rounds to1.
fn scalar_exp_bf16_lane(input: f32) -> f32 {
    if input > f32::from_bits(0x42b1_7217) {
        return f32::INFINITY;
    }
    if input < f32::from_bits(0xc2cf_f1b4) {
        return 0.0;
    }
    let input = f64::from(input);
    let inv_ln2_32 = f64::from_bits(0x4047_1547_652b_82fe);
    let shift = f64::from_bits(0x4338_0000_0000_0000);
    // ELF7dc80 FMA includes SHIFT; ELF7dc98 reconstructs the residual with
    // another FMA. Separate multiply/add would change the selected profile.
    let shifted = inv_ln2_32.mul_add(input, shift);
    let integer_bits = shifted.to_bits();
    let rounded = shifted - shift;
    let remainder = inv_ln2_32.mul_add(input, -rounded);
    let scale = f64::from_bits(
        EXP2_TABLE[(integer_bits & 31) as usize].wrapping_add(integer_bits.wrapping_shl(47)),
    );
    let high = f64::from_bits(0x3ebc_6af8_4b91_2394)
        .mul_add(remainder, f64::from_bits(0x3f2e_bfce_50fa_c4f3));
    let squared = remainder * remainder;
    let low = f64::from_bits(0x3f96_2e42_ff0c_52d6).mul_add(remainder, 1.0);
    // Source final scale multiply and F64→F32 cast are separate roundings.
    (high.mul_add(squared, low) * scale) as f32
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Pure private BF16 sigmoid result for one explicit contiguous iterator span.
/// Exact finite BF16 inputs stored in F32, nonempty/max1M. Vector32 prefix and
/// scalar remainder preserve the original call shape; do not flatten multiple
/// spans or treat a source identity as authentication/eligibility of payloads.
pub fn sigmoid_bf16_widened_f32_contiguous_span(
    source: SourceIdentity,
    input: &[f32],
) -> Result<Vec<f32>> {
    if !source.is_bound() || input.is_empty() {
        return Err(Error::Malformed(
            "pinned CPU BF16 sigmoid source/span".into(),
        ));
    }
    if input.len() > MAX_ELEMENTS {
        return Err(Error::Limit("pinned CPU BF16 sigmoid element bound".into()));
    }
    if input
        .iter()
        .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "pinned CPU sigmoid requires finite exact BF16 input".into(),
        ));
    }
    let prefix = input.len() / VECTOR_BLOCK * VECTOR_BLOCK;
    let output = input
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let exponential = if index < prefix {
                sleef_exp_finite_lane(-value)
            } else {
                scalar_exp_bf16_lane(-value)
            };
            // UnaryOpsKernel: separate FP32 add and division, then BF16 RNE.
            round_bf16(1.0 / (1.0 + exponential))
        })
        .collect::<Vec<_>>();
    if output
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(Error::Malformed(
            "nonfinite/out-of-range pinned CPU sigmoid".into(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn actual() -> (Vec<f32>, Vec<f32>) {
        include_str!("sigmoid_beta_golden.txt")
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut words = line.split_whitespace();
                let input = u16::from_str_radix(words.next().unwrap(), 16).unwrap();
                let beta = u16::from_str_radix(words.next().unwrap(), 16).unwrap();
                assert!(words.next().is_none());
                (
                    f32::from_bits(u32::from(input) << 16),
                    f32::from_bits(u32::from(beta) << 16),
                )
            })
            .unzip()
    }
    #[test]
    fn actual_original_beta_uses_literal_sixteen_element_scalar_calls() {
        let (input, expected) = actual();
        assert_eq!(input.len(), 32);
        for (input, expected) in input.chunks_exact(16).zip(expected.chunks_exact(16)) {
            let output = sigmoid_bf16_widened_f32_contiguous_span(source(), input).unwrap();
            for (actual, expected) in output.iter().zip(expected) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
    #[test]
    fn scalar_vector_and_mixed_spans_preserve_closed_edges_and_bf16_encoding() {
        let words = [
            0, 0x80000000, 0x00010000, 0x80010000, 0x00800000, 0x80800000, 0x7f7f0000, 0xff7f0000,
        ];
        let expected = [
            0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f800000, 0,
        ];
        for length in [1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65] {
            let input = (0..length)
                .map(|index| f32::from_bits(words[index % words.len()]))
                .collect::<Vec<_>>();
            let output = sigmoid_bf16_widened_f32_contiguous_span(source(), &input).unwrap();
            for (index, actual) in output.iter().enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected[index % expected.len()],
                    "span{length} lane{index}"
                );
            }
        }
    }
    #[test]
    fn source_encoding_finite_and_allocation_guards_refuse_before_math() {
        assert!(matches!(
            sigmoid_bf16_widened_f32_contiguous_span(SourceIdentity::unbound(), &[1.0]),
            Err(Error::Malformed(_))
        ));
        assert!(matches!(
            sigmoid_bf16_widened_f32_contiguous_span(source(), &[]),
            Err(Error::Malformed(_))
        ));
        for invalid in [
            f32::from_bits(0x3f800001),
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert!(matches!(
                sigmoid_bf16_widened_f32_contiguous_span(source(), &[invalid]),
                Err(Error::Malformed(_))
            ));
        }
        assert!(matches!(
            sigmoid_bf16_widened_f32_contiguous_span(source(), &vec![1.0; MAX_ELEMENTS + 1]),
            Err(Error::Limit(_))
        ));
    }
}
