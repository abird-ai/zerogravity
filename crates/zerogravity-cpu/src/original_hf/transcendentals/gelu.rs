//! Torch cf30153… Gelu.h BF16 Vec16→two FP32 Vec8→BF16.
//! SLEEF3.8/5a1d179 xtanhf/expk2f + df.h FMA_SP. Source attribution and
//! Boost license are retained in the parent/compensated module; Torch BSD
//! license is in softmax/PYTORCH-LICENSE.txt. No model/profile qualification.
use super::{MAX_ELEMENTS, SPAN, compensated::Compensated as Df};
use crate::{Error, Result, accelerator::SourceIdentity};

// Input abs<=source tanh saturation threshold. q0..13, so both scaling
// powers are normal and no overflow/subnormal exponent special case selects.
fn expk2_nonnegative(input: f32) -> Df {
    let quotient = (input * f32::from_bits(0x3fb8aa3b)).round_ties_even() as i32;
    let q = quotient as f32;
    let s = Df {
        high: input,
        low: 0.0,
    }
    .add_general_scalar(q * -f32::from_bits(0x3f317200));
    let s = s.add_general_scalar(q * -f32::from_bits(0x35bfbe8e));
    let u = f32::from_bits(0x394fb7ff)
        .mul_add(s.high, f32::from_bits(0x3ab6bf7c))
        .mul_add(s.high, f32::from_bits(0x3c08890d))
        .mul_add(s.high, f32::from_bits(0x3d2aaa5c));
    let t = s
        .multiply_scalar(u)
        .add_general_scalar(f32::from_bits(0x3e2aaaaa));
    let t = s.multiply_pair(t).add_general_scalar(0.5);
    let t = s.add_general_pair(s.square().multiply_pair(t));
    let t = Df::scalar_plus_fast(1.0, t);
    let first = quotient >> 1;
    let second = quotient - first;
    let a = f32::from_bits(((first + 127) as u32) << 23);
    let b = f32::from_bits(((second + 127) as u32) << 23);
    Df {
        high: (t.high * a) * b,
        low: (t.low * a) * b,
    }
}

/// Internal original vector lane, including source-defined saturation/sign.
/// NaN stays NaN; infinities select saturation before integer conversion.
pub(crate) fn tanh_finite_lane(input: f32) -> f32 {
    let magnitude = input.abs();
    let result = if magnitude > f32::from_bits(0x410aa123) {
        1.0
    } else {
        let positive = expk2_nonnegative(magnitude);
        let negative = positive.reciprocal();
        let ratio = positive
            .add_pair(negative.negate())
            .divide(positive.add_pair(negative));
        ratio.high + ratio.low
    };
    f32::from_bits(result.to_bits() ^ (input.to_bits() & 0x80000000))
}

/// Pinned complete vector spans only. No scalar-tail, general F32 GELU,
/// normal model-session grant, expected words or alternate mathematical path.
pub fn gelu_tanh_bf16_widened_f32_contiguous16(
    source: SourceIdentity,
    input: &[f32],
) -> Result<Vec<f32>> {
    if !source.is_bound() || input.is_empty() || !input.len().is_multiple_of(SPAN) {
        return Err(Error::Malformed(
            "pinned BF16 GELUtanh source/contiguous16 span".into(),
        ));
    }
    if input.len() > MAX_ELEMENTS {
        return Err(Error::Limit("pinned GELUtanh element bound".into()));
    }
    if input
        .iter()
        .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "pinned GELUtanh exact finite BF16 input".into(),
        ));
    }
    // Gelu.h uses double constants cast to FP32, explicit vector multiply/add.
    let beta = f32::from_bits(0x3f4c422a);
    let kappa = 0.044715f32;
    let mut output_values = Vec::with_capacity(input.len());
    for &x in input {
        let cube = (x * x) * x;
        let inner = beta * (x + kappa * cube);
        let output = (0.5 * x) * (1.0 + tanh_finite_lane(inner));
        output_values.push(crate::original_hf::norm::round_bf16(output)?);
    }
    Ok(output_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn authentic_initial_gelu_span_matches_strict_original_bf16_words() {
        let rows = include_str!("gelu_golden.txt")
            .lines()
            .filter(|x| !x.starts_with('#') && !x.is_empty())
            .map(|line| {
                let mut x = line.split_whitespace();
                let a = u32::from_str_radix(x.next().unwrap(), 16).unwrap();
                let b = u32::from_str_radix(x.next().unwrap(), 16).unwrap();
                assert!(x.next().is_none());
                (a, b)
            })
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 32);
        let input = rows
            .iter()
            .map(|(x, _)| f32::from_bits(*x))
            .collect::<Vec<_>>();
        let actual = gelu_tanh_bf16_widened_f32_contiguous16(
            SourceIdentity::for_new_allocation().unwrap(),
            &input,
        )
        .unwrap();
        for (i, (actual, (_, expected))) in actual.iter().zip(rows).enumerate() {
            assert_eq!(actual.to_bits(), expected, "authentic source lane{i}");
        }
    }
    #[test]
    fn source_saturation_signed_zero_and_finite_guards_are_preserved() {
        assert_eq!(tanh_finite_lane(0.0).to_bits(), 0);
        assert_eq!(tanh_finite_lane(-0.0).to_bits(), 0x80000000);
        assert_eq!(tanh_finite_lane(9.0), 1.0);
        assert_eq!(tanh_finite_lane(-9.0), -1.0);
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut input = [-5.125; 16];
        assert_eq!(
            gelu_tanh_bf16_widened_f32_contiguous16(source, &input).unwrap()[0].to_bits(),
            0x80000000
        );
        assert!(
            gelu_tanh_bf16_widened_f32_contiguous16(SourceIdentity::unbound(), &input).is_err()
        );
        assert!(gelu_tanh_bf16_widened_f32_contiguous16(source, &input[..15]).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::from_bits(0x3f800001)] {
            input[15] = bad;
            assert!(gelu_tanh_bf16_widened_f32_contiguous16(source, &input).is_err());
        }
    }
}
