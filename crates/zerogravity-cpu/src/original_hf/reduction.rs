//! Pinned Torch CPU AVX2 FP32 reductions, independent of model/token loops.
//!
//! Explicit closed operators only. GGUF/source kernels remain unchanged.
//! Source: PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb,
//! SumKernel.cpp cascade_sum and vec256_float.h sqrt/div rsqrt. This does
//! not qualify CUDA FLA, full HF inference or another runtime/shape profile.

use crate::accelerator::SourceIdentity;
use crate::{Error, Result};

const WIDTH: usize = 128;
const MAX_ELEMENTS: usize = 1_048_576;

fn validate(source: SourceIdentity, input: &[f32]) -> Result<()> {
    if !source.is_bound() || input.is_empty() || !input.len().is_multiple_of(WIDTH) {
        return Err(Error::Malformed(
            "pinned CPU FP32 reduction source/shape".into(),
        ));
    }
    if input.len() > MAX_ELEMENTS {
        return Err(Error::Limit(
            "pinned CPU FP32 reduction element bound".into(),
        ));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite pinned CPU FP32 reduction input".into(),
        ));
    }
    Ok(())
}

fn finite(output: Vec<f32>) -> Result<Vec<f32>> {
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite pinned CPU FP32 reduction output".into(),
        ));
    }
    Ok(output)
}

/// Sum independent contiguous128-element FP32 rows. Source order is four
/// 8-lane partial vectors, combined sequentially, then eight sequential lanes.
pub fn sum_inner_128(source: SourceIdentity, input: &[f32]) -> Result<Vec<f32>> {
    validate(source, input)?;
    let mut output = Vec::with_capacity(input.len() / WIDTH);
    for row in input.chunks_exact(WIDTH) {
        let mut partial = [[0.0f32; 8]; 4];
        for block in 0..4 {
            for (group, lanes) in partial.iter_mut().enumerate() {
                for (lane, sum) in lanes.iter_mut().enumerate() {
                    *sum += row[block * 32 + group * 8 + lane];
                }
            }
        }
        let mut vector = partial[0];
        for other in &partial[1..] {
            for (sum, &value) in vector.iter_mut().zip(other) {
                *sum += value;
            }
        }
        let mut sum = 0.0f32;
        for value in vector {
            sum += value;
        }
        output.push(sum);
    }
    finite(output)
}

/// Actual eager L2 order: materialized FP32 squares, source FP32 sum, +eps,
/// FP32 sqrt then divide1, and FP32 input*inverse. No mean/gamma rewrite.
pub fn l2_norm_128(source: SourceIdentity, input: &[f32], eps: f32) -> Result<Vec<f32>> {
    validate(source, input)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Malformed(
            "invalid pinned CPU FP32 L2 epsilon".into(),
        ));
    }
    let squares = input.iter().map(|&value| value * value).collect::<Vec<_>>();
    let sums = sum_inner_128(source, &squares)?;
    let mut output = Vec::with_capacity(input.len());
    for (row, sum) in input.chunks_exact(WIDTH).zip(sums) {
        let denominator = sum + eps;
        if !denominator.is_finite() {
            return Err(Error::Malformed(
                "nonfinite pinned CPU FP32 L2 denominator".into(),
            ));
        }
        let inverse = denominator.sqrt().recip();
        output.extend(row.iter().map(|&value| value * inverse));
    }
    finite(output)
}

/// Reduce already materialized `[head,key128,value128]` FP32 products across
/// key. Each value retains eight sequential16-key subtotals; no FMA or head
/// mixing. Key-major input is explicit; other layouts need bit transport.
pub fn sum_key_128(source: SourceIdentity, input: &[f32], heads: usize) -> Result<Vec<f32>> {
    let expected = heads
        .checked_mul(WIDTH)
        .and_then(|elements| elements.checked_mul(WIDTH))
        .filter(|&elements| heads > 0 && elements <= MAX_ELEMENTS)
        .ok_or_else(|| Error::Limit("pinned CPU key reduction head/element bound".into()))?;
    if input.len() != expected {
        return Err(Error::Malformed(
            "pinned CPU key reduction layout/length".into(),
        ));
    }
    validate(source, input)?;
    let mut output = Vec::with_capacity(heads * WIDTH);
    for head in input.chunks_exact(WIDTH * WIDTH) {
        let mut carry = [0.0f32; WIDTH];
        for block in 0..8 {
            let mut subtotal = [0.0f32; WIDTH];
            for key in block * 16..(block + 1) * 16 {
                for (value, sum) in subtotal.iter_mut().enumerate() {
                    *sum += head[key * WIDTH + value];
                }
            }
            for (sum, value) in carry.iter_mut().zip(subtotal) {
                *sum += value;
            }
        }
        output.extend(carry);
    }
    finite(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }

    fn exact(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "word {index}");
        }
    }

    #[test]
    fn contiguous128_retains_partial_lanes_before_sequential_reduction() {
        let mut input = [0.0; WIDTH];
        input[0] = 16384.0;
        input[1] = 2f32.powi(-10);
        input[32] = -16384.0;
        exact(&sum_inner_128(source(), &input).unwrap(), &[2f32.powi(-10)]);
        assert_eq!(input.iter().copied().sum::<f32>().to_bits(), 0);
    }

    #[test]
    fn key_axis128_uses_subtotals_and_preserves_head_value_coordinates() {
        let mut products = vec![0.0; 2 * WIDTH * WIDTH];
        // Pinned Torch outer sum loses this half-ULP in the second subtotal.
        // 16384 has a finer toward-zero ULP and does not distinguish the paths.
        products[0] = 16385.0;
        products[16 * WIDTH] = -16385.0;
        products[17 * WIDTH] = 2f32.powi(-10);
        for key in 0..WIDTH {
            products[key * WIDTH + 1] = 1.0;
            products[WIDTH * WIDTH + key * WIDTH + 127] = -2.0;
        }
        let sums = sum_key_128(source(), &products, 2).unwrap();
        assert_eq!(sums[0].to_bits(), 0);
        assert_eq!(sums[1], 128.0);
        assert_eq!(sums[WIDTH + 127], -256.0);
        assert!(
            sums[2..WIDTH + 127]
                .iter()
                .all(|value| value.to_bits() == 0)
        );
        let serial = (0..WIDTH).map(|key| products[key * WIDTH]).sum::<f32>();
        assert_eq!(serial, 2f32.powi(-10));
    }

    #[test]
    fn l2_preserves_zero_sign_and_rejects_invalid_bindings_before_math() {
        let source = source();
        let mut input = [0.0; WIDTH];
        input[0] = -0.0;
        let output = l2_norm_128(source, &input, 1e-6).unwrap();
        assert_eq!(output[0].to_bits(), (-0.0f32).to_bits());
        assert!(output[1..].iter().all(|value| value.to_bits() == 0));
        for invalid in [&[][..], &input[..127], &input[..1]] {
            assert!(sum_inner_128(source, invalid).is_err());
            assert!(l2_norm_128(source, invalid, 1e-6).is_err());
        }
        assert!(sum_inner_128(SourceIdentity::unbound(), &input).is_err());
        assert!(l2_norm_128(SourceIdentity::unbound(), &input, 1e-6).is_err());
        for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(l2_norm_128(source, &input, eps).is_err());
        }
        input[1] = f32::NAN;
        assert!(sum_inner_128(source, &input).is_err());
        assert!(l2_norm_128(source, &input, 1e-6).is_err());
        input.fill(f32::MAX);
        assert!(sum_inner_128(source, &input).is_err());
        assert!(l2_norm_128(source, &input, 1e-6).is_err());
    }

    #[test]
    fn reduction_limits_refuse_malformed_layout_and_overflow() {
        let source = source();
        assert!(sum_key_128(source, &[], 0).is_err());
        assert!(sum_key_128(source, &[], usize::MAX).is_err());
        assert!(sum_key_128(source, &[0.0; WIDTH], 1).is_err());
        let mut products = vec![0.0; WIDTH * WIDTH];
        assert!(sum_key_128(SourceIdentity::unbound(), &products, 1).is_err());
        products[255] = f32::INFINITY;
        assert!(sum_key_128(source, &products, 1).is_err());
        products.fill(f32::MAX);
        assert!(sum_key_128(source, &products, 1).is_err());
        let oversized = vec![0.0; MAX_ELEMENTS + WIDTH];
        assert!(sum_inner_128(source, &oversized).is_err());
        assert!(l2_norm_128(source, &oversized, 1e-6).is_err());
        assert!(sum_key_128(source, &[], MAX_ELEMENTS / (WIDTH * WIDTH) + 1).is_err());
    }

    // Captured original .8B source36F32 profile, tokens11,353, layer0.
    // Torch cf30153..., modeling762feb6c...412c3b, stock pinned CPU AVX2;
    // actual public replays match untouched original frame outputs bitwise.
    // Four head0 L2 records and four actual key-product columns per recurrent
    // reduction. Other test columns are zero; no cross-column arithmetic.
    // Expected words come from actual Torch outputs, not these Rust helpers.
    // Full16-head observed replay is retained in the ignored run.
    // Fixture SHA256:9ee2e4e97fc8a2b2a2c48db4e9bec2fed9d0350f1e6b319031b879274c62edec.
    #[test]
    fn actual_pinned_qwen_l2_and_recurrent_products_match_all_golden_bits() {
        let bytes = include_bytes!("reduction/pinned_avx2_128.f32le.bin");
        let mut cursor = 0;
        fn word(bytes: &[u8], cursor: &mut usize) -> u32 {
            let word = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
            *cursor += 4;
            word
        }
        fn values(bytes: &[u8], cursor: &mut usize, count: usize) -> Vec<f32> {
            (0..count)
                .map(|_| f32::from_bits(word(bytes, cursor)))
                .collect()
        }
        assert_eq!(&bytes[..8], b"ZGTFP128");
        cursor += 8;
        assert_eq!(word(bytes, &mut cursor), 6);
        let source = source();
        for case in 0..6 {
            let kind = word(bytes, &mut cursor);
            let heads = word(bytes, &mut cursor) as usize;
            let counts = std::array::from_fn::<_, 4, _>(|_| word(bytes, &mut cursor) as usize);
            let input = values(bytes, &mut cursor, counts[0]);
            let sums = values(bytes, &mut cursor, counts[1]);
            let inverse = values(bytes, &mut cursor, counts[2]);
            let normalized = values(bytes, &mut cursor, counts[3]);
            if kind == 1 {
                assert!(case < 4);
                assert_eq!(heads, 1);
                assert_eq!(input.len(), heads * WIDTH);
                assert_eq!(inverse.len(), heads);
                let products = input.iter().map(|&value| value * value).collect::<Vec<_>>();
                let actual_sums = sum_inner_128(source, &products).unwrap();
                exact(&actual_sums, &sums);
                let actual_inverse = actual_sums
                    .into_iter()
                    .map(|value| (value + 1e-6).sqrt().recip())
                    .collect::<Vec<_>>();
                exact(&actual_inverse, &inverse);
                exact(&l2_norm_128(source, &input, 1e-6).unwrap(), &normalized);
            } else {
                assert_eq!(kind, 2);
                assert!(case >= 4);
                assert_eq!(heads, 1);
                assert!(inverse.is_empty() && normalized.is_empty());
                assert_eq!(input.len(), 4 * WIDTH);
                assert_eq!(sums.len(), 4);
                let mut products = vec![0.0; WIDTH * WIDTH];
                let mut expected = vec![0.0; WIDTH];
                for (compact, column) in [0, 7, 63, 127].into_iter().enumerate() {
                    for key in 0..WIDTH {
                        products[key * WIDTH + column] = input[compact * WIDTH + key];
                    }
                    expected[column] = sums[compact];
                }
                exact(&sum_key_128(source, &products, heads).unwrap(), &expected);
            }
        }
        assert_eq!(cursor, bytes.len());
    }
}
