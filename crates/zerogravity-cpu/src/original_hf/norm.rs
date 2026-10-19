//! Closed pinned CPU BF16 head RMS prefix, before gated SiLU.
//!
//! Torch cf30153c4c131c8164ee7798e5022d810682e2cb CPU mean uses sum then
//! divide128; rsqrt uses FP32 sqrt/divide1. Modeling762feb6c...412c3b casts
//! normalized values to BF16 before multiplying original F32 gamma. This
//! source-36-F32 profile enables neither a model nor CUDA/FLA execution.

use super::reduction::sum_inner_128;
use crate::accelerator::SourceIdentity;
use crate::quant::QuantKind;
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

const WIDTH: usize = 128;
const MAX_HEADS: usize = 64;
const EPSILON: f32 = 1e-6;

/// Private operator outputs; publication/source eligibility remain caller-owned.
#[derive(Debug)]
pub struct HeadRms128 {
    source: SourceIdentity,
    mean: Vec<f32>,
    inverse: Vec<f32>,
    normalized_bf16: Vec<f32>,
    weighted_fp32: Vec<f32>,
}

impl HeadRms128 {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }
    pub fn inverse(&self) -> &[f32] {
        &self.inverse
    }
    /// Exactly widened BF16 words in original `[head,128]` order.
    pub fn normalized_bf16(&self) -> &[f32] {
        &self.normalized_bf16
    }
    /// Original F32 gamma product; no BF16 cast follows this prefix.
    pub fn weighted_fp32(&self) -> &[f32] {
        &self.weighted_fp32
    }
}

fn validate(
    source: SourceIdentity,
    gamma: SourceTensorRef<'_>,
    core: &[f32],
    heads: usize,
) -> Result<[f32; WIDTH]> {
    if !source.is_bound() {
        return Err(Error::Malformed("pinned head RMS source binding".into()));
    }
    if !(1..=MAX_HEADS).contains(&heads) {
        return Err(Error::Limit("pinned head RMS head bound".into()));
    }
    if gamma.kind != QuantKind::F32 {
        return Err(Error::Unsupported(
            "pinned head RMS requires original F32 gamma".into(),
        ));
    }
    if gamma.shape != [WIDTH as u64] || core.len() != heads * WIDTH {
        return Err(Error::Malformed("pinned head RMS geometry".into()));
    }
    gamma.validate()?;
    if core
        .iter()
        .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed(
            "pinned head RMS requires finite exact BF16 core".into(),
        ));
    }
    let gamma = std::array::from_fn(|index| {
        let start = index * 4;
        f32::from_le_bytes(
            gamma.packed_bytes[start..start + 4]
                .try_into()
                .expect("validated F32 source gamma"),
        )
    });
    if gamma.iter().any(|value: &f32| !value.is_finite()) {
        return Err(Error::Malformed("nonfinite pinned head RMS gamma".into()));
    }
    Ok(gamma)
}

fn round_bf16(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(Error::Malformed("nonfinite pinned head RMS value".into()));
    }
    let word = value.to_bits();
    let bias = 0x7fff + ((word >> 16) & 1);
    let rounded = f32::from_bits(word.wrapping_add(bias) & 0xffff_0000);
    if !rounded.is_finite() {
        return Err(Error::Limit("pinned head RMS BF16 overflow".into()));
    }
    Ok(rounded)
}

/// Original contiguous `[head,128]` BF16 core and rank1 F32 source gamma128.
/// Fixed epsilon1e-6 and pinned CPU AVX2 mean order. No gate/SiLU, token loop,
/// provider dispatch, source mutation, SessionAuthority or StateRoot changes.
pub fn head_rms_128(
    source: SourceIdentity,
    gamma: SourceTensorRef<'_>,
    core: &[f32],
    heads: usize,
) -> Result<HeadRms128> {
    let gamma = validate(source, gamma, core, heads)?;
    // pow(2) materializes FP32 squares before CPU sum_out(...).div_(128).
    let squares = core.iter().map(|&value| value * value).collect::<Vec<_>>();
    let sums = sum_inner_128(source, &squares)?;
    let mut mean = Vec::with_capacity(heads);
    let mut inverse = Vec::with_capacity(heads);
    let mut normalized_bf16 = Vec::with_capacity(core.len());
    let mut weighted_fp32 = Vec::with_capacity(core.len());
    for (row, sum) in core.chunks_exact(WIDTH).zip(sums) {
        let row_mean = sum / WIDTH as f32;
        let denominator = row_mean + EPSILON;
        let row_inverse = 1.0f32 / denominator.sqrt();
        if !denominator.is_finite() || !row_inverse.is_finite() {
            return Err(Error::Malformed("nonfinite pinned head RMS inverse".into()));
        }
        mean.push(row_mean);
        inverse.push(row_inverse);
        for (&value, &weight) in row.iter().zip(&gamma) {
            let normalized = round_bf16(value * row_inverse)?;
            let weighted = weight * normalized;
            if !weighted.is_finite() {
                return Err(Error::Malformed("nonfinite pinned head RMS product".into()));
            }
            normalized_bf16.push(normalized);
            weighted_fp32.push(weighted);
        }
    }
    Ok(HeadRms128 {
        source,
        mean,
        inverse,
        normalized_bf16,
        weighted_fp32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn packed(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
    fn gamma(bytes: &[u8]) -> SourceTensorRef<'_> {
        SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[WIDTH as u64],
            packed_bytes: bytes,
        }
    }
    fn exact(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "word {index}");
        }
    }

    #[test]
    fn mean_keeps_pinned_sum_partial_order_before_dividing128() {
        let mut core = [0.0; WIDTH];
        core[0] = 128.0;
        core[1] = 1.0 / 32.0;
        core[33] = 1.0 / 32.0;
        let weights = packed(&[1.0; WIDTH]);
        let result = head_rms_128(source(), gamma(&weights), &core, 1).unwrap();
        // Two half-ULPs accumulate in a lane before the large lane is added.
        assert_eq!(result.mean()[0].to_bits(), 0x4300_0001);
        let serial = core.iter().map(|value| value * value).sum::<f32>() / 128.0;
        assert_eq!(serial.to_bits(), 128.0f32.to_bits());
    }

    #[test]
    fn independent_heads_preserve_signed_zero_and_unrounded_f32_gamma() {
        let mut core = vec![1.0; WIDTH * 2];
        core[WIDTH..].fill(0.0);
        core[WIDTH] = -0.0;
        let weights = packed(&[1.0001; WIDTH]);
        let source = source();
        let result = head_rms_128(source, gamma(&weights), &core, 2).unwrap();
        assert_eq!(result.source_identity(), source);
        exact(result.mean(), &[1.0, 0.0]);
        exact(&result.normalized_bf16()[..WIDTH], &[1.0; WIDTH]);
        exact(&result.weighted_fp32()[..WIDTH], &[1.0001; WIDTH]);
        assert_ne!(result.weighted_fp32()[0].to_bits() & 0xffff, 0);
        assert_eq!(
            result.normalized_bf16()[WIDTH].to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(result.weighted_fp32()[WIDTH].to_bits(), (-0.0f32).to_bits());
        assert!(
            result.normalized_bf16()[WIDTH + 1..]
                .iter()
                .all(|value| value.to_bits() == 0)
        );
        assert!(result.inverse().iter().all(|value| value.is_finite()));
        let max = head_rms_128(
            source,
            gamma(&weights),
            &vec![0.0; MAX_HEADS * WIDTH],
            MAX_HEADS,
        )
        .unwrap();
        assert_eq!(max.mean().len(), MAX_HEADS);
        assert_eq!(max.weighted_fp32().len(), MAX_HEADS * WIDTH);
    }

    #[test]
    fn malformed_or_nonfinite_bindings_refuse_before_returning_outputs() {
        let source = source();
        let weights = packed(&[1.0; WIDTH]);
        let valid = gamma(&weights);
        let core = [1.0; WIDTH];
        assert!(head_rms_128(SourceIdentity::unbound(), valid, &core, 1).is_err());
        for heads in [0, MAX_HEADS + 1, usize::MAX] {
            assert!(matches!(
                head_rms_128(source, valid, &core, heads),
                Err(Error::Limit(_))
            ));
        }
        for input in [&[][..], &core[..WIDTH - 1]] {
            assert!(head_rms_128(source, valid, input, 1).is_err());
        }
        for shape in [&[][..], &[127][..], &[1, 128][..], &[u64::MAX][..]] {
            assert!(head_rms_128(source, SourceTensorRef { shape, ..valid }, &core, 1).is_err());
        }
        assert!(matches!(
            head_rms_128(
                source,
                SourceTensorRef {
                    kind: QuantKind::Bf16,
                    ..valid
                },
                &core,
                1
            ),
            Err(Error::Unsupported(_))
        ));
        for bytes in [&weights[..511], &[0u8; 513][..]] {
            assert!(head_rms_128(source, gamma(bytes), &core, 1).is_err());
        }
        for invalid in [f32::NAN, f32::INFINITY, 1.0001] {
            let mut values = core;
            values[17] = invalid;
            assert!(head_rms_128(source, valid, &values, 1).is_err());
        }
        for invalid in [f32::NAN, f32::NEG_INFINITY] {
            let mut values = [1.0; WIDTH];
            values[127] = invalid;
            assert!(head_rms_128(source, gamma(&packed(&values)), &core, 1).is_err());
        }
        let large = f32::from_bits(0x7f7f_0000);
        assert!(head_rms_128(source, valid, &[large; WIDTH], 1).is_err());
        let mut sparse = [0.0; WIDTH];
        sparse[0] = 1.0;
        assert!(head_rms_128(source, gamma(&packed(&[f32::MAX; WIDTH])), &sparse, 1).is_err());
    }

    #[test]
    fn bf16_conversion_keeps_rne_ties_and_refuses_overflow() {
        for (input, expected) in [
            (0x3f80_8000, 0x3f80_0000),
            (0x3f81_8000, 0x3f82_0000),
            (0x8000_0000, 0x8000_0000),
        ] {
            assert_eq!(
                round_bf16(f32::from_bits(input)).unwrap().to_bits(),
                expected
            );
        }
        for input in [f32::INFINITY, f32::NAN, f32::MAX] {
            assert!(round_bf16(input).is_err());
        }
    }

    #[test]
    fn actual_original_head0_prefix_matches_all_captured_golden_bits() {
        // Original .8B source36F32, tokens11/353, layer0 head0. Expected
        // bytes copied from actual twelve-stage observed source calls; no
        // candidate-derived oracle. Full16 heads remain in the ignored run.
        // Fixture SHA256:62f5f75acb709c43c6aea91b0bb41dd4c4e5516c1354ebfae07cfb4ff7d5016d.
        let bytes = include_bytes!("norm/pinned_head128.bin");
        let mut cursor = 0;
        fn take<'a>(bytes: &'a [u8], cursor: &mut usize, count: usize) -> &'a [u8] {
            let output = &bytes[*cursor..*cursor + count];
            *cursor += count;
            output
        }
        fn word(bytes: &[u8], cursor: &mut usize) -> u32 {
            u32::from_le_bytes(take(bytes, cursor, 4).try_into().unwrap())
        }
        fn f32_values(bytes: &[u8], cursor: &mut usize, count: usize) -> Vec<f32> {
            (0..count)
                .map(|_| f32::from_bits(word(bytes, cursor)))
                .collect()
        }
        fn bf16_values(bytes: &[u8], cursor: &mut usize, count: usize) -> Vec<f32> {
            (0..count)
                .map(|_| {
                    let value = u16::from_le_bytes(take(bytes, cursor, 2).try_into().unwrap());
                    f32::from_bits(u32::from(value) << 16)
                })
                .collect()
        }
        assert_eq!(take(bytes, &mut cursor, 8), b"ZGHFN128");
        assert_eq!(word(bytes, &mut cursor), 2);
        for step in 0..2 {
            assert_eq!(word(bytes, &mut cursor), step);
            assert_eq!(word(bytes, &mut cursor), 1);
            let core = bf16_values(bytes, &mut cursor, WIDTH);
            let gamma = gamma(take(bytes, &mut cursor, WIDTH * 4));
            let mean = f32_values(bytes, &mut cursor, 1);
            let inverse = f32_values(bytes, &mut cursor, 1);
            let normalized = bf16_values(bytes, &mut cursor, WIDTH);
            let weighted = f32_values(bytes, &mut cursor, WIDTH);
            let result = head_rms_128(source(), gamma, &core, 1).unwrap();
            exact(result.mean(), &mean);
            exact(result.inverse(), &inverse);
            exact(result.normalized_bf16(), &normalized);
            exact(result.weighted_fp32(), &weighted);
        }
        assert_eq!(cursor, bytes.len());
    }
}
