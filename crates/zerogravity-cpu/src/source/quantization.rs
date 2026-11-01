//! Source-exact GGUF activation coding and integer-dot reduction schedules.
//! Source validation, Q4_K refusal and the dequantized diagnostic stay at the boundary.

use super::dot_row;
use crate::codecs::{f16_to_f32, f32_to_f16_rne};
use crate::{Error, Result};
use zerogravity_core::encoding::StoredTensorEncoding;
use zerogravity_core::numerics::ActivationEncoding;

/// Ephemeral source GGUF activation encoding, independent of model family.
/// Dynamic scales are recomputed for every GEMV input; not a weight codec.
#[derive(Clone)]
pub(super) struct Q8SmallBlock {
    d: f32, // f16-roundtripped GGUF Q8_0/Q8_1 scale
    s: f32, // f16-roundtripped sum(q_i)*unrounded_scale, Q8_1 only
    qs: [i8; 32],
}

#[derive(Clone)]
pub(super) struct Q8KBlock {
    d: f32, // GGUF Q8_K signed dynamic f32 scale
    qs: [i8; 256],
}

pub(super) enum QuantizedActivation {
    Direct,
    Small(Vec<Q8SmallBlock>),
    Big(Vec<Q8KBlock>),
}

impl QuantizedActivation {
    pub(super) fn prepare(encoding: ActivationEncoding, input: &[f32]) -> Result<Self> {
        if !input.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed("nonfinite GEMV input".into()));
        }
        match encoding {
            ActivationEncoding::GgmlQ8_0 | ActivationEncoding::GgmlQ8_1 => {
                if !input.len().is_multiple_of(32) {
                    return Err(Error::Malformed("Q8 activation width misaligned".into()));
                }
                let mut out = Vec::with_capacity(input.len() / 32);
                for block in input.chunks_exact(32) {
                    let amax = block.iter().map(|x| x.abs()).fold(0f32, f32::max);
                    let d = amax / 127.0;
                    let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                    let mut qs = [0i8; 32];
                    let mut sum = 0i32;
                    for (i, &x) in block.iter().enumerate() {
                        let value = (x * inv).round_ties_even().clamp(-128.0, 127.0) as i8;
                        qs[i] = value;
                        sum += i32::from(value);
                    }
                    let d_half = f16_to_f32(f32_to_f16_rne(d));
                    let s_half = f16_to_f32(f32_to_f16_rne(sum as f32 * d));
                    out.push(Q8SmallBlock {
                        d: d_half,
                        s: s_half,
                        qs,
                    });
                }
                Ok(Self::Small(out))
            }
            ActivationEncoding::GgmlQ8K => {
                if !input.len().is_multiple_of(256) {
                    return Err(Error::Malformed("Q8_K activation width misaligned".into()));
                }
                let mut out = Vec::with_capacity(input.len() / 256);
                for block in input.chunks_exact(256) {
                    // GGUF Q8_K chooses the *signed* input value with largest
                    // absolute magnitude and uses iscale = -127 / signed_max.
                    let mut max_signed = 0.0f32;
                    let mut amax = 0.0f32;
                    for &x in block {
                        if x.abs() > amax {
                            amax = x.abs();
                            max_signed = x;
                        }
                    }
                    let inv = if amax == 0.0 {
                        0.0
                    } else {
                        -127.0 / max_signed
                    };
                    let d = if amax == 0.0 { 0.0 } else { 1.0 / inv };
                    let mut qs = [0i8; 256];
                    for (i, &x) in block.iter().enumerate() {
                        qs[i] = (x * inv).round_ties_even().clamp(-128.0, 127.0) as i8;
                    }
                    out.push(Q8KBlock { d, qs });
                }
                Ok(Self::Big(out))
            }
            ActivationEncoding::Scalar(_) => Ok(Self::Direct),
        }
    }
}

/// Matches the x86 GGML Q4_0_8x8 versus Q8_0 GEMV numerical contract,
/// while reading original (unrepacked) GGUF row bytes directly.
///
/// The source optimized kernel pre-repacks eight weight rows for efficient
/// memory access, sums all 32 signed integer products into ONE i32 per
/// quantization block, then performs ONE fused f32 multiply-add per block.
/// The single-row GGML vec_dot kernel instead keeps eight partial f32
/// accumulators and horizontally reduces at the end. Both are valid
/// quantized dot implementations but they round DIFFERENTLY in f32.
/// Replication of the 8x8 reduction order does not require weight repacking
/// because each row has independent integer sums and f32 accumulators.
pub(super) fn dot_q4_0_repacked_8x8_source_row(row: &[u8], blocks: &[Q8SmallBlock]) -> f32 {
    let mut accumulator = 0.0f32;
    for (weights, activation) in row.chunks_exact(18).zip(blocks) {
        let packed = &weights[2..18];
        let mut integer_sum = 0i32;
        for (i, &nibbles) in packed.iter().enumerate() {
            let lo = i32::from(nibbles & 15) - 8;
            let hi = i32::from(nibbles >> 4) - 8;
            integer_sum += lo * i32::from(activation.qs[i]);
            integer_sum += hi * i32::from(activation.qs[i + 16]);
        }
        let scale = f16_to_f32(u16::from_le_bytes([weights[0], weights[1]])) * activation.d;
        accumulator = (integer_sum as f32).mul_add(scale, accumulator);
    }
    accumulator
}

/// Physical source-paired GGUF block integer-dot reference. This is Rust math
/// and never calls a foreign inference or quantization library. Keeping it
/// separate from pure dequantized FP32 matvec allows explicit source semantics
/// versus idealized FP32 diagnostic reference comparisons.
pub(super) fn dot_row_with_activation(
    kind: StoredTensorEncoding,
    row: &[u8],
    original: &[f32],
    activation: &QuantizedActivation,
) -> f32 {
    match (kind, activation) {
        (
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_0),
            QuantizedActivation::Small(blocks),
        ) => {
            // Source x86 GGML AVX2/FMA reduction topology, expressed as
            // portable Rust scalar lanes: 8 dot accumulators (four i8 pairs
            // each), fused scale*integer accumulation per 32-element block,
            // then a source-ordered horizontal sum. No ISA-specific SIMD is
            // required; this is a numerical specialization, not a throughput
            // optimization. Other architectures can supply their own
            // legally qualified kernel.
            let mut lanes = [0.0f32; 8];
            for (weight, quant) in row.chunks_exact(18).zip(blocks) {
                let d = f16_to_f32(u16::from_le_bytes([weight[0], weight[1]])) * quant.d;
                let packed = &weight[2..18];
                for (lane, accumulator) in lanes.iter_mut().enumerate() {
                    let mut dot = 0i32;
                    for j in lane * 4..lane * 4 + 4 {
                        let quantized = if j < 16 {
                            i32::from(packed[j] & 15) - 8
                        } else {
                            i32::from(packed[j - 16] >> 4) - 8
                        };
                        dot += quantized * i32::from(quant.qs[j]);
                    }
                    *accumulator = d.mul_add(dot as f32, *accumulator);
                }
            }
            let r0 = lanes[0] + lanes[4];
            let r1 = lanes[1] + lanes[5];
            let r2 = lanes[2] + lanes[6];
            let r3 = lanes[3] + lanes[7];
            (r0 + r2) + (r1 + r3)
        }
        (
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q8_0),
            QuantizedActivation::Small(blocks),
        ) => {
            // Source GGML Haswell nonrepacked Q8_0 x Q8_0 uses one f32
            // FMA accumulator per 4-element integer lane, then an ordered
            // 8-lane horizontal sum. A single int32 block sum accumulated in
            // f32 across blocks is algebraically equivalent but rounds
            // differently and can flip later Q8 quantization boundaries.
            let mut lanes = [0.0f32; 8];
            for (weight, quant) in row.chunks_exact(34).zip(blocks) {
                let scale = f16_to_f32(u16::from_le_bytes([weight[0], weight[1]])) * quant.d;
                for (lane, acc) in lanes.iter_mut().enumerate() {
                    let mut pair_dot = 0i32;
                    for i in lane * 4..lane * 4 + 4 {
                        pair_dot += i32::from(weight[2 + i] as i8) * i32::from(quant.qs[i]);
                    }
                    *acc = (pair_dot as f32).mul_add(scale, *acc);
                }
            }
            let r0 = lanes[0] + lanes[4];
            let r1 = lanes[1] + lanes[5];
            let r2 = lanes[2] + lanes[6];
            let r3 = lanes[3] + lanes[7];
            (r0 + r2) + (r1 + r3)
        }
        (
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4_1),
            QuantizedActivation::Small(blocks),
        ) => {
            // Source Haswell Q4_1 x Q8_1 uses a floating zero-point
            // correction accumulated separately across blocks. The 8 SIMD
            // integer partial dot sums (four values per lane) are combined
            // with FMA into 8 f32 lanes; one final ordered horizontal sum
            // is added to the cumulative zero-point correction.
            let mut lanes = [0.0f32; 8];
            let mut summs = 0.0f32;
            for (weight, quant) in row.chunks_exact(20).zip(blocks) {
                let d0 = f16_to_f32(u16::from_le_bytes([weight[0], weight[1]]));
                let m = f16_to_f32(u16::from_le_bytes([weight[2], weight[3]]));
                summs += m * quant.s;
                let scale = d0 * quant.d;
                let packed = &weight[4..20];
                for (lane, acc) in lanes.iter_mut().enumerate() {
                    let mut partial = 0i32;
                    for i in lane * 4..lane * 4 + 4 {
                        let w = if i < 16 {
                            i32::from(packed[i] & 15)
                        } else {
                            i32::from(packed[i - 16] >> 4)
                        };
                        partial += w * i32::from(quant.qs[i]);
                    }
                    *acc = (partial as f32).mul_add(scale, *acc);
                }
            }
            let r0 = lanes[0] + lanes[4];
            let r1 = lanes[1] + lanes[5];
            let r2 = lanes[2] + lanes[6];
            let r3 = lanes[3] + lanes[7];
            ((r0 + r2) + (r1 + r3)) + summs
        }
        (
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q6K),
            QuantizedActivation::Big(blocks),
        ) => {
            // Source Haswell GGML Q6_K × Q8_K: one signed integer
            // accumulator per 32 consecutive activation positions (8 lanes
            // per 256-value superblock), then FMA each lane with its f32
            // dynamic Q8_K × f16 Q6_K scale and horizontally reduce once.
            // Summing 256 weighted integer products into one f32 partial
            // before accumulation produces different numerical semantics.
            let mut lanes = [0.0f32; 8];
            for (weight, quant) in row.chunks_exact(210).zip(blocks) {
                let scale = quant.d * f16_to_f32(u16::from_le_bytes([weight[208], weight[209]]));
                let mut int_acc = [0i32; 8];
                for half in 0..2 {
                    let low = &weight[half * 64..half * 64 + 64];
                    let high = &weight[128 + half * 32..128 + half * 32 + 32];
                    let scales = &weight[192 + half * 8..192 + half * 8 + 8];
                    for lane in 0..32 {
                        let hi = high[lane];
                        for group in 0..4 {
                            let nibble = if group < 2 {
                                low[lane + group * 32] & 15
                            } else {
                                low[lane + (group - 2) * 32] >> 4
                            };
                            let code = i32::from(nibble | (((hi >> (2 * group)) & 3) << 4));
                            let subscale = i32::from(scales[group * 2 + lane / 16] as i8);
                            let at = half * 128 + group * 32 + lane;
                            // SSE2/AVX2 pairwise madd maps four consecutive
                            // elements into each 32-bit SIMD lane, then adds
                            // the corresponding positions from all eight
                            // 32-value groups. Offset correction below uses
                            // the original eight contiguous 32-value groups,
                            // a DISTINCT lane layout. The final integer sum
                            // is equal but FMA per-lane reduction is not.
                            int_acc[(at % 32) / 4] += code * subscale * i32::from(quant.qs[at]);
                        }
                    }
                }
                let scales = &weight[192..208];
                for (lane, acc) in int_acc.iter_mut().enumerate() {
                    let mut offset = 0i32;
                    for sub in 0..2 {
                        let group16 = lane * 2 + sub;
                        let q8sum = quant.qs[group16 * 16..(group16 + 1) * 16]
                            .iter()
                            .map(|&v| i32::from(v))
                            .sum::<i32>();
                        offset += q8sum * i32::from(scales[group16] as i8);
                    }
                    *acc -= 32 * offset;
                }
                for (acc, &code) in lanes.iter_mut().zip(&int_acc) {
                    *acc = (code as f32).mul_add(scale, *acc);
                }
            }
            let r0 = lanes[0] + lanes[4];
            let r1 = lanes[1] + lanes[5];
            let r2 = lanes[2] + lanes[6];
            let r3 = lanes[3] + lanes[7];
            (r0 + r2) + (r1 + r3)
        }
        // Q4_K is not present in the two initial Qwen checkpoints. Leave its
        // independent FP32 dequant numerical oracle operational pending a
        // separately source-qualified Q4_K x Q8_K block dot.
        (
            StoredTensorEncoding::Quantized(zerogravity_core::encoding::QuantScheme::Q4K),
            QuantizedActivation::Big(_),
        ) => dot_row(kind, row, original),
        (
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32)
            | StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F16)
            | StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16),
            QuantizedActivation::Direct,
        ) => dot_row(kind, row, original),
        _ => unreachable!("shape/codec validated before GEMV dispatch"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::encoding::QuantScheme;

    #[test]
    fn small_activation_keeps_ties_even_half_scales_and_zero() {
        for scheme in [QuantScheme::Q4_0, QuantScheme::Q4_1, QuantScheme::Q8_0] {
            let kind = StoredTensorEncoding::Quantized(scheme);
            let mut input = [0.0; 32];
            input[..7].copy_from_slice(&[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]);
            let QuantizedActivation::Small(blocks) = QuantizedActivation::prepare(
                zerogravity_core::numerics::ProjectionContract::Source
                    .activation(kind)
                    .unwrap(),
                &input,
            )
            .unwrap() else {
                panic!("expected small activation");
            };
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].d.to_bits(), 1.0f32.to_bits());
            assert_eq!(blocks[0].s.to_bits(), 127.0f32.to_bits());
            assert_eq!(&blocks[0].qs[..7], &[127, 0, 2, 2, 0, -2, -2]);
            assert!(blocks[0].qs[7..].iter().all(|&q| q == 0));
            let QuantizedActivation::Small(blocks) = QuantizedActivation::prepare(
                zerogravity_core::numerics::ProjectionContract::Source
                    .activation(kind)
                    .unwrap(),
                &[1.0; 32],
            )
            .unwrap() else {
                panic!("expected small activation");
            };
            assert_eq!(blocks[0].d.to_bits(), 0x3c01_0000); // f16 RNE of 1/127
            assert_eq!(blocks[0].s.to_bits(), 32.0f32.to_bits());
            assert_eq!(blocks[0].qs, [127; 32]);
            for zero in [0.0, -0.0] {
                let QuantizedActivation::Small(blocks) = QuantizedActivation::prepare(
                    zerogravity_core::numerics::ProjectionContract::Source
                        .activation(kind)
                        .unwrap(),
                    &[zero; 32],
                )
                .unwrap() else {
                    panic!("expected small activation");
                };
                assert_eq!(blocks[0].d.to_bits(), 0);
                assert_eq!(blocks[0].s.to_bits(), 0);
                assert_eq!(blocks[0].qs, [0; 32]);
            }
        }
    }

    #[test]
    fn q8_k_keeps_first_signed_maximum_and_zero() {
        let kind = StoredTensorEncoding::Quantized(QuantScheme::Q6K);
        for (first, scale, code) in [(127.0, -1.0f32, -2), (-127.0, 1.0, 2)] {
            let mut input = [0.0; 256];
            input[..4].copy_from_slice(&[first, -first, 0.5, 1.5]);
            let QuantizedActivation::Big(blocks) = QuantizedActivation::prepare(
                zerogravity_core::numerics::ProjectionContract::Source
                    .activation(kind)
                    .unwrap(),
                &input,
            )
            .unwrap() else {
                panic!("expected Q8_K activation");
            };
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].d.to_bits(), scale.to_bits());
            assert_eq!(&blocks[0].qs[..4], &[-127, 127, 0, code]);
            assert!(blocks[0].qs[4..].iter().all(|&q| q == 0));
        }
        for zero in [0.0, -0.0] {
            let QuantizedActivation::Big(blocks) = QuantizedActivation::prepare(
                zerogravity_core::numerics::ProjectionContract::Source
                    .activation(kind)
                    .unwrap(),
                &[zero; 256],
            )
            .unwrap() else {
                panic!("expected Q8_K activation");
            };
            assert_eq!(blocks[0].d.to_bits(), 0);
            assert_eq!(blocks[0].qs, [0; 256]);
        }
    }

    #[test]
    fn activation_rejects_misaligned_or_nonfinite_inputs() {
        for scheme in [
            QuantScheme::Q4_0,
            QuantScheme::Q4_1,
            QuantScheme::Q8_0,
            QuantScheme::Q6K,
        ] {
            let kind = StoredTensorEncoding::Quantized(scheme);
            let width = kind.block().0 as usize;
            assert!(matches!(
                QuantizedActivation::prepare(
                    zerogravity_core::numerics::ProjectionContract::Source
                        .activation(kind)
                        .unwrap(),
                    &vec![0.0; width - 1]
                ),
                Err(Error::Malformed(_))
            ));
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(matches!(
                    QuantizedActivation::prepare(
                        zerogravity_core::numerics::ProjectionContract::Source
                            .activation(kind)
                            .unwrap(),
                        &vec![value; width]
                    ),
                    Err(Error::Malformed(_))
                ));
            }
        }
    }

    #[test]
    fn source_exact_blocks_match_hand_computed_integer_fixtures() {
        // Exact small fixtures, not a dequantized-f32 comparison or general GEMV qualification.
        for (scheme, expected) in [
            (QuantScheme::Q4_0, 6096.0f32),
            (QuantScheme::Q4_1, 1016.0),
            (QuantScheme::Q8_0, -1016.0),
            (QuantScheme::Q6K, -24384.0),
        ] {
            let kind = StoredTensorEncoding::Quantized(scheme);
            let (width, bytes) = kind.block();
            let input = vec![127.0; width as usize];
            let activation = QuantizedActivation::prepare(
                zerogravity_core::numerics::ProjectionContract::Source
                    .activation(kind)
                    .unwrap(),
                &input,
            )
            .unwrap();
            let mut row = vec![0u8; bytes as usize];
            match scheme {
                QuantScheme::Q4_0 => {
                    row[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                    row[2..].fill(0xa9); // 16 ones + 16 twos
                }
                QuantScheme::Q4_1 => {
                    row[..2].copy_from_slice(&0x3800u16.to_le_bytes());
                    row[2..4].copy_from_slice(&0xbc00u16.to_le_bytes());
                    row[4..].fill(0x32); // d=1/2, m=-1: 16 zeros + 16 halves
                }
                QuantScheme::Q8_0 => {
                    row[..2].copy_from_slice(&0x3800u16.to_le_bytes());
                    for pair in row[2..].chunks_exact_mut(2) {
                        pair.copy_from_slice(&[(-3i8) as u8, 2]);
                    }
                }
                QuantScheme::Q6K => {
                    row[..128].fill(0x21);
                    row[128..192].fill(0xaa); // codes 33/34, before -32 correction
                    row[192..208].fill(255); // signed local scales = -1
                    row[208..].copy_from_slice(&0x3800u16.to_le_bytes());
                }
                _ => unreachable!(),
            }
            assert_eq!(
                dot_row_with_activation(kind, &row, &input, &activation).to_bits(),
                expected.to_bits(),
                "{scheme:?}"
            );
            if scheme == QuantScheme::Q4_0 {
                let QuantizedActivation::Small(blocks) = &activation else {
                    panic!("expected small activation");
                };
                assert_eq!(
                    dot_q4_0_repacked_8x8_source_row(&row, blocks).to_bits(),
                    expected.to_bits()
                );
            }
        }
    }

    #[test]
    fn q4_0_repacked_block_order_stays_distinct_from_eight_lane_reduction() {
        let kind = StoredTensorEncoding::Quantized(QuantScheme::Q4_0);
        let mut input = [0.0; 96];
        for block in input.chunks_exact_mut(32) {
            block[31] = 127.0; // d=1; this position's weight is zero
        }
        input[0] = -64.0;
        input[36] = 1.0;
        input[64] = 64.0;
        let activation = QuantizedActivation::prepare(
            zerogravity_core::numerics::ProjectionContract::Source
                .activation(kind)
                .unwrap(),
            &input,
        )
        .unwrap();
        let QuantizedActivation::Small(blocks) = &activation else {
            panic!("expected small activation");
        };
        let mut row = [0x88u8; 54];
        for block in row.chunks_exact_mut(18) {
            block[..2].copy_from_slice(&0x7800u16.to_le_bytes()); // 32768
            block[2] = 0x80; // first weight = -8
        }
        row[18..20].copy_from_slice(&0x3c00u16.to_le_bytes());
        row[20] = 0x88;
        row[24] = 0x89; // second block's lane 1 contributes +1
        // Block sums: 2^24, 1, -2^24. Serial block FMA loses the 1;
        // lane 0 cancels before lane 1 is added in the nonrepacked schedule.
        assert_eq!(dot_q4_0_repacked_8x8_source_row(&row, blocks).to_bits(), 0);
        assert_eq!(
            dot_row_with_activation(kind, &row, &input, &activation).to_bits(),
            1.0f32.to_bits()
        );
    }
}
